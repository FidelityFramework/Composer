/// The Espressif ROM-loadable image container.
///
/// The ESP32-S3 mask ROM at 0x4000_0400 reads this structure from flash offset
/// 0, loads each segment to its stated address, and jumps to the entry point.
/// Writing it here rather than shelling out to esptool keeps image production
/// inside the owned path: Composer already verifies ELF extents and vector
/// placement against BAREWire declarations, and the container is the last step
/// of that same verification rather than a separate tool's opinion.
///
/// Layout, from esp_app_format.h (esp_image_header_t is asserted to be 24
/// bytes there, and that assertion is reproduced below):
///
///   esp_image_header_t                       24 bytes, packed
///   per segment: { load_addr: u32; data_len: u32 } then data_len bytes
///   zero padding so that the checksum byte lands on a 16-byte boundary
///   1 checksum byte
///   32-byte SHA-256 of everything above, when HashAppended
///
/// This module computes bytes from bytes. It holds no opinion about which
/// segments an image should have; XtensaImage supplies those from the linked
/// ELF, and the declarations decide whether they are placeable.
module BackEnd.MCU.EspImage

open System
open System.Buffers.Binary
open System.Security.Cryptography

/// esp_image_header_t.magic
[<Literal>]
let HeaderMagic = 0xE9uy

/// ESPLoader.ESP_CHECKSUM_MAGIC: the checksum accumulator's seed.
[<Literal>]
let ChecksumSeed = 0xEFuy

/// esp_chip_id_t.ESP_CHIP_ID_ESP32S3
[<Literal>]
let ChipIdEsp32S3 = 0x0009

/// ESP_IMAGE_MAX_SEGMENTS
[<Literal>]
let MaxSegments = 16

/// esp_image_header_t is packed and fixed-size; the C header static-asserts it.
[<Literal>]
let HeaderBytes = 24

/// Each segment carries an 8-byte { load_addr; data_len } prologue.
[<Literal>]
let SegmentHeaderBytes = 8

/// The checksum byte is placed so that it ends a 16-byte block.
[<Literal>]
let ChecksumAlignment = 16

/// wp_pin = 0xEE disables the ROM loader's write-protect pin handling, which
/// is what an image that does not reconfigure the flash pins should say.
[<Literal>]
let WriteProtectDisabled = 0xEEuy

/// esp_image_flash_size_t. The ROM loader reads this before it can fetch
/// anything, so a wrong value is a boot failure rather than a runtime one.
/// The badge's ESP32-S3-WROOM-1-N8 carries 8 MB: FlashSize8MB.
module FlashSize =
    [<Literal>]
    let Size1MB = 0
    [<Literal>]
    let Size2MB = 1
    [<Literal>]
    let Size4MB = 2
    [<Literal>]
    let Size8MB = 3
    [<Literal>]
    let Size16MB = 4
    [<Literal>]
    let Size32MB = 5
    [<Literal>]
    let Size64MB = 6
    [<Literal>]
    let Size128MB = 7

/// esp_image_spi_mode_t
module SpiMode =
    [<Literal>]
    let Qio = 0
    [<Literal>]
    let Qout = 1
    [<Literal>]
    let Dio = 2
    [<Literal>]
    let Dout = 3
    [<Literal>]
    let FastRead = 4
    [<Literal>]
    let SlowRead = 5

/// One loadable segment: where the ROM loader puts it, and what goes there.
type Segment = {
    /// CPU address the ROM loader copies this data to. For an all-SRAM image
    /// every segment addresses internal SRAM, and no flash MMU mapping is
    /// established by the loader.
    LoadAddress: uint32
    Data: byte array
}

/// The fields of esp_image_header_t this writer does not take from the
/// segments themselves. Flash access parameters are read by the ROM loader
/// before it can fetch anything, so they are image facts, not runtime ones.
type Header = {
    EntryAddress: uint32
    ChipId: int
    /// Minimal and maximal supported chip revision, as major * 100 + minor.
    MinChipRevFull: int
    MaxChipRevFull: int
    /// esp_image_spi_mode_t
    SpiMode: int
    /// esp_image_spi_freq_t
    SpiSpeed: int
    /// esp_image_flash_size_t
    SpiSize: int
    /// Append a SHA-256 of the whole image for corruption detection. This is
    /// not secure boot and is not a signature.
    HashAppended: bool
}

/// Segment data is written in 4-byte units; the ROM loader's copy is
/// word-oriented and esptool pads to the same boundary.
let private padTo4 (data: byte array) =
    match data.Length % 4 with
    | 0 -> data
    | r -> Array.append data (Array.zeroCreate (4 - r))

/// Everything that makes a segment set unwritable, checked before any bytes
/// are produced so that a rejection names the cause rather than a bad image.
let private validate (header: Header) (segments: Segment list) =
    if List.isEmpty segments then failwith "ESP image needs at least one segment"
    if List.length segments > MaxSegments then
        failwithf "ESP image allows at most %d segments; got %d" MaxSegments (List.length segments)
    if header.ChipId < 0 || header.ChipId > 0xFFFF then failwith "Invalid esp_chip_id_t"
    if header.SpiMode < 0 || header.SpiMode > 0xFF then failwith "Invalid SpiMode"
    // spi_speed and spi_size share one byte as two nibbles.
    for field, value in [ "SpiSpeed", header.SpiSpeed; "SpiSize", header.SpiSize ] do
        if value < 0 || value > 0xF then failwithf "%s must fit in a nibble" field
    for field, value in [ "MinChipRevFull", header.MinChipRevFull; "MaxChipRevFull", header.MaxChipRevFull ] do
        if value < 0 || value > 0xFFFF then failwithf "%s must fit in a uint16" field
    if header.MaxChipRevFull < header.MinChipRevFull then
        failwith "ESP image maximal chip revision precedes its minimal revision"
    for segment in segments do
        if segment.Data.Length = 0 then failwithf "Empty segment at 0x%08X" segment.LoadAddress
        // The loader copies to a word-addressed destination.
        if segment.LoadAddress % 4u <> 0u then
            failwithf "Segment load address 0x%08X is not word-aligned" segment.LoadAddress
    // Overlapping segments would make the loaded image depend on load order.
    let extents =
        segments
        |> List.map (fun s -> uint64 s.LoadAddress, uint64 s.LoadAddress + uint64 (padTo4 s.Data).Length)
        |> List.sortBy fst
    extents
    |> List.pairwise
    |> List.iter (fun ((_, firstEnd), (secondStart, _)) ->
        if secondStart < firstEnd then
            failwithf "ESP image segments overlap at 0x%X" secondStart)

/// Produce the flash image. The returned bytes are what goes at flash offset 0.
let build (header: Header) (segments: Segment list) : byte array =
    validate header segments
    let padded = segments |> List.map (fun s -> { s with Data = padTo4 s.Data })
    let out = ResizeArray<byte>()
    let u32 (value: uint32) =
        let buffer = Array.zeroCreate 4
        BinaryPrimitives.WriteUInt32LittleEndian(Span(buffer), value)
        out.AddRange buffer
    let u16 (value: int) =
        let buffer = Array.zeroCreate 2
        BinaryPrimitives.WriteUInt16LittleEndian(Span(buffer), uint16 value)
        out.AddRange buffer

    // --- esp_image_header_t ------------------------------------------------
    out.Add HeaderMagic
    out.Add(byte (List.length padded))
    out.Add(byte header.SpiMode)
    // spi_speed occupies the low nibble, spi_size the high nibble.
    out.Add(byte ((header.SpiSize <<< 4) ||| header.SpiSpeed))
    u32 header.EntryAddress
    out.Add WriteProtectDisabled
    out.AddRange [| 0uy; 0uy; 0uy |]          // spi_pin_drv[3]
    u16 header.ChipId
    // min_chip_rev is superseded by min_chip_rev_full but still present.
    out.Add 0uy
    u16 header.MinChipRevFull
    u16 header.MaxChipRevFull
    out.AddRange(Array.zeroCreate 4)          // reserved[4]
    out.Add(if header.HashAppended then 1uy else 0uy)
    if out.Count <> HeaderBytes then
        failwithf "esp_image_header_t must be %d bytes; wrote %d" HeaderBytes out.Count

    // --- segments, accumulating the checksum over segment DATA only --------
    let mutable checksum = ChecksumSeed
    for segment in padded do
        u32 segment.LoadAddress
        u32 (uint32 segment.Data.Length)
        out.AddRange segment.Data
        for b in segment.Data do
            checksum <- checksum ^^^ b

    // --- padding so the checksum byte ends a 16-byte block -----------------
    // esptool seeks to (alignment - 1) mod alignment, writes one byte, and so
    // leaves the position on the boundary.
    while (out.Count + 1) % ChecksumAlignment <> 0 do
        out.Add 0uy
    out.Add checksum

    // --- optional SHA-256 over everything above ----------------------------
    if header.HashAppended then
        out.AddRange(SHA256.HashData(out.ToArray()))

    out.ToArray()

/// Read back what `build` wrote, so a verifier states the same facts from the
/// bytes rather than from the inputs that produced them.
type Parsed = {
    SegmentCount: int
    EntryAddress: uint32
    ChipId: int
    Checksum: byte
    ComputedChecksum: byte
    HashAppended: bool
    /// Present and verified only when HashAppended.
    HashMatches: bool
    Segments: (uint32 * int) list
    TotalBytes: int
}

/// Parse and check an image. Fails on anything a ROM loader would reject, so
/// that a broken writer is caught here rather than on a board.
let parse (image: byte array) : Parsed =
    if image.Length < HeaderBytes then failwith "Image shorter than esp_image_header_t"
    if image.[0] <> HeaderMagic then failwithf "Bad image magic 0x%02X" image.[0]
    let readU32 offset = BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(image).Slice(offset, 4))
    let readU16 offset = int (BinaryPrimitives.ReadUInt16LittleEndian(ReadOnlySpan(image).Slice(offset, 2)))
    let segmentCount = int image.[1]
    if segmentCount < 1 || segmentCount > MaxSegments then failwithf "Bad segment count %d" segmentCount
    let hashAppended = image.[23] = 1uy

    let mutable cursor = HeaderBytes
    let mutable checksum = ChecksumSeed
    let segments =
        [ for _ in 1 .. segmentCount do
            if cursor + SegmentHeaderBytes > image.Length then failwith "Truncated segment header"
            let loadAddress = readU32 cursor
            let length = int (readU32 (cursor + 4))
            cursor <- cursor + SegmentHeaderBytes
            if length <= 0 || cursor + length > image.Length then failwithf "Truncated segment at 0x%08X" loadAddress
            for i in cursor .. cursor + length - 1 do
                checksum <- checksum ^^^ image.[i]
            cursor <- cursor + length
            yield loadAddress, length ]

    // The checksum byte ends a 16-byte block; padding precedes it.
    let checksumIndex =
        let mutable index = cursor
        while (index + 1) % ChecksumAlignment <> 0 do index <- index + 1
        index
    if checksumIndex >= image.Length then failwith "Image ends before its checksum"
    let stated = image.[checksumIndex]
    let hashMatches =
        if not hashAppended then true
        else
            let hashStart = checksumIndex + 1
            if hashStart + 32 <> image.Length then
                failwithf "Image declares an appended hash but has %d trailing bytes" (image.Length - hashStart)
            let expected = SHA256.HashData(image.[0 .. checksumIndex])
            expected = image.[hashStart ..]

    { SegmentCount = segmentCount
      EntryAddress = readU32 4
      ChipId = readU16 12
      Checksum = stated
      ComputedChecksum = checksum
      HashAppended = hashAppended
      HashMatches = hashMatches
      Segments = segments
      TotalBytes = image.Length }

/// The check a build runs against its own output.
let verify (image: byte array) =
    let parsed = parse image
    if parsed.Checksum <> parsed.ComputedChecksum then
        failwithf "ESP image checksum 0x%02X disagrees with computed 0x%02X" parsed.Checksum parsed.ComputedChecksum
    if not parsed.HashMatches then failwith "ESP image SHA-256 does not match its contents"
    parsed
