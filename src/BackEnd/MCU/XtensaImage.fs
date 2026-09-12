/// Build a ROM-loadable ESP32-S3 image from Clef-generated LLVM IR.
///
/// The shape follows Image.fs deliberately: optimize, refuse constructs this
/// profile cannot support, assemble the owned startup, link with a generated
/// script, verify the result against the declarations, then write build
/// evidence. What differs is every check's content, and the toolchain.
///
/// Toolchain note. Per Farscape's toolchain-sovereignty division, clang is
/// confined to Farscape's offline archive production and takes no part in
/// lowering. This path therefore uses `opt`, `llc`, `llvm-mc` and `ld.lld`
/// only -- `llvm-mc` rather than clang's integrated assembler for the startup
/// source, and an in-process image writer rather than esptool.
///
/// STATUS: unexercised. No Xtensa-capable LLVM is installed, so this has never
/// been run end to end. The ELF segment reader and the image writer it feeds
/// are tested independently; the subprocess sequence is not.
module BackEnd.MCU.XtensaImage

open System
open System.IO
open System.Buffers.Binary
open System.Text.RegularExpressions
open Core.Types.Pipeline

/// EM_XTENSA. Image.fs checks for 40 (EM_ARM) in the same position.
[<Literal>]
let ElfMachineXtensa = 94

/// PT_LOAD
[<Literal>]
let private ProgramTypeLoad = 1u

let symbols text =
    Regex.Matches(text, @"(?m)^([0-9a-fA-F]+)\s+\w\s+(\S+)$")
    |> Seq.map (fun m -> m.Groups.[2].Value, Convert.ToUInt32(m.Groups.[1].Value, 16))
    |> Map.ofSeq

/// One loadable piece of a linked ELF32: where it goes and what goes there.
type LoadSegment = {
    VirtualAddress: uint32
    /// Bytes present in the file. A segment whose memory size exceeds this has
    /// a zero-fill tail (.bss), which an ESP image must not try to carry.
    FileBytes: int
    MemoryBytes: int
    FileOffset: int
    Data: byte array
}

/// The machine check is separate from the segment reader so that the reader is
/// architecture-neutral and can be exercised against any ELF32 LE file.
let requireXtensaElf (elf: byte array) =
    if elf.Length < 52 then failwith "Truncated ELF header"
    let machine = int (BinaryPrimitives.ReadUInt16LittleEndian(ReadOnlySpan(elf).Slice(18, 2)))
    if machine <> ElfMachineXtensa then
        failwithf "Expected ELF32 Xtensa (machine %d), got %d" ElfMachineXtensa machine

/// Read the ELF32 little-endian program headers. The ESP image's segments are
/// exactly the PT_LOAD segments, so this is the projection rather than a
/// flattened `objcopy -O binary`, which would lose the load addresses the ROM
/// loader needs.
let readLoadSegments (elf: byte array) : LoadSegment list =
    if elf.Length < 52 then failwith "Truncated ELF header"
    if elf.[0..3] <> [| 0x7Fuy; 0x45uy; 0x4Cuy; 0x46uy |] then failwith "Not an ELF file"
    if elf.[4] <> 1uy then failwith "Expected ELF32"
    if elf.[5] <> 1uy then failwith "Expected little-endian ELF"
    let u16 offset = int (BinaryPrimitives.ReadUInt16LittleEndian(ReadOnlySpan(elf).Slice(offset, 2)))
    let u32 offset = BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(elf).Slice(offset, 4))
    let programHeaderOffset = int (u32 28)
    let entrySize, count = u16 42, u16 44
    if entrySize < 32 then failwithf "Unexpected program header size %d" entrySize
    [ for i in 0 .. count - 1 do
        let header = programHeaderOffset + i * entrySize
        if header + 32 > elf.Length then failwith "Truncated program header table"
        if u32 header = ProgramTypeLoad then
            let fileOffset = int (u32 (header + 4))
            let virtualAddress = u32 (header + 8)
            let fileBytes = int (u32 (header + 16))
            let memoryBytes = int (u32 (header + 20))
            if fileOffset + fileBytes > elf.Length then
                failwithf "PT_LOAD at 0x%08X runs past the file" virtualAddress
            yield { VirtualAddress = virtualAddress
                    FileBytes = fileBytes
                    MemoryBytes = memoryBytes
                    FileOffset = fileOffset
                    Data = elf.[fileOffset .. fileOffset + fileBytes - 1] } ]

/// Check a linked ELF and its container against the declarations.
let verify (target: XtensaTarget) (elf: byte array) (image: byte array) (table: Map<string, uint32>) =
    let windows = XtensaLayout.windows target
    requireXtensaElf elf
    let segments = readLoadSegments elf

    if List.isEmpty segments then failwith "Linked image has no loadable segments"

    // Every segment must land in a window this image actually owns, and the
    // ROM loader cannot zero-fill, so a segment with a .bss tail must not be
    // handed to it as data.
    let inWindow (address: int64) (length: int64) =
        let within origin extent = address >= origin && address + length <= origin + extent
        within windows.IramOrigin windows.IramBytes || within windows.DramOrigin windows.DramBytes
    for segment in segments do
        if segment.FileBytes > 0 && not (inWindow (int64 segment.VirtualAddress) (int64 segment.FileBytes)) then
            failwithf "Loadable segment at 0x%08X (%d bytes) lies outside the declared IRAM and DRAM windows"
                segment.VirtualAddress segment.FileBytes
        if segment.MemoryBytes > segment.FileBytes && segment.FileBytes > 0 then
            failwithf "Segment at 0x%08X declares %d bytes in memory but %d in the file; the ROM loader does not zero-fill, so .bss must be a separate NOLOAD section that startup clears"
                segment.VirtualAddress segment.MemoryBytes segment.FileBytes

    // The vector block must exist, be the declared size, and sit where VECBASE
    // can point at it.
    let vectorSize = target.Vectors.Layout.Size
    let vectorAddress =
        segments
        |> List.tryFind (fun s -> int64 s.VirtualAddress = windows.IramOrigin)
        |> Option.map (fun s -> s.VirtualAddress)
        |> Option.defaultWith (fun () -> failwith "No loadable segment begins at the instruction window's origin, where the vector block belongs")
    if vectorAddress % uint32 target.Image.VectorAlignment <> 0u then
        failwithf "Vector block at 0x%08X is not %d-byte aligned; VECBASE ignores those low bits"
            vectorAddress target.Image.VectorAlignment

    // Every declared vector entry must resolve, and land inside the block.
    for KeyValue(name, handler) in target.VectorEntries do
        let offset =
            Map.tryFind name XtensaTarget.vectorOffsets
            |> Option.defaultWith (fun () -> failwithf "Unknown vector entry '%s'" name)
        let address =
            Map.tryFind handler table
            |> Option.defaultWith (fun () -> failwithf "Missing vector handler symbol: %s" handler)
        if address < vectorAddress || address >= vectorAddress + uint32 vectorSize then
            failwithf "Vector handler %s at 0x%08X lies outside the %d-byte vector block at 0x%08X"
                handler address vectorSize vectorAddress
        if address <> vectorAddress + uint32 offset then
            failwithf "Vector entry '%s' must be at VECBASE+0x%03X (0x%08X); %s is at 0x%08X"
                name offset (vectorAddress + uint32 offset) handler address

    // The ELF's entry point is what the ROM loader jumps to, so it must agree
    // with the image header the container carries.
    let elfEntry = BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(elf).Slice(24, 4))
    let entrySymbol =
        Map.tryFind target.Image.EntrySymbol table
        |> Option.defaultWith (fun () -> failwithf "Missing entry symbol: %s" target.Image.EntrySymbol)
    if elfEntry <> entrySymbol then
        failwithf "ELF entry 0x%08X disagrees with %s at 0x%08X" elfEntry target.Image.EntrySymbol entrySymbol

    let parsed = EspImage.verify image
    if parsed.EntryAddress <> elfEntry then
        failwithf "Image header entry 0x%08X disagrees with the ELF entry 0x%08X" parsed.EntryAddress elfEntry
    if parsed.ChipId <> target.Image.ChipId then
        failwithf "Image header chip id %d disagrees with the declared %d" parsed.ChipId target.Image.ChipId
    if int64 image.Length > target.FlashStore.Capacity then
        failwithf "Image is %d bytes; the declared flash store holds %d" image.Length target.FlashStore.Capacity
    parsed

let build llPath (ctx: BackEndContext) (target: XtensaTarget) =
    let core = target.Image
    if ctx.TargetPointerBits <> Some 32 then failwith "The Xtensa image target is 32-bit"
    let triple =
        ctx.TargetTripleOverride
        |> Option.defaultWith (fun () -> failwith "The Xtensa image target requires a declared triple")
    if not (triple.StartsWith("xtensa", StringComparison.Ordinal)) then
        failwithf "The Xtensa image backend was given the triple '%s'" triple
    if ctx.NativeLink <> NativeLinkOptions.Empty then
        failwith "The MCU image backend owns linking; remove hosted link overrides"
    if not (Set.isSubset ctx.ExternLibraries target.ProvidedLibraries) then
        failwithf "Unprovided MCU native libraries: %A" (Set.difference ctx.ExternLibraries target.ProvidedLibraries)

    let elfPath = Path.GetFullPath ctx.OutputPath
    let outDir = Path.GetDirectoryName elfPath
    let boot = Path.Combine(outDir, "boot")
    let artifact ext = Path.ChangeExtension(elfPath, ext)
    let evidencePath = artifact "build-evidence.json"
    // A failed attempt must not leave an older build marked eligible for deploy.
    if File.Exists evidencePath then File.Delete evidencePath

    XtensaLayout.generate target boot |> ignore

    let text = File.ReadAllText llPath
    if not (text.Contains(sprintf "target triple = \"%s\"" triple)) then
        failwithf "Missing pre-lowering target selection for %s" triple

    // The tools come from one directory when declared, so which toolchain ran
    // is recoverable from the build rather than inferred from PATH order.
    let tool name =
        match target.ToolDirectory with
        | Some directory -> Path.Combine(directory, name)
        | None -> name
    let run name args = Tools.run (tool name) args None |> ignore

    let optimized = artifact "ll"
    run "opt" [ "-S"; "-passes=default<O2>"; llPath; "-o"; optimized ]
    if Regex.IsMatch(File.ReadAllText optimized, @"\b(?:invoke|landingpad|resume)\b") then
        failwith "This MCU profile has no exception unwinder"

    // Select the part by CPU model when the toolchain has one, and by feature
    // flags when it does not: upstream LLVM carries 26 of the esp32s3 bundle's
    // 29 features but no esp32s3 CPU.
    let selection =
        match target.Cpu, target.Features with
        | Some cpu, _ -> [ "-mcpu=" + cpu ]
        | None, [] -> failwith "Declare either a CPU model or the -mattr features for this Xtensa part"
        | None, features -> [ "-mattr=" + String.Join(",", features |> List.map (fun f -> "+" + f)) ]

    let obj = artifact "o"
    run "llc" ([ "-mtriple=" + triple ] @ selection @ [ "-O=2"; "-filetype=obj"; optimized; "-o"; obj ])

    // llvm-mc assembles the owned startup and vector block. No clang, and no
    // GNU binutils: this target has no `as` of its own in the sovereign path.
    let startup = Path.Combine(boot, "startup.o")
    run "llvm-mc" ([ "-triple=" + triple ] @ selection @
                   [ "-filetype=obj"; "-I"; boot; target.StartupSource; "-o"; startup ])

    run "ld.lld" [ "--static"; "--fatal-warnings"; "--gc-sections"
                   "--entry=" + core.EntrySymbol
                   "-T"; Path.Combine(boot, "memory.ld")
                   "-Map=" + artifact "map"
                   startup; obj; "-o"; elfPath ]

    Tools.run (tool "llvm-objdump") [ "-d"; elfPath ] (Some (artifact "disassembly.txt")) |> ignore
    Tools.run (tool "llvm-readelf") [ "-h"; "-l"; "-S"; elfPath ] (Some (artifact "elf-report.txt")) |> ignore
    let undefined = Tools.run (tool "llvm-nm") [ "--undefined-only"; elfPath ] None
    if not (String.IsNullOrWhiteSpace undefined) then failwith ("Unresolved MCU symbols: " + undefined)
    let table = Tools.run (tool "llvm-nm") [ "-n"; elfPath ] (Some (artifact "symbols.txt")) |> symbols

    // The container is written here, from the ELF's own PT_LOAD segments.
    let elf = File.ReadAllBytes elfPath
    let loadSegments = readLoadSegments elf
    let header: EspImage.Header = {
        EntryAddress = BinaryPrimitives.ReadUInt32LittleEndian(ReadOnlySpan(elf).Slice(24, 4))
        ChipId = core.ChipId
        MinChipRevFull = core.MinChipRevFull
        MaxChipRevFull = core.MaxChipRevFull
        SpiMode = core.SpiMode
        SpiSpeed = core.SpiSpeed
        SpiSize = core.SpiSize
        HashAppended = core.HashAppended
    }
    let image =
        loadSegments
        |> List.filter (fun s -> s.FileBytes > 0)
        |> List.map (fun s -> { EspImage.LoadAddress = s.VirtualAddress; EspImage.Data = s.Data })
        |> EspImage.build header
    let binary = artifact "bin"
    File.WriteAllBytes(binary, image)

    let parsed = verify target elf image table

    Tools.writeJson evidencePath
        {| artifact = Path.GetFileName elfPath
           sha256 = Tools.sha256 elf
           binarySha256 = Tools.sha256 image
           binaryBytes = image.Length
           segmentCount = parsed.SegmentCount
           entryAddress = sprintf "0x%08X" parsed.EntryAddress
           chipId = parsed.ChipId
           vectorBytes = target.Vectors.Layout.Size
           stackBytes = core.StackBytes
           platform = target.PlatformId
           orchestrator = "Composer"
           unresolvedSymbols = ([||] : string array)
           toolchain = "llvm (opt/llc/llvm-mc/ld.lld); no clang in the lowering path"
           boardExecution = "not established by this build" |}
    printfn "ESP image verified: %d bytes, %d segments, %d-byte vector block; BAREWire layout and SRAM partition passed"
        image.Length parsed.SegmentCount (target.Vectors.Layout.Size)
    elfPath
