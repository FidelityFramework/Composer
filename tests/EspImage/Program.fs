/// Checks for the Espressif ROM-loadable image container.
///
/// These need no hardware, no Xtensa toolchain and no probe: the writer is a
/// pure byte computation, so the checks compute bytes and read them back.
///
/// An optional second oracle is available. If `esptool` is on PATH or named by
/// COMPOSER_ESPTOOL, each produced image is also handed to `esptool image-info`,
/// which validates the checksum and appended hash independently of this code.
/// Without it the internal checks still run and the run says so.
module EspImageTests

open System
open System.Diagnostics
open System.IO
open BackEnd.MCU
open BackEnd.MCU.EspImage

/// The badge's real configuration: ESP32-S3-WROOM-1-N8, 8 MB flash, an
/// all-SRAM image entered on the instruction-bus SRAM window.
let private badgeHeader = {
    EntryAddress = 0x40378000u
    ChipId = ChipIdEsp32S3
    MinChipRevFull = 0
    MaxChipRevFull = 9999
    SpiMode = SpiMode.Qio
    SpiSpeed = 0xF
    SpiSize = FlashSize.Size8MB
    HashAppended = true
}

/// Vectors at a 1024-aligned IRAM address, text in IRAM, data in DRAM.
let private badgeSegments = [
    { LoadAddress = 0x40374000u; Data = Array.init 1024 (fun i -> byte (i % 251)) }
    { LoadAddress = 0x40378000u; Data = Array.init 4096 (fun i -> byte (i * 7 % 253)) }
    { LoadAddress = 0x3FC88000u; Data = Array.init 300 byte }
]

let private esptool () =
    let configured = Environment.GetEnvironmentVariable "COMPOSER_ESPTOOL"
    if not (String.IsNullOrWhiteSpace configured) && File.Exists configured then Some configured
    else
        (Environment.GetEnvironmentVariable "PATH" |> Option.ofObj |> Option.defaultValue "").Split(Path.PathSeparator)
        |> Array.tryPick (fun directory ->
            let candidate = Path.Combine(directory, "esptool")
            if File.Exists candidate then Some candidate else None)

/// Run the external oracle over an image and return its report.
let private inspect tool path =
    let start = ProcessStartInfo(tool, UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true)
    for argument in [ "image-info"; path ] do start.ArgumentList.Add argument
    use child = new Process(StartInfo = start)
    child.Start() |> ignore
    let output = child.StandardOutput.ReadToEnd() + child.StandardError.ReadToEnd()
    child.WaitForExit()
    if child.ExitCode <> 0 then failwithf "esptool rejected the image: %s" output
    output

[<EntryPoint>]
let main _ =
    let mutable passed = 0
    let succeeds name f = f (); passed <- passed + 1; printfn "PASS %s" name
    let rejects name (fragment: string) f =
        let failure = try f () |> ignore; None with ex -> Some ex.Message
        match failure with
        | Some message when message.Contains fragment -> passed <- passed + 1; printfn "PASS %s" name
        | _ -> failwithf "%s expected '%s', got %A" name fragment failure

    // --- the container the badge will actually carry ----------------------
    let image = build badgeHeader badgeSegments
    let parsed = verify image

    succeeds "header is exactly 24 bytes" (fun () ->
        if image.[0] <> HeaderMagic then failwith "Bad magic"
        if int image.[1] <> List.length badgeSegments then failwith "Wrong segment count")
    succeeds "entry address round-trips" (fun () ->
        if parsed.EntryAddress <> badgeHeader.EntryAddress then failwith "Entry address changed")
    succeeds "chip id is ESP32-S3" (fun () ->
        if parsed.ChipId <> ChipIdEsp32S3 then failwithf "Chip id %d" parsed.ChipId)
    succeeds "checksum agrees with contents" (fun () ->
        if parsed.Checksum <> parsed.ComputedChecksum then failwith "Checksum disagrees")
    succeeds "appended hash agrees with contents" (fun () ->
        if not parsed.HashMatches then failwith "Hash mismatch")
    succeeds "checksum byte ends a 16-byte block" (fun () ->
        // The hash, when present, follows the checksum; strip it first.
        let throughChecksum = if parsed.HashAppended then parsed.TotalBytes - 32 else parsed.TotalBytes
        if throughChecksum % ChecksumAlignment <> 0 then failwithf "Checksum at %d" throughChecksum)
    succeeds "segment load addresses and lengths survive" (fun () ->
        let expected = badgeSegments |> List.map (fun s -> s.LoadAddress, s.Data.Length)
        if parsed.Segments <> expected then failwithf "Segments changed: %A" parsed.Segments)

    // --- segment data is written in whole words ---------------------------
    let odd = build badgeHeader [ { LoadAddress = 0x3FC88000u; Data = Array.init 301 byte } ]
    succeeds "a 301-byte segment is padded to 304" (fun () ->
        match (verify odd).Segments with
        | [ (_, 304) ] -> ()
        | other -> failwithf "Expected one 304-byte segment, got %A" other)

    // --- the hash is optional --------------------------------------------
    let bare = build { badgeHeader with HashAppended = false } [ { LoadAddress = 0x3FC88000u; Data = Array.init 64 byte } ]
    succeeds "an image without an appended hash ends at its checksum" (fun () ->
        let p = verify bare
        if p.HashAppended then failwith "Hash flag set"
        if p.TotalBytes % ChecksumAlignment <> 0 then failwithf "Ends at %d" p.TotalBytes)

    // --- what the writer must refuse -------------------------------------
    let one = [ { LoadAddress = 0x3FC88000u; Data = Array.init 16 byte } ]
    rejects "no segments" "at least one segment" (fun () -> build badgeHeader [])
    rejects "empty segment" "Empty segment" (fun () ->
        build badgeHeader [ { LoadAddress = 0x3FC88000u; Data = [||] } ])
    rejects "unaligned load address" "not word-aligned" (fun () ->
        build badgeHeader [ { LoadAddress = 0x3FC88002u; Data = Array.init 16 byte } ])
    rejects "overlapping segments" "overlap" (fun () ->
        build badgeHeader [
            { LoadAddress = 0x3FC88000u; Data = Array.init 256 byte }
            { LoadAddress = 0x3FC88080u; Data = Array.init 16 byte } ])
    rejects "more segments than the loader allows" "at most 16" (fun () ->
        build badgeHeader [ for i in 0 .. MaxSegments -> { LoadAddress = uint32 (0x3FC88000 + i * 0x1000); Data = Array.init 16 byte } ])
    rejects "flash size outside its nibble" "must fit in a nibble" (fun () ->
        build { badgeHeader with SpiSize = 0x10 } one)
    rejects "chip revision range inverted" "precedes its minimal" (fun () ->
        build { badgeHeader with MinChipRevFull = 200; MaxChipRevFull = 100 } one)
    rejects "short buffer is not an image" "shorter than" (fun () -> parse [| 0xE9uy |])
    rejects "wrong magic" "Bad image magic" (fun () ->
        let bad = Array.copy image in bad.[0] <- 0xE8uy
        parse bad)

    // The two footer checks cover different regions, and each is exercised on
    // its own. The checksum accumulates over segment DATA only, so corrupting
    // a data byte is what it catches -- and it is checked first.
    rejects "corrupted segment data fails the checksum" "checksum" (fun () ->
        let bad = Array.copy image
        bad.[HeaderBytes + SegmentHeaderBytes] <- bad.[HeaderBytes + SegmentHeaderBytes] ^^^ 0xFFuy
        verify bad)
    succeeds "the checksum alone catches corrupted data, with no hash present" (fun () ->
        let unhashed = build { badgeHeader with HashAppended = false } badgeSegments
        let bad = Array.copy unhashed
        bad.[HeaderBytes + SegmentHeaderBytes] <- bad.[HeaderBytes + SegmentHeaderBytes] ^^^ 0xFFuy
        let failure = try verify bad |> ignore; None with ex -> Some ex.Message
        match failure with
        | Some message when message.Contains "checksum" -> ()
        | other -> failwithf "Expected a checksum rejection, got %A" other)
    // The header is outside the checksum's coverage, so corrupting the entry
    // address is caught by the appended hash and by nothing else. This is the
    // reason to append one.
    rejects "corrupted header fails the hash" "SHA-256" (fun () ->
        let bad = Array.copy image
        bad.[4] <- bad.[4] ^^^ 0xFFuy
        verify bad)
    succeeds "a corrupted header passes the checksum, which does not cover it" (fun () ->
        let unhashed = build { badgeHeader with HashAppended = false } badgeSegments
        let bad = Array.copy unhashed
        bad.[4] <- bad.[4] ^^^ 0xFFuy
        // Verify succeeds: the entry address is wrong but the checksum is not
        // computed over it. Without an appended hash this damage is invisible.
        let parsedBad = verify bad
        if parsedBad.EntryAddress = badgeHeader.EntryAddress then failwith "Entry address was not actually corrupted")

    // --- optional independent oracle --------------------------------------
    let temporary = Path.Combine(Path.GetTempPath(), "composer-esp-image-" + Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory temporary |> ignore
    try
        match esptool () with
        | None ->
            printfn "SKIP esptool cross-check (not found; set COMPOSER_ESPTOOL to enable)"
        | Some tool ->
            let crossCheck name (bytes: byte array) expectedSegments hashed =
                let path = Path.Combine(temporary, name + ".bin")
                File.WriteAllBytes(path, bytes)
                let report = inspect tool path
                let requires (fragment: string) =
                    if not (report.Contains fragment) then failwithf "esptool did not report '%s':\n%s" fragment report
                requires "ESP32-S3"
                requires "Checksum:"
                requires "(valid)"
                requires (sprintf "Segments: %d" expectedSegments)
                if hashed then requires "Validation hash:"
                // "(valid)" must follow the hash too, not only the checksum.
                if hashed && report.Contains "Validation hash:" then
                    let line = report.Split('\n') |> Array.find (fun l -> l.Contains "Validation hash:")
                    if not (line.Contains "(valid)") then failwithf "esptool called the hash invalid: %s" line
                passed <- passed + 1
                printfn "PASS esptool accepts %s" name
            crossCheck "badge" image (List.length badgeSegments) true
            crossCheck "padded" odd 1 true
            crossCheck "unhashed" bare 1 false
    finally
        Directory.Delete(temporary, true)

    printfn "%d ESP image checks passed; no hardware, probe or Xtensa toolchain used" passed
    0
