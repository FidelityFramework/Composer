/// Checks for the ROM-loaded Xtensa image path: the SRAM partition, the
/// generated linker script, and the ELF program-header reader that feeds the
/// image writer.
///
/// The central property is one the Cortex-M layout never has to think about:
/// the ESP32-S3's middle SRAM bank answers to BOTH CPU buses at different
/// addresses, so an instruction window and a data window that each start at
/// their own base silently share silicon. The partition must therefore account
/// for exactly the part's real SRAM -- 512 KB, once -- and the emitted script
/// must carry an assertion that catches a crossing at link time.
///
/// Nothing here needs hardware, a probe or an Xtensa toolchain. It is
/// arithmetic over declarations, a linker-script parse by the real ld.lld, and
/// program-header parsing checked against values llvm-readelf reports for a
/// real ELF32. Each external check is skipped, with a note, when its tool or
/// fixture is absent.
module XtensaLayoutTests

open System
open System.Diagnostics
open System.IO
open BackEnd.MCU
open Core.Types.Pipeline

/// The real ESP32-S3 banks, from TRM v1.8 Table 4.3-2 and the aliasing in
/// Table 15.3-1. SRAM1 is declared once, with capacity; its data-bus address
/// is a second base on the image descriptor, not a second space.
let private space name kind access baseAddress capacity : BAREWire.Platform.MemorySpace =
    { Name = name; Kind = kind; Base = Some baseAddress; Capacity = capacity
      Alignment = 16; Granularity = 1; Growth = BAREWire.Platform.Growth.Fixed
      Access = access; Notes = ""; MapKind = ""; Since = ""; Until = "" }

let private sram = BAREWire.Platform.MemoryKind.Sram
let private rx = BAREWire.Platform.Access.ReadExecute
let private rw = BAREWire.Platform.Access.ReadWrite

let private sram0 = space "sram0" sram rx 0x40370000L 32768L
let private sram1 = space "sram1" sram rx 0x40378000L 425984L
let private sram1DataBase = 0x3FC88000L
let private sram2 = space "sram2" sram rw 0x3FCF0000L 65536L
/// Where the ROM's own memory begins at handover: 0x3FCD7E00 on this part.
let private dataLimit = 0x3FCD7E00L
/// Data-bus SRAM above the limit, spoken for by the ROM and the data cache.
let private reservedAboveLimit = (0x3FCF0000L + 65536L) - dataLimit
let private flashStore =
    { space "flash-store" BAREWire.Platform.MemoryKind.PersistentStore rw 0L 8388608L with
        Alignment = 4096; Granularity = 4096 }

/// 1024 bytes of vector CODE, not a handler-address table.
let private vectors: BAREWire.Hardware.StructDescriptor = {
    Name = "XtensaLX7Vectors"
    Documentation = None
    Layout = { Size = 1024; Alignment = 4
               Fields = [| { Name = "entries"; Offset = 0; Repr = BAREWire.Hardware.Repr.U8; Count = 1024
                             Access = BAREWire.Hardware.AccessKind.ReadOnly
                             BitFields = [||]; Documentation = None } |] }
}

let private image instructionBytes : BAREWire.Hardware.XtensaImageDescriptor = {
    Sram0Space = "sram0"; Sram1Space = "sram1"; Sram1DataBase = sram1DataBase; Sram2Space = "sram2"
    DataLimit = dataLimit
    FlashStoreSpace = "flash-store"
    Sram1InstructionBytes = instructionBytes
    VectorLayout = "XtensaLX7Vectors"; VectorAlignment = 1024
    StackBytes = 8192
    EntrySymbol = "_start"
    PartNumber = "ESP32-S3-WROOM-1-N8"
    ChipId = EspImage.ChipIdEsp32S3
    MinChipRevFull = 0; MaxChipRevFull = 9999
    SpiMode = EspImage.SpiMode.Qio; SpiSpeed = 0xF; SpiSize = EspImage.FlashSize.Size8MB
    HashAppended = 1
}

let private target instructionBytes : XtensaTarget = {
    PlatformId = "CCC2026Badge"
    Image = image instructionBytes
    Vectors = vectors
    Sram0 = sram0; Sram1 = sram1; Sram1DataBase = sram1DataBase; DataLimit = dataLimit; Sram2 = sram2
    FlashStore = flashStore
    StartupSource = "boot/startup.S"
    ProvidedLibraries = Set.empty
    VectorEntries = Map.ofList [ "window", "WindowVectors"; "user", "UserExceptionVector" ]
    RecoveryDirectory = "recovery"
    ToolDirectory = None
    Cpu = None
    Features = [ "density"; "windowed" ]
}

/// The part's real SRAM: 32 KB + 416 KB + 64 KB.
let private totalSram = 32768L + 425984L + 65536L

let private which tool =
    (Environment.GetEnvironmentVariable "PATH" |> Option.ofObj |> Option.defaultValue "").Split(Path.PathSeparator)
    |> Array.tryPick (fun d ->
        let candidate = Path.Combine(d, tool)
        if File.Exists candidate then Some candidate else None)

let private run tool args =
    let start = ProcessStartInfo(tool, UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true)
    for a in (args: string list) do start.ArgumentList.Add a
    use child = new Process(StartInfo = start)
    child.Start() |> ignore
    let output = child.StandardOutput.ReadToEnd() + child.StandardError.ReadToEnd()
    child.WaitForExit()
    child.ExitCode, output

[<EntryPoint>]
let main _ =
    let mutable passed = 0
    let succeeds name f = f (); passed <- passed + 1; printfn "PASS %s" name
    let rejects name (fragment: string) f =
        let failure = try f () |> ignore; None with ex -> Some ex.Message
        match failure with
        | Some message when message.Contains fragment -> passed <- passed + 1; printfn "PASS %s" name
        | _ -> failwithf "%s expected '%s', got %A" name fragment failure

    // --- the partition must account for the real SRAM exactly once ---------
    // This is the whole point. 192 KB of the shared bank to the instruction
    // side leaves 224 KB of it to the data side.
    let w = XtensaLayout.windows (target 0x30000)
    succeeds "instruction window is SRAM0 plus its share of the shared bank" (fun () ->
        if w.IramOrigin <> 0x40370000L then failwithf "IRAM origin 0x%X" w.IramOrigin
        if w.IramBytes <> 0x8000L + 0x30000L then failwithf "IRAM bytes 0x%X" w.IramBytes)
    succeeds "data window starts after the instruction share, in the data view, and ends at the ROM's limit" (fun () ->
        if w.DramOrigin <> 0x3FC88000L + 0x30000L then failwithf "DRAM origin 0x%X" w.DramOrigin
        if w.DramBytes <> dataLimit - (0x3FC88000L + 0x30000L) then failwithf "DRAM bytes 0x%X" w.DramBytes)
    succeeds "the two windows plus the ROM's reserve sum to exactly the part's 512 KB of SRAM" (fun () ->
        let total = w.IramBytes + w.DramBytes + reservedAboveLimit
        if total <> totalSram then
            failwithf "Windows total %d bytes; the part has %d. The shared bank is being counted twice or lost." total totalSram)

    // The invariant must hold for every legal split, not just the chosen one.
    succeeds "the sum is 512 KB at every legal split of the shared bank" (fun () ->
        for share in 0 .. 1024 .. int (dataLimit - sram1DataBase) - 1024 do
            let v = XtensaLayout.windows (target share)
            if v.IramBytes + v.DramBytes + reservedAboveLimit <> totalSram then
                failwithf "Split 0x%X totals %d, not %d" share (v.IramBytes + v.DramBytes) totalSram
            // The data window must never begin inside the instruction share.
            let dataOffsetInBank = v.DramOrigin - 0x3FC88000L
            if dataOffsetInBank <> int64 share then
                failwithf "Split 0x%X puts the data window at bank offset 0x%X" share dataOffsetInBank)

    succeeds "giving the whole shared bank to instructions leaves only SRAM2 for data, if the limit allows it" (fun () ->
        let v = XtensaLayout.windows { target (int sram1.Capacity) with DataLimit = 0x3FD00000L }
        if v.DramBytes <> sram2.Capacity then failwithf "DRAM bytes 0x%X" v.DramBytes
        if v.IramBytes <> sram0.Capacity + sram1.Capacity then failwithf "IRAM bytes 0x%X" v.IramBytes)
    // Learned on the badge: the ROM hands over with its own stacks and data at
    // the top of the shared bank and the data cache holding half of SRAM2. A
    // stack placed up there ran until its first spill and then parked.
    rejects "an instruction share that reaches the ROM's data limit" "leaves no data window" (fun () ->
        XtensaLayout.windows (target 0x50000))
    rejects "a data limit beyond the data-bus SRAM" "beyond the data-bus SRAM" (fun () ->
        XtensaLayout.windows { target 0x30000 with DataLimit = 0x3FD00010L })
    succeeds "giving none of it to instructions leaves only SRAM0 for code" (fun () ->
        let v = XtensaLayout.windows (target 0)
        if v.IramBytes <> sram0.Capacity then failwithf "IRAM bytes 0x%X" v.IramBytes
        if v.DramOrigin <> 0x3FC88000L then failwithf "DRAM origin 0x%X" v.DramOrigin)

    // --- declarations the partition cannot be computed from ----------------
    rejects "a split larger than the shared bank" "outside the shared bank" (fun () ->
        XtensaLayout.windows (target (int sram1.Capacity + 1024)))
    rejects "a negative split" "outside the shared bank" (fun () ->
        XtensaLayout.windows (target -1024))
    rejects "a split that leaves the data side misaligned" "multiple of the" (fun () ->
        XtensaLayout.windows (target 0x30004))
    rejects "a data-bus base that does not abut the data-only bank" "does not abut" (fun () ->
        XtensaLayout.windows { target 0x30000 with Sram1DataBase = 0x3FC80000L })
    rejects "an instruction bank that does not abut the shared bank" "does not abut" (fun () ->
        XtensaLayout.windows { target 0x30000 with Sram0 = { sram0 with Capacity = 16384L } })
    rejects "a data-only bank that does not abut the shared bank" "does not abut" (fun () ->
        XtensaLayout.windows { target 0x30000 with Sram2 = { sram2 with Base = Some 0x3FCF8000L } })

    // --- the emitted script -----------------------------------------------
    let temporary = Path.Combine(Path.GetTempPath(), "composer-xtensa-layout-" + Guid.NewGuid().ToString("N"))
    try
        let boot = Path.Combine(temporary, "boot")
        XtensaLayout.generate (target 0x30000) boot |> ignore
        let script = File.ReadAllText(Path.Combine(boot, "memory.ld"))

        succeeds "script declares the computed windows" (fun () ->
            if not (script.Contains "IRAM (rx) : ORIGIN = 0x40370000, LENGTH = 0x38000") then failwith "IRAM region wrong"
            if not (script.Contains "DRAM (rw) : ORIGIN = 0x3FCB8000, LENGTH = 0x1FE00") then failwith "DRAM region wrong")
        succeeds "the stack tops out exactly at the ROM's data limit, never in SRAM2" (fun () ->
            if not (script.Contains "__stack_top = 0x3FCD7E00;") then failwith "Stack top is not the data limit"
            if not (script.Contains "__stack_bottom = 0x3FCD5E00;") then failwith "Stack bottom wrong for 8192 bytes")
        succeeds "read-only data is placed on the data bus, not with the code" (fun () ->
            let text = script.Substring(script.IndexOf ".text :", script.IndexOf "__iram_end" - script.IndexOf ".text :")
            let data = script.Substring(script.IndexOf ".data ORIGIN(DRAM)", script.IndexOf "__data_end" - script.IndexOf ".data ORIGIN(DRAM)")
            if text.Contains ".rodata" then failwith "rodata in the instruction window: halfword tables fault there"
            if not (data.Contains "*(.rodata .rodata.*)") then failwith "rodata not placed in DRAM")
        succeeds "script asserts instruction use cannot cross into the data side" (fun () ->
            if not (script.Contains "__shared_bank_data_split") then failwith "No split symbol"
            if not (script.Contains "crosses into the data side") then failwith "No crossing assertion")
        succeeds "script has no load-address split, because the ROM loader places .data" (fun () ->
            if script.Contains "AT>" || script.Contains "LOADADDR" then
                failwith "An all-SRAM ROM-loaded image needs no .data copy")
        succeeds "script keeps Xtensa literal pools" (fun () ->
            if not (script.Contains "*(.literal .literal.*)") then failwith "L32R literal pools would be dropped")
        // Learned from a real link, not from reading the manual: ld.lld rejected
        // every movi in the startup code with "relocation R_XTENSA_SLOT0_OP out
        // of range" because the literal pool was emitted after the text that
        // referenced it. L32R reaches BACKWARDS only. Presence is not enough --
        // order is the property that matters.
        succeeds "literal pools are placed before the text that references them" (fun () ->
            let literalAt = script.IndexOf "*(.literal .literal.*)"
            let textAt = script.IndexOf "*(.text .text.*)"
            if literalAt < 0 || textAt < 0 then failwith "Missing literal or text placement"
            if literalAt > textAt then
                failwith "Literal pool follows .text; every L32R in startup will fail to link")
        succeeds "layout.inc exports the vector facts to startup assembly" (fun () ->
            let inc = File.ReadAllText(Path.Combine(boot, "layout.inc"))
            if not (inc.Contains ".set VECTOR_BYTES, 1024") then failwith "No VECTOR_BYTES"
            if not (inc.Contains ".set VECTOR_ALIGNMENT, 1024") then failwith "No VECTOR_ALIGNMENT")
        succeeds "layout.json records the partition for review" (fun () ->
            let json = File.ReadAllText(Path.Combine(boot, "layout.json"))
            if not (json.Contains "sharedBankInstructionBytes") then failwith "Partition not recorded"
            if not (json.Contains "dataCopiedByRomLoader") then failwith "Loader responsibility not recorded")

        rejects "a stack that does not fit the data window" "Invalid Xtensa stack" (fun () ->
            let t = target 0x30000
            XtensaLayout.generate { t with Image = { t.Image with StackBytes = 0x50000 } } boot)
        rejects "a stack violating the call ABI's 16-byte alignment" "Invalid Xtensa stack" (fun () ->
            let t = target 0x30000
            XtensaLayout.generate { t with Image = { t.Image with StackBytes = 8200 } } boot)

        // Syntax-check the script with the real linker when it is available.
        // A malformed script is reported as a parse error against the script
        // file; placement errors against a foreign-architecture object are not
        // what is being checked here.
        match which "ld.lld" with
        | None -> printfn "SKIP ld.lld script parse (ld.lld not found)"
        | Some lld ->
            XtensaLayout.generate (target 0x30000) boot |> ignore
            let scriptPath = Path.Combine(boot, "memory.ld")
            let _, output = run lld [ "-T"; scriptPath; "-o"; Path.Combine(temporary, "out.elf") ]
            let parseErrors =
                output.Split('\n')
                |> Array.filter (fun line ->
                    let l = line.ToLowerInvariant()
                    l.Contains "memory.ld" &&
                    (l.Contains "syntax" || l.Contains "unknown" || l.Contains "expected"
                     || l.Contains "unexpected" || l.Contains "malformed"))
            if parseErrors.Length > 0 then
                failwithf "ld.lld could not parse the generated script:\n%s" (String.Join("\n", parseErrors))
            passed <- passed + 1
            printfn "PASS ld.lld parses the generated script"
    finally
        if Directory.Exists temporary then Directory.Delete(temporary, true)

    // --- the ELF PT_LOAD reader -------------------------------------------
    // Exercised against a real linked ELF32 LE file and checked against values
    // independently reported by llvm-readelf. The reader is architecture-neutral
    // on purpose, so the only ELF32 available here -- the Cortex-M HelloBlinky
    // image -- is a valid subject for it.
    let helloBlinky = "/home/hhh/repos/MCU/Renesas/EK-RA6M5/HelloBlinky/targets/HelloBlinky.elf"
    if not (File.Exists helloBlinky) then
        printfn "SKIP ELF PT_LOAD reader (no ELF32 fixture at %s)" helloBlinky
    else
        let elf = File.ReadAllBytes helloBlinky
        let loads = XtensaImage.readLoadSegments elf

        succeeds "reader finds every PT_LOAD segment" (fun () ->
            if loads.Length <> 3 then failwithf "Expected 3 PT_LOAD segments, got %d" loads.Length)
        succeeds "reader agrees with llvm-readelf on addresses and sizes" (fun () ->
            // llvm-readelf -l reports, for this image:
            //   LOAD 0x010000 0x00000000 filesz 0x001c0 memsz 0x001c0
            //   LOAD 0x0101c0 0x000001c0 filesz 0x0076e memsz 0x0076e
            //   LOAD 0x020000 0x20000000 filesz 0x00000 memsz 0x80000
            let expected =
                [ 0x00000000u, 0x1c0, 0x1c0, 0x010000
                  0x000001c0u, 0x76e, 0x76e, 0x0101c0
                  0x20000000u, 0x00000, 0x80000, 0x020000 ]
            let actual = loads |> List.map (fun s -> s.VirtualAddress, s.FileBytes, s.MemoryBytes, s.FileOffset)
            if actual <> expected then failwithf "Program headers read as %A" actual)
        succeeds "reader extracts exactly FileBytes of data per segment" (fun () ->
            for s in loads do
                if s.Data.Length <> s.FileBytes then
                    failwithf "Segment 0x%08X carries %d bytes for a declared %d" s.VirtualAddress s.Data.Length s.FileBytes)
        succeeds "segment data matches the file at its stated offset" (fun () ->
            for s in loads do
                if s.FileBytes > 0 then
                    let fromFile = elf.[s.FileOffset .. s.FileOffset + s.FileBytes - 1]
                    if fromFile <> s.Data then failwithf "Segment 0x%08X data differs from the file" s.VirtualAddress)
        succeeds "a NOLOAD .bss segment carries no file bytes and is excluded from an image" (fun () ->
            // The third segment is memsz 0x80000 with filesz 0: RAM the linker
            // reserved, which the ROM loader must not be asked to copy.
            match loads |> List.filter (fun s -> s.FileBytes = 0) with
            | [ bss ] ->
                if bss.MemoryBytes <> 0x80000 then failwithf "Expected a 0x80000-byte reservation, got 0x%X" bss.MemoryBytes
                let carried = loads |> List.filter (fun s -> s.FileBytes > 0)
                if carried.Length <> 2 then failwithf "Expected 2 carried segments, got %d" carried.Length
            | other -> failwithf "Expected exactly one zero-filesz segment, got %d" other.Length)
        succeeds "the machine check is separate and rejects a non-Xtensa ELF" (fun () ->
            let failure = try XtensaImage.requireXtensaElf elf; None with ex -> Some ex.Message
            match failure with
            | Some message when message.Contains "Xtensa" -> ()
            | other -> failwithf "Expected an Xtensa machine rejection, got %A" other)

        rejects "not an ELF file" "Not an ELF" (fun () ->
            XtensaImage.readLoadSegments (Array.append [| 0x7Fuy; 0x00uy; 0x00uy; 0x00uy |] (Array.zeroCreate 60)))
        rejects "truncated ELF header" "Truncated ELF" (fun () ->
            XtensaImage.readLoadSegments [| 0x7Fuy; 0x45uy; 0x4Cuy; 0x46uy |])
        rejects "ELF64 is not this path" "Expected ELF32" (fun () ->
            let bad = Array.copy elf in bad.[4] <- 2uy
            XtensaImage.readLoadSegments bad)
        rejects "big-endian ELF is not this path" "little-endian" (fun () ->
            let bad = Array.copy elf in bad.[5] <- 2uy
            XtensaImage.readLoadSegments bad)

        // An ESP image built from the carried segments must be well-formed --
        // the composition of the two tested pieces.
        succeeds "carried PT_LOAD segments compose into a valid ESP image" (fun () ->
            let carried =
                loads
                |> List.filter (fun s -> s.FileBytes > 0)
                |> List.map (fun s -> { EspImage.LoadAddress = s.VirtualAddress; EspImage.Data = s.Data })
            let header: EspImage.Header =
                { EntryAddress = 0x1C1u; ChipId = EspImage.ChipIdEsp32S3
                  MinChipRevFull = 0; MaxChipRevFull = 9999
                  SpiMode = EspImage.SpiMode.Qio; SpiSpeed = 0xF
                  SpiSize = EspImage.FlashSize.Size8MB; HashAppended = true }
            let built = EspImage.build header carried
            let p = EspImage.verify built
            if p.SegmentCount <> 2 then failwithf "Expected 2 segments, got %d" p.SegmentCount)

    printfn "%d Xtensa backend checks passed; no hardware, probe or Xtensa toolchain used" passed
    0
