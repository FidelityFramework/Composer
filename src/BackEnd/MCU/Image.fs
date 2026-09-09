module BackEnd.MCU.Image

open System
open System.IO
open System.Text.RegularExpressions
open System.Buffers.Binary
open Core.Types.Pipeline

let symbols text =
    Regex.Matches(text, @"(?m)^([0-9a-fA-F]+)\s+\w\s+(\S+)$")
    |> Seq.map (fun m -> m.Groups.[2].Value, Convert.ToUInt32(m.Groups.[1].Value, 16)) |> Map.ofSeq

let verify (target: EmbeddedTarget) (elf: byte array) (image: byte array) table =
    let word offset = BinaryPrimitives.ReadUInt32LittleEndian(image.AsSpan(offset, 4))
    if image.Length < target.Vectors.Layout.Size || int64 image.Length > target.Flash.Capacity then failwith "Invalid flash image extent"
    if elf.Length < 52 || elf.[0..5] <> [|127uy;69uy;76uy;70uy;1uy;1uy|] || BinaryPrimitives.ReadUInt16LittleEndian(elf.AsSpan(18,2)) <> 40us then
        failwith "Expected little-endian ELF32 ARM image"
    if word 0 <> uint32 (Layout.origin target.Ram + target.Ram.Capacity) || word 0 % 8u <> 0u then failwith "Bad initial MSP"
    if BinaryPrimitives.ReadUInt32LittleEndian(elf.AsSpan(24,4)) <> word 4 then failwith "ELF entry and reset vector disagree"
    for KeyValue(slot, name) in target.VectorHandlers do
        let address = Map.tryFind name table |> Option.defaultWith (fun () -> failwith ("Missing vector handler: " + name))
        if word (4 * slot) <> (address ||| 1u) then failwithf "Wrong vector %d: %s" slot name
    for slot in 1 .. target.Vectors.Layout.Size / 4 - 1 do
        let value = word (4 * slot)
        if List.contains slot [8;9;10;13] then
            if value <> 0u then failwithf "Reserved vector %d must be zero" slot
        elif value &&& 1u = 0u || int64 (value &&& ~~~1u) < Layout.origin target.Flash || int64 (value &&& ~~~1u) >= Layout.origin target.Flash + int64 image.Length then
            failwithf "Invalid Thumb vector %d" slot

let build llPath (ctx: BackEndContext) (target: EmbeddedTarget) =
    if ctx.TargetTripleOverride <> Some "thumbv8m.main-none-eabi" || ctx.TargetPointerBits <> Some 32 || ctx.TargetCpu <> Some "cortex-m33" then
        failwith "Cortex-M image target must agree with the declared triple, CPU and 32-bit pointers"
    if ctx.NativeLink <> NativeLinkOptions.Empty then failwith "The MCU image backend owns linking; remove hosted link overrides"
    if not (Set.isSubset ctx.ExternLibraries target.ProvidedLibraries) then
        failwithf "Unprovided MCU native libraries: %A" (Set.difference ctx.ExternLibraries target.ProvidedLibraries)
    let elfPath = Path.GetFullPath ctx.OutputPath
    let outDir = Path.GetDirectoryName elfPath
    let boot = Path.Combine(outDir, "boot")
    let artifact ext = Path.ChangeExtension(elfPath, ext)
    let evidencePath = artifact "build-evidence.json"
    // A failed attempt must not leave an older build marked eligible for deploy.
    if File.Exists evidencePath then File.Delete evidencePath
    Layout.generate target boot
    let text = File.ReadAllText llPath
    let signature = Regex.Match(text, @"define i32 @main\(([^)]*)\)")
    if not signature.Success || (signature.Groups.[1].Value.Split(',') |> Array.map (fun p -> p.Trim().Split(' ').[0])) <> [|"ptr";"ptr";"i32";"i32";"i32"|] then
        failwith "Clef entry ABI changed: review the owned startup adapter before linking"
    if not (text.Contains("target triple = \"thumbv8m.main-none-eabi\"")) then failwith "Missing pre-lowering MCU target selection"
    let run tool args = Tools.run tool args None |> ignore
    let binutils = Tools.armToolDirectory target.ToolDirectory
    let tool name = Path.Combine(binutils, "arm-none-eabi-" + name)
    let optimized = artifact "ll"
    run "opt" ["-S"; "-passes=default<O2>"; llPath; "-o"; optimized]
    if Regex.IsMatch(File.ReadAllText optimized, @"\b(?:invoke|landingpad|resume)\b") then failwith "This MCU profile has no exception unwinder"
    let obj = artifact "o"
    run "llc" ["-mtriple=thumbv8m.main-none-eabi"; "-mcpu=cortex-m33"; "-float-abi=soft"; "-O=2"; "-filetype=obj"; optimized; "-o"; obj]
    let startup = Path.Combine(boot, "startup.o")
    run (tool "as") ["-mcpu=cortex-m33"; "-mthumb"; "-mfloat-abi=soft"; "-I"; boot; target.StartupSource; "-o"; startup]
    run "ld.lld" ["--static"; "--fatal-warnings"; "--gc-sections"; "--entry=" + target.Image.EntrySymbol; "-T"; Path.Combine(boot,"memory.ld"); "-Map=" + artifact "map"; startup; obj; "-o"; elfPath]
    let binary = artifact "bin"
    run (tool "objcopy") ["-O"; "binary"; elfPath; binary]
    Tools.run (tool "objdump") ["-d"; elfPath] (Some (artifact "disassembly.txt")) |> ignore
    Tools.run (tool "readelf") ["-h";"-l";"-S";"-A";elfPath] (Some (artifact "elf-report.txt")) |> ignore
    let undefined = Tools.run (tool "nm") ["--undefined-only";elfPath] None
    if not (String.IsNullOrWhiteSpace undefined) then failwith ("Unresolved MCU symbols: " + undefined)
    let table = Tools.run (tool "nm") ["-n";elfPath] (Some (artifact "symbols.txt")) |> symbols
    let image, elf = File.ReadAllBytes binary, File.ReadAllBytes elfPath
    verify target elf image table
    Tools.writeJson evidencePath
        {| artifact = Path.GetFileName elfPath; sha256 = Tools.sha256 elf; binarySha256 = Tools.sha256 image
           binaryBytes = image.Length; vectorCount = target.Vectors.Layout.Size / 4; stackBytes = target.Image.StackBytes
           platform = target.PlatformId; orchestrator = "Composer"; unresolvedSymbols = ([||] : string array)
           boardExecution = "not established by this build" |}
    printfn "MCU image verified: %d bytes, %d vectors; BAREWire layout and entry ABI passed" image.Length (target.Vectors.Layout.Size / 4)
    elfPath
