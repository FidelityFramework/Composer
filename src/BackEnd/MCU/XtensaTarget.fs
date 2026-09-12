/// Project inputs and BAREWire declarations for the ROM-loaded Xtensa image path.
///
/// The Cortex-M resolver in Target.fs insists on flash at address zero and on a
/// vector table of contiguous U32 handler addresses. Both are false here, so
/// this is a separate resolver rather than a relaxation of that one -- a
/// relaxation would weaken the Cortex-M checks to accommodate a target that
/// does not share its shape.
module BackEnd.MCU.XtensaTarget

open System
open System.IO
open System.Text.RegularExpressions
open Fidelity.Data.TOML
open BAREWire.Hardware
open BAREWire.Platform
open Core.Types.Pipeline
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution

let private required label = Option.defaultWith (fun () -> failwith ("Missing or malformed " + label))

let private symbol (s: string) =
    if not (Regex.IsMatch(s, "^[A-Za-z_][A-Za-z0-9_]*$")) then failwith ("Invalid native symbol: " + s)
    s

/// The ten named entries of an Xtensa vector block, with their offsets from
/// VECBASE. Names rather than slot indices: these are not interchangeable
/// positions in an array, and "slot 15" has no meaning on this core.
let vectorOffsets =
    Map.ofList [
        "window", 0x000          // overflow/underflow; 6 x 64-byte entries
        "level2", 0x180
        "level3", 0x1C0
        "level4", 0x200
        "level5", 0x240
        "debug", 0x280           // level 6
        "nmi", 0x2C0             // level 7
        "kernel", 0x300
        "user", 0x340            // level-1 interrupts arrive here too
        "double", 0x3C0
    ]

/// The entries an image must supply. The window vectors are not optional: a
/// call that exhausts the register window traps into them, which ordinary
/// nested calls do. The user vector is where level-1 interrupts and genuine
/// exceptions both arrive, so an image without one cannot survive either.
let requiredVectorEntries = [ "window"; "user" ]

/// True when the selected platform's core is an Xtensa target. Takes the
/// graph's DeclaredCore, which is what platform resolution yields.
let isXtensa (core: DeclaredCore option) =
    match core with
    | Some c -> c.Arch.StartsWith("xtensa", StringComparison.Ordinal)
    | None -> false

let resolve (projectPath: string) (graph: SemanticGraph) : XtensaTarget =
    let projectPath = Path.GetFullPath projectPath
    let projectDir = Path.GetDirectoryName projectPath
    let doc = Toml.parse (File.ReadAllText projectPath) |> function Ok d -> d | Error e -> failwith e
    let str key = Toml.getString key doc |> required key
    let strings key =
        match Toml.getValue key doc with
        | None -> []
        | Some (TomlValue.Array xs) ->
            xs |> List.map (function TomlValue.String s when s <> "" -> s | _ -> failwith ("Expected strings: " + key))
        | _ -> failwith ("Expected array: " + key)
    let onProject p = Path.GetFullPath(p, projectDir)

    let platform = resolve graph |> required "BAREWire PlatformDescription"
    let core = platform.Core |> required "platform core"
    if not (isXtensa (Some core)) then
        failwithf "The Xtensa image backend was selected for architecture '%s'" core.Arch
    // The declared core states its widths; this image path is 32-bit only.
    let pointerBits =
        core.Widths |> List.tryFind (fun w -> w.Name = "Pointer") |> Option.map (fun w -> w.Bits)
        |> required "declared Pointer width"
    if pointerBits <> 32 then failwithf "This Xtensa image path is 32-bit; the core declares %d-bit pointers" pointerBits

    // One XtensaImageDescriptor, found the way Target.fs finds its Cortex-M twin.
    let declarations =
        graph.Nodes |> Map.toList |> List.choose (fun (_, node) ->
            match node.Kind with
            | SemanticKind.Binding _ ->
                node.Children |> List.tryLast |> Option.bind (recordOf graph)
                |> Option.bind (fun (record, fields) ->
                    match record.Type with
                    | NativeType.TApp (tc, _) when tc.Name.Split('.') |> Array.last = "XtensaImageDescriptor" -> Some fields
                    | _ -> None)
            | _ -> None)
    let fields = match declarations with [one] -> one | _ -> failwith "Declare exactly one BAREWire XtensaImageDescriptor"
    let field name reader =
        fields |> List.tryFind (fst >> (=) name) |> Option.bind (snd >> reader graph)
        |> required ("XtensaImageDescriptor." + name)
    let image: XtensaImageDescriptor = {
        Sram0Space = field "Sram0Space" stringOf
        Sram1Space = field "Sram1Space" stringOf
        Sram1DataSpace = field "Sram1DataSpace" stringOf
        Sram2Space = field "Sram2Space" stringOf
        FlashStoreSpace = field "FlashStoreSpace" stringOf
        Sram1InstructionBytes = int (field "Sram1InstructionBytes" int64Of)
        VectorLayout = field "VectorLayout" stringOf
        VectorAlignment = int (field "VectorAlignment" int64Of)
        StackBytes = int (field "StackBytes" int64Of)
        EntrySymbol = field "EntrySymbol" stringOf |> symbol
        PartNumber = field "PartNumber" stringOf
        ChipId = int (field "ChipId" int64Of)
        MinChipRevFull = int (field "MinChipRevFull" int64Of)
        MaxChipRevFull = int (field "MaxChipRevFull" int64Of)
        SpiMode = int (field "SpiMode" int64Of)
        SpiSpeed = int (field "SpiSpeed" int64Of)
        SpiSize = int (field "SpiSize" int64Of)
        HashAppended = field "HashAppended" int64Of <> 0L
    }

    let spaces: MemorySpace array = platform.Spaces |> List.map memorySpace |> List.toArray
    // Same in-process projection check the Cortex-M path runs: BAREWire's own
    // alignment and overlap rules over the declared physical spaces.
    let memoryProjection: PlatformDescription = {
        Id = platform.Id; DisplayName = platform.Id; Substrate = "mcu"; Core = None
        Spaces = spaces; Surfaces = [||]; Buffers = [||]; Transports = [||]; Notes = [||]; Limits = [||]
        Lifecycle = { Clocks = [||]; Resets = [||]; Entry = image.EntrySymbol; Teardown = ""; Persistence = Persistence.Volatile }
    }
    let findings = Check.run memoryProjection
    if findings.Length <> 0 then failwithf "BAREWire memory declaration: %A" findings

    let space name = spaces |> Array.tryFind (fun s -> s.Name = name) |> required ("memory space " + name)
    let sram0, sram1, sram1Data, sram2 = space image.Sram0Space, space image.Sram1Space, space image.Sram1DataSpace, space image.Sram2Space
    let flashStore = space image.FlashStoreSpace

    // Bus views carry different access because the buses differ: code is
    // fetched through one window and stored through the other.
    if sram0.Kind <> MemoryKind.Sram || sram1.Kind <> MemoryKind.Sram || sram2.Kind <> MemoryKind.Sram then
        failwith "The Xtensa image requires three declared SRAM banks"
    if sram0.Access <> Access.ReadExecute || sram1.Access <> Access.ReadExecute then
        failwith "Instruction-bus SRAM windows must declare read/execute access"
    if sram1Data.Access <> Access.ReadWrite || sram2.Access <> Access.ReadWrite then
        failwith "Data-bus SRAM windows must declare read/write access"
    // The ROM loader reads the image at offset 0 of a store, not a mapped space.
    if flashStore.Base <> Some 0L then
        failwith "The flash store is addressed by offset; its declared base must be 0"
    for s in [ sram0; sram1; sram1Data; sram2 ] do
        let origin = s.Base |> required ("base of " + s.Name)
        if origin < 0L || origin + s.Capacity > 0x100000000L then
            failwithf "%s exceeds the 32-bit address extent" s.Name

    // The vector block is code: U8[n], not an array of handler addresses.
    let layouts = readDescriptors graph
    let layout =
        layouts.Layouts |> List.filter (fun l -> l.Name = image.VectorLayout)
        |> function [one] -> one | _ -> failwith "VectorLayout must name exactly one BAREWire StructDescriptor"
    let vectors: StructDescriptor = {
        Name = layout.Name; Documentation = None
        Layout = { Size = layout.Size |> required "vector size"
                   Alignment = layout.Alignment |> required "vector natural alignment"
                   Fields = layout.PhysicalFields |> List.map (fun f -> {
                       Name = f.Name; Repr = f.Repr; Count = f.Count; Offset = f.Offset
                       Access = AccessKind.ReadOnly; BitFields = [||]; Documentation = None }) |> List.toArray }
    }
    match vectors.Layout.Fields with
    | [| f |] when f.Repr = Repr.U8 && f.Offset = 0 && f.Count = 1024 && vectors.Layout.Size = 1024 -> ()
    | _ -> failwith "An Xtensa vector block is one contiguous U8[1024] of code, not a handler-address table"
    // VECBASE ignores its low bits, so the block's alignment is not a choice.
    if image.VectorAlignment <> 1024 then failwith "VECBASE placement requires 1024-byte alignment"

    let startup = str "embedded.startup" |> onProject
    if Path.GetExtension(startup) <> ".S" || not (File.Exists startup) then
        failwith "embedded.startup must name an owned .S assembly source"

    // Named vector entries, validated against the block's fixed offsets.
    let entries =
        Toml.getTable "embedded.vector_entries" doc |> required "embedded.vector_entries"
        |> Map.toList
        |> List.map (fun (name, value) ->
            if not (Map.containsKey name vectorOffsets) then
                failwithf "Unknown Xtensa vector entry '%s'; expected one of %s"
                    name (vectorOffsets |> Map.toList |> List.map fst |> String.concat ", ")
            name, (match value with TomlValue.String s -> symbol s | _ -> failwith "Vector entry must be a symbol"))
        |> Map.ofList
    for name in requiredVectorEntries do
        if not (Map.containsKey name entries) then
            failwithf "The '%s' vector entry is mandatory on this core" name
    // The reset path is the image entry symbol, which the ROM loader jumps to
    // directly; it is not a vector entry, unlike Cortex-M's slot 1.
    if entries |> Map.exists (fun _ handler -> handler = image.EntrySymbol) then
        failwith "The entry symbol is reached by the ROM loader's jump, not through a vector entry"

    { PlatformId = platform.Id; Image = image; Vectors = vectors
      Sram0 = sram0; Sram1 = sram1; Sram1Data = sram1Data; Sram2 = sram2; FlashStore = flashStore
      StartupSource = startup
      ProvidedLibraries = strings "embedded.provided_libraries" |> Set.ofList
      VectorEntries = entries
      RecoveryDirectory = str "embedded.recovery" |> onProject
      ToolDirectory = Toml.getString "embedded.tool_directory" doc |> Option.map onProject
      Cpu = (if String.IsNullOrWhiteSpace core.CpuModel then None else Some core.CpuModel)
      Features = strings "embedded.features" }
