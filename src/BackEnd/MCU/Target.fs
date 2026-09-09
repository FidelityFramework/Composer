/// Project inputs and BAREWire declarations for the owned Cortex-M image path.
module BackEnd.MCU.Target

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

let resolve (projectPath: string) (graph: SemanticGraph) : EmbeddedTarget =
    let projectPath = Path.GetFullPath projectPath
    let projectDir = Path.GetDirectoryName projectPath
    let doc = Toml.parse (File.ReadAllText projectPath) |> function Ok d -> d | Error e -> failwith e
    let str key = Toml.getString key doc |> required key
    let strings key =
        match Toml.getValue key doc with
        | None -> []
        | Some (TomlValue.Array xs) -> xs |> List.map (function TomlValue.String s when s <> "" -> s | _ -> failwith ("Expected strings: " + key))
        | _ -> failwith ("Expected array: " + key)
    let onProject p = Path.GetFullPath(p, projectDir)
    let platform = resolve graph |> required "BAREWire PlatformDescription"
    let core = platform.Core |> required "platform core"
    if core.Arch <> "arm_cortex_m33" || core.Triple <> "thumbv8m.main-none-eabi" || core.CpuModel <> "cortex-m33" then
        failwith "The MCU image backend currently supports the declared Cortex-M33 Thumb soft-float profile only"
    let declarations =
        graph.Nodes |> Map.toList |> List.choose (fun (_, node) ->
            match node.Kind with
            | SemanticKind.Binding _ ->
                node.Children |> List.tryLast |> Option.bind (recordOf graph)
                |> Option.bind (fun (record, fields) ->
                    match record.Type with
                    | NativeType.TApp (tc, _) when tc.Name.Split('.') |> Array.last = "CortexMImageDescriptor" -> Some fields
                    | _ -> None)
            | _ -> None)
    let fields = match declarations with [one] -> one | _ -> failwith "Declare exactly one BAREWire CortexMImageDescriptor"
    let field name reader = fields |> List.tryFind (fst >> (=) name) |> Option.bind (snd >> reader graph) |> required ("CortexMImageDescriptor." + name)
    let image: CortexMImageDescriptor = {
        FlashSpace = field "FlashSpace" stringOf; RamSpace = field "RamSpace" stringOf
        VectorLayout = field "VectorLayout" stringOf; VectorAlignment = int (field "VectorAlignment" int64Of)
        StackBytes = int (field "StackBytes" int64Of); EntrySymbol = field "EntrySymbol" stringOf |> symbol
        DebugDevice = field "DebugDevice" stringOf; PartNumber = field "PartNumber" stringOf
        PartNumberAddress = field "PartNumberAddress" int64Of
        PreservedOptionAddress = field "PreservedOptionAddress" int64Of
        PreservedOptionBytes = int (field "PreservedOptionBytes" int64Of)
    }
    let spaces: MemorySpace array =
        platform.Spaces |> List.map (fun s -> {
            Name = s.Name; Kind = s.Kind; Base = s.Base; Capacity = s.Capacity
            Alignment = s.Alignment; Granularity = s.Granularity; Growth = s.Growth; Access = s.Access
            Notes = ""; MapKind = ""; Since = ""; Until = ""
        }) |> List.toArray
    // CCS checks the entire declaration. This observer checks its physical
    // address-space projection in-process, with BAREWire's overlap/alignment rules.
    let memoryProjection: PlatformDescription = {
        Id = platform.Id; DisplayName = platform.Id; Substrate = "mcu"; Core = None
        Spaces = spaces; Surfaces = [||]; Buffers = [||]; Transports = [||]; Notes = [||]; Limits = [||]
        Lifecycle = { Clocks = [||]; Resets = [||]; Entry = image.EntrySymbol; Teardown = ""; Persistence = Persistence.Volatile }
    }
    let findings = Check.run memoryProjection
    if findings.Length <> 0 then failwithf "BAREWire memory declaration: %A" findings
    let space name = spaces |> Array.tryFind (fun s -> s.Name = name) |> required ("memory space " + name)
    let flash, ram = space image.FlashSpace, space image.RamSpace
    if flash.Kind <> MemoryKind.Flash || flash.Access <> Access.ReadExecute || ram.Kind <> MemoryKind.Sram || ram.Access <> Access.ReadWrite then
        failwith "Image requires executable flash and read/write SRAM"
    let origin (s: MemorySpace) = s.Base |> required ("base of " + s.Name)
    for s in [flash; ram] do
        if origin s < 0L || origin s + s.Capacity > 0x100000000L then failwith "Image space exceeds 32-bit address extent"
    // Reset-vector placement and the supported RA6 J-Link identity reader are explicit.
    if origin flash <> 0L then failwith "This reset profile requires code flash at zero"
    if image.StackBytes <= 0 || int64 image.StackBytes >= ram.Capacity || image.StackBytes % 8 <> 0 || (origin ram + ram.Capacity) % 8L <> 0L then
        failwith "Invalid Cortex-M stack reservation"
    let layouts = readDescriptors graph
    let layout = layouts.Layouts |> List.filter (fun l -> l.Name = image.VectorLayout) |> function
        | [one] -> one | _ -> failwith "VectorLayout must name exactly one BAREWire StructDescriptor"
    let vectors: StructDescriptor = {
        Name = layout.Name; Documentation = None
        Layout = { Size = layout.Size |> required "vector size"; Alignment = layout.Alignment |> required "vector natural alignment"
                   Fields = layout.PhysicalFields |> List.map (fun f -> {
                       Name = f.Name; Repr = f.Repr; Count = f.Count; Offset = f.Offset
                       Access = AccessKind.ReadOnly; BitFields = [||]; Documentation = None }) |> List.toArray }
    }
    let checkedLayout = Validator.validate Abi.armAapcs vectors
    if not checkedLayout.Agrees then failwith (Validator.explain checkedLayout)
    match vectors.Layout.Fields with
    | [| f |] when f.Repr = Repr.U32 && f.Offset = 0 && f.Count >= 16 && f.Count <= 512 && vectors.Layout.Size = f.Count * 4 -> ()
    | _ -> failwith "Cortex-M vectors must be one contiguous U32 array"
    let alignment = image.VectorAlignment
    if alignment < 128 || alignment &&& (alignment - 1) <> 0 || alignment < vectors.Layout.Size then failwith "Invalid VTOR placement alignment"
    if image.PartNumber.Length < 1 || image.PartNumber.Length > 16 || image.PartNumberAddress < 0L || image.PartNumberAddress % 4L <> 0L || image.PartNumberAddress + 16L > 0x100000000L then
        failwith "Invalid RA6 part-number register extent"
    if image.PreservedOptionBytes <= 0 || image.PreservedOptionBytes > 4096 || image.PreservedOptionBytes % 4 <> 0 || image.PreservedOptionAddress < 0L || image.PreservedOptionAddress % 4L <> 0L || image.PreservedOptionAddress + int64 image.PreservedOptionBytes > 0x100000000L then
        failwith "Invalid preserved-option extent"
    let startup = str "embedded.startup" |> onProject
    if Path.GetExtension(startup) <> ".S" || not (File.Exists startup) then failwith "embedded.startup must name an owned .S assembly source"
    if str "embedded.entry_abi" <> "clef-empty-string-array32" then failwith "Unsupported embedded entry ABI"
    let handlers =
        Toml.getTable "embedded.vector_handlers" doc |> required "embedded.vector_handlers"
        |> Map.toList |> List.map (fun (slot, value) ->
            let index = match Int32.TryParse slot with true, n -> n | _ -> failwith "Vector slot must be an integer"
            if index < 1 || index >= vectors.Layout.Size / 4 || List.contains index [8;9;10;13] then failwith "Invalid/reserved vector slot"
            index, (match value with TomlValue.String name -> symbol name | _ -> failwith "Vector handler must be a symbol")) |> Map.ofList
    if Map.tryFind 1 handlers <> Some image.EntrySymbol then failwith "Reset vector must match the platform entry symbol"
    { PlatformId = platform.Id; Image = image; Vectors = vectors; Flash = flash; Ram = ram
      StartupSource = startup; ProvidedLibraries = strings "embedded.provided_libraries" |> Set.ofList
      VectorHandlers = handlers; RecoveryDirectory = str "embedded.recovery" |> onProject
      ToolDirectory = Toml.getString "embedded.tool_directory" doc |> Option.map onProject
      ProbeLibrary = Toml.getString "embedded.probe_library" doc |> Option.map onProject
      WatchSymbols = Toml.getTable "embedded.watch" doc |> Option.defaultValue Map.empty |> Map.toList
          |> List.map (fun (name, value) -> symbol name, (match value with TomlValue.Integer n when n > 0L && n <= 64L -> int n | _ -> failwith "Watch extent must be 1..64 words")) |> Map.ofList }
