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

let triple = function
    | CortexMProfile.CortexM33SoftFloat -> "thumbv8m.main-none-eabi"
    | CortexMProfile.Stm32H747HardFloat -> "thumbv7em-none-eabihf"

let cpu = function
    | CortexMProfile.CortexM33SoftFloat -> "cortex-m33"
    | CortexMProfile.Stm32H747HardFloat -> "cortex-m7"

let floatAbi = function
    | CortexMProfile.CortexM33SoftFloat -> "soft"
    | CortexMProfile.Stm32H747HardFloat -> "hard"

let reservedVectorSlots = function
    | CortexMProfile.CortexM33SoftFloat -> [8;9;10;13]
    | CortexMProfile.Stm32H747HardFloat -> [7;8;9;10;13]

/// The existing probe owns an RA6 J-Link transaction, never an STM32 one.
let requireProbeSupport = function
    | CortexMProfile.CortexM33SoftFloat -> ()
    | CortexMProfile.Stm32H747HardFloat ->
        failwith "STM32H747 supports image builds only; Composer has no STM32 ST-LINK deployment or device transaction"

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
    let profile =
        match core.Arch, core.Triple, core.CpuModel with
        | "arm_cortex_m33", "thumbv8m.main-none-eabi", "cortex-m33" -> CortexMProfile.CortexM33SoftFloat
        | "arm_cortex_m7", "thumbv7em-none-eabihf", "cortex-m7" -> CortexMProfile.Stm32H747HardFloat
        | _ -> failwith "The MCU image backend requires the declared Cortex-M33 soft-float or STM32H747 Cortex-M7 FPv5-D16 hard-float target"
    if core.Widths |> List.tryFind (fun width -> width.Name = "Pointer") |> Option.map (fun width -> width.Bits) <> Some 32 then
        failwith "Cortex-M images require a declared 32-bit Pointer dimension"
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
        platform.Spaces |> List.map memorySpace |> List.toArray
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
    // A selected part owns its reset placement. Do not generalize the RA6
    // flash-at-zero rule into arbitrary nonzero flash support.
    match profile with
    | CortexMProfile.CortexM33SoftFloat ->
        if origin flash <> 0L then failwith "This reset profile requires code flash at zero"
    | CortexMProfile.Stm32H747HardFloat ->
        if flash.Name <> "flash-bank1" || origin flash <> 0x08000000L || flash.Capacity <> 0x100000L then
            failwith "STM32H747 first image requires the complete 1 MiB flash-bank1 at 0x08000000"
        match ram.Name, origin ram, ram.Capacity with
        | "dtcm", 0x20000000L, 0x20000L
        | "axi-sram", 0x24000000L, 0x80000L -> ()
        | _ -> failwith "STM32H747 requires the complete 128 KiB DTCM at 0x20000000 or 512 KiB AXI SRAM at 0x24000000"
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
    match profile with
    | CortexMProfile.CortexM33SoftFloat ->
        if image.PartNumber.Length < 1 || image.PartNumber.Length > 16 || image.PartNumberAddress < 0L || image.PartNumberAddress % 4L <> 0L || image.PartNumberAddress + 16L > 0x100000000L then
            failwith "Invalid RA6 part-number register extent"
    | CortexMProfile.Stm32H747HardFloat ->
        if image.VectorLayout <> "ArmV7MVectors" || vectors.Layout.Size <> 166 * 4 || alignment <> 1024 then
            failwith "STM32H747 requires ArmV7MVectors with 166 words and 1024-byte placement alignment"
        if image.DebugDevice <> "STM32H747XIH6" || image.PartNumber <> "STM32H747XIH6" || image.PartNumberAddress <> 0x5C001000L then
            failwith "STM32H747 requires the STM32H747XIH6 identity declaration and DBGMCU_IDC at 0x5C001000"
        if image.PreservedOptionAddress <> 0x5200201CL || image.PreservedOptionBytes <> 48 then
            failwith "STM32H747 requires the declared 48-byte bank-1 option-register observation window at 0x5200201C"
    if image.PreservedOptionBytes <= 0 || image.PreservedOptionBytes > 4096 || image.PreservedOptionBytes % 4 <> 0 || image.PreservedOptionAddress < 0L || image.PreservedOptionAddress % 4L <> 0L || image.PreservedOptionAddress + int64 image.PreservedOptionBytes > 0x100000000L then
        failwith "Invalid preserved-option extent"
    let startup = str "embedded.startup" |> onProject
    if Path.GetExtension(startup) <> ".S" || not (File.Exists startup) then failwith "embedded.startup must name an owned .S assembly source"
    if str "embedded.entry_abi" <> "clef-empty-string-array32" then failwith "Unsupported embedded entry ABI"
    let handlers =
        Toml.getTable "embedded.vector_handlers" doc |> required "embedded.vector_handlers"
        |> Map.toList |> List.map (fun (slot, value) ->
            let index = match Int32.TryParse slot with true, n -> n | _ -> failwith "Vector slot must be an integer"
            if index < 1 || index >= vectors.Layout.Size / 4 || List.contains index (reservedVectorSlots profile) then failwith "Invalid/reserved vector slot"
            index, (match value with TomlValue.String name -> symbol name | _ -> failwith "Vector handler must be a symbol")) |> Map.ofList
    if Map.tryFind 1 handlers <> Some image.EntrySymbol then failwith "Reset vector must match the platform entry symbol"
    { PlatformId = platform.Id; Profile = profile; Image = image; Vectors = vectors; Flash = flash; Ram = ram
      StartupSource = startup; ProvidedLibraries = strings "embedded.provided_libraries" |> Set.ofList
      VectorHandlers = handlers; RecoveryDirectory = str "embedded.recovery" |> onProject
      ToolDirectory = Toml.getString "embedded.tool_directory" doc |> Option.map onProject
      ProbeLibrary = Toml.getString "embedded.probe_library" doc |> Option.map onProject
      WatchSymbols = Toml.getTable "embedded.watch" doc |> Option.defaultValue Map.empty |> Map.toList
          |> List.map (fun (name, value) -> symbol name, (match value with TomlValue.Integer n when n > 0L && n <= 64L -> int n | _ -> failwith "Watch extent must be 1..64 words")) |> Map.ofList }
