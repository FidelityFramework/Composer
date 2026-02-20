/// PlatformPinResolution — Pre-compute FPGA pin mapping coeffect
///
/// Walks the PSG to extract:
///   1. [<Pin>]/[<Pins>] attributes from Input/Output TypeConRefs (FieldPinAttributes)
///   2. PinEndpoint records from platform bindings (LogicalName → PackagePin + IOStandard)
///   3. ClockEndpoint record (PackagePin, FrequencyHz, IOStandard)
///   4. PlatformDescriptor (Device + Package + SpeedGrade → Vivado device part string)
///
/// Joins (1) with (2) to produce PlatformPinMapping, consumed by:
///   - HardwareModulePatterns (flat hw.module ports)
///   - XDCTransfer (XDC constraint file generation)
///
/// Two observers, one truth, two residuals.
module PSGElaboration.PlatformPinResolution

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open PSGElaboration.Coeffects

// ═══════════════════════════════════════════════════════════════════════════
// PSG VALUE EXTRACTION HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Resolve a PSG node to a string literal, following VarRef and TypeAnnotation chains.
let rec private resolveToString (graph: SemanticGraph) (nodeId: NodeId) : string option =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Literal (NativeLiteral.String s) -> Some s
        | SemanticKind.VarRef (_, Some bindingId) ->
            match SemanticGraph.tryGetNode bindingId graph with
            | Some bindingNode ->
                match bindingNode.Children with
                | [valueId] -> resolveToString graph valueId
                | _ -> None
            | None -> None
        | SemanticKind.TypeAnnotation (wrappedId, _) ->
            resolveToString graph wrappedId
        | SemanticKind.Application (funcId, [argId]) ->
            // Handle cases like ElectricalStandard.Unknown "text" (function application)
            resolveToString graph argId
        | _ -> None
    | None -> None

/// Resolve a PSG node to an int64 literal, following VarRef and TypeAnnotation chains.
let rec private resolveToInt64 (graph: SemanticGraph) (nodeId: NodeId) : int64 option =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Literal (NativeLiteral.Int (v, _)) -> Some v
        | SemanticKind.VarRef (_, Some bindingId) ->
            match SemanticGraph.tryGetNode bindingId graph with
            | Some bindingNode ->
                match bindingNode.Children with
                | [valueId] -> resolveToInt64 graph valueId
                | _ -> None
            | None -> None
        | SemanticKind.TypeAnnotation (wrappedId, _) ->
            resolveToInt64 graph wrappedId
        | _ -> None
    | None -> None

/// Get short TypeConRef name from a NativeType (last segment of fully-qualified name).
/// E.g., "Platform.Contracts.PinEndpoint" → "PinEndpoint"
let private typeConName (ty: NativeType) : string option =
    match ty with
    | NativeType.TApp (tycon, _) ->
        let name = tycon.Name
        match name.LastIndexOf('.') with
        | -1 -> Some name
        | i  -> Some (name.Substring(i + 1))
    | _ -> None

/// Extract RecordExpr fields from a node, unwrapping TypeAnnotation wrappers.
let private extractRecordFields (graph: SemanticGraph) (nodeId: NodeId) : (string * NodeId) list option =
    let rec unwrap nid =
        match SemanticGraph.tryGetNode nid graph with
        | Some n ->
            match n.Kind with
            | SemanticKind.TypeAnnotation (innerNodeId, _) -> unwrap innerNodeId
            | SemanticKind.RecordExpr (fields, _) -> Some fields
            | _ -> None
        | None -> None
    unwrap nodeId

/// Lookup a field value NodeId by name from a RecordExpr field list.
let private fieldValue (name: string) (fields: (string * NodeId) list) : NodeId option =
    fields |> List.tryFind (fun (n, _) -> n = name) |> Option.map snd

/// Normalize a pin logical name to a valid MLIR/Verilog identifier.
/// Applied once here in the coeffect — observers see the final name.
let private normalizePortName (name: string) =
    name.Replace("[", "_").Replace("]", "")

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM BINDING EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════

/// Extract PinConstraint from a PinEndpoint RecordExpr's fields.
let private extractPinEndpoint (graph: SemanticGraph) (fields: (string * NodeId) list) : PinConstraint option =
    let logicalName = fieldValue "LogicalName" fields |> Option.bind (resolveToString graph)
    let packagePin  = fieldValue "PackagePin"  fields |> Option.bind (resolveToString graph)
    let standard    = fieldValue "Standard"    fields |> Option.bind (resolveToString graph)
    let direction   = fieldValue "Direction"   fields |> Option.bind (resolveToString graph)

    match logicalName, packagePin with
    | Some ln, Some pp ->
        Some {
            PortName   = normalizePortName ln
            PackagePin = pp
            IOStandard = standard  |> Option.defaultValue "LVCMOS33"
            Direction  = direction |> Option.defaultValue "InOut"
        }
    | _ -> None

/// Extract ClockConstraint from a ClockEndpoint RecordExpr's fields.
let private extractClockEndpoint (graph: SemanticGraph) (fields: (string * NodeId) list) : ClockConstraint option =
    let name        = fieldValue "Name"        fields |> Option.bind (resolveToString graph)
    let packagePin  = fieldValue "PackagePin"  fields |> Option.bind (resolveToString graph)
    let standard    = fieldValue "Standard"    fields |> Option.bind (resolveToString graph)
    let frequencyHz = fieldValue "FrequencyHz" fields |> Option.bind (resolveToInt64 graph)

    match name, packagePin, frequencyHz with
    | Some n, Some pp, Some freq ->
        Some {
            PortName    = normalizePortName n
            PackagePin  = pp
            IOStandard  = standard |> Option.defaultValue "LVCMOS33"
            FrequencyHz = freq
        }
    | _ -> None

/// Extract ResetConstraint from a ResetEndpoint RecordExpr's fields.
let private extractResetEndpoint (graph: SemanticGraph) (fields: (string * NodeId) list) : ResetConstraint option =
    let name        = fieldValue "Name"        fields |> Option.bind (resolveToString graph)
    let kind        = fieldValue "Kind"        fields |> Option.bind (resolveToString graph)
    let packagePin  = fieldValue "PackagePin"  fields |> Option.bind (resolveToString graph)
    let standard    = fieldValue "Standard"    fields |> Option.bind (resolveToString graph)
    let activeLevel = fieldValue "ActiveLevel" fields |> Option.bind (resolveToString graph)

    match name, kind with
    | Some n, Some k ->
        let isExternal = k = "External"
        Some {
            PortName    = normalizePortName n
            IsExternal  = isExternal
            PackagePin  = packagePin  |> Option.defaultValue "NONE"
            IOStandard  = standard    |> Option.defaultValue "LVCMOS33"
            ActiveHigh  = (activeLevel |> Option.defaultValue "High") = "High"
        }
    | _ -> None

/// Extract Vivado device part string from PlatformDescriptor fields.
/// Format: xc7a100tcsg324-1 (lowercase device + lowercase package + speedGrade)
let private extractDevicePart (graph: SemanticGraph) (fields: (string * NodeId) list) : string option =
    let device    = fieldValue "Device"     fields |> Option.bind (resolveToString graph)
    let package'  = fieldValue "Package"    fields |> Option.bind (resolveToString graph)
    let speedGrade = fieldValue "SpeedGrade" fields |> Option.bind (resolveToString graph)

    match device, package', speedGrade with
    | Some d, Some p, Some sg ->
        Some (sprintf "%s%s%s" (d.ToLowerInvariant()) (p.ToLowerInvariant()) sg)
    | _ -> None

/// Walk all PSG Binding nodes and collect platform data by type name.
let private collectPlatformBindings (graph: SemanticGraph) =
    let mutable pins = []
    let mutable clocks = []
    let mutable resets = []
    let mutable deviceParts = []

    for KeyValue(_, node) in graph.Nodes do
        // Only process Binding nodes (avoid duplicates from VarRef etc.)
        match node.Kind with
        | SemanticKind.Binding _ ->
            match typeConName node.Type with
            | Some "PinEndpoint" ->
                match node.Children with
                | [childId] ->
                    match extractRecordFields graph childId with
                    | Some fields ->
                        match extractPinEndpoint graph fields with
                        | Some pin -> pins <- pin :: pins
                        | None -> ()
                    | None -> ()
                | _ -> ()
            | Some "ClockEndpoint" ->
                match node.Children with
                | [childId] ->
                    match extractRecordFields graph childId with
                    | Some fields ->
                        match extractClockEndpoint graph fields with
                        | Some clk -> clocks <- clk :: clocks
                        | None -> ()
                    | None -> ()
                | _ -> ()
            | Some "ResetEndpoint" ->
                match node.Children with
                | [childId] ->
                    match extractRecordFields graph childId with
                    | Some fields ->
                        match extractResetEndpoint graph fields with
                        | Some rst -> resets <- rst :: resets
                        | None -> ()
                    | None -> ()
                | _ -> ()
            | Some "PlatformDescriptor" ->
                match node.Children with
                | [childId] ->
                    match extractRecordFields graph childId with
                    | Some fields ->
                        match extractDevicePart graph fields with
                        | Some part -> deviceParts <- part :: deviceParts
                        | None -> ()
                    | None -> ()
                | _ -> ()
            | _ -> ()
        | _ -> ()

    (pins, clocks, resets, deviceParts)

// ═══════════════════════════════════════════════════════════════════════════
// DESIGN TYPE PIN ATTRIBUTE EXTRACTION
// ═══════════════════════════════════════════════════════════════════════════

/// Walk ALL types in PSG and collect FieldPinAttributes from any TypeConRef that has them.
/// This is simpler and more robust than navigating the Design type tree: pin attributes
/// are on leaf record types (Inputs, LedOutputs, RgbOutputs) regardless of nesting depth.
let private extractDesignPinAttributes (graph: SemanticGraph) : Map<string, string list> =
    let mutable allAttrs = Map.empty

    let rec collectFromType (ty: NativeType) =
        match ty with
        | NativeType.TApp (tycon, args) ->
            // Collect own FieldPinAttributes
            for KeyValue(k, v) in tycon.FieldPinAttributes do
                allAttrs <- Map.add k (v |> List.map normalizePortName) allAttrs
            // Recurse into type arguments
            for arg in args do collectFromType arg
        | NativeType.TTuple (types, _) ->
            for t in types do collectFromType t
        | _ -> ()

    for KeyValue(_, node) in graph.Nodes do
        collectFromType node.Type

    allAttrs

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// Resolve FPGA pin mapping from the PSG.
/// Returns None if no HardwareModule with pin attributes is found.
let resolve (graph: SemanticGraph) : PlatformPinMapping option =
    let designPinAttrs = extractDesignPinAttributes graph

    // If no pin attributes found, no mapping needed
    if Map.isEmpty designPinAttrs then None
    else

    let (pinEndpoints, clockEndpoints, resetEndpoints, deviceParts) = collectPlatformBindings graph

    // Build lookup from logical name → PinConstraint
    let pinLookup =
        pinEndpoints
        |> List.map (fun p -> p.PortName, p)
        |> Map.ofList

    // Collect all pin constraints referenced by the Design's Pin attributes
    let designPins =
        designPinAttrs
        |> Map.toList
        |> List.collect (fun (_, pinNames) ->
            pinNames |> List.choose (fun name -> Map.tryFind name pinLookup))

    match clockEndpoints, deviceParts with
    | clk :: _, dev :: _ ->
        Some {
            Pins = designPins
            Clock = clk
            Reset = List.tryHead resetEndpoints
            DevicePart = dev
            FieldPinAttrs = designPinAttrs
        }
    | _ -> None
