/// TypeMapping - CCS NativeType to MLIR type conversion
///
/// Maps CCS native types to their MLIR representations, reading every width and every size from
/// the graph CCS settled (Dimensional_Range_Design.md §3.1, §3.3, §8.3; Horizon C3):
///
/// - The bare integer kind (`int`, `uint`: `NTUWidth.Resolved Register`) maps to the sentinel
///   `TInt (IntWidth 0)` on every substrate. Its width is the node's: `nodeWidth` reads
///   `RangeAnalysis.heldWidth` (the selected representation of the node's range, the Register width
///   at the value-call boundary, a width-named carrier's own bits, or the one interim word for an
///   unobservable range on a core) and `narrowType` (PSGCombinators) puts it on the sentinel. A
///   sentinel that reaches serialization unnarrowed is a stop there, never a width.
/// - A record, a tuple, a union, an option and a Result take their field widths, offsets and sizes
///   from `SemanticGraph.Layouts`, the layout Placement settled at saturation (ruling 2): a
///   `TStruct` carries its settled bytes, and `mlirTypeSize` and `structFieldOffset` read them.
/// - An array element of the bare kind is held at the width of the element type's settled range
///   (`SemanticGraph.ElementRanges`).
/// - A width-named carrier (`int32`, `uint8`, ...) is its own declared width, an interim boundary
///   until CS-12 deletes the spellings. Pointer-width kinds are `index`.
///
/// Composer reads; it decides no width and computes no size here.
module Alex.CodeGeneration.TypeMapping

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.NativeTypedTree.UnionFind
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Core.Types.Dialects

module RangeAnalysis = Clef.Compiler.PSGSaturation.SemanticGraph.RangeAnalysis

// ═══════════════════════════════════════════════════════════════════════════
// TYPE MAPPING DIAGNOSTIC COLLECTION
// ═══════════════════════════════════════════════════════════════════════════

/// Collects AX1001 diagnostics during type mapping so compilation can continue
/// and report ALL unbound type variables, not just the first one.
/// Drained by mapType after each call.
let private typeMappingErrors = System.Collections.Generic.List<string>()

/// Drain collected type mapping errors. Returns the list and clears the collector.
let drainTypeMappingErrors () : string list =
    let errors = typeMappingErrors |> Seq.toList
    typeMappingErrors.Clear()
    errors

/// The target platform the current compilation lowers to. Set once by MLIRGeneration
/// before any type is mapped. It decides representations that differ per target but are
/// reached through platform-agnostic entry points: the enum DU tag, and the graph-aware
/// mapping's leg.
let mutable private currentTargetPlatform : TargetPlatform option = None

/// Record the target platform for representation decisions made during type mapping.
let setTargetPlatform (platform: TargetPlatform) : unit =
    currentTargetPlatform <- Some platform

/// Representation of a nullary-cases-only DU (an enumeration) for the current target.
/// FPGA: abstract TTag (platform elision picks the width). CPU/MCU: a one-byte memref,
/// the same shape every other DU has, so construction, tag extraction and field storage
/// share one path.
let private enumTagRepresentation (caseCount: int) : MLIRType =
    match currentTargetPlatform with
    | Some FPGA -> TTag caseCount
    | _ -> TMemRefStatic (1, TInt (IntWidth 8))

// ═══════════════════════════════════════════════════════════════════════════
// WIDTH READS (the node's selection, the settled slot, the element range)
// ═══════════════════════════════════════════════════════════════════════════

/// Whether a kind is an integer the leg holds as an MLIR integer of a selected width: the bare
/// kind or a width-named carrier. A pointer-width kind, a size and a difference are `index`
/// and are never narrowed; a bool and a char have their own fixed types.
let isWordInteger (kind: NTUKind) : bool =
    match kind with
    | NTUKind.NTUint (NTUWidth.Resolved WidthDimension.Register) | NTUKind.NTUuint (NTUWidth.Resolved WidthDimension.Register)
    | NTUKind.NTUint (NTUWidth.Fixed _) | NTUKind.NTUuint (NTUWidth.Fixed _) -> true
    | _ -> false

/// The width a node's integer value is held at, read from CCS (`RangeAnalysis.heldWidth`): the
/// selected representation of its range, the Register width at the value-call boundary, a
/// width-named carrier's bits, or the interim word for an unobservable range on a core. None for a
/// node that is not a word integer (a bool, a char, an index, a real, an aggregate), or on fabric
/// for a node whose range has no width (a stop for the reader that needs it).
let nodeWidth (graph: SemanticGraph) (nodeId: NodeId) : IntWidth option =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node when Types.tryGetNTUKind node.Type |> Option.exists isWordInteger ->
        RangeAnalysis.heldWidth graph nodeId |> Option.map IntWidth
    | _ -> None

/// The width a node's integer value is held at, for a reader that cannot proceed without one.
let requireNodeWidth (graph: SemanticGraph) (nodeId: NodeId) : IntWidth =
    match nodeWidth graph nodeId with
    | Some w -> w
    | None ->
        let describe =
            match SemanticGraph.tryGetNode nodeId graph with
            | Some node ->
                let kind = sprintf "%A" node.Kind
                sprintf "node %d (%s, %s, range %s)" (NodeId.value nodeId) (kind.Substring(0, min 40 kind.Length)) (formatType node.Type)
                    (node.ValueRange |> Option.map ValueRange.render |> Option.defaultValue "none")
            | None -> sprintf "node %d (not in the graph)" (NodeId.value nodeId)
        failwithf "TypeMapping: %s has no width to hold its value at: it is not a word integer, or its range has no width on a substrate that holds no interim word (CCS8011 reports that before Composer runs)" describe

/// The rendered key of a type in `Layouts` and `ElementRanges`: the same rendering CCS keys by.
let layoutKey (ty: NativeType) : string = formatType (applySubst ty)

/// The settled layout of an aggregate type, read from the graph: a record or union by the
/// constructor's name, a tuple, an option or a Result by its rendered form.
let settledLayout (graph: SemanticGraph) (ty: NativeType) : SettledLayout option =
    match applySubst ty with
    | NativeType.TApp (tycon, _) as t when tycon.Name = "option" || tycon.Name = "voption" || tycon.Name = "Result" || tycon.Name = "result" ->
        Map.tryFind (layoutKey t) graph.Layouts.Value
    | NativeType.TApp (tycon, _) as t ->
        // a record or a user union by its constructor's name (as FieldRanges), else by rendering
        match Map.tryFind tycon.Name graph.Layouts.Value with
        | Some layout -> Some layout
        | None -> Map.tryFind (layoutKey t) graph.Layouts.Value
    | NativeType.TUnion (tycon, _) -> Map.tryFind tycon.Name graph.Layouts.Value
    | t -> Map.tryFind (layoutKey t) graph.Layouts.Value

/// The width an array element of the bare kind is held at: the settled range of its element type
/// (`ElementRanges`), or the range of a type nothing reachable stores into (unbounded, which the
/// interim word holds while CCS8011 is information on cores).
let elementWidth (graph: SemanticGraph) (elemTy: NativeType) : IntWidth =
    let range = Map.tryFind (layoutKey elemTy) graph.ElementRanges.Value |> Option.defaultValue ValueRange.Unbounded
    match RangeAnalysis.heldWidthOf graph range with
    | Some bits -> IntWidth bits
    | None -> failwithf "TypeMapping: the element type %s has the range %s, which has no width on this substrate (CCS8011)" (layoutKey elemTy) (ValueRange.render range)

/// The MLIR type of a settled scalar slot; None for a slot the mapping keeps as the field's own
/// mapped type (a pointer-sized field, an opaque one).
let private slotScalarType (slot: SettledSlot) : MLIRType option =
    match slot with
    | SettledSlot.Integer (bits, _) -> Some (TInt (IntWidth bits))
    | SettledSlot.Bool -> Some (TInt (IntWidth 1))
    | SettledSlot.Char -> Some (TInt (IntWidth 32))
    | SettledSlot.Real 32 -> Some (TFloat F32)
    | SettledSlot.Real _ -> Some (TFloat F64)
    | SettledSlot.Unit -> Some (TInt (IntWidth 32))
    | SettledSlot.Pointer _ | SettledSlot.Opaque _ -> None

/// The settled bytes of a record layout, when every field of it is placed.
let private bytesOf (fields: SettledField list) (size: int option) (align: int option) : StructBytes option =
    let offsets = fields |> List.map (fun f -> f.Offset)
    match size, align with
    | Some size, Some align when offsets |> List.forall Option.isSome ->
        Some { Offsets = offsets |> List.choose id; Size = size; Align = align }
    | _ -> None

// ═══════════════════════════════════════════════════════════════════════════
// NTUKind DIRECT MAPPING (for literals)
// ═══════════════════════════════════════════════════════════════════════════

/// Map NTUKind directly to MLIRType. Used for NativeLiteral where we have the kind without a
/// full NativeType. The bare integer kind is the sentinel on every substrate: a literal's width
/// is its point range's selection, put on the sentinel by `narrowType` at the literal's node.
/// A width-named spelling here, with no graph and no node, is the spelling's own bits, the
/// value its declaration takes on a description that offers no representation of that name
/// (CS-12 step 5a); a node's width is `nodeWidth`, a type's `mapNativeTypeForTarget`, both
/// reading the declaration through CCS. Owed to the promotion step with the spellings.
let mapNTUKindToMLIRType (kind: NTUKind) : MLIRType =
    match kind with
    | NTUKind.NTUint (NTUWidth.Fixed bits) | NTUKind.NTUuint (NTUWidth.Fixed bits) -> TInt (IntWidth bits)
    | NTUKind.NTUint (NTUWidth.Resolved WidthDimension.Register)
    | NTUKind.NTUuint (NTUWidth.Resolved WidthDimension.Register) -> TInt (IntWidth 0)
    // Native pointer-sized types - map to MLIR index for memref operations
    | NTUKind.NTUint (NTUWidth.Resolved WidthDimension.Pointer)  // nativeint
    | NTUKind.NTUuint (NTUWidth.Resolved WidthDimension.Pointer) // unativeint
    | NTUKind.NTUsize     // size_t
    | NTUKind.NTUdiff     // ptrdiff_t
        -> TIndex
    // Floating point
    | NTUKind.NTUfloat (NTUWidth.Fixed 32) -> TFloat F32
    | NTUKind.NTUfloat (NTUWidth.Fixed 64) -> TFloat F64
    // Boolean
    | NTUKind.NTUbool -> TInt (IntWidth 1)
    // Character (Unicode codepoint = i32)
    | NTUKind.NTUchar -> TInt (IntWidth 32)
    // Unit
    | NTUKind.NTUunit -> TInt (IntWidth 32)  // Unit represented as i32 0
    // Pointers
    | NTUKind.NTUptr | NTUKind.NTUfnptr -> TIndex
    // String as memref (portable MLIR type, not LLVM struct)
    // memref<?xi8> represents a dynamic-sized buffer with length tracked in descriptor
    | NTUKind.NTUstring -> TMemRef (TInt (IntWidth 8))
    // Composite/complex types - representation comes from platform tier, not here
    | kind -> failwithf "NTUKind %A requires platform-tier resolution, not scalar mapping" kind

// ═══════════════════════════════════════════════════════════════════════════
// LEAF TYPE MAPPING (no graph: scalars, handles, function values)
// ═══════════════════════════════════════════════════════════════════════════

/// Map a leaf CCS NativeType to MLIRType: a numeric carrier by its kind, a handle to `index`,
/// a string to its view, a function value to its closure pair. An aggregate (a record, a tuple,
/// a union, an option, a Result, an array) is mapped through the graph (`mapNativeTypeForTarget`),
/// whose settled layout it needs; reaching one here is a stop naming it.
let rec mapNativeTypeForArch (arch: Architecture) (ty: NativeType) : MLIRType =
    let mapTyCon (tycon: TypeConRef) (args: NativeType list) : MLIRType =
        match tycon.NTUKind with
        | Some NTUKind.NTUunit -> TInt (IntWidth 32)
        | Some NTUKind.NTUbool -> TInt (IntWidth 1)
        | Some NTUKind.NTUchar -> TInt (IntWidth 32)
        | Some (NTUKind.NTUint _ as kind) | Some (NTUKind.NTUuint _ as kind) -> mapNTUKindToMLIRType kind
        | Some NTUKind.NTUsize | Some NTUKind.NTUdiff | Some NTUKind.NTUptr | Some NTUKind.NTUfnptr -> TIndex
        | Some (NTUKind.NTUfloat (NTUWidth.Fixed 32)) -> TFloat F32
        | Some (NTUKind.NTUfloat (NTUWidth.Fixed 64)) -> TFloat F64
        | Some NTUKind.NTUstring -> TMemRef (TInt (IntWidth 8))
        | Some NTUKind.NTUlist | Some NTUKind.NTUmap | Some NTUKind.NTUset -> TIndex
        | _ ->
            match tycon.Name, args with
            | ("byref" | "inref" | "outref"), _ -> TIndex
            | "list", _ -> TIndex  // PRD-13a: list<'T> is a pointer to cons cell (linked list)
            | ("array" | "Array"), _ | ("option" | "voption"), _ | ("Result" | "result"), _ ->
                failwithf "mapNativeTypeForArch: '%s' is an aggregate whose element and payload widths and size are settled on the graph; map it through mapNativeTypeForTarget" (formatType ty)
            | _ ->
                match TypeLayout.baseLayout tycon.Layout with
                | TypeLayout.Union when currentTargetPlatform = Some FPGA ->
                    // The fabric leg holds a union as its tag (an enumeration; a payload union is
                    // not yet supported there, DUPatterns): the abstract tag the platform elides
                    enumTagRepresentation tycon.CaseCount
                | TypeLayout.Record | TypeLayout.Union ->
                    failwithf "mapNativeTypeForArch: the %s '%s' has its layout settled on the graph; map it through mapNativeTypeForTarget"
                        (match tycon.Layout with TypeLayout.Record -> "record" | _ -> "union") tycon.Name
                | TypeLayout.Inline (size, _) when size > 0 && tycon.CaseCount = 0 ->
                    // A C-style enum, declared at four bytes (NativeService): an integer of that width
                    TInt (IntWidth (size * 8))
                | TypeLayout.Inline _ when tycon.CaseCount > 0 -> enumTagRepresentation tycon.CaseCount
                | TypeLayout.PlatformWord -> TIndex
                | TypeLayout.NTUCompound n ->
                    // Arena<'lifetime> and similar compound types: N platform words
                    if n = 1 then TIndex else TMemRefStatic (n, TIndex)
                | TypeLayout.FatPointer ->
                    failwithf "FatPointer type '%s' lacks proper NTUKind or name match - fix CCS metadata" tycon.Name
                | TypeLayout.Opaque -> failwithf "TApp with Opaque layout - CCS must resolve type '%s'" tycon.Name
                | TypeLayout.Reference _ -> failwithf "Reference type not yet implemented: %s" tycon.Name
                | TypeLayout.Qualified _ -> failwithf "Qualified layout should have been normalized before mapping: %s" tycon.Name
                | TypeLayout.Inline _ -> failwithf "Unknown inline type '%s' with no fields" tycon.Name

    match ty with
    | NativeType.TApp (tycon, args) -> mapTyCon tycon args
    // The carrier is read through the one carrier read; a carrier variable the checker left
    // unresolved is a checker failure surfaced here, never a width chosen by Composer.
    | NativeType.TNum (carrier, _) ->
        match CarrierRef.tryConstructor carrier with
        | Some tc -> mapTyCon tc []
        | None -> failwithf "mapNativeTypeForArch: unresolved carrier variable in numeric type '%s'; CCS must resolve it" (formatType ty)
    | NativeType.TFun _ ->
        // Closures: {codePtr: ptr, envPtr: ptr} - homogeneous, use memref array
        // Use TIndex (not TPtr) because index can be memref element type
        TMemRefStatic (2, TIndex)
    | NativeType.TVar tvar ->
        // Use Union-Find to resolve type variable chains
        match find tvar with
        | (_, Some boundTy) -> mapNativeTypeForArch arch boundTy
        | (root, None) ->
            // AX1001: Unbound type variable at MLIR generation time.
            // All type variables must be resolved by CCS/Baker before Alex runs.
            // Collect diagnostic and continue with TIndex so all errors are reported.
            typeMappingErrors.Add(sprintf "AX1001: Unbound type variable '%s' — CCS/Baker must resolve all type variables before MLIR generation" root.Name)
            TIndex
    | NativeType.TByref _ -> TIndex
    | NativeType.TNativePtr _ -> TIndex
    | NativeType.TForall (_, body) -> mapNativeTypeForArch arch body
    // PRD-13a: Immutable collection types - all are reference types (pointer to nodes)
    | NativeType.TList _ -> TIndex  // Pointer to cons cell
    | NativeType.TMap _ -> TIndex   // Pointer to tree root
    | NativeType.TSet _ -> TIndex   // Pointer to tree root
    | NativeType.TTuple _ | NativeType.TUnion _ | NativeType.TAnon _ | NativeType.TLazy _ | NativeType.TSeq _ | NativeType.TSeqEnumerator _ ->
        failwithf "mapNativeTypeForArch: '%s' is an aggregate whose layout is settled on the graph; map it through mapNativeTypeForTarget" (formatType ty)
    | NativeType.TMeasure _ ->
        failwith "Measure type should have been stripped - this is an CCS issue"
    | NativeType.TError msg ->
        failwithf "NativeType.TError: %s" msg

// ═══════════════════════════════════════════════════════════════════════════
// GRAPH-AWARE TYPE MAPPING (aggregates at their settled layouts)
// ═══════════════════════════════════════════════════════════════════════════

/// Case payloads of a user union type, from its TypeDef node (None for records, options,
/// abbreviations and primitives).
let private tryGetUnionCases (typeName: string) (graph: SemanticGraph) : (string * (string option * NativeType) list) list option =
    match SemanticGraph.recallType typeName graph with
    | Some nodeId ->
        match SemanticGraph.tryGetNode nodeId graph with
        | Some node ->
            match node.Kind with
            | SemanticKind.TypeDef (_, TypeDefKind.UnionDef cases, _) -> Some cases
            | _ -> None
        | None -> None
    | None -> None

/// Collect unique unbound type variables from a NativeType, in order of first appearance.
/// Follows Union-Find chains to find root TVars that are Unbound.
let rec private collectUnboundTVars (seen: Set<int>) (ty: NativeType) : (TypeParam * Set<int>) list =
    match ty with
    | NativeType.TVar tvar ->
        match find tvar with
        | (root, None) when not (Set.contains root.Id seen) ->
            [(root, Set.add root.Id seen)]
        | _ -> []
    | NativeType.TApp(_, args) ->
        args |> List.fold (fun acc arg ->
            let currentSeen = match acc with [] -> seen | _ -> snd (List.last acc)
            let results = collectUnboundTVars currentSeen arg
            acc @ results) []
    | NativeType.TFun(a, b) ->
        let aVars = collectUnboundTVars seen a
        let bSeen = match aVars with [] -> seen | _ -> snd (List.last aVars)
        aVars @ collectUnboundTVars bSeen b
    | NativeType.TTuple(elements, _) ->
        elements |> List.fold (fun acc elem ->
            let currentSeen = match acc with [] -> seen | _ -> snd (List.last acc)
            acc @ collectUnboundTVars currentSeen elem) []
    | _ -> []

/// When a TApp carries type arguments but the TypeDef's fields contain unbound TVars,
/// bind them in the Union-Find so downstream type mapping resolves correctly.
/// This compensates for clef not propagating type arguments to expression-level nodes.
let private bindTypeArgsToFieldTVars (fields: (string * NativeType) list) (args: NativeType list) =
    if args.IsEmpty then ()
    else
        let unboundTVars =
            fields
            |> List.collect (fun (_, fieldTy) -> collectUnboundTVars Set.empty fieldTy)
            |> List.map fst
        // Bind positionally: first unique unbound TVar → first type arg, etc.
        for i in 0 .. min (unboundTVars.Length - 1) (args.Length - 1) do
            unboundTVars.[i].Parent <- TypeParamState.Bound args.[i]

/// Eagerly walk a NativeType to bind type parameters from TApp type arguments.
/// Call this on function signatures BEFORE traversing body nodes, so inner expression
/// types resolve correctly through the Union-Find.
let rec resolveTypeParams (graph: SemanticGraph) (ty: NativeType) =
    match ty with
    | NativeType.TApp(tycon, args) ->
        if not args.IsEmpty then
            match SemanticGraph.tryGetRecordFields tycon.Name graph with
            | Some fields -> bindTypeArgsToFieldTVars fields args
            | None -> ()
        args |> List.iter (resolveTypeParams graph)
    | NativeType.TFun(a, b) ->
        resolveTypeParams graph a
        resolveTypeParams graph b
    | NativeType.TTuple(elements, _) ->
        elements |> List.iter (resolveTypeParams graph)
    | _ -> ()

/// The physical storage of a value held in a container (an array element, a slot): a record or
/// tuple is a semantic `TStruct` whose storage is a byte memref of its settled size; every other
/// type is stored as itself. A struct with no settled bytes is a stop (`mlirTypeSize`).
let physicalStorageType (arch: Architecture) (ty: MLIRType) : MLIRType =
    match ty with
    | TStruct (_, Some bytes) -> TMemRefStatic (bytes.Size, TInt (IntWidth 8))
    | TStruct _ -> TMemRefStatic (mlirTypeSize arch ty, TInt (IntWidth 8))
    | t -> t

/// The settled record fields of a type, mapped: each field at its settled slot (an integer at its
/// selected representation) or, for a pointer-sized field, at the mapping of its own type; with
/// the layout's bytes. A record the placement did not settle on a core is a stop.
let rec private settledStruct (platform: TargetPlatform) (arch: Architecture) (graph: SemanticGraph)
                              (describe: string) (fields: (string * NativeType) list) (layout: SettledLayout option) : MLIRType =
    let recurse = mapNativeTypeForTarget platform arch graph
    match layout with
    | Some (SettledLayout.Record (settled, size, align)) when settled.Length = fields.Length ->
        let mapped =
            List.zip fields settled
            |> List.map (fun ((name, fieldTy), slot) ->
                match slotScalarType slot.Slot with
                | Some scalar -> (name, scalar)
                | None -> (name, recurse fieldTy))
        TStruct (mapped, bytesOf settled size align)
    | Some other ->
        failwithf "TypeMapping: %s has the settled layout %A, not a record's of %d fields" describe other fields.Length
    | None ->
        match platform with
        | FPGA -> TStruct (fields |> List.map (fun (name, fieldTy) -> (name, recurse fieldTy)), None)
        | _ -> failwithf "TypeMapping: %s has no settled layout on the graph (Placement settles every reachable record, tuple, option and Result; an unreachable or generic instance reaches this)" describe

/// A union, option or Result at its settled size: a byte memref of the tag and the widest payload.
and private settledUnion (describe: string) (layout: SettledLayout option) (caseCount: int) : MLIRType =
    match layout with
    | Some (SettledLayout.Union (cases, _, Some size, _)) ->
        if cases |> List.forall (fun (_, slot) -> slot.IsNone) then enumTagRepresentation caseCount
        else TMemRefStatic (size, TInt (IntWidth 8))
    | Some (SettledLayout.Union (cases, _, None, _)) ->
        failwithf "TypeMapping: %s has a settled union layout with no size: a case payload the placement could not settle (%s)" describe
            (cases |> List.choose (fun (n, s) -> match s with Some (SettledSlot.Opaque what) -> Some (sprintf "%s: %s" n what) | _ -> None) |> String.concat "; ")
    | Some other -> failwithf "TypeMapping: %s has the settled layout %A, not a union's" describe other
    | None -> failwithf "TypeMapping: %s has no settled layout on the graph" describe

/// Platform-aware type mapping — the canonical entry point for target-dependent code. On every
/// substrate the bare integer kind is the sentinel, narrowed at its node; an aggregate takes its
/// widths, offsets and size from the graph's settled layouts on a core, and on fabric its field
/// widths from `FieldRanges` through `narrowType`. Recursive: nested aggregates likewise.
and mapNativeTypeForTarget (platform: TargetPlatform) (arch: Architecture) (graph: SemanticGraph) (ty: NativeType) : MLIRType =
    let recurse = mapNativeTypeForTarget platform arch graph
    let core = platform <> FPGA
    let layoutOf (t: NativeType) = if core then settledLayout graph t else None
    match ty with
    // A value of a width-named spelling (CS-12 step 5a, the alias): its width is the declaration
    // the spelling writes, read from CCS (`RangeAnalysis.declaredWidthOfKind`, the platform's
    // representation of that name), never the spelling's own bits. A node at hand reads
    // `nodeWidth` instead; this is the read for a type with no node (a signature, a capture).
    | NativeType.TNum (carrier, _) when core ->
        match CarrierRef.tryConstructor carrier |> Option.bind (fun tc -> tc.NTUKind) with
        | Some (NTUKind.NTUint (NTUWidth.Fixed _) as kind) | Some (NTUKind.NTUuint (NTUWidth.Fixed _) as kind) ->
            match RangeAnalysis.declaredWidthOfKind graph kind with
            | Some bits -> TInt (IntWidth bits)
            | None -> failwithf "mapNativeTypeForTarget: the spelled kind %s has no declared representation on this platform" (NTUKind.name kind)
        | _ -> mapNativeTypeForArch arch ty
    | NativeType.TApp(tycon, args) when tycon.FieldCount > 0 ->
        // Record type: look up field types from TypeDef → TStruct with named fields
        match SemanticGraph.tryGetRecordFields tycon.Name graph with
        | Some fields ->
            // Bind type arguments to unbound TVars in fields (compensates for clef gap)
            bindTypeArgsToFieldTVars fields args
            settledStruct platform arch graph (sprintf "the record '%s'" tycon.Name) fields (layoutOf ty)
        | None ->
            failwithf "Record type '%s' not found in TypeDef nodes - CCS must create TypeDef for records" tycon.Name
    | NativeType.TApp(tycon, args) ->
        // Non-record TApp (FieldCount = 0) - check if it might be a record by name lookup
        match SemanticGraph.tryGetRecordFields tycon.Name graph with
        | Some fields ->
            bindTypeArgsToFieldTVars fields args
            settledStruct platform arch graph (sprintf "the record '%s'" tycon.Name) fields (layoutOf ty)
        | None ->
            match platform with
            | FPGA ->
                // FPGA: option/voption → hw.struct with tag + value
                match tycon.Name with
                | "option" | "voption" ->
                    match args with
                    | [innerTy] ->
                        let innerMlir = recurse innerTy
                        TStruct ([("tag", TInt (IntWidth 1)); ("value", innerMlir)], None)
                    | _ -> mapNativeTypeForArch arch ty
                | _ -> mapNativeTypeForArch arch ty
            | _ ->
                match tryGetUnionCases tycon.Name graph with
                | Some cases when not (List.isEmpty cases) ->
                    // User union: its settled size, or an enumeration tag
                    settledUnion (sprintf "the union '%s'" tycon.Name) (layoutOf ty) cases.Length
                | _ ->
                // CPU/MCU containers: element/payload types are the graph-aware PHYSICAL types,
                // so an array of records or an option of a record agrees with the record's storage.
                match tycon.Name, args with
                | ("array" | "Array"), [elemTy] ->
                    let elem =
                        match recurse elemTy with
                        | TInt (IntWidth 0) -> TInt (elementWidth graph elemTy)
                        | mapped -> physicalStorageType arch mapped
                    TMemRef elem
                | ("option" | "voption"), [_] -> settledUnion (sprintf "the option '%s'" (layoutKey ty)) (layoutOf ty) 2
                | ("Result" | "result"), [_; _] -> settledUnion (sprintf "the Result '%s'" (layoutKey ty)) (layoutOf ty) 2
                | _ -> mapNativeTypeForArch arch ty
    | NativeType.TTuple(elements, _) ->
        // Tuples are materialized as TStruct with positional field names on all platforms.
        // CPU uses memref alloca + byte-offset stores; FPGA uses hw.struct_create.
        // Both need TStruct for field-level access (pRecordFieldGet, TupleGet extraction).
        let fields = elements |> List.mapi (fun i e -> sprintf "Item%d" (i + 1), e)
        settledStruct platform arch graph (sprintf "the tuple '%s'" (layoutKey ty)) fields (layoutOf ty)
    | NativeType.TAnon(fields, _) ->
        // Anonymous records → TStruct with named fields (no settled layout: a stop where sized)
        TStruct (fields |> List.map (fun (name, fieldTy) -> (name, recurse fieldTy)), None)
    | NativeType.TUnion (tycon, cases) ->
        settledUnion (sprintf "the union '%s'" tycon.Name) (layoutOf ty) cases.Length
    | NativeType.TLazy elemTy ->
        // Lazy<T> - flat closure {computed: i1, value: T, code_ptr}: a Composer-realised
        // aggregate (PRD-14) whose bytes are its fields' reads; owed to the settled layouts
        let elemMlir = recurse elemTy
        let totalBytes = 1 + mlirTypeSize arch elemMlir + mlirTypeSize arch TIndex
        TMemRefStatic(totalBytes, TInt (IntWidth 8))
    | NativeType.TSeq elemTy ->
        // PRD-15: Seq<T> - flat closure {state: i32, current: T, moveNext_ptr: ptr}; as TLazy
        let elemMlir = recurse elemTy
        let totalSize = mlirTypeSize arch (TInt (IntWidth 32)) + mlirTypeSize arch elemMlir + mlirTypeSize arch TIndex
        TMemRefStatic (totalSize, TInt (IntWidth 8))
    | NativeType.TSeqEnumerator elemTy ->
        // PRD-15/16: { seq_ptr: ptr, state: i32, current: T, hasValue: i1 }; as TLazy
        let elemMlir = recurse elemTy
        let totalSize = mlirTypeSize arch TIndex + mlirTypeSize arch (TInt (IntWidth 32)) + mlirTypeSize arch elemMlir + mlirTypeSize arch (TInt (IntWidth 1))
        TMemRefStatic (totalSize, TInt (IntWidth 8))
    | NativeType.TVar tvar ->
        // Resolve type variable through Union-Find and recurse through target-aware mapper
        match find tvar with
        | (_, Some boundTy) -> recurse boundTy
        | (root, None) ->
            // AX1001: Unbound type variable at MLIR generation time.
            // All type variables must be resolved by CCS/Baker before Alex runs.
            // Collect diagnostic and continue with TIndex so all errors are reported.
            typeMappingErrors.Add(sprintf "AX1001: Unbound type variable '%s' — CCS/Baker must resolve all type variables before MLIR generation" root.Name)
            TIndex
    | NativeType.TForall (_, body) -> recurse body
    | _ ->
        // Leaf types: a scalar, a handle, a function value
        mapNativeTypeForArch arch ty

/// The graph-aware mapping for the current target (set once by MLIRGeneration): the entry point
/// of the patterns that map a node's type on the leg being compiled.
let mapNativeTypeWithGraphForArch (arch: Architecture) (graph: SemanticGraph) (ty: NativeType) : MLIRType =
    mapNativeTypeForTarget (currentTargetPlatform |> Option.defaultValue CPU) arch graph ty

/// The element type of an array node's type, at its settled element width (an array of the bare
/// kind) or its physical storage: what an allocation, a store and a load of its elements use.
let arrayElementType (arch: Architecture) (graph: SemanticGraph) (arrayTy: NativeType) : MLIRType =
    match mapNativeTypeWithGraphForArch arch graph arrayTy with
    | TMemRef elem | TMemRefStatic (_, elem) -> elem
    | other -> failwithf "TypeMapping: '%s' is not an array (mapped to %A)" (formatType arrayTy) other
