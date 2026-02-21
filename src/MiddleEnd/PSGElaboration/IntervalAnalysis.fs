/// IntervalAnalysis - FPGA width inference from value range analysis
///
/// ARCHITECTURAL FOUNDATION:
/// This module performs ONCE-per-graph analysis before transfer begins.
/// It computes value intervals (min, max) for PSG nodes and derives
/// minimum bit widths from those intervals.
///
/// On FPGA, every wire is exactly as wide as it needs to be.
/// This nanopass discovers how wide each value needs to be by
/// propagating ranges through the PSG:
///   - Constants have exact ranges
///   - x % K bounds to [0, K-1]
///   - DU tags have [0, numCases-1]
///   - Booleans are [0, 1]
///   - Arithmetic propagates through operand ranges
///
/// The result is a coeffect: pre-computed, observed by Patterns during elision.
module PSGElaboration.IntervalAnalysis

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════════════════════
// TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Value range for a PSG node — [Min, Max] inclusive
type ValueInterval = {
    Min: int64
    Max: int64
}

/// Inferred width for a node
type InferredWidth = {
    Interval: ValueInterval
    Bits: int
    IsSigned: bool
}

/// Complete width inference result (coeffect)
type WidthInferenceResult = {
    /// Per-node inferred widths for scalar values, keyed by NodeId int value
    NodeWidths: Map<int, InferredWidth>
    /// Per-node struct field widths, keyed by NodeId int value.
    /// Maps a struct-typed node's ID to [(fieldName, bits)].
    /// Populated from RecordExpr field value intervals, then propagated
    /// interprocedurally (call site → parameter, return → caller).
    /// No string keys — pure per-node codata.
    StructNodeWidths: Map<int, (string * int) list>
}

// ═══════════════════════════════════════════════════════════════════════════
// BIT WIDTH COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

/// Minimum unsigned bits to represent value v ≥ 0.
/// ceil(log2(v+1)), with floor cases: 0→1, 1→1.
let private unsignedBitsFor (v: int64) : int =
    if v <= 1L then 1
    else
        let mutable bits = 0
        let mutable x = v
        while x > 0L do
            bits <- bits + 1
            x <- x >>> 1
        bits

/// Minimum two's complement signed bits for range [lo, hi].
///
/// Two's complement iN represents [-2^(N-1), 2^(N-1)-1].
/// Key: the negative side holds one more value than the positive side.
///   i1:  [-1, 0]  (boolean convention: 0/1 maps to i1)
///   i8:  [-128, 127]
///   i10: [-512, 511]
///
/// For non-negative [0, max]:
///   max ≤ 1         → i1 (boolean)
///   max > 1         → i(unsignedBitsFor(max) + 1) — sign bit keeps MSB non-negative
///
/// For mixed/negative ranges:
///   Positive side needs N where 2^(N-1)-1 ≥ hi   → N = unsignedBitsFor(hi) + 1
///   Negative side needs N where 2^(N-1)   ≥ |lo| → N = unsignedBitsFor(|lo| - 1) + 1
///   (The -1 accounts for two's complement asymmetry: -128 fits in i8 because 2^7 = 128)
///   Take the max of both sides.
///
/// Round-trip stable with maxPositiveForBits:
///   bitsFromInterval [0, maxPositiveForBits(N)] = N for all N ≥ 1.
let private minSignedBits (lo: int64) (hi: int64) : int =
    if lo >= 0L && hi <= 1L then 1  // boolean
    elif lo >= 0L then
        unsignedBitsFor hi + 1
    elif hi < 0L then
        // All negative: need N where -2^(N-1) ≤ lo, i.e. 2^(N-1) ≥ |lo|
        unsignedBitsFor (abs lo - 1L) + 1
    else
        // Mixed: both sides contribute
        let posBits = unsignedBitsFor hi + 1
        let negBits = unsignedBitsFor (abs lo - 1L) + 1
        max posBits negBits

/// Maximum positive value representable in N signed two's complement bits.
/// Inverse of minSignedBits for non-negative intervals: maxPositiveForBits(minSignedBits(0, v)) ≥ v.
///   i1: 1 (boolean). iN (N>1): 2^(N-1) - 1.
let private maxPositiveForBits (bits: int) : int64 =
    if bits <= 1 then 1L
    else (1L <<< (bits - 1)) - 1L

/// Compute minimum signed two's complement bits from an interval.
/// Returns (bits, isSigned) where isSigned indicates negative values in the range.
let private bitsFromInterval (interval: ValueInterval) : int * bool =
    let bits = minSignedBits interval.Min interval.Max
    let isSigned = interval.Min < 0L
    (bits, isSigned)

// ═══════════════════════════════════════════════════════════════════════════
// INTERVAL ARITHMETIC
// ═══════════════════════════════════════════════════════════════════════════

let private addIntervals (a: ValueInterval) (b: ValueInterval) : ValueInterval =
    { Min = a.Min + b.Min; Max = a.Max + b.Max }

let private subIntervals (a: ValueInterval) (b: ValueInterval) : ValueInterval =
    { Min = a.Min - b.Max; Max = a.Max - b.Min }

let private mulIntervals (a: ValueInterval) (b: ValueInterval) : ValueInterval =
    // Four corners for general case
    let products = [
        a.Min * b.Min
        a.Min * b.Max
        a.Max * b.Min
        a.Max * b.Max
    ]
    { Min = List.min products; Max = List.max products }

/// Division of two intervals. Four-corner analysis for non-zero-crossing divisors.
let private divIntervals (a: ValueInterval) (b: ValueInterval) : ValueInterval =
    if b.Min <= 0L && b.Max >= 0L then a  // divisor spans zero: can't bound
    else
        let corners = [
            a.Min / b.Min; a.Min / b.Max
            a.Max / b.Min; a.Max / b.Max ]
        { Min = List.min corners; Max = List.max corners }

let private modInterval (_a: ValueInterval) (k: int64) : ValueInterval =
    // x % K always in [0, K-1] for positive K (F# semantics for natural values)
    if k > 0L then { Min = 0L; Max = k - 1L }
    else { Min = 0L; Max = abs k - 1L }

// ═══════════════════════════════════════════════════════════════════════════
// STRUCT FIELD MAXIMA — Pure combinators for merge-by-name-take-max
// ═══════════════════════════════════════════════════════════════════════════

/// Merge two struct field max-value lists: take max per field, union on names.
/// This is the algebraic join operation for the struct width lattice.
let private mergeFieldMaxima (existing: (string * int64) list) (incoming: (string * int64) list)
                             : (string * int64) list =
    if incoming.IsEmpty then existing
    elif existing.IsEmpty then incoming
    else
        let existingMap = Map.ofList existing
        incoming
        |> List.fold (fun acc (name, maxVal) ->
            match Map.tryFind name acc with
            | Some old -> Map.add name (max old maxVal) acc
            | None -> Map.add name maxVal acc) existingMap
        |> Map.toList

// ═══════════════════════════════════════════════════════════════════════════
// ANALYSIS
// ═══════════════════════════════════════════════════════════════════════════

/// Infer value intervals and minimum bit widths for all PSG nodes
let analyze (graph: SemanticGraph) : WidthInferenceResult =
    let mutable intervals : Map<int, ValueInterval> = Map.empty

    /// Try to get interval for a node (already computed)
    let tryGetInterval (NodeId id) = Map.tryFind id intervals

    /// Record an interval for a node; MERGES (widens) with any existing interval.
    /// This guarantees monotonic convergence: intervals only grow, never shrink.
    let mutable intervalsChanged = false
    let recordInterval (NodeId id) (interval: ValueInterval) =
        match Map.tryFind id intervals with
        | Some existing ->
            let merged = { Min = min existing.Min interval.Min; Max = max existing.Max interval.Max }
            if merged <> existing then
                intervals <- Map.add id merged intervals
                intervalsChanged <- true
        | None ->
            intervals <- Map.add id interval intervals
            intervalsChanged <- true

    // Phase 1: Walk all nodes, compute intervals bottom-up
    // We do two passes: first constants/leaves, then operations that depend on them
    for KeyValue(nodeId, node) in graph.Nodes do
        if not node.IsReachable then () else

        match node.Kind with
        // Integer literals — exact interval
        | SemanticKind.Literal (NativeLiteral.Int (value, _kind)) ->
            recordInterval nodeId { Min = value; Max = value }

        | SemanticKind.Literal (NativeLiteral.UInt (value, _kind)) ->
            let v = int64 value
            recordInterval nodeId { Min = v; Max = v }

        // Boolean literals — [0, 1]
        | SemanticKind.Literal (NativeLiteral.Bool _) ->
            recordInterval nodeId { Min = 0L; Max = 1L }

        // Char literals — [0, 1114111] (Unicode max)
        | SemanticKind.Literal (NativeLiteral.Char c) ->
            let v = int64 c
            recordInterval nodeId { Min = v; Max = v }

        // DU tag extraction — [0, numCases-1]
        | SemanticKind.DUGetTag (_duValue, duType) ->
            match duType with
            | NativeType.TUnion (_tycon, cases) ->
                let caseCount = int64 (List.length cases)
                recordInterval nodeId { Min = 0L; Max = max 0L (caseCount - 1L) }
            | _ -> ()

        // Union case construction — the tag value is the case index
        | SemanticKind.UnionCase (_caseName, caseIndex, _payload) ->
            recordInterval nodeId { Min = int64 caseIndex; Max = int64 caseIndex }

        // Boolean-producing nodes
        | SemanticKind.TypeTest _ ->
            recordInterval nodeId { Min = 0L; Max = 1L }

        | _ -> ()

    // Helper: Find the body's last value-producing node
    // Traverses through Sequential (to last child) and TypeAnnotation (unwrap)
    let rec findLastValue (nid: NodeId) : NodeId =
        match Map.tryFind nid graph.Nodes with
        | Some n ->
            match n.Kind with
            | SemanticKind.Sequential children ->
                match List.tryLast children with
                | Some lastId -> findLastValue lastId
                | None -> nid
            | SemanticKind.TypeAnnotation (wrappedId, _) ->
                findLastValue wrappedId
            | _ -> nid
        | None -> nid

    // Helper: Resolve Application → VarRef → Binding → Lambda(params)
    let resolveLambdaParams (funcNodeId: NodeId) : (string * NativeType * NodeId) list option =
        match Map.tryFind funcNodeId graph.Nodes with
        | Some { Kind = SemanticKind.VarRef (_, Some defId) } ->
            match Map.tryFind defId graph.Nodes with
            | Some defNode ->
                match defNode.Kind with
                | SemanticKind.Binding _ ->
                    defNode.Children |> List.tryPick (fun childId ->
                        match Map.tryFind childId graph.Nodes with
                        | Some { Kind = SemanticKind.Lambda (params', _, _, _, _) } -> Some params'
                        | _ -> None)
                | _ -> None
            | None -> None
        | _ -> None

    // Helper: Resolve Application → VarRef → Binding → Lambda → bodyId
    let resolveLambdaBodyId (funcNodeId: NodeId) : NodeId option =
        match Map.tryFind funcNodeId graph.Nodes with
        | Some { Kind = SemanticKind.VarRef (_, Some defId) } ->
            match Map.tryFind defId graph.Nodes with
            | Some defNode ->
                match defNode.Kind with
                | SemanticKind.Binding _ ->
                    defNode.Children |> List.tryPick (fun childId ->
                        match Map.tryFind childId graph.Nodes with
                        | Some { Kind = SemanticKind.Lambda (_, bodyId, _, _, _) } -> Some bodyId
                        | _ -> None)
                | _ -> None
            | None -> None
        | _ -> None

    // Phase 1B: Type-based intervals (booleans, DU types from NativeType)
    for KeyValue(nodeId, node) in graph.Nodes do
        if not node.IsReachable then () else
        let (NodeId id) = nodeId
        if Map.containsKey id intervals then () else

        match node.Type with
        | NativeType.TApp (tycon, _) ->
            match tycon.NTUKind with
            | Some NTUKind.NTUbool ->
                recordInterval nodeId { Min = 0L; Max = 1L }
            | _ ->
                if tycon.CaseCount > 0 then
                    recordInterval nodeId { Min = 0L; Max = int64 (tycon.CaseCount - 1) }

        | NativeType.TUnion (_tycon, cases) ->
            let caseCount = int64 (List.length cases)
            if caseCount > 0L then
                recordInterval nodeId { Min = 0L; Max = caseCount - 1L }

        | _ -> ()

    // ═══════════════════════════════════════════════════════════════════════════
    // MAIN CONVERGENCE LOOP
    // ═══════════════════════════════════════════════════════════════════════════
    // All scalar and struct phases iterate together until stable.
    // This handles nested function calls (step → applyCadenceButtons → clampPeriod),
    // struct width propagation through Bindings/VarRefs, and Mealy machine feedback.
    // Struct field max values: (fieldName, maxValue) during convergence.
    // Converted to (fieldName, bits) only at the end — no lossy round-trip.
    let mutable structFieldMaxima : Map<int, (string * int64) list> = Map.empty

    /// Propagate incoming field maxima into structFieldMaxima for a node.
    /// Pure merge via mergeFieldMaxima; only writes if the result changes.
    let propagateFieldMaxima (id: int) (incoming: (string * int64) list) =
        if incoming.IsEmpty then () else
        match Map.tryFind id structFieldMaxima with
        | Some existing ->
            let merged = mergeFieldMaxima existing incoming
            if merged <> existing then
                structFieldMaxima <- Map.add id merged structFieldMaxima
        | None ->
            structFieldMaxima <- Map.add id incoming structFieldMaxima

    /// Propagate from one node to another (transparent wrapper pattern).
    let propagateFieldMaximaFrom (sourceId: int) (targetId: int) =
        match Map.tryFind sourceId structFieldMaxima with
        | Some fields -> propagateFieldMaxima targetId fields
        | None -> ()

    let structFingerprint () =
        let count = structFieldMaxima.Count
        let totalMaxima = structFieldMaxima |> Map.fold (fun acc _ fields ->
            fields |> List.fold (fun a (_, v) -> a + v) acc) 0L
        (count, totalMaxima)

    let mutable mainChanged = true
    let mutable mainIter = 0
    while mainChanged && mainIter < 20 do
        mainChanged <- false
        mainIter <- mainIter + 1
        intervalsChanged <- false
        let prevIntervalCount = intervals.Count
        let prevStruct = structFingerprint ()

        // ── A: Scalar operations (arithmetic, comparison, VarRef, Binding) ──
        // No write-once guard: recordInterval merges (widens), so recomputation
        // is safe and necessary when operand intervals widen across iterations.
        for KeyValue(nodeId, node) in graph.Nodes do
            if not node.IsReachable then () else

            match node.Kind with
            | SemanticKind.Application (funcNodeId, argNodeIds) ->
                match Map.tryFind funcNodeId graph.Nodes with
                | Some funcNode ->
                    match funcNode.Kind with
                    | SemanticKind.Intrinsic info when info.Category = IntrinsicCategory.Arithmetic ->
                        match info.Module, info.Operation, argNodeIds with
                        | IntrinsicModule.Operators, "op_Addition", [lhsId; rhsId] ->
                            match tryGetInterval lhsId, tryGetInterval rhsId with
                            | Some lhs, Some rhs -> recordInterval nodeId (addIntervals lhs rhs)
                            | _ -> ()
                        | IntrinsicModule.Operators, "op_Subtraction", [lhsId; rhsId] ->
                            match tryGetInterval lhsId, tryGetInterval rhsId with
                            | Some lhs, Some rhs -> recordInterval nodeId (subIntervals lhs rhs)
                            | _ -> ()
                        | IntrinsicModule.Operators, "op_Multiply", [lhsId; rhsId] ->
                            match tryGetInterval lhsId, tryGetInterval rhsId with
                            | Some lhs, Some rhs -> recordInterval nodeId (mulIntervals lhs rhs)
                            | _ -> ()
                        | IntrinsicModule.Operators, "op_Division", [lhsId; rhsId] ->
                            match tryGetInterval lhsId, tryGetInterval rhsId with
                            | Some lhs, Some rhs when rhs.Min > 0L || rhs.Max < 0L ->
                                recordInterval nodeId (divIntervals lhs rhs)
                            | _ -> ()
                        | IntrinsicModule.Operators, "op_Modulus", [_lhsId; rhsId] ->
                            match tryGetInterval rhsId with
                            | Some rhs when rhs.Min > 0L ->
                                // x % y where y ∈ [ymin, ymax], ymin > 0 → result ∈ [0, ymax - 1]
                                recordInterval nodeId (modInterval { Min = 0L; Max = 0L } rhs.Max)
                            | _ -> ()
                        | _ -> ()
                    | SemanticKind.Intrinsic info when info.Category = IntrinsicCategory.Comparison ->
                        recordInterval nodeId { Min = 0L; Max = 1L }
                    | _ -> ()
                | None -> ()

            | SemanticKind.VarRef (_name, Some defId) ->
                match tryGetInterval defId with
                | Some interval -> recordInterval nodeId interval
                | None -> ()

            | SemanticKind.Binding _ ->
                match List.tryLast node.Children with
                | Some valueId ->
                    match tryGetInterval valueId with
                    | Some interval -> recordInterval nodeId interval
                    | None -> ()
                | None -> ()

            | SemanticKind.Sequential children ->
                match List.tryLast children with
                | Some lastId ->
                    match tryGetInterval lastId with
                    | Some interval -> recordInterval nodeId interval
                    | None -> ()
                | None -> ()

            | SemanticKind.TypeAnnotation (wrappedId, _) ->
                match tryGetInterval wrappedId with
                | Some interval -> recordInterval nodeId interval
                | None -> ()

            | _ -> ()

        // ── B: IfThenElse propagation ──
        // Union of branches with known intervals, widened across iterations.
        // No write-once guard: branches may resolve in later iterations,
        // requiring the union to widen (e.g. [100,100] → [100,500] as 'candidate' resolves).
        // recordInterval's change-tracking ensures convergence detection works correctly.
        for KeyValue(nodeId, node) in graph.Nodes do
            if not node.IsReachable then () else
            match node.Kind with
            | SemanticKind.IfThenElse (_condId, thenId, Some elseId) ->
                let branchIntervals = [ tryGetInterval thenId; tryGetInterval elseId ] |> List.choose id
                if not branchIntervals.IsEmpty then
                    let branchUnion = {
                        Min = branchIntervals |> List.map (fun iv -> iv.Min) |> List.min
                        Max = branchIntervals |> List.map (fun iv -> iv.Max) |> List.max }
                    // Merge with existing: only widen, never narrow
                    let merged =
                        match tryGetInterval nodeId with
                        | Some existing ->
                            { Min = min existing.Min branchUnion.Min
                              Max = max existing.Max branchUnion.Max }
                        | None -> branchUnion
                    recordInterval nodeId merged
            | SemanticKind.IfThenElse (_condId, thenId, None) ->
                match tryGetInterval thenId with
                | Some thenIv ->
                    let merged =
                        match tryGetInterval nodeId with
                        | Some existing ->
                            { Min = min existing.Min thenIv.Min
                              Max = max existing.Max thenIv.Max }
                        | None -> thenIv
                    recordInterval nodeId merged
                | None -> ()
            | _ -> ()

        // ── B2: Comparison-derived interval seeding ──
        // MUST run outside the write-once guard: comparison result nodes already
        // have [0,1] from Phase 1B (bool type), so Phase A's guarded loop skips them.
        // Both comparison operands need the same bit width in hardware.
        // Seed/widen both to the max of either range, and propagate to definitions
        // so all VarRefs of the same Binding see the widened interval.
        for KeyValue(_, node) in graph.Nodes do
            if not node.IsReachable then () else
            match node.Kind with
            | SemanticKind.Application (funcNodeId, argNodeIds) ->
                match Map.tryFind funcNodeId graph.Nodes with
                | Some { Kind = SemanticKind.Intrinsic info } when info.Category = IntrinsicCategory.Comparison ->
                    let seedWithMax (targetId: NodeId) (maxVal: int64) =
                        recordInterval targetId { Min = 0L; Max = maxVal }
                        // Also seed the definition if target is a VarRef
                        match Map.tryFind targetId graph.Nodes with
                        | Some { Kind = SemanticKind.VarRef (_, Some defId) } ->
                            recordInterval defId { Min = 0L; Max = maxVal }
                        | _ -> ()
                    match argNodeIds with
                    | [lhsId; rhsId] ->
                        match tryGetInterval lhsId, tryGetInterval rhsId with
                        | Some lhs, None ->
                            seedWithMax rhsId lhs.Max
                        | None, Some rhs ->
                            seedWithMax lhsId rhs.Max
                        | Some lhs, Some rhs ->
                            let maxMax = max lhs.Max rhs.Max
                            seedWithMax lhsId maxMax
                            seedWithMax rhsId maxMax
                        | None, None -> ()
                    | _ -> ()
                | _ -> ()
            | _ -> ()

        // ── C: Scalar interprocedural (arg → param) ──
        for KeyValue(_, node) in graph.Nodes do
            if not node.IsReachable then () else
            match node.Kind with
            | SemanticKind.Application (funcNodeId, argNodeIds) ->
                match resolveLambdaParams funcNodeId with
                | Some params' ->
                    let paramNodeIds = params' |> List.map (fun (_, _, pid) -> pid)
                    let pairCount = min (List.length argNodeIds) (List.length paramNodeIds)
                    let pairs = List.zip (List.truncate pairCount argNodeIds) (List.truncate pairCount paramNodeIds)
                    for (argId, paramId) in pairs do
                        match tryGetInterval argId with
                        | Some interval ->
                            let (NodeId paramIdVal) = paramId
                            match Map.tryFind paramIdVal intervals with
                            | Some existing ->
                                let merged = { Min = min existing.Min interval.Min; Max = max existing.Max interval.Max }
                                if merged <> existing then recordInterval paramId merged
                            | None -> recordInterval paramId interval
                        | None -> ()
                | None -> ()
            | _ -> ()

        // ── D: Scalar return propagation (body → call site) ──
        for KeyValue(nodeId, node) in graph.Nodes do
            if not node.IsReachable then () else
            match node.Kind with
            | SemanticKind.Application (funcNodeId, _) ->
                match resolveLambdaBodyId funcNodeId with
                | Some bodyId ->
                    let lastValueId = findLastValue bodyId
                    match tryGetInterval lastValueId with
                    | Some interval -> recordInterval nodeId interval
                    | None -> ()
                | None -> ()
            | _ -> ()

        // ── E: RecordExpr/TupleExpr → structFieldMaxima (store max values, not bits) ──
        for KeyValue(nodeId, node) in graph.Nodes do
            if not node.IsReachable then () else
            let (NodeId id) = nodeId
            match node.Kind with
            | SemanticKind.RecordExpr (fields, _) ->
                let fieldMaxima =
                    fields |> List.choose (fun (fieldName, valueId) ->
                        match tryGetInterval valueId with
                        | Some interval -> Some (fieldName, max 0L interval.Max)
                        | None -> None)
                propagateFieldMaxima id fieldMaxima
            | SemanticKind.TupleExpr elementIds ->
                let tupleMaxima =
                    elementIds |> List.mapi (fun i elemId ->
                        let itemName = sprintf "Item%d" (i + 1)
                        match tryGetInterval elemId with
                        | Some interval -> Some (itemName, max 0L interval.Max)
                        | None -> None)
                    |> List.choose (fun x -> x)
                propagateFieldMaxima id tupleMaxima
            | _ -> ()

        // ── F: Struct field maxima through transparent wrappers ──
        // Binding, VarRef, TypeAnnotation, Sequential, IfThenElse — propagate
        // struct field maxima from source to wrapper node via mergeFieldMaxima.
        for KeyValue(nodeId, node) in graph.Nodes do
            if not node.IsReachable then () else
            let (NodeId nid) = nodeId
            match node.Kind with
            | SemanticKind.Binding _ ->
                match List.tryLast node.Children with
                | Some (NodeId valId) -> propagateFieldMaximaFrom valId nid
                | None -> ()
            | SemanticKind.VarRef (_, Some (NodeId defIdVal)) ->
                propagateFieldMaximaFrom defIdVal nid
            | SemanticKind.TypeAnnotation ((NodeId wrappedIdVal), _) ->
                propagateFieldMaximaFrom wrappedIdVal nid
            | SemanticKind.Sequential children ->
                match List.tryLast children with
                | Some (NodeId lastIdVal) -> propagateFieldMaximaFrom lastIdVal nid
                | None -> ()
            | SemanticKind.IfThenElse (_condId, thenId, elseIdOpt) ->
                // Union struct field maxima from both branches
                let (NodeId thenIdVal) = thenId
                let branchIds = thenIdVal :: (elseIdOpt |> Option.map (fun (NodeId eid) -> eid) |> Option.toList)
                let merged =
                    branchIds |> List.fold (fun acc bid ->
                        match Map.tryFind bid structFieldMaxima with
                        | Some fields -> mergeFieldMaxima acc fields
                        | None -> acc) []
                propagateFieldMaxima nid merged
            | _ -> ()

        // ── G: FieldGet/TupleGet → scalar intervals (via structFieldMaxima) ──
        // Uses stored max values directly — no lossy bits→interval round-trip.
        // No write-once guard: struct field maxima may widen across iterations.
        for KeyValue(nodeId, node) in graph.Nodes do
            if not node.IsReachable then () else
            match node.Kind with
            | SemanticKind.FieldGet (objId, fieldName) ->
                let resolvedId =
                    match Map.tryFind objId graph.Nodes with
                    | Some { Kind = SemanticKind.VarRef (_, Some defId) } -> defId
                    | _ -> objId
                let (NodeId resId) = resolvedId
                match Map.tryFind resId structFieldMaxima with
                | Some fieldMaxima ->
                    match fieldMaxima |> List.tryFind (fun (n, _) -> n = fieldName) with
                    | Some (_, maxVal) ->
                        recordInterval nodeId { Min = 0L; Max = maxVal }
                    | None -> ()
                | None -> ()
            | SemanticKind.TupleGet (tupleId, index) ->
                // Tuple destructuring: extract scalar interval from tuple's struct field maxima.
                // Phase E stores tuple element maxima as "Item1", "Item2", etc.
                let resolvedId =
                    match Map.tryFind tupleId graph.Nodes with
                    | Some { Kind = SemanticKind.VarRef (_, Some defId) } -> defId
                    | _ -> tupleId
                let (NodeId resId) = resolvedId
                let itemName = sprintf "Item%d" (index + 1)
                match Map.tryFind resId structFieldMaxima with
                | Some fieldMaxima ->
                    match fieldMaxima |> List.tryFind (fun (n, _) -> n = itemName) with
                    | Some (_, maxVal) ->
                        recordInterval nodeId { Min = 0L; Max = maxVal }
                    | None -> ()
                | None -> ()
            | _ -> ()

        // ── H: Struct interprocedural (arg → param) ──
        for KeyValue(_, node) in graph.Nodes do
            if not node.IsReachable then () else
            match node.Kind with
            | SemanticKind.Application (funcNodeId, argNodeIds) ->
                match resolveLambdaParams funcNodeId with
                | Some params' ->
                    let paramNodeIds = params' |> List.map (fun (_, _, pid) -> pid)
                    let pairCount = min (List.length argNodeIds) (List.length paramNodeIds)
                    let pairs = List.zip (List.truncate pairCount argNodeIds) (List.truncate pairCount paramNodeIds)
                    for (NodeId argIdVal, NodeId paramIdVal) in pairs do
                        propagateFieldMaximaFrom argIdVal paramIdVal
                | None -> ()
            | _ -> ()

        // ── I: Struct return propagation (body → call site) ──
        for KeyValue(nodeId, node) in graph.Nodes do
            if not node.IsReachable then () else
            match node.Kind with
            | SemanticKind.Application (funcNodeId, _) ->
                match resolveLambdaBodyId funcNodeId with
                | Some bodyId ->
                    let (NodeId lastIdVal) = findLastValue bodyId
                    let (NodeId appIdVal) = nodeId
                    propagateFieldMaximaFrom lastIdVal appIdVal
                | None -> ()
            | _ -> ()

        // ── J: Lambda feedback (return → parameter) ──
        // For Mealy machines: step(state, inputs) → (newState, outputs)
        // Propagate return struct field maxima back to matching parameters.
        for KeyValue(_, node) in graph.Nodes do
            if not node.IsReachable then () else
            match node.Kind with
            | SemanticKind.Lambda (params', bodyId, _, _, _) ->
                let lastNodeId = findLastValue bodyId
                let (NodeId lastIdVal) = lastNodeId

                let tryPropagateToParam (fieldMaxima: (string * int64) list) =
                    if fieldMaxima.IsEmpty then () else
                    for (_, paramType, paramNodeId) in params' do
                        match paramType with
                        | NativeType.TApp(tycon, _) when tycon.FieldCount > 0 &&
                                                         fieldMaxima.Length <= tycon.FieldCount ->
                            let (NodeId paramIdVal) = paramNodeId
                            propagateFieldMaxima paramIdVal fieldMaxima
                        | _ -> ()

                // Case A: Direct — return struct matches a parameter
                match Map.tryFind lastIdVal structFieldMaxima with
                | Some returnMaxima -> tryPropagateToParam returnMaxima
                | None -> ()

                // Case B: Decompose TupleExpr/RecordExpr components
                match Map.tryFind lastNodeId graph.Nodes with
                | Some { Kind = SemanticKind.RecordExpr (fields, _) } ->
                    for (_, NodeId fieldValId) in fields do
                        match Map.tryFind fieldValId structFieldMaxima with
                        | Some inner -> tryPropagateToParam inner
                        | None -> ()
                | Some { Kind = SemanticKind.TupleExpr elementIds } ->
                    for (NodeId elemIdVal) in elementIds do
                        match Map.tryFind elemIdVal structFieldMaxima with
                        | Some inner -> tryPropagateToParam inner
                        | None -> ()
                | _ -> ()
            | _ -> ()

        let newIntervalCount = intervals.Count
        let newStruct = structFingerprint ()
        mainChanged <- intervalsChanged || newIntervalCount > prevIntervalCount || newStruct <> prevStruct

    // Converged — convert struct field maxima to bits (once, no round-trip)
    ignore (mainIter, intervals.Count, structFieldMaxima.Count)

    let structNodeWidths =
        structFieldMaxima |> Map.map (fun _ fields ->
            fields |> List.map (fun (name, maxVal) ->
                let (bits, _) = bitsFromInterval { Min = 0L; Max = maxVal }
                (name, max 1 bits)))

    // ═══════════════════════════════════════════════════════════════════════════
    // OPERAND WIDTH HARMONIZATION
    // ═══════════════════════════════════════════════════════════════════════════
    // In hardware, all operands of arithmetic/comparison must have the same width.
    // Widen operand bit widths to the maximum among siblings. This doesn't change
    // intervals — only the final Bits field in NodeWidths.
    let mutable nodeWidths =
        intervals
        |> Map.map (fun _id interval ->
            let (bits, isSigned) = bitsFromInterval interval
            { Interval = interval; Bits = max 1 bits; IsSigned = isSigned })

    let mutable harmChanged = true
    while harmChanged do
        harmChanged <- false
        for KeyValue(_, node) in graph.Nodes do
            if not node.IsReachable then () else
            match node.Kind with
            | SemanticKind.Application (funcNodeId, argNodeIds) ->
                match Map.tryFind funcNodeId graph.Nodes with
                | Some funcNode ->
                    match funcNode.Kind with
                    | SemanticKind.Intrinsic info when info.Category = IntrinsicCategory.Arithmetic
                                                    || info.Category = IntrinsicCategory.Comparison ->
                        let argWidths =
                            argNodeIds |> List.choose (fun (NodeId aid) ->
                                Map.tryFind aid nodeWidths |> Option.map (fun w -> (aid, w)))
                        if argWidths.Length > 0 then
                            let maxBits = argWidths |> List.map (fun (_, w) -> w.Bits) |> List.max
                            for (aid, w) in argWidths do
                                if w.Bits < maxBits then
                                    nodeWidths <- Map.add aid { w with Bits = maxBits } nodeWidths
                                    harmChanged <- true
                    | _ -> ()
                | None -> ()
            | _ -> ()

    { NodeWidths = nodeWidths; StructNodeWidths = structNodeWidths }