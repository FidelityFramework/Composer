/// SSA Assignment: the post-saturation coeffect nanopass
///
/// Runs once over the saturated graph, after every range and selection is settled and
/// before any witness runs, and derives each node's SSAs exactly from its structure:
/// the SSA count of a node is a deterministic function of the instance's shape. The
/// witnesses read the result as codata; nothing is minted, counted or allocated at
/// emission, and no witness or pattern holds a counter. A value that a witness emits
/// on a node's behalf (a meet, an extraction, a cast) is derived here for that node.
///
/// This is the coeffect the design places before traversal ("SSA Pre-assignment",
/// clef-lang-site docs/internals/pipeline/learning-to-walk.md; the coeffect table in
/// docs/CCS_Architecture.md). It is computed in Composer today and scheduled to move
/// into CCS as a hyperedge consequence; until then this pass is the interim computation
/// of a graph-resident fact, and the graph is the authority wherever it carries one.
///
/// Key design:
/// - Values are numbered per function: each Lambda boundary starts its own derivation
/// - Post-order derivation: a value's producers are derived before its uses
/// - Returns Map<NodeId, NodeSSAAllocation> that witnesses read (coeffect lookup, no generation during emission)
/// - Uses structured SSA type (V of int | Arg of int), not strings
/// - Knows MLIR expansion costs: one PSG node may need multiple SSAs
module PSGElaboration.SSAAssignment

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.NativeTypedTree.UnionFind
open Alex.Dialects.Core.Types
open Alex.CodeGeneration.TypeMapping
open PSGElaboration.Coeffects

// ═══════════════════════════════════════════════════════════════════════════
// STRUCTURAL SSA DERIVATION
// ═══════════════════════════════════════════════════════════════════════════
//
// SSA counts are derived from actual node STRUCTURE, not just node KIND.
// Since the graph is statically resolved, we know exactly what emission will do.
// This eliminates heuristics and prevents "not enough SSAs" errors.
//
// Key insight: SSA count is a deterministic function of instance structure.

/// Get the number of SSAs needed for a literal value
let private literalExpansionCost (lit: NativeLiteral) : int =
    match lit with
    | NativeLiteral.String _ -> 3  // memref.get_global (storage) + memref.reinterpret_cast (content view) + memref.cast (dynamic)
    | NativeLiteral.Unit -> 1     // constI
    | NativeLiteral.Bool _ -> 1   // constI
    | NativeLiteral.Int _ -> 1    // All integer types (int8..int64, uint8..uint64, nativeint)
    | NativeLiteral.UInt _ -> 1   // Unsigned integers > int64.MaxValue
    | NativeLiteral.Char _ -> 1
    | NativeLiteral.Float _ -> 1  // float32 and float64
    | NativeLiteral.Decimal _ -> 1
    | NativeLiteral.ByteArray _ -> 1
    | NativeLiteral.UInt16Array _ -> 1

/// SSA traversal context — bundles all invariant state for the recursive traversal.
/// Only `scope` and `nodeId` vary per call; everything else is created once in `assignSSA`.
type private SSAContext = {
    /// The target: the fabric leg holds a DU or a record as an hw.struct at its settled widths
    /// and derives no byte layout for it
    TargetPlatform: Core.Types.Dialects.TargetPlatform
    Arch: Architecture
    Graph: SemanticGraph
    ClosureLayouts: System.Collections.Generic.Dictionary<int, ClosureLayout>
    DULayouts: System.Collections.Generic.Dictionary<int, DULayout>
    InnerScopeAssignments: System.Collections.Generic.Dictionary<int, NodeSSAAllocation>
    /// The meets of nodes in nested lambda scopes, merged with InnerScopeAssignments
    InnerMeets: System.Collections.Generic.Dictionary<int, Meet list>
    /// The zero constant each unit-typed function returns, by Lambda NodeId.value
    UnitReturns: System.Collections.Generic.Dictionary<int, SSA>
    /// The return meet of each function whose last value is held narrower or wider than its
    /// result, by Lambda NodeId.value: the last value of the body's scope
    ReturnMeets: System.Collections.Generic.Dictionary<int, Meet>
    /// Curry flattening: the saturated calls (the target binding and every argument) and the
    /// partial applications, which emit nothing
    Curry: CurryFlattening.CurryFlatteningResult
}

/// The type a capture is held at in its closure slot: a read of the captured binding's emitted
/// type (`TypeMapping.mapNativeTypeForTarget`, the bare integer kind at the source node's held
/// width, `TypeMapping.nodeWidth`). A mutable capture holds the address of its cell; a string
/// is decomposed into its base index and its extent (two words, a layout derived here from the
/// declared Pointer width); every other memref-backed value (an array, a record, a tuple) holds
/// its base index.
let private captureSlotType (platform: Core.Types.Dialects.TargetPlatform) (arch: Architecture) (graph: SemanticGraph) (capture: CaptureInfo) : MLIRType =
    if capture.IsMutable then
        // Mutable capture: store pointer to the alloca
        TIndex
    else
        match mapNativeTypeForTarget platform arch graph capture.Type with
        | TInt (IntWidth 0) ->
            match capture.SourceNodeId |> Option.bind (nodeWidth graph) with
            | Some w -> TInt w
            | None ->
                failwithf "SSAAssignment: capture '%s' of the bare integer kind has no source node whose held width the slot can read" capture.Name
        | TMemRef (TInt (IntWidth 8)) ->
            // String: decomposed memref capture — {data_ptr, length} as two platform words.
            // MLIR memref descriptors can't nest inside byte-level structs; we decompose on
            // construction and reconstruct on extraction.
            let word = declaredPointerBytes arch
            TStruct ([("ptr", TIndex); ("len", TIndex)], Some { Offsets = [0; word]; Size = 2 * word; Align = word })
        | TMemRef _ | TMemRefStatic _ | TStruct _ -> TIndex  // memref-backed: the base index
        | other -> other

/// A slot that holds the base index of a memref value: the capture's type is a memref in
/// MLIR (an array, a mutable cell's alloca) and the slot is an index, so the
/// construction extracts the base pointer before storing it. Read from the same type
/// mapping the witnesses use; the witness checks the accumulator's type against this.
let private captureExtractsBasePointer (platform: Core.Types.Dialects.TargetPlatform) (arch: Architecture) (graph: SemanticGraph) (capture: CaptureInfo) (slotType: MLIRType) : bool =
    slotType = TIndex &&
    (capture.IsMutable ||
     (match mapNativeTypeForTarget platform arch graph capture.Type with
      | TMemRef _ | TMemRefStatic _ -> true
      | _ -> false))

/// Construction SSA cost per capture, derived from the slot.
/// A decomposed memref capture (a string) stores {ptr, len} separately: ExtractBasePtr,
/// Dim and two typed stores. A slot extracting a base pointer takes the extraction value
/// and its view. A scalar takes its view (for reinterpret_cast).
let private captureConstructionSSACount (slotType: MLIRType) (extractsBasePointer: bool) : int =
    match slotType with
    | TStruct ([("ptr", TIndex); ("len", TIndex)], _) -> 5  // ptrSSA, dimZeroSSA, lenSSA, ptrViewSSA, lenViewSSA
    | _ when extractsBasePointer -> 2  // viewSSA, extractSSA
    | _ -> 1  // viewSSA

/// Extraction work SSA cost per capture (excludes the result SSA).
/// Work SSAs follow the capture result SSAs [V(0)..V(n-1)].
let private captureExtractionWorkSSACount (slotType: MLIRType) : int =
    match slotType with
    | TStruct ([("ptr", TIndex); ("len", TIndex)], _) -> 7  // ptrView,ptrZero,ptr, lenView,lenZero,len, rawMemref
    | _ -> 2  // view, zero

/// The callee prologue of a closure, in the order pExtractCaptures consumes it: per capture
/// its work values then its result (the results are V 0 .. V (n-1); the work follows them),
/// and then the two env-reconstruction values (the memref view of Arg 0, its static cast).
let private closurePrologue (captureSlots: MLIRType list) : SSA list list * (SSA * SSA) =
    let n = captureSlots.Length
    let perCapture, totalWork =
        captureSlots
        |> List.mapi (fun i slot -> (i, slot))
        |> List.fold (fun (acc, work) (i, slot) ->
            let w = captureExtractionWorkSSACount slot
            (acc @ [ List.init w (fun k -> V (n + work + k)) @ [ V i ] ], work + w)) ([], 0)
    perCapture, (V (n + totalWork), V (n + totalWork + 1))

/// The number of values the callee prologue takes; the body's own values follow it
let private closurePrologueCount (captureSlots: MLIRType list) : int =
    let perCapture, _ = closurePrologue captureSlots
    (perCapture |> List.sumBy List.length) + 2

/// A unit-typed body: the function returns no value and its func.return needs a zero
/// constant, derived here as the first value of the body's scope
let private isUnitTyped (ty: NativeType) : bool =
    let rec go t =
        match t with
        | NativeType.TApp ({ NTUKind = Some NTUKind.NTUunit }, []) -> true
        | NativeType.TVar tv ->
            match find tv with
            | (_, Some bound) -> go bound
            | (_, None) -> false
        | _ -> false
    go ty

/// Compute exact SSA count for Lambda based on captures list
/// This is DETERMINISTIC - derived directly from PSG structure (captures list from CCS)
///
/// CLOSURE CONSTRUCTION with heap allocation:
/// - Simple Lambda (0 captures): 0 SSAs (emits func.func, no local value needed)
/// - Closing Lambda (C construction SSAs across all captures):
///   Flat struct construction (C + 3):
///     - 1 SSA: addressof for code pointer
///     - 1 SSA: undef for flat closure struct
///     - 1 SSA: insertvalue for code_ptr at [0]
///     - C SSAs: per-capture construction SSAs (1 for scalar, 5 for decomposed memref)
///   Heap allocation (5):
///     - 5 SSAs: posPtrSSA, posSSA, heapBaseSSA, resultPtrSSA, newPosSSA
///   Size computation (3):
///     - 3 SSAs: gepSSA, sizeSSA, oneSSA (size computed at compile time, no null GEP trick)
///   Uniform pair construction (3):
///     - 3 SSAs: pairUndefSSA, pairWithCodeSSA, closureResultSSA
///   Total: C + 14 SSAs
let private computeLambdaSSACost (platform: Core.Types.Dialects.TargetPlatform) (arch: Architecture) (graph: SemanticGraph) (captures: CaptureInfo list) : int =
    let n = List.length captures
    if n = 0 then
        0  // Simple function - no closure struct needed
    else
        let c =
            captures |> List.sumBy (fun cap ->
                let slot = captureSlotType platform arch graph cap
                captureConstructionSSACount slot (captureExtractsBasePointer platform arch graph cap slot))
        c + 14  // flat struct (c+3) + heap (5) + size (3) + pair (3)

/// The byte layout of a closure's environment, derived once here from its slots: each slot at
/// the next offset, the whole a byte memref. A closure is an aggregate the leg realises (no
/// Clef type names it), so its layout is this nanopass's derivation from the slot types, each of
/// which is a read of a selected width or of the declared Pointer width; the witnesses read the
/// offsets from the layout.
let private tileSlots (arch: Architecture) (slotTypes: MLIRType list) : int list * int =
    slotTypes
    |> List.fold (fun (offsets, cursor) slot -> (cursor :: offsets, cursor + mlirTypeSize arch slot)) ([], 0)
    |> fun (offsets, size) -> (List.rev offsets, size)

/// Build the environment struct type from captures
let private buildEnvStructType (platform: Core.Types.Dialects.TargetPlatform) (arch: Architecture) (graph: SemanticGraph) (captures: CaptureInfo list) : MLIRType =
    let slotTypes = captures |> List.map (captureSlotType platform arch graph)
    let (_, totalBytes) = tileSlots arch slotTypes
    TMemRefStatic(totalBytes, TInt (IntWidth 8))

/// Build complete ClosureLayout from Lambda captures and pre-assigned SSAs
/// This is computed once during SSAAssignment - witnesses observe the result
///
/// TRUE FLAT CLOSURE SSA layout for N captures (total N+3 SSAs):
///   ssas[0]           = addressof code_ptr
///   ssas[1]           = undef closure struct
///   ssas[2]           = insertvalue code_ptr at [0]
///   ssas[3..N+2]      = insertvalue for each capture at [1..N]
///   ssas[N+2]         = final result (last insertvalue)
///
/// NOTE: Capture EXTRACTION SSAs (used in callee) are derived at emission time
/// from capture count, not pre-allocated here. This cleanly separates:
/// - Parent scope: closure CONSTRUCTION (these SSAs)
/// - Child scope: capture EXTRACTION (v0..v(N-1), derived from PSG structure)
let private buildClosureLayout
    (platform: Core.Types.Dialects.TargetPlatform)
    (arch: Architecture)
    (graph: SemanticGraph)
    (lambdaNodeId: NodeId)
    (bodyNodeId: NodeId)
    (captures: CaptureInfo list)
    (ssas: SSA list)
    (context: LambdaContext)
    : ClosureLayout =

    let n = List.length captures
    let captureTypes = captures |> List.map (captureSlotType platform arch graph)
    let extracts = List.map2 (captureExtractsBasePointer platform arch graph) captures captureTypes

    // Total construction SSAs varies by capture: 1 for a scalar, 2 for a base-pointer
    // extraction, 5 for a decomposed memref
    let perCaptureCounts = List.map2 captureConstructionSSACount captureTypes extracts
    let c = List.sum perCaptureCounts

    // Extract SSAs by position for closure CONSTRUCTION
    // Flat struct construction (0 to c+2)
    let codeAddrSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withCodeSSA = ssas.[2]
    // The construction values per capture, in slot order, sliced by each capture's count
    let captureInsertSSAs =
        perCaptureCounts
        |> List.fold (fun (acc, position) count -> (ssas.[position .. position + count - 1] :: acc, position + count)) ([], 3)
        |> fst |> List.rev

    // Heap allocation (c+3 to c+7)
    let heapPosPtrSSA = ssas.[c + 3]
    let heapPosSSA = ssas.[c + 4]
    let heapBaseSSA = ssas.[c + 5]
    let heapResultPtrSSA = ssas.[c + 6]
    let heapNewPosSSA = ssas.[c + 7]

    // Size computation (c+8 to c+10) - no null GEP trick, size computed at compile time
    let sizeGepSSA = ssas.[c + 8]
    let sizeSSA = ssas.[c + 9]
    let sizeOneSSA = ssas.[c + 10]

    // Uniform pair construction (c+11 to c+13)
    let pairUndefSSA = ssas.[c + 11]
    let pairWithCodeSSA = ssas.[c + 12]
    let closureResultSSA = ssas.[c + 13]

    // Build env struct type (for internal tracking, kept for compatibility)
    let envStructType = buildEnvStructType platform arch graph captures

    // TRUE FLAT CLOSURE: {code_ptr, capture_0, capture_1, ...}
    // Captures are inlined directly, not via env_ptr indirection
    // This eliminates lifetime issues - closure is returned by value with all state inline.
    // The header before the first capture: the code pointer (a regular closure), or the lazy
    // header {computed: i1, value: T, code_ptr} or the seq header {state: i32, current, code_ptr}.
    let header =
        match context with
        | LambdaContext.RegularClosure -> [ TIndex ]
        | LambdaContext.LazyThunk ->
            match SemanticGraph.tryGetNode bodyNodeId graph with
            | Some bodyNode ->
                let elementType =
                    match mapNativeTypeForTarget platform arch graph bodyNode.Type with
                    | TInt (IntWidth 0) -> TInt (requireNodeWidth graph bodyNodeId)
                    | mapped -> mapped
                [ TInt (IntWidth 1); elementType; TIndex ]
            | None -> failwithf "LazyThunk Lambda body node %d not found" (NodeId.value bodyNodeId)
        | LambdaContext.SeqGenerator -> [ TInt (IntWidth 32); TIndex; TIndex ]
    let (offsets, totalBytes) = tileSlots arch (header @ captureTypes)
    let captureOffsets = offsets |> List.skip header.Length
    let closureStructType = TMemRefStatic((tileSlots arch (TIndex :: captureTypes) |> snd), TInt (IntWidth 8))

    // Build capture slots: the slot's type, its byte offset in the struct, whether it extracts
    let captureSlots =
        captures
        |> List.mapi (fun i capture ->
            {
                Name = capture.Name
                SlotIndex = i
                SlotType = captureTypes.[i]
                ByteOffset = captureOffsets.[i]
                SourceNodeId = capture.SourceNodeId
                Mode = if capture.IsMutable then ByRef else ByValue
                ExtractsBasePointer = extracts.[i]
            })

    // PRD-14 Option B: For lazy thunks, the FULL lazy struct type
    // {computed: i1, value: T, code_ptr: ptr, cap0, cap1, ...}
    let lazyStructType =
        match context with
        | LambdaContext.LazyThunk -> Some (TMemRefStatic(totalBytes, TInt (IntWidth 8)))
        | _ -> None

    // StructLoadSSA is for the CALLEE (inner function) - not from parent scope's ssas
    // It's V(captureCount) because extraction SSAs are v0..v(N-1), body starts at v(N+1)
    let structLoadSSA = V n

    // The callee prologue, derived once here and read by LambdaWitness
    let captureExtractionSSAs, envReconstructionSSAs = closurePrologue captureTypes

    {
        LambdaNodeId = lambdaNodeId
        Captures = captureSlots
        CodeAddrSSA = codeAddrSSA
        ClosureUndefSSA = undefSSA
        ClosureWithCodeSSA = withCodeSSA
        CaptureInsertSSAs = captureInsertSSAs
        HeapPosPtrSSA = heapPosPtrSSA
        HeapPosSSA = heapPosSSA
        HeapBaseSSA = heapBaseSSA
        HeapResultPtrSSA = heapResultPtrSSA
        HeapNewPosSSA = heapNewPosSSA
        SizeGepSSA = sizeGepSSA
        SizeSSA = sizeSSA
        SizeOneSSA = sizeOneSSA
        PairUndefSSA = pairUndefSSA
        PairWithCodeSSA = pairWithCodeSSA
        ClosureResultSSA = closureResultSSA
        StructLoadSSA = structLoadSSA
        CaptureExtractionSSAs = captureExtractionSSAs
        EnvReconstructionSSAs = envReconstructionSSAs
        EnvStructType = envStructType
        ClosureStructType = closureStructType
        Context = context
        LazyStructType = lazyStructType
    }

/// Build complete DULayout from DUConstruct node and pre-assigned SSAs
/// This follows the flat closure model: build case-specific struct, store to arena, return pointer
///
/// SSA layout for DU with payload (13 SSAs total):
///   ssas[0]           = undef case struct
///   ssas[1]           = tag constant
///   ssas[2]           = insertvalue tag at [0]
///   ssas[3]           = insertvalue payload at [1]
///   ssas[4..6]        = size computation (const1, gep, size from compile-time)
///   ssas[7..11]       = arena allocation (posPtrSSA, posSSA, baseSSA, resultPtrSSA, newPosSSA)
///
/// SSA layout for nullary DU (11 SSAs total):
///   ssas[0]           = undef case struct
///   ssas[1]           = tag constant
///   ssas[2]           = insertvalue tag at [0]
///   ssas[3..5]        = size computation
///   ssas[6..10]       = arena allocation
let private buildDULayout
    (platform: Core.Types.Dialects.TargetPlatform)
    (arch: Architecture)
    (graph: SemanticGraph)
    (duConstructNodeId: NodeId)
    (duType: NativeType)
    (caseName: string)
    (caseIndex: int)
    (payloadOpt: NodeId option)
    (ssas: SSA list)
    : DULayout =

    let hasPayload = Option.isSome payloadOpt

    // Extract SSAs by position
    // Struct construction
    let structUndefSSA = ssas.[0]
    let tagConstSSA = ssas.[1]
    let withTagSSA = ssas.[2]
    let withPayloadSSA, sizeOffset =
        if hasPayload then
            Some ssas.[3], 4
        else
            None, 3

    // Size computation (3 SSAs - no null GEP trick, size computed at compile time)
    let sizeOneSSA = ssas.[sizeOffset]
    let sizeGepSSA = ssas.[sizeOffset + 1]
    let sizeSSA = ssas.[sizeOffset + 2]

    // Arena allocation (5 SSAs)
    let arenaOffset = sizeOffset + 3
    let heapPosPtrSSA = ssas.[arenaOffset]
    let heapPosSSA = ssas.[arenaOffset + 1]
    let heapBaseSSA = ssas.[arenaOffset + 2]
    let heapResultPtrSSA = ssas.[arenaOffset + 3]
    let heapNewPosSSA = ssas.[arenaOffset + 4]

    // The payload's type: the payload node's emitted type (the bare integer kind at the node's
    // held width)
    let payloadType =
        payloadOpt
        |> Option.bind (fun payloadId -> Map.tryFind payloadId graph.Nodes |> Option.map (fun node -> payloadId, node))
        |> Option.map (fun (payloadId, node) ->
            match mapNativeTypeForTarget platform arch graph node.Type with
            | TInt (IntWidth 0) -> TInt (requireNodeWidth graph payloadId)
            | mapped -> mapped)

    // The union's settled bytes, read from the graph's layouts: the tag and the widest payload
    let caseStructType =
        match mapNativeTypeForTarget platform arch graph duType with
        | TMemRefStatic _ as settled -> settled
        | other -> failwithf "SSAAssignment: the union '%s' maps to %A, not to its settled byte memref" (formatType duType) other

    {
        DUConstructNodeId = duConstructNodeId
        CaseName = caseName
        CaseIndex = caseIndex
        HasPayload = hasPayload
        StructUndefSSA = structUndefSSA
        TagConstSSA = tagConstSSA
        WithTagSSA = withTagSSA
        WithPayloadSSA = withPayloadSSA
        SizeOneSSA = sizeOneSSA
        SizeGepSSA = sizeGepSSA
        SizeSSA = sizeSSA
        HeapPosPtrSSA = heapPosPtrSSA
        HeapPosSSA = heapPosSSA
        HeapBaseSSA = heapBaseSSA
        HeapResultPtrSSA = heapResultPtrSSA
        HeapNewPosSSA = heapNewPosSSA
        CaseStructType = caseStructType
        PayloadType = payloadType
    }

/// Calculate SSA cost for interpolated string based on parts
let private interpolatedStringCost (parts: InterpolatedPart list) : int =
    // Count string parts (each needs 5 SSAs for fat pointer construction)
    let stringPartCount =
        parts |> List.sumBy (fun p ->
            match p with
            | InterpolatedPart.StringPart _ -> 1
            | InterpolatedPart.ExprPart _ -> 0)  // Already computed, no new SSAs

    // Concatenations: each needs 10 SSAs (4 extract, 1 add, 1 alloca, 1 gep, 3 build)
    let concatCount = max 0 (List.length parts - 1)

    // Total: 5 per string part + 10 per concat
    (stringPartCount * 5) + (concatCount * 10)

/// Count tuple elements from a scrutinee type
/// Returns 1 for non-tuple types (single DU), N for N-element tuples
let private countScrutineeTupleElements (graph: SemanticGraph) (scrutineeId: NodeId) : int =
    match Map.tryFind scrutineeId graph.Nodes with
    | Some node ->
        match node.Type with
        | NativeType.TTuple (elements, _) -> List.length elements
        | _ -> 1  // Non-tuple scrutinee = single DU
    | None -> 1

/// Count tags in a pattern (for tuple patterns, count nested Union patterns)
let rec private countPatternTags (pattern: Pattern) : int =
    match pattern with
    | Pattern.Union _ -> 1
    | Pattern.Tuple elements -> elements |> List.sumBy countPatternTags
    | Pattern.Var _ | Pattern.Wildcard -> 0
    | _ -> 0

/// Compute exact SSA count for Match expression from its structure
/// This mirrors what ControlFlowWitness.witnessMatch actually emits
let private computeMatchSSACost (graph: SemanticGraph) (scrutineeId: NodeId) (cases: MatchCase list) : int =
    // Determine pattern complexity from actual cases
    let numTags =
        match cases with
        | case :: _ -> countPatternTags case.Pattern
        | [] -> 1

    let numCases = List.length cases

    // Tag extraction SSAs (mirrors lines 224-251 in ControlFlowWitness.fs)
    let extractionSSAs =
        if numTags <= 1 then
            1  // Single tag extraction
        else
            // Tuple pattern: extract each element (N) + extract each tag (N)
            numTags * 2

    // Tag comparison SSAs per case (mirrors buildTagComparison)
    // For each tag: expectedSSA + cmpSSA = 2
    // For multiple tags: add (numTags - 1) AND operations
    let comparisonSSAsPerCase =
        if numTags <= 1 then
            2  // Single tag: expected + cmp
        else
            (numTags * 2) + max 0 (numTags - 1)  // 2 per tag + ANDs

    // If-chain SSAs (mirrors buildIfChain)
    // Each non-final case: comparison SSAs + potentially if/else structure
    // Final case: just body (no comparison)
    // Plus result phi and zero constants for void cases
    let ifChainSSAs =
        let nonFinalCases = max 0 (numCases - 1)
        // Each branch may need: result accumulation + zero for void
        (nonFinalCases * comparisonSSAsPerCase) + (numCases * 2) + 5

    // Total: extraction + comparisons + if-chain + buffer
    extractionSSAs + ifChainSSAs + 10  // 10 for safety margin

/// Compute exact SSA count for Application based on intrinsic analysis
let private computeApplicationSSACost (ctx: SSAContext) (node: SemanticNode) : int =
    // Check if this is a saturated call (curry flattening) — use effective arg count
    match Map.tryFind node.Id ctx.Curry.SaturatedCalls with
    | Some info ->
        1 + List.length info.AllArgNodes  // 1 result + N potential memref casts
    | None ->
    // Look at what we're applying to determine SSA needs
    match node.Children with
    | funcId :: _ ->
        match Map.tryFind funcId ctx.Graph.Nodes with
        | Some funcNode ->
            match funcNode.Kind with
            | SemanticKind.Intrinsic info ->
                // Intrinsics have known SSA costs based on operation
                match info.Module, info.Operation with
                | IntrinsicModule.Format, "int" -> 2      // func.call + result (platform library function)
                | IntrinsicModule.Format, "int64" -> 2    // func.call + result (platform library function)
                | IntrinsicModule.Format, "float" -> 2    // func.call + result (platform library function)
                | IntrinsicModule.Parse, "int" -> 2      // func.call + result (platform library function)
                | IntrinsicModule.Parse, "float" -> 2    // func.call + result (platform library function)
                | IntrinsicModule.String, "charAt" -> 3   // index_cast + memref.load + extui
                | IntrinsicModule.String, "length" -> 3   // dim_const + memref.dim + index_cast
                | IntrinsicModule.String, "contains" -> 20 // 6 setup + 5 cond + 7 body + 2 post (scf.while byte scan, TIndex iteration)
                | IntrinsicModule.String, "concat2" -> 18  // concatenation (18 SSAs - pure index arithmetic, NO i64 round-trip)
                | IntrinsicModule.String, "concat" -> 18   // alias for concat2
                | IntrinsicModule.String, _ -> 20          // other String ops
                | IntrinsicModule.Sys, "write" -> 6        // FFI extraction + length (2 ptr + 3 len + 1 result)
                | IntrinsicModule.Sys, "read" -> 6         // FFI extraction + capacity (2 ptr + 3 cap + 1 result)
                | IntrinsicModule.Sys, _ -> 16             // other syscalls (clock_gettime needs 16 for ms computation)
                | IntrinsicModule.DateTime, "now" -> 16    // delegates to clock_gettime
                | IntrinsicModule.DateTime, "utcNow" -> 16 // delegates to clock_gettime
                | IntrinsicModule.DateTime, "hour" -> 5    // const + div + rem + trunc
                | IntrinsicModule.DateTime, "minute" -> 5
                | IntrinsicModule.DateTime, "second" -> 5
                | IntrinsicModule.DateTime, "millisecond" -> 3  // just mod + trunc
                | IntrinsicModule.DateTime, "toTimeString" -> 60  // complex formatting
                | IntrinsicModule.DateTime, "toDateString" -> 60
                | IntrinsicModule.DateTime, "toString" -> 80
                | IntrinsicModule.DateTime, "toDateTimeString" -> 100  // full ISO 8601 format
                | IntrinsicModule.DateTime, _ -> 20        // other DateTime ops
                | IntrinsicModule.TimeSpan, "fromMilliseconds" -> 2
                | IntrinsicModule.TimeSpan, "fromSeconds" -> 3
                | IntrinsicModule.TimeSpan, "hours" -> 5
                | IntrinsicModule.TimeSpan, "minutes" -> 5
                | IntrinsicModule.TimeSpan, "seconds" -> 5
                | IntrinsicModule.TimeSpan, "milliseconds" -> 3
                | IntrinsicModule.TimeSpan, _ -> 10        // other TimeSpan ops
                | IntrinsicModule.Lazy, "create" -> 10     // PRD-14: undef + flag store + thunk store
                | IntrinsicModule.Lazy, "force" -> 20      // PRD-14: check + branch + cached/compute paths + phi
                | IntrinsicModule.Lazy, "isValueCreated" -> 3  // PRD-14: GEP + load flag
                | IntrinsicModule.Lazy, _ -> 10            // other Lazy ops
                // PRD-16: Seq operations - wrapper creation costs
                | IntrinsicModule.Seq, "map" -> 15         // undef + insertvalue x 5 (state, current, moveNext_ptr, inner, mapper)
                | IntrinsicModule.Seq, "filter" -> 15      // same structure as map
                | IntrinsicModule.Seq, "take" -> 15        // undef + insertvalue x 5 (state, current, moveNext_ptr, inner, remaining)
                | IntrinsicModule.Seq, "fold" -> 25        // loop setup: alloca seq + alloca acc + moveNext calls
                | IntrinsicModule.Seq, "collect" -> 20     // complex: outer + mapper + inner seq slot
                | IntrinsicModule.Seq, "iter" -> 20        // loop like fold but no accumulator
                | IntrinsicModule.Seq, "toArray" -> 30     // iteration + dynamic array building
                | IntrinsicModule.Seq, "toList" -> 30      // iteration + list cons
                | IntrinsicModule.Seq, "isEmpty" -> 10     // single MoveNext call
                | IntrinsicModule.Seq, "head" -> 12        // MoveNext + extract current
                | IntrinsicModule.Seq, "length" -> 20      // full iteration with counter
                | IntrinsicModule.Seq, _ -> 15             // default for other Seq ops
                
                // PRD-13a: List operations
                | IntrinsicModule.List, "empty" -> 1           // flat closure (Undef)
                | IntrinsicModule.List, "isEmpty" -> 2         // Baker decomposes to structural check
                | IntrinsicModule.List, "head" -> 2            // GEP + load
                | IntrinsicModule.List, "tail" -> 2            // GEP + load
                | IntrinsicModule.List, "cons" -> 4            // const + alloca + 2 stores
                | IntrinsicModule.List, "length" -> 15         // loop with counter
                | IntrinsicModule.List, "map" -> 20            // recursive structure
                | IntrinsicModule.List, "filter" -> 20         // recursive structure
                | IntrinsicModule.List, "fold" -> 20           // recursive structure
                | IntrinsicModule.List, "rev" -> 15            // iterative reverse
                | IntrinsicModule.List, "append" -> 20         // copy + link
                | IntrinsicModule.List, _ -> 15                // default for other List ops
                
                // PRD-13a: Map operations
                | IntrinsicModule.Map, "empty" -> 1            // flat closure (Undef)
                | IntrinsicModule.Map, "isEmpty" -> 2          // Baker decomposes to structural check
                | IntrinsicModule.Map, "key" -> 2              // offset constant + load
                | IntrinsicModule.Map, "value" -> 2            // offset constant + load
                | IntrinsicModule.Map, "left" -> 2             // offset constant + load
                | IntrinsicModule.Map, "right" -> 2            // offset constant + load
                | IntrinsicModule.Map, "height" -> 2           // offset constant + load
                | IntrinsicModule.Map, "tryFind" -> 20         // tree traversal
                | IntrinsicModule.Map, "find" -> 18            // tree traversal (may fail)
                | IntrinsicModule.Map, "add" -> 25             // tree traversal + rebalance
                | IntrinsicModule.Map, "remove" -> 25          // tree traversal + rebalance
                | IntrinsicModule.Map, "containsKey" -> 15     // tree traversal
                | IntrinsicModule.Map, "count" -> 20           // full traversal
                | IntrinsicModule.Map, "keys" -> 30            // in-order traversal
                | IntrinsicModule.Map, "values" -> 30          // in-order traversal
                | IntrinsicModule.Map, "toList" -> 30          // in-order traversal
                | IntrinsicModule.Map, "ofList" -> 40          // build tree from list
                | IntrinsicModule.Map, _ -> 20                 // default for other Map ops
                
                // PRD-13a: Set operations
                | IntrinsicModule.Set, "empty" -> 1            // flat closure (Undef)
                | IntrinsicModule.Set, "isEmpty" -> 2          // Baker decomposes to structural check
                | IntrinsicModule.Set, "value" -> 2            // offset constant + load
                | IntrinsicModule.Set, "left" -> 2             // offset constant + load
                | IntrinsicModule.Set, "right" -> 2            // offset constant + load
                | IntrinsicModule.Set, "height" -> 2           // offset constant + load
                | IntrinsicModule.Set, "contains" -> 15        // tree traversal
                | IntrinsicModule.Set, "add" -> 20             // tree traversal + rebalance
                | IntrinsicModule.Set, "remove" -> 20          // tree traversal + rebalance
                | IntrinsicModule.Set, "count" -> 20           // full traversal
                | IntrinsicModule.Set, "union" -> 30           // tree merge
                | IntrinsicModule.Set, "intersect" -> 30       // tree filter
                | IntrinsicModule.Set, "difference" -> 30      // tree filter
                | IntrinsicModule.Set, "toList" -> 25          // in-order traversal
                | IntrinsicModule.Set, "ofList" -> 35          // build tree from list
                | IntrinsicModule.Set, _ -> 20                 // default for other Set ops
                
                // PRD-13a: Option operations
                | IntrinsicModule.Option, "isSome" -> 3        // extract + const + icmp
                | IntrinsicModule.Option, "isNone" -> 3        // extract + const + icmp
                | IntrinsicModule.Option, "get" -> 1           // extract value
                | IntrinsicModule.Option, "defaultValue" -> 5  // extract + icmp + select
                | IntrinsicModule.Option, "defaultWith" -> 10  // conditional + closure call
                | IntrinsicModule.Option, "map" -> 15          // conditional + closure call
                | IntrinsicModule.Option, "bind" -> 20         // conditional + closure call + option handling
                | IntrinsicModule.Option, "toList" -> 8        // conditional + list cons
                | IntrinsicModule.Option, _ -> 10              // default for other Option ops
                
                | IntrinsicModule.Bits, "htons" | IntrinsicModule.Bits, "ntohs" -> 2  // byte swap uint16
                | IntrinsicModule.Bits, "htonl" | IntrinsicModule.Bits, "ntohl" -> 2  // byte swap uint32
                | IntrinsicModule.Bits, _ -> 1             // bitcast operations

                // NativeDefault.zeroed — FidelityExtern placeholder body (unreachable at runtime)
                | IntrinsicModule.NativeDefault, _ -> 1

                // MemRef operations (MLIR memref semantics)
                // MemRef ops are Baker's synthesized internal memory vocabulary
                | IntrinsicModule.Array, _ -> 10           // array ops
                | IntrinsicModule.Operators, ("op_Equality" | "op_Inequality") ->
                    // Structural equality on strings / byte arrays (pStringEquality) needs 16 SSAs
                    let firstArgIsMemRef =
                        match node.Children with
                        | _ :: argId :: _ ->
                            match Map.tryFind argId ctx.Graph.Nodes with
                            | Some argNode ->
                                match argNode.Type with
                                | NativeType.TApp (tycon, _) when tycon.NTUKind = Some NTUKind.NTUstring -> true
                                | NativeType.TApp (tycon, _) when tycon.Name = "array" || tycon.Name = "Array" -> true
                                | _ -> false
                            | None -> false
                        | _ -> false
                    if firstArgIsMemRef then 16 else 5
                | IntrinsicModule.Operators, _ -> 5        // arithmetic
                | IntrinsicModule.Convert, _ -> 3          // type conversions
                | IntrinsicModule.Math, _ -> 5             // math functions
                | unhandledModule, unhandledOp ->
                    failwith $"SSAAssignment: Unhandled intrinsic '{unhandledModule}.{unhandledOp}' at node {NodeId.value node.Id}. Add an explicit SSA cost."
            | SemanticKind.VarRef _ ->
                // Function call: 1 result + N potential type compatibility casts (static→dynamic memref)
                // Argument count = Children.Length - 1 (first child is function)
                let argCount = max 0 (node.Children.Length - 1)
                // ExternCall FFI marshaling: detect FidelityExtern metadata on the target binding.
                // Distinguishes static (library="c") from dynamic (library!="c") binding paths.
                let externLibrary =
                    match funcNode.Kind with
                    | SemanticKind.VarRef (_, Some definitionId) ->
                        match Map.tryFind definitionId ctx.Graph.Nodes with
                        | Some bindingNode ->
                            match Map.tryFind "FidelityExtern.Library" bindingNode.Metadata with
                            | Some (Clef.Compiler.PSGSaturation.SemanticGraph.Types.MetadataValue.String lib) -> Some lib
                            | _ -> None
                        | None -> None
                    | _ -> None
                let isExternCall = externLibrary.IsSome
                let isDynamicExtern = match externLibrary with Some lib -> lib <> "c" | None -> false
                let isOptionReturn =
                    match node.Type with
                    | NativeType.TApp(tycon, [_]) when tycon.Name = "option" || tycon.Name = "voption" -> true
                    | _ -> false
                if isDynamicExtern && isOptionReturn then
                    // Dynamic extern with option return:
                    // 11 (dlopen/dlsym preamble, SSAs [0..10])
                    // + 11 (option return handling, SSAs [11..21])
                    // + 11 per arg (FFI marshaling, starting at SSA [22])
                    26 + 11 * argCount
                elif isDynamicExtern then
                    // Dynamic extern with direct return:
                    // 11 (dlopen/dlsym preamble, SSAs [0..10])
                    // + 2 (result + potential return cast, SSAs [11..12])
                    // + 11 per arg (FFI marshaling, starting at SSA [13])
                    13 + 11 * argCount
                elif isExternCall && isOptionReturn then
                    15 + 11 * argCount  // 11 base for option return handling (ssas[0..10])
                                        // + 11 per arg: option unwrap needs 11 SSAs (tag extract,
                                        // payload extract, cmp, select); memref/index use ≤2.
                                        // Unused SSAs are harmless — elided if not consumed (Pillar 1).
                else
                    // Static extern or regular function call:
                    // 1 for result + 1 for potential return-side index.casts
                    // (nativeint returns as platformWordTy at boundary, cast back to index)
                    // + 11 per arg for FFI boundary marshaling
                    // (option unwrap=11, memref extract+cast=2, index cast=1)
                    // +6 for potential closure call
                    // Unused SSAs are harmless — elided if not consumed
                    8 + 11 * argCount
            | _ -> 10  // Other applications
        | None -> 10
    | [] -> 5

/// Compute exact SSA count for TupleExpr based on element count
/// Tuples are materialized as TStruct on all platforms via pBuildRecord.
/// CPU: alloca(1) + per element: byte-offset(1) + view(1) + zero-index(1)
/// FPGA: result(1) — but we allocate for worst case (CPU)
let private computeTupleSSACost (childIds: NodeId list) : int =
    1 + 3 * List.length childIds

/// Compute exact SSA count for RecordExpr based on field count and copy-from
let private computeRecordSSACost (fields: (string * NodeId) list) (copyFrom: NodeId option) : int =
    match copyFrom with
    | None ->
        // alloca(1) + per field: byte-offset constant(1) + view(1) + zero-index(1)
        1 + 3 * List.length fields
    | Some _ ->
        // alloca(1) + extractPtr src(1) + extractPtr dst(1) + castSrc(1) + castDst(1) + sizeConst(1) + memcpyResult(1)
        // + per updated field: byte-offset constant(1) + view(1) + zero-index(1)
        7 + 3 * List.length fields

/// Compute exact SSA count for UnionCase based on payload presence
let private computeUnionCaseSSACost (payloadOpt: NodeId option) : int =
    match payloadOpt with
    | Some _ -> 6  // tag + undef + withTag + payload insert + conversion + result
    | None -> 3    // tag + undef + withTag (no payload)

/// Check if a DU type needs arena allocation
/// Heterogeneous DUs (like Result<'T, 'E>) need arena; homogeneous DUs (like Option<'T>) use inline struct
let private needsDUArenaAllocation (duType: NativeType) : bool =
    match duType with
    | NativeType.TApp (tycon, _) ->
        match tycon.Name with
        | "result" -> true   // Result is heterogeneous, needs arena
        | "option" | "voption" -> false  // Option is homogeneous, inline struct
        | _ -> false  // Default to inline for other DUs
    | _ -> false

/// Compute exact SSA count for DUConstruct based on actual types
///
/// DU construction uses pDUCase pattern which needs: 4 + 2 * payload.Length
/// - 4 base SSAs: undef, tag const, tag offset, tag result (insertvalue for tag)
/// - 2 SSAs per payload field: offset constant + insertvalue result
///
/// Examples:
/// - Option None: 4 + 2*0 = 4 SSAs
/// - Option Some(x): 4 + 2*1 = 6 SSAs
/// - Result Ok(x): 4 + 2*1 = 6 SSAs
/// - Result Error(e): 4 + 2*1 = 6 SSAs
let private computeDUConstructSSACost (_arch: Architecture) (_graph: SemanticGraph) (_duType: NativeType) (payloadOpt: NodeId option) : int =
    // Tag insert: 2 SSAs (reinterpret_cast + store index) — same element type (i8→i8)
    // Payload insert: 3 SSAs (offset const + memref.view + store index) — different element type
    // Base: 4 SSAs (alloca, tag const, + 2 for tag insert via reinterpret_cast)
    // Arena allocation (if needed) is handled by DULayout coeffect, not SSA count
    let payloadCount = if Option.isSome payloadOpt then 1 else 0
    4 + 3 * payloadCount

/// Count mutable bindings in a subtree (internal state fields for seq)
/// PRD-15 THROUGH-LINE: Internal state is unique to Seq - mutable vars declared inside body
/// that persist between MoveNext calls. This is distinct from captures (read-only from enclosing scope).
let rec private countMutableBindingsInSubtree (graph: SemanticGraph) (nodeId: NodeId) : int =
    match Map.tryFind nodeId graph.Nodes with
    | None -> 0
    | Some node ->
        // Count this node if it's a mutable binding
        let thisCount =
            match node.Kind with
            | SemanticKind.Binding (_, isMutable, _, _) when isMutable -> 1
            | _ -> 0
        // Count in children
        let childCount =
            node.Children
            |> List.sumBy (fun childId -> countMutableBindingsInSubtree graph childId)
        thisCount + childCount

/// Compute SSA cost for SeqExpr based on captures and internal state
/// PRD-15: SeqExpr SSA cost = 5 base + numCaptures + (2 * numInternalState)
/// The 2 per internal state is for: const zero + InsertValue
let private computeSeqExprSSACost (graph: SemanticGraph) (bodyId: NodeId) (captures: CaptureInfo list) : int =
    let numCaptures = List.length captures
    let numInternalState = countMutableBindingsInSubtree graph bodyId
    // 5 base (zero, undef, insert state, addressof, insert code_ptr)
    // + 1 per capture (InsertValue)
    // + 2 per internal state (const zero + InsertValue)
    5 + numCaptures + (numInternalState * 2)

/// A module-level value binding realized as a program-lifetime slot (memref.global):
/// a direct ModuleDef member (ModuleInit of its module classification), EmissionStrategy.MainPrologue,
/// and a non-Lambda value. Mirrors TransferTypes.ModuleValues.isSlotBinding.
let private isModuleValueSlotBinding (graph: SemanticGraph) (node: SemanticNode) : bool =
    match node.Kind with
    | SemanticKind.Binding _ when node.EmissionStrategy = EmissionStrategy.MainPrologue ->
        let isModuleMember =
            graph.ModuleClassifications.Value
            |> Map.exists (fun _ classification -> List.contains node.Id classification.ModuleInit)
        isModuleMember &&
        (match node.Children with
         | childId :: _ ->
             match Map.tryFind childId graph.Nodes with
             | Some child -> (match child.Kind with SemanticKind.Lambda _ -> false | _ -> true)
             | None -> false
         | [] -> false)
    | _ -> false

/// Get the number of SSAs needed for a node based on its STRUCTURE
/// This is the key function - it analyzes actual instance structure, not just kind
let private nodeExpansionCost (ctx: SSAContext) (node: SemanticNode) : int =
    match node.Kind with
    // Structural analysis - exact counts from instance
    | SemanticKind.Match (scrutineeId, cases) ->
        computeMatchSSACost ctx.Graph scrutineeId cases

    | SemanticKind.CaseElimination (_scrutineeId, arms) ->
        // Tag extraction: 3 SSAs (reinterpret_cast + zero + load)
        // Per non-final arm: tag literal + comparison = 2 SSAs
        // Result SSA + control flow structure
        let numArms = List.length arms
        let tagSSAs = 3
        let comparisonSSAs = max 0 (numArms - 1) * 2
        let resultSSA = 1
        tagSSAs + comparisonSSAs + resultSSA + (numArms * 2) + 5

    | SemanticKind.Application _ ->
        computeApplicationSSACost ctx node

    | SemanticKind.TupleExpr childIds ->
        computeTupleSSACost childIds

    | SemanticKind.RecordExpr (fields, copyFrom) ->
        computeRecordSSACost fields copyFrom

    | SemanticKind.UnionCase (_, _, payloadOpt) ->
        computeUnionCaseSSACost payloadOpt

    // Literal-based costs
    | SemanticKind.Literal lit -> literalExpansionCost lit
    | SemanticKind.InterpolatedString parts -> interpolatedStringCost parts

    // Lambda: cost depends on captures (structural analysis, capture types vary by arch)
    | SemanticKind.Lambda (_, _, captures, _, _) ->
        computeLambdaSSACost ctx.TargetPlatform ctx.Arch ctx.Graph captures

    // Fixed costs (these don't vary by structure)
    | SemanticKind.ForLoop _ -> 2
    | SemanticKind.IfThenElse _ -> 4  // FPGA: up to 2 ext + mux + trunc
    | SemanticKind.Binding _ -> 3
    | SemanticKind.IndexGet _ -> 2
    | SemanticKind.IndexSet _ -> 1
    | SemanticKind.AddressOf _ -> 3  // alloca, zero-index, extract-base-ptr
    | SemanticKind.VarRef (_, defIdOpt) ->
        // A reference to a module-level value slot needs 3 (get_global + zero + load); any
        // other reference 2. A named function in value position is no VarRef by now: Baker
        // elaborates it into a Lambda marked for closure pair construction.
        match defIdOpt |> Option.bind (fun d -> Map.tryFind d ctx.Graph.Nodes) with
        | Some def when isModuleValueSlotBinding ctx.Graph def -> 3
        | _ -> 2
    | SemanticKind.TupleGet _ -> 4  // Struct field extraction: offset + view + zero + result (pass-through uses 0 but over-allocate for safety)
    | SemanticKind.FieldGet _ -> 4  // View-based: offset + view + zero + result (for TStruct records)
    | SemanticKind.FieldSet _ -> 3  // Offset constant + typed view + zero index (pRecordFieldSet)
    | SemanticKind.Set _ -> 2  // zero index; plus get_global when the target is a module-level slot
    | SemanticKind.TraitCall _ -> 1
    | SemanticKind.ArrayExpr elements -> 2 + List.length elements  // size constant + alloc + one index per element (pBuildArrayLiteral)
    | SemanticKind.ListExpr _ -> 20
    // PatternBinding needs SSAs for tuple element extraction via pRecordFieldGet:
    // offset + view + zero + result = 4
    | SemanticKind.PatternBinding _ -> 4
    // PRD-14: Lazy values - SSA costs derived from PSG structure
    // LazyExpr: 5 base + N captures (per LazyWitness documentation)
    //   - 1: false constant, 1: undef struct, 1: insert computed
    //   - 1: addressof code_ptr, 1: insert code_ptr
    //   - N: insert each capture
    | SemanticKind.LazyExpr (_, captures) -> 5 + List.length captures
    // LazyForce: 4 fixed (per LazyWitness documentation)
    //   - 1: extract code_ptr, 1: const 1 for alloca
    //   - 1: alloca for lazy struct, 1: indirect call result
    | SemanticKind.LazyForce _ -> 4
    // PRD-15: Sequence expressions
    // SeqExpr: 5 base + captures + (2 * internal state fields)
    // Internal state = let mutable bindings inside seq body (through-line from PRD-11)
    | SemanticKind.SeqExpr (bodyId, captures) -> computeSeqExprSSACost ctx.Graph bodyId captures
    // Yield: 4 (gep current + store value + gep state + store state)
    | SemanticKind.Yield _ -> 4
    // YieldBang: 12 (nested iteration setup)
    | SemanticKind.YieldBang _ -> 12
    // ForEach: 7 (setup: 4 + condition: 1 + body: 2)
    | SemanticKind.ForEach _ -> 7
    // DU Operations (January 2026, updated February 2026 for reinterpret_cast)
    // DUGetTag: pointer-based = constI + pLoad (2 SSAs)
    //           inline = reinterpret_cast + constI(0) + typed load (3 SSAs)
    | SemanticKind.DUGetTag (_, duType) ->
        if needsDUArenaAllocation duType then 2 else 3
    // DUEliminate: memref.view path (different element type: i8 → payload type)
    // constI(offset) + memref.view + constI(0) + typed load = 4 SSAs
    | SemanticKind.DUEliminate (_, _, _, _) -> 4
    // DUConstruct: SSA count depends on whether payload type matches slot type
    // Computed deterministically from PSG structure
    | SemanticKind.DUConstruct (_, _, payloadOpt, _) ->
        computeDUConstructSSACost ctx.Arch ctx.Graph node.Type payloadOpt
    // Standalone Intrinsic nodes (not applied via Application)
    // These appear in entry point elaboration and other structural expansions
    | SemanticKind.Intrinsic info ->
        match info.Module, info.Operation with
        | IntrinsicModule.Sys, "emptyStringArray" -> 5  // zeroPtr + const0 + undef + insertvalue*2
        | IntrinsicModule.Sys, "exit" -> 16             // syscall with full register setup
        | _ -> 1  // Default for standalone intrinsics
    | _ -> 1

// ═══════════════════════════════════════════════════════════════════════════
// THE DERIVATION TABLE OF MEETS (Dimensional_Range_Design.md §3.1, §8.3; rulings 1 and 3;
// CS-11 slice 2(b))
// ═══════════════════════════════════════════════════════════════════════════
//
// Every place a value meets a slot held at another width is a consumer kind below, one function
// each. A meet's value is one more SSA in the consumer's allocation, yielded right after the
// node's own values in the scope's numbering, in the order the table gives (operand order):
// `Meets[consumer]`, read by the witness through `lookupMeet consumer operand`. Both widths are
// reads of CCS's selection (`TypeMapping.nodeWidth`, the settled layouts, the element ranges);
// the extension's sign is the operand's range's; nothing is decided here.
//
//   Application, direct call to a lambda   each argument -> the parameter node's width; the
//                                             result read from the callee's body width to the
//                                             call node's own (keyed with the call as operand)
//   Application, through a value / to an   each argument -> the declared Register width
//     escaping lambda, and a closure call    (ruling 1: the value-call boundary)
//   Application, the syscall ABI            the descriptor -> the Register width; Array.blit's
//     (Sys.write, read, readline)             indices likewise (pointer arithmetic at the word)
//   Application, Array.set / Array.create   the value -> the element's settled width
//   Set / mutable Binding                   the value -> the cell's width (the Binding node's)
//   RecordExpr / TupleExpr                  each field value -> the field's settled representation
//   IfThenElse / CaseElimination / Match    each arm's value -> the join's width (the node's)
//   Lambda (the return)                     the body's last value -> the body node's width; the
//                                             last value of the body's scope (`ReturnMeets`)
//   VarRef / FieldGet / TupleGet /          the slot's width -> the read's width (ruling 3: a
//     IndexGet / Array.get / DUEliminate      refined read truncates; a boundary read extends);
//                                             keyed with the consumer as its own operand
//   IndexSet / ArrayExpr                    each value -> the element's settled width
//   DUConstruct / Option.Some               the payload -> the payload slot's width
//   Operators (pBinaryArithOp, pComparisonOp, the unary ops) and Convert (pTypeConversion)
//     adapt within the node's own five (three) values at the CS-10 positions; a Literal has
//     its point range's width and needs no meet.

/// A meet still to be given its value: the operand and the widths it adapts between.
type private Pending = { Operand: NodeId; From: IntWidth; To: IntWidth; Kind: MeetKind }

/// The last value a body evaluates to: through a block's last child and an annotation.
let rec private lastValueOf (graph: SemanticGraph) (id: NodeId) : NodeId =
    match Map.tryFind id graph.Nodes with
    | Some { Kind = SemanticKind.Sequential ids } ->
        match List.tryLast ids with
        | Some last -> lastValueOf graph last
        | None -> id
    | Some { Kind = SemanticKind.TypeAnnotation (inner, _) } -> lastValueOf graph inner
    | _ -> id

/// The function node of an application, through an annotation.
let private functionNodeOf (graph: SemanticGraph) (funcId: NodeId) : SemanticNode option =
    match Map.tryFind funcId graph.Nodes with
    | Some { Kind = SemanticKind.TypeAnnotation (inner, _) } -> Map.tryFind inner graph.Nodes
    | other -> other

/// The Lambda a binding holds, through an annotation.
let private lambdaOfBinding (graph: SemanticGraph) (bindingId: NodeId) : SemanticNode option =
    match Map.tryFind bindingId graph.Nodes with
    | Some ({ Kind = SemanticKind.Binding _ } as binding) ->
        binding.Children
        |> List.tryPick (fun childId ->
            match Map.tryFind childId graph.Nodes with
            | Some ({ Kind = SemanticKind.Lambda _ } as lambda) -> Some lambda
            | Some { Kind = SemanticKind.TypeAnnotation (inner, _) } ->
                match Map.tryFind inner graph.Nodes with
                | Some ({ Kind = SemanticKind.Lambda _ } as lambda) -> Some lambda
                | _ -> None
            | _ -> None)
    | _ -> None

/// The meet that brings `operand` from `from` to `target`, if the widths differ: an extension
/// by the sign of the operand's range, or a truncation.
let private pending (graph: SemanticGraph) (operand: NodeId) (from: IntWidth) (target: IntWidth) : Pending option =
    if from = target then None
    elif intWidthBits from < intWidthBits target then
        let sign =
            match Map.tryFind operand graph.Nodes |> Option.bind (fun n -> n.ValueRange) with
            | Some r when ValueRange.isNonNegative r -> MeetKind.ExtendUnsigned
            | Some _ -> MeetKind.ExtendSigned
            | None -> failwithf "SSAAssignment: node %d is extended to a wider slot but has no analysed range to read the extension's sign from" (NodeId.value operand)
        Some { Operand = operand; From = from; To = target; Kind = sign }
    else Some { Operand = operand; From = from; To = target; Kind = MeetKind.Truncate }

/// The meet of a value node into a slot of the given width, where both are word integers. The
/// value is read at its last value (through a block or an annotation), the node whose value the
/// witness recalls.
let private meetInto (graph: SemanticGraph) (valueId: NodeId) (slot: IntWidth option) : Pending option =
    let valueId = lastValueOf graph valueId
    match nodeWidth graph valueId, slot with
    | Some from, Some target -> pending graph valueId from target
    | _ -> None

/// The settled integer slot of a record's field, if the field is one.
let private fieldSlotWidth (graph: SemanticGraph) (recordTy: NativeType) (field: string) : IntWidth option =
    match settledLayout graph recordTy with
    | Some (SettledLayout.Record (fields, _, _)) ->
        fields |> List.tryFind (fun f -> f.Name = field) |> Option.bind (fun f ->
            match f.Slot with
            | SettledSlot.Integer (bits, _) -> Some (IntWidth bits)
            | _ -> None)
    | _ -> None

/// The settled integer slot of a union case's payload, if it is one.
let private payloadSlotWidth (graph: SemanticGraph) (unionTy: NativeType) (caseIndex: int) : IntWidth option =
    match settledLayout graph unionTy with
    | Some (SettledLayout.Union (cases, _, _, _)) ->
        List.tryItem caseIndex cases |> Option.bind snd |> Option.bind (fun slot ->
            match slot with
            | SettledSlot.Integer (bits, _) -> Some (IntWidth bits)
            | _ -> None)
    | _ -> None

/// The settled width of an array's elements, if they are word integers.
let private elementSlotWidth (graph: SemanticGraph) (arrayId: NodeId) : IntWidth option =
    match Map.tryFind arrayId graph.Nodes |> Option.map (fun n -> applySubst n.Type) with
    | Some (NativeType.TApp (tycon, [ elemTy ])) when tycon.Name = "array" || tycon.Name = "Array" ->
        if Types.tryGetNTUKind elemTy |> Option.exists isWordInteger then Some (elementWidth graph elemTy) else None
    | _ -> None

/// The result of a direct call read from the callee's body width (the width the callee returns
/// at) to the call node's own width: a read meet keyed with the call as its own operand. A call
/// through a value has both at the declared Register width and no meet.
let private callResultMeet (graph: SemanticGraph) (node: SemanticNode) (lambda: SemanticNode) : Pending list =
    match lambda.Kind with
    | SemanticKind.Lambda (_, bodyId, _, _, _) ->
        match nodeWidth graph bodyId, nodeWidth graph node.Id with
        | Some from, Some target -> Option.toList (pending graph node.Id from target)
        | _ -> []
    | _ -> []

/// The meets of an application: its arguments against the parameters they meet, and its
/// result against the callee's.
let private applicationMeets (ctx: SSAContext) (node: SemanticNode) (funcId: NodeId) (args: NodeId list) : Pending list =
    let graph = ctx.Graph
    let word = declaredWordWidth ctx.Arch
    let toWord (argId: NodeId) = meetInto graph argId (Some word)
    let toParameters (lambda: SemanticNode) (parameters: (string * NativeType * NodeId) list) (args: NodeId list) =
        (List.zip (List.truncate (min parameters.Length args.Length) parameters) (List.truncate (min parameters.Length args.Length) args)
         |> List.choose (fun ((_, _, paramId), argId) -> meetInto graph argId (nodeWidth graph paramId)))
        @ callResultMeet graph node lambda
    match functionNodeOf graph funcId with
    | Some { Kind = SemanticKind.Intrinsic info } ->
        match info.Module, info.Operation, args with
        // the syscall ABI: the descriptor at the declared Register width
        | IntrinsicModule.Sys, ("write" | "read" | "readline"), fd :: _ -> Option.toList (toWord fd)
        // an element store: the value at the element's settled width
        | IntrinsicModule.Array, "set", [ arr; _; value ] -> Option.toList (meetInto graph value (elementSlotWidth graph arr))
        | IntrinsicModule.Array, "create", [ _; seed ] -> Option.toList (meetInto graph seed (elementSlotWidth graph node.Id))
        // pointer arithmetic at the word
        | IntrinsicModule.Array, "blit", [ _; srcIdx; _; dstIdx; count ] -> [ srcIdx; dstIdx; count ] |> List.choose toWord
        | _ -> []
    | Some funcNode ->
        match Map.tryFind node.Id ctx.Curry.SaturatedCalls with
        | Some info ->
            match lambdaOfBinding graph info.TargetBindingId with
            | Some ({ Kind = SemanticKind.Lambda (parameters, _, _, _, _) } as lambda) -> toParameters lambda parameters info.AllArgNodes
            | _ -> info.AllArgNodes |> List.choose toWord
        | None when Map.containsKey node.Id ctx.Curry.PartialApplications -> []
        | None ->
            match funcNode.Kind with
            | SemanticKind.VarRef (_, Some defId) ->
                match lambdaOfBinding graph defId with
                | Some ({ Kind = SemanticKind.Lambda (parameters, _, _, _, _) } as lambda) when args.Length <= parameters.Length ->
                    // a direct call: the parameter nodes carry the boundary rule where the lambda escapes
                    toParameters lambda parameters args
                | _ -> args |> List.choose toWord   // a function value, or a surplus handed to the returned value
            | SemanticKind.Lambda (parameters, _, _, _, _) when args.Length <= parameters.Length -> toParameters funcNode parameters args
            | _ -> args |> List.choose toWord
    | None -> []

/// The read of a slot into a node held at another width (ruling 3), keyed with the node as its
/// own operand.
let private readMeet (graph: SemanticGraph) (node: SemanticNode) (slot: IntWidth option) : Pending list =
    match slot, nodeWidth graph node.Id with
    | Some from, Some target -> Option.toList (pending graph node.Id from target)
    | _ -> []

/// The meets a node's consumers-of-slots derive, by kind (the table above). The fabric leg
/// derives none here: its meets are the CS-10 harmonisations within each pattern's own values
/// (an hw.struct field, a mux operand, a comb operand), and it reads no Register width.
let private nodeMeets (ctx: SSAContext) (node: SemanticNode) : Pending list =
    let graph = ctx.Graph
    // an unreachable node is never witnessed and carries no range to read
    if ctx.TargetPlatform = Core.Types.Dialects.TargetPlatform.FPGA || not node.IsReachable then [] else
    match node.Kind with
    | SemanticKind.Application (funcId, args) -> applicationMeets ctx node funcId args
    | SemanticKind.Intrinsic { Module = IntrinsicModule.Option; Operation = "Some" } ->
        match node.Children with
        | [ payloadId ] -> Option.toList (meetInto graph payloadId (payloadSlotWidth graph node.Type 1))
        | _ -> []
    | SemanticKind.DUConstruct (_, caseIndex, Some payloadId, _) ->
        Option.toList (meetInto graph payloadId (payloadSlotWidth graph node.Type caseIndex))
    | SemanticKind.Set (targetId, valueId) ->
        match Map.tryFind targetId graph.Nodes with
        | Some { Kind = SemanticKind.VarRef (_, Some defId) } -> Option.toList (meetInto graph valueId (nodeWidth graph defId))
        | _ -> []
    | SemanticKind.Binding (_, true, _, _) ->
        match node.Children with
        | [ valueId ] -> Option.toList (meetInto graph valueId (nodeWidth graph node.Id))
        | _ -> []
    | SemanticKind.RecordExpr (fields, _) ->
        fields |> List.choose (fun (field, valueId) -> meetInto graph valueId (fieldSlotWidth graph node.Type field))
    | SemanticKind.TupleExpr elements ->
        elements |> List.mapi (fun i elementId -> meetInto graph elementId (fieldSlotWidth graph node.Type (sprintf "Item%d" (i + 1)))) |> List.choose id
    | SemanticKind.IfThenElse (_, thenId, elseId) ->
        (thenId :: Option.toList elseId) |> List.choose (fun armId -> meetInto graph armId (nodeWidth graph node.Id))
    | SemanticKind.CaseElimination (_, arms) ->
        arms |> List.choose (fun arm -> meetInto graph arm.Body (nodeWidth graph node.Id))
    | SemanticKind.Match (_, cases) ->
        cases |> List.choose (fun case -> meetInto graph case.Body (nodeWidth graph node.Id))
    | SemanticKind.IndexSet (arrId, _, valueId) -> Option.toList (meetInto graph valueId (elementSlotWidth graph arrId))
    | SemanticKind.ArrayExpr elements ->
        elements |> List.choose (fun elementId -> meetInto graph elementId (elementSlotWidth graph node.Id))
    | SemanticKind.VarRef (_, Some defId) ->
        match Map.tryFind defId graph.Nodes with
        | Some { Kind = SemanticKind.Binding _ } -> readMeet graph node (nodeWidth graph defId)
        | Some ({ Kind = SemanticKind.PatternBinding _ } as def) ->
            // a lambda parameter: its own held width (a match arm's binding aliases its scrutinee)
            match def.Parent |> Option.bind (fun p -> Map.tryFind p graph.Nodes) with
            | Some { Kind = SemanticKind.Lambda _ } -> readMeet graph node (nodeWidth graph defId)
            | _ -> []
        | _ -> []
    | SemanticKind.FieldGet (exprId, field) ->
        match Map.tryFind exprId graph.Nodes with
        | Some expr -> readMeet graph node (fieldSlotWidth graph expr.Type field)
        | None -> []
    | SemanticKind.TupleGet (tupleId, index) ->
        match Map.tryFind tupleId graph.Nodes with
        | Some { Kind = SemanticKind.TupleExpr elements } when index < elements.Length ->
            readMeet graph node (nodeWidth graph elements.[index])
        | Some tuple -> readMeet graph node (fieldSlotWidth graph tuple.Type (sprintf "Item%d" (index + 1)))
        | None -> []
    | SemanticKind.IndexGet (arrId, _) -> readMeet graph node (elementSlotWidth graph arrId)
    | SemanticKind.DUEliminate (duId, caseIndex, _, _) ->
        match Map.tryFind duId graph.Nodes with
        | Some du -> readMeet graph node (payloadSlotWidth graph du.Type caseIndex)
        | None -> []
    | _ -> []

/// The application form of a read of an element (`Array.get`), which the table lists with the reads.
let private applicationReadMeets (ctx: SSAContext) (node: SemanticNode) : Pending list =
    if ctx.TargetPlatform = Core.Types.Dialects.TargetPlatform.FPGA || not node.IsReachable then [] else
    match node.Kind with
    | SemanticKind.Application (funcId, [ arr; _ ]) ->
        match functionNodeOf ctx.Graph funcId with
        | Some { Kind = SemanticKind.Intrinsic { Module = IntrinsicModule.Array; Operation = "get" } } ->
            readMeet ctx.Graph node (elementSlotWidth ctx.Graph arr)
        | _ -> []
    | _ -> []

// ═══════════════════════════════════════════════════════════════════════════
// FUNCTION SCOPE STATE
// ═══════════════════════════════════════════════════════════════════════════

/// SSA assignment state for a single function scope
type private FunctionScope = {
    Counter: int
    Assignments: Map<int, NodeSSAAllocation>  // NodeId.value -> SSA allocation
    /// The meets derived for each consumer in this scope (NodeId.value -> meets in table order)
    Meets: Map<int, Meet list>
}

module private FunctionScope =
    let empty = { Counter = 0; Assignments = Map.empty; Meets = Map.empty }

    /// Yield a single SSA
    let yieldSSA (scope: FunctionScope) : SSA * FunctionScope =
        let ssa = V scope.Counter
        ssa, { scope with Counter = scope.Counter + 1 }

    /// Yield multiple SSAs based on expansion cost
    let yieldSSAs (count: int) (scope: FunctionScope) : SSA list * FunctionScope =
        let ssas = List.init count (fun i -> V (scope.Counter + i))
        ssas, { scope with Counter = scope.Counter + count }

    /// Assign a node's SSA allocation
    let assign (nodeId: NodeId) (alloc: NodeSSAAllocation) (scope: FunctionScope) : FunctionScope =
        { scope with Assignments = Map.add (NodeId.value nodeId) alloc scope.Assignments }

    /// Give each pending meet of a consumer its value, right after the consumer's own values
    let meet (consumer: NodeId) (pendings: Pending list) (scope: FunctionScope) : FunctionScope =
        if List.isEmpty pendings then scope
        else
            let ssas, scope' = yieldSSAs pendings.Length scope
            let meets =
                List.zip pendings ssas
                |> List.map (fun (p, ssa) -> { Consumer = consumer; Operand = p.Operand; SSA = ssa; From = p.From; To = p.To; Kind = p.Kind })
            { scope' with Meets = Map.add (NodeId.value consumer) meets scope'.Meets }

/// Check if a node kind produces an SSA value
let private producesValue (kind: SemanticKind) : bool =
    match kind with
    | SemanticKind.Literal _ -> true
    | SemanticKind.Application _ -> true
    | SemanticKind.Lambda _ -> true
    | SemanticKind.Binding _ -> true
    | SemanticKind.Sequential _ -> true
    | SemanticKind.IfThenElse _ -> true
    | SemanticKind.Match _ -> true
    | SemanticKind.CaseElimination _ -> true
    | SemanticKind.TupleExpr _ -> true
    | SemanticKind.TupleGet _ -> true
    | SemanticKind.RecordExpr _ -> true
    | SemanticKind.UnionCase _ -> true
    // DU Operations (January 2026)
    | SemanticKind.DUGetTag _ -> true
    | SemanticKind.DUEliminate _ -> true
    | SemanticKind.DUConstruct _ -> true
    | SemanticKind.ArrayExpr _ -> true
    | SemanticKind.ListExpr _ -> true
    | SemanticKind.FieldGet _ -> true
    | SemanticKind.IndexGet _ -> true
    | SemanticKind.Upcast _ -> true
    | SemanticKind.Downcast _ -> true
    | SemanticKind.TypeTest _ -> true
    | SemanticKind.AddressOf _ -> true
    | SemanticKind.VarRef _ -> true
    | SemanticKind.Deref _ -> true
    | SemanticKind.TraitCall _ -> true
    | SemanticKind.Intrinsic _ -> true
    | SemanticKind.LazyExpr _ -> true
    | SemanticKind.LazyForce _ -> true
    // PRD-15: Sequence expressions
    | SemanticKind.SeqExpr _ -> true   // Produces seq struct value
    | SemanticKind.Yield _ -> true     // Needs SSAs for state machine operations
    | SemanticKind.YieldBang _ -> true // Needs SSAs for nested iteration
    | SemanticKind.PlatformBinding _ -> true
    | SemanticKind.InterpolatedString _ -> true
    // Set needs SSAs for module-level mutable address operations
    | SemanticKind.Set _ -> true
    | SemanticKind.FieldSet _ -> true   // offset/view/zero SSAs for the in-place store (pRecordFieldSet)
    | SemanticKind.IndexSet _ -> true   // index cast SSA for the store (pIndexSetArray)
    | SemanticKind.NamedIndexedPropertySet _ -> false
    | SemanticKind.WhileLoop _ -> false
    | SemanticKind.ForLoop _ -> false
    | SemanticKind.ForEach _ -> true  // PRD-15: Needs SSAs for loop control
    | SemanticKind.TryWith _ -> false
    | SemanticKind.TryFinally _ -> false
    | SemanticKind.Quote _ -> false
    | SemanticKind.ObjectExpr _ -> false
    | SemanticKind.ModuleDef _ -> false
    | SemanticKind.TypeDef _ -> false
    | SemanticKind.MemberDef _ -> false
    | SemanticKind.TypeAnnotation _ -> true  // Passes through the inner value
    | SemanticKind.PatternBinding _ -> true  // Pattern binding introduces a variable
    | SemanticKind.Error _ -> false
    | SemanticKind.Obligation _ -> false  // Off the emission spine: cited through F, never witnessed

/// Result of SSA assignment pass
type SSAAssignment = {
    /// Map from NodeId.value to SSA allocation (supports multi-SSA expansion)
    NodeSSA: Map<int, NodeSSAAllocation>
    /// Map from Lambda NodeId.value to its function name
    LambdaNames: Map<int, string>
    /// Map from declaration root Lambda IDs to their DeclRoot flavor
    DeclarationRootLambdas: Map<int, DeclRoot>
    /// Closure layouts for Lambdas with captures (NodeId.value -> ClosureLayout)
    /// Empty for simple lambdas (no captures)
    ClosureLayouts: Map<int, ClosureLayout>
    /// DU layouts for DUConstruct nodes needing arena allocation (NodeId.value -> DULayout)
    /// Empty for homogeneous DUs like Option that use inline struct
    DULayouts: Map<int, DULayout>
    /// The Mealy machine values of each [<HardwareModule>] binding (NodeId.value -> layout)
    HardwareModuleLayouts: Map<int, HardwareModuleLayout>
    /// The zero constant a unit-typed function returns (Lambda NodeId.value -> its value),
    /// the first value of the body's scope
    UnitReturns: Map<int, SSA>
    /// The meets derived for each consumer (NodeId.value -> meets, in the derivation table's order)
    Meets: Map<int, Meet list>
    /// The return meet of each function whose last value is held at another width than its
    /// result (Lambda NodeId.value -> the meet), the last value of the body's scope
    ReturnMeets: Map<int, Meet>
}

/// The return meet of a lambda: its body's last value brought to the body node's width (the
/// result the callers read), the last value of the body's scope. Recorded in `ReturnMeets`.
let private returnMeet (ctx: SSAContext) (lambdaId: NodeId) (bodyId: NodeId) (scope: FunctionScope) : FunctionScope =
    let lastValue = lastValueOf ctx.Graph bodyId
    let reachable = Map.tryFind bodyId ctx.Graph.Nodes |> Option.exists (fun n -> n.IsReachable)
    if lastValue = bodyId || not reachable || ctx.TargetPlatform = Core.Types.Dialects.TargetPlatform.FPGA then scope
    else
        match meetInto ctx.Graph lastValue (nodeWidth ctx.Graph bodyId) with
        | Some p ->
            let ssa, scope' = FunctionScope.yieldSSA scope
            ctx.ReturnMeets.[NodeId.value lambdaId] <- { Consumer = bodyId; Operand = p.Operand; SSA = ssa; From = p.From; To = p.To; Kind = p.Kind }
            scope'
        | None -> scope

/// Assign SSA names to all nodes in a function body
/// Returns updated scope with assignments
/// innerScopeAssignments: mutable collection for nested lambda body SSAs (separate MLIR functions)
let rec private assignFunctionBody
    (ctx: SSAContext)
    (scope: FunctionScope)
    (nodeId: NodeId)
    : FunctionScope =

    match Map.tryFind nodeId ctx.Graph.Nodes with
    | None -> scope
    | Some node ->
        // Architectural fix (January 2026): Filter out SeparateFunction children.
        // These are Lambda/SeqExpr body nodes - processed by their parent, not during children traversal.
        // This makes SSA assignment deterministic based on EmissionStrategy, not SemanticKind special-cases.
        let childrenToProcess =
            node.Children
            |> List.filter (fun childId ->
                match Map.tryFind childId ctx.Graph.Nodes with
                | Some child ->
                    match child.EmissionStrategy with
                    | EmissionStrategy.SeparateFunction _ -> false  // Skip, parent handles it
                    | EmissionStrategy.MainPrologue ->
                        // A module-level value slot is assigned by Pass 1 (main's prologue). A
                        // binding nested inside such a slot's initializer (`let a = ... in {..}`)
                        // also carries MainPrologue but is an ordinary local of the initializer.
                        not (isModuleValueSlotBinding ctx.Graph child)
                    | EmissionStrategy.Inline -> true
                | None -> true)

        // Post-order: process filtered children first
        let scopeAfterChildren =
            childrenToProcess |> List.fold (fun s childId -> assignFunctionBody ctx s childId) scope

        // Special handling for nested Lambdas - they get their own scope
        // (but we still assign this Lambda node SSAs in parent scope for closure construction)
        match node.Kind with
        | SemanticKind.Lambda(_params, bodyId, captures, enclosingFuncOpt, context) ->
            // Process Lambda body in a NEW scope.
            // Architectural fix: Start SSA counter AFTER capture extraction SSAs.
            // The body's EmissionStrategy.SeparateFunction carries the captureCount.
            // SSA layout in inner function: v0..v(N-1) = capture results, then work SSAs.
            // Work SSA count per capture varies by slot type (2 for scalar, 7 for decomposed memref).
            // Body starts after all capture result + work SSAs.
            let startCounter =
                match Map.tryFind bodyId ctx.Graph.Nodes with
                | Some bodyNode ->
                    match bodyNode.EmissionStrategy with
                    | EmissionStrategy.SeparateFunction captureCount ->
                        if captureCount > 0 then
                            // The callee prologue (closurePrologue): capture results, work, env reconstruction
                            closurePrologueCount (captures |> List.map (captureSlotType ctx.TargetPlatform ctx.Arch ctx.Graph))
                        else 0
                    | _ -> 0  // Shouldn't happen - Lambda bodies are marked SeparateFunction
                | None -> 0
            // A unit-typed body returns a zero constant: the first value after the prologue
            let unitReturnCount =
                match Map.tryFind bodyId ctx.Graph.Nodes with
                | Some bodyNode when isUnitTyped bodyNode.Type ->
                    ctx.UnitReturns.[NodeId.value nodeId] <- V startCounter
                    1
                | _ -> 0
            let innerStartScope = { FunctionScope.empty with Counter = startCounter + unitReturnCount }

            // Assign Arg SSAs for nested lambda parameters (mirrors top-level handling in assignSSA)
            // For closures: offset by 1 because Arg 0 = env_ptr (closure struct)
            let requiresClosurePairNested =
                node.Metadata
                |> Map.tryFind ClosureMetadata.RequiresClosurePair
                |> Option.map (function MetadataValue.Bool b -> b | _ -> false)
                |> Option.defaultValue false
            let argOffset = if not (List.isEmpty captures) || requiresClosurePairNested then 1 else 0
            let paramScope =
                _params
                |> List.mapi (fun i (_name, _ty, paramNodeId) -> i + argOffset, paramNodeId)
                |> List.fold (fun (s: FunctionScope) (i, paramNodeId) ->
                    FunctionScope.assign paramNodeId (NodeSSAAllocation.single (Arg i)) s
                ) innerStartScope

            let innerScope = returnMeet ctx node.Id bodyId (assignFunctionBody ctx paramScope bodyId)

            // Merge nested lambda's parameter and body SSAs into the shared collection.
            // These are a separate MLIR function's namespace, collected for the global SSA map.
            for kvp in paramScope.Assignments do
                if not (ctx.InnerScopeAssignments.ContainsKey(kvp.Key)) then
                    ctx.InnerScopeAssignments.Add(kvp.Key, kvp.Value)
            for kvp in innerScope.Assignments do
                if not (ctx.InnerScopeAssignments.ContainsKey(kvp.Key)) then
                    ctx.InnerScopeAssignments.Add(kvp.Key, kvp.Value)
            for kvp in innerScope.Meets do
                if not (ctx.InnerMeets.ContainsKey(kvp.Key)) then
                    ctx.InnerMeets.Add(kvp.Key, kvp.Value)

            // DISTINCTION: Nested NAMED functions with captures use parameter-passing, NOT closure structs.
            // Anonymous lambdas (fun x -> ...) that escape STILL need closure structs.
            // Check: Lambda is nested (enclosingFuncOpt = Some _) AND its parent is a Binding (named function).
            // Anonymous lambdas have non-Binding parents (Application, Sequential, etc.).
            let isNestedNamedFunction =
                Option.isSome enclosingFuncOpt &&
                match node.Parent with
                | Some parentId ->
                    match Map.tryFind parentId ctx.Graph.Nodes with
                    | Some parentNode ->
                        match parentNode.Kind with
                        | SemanticKind.Binding _ -> true  // Named function definition
                        | _ -> false  // Anonymous lambda (value, argument, etc.)
                    | None -> false
                | None -> false

            if isNestedNamedFunction then
                // Nested function: NO closure layout, captures passed as explicit parameters
                // No SSAs needed in parent scope for closure construction
                scopeAfterChildren
            else
                // Potentially escaping closure: use closure struct model
                // Lambda itself gets SSAs in the PARENT scope for closure struct construction
                // SSA count is deterministic based on captures (from CCS)
                // Baker marks zero-capture lambdas in value position with ClosureMetadata.RequiresClosurePair.
                // These need closure pair construction ({code_ptr, null_env}) even with empty captures.
                let requiresClosurePair =
                    node.Metadata
                    |> Map.tryFind ClosureMetadata.RequiresClosurePair
                    |> Option.map (function MetadataValue.Bool b -> b | _ -> false)
                    |> Option.defaultValue false

                let cost =
                    if requiresClosurePair && List.isEmpty captures then
                        // Zero-capture closure pair: need 14 SSAs (c+14 where c=0)
                        // for code_ptr, closure struct alloca, and uniform pair construction
                        14
                    else
                        computeLambdaSSACost ctx.TargetPlatform ctx.Arch ctx.Graph captures

                if cost > 0 then
                    let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                    let alloc = NodeSSAAllocation.multi ssas
                    let scopeWithAlloc = FunctionScope.assign node.Id alloc scopeWithSSAs

                    // Compute ClosureLayout immediately using the allocated SSAs from the parent scope
                    // Pass the context so witnesses know how to extract captures
                    // PRD-14: Pass graph and bodyId for lazy struct type computation
                    // A closure's byte layout is a core's, sized by the declared Pointer width; the
                    // fabric leg holds no closure (a function is an hw.module, a value its instance)
                    // and derives none (HelloArty, CS-11 review)
                    if ctx.TargetPlatform <> Core.Types.Dialects.TargetPlatform.FPGA && (not (List.isEmpty captures) || requiresClosurePair) then
                        let layout = buildClosureLayout ctx.TargetPlatform ctx.Arch ctx.Graph node.Id bodyId captures ssas context
                        if not (ctx.ClosureLayouts.ContainsKey(NodeId.value node.Id)) then
                            ctx.ClosureLayouts.Add(NodeId.value node.Id, layout)
                    scopeWithAlloc
                else
                    // Simple lambda (no captures, not in value position) - no SSAs needed in parent scope
                    scopeAfterChildren
        // VarRef now gets SSAs for mutable variable loads
        // (Regular VarRefs to immutable values may not use their SSAs, but unused SSAs are harmless)

        // ForLoop needs SSAs for internal operation (ivSSA + stepSSA)
        // even though it doesn't "produce a value" in the semantic sense
        | SemanticKind.ForLoop _ ->
            let cost = nodeExpansionCost ctx node  // Structural derivation
            let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
            let alloc = NodeSSAAllocation.multi ssas
            FunctionScope.assign node.Id alloc scopeWithSSAs

        // DUConstruct: Build DULayout coeffect for arena-allocated heterogeneous DUs
        | SemanticKind.DUConstruct (caseName, caseIndex, payloadOpt, _guardExpr) ->
            let cost = nodeExpansionCost ctx node
            let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
            let alloc = NodeSSAAllocation.multi ssas
            let scopeWithAlloc = FunctionScope.assign node.Id alloc scopeWithSSAs |> FunctionScope.meet node.Id (nodeMeets ctx node)

            // Build DULayout for heterogeneous DUs needing arena allocation: a core's settled
            // layout; the fabric leg holds a DU as an hw.struct and reads no such layout
            // (HelloArty, CS-11 review)
            if ctx.TargetPlatform <> Core.Types.Dialects.TargetPlatform.FPGA && needsDUArenaAllocation node.Type then
                let layout = buildDULayout ctx.TargetPlatform ctx.Arch ctx.Graph node.Id node.Type caseName caseIndex payloadOpt ssas
                if not (ctx.DULayouts.ContainsKey(NodeId.value node.Id)) then
                    ctx.DULayouts.Add(NodeId.value node.Id, layout)

            scopeWithAlloc

        // ─────────────────────────────────────────────────────────────────────
        // PATTERN BINDING SSA ALIASING (January 2026)
        // Record pattern bindings: `{ Age = a }` creates PatternBinding with FieldGet child.
        // The PatternBinding is just a NAME for the FieldGet result - it should ALIAS
        // the child's SSA, not allocate new SSAs.
        // Lambda parameter PatternBindings: have no children, Lambda assigns them as Arg.
        // ─────────────────────────────────────────────────────────────────────
        | SemanticKind.PatternBinding _ ->
            if not (List.isEmpty node.Children) then
                // Record pattern binding - alias the first child's SSA (the FieldGet)
                let childId = List.head node.Children
                match Map.tryFind (NodeId.value childId) scopeAfterChildren.Assignments with
                | Some childSSA ->
                    // Alias: PatternBinding gets the same SSA as its FieldGet child
                    FunctionScope.assign node.Id childSSA scopeAfterChildren
                | None ->
                    // Child not in assignments - shouldn't happen with post-order
                    // Fall back to normal allocation
                    let cost = nodeExpansionCost ctx node
                    let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                    let alloc = NodeSSAAllocation.multi ssas
                    FunctionScope.assign node.Id alloc scopeWithSSAs
            else
                // No children: either Lambda parameter (Arg SSA from Lambda processing)
                // or match-arm tuple pattern binding (needs SSAs for field extraction).
                // Check if Lambda processing already assigned an Arg SSA:
                match Map.tryFind (NodeId.value node.Id) scopeAfterChildren.Assignments with
                | Some _ ->
                    // Already assigned by Lambda parameter processing — keep it
                    scopeAfterChildren
                | None ->
                    // Match-arm binding — allocate SSAs for tuple element extraction
                    let cost = nodeExpansionCost ctx node
                    let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                    let alloc = NodeSSAAllocation.multi ssas
                    FunctionScope.assign node.Id alloc scopeWithSSAs

        // ─────────────────────────────────────────────────────────────────────
        // IMMUTABLE BINDING SSA ALIASING (March 2026)
        // Immutable `let` bindings are transparent in MLIR — they forward the
        // child expression's SSA without emitting any ops (BindingWitness).
        // Their SSA assignment must ALIAS the child's, not allocate fresh SSAs,
        // so that closure captures referencing the binding's SourceNodeId resolve
        // to the actually-emitted SSA value.
        // Mutable bindings need their own SSAs for memref.alloca + initialization.
        // ─────────────────────────────────────────────────────────────────────
        | SemanticKind.Binding (_name, isMutable, _isRec, _declRoot) when not isMutable && not (isModuleValueSlotBinding ctx.Graph node) ->
            if not (List.isEmpty node.Children) then
                let childId = List.head node.Children
                match Map.tryFind (NodeId.value childId) scopeAfterChildren.Assignments with
                | Some childSSA ->
                    // Alias: immutable Binding gets the same SSA as its value expression
                    FunctionScope.assign node.Id childSSA scopeAfterChildren
                | None ->
                    // Child not in assignments — fall back to normal allocation
                    let cost = nodeExpansionCost ctx node
                    let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                    let alloc = NodeSSAAllocation.multi ssas
                    FunctionScope.assign node.Id alloc scopeWithSSAs
            else
                // No children — shouldn't happen for Binding, but be safe
                let cost = nodeExpansionCost ctx node
                let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                let alloc = NodeSSAAllocation.multi ssas
                FunctionScope.assign node.Id alloc scopeWithSSAs

        | _ ->
            // Regular node - assign SSAs based on structural analysis, then the meets its
            // consumption derives (the derivation table), right after its own values
            let meets = nodeMeets ctx node @ applicationReadMeets ctx node
            if producesValue node.Kind then
                let cost = nodeExpansionCost ctx node  // Structural derivation
                if cost > 0 then
                    let ssas, scopeWithSSAs = FunctionScope.yieldSSAs cost scopeAfterChildren
                    let alloc = NodeSSAAllocation.multi ssas
                    FunctionScope.assign node.Id alloc scopeWithSSAs |> FunctionScope.meet node.Id meets
                else
                    // Node produces a value conceptually but has 0 SSA cost (e.g., MemRef.store returning unit)
                    scopeAfterChildren |> FunctionScope.meet node.Id meets
            else
                scopeAfterChildren |> FunctionScope.meet node.Id meets

/// Collect all Lambdas in the graph and assign function names
let private collectLambdas (graph: SemanticGraph) : Map<int, string> * Map<int, DeclRoot> =
    let mutable lambdaCounter = 0
    let mutable lambdaNames = Map.empty
    let mutable declRootLambdas : Map<int, DeclRoot> = Map.empty

    // First, identify declaration root lambdas
    // Roots may be ModuleDef nodes containing Binding nodes with declRoot.IsSome
    for (entryId, rootKind) in graph.DeclarationRoots do
        match Map.tryFind entryId graph.Nodes with
        | Some node ->
            match node.Kind with
            | SemanticKind.Binding(_, _, _, dr) when dr.IsSome ->
                // The binding's first child is typically the Lambda
                match node.Children with
                | lambdaId :: _ -> declRootLambdas <- Map.add (NodeId.value lambdaId) dr.Value declRootLambdas
                | _ -> ()
            | SemanticKind.Lambda _ ->
                declRootLambdas <- Map.add (NodeId.value entryId) rootKind declRootLambdas
            | SemanticKind.ModuleDef (_, memberIds) ->
                // ModuleDef root - check members for root Binding
                for memberId in memberIds do
                    match Map.tryFind memberId graph.Nodes with
                    | Some memberNode ->
                        match memberNode.Kind with
                        | SemanticKind.Binding(_, _, _, Some DeclRoot.HardwareModule) ->
                            ()  // HardwareModule bindings are structural metadata, not Lambda entry points
                        | SemanticKind.Binding(_, _, _, Some dr) ->
                            match memberNode.Children with
                            | lambdaId :: _ -> declRootLambdas <- Map.add (NodeId.value lambdaId) dr declRootLambdas
                            | _ -> ()
                        | SemanticKind.Binding(name, _, _, None) when name = "main" ->
                            match memberNode.Children with
                            | lambdaId :: _ -> declRootLambdas <- Map.add (NodeId.value lambdaId) DeclRoot.EntryPoint declRootLambdas
                            | _ -> ()
                        | _ -> ()
                    | None -> ()
            | _ -> ()
        | None -> ()

    // PRD-13: Find enclosing function by walking Parent chain
    // Structure: Lambda -> Binding("loop") -> ... -> Lambda -> Binding("factorialTail")
    let findEnclosingFunctionName (startId: NodeId) : string option =
        let rec walk (nodeId: NodeId) (passedFirstLambda: bool) =
            match Map.tryFind nodeId graph.Nodes with
            | None -> None
            | Some n ->
                match n.Kind with
                | SemanticKind.Lambda _ when passedFirstLambda ->
                    // Found enclosing Lambda - get its parent Binding's name
                    match n.Parent with
                    | Some parentId ->
                        match Map.tryFind parentId graph.Nodes with
                        | Some { Kind = SemanticKind.Binding(name, _, _, _) } -> Some name
                        | _ -> None
                    | None -> None
                | _ ->
                    match n.Parent with
                    | Some parentId -> walk parentId true
                    | None -> None
        walk startId false

    // Assign names to all Lambdas
    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda (_, _, _, lambdaNameOpt, _) ->
            let nodeIdVal = NodeId.value node.Id
            let name =
                // Get base name from parent Binding first
                let parentBindingName =
                    match node.Parent with
                    | Some parentId ->
                        match Map.tryFind parentId graph.Nodes with
                        | Some { Kind = SemanticKind.Binding(bindingName, _, _, _) } -> Some bindingName
                        | _ -> None
                    | None -> None

                // Only use lambdaNameOpt if it matches the parent Binding name
                // This distinguishes explicit names (like "_start") from inherited context
                // (where lambdaNameOpt = env.EnclosingFunction for capture analysis)
                match lambdaNameOpt, parentBindingName with
                | Some explicitName, Some parentName when explicitName = parentName -> explicitName
                | _, _ ->
                    if Map.containsKey nodeIdVal declRootLambdas then
                        "main"
                    else
                        // Use already-computed parentBindingName
                        match parentBindingName with
                        | Some bname ->
                            // Check if nested by walking Parent chain
                            match findEnclosingFunctionName node.Id with
                            | Some enclosing -> sprintf "%s_%s" enclosing bname
                            | None -> bname
                        | None ->
                            let n = sprintf "lambda_%d" lambdaCounter
                            lambdaCounter <- lambdaCounter + 1
                            n

            lambdaNames <- Map.add nodeIdVal name lambdaNames

            // Also store for parent Binding's NodeId (VarRefs point to Bindings)
            match node.Parent with
            | Some parentId ->
                match Map.tryFind parentId graph.Nodes with
                | Some { Kind = SemanticKind.Binding _ } ->
                    lambdaNames <- Map.add (NodeId.value parentId) name lambdaNames
                | _ -> ()
            | None -> ()
        | _ -> ()

    lambdaNames, declRootLambdas

/// Check if a Binding contains a Lambda (function definition vs value binding)
let private isLambdaBinding (graph: SemanticGraph) (bindingNodeId: NodeId) : bool =
    match Map.tryFind bindingNodeId graph.Nodes with
    | Some node ->
        match node.Kind with
        | SemanticKind.Binding _ ->
            match node.Children with
            | childId :: _ ->
                match Map.tryFind childId graph.Nodes with
                | Some childNode ->
                    match childNode.Kind with
                    | SemanticKind.Lambda _ -> true
                    | _ -> false
                | None -> false
            | [] -> false
        | _ -> false
    | None -> false

/// Find the main Lambda NodeId from declaration roots
let private findMainLambdaId (graph: SemanticGraph) (declRootLambdas: Map<int, DeclRoot>) : NodeId option =
    // Declaration roots are either:
    // 1. ModuleDef containing Bindings (one of which is main)
    // 2. Direct Lambda node
    graph.DeclarationRoots
    |> List.tryPick (fun (entryId, _) ->
        match Map.tryFind entryId graph.Nodes with
        | Some node ->
            match node.Kind with
            | SemanticKind.Lambda _ when Map.containsKey (NodeId.value entryId) declRootLambdas ->
                Some entryId
            | SemanticKind.ModuleDef (_, memberIds) ->
                // Find the main Lambda among module members
                memberIds |> List.tryPick (fun memberId ->
                    match Map.tryFind memberId graph.Nodes with
                    | Some memberNode ->
                        match memberNode.Kind with
                        | SemanticKind.Binding (_, _, _, dr) when dr.IsSome ->
                            // Get the Lambda child of this binding
                            match memberNode.Children with
                            | lambdaId :: _ when Map.containsKey (NodeId.value lambdaId) declRootLambdas ->
                                Some lambdaId
                            | _ -> None
                        | _ -> None
                    | None -> None)
            | _ -> None
        | None -> None)

/// Find module-level VALUE bindings (non-Lambda bindings that are siblings of main)
/// These need SSAs in main's scope because they're emitted in main's prologue
let private findModuleLevelValueBindings (graph: SemanticGraph) (mainLambdaId: NodeId) : NodeId list =
    // Find the ModuleDef containing main
    graph.DeclarationRoots
    |> List.collect (fun (entryId, _) ->
        match Map.tryFind entryId graph.Nodes with
        | Some node ->
            match node.Kind with
            | SemanticKind.ModuleDef (_, memberIds) ->
                // Get all Binding members that are NOT Lambda bindings
                // and are NOT the main binding itself
                memberIds
                |> List.filter (fun memberId ->
                    match Map.tryFind memberId graph.Nodes with
                    | Some memberNode ->
                        match memberNode.Kind with
                        | SemanticKind.Binding _ ->
                            // Check if this binding contains a Lambda
                            let isLambda = isLambdaBinding graph memberId
                            // Also check if this binding's child is main
                            let containsMain =
                                match memberNode.Children with
                                | childId :: _ -> childId = mainLambdaId
                                | [] -> false
                            // Include if it's a VALUE binding (not Lambda, not main)
                            not isLambda && not containsMain
                        | _ -> false
                    | None -> false)
            | _ -> []
        | None -> [])

/// Main entry point: assign SSA names to all nodes in the graph
///
/// TWO-PASS SSA ASSIGNMENT:
/// Pass 1: Module-level VALUE bindings get SSAs in main's scope (emitted in main's prologue)
/// Pass 2: Each Lambda body gets its own scope with counter starting at 0
///
/// This prevents SSA collisions between module-level bindings and function bodies.
/// Find ALL module-level VALUE bindings from ALL ModuleDefs (no main Lambda required).
/// Used for FPGA and other targets where there is no "main" entry point Lambda.
/// Excludes Lambda bindings (functions) and HardwareModule bindings (FPGA metadata).
let private findAllModuleLevelValueBindings (graph: SemanticGraph) : NodeId list =
    // Scan ALL ModuleDef nodes in the graph, not just DeclarationRoots.
    // On FPGA, platform modules (Prelude, Behavior) contain module-level constants
    // that are referenced via VarRef from the HardwareModule's step function.
    graph.Nodes
    |> Map.toList
    |> List.collect (fun (_, node) ->
        match node.Kind with
        | SemanticKind.ModuleDef (_, memberIds) ->
            memberIds
            |> List.filter (fun memberId ->
                match Map.tryFind memberId graph.Nodes with
                | Some memberNode ->
                    match memberNode.Kind with
                    | SemanticKind.Binding (_, _, _, Some DeclRoot.HardwareModule) ->
                        false  // Skip HardwareModule bindings (structural metadata)
                    | SemanticKind.Binding _ ->
                        not (isLambdaBinding graph memberId)  // Include value bindings, exclude functions
                    | _ -> false
                | None -> false)
        | _ -> [])


// ═══════════════════════════════════════════════════════════════════════════
// HARDWARE MODULE LAYOUT DERIVATION (FPGA)
// ═══════════════════════════════════════════════════════════════════════════

/// See through TypeAnnotation to the node it annotates
let rec private unwrapTypeAnnotation (graph: SemanticGraph) (nodeId: NodeId) : NodeId =
    match Map.tryFind nodeId graph.Nodes with
    | Some { Kind = SemanticKind.TypeAnnotation (innerId, _) } -> unwrapTypeAnnotation graph innerId
    | _ -> nodeId

/// The values of the Mealy machine an [<HardwareModule>] binding describes, derived from the
/// Design record's structure (the InitialState fields, the Step function's signature) and the
/// platform's pin facts, numbered from zero in the order HardwareModulePatterns emits them.
/// None when the binding does not carry a Design the pattern can build; the witness reports
/// the shape it found.
let private deriveHardwareModuleLayout (targetPlatform: Core.Types.Dialects.TargetPlatform) (arch: Architecture) (graph: SemanticGraph) (pinMapping: PlatformPinMapping option) (binding: SemanticNode) : HardwareModuleLayout option =
    let designFields =
        match binding.Children with
        | [valueId] ->
            match Map.tryFind (unwrapTypeAnnotation graph valueId) graph.Nodes with
            | Some { Kind = SemanticKind.RecordExpr (fields, _) } -> Some fields
            | _ -> None
        | _ -> None
    let field name = designFields |> Option.bind (List.tryFind (fun (n, _) -> n = name)) |> Option.map snd
    let stateFieldCount =
        field "InitialState" |> Option.bind (fun initId ->
            match Map.tryFind (unwrapTypeAnnotation graph initId) graph.Nodes with
            | Some { Kind = SemanticKind.RecordExpr (fields, _) } -> Some fields.Length
            | _ -> None)
    let stepLambda =
        field "Step" |> Option.bind (fun stepId ->
            match Map.tryFind stepId graph.Nodes with
            | Some { Kind = SemanticKind.VarRef (_, Some defId) } ->
                match Map.tryFind defId graph.Nodes with
                | Some defNode ->
                    match defNode.Children with
                    | [lambdaId] ->
                        match Map.tryFind lambdaId graph.Nodes with
                        | Some { Kind = SemanticKind.Lambda (params', bodyId, _, _, _) } -> Some (params', bodyId)
                        | _ -> None
                    | _ -> None
                | None -> None
            | _ -> None)
    match stateFieldCount, stepLambda with
    | Some n, Some (params', bodyId) ->
        // The witness's own mapping (TransferTypes.mapType): the target's shape of a record, a tuple,
        // an option; narrowing afterwards changes widths only, so the shapes agree
        let mapTy ty = mapNativeTypeForTarget targetPlatform arch graph ty
        let inputType =
            match params' with
            | _ :: (_, inputTy, _) :: _ -> Some (mapTy inputTy)
            | _ -> None
        let outputType =
            match Map.tryFind bodyId graph.Nodes with
            | Some bodyNode ->
                match mapTy bodyNode.Type with
                | TStruct (("Item1", _) :: ("Item2", outTy) :: _, _) -> Some outTy
                | _ -> None
            | None -> None
        let pinAttrs = pinMapping |> Option.map (fun m -> m.FieldPinAttrs) |> Option.defaultValue Map.empty
        // The flat-port module (a pin mapping) synthesises a power-on reset unless the platform
        // declares an external one; the struct-port module takes rst as a port
        let internalReset =
            match pinMapping with
            | Some m -> not (m.Reset |> Option.map (fun r -> r.IsExternal) |> Option.defaultValue false)
            | None -> false
        let inputPackCount, hasInputStruct =
            match pinMapping, inputType with
            | Some _, Some (TStruct (fields, _)) ->
                (fields |> List.sumBy (fun (name, ty) ->
                    match Map.tryFind name pinAttrs, ty with
                    | Some pins, TStruct _ when pins.Length > 1 -> 1
                    | _ -> 0)), true
            | _ -> 0, false
        let flattenCount =
            match pinMapping, outputType with
            | Some _, Some outTy -> (outputExtractions pinAttrs outTy).Length
            | _ -> 0
        // One numbering, in emission order: power-on reset, reset constants, registers, input
        // packs, input struct, state, instance, step result, output flatten, next fields
        let counts =
            [ (if internalReset then 3 else 0); n; n; inputPackCount; (if hasInputStruct then 1 else 0)
              1; 1; (if outputType.IsSome then 2 else 0); flattenCount; n ]
        let starts = counts |> List.scan (+) 0
        let values i = List.init counts.[i] (fun k -> V (starts.[i] + k))
        Some {
            BindingNodeId = binding.Id
            PowerOnReset = (match values 0 with [a; b; c] -> Some (a, b, c) | _ -> None)
            ResetValues = values 1
            Registers = values 2
            InputPacks = values 3
            InputStruct = List.tryHead (values 4)
            State = List.head (values 5)
            Instance = List.head (values 6)
            StepResult = (match values 7 with [a; b] -> Some (a, b) | _ -> None)
            OutputFlatten = values 8
            NextFields = values 9
        }
    | _ -> None

let assignSSA (targetPlatform: Core.Types.Dialects.TargetPlatform) (arch: Architecture) (graph: SemanticGraph) (curry: CurryFlattening.CurryFlatteningResult) (pinMapping: PlatformPinMapping option) : SSAAssignment =
    let lambdaNames, declRootLambdas = collectLambdas graph

    let mutable allAssignments = Map.empty
    let mutable allMeets : Map<int, Meet list> = Map.empty
    let mutableClosureLayouts = System.Collections.Generic.Dictionary<int, ClosureLayout>()
    let mutableDULayouts = System.Collections.Generic.Dictionary<int, DULayout>()
    let mutableInnerScopeAssignments = System.Collections.Generic.Dictionary<int, NodeSSAAllocation>()
    let mutableInnerMeets = System.Collections.Generic.Dictionary<int, Meet list>()
    let mutableUnitReturns = System.Collections.Generic.Dictionary<int, SSA>()
    let mutableReturnMeets = System.Collections.Generic.Dictionary<int, Meet>()

    let ctx : SSAContext = {
        TargetPlatform = targetPlatform
        Arch = arch
        Graph = graph
        ClosureLayouts = mutableClosureLayouts
        DULayouts = mutableDULayouts
        InnerScopeAssignments = mutableInnerScopeAssignments
        InnerMeets = mutableInnerMeets
        UnitReturns = mutableUnitReturns
        ReturnMeets = mutableReturnMeets
        Curry = curry
    }

    // Find the main Lambda
    let mainLambdaIdOpt = findMainLambdaId graph declRootLambdas

    // ═══════════════════════════════════════════════════════════════════════════
    // PASS 1: Module-level VALUE bindings (emitted in main's prologue)
    // These share main's SSA namespace, so process them first and continue counter
    // ═══════════════════════════════════════════════════════════════════════════
    // Module-level value bindings from EVERY module: they are all initialized in the entry
    // point's prologue (slots), so they all live in main's SSA namespace.
    let moduleLevelValueBindings =
        match mainLambdaIdOpt with
        | Some _ -> findAllModuleLevelValueBindings graph
        | None -> findAllModuleLevelValueBindings graph

    // Assign SSAs to module-level value bindings
    // These will use %v0, %v1, ... and main's body continues from there
    let moduleLevelScope =
        moduleLevelValueBindings
        |> List.fold (fun scope bindingId ->
            assignFunctionBody ctx scope bindingId
        ) FunctionScope.empty

    // Track the counter after module-level bindings
    let moduleLevelCounter = moduleLevelScope.Counter

    // Merge module-level assignments
    for kvp in moduleLevelScope.Assignments do
        allAssignments <- Map.add kvp.Key kvp.Value allAssignments
    for kvp in moduleLevelScope.Meets do
        allMeets <- Map.add kvp.Key kvp.Value allMeets

    // ═══════════════════════════════════════════════════════════════════════════
    // PASS 2: Lambda bodies (each gets its own scope)
    // Non-main Lambdas start at counter 0
    // Main Lambda starts at moduleLevelCounter (continues from Pass 1)
    // ═══════════════════════════════════════════════════════════════════════════

    // Track module-level SSA counter for top-level Lambda nodes
    let mutable topLevelCounter = moduleLevelCounter

    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.Lambda(params', bodyId, captures, enclosingFuncOpt, _context) ->
            // FIX (January 2026): Only process top-level Lambdas to avoid double-processing
            // Nested Lambdas (with enclosingFuncOpt = Some _) are already handled recursively
            // by assignFunctionBody when processing their enclosing Lambda's body
            let isTopLevel = Option.isNone enclosingFuncOpt

            if isTopLevel then
                let nodeIdVal = NodeId.value node.Id
                let isMain = Map.containsKey nodeIdVal declRootLambdas &&
                             (mainLambdaIdOpt |> Option.map (fun id -> NodeId.value id = nodeIdVal) |> Option.defaultValue false)

                // Main Lambda continues from module-level counter; others start fresh
                // FPGA FIX (February 2026): When there's no main Lambda, ALL top-level Lambdas
                // start from module-level counter. Module-level constants are VarRef-followed
                // into Lambda bodies during FPGA walk, so their SSAs must not collide.
                let initialCounter =
                    if isMain then moduleLevelCounter
                    elif Option.isNone mainLambdaIdOpt then moduleLevelCounter
                    else 0
                // A unit-typed body returns a zero constant: the first value of the body's scope
                let unitReturnCount =
                    match Map.tryFind bodyId graph.Nodes with
                    | Some bodyNode when isUnitTyped bodyNode.Type ->
                        mutableUnitReturns.[nodeIdVal] <- V initialCounter
                        1
                    | _ -> 0
                let initialScope = { FunctionScope.empty with Counter = initialCounter + unitReturnCount }

                // Assign SSAs to parameter PatternBindings (Arg 0, Arg 1, etc.)
                // For closures: offset by 1 because Arg 0 = env_ptr (closure struct)
                let requiresClosurePairTopLevel =
                    node.Metadata
                    |> Map.tryFind ClosureMetadata.RequiresClosurePair
                    |> Option.map (function MetadataValue.Bool b -> b | _ -> false)
                    |> Option.defaultValue false
                let argOffsetTopLevel = if not (List.isEmpty captures) || requiresClosurePairTopLevel then 1 else 0
                let paramScope =
                    params'
                    |> List.mapi (fun i (_name, _ty, nodeId) -> i + argOffsetTopLevel, nodeId)
                    |> List.fold (fun (scope: FunctionScope) (i, nodeId) ->
                        FunctionScope.assign nodeId (NodeSSAAllocation.single (Arg i)) scope
                    ) initialScope

                // Assign SSAs to body nodes, then the return meet as the body's last value
                // This will also compute ClosureLayouts for any nested lambdas found in the body
                let bodyScope = returnMeet ctx node.Id bodyId (assignFunctionBody ctx paramScope bodyId)

                // Merge into global assignments (including parameter SSAs)
                for kvp in paramScope.Assignments do
                    allAssignments <- Map.add kvp.Key kvp.Value allAssignments
                for kvp in bodyScope.Assignments do
                    allAssignments <- Map.add kvp.Key kvp.Value allAssignments
                for kvp in bodyScope.Meets do
                    allMeets <- Map.add kvp.Key kvp.Value allMeets

                // Merge nested lambda scope assignments collected during recursive traversal
                for kvp in mutableInnerScopeAssignments do
                    allAssignments <- Map.add kvp.Key kvp.Value allAssignments
                mutableInnerScopeAssignments.Clear()
                for kvp in mutableInnerMeets do
                    allMeets <- Map.add kvp.Key kvp.Value allMeets
                mutableInnerMeets.Clear()

                // Assign SSAs to the Lambda node itself (for closure value)
                // Top-level Lambdas (not visited during body traversal) need SSA assignments
                // for closure construction if they have captures
                let cost = computeLambdaSSACost targetPlatform arch graph captures
                if cost > 0 then
                    // Lambda with captures needs SSAs for closure struct construction
                    let ssas = List.init cost (fun i -> V (topLevelCounter + i))
                    let alloc = NodeSSAAllocation.multi ssas
                    allAssignments <- Map.add nodeIdVal alloc allAssignments
                    topLevelCounter <- topLevelCounter + cost
                // Note: Lambdas with no captures (cost=0) don't need SSA assignments
                // They are emitted as direct function symbols

                // Assign SSAs to parent Binding if this Lambda has one
                // Lambda Bindings are filtered out of Pass 1, so they need SSA assignment here
                match node.Parent with
                | Some parentId ->
                    match Map.tryFind parentId graph.Nodes with
                    | Some parentNode when (match parentNode.Kind with SemanticKind.Binding _ -> true | _ -> false) ->
                        let parentIdVal = NodeId.value parentId
                        if not (Map.containsKey parentIdVal allAssignments) then
                            // Binding needs SSAs (fixed cost of 3)
                            let bindingCost = 3
                            let bindingSSAs = List.init bindingCost (fun i -> V (topLevelCounter + i))
                            let bindingAlloc = NodeSSAAllocation.multi bindingSSAs
                            allAssignments <- Map.add parentIdVal bindingAlloc allAssignments
                            topLevelCounter <- topLevelCounter + bindingCost
                    | _ -> ()
                | None -> ()

        // PRD-15 FIX (January 2026): SeqExpr MoveNext bodies need their own SSA scope
        // MoveNext is a separate function with seqPtr as %arg0, body SSAs start at 1
        | SemanticKind.SeqExpr (bodyId, _captures) ->
            // MoveNext function: %arg0 = seqPtr, body SSAs start at v1
            let initialScope = { FunctionScope.empty with Counter = 1 }
            let bodyScope = assignFunctionBody ctx initialScope bodyId

            // Merge SeqExpr body assignments
            for kvp in bodyScope.Assignments do
                allAssignments <- Map.add kvp.Key kvp.Value allAssignments
            for kvp in bodyScope.Meets do
                allMeets <- Map.add kvp.Key kvp.Value allMeets

        | _ -> ()

    // ═══════════════════════════════════════════════════════════════════════════
    // VALIDATION: Check for unassigned value-producing nodes
    // ═══════════════════════════════════════════════════════════════════════════
    let unassignedNodes =
        graph.Nodes
        |> Map.toList
        |> List.filter (fun (_, node) ->
            let nodeIdVal = NodeId.value node.Id
            if not node.IsReachable then false
            elif not (producesValue node.Kind) then false
            elif Map.containsKey nodeIdVal allAssignments then false
            else
                // Node produces value but has no SSA - check if this is expected
                match node.Kind with
                | SemanticKind.Lambda (_, _, captures, _, _) ->
                    // No-capture Lambdas are emitted as direct function symbols, not SSA values
                    // Only Lambdas with captures need SSAs for closure struct construction
                    not (List.isEmpty captures)
                | SemanticKind.Intrinsic _ ->
                    // Intrinsic function nodes (TFun types) are just references, not values
                    // Only intrinsic CALLS (via Application) produce values
                    match node.Type with
                    | NativeType.TFun _ -> false  // Function reference, not a value
                    | _ -> true  // Non-function intrinsic should have SSA
                | _ -> true)
        |> List.map (fun (_, node) -> NodeId.value node.Id, node.Kind)

    // Convert mutable dictionaries to immutable maps
    let closureLayouts = 
        mutableClosureLayouts
        |> Seq.map (fun kvp -> kvp.Key, kvp.Value)
        |> Map.ofSeq

    let duLayouts =
        mutableDULayouts
        |> Seq.map (fun kvp -> kvp.Key, kvp.Value)
        |> Map.ofSeq

    {
        NodeSSA = allAssignments
        LambdaNames = lambdaNames
        DeclarationRootLambdas = declRootLambdas
        ClosureLayouts = closureLayouts
        DULayouts = duLayouts
        HardwareModuleLayouts =
            graph.Nodes
            |> Map.toList
            |> List.choose (fun (_, node) ->
                match node.Kind with
                | SemanticKind.Binding (_, _, _, Some DeclRoot.HardwareModule) ->
                    deriveHardwareModuleLayout targetPlatform arch graph pinMapping node
                    |> Option.map (fun layout -> (NodeId.value node.Id, layout))
                | _ -> None)
            |> Map.ofList
        UnitReturns = mutableUnitReturns |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Map.ofSeq
        Meets = allMeets
        ReturnMeets = mutableReturnMeets |> Seq.map (fun kv -> (kv.Key, kv.Value)) |> Map.ofSeq
    }

/// Look up the full SSA allocation for a node (coeffect lookup)
let lookupAllocation (nodeId: NodeId) (assignment: SSAAssignment) : NodeSSAAllocation option =
    Map.tryFind (NodeId.value nodeId) assignment.NodeSSA

/// Look up just the result SSA for a node (most common use case)
let lookupSSA (nodeId: NodeId) (assignment: SSAAssignment) : SSA option =
    lookupAllocation nodeId assignment |> Option.map (fun a -> a.Result)

/// Look up all SSAs for a node (for witnesses that need intermediates)
let lookupSSAs (nodeId: NodeId) (assignment: SSAAssignment) : SSA list option =
    lookupAllocation nodeId assignment |> Option.map (fun a -> a.SSAs)

/// Look up the function name for a Lambda
let lookupLambdaName (nodeId: NodeId) (assignment: SSAAssignment) : string option =
    Map.tryFind (NodeId.value nodeId) assignment.LambdaNames

/// Check if a Lambda is a declaration root and return its flavor
let tryGetDeclRoot (nodeId: NodeId) (assignment: SSAAssignment) : DeclRoot option =
    Map.tryFind (NodeId.value nodeId) assignment.DeclarationRootLambdas

/// Look up ClosureLayout for a Lambda with captures (coeffect lookup)
/// Returns None for simple lambdas (no captures)
let lookupClosureLayout (nodeId: NodeId) (assignment: SSAAssignment) : ClosureLayout option =
    Map.tryFind (NodeId.value nodeId) assignment.ClosureLayouts

/// Check if a Lambda has captures (is a closure)
let hasClosure (nodeId: NodeId) (assignment: SSAAssignment) : bool =
    Map.containsKey (NodeId.value nodeId) assignment.ClosureLayouts

/// Look up DULayout for a DUConstruct node needing arena allocation (coeffect lookup)
/// Returns None for homogeneous DUs like Option that use inline struct
let lookupDULayout (nodeId: NodeId) (assignment: SSAAssignment) : DULayout option =
    Map.tryFind (NodeId.value nodeId) assignment.DULayouts

/// Check if a DUConstruct node needs arena allocation
let hasDULayout (nodeId: NodeId) (assignment: SSAAssignment) : bool =
    Map.containsKey (NodeId.value nodeId) assignment.DULayouts

/// The Mealy machine values derived for an [<HardwareModule>] binding (coeffect lookup)
let lookupHardwareModuleLayout (bindingId: NodeId) (assignment: SSAAssignment) : HardwareModuleLayout option =
    Map.tryFind (NodeId.value bindingId) assignment.HardwareModuleLayouts

/// The zero constant a unit-typed function returns (coeffect lookup); None for a function
/// whose body has a value
let lookupUnitReturn (lambdaId: NodeId) (assignment: SSAAssignment) : SSA option =
    Map.tryFind (NodeId.value lambdaId) assignment.UnitReturns

/// The meet derived for a consumer's operand (coeffect lookup): the value that brings the
/// operand to the width of the slot it meets; None where the widths agree. A read of a slot
/// names the consumer as its own operand.
let lookupMeet (consumer: NodeId) (operand: NodeId) (assignment: SSAAssignment) : Meet option =
    Map.tryFind (NodeId.value consumer) assignment.Meets
    |> Option.bind (List.tryFind (fun m -> m.Operand = operand))

/// The return meet of a lambda (coeffect lookup); None where the last value is held at the
/// result's width
let lookupReturnMeet (lambdaId: NodeId) (assignment: SSAAssignment) : Meet option =
    Map.tryFind (NodeId.value lambdaId) assignment.ReturnMeets

/// PRD-14/PRD-15: Get the actual return type for a function that may return a lazy or seq with captures.
/// If the function body is a LazyExpr with captures, returns the actual lazy struct type
/// including the inlined captures: {i1, T, ptr, cap0, cap1, ...}
/// If the function body is a SeqExpr with captures, returns the actual seq struct type
/// including the inlined captures: {i32, T, ptr, cap0, cap1, ...}
/// Returns None if the function doesn't return a lazy/seq with captures.
let getActualFunctionReturnType (platform: Core.Types.Dialects.TargetPlatform) (arch: Architecture) (graph: SemanticGraph) (defId: NodeId) (assignment: SSAAssignment) : MLIRType option =
    // defId may be a Binding node - need to find the Lambda child
    let lambdaNode =
        match Map.tryFind defId graph.Nodes with
        | Some node ->
            match node.Kind with
            | SemanticKind.Lambda _ -> Some node
            | SemanticKind.Binding _ ->
                // Binding's first child is typically the Lambda
                match node.Children with
                | childId :: _ ->
                    match Map.tryFind childId graph.Nodes with
                    | Some childNode ->
                        match childNode.Kind with
                        | SemanticKind.Lambda _ -> Some childNode
                        | _ -> None
                    | None -> None
                | [] -> None
            | _ -> None
        | None -> None

    match lambdaNode with
    | None -> None
    | Some lambda ->
        match lambda.Kind with
        | SemanticKind.Lambda (_, bodyId, _, _, _) ->
            // Check if body is a LazyExpr
            match Map.tryFind bodyId graph.Nodes with
            | Some bodyNode ->
                match bodyNode.Kind with
                | SemanticKind.LazyExpr (_, captures) when not (List.isEmpty captures) ->
                    // Function returns a lazy with captures
                    // Compute the actual lazy struct type: {i1, T, ptr, cap0, cap1, ...}
                    // Get element type from the LazyExpr's type
                    let elemMlir =
                        match bodyNode.Type with
                        | NativeType.TLazy elemType -> mapNativeTypeForTarget platform arch graph elemType
                        | other -> failwithf "getActualFunctionReturnType: a lazy body of type %s" (formatType other)

                    // Compute capture types using the same logic as closure construction
                    let captureTypes = captures |> List.map (captureSlotType platform arch graph)

                    // Build the actual lazy struct type with captures inlined
                    let fieldTypes = TInt (IntWidth 1) :: elemMlir :: TIndex :: captureTypes
                    let totalBytes = fieldTypes |> List.sumBy (mlirTypeSize arch)
                    let actualLazyType = TMemRefStatic(totalBytes, TInt (IntWidth 8))
                    Some actualLazyType

                // PRD-15: Sequence expressions with captures and/or internal state
                | SemanticKind.SeqExpr (seqBodyId, captures) ->
                    // Function returns a seq - check if it has captures OR internal state
                    let numInternalState = countMutableBindingsInSubtree graph seqBodyId
                    if List.isEmpty captures && numInternalState = 0 then
                        None  // No captures, no internal state - use simple type
                    else
                        // Compute the actual seq struct type: {i32, T, ptr, cap0, ..., state0, ...}
                        // where i32 is state (vs i1 computed flag for lazy)
                        let elemMlir =
                            match bodyNode.Type with
                            | NativeType.TSeq elemType -> mapNativeTypeForTarget platform arch graph elemType
                            | other -> failwithf "getActualFunctionReturnType: a seq body of type %s" (formatType other)

                        // Compute capture types using the same logic as closure construction
                        let captureTypes = captures |> List.map (captureSlotType platform arch graph)

                        // PRD-15 THROUGH-LINE: Internal state fields are also part of struct
                        // They're initialized to default (0), MoveNext state 0 sets actual values.
                        // Held at the declared Register width: a seq's internal state is the leg's
                        // own aggregate (owed to the settled layouts with PRD-15)
                        let internalStateTypes = List.replicate numInternalState (TInt (declaredWordWidth arch))

                        // Build the actual seq struct type with captures + internal state inlined
                        // Layout: {state: i32, current: T, code_ptr: ptr, cap0..., state0...}
                        let fieldTypes = TInt (IntWidth 32) :: elemMlir :: TIndex :: captureTypes @ internalStateTypes
                        let totalBytes = fieldTypes |> List.sumBy (mlirTypeSize arch)
                        let actualSeqType = TMemRefStatic(totalBytes, TInt (IntWidth 8))
                        Some actualSeqType

                | _ -> None
            | None -> None
        | _ -> None
