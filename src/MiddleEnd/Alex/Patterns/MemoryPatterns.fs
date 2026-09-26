/// MemoryPatterns - Memory operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns to elide memory operations to MLIR.
/// Patterns compose Elements (internal) into semantic memory operations.
module Alex.Patterns.MemoryPatterns

open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId - MUST be before TransferTypes
open XParsec
open XParsec.Parsers     // fail, preturn
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.XParsec.Extensions // sequence combinator
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRAtomics
open Alex.Elements.MemRefElements
open Alex.Elements.ArithElements
open Alex.Elements.IndexElements
open Alex.Elements.FuncElements
open Alex.Elements.SCFElements
open Alex.CodeGeneration.TypeMapping
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core

// ═══════════════════════════════════════════════════════════
// FIELD EXTRACTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract single field from struct
/// SSA layout (2 total):
///   [0] = offsetConstSSA - index constant for memref.load
///   [1] = resultSSA - result of the load
let pExtractField (ssas: SSA list) (structSSA: SSA) (fieldIndex: int) (structTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 2) $"pExtractField: Expected 2 SSAs, got {ssas.Length}"
        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]
        return! pExtractValue resultSSA structSSA fieldIndex offsetSSA structTy
    }

// ═══════════════════════════════════════════════════════════
// DU PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract DU tag (handles both inline and pointer-based DUs)
/// Pointer-based: Load tag byte from offset 0
/// Inline: ExtractValue at index 0
/// SSAs extracted from coeffects via nodeId: [0] = indexZeroSSA, [1] = tagSSA (result)
let pExtractDUTag (nodeId: NodeId) (duSSA: SSA) (duType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        let tagTy = TInt (IntWidth 8)  // DU tags are always i8

        match duType with
        | TIndex ->
            // Pointer-based DU: load tag byte from offset 0
            do! ensure (ssas.Length >= 2) $"pExtractDUTag (pointer): Expected 2 SSAs, got {ssas.Length}"
            let indexZeroSSA = ssas.[0]
            let tagSSA = ssas.[1]
            let! indexZeroOp = pConstI indexZeroSSA 0L TIndex
            let! loadOp = pLoad tagSSA duSSA [indexZeroSSA]
            return ([indexZeroOp; loadOp], TRValue { SSA = tagSSA; Type = tagTy })
        | _ ->
            // Inline struct DU: typed extract via reinterpret_cast at byte offset 0
            do! ensure (ssas.Length >= 3) $"pExtractDUTag (inline): Expected 3 SSAs, got {ssas.Length}"
            let castSSA = ssas.[0]
            let zeroSSA = ssas.[1]
            let tagSSA = ssas.[2]
            let! ops = pTypedExtract tagSSA duSSA 0 castSSA zeroSSA tagTy duType
            return (ops, TRValue { SSA = tagSSA; Type = tagTy })
    }

/// The payload offset of a union, read from the settled layout of the union's type
/// (`SemanticGraph.Layouts`, ruling 2: the tag, then the payload slot of the widest case).
let unionPayloadOffset (graph: SemanticGraph) (unionTy: NativeType) : int =
    match settledLayout graph unionTy with
    | Some (SettledLayout.Union (_, Some offset, _, _)) -> offset
    | Some other -> failwithf "unionPayloadOffset: '%s' has the settled layout %A, not a union's with a payload offset" (formatType unionTy) other
    | None -> failwithf "unionPayloadOffset: '%s' has no settled layout on the graph" (formatType unionTy)

/// Extract DU payload via memref.view (different element type: byte buffer → typed payload)
/// SSAs extracted from coeffects via nodeId: [0] = offsetSSA, [1] = viewSSA, [2] = zeroSSA, [3] = extractSSA
let pExtractDUPayload (nodeId: NodeId) (duSSA: SSA) (duType: MLIRType) (unionNativeType: NativeType) (payloadType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 4) $"pExtractDUPayload: Expected 4 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let viewSSA = ssas.[1]
        let zeroSSA = ssas.[2]
        let extractSSA = ssas.[3]

        let! state = getUserState
        let payloadByteOffset = unionPayloadOffset state.Graph unionNativeType

        // Typed extract via memref.view — payload has different element type than byte buffer
        let! extractOps = pTypedExtractView extractSSA duSSA payloadByteOffset offsetSSA viewSSA zeroSSA payloadType duType
        return (extractOps, TRValue { SSA = extractSSA; Type = payloadType })
    }

// ═══════════════════════════════════════════════════════════
// ARRAY PATTERNS
// ═══════════════════════════════════════════════════════════

/// Build Arena.create pattern
/// Allocates an arena buffer on the stack
///
/// Arena.create<'lifetime>(sizeBytes: int) : Arena<'lifetime>
/// Returns: memref<sizeBytes x i8> (stack-allocated byte buffer)
/// SSA extracted from coeffects via nodeId: [0] = result
let pArenaCreate (nodeId: NodeId) (sizeBytes: int) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pArenaCreate: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Allocate arena memory block on stack as byte array
        // Arena IS the memref - no separate control struct in memref semantics
        let elemType = TInt (IntWidth 8)
        let! allocaOp = pAlloca resultSSA sizeBytes elemType None
        let memrefTy = TMemRefStatic (sizeBytes, elemType)

        // Return the arena memref (byte buffer)
        return ([allocaOp], TRValue { SSA = resultSSA; Type = memrefTy })
    }

/// Build Arena.alloc pattern
/// Allocates memory from an arena
///
/// Arena.alloc(arena: Arena<'lifetime> byref, sizeBytes: int) : nativeint
/// For now: returns the arena memref itself (simplified - proper bump allocation later)
/// TODO: Implement proper bump-pointer allocation with memref.subview and offset tracking
/// SSA extracted from coeffects via nodeId: [0] = result
let pArenaAlloc (nodeId: NodeId) (arenaSSA: SSA) (sizeSSA: SSA) (arenaType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pArenaAlloc: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Simplified implementation: return arena memref as the allocated pointer
        // The memref IS the allocation - caller can use memref.store directly
        // Future: Add offset tracking and memref.subview for true bump allocation

        // For now, just return the arena memref unchanged
        // This works for single allocation per arena (like String.concat2)
        return ([], TRValue { SSA = resultSSA; Type = arenaType })
    }

// ═══════════════════════════════════════════════════════════
// STRUCT FIELD ACCESS PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract field from struct (e.g., string.Pointer, string.Length)
/// SSA layout (max 3): [0] = intermediate (index or dim const), [1] = intermediate2 (dim result), [2] = result
let pStructFieldGet (nodeId: NodeId) (structSSA: SSA) (fieldName: string) (structTy: MLIRType) (fieldTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 3) $"pStructFieldGet: Expected 3 SSAs, got {ssas.Length}"
        let resultSSA = List.last ssas

        // Check if structTy is a memref (strings are now memref<?xi8>)
        match structTy with
        | TMemRef _ | TMemRefScalar _ ->
            // String as memref - use memref operations
            match fieldName with
            | "Pointer" | "ptr" ->  // Accept both capitalized (old) and lowercase (CCS)
                // Extract base pointer from memref descriptor as index.
                // Returns TIndex (MLIR index) which is the canonical type for pointer values.
                // Callers at FFI boundaries (pExternCallResolved) handle index→i64 conversion.
                match fieldTy with
                | TIndex ->
                    // An index field: the base pointer as index, no cast needed
                    let! extractOp = pExtractBasePtr resultSSA structSSA structTy
                    return ([extractOp], TRValue { SSA = resultSSA; Type = TIndex })
                | _ ->
                    // Non-index target type: cast index → targetTy
                    let indexSSA = ssas.[0]  // Intermediate index from coeffects
                    let! extractOp = pExtractBasePtr indexSSA structSSA structTy
                    let! castOp = pIndexCastS resultSSA indexSSA TIndex fieldTy
                    return ([extractOp; castOp], TRValue { SSA = resultSSA; Type = fieldTy })
            | "Length" | "len" ->  // Accept both capitalized (old) and lowercase (CCS)
                // Extract length using memref.dim (returns index type)
                let dimIndexSSA = ssas.[0]  // Dim constant (0) from coeffects
                let! constOp = pConstI dimIndexSSA 0L TIndex

                // Check if we need to cast index → fieldTy (for FFI boundaries)
                match fieldTy with
                | TIndex ->
                    // No cast needed - result is index
                    let! dimOp = pMemRefDim resultSSA structSSA dimIndexSSA structTy
                    return ([constOp; dimOp], TRValue { SSA = resultSSA; Type = fieldTy })
                | _ ->
                    // Cast index → fieldTy (e.g., index → i64 for x86-64 syscall, index → i32 for ARM32)
                    let dimResultSSA = ssas.[1]  // Dim result from coeffects
                    let! dimOp = pMemRefDim dimResultSSA structSSA dimIndexSSA structTy
                    let! castOp = pIndexCastS resultSSA dimResultSSA TIndex fieldTy
                    return ([constOp; dimOp; castOp], TRValue { SSA = resultSSA; Type = fieldTy })
            | _ ->
                return failwith $"Unknown memref field name: {fieldName}"
        | _ ->
            // LLVM struct - use extractvalue (for closures, option, etc.)
            let fieldIndex =
                match fieldName with
                | "Pointer" | "ptr" -> 0  // Accept both capitalized (old) and lowercase (CCS)
                | "Length" | "len" -> 1  // Accept both capitalized (old) and lowercase (CCS)
                | _ -> failwith $"Unknown field name: {fieldName}"

            // Extract field value - pExtractField needs [offsetSSA, resultSSA]
            let extractFieldSSAs = [ssas.[0]; resultSSA]
            let! ops = pExtractField extractFieldSSAs structSSA fieldIndex structTy
            return (ops, TRValue { SSA = resultSSA; Type = fieldTy })
    }

// ═══════════════════════════════════════════════════════════
// ESCAPE-AWARE ALLOCATION
// ═══════════════════════════════════════════════════════════

/// The static memref shape of a value's storage: a struct is a byte memref of its settled size
let extractMemRefShape (arch: Architecture) (ty: MLIRType) =
    match ty with
    | TMemRefStatic (count, elemType) -> (count, elemType)
    | TStruct _ -> (mlirTypeSize arch ty, TInt (IntWidth 8))
    | _ -> failwith $"pAllocValue: expected TMemRefStatic or TStruct, got {ty}"

/// One symbol convention for a source-admitted static allocation and its reads.
let staticValueName (nodeId: NodeId) = sprintf "__clef_static_value_%d" (NodeId.value nodeId)

/// A writable declaration consumes its exact source inventory entry. It cannot
/// infer residence from a symbol name or turn individual fit into pool capacity.
let programStorageType arch graph (entry: ProgramStorageEntry) =
    match entry.Shape with
    | ProgramStorageShape.Bytes -> Some(TMemRefStatic(entry.Bytes, TInt(IntWidth 8)))
    | ProgramStorageShape.Scalar slot ->
        let scalar =
            match slot with
            | SettledSlot.Pointer 1 -> Some TIndex
            | _ -> settledScalarType slot
        scalar |> Option.map (fun scalar -> TMemRefStatic(1, scalar))
    | ProgramStorageShape.ValueView ty ->
        mapNativeTypeWithGraphForArch arch graph ty
        |> physicalStorageType arch
        |> fun value -> Some(TMemRefStatic(1, value))

let pProgramStorageDeclaration identity storageTy : PSGParser<ProgramStorageEntry> =
    parser {
        let! state = getUserState
        let reading = Clef.Compiler.PSGSaturation.SemanticGraph.ProgramStorage.read state.Graph
        do! ensure reading.IsSome "Writable program inventory is absent or stale"
        let inventory = reading.Value
        do! ensure inventory.Unresolved.IsEmpty
                ("Writable program inventory is unresolved: " + (inventory.Unresolved.Values |> String.concat "; "))
        let entry = inventory.Entries.TryFind identity
        do! ensure entry.IsSome (sprintf "Writable declaration has no source inventory entry for %A" identity)
        let entry = entry.Value
        let expected = programStorageType state.Platform.TargetArch state.Graph entry
        do! ensure (expected = Some storageTy) (sprintf "Writable declaration %A disagrees with its source-held physical shape" identity)
        return entry
    }

/// Allocate memory for a constructed value — queries escape analysis coeffect
/// PULL model: pattern pulls allocation decision from pre-computed coeffects.
/// Four-point lifetime lattice (closure-representation.md §3.3):
///   StackScoped    → memref.alloca (stack)
///   StaticLifetime → program-lifetime, belongs in static storage (memref.global)
///   EscapesVia*    → memref.alloc  (heap)
///
/// StaticLifetime is a program-lifetime DU/record: constructed once at global scope, held to
/// program end, never freed. It is placed in a module-level memref.global and referenced inline
/// via memref.get_global — the same static-storage mechanism the flat-closure path uses. Because
/// a memref.global is only valid at module scope and this runs in the PSGParser layer, the decl
/// is queued (deduped) on the shared accumulator via tryEmitGlobalMemref; the owning DU/record
/// witness drains it into WitnessOutput.TopLevelOps for module-scope placement. On a heap-free
/// target this is the only non-stack placement, so a program-lifetime DU/record no longer routes
/// through the heap allocator.
let pAllocValue (nodeId: NodeId) (ssa: SSA) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let escapeKind = Alex.Traversal.TransferTypes.escapeOf state.Graph nodeId
        match escapeKind with
        | EscapeKind.StackScoped ->
            return! pUndef ssa ty
        | EscapeKind.StaticLifetime ->
            let count, elemType = extractMemRefShape state.Platform.TargetArch ty
            let storageTy = TMemRefStatic (count, elemType)
            let globalName = staticValueName nodeId
            let! authority = pProgramStorageDeclaration (ProgramStorageIdentity.Allocation nodeId) storageTy
            MLIRAccumulator.tryEmitGlobalMemref globalName storageTy (Some authority) state.Accumulator
            return! pMemRefGetGlobal ssa globalName storageTy
        | EscapeKind.EscapesViaReturn | EscapeKind.EscapesViaClosure _ | EscapeKind.EscapesViaByRef ->
            let count, elemType = extractMemRefShape state.Platform.TargetArch ty
            return! pAllocStatic ssa count elemType None
    }

// ═══════════════════════════════════════════════════════════
// DU CONSTRUCTION
// ═══════════════════════════════════════════════════════════

/// DU case construction: tag field (index 0) + payload fields
/// CRITICAL: This is the foundation for all collection patterns (Option, List, Map, Set, Result)
/// SSA layout: [0] = undefSSA, [1] = tagSSA, [2] = tagOffsetSSA, [3] = tagResultSSA,
///             then for each payload: [4+3*i] = offsetSSA, [5+3*i] = viewSSA, [6+3*i] = zeroSSA
let pDUCaseAt (nodeId: NodeId) (destination: Val) (nativeType: NativeType) (tag: int64) (payload: Val list) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let s = Alex.Traversal.Values.value nodeId
        let ty = destination.Type

        // Insert tag at byte offset 0 via reinterpret_cast (same element type: i8→i8)
        let tagTy = TInt (IntWidth 8)  // DU tags are always i8
        let! tagConstOp = pConstI (s 1) tag tagTy
        let! insertTagOps = pTypedInsert destination.SSA (s 1) 0 (s 2) (s 3) tagTy ty

        // Insert payload fields at the settled payload offset (after the tag) via memref.view
        // (different element type: byte buffer → typed payload)
        let! state = getUserState
        let payloadByteOffset = unionPayloadOffset state.Graph nativeType
        let! payloadOpLists =
            payload
            |> List.mapi (fun i field ->
                parser {
                    let offsetSSA = s (4 + 3*i)
                    let viewSSA = s (5 + 3*i)
                    let zeroSSA = s (6 + 3*i)
                    return! pTypedInsertView destination.SSA field.SSA payloadByteOffset offsetSSA viewSSA zeroSSA field.Type ty
                })
            |> sequence

        let payloadOps = List.concat payloadOpLists
        return (tagConstOp :: (insertTagOps @ payloadOps), TRVoid)
    }

/// Construct into newly allocated storage using the same settled tag/payload
/// insertion as an explicit Baker destination. Inactive payloads are untouched.
let pDUCase (nodeId: NodeId) (tag: int64) (payload: Val list) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState
        let! result = getNodeSSA nodeId
        let! allocation = pAllocValue nodeId result ty
        let! writes, _ = pDUCaseAt nodeId { SSA = result; Type = ty } state.Current.Type tag payload
        return allocation :: writes, TRValue { SSA = result; Type = ty }
    }

// ═══════════════════════════════════════════════════════════
// SIMPLE MEMORY STORE
// ═══════════════════════════════════════════════════════════

/// MemRef copy - bulk memory copy via memcpy library function
let pMemCopy (resultSSA: SSA) (destSSA: SSA) (srcSSA: SSA) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType
        let args = [
            { SSA = destSSA; Type = platformWordTy }
            { SSA = srcSSA; Type = platformWordTy }
            { SSA = countSSA; Type = platformWordTy }
        ]
        let! memcpyCall = pFuncCall (Some resultSSA) "memcpy" args platformWordTy
        let! memcpyDecl = pFuncDecl "memcpy" [platformWordTy; platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private
        return ([memcpyDecl; memcpyCall], TRVoid)
    }

// ═══════════════════════════════════════════════════════════
// MONADIC ARGUMENT RECALL
// ═══════════════════════════════════════════════════════════

/// Recall argument from accumulator.
/// VarRefWitness already auto-loads mutable variables (TMemRef) in post-order.
/// By the time Application recalls its arguments, loading is done.
/// This combinator provides a uniform (ops, ssa, type) triple interface.
let pRecallArgWithLoad (argId: NodeId) : PSGParser<MLIROp list * SSA * MLIRType> =
    parser {
        let! (ssa, ty) = pRecallNode argId
        return ([], ssa, ty)
    }

// ═══════════════════════════════════════════════════════════
// COMPOSED INTRINSIC PARSERS (per-operation, self-contained)
// ═══════════════════════════════════════════════════════════

/// Arena.create intrinsic — stack-allocated byte buffer
let pArenaCreateIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Arena
        do! ensure (info.Operation = "create") "Not Arena.create"
        let! state = getUserState
        let! node = getCurrentNode
        let sizeNodeId = argIds.[0]
        match SemanticGraph.tryGetNode sizeNodeId state.Graph with
        | Some sizeNode ->
            match sizeNode.Kind with
            | SemanticKind.Literal (NativeLiteral.Int (value, _)) ->
                return! pArenaCreate node.Id (int value)
            | _ -> return! fail (Message $"Arena.create: size must be a literal int")
        | None -> return! fail (Message "Arena.create: size node not found")
    }

/// Arena.alloc intrinsic — allocate from arena
let pArenaAllocIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Arena
        do! ensure (info.Operation = "alloc") "Not Arena.alloc"
        do! ensure (argIds.Length >= 2) "Arena.alloc: Expected 2 args"
        let! node = getCurrentNode
        let! (_, arenaSSA, arenaType) = pRecallArgWithLoad argIds.[0]
        let! (_, sizeSSA, _) = pRecallArgWithLoad argIds.[1]
        return! pArenaAlloc node.Id arenaSSA sizeSSA arenaType
    }

// ═══════════════════════════════════════════════════════════
// ARRAY INTRINSIC PARSERS
// ═══════════════════════════════════════════════════════════

/// Index extension follows the established operand range. In particular, an
/// unsigned narrow carrier's high bit is data, while a possibly negative index
/// must keep its sign. Empty does not establish a non-negative runtime value.
let indexCastForRange (range: ValueRange) (result: SSA) (operand: SSA) (operandType: MLIRType) : MLIROp =
    if range <> ValueRange.Empty && ValueRange.isNonNegative range then
        MLIROp.IndexOp (IndexOp.IndexCastU (result, operand, operandType, TIndex))
    else
        MLIROp.IndexOp (IndexOp.IndexCastS (result, operand, operandType, TIndex))

let private pArrayIndex (nodeId: NodeId) (result: SSA) (operand: SSA) (operandType: MLIRType) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let range = nodeRange state.Graph nodeId |> Option.defaultValue ValueRange.Unbounded
        return indexCastForRange range result operand operandType
    }

/// Array.zeroCreate<'T> intrinsic — allocate zeroed array
/// int -> 'T[]  (size -> memref<?xelemType>)
///
/// Node-owned SSAs: array, length, zero index, one index, element zero, counter,
/// condition index, condition, body index, incremented index.
let pArrayZeroCreateIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Array
        do! ensure (info.Operation = "zeroCreate") "Not Array.zeroCreate"
        do! ensure (argIds.Length >= 1) "Array.zeroCreate: Expected 1 arg"
        let! node = getCurrentNode
        let! ssas = getNodeSSAs node.Id
        do! ensure (ssas.Length >= 10) $"pArrayZeroCreate: Expected 10 SSAs, got {ssas.Length}"
        let resultSSA = ssas.[0]
        let sizeIndexSSA = ssas.[1]

        // Recall the size argument (may be i64, need index for memref.alloc)
        let! (_, sizeSSA, sizeType) = pRecallArgWithLoad argIds.[0]

        // Cast size to index type (memref.alloc requires index)
        let! castOp = pArrayIndex argIds.[0] sizeIndexSSA sizeSSA sizeType

        // The element type of the array node's type: an element of the bare kind at the element
        // range's settled width, a record at its physical storage
        let! state = getUserState
        let elemType = arrayElementTypeAt state.Platform.TargetArch state.Graph node.Id node.Type

        let! allocOp = pAlloc resultSSA sizeIndexSSA elemType
        let resultType = TMemRef elemType
        let! zeroOps =
            match elemType with
            | TInt _ -> parser { let! op = pConstI ssas.[4] 0L elemType in return [op] }
            | TFloat _ -> parser { let! op = pConstF ssas.[4] 0.0 elemType in return [op] }
            | TMemRefStatic (bytes, TInt (IntWidth 8)) when
                (match state.Current.Type with NativeType.TApp (_, [elem]) -> isNullableHandle elem | _ -> false) ->
                parser {
                    // Immutable None value shared by initially empty cells. C pointer
                    // words are produced only by the foreign reference adapter.
                    let! alloc = pAllocStatic ssas.[4] bytes (TInt (IntWidth 8)) None
                    let! tag = pConstI ssas.[10] 0L (TInt (IntWidth 8))
                    let! tagOps = pTypedInsert ssas.[4] ssas.[10] 0 ssas.[11] ssas.[12] (TInt (IntWidth 8)) elemType
                    let! word = pConstI ssas.[13] 0L TIndex
                    let inner = match state.Current.Type with NativeType.TApp (_, [elem]) -> elem | _ -> failwith "Expected array type"
                    let offset = unionPayloadOffset state.Graph inner
                    let! payload = pTypedInsertView ssas.[4] ssas.[13] offset ssas.[14] ssas.[15] ssas.[16] TIndex elemType
                    return [alloc; tag; word] @ tagOps @ payload
                }
            | other -> fail (Message $"Array.zeroCreate: no valid scalar zero initialization for {other}")
        let! zeroIndex = pConstI ssas.[2] 0L TIndex
        let! oneIndex = pConstI ssas.[3] 1L TIndex
        let! counter = pAlloca ssas.[5] 1 TIndex None
        let counterType = TMemRefStatic (1, TIndex)
        let! initialize = pStore ssas.[2] ssas.[5] [ssas.[2]] TIndex counterType
        let! conditionIndex = pLoad ssas.[6] ssas.[5] [ssas.[2]]
        let! condition = pIndexCmp ssas.[7] IndexCmpPred.Slt ssas.[6] sizeIndexSSA
        let! continuation = pSCFCondition ssas.[7] []
        let! bodyIndex = pLoad ssas.[8] ssas.[5] [ssas.[2]]
        let! storeZero = pStore ssas.[4] resultSSA [ssas.[8]] elemType resultType
        let! increment = pIndexAdd ssas.[9] ssas.[8] ssas.[3]
        let! storeIndex = pStore ssas.[9] ssas.[5] [ssas.[2]] TIndex counterType
        let! yieldOp = pSCFYield []
        let! fill = pSCFWhile [conditionIndex; condition; continuation]
                             [bodyIndex; storeZero; increment; storeIndex; yieldOp]
        return ([castOp; allocOp] @ zeroOps @ [zeroIndex; oneIndex; counter; initialize; fill],
                TRValue { SSA = resultSSA; Type = resultType })
    }

/// Array.set intrinsic — store element at index
/// 'T[] -> int -> 'T -> unit  (array -> index -> value -> unit)
///
/// SSA layout (1 SSA):
///   [0] = indexCastSSA (index.casts for memref index)
let pArraySetIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Array
        do! ensure (info.Operation = "set") "Not Array.set"
        do! ensure (argIds.Length >= 3) "Array.set: Expected 3 args"
        let! node = getCurrentNode
        let! ssas = getNodeSSAs node.Id
        do! ensure (ssas.Length >= 1) $"pArraySet: Expected 1 SSA, got {ssas.Length}"
        let indexCastSSA = ssas.[0]

        let! (_, arraySSA, arrayType) = pRecallArgWithLoad argIds.[0]
        let! (_, indexSSA, indexType) = pRecallArgWithLoad argIds.[1]
        let! (_, rawValueSSA, rawValueTy) = pRecallArgWithLoad argIds.[2]
        // the value at the element's settled width (its derived meet)
        let! (meetOps, valueSSA, _) = pAdapt node.Id argIds.[2] rawValueSSA rawValueTy

        // Cast index to index type (memref.store requires index-typed indices)
        let! castOp = pArrayIndex argIds.[1] indexCastSSA indexSSA indexType

        // Element type from the array type (NOT current node type which is unit)
        let! elemType =
            match arrayType with
            | TMemRef t -> preturn t
            | TMemRefStatic (_, t) -> preturn t
            | other -> fail (Message $"Array.set: expected an array (memref), got {other}")

        let! storeOp = pStore valueSSA arraySSA [indexCastSSA] elemType arrayType
        return (meetOps @ [castOp; storeOp], TRVoid)
    }

/// An element read at its own width: a scalar element through the read's derived meet; a
/// record or tuple element keeps the logical struct type of the read's node over the byte
/// memref the element holds (the settled layout the struct carries).
let pReadElement (nodeId: NodeId) (elemType: MLIRType) (loaded: SSA) (nodeType: NativeType) : PSGParser<MLIROp list * SSA * MLIRType> =
    parser {
        let! state = getUserState
        match elemType with
        | TInt _ -> return! pAdapt nodeId nodeId loaded elemType
        | _ ->
            let logical = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph nodeType
            match logical with
            | TStruct _ -> return ([], loaded, logical)
            | _ -> return ([], loaded, elemType)
    }

/// Array.get intrinsic — load element at index
/// 'T[] -> int -> 'T  (array -> index -> element)
///
/// SSA layout (2 SSAs):
///   [0] = indexCastSSA (index.casts for memref index)
///   [1] = resultSSA (memref.load result)
let pArrayGetIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Array
        do! ensure (info.Operation = "get") "Not Array.get"
        do! ensure (argIds.Length >= 2) "Array.get: Expected 2 args"
        let! node = getCurrentNode
        let! ssas = getNodeSSAs node.Id
        do! ensure (ssas.Length >= 2) $"pArrayGet: Expected 2 SSAs, got {ssas.Length}"
        let indexCastSSA = ssas.[0]
        let resultSSA = ssas.[1]

        let! (_, arraySSA, arrayType) = pRecallArgWithLoad argIds.[0]
        let! (_, indexSSA, indexType) = pRecallArgWithLoad argIds.[1]

        // Cast index to index type (memref.load requires index-typed indices)
        let! castOp = pArrayIndex argIds.[1] indexCastSSA indexSSA indexType

        // Direct memref.load at cast index: the element's slot, then the read's own width
        // (a scalar through its derived meet; a record or tuple keeps its logical struct type
        // over the byte memref the element holds)
        let! loadOp = pLoad resultSSA arraySSA [indexCastSSA]
        let! elemType =
            match arrayType with
            | TMemRef t | TMemRefStatic (_, t) -> preturn t
            | other -> fail (Message $"Array.get: expected an array (memref), got {other}")
        let! state = getUserState
        let! (meetOps, readSSA, readTy) = pReadElement node.Id elemType resultSSA state.Current.Type

        return ([castOp; loadOp] @ meetOps, TRValue { SSA = readSSA; Type = readTy })
    }

/// Array.sub intrinsic — extract subarray (offset + length)
/// 'T[] -> int -> int -> 'T[]  (source -> startIndex -> count -> result)
///
/// Creates a contiguous copy of source[offset..offset+count].
/// SubViewCopy: subview → alloc → copy (fresh buffer for correct FFI pointer extraction).
///
/// SSA layout (3 SSAs):
///   [0] = resultSSA (fresh contiguous alloc)
///   [1] = offsetIndexSSA (index.casts for offset)
///   [2] = countIndexSSA (index.casts for count)
let pArraySubIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Array
        do! ensure (info.Operation = "sub") "Not Array.sub"
        do! ensure (argIds.Length >= 3) "Array.sub: Expected 3 args"
        let! node = getCurrentNode
        let! ssas = getNodeSSAs node.Id
        do! ensure (ssas.Length >= 3) $"pArraySub: Expected 3 SSAs, got {ssas.Length}"
        let resultSSA = ssas.[0]
        let offsetIndexSSA = ssas.[1]
        let countIndexSSA = ssas.[2]

        let! (_, sourceSSA, sourceType) = pRecallArgWithLoad argIds.[0]
        let! (_, offsetSSA, offsetType) = pRecallArgWithLoad argIds.[1]
        let! (_, countSSA, countType) = pRecallArgWithLoad argIds.[2]

        // Cast offset and count to index type (memref.subview requires index)
        let! offsetCastOp = pArrayIndex argIds.[1] offsetIndexSSA offsetSSA offsetType
        let! countCastOp = pArrayIndex argIds.[2] countIndexSSA countSSA countType

        // SubViewCopy: subview + alloc + copy → fresh contiguous buffer
        let subviewCopyOp = MLIROp.MemRefOp (MemRefOp.SubViewCopy (resultSSA, sourceSSA, [offsetIndexSSA], [SubViewParam.Dynamic countIndexSSA], [SubViewParam.Static 1L], countIndexSSA, sourceType))
        return ([offsetCastOp; countCastOp; subviewCopyOp], TRValue { SSA = resultSSA; Type = sourceType })
    }


/// Array.length intrinsic — memref.dim on dimension 0, cast to the platform int
/// 'T[] -> int
///
/// SSA layout (3 SSAs):
///   [0] = dimConstSSA (index 0)
///   [1] = lenIndexSSA (memref.dim result, index)
///   [2] = resultSSA (index.casts to int)
let pArrayLengthIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Array
        do! ensure (info.Operation = "length") "Not Array.length"
        do! ensure (argIds.Length >= 1) "Array.length: Expected 1 arg"
        let! node = getCurrentNode
        let! ssas = getNodeSSAs node.Id
        do! ensure (ssas.Length >= 3) $"pArrayLength: Expected 3 SSAs, got {ssas.Length}"
        let! (_, arraySSA, arrayType) = pRecallArgWithLoad argIds.[0]
        let! state = getUserState
        let intTy = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph state.Current.Type |> narrowForCurrent state
        let! dimConstOp = pConstI ssas.[0] 0L TIndex
        let! dimOp = pMemRefDim ssas.[1] arraySSA ssas.[0] arrayType
        let! castOp = pIndexCastS ssas.[2] ssas.[1] TIndex intTy
        return ([dimConstOp; dimOp; castOp], TRValue { SSA = ssas.[2]; Type = intTy })
    }

/// Physical storage type of an element (TypeMapping.physicalStorageType).
let private physicalElementType (arch: Architecture) (elemTy: MLIRType) : MLIRType =
    physicalStorageType arch elemTy

/// Array.blit intrinsic — byte copy between two arrays via memcpy
/// 'T[] -> int -> 'T[] -> int -> int -> unit  (source, sourceIndex, target, targetIndex, count)
///
/// SSA layout (10 SSAs):
///   [0] = srcBaseIdx (extract_aligned_pointer_as_index source)
///   [1] = dstBaseIdx (extract_aligned_pointer_as_index target)
///   [2] = srcBase (index.casts to platform word)
///   [3] = dstBase (index.casts to platform word)
///   [4] = elemSizeSSA (constant element size in bytes)
///   [5] = srcOffset (sourceIndex * elemSize)
///   [6] = dstOffset (targetIndex * elemSize)
///   [7] = byteCount (count * elemSize)
///   [8] = srcPtr (srcBase + srcOffset)
///   [9] = dstPtr (dstBase + dstOffset)
///   [10] = memcpy's required pointer result (unused by Array.blit)
let pArrayBlitIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Array
        do! ensure (info.Operation = "blit") "Not Array.blit"
        do! ensure (argIds.Length >= 5) "Array.blit: Expected 5 args"
        let! node = getCurrentNode
        let! ssas = getNodeSSAs node.Id
        do! ensure (ssas.Length >= 11) $"pArrayBlit: Expected 11 SSAs, got {ssas.Length}"
        let! (_, srcSSA, srcType) = pRecallArgWithLoad argIds.[0]
        let! (_, rawSrcIdx, rawSrcIdxTy) = pRecallArgWithLoad argIds.[1]
        let! (_, dstSSA, dstType) = pRecallArgWithLoad argIds.[2]
        let! (_, rawDstIdx, rawDstIdxTy) = pRecallArgWithLoad argIds.[3]
        let! (_, rawCount, rawCountTy) = pRecallArgWithLoad argIds.[4]
        // pointer arithmetic at the declared word: each index and the count at its derived meet
        let! (srcIdxMeet, srcIdxSSA, idxType) = pAdapt node.Id argIds.[1] rawSrcIdx rawSrcIdxTy
        let! (dstIdxMeet, dstIdxSSA, _) = pAdapt node.Id argIds.[3] rawDstIdx rawDstIdxTy
        let! (countMeet, countSSA, _) = pAdapt node.Id argIds.[4] rawCount rawCountTy
        let! state = getUserState
        let arch = state.Platform.TargetArch
        let wordTy = state.Platform.PlatformWordType
        let! elemTy =
            match srcType with
            | TMemRef t | TMemRefStatic (_, t) -> preturn t
            | other -> fail (Message $"Array.blit: expected an array (memref), got {other}")
        let elemSize = int64 (mlirTypeSize arch (physicalElementType arch elemTy))
        let! srcBaseIdxOp = pExtractBasePtr ssas.[0] srcSSA srcType
        let! dstBaseIdxOp = pExtractBasePtr ssas.[1] dstSSA dstType
        let! srcBaseOp = pIndexCastS ssas.[2] ssas.[0] TIndex wordTy
        let! dstBaseOp = pIndexCastS ssas.[3] ssas.[1] TIndex wordTy
        let! elemSizeOp = pConstI ssas.[4] elemSize idxType
        let srcOffsetOp = MLIROp.ArithOp (ArithOp.MulI (ssas.[5], srcIdxSSA, ssas.[4], idxType))
        let dstOffsetOp = MLIROp.ArithOp (ArithOp.MulI (ssas.[6], dstIdxSSA, ssas.[4], idxType))
        let byteCountOp = MLIROp.ArithOp (ArithOp.MulI (ssas.[7], countSSA, ssas.[4], idxType))
        let srcPtrOp = MLIROp.ArithOp (ArithOp.AddI (ssas.[8], ssas.[2], ssas.[5], wordTy))
        let dstPtrOp = MLIROp.ArithOp (ArithOp.AddI (ssas.[9], ssas.[3], ssas.[6], wordTy))
        let! (copyOps, _) = pMemCopy ssas.[10] ssas.[9] ssas.[8] ssas.[7]
        let ops =
            srcIdxMeet @ dstIdxMeet @ countMeet @
            [srcBaseIdxOp; dstBaseIdxOp; srcBaseOp; dstBaseOp; elemSizeOp;
             srcOffsetOp; dstOffsetOp; byteCountOp; srcPtrOp; dstPtrOp] @ copyOps
        return (ops, TRVoid)
    }

// ═══════════════════════════════════════════════════════════
// ARRAY INDEXER AND LITERAL PATTERNS (non-intrinsic node kinds)
// ═══════════════════════════════════════════════════════════

/// Indexer read `arr.[i]` on an array-typed expression (SemanticKind.IndexGet).
/// Same elision as Array.get: index cast + memref.load.
///
/// SSA layout (2 SSAs): [0] = indexCastSSA, [1] = resultSSA
let pIndexGetArray : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.IndexGet (arrId, idxId) ->
            let! (_, arraySSA, arrayType) = pRecallArgWithLoad arrId
            match arrayType with
            | TMemRef elemTy | TMemRefStatic (_, elemTy) ->
                let! (_, indexSSA, indexType) = pRecallArgWithLoad idxId
                let! ssas = getNodeSSAs node.Id
                do! ensure (ssas.Length >= 2) $"pIndexGetArray: Expected 2 SSAs, got {ssas.Length}"
                let! castOp = pIndexCastS ssas.[0] indexSSA indexType TIndex
                let! loadOp = pLoad ssas.[1] arraySSA [ssas.[0]]
                // the element's slot, then the read's own width (its derived meet)
                let! (meetOps, readSSA, readTy) = pReadElement node.Id elemTy ssas.[1] node.Type
                return ([castOp; loadOp] @ meetOps, TRValue { SSA = readSSA; Type = readTy })
            | _ -> return! fail (Message $"IndexGet: expected an array (memref), got {arrayType}")
        | _ -> return! fail (Message "Expected IndexGet")
    }

/// Indexer write `arr.[i] <- v` on an array-typed expression (SemanticKind.IndexSet).
/// Same elision as Array.set: index cast + memref.store.
///
/// SSA layout (1 SSA): [0] = indexCastSSA
let pIndexSetArray : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.IndexSet (arrId, idxId, valId) ->
            let! (_, arraySSA, arrayType) = pRecallArgWithLoad arrId
            match arrayType with
            | TMemRef elemTy | TMemRefStatic (_, elemTy) ->
                let! (_, indexSSA, indexType) = pRecallArgWithLoad idxId
                let! (_, rawValueSSA, rawValueTy) = pRecallArgWithLoad valId
                let! (meetOps, valueSSA, _) = pAdapt node.Id valId rawValueSSA rawValueTy
                let! ssas = getNodeSSAs node.Id
                do! ensure (ssas.Length >= 1) $"pIndexSetArray: Expected 1 SSA, got {ssas.Length}"
                let! castOp = pIndexCastS ssas.[0] indexSSA indexType TIndex
                let! storeOp = pStore valueSSA arraySSA [ssas.[0]] elemTy arrayType
                return (meetOps @ [castOp; storeOp], TRVoid)
            | _ -> return! fail (Message $"IndexSet: expected an array (memref), got {arrayType}")
        | _ -> return! fail (Message "Expected IndexSet")
    }

/// Array literal `[| a; b; c |]` (SemanticKind.ArrayExpr): heap allocation plus one store per element.
/// The allocation mirrors Array.zeroCreate so literals and created arrays share one representation.
///
/// SSA layout (2 + N SSAs): [0] = sizeSSA (index constant N), [1] = arraySSA (memref.alloc), [2+i] = index constant i
let pBuildArrayLiteral : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! node = getCurrentNode
        match node.Kind with
        | SemanticKind.ArrayExpr elemIds ->
            let n = List.length elemIds
            let! ssas = getNodeSSAs node.Id
            do! ensure (ssas.Length >= 2 + n) $"pBuildArrayLiteral: Expected {2 + n} SSAs, got {ssas.Length}"
            let! state = getUserState
            let arch = state.Platform.TargetArch
            let elemType = arrayElementTypeAt arch state.Graph node.Id node.Type
            let arrayType = TMemRef elemType
            let! sizeOp = pConstI ssas.[0] (int64 n) TIndex
            let! allocOp = pAlloc ssas.[1] ssas.[0] elemType
            let! storeOpLists =
                elemIds
                |> List.mapi (fun i elemId ->
                    parser {
                        let! (_, rawValueSSA, rawValueTy) = pRecallArgWithLoad elemId
                        let! (meetOps, valueSSA, _) = pAdapt node.Id elemId rawValueSSA rawValueTy
                        let! idxOp = pConstI ssas.[2 + i] (int64 i) TIndex
                        let! storeOp = pStore valueSSA ssas.[1] [ssas.[2 + i]] elemType arrayType
                        return meetOps @ [idxOp; storeOp]
                    })
                |> sequence
            return (sizeOp :: allocOp :: List.concat storeOpLists, TRValue { SSA = ssas.[1]; Type = arrayType })
        | _ -> return! fail (Message "Expected ArrayExpr")
    }
