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
open Alex.CodeGeneration.TypeMapping
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open PSGElaboration.EscapeAnalysis

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
// FIELD ACCESS PATTERNS (Byte-Offset)
// ═══════════════════════════════════════════════════════════

/// Field access via byte-offset memref operations
/// structType: The NativeType of the struct (for calculating field offset)
let pFieldAccess (structPtr: SSA) (structType: NativeType) (fieldIndex: int) (gepSSA: SSA) (loadSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! state = getUserState
        let arch = state.Platform.TargetArch

        // Calculate byte offset for the field using CCS-provided type structure
        let fieldOffset = calculateFieldOffsetForArch arch structType fieldIndex

        // Emit offset constant using SSA observed from coeffects via witness
        let! offsetOp = pConstI gepSSA (int64 fieldOffset) TIndex

        // Memref.load with byte offset
        // Note: This assumes structPtr is memref<Nxi8> and we load at byte offset
        let! loadOp = Alex.Elements.MemRefElements.pLoad loadSSA structPtr [gepSSA]

        return ([offsetOp; loadOp])
    }

/// Field set via byte-offset memref operations
/// structType: The NativeType of the struct (for calculating field offset)
let pFieldSet (structPtr: SSA) (structType: NativeType) (fieldIndex: int) (value: SSA) (gepSSA: SSA) (_indexSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! state = getUserState
        let arch = state.Platform.TargetArch
        let elemType = mapNativeTypeWithGraphForArch arch state.Graph state.Current.Type

        // Calculate byte offset for the field using CCS-provided type structure
        let fieldOffset = calculateFieldOffsetForArch arch structType fieldIndex

        // Emit offset constant using SSA observed from coeffects via witness
        let! offsetOp = pConstI gepSSA (int64 fieldOffset) TIndex

        // Memref.store with byte offset
        let memrefType = TMemRefStatic (1, elemType)
        let! storeOp = pStore value structPtr [gepSSA] elemType memrefType

        return ([offsetOp; storeOp])
    }

// ═══════════════════════════════════════════════════════════
// ALLOCATION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Address-of for immutable values: const 1, allocate, store, return pointer
/// SSAs: [0] = const 1, [1] = alloca result
let pAllocaImmutable (valueSSA: SSA) (valueType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 3) $"pAllocaImmutable: Expected 3 SSAs, got {ssas.Length}"

        let constOneSSA = ssas.[0]
        let allocaSSA = ssas.[1]
        let indexSSA = ssas.[2]

        let constOneTy = TInt (IntWidth 64)
        let! constOp = pConstI constOneSSA 1L constOneTy
        let! allocaOp = pAlloca allocaSSA 1 valueType None
        let! indexOp = pConstI indexSSA 0L TIndex  // Index 0 for 1-element memref
        let memrefType = TMemRefStatic (1, valueType)
        let! storeOp = pStore valueSSA allocaSSA [indexSSA] valueType memrefType

        return ([constOp; allocaOp; indexOp; storeOp])
    }

// ═══════════════════════════════════════════════════════════
// TYPE CONVERSION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Type conversion dispatcher - chooses appropriate conversion Element
let pConvertType (srcSSA: SSA) (srcType: MLIRType) (dstType: MLIRType) (resultSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        if srcType = dstType then
            // No conversion needed
            return []
        else
            let! convOp =
                match srcType, dstType with
                // Integer widening (sign-extend)
                | TInt srcWidth, TInt dstWidth when srcWidth < dstWidth ->
                    pExtSI resultSSA srcSSA srcType dstType
                // Integer narrowing (truncate)
                | TInt _, TInt _ ->
                    pTruncI resultSSA srcSSA srcType dstType
                // Float to int
                | TFloat _, TInt _ ->
                    pFPToSI resultSSA srcSSA srcType dstType
                // Int to float
                | TInt _, TFloat _ ->
                    pSIToFP resultSSA srcSSA srcType dstType
                // Unsupported conversion (bitcast removed - no portable memref equivalent)
                | _, _ ->
                    fail (Message $"Unsupported type conversion: {srcType} -> {dstType}")
            return ([convOp])
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

/// Extract DU payload via memref.view (different element type: byte buffer → typed payload)
/// SSAs extracted from coeffects via nodeId: [0] = offsetSSA, [1] = viewSSA, [2] = zeroSSA, [3] = extractSSA
let pExtractDUPayload (nodeId: NodeId) (duSSA: SSA) (duType: MLIRType) (_caseIndex: int) (payloadType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 4) $"pExtractDUPayload: Expected 4 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let viewSSA = ssas.[1]
        let zeroSSA = ssas.[2]
        let extractSSA = ssas.[3]

        // Payload byte offset = tag size (1 byte for i8 tags)
        let payloadByteOffset = 1

        // Typed extract via memref.view — payload has different element type than byte buffer
        let! extractOps = pTypedExtractView extractSSA duSSA payloadByteOffset offsetSSA viewSSA zeroSSA payloadType duType
        return (extractOps, TRValue { SSA = extractSSA; Type = payloadType })
    }

// ═══════════════════════════════════════════════════════════
// RECORD PATTERNS
// ═══════════════════════════════════════════════════════════

/// Record copy-and-update: start with original, insert updated fields
/// SSAs: one per updated field
/// Updates: (fieldIndex, valueSSA) pairs
let pRecordCopyWith (origSSA: SSA) (recordType: MLIRType) (updates: (int * SSA) list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        // Each update needs 2 SSAs: offsetSSA and targetSSA
        do! ensure (ssas.Length = 2 * updates.Length) $"pRecordCopyWith: Expected {2 * updates.Length} SSAs (2 per update), got {ssas.Length}"

        // Fold over updates, threading prevSSA through
        let! result =
            updates
            |> List.mapi (fun i (fieldIdx, valueSSA) ->
                let offsetSSA = ssas.[2*i]
                let targetSSA = ssas.[2*i + 1]
                (offsetSSA, targetSSA, fieldIdx, valueSSA))
            |> List.fold (fun accParser (offsetSSA, targetSSA, fieldIdx, valueSSA) ->
                parser {
                    let! (prevOps, prevSSA) = accParser
                    let! insertOps = pInsertValue targetSSA prevSSA valueSSA fieldIdx offsetSSA recordType
                    return (prevOps @ insertOps, targetSSA)
                }
            ) (preturn ([], origSSA))

        let (ops, _) = result
        return ops
    }

// ═══════════════════════════════════════════════════════════
// ARRAY PATTERNS
// ═══════════════════════════════════════════════════════════

/// Build array: allocate, initialize elements, construct the memref view
/// Array element access via SubView + Load
/// SSAs: gepSSA for subview, loadSSA for result, indexZeroSSA for memref index
let pArrayAccess (arrayPtr: SSA) (index: SSA) (indexTy: MLIRType) (gepSSA: SSA) (loadSSA: SSA) (indexZeroSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! subViewOp = pSubView gepSSA arrayPtr [index]
        let! indexZeroOp = pConstI indexZeroSSA 0L TIndex  // MLIR memrefs require indices
        let! loadOp = pLoad loadSSA gepSSA [indexZeroSSA]
        return ([subViewOp; indexZeroOp; loadOp])
    }

/// Array element set via SubView + Store
let pArraySet (arrayPtr: SSA) (index: SSA) (indexTy: MLIRType) (value: SSA) (gepSSA: SSA) (indexZeroSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        let! state = getUserState
        let elemType = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph state.Current.Type

        let! subViewOp = pSubView gepSSA arrayPtr [index]
        let! indexZeroOp = pConstI indexZeroSSA 0L TIndex  // Index 0 for 1-element memref
        let memrefType = TMemRefStatic (1, elemType)
        let! storeOp = pStore value gepSSA [indexZeroSSA] elemType memrefType
        return ([subViewOp; indexZeroOp; storeOp])
    }

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
// STRUCT CONSTRUCTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Record struct via Undef + InsertValue chain
/// SSA layout: [0] = undefSSA, then for each field: [2*i+1] = offsetSSA, [2*i+2] = resultSSA
let pRecordStruct (arch: Architecture) (fields: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length = 1 + 2 * fields.Length) $"pRecordStruct: Expected {1 + 2 * fields.Length} SSAs, got {ssas.Length}"

        // Compute struct type from field types
        let fieldTypes = fields |> List.map (fun f -> f.Type)
        let totalBytes = fieldTypes |> List.sumBy (mlirTypeSize arch)
        let structTy = TMemRefStatic(totalBytes, TInt (IntWidth 8))
        let! undefOp = pUndef ssas.[0] structTy

        let! insertOpLists =
            fields
            |> List.mapi (fun i field ->
                parser {
                    let offsetSSA = ssas.[2*i + 1]
                    let targetSSA = ssas.[2*i + 2]
                    let sourceSSA = if i = 0 then ssas.[0] else ssas.[2*(i-1) + 2]
                    return! pInsertValue targetSSA sourceSSA field.SSA i offsetSSA structTy
                })
            |> sequence

        let insertOps = List.concat insertOpLists
        return undefOp :: insertOps
    }

/// Tuple struct via Undef + InsertValue chain (same as record, but semantically different)
let pTupleStruct (arch: Architecture) (elements: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    pRecordStruct arch elements ssas  // Same implementation, different semantic context

// ═══════════════════════════════════════════════════════════
// ESCAPE-AWARE ALLOCATION
// ═══════════════════════════════════════════════════════════

/// Extract static memref shape from an MLIRType
let extractMemRefShape (arch: Architecture) (ty: MLIRType) =
    match ty with
    | TMemRefStatic (count, elemType) -> (count, elemType)
    | TStruct fields ->
        let totalBytes = fields |> List.sumBy (fun (_, ft) -> mlirTypeSize arch ft)
        (totalBytes, TInt (IntWidth 8))
    | _ -> failwith $"pAllocValue: expected TMemRefStatic or TStruct, got {ty}"

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
        let escapeKind = getEscapeKindOrDefault nodeId state.Coeffects.EscapeAnalysis
        match escapeKind with
        | StackScoped ->
            return! pUndef ssa ty
        | StaticLifetime ->
            let count, elemType = extractMemRefShape state.Platform.TargetArch ty
            let storageTy = TMemRefStatic (count, elemType)
            let globalName = sprintf "__clef_static_value_%d" (NodeId.value nodeId)
            MLIRAccumulator.tryEmitGlobalMemref globalName storageTy state.Accumulator
            return! pMemRefGetGlobal ssa globalName storageTy
        | EscapesViaReturn | EscapesViaClosure _ | EscapesViaByRef ->
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
let pDUCase (nodeId: NodeId) (tag: int64) (payload: Val list) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        let ssaCount = 4 + 3 * payload.Length
        do! ensure (ssas.Length >= ssaCount) $"pDUCase: Expected at least {ssaCount} SSAs, got {ssas.Length}"

        // Allocate byte-level memref (stack or heap based on escape analysis)
        let! allocOp = pAllocValue nodeId ssas.[0] ty

        // Insert tag at byte offset 0 via reinterpret_cast (same element type: i8→i8)
        let tagTy = TInt (IntWidth 8)  // DU tags are always i8
        let! tagConstOp = pConstI ssas.[1] tag tagTy
        let! insertTagOps = pTypedInsert ssas.[0] ssas.[1] 0 ssas.[2] ssas.[3] tagTy ty

        // Insert payload fields at byte offset 1 (after i8 tag) via memref.view
        // (different element type: byte buffer → typed payload)
        let payloadByteOffset = 1
        let! payloadOpLists =
            payload
            |> List.mapi (fun i field ->
                parser {
                    let offsetSSA = ssas.[4 + 3*i]
                    let viewSSA = ssas.[5 + 3*i]
                    let zeroSSA = ssas.[6 + 3*i]
                    return! pTypedInsertView ssas.[0] field.SSA payloadByteOffset offsetSSA viewSSA zeroSSA field.Type ty
                })
            |> sequence

        let payloadOps = List.concat payloadOpLists
        // Result is the allocated memref (stores are in-place)
        return (allocOp :: tagConstOp :: (insertTagOps @ payloadOps), TRValue { SSA = ssas.[0]; Type = ty })
    }

// ═══════════════════════════════════════════════════════════
// SIMPLE MEMORY STORE
// ═══════════════════════════════════════════════════════════

/// MemRef copy - bulk memory copy via memcpy library function
let pMemCopy (destSSA: SSA) (srcSSA: SSA) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType
        let args = [
            { SSA = destSSA; Type = platformWordTy }
            { SSA = srcSSA; Type = platformWordTy }
            { SSA = countSSA; Type = platformWordTy }
        ]
        let! memcpyCall = pFuncCall None "memcpy" args platformWordTy
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

/// Array.zeroCreate<'T> intrinsic — allocate zeroed array
/// int -> 'T[]  (size -> memref<?xelemType>)
///
/// SSA layout (1 SSA):
///   [0] = resultSSA (memref.alloc result)
let pArrayZeroCreateIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Array
        do! ensure (info.Operation = "zeroCreate") "Not Array.zeroCreate"
        do! ensure (argIds.Length >= 1) "Array.zeroCreate: Expected 1 arg"
        let! node = getCurrentNode
        let! ssas = getNodeSSAs node.Id
        do! ensure (ssas.Length >= 2) $"pArrayZeroCreate: Expected 2 SSAs, got {ssas.Length}"
        let resultSSA = ssas.[0]
        let sizeIndexSSA = ssas.[1]

        // Recall the size argument (may be i64, need index for memref.alloc)
        let! (_, sizeSSA, sizeType) = pRecallArgWithLoad argIds.[0]

        // Cast size to index type (memref.alloc requires index)
        let! castOp = pIndexCastS sizeIndexSSA sizeSSA sizeType TIndex

        // Element type from the result type (Array<byte> → memref<?xi8>)
        let! state = getUserState
        let elemType =
            match state.Current.Type with
            | NativeType.TApp(tycon, [innerTy]) when tycon.Name = "array" ->
                mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph innerTy
            | _ -> TInt (IntWidth 8)  // Default to byte for Array.zeroCreate<byte>

        let! allocOp = pAlloc resultSSA sizeIndexSSA elemType
        let resultType = TMemRef elemType
        return ([castOp; allocOp], TRValue { SSA = resultSSA; Type = resultType })
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
        let! (_, valueSSA, _) = pRecallArgWithLoad argIds.[2]

        // Cast index to index type (memref.store requires index-typed indices)
        let! castOp = pIndexCastS indexCastSSA indexSSA indexType TIndex

        // Element type from the array type (NOT current node type which is unit)
        let elemType =
            match arrayType with
            | TMemRef t -> t
            | TMemRefStatic (_, t) -> t
            | _ -> TInt (IntWidth 8)  // Default to byte

        // Direct memref.store (no SubView needed)
        let! storeOp = pStore valueSSA arraySSA [indexCastSSA] elemType arrayType
        return ([castOp; storeOp], TRVoid)
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

        let! (_, arraySSA, _) = pRecallArgWithLoad argIds.[0]
        let! (_, indexSSA, indexType) = pRecallArgWithLoad argIds.[1]

        // Cast index to index type (memref.load requires index-typed indices)
        let! castOp = pIndexCastS indexCastSSA indexSSA indexType TIndex

        // Result type from the current node's type
        let! state = getUserState
        let resultType = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph state.Current.Type

        // Direct memref.load at cast index
        let! loadOp = pLoad resultSSA arraySSA [indexCastSSA]

        return ([castOp; loadOp], TRValue { SSA = resultSSA; Type = resultType })
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
        let! offsetCastOp = pIndexCastS offsetIndexSSA offsetSSA offsetType TIndex
        let! countCastOp = pIndexCastS countIndexSSA countSSA countType TIndex

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
        let intTy = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph state.Current.Type
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
let pArrayBlitIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Array
        do! ensure (info.Operation = "blit") "Not Array.blit"
        do! ensure (argIds.Length >= 5) "Array.blit: Expected 5 args"
        let! node = getCurrentNode
        let! ssas = getNodeSSAs node.Id
        do! ensure (ssas.Length >= 10) $"pArrayBlit: Expected 10 SSAs, got {ssas.Length}"
        let! (_, srcSSA, srcType) = pRecallArgWithLoad argIds.[0]
        let! (_, srcIdxSSA, idxType) = pRecallArgWithLoad argIds.[1]
        let! (_, dstSSA, dstType) = pRecallArgWithLoad argIds.[2]
        let! (_, dstIdxSSA, _) = pRecallArgWithLoad argIds.[3]
        let! (_, countSSA, _) = pRecallArgWithLoad argIds.[4]
        let! state = getUserState
        let arch = state.Platform.TargetArch
        let wordTy = state.Platform.PlatformWordType
        let elemTy =
            match srcType with
            | TMemRef t | TMemRefStatic (_, t) -> t
            | _ -> TInt (IntWidth 8)
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
        let! (copyOps, _) = pMemCopy ssas.[9] ssas.[8] ssas.[7]
        let ops =
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
                let! state = getUserState
                let resultType = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph node.Type
                let physical = physicalElementType state.Platform.TargetArch resultType
                do! ensure (physical = elemTy) $"IndexGet: element type {elemTy} does not match result type {resultType} (string indexing is String.charAt)"
                let! castOp = pIndexCastS ssas.[0] indexSSA indexType TIndex
                let! loadOp = pLoad ssas.[1] arraySSA [ssas.[0]]
                return ([castOp; loadOp], TRValue { SSA = ssas.[1]; Type = resultType })
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
                let! (_, valueSSA, _) = pRecallArgWithLoad valId
                let! ssas = getNodeSSAs node.Id
                do! ensure (ssas.Length >= 1) $"pIndexSetArray: Expected 1 SSA, got {ssas.Length}"
                let! castOp = pIndexCastS ssas.[0] indexSSA indexType TIndex
                let! storeOp = pStore valueSSA arraySSA [ssas.[0]] elemTy arrayType
                return ([castOp; storeOp], TRVoid)
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
            let! elemType =
                match node.Type with
                | NativeType.TApp (tycon, [innerTy]) when tycon.Name = "array" || tycon.Name = "Array" ->
                    preturn (physicalElementType arch (mapNativeTypeWithGraphForArch arch state.Graph innerTy))
                | other -> fail (Message $"ArrayExpr: expected an array type, got {other}")
            let arrayType = TMemRef elemType
            let! sizeOp = pConstI ssas.[0] (int64 n) TIndex
            let! allocOp = pAlloc ssas.[1] ssas.[0] elemType
            let! storeOpLists =
                elemIds
                |> List.mapi (fun i elemId ->
                    parser {
                        let! (_, valueSSA, _) = pRecallArgWithLoad elemId
                        let! idxOp = pConstI ssas.[2 + i] (int64 i) TIndex
                        let! storeOp = pStore valueSSA ssas.[1] [ssas.[2 + i]] elemType arrayType
                        return [idxOp; storeOp]
                    })
                |> sequence
            return (sizeOp :: allocOp :: List.concat storeOpLists, TRValue { SSA = ssas.[1]; Type = arrayType })
        | _ -> return! fail (Message "Expected ArrayExpr")
    }

