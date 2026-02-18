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

        // Calculate byte offset for the field using FNCS-provided type structure
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

        // Calculate byte offset for the field using FNCS-provided type structure
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

        let constOneTy = TInt I64
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
        let tagTy = TInt I8  // DU tags are always i8

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

/// Build array: allocate, initialize elements, construct fat pointer
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

// ═══════════════════════════════════════════════════════════
// NATIVEPTR OPERATIONS (FNCS Intrinsics)
// ═══════════════════════════════════════════════════════════

/// Build NativePtr.stackalloc pattern
/// Allocates memory on the stack and returns a pointer
///
/// NativePtr.stackalloc<'T>(count) : nativeptr<'T>
/// SSA extracted from coeffects via nodeId: [0] = result
let pNativePtrStackAlloc (nodeId: NodeId) (count: int) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pNativePtrStackAlloc: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        let! state = getUserState

        // Extract element type from nativeptr<'T> in state.Current.Type
        match state.Current.Type with
        | NativeType.TApp(tycon, [innerTy]) when tycon.Name = "nativeptr" ->
            let elemType = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph innerTy

            // Use the provided count for memref allocation
            let! allocaOp = pAlloca resultSSA count elemType None
            let memrefTy = TMemRefStatic (count, elemType)

            // Return memref type (not TIndex) - conversion to pointer happens at FFI boundary
            return ([allocaOp], TRValue { SSA = resultSSA; Type = memrefTy })
        | NativeType.TNativePtr innerTy ->
            let elemType = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph innerTy

            // Use the provided count for memref allocation
            let! allocaOp = pAlloca resultSSA count elemType None
            let memrefTy = TMemRefStatic (count, elemType)

            // Return memref type (not TIndex) - conversion to pointer happens at FFI boundary
            return ([allocaOp], TRValue { SSA = resultSSA; Type = memrefTy })
        | _ ->
            return! fail (Message $"NativePtr.stackalloc: expected nativeptr<'T> type, got {state.Current.Type}")
    }

/// Build NativePtr.write pattern
/// Writes a value to a pointer location
///
/// NativePtr.write (ptr: nativeptr<'T>) (value: 'T) : unit
let pNativePtrWrite (valueSSA: SSA) (ptrSSA: SSA) (elemType: MLIRType) (indexSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Emit memref.store operation with index
        let! indexOp = pConstI indexSSA 0L TIndex  // Index 0 for 1-element memref
        let memrefType = TMemRefStatic (1, elemType)
        let! storeOp = pStore valueSSA ptrSSA [indexSSA] elemType memrefType

        return ([indexOp; storeOp], TRVoid)
    }

/// Build NativePtr.read pattern
/// Reads a value from a pointer location
///
/// NativePtr.read (ptr: nativeptr<'T>) : 'T
/// SSA extracted from coeffects via nodeId: [0] = indexZeroSSA, [1] = resultSSA
let pNativePtrRead (nodeId: NodeId) (ptrSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pNativePtrRead: Expected 2 SSAs, got {ssas.Length}"
        let indexZeroSSA = ssas.[0]
        let resultSSA = ssas.[1]

        // Get result type from XParsec state (type of value being loaded)
        let! state = getUserState
        let arch = state.Platform.TargetArch
        let resultType = mapNativeTypeWithGraphForArch arch state.Graph state.Current.Type

        // Emit index constant for memref.load (always load from index 0 for scalar pointer read)
        let! constOp = pConstI indexZeroSSA 0L TIndex

        // Emit memref.load operation with index
        let! loadOp = pLoad resultSSA ptrSSA [indexZeroSSA]

        return ([constOp; loadOp], TRValue { SSA = resultSSA; Type = resultType })
    }

// ═══════════════════════════════════════════════════════════
// ARENA OPERATIONS (F-02: Arena Allocation)
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
        let elemType = TInt I8
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
            | "Pointer" | "ptr" ->  // Accept both capitalized (old) and lowercase (FNCS)
                // Extract base pointer from memref descriptor as index, then cast to target type
                let! state = getUserState
                let targetTy =
                    match fieldTy with
                    | TIndex -> state.Platform.PlatformWordType  // TIndex → i64/i32 (portable!)
                    | ty -> ty

                match targetTy with
                | TIndex ->
                    // No cast needed - result is index
                    let! extractOp = pExtractBasePtr resultSSA structSSA structTy
                    return ([extractOp], TRValue { SSA = resultSSA; Type = targetTy })
                | _ ->
                    // Cast index → targetTy (e.g., index → i64 for x86-64, index → i32 for ARM32)
                    let indexSSA = ssas.[0]  // Intermediate index from coeffects
                    let! extractOp = pExtractBasePtr indexSSA structSSA structTy
                    let! castOp = pIndexCastS resultSSA indexSSA TIndex targetTy
                    return ([extractOp; castOp], TRValue { SSA = resultSSA; Type = targetTy })
            | "Length" | "len" ->  // Accept both capitalized (old) and lowercase (FNCS)
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
                | "Pointer" | "ptr" -> 0  // Accept both capitalized (old) and lowercase (FNCS)
                | "Length" | "len" -> 1  // Accept both capitalized (old) and lowercase (FNCS)
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
let pRecordStruct (fields: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length = 1 + 2 * fields.Length) $"pRecordStruct: Expected {1 + 2 * fields.Length} SSAs, got {ssas.Length}"

        // Compute struct type from field types
        let fieldTypes = fields |> List.map (fun f -> f.Type)
        let totalBytes = fieldTypes |> List.sumBy mlirTypeSize
        let structTy = TMemRefStatic(totalBytes, TInt I8)
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
let pTupleStruct (elements: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    pRecordStruct elements ssas  // Same implementation, different semantic context

// ═══════════════════════════════════════════════════════════
// ESCAPE-AWARE ALLOCATION
// ═══════════════════════════════════════════════════════════

/// Extract static memref shape from an MLIRType
let private extractMemRefShape (ty: MLIRType) =
    match ty with
    | TMemRefStatic (count, elemType) -> (count, elemType)
    | _ -> failwith $"pAllocValue: expected TMemRefStatic, got {ty}"

/// Allocate memory for a constructed value — queries escape analysis coeffect
/// PULL model: pattern pulls allocation decision from pre-computed coeffects
/// StackScoped → memref.alloca (stack), EscapesViaReturn → memref.alloc (heap)
let pAllocValue (nodeId: NodeId) (ssa: SSA) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let escapeKind = getEscapeKindOrDefault nodeId state.Coeffects.EscapeAnalysis
        match escapeKind with
        | StackScoped ->
            return! pUndef ssa ty
        | EscapesViaReturn | EscapesViaClosure _ | EscapesViaByRef ->
            let count, elemType = extractMemRefShape ty
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
        let tagTy = TInt I8  // DU tags are always i8
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

/// Simple memref.store with no indices (scalar store)
/// Store value to scalar memref (requires index even for 1-element memrefs)
/// Allocates 1 SSA for the index constant
let pMemRefStore (indexSSA: SSA) (value: SSA) (memref: SSA) (elemType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! indexOp = pConstI indexSSA 0L TIndex  // Index 0 for scalar/1-element memref
        let memrefType = TMemRefStatic (1, elemType)  // 1-element memref for scalar stores
        let! storeOp = pStore value memref [indexSSA] elemType memrefType
        return ([indexOp; storeOp], TRVoid)
    }

/// Indexed memref.store (for NativePtr.write with NativePtr.add)
/// Store value to memref at computed index
/// Handles: NativePtr.write (NativePtr.add base offset) value -> memref.store value, base[offset]
/// Note: offsetSSA must be index type (nativeint in F# source)
let pMemRefStoreIndexed (memref: SSA) (value: SSA) (offsetSSA: SSA) (elemType: MLIRType) (memrefType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Store value at computed index (offsetSSA is already index type from nativeint)
        let! storeOp = pStore value memref [offsetSSA] elemType memrefType
        return ([storeOp], TRVoid)
    }

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
        return ([memcpyCall], TRVoid)
    }

/// MemRef load operation - MLIR memref.load (NOT LLVM pointer load)
/// Baker has already transformed NativePtr.read → MemRef.load
/// This emits: %result = memref.load %memref[%index] : memref<?xT>
///
/// SSA Coeffects (1 SSA allocated by SSAAssignment):
///   [0] = result (loaded value)
///
/// Parameters:
/// - nodeId: NodeId for extracting result SSA from coeffects
/// - memrefSSA: The memref to load from
/// - indexSSA: The index to load at (already computed by witness from MemRef.add marker)
let pMemRefLoad (nodeId: NodeId) (memrefSSA: SSA) (indexSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pMemRefLoad: Expected 1 SSA, got {ssas.Length}"

        let resultSSA = ssas.[0]
        let resultType = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph state.Current.Type

        // Emit memref.load %memref[%index]
        let! loadOp = pLoad resultSSA memrefSSA [indexSSA]

        return ([loadOp], TRValue { SSA = resultSSA; Type = resultType })
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
// MEMREF.ADD FUSION COMBINATOR
// ═══════════════════════════════════════════════════════════

/// Detect MemRef.add(base, offset) fusion on an argument node.
/// If the argument was produced by MemRef.add, returns base memref + loaded offset.
/// Uses pRecallArgWithLoad for offset — monadic, no raw MemRefOp.Load.
let pDetectMemRefAddFusion (argId: NodeId) : PSGParser<SSA * SSA option * MLIRType * MLIROp list> =
    parser {
        let! (argSSA, argType) = pRecallNode argId
        let! state = getUserState
        match SemanticGraph.tryGetNode argId state.Graph with
        | Some { Kind = SemanticKind.Application (funcId, addArgIds) } ->
            match SemanticGraph.tryGetNode funcId state.Graph with
            | Some { Kind = SemanticKind.Intrinsic info }
                when info.Module = IntrinsicModule.MemRef && info.Operation = "add" ->
                let! (_, baseSSA, baseTy) = pRecallArgWithLoad addArgIds.[0]
                let! (loadOps, offsetSSA, _) = pRecallArgWithLoad addArgIds.[1]
                return (baseSSA, Some offsetSSA, baseTy, loadOps)
            | _ -> return (argSSA, None, argType, [])
        | _ -> return (argSSA, None, argType, [])
    }

// ═══════════════════════════════════════════════════════════
// COMPOSED INTRINSIC PARSERS (per-operation, self-contained)
// ═══════════════════════════════════════════════════════════

/// MemRef.alloca intrinsic — stack allocation with compile-time size
let pMemRefAllocaIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.MemRef
        do! ensure (info.Operation = "alloca") "Not MemRef.alloca"
        let! state = getUserState
        let! node = getCurrentNode
        let countNodeId = argIds.[0]
        match SemanticGraph.tryGetNode countNodeId state.Graph with
        | Some countNode ->
            match countNode.Kind with
            | SemanticKind.Literal (NativeLiteral.Int (value, _)) ->
                return! pNativePtrStackAlloc node.Id (int value)
            | _ -> return! fail (Message $"MemRef.alloca: count must be a literal (node {countNodeId})")
        | None -> return! fail (Message $"MemRef.alloca: count node not found")
    }

/// MemRef.store intrinsic — store value to memref with MemRef.add fusion
let pMemRefStoreIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.MemRef
        do! ensure (info.Operation = "store") "Not MemRef.store"
        do! ensure (argIds.Length >= 3) "MemRef.store: Expected 3 args"

        // Recall value argument
        let! (_, valueSSA, _) = pRecallArgWithLoad argIds.[0]

        // Detect MemRef.add fusion on pointer argument
        let! (memrefSSA, fusedOffsetOpt, memrefType, fusionOps) = pDetectMemRefAddFusion argIds.[1]

        // Use fused offset or recall original index argument
        let! (indexOps, offsetSSA) =
            match fusedOffsetOpt with
            | Some offset -> parser { return ([], offset) }
            | None ->
                parser {
                    let! (ops, ssa, _) = pRecallArgWithLoad argIds.[2]
                    return (ops, ssa)
                }

        match memrefType with
        | TMemRef elemType
        | TMemRefStatic (_, elemType) ->
            let! (storeOps, result) = pMemRefStoreIndexed memrefSSA valueSSA offsetSSA elemType memrefType
            return (fusionOps @ indexOps @ storeOps, result)
        | _ ->
            return! fail (Message $"MemRef.store: expected memref destination type, got {memrefType}")
    }

/// MemRef.load intrinsic — load from memref at index
let pMemRefLoadIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.MemRef
        do! ensure (info.Operation = "load") "Not MemRef.load"
        do! ensure (argIds.Length >= 2) "MemRef.load: Expected 2 args"
        let! node = getCurrentNode
        let! (_, memrefSSA, _) = pRecallArgWithLoad argIds.[0]
        let! (_, indexSSA, _) = pRecallArgWithLoad argIds.[1]
        return! pMemRefLoad node.Id memrefSSA indexSSA
    }

/// MemRef.copy intrinsic — bulk memory copy
let pMemRefCopyIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.MemRef
        do! ensure (info.Operation = "copy") "Not MemRef.copy"
        do! ensure (argIds.Length >= 3) "MemRef.copy: Expected 3 args"
        let! (_, destSSA, _) = pRecallArgWithLoad argIds.[0]
        let! (_, srcSSA, _) = pRecallArgWithLoad argIds.[1]
        let! (_, countSSA, _) = pRecallArgWithLoad argIds.[2]
        return! pMemCopy destSSA srcSSA countSSA
    }

/// MemRef.add intrinsic — marker operation, returns offset only
let pMemRefAddIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.MemRef
        do! ensure (info.Operation = "add") "Not MemRef.add"
        do! ensure (argIds.Length >= 2) "MemRef.add: Expected 2 args"
        let! (_, offsetSSA, _) = pRecallArgWithLoad argIds.[1]
        return ([], TRValue { SSA = offsetSSA; Type = TIndex })
    }

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
