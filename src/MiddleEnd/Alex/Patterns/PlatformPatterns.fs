/// PlatformPatterns - Platform syscall operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns for platform I/O operations.
/// Patterns compose Elements into platform-specific syscall sequences.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): All patterns use NodeId-based API.
/// Patterns extract SSAs monadically via getNodeSSAs - witnesses pass NodeIds, not SSAs.
module Alex.Patterns.PlatformPatterns

open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open XParsec
open XParsec.Parsers     // preturn
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemoryPatterns // pRecallArgWithLoad
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.FuncElements
open Alex.Elements.ArithElements
open Alex.Elements.MemRefElements
open Alex.Elements.IndexElements
open Alex.Elements.SCFElements
open Alex.Elements.MLIRAtomics
open Alex.Patterns.LiteralPatterns  // deriveGlobalRef, deriveByteLength (for dynamic extern string constants)
open Alex.Traversal.TransferTypes

// ═══════════════════════════════════════════════════════════
// FFI BOUNDARY MARSHALING
// ═══════════════════════════════════════════════════════════

/// Marshal an internal MLIR type to its C ABI representation.
/// At the FFI boundary, TIndex (Clef's nativeint/pointer) must become
/// PlatformWordType (i64 on x86_64). MLIR's index type is "in here" —
/// the C world sees platform-word-sized integers for pointer values.
/// This prevents symbol collisions with MLIR's own runtime symbols
/// (e.g., finalize-memref-to-llvm's @malloc(i64) -> !llvm.ptr).
let private marshalToCType (platformWordTy: MLIRType) (ty: MLIRType) : MLIRType =
    match ty with
    | TIndex -> platformWordTy
    | other -> other

let private pForeignReturnType (funcId: NodeId) (ty: NativeType) : PSGParser<MLIRType> =
    parser {
        let! state = getUserState
        let! mapped = pMapType ty
        match mapped with
        | _ when Clef.Compiler.NativeTypedTree.NativeTypes.Types.tryGetNTUKind ty = Some NTUKind.NTUunit -> return TVoid
        | TInt (IntWidth 0) ->
            match Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.returnOfCall state.Graph funcId with
            | Some declared -> return TInt (IntWidth declared.Bits)
            | None -> return narrowForCurrent state mapped
        | _ -> return mapped
    }

/// Unwrap an option-typed argument at the FFI boundary.
/// Options retain their settled tagged DU storage; payload offsets come from CCS.
///   None (tag=0) → NULL (0 as PlatformWordType)
///   Some (tag≠0) → payload extracted as PlatformWordType
///
/// Composes from MLIRAtomics (pTypedExtract, pTypedExtractView) and
/// ArithElements (pCmpI, pSelect, pConstI) — proper layer composition.
///
/// SSA layout (13 slots from coeffects; opaque pointers use the first 11):
///   0-2: tag extraction (tagSSA, tagViewSSA, tagZeroSSA)
///   3-6: payload extraction (payloadSSA, payOffsetSSA, payViewSSA, payZeroSSA)
///   7:   zero i8 constant for tag comparison
///   8:   isSome comparison result (i1)
///   9:   null constant (0 as PlatformWordType)
///   10:  select result
///   11-12: string payload descriptor to native pointer (when applicable)
let pUnwrapOptionArgForFFI
    (nativeOptionType: NativeType) (optionSSA: SSA) (optionType: MLIRType) (platformWordTy: MLIRType)
    (ssas: SSA list) (baseIdx: int)
    : PSGParser<MLIROp list * Val> =
    parser {
        let tagSSA       = ssas.[baseIdx]
        let tagViewSSA   = ssas.[baseIdx + 1]
        let tagZeroSSA   = ssas.[baseIdx + 2]
        let payloadSSA   = ssas.[baseIdx + 3]
        let payOffsetSSA = ssas.[baseIdx + 4]
        let payViewSSA   = ssas.[baseIdx + 5]
        let payZeroSSA   = ssas.[baseIdx + 6]
        let zeroI8SSA    = ssas.[baseIdx + 7]
        let isSomeSSA    = ssas.[baseIdx + 8]
        let nullSSA      = ssas.[baseIdx + 9]
        let selectSSA    = ssas.[baseIdx + 10]

        let tagTy = TInt (IntWidth 8)

        // 1. Extract tag (i8) from byte offset 0 — pTypedExtract (MLIRAtomics)
        let! tagOps = pTypedExtract tagSSA optionSSA 0 tagViewSSA tagZeroSSA tagTy optionType

        let! state = getUserState
        let! inner =
            match nativeOptionType with
            | NativeType.TApp (tc, [inner]) when tc.Name = "option" || tc.Name = "voption" -> preturn inner
            | _ -> fail (Message "Nullable foreign argument requires a settled option type")
        let payloadOffset = unionPayloadOffset state.Graph nativeOptionType
        let! payloadOps, nativePayload =
            match Clef.Compiler.NativeTypedTree.NativeTypes.Types.tryGetNTUKind inner with
            | Some NTUKind.NTUptr ->
                parser {
                    let! ops = pTypedExtractView payloadSSA optionSSA payloadOffset payOffsetSSA payViewSSA payZeroSSA platformWordTy optionType
                    return ops, payloadSSA
                }
            | Some NTUKind.NTUstring ->
                parser {
                    let! innerType = pMapType inner
                    let! ops = pTypedExtractView payloadSSA optionSSA payloadOffset payOffsetSSA payViewSSA payZeroSSA innerType optionType
                    let pointer, word = ssas.[baseIdx + 11], ssas.[baseIdx + 12]
                    let! extract = pExtractBasePtr pointer payloadSSA innerType
                    let! cast = pIndexCastS word pointer TIndex platformWordTy
                    return ops @ [extract; cast], word
                }
            | _ -> fail (Message "Nullable foreign arguments require an opaque handle or a string adapter")

        // 3. Compare tag ≠ 0 → isSome — pCmpI (ArithElements)
        let! zeroI8Op = pConstI zeroI8SSA 0L tagTy
        let! cmpOp = pCmpI isSomeSSA ICmpPred.Ne tagSSA zeroI8SSA tagTy

        // 4. Null constant for None case
        let! nullOp = pConstI nullSSA 0L platformWordTy

        // 5. Select: isSome ? payload : null — pSelect (ArithElements)
        let! selectOp = pSelect selectSSA isSomeSSA nativePayload nullSSA platformWordTy

        return (tagOps @ payloadOps @ [zeroI8Op; cmpOp; nullOp; selectOp],
                { SSA = selectSSA; Type = platformWordTy })
    }

// ═══════════════════════════════════════════════════════════
// RESOLVED BINDING LOOKUP
// ═══════════════════════════════════════════════════════════

/// Resolve the target function name for a platform call from pre-computed coeffects.
/// For LibcCall/ExternCall, returns the target function name.
/// For Syscall/InlineAsm, falls back to the provided default (inline asm is future work).
let private resolveCallTarget (nodeId: NodeId) (defaultName: string) (platform: PlatformReads) : string =
    match Map.tryFind nodeId platform.Bindings.Bindings with
    | Some binding ->
        match binding.Resolved with
        | ResolvedBinding.LibcCall funcName -> funcName
        | ResolvedBinding.ExternCall (_, symbol) -> symbol
        | ResolvedBinding.Syscall _ -> defaultName   // the freestanding leg's inline asm is owed
    | None -> defaultName

// ═══════════════════════════════════════════════════════════
// PLATFORM I/O SYSCALLS
// ═══════════════════════════════════════════════════════════

/// Build Sys.write syscall pattern (portable)
/// Uses func.call (portable) for direct syscall
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes written)
/// - fdSSA: File descriptor SSA (typically constant 1 for stdout)
/// - bufferSSA: Buffer SSA (memref or ptr depending on source)
/// - bufferType: Actual MLIR type of buffer (TMemRefScalar, TMemRef, or TIndex)
/// - countSSA: Number of bytes to write SSA
/// Build Sys.write syscall pattern with FFI pointer extraction
/// Uses MLIR standard dialects (memref.extract_aligned_pointer_as_index + index.casts)
/// to extract pointers from memrefs at FFI boundaries.
///
/// Buffers are ALWAYS memrefs at syscall boundaries - we ALWAYS extract pointers.
/// Length is extracted via memref.dim (NO explicit count parameter).
///
/// SSA layout (6 SSAs):
///   [0] = buf_ptr_index (memref.extract_aligned_pointer_as_index)
///   [1] = buf_ptr_i64 (index.casts)
///   [2] = dim_index_const (constant 0 for dimension)
///   [3] = count_index (memref.dim result, index type)
///   [4] = count_i64 (count cast to platform word)
///   [5] = result (func.call return value)
///
/// Parameters:
/// - nodeId: NodeId for extracting SSAs from coeffects (6 SSAs allocated)
/// - fdSSA: File descriptor SSA (typically constant 1 for stdout)
/// - bufferSSA: Buffer SSA (ALWAYS memref)
/// - bufferType: MLIR type of buffer (ALWAYS TMemRef or TMemRefStatic)
let pSysWrite (nodeId: NodeId) (fdSSA: SSA) (bufferSSA: SSA) (bufferType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Buffer is ALWAYS a memref at syscall boundary
        // We ALWAYS need FFI pointer extraction (no conditionals)
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 6) $"pSysWrite: Expected 6 SSAs, got {ssas.Length}"

        let buf_ptr_index = ssas.[0]
        let buf_ptr_i64 = ssas.[1]
        let dim_index_const = ssas.[2]
        let count_index = ssas.[3]
        let count_i64 = ssas.[4]
        let resultSSA = ssas.[5]

        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Extract pointer from memref: memref.extract_aligned_pointer_as_index
        let! extractOp = pExtractBasePtr buf_ptr_index bufferSSA bufferType

        // Cast index to i64: index.casts
        let! castOp = pIndexCastS buf_ptr_i64 buf_ptr_index TIndex platformWordTy

        // Extract length via memref.dim (dimension 0)
        let! dimConstOp = pConstI dim_index_const 0L TIndex
        let! dimOp = pMemRefDim count_index bufferSSA dim_index_const bufferType

        // Cast count to platform word for syscall
        let! countCastOp = pIndexCastS count_i64 count_index TIndex platformWordTy

        // Resolve target function name from pre-computed binding coeffects
        let callTarget = resolveCallTarget nodeId "write" state.Platform

        // Call with extracted i64 pointer and length
        let vals = [
            { SSA = fdSSA; Type = platformWordTy }
            { SSA = buf_ptr_i64; Type = platformWordTy }  // ALWAYS i64 after extraction
            { SSA = count_i64; Type = platformWordTy }    // Length from memref.dim
        ]
        let! writeCall = pFuncCall (Some resultSSA) callTarget vals platformWordTy

        // External function — emit declaration alongside call
        let! writeDecl = pFuncDecl callTarget [platformWordTy; platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private

        return ([writeDecl; extractOp; castOp; dimConstOp; dimOp; countCastOp; writeCall], TRValue { SSA = resultSSA; Type = platformWordTy })
    }

/// Build Sys.read syscall pattern with FFI pointer extraction
/// Uses MLIR standard dialects (memref.extract_aligned_pointer_as_index + index.casts)
/// to extract pointers from memrefs at FFI boundaries.
///
/// Buffers are ALWAYS memrefs at syscall boundaries - we ALWAYS extract pointers.
/// Buffer capacity (maxCount) is extracted via memref.dim (NO explicit count parameter).
///
/// SSA layout (6 SSAs):
///   [0] = buf_ptr_index (memref.extract_aligned_pointer_as_index)
///   [1] = buf_ptr_i64 (index.casts)
///   [2] = dim_index_const (constant 0 for dimension)
///   [3] = capacity_index (memref.dim result, index type)
///   [4] = capacity_i64 (capacity cast to platform word)
///   [5] = result (func.call return value)
///
/// Parameters:
/// - nodeId: NodeId for extracting SSAs from coeffects (6 SSAs allocated)
/// - fdSSA: File descriptor SSA (typically constant 0 for stdin)
/// - bufferSSA: Buffer SSA (ALWAYS memref)
/// - bufferType: MLIR type of buffer (ALWAYS TMemRef or TMemRefStatic)
let pSysRead (nodeId: NodeId) (fdSSA: SSA) (bufferSSA: SSA) (bufferType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Buffer is ALWAYS a memref at syscall boundary
        // We ALWAYS need FFI pointer extraction (no conditionals)
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 6) $"pSysRead: Expected 6 SSAs, got {ssas.Length}"

        let buf_ptr_index = ssas.[0]
        let buf_ptr_i64 = ssas.[1]
        let dim_index_const = ssas.[2]
        let capacity_index = ssas.[3]
        let capacity_i64 = ssas.[4]
        let resultSSA = ssas.[5]

        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Extract pointer from memref: memref.extract_aligned_pointer_as_index
        let! extractOp = pExtractBasePtr buf_ptr_index bufferSSA bufferType

        // Cast index to i64: index.casts
        let! castOp = pIndexCastS buf_ptr_i64 buf_ptr_index TIndex platformWordTy

        // Extract buffer capacity via memref.dim (dimension 0)
        let! dimConstOp = pConstI dim_index_const 0L TIndex
        let! dimOp = pMemRefDim capacity_index bufferSSA dim_index_const bufferType

        // Cast capacity to platform word for syscall
        let! capacityCastOp = pIndexCastS capacity_i64 capacity_index TIndex platformWordTy

        // Resolve target function name from pre-computed binding coeffects
        let callTarget = resolveCallTarget nodeId "read" state.Platform

        // Call with extracted i64 pointer and capacity
        let vals = [
            { SSA = fdSSA; Type = platformWordTy }
            { SSA = buf_ptr_i64; Type = platformWordTy }  // ALWAYS i64 after extraction
            { SSA = capacity_i64; Type = platformWordTy } // Capacity from memref.dim
        ]
        let! readCall = pFuncCall (Some resultSSA) callTarget vals platformWordTy

        // External function — emit declaration alongside call
        let! readDecl = pFuncDecl callTarget [platformWordTy; platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private

        return ([readDecl; extractOp; castOp; dimConstOp; dimOp; capacityCastOp; readCall], TRValue { SSA = resultSSA; Type = platformWordTy })
    }

/// Build Sys.readline pattern — read line from fd, return the framed line.
/// The buffer capacity and the delimiter trim are READ from the site's
/// Buffer.* annotation, which the CCS obligation pass projected from the
/// platform's declared `consoleReadln` schema (BAREWire docs/11: three
/// observers, one truth). This witness authors no number. A site without the
/// annotation is a compiler fault, not a default.
///
/// SSA layout (14 SSAs):
///   [0]  = bufferSSA (memref.alloc <capacity>xi8)
///   [1]  = buf_ptr_index (extract base pointer)
///   [2]  = buf_ptr_word (index.casts to platform word)
///   [3]  = capacity_const (declared capacity as platform word)
///   [4]  = bytesReadSSA (func.call read result)
///   [5]  = bytesReadIndex (index.casts from platform word to index)
///   [6]  = oneConst (constant 1 as index)
///   [7]  = trimmedLen (bytes_read - 1, trims newline)
///   [8]  = resultSSA (memref.subview of buffer[0..trimmedLen])
///   [9]  = sizeConst (declared capacity as index for alloc)
///   [10] = zeroConst (constant 0 as index for subview offset)
///   [11] = oneStrideConst (constant 1 as index for subview stride)
///   [12] = readDeclSlot (read function decl)
///   [13] = (reserved)
let pSysReadline (node: SemanticNode) (fdSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let nodeId = node.Id
        // The declared buffer, projected onto this site by CCS Pass 5.
        let capacity =
            match Map.tryFind BufferMetadata.Capacity node.Metadata with
            | Some (MetadataValue.Int64 c) -> Some c
            | _ -> None
        let trimDelimiter =
            match Map.tryFind BufferMetadata.TrimDelimiter node.Metadata with
            | Some (MetadataValue.Bool b) -> b
            | _ -> false
        do! ensure capacity.IsSome
                $"pSysReadline: site {NodeId.value nodeId} carries no Buffer.Capacity annotation; the platform's consoleReadln declaration was not cross-applied"
        let capacity = capacity.Value
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 12) $"pSysReadline: Expected 12 SSAs, got {ssas.Length}"

        let bufferSSA     = ssas.[0]
        let buf_ptr_index = ssas.[1]
        let buf_ptr_word  = ssas.[2]
        let capacity_const = ssas.[3]
        let bytesReadSSA  = ssas.[4]
        let bytesReadIdx  = ssas.[5]
        let oneConst      = ssas.[6]
        let trimmedLen    = ssas.[7]
        let resultSSA     = ssas.[8]
        let sizeConst     = ssas.[9]
        let zeroConst     = ssas.[10]
        let oneStrideConst = ssas.[11]

        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType
        let bufferType = TMemRef (TInt (IntWidth 8))

        // Resolve target function name from pre-computed binding coeffects
        let callTarget = resolveCallTarget nodeId "read" state.Platform

        // 1. Allocate the declared buffer
        let! sizeOp = pConstI sizeConst capacity TIndex
        let! allocOp = pAlloc bufferSSA sizeConst (TInt (IntWidth 8))

        // 2. Extract pointer for read syscall
        let! extractOp = pExtractBasePtr buf_ptr_index bufferSSA bufferType
        let! castPtrOp = pIndexCastS buf_ptr_word buf_ptr_index TIndex platformWordTy

        // 3. Declared capacity as platform word
        let! capacityOp = pConstI capacity_const capacity platformWordTy

        // 4. Call read(fd, buffer, capacity)
        let readArgs = [
            { SSA = fdSSA; Type = platformWordTy }
            { SSA = buf_ptr_word; Type = platformWordTy }
            { SSA = capacity_const; Type = platformWordTy }
        ]
        let! readCall = pFuncCall (Some bytesReadSSA) callTarget readArgs platformWordTy
        let! readDecl = pFuncDecl callTarget [platformWordTy; platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private

        // 5. Convert bytes read to index; trim the framing delimiter iff the
        //    declared schema says it is trimmed (Framing.Delimited, TrimDelimiter)
        let! castBytesOp = pIndexCastS bytesReadIdx bytesReadSSA platformWordTy TIndex
        let! oneOp = pConstI oneConst (if trimDelimiter then 1L else 0L) TIndex
        let trimOp = MLIROp.ArithOp (ArithOp.SubI (trimmedLen, bytesReadIdx, oneConst, TIndex))

        // 6. SubViewCopy: subview + alloc + copy → fresh contiguous buffer for FFI
        let! zeroOp = pConstI zeroConst 0L TIndex
        let subviewCopyOp = MLIROp.MemRefOp (MemRefOp.SubViewCopy (resultSSA, bufferSSA, [zeroConst], [SubViewParam.Dynamic trimmedLen], [SubViewParam.Static 1L], trimmedLen, bufferType))

        let ops = [readDecl; sizeOp; allocOp; extractOp; castPtrOp; capacityOp; readCall;
                   castBytesOp; oneOp; trimOp; zeroOp; subviewCopyOp]
        return (ops, TRValue { SSA = resultSSA; Type = bufferType })
    }

// ═══════════════════════════════════════════════════════════
// EXTERN CALL PATTERN (FidelityExtern bindings)
// ═══════════════════════════════════════════════════════════

/// Monadically recall a list of argument nodes from the accumulator, each brought to the width
/// of the C parameter it meets (the binding's parameter node: its descriptor's carrier, or the
/// declared Register width for the bare kind) by the meet SSAAssignment derived for (call, arg).
/// Returns the meet ops and the (SSA * MLIRType) pairs in argument order.
let private recallArgs (callId: NodeId) (argIds: NodeId list) : PSGParser<MLIROp list * (SSA * MLIRType) list> =
    let rec loop (ids: NodeId list) : PSGParser<MLIROp list * (SSA * MLIRType) list> =
        parser {
            match ids with
            | [] -> return ([], [])
            | id :: rest ->
                let! (ssa, ty) = pRecallNode id
                let! (meetOps, ssa', ty') = pAdapt callId id ssa ty
                let! (restOps, restPairs) = loop rest
                return (meetOps @ restOps, (ssa', ty') :: restPairs)
        }
    loop argIds

/// Pack a bounded scalar array according to its declared foreign element ABI.
/// Source storage remains unchanged until writable results are copied back;
/// every narrowing input is checked, and the temporary ends with this call.
let private projectScalarReference (graph: SemanticGraph) argId value ty
    (declared: Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.DeclaredParameter)
    readOnly (ssas: SSA list) cursor : PSGParser<MLIROp list * MLIROp list * (SSA * MLIRType)> =
    parser {
        do! ensure (cursor + 48 <= ssas.Length) "Foreign scalar projection exceeds the node's assigned SSA family"
        let s i = ssas.[cursor + i]
        let! sourceBits =
            match ty with
            | TMemRef (TInt (IntWidth bits)) | TMemRefStatic (_, TInt (IntWidth bits)) -> preturn bits
            | _ -> fail (Message "Foreign scalar projection requires an integer array")
        do! ensure (declared.Bits > 0 && declared.Bits < sourceBits && sourceBits <= 64)
                "Foreign scalar projection requires a narrower native integer representation"
        let sourceTy = TInt (IntWidth sourceBits)
        let nativeTy = TInt (IntWidth declared.Bits)
        let bufferTy = TMemRef nativeTy
        let unsignedNative = ValueRange.isNonNegative declared.Range
        let sourceRange =
            match graph.Nodes.[argId].Type with
            | NativeType.TApp (_, [elem]) ->
                Map.tryFind (formatType (Clef.Compiler.NativeTypedTree.UnionFind.applySubst elem)) graph.ElementRanges.Value
                |> Option.defaultValue ValueRange.Unbounded
            | _ -> ValueRange.Unbounded
        let unsignedSource = ValueRange.isNonNegative sourceRange
        let! bounds =
            match ValueRange.endpoints declared.Range with
            | Some (ValueRange.Endpoint.Finite lo, ValueRange.Endpoint.Finite hi) -> preturn (int64 lo, int64 hi)
            | _ -> fail (Message "Foreign scalar projection requires finite declared element bounds")
        let! zero = pConstI (s 0) 0L TIndex
        let! one = pConstI (s 1) 1L TIndex
        let length = MLIROp.MemRefOp (MemRefOp.Dim (s 2, value, s 0, ty))
        let! nonempty = pCmpI (s 3) ICmpPred.Uge (s 2) (s 1) TIndex
        let! allocate = pAlloc (s 4) (s 2) nativeTy
        let! lower = pConstI (s 5) (fst bounds) sourceTy
        let! upper = pConstI (s 6) (snd bounds) sourceTy
        let! counter = pAlloca (s 7) 1 TIndex None
        let counterTy = TMemRefStatic (1, TIndex)
        let! initialize = pStore (s 0) (s 7) [s 0] TIndex counterTy
        let! condIndex = pLoad (s 8) (s 7) [s 0]
        let! condition = pIndexCmp (s 9) IndexCmpPred.Ult (s 8) (s 2)
        let! continuation = pSCFCondition (s 9) []
        let! index = pLoad (s 10) (s 7) [s 0]
        let! input = pLoad (s 11) value [s 10]
        // Unsigned comparison rejects a negative input to an unsigned target.
        // A nonnegative source also cannot reinterpret its high bit as a sign.
        let! high = pCmpI (s 12) (if unsignedNative || unsignedSource then ICmpPred.Ule else ICmpPred.Sle) (s 11) (s 6) sourceTy
        let! low = pCmpI (s 13) ICmpPred.Sge (s 11) (s 5) sourceTy
        let! inBounds = pAndI (s 14) (s 12) (s 13) (TInt (IntWidth 1))
        let valid = if unsignedNative || unsignedSource then s 12 else s 14
        let! narrow = pTruncI (s 15) (s 11) sourceTy nativeTy
        let! store = pStore (s 15) (s 4) [s 10] nativeTy bufferTy
        let! increment = pIndexAdd (s 16) (s 10) (s 1)
        let! next = pStore (s 16) (s 7) [s 0] TIndex counterTy
        let! yieldOp = pSCFYield []
        let! pack = pSCFWhile [condIndex; condition; continuation]
                              [index; input; high; low; inBounds; MLIROp.Assert(valid, $"Foreign reference {declared.Name} element is outside its declared range"); narrow; store; increment; next; yieldOp]
        let! copyback =
            if readOnly then preturn []
            else parser {
                let! reset = pStore (s 0) (s 7) [s 0] TIndex counterTy
                let! condIndex = pLoad (s 17) (s 7) [s 0]
                let! condition = pIndexCmp (s 18) IndexCmpPred.Ult (s 17) (s 2)
                let! continuation = pSCFCondition (s 18) []
                let! index = pLoad (s 19) (s 7) [s 0]
                let! output = pLoad (s 20) (s 4) [s 19]
                let! widen = if unsignedNative then pExtUI (s 21) (s 20) nativeTy sourceTy else pExtSI (s 21) (s 20) nativeTy sourceTy
                let! store = pStore (s 21) value [s 19] sourceTy ty
                let! increment = pIndexAdd (s 22) (s 19) (s 1)
                let! next = pStore (s 22) (s 7) [s 0] TIndex counterTy
                let! copy = pSCFWhile [condIndex; condition; continuation] [index; output; widen; store; increment; next; yieldOp]
                return [reset; copy]
            }
        let! release = pDealloc (s 4) bufferTy
        return [zero; one; length; nonempty; MLIROp.Assert(s 3, $"Foreign reference {declared.Name} requires at least one element"); allocate; lower; upper; counter; initialize; pack],
               copyback @ [release], (s 4, bufferTy)
    }

/// Project rich Clef values into call-scoped C storage. Source options retain
/// their tags and payloads; only this adapter encodes a C null pointer. Writable
/// pointer output cells are copied back as source options after the native call.
let private projectForeignArguments (graph: SemanticGraph) funcId argIds argPairs (ssas: SSA list) : PSGParser<MLIROp list * MLIROp list * (SSA * MLIRType) list> =
    let scalarReferenceArguments = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.referenceArguments graph funcId argIds
    let scalarReferences = scalarReferenceArguments |> Map.ofList
    let readOnlyScalars = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.readOnlyScalarReferenceArguments graph funcId argIds
    let pointerReferenceArguments = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.pointerReferenceArguments graph funcId argIds
    let pointerReferences = pointerReferenceArguments |> Map.ofList
    let readOnlyRecords = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.readOnlyRecordReferenceArguments graph funcId argIds
    let recordReferences = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.recordReferenceArguments graph funcId argIds
    let layouts = (Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.readDescriptors graph).Layouts
    let rec loop cursor items =
        parser {
            match items with
            | [] -> return [], [], []
            | (argId, (value, ty)) :: rest ->
                match Map.tryFind argId scalarReferences, Map.tryFind argId pointerReferences with
                | Some declared, _ when ty <> TMemRef (TInt (IntWidth declared.Bits)) && (match ty with TMemRefStatic (_, element) -> element <> TInt (IntWidth declared.Bits) | _ -> true) ->
                    // Two scalar references can alias even through different source names.
                    // Separate temporaries would change the native call's alias semantics.
                    do! ensure (scalarReferenceArguments.Length <= 1) "Multiple scalar references require an alias-preserving native projection"
                    let! before, after, pair = projectScalarReference graph argId value ty declared (Set.contains argId readOnlyScalars) ssas cursor
                    let! beforeRest, afterRest, pairs = loop (cursor + 48) rest
                    return before @ beforeRest, after @ afterRest, pair :: pairs
                | _, Some declared ->
                    do! ensure (cursor + 32 <= ssas.Length) "Foreign pointer projection exceeds the node's assigned SSA family"
                    let s i = ssas.[cursor + i]
                    let! state = getUserState
                    let pointerBytes = PlatformContext.pointerSize state.Graph.Platform.Value |> Result.toOption
                    do! ensure (pointerBytes = Some (declared.Bits / 8)) "Foreign pointer reference width disagrees with target Pointer dimension"
                    let! optionTy = match ty with TMemRef elem | TMemRefStatic (_, elem) -> preturn elem | _ -> fail (Message "Pointer output reference requires an array")
                    let! optionBytes = match optionTy with TMemRefStatic (n, TInt (IntWidth 8)) -> preturn n | _ -> fail (Message "Pointer output reference requires source option storage")
                    let! zero = pConstI (s 0) 0L TIndex
                    let! one = pConstI (s 1) 1L TIndex
                    let dim = MLIROp.MemRefOp (MemRefOp.Dim(s 2, value, s 0, ty))
                    let! valid = pCmpI (s 3) ICmpPred.Uge (s 2) (s 1) TIndex
                    let! loaded = pLoad (s 4) value [s 0]
                    let inner = match graph.Nodes.[argId].Type with NativeType.TApp (_, [inner]) -> inner | _ -> failwith "Expected option array"
                    let! unpack, raw = pUnwrapOptionArgForFFI inner (s 4) optionTy TIndex ssas (cursor + 5)
                    let! allocate = pAlloca (s 16) 1 TIndex None
                    let nativeTy = TMemRefStatic (1, TIndex)
                    let! initialize = pStore raw.SSA (s 16) [s 0] TIndex nativeTy
                    let! returned = pLoad (s 17) (s 16) [s 0]
                    let! changed = pCmpI (s 18) ICmpPred.Ne (s 17) raw.SSA TIndex
                    let! present = pCmpI (s 19) ICmpPred.Ne (s 17) (s 0) TIndex
                    let! tag = pExtUI (s 20) (s 19) (TInt (IntWidth 1)) (TInt (IntWidth 8))
                    let! option = pAllocStatic (s 21) optionBytes (TInt (IntWidth 8)) None
                    let! tagOps = pTypedInsert (s 21) (s 20) 0 (s 22) (s 23) (TInt (IntWidth 8)) optionTy
                    let! payloadOps = pTypedInsertView (s 21) (s 17) (unionPayloadOffset graph inner) (s 24) (s 25) (s 26) TIndex optionTy
                    let! store = pStore (s 21) value [s 0] optionTy ty
                    let! yieldOp = pSCFYield []
                    let! copyback = pSCFIf (s 18) ([present; tag; option] @ tagOps @ payloadOps @ [store; yieldOp]) None None
                    let! beforeRest, afterRest, pairs = loop (cursor + 32) rest
                    return [zero; one; dim; valid; MLIROp.Assert(s 3, $"Foreign reference {declared.Name} requires at least one element"); loaded]
                           @ unpack @ [allocate; initialize] @ beforeRest,
                           [returned; changed; copyback] @ afterRest, ((s 16, nativeTy) :: pairs)
                | _, None ->
                    match ty with
                    | TStruct (fields, Some bytes) ->
                        let name = match graph.Nodes.[argId].Type with NativeType.TApp (tc, _) -> Some tc.Name | _ -> None
                        let descriptor = layouts |> List.tryFind (fun d -> d.RecordType = name && name.IsSome)
                        match descriptor with
                        | Some d when d.Size.IsSome && d.Alignment.IsSome && d.PhysicalFields.Length = fields.Length ->
                            let nativeField (field: Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.DeclaredPhysicalField) =
                                match field.Repr with
                                | "pointer" -> TIndex
                                | "f32" -> TFloat F32 | "f64" -> TFloat F64
                                | "u8" | "i8" -> TInt (IntWidth 8)
                                | "u16" | "i16" -> TInt (IntWidth 16)
                                | "u32" | "i32" -> TInt (IntWidth 32)
                                | "u64" | "i64" -> TInt (IntWidth 64)
                                | other -> failwithf "Unsupported foreign record field representation '%s'" other
                            let nativeFields = d.PhysicalFields |> List.map (fun f -> f.Name, nativeField f)
                            let nativeBytes = { Size = d.Size.Value; Align = d.Alignment.Value; Offsets = d.PhysicalFields |> List.map (fun f -> f.Offset) }
                            let nativeTy = TStruct (nativeFields, Some nativeBytes)
                            if ty = nativeTy then
                                let! before, after, pairs = loop cursor rest
                                return before, after, ((value, ty) :: pairs)
                            else
                                do! ensure (not (Set.contains argId recordReferences) || Set.contains argId readOnlyRecords)
                                        "A writable foreign record requires identical source/native storage or an explicit copy-back adapter"
                                let sourceFields = name |> Option.bind (fun name -> Clef.Compiler.PSGSaturation.SemanticGraph.Core.SemanticGraph.tryGetRecordFields name graph) |> Option.defaultValue []
                                do! ensure (sourceFields.Length = fields.Length) "Foreign record projection requires source field types"
                                do! ensure (cursor + 1 + 20 * fields.Length <= ssas.Length) "Foreign record projection exceeds the node's assigned SSA family"
                                do! ensure (d.PhysicalFields |> List.forall (fun f -> f.Count = 1)) "Foreign record array fields require an explicit bounded projection"
                                let buffer = ssas.[cursor]
                                let! alloc = pAlloca buffer nativeBytes.Size (TInt (IntWidth 8)) (Some nativeBytes.Align)
                                let rec copy index (remaining: ((string * MLIRType) * Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.DeclaredPhysicalField) list) : PSGParser<MLIROp list> =
                                    parser {
                                        match remaining with
                                        | [] -> return []
                                        | ((fieldName, sourceTy), native) :: tail ->
                                            do! ensure (fieldName = native.Name) "Foreign record field order disagrees with its descriptor"
                                            let start = cursor + 1 + 20*index
                                            let s n = ssas.[start + n]
                                            let! extract = pTypedExtractView (s 0) value bytes.Offsets.[index] (s 1) (s 2) (s 3) sourceTy ty
                                            let expected = nativeField native
                                            let! conversion, stored =
                                                match sourceTy, expected with
                                                | TMemRefStatic (_, TInt (IntWidth 8)), TIndex ->
                                                    pUnwrapOptionArgForFFI (snd sourceFields.[index]) (s 0) sourceTy TIndex ssas (start + 4)
                                                | _ when sourceTy = expected -> preturn ([], { SSA = s 0; Type = expected })
                                                | _ -> fail (Message $"Foreign field {fieldName}: cannot project {sourceTy} into {expected}")
                                            let! insert = pTypedInsertView buffer stored.SSA native.Offset (s 15) (s 16) (s 17) expected nativeTy
                                            let! remainder = copy (index + 1) tail
                                            return extract @ conversion @ insert @ remainder
                                    }
                                let! copies = copy 0 (List.zip fields d.PhysicalFields)
                                let! before, after, pairs = loop (cursor + 1 + 20*fields.Length) rest
                                return [alloc] @ copies @ before, after, ((buffer, nativeTy) :: pairs)
                        | _ ->
                            let! before, after, pairs = loop cursor rest
                            return before, after, ((value, ty) :: pairs)
                    | _ ->
                        let! before, after, pairs = loop cursor rest
                        return before, after, ((value, ty) :: pairs)
        }
    parser {
        do! ensure (pointerReferenceArguments.Length <= 1) "Multiple native pointer output references require an alias-preserving adapter"
        do! ensure (22 + 13 * argIds.Length <= 128) "Foreign argument marshaling exceeds its assigned SSA partition"
        return! loop 128 (List.zip argIds argPairs)
    }

/// A foreign record must match its measured BAREWire layout. Passing by
/// reference lends that storage; passing by value additionally requires the
/// target's declared calling convention and the supported aggregate class.
let private byvalOf (graph: SemanticGraph) (references: Set<NodeId>) (platformId: string) (argWithIds: (NodeId * (SSA * MLIRType)) list) (isOptionArgument: NodeId -> bool) : ByvalParam list =
    let layouts = (Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.readDescriptors graph).Layouts
    let declaredAbi = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.cAbiOfGraph graph
    argWithIds
    |> List.mapi (fun i (argId, (_ssa, ty)) ->
        match ty with
        | TStruct (fields, Some bytes) when not (isOptionArgument argId) ->
            let nativeName = match graph.Nodes.[argId].Type with NativeType.TApp (tc, _) -> Some tc.Name | _ -> None
            let layout = layouts |> List.tryFind (fun d -> d.RecordType = nativeName && nativeName.IsSome)
            match layout with
            | Some d when d.Size = Some bytes.Size && d.Alignment = Some bytes.Align && d.PhysicalFields.Length = fields.Length ->
                let compatible =
                    List.zip3 fields bytes.Offsets d.PhysicalFields |> List.forall (fun ((name, fieldType), offset, declared) ->
                        let reprMatches =
                            match declared.Repr, fieldType with
                            | "pointer", TIndex -> true
                            | "f32", TFloat F32 | "f64", TFloat F64 -> true
                            | ("u8" | "i8"), TInt (IntWidth 8)
                            | ("u16" | "i16"), TInt (IntWidth 16)
                            | ("u32" | "i32"), TInt (IntWidth 32)
                            | ("u64" | "i64"), TInt (IntWidth 64) -> true
                            | _ -> false
                        name = declared.Name && offset = declared.Offset && declared.Count = 1 && reprMatches)
                if not compatible then failwithf "CCS8207: foreign record '%s' storage disagrees with its measured fields" d.Name
                if Set.contains argId references then None
                else
                    match declaredAbi, graph.Platform |> Option.bind (fun p -> PlatformContext.pointerSize p |> Result.toOption) with
                    | [ ("sysv-amd64", 64, 16) ], Some 8 when bytes.Size > 16 ->
                        Some { ParamIndex = i; SizeBytes = bytes.Size; AlignBytes = bytes.Align }
                    | _ -> failwithf "CCS8203: '%s' has no supported C ABI aggregate passing rule for '%s' (%d bytes)" platformId d.Name bytes.Size
            | _ -> failwithf "CCS8207: foreign record argument %d has no matching measured BAREWire layout (%d bytes, alignment %d)" i bytes.Size bytes.Align
        | TStruct _ -> failwithf "CCS8203: foreign record argument %d has no settled layout" i
        | _ -> None)
    |> List.choose id

/// Source `f ()` supplies unit; a C `f(void)` call has no argument.
let private foreignValues (graph: SemanticGraph) argIds marshaled =
    List.zip argIds marshaled
    |> List.choose (fun (arg, (_, value)) ->
        match Map.tryFind arg graph.Nodes with
        | Some node when Types.tryGetNTUKind node.Type = Some NTUKind.NTUunit -> None
        | _ -> Some value)

/// A declared scalar reference lends the first element of a typed array to C.
/// Its element width must agree with the descriptor, and even an empty array
/// must fail before C can write to it. The slots belong to the extern's existing
/// per-argument allocation; this guard allocates no ad-hoc SSA identifiers.
let private referenceGuards graph funcId argIds argPairs (ssas: SSA list) start : PSGParser<MLIROp list> =
    let scalar = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.referenceArguments graph funcId argIds |> List.map (fun (id, d) -> id, (d.Name, TInt (IntWidth d.Bits)))
    let pointers = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.pointerReferenceArguments graph funcId argIds |> List.map (fun (id, d) -> id, (d.Name, TIndex))
    let declared = scalar @ pointers |> Map.ofList
    let rec loop items =
        parser {
            match items with
            | [] -> return []
            | (i, (argId, (ssa, ty))) :: rest ->
                match Map.tryFind argId declared with
                | None -> return! loop rest
                | Some (name, expected) ->
                    let element = match ty with TMemRef elem | TMemRefStatic (_, elem) -> Some elem | _ -> None
                    do! ensure (element = Some expected) $"The foreign reference '{name}' requires an array with {expected} elements; the settled storage is {ty}."
                    let zero, one, length, valid = ssas.[start + 13*i + 2], ssas.[start + 13*i + 3], ssas.[start + 13*i + 4], ssas.[start + 13*i + 5]
                    let! zeroOp = pConstI zero 0L TIndex
                    let! oneOp = pConstI one 1L TIndex
                    let! validOp = pCmpI valid ICmpPred.Uge length one TIndex
                    let! restOps = loop rest
                    return [ zeroOp; oneOp; MLIROp.MemRefOp (MemRefOp.Dim(length, ssa, zero, ty)); validOp
                             MLIROp.Assert(valid, $"Foreign reference {name} requires at least one element") ] @ restOps
        }
    loop (List.indexed (List.zip argIds argPairs))

/// Explicit project/dependency link declarations choose direct symbol calls;
/// absent declarations retain dynamic lookup. Library identity is unchanged.
let isLinkedExtern (platform: PlatformReads) library = library = "c" || Set.contains library platform.LinkedLibraries

/// ExternCall resolved pattern — emits func.call for STATIC [<FidelityExtern>] bindings.
/// Matches Application nodes with ExternCall(library="c") in pre-computed coeffects.
/// Uses the C symbol name from the binding (not the Clef function name).
///
/// STATIC vs DYNAMIC (Mar 2026): This pattern handles only statically-linked libraries
/// (library = "c"). Dynamic libraries (library != "c") are handled by
/// pDynamicExternCallResolved, which emits dlopen/dlsym/call_indirect sequences.
///
/// FFI MARSHALING: When the Fidelity-level return type differs from the
/// C-level return type, this pattern inserts marshaling at the boundary:
///   - option<T> return → C returns nullable pointer (index); null-check + option construction
///   - Direct types → no marshaling needed (passthrough)
///
/// This is the FFI boundary where DTS concretization is legitimate — C's types ARE
/// genuine width demands. The pattern observes the ExternCall coeffect and the marshaling
/// code elides naturally as the residual (Pillar 4: Elision).
///
/// Discriminator: coeffect-based. If Platform.Bindings[nodeId] has ExternCall(library="c"),
/// PlatformWitness handles it via this pattern. Dynamic externs are handled separately.
///
/// SSA layout:
///   Direct return: [0] = result, [1..N] = FFI arg extraction (memref→index)
///   Option return: [0] = option memref result, [1] = raw C return, [2] = null const,
///     [3] = cmp result, [4] = tag extended, [5..6] = tag insert views,
///     [7..9] = payload insert views + offset, [10] = alloca,
///     [11..11+N] = FFI arg extraction (memref→index)
let pExternCallResolved : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Match Application node
        let! (funcId, argIds) = pApplication
        let! node = getCurrentNode
        let! state = getUserState

        // Guard: check if this node has a STATIC ExternCall binding (library = "c")
        match Map.tryFind node.Id state.Platform.Bindings.Bindings with
        | Some { Resolved = ResolvedBinding.ExternCall (library, symbol) } when isLinkedExtern state.Platform library ->
            // FFI namespace prefix: all extern symbols get "ffi." prefix in MLIR
            // to avoid collisions with MLIR infrastructure symbols (e.g., @malloc
            // from finalize-memref-to-llvm). The reconcile-ffi-externs plugin
            // strips the prefix after standard lowering passes complete.
            let ffiSymbol = "ffi." + symbol

            // Get pre-allocated SSAs
            let! ssas = getNodeSSAs node.Id

            // Recall and marshal arguments in one monadic fold.
            // At the FFI boundary, memref-typed args need pointer extraction
            // via pExtractBasePtr (Element) — the extraction op elides naturally
            // as the residual of observing memref at the C boundary (Pillar 4).
            // Cast SSAs are pre-allocated in coeffects (Pillar 1).
            let! (argMeetOps, argPairs) = recallArgs node.Id argIds
            let! boundaryBefore, boundaryAfter, argPairs = projectForeignArguments state.Graph funcId argIds argPairs ssas
            let argMeetOps = argMeetOps @ boundaryBefore

            // Helper: check if an argument's original NativeType is option/voption.
            // Record types (e.g., resvg_transform) also lower to TMemRefStatic(N, i8)
            // in MLIR, so we must check the NativeType to distinguish them from options.
            let isOptionArgument (argId: NodeId) =
                match Map.tryFind argId state.Graph.Nodes with
                | Some argNode ->
                    match argNode.Type with
                    | NativeType.TApp(tycon, _) when tycon.Name = "option" || tycon.Name = "voption" -> true
                    | _ -> false
                | None -> false
            let argWithIds = List.zip argIds argPairs

            let byvalParams = byvalOf state.Graph (Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.recordReferenceArguments state.Graph funcId argIds) state.Graph.Platform.Value.PlatformId argWithIds isOptionArgument

            // Detect option<T> return type — requires FFI marshaling at the boundary.
            // C returns a nullable pointer; we must null-check and construct the option.
            match node.Type with
            | NativeType.TApp(tycon, [innerTy]) when tycon.Name = "option" || tycon.Name = "voption" ->
                // FFI MARSHALING: option<T> return
                // C returns the inner type (pointer); null = None, non-null = Some(value)
                do! ensure (ssas.Length >= 11) $"pExternCallResolved (option): Expected at least 11 SSAs, got {ssas.Length}"
                let resultSSA   = ssas.[0]   // Final option memref
                let rawRetSSA   = ssas.[1]   // Raw C return value
                let nullConstSSA = ssas.[2]  // Null constant for comparison
                let cmpSSA      = ssas.[3]   // Comparison result (i1)
                let tagExtSSA   = ssas.[4]   // Extended tag (i8)
                // ssas.[5..6] for tag insert (viewSSA, zeroSSA)
                let tagViewSSA  = ssas.[5]
                let tagZeroSSA  = ssas.[6]
                // ssas.[7..9] for payload insert (offsetSSA, viewSSA, zeroSSA)
                let payOffsetSSA = ssas.[7]
                let payViewSSA  = ssas.[8]
                let payZeroSSA  = ssas.[9]
                // ssas.[10] = alloca for option memref
                let allocaSSA   = ssas.[10]

                // FFI argument marshaling: extract pointers from memref args,
                // unwrap option-typed args (None→NULL, Some→payload), and
                // cast TIndex values to PlatformWordType at the boundary.
                // Each arg gets 11 SSA slots from coeffects (Pillar 1) starting at ssas.[11]:
                //   Option args: all 11 slots used by pUnwrapOptionArgForFFI
                //   Memref args: slots 0-1 used (extraction + boundary cast)
                //   Index args: slot 1 used (boundary cast)
                // "Good fences make good neighbors" — internal types stay inside,
                // C ABI types cross the boundary.
                let platformWordTy = state.Platform.PlatformWordType
                let! marshaledArgs =
                    let rec fold i items = parser {
                        match items with
                        | [] -> return []
                        | (argId, (ssa, ty)) :: rest ->
                            match ty with
                            | TMemRefStatic (_, TInt(IntWidth 8)) when isOptionArgument argId ->
                                // Option/DU at FFI boundary: unwrap via composed pattern
                                // pTypedExtract (tag) → pTypedExtractView (payload) → pCmpI → pSelect
                                let! (ops, v) = pUnwrapOptionArgForFFI state.Graph.Nodes.[argId].Type ssa ty platformWordTy ssas (11 + 13*i)
                                let! restResult = fold (i + 1) rest
                                return (ops, v) :: restResult
                            | TMemRef _ | TMemRefStatic _ | TStruct _ ->
                                // Memref/struct → extract pointer (index) → cast to PlatformWordType.
                                // TStruct (record types) recalled from accumulator also need pointer
                                // extraction at FFI boundaries, same as memrefs.
                                let extractSlot = ssas.[11 + 13*i]
                                let castSlot = ssas.[11 + 13*i + 1]
                                let! extractOp = pExtractBasePtr extractSlot ssa ty
                                let! castOp = pIndexCastS castSlot extractSlot TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([extractOp; castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | TIndex ->
                                // Index value → cast to PlatformWordType at boundary
                                let castSlot = ssas.[11 + 13*i + 1]
                                let! castOp = pIndexCastS castSlot ssa TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | _ ->
                                // Non-pointer arg — passes through unchanged
                                let! restResult = fold (i + 1) rest
                                return ([], { SSA = ssa; Type = ty }) :: restResult
                    }
                    fold 0 argWithIds
                let marshalOps = marshaledArgs |> List.collect fst
                let vals = foreignValues state.Graph argIds marshaledArgs
                let cArgTypes = vals |> List.map (fun v -> v.Type)

                // Map inner type to MLIR, then marshal to C ABI type at the boundary
                let! internalRetType = pMapType innerTy
                let cRetType = marshalToCType platformWordTy internalRetType
                // Map full option type to MLIR (for the constructed result)
                let! optionType = pMapType node.Type

                // 1. Declare extern with C-level (marshaled) arg types and return type
                let! declOp = pFuncDeclByval ffiSymbol cArgTypes cRetType FuncVisibility.Private byvalParams
                // 2. Call extern — get raw C value back
                let! callOp = pFuncCall (Some rawRetSSA) ffiSymbol vals cRetType

                // 3. Alloc option memref on HEAP (tag byte + payload)
                // Must be heap-allocated: this memref is returned from the wrapper function.
                // Stack alloca would produce a dangling pointer after the callee's frame is destroyed.
                let optionElemTy = TInt (IntWidth 8)
                let! optionSize = match optionType with TMemRefStatic (n, TInt (IntWidth 8)) -> preturn n | _ -> fail (Message "Foreign option result lacks settled byte storage")
                let! allocaOp = pAllocStatic allocaSSA optionSize optionElemTy None

                // 4. Null-check: compare raw result to 0 (null)
                // Comparison uses C-boundary type (PlatformWordType)
                let! nullOp = pConstI nullConstSSA 0L cRetType
                let! cmpOp = pCmpI cmpSSA ICmpPred.Ne rawRetSSA nullConstSSA cRetType

                // 5. Extend i1 → i8 for tag value (0 = None, 1 = Some)
                let! extOp = pExtUI tagExtSSA cmpSSA (TInt (IntWidth 1)) (TInt (IntWidth 8))

                // 6. Store tag at offset 0
                let! tagInsertOps = pTypedInsert allocaSSA tagExtSSA 0 tagViewSSA tagZeroSSA (TInt (IntWidth 8)) optionType

                // 7. Store payload at offset 1 (always — value is meaningless for None,
                //    CaseElimination checks tag before reading payload)
                let! payInsertOps = pTypedInsertView allocaSSA rawRetSSA (unionPayloadOffset state.Graph node.Type) payOffsetSSA payViewSSA payZeroSSA cRetType optionType

                let! guards = referenceGuards state.Graph funcId argIds argPairs ssas 11
                let allOps = argMeetOps @ guards @ marshalOps @ [declOp; callOp] @ boundaryAfter @ [allocaOp; nullOp; cmpOp; extOp]
                             @ tagInsertOps @ payInsertOps

                return (allOps, TRValue { SSA = allocaSSA; Type = optionType })

            | _ ->
                // DIRECT RETURN: type passes through with boundary marshaling
                let platformWordTy = state.Platform.PlatformWordType
                do! ensure (ssas.Length >= 2) $"pExternCallResolved: Expected at least 2 SSAs, got {ssas.Length}"
                let resultSSA = ssas.[0]
                let returnCastSSA = ssas.[1]  // potential platformWordTy → index cast on return

                // FFI argument marshaling: extract pointers from memref args,
                // unwrap option-typed args (None→NULL, Some→payload), and
                // cast TIndex values to PlatformWordType at the boundary.
                // Each arg gets 11 SSA slots from coeffects (Pillar 1) starting at ssas.[2].
                let! marshaledArgs =
                    let rec fold i items = parser {
                        match items with
                        | [] -> return []
                        | (argId, (ssa, ty)) :: rest ->
                            match ty with
                            | TMemRefStatic (_, TInt(IntWidth 8)) when isOptionArgument argId ->
                                // Option/DU at FFI boundary: unwrap via composed pattern
                                let! (ops, v) = pUnwrapOptionArgForFFI state.Graph.Nodes.[argId].Type ssa ty platformWordTy ssas (2 + 13*i)
                                let! restResult = fold (i + 1) rest
                                return (ops, v) :: restResult
                            | TMemRef _ | TMemRefStatic _ | TStruct _ ->
                                // Memref/struct → extract pointer (index) → cast to PlatformWordType.
                                // TStruct (record types) serialize as memref<Nxi8> in MLIR but are
                                // recalled as TStruct in the accumulator. At FFI boundaries, we extract
                                // the base pointer — on SysV x86_64, structs >16 bytes are passed by
                                // a pointer names the source of the copy described by LLVM byval.
                                let extractSlot = ssas.[2 + 13*i]
                                let castSlot = ssas.[2 + 13*i + 1]
                                let! extractOp = pExtractBasePtr extractSlot ssa ty
                                let! castOp = pIndexCastS castSlot extractSlot TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([extractOp; castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | TIndex ->
                                // Index value → cast to PlatformWordType at boundary
                                let castSlot = ssas.[2 + 13*i + 1]
                                let! castOp = pIndexCastS castSlot ssa TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | _ ->
                                // Non-pointer arg — passes through unchanged
                                let! restResult = fold (i + 1) rest
                                return ([], { SSA = ssa; Type = ty }) :: restResult
                    }
                    fold 0 argWithIds
                let marshalOps = marshaledArgs |> List.collect fst
                let vals = foreignValues state.Graph argIds marshaledArgs
                let cArgTypes = vals |> List.map (fun v -> v.Type)

                // Map return type from Clef NativeType to MLIR, then marshal at boundary
                let! internalRetType = pForeignReturnType funcId node.Type
                let cRetType = marshalToCType platformWordTy internalRetType

                // Emit external function declaration + call with C-boundary types
                let! declOp = pFuncDeclByval ffiSymbol cArgTypes cRetType FuncVisibility.Private byvalParams
                let! callOp = pFuncCall (if cRetType = TVoid then None else Some resultSSA) ffiSymbol vals cRetType

                let! guards = referenceGuards state.Graph funcId argIds argPairs ssas 2

                // Return-side demarshal: if the internal type is TIndex (nativeint),
                // the call returned platformWordTy (i64). Cast back to index so
                // the rest of the middle-end stays width-abstract until LLVM lowering.
                match internalRetType with
                | TVoid ->
                    let! unitTy = pMapType node.Type
                    let! unitOp = pConstI resultSSA 0L unitTy
                    return (argMeetOps @ guards @ marshalOps @ [declOp; callOp] @ boundaryAfter @ [unitOp], TRValue { SSA = resultSSA; Type = unitTy })
                | TIndex ->
                    let! returnCastOp = pIndexCastS returnCastSSA resultSSA platformWordTy TIndex
                    return (argMeetOps @ guards @ marshalOps @ [declOp; callOp] @ boundaryAfter @ [returnCastOp], TRValue { SSA = returnCastSSA; Type = TIndex })
                | _ ->
                    return (argMeetOps @ guards @ marshalOps @ [declOp; callOp] @ boundaryAfter, TRValue { SSA = resultSSA; Type = cRetType })
        | _ -> return! fail (Message "Not a static ExternCall")
    }

// ═══════════════════════════════════════════════════════════
// DYNAMIC EXTERN CALL PATTERN (dlopen/dlsym/call_indirect)
// ═══════════════════════════════════════════════════════════

/// Construct the shared-object filename from a library name.
/// "xrt_coreutil" → "libxrt_coreutil.so"
let private soName (library: string) : string =
    sprintf "lib%s.so" library

/// Dynamic extern call — emits dlopen/dlsym/call_indirect for runtime-resolved symbols.
/// Handles FidelityExtern bindings where library != "c" (not statically linked).
///
/// At each call site the pattern emits:
///   1. dlopen(lib_path, RTLD_LAZY) → handle
///   2. dlsym(handle, symbol_name) → raw function pointer (i64)
///   3. Cast i64 → index → typed function pointer (IndexToFunc)
///   4. func.call_indirect through the typed pointer with marshaled args
///
/// String constants for the library path and symbol name are returned as
/// pending globals (name, content, storageLength) for the witness to emit
/// via tryEmitGlobal / GlobalString TopLevelOps.
///
/// dlopen is idempotent; calling it multiple times for the same library
/// returns the same handle. Per-call overhead is acceptable for an MVP;
/// a future caching pass can hoist dlopen/dlsym to module init.
///
/// SSA layout (direct return):
///   [0]  = lib_memref   (memref.get_global for library path)
///   [1]  = lib_ptr_idx  (memref.extract_aligned_pointer_as_index)
///   [2]  = lib_ptr      (index.casts index → PlatformWordType)
///   [3]  = dlopen_mode  (arith.constant RTLD_LAZY = 1)
///   [4]  = handle       (func.call @ffi.dlopen)
///   [5]  = sym_memref   (memref.get_global for symbol name)
///   [6]  = sym_ptr_idx  (memref.extract_aligned_pointer_as_index)
///   [7]  = sym_ptr      (index.casts index → PlatformWordType)
///   [8]  = raw_ptr      (func.call @ffi.dlsym → PlatformWordType)
///   [9]  = raw_ptr_idx  (index.casts PlatformWordType → index)
///   [10] = func_ptr     (IndexToFunc: index → typed function pointer)
///   [11] = result       (func.call_indirect result)
///   [12] = return_cast  (potential return-side demarshal)
///   [13 + 13*i ..] = per-arg FFI marshaling
///
/// SSA layout (option return):
///   [0..10] = dlopen/dlsym preamble (same as direct return)
///   [11] = option_memref  (final option result)
///   [12] = raw_ret        (raw C return from call_indirect)
///   [13] = null_const     (0 for null comparison)
///   [14] = cmp_result     (ne comparison result)
///   [15] = tag_ext        (i1 → i8 for tag)
///   [16..17] = tag insert views
///   [18..20] = payload insert views + offset
///   [21] = alloca         (option heap alloc)
///   [22 + 13*i ..] = per-arg FFI marshaling
let pDynamicExternCallResolved : PSGParser<MLIROp list * (string * string * int) list * TransferResult> =
    parser {
        let! (funcId, argIds) = pApplication
        let! node = getCurrentNode
        let! state = getUserState

        match Map.tryFind node.Id state.Platform.Bindings.Bindings with
        | Some { Resolved = ResolvedBinding.ExternCall (library, symbol) } when not (isLinkedExtern state.Platform library) ->
            let platformWordTy = state.Platform.PlatformWordType
            let! ssas = getNodeSSAs node.Id
            let! (argMeetOps, argPairs) = recallArgs node.Id argIds
            let! boundaryBefore, boundaryAfter, argPairs = projectForeignArguments state.Graph funcId argIds argPairs ssas
            let argMeetOps = argMeetOps @ boundaryBefore

            let isOptionArgument (argId: NodeId) =
                match Map.tryFind argId state.Graph.Nodes with
                | Some argNode ->
                    match argNode.Type with
                    | NativeType.TApp(tycon, _) when tycon.Name = "option" || tycon.Name = "voption" -> true
                    | _ -> false
                | None -> false
            let argWithIds = List.zip argIds argPairs

            let byvalParams = byvalOf state.Graph (Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.recordReferenceArguments state.Graph funcId argIds) state.Graph.Platform.Value.PlatformId argWithIds isOptionArgument

            do! ensure (List.isEmpty byvalParams) "Dynamic aggregate-by-value calls require explicit linked-library dispatch"

            // ── dlopen/dlsym preamble (SSAs [0..10]) ──

            // String constants: library path and symbol name
            let libPath = soName library
            let libGlobalName = deriveGlobalRef libPath
            let libByteLen = deriveByteLength libPath
            let libStorageLen = libByteLen + 1  // null sentinel
            let libStorageTy = TMemRefStatic (libStorageLen, TInt (IntWidth 8))

            let symGlobalName = deriveGlobalRef symbol
            let symByteLen = deriveByteLength symbol
            let symStorageLen = symByteLen + 1
            let symStorageTy = TMemRefStatic (symStorageLen, TInt (IntWidth 8))

            // Pending globals to emit as TopLevelOps (witness handles deduplication)
            let pendingGlobals = [
                (libGlobalName, libPath, libStorageLen)
                (symGlobalName, symbol, symStorageLen)
            ]

            // SSAs [0..10]: dlopen/dlsym preamble
            let libMemrefSSA  = ssas.[0]
            let libPtrIdxSSA  = ssas.[1]
            let libPtrSSA     = ssas.[2]
            let dlopenModeSSA = ssas.[3]
            let handleSSA     = ssas.[4]
            let symMemrefSSA  = ssas.[5]
            let symPtrIdxSSA  = ssas.[6]
            let symPtrSSA     = ssas.[7]
            let rawPtrSSA     = ssas.[8]
            let rawPtrIdxSSA  = ssas.[9]
            let funcPtrSSA    = ssas.[10]

            // 1. Load library path string → extract pointer → cast to platform word
            let! libGetGlobalOp = pMemRefGetGlobal libMemrefSSA libGlobalName libStorageTy
            let! libExtractOp = pExtractBasePtr libPtrIdxSSA libMemrefSSA libStorageTy
            let! libCastOp = pIndexCastS libPtrSSA libPtrIdxSSA TIndex platformWordTy

            // 2. RTLD_LAZY = 1 (mode for dlopen)
            let! modeOp = pConstI dlopenModeSSA 1L platformWordTy

            // 3. Call dlopen — returns library handle as platform word
            //    dlopen is FidelityExtern("c", "dlopen"), resolved statically via libc.
            let dlopenArgs = [
                { SSA = libPtrSSA; Type = platformWordTy }
                { SSA = dlopenModeSSA; Type = platformWordTy }
            ]
            let! dlopenDeclOp = pFuncDecl "ffi.dlopen" [platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private
            let! dlopenCallOp = pFuncCall (Some handleSSA) "ffi.dlopen" dlopenArgs platformWordTy

            // 4. Load symbol name string → extract pointer → cast to platform word
            let! symGetGlobalOp = pMemRefGetGlobal symMemrefSSA symGlobalName symStorageTy
            let! symExtractOp = pExtractBasePtr symPtrIdxSSA symMemrefSSA symStorageTy
            let! symCastOp = pIndexCastS symPtrSSA symPtrIdxSSA TIndex platformWordTy

            // 5. Call dlsym — returns raw function pointer as platform word
            let dlsymArgs = [
                { SSA = handleSSA; Type = platformWordTy }
                { SSA = symPtrSSA; Type = platformWordTy }
            ]
            let! dlsymDeclOp = pFuncDecl "ffi.dlsym" [platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private
            let! dlsymCallOp = pFuncCall (Some rawPtrSSA) "ffi.dlsym" dlsymArgs platformWordTy

            // 6. Cast raw pointer (i64) → index → typed function pointer
            let! ptrToIdxOp = pIndexCastS rawPtrIdxSSA rawPtrSSA platformWordTy TIndex

            let preambleOps = [
                dlopenDeclOp; dlsymDeclOp;
                libGetGlobalOp; libExtractOp; libCastOp; modeOp; dlopenCallOp;
                symGetGlobalOp; symExtractOp; symCastOp; dlsymCallOp; ptrToIdxOp
            ]

            // ── Dispatch: option return vs direct return ──

            match node.Type with
            | NativeType.TApp(tycon, [innerTy]) when tycon.Name = "option" || tycon.Name = "voption" ->
                // OPTION RETURN with dynamic binding
                do! ensure (ssas.Length >= 22) $"pDynamicExternCallResolved (option): Expected at least 22 SSAs, got {ssas.Length}"
                let resultSSA    = ssas.[11]
                let rawRetSSA    = ssas.[12]
                let nullConstSSA = ssas.[13]
                let cmpSSA       = ssas.[14]
                let tagExtSSA    = ssas.[15]
                let tagViewSSA   = ssas.[16]
                let tagZeroSSA   = ssas.[17]
                let payOffsetSSA = ssas.[18]
                let payViewSSA   = ssas.[19]
                let payZeroSSA   = ssas.[20]
                let allocaSSA    = ssas.[21]

                // Marshal arguments (same logic as static extern)
                let! marshaledArgs =
                    let rec fold i items = parser {
                        match items with
                        | [] -> return []
                        | (argId, (ssa, ty)) :: rest ->
                            match ty with
                            | TMemRefStatic (_, TInt(IntWidth 8)) when isOptionArgument argId ->
                                let! (ops, v) = pUnwrapOptionArgForFFI state.Graph.Nodes.[argId].Type ssa ty platformWordTy ssas (22 + 13*i)
                                let! restResult = fold (i + 1) rest
                                return (ops, v) :: restResult
                            | TMemRef _ | TMemRefStatic _ | TStruct _ ->
                                let extractSlot = ssas.[22 + 13*i]
                                let castSlot = ssas.[22 + 13*i + 1]
                                let! extractOp = pExtractBasePtr extractSlot ssa ty
                                let! castOp = pIndexCastS castSlot extractSlot TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([extractOp; castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | TIndex ->
                                let castSlot = ssas.[22 + 13*i + 1]
                                let! castOp = pIndexCastS castSlot ssa TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | _ ->
                                let! restResult = fold (i + 1) rest
                                return ([], { SSA = ssa; Type = ty }) :: restResult
                    }
                    fold 0 argWithIds
                let marshalOps = marshaledArgs |> List.collect fst
                let vals = foreignValues state.Graph argIds marshaledArgs
                let cArgTypes = vals |> List.map (fun v -> v.Type)

                let! internalRetType = pMapType innerTy
                let cRetType = marshalToCType platformWordTy internalRetType
                let! optionType = pMapType node.Type

                // IndexToFunc: cast index → typed function pointer for call_indirect
                let funcTyArgs = cArgTypes
                let indexToFuncOp = MLIROp.FuncOp (FuncOp.IndexToFunc (funcPtrSSA, rawPtrIdxSSA, funcTyArgs, cRetType))

                // call_indirect through resolved function pointer
                let! callOp = pFuncCallIndirect (Some rawRetSSA) funcPtrSSA vals cRetType

                // Option construction (same as static path)
                let optionElemTy = TInt (IntWidth 8)
                let! optionSize = match optionType with TMemRefStatic (n, TInt (IntWidth 8)) -> preturn n | _ -> fail (Message "Foreign option result lacks settled byte storage")
                let! allocaOp = pAllocStatic allocaSSA optionSize optionElemTy None
                let! nullOp = pConstI nullConstSSA 0L cRetType
                let! cmpOp = pCmpI cmpSSA ICmpPred.Ne rawRetSSA nullConstSSA cRetType
                let! extOp = pExtUI tagExtSSA cmpSSA (TInt (IntWidth 1)) (TInt (IntWidth 8))
                let! tagInsertOps = pTypedInsert allocaSSA tagExtSSA 0 tagViewSSA tagZeroSSA (TInt (IntWidth 8)) optionType
                let! payInsertOps = pTypedInsertView allocaSSA rawRetSSA (unionPayloadOffset state.Graph node.Type) payOffsetSSA payViewSSA payZeroSSA cRetType optionType

                let! guards = referenceGuards state.Graph funcId argIds argPairs ssas 22
                let allOps = argMeetOps @ guards @ preambleOps @ [indexToFuncOp] @ marshalOps @ [callOp] @ boundaryAfter @ [allocaOp; nullOp; cmpOp; extOp]
                             @ tagInsertOps @ payInsertOps

                return (allOps, pendingGlobals, TRValue { SSA = allocaSSA; Type = optionType })

            | _ ->
                // DIRECT RETURN with dynamic binding
                do! ensure (ssas.Length >= 13) $"pDynamicExternCallResolved: Expected at least 13 SSAs, got {ssas.Length}"
                let resultSSA = ssas.[11]
                let returnCastSSA = ssas.[12]

                // Marshal arguments
                let! marshaledArgs =
                    let rec fold i items = parser {
                        match items with
                        | [] -> return []
                        | (argId, (ssa, ty)) :: rest ->
                            match ty with
                            | TMemRefStatic (_, TInt(IntWidth 8)) when isOptionArgument argId ->
                                let! (ops, v) = pUnwrapOptionArgForFFI state.Graph.Nodes.[argId].Type ssa ty platformWordTy ssas (13 + 13*i)
                                let! restResult = fold (i + 1) rest
                                return (ops, v) :: restResult
                            | TMemRef _ | TMemRefStatic _ | TStruct _ ->
                                let extractSlot = ssas.[13 + 13*i]
                                let castSlot = ssas.[13 + 13*i + 1]
                                let! extractOp = pExtractBasePtr extractSlot ssa ty
                                let! castOp = pIndexCastS castSlot extractSlot TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([extractOp; castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | TIndex ->
                                let castSlot = ssas.[13 + 13*i + 1]
                                let! castOp = pIndexCastS castSlot ssa TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | _ ->
                                let! restResult = fold (i + 1) rest
                                return ([], { SSA = ssa; Type = ty }) :: restResult
                    }
                    fold 0 argWithIds
                let marshalOps = marshaledArgs |> List.collect fst
                let vals = foreignValues state.Graph argIds marshaledArgs
                let cArgTypes = vals |> List.map (fun v -> v.Type)

                let! internalRetType = pForeignReturnType funcId node.Type
                let cRetType = marshalToCType platformWordTy internalRetType

                // IndexToFunc: cast index → typed function pointer for call_indirect
                let funcTyArgs = cArgTypes
                let indexToFuncOp = MLIROp.FuncOp (FuncOp.IndexToFunc (funcPtrSSA, rawPtrIdxSSA, funcTyArgs, cRetType))

                // call_indirect through resolved function pointer
                let! callOp = pFuncCallIndirect (if cRetType = TVoid then None else Some resultSSA) funcPtrSSA vals cRetType

                let! guards = referenceGuards state.Graph funcId argIds argPairs ssas 13

                // Return-side demarshal (same as static path)
                match internalRetType with
                | TVoid ->
                    let! unitTy = pMapType node.Type
                    let! unitOp = pConstI resultSSA 0L unitTy
                    return (argMeetOps @ guards @ preambleOps @ [indexToFuncOp] @ marshalOps @ [callOp] @ boundaryAfter @ [unitOp], pendingGlobals, TRValue { SSA = resultSSA; Type = unitTy })
                | TIndex ->
                    let! returnCastOp = pIndexCastS returnCastSSA resultSSA platformWordTy TIndex
                    return (argMeetOps @ guards @ preambleOps @ [indexToFuncOp] @ marshalOps @ [callOp] @ boundaryAfter @ [returnCastOp], pendingGlobals, TRValue { SSA = returnCastSSA; Type = TIndex })
                | _ ->
                    return (argMeetOps @ guards @ preambleOps @ [indexToFuncOp] @ marshalOps @ [callOp] @ boundaryAfter, pendingGlobals, TRValue { SSA = resultSSA; Type = cRetType })

        | _ -> return! fail (Message "Not a dynamic ExternCall")
    }

// ═══════════════════════════════════════════════════════════
// COMPOSED INTRINSIC PARSERS (per-operation, self-contained)
// ═══════════════════════════════════════════════════════════

/// Sys.write intrinsic — write buffer to file descriptor
let pSysWriteIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Sys
        do! ensure (info.Operation = "write") "Not Sys.write"
        do! ensure (argIds.Length >= 2) "Sys.write: Expected 2 args"
        let! node = getCurrentNode
        let! (_, rawFdSSA, rawFdTy) = pRecallArgWithLoad argIds.[0]
        // the descriptor at the declared Register width, the syscall ABI (its derived meet)
        let! (fdMeetOps, fdSSA, _) = pAdapt node.Id argIds.[0] rawFdSSA rawFdTy
        let! (_, bufferSSA, bufferType) = pRecallArgWithLoad argIds.[1]
        let! (ops, result) = pSysWrite node.Id fdSSA bufferSSA bufferType
        return (fdMeetOps @ ops, result)
    }

/// Sys.read intrinsic — read from file descriptor into buffer
let pSysReadIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Sys
        do! ensure (info.Operation = "read") "Not Sys.read"
        do! ensure (argIds.Length >= 2) "Sys.read: Expected 2 args"
        let! node = getCurrentNode
        let! (_, rawFdSSA, rawFdTy) = pRecallArgWithLoad argIds.[0]
        let! (fdMeetOps, fdSSA, _) = pAdapt node.Id argIds.[0] rawFdSSA rawFdTy
        let! (_, bufferSSA, bufferType) = pRecallArgWithLoad argIds.[1]
        let! (ops, result) = pSysRead node.Id fdSSA bufferSSA bufferType
        return (fdMeetOps @ ops, result)
    }

/// Sys.readline intrinsic — read line from fd, return trimmed string
let pSysReadlineIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Sys
        do! ensure (info.Operation = "readline") "Not Sys.readline"
        do! ensure (argIds.Length >= 1) "Sys.readline: Expected 1 arg"
        let! node = getCurrentNode
        let! (_, rawFdSSA, rawFdTy) = pRecallArgWithLoad argIds.[0]
        let! (fdMeetOps, fdSSA, _) = pAdapt node.Id argIds.[0] rawFdSSA rawFdTy
        let! (ops, result) = pSysReadline node fdSSA
        return (fdMeetOps @ ops, result)
    }
