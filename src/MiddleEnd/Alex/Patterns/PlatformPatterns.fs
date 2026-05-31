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
open Alex.Elements.MLIRAtomics
open Alex.Patterns.LiteralPatterns  // deriveGlobalRef, deriveByteLength (for dynamic extern string constants)
open PSGElaboration.PlatformConfig

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

/// Unwrap an option-typed argument at the FFI boundary.
/// Options are inline DUs: memref<Nxi8> with tag at byte 0, payload at byte 1.
///   None (tag=0) → NULL (0 as PlatformWordType)
///   Some (tag≠0) → payload extracted as PlatformWordType
///
/// Composes from MLIRAtomics (pTypedExtract, pTypedExtractView) and
/// ArithElements (pCmpI, pSelect, pConstI) — proper layer composition.
///
/// SSA layout (11 slots from coeffects, Pillar 1):
///   0-2: tag extraction (tagSSA, tagViewSSA, tagZeroSSA)
///   3-6: payload extraction (payloadSSA, payOffsetSSA, payViewSSA, payZeroSSA)
///   7:   zero i8 constant for tag comparison
///   8:   isSome comparison result (i1)
///   9:   null constant (0 as PlatformWordType)
///   10:  select result
let private pUnwrapOptionArgForFFI
    (optionSSA: SSA) (optionType: MLIRType) (platformWordTy: MLIRType)
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

        // 2. Extract payload from byte offset 1 as PlatformWordType — pTypedExtractView (MLIRAtomics)
        let! payloadOps = pTypedExtractView payloadSSA optionSSA 1 payOffsetSSA payViewSSA payZeroSSA platformWordTy optionType

        // 3. Compare tag ≠ 0 → isSome — pCmpI (ArithElements)
        let! zeroI8Op = pConstI zeroI8SSA 0L tagTy
        let! cmpOp = pCmpI isSomeSSA ICmpPred.Ne tagSSA zeroI8SSA tagTy

        // 4. Null constant for None case
        let! nullOp = pConstI nullSSA 0L platformWordTy

        // 5. Select: isSome ? payload : null — pSelect (ArithElements)
        let! selectOp = pSelect selectSSA isSomeSSA payloadSSA nullSSA platformWordTy

        return (tagOps @ payloadOps @ [zeroI8Op; cmpOp; nullOp; selectOp],
                { SSA = selectSSA; Type = platformWordTy })
    }

// ═══════════════════════════════════════════════════════════
// RESOLVED BINDING LOOKUP
// ═══════════════════════════════════════════════════════════

/// Resolve the target function name for a platform call from pre-computed coeffects.
/// For LibcCall/ExternCall, returns the target function name.
/// For Syscall/InlineAsm, falls back to the provided default (inline asm is future work).
let private resolveCallTarget (nodeId: NodeId) (defaultName: string) (platform: PlatformResolutionResult) : string =
    let (NodeId nodeIdInt) = nodeId
    match Map.tryFind nodeIdInt platform.Bindings with
    | Some binding ->
        match binding.Resolved with
        | LibcCall funcName -> funcName
        | ExternCall (_, symbol) -> symbol
        | Syscall _ -> defaultName   // TODO: emit inline asm for freestanding
        | InlineAsm _ -> defaultName // TODO: emit llvm.inline_asm
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

/// Build Sys.readline pattern — read line from fd, return trimmed string
/// Allocates a 1024-byte buffer, calls read(), creates result subview trimmed of newline.
///
/// SSA layout (14 SSAs):
///   [0]  = bufferSSA (memref.alloc 1024xi8)
///   [1]  = buf_ptr_index (extract base pointer)
///   [2]  = buf_ptr_word (index.casts to platform word)
///   [3]  = capacity_const (1024 as platform word)
///   [4]  = bytesReadSSA (func.call read result)
///   [5]  = bytesReadIndex (index.casts from platform word to index)
///   [6]  = oneConst (constant 1 as index)
///   [7]  = trimmedLen (bytes_read - 1, trims newline)
///   [8]  = resultSSA (memref.subview of buffer[0..trimmedLen])
///   [9]  = sizeConst (1024 as index for alloc)
///   [10] = zeroConst (constant 0 as index for subview offset)
///   [11] = oneStrideConst (constant 1 as index for subview stride)
///   [12] = readDeclSlot (read function decl)
///   [13] = (reserved)
let pSysReadline (nodeId: NodeId) (fdSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
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

        // 1. Allocate 1024-byte read buffer on heap
        let! sizeOp = pConstI sizeConst 1024L TIndex
        let! allocOp = pAlloc bufferSSA sizeConst (TInt (IntWidth 8))

        // 2. Extract pointer for read syscall
        let! extractOp = pExtractBasePtr buf_ptr_index bufferSSA bufferType
        let! castPtrOp = pIndexCastS buf_ptr_word buf_ptr_index TIndex platformWordTy

        // 3. Capacity as platform word
        let! capacityOp = pConstI capacity_const 1024L platformWordTy

        // 4. Call read(fd, buffer, capacity)
        let readArgs = [
            { SSA = fdSSA; Type = platformWordTy }
            { SSA = buf_ptr_word; Type = platformWordTy }
            { SSA = capacity_const; Type = platformWordTy }
        ]
        let! readCall = pFuncCall (Some bytesReadSSA) callTarget readArgs platformWordTy
        let! readDecl = pFuncDecl callTarget [platformWordTy; platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private

        // 5. Convert bytes read to index, subtract 1 for newline
        let! castBytesOp = pIndexCastS bytesReadIdx bytesReadSSA platformWordTy TIndex
        let! oneOp = pConstI oneConst 1L TIndex
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

/// Monadically recall a list of argument nodes from the accumulator.
/// Returns (SSA * MLIRType) pairs preserving argument order.
let private recallArgs (argIds: NodeId list) : PSGParser<(SSA * MLIRType) list> =
    let rec loop (ids: NodeId list) : PSGParser<(SSA * MLIRType) list> =
        parser {
            match ids with
            | [] -> return []
            | id :: rest ->
                let! pair = pRecallNode id
                let! restPairs = loop rest
                return pair :: restPairs
        }
    loop argIds

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
        let! (_funcId, argIds) = pApplication
        let! node = getCurrentNode
        let! state = getUserState

        // Guard: check if this node has a STATIC ExternCall binding (library = "c")
        let (NodeId nodeIdInt) = node.Id
        match Map.tryFind nodeIdInt state.Platform.Bindings with
        | Some { Resolved = ExternCall (library, symbol) } when library = "c" ->
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
            let! argPairs = recallArgs argIds

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

            // Collect byval metadata for struct parameters at FFI boundaries.
            // SysV x86_64: structs > 16 bytes are MEMORY class — passed on the stack
            // via invisible reference. LLVM needs `byval` attribute to generate correct ABI.
            let byvalParams =
                argWithIds
                |> List.mapi (fun i (_argId, (_ssa, ty)) ->
                    match ty with
                    | TStruct _ when not (isOptionArgument (fst (argWithIds.[i]))) ->
                        let size = mlirTypeSize ty
                        if size > 16 then Some { ParamIndex = i; SizeBytes = size; AlignBytes = 8 }
                        else None
                    | _ -> None)
                |> List.choose id

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
                                let! (ops, v) = pUnwrapOptionArgForFFI ssa ty platformWordTy ssas (11 + 11*i)
                                let! restResult = fold (i + 1) rest
                                return (ops, v) :: restResult
                            | TMemRef _ | TMemRefStatic _ | TStruct _ ->
                                // Memref/struct → extract pointer (index) → cast to PlatformWordType.
                                // TStruct (record types) recalled from accumulator also need pointer
                                // extraction at FFI boundaries, same as memrefs.
                                let extractSlot = ssas.[11 + 11*i]
                                let castSlot = ssas.[11 + 11*i + 1]
                                let! extractOp = pExtractBasePtr extractSlot ssa ty
                                let! castOp = pIndexCastS castSlot extractSlot TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([extractOp; castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | TIndex ->
                                // Index value → cast to PlatformWordType at boundary
                                let castSlot = ssas.[11 + 11*i + 1]
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
                let vals = marshaledArgs |> List.map snd
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
                let optionSize = match optionType with TMemRefStatic (n, _) -> n | _ -> 9
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
                let! payInsertOps = pTypedInsertView allocaSSA rawRetSSA 1 payOffsetSSA payViewSSA payZeroSSA cRetType optionType

                let allOps = marshalOps @ [declOp; callOp; allocaOp; nullOp; cmpOp; extOp]
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
                                let! (ops, v) = pUnwrapOptionArgForFFI ssa ty platformWordTy ssas (2 + 11*i)
                                let! restResult = fold (i + 1) rest
                                return (ops, v) :: restResult
                            | TMemRef _ | TMemRefStatic _ | TStruct _ ->
                                // Memref/struct → extract pointer (index) → cast to PlatformWordType.
                                // TStruct (record types) serialize as memref<Nxi8> in MLIR but are
                                // recalled as TStruct in the accumulator. At FFI boundaries, we extract
                                // the base pointer — on SysV x86_64, structs >16 bytes are passed by
                                // invisible reference (pointer in register), so this is ABI-correct.
                                let extractSlot = ssas.[2 + 11*i]
                                let castSlot = ssas.[2 + 11*i + 1]
                                let! extractOp = pExtractBasePtr extractSlot ssa ty
                                let! castOp = pIndexCastS castSlot extractSlot TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([extractOp; castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | TIndex ->
                                // Index value → cast to PlatformWordType at boundary
                                let castSlot = ssas.[2 + 11*i + 1]
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
                let vals = marshaledArgs |> List.map snd
                let cArgTypes = vals |> List.map (fun v -> v.Type)

                // Map return type from Clef NativeType to MLIR, then marshal at boundary
                let! internalRetType = pMapType node.Type
                let cRetType = marshalToCType platformWordTy internalRetType

                // Emit external function declaration + call with C-boundary types
                let! declOp = pFuncDeclByval ffiSymbol cArgTypes cRetType FuncVisibility.Private byvalParams
                let! callOp = pFuncCall (Some resultSSA) ffiSymbol vals cRetType

                // Return-side demarshal: if the internal type is TIndex (nativeint),
                // the call returned platformWordTy (i64). Cast back to index so
                // the rest of the middle-end stays width-abstract until LLVM lowering.
                match internalRetType with
                | TIndex ->
                    let! returnCastOp = pIndexCastS returnCastSSA resultSSA platformWordTy TIndex
                    return (marshalOps @ [declOp; callOp; returnCastOp], TRValue { SSA = returnCastSSA; Type = TIndex })
                | _ ->
                    return (marshalOps @ [declOp; callOp], TRValue { SSA = resultSSA; Type = cRetType })
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
///   [13 + 11*i ..] = per-arg FFI marshaling
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
///   [22 + 11*i ..] = per-arg FFI marshaling
let pDynamicExternCallResolved : PSGParser<MLIROp list * (string * string * int) list * TransferResult> =
    parser {
        let! (_funcId, argIds) = pApplication
        let! node = getCurrentNode
        let! state = getUserState

        let (NodeId nodeIdInt) = node.Id
        match Map.tryFind nodeIdInt state.Platform.Bindings with
        | Some { Resolved = ExternCall (library, symbol) } when library <> "c" ->
            let platformWordTy = state.Platform.PlatformWordType
            let! ssas = getNodeSSAs node.Id
            let! argPairs = recallArgs argIds

            let isOptionArgument (argId: NodeId) =
                match Map.tryFind argId state.Graph.Nodes with
                | Some argNode ->
                    match argNode.Type with
                    | NativeType.TApp(tycon, _) when tycon.Name = "option" || tycon.Name = "voption" -> true
                    | _ -> false
                | None -> false
            let argWithIds = List.zip argIds argPairs

            let byvalParams =
                argWithIds
                |> List.mapi (fun i (_argId, (_ssa, ty)) ->
                    match ty with
                    | TStruct _ when not (isOptionArgument (fst (argWithIds.[i]))) ->
                        let size = mlirTypeSize ty
                        if size > 16 then Some { ParamIndex = i; SizeBytes = size; AlignBytes = 8 }
                        else None
                    | _ -> None)
                |> List.choose id

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
                                let! (ops, v) = pUnwrapOptionArgForFFI ssa ty platformWordTy ssas (22 + 11*i)
                                let! restResult = fold (i + 1) rest
                                return (ops, v) :: restResult
                            | TMemRef _ | TMemRefStatic _ | TStruct _ ->
                                let extractSlot = ssas.[22 + 11*i]
                                let castSlot = ssas.[22 + 11*i + 1]
                                let! extractOp = pExtractBasePtr extractSlot ssa ty
                                let! castOp = pIndexCastS castSlot extractSlot TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([extractOp; castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | TIndex ->
                                let castSlot = ssas.[22 + 11*i + 1]
                                let! castOp = pIndexCastS castSlot ssa TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | _ ->
                                let! restResult = fold (i + 1) rest
                                return ([], { SSA = ssa; Type = ty }) :: restResult
                    }
                    fold 0 argWithIds
                let marshalOps = marshaledArgs |> List.collect fst
                let vals = marshaledArgs |> List.map snd
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
                let optionSize = match optionType with TMemRefStatic (n, _) -> n | _ -> 9
                let! allocaOp = pAllocStatic allocaSSA optionSize optionElemTy None
                let! nullOp = pConstI nullConstSSA 0L cRetType
                let! cmpOp = pCmpI cmpSSA ICmpPred.Ne rawRetSSA nullConstSSA cRetType
                let! extOp = pExtUI tagExtSSA cmpSSA (TInt (IntWidth 1)) (TInt (IntWidth 8))
                let! tagInsertOps = pTypedInsert allocaSSA tagExtSSA 0 tagViewSSA tagZeroSSA (TInt (IntWidth 8)) optionType
                let! payInsertOps = pTypedInsertView allocaSSA rawRetSSA 1 payOffsetSSA payViewSSA payZeroSSA cRetType optionType

                let allOps = preambleOps @ [indexToFuncOp] @ marshalOps @ [callOp; allocaOp; nullOp; cmpOp; extOp]
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
                                let! (ops, v) = pUnwrapOptionArgForFFI ssa ty platformWordTy ssas (13 + 11*i)
                                let! restResult = fold (i + 1) rest
                                return (ops, v) :: restResult
                            | TMemRef _ | TMemRefStatic _ | TStruct _ ->
                                let extractSlot = ssas.[13 + 11*i]
                                let castSlot = ssas.[13 + 11*i + 1]
                                let! extractOp = pExtractBasePtr extractSlot ssa ty
                                let! castOp = pIndexCastS castSlot extractSlot TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([extractOp; castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | TIndex ->
                                let castSlot = ssas.[13 + 11*i + 1]
                                let! castOp = pIndexCastS castSlot ssa TIndex platformWordTy
                                let! restResult = fold (i + 1) rest
                                return ([castOp], { SSA = castSlot; Type = platformWordTy }) :: restResult
                            | _ ->
                                let! restResult = fold (i + 1) rest
                                return ([], { SSA = ssa; Type = ty }) :: restResult
                    }
                    fold 0 argWithIds
                let marshalOps = marshaledArgs |> List.collect fst
                let vals = marshaledArgs |> List.map snd
                let cArgTypes = vals |> List.map (fun v -> v.Type)

                let! internalRetType = pMapType node.Type
                let cRetType = marshalToCType platformWordTy internalRetType

                // IndexToFunc: cast index → typed function pointer for call_indirect
                let funcTyArgs = cArgTypes
                let indexToFuncOp = MLIROp.FuncOp (FuncOp.IndexToFunc (funcPtrSSA, rawPtrIdxSSA, funcTyArgs, cRetType))

                // call_indirect through resolved function pointer
                let! callOp = pFuncCallIndirect (Some resultSSA) funcPtrSSA vals cRetType

                // Return-side demarshal (same as static path)
                match internalRetType with
                | TIndex ->
                    let! returnCastOp = pIndexCastS returnCastSSA resultSSA platformWordTy TIndex
                    return (preambleOps @ [indexToFuncOp] @ marshalOps @ [callOp; returnCastOp], pendingGlobals, TRValue { SSA = returnCastSSA; Type = TIndex })
                | _ ->
                    return (preambleOps @ [indexToFuncOp] @ marshalOps @ [callOp], pendingGlobals, TRValue { SSA = resultSSA; Type = cRetType })

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
        let! (_, fdSSA, _) = pRecallArgWithLoad argIds.[0]
        let! (_, bufferSSA, bufferType) = pRecallArgWithLoad argIds.[1]
        return! pSysWrite node.Id fdSSA bufferSSA bufferType
    }

/// Sys.read intrinsic — read from file descriptor into buffer
let pSysReadIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Sys
        do! ensure (info.Operation = "read") "Not Sys.read"
        do! ensure (argIds.Length >= 2) "Sys.read: Expected 2 args"
        let! node = getCurrentNode
        let! (_, fdSSA, _) = pRecallArgWithLoad argIds.[0]
        let! (_, bufferSSA, bufferType) = pRecallArgWithLoad argIds.[1]
        return! pSysRead node.Id fdSSA bufferSSA bufferType
    }

/// Sys.readline intrinsic — read line from fd, return trimmed string
let pSysReadlineIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Sys
        do! ensure (info.Operation = "readline") "Not Sys.readline"
        do! ensure (argIds.Length >= 1) "Sys.readline: Expected 1 arg"
        let! node = getCurrentNode
        let! (_, fdSSA, _) = pRecallArgWithLoad argIds.[0]
        return! pSysReadline node.Id fdSSA
    }
