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

        // Syscall with extracted i64 pointer and length
        let vals = [
            { SSA = fdSSA; Type = platformWordTy }
            { SSA = buf_ptr_i64; Type = platformWordTy }  // ALWAYS i64 after extraction
            { SSA = count_i64; Type = platformWordTy }    // Length from memref.dim
        ]
        let! writeCall = pFuncCall (Some resultSSA) "write" vals platformWordTy

        // External function — emit declaration alongside call
        let! writeDecl = pFuncDecl "write" [platformWordTy; platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private

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

        // Syscall with extracted i64 pointer and capacity
        let vals = [
            { SSA = fdSSA; Type = platformWordTy }
            { SSA = buf_ptr_i64; Type = platformWordTy }  // ALWAYS i64 after extraction
            { SSA = capacity_i64; Type = platformWordTy } // Capacity from memref.dim
        ]
        let! readCall = pFuncCall (Some resultSSA) "read" vals platformWordTy

        // External function — emit declaration alongside call
        let! readDecl = pFuncDecl "read" [platformWordTy; platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private

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
        let! readCall = pFuncCall (Some bytesReadSSA) "read" readArgs platformWordTy
        let! readDecl = pFuncDecl "read" [platformWordTy; platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private

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
