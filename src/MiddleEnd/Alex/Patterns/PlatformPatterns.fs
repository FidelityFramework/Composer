/// PlatformPatterns - Platform syscall operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns for platform I/O operations.
/// Patterns compose Elements into platform-specific syscall sequences.
module Alex.Patterns.PlatformPatterns

open XParsec
open XParsec.Parsers     // preturn
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.FuncElements
open Alex.Elements.ArithElements
open Alex.Elements.MemRefElements

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
/// - bufferType: Actual MLIR type of buffer (TMemRefScalar, TMemRef, or TPtr)
/// - countSSA: Number of bytes to write SSA
/// Build Sys.write syscall pattern with FFI boundary normalization
/// Normalizes memref→ptr at FFI boundaries (Patterns handle FFI concern)
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes written)
/// - fdSSA: File descriptor SSA (typically constant 1 for stdout)
/// - bufferSSA: Buffer SSA (memref or ptr depending on source)
/// - bufferType: Actual MLIR type of buffer (TMemRefScalar, TMemRef, or TPtr)
/// - countSSA: Number of bytes to write SSA
/// - conversionSSA: Optional SSA for memref→ptr conversion (witness allocates if needed)
let pSysWriteTyped (resultSSA: SSA) (fdSSA: SSA) (bufferSSA: SSA) (bufferType: MLIRType) (countSSA: SSA) (conversionSSA: SSA option) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // FFI BOUNDARY NORMALIZATION
        // Sys.write is an FFI function that expects pointer types, not memref.
        // Pattern normalizes memref→ptr at FFI boundary (no coordination needed).
        
        // Check if buffer needs conversion from memref to ptr
        let! (bufferCallSSA, bufferCallType, conversionOps) =
            match bufferType, conversionSSA with
            | (TMemRefScalar _ | TMemRef _), Some ptrSSA ->
                // Buffer is memref AND witness provided conversion SSA - insert conversion
                parser {
                    let! convOp = pExtractBasePtr ptrSSA bufferSSA bufferType
                    return (ptrSSA, TPtr, [convOp])
                }
            | _ ->
                // Buffer is already ptr or no conversion SSA provided - use as-is
                preturn (bufferSSA, bufferType, [])
        
        // Emit call with normalized types (ptr for FFI)
        let vals = [
            { SSA = fdSSA; Type = TInt I64 }
            { SSA = bufferCallSSA; Type = bufferCallType }  // Normalized to ptr if memref
            { SSA = countSSA; Type = TInt I64 }
        ]
        let! writeCall = pFuncCall (Some resultSSA) "write" vals (TInt I64)

        return (conversionOps @ [writeCall], TRValue { SSA = resultSSA; Type = TInt I64 })
    }

/// Build Sys.write syscall pattern (portable) - deprecated, use pSysWriteTyped
/// Kept for compatibility, but hardcodes TPtr which is incorrect for memref buffers
let pSysWrite (resultSSA: SSA) (fdSSA: SSA) (bufferSSA: SSA) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    pSysWriteTyped resultSSA fdSSA bufferSSA TPtr countSSA None

/// Build Sys.read syscall pattern (portable)
/// Uses func.call (portable) for direct syscall
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes read)
/// - fdSSA: File descriptor SSA (typically constant 0 for stdin)
/// - bufferSSA: Pointer to buffer SSA
/// - countSSA: Maximum bytes to read SSA
let pSysRead (resultSSA: SSA) (fdSSA: SSA) (bufferSSA: SSA) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Portable direct call to read syscall: read(fd: i64, buffer: ptr, count: i64) -> i64
        let vals = [
            { SSA = fdSSA; Type = TInt I64 }
            { SSA = bufferSSA; Type = TPtr }
            { SSA = countSSA; Type = TInt I64 }
        ]
        let! readCall = pFuncCall (Some resultSSA) "read" vals (TInt I64)

        return ([readCall], TRValue { SSA = resultSSA; Type = TInt I64 })
    }
