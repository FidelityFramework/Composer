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
open Alex.Elements.LLVMElements
open Alex.Elements.ArithElements

// ═══════════════════════════════════════════════════════════
// PLATFORM I/O SYSCALLS
// ═══════════════════════════════════════════════════════════

/// Build Sys.write syscall pattern
/// syscall(1, fd, buffer_ptr, count) -> bytes_written
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes written)
/// - fdSSA: File descriptor SSA (typically constant 1 for stdout)
/// - bufferSSA: Pointer to buffer SSA
/// - countSSA: Number of bytes to write SSA
let pSysWrite (resultSSA: SSA) (fdSSA: SSA) (bufferSSA: SSA) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Emit llvm.call to write syscall
        // write(fd, buffer, count) returns bytes written
        let! writeCall = pCall resultSSA "write" [fdSSA; bufferSSA; countSSA]

        return ([writeCall], TRValue { SSA = resultSSA; Type = TInt I64 })
    }

/// Build Sys.read syscall pattern
/// syscall(0, fd, buffer_ptr, count) -> bytes_read
///
/// Parameters:
/// - resultSSA: SSA value for result (bytes read)
/// - fdSSA: File descriptor SSA (typically constant 0 for stdin)
/// - bufferSSA: Pointer to buffer SSA
/// - countSSA: Maximum bytes to read SSA
let pSysRead (resultSSA: SSA) (fdSSA: SSA) (bufferSSA: SSA) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Emit llvm.call to read syscall
        // read(fd, buffer, count) returns bytes read
        let! readCall = pCall resultSSA "read" [fdSSA; bufferSSA; countSSA]

        return ([readCall], TRValue { SSA = resultSSA; Type = TInt I64 })
    }
