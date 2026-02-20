/// SeqElements - Atomic CIRCT sequential logic operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides sequential logic (seq dialect) for FPGA targets — clocked registers.
module internal Alex.Elements.SeqElements

open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════
// CLOCKED REGISTERS
// ═══════════════════════════════════════════════════════════

/// Emit seq.compreg (clocked register — flip-flop)
/// result: register output, input: next-state value, clk: clock signal
/// reset: optional (resetSignal, resetValue) pair for power-on initialization
let pSeqCompreg (ssa: SSA) (input: SSA) (clk: SSA) (reset: (SSA * SSA) option) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.SeqOp (SeqOp.SeqCompreg (ssa, input, clk, reset, ty))
    }
