/// SeqElements - Atomic CIRCT sequential logic operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides sequential logic (seq dialect) for FPGA targets — clocked registers.
/// Stub: full integration requires Design<State, Report> handling.
module internal Alex.Elements.SeqElements

open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════
// CLOCKED REGISTERS
// ═══════════════════════════════════════════════════════════

/// Emit seq.compreg (clocked register — flip-flop)
/// result: register output, input: next-state value, clk: clock signal
/// resetVal: optional reset value (for power-on initialization)
let pSeqCompreg (ssa: SSA) (input: SSA) (clk: SSA) (resetVal: SSA option) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.SeqOp (SeqOp.SeqCompreg (ssa, input, clk, resetVal, ty))
    }
