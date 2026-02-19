/// CombElements - Atomic CIRCT combinational logic operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides combinational operations (comb dialect) for FPGA targets via XParsec state threading.
/// Parallel to ArithElements — same interface, different MLIR dialect.
module internal Alex.Elements.CombElements

open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types

// All Elements take explicit operand types from the Pattern that calls them

// ═══════════════════════════════════════════════════════════
// COMBINATIONAL ARITHMETIC
// ═══════════════════════════════════════════════════════════

/// Emit comb.add (combinational addition)
let pCombAdd (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombAdd (ssa, lhs, rhs, operandTy))
    }

/// Emit comb.sub (combinational subtraction)
let pCombSub (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombSub (ssa, lhs, rhs, operandTy))
    }

/// Emit comb.mul (combinational multiplication)
let pCombMul (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombMul (ssa, lhs, rhs, operandTy))
    }

/// Emit comb.divs (signed combinational division)
let pCombDivS (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombDivS (ssa, lhs, rhs, operandTy))
    }

/// Emit comb.divu (unsigned combinational division)
let pCombDivU (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombDivU (ssa, lhs, rhs, operandTy))
    }

/// Emit comb.mods (combinational modulus)
let pCombMod (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombMod (ssa, lhs, rhs, operandTy))
    }

// ═══════════════════════════════════════════════════════════
// BITWISE OPERATIONS
// ═══════════════════════════════════════════════════════════

/// Emit comb.and (bitwise AND)
let pCombAnd (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombAnd (ssa, lhs, rhs, operandTy))
    }

/// Emit comb.or (bitwise OR)
let pCombOr (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombOr (ssa, lhs, rhs, operandTy))
    }

/// Emit comb.xor (bitwise XOR)
let pCombXor (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombXor (ssa, lhs, rhs, operandTy))
    }

/// Emit comb.shl (shift left)
let pCombShl (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombShl (ssa, lhs, rhs, operandTy))
    }

/// Emit comb.shru (logical shift right)
let pCombShrU (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombShrU (ssa, lhs, rhs, operandTy))
    }

/// Emit comb.shrs (arithmetic shift right)
let pCombShrS (ssa: SSA) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombShrS (ssa, lhs, rhs, operandTy))
    }

// ═══════════════════════════════════════════════════════════
// COMPARISON
// ═══════════════════════════════════════════════════════════

/// Emit comb.icmp (integer comparison)
let pCombICmp (ssa: SSA) (pred: ICmpPred) (lhs: SSA) (rhs: SSA) (operandTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombICmp (ssa, pred, lhs, rhs, operandTy))
    }

// ═══════════════════════════════════════════════════════════
// MULTIPLEXER
// ═══════════════════════════════════════════════════════════

/// Emit comb.mux (combinational multiplexer — if/else for hardware)
let pCombMux (ssa: SSA) (cond: SSA) (trueVal: SSA) (falseVal: SSA) (resultTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.CombOp (CombOp.CombMux (ssa, cond, trueVal, falseVal, resultTy))
    }
