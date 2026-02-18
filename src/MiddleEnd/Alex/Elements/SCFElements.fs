/// SCFElements - Atomic Structured Control Flow dialect operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides SCF dialect operations via XParsec state threading.
module internal Alex.Elements.SCFElements

open XParsec
open XParsec.Parsers     // getUserState
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

// All Elements use XParsec state for platform/type context

/// Emit SCF If operation (structured control flow)
/// resultOpt: Some (resultSSA, resultType) for expression-valued if, None for void
let pSCFIf (cond: SSA) (thenOps: MLIROp list) (elseOps: MLIROp list option) (resultOpt: (SSA * MLIRType) option) : PSGParser<MLIROp> =
    parser {
        return MLIROp.SCFOp (SCFOp.If (cond, thenOps, elseOps, resultOpt))
    }

/// Emit SCF While operation
let pSCFWhile (condOps: MLIROp list) (bodyOps: MLIROp list) : PSGParser<MLIROp> =
    parser {
        return MLIROp.SCFOp (SCFOp.While (condOps, bodyOps))
    }

/// Emit SCF For operation
let pSCFFor (lower: SSA) (upper: SSA) (step: SSA) (bodyOps: MLIROp list) : PSGParser<MLIROp> =
    parser {
        return MLIROp.SCFOp (SCFOp.For (lower, upper, step, bodyOps))
    }

/// Emit SCF Yield (return value from SCF region)
/// For expression-valued scf.if: pass typed values. For void: pass empty list.
let pSCFYield (values: (SSA * MLIRType) list) : PSGParser<MLIROp> =
    parser {
        return MLIROp.SCFOp (SCFOp.Yield values)
    }

/// Emit SCF Condition (condition check in while loop)
let pSCFCondition (cond: SSA) (args: SSA list) : PSGParser<MLIROp> =
    parser {
        return MLIROp.SCFOp (SCFOp.Condition (cond, args))
    }
