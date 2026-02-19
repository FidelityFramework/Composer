/// HWElements - Atomic CIRCT hardware module operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides hardware module structure (hw dialect) for FPGA targets.
module internal Alex.Elements.HWElements

open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════
// HARDWARE MODULE STRUCTURE
// ═══════════════════════════════════════════════════════════

/// Emit hw.module (hardware module definition)
/// inputs: named input ports, outputs: named output ports, body: combinational logic
let pHWModule (name: string) (inputs: (string * MLIRType) list) (outputs: (string * MLIRType) list) (bodyOps: MLIROp list) : PSGParser<MLIROp> =
    parser {
        return MLIROp.HWOp (HWOp.HWModule (name, inputs, outputs, bodyOps))
    }

/// Emit hw.output (module output terminator)
let pHWOutput (vals: (SSA * MLIRType) list) : PSGParser<MLIROp> =
    parser {
        return MLIROp.HWOp (HWOp.HWOutput vals)
    }
