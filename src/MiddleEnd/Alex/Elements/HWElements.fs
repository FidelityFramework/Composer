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

/// hw.struct_create — create a struct from field values
let internal pHWStructCreate (result: SSA) (fieldVals: (SSA * MLIRType) list) (structTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.HWOp (HWOp.HWStructCreate (result, fieldVals, structTy))
    }

/// hw.struct_extract — extract a named field from a struct
let internal pHWStructExtract (result: SSA) (input: SSA) (fieldName: string) (structTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.HWOp (HWOp.HWStructExtract (result, input, fieldName, structTy))
    }

/// hw.struct_inject — produce new struct with one field replaced
let internal pHWStructInject (result: SSA) (input: SSA) (fieldName: string) (newValue: SSA) (structTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.HWOp (HWOp.HWStructInject (result, input, fieldName, newValue, structTy))
    }

/// hw.instance — instantiate a sub-module
let internal pHWInstance (result: SSA) (instName: string) (moduleName: string)
    (inputs: (string * SSA * MLIRType) list) (outputs: (string * MLIRType) list) : PSGParser<MLIROp> =
    parser {
        return MLIROp.HWOp (HWOp.HWInstance (result, instName, moduleName, inputs, outputs))
    }

/// hw.aggregate_constant — zero-initialized struct constant
let internal pHWAggregateConstant (result: SSA) (structTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.HWOp (HWOp.HWAggregateConstant (result, structTy))
    }
