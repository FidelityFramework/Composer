/// CIRCT Pipeline - MLIR → hw/comb/seq → Verilog → synthesis
///
/// The CIRCT backend for FPGA targets. Converts target-agnostic MLIR
/// (func/arith/scf/cf) to hardware dialects (hw/comb/seq) via circt-opt,
/// then exports to Verilog for FPGA synthesis.
///
/// LLVM is never involved in this path.
module BackEnd.CIRCT.Pipeline

open Core.Types.Pipeline

/// The CIRCT backend: MLIR text → Verilog (stub)
let backend : BackEnd = {
    Name = "CIRCT"
    Compile = fun _mlirText _ctx ->
        Error "FPGA backend (CIRCT) not yet implemented. \
               The MLIR output has been generated successfully. \
               Next step: install circt-opt and implement lowering passes \
               (func/arith/scf → hw/comb/seq → ExportVerilog)."
}
