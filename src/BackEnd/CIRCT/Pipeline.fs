/// CIRCT Pipeline - hw/comb/seq MLIR → SystemVerilog
///
/// The CIRCT backend for FPGA targets. Alex elides directly to hw/comb/seq
/// dialects, so the backend only needs:
///   1. circt-opt: canonicalize + CSE (optimization)
///   2. circt-opt: lower-seq-to-sv + lower-hw-to-sv + export-verilog
///
/// LLVM is never involved in this path.
module BackEnd.CIRCT.Pipeline

open System.IO
open Core.Types.Pipeline
open Core.Timing

/// The CIRCT backend: hw/comb/seq MLIR → SystemVerilog
let backend : BackEnd = {
    Name = "CIRCT"
    Compile = fun mlirText ctx ->
        let intermediateFile name =
            match ctx.IntermediatesDir with
            | Some dir -> Path.Combine(dir, name)
            | None -> Path.Combine(Path.GetTempPath(), name)

        // Write MLIR to file for tool input
        let mlirPath = intermediateFile "output.mlir"
        File.WriteAllText(mlirPath, mlirText)

        // Step 1: Optimize hw/comb/seq (canonicalize + CSE)
        let optimizedPath = intermediateFile "output.opt.mlir"
        timePhase "BackEnd.CIRCTOptimize" "Optimizing hardware MLIR" (fun () ->
            Lowering.optimizeHW mlirPath optimizedPath)
        |> Result.bind (fun () ->
            if ctx.EmitIntermediateOnly then
                printfn "Stopped after CIRCT optimization"
                Ok (IntermediateOnly "CIRCT hw/comb/seq optimized")
            else
                // Step 2: Export to SystemVerilog
                let svPath =
                    match ctx.IntermediatesDir with
                    | Some dir -> Path.Combine(dir, "output.sv")
                    | None -> Path.ChangeExtension(ctx.OutputPath, ".sv")

                timePhase "BackEnd.VerilogExport" "Exporting SystemVerilog" (fun () ->
                    Lowering.exportToVerilog optimizedPath svPath)
                |> Result.map (fun () -> Verilog svPath))
}
