/// CIRCT Pipeline - hw/comb/seq MLIR → SystemVerilog
///
/// Realize the source hardware contract, then optimize hw/comb/seq and
/// lower through CIRCT to SystemVerilog.
///
/// LLVM is never involved in this path.
module BackEnd.CIRCT.Pipeline

open System.IO
open Core.Types.Pipeline
open Core.Timing

/// The CIRCT backend: hw/comb/seq MLIR → SystemVerilog
let private implementation : BackEnd = {
    Name = "CIRCT"
    Compile = fun witnessed ctx ->
        HardwareRealization.realize witnessed |> Result.bind (fun realized ->
        let mlirText = realized.Text
        let intermediateFile name =
            match ctx.IntermediatesDir with
            | Some dir -> Path.Combine(dir, name)
            | None -> Core.Utilities.IntermediateWriter.scratchPath name

        // Write MLIR to file for tool input
        let mlirPath = intermediateFile "output.mlir"
        File.WriteAllText(mlirPath, mlirText)

        // Step 1: Optimize hw/comb/seq (canonicalize + CSE)
        let optimizedPath = intermediateFile "output.opt.mlir"
        timePhase ctx.Timing "BackEnd.CIRCTOptimize" "Optimizing hardware MLIR" (fun () ->
            Lowering.optimizeHW mlirPath optimizedPath)
        |> Result.bind (fun () ->
            if ctx.EmitIntermediateOnly then
                printfn "Stopped after CIRCT optimization"
                Ok (IntermediateOnly "CIRCT hw/comb/seq optimized")
            elif not witnessed.WritableStorage.IsEmpty then
                Error "CIRCT writable program storage requires a target allocation commitment for the source inventory"
            else
                // Step 2: Export to SystemVerilog
                let svPath =
                    match ctx.IntermediatesDir with
                    | Some dir -> Path.Combine(dir, "output.sv")
                    | None -> Path.ChangeExtension(ctx.OutputPath, ".sv")

                timePhase ctx.Timing "BackEnd.VerilogExport" "Exporting SystemVerilog" (fun () ->
                    Lowering.exportToVerilog optimizedPath svPath)
                |> Result.map (fun () -> Verilog svPath)))
}

/// Current source/witness ownership is validated before target realization.
let backend: BackEnd =
    { implementation with Compile = WitnessedInput.compileWithBoundary HardwareRealization.validate implementation.Compile }
