/// AIE Pipeline - Composes MLIR-AIE lowering into a BackEnd value
///
/// This is the AIE backend: MLIR-AIE → native target tools → xclbin + insts.bin.
/// Assembled as a function value, consumed by the orchestrator without dispatch.
module BackEnd.AIE.Pipeline

open System.IO
open Core.Types.Pipeline
open Core.Timing

/// The AIE backend: MLIR-AIE text → xclbin + NPU instructions
let private implementation : BackEnd = {
    Name = "AIE"
    Compile = fun witnessed ctx ->
        match KernelRealization.realize ctx witnessed with
        | Error reason -> Error reason
        | Ok (realized,plan) ->
        let mlirText = realized.Text
        // Write the target realization for the native AIE tools.
        let mlirPath =
            match ctx.IntermediatesDir with
            | Some dir -> Path.Combine(dir, "output.mlir")
            | None -> Core.Utilities.IntermediateWriter.scratchPath "output_aie.mlir"
        File.WriteAllText(mlirPath, mlirText)

        if ctx.EmitIntermediateOnly then
            printfn "Stopped after MLIR-AIE generation (--emit-mlir)"
            Ok (IntermediateOnly "MLIR-AIE")
        elif not witnessed.WritableStorage.IsEmpty then
            Error "AIE writable program storage requires a target allocation commitment for the source inventory"
        else
            // Derive output paths from the target output path
            // ctx.OutputPath is the project output (e.g., targets/HelloNappyKernel)
            let outputDir = Path.GetDirectoryName(ctx.OutputPath)
            let baseName = Path.GetFileNameWithoutExtension(ctx.OutputPath)
            let xclbinPath = Path.Combine(outputDir, baseName + ".xclbin")
            let instsPath = Path.Combine(outputDir, baseName + "_insts.bin")

            timePhase ctx.Timing "BackEnd.AIECompile" "Compiling MLIR-AIE to xclbin" (fun () ->
                Lowering.lowerToXclbin plan.Target.Device (plan.Tiles |> List.map (fun tile -> tile.Column,tile.ComputeRow)) plan.Tiles.Length mlirPath xclbinPath instsPath)
            |> Result.map (fun () -> Xclbin (xclbinPath, instsPath))
}

/// Current source/witness ownership is validated before target realization.
let backend: BackEnd =
    { implementation with Compile = WitnessedInput.compileWithBoundary KernelRealization.validate implementation.Compile }
