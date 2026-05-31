/// AIE Pipeline - Composes MLIR-AIE lowering into a BackEnd value
///
/// This is the AIE backend: MLIR-AIE → aiecc.py → xclbin + insts.bin.
/// Assembled as a function value, consumed by the orchestrator without dispatch.
module BackEnd.AIE.Pipeline

open System.IO
open Core.Types.Pipeline
open Core.Timing

/// The AIE backend: MLIR-AIE text → xclbin + NPU instructions
let backend : BackEnd = {
    Name = "AIE"
    Compile = fun mlirText ctx ->
        // Write MLIR-AIE to file for aiecc.py input
        let mlirPath =
            match ctx.IntermediatesDir with
            | Some dir -> Path.Combine(dir, "output.mlir")
            | None -> Path.Combine(Path.GetTempPath(), "output_aie.mlir")
        File.WriteAllText(mlirPath, mlirText)

        if ctx.EmitIntermediateOnly then
            printfn "Stopped after MLIR-AIE generation (--emit-mlir)"
            Ok (IntermediateOnly "MLIR-AIE")
        else
            // Derive output paths from the target output path
            // ctx.OutputPath is the project output (e.g., targets/HelloNappyKernel)
            let outputDir = Path.GetDirectoryName(ctx.OutputPath)
            let baseName = Path.GetFileNameWithoutExtension(ctx.OutputPath)
            let xclbinPath = Path.Combine(outputDir, baseName + ".xclbin")
            let instsPath = Path.Combine(outputDir, baseName + "_insts.bin")

            timePhase "BackEnd.AIECompile" "Compiling MLIR-AIE to xclbin" (fun () ->
                Lowering.lowerToXclbin mlirPath xclbinPath instsPath)
            |> Result.map (fun () -> Xclbin (xclbinPath, instsPath))
}
