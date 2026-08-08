/// GPU Pipeline - Composes AMD GPU lowering into a BackEnd value
///
/// This is the GPU backend: portable MLIR → gpu.module → AMDGPU code object.
/// Assembled as a function value, consumed by the orchestrator without dispatch.
module BackEnd.GPU.Pipeline

open System.IO
open Core.Types.Pipeline
open Core.Timing

/// The GPU backend: portable MLIR text → .hsaco code object
let backend : BackEnd = {
    Name = "GPU"
    Compile = fun mlirText ctx ->
        // Write the middle end's portable MLIR for the lowering to consume
        let mlirPath =
            match ctx.IntermediatesDir with
            | Some dir -> Path.Combine(dir, "output.mlir")
            | None -> Path.Combine(Path.GetTempPath(), "output_gpu.mlir")
        File.WriteAllText(mlirPath, mlirText)

        if ctx.EmitIntermediateOnly then
            printfn "Stopped after MLIR generation (--emit-llvm)"
            Ok (IntermediateOnly "MLIR-GPU")
        else
            // Derive the code object path as a sibling of the requested output
            let outputDir = Path.GetDirectoryName(ctx.OutputPath)
            let baseName = Path.GetFileNameWithoutExtension(ctx.OutputPath)
            let hsacoPath = Path.Combine(outputDir, baseName + ".hsaco")
            // `output` may name a subdirectory (e.g. "gpu/Kernel"), so the
            // artifact directory is not guaranteed to exist yet.
            Directory.CreateDirectory(outputDir) |> ignore

            timePhase "BackEnd.GPUCompile" "Compiling MLIR to AMDGPU code object" (fun () ->
                Lowering.lowerToCodeObject mlirPath hsacoPath)
            |> Result.map (fun () -> GpuCodeObject hsacoPath)
}
