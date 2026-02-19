/// LLVM Pipeline - Composes MLIR lowering + native codegen into a BackEnd value
///
/// This is the LLVM backend: MLIR → mlir-opt → mlir-translate → llc → clang → native binary.
/// Assembled as a function value, consumed by the orchestrator without dispatch.
module BackEnd.LLVM.Pipeline

open System.IO
open Core.Types.Pipeline
open Core.Timing
open Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig

/// The LLVM backend: MLIR text → native binary
let backend : BackEnd = {
    Name = "LLVM"
    Compile = fun mlirText ctx ->
        // Write MLIR to temp file for mlir-opt input
        let mlirPath =
            match ctx.IntermediatesDir with
            | Some dir -> Path.Combine(dir, artifactFilename ArtifactId.Mlir)
            | None -> Path.Combine(Path.GetTempPath(), "output.mlir")
        File.WriteAllText(mlirPath, mlirText)

        // Phase 1: Lower MLIR → LLVM IR (mlir-opt + mlir-translate)
        let llPath =
            match ctx.IntermediatesDir with
            | Some dir -> Path.Combine(dir, artifactFilename ArtifactId.Llvm)
            | None -> Path.Combine(Path.GetTempPath(), "output.ll")

        timePhase "BackEnd.MLIRLower" "Lowering MLIR to LLVM IR" (fun () ->
            Lowering.lowerToLLVM mlirPath llPath)
        |> Result.bind (fun () ->
            if ctx.EmitIntermediateOnly then
                printfn "Stopped after LLVM IR generation (--emit-llvm)"
                Ok (IntermediateOnly "LLVM IR")
            else
                // Phase 2: LLVM IR → native binary (llc + clang)
                timePhase "BackEnd.Link" "Linking to native binary" (fun () ->
                    let targetTriple = ctx.TargetTripleOverride |> Option.defaultValue (Codegen.getDefaultTarget())
                    Codegen.compileToNative llPath ctx.OutputPath targetTriple ctx.DeploymentMode)
                |> Result.map (fun () -> NativeBinary ctx.OutputPath))
}
