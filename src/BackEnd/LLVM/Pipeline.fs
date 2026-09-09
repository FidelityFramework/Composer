/// LLVM Pipeline - Composes MLIR lowering + native codegen into a BackEnd value
///
/// This is the LLVM backend: MLIR → mlir-opt → mlir-translate → opt (target bitcode) → ld.lld → native binary.
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
            | None -> Core.Utilities.IntermediateWriter.scratchPath "output.mlir"
        let targetTriple = ctx.TargetTripleOverride |> Option.defaultValue (Codegen.getDefaultTarget())
        File.WriteAllText(mlirPath, mlirText)

        // Phase 1: Lower MLIR → LLVM IR (mlir-opt + mlir-translate)
        let llPath =
            match ctx.IntermediatesDir with
            | Some dir -> Path.Combine(dir, artifactFilename ArtifactId.Llvm)
            | None -> Core.Utilities.IntermediateWriter.scratchPath "output.ll"

        timePhase "BackEnd.MLIRLower" "Lowering MLIR to LLVM IR" (fun () ->
            Lowering.lowerToLLVM mlirPath llPath targetTriple ctx.TargetPointerBits)
        |> Result.bind (fun () ->
            if ctx.EmitIntermediateOnly then
                printfn "Stopped after LLVM IR generation (--emit-llvm)"
                Ok (IntermediateOnly "LLVM IR")
            else
                // Phase 2: LLVM IR → native binary (target bitcode + LLD)
                timePhase "BackEnd.Link" "Linking to native binary" (fun () ->
                    Codegen.compileToNative llPath ctx.OutputPath targetTriple ctx.DeploymentMode ctx.ExternLibraries ctx.NativeLink ctx.TargetCpu)
                |> Result.map (fun () -> NativeBinary ctx.OutputPath))
}
