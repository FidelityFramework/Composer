/// LLVM Pipeline - Composes MLIR lowering + native codegen into a BackEnd value
///
/// This is the LLVM backend: MLIR → mlir-opt → mlir-translate → opt (target bitcode) → ld.lld → native binary.
/// Assembled as a function value, consumed by the orchestrator without dispatch.
module BackEnd.LLVM.Pipeline

open System.IO
open Core.Types.Pipeline
open Core.Timing
open Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig

/// The LLVM backend: witnessed portable module → target realization → native binary
let backend : BackEnd = {
    Name = "LLVM"
    Compile = fun witnessed ctx ->
        // Write MLIR to temp file for mlir-opt input
        let mlirPath =
            match ctx.IntermediatesDir with
            | Some dir -> Path.Combine(dir, artifactFilename ArtifactId.Mlir)
            | None -> Core.Utilities.IntermediateWriter.scratchPath "output.mlir"
        let targetTriple = ctx.TargetTripleOverride |> Option.defaultValue (Codegen.getDefaultTarget())
        File.WriteAllText(mlirPath, witnessed.Text)

        // Phase 1: Lower MLIR → LLVM IR (mlir-opt + mlir-translate)
        let llPath =
            match ctx.IntermediatesDir with
            | Some dir -> Path.Combine(dir, artifactFilename ArtifactId.Llvm)
            | None -> Core.Utilities.IntermediateWriter.scratchPath "output.ll"

        RequirementRealization.realize (RequirementRealization.selectRuntime ctx targetTriple) witnessed
        |> Result.bind (fun realized ->
            // Keep the portable artifact intact. Runtime realization is a
            // separate backend input, selected by the declared process ABI.
            let inputPath =
                if System.Object.ReferenceEquals(realized, witnessed) then mlirPath
                else
                    let path = Path.ChangeExtension(mlirPath, ".runtime.mlir")
                    File.WriteAllText(path, realized.Text)
                    path
            timePhase "BackEnd.MLIRLower" "Lowering MLIR to LLVM IR" (fun () ->
                Lowering.lowerToLLVM inputPath llPath targetTriple ctx.TargetPointerBits))
        |> Result.bind (fun () ->
            if ctx.EmitIntermediateOnly then
                printfn "Stopped after LLVM IR generation (--emit-llvm)"
                Ok (IntermediateOnly "LLVM IR")
            else
                // Phase 2: LLVM IR → native binary (target bitcode + LLD)
                timePhase "BackEnd.Link" "Linking to native binary" (fun () ->
                    Codegen.compileToNativeWithStorage llPath ctx.OutputPath targetTriple ctx.DeploymentMode ctx.ExternLibraries ctx.NativeLink ctx.TargetCpu witnessed.WritableStorage)
                |> Result.map (fun () -> NativeBinary ctx.OutputPath))
}
