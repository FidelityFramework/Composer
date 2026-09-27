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
let private implementation : BackEnd = {
    Name = "LLVM"
    Compile = fun witnessed ctx ->
        // The triple is the selected platform's declared core triple (or an
        // explicit --target). It is never the build host's triple.
        match ctx.TargetTripleOverride with
        | None ->
            Error "PSG settlement (platform resolution) did not settle the target triple for the LLVM backend: the selected platform declares no core triple and no --target was given"
        | Some targetTriple ->
        // Write MLIR to temp file for mlir-opt input
        let mlirPath =
            match ctx.IntermediatesDir with
            | Some dir -> Path.Combine(dir, artifactFilename ArtifactId.Mlir)
            | None -> Core.Utilities.IntermediateWriter.scratchPath "output.mlir"
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
            timePhase ctx.Timing "BackEnd.MLIRLower" "Lowering MLIR to LLVM IR" (fun () ->
                Lowering.lowerToLLVM inputPath llPath targetTriple ctx.TargetPointerBits))
        |> Result.bind (fun () ->
            if ctx.EmitIntermediateOnly then
                printfn "Stopped after LLVM IR generation (--emit-llvm)"
                Ok (IntermediateOnly "LLVM IR")
            else
                // Phase 2: LLVM IR → native binary (target bitcode + LLD)
                timePhase ctx.Timing "BackEnd.Link" "Linking to native binary" (fun () ->
                    Codegen.compileToNativeWithStorage llPath ctx.OutputPath targetTriple ctx.DeploymentMode ctx.ExternLibraries ctx.NativeLink ctx.TargetCpu witnessed.WritableStorage)
                |> Result.map (fun () -> NativeBinary ctx.OutputPath))
}

/// Current source/witness ownership is validated before target realization.
let backend: BackEnd =
    { implementation with Compile = WitnessedInput.compileWithBoundary BoundaryAdmission.validate implementation.Compile }
