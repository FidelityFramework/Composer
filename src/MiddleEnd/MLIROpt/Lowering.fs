/// MLIROpt Lowering - MLIR dialect lowering to LLVM dialect
///
/// This module handles MLIR-to-LLIR conversion:
/// - Dialect lowering (vector, scf, cf, func, arith → llvm)
/// - MLIR validation passes
/// - mlir-translate to LLVM IR
///
/// When Composer becomes self-hosted, this module gets replaced with
/// native MLIR dialect lowering.
module MiddleEnd.MLIROpt.Lowering

open System.IO

/// Lower MLIR to LLVM IR using mlir-opt and mlir-translate
let lowerToLLVM (mlirPath: string) (llvmPath: string) : Result<unit, string> =
    try
        // Step 1: mlir-opt to convert to LLVM dialect
        // Conversion order: memref -> vector -> scf -> cf -> index -> func -> arith -> llvm, then cleanup casts
        // Note: --convert-vector-to-llvm enables SIMD code generation
        // Note: memref passes MUST come before func/arith lowering (func lowering may reference memref types)
        // Note: --convert-index-to-llvm converts portable index types to platform i32/i64 (BEFORE func/arith)
        // Note: --canonicalize after reconciliation helps clean up remaining artifacts
        let mlirOptArgs = sprintf "%s --memref-expand --finalize-memref-to-llvm --convert-vector-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-index-to-llvm --convert-func-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts --canonicalize" mlirPath
        let mlirOptProcess = new System.Diagnostics.Process()
        mlirOptProcess.StartInfo.FileName <- "mlir-opt"
        mlirOptProcess.StartInfo.Arguments <- mlirOptArgs
        mlirOptProcess.StartInfo.UseShellExecute <- false
        mlirOptProcess.StartInfo.RedirectStandardOutput <- true
        mlirOptProcess.StartInfo.RedirectStandardError <- true
        mlirOptProcess.Start() |> ignore
        let mlirOptOutput = mlirOptProcess.StandardOutput.ReadToEnd()
        let mlirOptError = mlirOptProcess.StandardError.ReadToEnd()
        mlirOptProcess.WaitForExit()

        if mlirOptProcess.ExitCode <> 0 then
            Error (sprintf "mlir-opt failed: %s" mlirOptError)
        else
            // Step 2: mlir-translate to convert LLVM dialect to LLVM IR
            let mlirTranslateProcess = new System.Diagnostics.Process()
            mlirTranslateProcess.StartInfo.FileName <- "mlir-translate"
            mlirTranslateProcess.StartInfo.Arguments <- "--mlir-to-llvmir"
            mlirTranslateProcess.StartInfo.UseShellExecute <- false
            mlirTranslateProcess.StartInfo.RedirectStandardInput <- true
            mlirTranslateProcess.StartInfo.RedirectStandardOutput <- true
            mlirTranslateProcess.StartInfo.RedirectStandardError <- true
            mlirTranslateProcess.Start() |> ignore
            mlirTranslateProcess.StandardInput.Write(mlirOptOutput)
            mlirTranslateProcess.StandardInput.Close()
            let llvmOutput = mlirTranslateProcess.StandardOutput.ReadToEnd()
            let translateError = mlirTranslateProcess.StandardError.ReadToEnd()
            mlirTranslateProcess.WaitForExit()

            if mlirTranslateProcess.ExitCode <> 0 then
                Error (sprintf "mlir-translate failed: %s" translateError)
            else
                File.WriteAllText(llvmPath, llvmOutput)
                Ok ()
    with ex ->
        Error (sprintf "MLIR lowering failed: %s" ex.Message)
