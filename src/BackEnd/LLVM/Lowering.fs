/// LLVM Lowering - MLIR dialect lowering to LLVM dialect
///
/// This module handles MLIR-to-LLVM IR conversion:
/// - Dialect lowering via mlir-opt (vector, scf, cf, func, arith → llvm)
/// - mlir-translate to LLVM IR
///
/// When Composer becomes self-hosted, this module gets replaced with
/// native MLIR dialect lowering.
module BackEnd.LLVM.Lowering

open System.IO

/// Lower MLIR to LLVM IR using mlir-opt and mlir-translate
let lowerToLLVM (mlirPath: string) (llvmPath: string) (triple: string) (pointerBits: int option) : Result<unit, string> =
    try
        // Set index width BEFORE converting memrefs/functions. A late LLVM
        // target triple cannot repair already materialized i64 descriptors.
        let width = pointerBits |> Option.defaultValue 64
        let isCortexM = triple.StartsWith("thumbv8m.main-") || triple.StartsWith("thumbv7em-")
        let isXtensa = triple.StartsWith("xtensa")
        if (isCortexM || isXtensa) && width <> 32 then failwith "This MCU target requires a declared 32-bit Pointer dimension."
        let source = File.ReadAllText mlirPath
        // The data layout strings are the target's own, obtained from
        // `opt -mtriple=<triple> -passes=no-op-module` on an empty module rather
        // than hand-written; the Xtensa one was read from the 22.1.8 build.
        let attributes =
            if isCortexM then
                sprintf " attributes {llvm.target_triple = \"%s\", llvm.data_layout = \"e-m:e-p:32:32-Fi8-i64:64-v128:64:128-a:0:32-n32-S64\"} " triple
            elif isXtensa then
                sprintf " attributes {llvm.target_triple = \"%s\", llvm.data_layout = \"e-m:e-p:32:32-i8:8:32-i16:16:32-i64:64-n32\"} " triple
            else " "
        let targetedPath = Path.ChangeExtension(mlirPath, ".target.mlir")
        let brace = source.IndexOf('{')
        if brace < 0 then failwith "Expected an MLIR module."
        File.WriteAllText(targetedPath, source.Substring(0, brace).TrimEnd() + attributes + source.Substring(brace))
        let mlirPath = targetedPath
        let indexPass name = sprintf "%s{index-bitwidth=%d}" name width
        // Target realization uses stock dialect conversions. Callable and FFI
        // semantics must already be settled in the PSG and witnessed faithfully.
        let passes =
            [ "expand-strided-metadata"; "memref-expand"; indexPass "finalize-memref-to-llvm"
              "convert-vector-to-llvm"; "convert-scf-to-cf"; "convert-cf-to-llvm"
              indexPass "convert-index-to-llvm"; indexPass "convert-func-to-llvm"
              indexPass "convert-arith-to-llvm"
              "reconcile-unrealized-casts"; "canonicalize" ]
        let pipeline = "builtin.module(" + String.concat "," passes + ")"
        use mlirOptProcess = new System.Diagnostics.Process()
        mlirOptProcess.StartInfo.FileName <- "mlir-opt"
        mlirOptProcess.StartInfo.ArgumentList.Add("--pass-pipeline=" + pipeline)
        mlirOptProcess.StartInfo.ArgumentList.Add mlirPath
        mlirOptProcess.StartInfo.UseShellExecute <- false
        mlirOptProcess.StartInfo.RedirectStandardOutput <- true
        mlirOptProcess.StartInfo.RedirectStandardError <- true
        mlirOptProcess.Start() |> ignore
        // Diagnostic globals can be large. Drain both pipes concurrently so
        // an invalid input's stderr cannot block completion of stdout.
        let mlirOptOutputTask = mlirOptProcess.StandardOutput.ReadToEndAsync()
        let mlirOptErrorTask = mlirOptProcess.StandardError.ReadToEndAsync()
        mlirOptProcess.WaitForExit()
        let mlirOptOutput = mlirOptOutputTask.GetAwaiter().GetResult()
        let mlirOptError = mlirOptErrorTask.GetAwaiter().GetResult()

        if mlirOptProcess.ExitCode <> 0 then
            Error (sprintf "mlir-opt failed: %s" mlirOptError)
        else
            // Step 2: mlir-translate to convert LLVM dialect to LLVM IR
            use mlirTranslateProcess = new System.Diagnostics.Process()
            mlirTranslateProcess.StartInfo.FileName <- "mlir-translate"
            mlirTranslateProcess.StartInfo.Arguments <- "--mlir-to-llvmir"
            mlirTranslateProcess.StartInfo.UseShellExecute <- false
            mlirTranslateProcess.StartInfo.RedirectStandardInput <- true
            mlirTranslateProcess.StartInfo.RedirectStandardOutput <- true
            mlirTranslateProcess.StartInfo.RedirectStandardError <- true
            mlirTranslateProcess.Start() |> ignore
            let llvmOutputTask = mlirTranslateProcess.StandardOutput.ReadToEndAsync()
            let translateErrorTask = mlirTranslateProcess.StandardError.ReadToEndAsync()
            mlirTranslateProcess.StandardInput.Write(mlirOptOutput)
            mlirTranslateProcess.StandardInput.Close()
            mlirTranslateProcess.WaitForExit()
            let llvmOutput = llvmOutputTask.GetAwaiter().GetResult()
            let translateError = translateErrorTask.GetAwaiter().GetResult()

            if mlirTranslateProcess.ExitCode <> 0 then
                Error (sprintf "mlir-translate failed: %s" translateError)
            else
                File.WriteAllText(llvmPath, llvmOutput)
                Ok ()
    with ex ->
        Error (sprintf "MLIR lowering failed: %s" ex.Message)
