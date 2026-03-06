/// LLVM Lowering - MLIR dialect lowering to LLVM dialect
///
/// This module handles MLIR-to-LLVM IR conversion:
/// - Dialect lowering via mlir-opt (vector, scf, cf, func, arith → llvm)
/// - flat-closure-lowering plugin resolves closure cast patterns
/// - mlir-translate to LLVM IR
///
/// When Composer becomes self-hosted, this module gets replaced with
/// native MLIR dialect lowering.
module BackEnd.LLVM.Lowering

open System.IO

/// Resolve the path to the flat-closure-lowering MLIR pass plugin.
/// Checks FIDELITY_MLIR_PLUGINS environment variable first, then falls back
/// to the standard install location.
let internal closurePluginPath =
    let envPath = System.Environment.GetEnvironmentVariable("FIDELITY_MLIR_PLUGINS")
    let pluginDir =
        if not (System.String.IsNullOrEmpty(envPath)) then envPath
        else
            // Standard location: ~/repos/mlir-plugins/build/flat-closure-lowering
            let home = System.Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile)
            Path.Combine(home, "repos", "mlir-plugins", "build", "flat-closure-lowering")
    Path.Combine(pluginDir, "flat-closure-lowering.so")

/// Lower MLIR to LLVM IR using mlir-opt and mlir-translate
let lowerToLLVM (mlirPath: string) (llvmPath: string) : Result<unit, string> =
    try
        // Step 1: mlir-opt to convert to LLVM dialect
        // Uses --pass-pipeline syntax (required for dynamically loaded pass plugins).
        //
        // Pipeline order:
        //   memref preparation → vector → scf → cf → index → func → arith →
        //   resolve-closure-casts → reconcile-unrealized-casts → canonicalize
        //
        // resolve-closure-casts (flat-closure-lowering plugin) resolves three cast patterns:
        //   - !llvm.ptr → i64  (FuncToIndex: function pointer stored as index)
        //   - i64 → !llvm.ptr  (IndexToFunc: index recovered as function pointer)
        //   - i64 → memref     (IndexToMemRef: pointer to captured memref data)
        // These arise from the flat closure representation (memref<Nxindex> pairs).
        // Must run AFTER convert-func-to-llvm (which creates the !llvm.ptr casts)
        // and BEFORE reconcile-unrealized-casts (which cannot resolve cross-domain casts).
        let pluginArg =
            if File.Exists(closurePluginPath) then
                sprintf "--load-pass-plugin=\"%s\" " closurePluginPath
            else ""
        let pipeline = "builtin.module(expand-strided-metadata,memref-expand,finalize-memref-to-llvm,convert-vector-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,convert-index-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,resolve-closure-casts,reconcile-unrealized-casts,canonicalize)"
        let mlirOptArgs =
            if File.Exists(closurePluginPath) then
                sprintf "%s--pass-pipeline=\"%s\" %s" pluginArg pipeline mlirPath
            else
                // Fallback: no plugin available, use individual flags (will fail on closure code)
                sprintf "%s --expand-strided-metadata --memref-expand --finalize-memref-to-llvm --convert-vector-to-llvm --convert-scf-to-cf --convert-cf-to-llvm --convert-index-to-llvm --convert-func-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts --canonicalize" mlirPath
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
