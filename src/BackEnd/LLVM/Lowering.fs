/// LLVM Lowering - MLIR dialect lowering to LLVM dialect
///
/// This module handles MLIR-to-LLVM IR conversion:
/// - Dialect lowering via mlir-opt (vector, scf, cf, func, arith → llvm)
/// - reconcile-ffi-externs plugin resolves FFI extern symbol collisions
/// - flat-closure-lowering plugin resolves closure cast patterns
/// - mlir-translate to LLVM IR
///
/// When Composer becomes self-hosted, this module gets replaced with
/// native MLIR dialect lowering.
module BackEnd.LLVM.Lowering

open System.IO

/// Resolve the path to an MLIR pass plugin.
/// Checks FIDELITY_MLIR_PLUGINS environment variable first, then falls back
/// to the standard build location under ~/repos/mlir-plugins/build/<name>/.
let private resolvePluginPath (name: string) =
    let envPath = System.Environment.GetEnvironmentVariable("FIDELITY_MLIR_PLUGINS")
    let pluginDir =
        if not (System.String.IsNullOrEmpty(envPath)) then envPath
        else
            let home = System.Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile)
            Path.Combine(home, "repos", "mlir-plugins", "build", name)
    Path.Combine(pluginDir, name + ".so")

let internal closurePluginPath = resolvePluginPath "flat-closure-lowering"
let internal ffiPluginPath = resolvePluginPath "reconcile-ffi-externs"

/// Lower MLIR to LLVM IR using mlir-opt and mlir-translate
let lowerToLLVM (mlirPath: string) (llvmPath: string) : Result<unit, string> =
    try
        // Step 1: mlir-opt to convert to LLVM dialect
        // Uses --pass-pipeline syntax (required for dynamically loaded pass plugins).
        //
        // Pipeline order:
        //   memref preparation → vector → scf → cf → index → func → arith →
        //   reconcile-ffi-externs → resolve-closure-casts →
        //   reconcile-unrealized-casts → canonicalize
        //
        // reconcile-ffi-externs strips the "ffi." prefix from FFI extern
        // declarations, reconciling any type differences with MLIR infrastructure
        // declarations (e.g., @malloc from finalize-memref-to-llvm) via
        // explicit LLVM casts (ptrtoint/inttoptr).
        //
        // resolve-closure-casts resolves three cast patterns from flat closures:
        //   - !llvm.ptr → i64  (FuncToIndex: function pointer stored as index)
        //   - i64 → !llvm.ptr  (IndexToFunc: index recovered as function pointer)
        //   - i64 → memref     (IndexToMemRef: pointer to captured memref data)
        //
        // Both plugins run AFTER all standard dialect conversions and BEFORE
        // reconcile-unrealized-casts.
        let pluginArgs =
            [ closurePluginPath; ffiPluginPath ]
            |> List.filter File.Exists
            |> List.map (sprintf "--load-pass-plugin=\"%s\"")
            |> String.concat " "
        let hasPlugins = pluginArgs.Length > 0
        let pipeline = "builtin.module(expand-strided-metadata,memref-expand,finalize-memref-to-llvm,convert-vector-to-llvm,convert-scf-to-cf,convert-cf-to-llvm,convert-index-to-llvm,convert-func-to-llvm,convert-arith-to-llvm,reconcile-ffi-externs,resolve-closure-casts,reconcile-unrealized-casts,canonicalize)"
        let mlirOptArgs =
            if hasPlugins then
                sprintf "%s --pass-pipeline=\"%s\" %s" pluginArgs pipeline mlirPath
            else
                // Fallback: no plugins available, use individual flags (will fail on closure/FFI code)
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
