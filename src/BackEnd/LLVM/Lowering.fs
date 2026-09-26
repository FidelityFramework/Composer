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
        let passes =
            [ "expand-strided-metadata"; "memref-expand"; indexPass "finalize-memref-to-llvm"
              "convert-vector-to-llvm"; "convert-scf-to-cf"; "convert-cf-to-llvm"
              indexPass "convert-index-to-llvm"; indexPass "convert-func-to-llvm"
              indexPass "convert-arith-to-llvm" ]
            @ (if File.Exists ffiPluginPath then ["reconcile-ffi-externs"] else [])
            @ (if File.Exists closurePluginPath then ["resolve-closure-casts"] else [])
            @ ["reconcile-unrealized-casts"; "canonicalize"]
        let pipeline = "builtin.module(" + String.concat "," passes + ")"
        let mlirOptArgs = sprintf "%s --pass-pipeline=\"%s\" \"%s\"" pluginArgs pipeline mlirPath
        use mlirOptProcess = new System.Diagnostics.Process()
        mlirOptProcess.StartInfo.FileName <- "mlir-opt"
        mlirOptProcess.StartInfo.Arguments <- mlirOptArgs
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
