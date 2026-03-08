/// LLVM Codegen - LLVM IR to native binary compilation
///
/// This module handles final native code generation:
/// - llc: LLVM IR → object files
/// - clang: object files → linked executable
/// - Target-specific optimizations
///
/// When Composer becomes self-hosted, this module gets replaced with
/// native LLVM code generation.
module BackEnd.LLVM.Codegen

open System.IO
open Core.Types.Dialects

/// Get default target triple based on host platform
let getDefaultTarget() =
    if System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Windows) then
        "x86_64-pc-windows-gnu"
    elif System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.Linux) then
        "x86_64-unknown-linux-gnu"
    elif System.Runtime.InteropServices.RuntimeInformation.IsOSPlatform(System.Runtime.InteropServices.OSPlatform.OSX) then
        "x86_64-apple-darwin"
    else
        "x86_64-unknown-linux-gnu"

/// Compile LLVM IR to native binary using llc and clang
let compileToNative
    (llvmPath: string)
    (outputPath: string)
    (targetTriple: string)
    (deploymentMode: DeploymentMode)
    (externLibraries: Set<string>) : Result<unit, string> =
    try
        let objPath = Path.ChangeExtension(llvmPath, ".o")

        // Step 1: llc to compile LLVM IR to object file
        // Note: -mcpu=native auto-detects host CPU features (AVX2, etc.) for SIMD optimization
        let llcArgs = sprintf "-mcpu=native -O0 -filetype=obj %s -o %s" llvmPath objPath
        let llcProcess = new System.Diagnostics.Process()
        llcProcess.StartInfo.FileName <- "llc"
        llcProcess.StartInfo.Arguments <- llcArgs
        llcProcess.StartInfo.UseShellExecute <- false
        llcProcess.StartInfo.RedirectStandardError <- true
        llcProcess.Start() |> ignore
        let llcError = llcProcess.StandardError.ReadToEnd()
        llcProcess.WaitForExit()

        if llcProcess.ExitCode <> 0 then
            Error (sprintf "llc failed: %s" llcError)
        else
            // Step 2: clang to link into executable
            // Library flags are data-driven from binding resolution (ExternLibraries)
            let libraryFlags =
                externLibraries
                |> Set.toList
                |> List.map (sprintf "-l%s")
                |> String.concat " "

            let clangArgs =
                match deploymentMode with
                | Console ->
                    // Use -no-pie to avoid relocation issues with LLVM-generated code
                    // -rdynamic exports all symbols to dynamic symbol table so that
                    // dlsym(RTLD_DEFAULT, "symbol") can find functions in the binary itself
                    // (required for Fidelity callback resolution via string-literal dlsym)
                    // Console mode always links libc; additional libraries come from ExternLibraries
                    let libs = if externLibraries.IsEmpty then "-lc" else "-lc " + libraryFlags
                    sprintf "-O0 -no-pie -rdynamic %s -o %s %s" objPath outputPath libs
                | Freestanding | Embedded ->
                    // Use _start as entry point - it handles argc/argv and calls exit syscall
                    sprintf "-O0 %s -o %s -nostdlib -static -ffreestanding -Wl,-e,_start" objPath outputPath
                | Library ->
                    let libs = if externLibraries.IsEmpty then "" else " " + libraryFlags
                    sprintf "-O0 -shared %s -o %s%s" objPath outputPath libs

            let clangProcess = new System.Diagnostics.Process()
            clangProcess.StartInfo.FileName <- "clang"
            clangProcess.StartInfo.Arguments <- clangArgs
            clangProcess.StartInfo.UseShellExecute <- false
            clangProcess.StartInfo.RedirectStandardError <- true
            clangProcess.Start() |> ignore
            let clangError = clangProcess.StandardError.ReadToEnd()
            clangProcess.WaitForExit()

            if clangProcess.ExitCode <> 0 then
                Error (sprintf "clang failed: %s" clangError)
            else
                // Clean up object file
                if File.Exists(objPath) then
                    File.Delete(objPath)
                Ok ()
    with ex ->
        Error (sprintf "Native compilation failed: %s" ex.Message)
