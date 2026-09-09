/// Composer CLI - Thin wrapper around CompilationOrchestrator
///
/// This is intentionally minimal. All compilation logic lives in the orchestrator.
/// The CLI only handles:
///   - Argument parsing
///   - Calling the orchestrator
///   - Returning exit codes
module CLI.Program

open System
open System.IO
open System.Reflection
open Argu
open Core.CompilationOrchestrator
open CLI.Commands.VerifyCommand
open CLI.Commands.DoctorCommand

// ═══════════════════════════════════════════════════════════════════════════
// Command Line Arguments
// ═══════════════════════════════════════════════════════════════════════════

/// Compile command arguments
type CompileArgs =
    | [<MainCommand; Unique>] Project of path: string
    | [<AltCommandLine("-o")>] Output of path: string
    | [<AltCommandLine("-t")>] Target of target: string
    | [<AltCommandLine("-k")>] Keep_Intermediates
    | [<AltCommandLine("-v")>] Verbose
    | [<AltCommandLine("-T")>] Timing
    | Emit_MLIR
    | Emit_LLVM
    | [<AltCommandLine("--warnaserror")>] Warn_As_Error
    | No_Color
    | Sysroot of path: string
    | Link_Library_Path of path: string
    | Link_Start_File of path: string
    | Link_End_File of path: string
    | Dynamic_Linker of path: string
    | Linker_Script of path: string

    interface IArgParserTemplate with
        member this.Usage =
            match this with
            | Project _ -> ".fidproj file or Clef source file to compile"
            | Output _ -> "Output executable path"
            | Target _ -> "Target triple (default: host platform)"
            | Keep_Intermediates -> "Keep intermediate files (.mlir, .ll, .bc) for debugging"
            | Verbose -> "Enable verbose output"
            | Timing -> "Show timing for each compilation phase"
            | Emit_MLIR -> "Emit MLIR and stop (don't generate executable)"
            | Emit_LLVM -> "Emit LLVM IR and stop (don't generate executable)"
            | Warn_As_Error -> "Treat warnings as errors"
            | No_Color -> "Disable colored output"
            | Sysroot _ -> "Target runtime root for direct ELF linking"
            | Link_Library_Path _ -> "Target library directory (repeatable)"
            | Link_Start_File _ -> "Object before program bitcode (repeatable; replaces discovered startup files)"
            | Link_End_File _ -> "Object after program libraries (repeatable; replaces discovered end files)"
            | Dynamic_Linker _ -> "Runtime loader path recorded in the ELF (target path, not sysroot path)"
            | Linker_Script _ -> "LLD script controlling target section/segment layout"

// ═══════════════════════════════════════════════════════════════════════════
// Compile Command Handler
// ═══════════════════════════════════════════════════════════════════════════

/// Execute compile command - delegates to orchestrator
let private executeCompile (args: ParseResults<CompileArgs>) : int =
    if args.Contains(No_Color) then CLI.Output.disableColor()

    // Find project path
    let projectPath =
        match args.TryGetResult(Project) with
        | Some p -> p
        | None ->
            let fidprojs = Directory.GetFiles(".", "*.fidproj")
            if fidprojs.Length = 1 then
                fidprojs.[0]
            elif fidprojs.Length > 1 then
                printfn "Error: Multiple .fidproj files found. Please specify which one to compile."
                exit 1
            else
                printfn "Error: No .fidproj file found and no project specified."
                printfn "Usage: composer compile <project.fidproj>"
                exit 1

    // Build options and delegate to orchestrator
    let options : CompilationOptions = {
        ProjectPath = projectPath
        OutputPath = args.TryGetResult(Output)
        TargetTriple = args.TryGetResult(Target)
        NativeLink = {
            Sysroot = args.TryGetResult(Sysroot)
            LibraryPaths = args.GetResults(Link_Library_Path)
            StartFiles = args.GetResults(Link_Start_File)
            EndFiles = args.GetResults(Link_End_File)
            DynamicLinker = args.TryGetResult(Dynamic_Linker)
            LinkerScript = args.TryGetResult(Linker_Script)
        }
        KeepIntermediates = args.Contains(Keep_Intermediates)
        EmitMLIROnly = args.Contains(Emit_MLIR)
        EmitLLVMOnly = args.Contains(Emit_LLVM)
        Verbose = args.Contains(Verbose)
        ShowTiming = args.Contains(Timing)
        TreatWarningsAsErrors = args.Contains(Warn_As_Error)
    }

    // THE single entry point for compilation
    compileProject options

// ═══════════════════════════════════════════════════════════════════════════
// CLI Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Display version information
/// Version is read from assembly (set in Composer.fsproj <Version>)
let private showVersion() =
    let assembly = Assembly.GetExecutingAssembly()
    let version =
        assembly.GetCustomAttribute<AssemblyInformationalVersionAttribute>()
        |> Option.ofObj
        |> Option.map (fun a -> a.InformationalVersion)
        |> Option.defaultValue (assembly.GetName().Version.ToString())
    printfn "Composer %s - Clef to Native Compiler with Deterministic Memory Management" version
    printfn "Copyright (c) 2025-2026 SpeakEZ Technologies, Inc."
    0

/// Display usage information
let private showUsage() =
    printfn "Composer - Clef to Native Compiler"
    printfn ""
    printfn "Usage:"
    printfn "  composer compile [options]    Compile Clef to native code"
    printfn "  composer verify [options]     Verify binary meets constraints"
    printfn "  composer doctor [options]     Diagnose toolchain issues"
    printfn "  composer --version            Display version information"
    printfn ""
    printfn "Use 'composer <subcommand> --help' for more information about a subcommand."

[<EntryPoint>]
let main argv =
    let errorHandler = ProcessExiter(colorizer = function ErrorCode.HelpText -> None | _ -> Some ConsoleColor.Red)

    try
        if argv.Length = 0 then
            showUsage()
            0
        elif argv.[0] = "--version" then
            showVersion()
        elif argv.[0] = "compile" then
            let compileParser = ArgumentParser.Create<CompileArgs>(programName = "composer compile", errorHandler = errorHandler)
            let compileArgs = Array.skip 1 argv
            let compileResults = compileParser.ParseCommandLine(compileArgs)
            executeCompile compileResults
        elif argv.[0] = "verify" then
            let verifyParser = ArgumentParser.Create<VerifyArgs>(programName = "composer verify", errorHandler = errorHandler)
            let verifyArgs = Array.skip 1 argv
            let verifyResults = verifyParser.ParseCommandLine(verifyArgs)
            verify verifyResults
        elif argv.[0] = "doctor" then
            let doctorParser = ArgumentParser.Create<DoctorArgs>(programName = "composer doctor", errorHandler = errorHandler)
            let doctorArgs = Array.skip 1 argv
            let doctorResults = doctorParser.ParseCommandLine(doctorArgs)
            doctor doctorResults
        elif argv.[0] = "--help" || argv.[0] = "-h" then
            showUsage()
            0
        else
            printfn "Error: Unknown subcommand '%s'" argv.[0]
            printfn ""
            showUsage()
            1
    with
    | :? ArguParseException as ex ->
        printfn "%s" ex.Message
        1
    | ex ->
        printfn "Error: %s" ex.Message
        if argv |> Array.exists (fun arg -> arg = "-v" || arg = "--verbose") then
            eprintfn "%s" (ex.ToString())
        1
