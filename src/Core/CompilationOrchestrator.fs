/// CompilationOrchestrator - Top-level compiler pipeline coordination
///
/// Orchestrates the full compilation pipeline:
/// FrontEnd (FNCS) → MiddleEnd (Alex) → BackEnd (resolved from target platform)
///
/// The orchestrator is backend-agnostic. It resolves the backend once at
/// pipeline assembly time and delegates to it. No dispatch, no branching
/// on target type.
module Core.CompilationOrchestrator

open System.IO
open System.Reflection
open Clef.Compiler.Project

open Core.Timing
open Core.CompilerConfig
open Core.Types.Pipeline
open Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

type CompilationOptions = {
    ProjectPath: string
    OutputPath: string option
    TargetTriple: string option
    KeepIntermediates: bool
    EmitMLIROnly: bool
    EmitLLVMOnly: bool
    Verbose: bool
    ShowTiming: bool
    TreatWarningsAsErrors: bool
}

type CompilationContext = {
    ProjectName: string
    BuildDir: string
    IntermediatesDir: string option
    OutputPath: string
    TargetPlatform: Core.Types.Dialects.TargetPlatform
    DeploymentMode: Core.Types.Dialects.DeploymentMode
}

// ═══════════════════════════════════════════════════════════════════════════
// Phase 1: FrontEnd - Compile F# → PSG
// ═══════════════════════════════════════════════════════════════════════════

let private runFrontEnd (projectPath: string) : Result<ProjectCheckResult, string> =
    timePhase "FrontEnd" "F# → PSG (Type Checking & Semantic Graph)" (fun () ->
        FrontEnd.ProjectLoader.load projectPath)

/// Diagnostic gate — emits all diagnostics with colored formatting, short-circuits on errors.
/// Tiered model: unreachable diagnostics are demoted to info by effective severity.
/// When warnaserror is set, warnings (from reachable code) promote to errors.
let private requireCleanDiagnostics (warnaserror: bool) (project: ProjectCheckResult) : Result<ProjectCheckResult, string> =
    let projectDir = Some project.Options.ProjectDirectory
    let diagnostics = project.CheckResult.Diagnostics

    // Emit all diagnostics with colored formatting (warnaserror elevates warnings to errors)
    let (errors, warnings, infos) = CLI.Output.emitAllDiagnostics warnaserror projectDir diagnostics
    CLI.Output.emitSummary errors warnings infos

    // Short-circuit on errors (includes elevated warnings when warnaserror is set)
    if errors > 0 then
        Error (sprintf "Compilation failed with %d error(s)" errors)
    else
        Ok project

// ═══════════════════════════════════════════════════════════════════════════
// Phase 2: MiddleEnd (Alex + PSGElaboration)
// ═══════════════════════════════════════════════════════════════════════════

let private runMiddleEnd (project: ProjectCheckResult) (ctx: CompilationContext) : Result<string, string> =
    timePhase "MiddleEnd" "MLIR Generation" (fun () ->
        // Get platform context from FNCS
        match Core.FNCS.Integration.platformContext project.CheckResult with
        | None -> Error "No platform context available from FNCS"
        | Some platformCtx ->
            // Delegate to MiddleEnd - it orchestrates PSGElaboration + Alex
            MiddleEnd.MLIRGeneration.generate
                project.CheckResult.Graph
                platformCtx
                ctx.DeploymentMode
                ctx.TargetPlatform
                ctx.IntermediatesDir)

// ═══════════════════════════════════════════════════════════════════════════
// Context Setup
// ═══════════════════════════════════════════════════════════════════════════

let private setupContext (options: CompilationOptions) (project: ProjectCheckResult) : CompilationContext =
    let config = project.Options
    let buildDir = Path.Combine(config.ProjectDirectory, "targets")
    Directory.CreateDirectory(buildDir) |> ignore

    let intermediatesDir =
        if options.KeepIntermediates || options.EmitMLIROnly || options.EmitLLVMOnly then
            let dir = Path.Combine(buildDir, "intermediates")
            Directory.CreateDirectory(dir) |> ignore
            enableAllPhases dir
            Some dir
        else
            None

    let targetPlatform =
        match config.TargetPlatform with
        | TargetPlatform.CPU -> Core.Types.Dialects.TargetPlatform.CPU
        | TargetPlatform.FPGA -> Core.Types.Dialects.TargetPlatform.FPGA
        | TargetPlatform.GPU -> Core.Types.Dialects.TargetPlatform.GPU
        | TargetPlatform.MCU -> Core.Types.Dialects.TargetPlatform.MCU
        | TargetPlatform.NPU -> Core.Types.Dialects.TargetPlatform.NPU

    let deploymentMode =
        match config.DeploymentMode with
        | DeploymentMode.Freestanding -> Core.Types.Dialects.DeploymentMode.Freestanding
        | DeploymentMode.Console -> Core.Types.Dialects.DeploymentMode.Console
        | DeploymentMode.Library -> Core.Types.Dialects.DeploymentMode.Library
        | DeploymentMode.Embedded -> Core.Types.Dialects.DeploymentMode.Embedded

    {
        ProjectName = config.Name
        BuildDir = buildDir
        IntermediatesDir = intermediatesDir
        OutputPath = options.OutputPath |> Option.defaultValue (Path.Combine(buildDir, config.OutputName |> Option.defaultValue config.Name))
        TargetPlatform = targetPlatform
        DeploymentMode = deploymentMode
    }

// ═══════════════════════════════════════════════════════════════════════════
// Main Pipeline
// ═══════════════════════════════════════════════════════════════════════════

let compileProject (options: CompilationOptions) : int =
    // Setup
    setEnabled options.ShowTiming
    if options.Verbose then
        enableVerboseMode()
        enableVerbose()

    let version = Assembly.GetExecutingAssembly().GetCustomAttribute<AssemblyInformationalVersionAttribute>()
                  |> Option.ofObj
                  |> Option.map (fun a -> a.InformationalVersion)
                  |> Option.defaultValue "dev"

    printfn "Composer Compiler v%s" version
    printfn "======================"
    printfn ""

    // Setup intermediates directory BEFORE loading project (enables FNCS phase emission)
    let needsIntermediates = options.KeepIntermediates || options.EmitMLIROnly || options.EmitLLVMOnly
    if needsIntermediates then
        let projectDir = Path.GetDirectoryName(options.ProjectPath)
        let intermediatesDir = Path.Combine(projectDir, "targets", "intermediates")
        Directory.CreateDirectory(intermediatesDir) |> ignore
        enableAllPhases intermediatesDir

    // Run pipeline: FrontEnd → MiddleEnd → BackEnd
    let result =
        // Phase 1: FrontEnd - Compile F# to PSG
        runFrontEnd options.ProjectPath
        |> Result.bind (requireCleanDiagnostics options.TreatWarningsAsErrors)
        |> Result.bind (fun project ->
            let ctx = setupContext options project

            // Resolve backend from target platform (assembly time — once, not dispatch)
            let backEnd = PlatformPipeline.resolveBackEnd ctx.TargetPlatform

            printfn "Project:  %s" ctx.ProjectName
            printfn "Platform: %A" ctx.TargetPlatform
            printfn "Backend:  %s" backEnd.Name
            printfn "Output:   %s" ctx.OutputPath
            printfn ""

            // Phase 2: MiddleEnd - Generate MLIR from PSG (target-agnostic)
            runMiddleEnd project ctx
            |> Result.bind (fun mlirText ->
                // Write MLIR to intermediates (if enabled)
                if ctx.IntermediatesDir.IsSome then
                    let mlirPath = Path.Combine(ctx.IntermediatesDir.Value, artifactFilename ArtifactId.Mlir)
                    File.WriteAllText(mlirPath, mlirText)

                if options.EmitMLIROnly then
                    printfn "Stopped after MLIR generation (--emit-mlir)"
                    Ok ()
                else
                    // Phase 3+: BackEnd — the backend function runs its own pipeline
                    let backEndCtx = {
                        OutputPath = ctx.OutputPath
                        IntermediatesDir = ctx.IntermediatesDir
                        TargetTripleOverride = options.TargetTriple
                        DeploymentMode = ctx.DeploymentMode
                        EmitIntermediateOnly = options.EmitLLVMOnly
                    }
                    backEnd.Compile mlirText backEndCtx
                    |> Result.bind (fun artifact ->
                        printfn ""
                        match artifact with
                        | NativeBinary path ->
                            printfn "Compilation successful: %s" path
                            Ok ()
                        | Verilog path ->
                            printfn "Verilog generated: %s" path
                            // Copy XDC constraints alongside .sv, then verify consistency
                            match ctx.IntermediatesDir with
                            | Some dir ->
                                let xdcSrc = Path.Combine(dir, "constraints.xdc")
                                if File.Exists xdcSrc then
                                    let xdcDst = Path.ChangeExtension(path, ".xdc")
                                    File.Copy(xdcSrc, xdcDst, true)
                                    printfn "XDC constraints: %s" xdcDst
                                    // Closed-loop: verify HDL ports match constraint ports
                                    match BackEnd.ArtifactVerification.verifyArtifacts path xdcDst with
                                    | Ok summary ->
                                        printfn "%s" summary
                                        Ok ()
                                    | Error diag ->
                                        Error diag
                                else
                                    Ok ()
                            | None -> Ok ()
                        | IntermediateOnly fmt ->
                            printfn "Produced %s intermediate" fmt
                            Ok ())))

    printSummary()
    match result with
    | Ok () -> 0
    | Error msg ->
        printfn "Error: %s" msg
        1
