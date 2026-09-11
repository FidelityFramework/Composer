/// CompilationOrchestrator - Top-level compiler pipeline coordination
///
/// Orchestrates the full compilation pipeline:
/// FrontEnd (CCS) → MiddleEnd (Alex) → BackEnd (resolved from target platform)
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
    NativeLink: Core.Types.Pipeline.NativeLinkOptions
    KeepIntermediates: bool
    EmitMLIROnly: bool
    EmitLLVMOnly: bool
    Verbose: bool
    ShowTiming: bool
    TreatWarningsAsErrors: bool
    Deploy: bool
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
// Phase 1: FrontEnd - Compile Clef → PSG
// ═══════════════════════════════════════════════════════════════════════════

let private runFrontEnd (projectPath: string) : Result<ProjectCheckResult, string> =
    timePhase "FrontEnd" "Clef → PSG (Type Checking & Semantic Graph)" (fun () ->
        FrontEnd.ProjectLoader.load projectPath)

/// Diagnostic gate — emits all diagnostics with colored formatting, short-circuits on errors.
/// Tiered model: unreachable diagnostics are demoted to info by effective severity.
/// When warnaserror is set, warnings (from reachable code) promote to errors.
let private requireCleanDiagnostics (warnaserror: bool) (project: ProjectCheckResult) : Result<ProjectCheckResult, string> =
    let projectDir = Some project.Options.ProjectDirectory

    // A file that failed to parse is a hard error: the checker produced no graph for it, so its
    // absence would otherwise surface far downstream as a witness failure with no diagnostic.
    let parseErrors = CLI.Output.emitParseErrors projectDir project.ParseErrors
    if parseErrors > 0 then
        CLI.Output.emitSummary parseErrors 0 0
        Error (sprintf "Compilation failed with %d parse error(s)" parseErrors)
    else
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

/// Explicit CPU/MCU profiles fix the backend contract before any lowering.
/// An absent triple must not turn a selected target into the build host.
let private requireCompatibleTarget (options: CompilationOptions) (project: ProjectCheckResult) =
    let graph = project.CheckResult.Graph
    let selected = graph.Platform |> Option.bind (fun p -> p.PlatformDescription)
    let requiresCore = project.Options.TargetPlatform = TargetPlatform.CPU || project.Options.TargetPlatform = TargetPlatform.MCU
    let compatibleBackend =
        match selected, project.Options.PlatformPath with
        | Some _, Some path ->
            FidprojLoader.load path
            |> Result.bind (fun platform ->
                if platform.TargetPlatform = TargetPlatform.Library || platform.TargetPlatform = project.Options.TargetPlatform then Ok ()
                else Error (sprintf "Workload backend %A disagrees with selected platform backend %A" project.Options.TargetPlatform platform.TargetPlatform))
        | _ -> Ok ()
    compatibleBackend |> Result.bind (fun () ->
    match selected, requiresCore with
    | Some export, true ->
        let core =
            Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.resolve graph
            |> Option.bind (fun p -> p.Core)
        match core with
        | None -> Error (sprintf "Selected platform '%s' requires a declared TargetCore" export)
        | Some core when System.String.IsNullOrWhiteSpace core.Triple ->
            Error (sprintf "Selected platform '%s' requires a target triple; build-host fallback is unavailable" export)
        | Some core ->
            match options.TargetTriple with
            | Some requested when requested <> core.Triple ->
                Error (sprintf "CLI target triple '%s' disagrees with selected platform triple '%s'" requested core.Triple)
            | _ -> Ok project
    | _ -> Ok project)

let private runMiddleEnd (project: ProjectCheckResult) (ctx: CompilationContext) : Result<string * Set<string>, string> =
    timePhase "MiddleEnd" "MLIR Generation" (fun () ->
        // Get platform context from CCS
        match Core.CCS.Integration.platformContext project.CheckResult with
        | None -> Error "No platform context available from CCS"
        | Some platformCtx ->
            // Delegate to MiddleEnd - it orchestrates PSGElaboration + Alex
            MiddleEnd.MLIRGeneration.generateWithLinkedLibraries
                project.CheckResult.Graph
                platformCtx
                ctx.DeploymentMode
                ctx.TargetPlatform
                ctx.IntermediatesDir
                (Set.ofList project.Options.LinkedLibraries))

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
        | TargetPlatform.CPU     -> Core.Types.Dialects.TargetPlatform.CPU
        | TargetPlatform.FPGA    -> Core.Types.Dialects.TargetPlatform.FPGA
        | TargetPlatform.GPU     -> Core.Types.Dialects.TargetPlatform.GPU
        | TargetPlatform.MCU     -> Core.Types.Dialects.TargetPlatform.MCU
        | TargetPlatform.NPU     -> Core.Types.Dialects.TargetPlatform.NPU
        | TargetPlatform.Library -> Core.Types.Dialects.TargetPlatform.Library

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
    if options.Deploy && (options.EmitMLIROnly || options.EmitLLVMOnly) then
        invalidArg "Deploy" "Deployment requires a complete build; remove intermediate-only flags"
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

    // Setup intermediates directory BEFORE loading project (enables CCS phase emission)
    let accessEvidence = Path.Combine(Path.GetDirectoryName(options.ProjectPath), "targets", "intermediates", "device-access.json")
    if File.Exists accessEvidence then File.Delete accessEvidence
    let needsIntermediates = options.KeepIntermediates || options.EmitMLIROnly || options.EmitLLVMOnly
    if needsIntermediates then
        let projectDir = Path.GetDirectoryName(options.ProjectPath)
        let intermediatesDir = Path.Combine(projectDir, "targets", "intermediates")
        Directory.CreateDirectory(intermediatesDir) |> ignore
        enableAllPhases intermediatesDir

    // Run pipeline: FrontEnd → MiddleEnd → BackEnd
    let result =
        // Phase 1: FrontEnd - Compile Clef to PSG
        runFrontEnd options.ProjectPath
        |> Result.bind (requireCleanDiagnostics options.TreatWarningsAsErrors)
        |> Result.bind (requireCompatibleTarget options)
        |> Result.bind (fun project ->
            let ctx = setupContext options project
            ctx.IntermediatesDir |> Option.iter (fun directory ->
                Core.DeviceAccessEvidence.write (Path.Combine(directory, "device-access.json")) project.CheckResult.Graph)
            if options.Deploy && ctx.TargetPlatform <> Core.Types.Dialects.TargetPlatform.MCU then
                failwith "Composer-managed deployment is currently implemented for MCU images only"

            // Resolve backend from target platform (assembly time — once, not dispatch)
            let backEnd = PlatformPipeline.resolveBackEnd ctx.TargetPlatform

            printfn "Project:  %s" ctx.ProjectName
            printfn "Platform: %A" ctx.TargetPlatform
            printfn "Backend:  %s" backEnd.Name
            printfn "Output:   %s" ctx.OutputPath
            printfn ""

            // Phase 2: MiddleEnd - Generate MLIR from PSG (target-agnostic)
            runMiddleEnd project ctx
            |> Result.bind (fun (mlirText, externLibraries) ->
                // Write MLIR to intermediates (if enabled)
                if ctx.IntermediatesDir.IsSome then
                    let mlirPath = Path.Combine(ctx.IntermediatesDir.Value, artifactFilename ArtifactId.Mlir)
                    File.WriteAllText(mlirPath, mlirText)

                if options.EmitMLIROnly then
                    printfn "Stopped after MLIR generation (--emit-mlir)"
                    Ok ()
                else
                    // Phase 3+: BackEnd — the backend function runs its own pipeline
                    let declaredCore =
                        Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.resolve project.CheckResult.Graph
                        |> Option.bind (fun p -> p.Core)
                    let backEndCtx = {
                        OutputPath = ctx.OutputPath
                        IntermediatesDir = ctx.IntermediatesDir
                        TargetTripleOverride = options.TargetTriple |> Option.orElseWith (fun () -> declaredCore |> Option.map (fun c -> c.Triple) |> Option.filter (fun t -> t <> ""))
                        TargetPointerBits = declaredCore |> Option.bind (fun c -> c.Widths |> List.tryFind (fun w -> w.Name = "Pointer") |> Option.map (fun w -> w.Bits))
                        TargetCpu = declaredCore |> Option.map (fun c -> c.CpuModel) |> Option.filter (fun t -> t <> "")
                        DeploymentMode = ctx.DeploymentMode
                        EmitIntermediateOnly = options.EmitLLVMOnly
                        ExternLibraries = externLibraries
                        NativeLink = options.NativeLink
                        EmbeddedTarget =
                            if ctx.TargetPlatform = Core.Types.Dialects.TargetPlatform.MCU && not options.EmitLLVMOnly then
                                Some (BackEnd.MCU.Target.resolve project.Options.ProjectPath project.CheckResult.Graph)
                            else None
                        Deploy = options.Deploy
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
                        | Xclbin (xclbinPath, instsPath) ->
                            printfn "Xclbin generated: %s" xclbinPath
                            printfn "NPU instructions: %s" instsPath
                            Ok ()
                        | GpuCodeObject path ->
                            printfn "GPU code object generated: %s" path
                            Ok ()
                        | IntermediateOnly fmt ->
                            printfn "Produced %s intermediate" fmt
                            Ok ())))

    printSummary()
    match result with
    | Ok () -> 0
    | Error msg ->
        printfn "Error: %s" msg
        1

/// Read the same checked project/platform declarations for device operations.
/// Deployment itself always goes through compileProject --deploy and a fresh build.
let deviceProject projectPath action seconds =
    match runFrontEnd (Path.GetFullPath projectPath) |> Result.bind (requireCleanDiagnostics false) with
    | Error e -> eprintfn "%s" e; 1
    | Ok project ->
        if project.Options.TargetPlatform <> TargetPlatform.MCU then failwith "Device commands require an MCU project"
        let target = BackEnd.MCU.Target.resolve project.Options.ProjectPath project.CheckResult.Graph
        let output = Path.Combine(project.Options.ProjectDirectory, "targets", project.Options.OutputName |> Option.defaultValue project.Options.Name)
        BackEnd.MCU.Probe.device action seconds target output
        0
