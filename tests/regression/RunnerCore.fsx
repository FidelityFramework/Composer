// Testable implementation for Runner.fsx.
#r "../../src/bin/Debug/net10.0/XParsec.dll"
#r "../../src/bin/Debug/net10.0/Fidelity.Data.dll"

open System
open System.IO
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open Fidelity.Data.TOML

// =============================================================================
// Types
// =============================================================================

type ProcessResult =
    | Completed of exitCode: int * stdout: string * stderr: string
    | Timeout of timeoutMs: int
    | Failed of exn: Exception

type SampleDef = {
    Name: string
    ProjectFile: string
    BinaryName: string
    StdinFile: string option
    ExpectedOutput: string
    TimeoutSeconds: int
    Skip: bool
    SkipReason: string option
}

type CompileResult =
    | CompileSuccess of durationMs: int64
    | CompileFailed of exitCode: int * stdout: string * stderr: string * durationMs: int64
    | CompileTimeout of timeoutMs: int
    | CompileSkipped of reason: string

type RunResult =
    | RunSuccess of durationMs: int64
    | RunFailed of exitCode: int * stdout: string * stderr: string * durationMs: int64
    | OutputMismatch of expected: string * actual: string * durationMs: int64
    | RunTimeout of timeoutMs: int
    | RunSkipped of reason: string

type TestResult = { Sample: SampleDef; CompileResult: CompileResult; RunResult: RunResult option }
type TestConfig = { SamplesRoot: string; CompilerPath: string; DefaultTimeoutSeconds: int }
type TestReport = { RunId: string; ManifestPath: string; CompilerPath: string; StartTime: DateTime; EndTime: DateTime; Results: TestResult list }

type CliOptions = { ManifestPath: string; TargetSamples: string list; Verbose: bool; TimeoutOverride: int option }

// =============================================================================
// Process Runner
// =============================================================================

// Adapted from tests/Infrastructure/Process.fs: direct argument arrays,
// concurrent stream reads and process-tree termination. This runner also needs
// separate streams, a working directory and paced input for console samples.
let runProcess cmd (args: string list) workDir (stdin: string option) (timeoutMs: int) : ProcessResult * int64 =
    let sw = Stopwatch.StartNew()
    try
        use proc = new Process()
        use cancellation = new CancellationTokenSource()
        proc.StartInfo <- ProcessStartInfo(
            FileName = cmd,
            WorkingDirectory = workDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            RedirectStandardInput = Option.isSome stdin,
            UseShellExecute = false,
            CreateNoWindow = true)
        for argument in args do proc.StartInfo.ArgumentList.Add argument

        if not (proc.Start()) then failwith ("Cannot start " + cmd)

        let stdout = proc.StandardOutput.ReadToEndAsync(cancellation.Token)
        let stderr = proc.StandardError.ReadToEndAsync(cancellation.Token)
        let writeInput = task {
            // Preserve line pacing for the existing interactive samples. Input,
            // stream drainage and process exit all share the same deadline.
            match stdin with
            | Some input ->
                let lines = input.TrimEnd([|'\n'; '\r'|]).Split('\n')
                for i in 0 .. lines.Length - 1 do
                    do! proc.StandardInput.WriteLineAsync(lines.[i].TrimEnd('\r').AsMemory(), cancellation.Token)
                    do! proc.StandardInput.FlushAsync(cancellation.Token)
                    if i < lines.Length - 1 then
                        do! Task.Delay(50, cancellation.Token)
                proc.StandardInput.Close()
            | None -> ()
        }
        let finished = Task.WhenAll [| stdout :> Task; stderr :> Task; writeInput :> Task; proc.WaitForExitAsync(cancellation.Token) |]
        try
            finished.WaitAsync(TimeSpan.FromMilliseconds(float timeoutMs)).GetAwaiter().GetResult()
            sw.Stop()
            (Completed(proc.ExitCode, stdout.Result, stderr.Result), sw.ElapsedMilliseconds)
        with :? TimeoutException ->
            // Kill descendants too: they can otherwise retain redirected pipes
            // or survive a timed-out compiler invocation.
            if not proc.HasExited then proc.Kill(true)
            cancellation.Cancel()
            proc.WaitForExit(5000) |> ignore
            sw.Stop()
            (Timeout timeoutMs, sw.ElapsedMilliseconds)
    with ex ->
        sw.Stop()
        (Failed ex, sw.ElapsedMilliseconds)

let compileSample compilerPath projectDir projectFile binaryName timeoutMs =
    let outputPath = Path.Combine(projectDir, binaryName)
    let (result, ms) = runProcess compilerPath ["compile"; projectFile; "-o"; outputPath; "-k"; "--no-color"] projectDir None timeoutMs
    match result with
    | Completed (0, _, _) -> CompileSuccess ms
    | Completed (code, stdout, stderr) -> CompileFailed (code, stdout, stderr, ms)
    | Timeout t -> CompileTimeout t
    | Failed ex -> CompileFailed (-1, "", ex.Message, ms)

let runBinary binaryPath workDir stdin timeoutMs =
    runProcess binaryPath [] workDir stdin timeoutMs

// =============================================================================
// Output Normalization and Comparison
// =============================================================================

let normalizeOutput (s: string) =
    s.Replace("\r\n", "\n").TrimEnd([|'\n'; '\r'; ' '|])

let createDiffSummary (expected: string) (actual: string) maxLines =
    let expectedLines = expected.Split('\n')
    let actualLines = actual.Split('\n')
    let mutable diffLine = -1
    let mutable i = 0
    while i < min expectedLines.Length actualLines.Length && diffLine < 0 do
        if expectedLines.[i] <> actualLines.[i] then diffLine <- i
        i <- i + 1
    if diffLine < 0 && expectedLines.Length <> actualLines.Length then
        diffLine <- min expectedLines.Length actualLines.Length
    if diffLine >= 0 then
        let exp = if diffLine < expectedLines.Length then expectedLines.[diffLine] else "(missing)"
        let act = if diffLine < actualLines.Length then actualLines.[diffLine] else "(missing)"
        sprintf "First diff at line %d:\n    Expected: %s\n    Actual:   %s" (diffLine + 1) exp act
    else
        "Outputs differ but no specific line difference found"

// =============================================================================
// Manifest Loading
// =============================================================================

let loadManifest manifestPath =
    let toml =
        match File.ReadAllText manifestPath |> Fidelity.Data.TOML.Toml.parse with
        | Ok doc -> doc
        | Error e -> failwith $"Failed to parse manifest: {e}"
    let manifestDir = Path.GetDirectoryName(manifestPath)

    let getString key table =
        match Map.tryFind key table with
        | Some (Fidelity.Data.TOML.TomlValue.String s) -> s
        | _ -> ""
    let getInt key def table =
        match Map.tryFind key table with
        | Some (Fidelity.Data.TOML.TomlValue.Integer i) -> int i
        | _ -> def
    let getBool key def table =
        match Map.tryFind key table with
        | Some (Fidelity.Data.TOML.TomlValue.Boolean b) -> b
        | _ -> def

    let config =
        match Map.tryFind "config" toml with
        | Some (Fidelity.Data.TOML.TomlValue.Table t) ->
            { SamplesRoot = Path.GetFullPath(Path.Combine(manifestDir, getString "samples_root" t))
              CompilerPath = Path.GetFullPath(Path.Combine(manifestDir, getString "compiler" t))
              DefaultTimeoutSeconds = getInt "default_timeout_seconds" 30 t }
        | _ -> failwith "Missing [config] section"

    let samples =
        match Map.tryFind "samples" toml with
        | Some (Fidelity.Data.TOML.TomlValue.Array items) ->
            items |> List.choose (function
                | Fidelity.Data.TOML.TomlValue.Table t ->
                    Some {
                        Name = getString "name" t
                        ProjectFile = getString "project" t
                        BinaryName = getString "binary" t
                        StdinFile = match getString "stdin_file" t with "" -> None | s -> Some s
                        ExpectedOutput = getString "expected_output" t
                        TimeoutSeconds = getInt "timeout_seconds" config.DefaultTimeoutSeconds t
                        Skip = getBool "skip" false t
                        SkipReason = match getString "skip_reason" t with "" -> None | s -> Some s
                    }
                | _ -> None)
        | _ -> []

    (config, samples)

let getStdinContent config sample =
    match sample.StdinFile with
    | None -> None
    | Some file ->
        let path = Path.Combine(config.SamplesRoot, sample.Name, file)
        if File.Exists path then Some (File.ReadAllText path) else None

// =============================================================================
// Reporting
// =============================================================================

let generateReport (report: TestReport) verbose =
    printfn "\n=== Composer Regression Test ==="
    printfn "Run ID: %s" report.RunId
    printfn "Manifest: %s" report.ManifestPath
    printfn "Compiler: %s" report.CompilerPath

    printfn "\n=== Compilation Phase ==="
    for r in report.Results do
        match r.CompileResult with
        | CompileSuccess ms -> printfn "[PASS] %s (%.2fs)" r.Sample.Name (float ms / 1000.0)
        | CompileFailed (code, stdout, stderr, ms) ->
            printfn "[FAIL] %s (%.2fs)" r.Sample.Name (float ms / 1000.0)
            if verbose then
                let lastStdout = if stdout.Length > 500 then "..." + stdout.Substring(stdout.Length - 500) else stdout
                printfn "  Exit code: %d\n  Stderr: %s\n  Stdout (tail): %s" code (if stderr.Length > 500 then stderr.Substring(0,500) + "..." else stderr) lastStdout
        | CompileTimeout t -> printfn "[TIMEOUT] %s (%dms)" r.Sample.Name t
        | CompileSkipped reason -> printfn "[SKIP] %s (%s)" r.Sample.Name reason

    printfn "\n=== Execution Phase ==="
    for r in report.Results do
        match r.RunResult with
        | Some (RunSuccess ms) -> printfn "[PASS] %s (%dms)" r.Sample.Name ms
        | Some (RunFailed (code, _, stderr, ms)) ->
            printfn "[FAIL] %s (exit %d, %dms)" r.Sample.Name code ms
            if verbose && stderr <> "" then printfn "  Stderr: %s" stderr
        | Some (OutputMismatch (exp, act, ms)) ->
            printfn "[MISMATCH] %s (%dms)" r.Sample.Name ms
            if verbose then printfn "  %s" (createDiffSummary exp act 5)
        | Some (RunTimeout t) -> printfn "[TIMEOUT] %s (%dms)" r.Sample.Name t
        | Some (RunSkipped reason) -> printfn "[SKIP] %s (%s)" r.Sample.Name reason
        | None -> ()

    let compilePass = report.Results |> List.filter (fun r -> match r.CompileResult with CompileSuccess _ -> true | _ -> false) |> List.length
    let compileFail = report.Results |> List.filter (fun r -> match r.CompileResult with CompileFailed _ | CompileTimeout _ -> true | _ -> false) |> List.length
    let compileSkip = report.Results |> List.filter (fun r -> match r.CompileResult with CompileSkipped _ -> true | _ -> false) |> List.length
    let runPass = report.Results |> List.choose (fun r -> r.RunResult) |> List.filter (function RunSuccess _ -> true | _ -> false) |> List.length
    let runFail = report.Results |> List.choose (fun r -> r.RunResult) |> List.filter (function RunFailed _ | OutputMismatch _ | RunTimeout _ -> true | _ -> false) |> List.length
    let runSkip = report.Results |> List.choose (fun r -> r.RunResult) |> List.filter (function RunSkipped _ -> true | _ -> false) |> List.length

    printfn "\n=== Summary ==="
    printfn "Started: %s" (report.StartTime.ToString("s"))
    printfn "Completed: %s" (report.EndTime.ToString("s"))
    printfn "Duration: %.1fs" (report.EndTime - report.StartTime).TotalSeconds
    printfn "Compilation: %d/%d passed, %d failed, %d skipped" compilePass (List.length report.Results) compileFail compileSkip
    printfn "Execution: %d/%d passed, %d failed, %d skipped" runPass (runPass + runFail) runFail runSkip
    printfn "Status: %s" (if compileFail = 0 && runFail = 0 then "PASSED" else "FAILED")

let didPass report =
    let compileFail = report.Results |> List.exists (fun r -> match r.CompileResult with CompileFailed _ | CompileTimeout _ -> true | _ -> false)
    let runFail = report.Results |> List.exists (fun r -> match r.RunResult with Some (RunFailed _ | OutputMismatch _ | RunTimeout _) -> true | _ -> false)
    not compileFail && not runFail

// =============================================================================
// Test Execution - Three Phase: Compile All, Run All, Collect Results
// =============================================================================

/// Phase 1: Compile a single sample (returns compile result + sample info for phase 2)
let compileSamplePhase config sample =
    if sample.Skip then
        (sample, CompileSkipped (sample.SkipReason |> Option.defaultValue "marked skip"), None)
    else
        let sampleDir = Path.Combine(config.SamplesRoot, sample.Name)
        let timeoutMs = sample.TimeoutSeconds * 1000
        let compileResult = compileSample config.CompilerPath sampleDir sample.ProjectFile sample.BinaryName timeoutMs
        let binaryPath =
            match compileResult with
            | CompileSuccess _ -> Some (Path.Combine(sampleDir, sample.BinaryName))
            | _ -> None
        (sample, compileResult, binaryPath)

/// Phase 2: Run a single binary (takes compiled sample info)
let runBinaryPhase config (sample, compileResult, binaryPathOpt) =
    match compileResult, binaryPathOpt with
    | CompileSuccess compileMs, Some binaryPath ->
        if not (File.Exists binaryPath) then
            { Sample = sample; CompileResult = CompileFailed (-1, "", $"Binary not found: {binaryPath}", compileMs); RunResult = None }
        else
            let sampleDir = Path.Combine(config.SamplesRoot, sample.Name)
            let timeoutMs = sample.TimeoutSeconds * 1000
            let stdin = getStdinContent config sample
            let (outcome, runMs) = runBinary binaryPath sampleDir stdin timeoutMs
            match outcome with
            | Timeout duration ->
                { Sample = sample; CompileResult = compileResult; RunResult = Some (RunTimeout duration) }
            | Failed ex ->
                { Sample = sample; CompileResult = compileResult; RunResult = Some (RunFailed (-1, "", ex.Message, runMs)) }
            | Completed (code, output, stderr) when code <> 0 ->
                { Sample = sample; CompileResult = compileResult; RunResult = Some (RunFailed (code, output, stderr, runMs)) }
            | Completed (_, output, _) ->
                let normalizedOutput = normalizeOutput output
                let normalizedExpected = normalizeOutput sample.ExpectedOutput
                if normalizedOutput = normalizedExpected then
                    { Sample = sample; CompileResult = compileResult; RunResult = Some (RunSuccess runMs) }
                else
                    { Sample = sample; CompileResult = compileResult; RunResult = Some (OutputMismatch (normalizedExpected, normalizedOutput, runMs)) }
    | CompileSkipped reason, _ ->
        { Sample = sample; CompileResult = compileResult; RunResult = Some (RunSkipped "compile skipped") }
    | _, _ ->
        { Sample = sample; CompileResult = compileResult; RunResult = None }

/// Run all tests with strict phase separation:
/// Phase 1: Compile ALL samples sequentially (with progress)
/// Phase 2: Run ALL binaries sequentially (with progress)
let runAllTests config samples verbose : TestResult list =
    let total = List.length samples

    printfn "--- Compile Phase (%d samples) ---" total
    let compileResults =
        samples |> List.mapi (fun i s ->
            if verbose then printf "  [%d/%d] Compiling %s... " (i+1) total s.Name; Console.Out.Flush()
            let result = compileSamplePhase config s
            let (_, cr, _) = result
            match cr with
            | CompileSuccess ms ->
                if verbose then printfn "OK (%.2fs)" (float ms / 1000.0)
                else printf "."
            | CompileFailed (code, _, stderr, ms) ->
                if verbose then printfn "FAIL (exit %d, %.2fs)" code (float ms / 1000.0)
                else printf "X"
                if verbose && stderr.Length > 0 then
                    let truncated = if stderr.Length > 300 then stderr.Substring(0, 300) + "..." else stderr
                    printfn "    stderr: %s" truncated
            | CompileTimeout t ->
                if verbose then printfn "TIMEOUT (%dms)" t
                else printf "T"
            | CompileSkipped reason ->
                if verbose then printfn "SKIP (%s)" reason
                else printf "S"
            result)
    if not verbose then printfn ""

    printfn "\n--- Run Phase ---"
    let results =
        compileResults |> List.mapi (fun i r ->
            let (sample, _, _) = r
            if verbose then printf "  [%d/%d] Running %s... " (i+1) total sample.Name; Console.Out.Flush()
            let result = runBinaryPhase config r
            match result.RunResult with
            | Some (RunSuccess ms) ->
                if verbose then printfn "PASS (%dms)" ms
                else printf "."
            | Some (RunFailed (code, _, _, ms)) ->
                if verbose then printfn "FAIL (exit %d, %dms)" code ms
                else printf "X"
            | Some (OutputMismatch (exp, act, ms)) ->
                if verbose then
                    printfn "MISMATCH (%dms)" ms
                    printfn "    %s" (createDiffSummary exp act 5)
                else printf "M"
            | Some (RunTimeout t) ->
                if verbose then printfn "TIMEOUT (%dms)" t
                else printf "T"
            | Some (RunSkipped reason) ->
                if verbose then printfn "SKIP (%s)" reason
                else printf "S"
            | None ->
                if verbose then printfn "(no binary)"
                else printf "-"
            result)
    if not verbose then printfn ""
    results

let buildCompiler compilerDir =
    let (result, ms) = runProcess "dotnet" ["build"] compilerDir None 120000
    match result with
    | Completed (0, _, _) -> Some ms
    | Completed (code, _, stderr) ->
        printfn "Build failed (exit %d): %s" code (if stderr.Length > 500 then stderr.Substring(0, 500) else stderr)
        None
    | Timeout _ ->
        printfn "Build timed out"
        None
    | Failed ex ->
        printfn "Build exception: %s" ex.Message
        None

// =============================================================================
// CLI and Main
// =============================================================================

let defaultOptions = { ManifestPath = Path.Combine(__SOURCE_DIRECTORY__, "Manifest.toml"); TargetSamples = []; Verbose = false; TimeoutOverride = None }

/// Every requested substring must select at least one oracle. A misspelled
/// sample must not turn an empty run, or a partially selected run, green.
let selectSamples targets (samples: SampleDef list) =
    let unmatched = targets |> List.distinct |> List.filter (fun target ->
        samples |> List.exists (fun sample -> sample.Name.Contains(target: string)) |> not)
    if not unmatched.IsEmpty then
        Error ("No manifest sample matches: " + String.concat ", " unmatched)
    elif samples.IsEmpty then
        Error "The manifest contains no samples."
    elif targets.IsEmpty then Ok samples
    else Ok (samples |> List.filter (fun sample -> targets |> List.exists sample.Name.Contains))

let rec parseArgs args opts =
    match args with
    | [] -> opts
    | "--sample" :: name :: rest -> parseArgs rest { opts with TargetSamples = name :: opts.TargetSamples }
    | "--verbose" :: rest -> parseArgs rest { opts with Verbose = true }
    | "--timeout" :: sec :: rest -> parseArgs rest { opts with TimeoutOverride = Some (int sec) }
    | "--" :: rest -> parseArgs rest opts
    | "--help" :: _ ->
        printfn "Usage: dotnet fsi Runner.fsx [options]"
        printfn "  --sample NAME    Run specific sample(s)"
        printfn "  --verbose        Show detailed output"
        printfn "  --timeout SEC    Override timeout for all samples"
        printfn "  --help           Show this help"
        exit 0
    | _ :: rest -> parseArgs rest opts

let main argv =
    let opts = parseArgs (Array.toList argv) defaultOptions
    printfn "=== Composer Regression Test Runner ===\n"
    if not (File.Exists opts.ManifestPath) then
        printfn "ERROR: Manifest not found at %s" opts.ManifestPath
        1
    else
        let (config, allSamples) = loadManifest opts.ManifestPath
        let samples = match opts.TimeoutOverride with Some t -> allSamples |> List.map (fun s -> { s with TimeoutSeconds = t }) | None -> allSamples
        match selectSamples opts.TargetSamples samples with
        | Error message ->
            eprintfn "ERROR: %s" message
            1
        | Ok samplesToRun ->
            printfn "Manifest: %s" opts.ManifestPath
            printfn "Compiler: %s" config.CompilerPath
            printfn "Samples: %d\n" (List.length samplesToRun)

            // CompilerPath points to bin/Debug/net10.0/Composer, we need src/ for build
            let compilerSourceDir = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(config.CompilerPath), "..", "..", ".."))
            printfn "Building compiler..."
            match buildCompiler compilerSourceDir with
            | None -> 1
            | Some ms ->
                printfn "Built in %dms\n" ms
                printfn "Running %d tests...\n" (List.length samplesToRun)

                let startTime = DateTime.Now
                let results = runAllTests config samplesToRun opts.Verbose
                let endTime = DateTime.Now

                let report = {
                    RunId = startTime.ToString("s")
                    ManifestPath = opts.ManifestPath
                    CompilerPath = config.CompilerPath
                    StartTime = startTime
                    EndTime = endTime
                    Results = results
                }

                generateReport report opts.Verbose
                if didPass report then 0 else 1
