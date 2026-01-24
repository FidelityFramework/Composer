#!/usr/bin/env dotnet fsi
// Runner.fsx - Firefly Regression Test Runner
// Usage: dotnet fsi Runner.fsx [options]
//
// Fixed: Proper async process handling to avoid pipe buffer deadlocks
// Fixed: Task-based parallelism with proper fan-out/fold-in

#r "/home/hhh/repos/Firefly/src/bin/Debug/net10.0/XParsec.dll"
#r "/home/hhh/repos/Fidelity.Toml/src/bin/Debug/net10.0/Fidelity.Toml.dll"

open System
open System.IO
open System.Diagnostics
open System.Text
open System.Threading
open System.Threading.Tasks
open Fidelity.Toml

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
    | CompileFailed of exitCode: int * stderr: string * durationMs: int64
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

type CliOptions = { ManifestPath: string; TargetSamples: string list; Verbose: bool; TimeoutOverride: int option; Parallel: bool }

// =============================================================================
// Process Runner - Fixed async stream reading
// =============================================================================

/// Run a process with proper async stdout/stderr reading to avoid pipe buffer deadlocks.
/// This is the correct pattern for .NET Process - read streams WHILE process runs, not after.
let runProcessAsync cmd args workDir (stdin: string option) (timeoutMs: int) : Task<ProcessResult * int64> =
    task {
        let sw = Stopwatch.StartNew()
        try
            use proc = new Process()
            proc.StartInfo <- ProcessStartInfo(
                FileName = cmd,
                Arguments = args,
                WorkingDirectory = workDir,
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                RedirectStandardInput = Option.isSome stdin,
                UseShellExecute = false,
                CreateNoWindow = true)

            proc.Start() |> ignore

            // Write stdin if provided
            match stdin with
            | Some input ->
                proc.StandardInput.Write(input)
                proc.StandardInput.Close()
            | None -> ()

            // Read stdout and stderr asynchronously WHILE process runs
            // This prevents pipe buffer deadlock
            let! stdoutTask = proc.StandardOutput.ReadToEndAsync()
            let! stderrTask = proc.StandardError.ReadToEndAsync()

            // Now wait for exit with timeout
            use cts = new CancellationTokenSource(timeoutMs)
            try
                do! proc.WaitForExitAsync(cts.Token)
                sw.Stop()
                return (Completed(proc.ExitCode, stdoutTask, stderrTask), sw.ElapsedMilliseconds)
            with :? OperationCanceledException ->
                try proc.Kill() with _ -> ()
                sw.Stop()
                return (Timeout timeoutMs, sw.ElapsedMilliseconds)
        with ex ->
            sw.Stop()
            return (Failed ex, sw.ElapsedMilliseconds)
    }

/// Synchronous wrapper for compatibility
let runWithTimeout cmd args workDir stdin timeoutMs =
    let (result, ms) = (runProcessAsync cmd args workDir stdin timeoutMs).Result
    (result, ms)

let compileSample compilerPath projectDir projectFile timeoutMs =
    let (result, ms) = runWithTimeout compilerPath $"compile {projectFile} -k" projectDir None timeoutMs
    match result with
    | Completed (0, _, _) -> CompileSuccess ms
    | Completed (code, _, stderr) -> CompileFailed (code, stderr, ms)
    | Timeout t -> CompileTimeout t
    | Failed ex -> CompileFailed (-1, ex.Message, ms)

let runBinary binaryPath workDir stdin timeoutMs =
    let (result, ms) = runWithTimeout binaryPath "" workDir stdin timeoutMs
    match result with
    | Completed (0, stdout, _) -> (RunSuccess ms, stdout)
    | Completed (code, stdout, stderr) -> (RunFailed (code, stdout, stderr, ms), stdout)
    | Timeout t -> (RunTimeout t, "")
    | Failed ex -> (RunFailed (-1, "", ex.Message, ms), "")

// =============================================================================
// Output Verifier
// =============================================================================

let normalizeOutput (s: string) =
    s.Replace("\r\n", "\n").Split('\n')
    |> Array.map (fun line -> line.TrimEnd())
    |> String.concat "\n" |> fun s -> s.TrimEnd()

let outputMatches expected actual = normalizeOutput expected = normalizeOutput actual

let createDiffSummary expected actual maxLines =
    let expLines = (normalizeOutput expected).Split('\n')
    let actLines = (normalizeOutput actual).Split('\n')
    let sb = StringBuilder()
    let rec findDiff i =
        if i >= max expLines.Length actLines.Length then None
        else
            let e = if i < expLines.Length then expLines.[i] else "<end>"
            let a = if i < actLines.Length then actLines.[i] else "<end>"
            if e <> a then Some (i + 1, e, a) else findDiff (i + 1)
    match findDiff 0 with
    | Some (n, e, a) -> sb.AppendLine($"  First diff at line {n}:").AppendLine($"    Expected: {e}").AppendLine($"    Actual:   {a}") |> ignore
    | None -> sb.AppendLine("  No line diff found") |> ignore
    sb.AppendLine().AppendLine($"  Expected (first {maxLines} lines):") |> ignore
    expLines |> Array.truncate maxLines |> Array.iter (fun l -> sb.AppendLine($"    {l}") |> ignore)
    sb.AppendLine().AppendLine($"  Actual (first {maxLines} lines):") |> ignore
    actLines |> Array.truncate maxLines |> Array.iter (fun l -> sb.AppendLine($"    {l}") |> ignore)
    sb.ToString()

// =============================================================================
// Sample Discovery (Manifest Parsing)
// =============================================================================

let loadManifest manifestPath =
    let manifestDir = Path.GetDirectoryName(Path.GetFullPath(manifestPath))
    let doc = Toml.parseOrFail (File.ReadAllText(manifestPath))
    let resolve (path: string) = if Path.IsPathRooted(path) then path else Path.GetFullPath(Path.Combine(manifestDir, path))
    let config = {
        SamplesRoot = Toml.getString "config.samples_root" doc |> Option.defaultValue "" |> resolve
        CompilerPath = Toml.getString "config.compiler" doc |> Option.defaultValue "" |> resolve
        DefaultTimeoutSeconds = Toml.getInt "config.default_timeout_seconds" doc |> Option.map int |> Option.defaultValue 30
    }
    let samples =
        match Toml.getValue "samples" doc with
        | Some (TomlValue.Array items) ->
            items |> List.choose (function
                | TomlValue.Table tbl ->
                    let str k = TomlTable.tryFind k tbl |> Option.bind (function TomlValue.String s -> Some s | _ -> None) |> Option.defaultValue ""
                    let strOpt k = TomlTable.tryFind k tbl |> Option.bind (function TomlValue.String s -> Some s | _ -> None)
                    let intVal k d = TomlTable.tryFind k tbl |> Option.bind (function TomlValue.Integer i -> Some (int i) | _ -> None) |> Option.defaultValue d
                    let boolVal k d = TomlTable.tryFind k tbl |> Option.bind (function TomlValue.Boolean b -> Some b | _ -> None) |> Option.defaultValue d
                    Some { Name = str "name"; ProjectFile = str "project"; BinaryName = str "binary"
                           StdinFile = strOpt "stdin_file"; ExpectedOutput = str "expected_output"
                           TimeoutSeconds = intVal "timeout_seconds" config.DefaultTimeoutSeconds
                           Skip = boolVal "skip" false; SkipReason = strOpt "skip_reason" }
                | _ -> None)
        | _ -> []
    (config, samples)

let getSampleDir config sample = Path.Combine(config.SamplesRoot, sample.Name)
let getBinaryPath config sample = Path.Combine(getSampleDir config sample, sample.BinaryName)
let getStdinContent config sample =
    match sample.StdinFile with
    | Some file -> let p = Path.Combine(getSampleDir config sample, file) in if File.Exists(p) then Some (File.ReadAllText(p)) else None
    | None -> None

// =============================================================================
// Report Generator
// =============================================================================

let formatDuration ms = if ms < 1000L then $"{ms}ms" else $"{float ms / 1000.0:F2}s"
let compileStatusStr = function CompileSuccess _ -> "PASS" | CompileFailed _ -> "FAIL" | CompileTimeout _ -> "TIMEOUT" | CompileSkipped _ -> "SKIP"
let runStatusStr = function RunSuccess _ -> "PASS" | RunFailed _ -> "FAIL" | OutputMismatch _ -> "MISMATCH" | RunTimeout _ -> "TIMEOUT" | RunSkipped _ -> "SKIP"

let generateReport (report: TestReport) verbose =
    let sb = StringBuilder()
    sb.AppendLine("=== Firefly Regression Test ===").AppendLine($"Run ID: {report.RunId}")
      .AppendLine($"Manifest: {report.ManifestPath}").AppendLine($"Compiler: {report.CompilerPath}").AppendLine() |> ignore
    sb.AppendLine("=== Compilation Phase ===") |> ignore
    for r in report.Results do
        let dur = match r.CompileResult with CompileSuccess ms | CompileFailed (_,_,ms) -> formatDuration ms | CompileTimeout ms -> $">{ms}ms" | CompileSkipped _ -> "-"
        let extra =
            match r.CompileResult with
            | CompileSkipped reason -> $" ({reason})"
            | CompileFailed (_, stderr, _) when verbose && stderr.Length > 0 ->
                let preview = if stderr.Length > 200 then stderr.Substring(0, 200) + "..." else stderr
                let formatted = preview.Replace("\n", "\n            ")
                $"\n    stderr: {formatted}"
            | _ -> ""
        sb.AppendLine($"[{compileStatusStr r.CompileResult}] {r.Sample.Name} ({dur}){extra}") |> ignore
    sb.AppendLine().AppendLine("=== Execution Phase ===") |> ignore
    for r in report.Results do
        match r.RunResult with
        | Some rr ->
            let dur = match rr with RunSuccess ms | RunFailed (_,_,_,ms) | OutputMismatch (_,_,ms) -> formatDuration ms | RunTimeout ms -> $">{ms}ms" | RunSkipped _ -> "-"
            let extra = match rr with RunSkipped reason -> $" ({reason})" | OutputMismatch (e,a,_) -> "\n" + createDiffSummary e a 5 | _ -> ""
            sb.AppendLine($"[{runStatusStr rr}] {r.Sample.Name} ({dur}){extra}") |> ignore
        | None -> sb.AppendLine($"[SKIP] {r.Sample.Name} (compile failed)") |> ignore
    sb.AppendLine().AppendLine("=== Summary ===") |> ignore
    let startStr = report.StartTime.ToString("yyyy-MM-ddTHH:mm:ss")
    let endStr = report.EndTime.ToString("yyyy-MM-ddTHH:mm:ss")
    sb.AppendLine($"Started: {startStr}").AppendLine($"Completed: {endStr}") |> ignore
    let dur = (report.EndTime - report.StartTime).TotalSeconds
    sb.AppendLine($"Duration: {dur:F1}s") |> ignore
    let cPass = report.Results |> List.filter (fun r -> match r.CompileResult with CompileSuccess _ -> true | _ -> false) |> List.length
    let cFail = report.Results |> List.filter (fun r -> match r.CompileResult with CompileFailed _ | CompileTimeout _ -> true | _ -> false) |> List.length
    let cSkip = report.Results |> List.filter (fun r -> match r.CompileResult with CompileSkipped _ -> true | _ -> false) |> List.length
    sb.AppendLine($"Compilation: {cPass}/{report.Results.Length} passed, {cFail} failed, {cSkip} skipped") |> ignore
    let runs = report.Results |> List.choose (fun r -> r.RunResult)
    let rPass = runs |> List.filter (function RunSuccess _ -> true | _ -> false) |> List.length
    let rFail = runs |> List.filter (function RunFailed _ | OutputMismatch _ | RunTimeout _ -> true | _ -> false) |> List.length
    let rSkip = runs |> List.filter (function RunSkipped _ -> true | _ -> false) |> List.length
    sb.AppendLine($"Execution: {rPass}/{runs.Length} passed, {rFail} failed, {rSkip} skipped") |> ignore
    let status = if cFail = 0 && rFail = 0 then "PASSED" else "FAILED"
    sb.AppendLine($"Status: {status}").ToString()

let didPass report =
    let cFail = report.Results |> List.filter (fun r -> match r.CompileResult with CompileFailed _ | CompileTimeout _ -> true | _ -> false) |> List.length
    let rFail = report.Results |> List.choose (fun r -> r.RunResult) |> List.filter (function RunFailed _ | OutputMismatch _ | RunTimeout _ -> true | _ -> false) |> List.length
    cFail = 0 && rFail = 0

// =============================================================================
// Test Execution - Task-based with proper fan-out/fold-in
// =============================================================================

/// Run a single sample test (compile + execute + verify) - completely isolated
let runSampleTestAsync config sample : Task<TestResult> =
    task {
        let sampleDir = getSampleDir config sample
        let binaryPath = getBinaryPath config sample
        let stdinContent = getStdinContent config sample
        let timeoutMs = sample.TimeoutSeconds * 1000

        if sample.Skip then
            let reason = defaultArg sample.SkipReason "No reason"
            return { Sample = sample; CompileResult = CompileSkipped reason; RunResult = Some (RunSkipped reason) }
        else
            // Compile
            let! (compileResult, compileMs) = runProcessAsync config.CompilerPath $"compile {sample.ProjectFile} -k" sampleDir None timeoutMs
            match compileResult with
            | Completed (0, _, _) ->
                // Execute
                let! (runResult, runMs) = runProcessAsync binaryPath "" sampleDir stdinContent timeoutMs
                match runResult with
                | Completed (0, stdout, _) ->
                    if outputMatches sample.ExpectedOutput stdout then
                        return { Sample = sample; CompileResult = CompileSuccess compileMs; RunResult = Some (RunSuccess runMs) }
                    else
                        return { Sample = sample; CompileResult = CompileSuccess compileMs; RunResult = Some (OutputMismatch (sample.ExpectedOutput, stdout, runMs)) }
                | Completed (code, stdout, stderr) ->
                    return { Sample = sample; CompileResult = CompileSuccess compileMs; RunResult = Some (RunFailed (code, stdout, stderr, runMs)) }
                | Timeout t ->
                    return { Sample = sample; CompileResult = CompileSuccess compileMs; RunResult = Some (RunTimeout t) }
                | Failed ex ->
                    return { Sample = sample; CompileResult = CompileSuccess compileMs; RunResult = Some (RunFailed (-1, "", ex.Message, runMs)) }
            | Completed (code, _, stderr) ->
                return { Sample = sample; CompileResult = CompileFailed (code, stderr, compileMs); RunResult = None }
            | Timeout t ->
                return { Sample = sample; CompileResult = CompileTimeout t; RunResult = None }
            | Failed ex ->
                return { Sample = sample; CompileResult = CompileFailed (-1, ex.Message, compileMs); RunResult = None }
    }

/// Fan-out: Start all tests as independent tasks
/// Fold-in: Await all results, then tabulate
let runAllTestsAsync config samples runParallel =
    task {
        if runParallel then
            // Fan-out: Create all tasks (they start immediately)
            let tasks = samples |> List.map (fun s -> runSampleTestAsync config s)
            // Fold-in: Wait for all to complete
            let! results = Task.WhenAll(tasks)
            return Array.toList results
        else
            // Sequential execution
            let mutable results = []
            for sample in samples do
                let! result = runSampleTestAsync config sample
                results <- results @ [result]
            return results
    }

let buildCompilerAsync compilerDir =
    task {
        let! (result, ms) = runProcessAsync "dotnet" "build" compilerDir None 120000
        match result with
        | Completed (0, _, _) -> return Some ms
        | Completed (code, _, stderr) ->
            printfn "Build failed (exit %d): %s" code (if stderr.Length > 500 then stderr.Substring(0, 500) else stderr)
            return None
        | Timeout _ ->
            printfn "Build timed out"
            return None
        | Failed ex ->
            printfn "Build exception: %s" ex.Message
            return None
    }

// =============================================================================
// CLI and Main
// =============================================================================

let defaultOptions = { ManifestPath = Path.Combine(__SOURCE_DIRECTORY__, "Manifest.toml"); TargetSamples = []; Verbose = false; TimeoutOverride = None; Parallel = false }

let rec parseArgs args opts =
    match args with
    | [] -> opts
    | "--sample" :: name :: rest -> parseArgs rest { opts with TargetSamples = name :: opts.TargetSamples }
    | "--verbose" :: rest -> parseArgs rest { opts with Verbose = true }
    | "--parallel" :: rest -> parseArgs rest { opts with Parallel = true }
    | "--timeout" :: sec :: rest -> match Int32.TryParse(sec) with true, n -> parseArgs rest { opts with TimeoutOverride = Some n } | _ -> parseArgs rest opts
    | "--help" :: _ ->
        printfn "Usage: dotnet fsi Runner.fsx [options]"
        printfn "Options:"
        printfn "  --sample NAME    Run specific sample(s) (can be repeated)"
        printfn "  --verbose        Show detailed output"
        printfn "  --parallel       Run samples in parallel (fan-out/fold-in)"
        printfn "  --timeout SEC    Override timeout for all samples"
        printfn "  --help           Show this help"
        exit 0
    | _ :: rest -> parseArgs rest opts

let mainAsync argv =
    task {
        let opts = parseArgs (Array.toList argv) defaultOptions
        printfn "=== Firefly Regression Test Runner ===\n"
        if not (File.Exists opts.ManifestPath) then
            printfn "ERROR: Manifest not found at %s" opts.ManifestPath
            return 1
        else
            let (config, allSamples) = loadManifest opts.ManifestPath
            let samples = match opts.TimeoutOverride with Some t -> allSamples |> List.map (fun s -> { s with TimeoutSeconds = t }) | None -> allSamples
            let samplesToRun =
                match opts.TargetSamples with
                | [] -> samples
                | targets ->
                    let targetSet = Set.ofList targets
                    let found = samples |> List.filter (fun s -> Set.contains s.Name targetSet)
                    let missing = targets |> List.filter (fun t -> not (List.exists (fun s -> s.Name = t) found))
                    if not (List.isEmpty missing) then
                        printfn "WARNING: Sample(s) not found: %s" (String.concat ", " missing)
                    found
            printfn "Manifest: %s\nCompiler: %s\nSamples: %d%s\n"
                opts.ManifestPath config.CompilerPath samplesToRun.Length
                (if opts.Parallel then " (parallel)" else "")

            // Build compiler
            let srcDir = Path.GetFullPath(Path.Combine(Path.GetDirectoryName(config.CompilerPath), "..", "..", "..", ".."))
            printfn "Building compiler..."
            let! buildResult = buildCompilerAsync srcDir
            match buildResult with
            | None -> return 1
            | Some buildMs ->
                printfn "Built in %dms\n" buildMs
                printfn "Running %d tests...\n" samplesToRun.Length

                let startTime = DateTime.Now

                // Fan-out/fold-in test execution
                let! results = runAllTestsAsync config samplesToRun opts.Parallel

                let endTime = DateTime.Now
                let report = {
                    RunId = startTime.ToString("yyyy-MM-ddTHH:mm:ss")
                    ManifestPath = opts.ManifestPath
                    CompilerPath = config.CompilerPath
                    StartTime = startTime
                    EndTime = endTime
                    Results = results
                }
                printfn "\n%s" (generateReport report opts.Verbose)
                return if didPass report then 0 else 1
    }

exit ((mainAsync (Environment.GetCommandLineArgs() |> Array.skip 2)).Result)
