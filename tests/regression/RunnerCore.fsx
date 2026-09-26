// Testable implementation for Runner.fsx.
#r "../../src/bin/Debug/net10.0/XParsec.dll"
#r "../../src/bin/Debug/net10.0/Fidelity.Data.dll"

open System
open System.IO
open System.Diagnostics
open System.Threading
open System.Threading.Tasks
open System.Security.Cryptography
open System.Text
open System.Runtime.InteropServices
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

type CliOptions = {
    ManifestPath: string
    TargetSamples: string list
    Verbose: bool
    TimeoutOverride: int option
    Jobs: int
    ResultsDirectory: string option
}

// =============================================================================
// Process Runner
// =============================================================================

module private ProcessGroup =
    [<DllImport("libc", SetLastError = true)>]
    extern int kill(int pid, int signal)

let mutable private processHostAssembly =
    Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../Infrastructure/ProcessHost/bin/Debug/net10.0/ProcessHost.dll"))

// Adapted from tests/Infrastructure/Process.fs: direct argument arrays,
// concurrent stream reads and process-tree termination. This runner also needs
// separate streams, a working directory and paced input for console samples.
let private runProcessCoreAsync hostAssembly logDirectory stage cmd (args: string list) workDir (stdin: string option) (timeoutMs: int) = task {
    let sw = Stopwatch.StartNew()
    let readyFile = hostAssembly |> Option.map (fun _ -> Path.Combine(Path.GetTempPath(), "composer-process-" + Guid.NewGuid().ToString("N") + ".ready"))
    use readyCleanup =
        { new IDisposable with
            member _.Dispose() =
                readyFile |> Option.iter (fun path ->
                    for file in [path; path + ".error"] do
                        if File.Exists file then File.Delete file) }
    try
        use proc = new Process()
        use cancellation = new CancellationTokenSource()
        let executable, arguments =
            match hostAssembly, readyFile with
            | Some host, Some ready -> "dotnet", [host; "--ready"; ready; "--"; cmd] @ args
            | _ -> cmd, args
        proc.StartInfo <- ProcessStartInfo(
            FileName = executable,
            WorkingDirectory = workDir,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            // A job without supplied input gets EOF, never the runner's terminal.
            RedirectStandardInput = true,
            UseShellExecute = false,
            CreateNoWindow = true)
        for argument in arguments do proc.StartInfo.ArgumentList.Add argument

        if not (proc.Start()) then failwith ("Cannot start " + cmd)

        let terminateJob () =
            // The host establishes a fresh Linux session before starting the
            // command. Its process group survives the leader exiting, so even
            // an orphan that holds our pipes remains owned by this job.
            if OperatingSystem.IsLinux() then
                match readyFile with
                | Some path when File.Exists path ->
                    match Int32.TryParse(File.ReadAllText path) with
                    | true, pid when pid = proc.Id -> ProcessGroup.kill(-pid, 9) |> ignore
                    | _ -> ()
                | _ -> ()
            if not proc.HasExited then
                try proc.Kill(true) with :? InvalidOperationException -> ()

        let capture (reader: StreamReader) streamName = task {
            let writer = logDirectory |> Option.map (fun directory ->
                Directory.CreateDirectory directory |> ignore
                new StreamWriter(Path.Combine(directory, stage + "." + streamName + ".log"), false))
            let text = StringBuilder()
            let buffer = Array.zeroCreate<char> 4096
            try
                let mutable reading = true
                while reading do
                    let! count = reader.ReadAsync(buffer.AsMemory(), cancellation.Token)
                    if count = 0 then reading <- false
                    else
                        text.Append(buffer, 0, count) |> ignore
                        match writer with
                        | Some output ->
                            do! output.WriteAsync(buffer.AsMemory(0, count), cancellation.Token)
                            do! output.FlushAsync(cancellation.Token)
                        | None -> ()
                return text.ToString()
            finally
                writer |> Option.iter _.Dispose()
        }
        let stdout = capture proc.StandardOutput "stdout"
        let stderr = capture proc.StandardError "stderr"
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
            | None -> proc.StandardInput.Close()
        }
        let finished = Task.WhenAll [| stdout :> Task; stderr :> Task; writeInput :> Task; proc.WaitForExitAsync(cancellation.Token) |]
        try
            do! finished.WaitAsync(TimeSpan.FromMilliseconds(float timeoutMs))
            terminateJob ()
            sw.Stop()
            let result =
                match readyFile with
                | Some ready when File.Exists(ready + ".error") -> Failed (InvalidOperationException(File.ReadAllText(ready + ".error")))
                | _ -> Completed(proc.ExitCode, stdout.Result, stderr.Result)
            return (result, sw.ElapsedMilliseconds)
        with ex ->
            // Kill descendants too: they can otherwise retain redirected pipes
            // or survive a timed-out compiler invocation.
            terminateJob ()
            cancellation.Cancel()
            // Observe cancellation before disposing the streams. Bound cleanup
            // too, including when a descendant keeps a redirected pipe open.
            try do! finished.WaitAsync(TimeSpan.FromSeconds 5.) with _ -> ()
            try do! proc.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds 5.) with _ -> ()
            sw.Stop()
            return ((match ex with :? TimeoutException -> Timeout timeoutMs | _ -> Failed ex), sw.ElapsedMilliseconds)
    with ex ->
        sw.Stop()
        return (Failed ex, sw.ElapsedMilliseconds)
}

let private runProcessWithLogsAsync logDirectory stage cmd args workDir stdin timeoutMs =
    if not (File.Exists processHostAssembly) then
        failwith "ProcessHost is not built. Build tests/Infrastructure/ProcessHost/ProcessHost.fsproj before loading harness tests; Runner.fsx builds its own private host."
    runProcessCoreAsync (Some processHostAssembly) logDirectory stage cmd args workDir stdin timeoutMs

let runProcessAsync cmd args workDir stdin timeoutMs =
    runProcessWithLogsAsync None "" cmd args workDir stdin timeoutMs

let runProcess cmd args workDir stdin timeoutMs =
    (runProcessAsync cmd args workDir stdin timeoutMs).GetAwaiter().GetResult()

/// A fixed number of async workers own independent process jobs. Queued jobs do
/// not start their process deadline until a worker starts them. Results keep the
/// input order regardless of completion order; no thread blocks on a semaphore.
let mapBounded jobs (work: int -> 'a -> Task<'b>) (items: 'a list) = task {
    if jobs < 1 then invalidArg "jobs" "The worker count must be positive."
    let input = List.toArray items
    let output = Array.zeroCreate<'b> input.Length
    let mutable next = -1
    let worker () = task {
        let mutable index = Interlocked.Increment(&next)
        while index < input.Length do
            let! result = work index input[index]
            output[index] <- result
            index <- Interlocked.Increment(&next)
    }
    let! _ = Array.init (min jobs input.Length) (fun _ -> worker ()) |> Task.WhenAll
    return Array.toList output
}

/// Keep the build/compile/execute barriers: native execution never overlaps a
/// still-running compiler job from this run. Each phase has the same job bound.
let runPhases jobs compile run items = task {
    let! compiled = mapBounded jobs compile items
    return! mapBounded jobs run compiled
}

let private saveProcessLog directory stage (result: ProcessResult) elapsed =
    Directory.CreateDirectory directory |> ignore
    let stdout, stderr, status =
        match result with
        | Completed(code, stdout, stderr) -> stdout, stderr, sprintf "exit=%d" code
        | Timeout ms -> "", "", sprintf "timeout=%dms" ms
        | Failed ex -> "", ex.ToString(), "launch-or-io-failure\n" + ex.ToString()
    // Timed-out jobs keep whatever their streams had already captured.
    let writeStream name (text: string) =
        let path = Path.Combine(directory, stage + "." + name + ".log")
        if not (File.Exists path) then File.WriteAllText(path, text)
    writeStream "stdout" stdout
    writeStream "stderr" stderr
    File.WriteAllText(Path.Combine(directory, stage + ".status"), sprintf "%s\nelapsed=%dms\n" status elapsed)

let private compileSampleAsync compilerPath projectDir projectFile outputPath artifactsDirectory timeoutMs = task {
    let artifactsArgs = artifactsDirectory |> Option.map (fun dir -> ["--artifacts-dir"; dir]) |> Option.defaultValue []
    let! result, ms = runProcessWithLogsAsync artifactsDirectory "compile" compilerPath (["compile"; projectFile; "-o"; outputPath; "-k"; "--no-color"] @ artifactsArgs) projectDir None timeoutMs
    artifactsDirectory |> Option.iter (fun dir -> saveProcessLog dir "compile" result ms)
    return
        match result with
        | Completed (0, _, _) -> CompileSuccess ms
        | Completed (code, stdout, stderr) -> CompileFailed (code, stdout, stderr, ms)
        | Timeout t -> CompileTimeout t
        | Failed ex -> CompileFailed (-1, "", ex.Message, ms)
}

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
        Some (File.ReadAllText path)

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
let compileSamplePhaseAsync config (artifactsDirectory: string option) sample = task {
    if sample.Skip then
        return (sample, CompileSkipped (sample.SkipReason |> Option.defaultValue "marked skip"), None)
    else
        let sampleDir = Path.Combine(config.SamplesRoot, sample.Name)
        let timeoutMs = sample.TimeoutSeconds * 1000
        let outputPath =
            match artifactsDirectory with
            | Some dir -> Path.Combine(dir, Path.GetFileName sample.BinaryName)
            | None -> Path.Combine(sampleDir, sample.BinaryName)
        let! compileResult = compileSampleAsync config.CompilerPath sampleDir sample.ProjectFile outputPath artifactsDirectory timeoutMs
        let binaryPath =
            match compileResult with
            | CompileSuccess _ -> Some outputPath
            | _ -> None
        return (sample, compileResult, binaryPath)
}

/// Phase 2: Run a single binary (takes compiled sample info)
let runBinaryPhaseAsync config (sample, compileResult, binaryPathOpt) = task {
    match compileResult, binaryPathOpt with
    | CompileSuccess compileMs, Some binaryPath ->
        if not (File.Exists binaryPath) then
            return { Sample = sample; CompileResult = CompileFailed (-1, "", $"Binary not found: {binaryPath}", compileMs); RunResult = None }
        else
            let sampleDir = Path.Combine(config.SamplesRoot, sample.Name)
            let timeoutMs = sample.TimeoutSeconds * 1000
            try
                let stdin = getStdinContent config sample
                let logDirectory = Path.GetDirectoryName binaryPath
                let! outcome, runMs = runProcessWithLogsAsync (Some logDirectory) "run" binaryPath [] sampleDir stdin timeoutMs
                saveProcessLog logDirectory "run" outcome runMs
                return
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
            with ex ->
                saveProcessLog (Path.GetDirectoryName binaryPath) "run" (Failed ex) 0L
                return { Sample = sample; CompileResult = compileResult; RunResult = Some (RunFailed (-1, "", ex.Message, 0L)) }
    | CompileSkipped reason, _ ->
        return { Sample = sample; CompileResult = compileResult; RunResult = Some (RunSkipped "compile skipped") }
    | _, _ ->
        return { Sample = sample; CompileResult = compileResult; RunResult = None }
}

let runBinaryPhase config compiled = (runBinaryPhaseAsync config compiled).GetAwaiter().GetResult()

/// Every ordinal has a private output root even when manifest entries name the
/// same source project. Compiler globals remain isolated in child processes.
let jobDirectory runDirectory index = Path.Combine(runDirectory, sprintf "%04d" (index + 1))

let runAllTests jobs runDirectory config samples verbose = task {
    let total = List.length samples
    let consoleGate = obj()
    printfn "--- Compile, then execute (%d samples, at most %d jobs per phase) ---" total jobs
    let compile i s = task {
            let directory = jobDirectory runDirectory i
            Directory.CreateDirectory directory |> ignore
            File.WriteAllText(Path.Combine(directory, "sample.txt"), s.Name + "\n" + s.ProjectFile + "\n")
            let! result = compileSamplePhaseAsync config (Some directory) s
            let (_, cr, _) = result
            let status =
                match cr with
                | CompileSuccess ms -> sprintf "OK (%.2fs)" (float ms / 1000.)
                | CompileFailed(code, _, _, ms) -> sprintf "FAIL (exit %d, %.2fs)" code (float ms / 1000.)
                | CompileTimeout t -> sprintf "TIMEOUT (%dms)" t
                | CompileSkipped reason -> sprintf "SKIP (%s)" reason
            lock consoleGate (fun () ->
                if verbose then printfn "  [%d/%d] Compile %s: %s" (i + 1) total s.Name status
                else printf "."
                Console.Out.Flush())
            return result
    }
    let run i r = task {
            let (sample, _, _) = r
            let! result = runBinaryPhaseAsync config r
            let status =
                match result.RunResult with
                | Some (RunSuccess ms) -> sprintf "PASS (%dms)" ms
                | Some (RunFailed(code, _, _, ms)) -> sprintf "FAIL (exit %d, %dms)" code ms
                | Some (OutputMismatch(_, _, ms)) -> sprintf "MISMATCH (%dms)" ms
                | Some (RunTimeout t) -> sprintf "TIMEOUT (%dms)" t
                | Some (RunSkipped reason) -> sprintf "SKIP (%s)" reason
                | None -> "no binary"
            lock consoleGate (fun () ->
                if verbose then printfn "  [%d/%d] Run %s: %s" (i + 1) total sample.Name status
                else printf "."
                Console.Out.Flush())
            return result
    }
    let! results = runPhases jobs compile run samples
    if not verbose then printfn ""
    return results
}

// =============================================================================
// CLI and Main
// =============================================================================

let defaultOptions = {
    ManifestPath = Path.Combine(__SOURCE_DIRECTORY__, "Manifest.toml")
    TargetSamples = []; Verbose = false; TimeoutOverride = None
    Jobs = 1; ResultsDirectory = None
}

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
    let positive option (value: string) =
        match Int32.TryParse value with
        | true, n when n > 0 && (option <> "--timeout" || n <= Int32.MaxValue / 1000) -> n
        | _ -> invalidArg option (option + " requires a positive integer" )
    match args with
    | [] -> opts
    | "--sample" :: name :: rest -> parseArgs rest { opts with TargetSamples = name :: opts.TargetSamples }
    | "--verbose" :: rest -> parseArgs rest { opts with Verbose = true }
    | "--timeout" :: sec :: rest -> parseArgs rest { opts with TimeoutOverride = Some (positive "--timeout" sec) }
    | "--jobs" :: count :: rest -> parseArgs rest { opts with Jobs = positive "--jobs" count }
    | "--results" :: path :: rest -> parseArgs rest { opts with ResultsDirectory = Some (Path.GetFullPath path) }
    | "--manifest" :: path :: rest -> parseArgs rest { opts with ManifestPath = Path.GetFullPath path }
    | "--" :: rest -> parseArgs rest opts
    | "--help" :: _ ->
        printfn "Usage: dotnet fsi Runner.fsx [options]"
        printfn "  --sample NAME    Run specific sample(s)"
        printfn "  --verbose        Show detailed output"
        printfn "  --timeout SEC    Override timeout for all samples"
        printfn "  --jobs N         At most N compiler/native jobs per phase (default: 1)"
        printfn "  --results DIR    Parent directory for a unique run and its retained artifacts"
        printfn "  --manifest FILE  Use a different sample manifest"
        printfn "  --help           Show this help"
        exit 0
    | option :: _ -> invalidArg "args" ("Unknown option or missing value: " + option)

let createRunDirectory parent =
    let parent = parent |> Option.defaultValue (Path.Combine(Path.GetTempPath(), "composer-checks")) |> Path.GetFullPath
    let directory = Path.Combine(parent, DateTime.UtcNow.ToString("yyyyMMddTHHmmss") + "-" + Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory directory |> ignore
    directory

let rec private canonicalDirectory path =
    let directory = DirectoryInfo(Path.GetFullPath path)
    match directory.ResolveLinkTarget(true) with
    | null ->
        match directory.Parent with
        | null -> directory.FullName
        | parent -> Path.Combine(canonicalDirectory parent.FullName, directory.Name)
    | target -> canonicalDirectory target.FullName

/// Reject a snapshot inside its own source, including through directory links.
let validateRunDirectory compilerDirectory runDirectory =
    let relative = Path.GetRelativePath(canonicalDirectory compilerDirectory, canonicalDirectory runDirectory)
    if not (Path.IsPathRooted relative) && relative <> ".." && not (relative.StartsWith(".." + string Path.DirectorySeparatorChar, StringComparison.Ordinal)) then
        invalidArg "results" "The results directory cannot be inside the compiler output directory."

let private copyDirectory source destination =
    Directory.CreateDirectory destination |> ignore
    for directory in Directory.EnumerateDirectories(source, "*", SearchOption.AllDirectories) do
        Directory.CreateDirectory(Path.Combine(destination, Path.GetRelativePath(source, directory))) |> ignore
    for file in Directory.EnumerateFiles(source, "*", SearchOption.AllDirectories) do
        let target = Path.Combine(destination, Path.GetRelativePath(source, file))
        File.Copy(file, target)
        if not (OperatingSystem.IsWindows()) then File.SetUnixFileMode(target, File.GetUnixFileMode file)

/// Other runner invocations must not overwrite DLLs while this one snapshots
/// its build. Hold a cooperative inter-process lease only through build/copy;
/// compilation jobs use the private snapshot and release the shared build tree.
let private prepareCompiler (config: TestConfig) runDirectory = task {
    let outputDirectory = Path.GetDirectoryName config.CompilerPath
    validateRunDirectory outputDirectory runDirectory
    let sourceDirectory = Path.GetFullPath(Path.Combine(outputDirectory, "..", "..", ".."))
    let key = Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes sourceDirectory))
    let lockPath = Path.Combine(Path.GetTempPath(), "composer-check-build-" + key + ".lock")
    let deadline = Stopwatch.StartNew()
    let acquire () = task {
        let mutable opened = None
        while opened.IsNone do
            try opened <- Some (new FileStream(lockPath, FileMode.OpenOrCreate, FileAccess.ReadWrite, FileShare.None))
            with :? IOException when deadline.Elapsed < TimeSpan.FromMinutes 5. -> ()
            if opened.IsNone then
                do! Task.Delay 100
        return opened.Value
    }
    use! lease = acquire ()
    printfn "Building compiler..."
    // One build, before any workers. Its logs belong to this run too.
    let! outcome, ms = runProcessWithLogsAsync (Some runDirectory) "build" "dotnet" ["build"; "--disable-build-servers"] sourceDirectory None 120000
    saveProcessLog runDirectory "build" outcome ms
    match outcome with
    | Completed(0, _, _) ->
        let snapshot = Path.Combine(runDirectory, "compiler")
        copyDirectory outputDirectory snapshot
        let compiler = Path.Combine(snapshot, Path.GetFileName config.CompilerPath)
        let hashes =
            Directory.EnumerateFiles(snapshot, "*", SearchOption.AllDirectories) |> Seq.sort
            |> Seq.map (fun file ->
                use stream = File.OpenRead file
                sprintf "%s  %s" (Convert.ToHexString(SHA256.HashData stream)) (Path.GetRelativePath(snapshot, file)))
        File.WriteAllLines(Path.Combine(runDirectory, "compiler.sha256"), hashes)
        printfn "Built in %dms; compiler snapshot: %s\n" ms compiler
        return Some { config with CompilerPath = compiler }
    | _ ->
        eprintfn "Compiler build failed; see %s" runDirectory
        return None
}

let private prepareProcessHost runDirectory = task {
    let project = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../Infrastructure/ProcessHost/ProcessHost.fsproj"))
    let output = Path.Combine(runDirectory, "process-host")
    // Bootstrap just this tiny BCL host; separate obj/bin trees let concurrent
    // runners build it without sharing mutable build outputs.
    let! result, ms = runProcessCoreAsync None (Some runDirectory) "host-build" "dotnet"
                        ["build"; project; "--artifacts-path"; Path.Combine(runDirectory, "host-build");
                         "--output"; output; "--disable-build-servers"] __SOURCE_DIRECTORY__ None 120000
    saveProcessLog runDirectory "host-build" result ms
    match result with
    | Completed(0, _, _) ->
        processHostAssembly <- Path.Combine(output, "ProcessHost.dll")
        return true
    | _ ->
        eprintfn "Process host build failed; see %s" runDirectory
        return false
}

let private execute argv =
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

            let runDirectory = createRunDirectory opts.ResultsDirectory
            validateRunDirectory (Path.GetDirectoryName config.CompilerPath) runDirectory
            printfn "Jobs: %d\nArtifacts: %s\n" opts.Jobs runDirectory
            File.Copy(opts.ManifestPath, Path.Combine(runDirectory, "Manifest.toml"))
            File.WriteAllLines(Path.Combine(runDirectory, "selection.txt"), samplesToRun |> List.mapi (fun i sample ->
                sprintf "%04d\t%s\ttimeout=%ds" (i + 1) sample.Name sample.TimeoutSeconds))
            let provenance =
                {| Manifest = opts.ManifestPath; SamplesRoot = config.SamplesRoot
                   CompilerSourceOutput = config.CompilerPath; Jobs = opts.Jobs
                   Samples = samplesToRun |> List.mapi (fun i sample ->
                       {| Name = sample.Name; Project = Path.GetFullPath(Path.Combine(config.SamplesRoot, sample.Name, sample.ProjectFile))
                          Artifacts = jobDirectory runDirectory i; TimeoutSeconds = sample.TimeoutSeconds
                          ExpectedOutput = sample.ExpectedOutput; Skipped = sample.Skip |}) |> List.toArray |}
            let jsonOptions = System.Text.Json.JsonSerializerOptions(WriteIndented = true)
            File.WriteAllText(Path.Combine(runDirectory, "run.json"), System.Text.Json.JsonSerializer.Serialize(provenance, jsonOptions))
            let compiler = task {
                let! ready = prepareProcessHost runDirectory
                if ready then return! prepareCompiler config runDirectory
                else return None
            }
            match compiler.GetAwaiter().GetResult() with
            | None -> 1
            | Some isolatedConfig ->
                printfn "Running %d tests...\n" (List.length samplesToRun)

                let startTime = DateTime.Now
                let results = (runAllTests opts.Jobs runDirectory isolatedConfig samplesToRun opts.Verbose).GetAwaiter().GetResult()
                let endTime = DateTime.Now

                let report = {
                    RunId = startTime.ToString("s")
                    ManifestPath = opts.ManifestPath
                    CompilerPath = isolatedConfig.CompilerPath
                    StartTime = startTime
                    EndTime = endTime
                    Results = results
                }

                generateReport report opts.Verbose
                let outcomes = results |> List.map (fun result ->
                    {| Name = result.Sample.Name; Compile = sprintf "%A" result.CompileResult
                       Run = result.RunResult |> Option.map (sprintf "%A") |> Option.defaultValue "Not run" |}) |> List.toArray
                File.WriteAllText(Path.Combine(runDirectory, "results.json"), System.Text.Json.JsonSerializer.Serialize(outcomes, jsonOptions))
                if didPass report then 0 else 1

let main argv =
    try execute argv
    with ex ->
        eprintfn "ERROR: %s" ex.Message
        1
