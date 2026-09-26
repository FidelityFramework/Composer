#load "RunnerCore.fsx"

open System
open System.Diagnostics
open System.IO
open System.Threading
open System.Threading.Tasks
open RunnerCore

// Bounded scheduler and real process tests. No compiler build, native sample,
// shell fixture or Python process is used by this script.
let check condition message = if not condition then failwith message
let work = Path.Combine(Path.GetTempPath(), "composer-parallel-runner-" + Guid.NewGuid().ToString("N"))
Directory.CreateDirectory work |> ignore

let boundedAndOrdered jobs = task {
    let artifactRoot = createRunDirectory (Some (Path.Combine(work, "artifacts-" + string jobs)))
    let released = TaskCompletionSource<unit>(TaskCreationOptions.RunContinuationsAsynchronously)
    let occupied = TaskCompletionSource<unit>(TaskCreationOptions.RunContinuationsAsynchronously)
    let mutable active = 0
    let mutable peak = 0
    let counts = obj()
    let values = [0 .. 7]
    let execute index value = task {
        lock counts (fun () ->
            active <- active + 1
            peak <- max peak active
            check (active <= jobs) "The scheduler exceeded its requested concurrency."
            if active = jobs then occupied.TrySetResult(()) |> ignore)
        try
            do! released.Task
            // Complete the initial group in reverse order. Result order must
            // still follow the input, not the order in which workers finish.
            do! Task.Delay((jobs - index % jobs) * 15)
            let directory = jobDirectory artifactRoot index
            Directory.CreateDirectory directory |> ignore
            let artifact = Path.Combine(directory, "result.txt")
            File.WriteAllText(artifact, string value)
            return index, value, artifact
        finally
            lock counts (fun () -> active <- active - 1)
    }
    let pending = mapBounded jobs execute values
    try
        do! occupied.Task.WaitAsync(TimeSpan.FromSeconds 5.0)
        check (not pending.IsCompleted) "Work completed before its asynchronous release."
    finally
        released.TrySetResult(()) |> ignore
    let! results = pending.WaitAsync(TimeSpan.FromSeconds 10.0)
    check (peak = jobs) "Independent tasks never occupied the requested worker count."
    check (active = 0) "A worker remained active after the scheduler completed."
    check ((results |> List.map (fun (index, value, _) -> index, value)) = List.indexed values)
        "Results were reordered by completion time."
    let artifacts = results |> List.map (fun (_, _, path) -> path)
    check (Set.count (Set.ofList artifacts) = values.Length) "Two jobs shared an artifact path."
    for _, value, artifact in results do
        check (File.ReadAllText artifact = string value) "A peer overwrote another job's artifact."
}

let phaseBarrier () = task {
    let lastCompileRelease = TaskCompletionSource<unit>(TaskCreationOptions.RunContinuationsAsynchronously)
    let lastCompileStarted = TaskCompletionSource<unit>(TaskCreationOptions.RunContinuationsAsynchronously)
    let mutable completedCompiles = 0
    let mutable startedRuns = 0
    let compile index value = task {
        if index = 4 then
            lastCompileStarted.TrySetResult(()) |> ignore
            do! lastCompileRelease.Task
        Interlocked.Increment(&completedCompiles) |> ignore
        return value * 10
    }
    let run index compiled = task {
        check (Volatile.Read(&completedCompiles) = 5) "A binary ran before the compile phase finished."
        Interlocked.Increment(&startedRuns) |> ignore
        do! Task.Delay((5 - index) * 5)
        return compiled + 1
    }
    let pending = runPhases 2 compile run [0 .. 4]
    try
        do! lastCompileStarted.Task.WaitAsync(TimeSpan.FromSeconds 5.0)
        check (Volatile.Read(&startedRuns) = 0) "The run phase crossed an unfinished compile."
    finally
        lastCompileRelease.TrySetResult(()) |> ignore
    let! results = pending.WaitAsync(TimeSpan.FromSeconds 10.0)
    check (results = [1; 11; 21; 31; 41]) "Phase results lost their input association."
    check (startedRuns = 5) "A scheduled run was omitted or repeated."
}

let failedCompilesNeverRun () = task {
    let sample = {
        Name = "not-executed"; ProjectFile = "absent.fidproj"; BinaryName = "absent"
        StdinFile = None; ExpectedOutput = ""; TimeoutSeconds = 1
        Skip = false; SkipReason = None }
    let config = { SamplesRoot = work; CompilerPath = "absent"; DefaultTimeoutSeconds = 1 }
    let! failed = runBinaryPhaseAsync config (sample, CompileFailed (9, "compile out", "compile error", 1L), None)
    check (failed.RunResult.IsNone) "A failed compile attempted native execution."
    let! timedOut = runBinaryPhaseAsync config (sample, CompileTimeout 1000, None)
    check (timedOut.RunResult.IsNone) "A timed-out compile attempted native execution."
    let! skipped = runBinaryPhaseAsync config ({ sample with Skip = true }, CompileSkipped "fixture skip", None)
    match skipped.RunResult with
    | Some (RunSkipped _) -> ()
    | actual -> failwithf "A skipped compile did not retain its skipped run: %A" actual
    // An existing host executable is only a sentinel. Missing declared input
    // must fail before any process can start, rather than silently supplying EOF.
    let missingInput = { sample with StdinFile = Some "required-input.stdin" }
    let! missing = runBinaryPhaseAsync config (missingInput, CompileSuccess 0L, Option.ofObj Environment.ProcessPath)
    match missing.RunResult with
    | Some (RunFailed (-1, _, message, _)) ->
        check (message.Contains "required-input.stdin") "Missing stdin failed for an unrelated reason."
    | actual -> failwithf "A missing stdin fixture was not an explicit run failure: %A" actual
}

let argumentAndRootChecks () =
    let parsed = parseArgs ["--jobs"; "3"; "--timeout"; "2"] defaultOptions
    check (parsed.Jobs = 3 && parsed.TimeoutOverride = Some 2) "Positive jobs/timeout arguments were not retained."
    for arguments in
        [ ["--jobs"; "0"]; ["--jobs"; "-1"]; ["--jobs"; "many"]; ["--jobs"]
          ["--timeout"; "0"]; ["--timeout"; "-1"]; ["--timeout"; "many"]; ["--timeout"]
          ["--timeout"; string Int32.MaxValue]; ["--unknown-option"] ] do
        let mutable rejected = false
        try parseArgs arguments defaultOptions |> ignore
        with :? ArgumentException -> rejected <- true
        check rejected (sprintf "Invalid CLI arguments were silently accepted: %A" arguments)
    let parent = Path.Combine(work, "independent-runs")
    let first, second = createRunDirectory (Some parent), createRunDirectory (Some parent)
    let relative = Path.GetRelativePath(Directory.GetCurrentDirectory(), parent)
    let fromRelative = createRunDirectory (Some relative)
    check (Path.IsPathFullyQualified fromRelative) "A relative results parent produced relative worker paths."
    check (Path.GetDirectoryName fromRelative = Path.GetFullPath parent) "Relative results changed their intended parent."
    check (first <> second && Directory.Exists first && Directory.Exists second) "Two invocations shared a run directory."
    let outputs = [jobDirectory first 0; jobDirectory first 1; jobDirectory second 0]
    check (Set.count (Set.ofList outputs) = 3) "Run/job ordinals collided across invocations."
    for index, directory in List.indexed outputs do
        Directory.CreateDirectory directory |> ignore
        File.WriteAllText(Path.Combine(directory, "same-binary-name"), string index)
    for index, directory in List.indexed outputs do
        check (File.ReadAllText(Path.Combine(directory, "same-binary-name")) = string index)
            "The same binary name in another job overwrote retained artifacts."

    let compilerOutput = Path.Combine(work, "compiler-output")
    let nested = Path.Combine(compilerOutput, "results", "run")
    let sibling = compilerOutput + "-results"
    Directory.CreateDirectory nested |> ignore
    Directory.CreateDirectory sibling |> ignore
    let rejectSnapshot source destination =
        let mutable rejected = false
        try validateRunDirectory source destination
        with :? ArgumentException as ex ->
            check (ex.ParamName = "results") "Snapshot containment failed for an unrelated reason."
            rejected <- true
        check rejected (sprintf "Snapshot source/destination containment was accepted: %s -> %s" source destination)
    rejectSnapshot compilerOutput compilerOutput
    rejectSnapshot compilerOutput nested
    validateRunDirectory compilerOutput sibling
    validateRunDirectory compilerOutput work
    if OperatingSystem.IsLinux() then
        let outputAlias = Path.Combine(work, "compiler-output-link")
        Directory.CreateSymbolicLink(outputAlias, compilerOutput) |> ignore
        rejectSnapshot compilerOutput outputAlias
        rejectSnapshot compilerOutput (Path.Combine(outputAlias, "results", "run"))
        rejectSnapshot outputAlias nested
        let safeAlias = Path.Combine(work, "separate-results-link")
        Directory.CreateSymbolicLink(safeAlias, sibling) |> ignore
        validateRunDirectory compilerOutput safeAlias

// The child uses only the .NET base class library. It is interpreted from a
// private temporary source file and does not build or replace shared outputs.
let child = Path.Combine(work, "process fixture with spaces.fsx")
File.WriteAllText(child, """
open System
open System.Diagnostics
open System.IO
open System.Threading

let args = fsi.CommandLineArgs.[1..] |> Array.skipWhile ((=) "--")
match args with
| [| "streams" |] ->
    for _ in 1 .. 10000 do
        Console.Out.WriteLine("stdout0123456789")
        Console.Error.WriteLine("stderr0123456789")
| [| "input" |] ->
    let mutable line = Console.ReadLine()
    while not (isNull line) do
        Console.WriteLine("input:" + line)
        line <- Console.ReadLine()
| [| "arguments"; first; second |] ->
    Console.WriteLine(first)
    Console.WriteLine(second)
| [| "exit"; code |] ->
    Console.WriteLine("child-exit")
    Console.Error.WriteLine("child-error")
    Environment.Exit(Int32.Parse code)
| [| "blocked-input" |] -> Thread.Sleep 30000
| [| "delayed-marker"; ready; marker |] ->
    File.WriteAllText(ready, string DateTime.UtcNow.Ticks)
    Console.Out.WriteLine("descendant holds stdout")
    Console.Error.WriteLine("descendant holds stderr")
    Console.Out.Flush()
    Console.Error.Flush()
    Thread.Sleep 10000
    File.WriteAllText(marker, "survived")
| [| "parent"; ready; marker |] ->
    let info = ProcessStartInfo("dotnet", UseShellExecute = false)
    for argument in ["fsi"; "--exec"; Path.Combine(__SOURCE_DIRECTORY__, __SOURCE_FILE__); "--"; "delayed-marker"; ready; marker] do
        info.ArgumentList.Add argument
    use descendant = Process.Start info
    descendant.WaitForExit()
| [| "orphan-parent"; ready; marker; parentExit |] ->
    let info = ProcessStartInfo("dotnet", UseShellExecute = false)
    for argument in ["fsi"; "--exec"; Path.Combine(__SOURCE_DIRECTORY__, __SOURCE_FILE__); "--"; "delayed-marker"; ready; marker] do
        info.ArgumentList.Add argument
    use descendant = Process.Start info
    let deadline = Stopwatch.StartNew()
    while not (File.Exists ready) && not descendant.HasExited && deadline.ElapsedMilliseconds < 5000L do
        Thread.Sleep 10
    if not (File.Exists ready) then failwith "The descendant did not become ready before its parent exited."
    File.WriteAllText(parentExit, string Environment.ProcessId)
    Environment.Exit 0
| _ -> failwithf "Unknown process fixture arguments: %A" args
""")

let invoke arguments input timeout =
    runProcess "dotnet" (["fsi"; "--exec"; child; "--"] @ arguments) work input timeout

let processChecks () = task {
    match invoke ["streams"] None 10000 |> fst with
    | Completed (0, stdout, stderr) ->
        check (stdout = String.replicate 10000 ("stdout0123456789" + Environment.NewLine)) "Stdout was truncated or mixed with a peer."
        check (stderr = String.replicate 10000 ("stderr0123456789" + Environment.NewLine)) "Stderr was truncated or not drained concurrently."
    | actual -> failwithf "Concurrent stream drainage failed: %A" actual
    match invoke ["input"] (Some "first\n\nlast\n") 10000 |> fst with
    | Completed (0, stdout, "") ->
        check (stdout = String.concat Environment.NewLine ["input:first"; "input:"; "input:last"; ""]) "Paced stdin lost line boundaries or EOF."
    | actual -> failwithf "Interactive input failed: %A" actual
    match invoke ["input"] None 10000 |> fst with
    | Completed (0, "", "") -> ()
    | actual -> failwithf "A no-input worker did not receive EOF: %A" actual
    match invoke ["arguments"; "path with spaces"; "$(literal)"] None 10000 |> fst with
    | Completed (0, stdout, "") ->
        check (stdout = String.concat Environment.NewLine ["path with spaces"; "$(literal)"; ""]) "Argument boundaries changed."
    | actual -> failwithf "Literal process arguments failed: %A" actual
    let invalidExecutable = Path.Combine(work, "invalid executable")
    File.WriteAllText(invalidExecutable, "This is not an executable image.\n")
    if not (OperatingSystem.IsWindows()) then
        File.SetUnixFileMode(invalidExecutable, UnixFileMode.UserRead ||| UnixFileMode.UserWrite ||| UnixFileMode.UserExecute)
    match runProcess invalidExecutable [] work None 10000 |> fst with
    | Failed ex -> check (ex.Message.Contains invalidExecutable) "Hosted launch failure lost the executable identity."
    | actual -> failwithf "Hosted launch failure was mislabeled as a child exit or timeout: %A" actual
    match invoke ["exit"; "1"] None 10000 |> fst with
    | Completed (1, stdout, stderr) ->
        check (stdout = "child-exit" + Environment.NewLine && stderr = "child-error" + Environment.NewLine)
            "A real nonzero child exit lost its streams."
    | actual -> failwithf "A real child exit was confused with a host failure: %A" actual
    let inputResult, inputElapsed = invoke ["blocked-input"] (Some (String.replicate 1048576 "x")) 2000
    match inputResult with
    | Timeout 2000 -> check (inputElapsed < 7000L) "Blocked stdin escaped the shared deadline."
    | actual -> failwithf "Blocked stdin was not a timeout: %A" actual

    let ready = Path.Combine(work, "descendant-ready")
    let marker = Path.Combine(work, "surviving-descendant")
    let processJob index () = Task.Run(fun () ->
        if index = 0 then invoke ["parent"; ready; marker] None 6000
        else invoke ["streams"] None 10000)
    let! results = mapBounded 2 processJob [(); ()]
    match results[0] with
    | Timeout 6000, elapsed -> check (elapsed < 11000L) "Held-open pipes escaped the timeout."
    | actual -> failwithf "Expected the parent/descendant process tree to time out: %A" actual
    match results[1] with
    | Completed (0, stdout, stderr), _ ->
        check (stdout = String.replicate 10000 ("stdout0123456789" + Environment.NewLine)) "Timeout damaged a peer's stdout."
        check (stderr = String.replicate 10000 ("stderr0123456789" + Environment.NewLine)) "Timeout damaged a peer's stderr."
    | actual -> failwithf "An independent process failed beside the timeout: %A" actual
    check (File.Exists ready) "The descendant never started; process-tree termination was not exercised."
    let started = DateTime(Int64.Parse(File.ReadAllText ready), DateTimeKind.Utc)
    let remaining = started.AddMilliseconds(10500.0) - DateTime.UtcNow
    if remaining > TimeSpan.Zero then do! Task.Delay remaining
    check (not (File.Exists marker)) "A timed-out process left its descendant alive."

    if OperatingSystem.IsLinux() then
        let orphanReady = Path.Combine(work, "orphan-ready")
        let orphanMarker = Path.Combine(work, "surviving-orphan")
        let parentExit = Path.Combine(work, "orphan-parent-exit")
        let orphanJob index () =
            let arguments =
                if index = 0 then ["orphan-parent"; orphanReady; orphanMarker; parentExit]
                else ["streams"]
            runProcessAsync "dotnet" (["fsi"; "--exec"; child; "--"] @ arguments) work None
                (if index = 0 then 8000 else 10000)
        let pending = mapBounded 2 orphanJob [(); ()]
        let parentHasExited () =
            if not (File.Exists parentExit) then false
            else
                match Int32.TryParse(File.ReadAllText parentExit) with
                | true, pid ->
                    try
                        use parent = Process.GetProcessById pid
                        parent.HasExited
                    with :? ArgumentException -> true
                | _ -> false
        let observation = Stopwatch.StartNew()
        while not (parentHasExited ()) && observation.ElapsedMilliseconds < 6500L do
            do! Task.Delay 20
        check (File.Exists orphanReady && parentHasExited ())
            "The orphan fixture did not establish a started descendant and an exited parent before timeout."
        check (not pending.IsCompleted) "Exited parent completed the job while its descendant still held redirected pipes."
        let! orphanResults = pending
        match orphanResults[0] with
        | Timeout 8000, elapsed -> check (elapsed < 13000L) "Orphan-held pipes escaped the process-group deadline."
        | actual -> failwithf "An exited parent with inherited-pipe descendants did not time out: %A" actual
        match orphanResults[1] with
        | Completed (0, stdout, stderr), _ ->
            check (stdout = String.replicate 10000 ("stdout0123456789" + Environment.NewLine)) "Orphan cleanup damaged a peer's stdout."
            check (stderr = String.replicate 10000 ("stderr0123456789" + Environment.NewLine)) "Orphan cleanup damaged a peer's stderr."
        | actual -> failwithf "An independent peer failed beside orphan cleanup: %A" actual
        let started = DateTime(Int64.Parse(File.ReadAllText orphanReady), DateTimeKind.Utc)
        let remaining = started.AddMilliseconds(10500.0) - DateTime.UtcNow
        if remaining > TimeSpan.Zero then do! Task.Delay remaining
        check (not (File.Exists orphanMarker)) "A timed-out job left an inherited-pipe descendant alive after its parent exited."
}

try
    let run = task {
        argumentAndRootChecks ()
        printfn "PASS strict jobs/timeouts, unique artifacts and snapshot containment including symlinks"
        do! boundedAndOrdered 1
        do! boundedAndOrdered 3
        printfn "PASS worker limits, actual overlap, deterministic results and independent artifacts"
        do! phaseBarrier ()
        do! failedCompilesNeverRun ()
        printfn "PASS compile/run phase barrier, failed/skipped compile admission and missing stdin failure"
        do! processChecks ()
        printfn "PASS .NET streams, stdin, hosted launch errors, nonzero child exits and timeout cleanup including orphans"
    }
    run.GetAwaiter().GetResult()
    printfn "All parallel runner tests passed."
finally
    Directory.Delete(work, true)
