#load "RunnerCore.fsx"

open System
open System.IO
open System.Threading
open RunnerCore

// Focused tests of the oracle harness. No Composer build or sample compilation.
let work = Path.Combine(Path.GetTempPath(), "composer-regression-runner-" + Guid.NewGuid().ToString("N"))
Directory.CreateDirectory work |> ignore
let check condition message = if not condition then failwith message
let script name body =
    let path = Path.Combine(work, name)
    File.WriteAllText(path, "#!/bin/sh\n" + body + "\n")
    File.SetUnixFileMode(path, UnixFileMode.UserRead ||| UnixFileMode.UserWrite ||| UnixFileMode.UserExecute)
    path

let sample = {
    Name = "oracle"; ProjectFile = "unused.fidproj"; BinaryName = "unused"
    StdinFile = None; ExpectedOutput = "expected\n"; TimeoutSeconds = 2
    Skip = false; SkipReason = None }
let config = { SamplesRoot = work; CompilerPath = "unused"; DefaultTimeoutSeconds = 2 }
Directory.CreateDirectory(Path.Combine(work, sample.Name)) |> ignore

let report result = {
    RunId = "harness-test"; ManifestPath = "unused"; CompilerPath = "unused"
    StartTime = DateTime.UtcNow; EndTime = DateTime.UtcNow; Results = [result] }

try
    let succeeds = script "matching output" "printf 'expected\\n'"
    let successful = runBinaryPhase config (sample, CompileSuccess 0L, Some succeeds)
    check (didPass (report successful)) "Matching stdout with exit zero must pass."

    let fails = script "matching output but failure" "printf 'expected\\n'; printf 'native failure\\n' >&2; exit 7"
    let failure = runBinaryPhase config (sample, CompileSuccess 0L, Some fails)
    match failure.RunResult with
    | Some (RunFailed (7, "expected\n", "native failure\n", _)) -> ()
    | result -> failwithf "Nonzero native exit was not preserved: %A" result
    check (not (didPass (report failure))) "Matching output must not hide a native failure."
    printfn "PASS matching stdout requires a zero native exit"

    let invalid = Path.Combine(work, "invalid executable")
    File.WriteAllText(invalid, "not an executable")
    let launchFailure = runBinaryPhase config (sample, CompileSuccess 0L, Some invalid)
    match launchFailure.RunResult with
    | Some (RunFailed (-1, _, _, _)) -> ()
    | result -> failwithf "Launch failure was mislabeled: %A" result
    printfn "PASS launch failure is distinct from timeout"

    // Both pipes exceed their typical capacity. Neither stream can be read only
    // after waiting for exit, or only after the other stream reaches EOF.
    let writes = script "large streams" "i=0; while [ \"$i\" -lt 10000 ]; do printf 'stdout0123456789\\n'; printf 'stderr0123456789\\n' >&2; i=$((i+1)); done"
    match runProcess writes [] work None 5000 |> fst with
    | Completed (0, stdout, stderr) ->
        check (stdout = String.replicate 10000 "stdout0123456789\n") "Stdout was truncated."
        check (stderr = String.replicate 10000 "stderr0123456789\n") "Stderr was truncated."
    | result -> failwithf "Concurrent stream drainage failed: %A" result
    printfn "PASS both redirected streams drain concurrently"

    let marker = Path.Combine(work, "surviving descendant")
    let hangs = script "holds both pipes" "(sleep 2; printf 'survived' > \"$1\") &\nprintf 'stdout is open\\n'; printf 'stderr is open\\n' >&2\nwait"
    let timeout, elapsed = runProcess hangs [marker] work None 200
    match timeout with
    | Timeout 200 -> check (elapsed < 2000L) "Timeout did not bound the process run."
    | result -> failwithf "Expected timeout while both pipes remained open: %A" result
    // A surviving descendant would create the file after its sleep completes.
    Thread.Sleep 2200
    check (not (File.Exists marker)) "The timed-out process left a descendant running."
    printfn "PASS held-open stdout/stderr time out and descendants are terminated"

    let noInput = script "does not read stdin" "sleep 30"
    let inputTimeout, inputElapsed = runProcess noInput [] work (Some (String.replicate 1048576 "x")) 200
    match inputTimeout with
    | Timeout 200 -> check (inputElapsed < 2000L) "Blocked stdin escaped the process deadline."
    | result -> failwithf "Expected timeout for blocked stdin: %A" result
    printfn "PASS blocked stdin shares the timeout"

    let argumentEcho = script "argument boundaries" "printf '%s\\n' \"$#\" \"$1\" \"$2\""
    match runProcess argumentEcho ["path with spaces"; "$(literal)"] work None 2000 |> fst with
    | Completed (0, "2\npath with spaces\n$(literal)\n", "") -> ()
    | result -> failwithf "Process arguments lost their boundaries: %A" result
    printfn "PASS argument lists preserve spaces and literal shell syntax"

    let samples = [sample; { sample with Name = "another-oracle" }]
    check (selectSamples ["oracle"] samples = Ok samples) "Existing substring selection changed."
    for targets in [["missing"]; ["oracle"; "missing"]] do
        match selectSamples targets samples with
        | Error message -> check (message.Contains "missing") "Selection error lost the unmatched filter."
        | Ok selected -> failwithf "Unmatched filter selected an apparent passing run: %A" selected
    match selectSamples [] [] with
    | Error _ -> ()
    | Ok _ -> failwith "An empty manifest must not pass."
    printfn "PASS every requested sample filter must match"

    let runner = Path.Combine(__SOURCE_DIRECTORY__, "Runner.fsx")
    match runProcess "dotnet" ["fsi"; runner; "--"; "--sample"; "__missing_oracle__"] work None 20000 |> fst with
    | Completed (1, stdout, stderr) ->
        check (stderr.Contains "No manifest sample matches") "Runner failed for the wrong reason."
        check (not (stdout.Contains "Building compiler")) "An unmatched filter started a compiler build."
    | result -> failwithf "Runner did not propagate its failure as process exit 1: %A" result
    match runProcess "dotnet" ["fsi"; runner; "--"; "--help"] work None 20000 |> fst with
    | Completed (0, stdout, _) -> check (stdout.Contains "--sample") "Runner help was not executed."
    | result -> failwithf "Runner help did not exit successfully: %A" result
    printfn "PASS real CLI failure and success exit codes propagate"
    printfn "All regression harness tests passed."
finally
    Directory.Delete(work, true)
