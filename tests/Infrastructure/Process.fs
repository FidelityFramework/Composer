module Tests.Process

open System
open System.Diagnostics
open System.IO

type Outcome = { ExitCode: int; Output: string }

/// Regression-test processes use direct argument arrays, never shell evaluation.
let run executable (arguments: string list) timeoutMs log =
    let info = ProcessStartInfo(executable, UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true)
    for argument in arguments do info.ArgumentList.Add argument
    use child = new Process(StartInfo = info)
    if not (child.Start()) then failwith ("Cannot start " + executable)
    let stdout, stderr = child.StandardOutput.ReadToEndAsync(), child.StandardError.ReadToEndAsync()
    let timedOut = not (child.WaitForExit(timeoutMs: int))
    if timedOut then child.Kill(true); child.WaitForExit()
    let output = stdout.GetAwaiter().GetResult() + stderr.GetAwaiter().GetResult()
    log |> Option.iter (fun path -> File.WriteAllText(path, output))
    if timedOut then failwithf "%s timed out after %d ms; log: %A" executable timeoutMs log
    { ExitCode = child.ExitCode; Output = output }

let requireSuccess executable args timeoutMs log =
    let result = run executable args timeoutMs log
    if result.ExitCode <> 0 then failwithf "%s failed (%d): %s" executable result.ExitCode result.Output
    result.Output
