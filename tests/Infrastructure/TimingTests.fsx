#load "../../src/Core/Timing.fs"

open System
open System.Threading.Tasks
open Core.Timing

let check condition message = if not condition then failwith message
let refuses action =
    try action(); false
    with :? InvalidOperationException -> true

// No compiler output lease: this exercises only the production host timer.
let session = TimingSession(true, ignore)
let first = session.StartPhase("unit", "first segment")
let second = session.StartPhase("unit", "second segment")
check (session.ActiveSpans = 2) "Starting a second phase stopped the first."
check (refuses session.Complete) "A live worker was counted as a completed run."
second.Dispose()
check (session.ActiveSpans = 1) "A span completed a different worker's span."
check (session.Phases |> List.map _.Description = ["second segment"]) "A completed span lost its own identity."
first.Dispose()
first.Dispose()
session.Complete()
check (session.Phases.Length = 2) "Disposing twice duplicated a phase."
check (session.Phases |> List.map _.Description = ["first segment"; "second segment"]) "Observation order was replaced by worker completion order."
check (refuses (fun () -> session.StartPhase("late", "late") |> ignore)) "Completed session accepted new work."
let finishedWall = session.WallElapsedMs
session.Complete()
check (session.WallElapsedMs = finishedWall) "Repeated completion changed the timing sample."

let concurrent = TimingSession(true, ignore)
let other = TimingSession(true, ignore)
let release = TaskCompletionSource<unit>(TaskCreationOptions.RunContinuationsAsynchronously)
let ready = TaskCompletionSource<unit>(TaskCreationOptions.RunContinuationsAsynchronously)
let mutable started = 0
let counts = obj()
let workers =
    [| for index in 0 .. 31 -> Task.Run(Func<Task>(fun () -> (task {
        use _span = concurrent.StartPhase("segment", string index)
        lock counts (fun () ->
            started <- started + 1
            if started = 32 then ready.SetResult(()))
        do! release.Task
        if index % 2 = 0 then
            use _nested = concurrent.StartPhase("verify", string index)
            () } :> Task))) |]
try
    ready.Task.WaitAsync(TimeSpan.FromSeconds 10.0).GetAwaiter().GetResult()
    check (concurrent.ActiveSpans = 32) "Parallel spans shared a current-phase slot."
    use _other = other.StartPhase("independent", "another compilation")
    check (concurrent.Phases.IsEmpty) "Another compilation altered this session."
finally
    release.TrySetResult(()) |> ignore
Task.WhenAll(workers).WaitAsync(TimeSpan.FromSeconds 10.0).GetAwaiter().GetResult() |> ignore
concurrent.Complete()
other.Complete()
check (concurrent.Phases.Length = 48 && other.Phases.Length = 1) "Concurrent compilation observations were lost or mixed."
check (concurrent.Phases |> List.map _.Id |> Set.ofList |> Set.count = 48) "Concurrent spans reused an identity."
check (concurrent.Phases |> List.forall (fun p -> p.StartOffsetTicks >= 0L && p.ElapsedTicks >= 0L)) "Monotonic duration was negative."

let disabled = silent()
let ignored = disabled.StartPhase("disabled", "disabled")
ignored.Dispose()
disabled.Complete()
check disabled.Phases.IsEmpty "Disabled timing recorded work."
let failed = TimingSession(true, ignore)
try timePhase failed "failed" "failed phase" (fun () -> invalidOp "deliberate")
with :? InvalidOperationException -> ()
failed.Complete()
check (failed.Phases.Length = 1 && failed.ActiveSpans = 0) "A throwing phase left the session active."
let report = System.IO.Path.GetTempFileName()
try
    writeReport report concurrent
    use json = System.Text.Json.JsonDocument.Parse(System.IO.File.ReadAllText report)
    check (json.RootElement.GetProperty("Phases").GetArrayLength() = 48) "Machine report lost spans."
    check (json.RootElement.GetProperty("WallElapsedMilliseconds").GetDouble() = concurrent.WallElapsedMs) "Machine report substituted a span sum for elapsed time."
finally
    System.IO.File.Delete report
printfn "PASS: overlapping, concurrent, isolated, disabled, failed and completed timing sessions"
