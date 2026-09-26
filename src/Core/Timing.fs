/// Compilation-owned monotonic timings. Independent workers share a session,
/// never a current-phase slot. Nested and overlapping spans remain independent.
module Core.Timing

open System
open System.Diagnostics
open System.Threading

type PhaseTiming = {
    Id: int64
    Name: string
    Description: string
    StartTime: DateTimeOffset
    StartOffsetTicks: int64
    ElapsedTicks: int64
} with
    member this.ElapsedMs = float this.ElapsedTicks * 1000.0 / float Stopwatch.Frequency

/// Host-side instrumentation, independent of source identities and semantics.
/// The output callback is serialized within this session; source compilation
/// and timed work never run under the instrumentation lock.
type TimingSession(enabled: bool, output: string -> unit) =
    let started = Stopwatch.GetTimestamp()
    let gate = obj()
    let phases = ResizeArray<PhaseTiming>()
    let mutable nextId = 0L
    let mutable active = 0
    let mutable completed: int64 option = None

    member _.Enabled = enabled

    member _.StartPhase(name: string, description: string) : IDisposable =
        if not enabled then { new IDisposable with member _.Dispose() = () }
        else
            let id, start, wall = lock gate (fun () ->
                if completed.IsSome then invalidOp "A completed timing session cannot accept new spans."
                nextId <- nextId + 1L
                active <- active + 1
                let start = Stopwatch.GetTimestamp()
                let wall = DateTimeOffset.UtcNow
                output (sprintf "[%s] [%s:%d] %s..." (wall.ToString("HH:mm:ss.fff")) name nextId description)
                nextId, start, wall)
            let mutable disposed = 0
            { new IDisposable with
                member _.Dispose() =
                    if Interlocked.Exchange(&disposed, 1) = 0 then
                        let ended = Stopwatch.GetTimestamp()
                        let phase = {
                            Id = id; Name = name; Description = description; StartTime = wall
                            StartOffsetTicks = start - started; ElapsedTicks = ended - start }
                        lock gate (fun () ->
                            phases.Add phase
                            active <- active - 1
                            output (sprintf "[%s:%d] Done (%.3fms)" name id phase.ElapsedMs)) }

    /// Call only after all workers and reconciliation have finished. Refusing
    /// active spans prevents a partial sample being presented as a full run.
    member _.Complete() =
        lock gate (fun () ->
            if active <> 0 then invalidOp "Cannot complete timing while worker spans are active."
            if completed.IsNone then completed <- Some (Stopwatch.GetTimestamp()))

    member _.Phases = lock gate (fun () -> phases |> Seq.sortBy _.Id |> Seq.toList)
    member _.ActiveSpans = lock gate (fun () -> active)
    member _.WallElapsedMs =
        lock gate (fun () ->
            let ended = completed |> Option.defaultWith Stopwatch.GetTimestamp
            float (ended - started) * 1000.0 / float Stopwatch.Frequency)

let create enabled = TimingSession(enabled, printfn "%s")
let silent () = TimingSession(false, ignore)

let timePhase (session: TimingSession) name description f =
    use _span = session.StartPhase(name, description)
    f()

/// Raw observations for comparative runs. The owner supplies unique output
/// directories and records inputs/toolchain/cache conditions with these spans.
let writeReport path (session: TimingSession) =
    session.Complete()
    let phases = session.Phases
    let report =
        {| SchemaVersion = 1
           Enabled = session.Enabled
           StopwatchFrequency = Stopwatch.Frequency
           WallElapsedMilliseconds = session.WallElapsedMs
           SumOfSpanMilliseconds = phases |> List.sumBy _.ElapsedMs
           Phases = phases |> List.map (fun phase ->
               {| Id = phase.Id; Name = phase.Name; Description = phase.Description; StartTime = phase.StartTime
                  StartOffsetTicks = phase.StartOffsetTicks; ElapsedTicks = phase.ElapsedTicks; ElapsedMilliseconds = phase.ElapsedMs |}) |> List.toArray |}
    let options = System.Text.Json.JsonSerializerOptions(WriteIndented = true)
    System.IO.File.WriteAllText(path, System.Text.Json.JsonSerializer.Serialize(report, options))

/// Wall time is measured, never obtained by adding overlapping phase durations.
/// Phase sums describe instrumented span time, not CPU time or elapsed latency.
let printSummary (session: TimingSession) =
    session.Complete()
    if session.Enabled then
        printfn ""
        printfn "Compilation timing"
        for phase in session.Phases do
            printfn "  %-24s %10.3f ms  %s" phase.Name phase.ElapsedMs phase.Description
        let phaseSum = session.Phases |> List.sumBy _.ElapsedMs
        printfn "  Wall elapsed             %10.3f ms" session.WallElapsedMs
        printfn "  Sum of phase spans       %10.3f ms (may overlap)" phaseSum
        printfn ""
