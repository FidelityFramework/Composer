namespace Alex.Tests

open Xunit

// Source-backed component fixtures share CCS's process-local variable supplies
// and node identities. Parallel check jobs run in separate processes; xUnit
// must not interleave compiler requests or builders inside one process.
[<assembly: CollectionBehavior(DisableTestParallelization = true)>]
do ()
