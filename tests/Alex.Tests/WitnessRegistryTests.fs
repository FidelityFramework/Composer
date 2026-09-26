module Alex.Tests.WitnessRegistryTests

open System.Threading.Tasks
open Xunit
open Core.Types.Dialects
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.WitnessRegistry

[<Fact>]
let ``Independent target registries retain their own witness selection`` () =
    let names (registry: NanopassRegistry) = registry.Nanopasses |> List.map _.Name
    let cpu = createRegistry CPU
    let fabric = createRegistry FPGA
    let accelerator = createRegistry NPU
    Assert.Contains("Memory", names cpu)
    Assert.DoesNotContain("HardwareModule", names cpu)
    Assert.Contains("HardwareModule", names fabric)
    Assert.DoesNotContain("Memory", names fabric)
    Assert.Contains("KernelModule", names accelerator)
    Assert.DoesNotContain("HardwareModule", names accelerator)
    let expected = [| CPU, names cpu; FPGA, names fabric; NPU, names accelerator |]
    // Only registry construction is exercised concurrently here. Source
    // checking and the still-migrating type mapper are not called by this test.
    Parallel.For(0, 96, fun index ->
        let target, expectedNames = expected[index % expected.Length]
        let own = createRegistry target
        createRegistry (fst expected[(index + 1) % expected.Length]) |> ignore
        Assert.True(names own = expectedNames)) |> ignore
    Assert.True(names cpu = snd expected[0])
    Assert.True(names fabric = snd expected[1])
    Assert.True(names accelerator = snd expected[2])
