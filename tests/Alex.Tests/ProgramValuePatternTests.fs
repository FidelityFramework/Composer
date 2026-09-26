module Alex.Tests.ProgramValuePatternTests

open Xunit
open Clef.Compiler.NativeService
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Patterns.MemRefPatterns
open Alex.Tests.Fixtures
module Startup = Clef.Compiler.PSGSaturation.SemanticGraph.ProgramInitialization
module Zipper = Alex.Traversal.PSGZipper

let private fixture () =
    let file = System.IO.Path.GetFullPath "program-value-pattern.clef"
    let context: PlatformContext = {
        PlatformId = "program-value-pattern"; Dimensions = Map.ofList ["Pointer", 64; "Register", 64]
        Representations = Map.empty; EndpointReturns = Map.empty; PlatformLibraryPath = None
        PlatformDescription = Some "ProgramValuePattern.description"; PlatformArchitecture = None; PlatformOS = None
        PlatformSourcePaths = Set.singleton file; Predicates = Map.empty; FreestandingStartup = None
        SubstrateKind = Some SubstrateKind.FPGA; RuntimeModel = None; AvailableMemorySpaces = []
        DefaultMemorySpace = None; ClockFrequencyMhz = None; NsPerWeightUnit = None }
    let source = """module ProgramValuePattern
 type MemorySpace = { Name: string; Kind: string; Capacity: int; Alignment: int; Granularity: int; Growth: string; Access: string; Base: int option }
 type ProgramLifetimeSpaces = { Immutable: string; Mutable: string option }
 type PlatformDescription = { Id: string; Spaces: MemorySpace array; ProgramLifetime: ProgramLifetimeSpaces option }
 let image = { Name = "image"; Kind = "rodata"; Capacity = 1024; Alignment = 16; Granularity = 16; Growth = "fixed"; Access = "r"; Base = None }
 let state = { Name = "state"; Kind = "data"; Capacity = 1024; Alignment = 16; Granularity = 16; Growth = "fixed"; Access = "rw"; Base = None }
 let description = { Id = "program-value-pattern"; Spaces = [| image; state |]; ProgramLifetime = Some { Immutable = "image"; Mutable = Some "state" } }
 let mutable flag = true
 [<EntryPoint>]
 let main _ = if flag then 0 else 1
"""
    let input = match parseStringWithDefaults source file with ParseSuccess input -> input | ParseError errors -> failwithf "Fixture parse failed: %A" errors
    let result = checkParsedInputsWithPlatform [input] (Some context)
    let errors = result.Diagnostics |> List.filter (fun diagnostic -> Diagnostic.effectiveSeverity diagnostic = NativeDiagnosticSeverity.Error)
    Assert.Empty errors
    let plan = Startup.read result.Graph |> require "Fixture startup was not settled"
    let binding = plan.ValueBindings |> Seq.filter (fun id ->
        match result.Graph.Nodes[id].Kind with SemanticKind.Binding("flag", true, _, _) -> true | _ -> false) |> Assert.Single
    Assert.True((Startup.tryValueAuthority result.Graph binding).IsSome)
    result.Graph, binding

[<Fact>]
let ``program slot initialization consumes the declared writable authority`` () =
    let graph, binding = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    let focus = Zipper.create graph binding |> require "Missing slot fixture"
    let ty = TInt(IntWidth 1)
    let ops =
        match matchAt (pGlobalSlotInit binding "program_slot" (Arg 0) ty) focus 64 accumulator with
        | Result.Ok ((ops, TRValue _), _) -> ops
        | result -> failwithf "Program slot initialization failed: %A" result
    let globals = MLIRAccumulator.drainPendingStaticGlobals accumulator
    Assert.Single globals |> ignore
    let zero = V(-2, 0)
    let resultType = TInt(IntWidth 32)
    let body = ops @ [MLIROp.ArithOp(ArithOp.ConstI(zero, 0L, resultType)); MLIROp.FuncOp(FuncOp.Return([{ SSA = zero; Type = resultType }]))]
    let functionOp = MLIROp.FuncOp(FuncOp.FuncDef("initialize_program_slot", [Arg 0, ty], [resultType], body, FuncVisibility.Public))
    let source = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "program_slot_component" (globals @ [functionOp])
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] source
    Assert.Contains("memref.global", verified)
    Assert.Contains("memref.store", verified)

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``missing or mismatched authority preserves slot intent and emits no storage`` malformed =
    let graph, binding = fixture ()
    let edges = graph.Edges |> List.choose (fun edge ->
        if edge.Role <> EdgeRole.ProgramValue then Some edge
        elif malformed then Some { edge with Sources = edge.Sources |> List.rev }
        else None)
    let graph = { graph with Edges = edges }
    Assert.True(Startup.isSlotBinding graph binding)
    let accumulator = MLIRAccumulator.empty ()
    let focus = Zipper.create graph binding |> require "Missing slot fixture"
    match matchAt (pGlobalSlotInit binding "program_slot" (Arg 0) (TInt(IntWidth 1))) focus 64 accumulator with
    | Result.Error message -> Assert.Contains("writable-space authority", message)
    | result -> failwithf "Expected a missing authority boundary: %A" result
    Assert.Empty(MLIRAccumulator.drainPendingStaticGlobals accumulator)
