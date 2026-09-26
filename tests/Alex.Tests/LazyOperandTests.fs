module Alex.Tests.LazyOperandTests

open Xunit
open Clef.Compiler.NativeService
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures
module Operands = Alex.Traversal.LazyOperands
module Zipper = Alex.Traversal.PSGZipper

let private ok = function Result.Ok value -> value | Result.Error reason -> failwith reason

let private platform : PlatformContext =
    { PlatformId = "lazy-operand-test"; Dimensions = Map.ofList ["Pointer", 64; "Register", 64]
      Representations = Map.empty; EndpointReturns = Map.empty
      PlatformLibraryPath = None; PlatformDescription = None; PlatformArchitecture = None; PlatformOS = None
      PlatformSourcePaths = Set.empty; Predicates = Map.empty; FreestandingStartup = None
      SubstrateKind = None; RuntimeModel = None; AvailableMemorySpaces = []; DefaultMemorySpace = None
      ClockFrequencyMhz = None; NsPerWeightUnit = None }

/// Real source instance, guard and lifetime proofs precede the component test.
/// The component supplies actual SSA operands; no new source authority is forged.
let private fixture () =
    let source = """
module LazyOperands
let make (value: bool) = lazy value
[<EntryPoint>]
let main _ =
    let first = make false
    let second = make true
    let alias = first
    ignore (Lazy.force alias)
    ignore (Lazy.force second)
    0
"""
    let raw =
        match parseAndCheck source "lazy-operands.clef" with
        | Success result -> result.Graph
        | other -> failwithf "Expected checked lazy fixture: %A" other
    let normalized = Clef.Compiler.Nanopass.LazyElaboration.normalize raw
    let prepared, _ = Clef.Compiler.Nanopass.LazyFactoryResults.prepare normalized normalized.Codata.Value.Curry
    let graph, reading = Clef.Compiler.Nanopass.LazyRuntime.settle { prepared with Platform = Some platform }
    Assert.Empty reading.Diagnostics
    let binding name = graph.Nodes.Values |> Seq.find (fun node ->
        node.IsReachable && match node.Kind with SemanticKind.Binding(actual, false, _, _) -> actual = name | _ -> false)
    graph, (binding "first").Id, (binding "second").Id, (binding "alias").Id

let private context graph occurrence accumulator =
    let scope = ref (Alex.Traversal.ScopeContext.ScopeContext.root ())
    let visited = ref Set.empty
    { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
      ScopeContext = scope; RootScopeContext = scope; Graph = graph
      Zipper = Zipper.create graph occurrence |> require "Missing lazy occurrence"
      GlobalVisited = visited; TraversalVisited = visited }

let private seed ctx occurrence environmentIndex =
    let shape = Operands.project ctx occurrence |> ok
    let code = { SSA = Arg 0; Type = Operands.functionType shape }
    let environment = { SSA = Arg environmentIndex; Type = Operands.environmentType shape }
    let value = Operands.create shape code environment |> ok
    MLIRAccumulator.bindLazy occurrence value ctx.Accumulator |> ok
    value

[<Fact>]
let ``same lazy schema preserves distinct actual environments and alias operands`` () =
    let graph, first, second, alias = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    let ctx = context graph alias accumulator
    let left, right = seed ctx first 1, seed ctx second 2
    Assert.Equal((Operands.layoutOf left).Owner, (Operands.layoutOf right).Owner)
    Assert.Equal(Operands.code left, Operands.code right)
    Assert.NotEqual((Operands.environment left).SSA, (Operands.environment right).SSA)
    let copied = Operands.reproject ctx first alias |> ok
    Assert.Equal<Val list>(Operands.values left, Operands.values copied)
    Assert.Equal(alias, Operands.occurrence copied)
    Assert.True((MLIRAccumulator.recallNode first accumulator).IsNone)
    Assert.Empty accumulator.AllOps

[<Fact>]
let ``operation scope restores the original lazy instance pair rather than schema lookup`` () =
    let graph, first, _, alias = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    let ctx = context graph alias accumulator
    let original = seed ctx first 1
    let saved = MLIRAccumulator.snapshotOperands accumulator
    let nested = seed ctx first 3
    Assert.Equal(Arg 3, (Operands.environment nested).SSA)
    MLIRAccumulator.restoreOperands saved accumulator
    let restored = Operands.reproject ctx first alias |> ok
    Assert.Equal<Val list>(Operands.values original, Operands.values restored)
    Assert.Empty accumulator.AllOps

[<Theory>]
[<InlineData("force-proof")>]
[<InlineData("layout-proof")>]
let ``changed lazy source proof retracts cached physical projection`` defect =
    let graph, first, _, alias = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    let ctx = context graph alias accumulator
    seed ctx first 1 |> ignore
    let removed = if defect = "force-proof" then EdgeRole.LazyMemoization else EdgeRole.LazyLayout
    let changed = { graph with Edges = graph.Edges |> List.filter (fun edge -> edge.Role <> removed) }
    let reading = Operands.project (context changed alias accumulator) alias
    Assert.True(Result.isError reading)
    Assert.Empty accumulator.AllOps

[<Fact>]
let ``physical thunk formal uses the exact environment extent of its lazy value`` () =
    let graph, first, _, _ = fixture ()
    let inputs: Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.Inputs =
        { Layouts = Map.empty; Origins = Map.empty; Known = Map.empty }
    let carriers, _ = Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.settle inputs graph
    let previous = graph.Codata.Value
    let graph = { graph with Codata = lazy { previous with CallableCarriers = carriers } }
    let ctx = context graph first (MLIRAccumulator.empty ())
    let shape = Operands.project ctx first |> ok
    let layout = Operands.contract shape
    let contract = Clef.Compiler.PSGSaturation.SemanticGraph.LazyValues.instance graph layout.Owner |> Option.get
    let actual = mapTypeAt contract.Formal graph.Nodes[contract.Formal].Type ctx
    Assert.Equal(Operands.environmentType shape, actual)
    match Operands.functionType shape with
    | TFunc([formal], _) -> Assert.Equal(actual, formal)
    | other -> failwithf "Expected one actual environment formal: %A" other
    let _, parameters, body = Alex.Traversal.CallableOperands.tryThunkDeclaration ctx contract.Thunk |> Option.get
    Assert.Equal(contract.ThunkBody, body)
    let _, _, formal = Assert.Single parameters
    Assert.Equal(contract.Formal, formal)
    for removed in [EdgeRole.LazyInstance; EdgeRole.LazyLayout; EdgeRole.LazyMemoization] do
        let changed = { graph with Edges = graph.Edges |> List.filter (fun edge -> edge.Role <> removed) }
        let changedContext = context changed first (MLIRAccumulator.empty ())
        Assert.True((Alex.Traversal.CallableOperands.tryThunkDeclaration changedContext contract.Thunk).IsNone)

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``only witnessed formation accounts for uninitialized cache declarations`` invalidated =
    let source = "module LazyCoverage\n[<EntryPoint>]\nlet main _ =\n    let value = lazy true\n    if Lazy.force value then 0 else 1\n"
    let raw =
        match parseAndCheck source "lazy-coverage.clef" with
        | Success result -> result.Graph
        | other -> failwithf "Expected checked lazy formation: %A" other
    let normalized = Clef.Compiler.Nanopass.LazyElaboration.normalize raw
    let graph, reading = Clef.Compiler.Nanopass.LazyRuntime.settle { normalized with Platform = Some platform }
    Assert.Empty reading.Diagnostics
    let _, layout = reading.Layouts |> Map.toList |> Assert.Single
    let contract = Clef.Compiler.PSGSaturation.SemanticGraph.LazyValues.instance graph layout.Owner |> Option.get
    let graph =
        if invalidated then { graph with Edges = graph.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.LazyLayout) }
        else graph
    let accumulator = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode contract.InitialComputed (Arg 0) (TInt(IntWidth 1)) accumulator
    let ctx = context graph contract.Environment accumulator
    let output = Alex.Witnesses.LazyWitness.nanopass.Witness ctx graph.Nodes[contract.Environment]
    if invalidated then
        match output.Result with TRError _ -> () | other -> failwithf "Stale layout was accepted: %A" other
        Assert.Empty ctx.GlobalVisited.Value
    else
        match output.Result with TRValue _ -> () | other -> failwithf "Formation was not witnessed: %A" other
        Assert.Equal<Set<NodeId>>(Set.ofList [layout.Computed; layout.Cached], ctx.GlobalVisited.Value)
        let stores =
            output.InlineOps |> List.choose (function
                | MLIROp.MemRefOp(MemRefOp.Store(value, view, _, _, _))
                | MLIROp.MemRefOp(MemRefOp.StoreAligned(value, view, _, _, _, _)) -> Some(value, view)
                | _ -> None)
        let value, view = Assert.Single stores
        Assert.Equal(Arg 0, value)
        let offset =
            output.InlineOps |> List.choose (function
                | MLIROp.MemRefOp(MemRefOp.View(actual, _, offset, _, _)) when actual = view -> Some offset
                | _ -> None) |> Assert.Single
        let computed = layout.Slots |> List.find (fun slot -> slot.Source = layout.Computed)
        Assert.Contains(output.InlineOps, function
            | MLIROp.ArithOp(ArithOp.ConstI(actual, bytes, TIndex)) -> actual = offset && bytes = int64 computed.Field.Offset.Value
            | _ -> false)
        Assert.Empty graph.Nodes[layout.Cached].Children
