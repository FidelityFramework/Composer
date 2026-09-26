module Alex.Tests.LazyTransportTests

open Xunit
open Clef.Compiler.NativeService
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures
open Alex.Patterns.LazyPatterns
module Operands = Alex.Traversal.LazyOperands
module Zipper = Alex.Traversal.PSGZipper

let private ok = function Result.Ok value -> value | Result.Error reason -> failwith reason

let private platform : PlatformContext =
    { PlatformId = "lazy-transport-test"; Dimensions = Map.ofList ["Pointer", 64; "Register", 64]
      Representations = Map.empty; EndpointReturns = Map.empty
      PlatformLibraryPath = None; PlatformDescription = None; PlatformArchitecture = None; PlatformOS = None
      PlatformSourcePaths = Set.empty; Predicates = Map.empty; FreestandingStartup = None
      SubstrateKind = None; RuntimeModel = None; AvailableMemorySpaces = []; DefaultMemorySpace = None
      ClockFrequencyMhz = None; NsPerWeightUnit = None }

/// Source settlement proves one schema, two caller-owned instances and all
/// force/lifetime obligations. Only already witnessed SSA values are supplied.
let private fixture () =
    let source = """
module LazyTransport
let make (value: bool) = lazy value
[<EntryPoint>]
let main _ =
    let first = make false
    let second = make true
    let forwarded =
        let saved = first
        (saved: Lazy<bool>)
    let chosen = if Lazy.force first then forwarded else second
    ignore (Lazy.force chosen)
    0
"""
    let raw =
        match parseAndCheck source "lazy-transport.clef" with
        | Success result -> result.Graph
        | other -> failwithf "Expected checked lazy transport fixture: %A" other
    let normalized = Clef.Compiler.Nanopass.LazyElaboration.normalize raw
    let prepared, _ = Clef.Compiler.Nanopass.LazyFactoryResults.prepare normalized normalized.Codata.Value.Curry
    let graph, reading = Clef.Compiler.Nanopass.LazyRuntime.settle { prepared with Platform = Some platform }
    Assert.Empty reading.Diagnostics
    let binding name = graph.Nodes.Values |> Seq.find (fun node ->
        node.IsReachable && match node.Kind with SemanticKind.Binding(actual, false, _, _) -> actual = name | _ -> false)
    let saved = binding "saved"
    let block = (binding "forwarded").Children |> Assert.Single
    let annotation =
        match graph.Nodes[block].Kind with SemanticKind.Sequential values -> List.last values | kind -> failwithf "Expected source block: %A" kind
    let reference =
        match graph.Nodes[annotation].Kind with SemanticKind.TypeAnnotation(value, _) -> value | kind -> failwithf "Expected source annotation: %A" kind
    let choice = (binding "chosen").Children |> Assert.Single
    let condition, left, right =
        match graph.Nodes[choice].Kind with SemanticKind.IfThenElse(condition, left, Some right) -> condition, left, right | _ -> failwith "Expected source choice"
    graph, saved.Id, reference, annotation, block, condition, left, right, choice

let private context graph position accumulator =
    let scope = ref (Alex.Traversal.ScopeContext.ScopeContext.root ())
    let visited = ref Set.empty
    { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
      ScopeContext = scope; RootScopeContext = scope; Graph = graph; Zipper = position
      GlobalVisited = visited; TraversalVisited = visited }

let private seed ctx occurrence codeIndex environmentIndex =
    let shape = Operands.project ctx occurrence |> ok
    let code = { SSA = Arg codeIndex; Type = Operands.functionType shape }
    let environment = { SSA = Arg environmentIndex; Type = Operands.environmentType shape }
    let value = Operands.create shape code environment |> ok
    MLIRAccumulator.bindLazy occurrence value ctx.Accumulator |> ok
    value

[<Fact>]
let ``passive lazy witnesses preserve the actual pair at each nested Huet occurrence`` () =
    let graph, saved, reference, annotation, block, _, _, _, _ = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    let root = Zipper.create graph block |> require "Missing source block"
    let original = seed (context graph root accumulator) (Assert.Single graph.Nodes[saved].Children) 0 1
    let witnesses =
        [saved, atChild saved root, Alex.Witnesses.BindingWitness.nanopass
         reference, atChild annotation root |> atChild reference, Alex.Witnesses.VarRefWitness.nanopass
         annotation, atChild annotation root, Alex.Witnesses.TypeAnnotationWitness.nanopass
         block, root, Alex.Witnesses.StructuralWitness.nanopass]
    for occurrence, position, witness in witnesses do
        let ctx = context graph position accumulator
        let output = witness.Witness ctx graph.Nodes[occurrence]
        match output.Result with
        | TRLazy value ->
            Assert.Equal(occurrence, Operands.occurrence value)
            Assert.Equal<Val list>(Operands.values original, Operands.values value)
            MLIRAccumulator.bindLazy occurrence value accumulator |> ok
        | other -> failwithf "Lost actual lazy pair: %A" other
        Assert.Empty output.InlineOps
        Assert.Empty output.TopLevelOps
        Assert.True((MLIRAccumulator.recallNode occurrence accumulator).IsNone)
        Assert.Same(position, ctx.Zipper)
        Assert.Empty ctx.TraversalVisited.Value

[<Fact>]
let ``lazy forwarding rejects a scalar substitute and a foreign Huet occurrence`` () =
    let graph, saved, reference, annotation, block, _, _, _, _ = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    let root = Zipper.create graph block |> require "Missing source block"
    let position = atChild saved root
    let ctx = context graph position accumulator
    let source = Assert.Single graph.Nodes[saved].Children
    let shape = Operands.project ctx source |> ok
    MLIRAccumulator.bindNode source (Arg 0) (Operands.environmentType shape) accumulator
    Assert.True(Result.isError (matchAt (pLazyForward ctx source) position 64 accumulator))
    seed ctx source 1 2 |> ignore
    let foreign = atChild annotation root |> atChild reference
    Assert.True(Result.isError (matchAt (pLazyForward ctx source) foreign 64 accumulator))
    Assert.Empty accumulator.AllOps

let private yields = function
    | [MLIROp.SCFOp(SCFOp.Yield values)] -> values
    | operations -> failwithf "Unexpected lazy arm operations: %A" operations

let private pair value = Operands.values value |> List.map (fun value -> value.SSA, value.Type)

[<Fact>]
let ``lazy conditional selects both actual components from the same runtime instance`` () =
    let graph, _, _, _, _, _, left, right, choice = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    let position = Zipper.create graph choice |> require "Missing conditional"
    let ctx = context graph position accumulator
    let first, second = seed ctx left 0 1, seed ctx right 2 3
    Assert.Equal((Operands.layoutOf first).Owner, (Operands.layoutOf second).Owner)
    Assert.NotEqual((Operands.environment first).SSA, (Operands.environment second).SSA)
    let condition = { SSA = Arg 4; Type = TInt(IntWidth 1) }
    match matchAt (pLazyConditional ctx condition left [] right []) position 64 accumulator with
    | Result.Ok (([MLIROp.IndexOp(IndexOp.IndexCastU(selector, test, _, TIndex));
                    MLIROp.SCFOp(SCFOp.IndexSwitch(actual, cases, fallback, results))], TRLazy value), _) ->
        Assert.Equal(condition.SSA, test)
        Assert.Equal(selector, actual)
        let label, branch = Assert.Single cases
        Assert.Equal(1L, label)
        Assert.Equal<(SSA * MLIRType) list>(pair first, yields branch)
        Assert.Equal<(SSA * MLIRType) list>(pair second, yields fallback)
        Assert.Equal<(SSA * MLIRType) list>(pair value, results)
    | other -> failwithf "Conditional did not retain the selected lazy pair: %A" other
    Assert.Empty ctx.TraversalVisited.Value

[<Fact>]
let ``lazy dispatch preserves each arm pair and refuses an unwitnessed alternative`` () =
    let graph, _, _, _, _, selector, left, right, choice = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    let position = Zipper.create graph choice |> require "Missing choice"
    let ctx = context graph position accumulator
    let first, second = seed ctx left 0 1, seed ctx right 2 3
    MLIRAccumulator.bindNode selector (Arg 4) TIndex accumulator
    match matchAt (pLazyDispatch ctx selector [7, left, []] (right, [])) position 64 accumulator with
    | Result.Ok (([MLIROp.SCFOp(SCFOp.IndexSwitch(actual, cases, fallback, results))], TRLazy value), _) ->
        Assert.Equal(Arg 4, actual)
        let label, branch = Assert.Single cases
        Assert.Equal(7L, label)
        Assert.Equal<(SSA * MLIRType) list>(pair first, yields branch)
        Assert.Equal<(SSA * MLIRType) list>(pair second, yields fallback)
        Assert.Equal<(SSA * MLIRType) list>(pair value, results)
    | other -> failwithf "Dispatch did not preserve its complete pairs: %A" other
    let missing = MLIRAccumulator.empty ()
    let missingCtx = context graph position missing
    seed missingCtx left 0 1 |> ignore
    MLIRAccumulator.bindNode selector (Arg 4) TIndex missing
    Assert.True(Result.isError (matchAt (pLazyDispatch missingCtx selector [7, left, []] (right, [])) position 64 missing))

[<Fact>]
let ``lazy witness forwarding retracts when the destination layout proof is removed`` () =
    let graph, saved, _, _, block, _, _, _, _ = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    let root = Zipper.create graph block |> require "Missing block"
    let source = Assert.Single graph.Nodes[saved].Children
    seed (context graph root accumulator) source 0 1 |> ignore
    let changed = { graph with Edges = graph.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.LazyLayout) }
    let position = Zipper.create changed block |> require "Missing changed block" |> atChild saved
    let ctx = context changed position accumulator
    let output = Alex.Witnesses.BindingWitness.nanopass.Witness ctx changed.Nodes[saved]
    match output.Result with
    | TRError _ -> ()
    | other -> failwithf "A stale source layout was reconstructed by passive forwarding: %A" other
    Assert.Empty output.InlineOps

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``validated lazy code is globally observed once while formations remain local`` invalidated =
    let original, _, _, _, _, _, _, _, _ = fixture ()
    let layout = original.Codata.Value.LazyLayouts.Values |> Assert.Single
    let graph =
        if invalidated then
            { original with Edges = original.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.LazyLayout) }
        else original
    let accumulator = MLIRAccumulator.empty ()
    let globalVisited = ref Set.empty
    let position = Zipper.create graph layout.Owner |> require "Missing formation"
    let observations = ResizeArray<NodeId>()
    // This is a traversal-ownership check. The real source contract supplies
    // the code/value distinction; the observer does not pretend to emit MLIR.
    let observer (_: WitnessContext) (node: SemanticNode) =
        observations.Add node.Id
        WitnessOutput.empty
    for _ in [1; 2] do
        let ctx = { context graph position accumulator with GlobalVisited = globalVisited }
        Alex.Traversal.NanopassArchitecture.visitAllNodes observer ctx position.Focus ctx.TraversalVisited
    Assert.Equal(2, observations |> Seq.filter ((=) layout.Owner) |> Seq.length)
    Assert.Equal((if invalidated then 2 else 1), observations |> Seq.filter ((=) layout.Thunk) |> Seq.length)
