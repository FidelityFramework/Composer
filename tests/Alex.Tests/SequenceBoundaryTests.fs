module Alex.Tests.SequenceBoundaryTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.ScopeContext
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper
module SourceChecker = Clef.Compiler.NativeService

/// An owner with its typed generator formal, but no suspension segments,
/// frame or resumption construction. A delimiter adds ownership only.
let private fixture delegated owned =
    let builder = NodeBuilder()
    let sequenceType = Types.mkSeqType Types.boolType
    let payload = builder.Create(SemanticKind.PatternBinding "input",
                                 (if delegated then sequenceType else Types.boolType), dummyRange)
    let site = builder.Create((if delegated then SemanticKind.YieldBang payload.Id else SemanticKind.Yield payload.Id),
                              Types.unitType, dummyRange, children = [payload.Id])
    let formalType = NativeType.TNativePtr sequenceType
    let formal = builder.Create(SemanticKind.PatternBinding "_seq_ptr", formalType, dummyRange)
    let generator = builder.Create(
        SemanticKind.Lambda(["_seq_ptr", formalType, formal.Id], site.Id, [], None, LambdaContext.SeqGenerator),
        NativeType.TFun(formalType, Types.boolType), dummyRange, children = [formal.Id; site.Id])
    let owner = builder.Create(SemanticKind.SeqExpr(generator.Id, []), sequenceType, dummyRange, children = [generator.Id])
    let binding = builder.Create(SemanticKind.Binding("values", false, false, None), sequenceType, dummyRange, children = [owner.Id])
    for child, parent in [payload.Id, site.Id; site.Id, generator.Id; formal.Id, generator.Id; generator.Id, owner.Id; owner.Id, binding.Id] do
        builder.SetParent(child, parent)
    let raw = builder.Build []
    let edges =
        if owned then
            [{ Sources = [owner.Id; generator.Id]; Target = site.Id
               Class = EdgeClass.Suspension; Role = EdgeRole.Delimiter; Ordinal = 0 }]
        else []
    let graph = { raw with Edges = edges }
    let position = Zipper.create graph binding.Id |> require "Missing sequence binding" |> atChild owner.Id
    position, generator.Id, site.Id, payload.Id

let private observe (position: Zipper.PSGZipper) =
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode position.Focus.Id (Arg 0) (TInt(IntWidth 1)) operands
    let nodes, types = operands.NodeAssoc, operands.SSATypes
    let rootScope = ref (ScopeContext.root ())
    let scope = ref (ScopeContext.createChild rootScope.Value FunctionLevel)
    let originalRoot, originalScope = rootScope.Value, scope.Value
    let visited = ref Set.empty
    let ctx: WitnessContext =
        { Coeffects = coeffects position.Graph 64
          Accumulator = operands; RootAccumulator = operands
          ScopeContext = scope; RootScopeContext = rootScope
          Graph = position.Graph; Zipper = position
          GlobalVisited = visited; TraversalVisited = visited }
    let output = Alex.Witnesses.SeqWitness.nanopass.Witness ctx position.Focus
    Assert.Empty(output.InlineOps)
    Assert.Empty(output.TopLevelOps)
    Assert.Same(position.Graph, ctx.Graph)
    Assert.Same(position.Graph.Nodes, ctx.Graph.Nodes)
    Assert.Same(position.Graph.Edges, ctx.Graph.Edges)
    Assert.Same(position, ctx.Zipper)
    Assert.Same(position.Path, ctx.Zipper.Path)
    Assert.Same(nodes, operands.NodeAssoc)
    Assert.Same(types, operands.SSATypes)
    Assert.Same(originalRoot, rootScope.Value)
    Assert.Same(originalScope, scope.Value)
    Assert.Empty(visited.Value)
    Assert.Empty(operands.AllOps)
    Assert.Empty(operands.Errors)
    Assert.Empty(operands.EmittedGlobals)
    Assert.Empty(operands.EmittedStaticGlobals)
    Assert.Empty(operands.PendingStaticGlobals)
    Assert.Empty(operands.DeferredInlineOps)
    output.Result

let private focusWithin (graph: SemanticGraph) owner site =
    let rec pathToOwner id path =
        if id = owner then path
        else
            let parent = graph.Nodes[id].Parent |> require "Sequence site lost its structural parent"
            pathToOwner parent (id :: path)
    let root = Zipper.create graph owner |> require "Missing checked sequence owner"
    pathToOwner site [] |> List.fold (fun position child -> atChild child position) root

[<Theory>]
[<InlineData("SeqExpr", false)>]
[<InlineData("SeqExpr", true)>]
[<InlineData("Yield", false)>]
[<InlineData("Yield", true)>]
[<InlineData("YieldBang", false)>]
[<InlineData("YieldBang", true)>]
let ``suspension boundaries require more than owner identity`` kind owned =
    let owner, generator, site, _ = fixture (kind = "YieldBang") owned
    if owned then
        let edge = Assert.Single owner.Graph.Edges
        Assert.Equal<NodeId list>([owner.Focus.Id; generator], edge.Sources)
        Assert.Equal(site, edge.Target)
    let position = if kind = "SeqExpr" then owner else owner |> atChild generator |> atChild site
    Assert.NotEmpty(position.Path)
    match observe position with
    | TRError diagnostic ->
        Assert.Equal(kind + " requires Baker-settled suspension segments, frame and resumption; delimiter ownership alone is insufficient", diagnostic.Message)
    | result -> failwithf "Unelaborated suspension was accepted: %A" result

[<Fact>]
let ``unrelated node remains available to other witnesses`` () =
    let owner, generator, site, payload = fixture false true
    let position = owner |> atChild generator |> atChild site |> atChild payload
    match observe position with
    | TRSkip -> ()
    | result -> failwithf "Sequence witness consumed an unrelated node: %A" result

[<Fact>]
let ``checked delegation retains origin and ownership without authorizing frame-less witnessing`` () =
    let source = """module SequenceBoundary
let input = seq { yield true }
let outer = seq { yield! input }
[<EntryPoint>]
let main _ = ignore outer; 0
"""
    let graph =
        match SourceChecker.parseAndCheck source "sequence-boundary.clef" with
        | SourceChecker.Success result -> result.Graph
        | SourceChecker.CheckFailure result -> failwithf "Delegation source was rejected: %A" result.Diagnostics
        | SourceChecker.ParseFailure errors -> failwithf "Delegation source did not parse: %A" errors
    Assert.DoesNotContain(graph.Nodes.Values, fun node ->
        match node.Kind with SemanticKind.Error _ -> true | _ -> false)
    let origin = graph.Edges |> List.filter (fun edge ->
        edge.Class = EdgeClass.Provenance && edge.Role = EdgeRole.DelegationOrigin) |> Assert.Single
    match origin.Sources with
    | [wrapper; input] ->
        Assert.Equal<NativeType>(Types.unitType, graph.Nodes[wrapper].Type)
        Assert.Equal<NativeType>(Types.mkSeqType Types.boolType, graph.Nodes[input].Type)
        match graph.Nodes[wrapper].Kind with
        | SemanticKind.Sequential [_] -> ()
        | kind -> failwithf "Delegation lost its source wrapper: %A" kind
    | sources -> failwithf "Delegation lost its source/input provenance: %A" sources
    match graph.Nodes[origin.Target].Kind with
    | SemanticKind.Yield current -> Assert.Equal<NativeType>(Types.boolType, graph.Nodes[current].Type)
    | kind -> failwithf "Baker did not produce a typed delegated yield: %A" kind
    let ownership = graph.Edges |> List.filter (fun edge ->
        edge.Class = EdgeClass.Suspension && edge.Role = EdgeRole.Delimiter && edge.Target = origin.Target) |> Assert.Single
    let owner =
        match ownership.Sources with
        | [owner; generator] ->
            match graph.Nodes[owner].Kind with
            | SemanticKind.SeqExpr(actual, _) -> Assert.Equal(generator, actual)
            | kind -> failwithf "Delegated yield has no sequence owner: %A" kind
            owner
        | sources -> failwithf "Delegated yield lost its owner/generator: %A" sources
    let position = focusWithin graph owner origin.Target
    Assert.NotEmpty(position.Path)
    match observe position with
    | TRError diagnostic ->
        Assert.Equal("Yield requires Baker-settled suspension segments, frame and resumption; delimiter ownership alone is insufficient", diagnostic.Message)
    | result -> failwithf "Ownership and delegation provenance authorized an unsettled frame: %A" result

[<Fact>]
let ``checked guarded loop retains evaluation facts without authorizing frame-less witnessing`` () =
    let source = """module SequenceEvaluationBoundary
let outer (gate: bool) = seq {
    let mutable active = true
    while active do
        if gate && active then yield gate
        active <- false
}
[<EntryPoint>]
let main _ = ignore (outer true); 0
"""
    let graph =
        match SourceChecker.parseAndCheck source "sequence-evaluation-boundary.clef" with
        | SourceChecker.Success result -> result.Graph
        | SourceChecker.CheckFailure result -> failwithf "Guarded sequence was rejected: %A" result.Diagnostics
        | SourceChecker.ParseFailure errors -> failwithf "Guarded sequence did not parse: %A" errors
    Assert.DoesNotContain(graph.Nodes.Values, fun node ->
        match node.Kind with SemanticKind.Error _ -> true | _ -> false)
    let site = graph.Nodes.Values |> Seq.filter (fun node ->
        match node.Kind with SemanticKind.Yield _ -> true | _ -> false) |> Assert.Single
    let payload = match site.Kind with SemanticKind.Yield value -> value | _ -> failwith "Missing yield"
    let ownership = graph.Edges |> List.filter (fun edge ->
        edge.Class = EdgeClass.Suspension && edge.Role = EdgeRole.Delimiter && edge.Target = site.Id) |> Assert.Single
    let owner = List.head ownership.Sources
    let facts = graph.Edges |> List.filter (fun edge -> edge.Class = EdgeClass.Evaluation)
    let flow target fromPort toPort transfer =
        facts |> List.filter (fun edge ->
            edge.Target = target && edge.Role = EdgeRole.EvaluationFlow(fromPort, toPort, transfer)) |> Assert.Single
    let demand = facts |> List.filter (fun edge ->
        edge.Target = site.Id && edge.Role = EdgeRole.EvaluationOperand EvaluationAccess.Value) |> Assert.Single
    Assert.Equal(0, demand.Ordinal)
    Assert.Equal<NodeId list>([owner; payload], demand.Sources)
    let ready = flow site.Id (EvaluationPort.OperandExit 0) EvaluationPort.Ready EvaluationTransfer.Continue
    Assert.Equal<NodeId list>([owner; payload], ready.Sources)
    let resume = flow site.Id EvaluationPort.Ready EvaluationPort.Exit EvaluationTransfer.Resume
    Assert.Equal<NodeId list>([owner], resume.Sources)
    let loop = graph.Nodes.Values |> Seq.filter (fun node ->
        match node.Kind with SemanticKind.WhileLoop _ -> true | _ -> false) |> Assert.Single
    match loop.Kind with
    | SemanticKind.WhileLoop(guard, body) ->
        let back = flow loop.Id (EvaluationPort.OperandExit 1) (EvaluationPort.OperandEntry 0) EvaluationTransfer.Continue
        Assert.Equal<NodeId list>([owner; body; guard], back.Sources)
        flow loop.Id (EvaluationPort.OperandExit 0) (EvaluationPort.OperandEntry 1) EvaluationTransfer.WhenTrue |> ignore
        flow loop.Id (EvaluationPort.OperandExit 0) EvaluationPort.Ready EvaluationTransfer.WhenFalse |> ignore
    | _ -> failwith "Missing loop"
    let position = focusWithin graph owner site.Id
    Assert.NotEmpty(position.Path)
    match observe position with
    | TRError diagnostic ->
        Assert.Equal("Yield requires Baker-settled suspension segments, frame and resumption; delimiter ownership alone is insufficient", diagnostic.Message)
    | result -> failwithf "Local evaluation contracts authorized an unsettled frame: %A" result
