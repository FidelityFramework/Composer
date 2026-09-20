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
