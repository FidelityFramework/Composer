module Alex.Tests.TraversalOccurrenceTests

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

let private context graph position visited =
    let accumulator = MLIRAccumulator.empty ()
    let scope = ref (ScopeContext.root ())
    { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
      ScopeContext = scope; RootScopeContext = scope; Graph = graph; Zipper = position
      GlobalVisited = visited; TraversalVisited = visited }

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``driver rejects a foreign occurrence even if its node was previously visited`` alreadyVisited =
    let builder = NodeBuilder()
    let value = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let root = builder.Create(SemanticKind.Sequential [value.Id], Types.boolType, dummyRange)
    let graph = builder.Build [] |> prepareSource
    let position = Zipper.create graph root.Id |> require "Missing root"
    let initial = if alreadyVisited then Set.singleton value.Id else Set.empty
    let visited = ref initial
    let ctx = context graph position visited
    let mutable observed = false
    visitAllNodes (fun _ _ -> observed <- true; WitnessOutput.skip) ctx value visited
    Assert.False observed
    Assert.Equal<Set<NodeId>>(initial, visited.Value)
    Assert.Single ctx.Accumulator.Errors |> ignore
    Assert.Empty (ScopeContext.getOps ctx.ScopeContext.Value)

[<Fact>]
let ``driver rejects a zipper from another graph snapshot before witnessing`` () =
    let builder = NodeBuilder()
    let value = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let oldGraph = builder.Build [] |> prepareSource
    let graph = { oldGraph with DeclarationRoots = [value.Id, DeclRoot.EntryPoint] }
    let position = Zipper.create oldGraph value.Id |> require "Missing old occurrence"
    let visited = ref Set.empty
    let ctx = context graph position visited
    let mutable observed = false
    visitAllNodes (fun _ _ -> observed <- true; WitnessOutput.skip) ctx graph.Nodes[value.Id] visited
    Assert.False observed
    Assert.Empty visited.Value
    Assert.Single ctx.Accumulator.Errors |> ignore

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``source-only deferred helpers emit nothing and stale source authority refuses traversal`` removeAuthority =
    let source = """module DeferredHelper
let helper () = "COLD-HELPER-BODY"
let discard (value: string) = 0
[<EntryPoint>]
let main _ = discard (helper ())
"""
    let graph = checkScalarProgram source "deferred-helper.clef"
    let helper =
        graph.Nodes.Values |> Seq.find (fun node ->
            node.IsReachable && (match node.Kind with SemanticKind.Binding("helper", false, _, _) -> true | _ -> false))
    let projection =
        match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryOrdinary graph with
        | Result.Ok projection -> projection
        | Result.Error reason -> failwith reason
    Assert.Contains(helper.Id, projection.DeferredOnly)
    let originalPosition = Zipper.create graph helper.Id |> require "Missing original helper declaration"
    let visited = ref Set.empty
    let originalContext = context graph originalPosition visited
    let graph =
        if removeAuthority then
            { graph with Edges = graph.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.OrdinaryUnusedActual) }
            |> unpublished
        else graph
    let position = Zipper.create graph helper.Id |> require "Missing actual helper declaration"
    // Intentionally withhold source publication after editing its premises.
    // Emission cannot infer an eager fallback for an unpublished graph.
    let ctx = { originalContext with Graph = graph; Zipper = position }
    let observed = ResizeArray<NodeId>()
    let witness (_: WitnessContext) (node: SemanticNode) =
        observed.Add node.Id
        WitnessOutput.empty
    visitAllNodes witness ctx position.Focus visited
    if removeAuthority then Assert.Single ctx.Accumulator.Errors |> ignore
    else Assert.Empty ctx.Accumulator.Errors
    Assert.Empty observed
    Assert.Empty visited.Value
    Assert.Empty ctx.GlobalVisited.Value
    Assert.Empty (ScopeContext.getOps ctx.ScopeContext.Value)
    Assert.True graph.Nodes[helper.Id].IsReachable

[<Fact>]
let ``an explicitly invalidated source projection cannot enter transfer or claim coverage`` () =
    let builder = NodeBuilder()
    let value = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let original = builder.Build [] |> prepareSource
    let supplied = coeffects original 64
    let copied = { original with Nodes = original.Nodes }
    Assert.True(Result.isOk(Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryRead copied))
    let graph = unpublished copied
    Assert.False(obj.ReferenceEquals(original, graph))
    let position = Zipper.create graph value.Id |> require "Missing copied occurrence"
    let accumulator = MLIRAccumulator.empty ()
    let scope = ref (ScopeContext.root ())
    let visited = ref Set.empty
    let ctx =
        { Coeffects = supplied; Accumulator = accumulator; RootAccumulator = accumulator
          ScopeContext = scope; RootScopeContext = scope; Graph = graph; Zipper = position
          GlobalVisited = visited; TraversalVisited = visited }
    let mutable witnessed = false
    visitAllNodes (fun _ _ -> witnessed <- true; WitnessOutput.empty) ctx position.Focus visited
    Assert.False witnessed
    Assert.Empty visited.Value
    Assert.Single accumulator.Errors |> ignore
    Assert.Empty (ScopeContext.getOps scope.Value)
    match Alex.Traversal.MLIRTransfer.transferWithCorrespondence graph value.Id supplied None with
    | Result.Error reason -> Assert.Contains("Source emission admission", reason)
    | Result.Ok _ -> failwith "An unpublished graph entered production witnessing"
    Assert.Single(Alex.Traversal.CoverageValidation.validateCoverage graph Set.empty) |> ignore

[<Fact>]
let ``match scrutinee bindings guard and body retain the declared structural occurrence`` () =
    let builder = NodeBuilder()
    let scrutinee = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let initial = builder.Create(SemanticKind.Literal(NativeLiteral.Bool false), Types.boolType, dummyRange)
    let binding = builder.Create(SemanticKind.Binding("condition", false, false, None), Types.boolType, dummyRange,
                                 children = [initial.Id])
    let read = builder.Create(SemanticKind.VarRef("condition", Some binding.Id), Types.boolType, dummyRange)
    let guard = builder.Create(SemanticKind.Sequential [read.Id], Types.boolType, dummyRange)
    let yes = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let no = builder.Create(SemanticKind.Literal(NativeLiteral.Bool false), Types.boolType, dummyRange)
    let fallback = builder.Create(SemanticKind.Literal(NativeLiteral.Bool false), Types.boolType, dummyRange)
    let guarded = builder.Create(SemanticKind.IfThenElse(guard.Id, yes.Id, Some no.Id), Types.boolType, dummyRange)
    let selected = builder.Create(SemanticKind.Sequential [binding.Id; guarded.Id], Types.boolType, dummyRange)
    let arms =
        [{ Pattern = Pattern.Const(NativeLiteral.Bool true); Bindings = []; Guard = None; Body = selected.Id }
         { Pattern = Pattern.Wildcard; Bindings = []; Guard = None; Body = fallback.Id }]
    let choice = builder.Create(SemanticKind.CaseElimination(scrutinee.Id, arms), Types.boolType, dummyRange)
    let root = builder.Create(SemanticKind.Sequential [choice.Id], Types.boolType, dummyRange)
    let graph = builder.Build [] |> prepareSource
    let position = Zipper.create graph root.Id |> require "Missing root"
    let visited = ref Set.empty
    let ctx = context graph position visited
    let occurrences = ResizeArray<NodeId * NodeId list>()
    let expected =
        Map.ofList [scrutinee.Id, [choice.Id; root.Id]
                    binding.Id, [selected.Id; choice.Id; root.Id]
                    guard.Id, [guarded.Id; selected.Id; choice.Id; root.Id]
                    read.Id, [guard.Id; guarded.Id; selected.Id; choice.Id; root.Id]
                    yes.Id, [guarded.Id; selected.Id; choice.Id; root.Id]
                    no.Id, [guarded.Id; selected.Id; choice.Id; root.Id]
                    fallback.Id, [choice.Id; root.Id]]
    let rec witness ctx node =
        Assert.Equal(node.Id, ctx.Zipper.Focus.Id)
        if Map.containsKey node.Id expected then
            occurrences.Add(node.Id, ctx.Zipper.Path |> List.map (fun step -> step.Parent.Id))
        match node.Kind with
        | SemanticKind.CaseElimination _ -> (Alex.Witnesses.MatchWitness.createNanopass (fun () -> witness)).Witness ctx node
        | SemanticKind.IfThenElse _ -> (Alex.Witnesses.ControlFlowWitness.createNanopass (fun () -> witness)).Witness ctx node
        | SemanticKind.Binding _ -> Alex.Witnesses.BindingWitness.nanopass.Witness ctx node
        | SemanticKind.VarRef _ -> Alex.Witnesses.VarRefWitness.nanopass.Witness ctx node
        | SemanticKind.Literal _ -> Alex.Witnesses.LiteralWitness.nanopass.Witness ctx node
        | _ -> Alex.Witnesses.StructuralWitness.nanopass.Witness ctx node
    visitAllNodes witness ctx position.Focus visited
    Assert.Empty ctx.Accumulator.Errors
    Assert.Equal(expected.Count, occurrences.Count)
    for id, path in occurrences do
        Assert.Equal<NodeId list>(expected[id], path)
    let result, resultType = MLIRAccumulator.recallNode root.Id ctx.Accumulator |> require "Match lost its result"
    let body = ScopeContext.getOps ctx.ScopeContext.Value @ [MLIROp.FuncOp(FuncOp.Return [{ SSA = result; Type = resultType }])]
    let declaration = MLIROp.FuncOp(FuncOp.FuncDef("match_occurrences", [], [resultType], body, FuncVisibility.Private))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "match_occurrences" [declaration]
    MlirComponentTests.mlirOpt ["--verify-each"] text |> ignore

[<Fact>]
let ``match refuses an unsettled source guard before visiting any child`` () =
    let builder = NodeBuilder()
    let value = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let arm = { Pattern = Pattern.Wildcard; Bindings = []; Guard = Some value.Id; Body = value.Id }
    let choice = builder.Create(SemanticKind.CaseElimination(value.Id, [arm]), Types.boolType, dummyRange)
    let graph = builder.Build [] |> prepareSource
    let position = Zipper.create graph choice.Id |> require "Missing match"
    let visited = ref Set.empty
    let ctx = context graph position visited
    let mutable observed = false
    let output =
        (Alex.Witnesses.MatchWitness.createNanopass (fun () -> fun _ _ -> observed <- true; WitnessOutput.skip)).Witness ctx position.Focus
    Assert.False observed
    match output.Result with TRError _ -> () | other -> failwithf "Expected selected-scope diagnostic, got %A" other
    Assert.Empty output.InlineOps
