module Alex.Tests.ControlFlowOccurrenceTests

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

[<Fact>]
let ``conditional guard descends through its actual occurrence and recalls the enclosing operand`` () =
    let builder = NodeBuilder()
    let formal = builder.Create(SemanticKind.PatternBinding "condition", Types.boolType, dummyRange)
    let read = builder.Create(SemanticKind.VarRef("condition", Some formal.Id), Types.boolType, dummyRange)
    let guard = builder.Create(SemanticKind.Sequential [read.Id], Types.boolType, dummyRange)
    let yes = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let no = builder.Create(SemanticKind.Literal(NativeLiteral.Bool false), Types.boolType, dummyRange)
    let choice = builder.Create(SemanticKind.IfThenElse(guard.Id, yes.Id, Some no.Id), Types.boolType, dummyRange)
    let root = builder.Create(SemanticKind.Sequential [choice.Id], Types.boolType, dummyRange)
    let graph = builder.Build []
    let accumulator = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode formal.Id (Arg 0) (TInt(IntWidth 1)) accumulator
    let scope = ref (ScopeContext.root ())
    let visited = ref (Set.singleton formal.Id)
    let position = Zipper.create graph root.Id |> require "Missing conditional root"
    let ctx =
        { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
          ScopeContext = scope; RootScopeContext = scope; Graph = graph; Zipper = position
          GlobalVisited = visited; TraversalVisited = visited }
    let reads = ResizeArray<NodeId list>()
    let rec witness ctx node =
        Assert.Equal(node.Id, ctx.Zipper.Focus.Id)
        if node.Id = read.Id then
            let parent = Zipper.up ctx.Zipper |> require "Read lost its guard"
            let grandparent = Zipper.up parent |> require "Guard lost its conditional"
            reads.Add([parent.Focus.Id; grandparent.Focus.Id])
        match node.Kind with
        | SemanticKind.IfThenElse _ -> (Alex.Witnesses.ControlFlowWitness.createNanopass (fun () -> witness)).Witness ctx node
        | SemanticKind.VarRef _ -> Alex.Witnesses.VarRefWitness.nanopass.Witness ctx node
        | SemanticKind.Literal _ -> Alex.Witnesses.LiteralWitness.nanopass.Witness ctx node
        | _ -> Alex.Witnesses.StructuralWitness.nanopass.Witness ctx node
    visitAllNodes witness ctx position.Focus visited
    Assert.Empty accumulator.Errors
    Assert.Equal<NodeId list>([guard.Id; choice.Id], Assert.Single reads)
    Assert.Equal(Some(Arg 0, TInt(IntWidth 1)), MLIRAccumulator.recallNode read.Id accumulator)
    let result, resultType = MLIRAccumulator.recallNode root.Id accumulator |> require "Conditional lost its result"
    let operations = ScopeContext.getOps scope.Value
    let functionBody = operations @ [MLIROp.FuncOp(FuncOp.Return [{ SSA = result; Type = resultType }])]
    let declaration = MLIROp.FuncOp(FuncOp.FuncDef("conditional_occurrence", [Arg 0, TInt(IntWidth 1)], [resultType], functionBody, FuncVisibility.Private))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "conditional_occurrence" [declaration]
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
    Assert.Contains("scf.if %arg0", verified)

[<Fact>]
let ``control flow rejects a branch missing its declared occurrence even when an operand was recalled`` () =
    let builder = NodeBuilder()
    let guard = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let yes = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let no = builder.Create(SemanticKind.Literal(NativeLiteral.Bool false), Types.boolType, dummyRange)
    let choice = builder.Create(SemanticKind.IfThenElse(guard.Id, yes.Id, Some no.Id), Types.boolType, dummyRange)
    let original = builder.Build []
    let node = original.Nodes[choice.Id]
    let graph = { original with Nodes = original.Nodes.Add(choice.Id, { node with Children = [guard.Id; no.Id] }) }
    let accumulator = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode yes.Id (Arg 0) (TInt(IntWidth 1)) accumulator
    let scope = ref (ScopeContext.root ())
    let visited = ref Set.empty
    let position = Zipper.create graph choice.Id |> require "Missing conditional root"
    let ctx =
        { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
          ScopeContext = scope; RootScopeContext = scope; Graph = graph; Zipper = position
          GlobalVisited = visited; TraversalVisited = visited }
    let rec witness ctx node =
        Assert.NotEqual(yes.Id, node.Id)
        match node.Kind with
        | SemanticKind.IfThenElse _ -> (Alex.Witnesses.ControlFlowWitness.createNanopass (fun () -> witness)).Witness ctx node
        | SemanticKind.Literal _ -> Alex.Witnesses.LiteralWitness.nanopass.Witness ctx node
        | _ -> Alex.Witnesses.StructuralWitness.nanopass.Witness ctx node
    visitAllNodes witness ctx position.Focus visited
    Assert.NotEmpty accumulator.Errors
    Assert.DoesNotContain(yes.Id, visited.Value)
