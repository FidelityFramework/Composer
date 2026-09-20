module Alex.Tests.ZipperTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper

[<Fact>]
let ``sibling round trip preserves the enclosing scope and settled graph facts`` () =
    let fixture = arrayRead true
    let atIndex = fixture.Position |> atChild fixture.Index
    let atArray = Zipper.left atIndex |> require "No preceding array operand"
    let roundTrip = Zipper.right atArray |> require "No following index operand"
    Assert.Equal(fixture.Index, roundTrip.Focus.Id)
    Assert.Equal(fixture.Lambda, (Zipper.findEnclosingLambda roundTrip |> require "Lost lambda scope").Id)
    Assert.Equal(fixture.Call, (Zipper.up roundTrip |> require "Lost call context").Focus.Id)
    Assert.Same(fixture.Graph, roundTrip.Graph)
    Assert.Same(fixture.Graph.Nodes[fixture.Index], roundTrip.Focus)
    Assert.Same(fixture.Graph.Codata, roundTrip.Graph.Codata)
    Assert.Same(fixture.Graph.Edges, roundTrip.Graph.Edges)
    Assert.Equal(Some(ValueRange.Bounded(128I, 255I)), roundTrip.Focus.ValueRange)
    let proofEdge = Assert.Single(roundTrip.Graph.Edges)
    Assert.Equal<NodeId list>([fixture.Index], proofEdge.Sources)
    Assert.Equal(fixture.Proof, proofEdge.Target)
    Assert.Equal(EdgeRole.Constrains, proofEdge.Role)

[<Fact>]
let ``one shared node retains a distinct enclosing lambda at each Huet position`` () =
    let builder = NodeBuilder()
    let shared = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let makeLambda name =
        builder.Create(SemanticKind.Lambda([], shared.Id, [], Some name, LambdaContext.RegularClosure),
                       NativeType.TFun(Types.unitType, Types.boolType), dummyRange)
    let first, second = makeLambda "first", makeLambda "second"
    let root = builder.Create(SemanticKind.Sequential [first.Id; second.Id], second.Type, dummyRange)
    let graph = builder.Build []
    let start = Zipper.create graph root.Id |> require "Missing root"
    let leftUse = start |> atChild first.Id |> atChild shared.Id
    let rightUse = start |> atChild second.Id |> atChild shared.Id
    Assert.Same(leftUse.Focus, rightUse.Focus)
    Assert.Equal(first.Id, (Zipper.findEnclosingLambda leftUse |> require "Missing first scope").Id)
    Assert.Equal(second.Id, (Zipper.findEnclosingLambda rightUse |> require "Missing second scope").Id)
    Assert.Same(graph, leftUse.Graph)
    Assert.Same(graph, rightUse.Graph)
    let reRooted = Zipper.focusOn shared.Id rightUse |> require "Cannot re-root"
    Assert.True(Zipper.isAtRoot reRooted)
    Assert.True((Zipper.findEnclosingLambda reRooted).IsNone)

[<Fact>]
let ``binding-name parser reads the zipper parent and restores its focus`` () =
    let fixture = arrayRead true
    let lambda = Zipper.up fixture.Position |> require "Missing lambda"
    let operands = MLIRAccumulator.empty ()
    // Composing another observation after pLambdaWithBinding detects leaked parser focus.
    let observation = XParsec.Combinators.parser {
        let! name, _, body, _ = pLambdaWithBinding
        let! current = getCurrentNode
        return name, body, current.Id
    }
    match matchAt observation lambda 64 operands with
    | Result.Ok ((name, body, current), position) ->
        Assert.Equal("read", name)
        Assert.Equal(fixture.Call, body)
        Assert.Equal(fixture.Lambda, current)
        Assert.Equal(fixture.Lambda, position.Focus.Id)
        Assert.Same(fixture.Graph, position.Graph)
        Assert.Empty(operands.AllOps)
    | Result.Error message -> failwith message

[<Fact>]
let ``invalid navigation does not manufacture a scope or graph node`` () =
    let fixture = arrayRead true
    let root = Zipper.create fixture.Graph fixture.Binding |> require "Missing root"
    Assert.True((Zipper.up root).IsNone)
    Assert.True((Zipper.left root).IsNone)
    Assert.True((Zipper.right root).IsNone)
    Assert.True((Zipper.down -1 root).IsNone)
    Assert.True((Zipper.down root.Focus.Children.Length root).IsNone)
    let absent = NodeId.fresh ()
    Assert.True((Zipper.create fixture.Graph absent).IsNone)
    Assert.True((Zipper.focusOn absent fixture.Position).IsNone)
    let error = Assert.ThrowsAny<System.Exception>(fun () -> Zipper.requireNode absent fixture.Position |> ignore)
    Assert.Contains("not found in graph", error.Message)
    Assert.Same(fixture.Graph, fixture.Position.Graph)
