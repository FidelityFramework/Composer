module Alex.Tests.IndexPatternTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures

[<Theory>]
[<InlineData(true, "index.castu")>]
[<InlineData(false, "index.casts")>]
let ``Array get pulls its index range from the graph and composes a typed load`` (unsigned: bool) (spelling: string) =
    let fixture = arrayRead unsigned
    let operands = recalledOperands fixture
    let originalNodes, originalTypes = operands.NodeAssoc, operands.SSATypes
    let operations, result, position = readArray fixture 64 operands
    let cast =
        operations |> List.choose (function MLIROp.IndexOp _ as operation -> Some operation | _ -> None)
        |> Assert.Single
    let castSSA = Alex.Traversal.Values.value fixture.Call 0
    Assert.Equal($"{ssaToString castSSA} = {spelling} {ssaToString (Arg 1)} : i8 to index", opToString (Ok 64) cast)
    let load =
        operations |> List.choose (function MLIROp.MemRefOp(MemRefOp.Load _ as load) -> Some load | _ -> None)
        |> Assert.Single
    match load, result with
    | MemRefOp.Load(loaded, Arg 0, [index], TInt(IntWidth 1), TMemRef(TInt(IntWidth 1))), TRValue value ->
        Assert.Equal(castSSA, index)
        Assert.Equal(loaded, value.SSA)
        Assert.Equal(TInt(IntWidth 1), value.Type)
    | other -> failwithf "The cast and typed load are disconnected: %A" other
    Assert.Same(fixture.Graph, position.Graph)
    Assert.Same(fixture.Graph.Nodes, position.Graph.Nodes)
    Assert.Same(fixture.Graph.Edges, position.Graph.Edges)
    Assert.Same(originalNodes, operands.NodeAssoc)
    Assert.Same(originalTypes, operands.SSATypes)
    Assert.Empty(operands.AllOps)
    Assert.Empty(operands.Errors)

[<Fact>]
let ``missing recalled index is a diagnostic rather than a fabricated operand`` () =
    let fixture = arrayRead true
    let operands = recalledOperands fixture
    operands.NodeAssoc <- operands.NodeAssoc.Remove fixture.Index
    match matchAt Alex.Patterns.MemoryPatterns.pArrayGetIntrinsic fixture.Position 64 operands with
    | Result.Error message ->
        Assert.Contains($"Node {NodeId.value fixture.Index} not yet witnessed", message)
    | Result.Ok _ -> failwith "Pattern accepted an index that has not been witnessed"
    Assert.Empty(operands.AllOps)

[<Fact>]
let ``missing memory carrier type is diagnosed by the composed load element`` () =
    let fixture = arrayRead true
    let operands = recalledOperands fixture
    operands.SSATypes <- operands.SSATypes.Remove(Arg 0)
    match matchAt Alex.Patterns.MemoryPatterns.pArrayGetIntrinsic fixture.Position 64 operands with
    | Result.Error message ->
        Assert.Contains("pLoad:", message)
        Assert.Contains($"memref SSA {Arg 0}", message)
        Assert.Contains("has no registered type", message)
    | Result.Ok _ -> failwith "Pattern accepted an array with no registered memory type"
    Assert.Empty(operands.AllOps)
