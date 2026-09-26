module Alex.Tests.StringBoundaryTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Patterns.StringPatterns
open Alex.Patterns.MemoryPatterns
open Alex.CodeGeneration.TypeMapping
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper

/// The source byte-array signature is fixed; tests supply the actual recalled
/// physical operand to exercise Alex's settled-representation boundary.
let private fixture () =
    let builder = NodeBuilder()
    let arrayType = Types.mkArrayType Types.uint8Type
    let input = builder.Create(SemanticKind.PatternBinding "bytes", arrayType, dummyRange)
    let intrinsic = builder.Create(
        SemanticKind.Intrinsic
            { Module = IntrinsicModule.String; Operation = "fromBytes"
              Category = IntrinsicCategory.StringOp; FullName = "String.fromBytes" },
        NativeType.TFun(arrayType, Types.stringType), dummyRange)
    let call = builder.Create(SemanticKind.Application(intrinsic.Id, [input.Id]), Types.stringType, dummyRange,
                              children = [intrinsic.Id; input.Id])
    let graph = builder.Build []
    let position = Zipper.create graph call.Id |> require "Missing string boundary fixture"
    position, input.Id

let private observe carrier =
    let position, input = fixture ()
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode input (Arg 0) carrier operands
    let associations, types = operands.NodeAssoc, operands.SSATypes
    let nodes, edges, codata = position.Graph.Nodes, position.Graph.Edges, position.Graph.Codata
    let result = matchAt pStringFromBytesIntrinsic position 64 operands
    Assert.Same(nodes, position.Graph.Nodes)
    Assert.Same(edges, position.Graph.Edges)
    Assert.Same(codata, position.Graph.Codata)
    Assert.Same(associations, operands.NodeAssoc)
    Assert.Same(types, operands.SSATypes)
    Assert.Empty operands.AllOps
    Assert.Empty operands.Errors
    position, input, result

[<Fact>]
let ``settled bytes retain their exact SSA and string carrier without allocation or conversion`` () =
    let stringType = TMemRef(TInt(IntWidth 8))
    let position, _, result = observe stringType
    match result with
    | Result.Ok ((operations, TRValue value), next) ->
        Assert.Empty operations
        Assert.Equal(Arg 0, value.SSA)
        Assert.Equal(stringType, value.Type)
        Assert.Same(position.Graph, next.Graph)
        Assert.Equal(position.Focus.Id, next.Focus.Id)
        let body = [MLIROp.FuncOp(FuncOp.Return([{ SSA = value.SSA; Type = value.Type }]))]
        let definition = MLIROp.FuncOp(FuncOp.FuncDef("from_settled_bytes", [Arg 0, stringType], [stringType], body, FuncVisibility.Public))
        let source = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "string_boundary" [definition]
        let verified = MlirComponentTests.mlirOpt ["--verify-each"] source
        Assert.Contains("func.func @from_settled_bytes", verified)
    | other -> failwithf "Settled byte identity was rejected: %A" other

[<Theory>]
[<InlineData(32)>]
[<InlineData(64)>]
let ``a wider recalled array cannot be relabeled as a string`` width =
    let carrier = TMemRef(TInt(IntWidth width))
    let position, input, result = observe carrier
    match result with
    | Result.Error message ->
        Assert.Contains("String.fromBytes: byte-buffer representation is not settled", message)
        Assert.Contains(sprintf "node %d" (NodeId.value position.Focus.Id), message)
        Assert.Contains(sprintf "operand %d" (NodeId.value input), message)
        Assert.Contains(sprintf "carrier %A" carrier, message)
        Assert.Contains(sprintf "expected %A" (TMemRef(TInt(IntWidth 8))), message)
    | other -> failwithf "Unsettled array acquired string identity: %A" other

/// This component receives Baker's complete storage relation and scalar meets.
/// It verifies their physical composition; it does not establish byte admission.
[<Fact>]
let ``an exact byte storage origin composes allocation writes reads and string identity`` () =
    let builder = NodeBuilder()
    let arrayType = Types.mkArrayType Types.intType
    let size = builder.Create(SemanticKind.PatternBinding "size", Types.intType, dummyRange)
    let index = builder.Create(SemanticKind.PatternBinding "index", Types.intType, dummyRange)
    let value = builder.Create(SemanticKind.PatternBinding "value", Types.intType, dummyRange)
    let allocator = builder.Create(
        SemanticKind.Intrinsic
            { Module = IntrinsicModule.Array; Operation = "zeroCreate"
              Category = IntrinsicCategory.Memory; FullName = "Array.zeroCreate" },
        NativeType.TFun(Types.intType, arrayType), dummyRange)
    let allocation = builder.Create(SemanticKind.Application(allocator.Id, [size.Id]), arrayType, dummyRange)
    let binding = builder.Create(SemanticKind.Binding("bytes", false, false, None), arrayType,
                                 dummyRange, children = [allocation.Id])
    let input = builder.Create(SemanticKind.VarRef("bytes", Some binding.Id), arrayType, dummyRange)
    let write = builder.Create(SemanticKind.IndexSet(input.Id, index.Id, value.Id), Types.unitType, dummyRange)
    let read = builder.Create(SemanticKind.IndexGet(input.Id, index.Id), Types.intType, dummyRange)
    let converter = builder.Create(
        SemanticKind.Intrinsic
            { Module = IntrinsicModule.String; Operation = "fromBytes"
              Category = IntrinsicCategory.StringOp; FullName = "String.fromBytes" },
        NativeType.TFun(arrayType, Types.stringType), dummyRange)
    let conversion = builder.Create(SemanticKind.Application(converter.Id, [input.Id]), Types.stringType, dummyRange)
    let unrelated = builder.Create(SemanticKind.PatternBinding "integers", arrayType, dummyRange)
    let raw = builder.Build []
    let storage =
        [allocation.Id; binding.Id; input.Id]
        |> List.map (fun target ->
            { Class = EdgeClass.Range; Role = EdgeRole.StringByteStorage(0I, 255I, "byte")
              Sources = [conversion.Id; input.Id; allocation.Id; write.Id; value.Id]
              Target = target; Ordinal = 0 })
    let writeMeet = { Consumer = write.Id; Operand = value.Id; From = 64; To = 8; Adapt = MeetKind.Truncate }
    let readMeet = { Consumer = read.Id; Operand = read.Id; From = 8; To = 64; Adapt = MeetKind.ExtendUnsigned }
    let graph =
        { raw with Edges = storage @ raw.Edges
                   Codata = lazy { raw.Codata.Value with Meets = Map.ofList [write.Id, [writeMeet]; read.Id, [readMeet]] } }
    let byteType, intType = TInt(IntWidth 8), TInt(IntWidth 64)
    for occurrence in [allocation.Id; binding.Id; input.Id] do
        Assert.Equal(Some byteType, tryArrayElementTypeAt graph occurrence)
        Assert.Equal(arrayType, graph.Nodes[occurrence].Type)
    Assert.Equal(None, tryArrayElementTypeAt graph unrelated.Id)
    let operands = MLIRAccumulator.empty ()
    for id, argument in [size.Id, Arg 0; index.Id, Arg 1; value.Id, Arg 2] do
        MLIRAccumulator.bindNode id argument intType operands
    let observeAt parser id =
        let position = Zipper.create graph id |> require "Missing byte storage component position"
        match matchAt parser position 64 operands with
        | Result.Ok ((operations, result), next) ->
            Assert.Same(graph, next.Graph)
            operations, result
        | Result.Error message -> failwith message
    let allocationOps, allocated = observeAt pArrayZeroCreateIntrinsic allocation.Id
    let buffer = match allocated with TRValue value -> value | other -> failwithf "No array value: %A" other
    Assert.Equal(TMemRef byteType, buffer.Type)
    MLIRAccumulator.bindNode input.Id buffer.SSA buffer.Type operands
    let writeOps, written = observeAt pIndexSetArray write.Id
    match written with TRVoid -> () | other -> failwithf "Unexpected indexed write result: %A" other
    let readOps, loaded = observeAt pIndexGetArray read.Id
    let loaded = match loaded with TRValue value -> value | other -> failwithf "No indexed value: %A" other
    Assert.Equal(intType, loaded.Type)
    let identityOps, converted = observeAt pStringFromBytesIntrinsic conversion.Id
    Assert.Empty identityOps
    match converted with
    | TRValue converted ->
        Assert.Equal(buffer.SSA, converted.SSA)
        Assert.Equal(buffer.Type, converted.Type)
    | other -> failwithf "No string value: %A" other
    let body = allocationOps @ writeOps @ readOps @ [MLIROp.FuncOp(FuncOp.Return([{ SSA = loaded.SSA; Type = loaded.Type }]))]
    let definition = MLIROp.FuncOp(FuncOp.FuncDef(
        "byte_storage", [Arg 0, intType; Arg 1, intType; Arg 2, intType], [intType], body, FuncVisibility.Public))
    let source = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "string_storage" [definition]
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] source
    Assert.Contains("memref<?xi8>", verified)
    Assert.Contains("arith.trunci", verified)
    Assert.Contains("arith.extui", verified)
    Assert.Same(raw.Nodes, graph.Nodes)
