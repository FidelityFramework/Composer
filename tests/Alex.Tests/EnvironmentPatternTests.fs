module Alex.Tests.EnvironmentPatternTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Patterns.EnvironmentPatterns
open Alex.Patterns.ContinuationPatterns
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper

let private fixture () =
    let builder = NodeBuilder()
    let byteArray = Types.mkArrayType Types.uint8Type
    let capture = builder.Create(SemanticKind.PatternBinding "cell", Types.boolType, dummyRange)
    let formal = builder.Create(SemanticKind.PatternBinding "environment", byteArray, dummyRange)
    let body = builder.Create(SemanticKind.EnvironmentRead(formal.Id, capture.Id), Types.boolType, dummyRange)
    let implementation = builder.Create(SemanticKind.Lambda(["environment", byteArray, formal.Id], body.Id, [], None, LambdaContext.RegularClosure), NativeType.TFun(byteArray, Types.boolType), dummyRange)
    let owner = (builder.Create(SemanticKind.PatternBinding "callableOwner", NativeType.TFun(Types.unitType, Types.boolType), dummyRange)).Id
    let first = builder.Create(SemanticKind.EnvironmentCreate(owner, [capture.Id, capture.Id]), byteArray, dummyRange)
    let second = builder.Create(SemanticKind.EnvironmentCreate(owner, [capture.Id, capture.Id]), byteArray, dummyRange)
    let proof = builder.Create(SemanticKind.Obligation {
        Id = "environment_layout"; Kind = "continuation-layout"; Logic = "QF_LIA"
        Statement = "The captured cell descriptor occupies the declared environment"
        Source = "Alex component fixture"; Refs = []
        Body = ObligationBody.ContinuationLayout([0, 40, 8], 40, 8) }, Types.unitType, dummyRange)
    let slot: ContinuationSlot = {
        Source = capture.Id; ValueType = Types.boolType; IsCapture = true
        Holds = CaptureSlotKind.CellView Types.boolType
        Field = { Name = "cell"; Slot = SettledSlot.Pointer 5; Offset = Some 0; Size = Some 40; Align = Some 8 } }
    let layout: EnvironmentLayout = {
        Owner = owner; Implementation = implementation.Id; Formal = formal.Id
        Slots = [slot]; Bytes = 40; Alignment = 8; Obligations = [proof.Id] }
    let raw = builder.Build []
    let codata =
        { raw.Codata.Value with
            EnvironmentLayouts = Map.ofList [owner, layout]
            EnvironmentOrigins = Map.ofList [first.Id, owner; second.Id, owner; formal.Id, owner]
            Escapes = Map.ofList [first.Id, EscapeKind.StackScoped; second.Id, EscapeKind.StackScoped] }
    let graph = { raw with Codata = lazy codata }
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode capture.Id (Arg 0) (TMemRefStatic(1, TInt(IntWidth 1))) operands
    let position = Zipper.create graph first.Id |> require "Missing environment fixture"
    position, first.Id, second.Id, capture.Id, body.Id, layout, operands

[<Fact>]
let ``ordinary environment construction preserves the complete mutable cell descriptor through standard lowering`` () =
    let position, first, _, capture, read, layout, operands = fixture ()
    let allocation, environment =
        match matchAt (pCreateEnvironment first layout [capture, capture]) position 64 operands with
        | Result.Ok ((operations, TRValue value), next) ->
            Assert.Same(position.Graph, next.Graph)
            Assert.Equal(position.Focus.Id, next.Focus.Id)
            operations, value
        | other -> failwithf "Environment construction failed: %A" other
    MLIRAccumulator.bindNode first environment.SSA environment.Type operands
    let reads, result =
        match matchAt (pReadContinuationSlot read first layout.Bytes layout.Slots.Head) position 64 operands with
        | Result.Ok ((operations, TRValue value), _) -> operations, value
        | other -> failwithf "Environment read failed: %A" other
    let definition = MLIROp.FuncOp(FuncOp.FuncDef("environment_component", [Arg 0, TMemRefStatic(1, TInt(IntWidth 1))], result.Type,
        allocation @ reads @ [MLIROp.FuncOp(FuncOp.Return(Some result.SSA, Some result.Type))], FuncVisibility.Public))
    let source = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "environment_component" [definition]
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] source
    Assert.Contains("memref<1xmemref<1xi1>>", verified)
    Assert.DoesNotContain("memref<2xindex>", verified)
    let pipeline = "builtin.module(expand-strided-metadata,memref-expand,finalize-memref-to-llvm{index-bitwidth=64},convert-index-to-llvm{index-bitwidth=64},convert-func-to-llvm{index-bitwidth=64},convert-arith-to-llvm{index-bitwidth=64},reconcile-unrealized-casts)"
    let lowered = MlirComponentTests.mlirOpt ["--verify-each"; "--pass-pipeline=" + pipeline] verified
    Assert.Contains("llvm.func @environment_component", lowered)
    Assert.DoesNotContain("unrealized_conversion_cast", lowered)

[<Fact>]
let ``two occurrences of the same environment layout recall their own actual values`` () =
    let position, first, second, _, _, layout, operands = fixture ()
    let ty = TMemRefStatic(layout.Bytes, TInt(IntWidth 8))
    MLIRAccumulator.bindNode first (Arg 1) ty operands
    MLIRAccumulator.bindNode second (Arg 2) ty operands
    for occurrence, expected in [first, Arg 1; second, Arg 2] do
        match matchAt (pRecallEnvironment occurrence layout) position 64 operands with
        | Result.Ok ((operations, TRValue value), _) ->
            Assert.Empty operations
            Assert.Equal(expected, value.SSA)
        | other -> failwithf "Environment occurrence failed: %A" other

[<Theory>]
[<InlineData("residence", "admitted allocation residence")>]
[<InlineData("duplicate", "exact unique initializer set")>]
[<InlineData("missing", "exact unique initializer set")>]
[<InlineData("proof", "resident layout obligations")>]
let ``environment construction rejects incomplete residence initialization and evidence`` defect reason =
    let position, first, _, capture, _, layout, operands = fixture ()
    let graph =
        if defect = "residence" then
            { position.Graph with Codata = lazy { position.Graph.Codata.Value with Escapes = Map.empty } }
        else position.Graph
    let position = Zipper.create graph first |> require "Missing environment fixture"
    let layout = if defect = "proof" then { layout with Obligations = [] } else layout
    let initializers =
        match defect with
        | "duplicate" -> [capture, capture; capture, capture]
        | "missing" -> []
        | _ -> [capture, capture]
    match matchAt (pCreateEnvironment first layout initializers) position 64 operands with
    | Result.Error message -> Assert.Contains(reason, message)
    | Result.Ok _ -> failwithf "Invalid environment %s was accepted" defect
