module Alex.Tests.UnitResultTests

open Xunit
open XParsec
open XParsec.Parsers
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
open Alex.Patterns.LiteralPatterns
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper

/// The unit type and branch operations have already been witnessed. Observe one
/// conditional at its Huet position; this fixture performs no graph elaboration.
let private fixture () =
    let builder = NodeBuilder()
    let guard = builder.Create(SemanticKind.PatternBinding "condition", Types.boolType, dummyRange)
    let thenBranch = builder.Create(SemanticKind.Literal NativeLiteral.Unit, Types.unitType, dummyRange)
    let elseBranch = builder.Create(SemanticKind.Literal NativeLiteral.Unit, Types.unitType, dummyRange)
    let conditional = builder.Create(SemanticKind.IfThenElse(guard.Id, thenBranch.Id, Some elseBranch.Id),
                                     Types.unitType, dummyRange)
    let binding = builder.Create(SemanticKind.Binding("effectResult", false, false, None),
                                 Types.unitType, dummyRange, children = [conditional.Id])
    builder.SetParent(conditional.Id, binding.Id)
    let graph = builder.Build []
    let position = Zipper.create graph binding.Id |> require "Missing unit fixture binding" |> atChild conditional.Id
    position, thenBranch.Id, elseBranch.Id

let private unitType = TInt(IntWidth 32)
let private cellType = TMemRefStatic(1, unitType)
let private store source = MLIROp.MemRefOp(MemRefOp.Store(source, Arg 1, [Arg 4], unitType, cellType))
let private thenEffects = [store (Arg 2); store (Arg 3)]
let private elseEffects = [store (Arg 3); store (Arg 2)]

let private observe body (position: Zipper.PSGZipper) =
    let operands = MLIRAccumulator.empty ()
    let nodes, types = operands.NodeAssoc, operands.SSATypes
    let graphNodes, edges, codata = position.Graph.Nodes, position.Graph.Edges, position.Graph.Codata
    let result = matchAt (pWithUnitResult position.Focus.Id body) position 64 operands
    Assert.Same(graphNodes, position.Graph.Nodes)
    Assert.Same(edges, position.Graph.Edges)
    Assert.Same(codata, position.Graph.Codata)
    Assert.Same(nodes, operands.NodeAssoc)
    Assert.Same(types, operands.SSATypes)
    Assert.Empty(operands.AllOps)
    Assert.Empty(operands.Errors)
    result

let private conditionalResult () =
    let position, thenId, elseId = fixture ()
    let body =
        Alex.Patterns.ControlFlowPatterns.pBuildConditional
            (Arg 0) thenEffects (Some elseEffects) thenId (Some elseId) None position.Focus.Id
    match observe body position with
    | Result.Ok ((operations, TRValue result), next) ->
        Assert.Same(position.Graph, next.Graph)
        Assert.Same(position.Graph.Nodes, next.Graph.Nodes)
        Assert.Equal(position.Focus.Id, next.Focus.Id)
        Assert.Same(position.Path, next.Path)
        operations, result
    | other -> failwithf "Unit conditional did not produce a value: %A" other

[<Fact>]
let ``unit conditional preserves ordered branch effects before its sole i32 zero value`` () =
    let operations, result = conditionalResult ()
    Assert.Equal(unitType, result.Type)
    match operations with
    | [MLIROp.SCFOp(SCFOp.If(Arg 0, thenOps, Some elseOps, None));
       MLIROp.ArithOp(ArithOp.ConstI(ssa, 0L, TInt(IntWidth 32)))] ->
        Assert.Equal<MLIROp list>(thenEffects @ [MLIROp.SCFOp(SCFOp.Yield [])], thenOps)
        Assert.Equal<MLIROp list>(elseEffects @ [MLIROp.SCFOp(SCFOp.Yield [])], elseOps)
        Assert.Equal(result.SSA, ssa)
    | other -> failwithf "Conditional effects or canonical unit result were changed: %A" other

[<Fact>]
let ``unit wrapper preserves a missing operand diagnostic from its body`` () =
    let position, thenId, _ = fixture ()
    let body = parser {
        let! _ = pRecallNode thenId
        return [], TRVoid
    }
    match observe body position with
    | Result.Error message ->
        Assert.Contains($"Node {NodeId.value thenId} not yet witnessed", message)
        Assert.DoesNotContain("Unit result requires", message)
    | Result.Ok _ -> failwith "Unit wrapper masked a missing witnessed operand"

[<Fact>]
let ``unit wrapper rejects a body that already returns a value`` () =
    let position, _, _ = fixture ()
    let body = preturn ([], TRValue { SSA = Arg 0; Type = unitType })
    match observe body position with
    | Result.Error message -> Assert.Contains("Unit result requires a witnessed void operation", message)
    | Result.Ok _ -> failwith "Unit wrapper silently replaced an existing value"

[<Fact>]
let ``effectful unit conditional verifies and lowers through standard MLIR`` () =
    let operations, result = conditionalResult ()
    let parameters = [Arg 0, TInt(IntWidth 1); Arg 1, cellType; Arg 2, unitType; Arg 3, unitType; Arg 4, TIndex]
    let body = operations @ [MLIROp.FuncOp(FuncOp.Return([{ SSA = result.SSA; Type = result.Type }]))]
    let definition = MLIROp.FuncOp(FuncOp.FuncDef("unit_effects", parameters, [result.Type], body, FuncVisibility.Public))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "unit_component" [definition]
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
    Assert.Contains("scf.if", verified)
    Assert.Contains("arith.constant 0 : i32", verified)
    let pipeline =
        "builtin.module(convert-scf-to-cf,expand-strided-metadata,memref-expand,"
        + "finalize-memref-to-llvm{index-bitwidth=64},convert-index-to-llvm{index-bitwidth=64},"
        + "convert-func-to-llvm{index-bitwidth=64},convert-arith-to-llvm{index-bitwidth=64},"
        + "convert-cf-to-llvm,reconcile-unrealized-casts)"
    let lowered = MlirComponentTests.mlirOpt ["--verify-each"; "--pass-pipeline=" + pipeline] verified
    Assert.Contains("llvm.func @unit_effects", lowered)
    Assert.Contains("llvm.store", lowered)
    Assert.Contains("llvm.return", lowered)
    Assert.DoesNotContain("scf.if", lowered)
    Assert.DoesNotContain("unrealized_conversion_cast", lowered)

let private unitComparison () =
    let builder = NodeBuilder()
    let left = builder.Create(SemanticKind.PatternBinding "left", Types.unitType, dummyRange)
    let right = builder.Create(SemanticKind.PatternBinding "right", Types.unitType, dummyRange)
    let equal = builder.Create(SemanticKind.Intrinsic
        { Module = IntrinsicModule.Operators; Operation = "op_Equality"
          Category = IntrinsicCategory.Comparison; FullName = "Operators.op_Equality" },
        NativeType.TFun(Types.unitType, NativeType.TFun(Types.unitType, Types.boolType)), dummyRange)
    let call = builder.Create(SemanticKind.Application(equal.Id, [left.Id; right.Id]), Types.boolType, dummyRange)
    let graph = builder.Build []
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode left.Id (Arg 0) unitType operands
    MLIRAccumulator.bindNode right.Id (Arg 1) unitType operands
    let position = Zipper.create graph call.Id |> require "Missing unit equality occurrence"
    position, left.Id, right.Id, operands

[<Theory>]
[<InlineData("eq")>]
[<InlineData("ne")>]
let ``unit equality compares already witnessed carriers without inventing numeric ranges`` predicate =
    let position, left, right, operands = unitComparison ()
    Assert.True(position.Graph.Nodes[left].ValueRange.IsNone)
    Assert.True(position.Graph.Nodes[right].ValueRange.IsNone)
    match matchAt (Alex.Patterns.ApplicationPatterns.pComparisonOp position.Focus.Id predicate) position 64 operands with
    | Result.Ok (([MLIROp.ArithOp(ArithOp.CmpI(_, actual, Arg 0, Arg 1, carrier))] as operations, TRValue result), _) ->
        Assert.Equal((if predicate = "eq" then ICmpPred.Eq else ICmpPred.Ne), actual)
        Assert.Equal(unitType, carrier)
        let definition = MLIROp.FuncOp(FuncOp.FuncDef("unit_equality", [Arg 0, unitType; Arg 1, unitType], [result.Type],
            operations @ [MLIROp.FuncOp(FuncOp.Return [result])], FuncVisibility.Private))
        let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "unit_equality" [definition]
        let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
        Assert.Contains("arith.cmpi " + predicate, verified)
    | other -> failwithf "Unit comparison lost its actual carriers: %A" other

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``unit equality refuses absent or noncanonical witnessed operands`` absent =
    let position, _, right, operands = unitComparison ()
    if absent then operands.NodeAssoc <- operands.NodeAssoc.Remove right
    else MLIRAccumulator.bindNode right (Arg 1) (TInt(IntWidth 8)) operands
    match matchAt (Alex.Patterns.ApplicationPatterns.pComparisonOp position.Focus.Id "eq") position 64 operands with
    | Result.Error message ->
        Assert.Contains((if absent then "not yet witnessed" else "canonical witnessed unit"), message)
    | Result.Ok _ -> failwith "Unit comparison manufactured an operand"
    Assert.Empty operands.AllOps
