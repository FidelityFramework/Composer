module Alex.Tests.IndexSwitchTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Patterns.ControlFlowPatterns
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper

// A dialect component fixture with already witnessed operands/arms. It does not
// establish source evaluation order, continuation segments or a frame layout.
let private position () =
    let builder = NodeBuilder()
    let selector = builder.Create(SemanticKind.PatternBinding "state", Types.intType, dummyRange)
    let binding = builder.Create(SemanticKind.Binding("switch", false, false, None),
                                 Types.intType, dummyRange, children = [selector.Id])
    builder.SetParent(selector.Id, binding.Id)
    Zipper.create (builder.Build []) binding.Id |> require "Missing switch binding" |> atChild selector.Id

let private integer = TInt(IntWidth 32)
let private cell = TMemRefStatic(1, integer)
let private value ssa ty : Val = { SSA = ssa; Type = ty }
let private selector = value (Arg 0) TIndex
let private first = value (Arg 3) integer
let private second = value (Arg 4) integer
let private store item = MLIROp.MemRefOp(MemRefOp.Store(item.SSA, Arg 1, [Arg 2], integer, cell))

let private observeAt focus parser =
    let accumulator = MLIRAccumulator.empty ()
    let associations, types = accumulator.NodeAssoc, accumulator.SSATypes
    let result = matchAt parser focus 64 accumulator
    Assert.Same(associations, accumulator.NodeAssoc)
    Assert.Same(types, accumulator.SSATypes)
    Assert.Empty(accumulator.AllOps)
    match result with
    | Result.Ok (_, after) ->
        Assert.Same(focus.Graph, after.Graph)
        Assert.Same(focus.Path, after.Path)
        Assert.Equal(focus.Focus.Id, after.Focus.Id)
    | _ -> ()
    result

let private observe parser = observeAt (position ()) parser

let private built arity =
    let focus = position ()
    let results = [0 .. arity - 1] |> List.map (fun ordinal -> value (Alex.Traversal.Values.value focus.Focus.Id ordinal) integer)
    let left = [first; second] |> List.take arity
    let right = [second; first] |> List.take arity
    let cases = [7L, ([store first; store second], left); -1L, ([store second; store first], right)]
    let fallback = [store first], left
    match observeAt focus (pBuildIndexSwitch selector cases fallback results) with
    | Result.Ok (operations, _) -> operations, results
    | Result.Error message -> failwith message

[<Fact>]
let ``index switch preserves settled labels arm effects and ordered result lanes`` () =
    let operations, results = built 2
    match operations with
    | [MLIROp.SCFOp(SCFOp.IndexSwitch(Arg 0, [7L, left; -1L, right], fallback, actualResults))] ->
        Assert.Equal<(SSA * MLIRType) list>(results |> List.map (fun v -> v.SSA, v.Type), actualResults)
        Assert.Equal<MLIROp list>([store first; store second; MLIROp.SCFOp(SCFOp.Yield [first.SSA, integer; second.SSA, integer])], left)
        Assert.Equal<MLIROp list>([store second; store first; MLIROp.SCFOp(SCFOp.Yield [second.SSA, integer; first.SSA, integer])], right)
        Assert.Equal<MLIROp list>([store first; MLIROp.SCFOp(SCFOp.Yield [first.SSA, integer; second.SSA, integer])], fallback)
    | other -> failwithf "Switch arm effects/results changed: %A" other

[<Theory>]
[<InlineData(0)>]
[<InlineData(1)>]
[<InlineData(2)>]
let ``index switch with zero one or multiple results verifies and lowers through stock MLIR`` (arity: int) =
    let operations, results = built arity
    let result = results |> List.tryHead |> Option.defaultValue first
    let parameters = [Arg 0, TIndex; Arg 1, cell; Arg 2, TIndex; Arg 3, integer; Arg 4, integer]
    let definition = MLIROp.FuncOp(FuncOp.FuncDef("switch_component", parameters, [integer],
        operations @ [MLIROp.FuncOp(FuncOp.Return([{ SSA = result.SSA; Type = integer }]))], FuncVisibility.Public))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "switch_component" [definition]
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
    Assert.Contains("scf.index_switch", verified)
    Assert.Contains("case -1", verified)
    let pipeline =
        "builtin.module(convert-scf-to-cf,expand-strided-metadata,memref-expand,"
        + "finalize-memref-to-llvm{index-bitwidth=64},convert-index-to-llvm{index-bitwidth=64},"
        + "convert-func-to-llvm{index-bitwidth=64},convert-arith-to-llvm{index-bitwidth=64},"
        + "convert-cf-to-llvm,reconcile-unrealized-casts)"
    let lowered = MlirComponentTests.mlirOpt ["--verify-each"; "--pass-pipeline=" + pipeline] verified
    Assert.Contains("llvm.func @switch_component", lowered)
    Assert.Contains("llvm.store", lowered)
    Assert.DoesNotContain("scf.index_switch", lowered)
    Assert.DoesNotContain("unrealized_conversion_cast", lowered)

[<Theory>]
[<InlineData("selector", "requires an index selector")>]
[<InlineData("labels", "requires distinct case labels")>]
[<InlineData("case", "case 7 yield types do not match")>]
[<InlineData("default", "default yield types do not match")>]
[<InlineData("terminator", "case 7 already has a yield terminator")>]
let ``index switch rejects incompatible supplied evidence with a precise reason`` (fault: string) (expected: string) =
    let input = if fault = "selector" then first else selector
    let outputs = [value (V(99, 0)) integer]
    let values = if fault = "case" then [] else [first]
    let body = if fault = "terminator" then [MLIROp.SCFOp(SCFOp.Yield [first.SSA, integer])] else []
    let cases = if fault = "labels" then [7L, (body, values); 7L, ([], [first])] else [7L, (body, values)]
    let fallback = [], (if fault = "default" then [] else [second])
    match observe (pBuildIndexSwitch input cases fallback outputs) with
    | Result.Error message -> Assert.Contains(expected, message)
    | Result.Ok _ -> failwithf "Switch accepted invalid %s evidence" fault

[<Fact>]
let ``declarations inside switch arms reach existing module collection`` () =
    let declaration = MLIROp.FuncOp(FuncOp.FuncDecl("external_effect", [integer], [integer], FuncVisibility.Private, []))
    let call = MLIROp.FuncOp(FuncOp.FuncCall([{ SSA = (V(99, 2)); Type = integer }], "external_effect", [first]))
    let operations =
        match observe (pBuildIndexSwitch selector [7L, ([declaration; call], [])] ([], []) []) with
        | Result.Ok (operations, _) -> operations
        | Result.Error message -> failwith message
    let definition = MLIROp.FuncOp(FuncOp.FuncDef("calls", [Arg 0, TIndex; Arg 3, integer], [integer],
        operations @ [MLIROp.FuncOp(FuncOp.Return([{ SSA = first.SSA; Type = integer }]))], FuncVisibility.Public))
    let collected = Alex.Pipeline.MLIRNanopass.declarationCollectionPass [definition]
    Assert.Equal(declaration, collected.Head)
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "switch_declarations" collected
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
    Assert.Contains("func.func private @external_effect", verified)
