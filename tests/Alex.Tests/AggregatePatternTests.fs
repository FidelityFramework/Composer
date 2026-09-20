module Alex.Tests.AggregatePatternTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Patterns.ContinuationPatterns
open Alex.Patterns.DUPatterns
open Alex.Patterns.LiteralPatterns
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper

// Component inputs are already settled. These tests verify physical witnessing
// and its missing-prerequisite boundary; Baker owns semantic copy admission.
let private fixture () =
    let builder = NodeBuilder()
    let option = NativeType.TApp(Types.optionTyCon, [Types.boolType])
    let frame = builder.Create(SemanticKind.PatternBinding "frame", Types.mkArrayType Types.uint8Type, dummyRange)
    let current = builder.Create(SemanticKind.PatternBinding "current", option, dummyRange)
    let view = builder.Create(SemanticKind.FrameRead(frame.Id, current.Id), option, dummyRange)
    let payload = builder.Create(SemanticKind.PatternBinding "payload", Types.boolType, dummyRange)
    let some = builder.Create(SemanticKind.DUInitialize(view.Id, "Some", 1, Some payload.Id), Types.unitType, dummyRange)
    let none = builder.Create(SemanticKind.DUInitialize(view.Id, "None", 0, None), Types.unitType, dummyRange)
    let storage = builder.Create(SemanticKind.AggregateStorage(current.Id), option, dummyRange)
    let raw = builder.Build []
    let graph = { raw with
                    Layouts = lazy (Map.ofList [formatType option, SettledLayout.Union(["None", None; "Some", Some SettledSlot.Bool], Some 1, Some 2, Some 1)])
                    Codata = lazy { raw.Codata.Value with Escapes = Map.ofList [storage.Id, EscapeKind.StackScoped] } }
    let slot = { Source = current.Id; ValueType = option; IsCapture = false; Holds = CaptureSlotKind.InlineValue option
                 Field = { Name = "current"; Slot = SettledSlot.InlineBytes(2, 1); Offset = Some 1; Size = Some 2; Align = Some 1 } }
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode frame.Id (Arg 0) (TMemRefStatic(4, TInt(IntWidth 8))) operands
    MLIRAccumulator.bindNode payload.Id (Arg 1) (TInt(IntWidth 1)) operands
    graph, frame, view, payload, some, none, storage, slot, operands

let private focus graph (node: SemanticNode) = Zipper.create graph node.Id |> require "Missing aggregate fixture focus"

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``selected Option initialization writes owned bytes without a descriptor copy`` hasPayload =
    let graph, frame, view, payload, some, none, _, slot, operands = fixture ()
    let views, value =
        match matchAt (pReadContinuationSlot view.Id frame.Id 4 slot) (focus graph view) 64 operands with
        | Result.Ok ((ops, TRValue value), _) -> ops, value
        | other -> failwithf "Inline view failed: %A" other
    MLIRAccumulator.bindNode view.Id value.SSA value.Type operands
    let destination = if hasPayload then some else none
    let name, index, input = if hasPayload then "Some", 1, Some payload.Id else "None", 0, None
    let writes, result =
        match matchAt (pWithUnitResult destination.Id (pBuildDUInitialize destination.Id view.Id name index input)) (focus graph destination) 64 operands with
        | Result.Ok ((ops, TRValue result), _) -> ops, result
        | other -> failwithf "Option initialization failed: %A" other
    let definition = MLIROp.FuncOp(FuncOp.FuncDef("initialize_option", [Arg 0, TMemRefStatic(4, TInt(IntWidth 8)); Arg 1, TInt(IntWidth 1)], result.Type,
        views @ writes @ [MLIROp.FuncOp(FuncOp.Return(Some result.SSA, Some result.Type))], FuncVisibility.Public))
    let source = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "aggregate_component" [definition]
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] source
    Assert.Contains("memref.view", verified)
    Assert.DoesNotContain("memref.load", verified)
    Assert.DoesNotContain("memref<1xmemref", verified)
    Assert.DoesNotContain("memref.alloc", verified)
    Assert.Equal((if hasPayload then 2 else 1), verified.Split("memref.store").Length - 1)
    let pipeline = "builtin.module(expand-strided-metadata,memref-expand,finalize-memref-to-llvm{index-bitwidth=64},convert-index-to-llvm{index-bitwidth=64},convert-func-to-llvm{index-bitwidth=64},convert-arith-to-llvm{index-bitwidth=64},reconcile-unrealized-casts)"
    let lowered = MlirComponentTests.mlirOpt ["--verify-each"; "--pass-pipeline=" + pipeline] verified
    Assert.Contains("llvm.func @initialize_option", lowered)
    Assert.DoesNotContain("unrealized_conversion_cast", lowered)

[<Fact>]
let ``aggregate allocation requires an explicit caller residence`` () =
    let graph, _, _, _, _, _, storage, _, operands = fixture ()
    match matchAt (pBuildAggregateStorage storage.Id) (focus graph storage) 64 operands with
    | Result.Ok (([MLIROp.MemRefOp(MemRefOp.Alloca(_, TMemRefStatic(2, TInt(IntWidth 8)), Some 1))], TRValue _), _) -> ()
    | other -> failwithf "Settled storage allocation failed: %A" other
    let missing = { graph with Codata = lazy { graph.Codata.Value with Escapes = Map.empty } }
    match matchAt (pBuildAggregateStorage storage.Id) (focus missing storage) 64 operands with
    | Result.Error diagnostic -> Assert.Contains("admitted caller activation", diagnostic)
    | other -> failwithf "Missing residence was accepted: %A" other

[<Theory>]
[<InlineData("capture")>]
[<InlineData("size")>]
[<InlineData("extent")>]
let ``inline continuation views reject inconsistent placement`` corruption =
    let graph, frame, view, _, _, _, _, slot, operands = fixture ()
    let corrupt =
        match corruption with
        | "capture" -> { slot with IsCapture = true }
        | "size" -> { slot with Field = { slot.Field with Size = Some 1 } }
        | _ -> { slot with Field = { slot.Field with Offset = Some 3 } }
    match matchAt (pReadContinuationSlot view.Id frame.Id 4 corrupt) (focus graph view) 64 operands with
    | Result.Error diagnostic ->
        Assert.Contains((if corruption = "extent" then "complete placement" else "owned union region"), diagnostic)
    | other -> failwithf "Invalid inline placement was accepted: %A" other

[<Fact>]
let ``an aggregate frame write requires Baker case decomposition`` () =
    let graph, frame, view, _, _, _, _, slot, operands = fixture ()
    MLIRAccumulator.bindNode view.Id (Arg 2) (TMemRefStatic(2, TInt(IntWidth 8))) operands
    match matchAt (pWriteContinuationSlot view.Id frame.Id view.Id 4 slot) (focus graph view) 64 operands with
    | Result.Error diagnostic -> Assert.Contains("explicit selected-case copy", diagnostic)
    | other -> failwithf "Implicit aggregate copy was accepted: %A" other

[<Theory>]
[<InlineData("case", "selected case payload")>]
[<InlineData("missing", "selected case payload")>]
[<InlineData("carrier", "settled scalar carrier")>]
let ``DU initialization rejects mismatched case and payload premises`` corruption message =
    let graph, _, view, payload, some, _, _, _, operands = fixture ()
    MLIRAccumulator.bindNode view.Id (Arg 0) (TMemRefStatic(2, TInt(IntWidth 8))) operands
    if corruption = "carrier" then MLIRAccumulator.bindNode payload.Id (Arg 1) (TInt(IntWidth 8)) operands
    let caseName = if corruption = "case" then "None" else "Some"
    let value = if corruption = "missing" then None else Some payload.Id
    match matchAt (pBuildDUInitialize some.Id view.Id caseName 1 value) (focus graph some) 64 operands with
    | Result.Error diagnostic -> Assert.Contains(message, diagnostic)
    | other -> failwithf "Mismatched DU premise was accepted: %A" other
