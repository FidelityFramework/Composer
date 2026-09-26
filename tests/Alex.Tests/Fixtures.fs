module Alex.Tests.Fixtures

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
module Zipper = Alex.Traversal.PSGZipper

let require label = Option.defaultWith (fun () -> failwith label)

let coeffects (graph: SemanticGraph) pointerBits : TransferCoeffects =
    { Platform =
        { TargetArch = { Isa = (if pointerBits = 32 then ARM32_Thumb else X86_64)
                         Register = Ok pointerBits; Pointer = Ok pointerBits }
          LinkedLibraries = Set.empty
          Bindings = graph.Codata.Value.Bindings }
      TargetPlatform = Core.Types.Dialects.CPU }

let atChild childId (position: Zipper.PSGZipper) =
    let index = position.Focus.Children |> List.findIndex ((=) childId)
    Zipper.down index position |> require "Fixture child is missing"

/// Component-boundary fixture, not a source program or a substitute for Baker.
/// The index already has its analysed range and an i8 operand representation.
/// Neither this fixture nor the tested pattern establishes array bounds.
type ArrayRead =
    { Graph: SemanticGraph
      Binding: NodeId
      Lambda: NodeId
      Array: NodeId
      Index: NodeId
      Call: NodeId
      Proof: NodeId
      Position: Zipper.PSGZipper }

let arrayRead unsigned =
    let builder = NodeBuilder()
    let arrayType = Types.mkArrayType Types.boolType
    let array = builder.Create(SemanticKind.PatternBinding "values", arrayType, dummyRange)
    let index = builder.Create(SemanticKind.PatternBinding "index", Types.intType, dummyRange)
    let intrinsic = builder.Create(
        SemanticKind.Intrinsic
            { Module = IntrinsicModule.Array; Operation = "get"
              Category = IntrinsicCategory.Memory; FullName = "Array.get" },
        NativeType.TFun(arrayType, NativeType.TFun(Types.intType, Types.boolType)), dummyRange)
    let call = builder.Create(SemanticKind.Application(intrinsic.Id, [array.Id; index.Id]), Types.boolType, dummyRange)
    let lambda = builder.Create(
        SemanticKind.Lambda(["values", arrayType, array.Id; "index", Types.intType, index.Id],
                            call.Id, [], None, LambdaContext.RegularClosure),
        intrinsic.Type, dummyRange, children = [array.Id; index.Id; call.Id])
    let binding = builder.Create(SemanticKind.Binding("read", false, false, None), lambda.Type,
                                 dummyRange, children = [lambda.Id])
    for child in [array; index; call] do builder.SetParent(child.Id, lambda.Id)
    builder.SetParent(lambda.Id, binding.Id)
    let lower, upper = if unsigned then 128I, 255I else -128I, 127I
    let minimum, maximum = if unsigned then 0I, 255I else -128I, 127I
    let proof = builder.Create(
        SemanticKind.Obligation
            { Id = "index_carrier"; Kind = "integer-representation-coverage"; Logic = "QF_LIA"
              Statement = "The supplied index range fits its supplied operand carrier"
              Source = "Alex component fixture"; Refs = []
              Body = ObligationBody.IntegerRepresentationCoverage(lower, upper, minimum, maximum) },
        Types.unitType, dummyRange)
    let raw = builder.Build []
    let graph =
        { raw with
            Nodes = raw.Nodes.Add(index.Id, { raw.Nodes[index.Id] with ValueRange = Some (ValueRange.Bounded(lower, upper)) })
            Edges =
                [{ Sources = [index.Id]; Target = proof.Id; Class = EdgeClass.Obligation
                   Role = EdgeRole.Constrains; Ordinal = 0 }] }
    let position =
        Zipper.create graph binding.Id |> require "Missing fixture binding"
        |> atChild lambda.Id |> atChild call.Id
    { Graph = graph; Binding = binding.Id; Lambda = lambda.Id; Array = array.Id
      Index = index.Id; Call = call.Id; Proof = proof.Id; Position = position }

/// The current public parser API requires a table of already witnessed operands.
/// Seed those two inputs only; tests do not run or reproduce the mutable traversal.
let recalledOperands fixture =
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode fixture.Array (Arg 0) (TMemRef(TInt(IntWidth 1))) operands
    MLIRAccumulator.bindNode fixture.Index (Arg 1) (TInt(IntWidth 8)) operands
    operands

let matchAt parser (position: Zipper.PSGZipper) pointerBits operands =
    tryMatchWithDiagnostics parser position.Graph position.Focus position
        (coeffects position.Graph pointerBits) operands

let readArray fixture pointerBits operands =
    match matchAt Alex.Patterns.MemoryPatterns.pArrayGetIntrinsic fixture.Position pointerBits operands with
    | Result.Ok ((operations, result), position) -> operations, result, position
    | Result.Error message -> failwith message

let readModule fixture pointerBits =
    let operations, result, _ = readArray fixture pointerBits (recalledOperands fixture)
    match result with
    | TRValue value ->
        let body = operations @ [MLIROp.FuncOp(FuncOp.Return([{ SSA = value.SSA; Type = value.Type }]))]
        let definition = MLIROp.FuncOp(FuncOp.FuncDef(
            "read_index", [Arg 0, TMemRef(TInt(IntWidth 1)); Arg 1, TInt(IntWidth 8)], [value.Type], body, FuncVisibility.Public))
        Alex.Dialects.Core.Serialize.moduleToString (Ok pointerBits) "index_component" [definition]
    | other -> failwithf "Array.get did not produce a value: %A" other
