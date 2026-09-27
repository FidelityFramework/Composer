module Alex.Tests.NumericOperationTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.CodeGeneration.TypeMapping
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures
module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission
module Zipper = Alex.Traversal.PSGZipper

let private fixture category =
    let expression, actuals =
        match category with
        | "divide" -> "first / second", "(-7) 700"
        | "unsigned-divide" -> "first / second", "7 3"
        | "compare" -> "if first < second then 1 else 0", "(-7) 700"
        | "negate" -> "-first", "(-7) 700"
        | "repeat" -> "first + first", "(-7) 700"
        | _ -> "first + second", "(-7) 700"
    let source = sprintf "module PublishedNumericOperation\nlet calculate first second = %s\n[<EntryPoint>]\nlet main _ = calculate %s\n" expression actuals
    let graph = checkScalarProgram source (category + "-numeric-operation.clef")
    let numeric = Publication.tryNumeric graph |> Result.defaultWith failwith
    let operation = numeric.Operations.Values |> Seq.filter (fun operation -> graph.Nodes[operation.Site].IsReachable) |> Assert.Single
    graph, operation

let private operands graph (operation: NumericOperationWitness) repeat wrong =
    let accumulator = MLIRAccumulator.empty ()
    let parameters = operation.Operands |> List.mapi (fun ordinal operand ->
        let ty = valueTypeAt graph operand.Actual
        let supplied = if wrong && ordinal = 0 then TInt(IntWidth 16) else ty
        let ssa = Arg(if repeat then 0 else ordinal)
        MLIRAccumulator.bindNode operand.Actual ssa supplied accumulator
        ssa, supplied)
    accumulator, List.distinct parameters

let private run graph (operation: NumericOperationWitness) accumulator =
    try
        let position = Zipper.create graph operation.Site |> require "Missing numeric operation occurrence"
        match matchAt (Alex.Patterns.ApplicationPatterns.pNumericOperation operation.Site) position 64 accumulator with
        | Result.Ok ((operations, result), _) -> Result.Ok(operations, result)
        | Result.Error reason -> Result.Error reason
    with error -> Result.Error error.Message

[<Theory>]
[<InlineData("add")>]
[<InlineData("divide")>]
[<InlineData("unsigned-divide")>]
[<InlineData("compare")>]
[<InlineData("negate")>]
[<InlineData("repeat")>]
let ``published numeric operation composes canonical adaptations and ordered actual values`` category =
    let graph, operation = fixture category
    let accumulator, parameters = operands graph operation (category = "repeat") false
    let operations, result = run graph operation accumulator |> Result.defaultWith failwith
    let value = match result with TRValue value -> value | other -> failwithf "Missing numeric result: %A" other
    Assert.Equal(scalarCarrierType operation.Result, value.Type)
    let adapted operand fallback =
        match meetFor graph operation.Site operand with
        | Some(_, ssa) -> ssa
        | None -> fallback
    let expected = operation.Operands |> List.mapi (fun ordinal operand -> adapted operand.Actual (Arg(if category = "repeat" then 0 else ordinal)))
    let operationType = operation.OperationCarrier |> Option.bind SettledScalar.tryType |> require "Missing published operation type"
    let position = Zipper.create graph operation.Site |> require "Missing numeric operation occurrence"
    let owners = Zipper.enclosingLambdaIds position
    let resultSSA = Alex.Traversal.Values.resultOf Core.Types.Dialects.CPU graph owners operation.Site
    let names = Alex.Traversal.Values.valuesOf Core.Types.Dialects.CPU graph owners operation.Site
    let exact =
        match category, expected with
        | "divide", [left; right] -> MLIROp.ArithOp(ArithOp.DivSI(resultSSA,left,right,operationType))
        | "unsigned-divide", [left; right] -> MLIROp.ArithOp(ArithOp.DivUI(resultSSA,left,right,operationType))
        | "compare", [left; right] -> MLIROp.ArithOp(ArithOp.CmpI(resultSSA,ICmpPred.Slt,left,right,operationType))
        | "negate", [operand] -> MLIROp.ArithOp(ArithOp.SubI(resultSSA,names[1],operand,operationType))
        | _, [left; right] -> MLIROp.ArithOp(ArithOp.AddI(resultSSA,left,right,operationType))
        | _ -> failwith "Unexpected source operation arity"
    Assert.Contains(exact, operations)
    if category <> "repeat" && category <> "unsigned-divide" then
        let actual = operation.Operands.Head.Actual
        let meet, ssa = meetFor graph operation.Site actual |> require "Expected source signed extension"
        Assert.Equal(MeetKind.ExtendSigned, meet.Adapt)
        Assert.Equal(8, meet.From)
        Assert.Equal(32, meet.To)
        Assert.Contains(MLIROp.ArithOp(ArithOp.ExtSI(ssa,Arg 0,TInt(IntWidth 8),TInt(IntWidth 32))), operations)
    let definition = MLIROp.FuncOp(FuncOp.FuncDef("numeric_operation",parameters,[value.Type],operations @ [MLIROp.FuncOp(FuncOp.Return[value])],FuncVisibility.Private))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Result.Ok 64) "numeric_operation" [definition]
    MlirComponentTests.mlirOpt ["--verify-each"] text |> ignore

[<Theory>]
[<InlineData("wrong-carrier")>]
[<InlineData("unpublished")>]
[<InlineData("stale-range")>]
[<InlineData("missing-meet")>]
let ``numeric operations refuse missing or contradictory premises without reselecting widths`` defect =
    let graph, operation = fixture "add"
    let accumulator, _ = operands graph operation false (defect = "wrong-carrier")
    let priorTypes = accumulator.SSATypes
    let changed =
        match defect with
        | "unpublished" -> Publication.invalidate graph
        | "stale-range" ->
            let id = operation.Operands.Head.Actual
            { graph with Nodes=graph.Nodes.Add(id,{graph.Nodes[id] with ValueRange=Some(ValueRange.Bounded(-65536I,65536I))}) }
        | "missing-meet" ->
            let previous = graph.Codata.Value
            {graph with Codata=lazy {previous with Meets=previous.Meets.Remove operation.Site}}
        | _ -> graph
    match run changed operation accumulator with
    | Result.Error reason ->
        Assert.Contains((if defect="wrong-carrier" then "operand differs from its source-published carrier" else "source Numeric publication is unavailable"), reason)
        Assert.Equal<Map<SSA, MLIRType>>(priorTypes, accumulator.SSATypes)
    | Result.Ok _ -> failwithf "Numeric operation accepted %s" defect
