module Alex.Tests.RawConstantMatchTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper

let private runOn target inputType inputCarrier patterns =
    let builder = NodeBuilder()
    let input = builder.Create(SemanticKind.PatternBinding "input", inputType, dummyRange)
    let bodies = patterns |> List.mapi (fun index _ -> builder.Create(SemanticKind.PatternBinding(sprintf "body%d" index), Types.boolType, dummyRange))
    let arms = List.map2 (fun pattern body -> { Pattern = pattern; Guard = None; Body = body.Id; Bindings = [] }) patterns bodies
    let choice = builder.Create(SemanticKind.CaseElimination(input.Id, arms), Types.boolType, dummyRange)
    let raw = builder.Build []
    let graph =
        match Types.isIntegerType inputType, inputCarrier with
        | true, TInt(IntWidth bits) ->
            { raw with Nodes = raw.Nodes.Add(input.Id, { raw.Nodes[input.Id] with ValueRange = Some(ValueRange.Bounded(0I, (1I <<< bits) - 1I)) }) }
        | _ -> raw
    let position = Zipper.create graph choice.Id |> require "Missing raw decision"
    let operands = MLIRAccumulator.empty ()
    for index, body in List.indexed bodies do
        MLIRAccumulator.bindNode body.Id (Arg(index + 1)) (TInt(IntWidth 1)) operands
    let parser =
        Alex.Patterns.ControlFlowPatterns.pBuildMatchElimination (Arg 0) inputCarrier input.Id
            (arms |> List.map (fun arm -> [], arm.Body, arm))
            (Some(Alex.Traversal.Values.value choice.Id 0, TInt(IntWidth 1))) choice.Id
    Alex.XParsec.PSGCombinators.tryMatchWithDiagnostics parser graph position.Focus position
        { coeffects graph 64 with TargetPlatform = target } operands

let private run inputType inputCarrier patterns = runOn Core.Types.Dialects.CPU inputType inputCarrier patterns

[<Theory>]
[<InlineData("bool")>]
[<InlineData("int64")>]
[<InlineData("uint64")>]
let ``FPGA raw scalar decisions preserve the actual full discriminant without ordinal or int32 narrowing`` category =
    let inputType, carrier, literal, expected =
        match category with
        | "bool" -> Types.boolType, TInt(IntWidth 1), NativeLiteral.Bool true, 1L
        | "int64" ->
            let value = (1L <<< 40) + 3L
            Types.intType, TInt(IntWidth 64), NativeLiteral.Int(value, NTUKind.NTUint(NTUWidth.Fixed 64)), value
        | _ -> Types.intType, TInt(IntWidth 64), NativeLiteral.UInt(System.UInt64.MaxValue, NTUKind.NTUuint(NTUWidth.Fixed 64)), -1L
    match runOn Core.Types.Dialects.FPGA inputType carrier [Pattern.Const literal; Pattern.Wildcard] with
    | Result.Ok ((operations, TRValue _), _) ->
        let actual = operations |> List.choose (function MLIROp.ArithOp(ArithOp.ConstI(_, value, ty)) -> Some(value, ty) | _ -> None) |> Assert.Single
        Assert.Equal((expected, carrier), actual)
    | other -> failwithf "FPGA scalar discriminant lost its source value: %A" other

[<Theory>]
[<InlineData("bool")>]
[<InlineData("int")>]
let ``raw scalar decisions use actual admitted literal values and an explicit default`` category =
    let inputType, inputCarrier, literal =
        if category = "bool" then Types.boolType, TInt(IntWidth 1), NativeLiteral.Bool true
        else Types.intType, TInt(IntWidth 16), NativeLiteral.Int(937L, NTUKind.NTUint(NTUWidth.Fixed 64))
    match run inputType inputCarrier [Pattern.Const literal; Pattern.Wildcard] with
    | Result.Ok ((operations, TRValue value), _) ->
        let declaration =
            MLIROp.FuncOp(FuncOp.FuncDef("raw_constant", [Arg 0, inputCarrier; Arg 1, value.Type; Arg 2, value.Type], [value.Type],
                operations @ [MLIROp.FuncOp(FuncOp.Return [value])], FuncVisibility.Private))
        let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "raw_constant" [declaration]
        let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
        Assert.Contains((if category = "bool" then "arith.constant true" else "arith.constant 937"), verified)
    | other -> failwithf "Admitted scalar decision failed: %A" other

[<Theory>]
[<InlineData("char")>]
[<InlineData("float")>]
[<InlineData("string")>]
[<InlineData("unit")>]
[<InlineData("missing-default")>]
[<InlineData("early-default")>]
[<InlineData("mixed-constant-category")>]
[<InlineData("wrong-input-type")>]
[<InlineData("wrong-carrier")>]
[<InlineData("integer-overflow")>]
[<InlineData("negative-literal-unsigned-carrier")>]
[<InlineData("multiple-record-arms")>]
[<InlineData("multiple-tuple-arms")>]
let ``raw constant decisions reject missing source settlement instead of using positional labels`` defect =
    let boolean = Pattern.Const(NativeLiteral.Bool true)
    let inputType, carrier, patterns =
        match defect with
        | "char" -> Types.charType, TInt(IntWidth 32), [Pattern.Const(NativeLiteral.Char 'Ω'); Pattern.Wildcard]
        | "float" -> Types.floatType, TFloat F64, [Pattern.Const(NativeLiteral.Float(1.5, NTUKind.NTUfloat(NTUWidth.Fixed 64))); Pattern.Wildcard]
        | "string" -> Types.stringType, TMemRef(TInt(IntWidth 8)), [Pattern.Const(NativeLiteral.String "same"); Pattern.Wildcard]
        | "unit" -> Types.unitType, TInt(IntWidth 32), [Pattern.Const NativeLiteral.Unit; Pattern.Wildcard]
        | "missing-default" -> Types.boolType, TInt(IntWidth 1), [boolean; Pattern.Const(NativeLiteral.Bool false)]
        | "early-default" -> Types.boolType, TInt(IntWidth 1), [Pattern.Wildcard; boolean; Pattern.Wildcard]
        | "mixed-constant-category" -> Types.boolType, TInt(IntWidth 1), [boolean; Pattern.Const(NativeLiteral.Int(1L, NTUKind.NTUint(NTUWidth.Fixed 64))); Pattern.Wildcard]
        | "wrong-input-type" -> Types.charType, TInt(IntWidth 1), [boolean; Pattern.Wildcard]
        | "integer-overflow" -> Types.intType, TInt(IntWidth 16), [Pattern.Const(NativeLiteral.Int(70000L, NTUKind.NTUint(NTUWidth.Fixed 64))); Pattern.Wildcard]
        | "negative-literal-unsigned-carrier" -> Types.intType, TInt(IntWidth 16), [Pattern.Const(NativeLiteral.Int(-1L, NTUKind.NTUint(NTUWidth.Fixed 64))); Pattern.Wildcard]
        | "multiple-record-arms" ->
            let recordType = NativeType.TAnon([("Value", Types.boolType)], false)
            recordType, TMemRef(TInt(IntWidth 8)), [Pattern.Record([], recordType); Pattern.Wildcard]
        | "multiple-tuple-arms" ->
            NativeType.TTuple([Types.boolType], false), TMemRef(TInt(IntWidth 8)), [Pattern.Tuple [Pattern.Wildcard]; Pattern.Wildcard]
        | _ -> Types.boolType, TInt(IntWidth 32), [boolean; Pattern.Wildcard]
    match run inputType carrier patterns with
    | Result.Error _ -> ()
    | other -> failwithf "Raw constant decision invented settlement for %s: %A" defect other
