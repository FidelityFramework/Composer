module Alex.Tests.RawConstantMatchTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper
module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission

// This fixture declares its complete numeric offer locally. Full-width literal
// acceptance is independent of a host default or the shared narrow fixture.
let private normalizedProgram category =
    let authority = """module MatchAuthority
type WidthDeclaration = { Name: string; Bits: int }
type Representation = { Name: string; Capability: string; Family: string; Bits: int; MinMagnitude: string; MaxMagnitude: string; Boundary: string }
type TargetCore = { Runtime: string; Widths: WidthDeclaration array; Representations: Representation array }
type PlatformDescription = { Id: string; Core: TargetCore option }
let description: PlatformDescription = {
    Id = "normalized-match"
    Core = Some {
        Runtime = "libc"
        Widths = [| { Name="Pointer"; Bits=64 }; { Name="Register"; Bits=64 } |]
        Representations = [|
            { Name="int8"; Capability="native"; Family="int"; Bits=8; MinMagnitude="-128"; MaxMagnitude="127"; Boundary="wrap" }
            { Name="uint8"; Capability="native"; Family="uint"; Bits=8; MinMagnitude="0"; MaxMagnitude="255"; Boundary="wrap" }
            { Name="int64"; Capability="native"; Family="int"; Bits=64; MinMagnitude="-9223372036854775808"; MaxMagnitude="9223372036854775807"; Boundary="wrap" }
            { Name="uint64"; Capability="native"; Family="uint"; Bits=64; MinMagnitude="0"; MaxMagnitude="18446744073709551615"; Boundary="wrap" } |] } }
"""
    let representation name family bits minimum maximum : NumericRepresentation =
        { Name=name; Family=family; Bits=bits; Capability="native"; MinMagnitude=minimum; MaxMagnitude=maximum; Boundary="wrap" }
    let offers =
        [ representation "int8" "int" 8 "-128" "127"
          representation "uint8" "uint" 8 "0" "255"
          representation "int64" "int" 64 "-9223372036854775808" "9223372036854775807"
          representation "uint64" "uint" 64 "0" "18446744073709551615" ]
    let platform : PlatformContext =
        { PlatformId="normalized-match"; Dimensions=Map.ofList ["Pointer",64; "Register",64]
          Representations=offers |> List.map (fun offer -> offer.Name,offer) |> Map.ofList; EndpointReturns=Map.empty
          PlatformLibraryPath=None; PlatformDescription=Some "MatchAuthority.description"; PlatformArchitecture=None; PlatformOS=None
          PlatformSourcePaths=Set.singleton (System.IO.Path.GetFullPath "match-authority.clef"); Predicates=Map.empty; FreestandingStartup=None; SubstrateKind=None
          RuntimeModel=Some RuntimeModel.Libc; AvailableMemorySpaces=[]; DefaultMemorySpace=None
          ClockFrequencyMhz=None; NsPerWeightUnit=None }
    let parameterType, literal, expected =
        match category with
        | "bool" -> "bool", "true", "arith.constant true"
        | "int64" -> "int", "-1099511627779", "arith.constant -1099511627779 : i64"
        | _ -> "int", "1099511627779", "arith.constant 1099511627779 : i64"
    let source = sprintf "module NormalizedMatch\nlet choose (value: %s) =\n    match value with\n    | %s -> true\n    | _ -> false\n[<EntryPoint>]\nlet main _ = if choose (%s) then 0 else 1\n" parameterType literal literal
    let parse source path =
        match Clef.Compiler.NativeService.parseStringWithDefaults source path with
        | Clef.Compiler.NativeService.ParseSuccess input -> input
        | Clef.Compiler.NativeService.ParseError errors -> failwithf "Match fixture parse failed: %A" errors
    let checkedProgram = Clef.Compiler.NativeService.checkParsedInputsWithPlatform
                            [parse authority "match-authority.clef"; parse source "normalized-match.clef"] (Some platform)
    let errors = checkedProgram.Diagnostics |> List.filter (fun diagnostic ->
        diagnostic.Severity = Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
    Assert.Empty errors
    prepareSource checkedProgram.Graph, expected

let private runOn target inputType inputCarrier patterns =
    let builder = NodeBuilder()
    let input = builder.Create(SemanticKind.PatternBinding "input", inputType, dummyRange)
    let bodies = patterns |> List.mapi (fun index _ -> builder.Create(SemanticKind.PatternBinding(sprintf "body%d" index), Types.boolType, dummyRange))
    let arms = List.map2 (fun pattern body -> { Pattern = pattern; Guard = None; Body = body.Id; Bindings = [] }) patterns bodies
    let choice = builder.Create(SemanticKind.CaseElimination(input.Id, arms), Types.boolType, dummyRange)
    // A negative component boundary: raw source syntax has no executable
    // publication. The Pattern must refuse before reading numeric premises.
    let graph = builder.Build []
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
let ``raw constants require source normalization even when their scalar values fit on fabric`` category =
    let inputType, carrier, literal =
        match category with
        | "bool" -> Types.boolType, TInt(IntWidth 1), NativeLiteral.Bool true
        | "int64" ->
            let value = (1L <<< 40) + 3L
            Types.intType, TInt(IntWidth 64), NativeLiteral.Int(value, NTUKind.NTUint(NTUWidth.Fixed 64))
        | _ -> Types.intType, TInt(IntWidth 64), NativeLiteral.UInt(System.UInt64.MaxValue, NTUKind.NTUuint(NTUWidth.Fixed 64))
    match runOn Core.Types.Dialects.FPGA inputType carrier [Pattern.Const literal; Pattern.Wildcard] with
    | Result.Error reason -> Assert.Contains("typed equality and conditional normalization", reason)
    | other -> failwithf "Raw scalar match was interpreted by a witness: %A" other

[<Theory>]
[<InlineData("bool")>]
[<InlineData("int")>]
let ``raw constants require source normalization even with an explicit final default`` category =
    let inputType, inputCarrier, literal =
        if category = "bool" then Types.boolType, TInt(IntWidth 1), NativeLiteral.Bool true
        else Types.intType, TInt(IntWidth 16), NativeLiteral.Int(937L, NTUKind.NTUint(NTUWidth.Fixed 64))
    match run inputType inputCarrier [Pattern.Const literal; Pattern.Wildcard] with
    | Result.Error reason -> Assert.Contains("typed equality and conditional normalization", reason)
    | other -> failwithf "Raw scalar match was interpreted by a witness: %A" other

[<Theory>]
[<InlineData("bool")>]
[<InlineData("int64")>]
[<InlineData("uint64")>]
let ``source normalized constant decisions retain exact full width values through the registry`` category =
    let graph, expected = normalizedProgram category
    let rawConstants = graph.Nodes.Values |> Seq.filter (fun node ->
        node.IsReachable &&
        (match node.Kind with
        | SemanticKind.CaseElimination(_, arms) -> arms |> List.exists (fun arm -> match arm.Pattern with Pattern.Const _ -> true | _ -> false)
        | _ -> false))
    Assert.Empty rawConstants
    let numeric = Publication.tryNumeric graph |> Result.defaultWith failwith
    let comparison = numeric.Operations.Values |> Seq.filter (fun operation -> operation.Kind = NumericOperationKind.Equal && graph.Nodes[operation.Site].IsReachable) |> Assert.Single
    let expectedSlot, expectedForm =
        match category with
        | "bool" -> SettledSlot.Bool, NumericOperationForm.Boolean
        | "int64" -> SettledSlot.Integer(64,Some "int64"), NumericOperationForm.Integer true
        | _ -> SettledSlot.Integer(64,Some "uint64"), NumericOperationForm.Integer false
    Assert.Equal(Some expectedSlot, comparison.OperationCarrier)
    Assert.Equal(expectedForm, comparison.Form)
    let sourceProof = Core.ProofDispatch.dischargeSource graph None |> Result.defaultWith failwith
    let witnessed, _ = MiddleEnd.MLIRGeneration.generateWithLinkedLibrariesAndProof graph graph.Platform.Value
                           Core.Types.Dialects.Console Core.Types.Dialects.CPU None Set.empty (Some sourceProof) |> Result.defaultWith failwith
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] witnessed.Text
    Assert.Contains(expected, verified)

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
