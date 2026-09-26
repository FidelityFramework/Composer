module Alex.Tests.RequirementTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.ScopeContext
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper

let private fixture () =
    let builder = NodeBuilder()
    let condition = builder.Create(SemanticKind.PatternBinding "condition", Types.boolType, dummyRange)
    let body = builder.Create(SemanticKind.Literal NativeLiteral.Unit, Types.unitType, dummyRange)
    let required = builder.Create(SemanticKind.Require(condition.Id, "Pattern match failed"), Types.unitType, dummyRange)
    let frontier = builder.Create(SemanticKind.Sequential [required.Id; body.Id], Types.unitType, dummyRange)
    let root = builder.Create(SemanticKind.Sequential [frontier.Id], Types.unitType, dummyRange)
    let raw = builder.Build []
    let row =
        { Sources = [required.Id; condition.Id; body.Id]; Target = frontier.Id
          Class = EdgeClass.Provenance; Role = EdgeRole.MatchRequirement; Ordinal = 1 }
    { raw with Edges = row :: raw.Edges }, root.Id, frontier.Id, required.Id, condition.Id

let private context graph root frontier site condition conditionType =
    let position = Zipper.create graph root |> require "Missing root" |> atChild frontier |> atChild site
    let accumulator = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode condition (Arg 0) conditionType accumulator
    let scope = ref (ScopeContext.root ())
    let visited = ref Set.empty
    { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
      ScopeContext = scope; RootScopeContext = scope; Graph = graph; Zipper = position
      GlobalVisited = visited; TraversalVisited = visited }

[<Fact>]
let ``ordered source requirement retains its diagnostic through backend realization and stock lowering`` () =
    let graph, root, frontier, site, condition = fixture ()
    let ctx = context graph root frontier site condition (TInt(IntWidth 1))
    let output = Alex.Witnesses.RequirementWitness.nanopass.Witness ctx ctx.Zipper.Focus
    match output.Result with TRValue _ -> () | other -> failwithf "Requirement lost its unit result: %A" other
    let check = output.InlineOps |> List.choose (function MLIROp.Assert(condition, diagnostic) -> Some(condition, diagnostic) | _ -> None) |> Assert.Single
    Assert.Equal((Arg 0, "Pattern match failed"), check)
    Assert.Empty output.TopLevelOps
    let declaration =
        MLIROp.FuncOp(FuncOp.FuncDef("requirement", [Arg 0, TInt(IntWidth 1)], [],
            output.InlineOps @ [MLIROp.FuncOp(FuncOp.Return [])], FuncVisibility.Private))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "requirement_component" [declaration]
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
    Assert.Contains("cf.assert %arg0", verified)
    let triple = "x86_64-unknown-linux-gnu"
    let backendContext: Core.Types.Pipeline.BackEndContext =
        { OutputPath = "unused"; IntermediatesDir = None; TargetTripleOverride = Some triple
          TargetPointerBits = Some 64; TargetCpu = None; PlatformOS = Some "linux"
          RuntimeModel = Some RuntimeModel.Libc; DeploymentMode = Core.Types.Dialects.Console
          EmitIntermediateOnly = false; ExternLibraries = Set.empty
          NativeLink = Core.Types.Pipeline.NativeLinkOptions.Empty
          EmbeddedTarget = None; XtensaTarget = None; Deploy = false }
    let witnessed: Core.Types.Pipeline.BackEndInput =
        { Operations = [declaration]; PointerBits = Ok 64; ModuleName = Some "requirement_component"; Text = text; WritableStorage = [] }
    let realized =
        BackEnd.LLVM.RequirementRealization.realize
            (BackEnd.LLVM.RequirementRealization.selectRuntime backendContext triple) witnessed
        |> Result.defaultWith failwith
    let lowered =
        MlirComponentTests.mlirOpt
            ["--verify-each"; "--pass-pipeline=builtin.module(convert-cf-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)"] realized.Text
    Assert.Contains("llvm.unreachable", lowered)
    Assert.Contains("Pattern match failed", lowered)

[<Theory>]
[<InlineData("missing-evidence")>]
[<InlineData("wrong-order")>]
[<InlineData("wrong-condition")>]
[<InlineData("foreign-position")>]
[<InlineData("foreign-snapshot")>]
[<InlineData("non-boolean")>]
let ``requirement rejects stale evidence operands and occurrences before emitting`` defect =
    let graph, root, frontier, site, condition = fixture ()
    let changed =
        match defect with
        | "missing-evidence" -> { graph with Edges = graph.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.MatchRequirement) }
        | "wrong-order" ->
            let node = graph.Nodes[frontier]
            { graph with Nodes = graph.Nodes.Add(frontier, { node with Children = List.rev node.Children }) }
        | "wrong-condition" ->
            let edges =
                graph.Edges |> List.map (fun edge ->
                    if edge.Role = EdgeRole.MatchRequirement then { edge with Sources = [site; site; List.last edge.Sources] } else edge)
            { graph with Edges = edges }
        | _ -> graph
    let carrier = if defect = "non-boolean" then TInt(IntWidth 32) else TInt(IntWidth 1)
    let ctx = context changed root frontier site condition carrier
    let ctx =
        match defect with
        | "foreign-position" -> { ctx with Zipper = Zipper.create changed site |> require "Missing detached requirement" }
        | "foreign-snapshot" -> { ctx with Graph = { changed with DeclarationRoots = [] } }
        | _ -> ctx
    let output = Alex.Witnesses.RequirementWitness.nanopass.Witness ctx ctx.Zipper.Focus
    match output.Result with TRError _ -> () | other -> failwithf "Expected requirement rejection: %A" other
    Assert.Empty output.InlineOps
    Assert.Empty output.TopLevelOps

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``a selected singleton returns its existing body carrier without an invented join`` unionPattern =
    // The component fixture supplies already selected operands. Source admission
    // is tested by RequirementCases and the terminal native match controls.
    let builder = NodeBuilder()
    let inputType = if unionPattern then NativeType.TApp(Types.optionTyCon, [Types.boolType]) else Types.boolType
    let input = builder.Create(SemanticKind.PatternBinding "input", inputType, dummyRange)
    let body = builder.Create(SemanticKind.PatternBinding "selected", Types.boolType, dummyRange)
    let selectedPattern =
        if unionPattern then Pattern.Union("Some", 1, Some Pattern.Wildcard, inputType)
        else Pattern.Const(NativeLiteral.Bool true)
    let arm = { Pattern = selectedPattern; Guard = None; Body = body.Id; Bindings = [] }
    let selected = builder.Create(SemanticKind.CaseElimination(input.Id, [arm]), Types.boolType, dummyRange)
    let graph = builder.Build []
    let position = Zipper.create graph selected.Id |> require "Missing selected match"
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode body.Id (Arg 1) (TInt(IntWidth 1)) operands
    let inputCarrier = if unionPattern then TMemRef(TInt(IntWidth 8)) else TInt(IntWidth 1)
    let inventedJoin = Alex.Traversal.Values.value selected.Id 0
    let parser =
        Alex.Patterns.ControlFlowPatterns.pBuildMatchElimination (Arg 0) inputCarrier input.Id
            [([], body.Id, arm)] (Some(inventedJoin, TInt(IntWidth 1))) selected.Id
    match matchAt parser position 64 operands with
    | Result.Ok ((operations, TRValue value), _) ->
        Assert.Empty operations
        Assert.Equal(Arg 1, value.SSA)
        Assert.NotEqual(inventedJoin, value.SSA)
        let definition = MLIROp.FuncOp(FuncOp.FuncDef("selected_body", [Arg 0, inputCarrier; Arg 1, value.Type],
            [value.Type], [MLIROp.FuncOp(FuncOp.Return [value])], FuncVisibility.Private))
        let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "selected_match" [definition]
        let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
        Assert.Contains("return %arg1 : i1", verified)
    | other -> failwithf "Selected body lost its carrier: %A" other

[<Fact>]
let ``portable requirement diagnostics preserve escaped UTF8 and embedded control bytes`` () =
    let message = "quoted \"message\" \\ newline\nUnicode λ 雪 NUL\000end"
    let declaration = MLIROp.FuncOp(FuncOp.FuncDef("diagnostic_bytes", [Arg 0, TInt(IntWidth 1)], [],
        [MLIROp.Assert(Arg 0, message); MLIROp.FuncOp(FuncOp.Return [])], FuncVisibility.Private))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "diagnostic_bytes" [declaration]
    Assert.Contains("\\22message\\22", text)
    Assert.Contains("\\CE\\BB", text)
    Assert.Contains("\\00end", text)
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
    let roundTrip = MlirComponentTests.mlirOpt ["--verify-each"] verified
    Assert.Equal(verified, roundTrip)
