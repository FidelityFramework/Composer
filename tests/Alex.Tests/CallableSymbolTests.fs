module Alex.Tests.CallableSymbolTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.CodeGeneration.CallableSymbols
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.ScopeContext
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper

type private Callable = { Binding: NodeId; Lambda: NodeId; Argument: NodeId; Reference: NodeId; Call: NodeId }

/// Resolved component inputs: two separate scopes deliberately use the same
/// source spelling. This does not reproduce name resolution or curry analysis.
let private fixture saturated =
    let builder = NodeBuilder()
    let callable () =
        let parameter = builder.Create(SemanticKind.PatternBinding "value", Types.boolType, dummyRange)
        let body = builder.Create(SemanticKind.VarRef("value", Some parameter.Id), Types.boolType, dummyRange)
        let ty = NativeType.TFun(Types.boolType, Types.boolType)
        let fn = builder.Create(SemanticKind.Lambda(["value", Types.boolType, parameter.Id], body.Id,
                                                   [], None, LambdaContext.RegularClosure), ty, dummyRange)
        let binding = builder.Create(SemanticKind.Binding("read", false, false, None), ty, dummyRange, children = [fn.Id])
        builder.SetParent(fn.Id, binding.Id)
        builder.SetParent(parameter.Id, fn.Id)
        builder.SetParent(body.Id, fn.Id)
        let argument = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
        let reference = builder.Create(SemanticKind.VarRef("read", Some binding.Id), ty, dummyRange)
        let call = builder.Create(SemanticKind.Application(reference.Id, [argument.Id]), Types.boolType, dummyRange)
        let scope = builder.Create(SemanticKind.Sequential [binding.Id; call.Id], Types.boolType, dummyRange)
        builder.SetParent(binding.Id, scope.Id)
        builder.SetParent(call.Id, scope.Id)
        builder.SetParent(reference.Id, call.Id)
        builder.SetParent(argument.Id, call.Id)
        { Binding = binding.Id; Lambda = fn.Id; Argument = argument.Id; Reference = reference.Id; Call = call.Id }
    let first, second = callable (), callable ()
    let raw = builder.Build []
    let codata =
        if saturated then
            let calls =
                [first; second]
                |> List.map (fun fn -> fn.Call, { TargetBindingId = fn.Binding; AllArgNodes = [fn.Argument] })
                |> Map.ofList
            { Codata.empty with Curry = { Codata.empty.Curry with SaturatedCalls = calls } }
        else Codata.empty
    { raw with Codata = lazy codata }, [first; second]

let private context (graph: SemanticGraph) site operands =
    let rootScope = ref (ScopeContext.root ())
    let scope = ref (ScopeContext.createChild rootScope.Value FunctionLevel)
    let visited = ref Set.empty
    { Coeffects = coeffects graph 64
      Accumulator = operands; RootAccumulator = operands
      ScopeContext = scope; RootScopeContext = rootScope
      Graph = graph; Zipper = Zipper.create graph site |> require "Missing component site"
      GlobalVisited = visited; TraversalVisited = visited }

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``ordinary and saturated calls preserve resolved identity across equal local names`` saturated =
    let graph, functions = fixture saturated
    let symbols =
        functions |> List.map (fun fn ->
            let definitionSymbol = lambda graph graph.Nodes[fn.Lambda] false
            let operands = MLIRAccumulator.empty ()
            MLIRAccumulator.bindNode fn.Argument (Arg 0) (TInt(IntWidth 1)) operands
            let ctx = context graph fn.Call operands
            let output = Alex.Witnesses.ApplicationWitness.nanopass.Witness ctx graph.Nodes[fn.Call]
            match output.Result with
            | TRValue _ -> ()
            | other -> failwithf "Call was not witnessed: %A" other
            let targets = output.InlineOps |> List.choose (function MLIROp.FuncOp(FuncOp.FuncCall(_, target, _, _)) -> Some target | _ -> None)
            Assert.Equal(definitionSymbol, Assert.Single targets)
            Assert.Equal(Some definitionSymbol, Alex.Patterns.HardwareModulePatterns.resolveStepFunctionName graph fn.Reference)
            Assert.Same(graph, ctx.Graph)
            Assert.Empty(ctx.TraversalVisited.Value)
            Assert.Empty(operands.Errors)
            definitionSymbol)
    Assert.NotEqual<string>(symbols[0], symbols[1])

[<Fact>]
let ``module and external spellings and anonymous closure identities are preserved`` () =
    let graph, functions = fixture false
    let fn = functions.Head
    let builder = NodeBuilder()
    let moduleNode = builder.Create(SemanticKind.ModuleDef("Library", [fn.Binding]), Types.unitType, dummyRange)
    let binding = graph.Nodes[fn.Binding]
    let moduleGraph = { graph with Nodes = graph.Nodes.Add(moduleNode.Id, moduleNode).Add(fn.Binding, { binding with Parent = Some moduleNode.Id }) }
    Assert.Equal(Some "Library.read", tryBinding moduleGraph fn.Binding)
    Assert.Equal("Library.read", lambda moduleGraph moduleGraph.Nodes[fn.Lambda] false)
    Assert.Equal(Some "Library.read", Alex.Patterns.HardwareModulePatterns.resolveStepFunctionName moduleGraph fn.Reference)
    let externalGraph = { graph with Nodes = graph.Nodes.Add(fn.Binding, { binding with Parent = None }) }
    Assert.Equal(Some "read", tryBinding externalGraph fn.Binding)
    Assert.Equal("read", lambda externalGraph externalGraph.Nodes[fn.Lambda] false)
    let original = graph.Nodes[fn.Lambda]
    for key in [ClosureMetadata.LambdaExpression; ClosureMetadata.RequiresClosurePair] do
        let anonymous = { original with Metadata = original.Metadata.Add(key, MetadataValue.Bool true) }
        Assert.Equal(sprintf "lambda_%d" (NodeId.value fn.Lambda), lambda graph anonymous true)

    // Native address plans are already settled by CCS. The witness retains that
    // symbol; this local-name projection does not re-resolve native entry plans.
    let addresses = Map.ofList [fn.Call, FunctionPointerPlan.Address("Library.read", fn.Lambda)]
    let addressGraph = { moduleGraph with Codata = lazy { Codata.empty with FunctionPointers = addresses } }
    let ctx = context addressGraph fn.Call (MLIRAccumulator.empty ())
    let output = Alex.Witnesses.FunctionPointerWitness.nanopass.Witness ctx addressGraph.Nodes[fn.Call]
    match output.Result with
    | TRValue _ -> ()
    | other -> failwithf "Native address was not witnessed: %A" other
    let targets = output.InlineOps |> List.choose (function MLIROp.FuncOp(FuncOp.FuncConstant(_, target, _)) -> Some target | _ -> None)
    Assert.Equal("Library.read", Assert.Single targets)
