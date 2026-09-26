module Alex.Tests.CallableFlowTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures
module Carriers = Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers
module Flows = Clef.Compiler.PSGSaturation.SemanticGraph.CallableFlows
module Operands = Alex.Traversal.CallableOperands
module Zipper = Alex.Traversal.PSGZipper

let private ok = function Result.Ok value -> value | Result.Error reason -> failwith reason
let private fixture () =
    let builder = NodeBuilder()
    let callbackType = NativeType.TFun(Types.boolType, Types.boolType)
    let make name =
        let argument = builder.Create(SemanticKind.PatternBinding "argument", Types.boolType, dummyRange)
        let body = builder.Create(SemanticKind.VarRef("argument", Some argument.Id), Types.boolType, dummyRange)
        let code = builder.Create(SemanticKind.Lambda(["argument", Types.boolType, argument.Id], body.Id,
                                                    [], None, LambdaContext.RegularClosure), callbackType, dummyRange)
        let binding = builder.Create(SemanticKind.Binding(name, false, false, None), callbackType, dummyRange, children = [code.Id])
        builder.SetParent(code.Id, binding.Id)
        builder.Create(SemanticKind.VarRef(name, Some binding.Id), callbackType, dummyRange)
    let first, second = make "first", make "second"
    let formal = builder.Create(SemanticKind.PatternBinding "callback", callbackType, dummyRange)
    let alias = builder.Create(SemanticKind.VarRef("callback", Some formal.Id), callbackType, dummyRange)
    let value = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let body = builder.Create(SemanticKind.Application(alias.Id, [value.Id]), Types.boolType, dummyRange)
    let ty = NativeType.TFun(callbackType, Types.boolType)
    let higher = builder.Create(SemanticKind.Lambda(["callback", callbackType, formal.Id], body.Id,
                                                  [], None, LambdaContext.RegularClosure), ty, dummyRange)
    let binding = builder.Create(SemanticKind.Binding("higher", false, false, None), ty, dummyRange, children = [higher.Id])
    let apply actual =
        let reference = builder.Create(SemanticKind.VarRef("higher", Some binding.Id), ty, dummyRange)
        builder.Create(SemanticKind.Application(reference.Id, [actual]), Types.boolType, dummyRange)
    let firstCall, secondCall = apply first.Id, apply second.Id
    let opaque = builder.Create(SemanticKind.PatternBinding "opaque", callbackType, dummyRange)
    let unit = builder.Create(SemanticKind.PatternBinding "unit", Types.unitType, dummyRange)
    let spine = builder.Create(SemanticKind.Sequential [firstCall.Id; secondCall.Id], Types.boolType, dummyRange)
    let entryType = NativeType.TFun(Types.unitType, Types.boolType)
    let entry = builder.Create(SemanticKind.Lambda(["unit", Types.unitType, unit.Id], spine.Id,
                                                 [], None, LambdaContext.RegularClosure), entryType, dummyRange)
    let main = builder.Create(SemanticKind.Binding("main", false, false, Some DeclRoot.EntryPoint), entryType, dummyRange, children = [entry.Id])
    builder.SetParent(entry.Id, main.Id)
    let raw, startupErrors = Clef.Compiler.Nanopass.ProgramInitialization.normalize [main.Id] (builder.Build [main.Id, DeclRoot.EntryPoint])
    Assert.Empty startupErrors
    let carriers, _ = Carriers.settle { Layouts = Map.empty; Origins = Map.empty; Known = Map.empty } raw
    let flows, residuals = Flows.settle
                            { Carriers = carriers; Joins = Map.empty; Layouts = Map.empty
                              SequenceFlows = Map.empty; SequenceFamilies = Map.empty } raw
    Assert.Empty residuals
    let codata = { raw.Codata.Value with CallableCarriers = carriers; CallableFlows = flows }
    { raw with Codata = lazy codata }, formal.Id, alias.Id, first.Id, secondCall.Id, opaque.Id

let private context graph id bits =
    let operands = MLIRAccumulator.empty ()
    let scope = ref (Alex.Traversal.ScopeContext.ScopeContext.root ())
    let visited = ref Set.empty
    { Coeffects = coeffects graph bits; Accumulator = operands; RootAccumulator = operands
      ScopeContext = scope; RootScopeContext = scope; Graph = graph
      Zipper = Zipper.create graph id |> require "Missing formal occurrence"
      GlobalVisited = visited; TraversalVisited = visited }

[<Theory>]
[<InlineData(32)>]
[<InlineData(64)>]
let ``ordinary multi-target formal forwards its actual code operand through a verified indirect call`` bits =
    let graph, formal, alias, first, _, _ = fixture ()
    let ctx = context graph formal bits
    let shape = Operands.project ctx formal |> ok
    Assert.Equal(None, Operands.environmentType shape)
    let code = { SSA = Arg 0; Type = Operands.functionType shape }
    Operands.bind ctx formal code None |> ok
    Operands.copy ctx formal alias |> ok
    let actual = MLIRAccumulator.recallCallable alias ctx.Accumulator |> require "Alias lost callable operand"
    Assert.Equal(Arg 0, (Operands.code actual).SSA)
    Assert.Single(Operands.values actual) |> ignore
    Assert.True((MLIRAccumulator.recallNode alias ctx.Accumulator).IsNone)
    // A join cannot become one arbitrary alternative even though signatures match.
    match Operands.reproject ctx alias first with
    | Result.Error _ -> ()
    | Result.Ok _ -> failwith "Callable flow was narrowed to a representative implementation"
    let boolean = TInt(IntWidth 1)
    let result = { SSA = V(501, 0); Type = boolean }
    let body = [MLIROp.FuncOp(FuncOp.FuncCallIndirect([result], code.SSA, [{ SSA = Arg 1; Type = boolean }]))
                MLIROp.FuncOp(FuncOp.Return [result])]
    let definition = MLIROp.FuncOp(FuncOp.FuncDef("higher_order_flow", [Arg 0, code.Type; Arg 1, boolean], [boolean], body, FuncVisibility.Public))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok bits) "callable_flow" [definition]
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
    Assert.Contains("call_indirect", verified)
    Assert.DoesNotContain("unrealized_conversion_cast", verified)
    Assert.DoesNotContain("memref", verified)

[<Fact>]
let ``an added opaque actual retracts a previously projected formal on a new graph snapshot`` () =
    let graph, formal, _, _, call, opaque = fixture ()
    Operands.project (context graph formal 64) formal |> ok |> ignore
    let node = graph.Nodes[call]
    let callee = match node.Kind with SemanticKind.Application(callee, _) -> callee | _ -> failwith "Missing fixture call"
    let changed = { node with Kind = SemanticKind.Application(callee, [opaque]); Children = [callee; opaque] }
    let revised = { graph with Nodes = graph.Nodes.Add(call, changed) }
    match Operands.project (context revised formal 64) formal with
    | Result.Error _ -> ()
    | Result.Ok _ -> failwith "Stale complete callable flow survived the changed actual"
