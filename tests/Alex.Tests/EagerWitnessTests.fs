module Alex.Tests.EagerWitnessTests

open Xunit
open Clef.Compiler.NativeService
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.ScopeContext
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper
module Demand = Clef.Compiler.Nanopass.EagerDemand

let private context graph position accumulator =
    let scope = ref (ScopeContext.root ())
    let visited = ref Set.empty
    { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
      ScopeContext = scope; RootScopeContext = scope; Graph = graph; Zipper = position
      GlobalVisited = visited; TraversalVisited = visited }

let private fixture () =
    let builder = NodeBuilder()
    let operand = builder.Create(SemanticKind.PatternBinding "alreadyComputed", Types.boolType, dummyRange)
    let marker = builder.Create(SemanticKind.EagerExpr operand.Id, Types.boolType, dummyRange)
    let root = builder.Create(SemanticKind.Sequential [marker.Id], Types.boolType, dummyRange)
    let graph = builder.Build [] |> Demand.normalize
    graph, root.Id, marker.Id, operand.Id

[<Fact>]
let ``explicit demand forwards its already witnessed operand without replaying effects`` () =
    let graph, root, marker, operand = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode operand (Arg 0) (TInt(IntWidth 1)) accumulator
    let position = Zipper.create graph root |> require "Missing root" |> atChild marker
    let ctx = context graph position accumulator
    let output = Alex.Witnesses.EagerWitness.nanopass.Witness ctx position.Focus
    Assert.Equal(TRValue { SSA = Arg 0; Type = TInt(IntWidth 1) }, output.Result)
    Assert.Empty output.InlineOps
    Assert.Empty output.TopLevelOps
    Assert.Empty accumulator.AllOps
    Assert.Empty ctx.TraversalVisited.Value
    Assert.Same(position, ctx.Zipper)

[<Theory>]
[<InlineData("missing")>]
[<InlineData("duplicate")>]
[<InlineData("changed operand")>]
[<InlineData("wrong class")>]
[<InlineData("wrong ordinal")>]
[<InlineData("wrong frontier")>]
[<InlineData("malformed marker")>]
[<InlineData("missing value")>]
[<InlineData("foreign graph")>]
let ``explicit demand refuses stale authority or absent scoped values`` defect =
    let graph, root, marker, operand = fixture ()
    let row = graph.Edges |> List.find (fun edge -> edge.Target = marker && edge.Role = EdgeRole.EagerDemand EagerFrontier.Expression)
    let replace updated =
        { graph with Edges = updated :: (graph.Edges |> List.filter (fun edge -> edge.Target <> marker)) }
    let graph =
        match defect with
        | "missing" -> { graph with Edges = graph.Edges |> List.filter (fun edge -> edge.Target <> marker) }
        | "duplicate" -> { graph with Edges = row :: graph.Edges }
        | "changed operand" -> replace { row with Sources = [marker; root] }
        | "wrong class" -> replace { row with Class = EdgeClass.Provenance }
        | "wrong ordinal" -> replace { row with Ordinal = 1 }
        | "wrong frontier" -> replace { row with Role = EdgeRole.EagerDemand EagerFrontier.Binding }
        | "malformed marker" -> { graph with Nodes = graph.Nodes.Add(marker, { graph.Nodes[marker] with Children = [] }) }
        | _ -> graph
    let accumulator = MLIRAccumulator.empty ()
    if defect <> "missing value" then MLIRAccumulator.bindNode operand (Arg 0) (TInt(IntWidth 1)) accumulator
    let position = Zipper.create graph root |> require "Missing root" |> atChild marker
    let ctx = context graph position accumulator
    let ctx = if defect = "foreign graph" then { ctx with Graph = { graph with DeclarationRoots = [] } } else ctx
    let output = Alex.Witnesses.EagerWitness.nanopass.Witness ctx position.Focus
    match output.Result with TRError _ -> () | other -> failwithf "Malformed demand was witnessed: %A" other
    Assert.Empty output.InlineOps
    Assert.Empty output.TopLevelOps

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``an eager effect remains inside its conditional arm for either guard value`` active =
    let builder = NodeBuilder()
    let guard = builder.Create(SemanticKind.Literal(NativeLiteral.Bool active), Types.boolType, dummyRange)
    let effect = builder.Create(SemanticKind.PatternBinding "effectfulOperand", Types.boolType, dummyRange)
    let marker = builder.Create(SemanticKind.EagerExpr effect.Id, Types.boolType, dummyRange)
    let fallback = builder.Create(SemanticKind.Literal(NativeLiteral.Bool false), Types.boolType, dummyRange)
    let choice = builder.Create(SemanticKind.IfThenElse(guard.Id, marker.Id, Some fallback.Id), Types.boolType, dummyRange)
    let graph = builder.Build [] |> Demand.normalize
    let accumulator = MLIRAccumulator.empty ()
    let position = Zipper.create graph choice.Id |> require "Missing conditional"
    let ctx = context graph position accumulator
    let effectValue = { SSA = Alex.Traversal.Values.value effect.Id 0; Type = TInt(IntWidth 1) }
    let visits = ResizeArray<NodeId>()
    let rec witness ctx node =
        Assert.Equal(node.Id, ctx.Zipper.Focus.Id)
        if node.Id = effect.Id then
            visits.Add node.Id
            let parent = Zipper.up ctx.Zipper |> require "Effect lost its marker"
            let conditional = Zipper.up parent |> require "Marker lost its conditional"
            Assert.Equal(marker.Id, parent.Focus.Id)
            Assert.Equal(choice.Id, conditional.Focus.Id)
            { InlineOps = [MLIROp.FuncOp(FuncOp.FuncCall([effectValue], "observe_effect", []))]
              TopLevelOps = []; Result = TRValue effectValue }
        else
            match node.Kind with
            | SemanticKind.IfThenElse _ -> (Alex.Witnesses.ControlFlowWitness.createNanopass (fun () -> witness)).Witness ctx node
            | SemanticKind.EagerExpr _ -> Alex.Witnesses.EagerWitness.nanopass.Witness ctx node
            | SemanticKind.Literal _ -> Alex.Witnesses.LiteralWitness.nanopass.Witness ctx node
            | _ -> Alex.Witnesses.StructuralWitness.nanopass.Witness ctx node
    visitAllNodes witness ctx position.Focus ctx.TraversalVisited
    Assert.Empty accumulator.Errors
    Assert.Equal(effect.Id, Assert.Single visits)
    let operations = ScopeContext.getOps ctx.ScopeContext.Value
    Assert.DoesNotContain(operations, function MLIROp.FuncOp(FuncOp.FuncCall(_, "observe_effect", _)) -> true | _ -> false)
    let yes, no = operations |> List.choose (function MLIROp.SCFOp(SCFOp.If(_, yes, Some no, _)) -> Some(yes, no) | _ -> None) |> Assert.Single
    Assert.Single(yes |> List.filter (function MLIROp.FuncOp(FuncOp.FuncCall(_, "observe_effect", _)) -> true | _ -> false)) |> ignore
    Assert.DoesNotContain(no, function MLIROp.FuncOp(FuncOp.FuncCall(_, "observe_effect", _)) -> true | _ -> false)
    let result, resultType = MLIRAccumulator.recallNode choice.Id accumulator |> require "Conditional lost its value"
    let declaration = MLIROp.FuncOp(FuncOp.FuncDef("conditional_demand", [], [resultType],
        operations @ [MLIROp.FuncOp(FuncOp.Return [{ SSA = result; Type = resultType }])], FuncVisibility.Private))
    let effectDeclaration = MLIROp.FuncOp(FuncOp.FuncDecl("observe_effect", [], [TInt(IntWidth 1)], FuncVisibility.Private, []))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "conditional_demand" [effectDeclaration; declaration]
    MlirComponentTests.mlirOpt ["--verify-each"] text |> ignore

[<Fact>]
let ``eager lazy value preserves its actual code and environment without forcing the payload`` () =
    let source = "module EagerLazy\n[<EntryPoint>]\nlet main _ =\n    let pending = lazy true\n    let demanded = eager pending\n    ignore (Lazy.force demanded)\n    0\n"
    let raw =
        match parseAndCheck source "eager-lazy.clef" with
        | Success result -> result.Graph
        | other -> failwithf "Expected checked lazy demand: %A" other
    let platform: PlatformContext =
        { PlatformId = "eager-test"; Dimensions = Map.ofList ["Pointer", 64; "Register", 64]
          Representations = Map.empty; EndpointReturns = Map.empty; PlatformLibraryPath = None
          PlatformDescription = None; PlatformArchitecture = None; PlatformOS = None; PlatformSourcePaths = Set.empty
          Predicates = Map.empty; FreestandingStartup = None; SubstrateKind = None; RuntimeModel = None
          AvailableMemorySpaces = []; DefaultMemorySpace = None; ClockFrequencyMhz = None; NsPerWeightUnit = None }
    let prepared, _ = Clef.Compiler.Nanopass.LazyFactoryResults.prepare raw raw.Codata.Value.Curry
    let settled, reading = Clef.Compiler.Nanopass.LazyRuntime.settle { prepared with Platform = Some platform }
    Assert.Empty reading.Diagnostics
    let graph = Demand.normalize settled
    let marker = graph.Nodes.Values |> Seq.filter (fun node -> node.IsReachable && match node.Kind with SemanticKind.EagerExpr _ -> true | _ -> false) |> Assert.Single
    let operand = Clef.Compiler.PSGSaturation.SemanticGraph.ExplicitDemand.operand graph marker.Id |> require "Missing source operand"
    let accumulator = MLIRAccumulator.empty ()
    let position = Zipper.create graph marker.Id |> require "Missing lazy marker"
    let ctx = context graph position accumulator
    let shape = Alex.Traversal.LazyOperands.project ctx operand |> Result.defaultWith failwith
    let code = { SSA = Arg 0; Type = Alex.Traversal.LazyOperands.functionType shape }
    let environment = { SSA = Arg 1; Type = Alex.Traversal.LazyOperands.environmentType shape }
    let value = Alex.Traversal.LazyOperands.create shape code environment |> Result.defaultWith failwith
    MLIRAccumulator.bindLazy operand value accumulator |> Result.defaultWith failwith
    let output = Alex.Witnesses.EagerWitness.nanopass.Witness ctx marker
    match output.Result with
    | TRLazy result ->
        Assert.Equal(marker.Id, Alex.Traversal.LazyOperands.occurrence result)
        Assert.Equal<Val list>([code; environment], Alex.Traversal.LazyOperands.values result)
    | other -> failwithf "Explicit demand changed the lazy pair: %A" other
    Assert.Empty output.InlineOps
    Assert.Empty output.TopLevelOps
    Assert.Empty accumulator.AllOps
    Assert.Empty ctx.TraversalVisited.Value

let private unitFixture () =
    let builder = NodeBuilder()
    let effect = builder.Create(SemanticKind.PatternBinding "consoleWritelnResult", Types.unitType, dummyRange)
    let marker = builder.Create(SemanticKind.EagerExpr effect.Id, Types.unitType, dummyRange)
    let root = builder.Create(SemanticKind.Sequential [marker.Id], Types.unitType, dummyRange)
    let graph = builder.Build [] |> Demand.normalize
    graph, root.Id, marker.Id, effect.Id

[<Theory>]
[<InlineData("absent")>]
[<InlineData("visited only")>]
[<InlineData("different scope")>]
[<InlineData("different path")>]
[<InlineData("different graph")>]
let ``unit demand requires successful completion at the actual child occurrence`` defect =
    let graph, root, marker, effect = unitFixture ()
    let operands = MLIRAccumulator.empty ()
    let position = Zipper.create graph root |> require "Missing unit root" |> atChild marker
    let ctx = context graph position operands
    match defect with
    | "visited only" -> ctx.GlobalVisited.Value <- Set.singleton effect
    | "different scope" -> MLIRAccumulator.completeVoid (atChild effect position) (ref (ScopeContext.root ())) operands
    | "different path" -> MLIRAccumulator.completeVoid (Zipper.create graph effect |> require "Missing detached effect") ctx.ScopeContext operands
    | "different graph" ->
        let other = { graph with DeclarationRoots = [] }
        let child = Zipper.create other root |> require "Missing other root" |> atChild marker |> atChild effect
        MLIRAccumulator.completeVoid child ctx.ScopeContext operands
    | _ -> ()
    let output = Alex.Witnesses.EagerWitness.nanopass.Witness ctx position.Focus
    match output.Result with TRError _ -> () | result -> failwithf "Unit result invented from %s: %A" defect result
    Assert.Empty output.InlineOps
    Assert.Empty output.TopLevelOps

[<Fact>]
let ``actual unit store completes once before the eager marker returns canonical unit`` () =
    let builder = NodeBuilder()
    let initial = builder.Create(SemanticKind.Literal(NativeLiteral.Bool false), Types.boolType, dummyRange)
    let cell = builder.Create(SemanticKind.Binding("cell", true, false, None), Types.boolType, dummyRange, children = [initial.Id])
    let target = builder.Create(SemanticKind.VarRef("cell", Some cell.Id), Types.boolType, dummyRange)
    let value = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let store = builder.Create(SemanticKind.Set(target.Id, value.Id), Types.unitType, dummyRange)
    let marker = builder.Create(SemanticKind.EagerExpr store.Id, Types.unitType, dummyRange)
    let graph = builder.Build [] |> Demand.normalize
    let accumulator = MLIRAccumulator.empty ()
    let cellType = TMemRefStatic(1, TInt(IntWidth 1))
    MLIRAccumulator.bindNode cell.Id (Arg 0) cellType accumulator
    let position = Zipper.create graph marker.Id |> require "Missing store marker"
    let ctx = context graph position accumulator
    ctx.TraversalVisited.Value <- Set.singleton cell.Id
    let witness ctx node =
        match node.Kind with
        | SemanticKind.Set _ -> Alex.Witnesses.MutableAssignmentWitness.nanopass.Witness ctx node
        | SemanticKind.VarRef _ -> Alex.Witnesses.VarRefWitness.nanopass.Witness ctx node
        | SemanticKind.Literal _ -> Alex.Witnesses.LiteralWitness.nanopass.Witness ctx node
        | SemanticKind.EagerExpr _ -> Alex.Witnesses.EagerWitness.nanopass.Witness ctx node
        | _ -> Alex.Witnesses.StructuralWitness.nanopass.Witness ctx node
    visitAllNodes witness ctx position.Focus ctx.TraversalVisited
    Assert.Empty accumulator.Errors
    Assert.True(MLIRAccumulator.completedVoid (atChild store.Id position) ctx.ScopeContext accumulator)
    let result, resultType = MLIRAccumulator.recallNode marker.Id accumulator |> require "Missing unit value after actual store"
    Assert.Equal(TInt(IntWidth 32), resultType)
    let operations = ScopeContext.getOps ctx.ScopeContext.Value
    Assert.Single(operations |> List.filter (function MLIROp.MemRefOp(MemRefOp.Store _) -> true | _ -> false)) |> ignore
    let declaration = MLIROp.FuncOp(FuncOp.FuncDef("eager_store", [Arg 0, cellType], [resultType],
        operations @ [MLIROp.FuncOp(FuncOp.Return [{ SSA = result; Type = resultType }])], FuncVisibility.Private))
    Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "eager_store" [declaration]
    |> MlirComponentTests.mlirOpt ["--verify-each"] |> ignore

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``void Console call output remains in its eager conditional arm before unit completion`` active =
    // The component supplies an already admitted void Console ABI operation.
    // It tests completion/placement, not source platform admission or linking.
    let builder = NodeBuilder()
    let guard = builder.Create(SemanticKind.Literal(NativeLiteral.Bool active), Types.boolType, dummyRange)
    let effect = builder.Create(SemanticKind.PatternBinding "Console.writeln result", Types.unitType, dummyRange)
    let marker = builder.Create(SemanticKind.EagerExpr effect.Id, Types.unitType, dummyRange)
    let fallback = builder.Create(SemanticKind.Literal NativeLiteral.Unit, Types.unitType, dummyRange)
    let choice = builder.Create(SemanticKind.IfThenElse(guard.Id, marker.Id, Some fallback.Id), Types.unitType, dummyRange)
    let graph = builder.Build [] |> Demand.normalize
    let accumulator = MLIRAccumulator.empty ()
    let position = Zipper.create graph choice.Id |> require "Missing unit conditional"
    let ctx = context graph position accumulator
    let rec witness ctx node =
        if node.Id = effect.Id then
            { InlineOps = [MLIROp.FuncOp(FuncOp.FuncCall([], "Console.writeln", []))]
              TopLevelOps = []; Result = TRVoid }
        else
            match node.Kind with
            | SemanticKind.IfThenElse _ -> (Alex.Witnesses.ControlFlowWitness.createNanopass (fun () -> witness)).Witness ctx node
            | SemanticKind.EagerExpr _ -> Alex.Witnesses.EagerWitness.nanopass.Witness ctx node
            | SemanticKind.Literal _ -> Alex.Witnesses.LiteralWitness.nanopass.Witness ctx node
            | _ -> Alex.Witnesses.StructuralWitness.nanopass.Witness ctx node
    visitAllNodes witness ctx position.Focus ctx.TraversalVisited
    Assert.Empty accumulator.Errors
    let operations = ScopeContext.getOps ctx.ScopeContext.Value
    let isCall = function MLIROp.FuncOp(FuncOp.FuncCall(_, "Console.writeln", _)) -> true | _ -> false
    Assert.DoesNotContain(operations, isCall)
    let yes, no = operations |> List.choose (function MLIROp.SCFOp(SCFOp.If(_, yes, Some no, _)) -> Some(yes, no) | _ -> None) |> Assert.Single
    Assert.Single(List.filter isCall yes) |> ignore
    Assert.DoesNotContain(no, isCall)
    Assert.True(yes |> List.exists (function MLIROp.ArithOp(ArithOp.ConstI(_, 0L, TInt(IntWidth 32))) -> true | _ -> false))
    let result, resultType = MLIRAccumulator.recallNode choice.Id accumulator |> require "Missing conditional unit"
    let declaration = MLIROp.FuncOp(FuncOp.FuncDef("eager_console", [], [resultType],
        operations @ [MLIROp.FuncOp(FuncOp.Return [{ SSA = result; Type = resultType }])], FuncVisibility.Private))
    let effectDeclaration = MLIROp.FuncOp(FuncOp.FuncDecl("Console.writeln", [], [], FuncVisibility.Private, []))
    Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "eager_console" [effectDeclaration; declaration]
    |> MlirComponentTests.mlirOpt ["--verify-each"] |> ignore

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``failed void operand or subtree cannot acquire successful completion`` subtree =
    let graph, root, marker, effect = unitFixture ()
    let failure = NodeId((graph.Nodes.Keys |> Seq.map NodeId.value |> Seq.max) + 1)
    let graph =
        if subtree then
            let child = { graph.Nodes[effect] with Id = failure; Kind = SemanticKind.Literal NativeLiteral.Unit; Parent = Some effect; Children = [] }
            let container = { graph.Nodes[effect] with Kind = SemanticKind.Sequential [failure]; Children = [failure] }
            { graph with Nodes = graph.Nodes.Add(failure, child).Add(effect, container) }
        else graph
    let accumulator = MLIRAccumulator.empty ()
    let position = Zipper.create graph root |> require "Missing unit root" |> atChild marker
    let ctx = context graph position accumulator
    let witness ctx node =
        if node.Id = (if subtree then failure else effect) then WitnessOutput.error "actual effect failed"
        elif node.Id = effect then Alex.Witnesses.StructuralWitness.nanopass.Witness ctx node
        else Alex.Witnesses.EagerWitness.nanopass.Witness ctx node
    visitAllNodes witness ctx position.Focus ctx.TraversalVisited
    Assert.NotEmpty accumulator.Errors
    Assert.False(MLIRAccumulator.completedVoid (atChild effect position) ctx.ScopeContext accumulator)
    Assert.True((MLIRAccumulator.recallNode marker accumulator).IsNone)
    Assert.Empty(ScopeContext.getOps ctx.ScopeContext.Value)

[<Fact>]
let ``void completion follows complete operand scope snapshot restoration`` () =
    let graph, root, marker, effect = unitFixture ()
    let accumulator = MLIRAccumulator.empty ()
    let position = Zipper.create graph root |> require "Missing unit root" |> atChild marker
    let ctx = context graph position accumulator
    let child = atChild effect position
    MLIRAccumulator.completeVoid child ctx.ScopeContext accumulator
    let saved = MLIRAccumulator.snapshotOperands accumulator
    MLIRAccumulator.completeVoid child (ref (ScopeContext.root ())) accumulator
    Assert.False(MLIRAccumulator.completedVoid child ctx.ScopeContext accumulator)
    MLIRAccumulator.restoreOperands saved accumulator
    Assert.True(MLIRAccumulator.completedVoid child ctx.ScopeContext accumulator)
    MLIRAccumulator.bindNode effect (Arg 0) (TInt(IntWidth 32)) accumulator
    Assert.False(MLIRAccumulator.completedVoid child ctx.ScopeContext accumulator)

[<Fact>]
let ``a transparent unit parent cannot certify emission of a deferred effect child`` () =
    let builder = NodeBuilder()
    let deferred = builder.Create(SemanticKind.PatternBinding "deferredEffect", Types.unitType, dummyRange)
    let completed = builder.Create(SemanticKind.PatternBinding "independentCompletedEffect", Types.unitType, dummyRange)
    let parent = builder.Create(SemanticKind.Sequential [deferred.Id; completed.Id], Types.unitType, dummyRange)
    let marker = builder.Create(SemanticKind.EagerExpr parent.Id, Types.unitType, dummyRange)
    let raw = builder.Build [] |> Demand.normalize
    let previous = raw.Codata.Value
    let graph = { raw with Codata = lazy { previous with Curry = { previous.Curry with DeferredArgNodes = Set.singleton deferred.Id } } }
    let accumulator = MLIRAccumulator.empty ()
    let position = Zipper.create graph marker.Id |> require "Missing deferred marker"
    let ctx = context graph position accumulator
    let witness ctx node =
        if node.Id = deferred.Id || node.Id = completed.Id then
            let symbol = if node.Id = deferred.Id then "deferred_effect" else "completed_effect"
            { InlineOps = [MLIROp.FuncOp(FuncOp.FuncCall([], symbol, []))]; TopLevelOps = []; Result = TRVoid }
        else
            match node.Kind with
            | SemanticKind.EagerExpr _ -> Alex.Witnesses.EagerWitness.nanopass.Witness ctx node
            | _ -> Alex.Witnesses.StructuralWitness.nanopass.Witness ctx node
    visitAllNodes witness ctx position.Focus ctx.TraversalVisited
    let parentPosition = atChild parent.Id position
    Assert.False(MLIRAccumulator.completedVoid parentPosition ctx.ScopeContext accumulator)
    Assert.False(MLIRAccumulator.completedVoid (atChild deferred.Id parentPosition) ctx.ScopeContext accumulator)
    Assert.True(MLIRAccumulator.completedVoid (atChild completed.Id parentPosition) ctx.ScopeContext accumulator)
    Assert.NotEmpty accumulator.Errors
    Assert.True((MLIRAccumulator.recallNode marker.Id accumulator).IsNone)
    Assert.Single(MLIRAccumulator.getDeferredInlineOps deferred.Id accumulator) |> ignore
    let operations = ScopeContext.getOps ctx.ScopeContext.Value
    Assert.Single operations |> ignore
    Assert.Contains(operations, function MLIROp.FuncOp(FuncOp.FuncCall(_, "completed_effect", _)) -> true | _ -> false)
