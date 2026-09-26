module Alex.Tests.CallableTransportTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures
module Carriers = Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers
module Operands = Alex.Traversal.CallableOperands
module Zipper = Alex.Traversal.PSGZipper

let private ok = function Result.Ok value -> value | Result.Error reason -> failwith reason

/// Settled component inputs use the source-owned carrier projector. These
/// fixtures establish passive physical transport, not source lifetime proof.
let private fixture captured =
    let builder = NodeBuilder()
    let ty = NativeType.TFun(Types.boolType, Types.boolType)
    let envTy = Types.mkArrayType Types.uint8Type
    let formal = builder.Create(SemanticKind.PatternBinding "environment", envTy, dummyRange)
    let argument = builder.Create(SemanticKind.PatternBinding "value", Types.boolType, dummyRange)
    let result = builder.Create(SemanticKind.VarRef("value", Some argument.Id), Types.boolType, dummyRange)
    let parameters = (if captured then ["environment", envTy, formal.Id] else []) @ ["value", Types.boolType, argument.Id]
    let physical = if captured then NativeType.TFun(envTy, ty) else ty
    let implementation = builder.Create(SemanticKind.Lambda(parameters, result.Id, [], None, LambdaContext.RegularClosure), physical, dummyRange)
    let code = builder.Create(SemanticKind.Binding("code", false, false, None), physical, dummyRange, children = [implementation.Id])
    builder.SetParent(implementation.Id, code.Id)
    let capture = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let pending = builder.Create(SemanticKind.PatternBinding "closure", ty, dummyRange)
    let environment = builder.Create(SemanticKind.EnvironmentCreate(pending.Id, [capture.Id, capture.Id]), envTy, dummyRange)
    let owner =
        if captured then builder.CompleteNode(pending.Id, SemanticKind.ClosureValue(implementation.Id, environment.Id), [implementation.Id; environment.Id])
        else code
    let alias = builder.Create(SemanticKind.Binding("alias", false, false, None), ty, dummyRange, children = [owner.Id])
    let reference = builder.Create(SemanticKind.VarRef("alias", Some alias.Id), ty, dummyRange)
    let annotation = builder.Create(SemanticKind.TypeAnnotation(reference.Id, ty), ty, dummyRange, children = [reference.Id])
    let named = builder.Create(SemanticKind.VarRef("code", Some code.Id), physical, dummyRange)
    let sequence = builder.Create(SemanticKind.Sequential [alias.Id; named.Id; annotation.Id], ty, dummyRange, children = [alias.Id; named.Id; annotation.Id])
    builder.SetParent(alias.Id, sequence.Id)
    builder.SetParent(named.Id, sequence.Id)
    builder.SetParent(annotation.Id, sequence.Id)
    builder.SetParent(reference.Id, annotation.Id)
    let ids = [owner.Id; alias.Id; reference.Id; annotation.Id; sequence.Id]
    let slot: ContinuationSlot =
        { Source = capture.Id; ValueType = Types.boolType; Holds = CaptureSlotKind.Scalar SettledSlot.Bool; IsCapture = true
          Field = { Name = "capture"; Slot = SettledSlot.Bool; Offset = Some 0; Size = Some 1; Align = Some 1 } }
    let layout: EnvironmentLayout =
        { Owner = owner.Id; Implementation = implementation.Id; Formal = formal.Id; Slots = [slot]; Bytes = 1; Alignment = 1; Obligations = [] }
    let inputs: Carriers.Inputs =
        { Layouts = if captured then Map.ofList [owner.Id, layout] else Map.empty
          Origins = if captured then (formal.Id :: ids) |> List.map (fun id -> id, owner.Id) |> Map.ofList else Map.empty
          Known = if captured then ids |> List.map (fun id -> id, { Implementation = implementation.Id; EnvironmentOwner = owner.Id }) |> Map.ofList else Map.empty }
    let raw = builder.Build []
    let edges =
        if captured then
            [{ Class = EdgeClass.Provenance; Role = EdgeRole.EnvironmentFormal; Sources = [owner.Id; implementation.Id]; Target = formal.Id; Ordinal = 0 }
             { Class = EdgeClass.Provenance; Role = EdgeRole.EnvironmentCapture false; Sources = [owner.Id; capture.Id; capture.Id]; Target = environment.Id; Ordinal = 0 }]
        else []
    let raw = { raw with Edges = raw.Edges @ edges }
    let carriers, residuals = Carriers.settle inputs raw
    Assert.Empty residuals
    let graph =
        { raw with Codata = lazy { raw.Codata.Value with
                                    CallableCarriers = carriers
                                    EnvironmentLayouts = inputs.Layouts
                                    EnvironmentOrigins = inputs.Origins
                                    KnownCallables = inputs.Known } }
    graph, owner.Id, alias.Id, reference.Id, annotation.Id, sequence.Id, named.Id

let private context graph position accumulator =
    let scope = ref (Alex.Traversal.ScopeContext.ScopeContext.root ())
    let visited = ref Set.empty
    { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
      ScopeContext = scope; RootScopeContext = scope; Graph = graph; Zipper = position
      GlobalVisited = visited; TraversalVisited = visited }

let private bindResult id (output: WitnessOutput) accumulator =
    match output.Result with
    | TRCallable value -> MLIRAccumulator.bindCallable id value accumulator |> ok; value
    | other -> failwithf "Expected separate callable operands: %A" other

[<Fact>]
let ``immutable binding reference annotation and sequence preserve the actual callable operands`` () =
    let graph, owner, alias, reference, annotation, sequence, _ = fixture true
    let operands = MLIRAccumulator.empty ()
    let root = Zipper.create graph sequence |> require "Missing sequence"
    let ownerContext = context graph root operands
    let shape = Operands.project ownerContext owner |> ok
    let code = { SSA = Arg 0; Type = Operands.functionType shape }
    let environment = { SSA = Arg 2; Type = Operands.environmentType shape |> require "No environment" }
    Operands.bind ownerContext owner code (Some environment) |> ok
    let witnesses =
        [alias, atChild alias root, Alex.Witnesses.BindingWitness.nanopass
         reference, atChild annotation root |> atChild reference, Alex.Witnesses.VarRefWitness.nanopass
         annotation, atChild annotation root, Alex.Witnesses.TypeAnnotationWitness.nanopass
         sequence, root, Alex.Witnesses.StructuralWitness.nanopass]
    for id, position, witness in witnesses do
        let ctx = context graph position operands
        let output = witness.Witness ctx graph.Nodes[id]
        let copied = bindResult id output operands
        Assert.Empty output.InlineOps
        Assert.Empty output.TopLevelOps
        Assert.Equal(id, (Operands.carrier copied).Occurrence)
        Assert.Equal(code, Operands.code copied)
        Assert.Equal(Some environment, Operands.environment copied)
        Assert.True((MLIRAccumulator.recallNode id operands).IsNone)
        Assert.Same(position, ctx.Zipper)
        Assert.Empty ctx.TraversalVisited.Value
    Assert.Empty operands.Errors

[<Fact>]
let ``named code value emits one typed function constant without an environment or thunk`` () =
    let graph, _, _, _, _, sequence, named = fixture false
    let root = Zipper.create graph sequence |> require "Missing sequence"
    let operands = MLIRAccumulator.empty ()
    let ctx = context graph (atChild named root) operands
    let output = Alex.Witnesses.VarRefWitness.nanopass.Witness ctx graph.Nodes[named]
    let value = bindResult named output operands
    Assert.Equal(None, Operands.environment value)
    Assert.Single(Operands.values value) |> ignore
    Assert.Empty output.TopLevelOps
    match Assert.Single output.InlineOps with
    | MLIROp.FuncOp(FuncOp.FuncConstant(ssa, symbol, ty)) ->
        Assert.Equal((Operands.code value).SSA, ssa)
        Assert.Equal((Operands.code value).Type, ty)
        Assert.Equal("code", symbol)
    | operation -> failwithf "Named code was not a direct function constant: %A" operation

[<Fact>]
let ``passive callable copy cannot recover an environment from a scalar packed value`` () =
    let graph, owner, alias, _, _, sequence, _ = fixture true
    let root = Zipper.create graph sequence |> require "Missing sequence"
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode owner (Arg 0) (TMemRefStatic(2, TIndex)) operands
    let ctx = context graph (atChild alias root) operands
    let output = Alex.Witnesses.BindingWitness.nanopass.Witness ctx graph.Nodes[alias]
    match output.Result with TRError _ -> () | other -> failwithf "Scalar fallback accepted: %A" other
    Assert.Empty output.InlineOps
    Assert.Empty operands.CallableAssoc

[<Fact>]
let ``transparent callable transport requires the destination occurrence carrier and actual Huet focus`` () =
    let graph, owner, alias, reference, annotation, sequence, _ = fixture true
    let operands = MLIRAccumulator.empty ()
    let root = Zipper.create graph sequence |> require "Missing sequence"
    let ctx = context graph (atChild alias root) operands
    let shape = Operands.project ctx owner |> ok
    Operands.bind ctx owner { SSA = Arg 0; Type = Operands.functionType shape }
        (Some { SSA = Arg 1; Type = Operands.environmentType shape |> require "No environment" }) |> ok
    let otherPosition = atChild annotation root |> atChild reference
    match matchAt (Alex.Patterns.CallablePatterns.pCallableForward ctx owner) otherPosition 64 operands with
    | Result.Error _ -> ()
    | Result.Ok _ -> failwith "A reused node identity must not replace its actual occurrence"
    let changed = { graph with Codata = lazy { graph.Codata.Value with CallableCarriers = graph.Codata.Value.CallableCarriers.Remove alias } }
    let position = Zipper.create changed sequence |> require "Missing sequence" |> atChild alias
    let ctx = context changed position operands
    let output = Alex.Witnesses.BindingWitness.nanopass.Witness ctx changed.Nodes[alias]
    match output.Result with TRError _ -> () | other -> failwithf "Missing destination carrier accepted: %A" other
    Assert.True((MLIRAccumulator.recallCallable alias operands).IsNone)

[<Theory>]
[<InlineData(true, 1)>]
[<InlineData(true, 2)>]
[<InlineData(false, 1)>]
[<InlineData(false, 2)>]
let ``intrinsic annotations are compile-time callees only at an actual application`` applied depth =
    let builder = NodeBuilder()
    let resultType = Types.mkArrayType Types.boolType
    let functionType = NativeType.TFun(Types.intType, resultType)
    let intrinsic = builder.Create(
        SemanticKind.Intrinsic { Module = IntrinsicModule.Array; Operation = "zeroCreate"
                                 Category = IntrinsicCategory.Memory; FullName = "Array.zeroCreate" },
        functionType, dummyRange)
    let annotations =
        [1 .. depth] |> List.scan (fun inner _ ->
            let node = builder.Create(SemanticKind.TypeAnnotation(inner, functionType), functionType, dummyRange, children = [inner])
            builder.SetParent(inner, node.Id)
            node.Id) intrinsic.Id |> List.tail
    let outer = List.last annotations
    let root =
        if applied then
            let argument = builder.Create(SemanticKind.Literal(NativeLiteral.Int(2L, NTUKind.NTUint(NTUWidth.Resolved WidthDimension.Register))), Types.intType, dummyRange)
            builder.Create(SemanticKind.Application(outer, [argument.Id]), resultType, dummyRange, children = [outer; argument.Id])
        else builder.Create(SemanticKind.Binding("value", false, false, None), functionType, dummyRange, children = [outer])
    builder.SetParent(outer, root.Id)
    let graph = builder.Build []
    let operands = MLIRAccumulator.empty ()
    let position = Zipper.create graph root.Id |> require "Missing annotation fixture"
    let mutable focus = position
    for annotation in List.rev annotations do
        focus <- atChild annotation focus
        let ctx = context graph focus operands
        let output = Alex.Witnesses.TypeAnnotationWitness.nanopass.Witness ctx graph.Nodes[annotation]
        match applied, output.Result with
        | true, TRVoid | false, TRError _ -> ()
        | _ -> failwithf "Incorrect annotation role: %A" output.Result
        Assert.Empty output.InlineOps
        Assert.Empty output.TopLevelOps
    Assert.Empty operands.CallableAssoc

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``continuation callable read preserves its loaded environment and requires the exact slot owner`` foreignOwner =
    let original, owner, _, _, _, _, _ = fixture true
    let source = original.Nodes[owner]
    let builder = NodeBuilder()
    let iteratorType = NativeType.TSeqEnumerator Types.boolType
    let frameOwner = builder.Create(SemanticKind.PatternBinding "owner", NativeType.TSeq Types.boolType, dummyRange)
    let frameFormal = builder.Create(SemanticKind.PatternBinding "frame", iteratorType, dummyRange)
    let frameValue = builder.Create(SemanticKind.VarRef("frame", Some frameFormal.Id), iteratorType, dummyRange)
    let read = builder.Create(SemanticKind.FrameRead(frameValue.Id, owner), source.Type, dummyRange)
    let finished = builder.Create(SemanticKind.Literal(NativeLiteral.Bool false), Types.boolType, dummyRange)
    let body = builder.Create(SemanticKind.Sequential [read.Id; finished.Id], Types.boolType, dummyRange)
    let generator = builder.Create(SemanticKind.Lambda(["frame", iteratorType, frameFormal.Id], body.Id, [], None, LambdaContext.SeqGenerator),
                                   NativeType.TFun(iteratorType, Types.boolType), dummyRange)
    let capture = { Name = "callback"; Type = source.Type; IsMutable = false; SourceNodeId = Some owner }
    builder.CompleteNode(frameOwner.Id, SemanticKind.SeqExpr(generator.Id, [capture]), [generator.Id]) |> ignore
    let state = builder.Create(SemanticKind.PatternBinding "state", Types.intType, dummyRange)
    let current = builder.Create(SemanticKind.PatternBinding "current", Types.boolType, dummyRange)
    let slot: ContinuationSlot =
        { Source = owner; ValueType = source.Type; IsCapture = true
          Holds = CaptureSlotKind.EnvironmentView(if foreignOwner then frameValue.Id else owner)
          Field = { Name = "callback"; Slot = SettledSlot.Pointer 5; Offset = Some 0; Size = Some 40; Align = Some 8 } }
    let known = original.Codata.Value.KnownCallables[owner]
    let frame: ContinuationFrame =
        { Owner = frameOwner.Id; Generator = generator.Id; Formal = frameFormal.Id
          State = state.Id; Current = current.Id; Slots = [slot]; Bytes = 40; Alignment = 8
          ScratchSlots = []; ScratchBytes = 0; ScratchAlignment = 1; Initializers = [owner, owner]
          ResumeStates = []; Obligations = [] }
    let inputs: Carriers.Inputs =
        { Layouts = original.Codata.Value.EnvironmentLayouts
          Origins = original.Codata.Value.EnvironmentOrigins.Add(read.Id, owner)
          Known = original.Codata.Value.KnownCallables.Add(read.Id, known) }
    let raw =
        let continuation = builder.Build []
        let access =
            [{ Sources = [owner]; Target = read.Id; Class = EdgeClass.Provenance; Role = EdgeRole.ContinuationValue; Ordinal = 0 }
             { Sources = [frameOwner.Id; generator.Id; frameFormal.Id; owner]; Target = read.Id
               Class = EdgeClass.Provenance; Role = EdgeRole.ContinuationSlotAccess; Ordinal = 0 }]
        { original with Nodes = continuation.Nodes |> Map.fold (fun nodes id node -> Map.add id node nodes) original.Nodes
                        Edges = original.Edges @ continuation.Edges @ access }
    let carriers, residuals = Carriers.settle inputs raw
    Assert.Empty residuals
    let graph =
        { raw with Codata = lazy { raw.Codata.Value with CallableCarriers = carriers; KnownCallables = inputs.Known
                                                         EnvironmentOrigins = inputs.Origins
                                                         SequenceOrigins = Map.ofList [frameValue.Id, frameOwner.Id; frameFormal.Id, frameOwner.Id]
                                                         ContinuationFrames = Map.ofList [frameOwner.Id, frame] } }
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode frameValue.Id (Arg 0) (TMemRefStatic(40, TInt(IntWidth 8))) operands
    let ctx = context graph (Zipper.create graph read.Id |> require "Missing frame read") operands
    let output = Alex.Witnesses.SeqWitness.nanopass.Witness ctx read
    if foreignOwner then
        match output.Result with TRError _ -> () | other -> failwithf "Foreign slot owner was accepted: %A" other
        Assert.Empty output.InlineOps
    else
        let callable = bindResult read.Id output operands
        let environment = Operands.environment callable |> require "Loaded environment was lost"
        Assert.Equal(V(NodeId.value read.Id, 0), environment.SSA)
        Assert.NotEqual(environment.SSA, (Operands.code callable).SSA)
        Assert.Equal(TMemRefStatic(1, TInt(IntWidth 8)), environment.Type)
        Assert.Single(output.InlineOps |> List.filter (function MLIROp.FuncOp(FuncOp.FuncConstant _) -> true | _ -> false)) |> ignore
        Assert.True((MLIRAccumulator.recallNode read.Id operands).IsNone)
