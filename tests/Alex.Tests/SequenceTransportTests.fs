module Alex.Tests.SequenceTransportTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures
open Alex.Patterns.SequencePatterns
module Operands = Alex.Traversal.SequenceOperands
module Zipper = Alex.Traversal.PSGZipper

let private ok = function Result.Ok value -> value | Result.Error reason -> failwith reason

/// Supplied component contracts, not a source lifetime/admission proof. Two
/// different generators have the same complete empty-enumerator protocol.
let private fixture () =
    let builder = NodeBuilder()
    let sequenceType = NativeType.TSeq Types.boolType
    let iteratorType = NativeType.TSeqEnumerator Types.boolType
    let stateField = { Name = "state"; Slot = SettledSlot.Integer(8, None); Offset = Some 0; Size = Some 1; Align = Some 1 }
    let memberFrame name =
        let state = builder.Create(SemanticKind.PatternBinding (name + "State"), Types.intType, dummyRange)
        let current = builder.Create(SemanticKind.PatternBinding (name + "Current"), Types.boolType, dummyRange)
        let formal = builder.Create(SemanticKind.PatternBinding (name + "Frame"), iteratorType, dummyRange)
        let body = builder.Create(SemanticKind.Literal(NativeLiteral.Bool false), Types.boolType, dummyRange)
        let signature = NativeType.TFun(iteratorType, Types.boolType)
        let generator = builder.Create(SemanticKind.Lambda([name + "Frame", iteratorType, formal.Id], body.Id, [], None, LambdaContext.SeqGenerator),
                                       signature, dummyRange, children = [formal.Id; body.Id])
        let owner = builder.Create(SemanticKind.SeqExpr(generator.Id, []), sequenceType, dummyRange, children = [generator.Id])
        builder.SetParent(generator.Id, owner.Id)
        builder.SetParent(formal.Id, generator.Id)
        builder.SetParent(body.Id, generator.Id)
        let slot: ContinuationSlot =
            { Source = state.Id; ValueType = Types.intType; IsCapture = false
              Holds = CaptureSlotKind.Scalar stateField.Slot; Field = stateField }
        let frame: ContinuationFrame =
            { Owner = owner.Id; Generator = generator.Id; Formal = formal.Id; State = state.Id; Current = current.Id
              Slots = [slot]; Bytes = 1; Alignment = 1; ScratchSlots = []; ScratchBytes = 0; ScratchAlignment = 1
              Initializers = []; ResumeStates = [0]; Obligations = [] }
        let contract: SequenceFamilyMember =
            { Generator = generator.Id; Formal = formal.Id; Signature = signature; State = state.Id; Current = None
              Slots = [slot]; Captures = Set.empty; Uninitialized = Set.empty; Obligations = [] }
        owner.Id, frame, contract
    let first, firstFrame, firstMember = memberFrame "first"
    let second, secondFrame, secondMember = memberFrame "second"
    let alias = builder.Create(SemanticKind.Binding("alias", false, false, None), sequenceType, dummyRange, children = [first])
    let reference = builder.Create(SemanticKind.VarRef("alias", Some alias.Id), sequenceType, dummyRange)
    let annotation = builder.Create(SemanticKind.TypeAnnotation(reference.Id, sequenceType), sequenceType, dummyRange, children = [reference.Id])
    let block = builder.Create(SemanticKind.Sequential [alias.Id; annotation.Id], sequenceType, dummyRange, children = [alias.Id; annotation.Id])
    builder.SetParent(alias.Id, block.Id)
    builder.SetParent(annotation.Id, block.Id)
    builder.SetParent(reference.Id, annotation.Id)
    let condition = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let left = builder.Create(SemanticKind.VarRef("first", Some first), sequenceType, dummyRange)
    let right = builder.Create(SemanticKind.VarRef("second", Some second), sequenceType, dummyRange)
    let choice = builder.Create(SemanticKind.IfThenElse(condition.Id, left.Id, Some right.Id), sequenceType, dummyRange,
                                children = [condition.Id; left.Id; right.Id])
    for id in [condition.Id; left.Id; right.Id] do builder.SetParent(id, choice.Id)
    let flows =
        [first, Set.singleton first; second, Set.singleton second
         alias.Id, Set.singleton first; reference.Id, Set.singleton first
         annotation.Id, Set.singleton first; block.Id, Set.singleton first
         left.Id, Set.singleton first; right.Id, Set.singleton second
         choice.Id, Set.ofList [first; second]]
        |> List.map (fun (id, owners) ->
            id, { Occurrence = id; ElementType = Types.boolType; IsEnumerator = false
                  Owners = owners; Unknown = Set.empty })
        |> Map.ofList
    let family: SequenceFamily =
        { Identity = first; ElementType = Types.boolType; Participants = flows.Keys |> Set.ofSeq
          Members = Map.ofList [first, firstMember; second, secondMember]
          Bytes = 1; Alignment = 1; StateField = stateField; CurrentField = None; CurrentRepresentation = None }
    let raw = builder.Build []
    let layoutEdges =
        family.Members |> Map.toList |> List.map (fun (owner, memberContract) ->
            { Sources = List.distinct (List.ofSeq family.Members.Keys @
                            [memberContract.Generator; memberContract.Formal; memberContract.State] @
                            (memberContract.Slots |> List.map _.Source) @ memberContract.Obligations)
              Target = owner; Class = EdgeClass.Suspension; Role = EdgeRole.SequenceFamilyLayout; Ordinal = 0 })
    let raw = { raw with Edges = raw.Edges @ layoutEdges }
    let graph =
        { raw with Codata = lazy { raw.Codata.Value with
                                    SequenceFlows = flows; SequenceFamilies = Map.ofList [first, family]
                                    ContinuationFrames = Map.ofList [first, firstFrame; second, secondFrame] } }
    graph, first, second, alias.Id, reference.Id, annotation.Id, block.Id, condition.Id, left.Id, right.Id, choice.Id

let private context graph position accumulator =
    let scope = ref (Alex.Traversal.ScopeContext.ScopeContext.root ())
    let visited = ref Set.empty
    { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
      ScopeContext = scope; RootScopeContext = scope; Graph = graph; Zipper = position
      GlobalVisited = visited; TraversalVisited = visited }

let private seed ctx occurrence codeIndex environmentIndex =
    let shape = Operands.project ctx occurrence |> ok
    let code = { SSA = Arg codeIndex; Type = Operands.functionType shape }
    let environment = { SSA = Arg environmentIndex; Type = Operands.environmentType shape }
    Operands.bind ctx occurrence code environment |> ok
    [code; environment]

[<Fact>]
let ``passive sequence binding read annotation and block retain the actual pair`` () =
    let graph, first, _, alias, reference, annotation, block, _, _, _, _ = fixture ()
    let operands = MLIRAccumulator.empty ()
    let root = Zipper.create graph block |> require "Missing block"
    let original = seed (context graph root operands) first 0 1
    let witnesses =
        [alias, atChild alias root, Alex.Witnesses.BindingWitness.nanopass
         reference, atChild annotation root |> atChild reference, Alex.Witnesses.VarRefWitness.nanopass
         annotation, atChild annotation root, Alex.Witnesses.TypeAnnotationWitness.nanopass
         block, root, Alex.Witnesses.StructuralWitness.nanopass]
    for id, position, witness in witnesses do
        let ctx = context graph position operands
        let output = witness.Witness ctx graph.Nodes[id]
        match output.Result with
        | TRSequence value ->
            Assert.Equal<Val list>(original, Operands.values value)
            MLIRAccumulator.bindSequence id value operands |> ok
        | other -> failwithf "Lost sequence pair: %A" other
        Assert.Empty output.InlineOps
        Assert.Empty output.TopLevelOps
        Assert.True((MLIRAccumulator.recallNode id operands).IsNone)
        Assert.Same(position, ctx.Zipper)
        Assert.Empty ctx.TraversalVisited.Value

[<Fact>]
let ``sequence forward rejects scalar fallback missing flow and wrong occurrence`` () =
    let graph, first, _, alias, reference, annotation, block, _, _, _, _ = fixture ()
    let operands = MLIRAccumulator.empty ()
    let root = Zipper.create graph block |> require "Missing block"
    let position = atChild alias root
    let ctx = context graph position operands
    MLIRAccumulator.bindNode first (Arg 0) (TMemRefStatic(1, TInt(IntWidth 8))) operands
    match matchAt (pSequenceForward ctx first) position 64 operands with
    | Result.Error _ -> ()
    | Result.Ok _ -> failwith "A raw environment silently substituted for a sequence pair"
    seed ctx first 1 2 |> ignore
    let different = atChild annotation root |> atChild reference
    match matchAt (pSequenceForward ctx first) different 64 operands with
    | Result.Error _ -> ()
    | Result.Ok _ -> failwith "A foreign Huet occurrence was accepted"
    let changed = { graph with Codata = lazy { graph.Codata.Value with SequenceFlows = graph.Codata.Value.SequenceFlows.Remove alias } }
    let position = Zipper.create changed block |> require "Missing block" |> atChild alias
    let ctx = context changed position operands
    match matchAt (pSequenceForward ctx first) position 64 operands with
    | Result.Error _ -> ()
    | Result.Ok _ -> failwith "Missing destination flow was reconstructed"

[<Fact>]
let ``conditional returns both components of the same selected alternative`` () =
    let graph, _, _, _, _, _, _, _, left, right, choice = fixture ()
    let operands = MLIRAccumulator.empty ()
    let position = Zipper.create graph choice |> require "Missing conditional"
    let ctx = context graph position operands
    let leftValues = seed ctx left 0 1
    let rightValues = seed ctx right 2 3
    let condition = { SSA = Arg 4; Type = TInt(IntWidth 1) }
    match matchAt (pSequenceConditional ctx condition left [] right []) position 64 operands with
    | Result.Ok (([MLIROp.IndexOp(IndexOp.IndexCastU(selector, test, _, TIndex));
                    MLIROp.SCFOp(SCFOp.IndexSwitch(actual, cases, fallback, results))], TRSequence value), _) ->
        Assert.Equal(condition.SSA, test)
        Assert.Equal(selector, actual)
        let label, branch = Assert.Single cases
        Assert.Equal(1L, label)
        let yielded = function
            | [MLIROp.SCFOp(SCFOp.Yield values)] -> values
            | operations -> failwithf "Wrong branch operations: %A" operations
        Assert.Equal<(SSA * MLIRType) list>(leftValues |> List.map (fun value -> value.SSA, value.Type), yielded branch)
        Assert.Equal<(SSA * MLIRType) list>(rightValues |> List.map (fun value -> value.SSA, value.Type), yielded fallback)
        Assert.Equal<(SSA * MLIRType) list>(Operands.values value |> List.map (fun value -> value.SSA, value.Type), results)
    | other -> failwithf "Conditional did not transport a selected pair: %A" other

[<Fact>]
let ``join cannot discard a possible source owner`` () =
    let graph, _, _, _, _, _, _, _, left, right, choice = fixture ()
    let narrowed = graph.Codata.Value.SequenceFlows[left]
    let changed =
        { graph with Codata = lazy { graph.Codata.Value with
                                      SequenceFlows = graph.Codata.Value.SequenceFlows.Add(choice, { narrowed with Occurrence = choice }) } }
    let operands = MLIRAccumulator.empty ()
    let position = Zipper.create changed choice |> require "Missing conditional"
    let ctx = context changed position operands
    seed ctx left 0 1 |> ignore
    seed ctx right 2 3 |> ignore
    match matchAt (pSequenceConditional ctx { SSA = Arg 4; Type = TInt(IntWidth 1) } left [] right []) position 64 operands with
    | Result.Error _ -> ()
    | Result.Ok _ -> failwith "Join selected one possible origin and silently discarded another"

[<Fact>]
let ``a generator raw frame formal cannot acquire a sequence pair from a flow row`` () =
    let graph, first, _, _, _, _, _, _, _, _, _ = fixture ()
    let codata = graph.Codata.Value
    let formal = codata.ContinuationFrames[first].Formal
    let family = codata.SequenceFamilies[first]
    let forgedFlow = { codata.SequenceFlows[first] with Occurrence = formal; IsEnumerator = true }
    let changed =
        { graph with Codata = lazy { codata with
                                      SequenceFlows = codata.SequenceFlows.Add(formal, forgedFlow)
                                      SequenceFamilies = codata.SequenceFamilies.Add(first, { family with Participants = family.Participants.Add formal }) } }
    let position = Zipper.create changed formal |> require "Missing generator formal"
    match Operands.project (context changed position (MLIRAccumulator.empty ())) formal with
    | Result.Error _ -> ()
    | Result.Ok _ -> failwith "An actual storage formal was expanded into a source sequence protocol"

[<Theory>]
[<InlineData("extent")>]
[<InlineData("holds")>]
[<InlineData("value-type")>]
[<InlineData("authority")>]
let ``sequence projection rejects changed actual frame under an unchanged family`` defect =
    let graph, first, second, _, _, _, _, _, _, _, choice = fixture ()
    let frame = graph.Codata.Value.ContinuationFrames[second]
    let changedFrame =
        match defect with
        | "holds" -> { frame with Slots = frame.Slots |> List.map (fun slot -> { slot with Holds = CaptureSlotKind.CellView Types.intType }) }
        | "value-type" -> { frame with Slots = frame.Slots |> List.map (fun slot -> { slot with ValueType = Types.boolType }) }
        | _ ->
            { frame with Bytes = 2
                         Slots = frame.Slots |> List.map (fun slot -> { slot with Field = { slot.Field with Offset = Some 1 } }) }
    let changed =
        { graph with Codata = lazy { graph.Codata.Value with
                                      ContinuationFrames = graph.Codata.Value.ContinuationFrames.Add(second, changedFrame) } }
    let changed =
        if defect = "authority" then
            { graph with Edges = graph.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.SequenceFamilyLayout || edge.Target <> second) }
        else changed
    let position = Zipper.create changed choice |> require "Missing conditional"
    let ctx = context changed position (MLIRAccumulator.empty ())
    // The first alternative itself is unchanged; the entire advertised family
    // must still agree with every actual member before any pair is admitted.
    match Operands.project ctx first with
    | Result.Error _ -> ()
    | Result.Ok _ -> failwith "A stale family concealed a changed alternative's frame layout"

[<Fact>]
let ``sequence projection rejects changed actual generator formal and signature`` () =
    let graph, first, second, _, _, _, _, _, _, _, choice = fixture ()
    let frame = graph.Codata.Value.ContinuationFrames[second]
    let formal = graph.Nodes[frame.Formal]
    let generator = graph.Nodes[frame.Generator]
    let wrongFormalType = NativeType.TSeqEnumerator Types.intType
    let changedGenerator =
        match generator.Kind with
        | SemanticKind.Lambda([name, _, parameter], body, captures, recursive, context) ->
            { generator with Type = NativeType.TFun(wrongFormalType, Types.boolType)
                             Kind = SemanticKind.Lambda([name, wrongFormalType, parameter], body, captures, recursive, context) }
        | _ -> failwith "Fixture lost its generator"
    let changed =
        { graph with Nodes = graph.Nodes.Add(formal.Id, { formal with Type = wrongFormalType }).Add(generator.Id, changedGenerator) }
    let position = Zipper.create changed choice |> require "Missing conditional"
    let ctx = context changed position (MLIRAccumulator.empty ())
    match Operands.project ctx first with
    | Result.Error _ -> ()
    | Result.Ok _ -> failwith "A stale family concealed a changed alternative's generator signature"

[<Fact>]
let ``operation scope restore keeps sequence pairs and their physical types together`` () =
    let graph, first, _, alias, _, _, block, _, _, _, _ = fixture ()
    let position = Zipper.create graph block |> require "Missing block"
    let operands = MLIRAccumulator.empty ()
    let ctx = context graph position operands
    let original = seed ctx first 0 1
    let outer = MLIRAccumulator.snapshotOperands operands
    let shape = Operands.project ctx first |> ok
    let code = { SSA = V(900, 0); Type = Operands.functionType shape }
    let environment = { SSA = V(900, 1); Type = Operands.environmentType shape }
    Operands.bind ctx first code environment |> ok
    Operands.copy ctx first alias |> ok
    MLIRAccumulator.restoreOperands outer operands
    Assert.Same(outer.Sequences, operands.SequenceAssoc)
    Assert.Same(outer.Scalars, operands.NodeAssoc)
    Assert.Same(outer.Types, operands.SSATypes)
    Assert.True((MLIRAccumulator.recallSequence alias operands).IsNone)
    Assert.True((MLIRAccumulator.recallSSAType code.SSA operands).IsNone)
    Assert.True((MLIRAccumulator.recallSSAType environment.SSA operands).IsNone)
    let restored = MLIRAccumulator.recallSequence first operands |> require "Sequence pair was lost"
    Assert.Equal<Val list>(original, Operands.values restored)
