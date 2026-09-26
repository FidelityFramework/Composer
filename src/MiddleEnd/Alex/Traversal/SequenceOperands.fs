/// Canonical transport for a Baker-settled sequence protocol. This reads exact
/// source participants and placed fields. It never chooses an origin, discovers
/// a frame from a type, emits a generator body or packs a function into data.
module Alex.Traversal.SequenceOperands

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.NativeTypedTree.UnionFind
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes

type Shape = private {
    Flow: SequenceFlow
    Family: SequenceFamily
    FunctionType: MLIRType
    EnvironmentType: MLIRType
}

let private sourceElement (node: SemanticNode) =
    match applySubst node.Type with
    | NativeType.TSeq item -> Some(item, false)
    | NativeType.TSeqEnumerator item -> Some(item, true)
    | _ -> None

let project (ctx: WitnessContext) occurrence : Result<Shape, string> =
    let codata = ctx.Graph.Codata.Value
    match ctx.Graph.Nodes.TryFind occurrence, codata.SequenceFlows.TryFind occurrence with
    | Some node, Some flow when flow.Occurrence = occurrence && flow.Unknown.IsEmpty && not flow.Owners.IsEmpty &&
                               Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.valueShape ctx.Graph node = CallableValueShape.Sequence occurrence ->
        match sourceElement node with
        | Some(item, iterator) when applySubst item = applySubst flow.ElementType && iterator = flow.IsEnumerator ->
            let families = codata.SequenceFamilies.Values |> Seq.filter (fun family -> family.Participants.Contains occurrence) |> Seq.toList
            match families with
            | [family] when family.Bytes > 0 && family.Alignment > 0 &&
                            applySubst family.ElementType = applySubst item &&
                            (flow.Owners |> Set.forall family.Members.ContainsKey) ->
                let validMember owner (memberContract: SequenceFamilyMember) =
                    let dependencies =
                        List.distinct (List.ofSeq family.Members.Keys @
                            [memberContract.Generator; memberContract.Formal; memberContract.State] @
                            (memberContract.Slots |> List.map _.Source) @ memberContract.Obligations)
                    let resident = ctx.Graph.Edges |> List.exists (fun edge ->
                        edge.Class = EdgeClass.Suspension && edge.Role = EdgeRole.SequenceFamilyLayout &&
                        edge.Target = owner && edge.Ordinal = 0 && edge.Sources = dependencies)
                    match codata.ContinuationFrames.TryFind owner, ctx.Graph.Nodes.TryFind owner,
                          ctx.Graph.Nodes.TryFind memberContract.Generator, ctx.Graph.Nodes.TryFind memberContract.Formal with
                    | Some frame, Some { Kind = SemanticKind.SeqExpr(generator, _); Type = NativeType.TSeq element },
                      Some { Kind = SemanticKind.Lambda([_, parameterType, formal], body, _, _, LambdaContext.SeqGenerator); Type = signature },
                      Some { Kind = SemanticKind.PatternBinding _; Type = formalType } ->
                        resident && frame.Owner = owner && generator = frame.Generator && generator = memberContract.Generator &&
                        formal = frame.Formal && formal = memberContract.Formal &&
                        applySubst element = applySubst item && applySubst formalType = NativeType.TSeqEnumerator(applySubst item) &&
                        applySubst parameterType = applySubst formalType && applySubst signature = applySubst memberContract.Signature &&
                        applySubst signature = NativeType.TFun(applySubst formalType, Types.boolType) &&
                        (ctx.Graph.Nodes.TryFind body |> Option.exists (fun node -> applySubst node.Type = Types.boolType)) &&
                        frame.Bytes = family.Bytes && frame.Alignment = family.Alignment && frame.State = memberContract.State &&
                        frame.Slots = memberContract.Slots &&
                        (frame.Slots |> List.tryFind (fun slot -> slot.Source = frame.State) |> Option.exists (fun slot ->
                            slot.Field.Slot = family.StateField.Slot && slot.Field.Offset = family.StateField.Offset &&
                            slot.Field.Size = family.StateField.Size && slot.Field.Align = family.StateField.Align)) &&
                        (frame.Slots |> List.filter _.IsCapture |> List.map _.Source |> Set.ofList) = memberContract.Captures &&
                        (frame.Slots |> List.filter (fun slot -> not slot.IsCapture && slot.Source <> frame.State) |> List.map _.Source |> Set.ofList) = memberContract.Uninitialized &&
                        frame.Obligations = memberContract.Obligations &&
                        (frame.Slots |> List.tryFind (fun slot -> slot.Source = frame.Current) |> Option.map _.Source) = memberContract.Current &&
                        (memberContract.Current |> Option.forall (fun current ->
                            match frame.Slots |> List.tryFind (fun slot -> slot.Source = current), family.CurrentField, family.CurrentRepresentation with
                            | Some slot, Some field, Some(valueType, holds) ->
                                slot.Field.Slot = field.Slot && slot.Field.Offset = field.Offset &&
                                slot.Field.Size = field.Size && slot.Field.Align = field.Align &&
                                slot.Holds = holds && applySubst slot.ValueType = applySubst valueType
                            | _ -> false))
                    | _ -> false
                let hasCurrent = family.Members.Values |> Seq.exists (fun memberContract -> memberContract.Current.IsSome)
                if (family.Members |> Map.forall validMember) &&
                   hasCurrent = family.CurrentField.IsSome && hasCurrent = family.CurrentRepresentation.IsSome &&
                   (family.CurrentRepresentation |> Option.forall (fun (ty, _) -> applySubst ty = applySubst item)) then
                    let environment = TMemRefStatic(family.Bytes, TInt(IntWidth 8))
                    Result.Ok { Flow = flow; Family = family; EnvironmentType = environment
                                FunctionType = TFunc([environment], [TInt(IntWidth 1)]) }
                else Result.Error "Sequence family no longer matches its actual generators, formals, fields and layout obligations."
            | _ -> Result.Error "Sequence occurrence has no unique complete source protocol."
        | _ -> Result.Error "Sequence flow does not match this source occurrence's element type and value role."
    | _ -> Result.Error "Sequence occurrence has incomplete source alternatives."

let functionType shape = shape.FunctionType
let environmentType shape = shape.EnvironmentType
let family shape = shape.Family
let flow shape = shape.Flow
let componentTypes shape = [shape.FunctionType; shape.EnvironmentType]

let private copyReadings = System.Runtime.CompilerServices.ConditionalWeakTable<SemanticGraph, Map<NodeId, SequenceTemplateCopy>>()

/// Recheck the exact source copy participants without emitting or granting a
/// new proof. Missing or changed representation/lifetime facts stay a refusal.
let copyContract (ctx: WitnessContext) acquisition =
    let codata = ctx.Graph.Codata.Value
    match codata.SequenceTemplateCopies.TryFind acquisition with
    | None -> Result.Error "Fresh sequence acquisition has no source representation-copy contract."
    | Some expected ->
        let current = copyReadings.GetValue(ctx.Graph, fun graph ->
            let codata = graph.Codata.Value
            let current, evidence, _ =
                Clef.Compiler.Nanopass.SequenceFamilies.copies graph codata.SequenceFamilies codata.SequenceFlows
                    codata.Escapes codata.ContinuationRegions codata.SequenceInitializers codata.SequenceDestinations
            current |> Map.filter (fun acquisition _ ->
                evidence.NewEdges |> List.exists (fun edge ->
                    edge.Target = acquisition && (graph.Edges |> List.exists (fun resident ->
                        resident.Class = edge.Class && resident.Role = edge.Role && resident.Sources = edge.Sources &&
                        resident.Target = edge.Target && resident.Ordinal = edge.Ordinal)))))
        match current.TryFind acquisition with
        | Some actual when actual = expected -> Result.Ok expected
        | _ -> Result.Error "Sequence representation-copy contract no longer matches its exact storage and initializer participants."

let create (shape: Shape) (code: Val) (environment: Val) : Result<SequenceOperand, string> =
    if code.Type <> shape.FunctionType || environment.Type <> shape.EnvironmentType then
        Result.Error "Sequence code and environment do not match their settled family signature."
    elif code.SSA = environment.SSA then Result.Error "Sequence code and environment must be separate values."
    else Result.Ok { Flow = shape.Flow; Family = shape.Family; Code = code; Environment = environment }

let bind (ctx: WitnessContext) occurrence code environment =
    project ctx occurrence
    |> Result.bind (fun shape -> create shape code environment)
    |> Result.bind (fun value -> MLIRAccumulator.bindSequence occurrence value ctx.Accumulator)

let reproject (ctx: WitnessContext) source destination =
    match MLIRAccumulator.recallSequence source ctx.Accumulator with
    | None -> Result.Error "Source sequence has not been witnessed in this operation scope."
    | Some value when value.Flow.Occurrence <> source -> Result.Error "Sequence was recalled under another occurrence's identity."
    | Some value ->
        project ctx destination |> Result.bind (fun shape ->
            if value.Family.Identity <> shape.Family.Identity || value.Flow.IsEnumerator <> shape.Flow.IsEnumerator ||
               applySubst value.Flow.ElementType <> applySubst shape.Flow.ElementType ||
               not (Set.isSubset value.Flow.Owners shape.Flow.Owners) then
                Result.Error "Sequence copy does not preserve its protocol, alternatives and actual value role."
            else create shape value.Code value.Environment)

let copy (ctx: WitnessContext) source destination =
    reproject ctx source destination
    |> Result.bind (fun value -> MLIRAccumulator.bindSequence destination value ctx.Accumulator)

let values (value: SequenceOperand) = [value.Code; value.Environment]
let code (value: SequenceOperand) = value.Code
let environment (value: SequenceOperand) = value.Environment
