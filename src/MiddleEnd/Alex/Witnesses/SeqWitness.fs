/// SeqWitness - Observe sequence operations at their graph focus.
///
/// Unelaborated suspension nodes require upstream Baker settlement; the
/// witness does not reconstruct a frame from body shape or mutable bindings.
///
/// NANOPASS: This witness handles ONLY Seq-related nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.SeqWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ContinuationPatterns
open Alex.Patterns.LiteralPatterns
open Alex.Patterns.CallablePatterns
open XParsec
open XParsec.Parsers
open XParsec.Combinators
module Operands = Alex.Traversal.CallableOperands
module Sequences = Alex.Traversal.SequenceOperands

let private failure (node: SemanticNode) phase message =
    WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "Continuation") (Some phase) message

let private observe (ctx: WitnessContext) (node: SemanticNode) pattern =
    match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Result.Ok ((ops, result), _) ->
        { InlineOps = ops; TopLevelOps = MLIRAccumulator.drainPendingStaticGlobals ctx.Accumulator; Result = result }
    | Result.Error message -> failure node "settled frame operands" message

/// Exact graph rows identify persistent and transient storage independently.
/// Slot references are never searched across unrelated continuation frames.
let private frameAt (ctx: WitnessContext) frameId =
    let codata = ctx.Graph.Codata.Value
    let owner =
        match codata.ContinuationStorage |> Map.tryFind frameId with
        | Some owner -> Some(owner, true)
        | None -> codata.SequenceOrigins |> Map.tryFind frameId |> Option.map (fun owner -> owner, false)
    owner |> Option.bind (fun (owner, scratch) -> codata.ContinuationFrames |> Map.tryFind owner |> Option.map (fun frame -> frame, scratch))

let private accessSlot (ctx: WitnessContext) (node: SemanticNode) frameId slotId borrow write =
    match frameAt ctx frameId with
    | None -> failure node "frame identity" $"Frame operand {NodeId.value frameId} has no settled continuation origin"
    | Some(frame, scratch) ->
        let slots, bytes = if scratch then frame.ScratchSlots, frame.ScratchBytes else frame.Slots, frame.Bytes
        match slots |> List.tryFind (fun slot -> slot.Source = slotId) with
        | None -> failure node "slot identity" $"Continuation {NodeId.value frame.Owner} has no slot {NodeId.value slotId} in the selected storage"
        | Some slot ->
            let pattern =
                match write with
                | Some value -> pWithUnitResult node.Id (pWriteContinuationSlot node.Id frameId value bytes slot)
                | None when borrow -> pBorrowContinuationSlot node.Id frameId bytes slot
                | None -> pReadContinuationSlot node.Id frameId bytes slot
            match Clef.Compiler.NativeTypedTree.UnionFind.applySubst node.Type, write, borrow with
            | (NativeType.TSeq _ | NativeType.TSeqEnumerator _), None, false
                when Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.valueShape ctx.Graph node = CallableValueShape.Sequence node.Id ->
                match ctx.Graph.Codata.Value.SequenceOrigins.TryFind node.Id, Sequences.project ctx node.Id with
                | Some owner, Result.Ok shape when (Sequences.flow shape).Owners = Set.singleton owner ->
                    let family = Sequences.family shape
                    let generator = ctx.Graph.Nodes[family.Members[owner].Generator]
                    let symbol = Alex.CodeGeneration.CallableSymbols.lambda ctx.Graph generator false
                    let sequence = parser {
                        let! operations, result = pattern
                        match result with
                        | TRValue environment ->
                            let! code, value = pSequenceValue node.Id shape symbol environment
                            return operations @ code, value
                        | _ -> return! fail (Message "Sequence frame read requires its actual descriptor")
                    }
                    observe ctx node sequence
                | _, Result.Error reason -> failure node "sequence frame carrier" reason
                | _ -> failure node "sequence slot identity" "Descriptor-only sequence capture lacks its exact source-proved function half"
            | NativeType.TFun _, None, false ->
                match slot.Holds, ctx.Graph.Codata.Value.CallableCarriers.TryFind node.Id with
                | CaptureSlotKind.EnvironmentView owner, Some { Environment = Some expected } when expected.Owner = owner ->
                    match Operands.project ctx node.Id with
                    | Result.Error reason -> failure node "callable frame carrier" reason
                    | Result.Ok shape ->
                        let carrier = ctx.Graph.Codata.Value.CallableCarriers[node.Id]
                        let implementation = ctx.Graph.Nodes[carrier.Implementation]
                        let symbol = Alex.CodeGeneration.CallableSymbols.lambda ctx.Graph implementation false
                        let callable = parser {
                            let! operations, result = pattern
                            match result with
                            | TRValue environment ->
                                let! code, value = pCallableValue node.Id shape symbol (Some environment)
                                return operations @ code, value
                            | _ -> return! fail (Message "Callable frame read requires its actual loaded environment descriptor")
                        }
                        observe ctx node callable
                | _ -> failure node "callable slot identity" "Callable frame read lacks its exact settled environment-view slot and occurrence carrier"
            | _ -> observe ctx node pattern

let private witnessIntrinsic (ctx: WitnessContext) (node: SemanticNode) =
    let matcher = pIntrinsicApplication IntrinsicModule.Seq <|> pIntrinsicApplication IntrinsicModule.SeqEnumerator
    match tryMatch matcher ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some ((info, [argument]), _) ->
        match MLIRAccumulator.recallSequence argument ctx.Accumulator with
        | Some sequence ->
            match info.Module, info.Operation with
            | IntrinsicModule.Seq, "getEnumerator" ->
                match Sequences.project ctx node.Id, Sequences.copyContract ctx node.Id with
                | Result.Ok shape, Result.Ok copy -> observe ctx node (pAcquireSequence node.Id sequence shape copy)
                | Result.Error reason, _ | _, Result.Error reason -> failure node "template copy" reason
            | IntrinsicModule.SeqEnumerator, "moveNext" ->
                observe ctx node (pPullSequence node.Id sequence)
            | IntrinsicModule.SeqEnumerator, "current" ->
                observe ctx node (pSequenceCurrent node.Id sequence)
            | _ -> failure node "intrinsic settlement" $"{info.FullName} requires Baker elaboration before continuation witnessing"
        | None -> failure node "sequence operands" $"{info.FullName} operand {NodeId.value argument} has no witnessed pull function and actual environment"
    | Some ((info, arguments), _) ->
        failure node "intrinsic arity" $"{info.FullName} requires one settled frame operand, received {arguments.Length}"

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Seq operations - category-selective (handles only Seq nodes)
let private witnessSeq (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.FrameRead(frameId, slotId) -> accessSlot ctx node frameId slotId false None
    | SemanticKind.FrameBorrow(frameId, slotId) -> accessSlot ctx node frameId slotId true None
    | SemanticKind.FrameWrite(frameId, slotId, value) -> accessSlot ctx node frameId slotId false (Some value)
    | SemanticKind.ContinuationAllocate owner ->
        match ctx.Graph.Codata.Value.ContinuationFrames |> Map.tryFind owner with
        | Some frame -> observe ctx node (pAllocateContinuationFrame node.Id frame)
        | None -> failure node "allocation layout" $"Continuation {NodeId.value owner} has no settled frame allocation"
    | SemanticKind.ContinuationStorage owner ->
        match ctx.Graph.Codata.Value.ContinuationFrames |> Map.tryFind owner with
        | Some frame -> observe ctx node (pAllocateContinuationStorage node.Id frame.ScratchBytes frame.ScratchAlignment)
        | None -> failure node "storage layout" $"Continuation {NodeId.value owner} has no settled activation storage"
    | SemanticKind.SeqExpr _ ->
        let codata = ctx.Graph.Codata.Value
        let owner = codata.SequenceOrigins |> Map.tryFind node.Id |> Option.defaultValue node.Id
        match codata.ContinuationFrames |> Map.tryFind owner with
        | Some frame ->
            match codata.SequenceInitializers |> Map.tryFind node.Id with
            | Some initializers ->
                match Sequences.project ctx node.Id, ctx.Graph.Nodes.TryFind frame.Generator with
                | Result.Ok shape, Some ({ Kind = SemanticKind.Lambda _ } as generator) ->
                    let symbol = Alex.CodeGeneration.CallableSymbols.lambda ctx.Graph generator false
                    let formation = parser {
                        let! operations, result = pConstructSequence node.Id frame initializers
                        match result with
                        | TRValue environment ->
                            let! code, sequence = pSequenceValue node.Id shape symbol environment
                            return operations @ code, sequence
                        | _ -> return! fail (Message "Sequence formation did not produce its actual environment")
                    }
                    observe ctx node formation
                | Result.Error reason, _ -> failure node "sequence carrier" reason
                | _ -> failure node "generator identity" "Sequence formation lacks its exact generated implementation"
            | None -> failure node "capture initialization" $"Sequence constructor {NodeId.value node.Id} has no settled capture initializers"
        | None -> WitnessOutput.error "SeqExpr requires Baker-settled suspension segments, frame and resumption; delimiter ownership alone is insufficient"
    | SemanticKind.Yield _ ->
        WitnessOutput.error "Yield requires Baker-settled suspension segments, frame and resumption; delimiter ownership alone is insufficient"
    | SemanticKind.YieldBang _ ->
        WitnessOutput.error "YieldBang requires Baker-settled suspension segments, frame and resumption; delimiter ownership alone is insufficient"
    | SemanticKind.ForEach _ ->
        failure node "consumer settlement" "ForEach requires Baker's explicit enumeration loop before continuation witnessing"
    | _ -> witnessIntrinsic ctx node

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Seq nanopass - observes settled frame operations and rejects remaining source suspensions.
let nanopass : Nanopass = {
    Name = "Seq"
    Witness = witnessSeq
}
