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
open XParsec
open XParsec.Parsers
open XParsec.Combinators

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
            observe ctx node pattern

let private witnessIntrinsic (ctx: WitnessContext) (node: SemanticNode) =
    let matcher = pIntrinsicApplication IntrinsicModule.Seq <|> pIntrinsicApplication IntrinsicModule.SeqEnumerator
    match tryMatch matcher ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some ((info, [argument]), _) ->
        match frameAt ctx argument with
        | Some (frame, false) ->
            match info.Module, info.Operation with
            | IntrinsicModule.Seq, "getEnumerator" -> observe ctx node (pGetEnumerator node.Id argument frame)
            | IntrinsicModule.SeqEnumerator, "moveNext" ->
                match ctx.Graph.Nodes |> Map.tryFind frame.Generator with
                | Some ({ Kind = SemanticKind.Lambda _ } as generator) ->
                    let symbol = Alex.CodeGeneration.CallableSymbols.lambda ctx.Graph generator false
                    observe ctx node (pMoveNext node.Id argument frame symbol)
                | _ -> failure node "generator identity" $"Continuation {NodeId.value frame.Owner} has no settled generator {NodeId.value frame.Generator}"
            | IntrinsicModule.SeqEnumerator, "current" ->
                if ctx.Graph.Codata.Value.SequenceCurrentReads.Contains node.Id then
                    accessSlot ctx node argument frame.Current false None
                else failure node "current admission" $"Current read {NodeId.value node.Id} has no settled successful-pull premise"
            | _ -> failure node "intrinsic settlement" $"{info.FullName} requires Baker elaboration before continuation witnessing"
        | _ -> failure node "frame identity" $"{info.FullName} operand {NodeId.value argument} has no settled sequence origin"
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
            | Some initializers -> observe ctx node (pConstructSequence node.Id frame initializers)
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
