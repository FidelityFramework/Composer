/// SeqWitness - Observe sequence operations at their graph focus.
///
/// Unelaborated suspension nodes require upstream Baker settlement; the
/// witness does not reconstruct a frame from body shape or mutable bindings.
///
/// NANOPASS: This witness handles ONLY Seq-related nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.SeqWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ControlFlowPatterns

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Seq operations - category-selective (handles only Seq nodes)
let private witnessSeq (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.SeqExpr _ ->
        WitnessOutput.error "SeqExpr requires Baker-settled suspension segments, frame and resumption; delimiter ownership alone is insufficient"
    | SemanticKind.Yield _ ->
        WitnessOutput.error "Yield requires Baker-settled suspension segments, frame and resumption; delimiter ownership alone is insufficient"
    | SemanticKind.YieldBang _ ->
        WitnessOutput.error "YieldBang requires Baker-settled suspension segments, frame and resumption; delimiter ownership alone is insufficient"
    | _ ->
        match tryMatch pForEach ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((_, collectionId, _), _) ->
            match MLIRAccumulator.recallNode collectionId ctx.Accumulator with
            | None -> WitnessOutput.error "ForEach: Collection not yet witnessed"
            | Some (collectionSSA, _) ->
                let arch = ctx.Coeffects.Platform.TargetArch
                let bodyOps = []
                match tryMatchWithDiagnostics (pBuildForEachLoop collectionSSA bodyOps arch) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | Result.Error diagnostic -> WitnessOutput.error $"ForEach: {diagnostic}"

        | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Seq nanopass - rejects unelaborated suspension nodes and handles ForEach
let nanopass : Nanopass = {
    Name = "Seq"
    Witness = witnessSeq
}
