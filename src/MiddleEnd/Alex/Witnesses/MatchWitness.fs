/// MatchWitness - Witness CaseElimination (structural match) via XParsec
///
/// Scope witness following the Y-combinator pattern (like ControlFlowWitness).
/// Platform-agnostic — the Pattern handles TargetPlatform.
///
/// CaseElimination preserves the fold structure from Baker:
/// - Each arm has enriched bindings (DUEliminate + Binding via letBindAt)
/// - No DUGetTag/comparison/IfThenElse nodes — those are elision concerns
///
/// The witness walks each arm's sub-tree via witnessBranchScope,
/// then delegates to pBuildMatchElimination for assembly.
module Alex.Witnesses.MatchWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.ScopeContext
open Alex.Traversal.PSGZipper
open Alex.XParsec.PSGCombinators

open Alex.Patterns.ControlFlowPatterns

// ═══════════════════════════════════════════════════════════════════════════
// BRANCH SCOPE HELPER (inlined from ControlFlowWitness pattern)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a branch scope and collect operations.
/// Creates child scope, visits sub-tree, returns collected ops.
let private witnessBranchScope (rootId: NodeId) (ctx: WitnessContext) (combinator: WitnessContext -> SemanticNode -> WitnessOutput) : MLIROp list =
    let branchScope = ref (ScopeContext.createChild !ctx.ScopeContext BlockLevel)
    match SemanticGraph.tryGetNode rootId ctx.Graph with
    | Some branchNode ->
        match focusOn rootId ctx.Zipper with
        | Some branchZipper ->
            let branchCtx = { ctx with
                                Zipper = branchZipper
                                ScopeContext = branchScope }
            visitAllNodes combinator branchCtx branchNode ctx.GlobalVisited
        | None -> ()
    | None -> ()
    ScopeContext.getOps !branchScope

// ═══════════════════════════════════════════════════════════════════════════
// MATCH WITNESS
// ═══════════════════════════════════════════════════════════════════════════

let private witnessMatchWith (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let combinator = getCombinator()

    match tryMatch pCaseElimination ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((scrutineeId, arms), _) ->

        // Step 1: Visit scrutinee in CURRENT scope (like ControlFlowWitness condition)
        match SemanticGraph.tryGetNode scrutineeId ctx.Graph with
        | Some scrutineeNode ->
            visitAllNodes combinator ctx scrutineeNode ctx.GlobalVisited
        | None -> ()

        // Recall scrutinee result
        match MLIRAccumulator.recallNode scrutineeId ctx.Accumulator with
        | None ->
            WitnessOutput.error "CaseElimination: Scrutinee witnessed but no result"
        | Some (scrutineeSSA, scrutineeMLIRType) ->

            // Step 2: For each arm, witness bindings + guard + body via branch scope
            let armResults =
                arms |> List.map (fun arm ->
                    // Visit binding nodes first (DUEliminate + Binding from Baker)
                    for bindingId in arm.Bindings do
                        match SemanticGraph.tryGetNode bindingId ctx.Graph with
                        | Some bindingNode ->
                            visitAllNodes combinator ctx bindingNode ctx.GlobalVisited
                        | None -> ()

                    // Visit guard if present
                    match arm.Guard with
                    | Some guardId ->
                        match SemanticGraph.tryGetNode guardId ctx.Graph with
                        | Some guardNode ->
                            visitAllNodes combinator ctx guardNode ctx.GlobalVisited
                        | None -> ()
                    | None -> ()

                    // Visit body in isolated scope
                    let armOps = witnessBranchScope arm.Body ctx combinator
                    let armValueNodeId = findLastValueNode arm.Body ctx.Graph
                    (armOps, armValueNodeId, arm))

            // Step 3: Determine if expression-valued
            let isExpressionValued =
                match node.Type with
                | NativeType.TApp ({ NTUKind = Some NTUKind.NTUunit }, []) -> false
                | _ -> true

            let result =
                if isExpressionValued then
                    let resultType = mapType node.Type ctx
                    match tryMatch (getNodeSSAs node.Id) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some (ssas, _) when ssas.Length >= 1 -> Some (ssas.[0], resultType)
                    | _ -> None
                else None

            // Step 4: Delegate to pattern for elision — diagnostic error flow preserved
            match tryMatchWithDiagnostics (pBuildMatchElimination scrutineeSSA scrutineeMLIRType scrutineeId armResults result node.Id) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((ops, transferResult), _) ->
                { InlineOps = ops; TopLevelOps = []; Result = transferResult }
            | Result.Error diagnostic ->
                WitnessOutput.error $"CaseElimination: {diagnostic}"

    | None -> WitnessOutput.skip

/// Create nanopass with Y-combinator for recursive sub-graph witnessing
let createNanopass (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) : Nanopass = {
    Name = "Match"
    Witness = witnessMatchWith getCombinator
}
