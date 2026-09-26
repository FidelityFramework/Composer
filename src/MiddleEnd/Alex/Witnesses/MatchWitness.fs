/// MatchWitness - Witness CaseElimination (structural match) via XParsec
///
/// Scope witness following the Y-combinator pattern (like ControlFlowWitness).
/// Platform-agnostic — the Pattern handles TargetPlatform.
///
/// CaseElimination preserves the fold structure from Baker:
/// - Each arm's body contains Baker's selected bindings and source guards
/// - This boundary selects only the already settled shallow pattern decision
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
module Requirements = Clef.Compiler.PSGSaturation.SemanticGraph.Requirements

// ═══════════════════════════════════════════════════════════════════════════
// BRANCH REGION COLLECTION THROUGH THE SCOPE TRAVERSAL DRIVER
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a branch scope and collect operations.
/// Creates child scope, visits sub-tree, returns collected ops.
let private visitChild (childId: NodeId) (ctx: WitnessContext) combinator =
    let position =
        ctx.Zipper.Focus.Children |> List.tryFindIndex ((=) childId)
        |> Option.bind (fun index -> down index ctx.Zipper)
    match position with
    | Some childZipper ->
        visitAllNodes combinator { ctx with Zipper = childZipper } childZipper.Focus ctx.TraversalVisited
    | None ->
        Diagnostic.error (Some ctx.Zipper.Focus.Id) (Some "CaseElimination") (Some "structural child")
            $"Cannot descend to declared match child {NodeId.value childId}"
        |> fun diagnostic -> MLIRAccumulator.addError diagnostic ctx.Accumulator

let private witnessBranchScope (rootId: NodeId) (ctx: WitnessContext) (combinator: WitnessContext -> SemanticNode -> WitnessOutput) : MLIROp list =
    let branchScope = ref (ScopeContext.createChild !ctx.ScopeContext BlockLevel)
    visitChild rootId { ctx with ScopeContext = branchScope } combinator
    ScopeContext.getOps !branchScope

/// A terminal refutable arm is selected only after Baker's requirement in this
/// exact frontier occurrence. A declaration's Parent field cannot establish it.
let private terminalAdmitted (ctx: WitnessContext) (node: SemanticNode) arms =
    match arms with
    | [{ Pattern = Pattern.Const _ | Pattern.Union _ }] ->
        match Requirements.tryPatternRequirement ctx.Graph node.Id, ctx.Zipper.Path with
        | Some contract, step :: _ ->
            step.Parent.Id = contract.Frontier && step.LeftSiblings = [contract.Site] && step.RightSiblings.IsEmpty
            && Set.contains contract.Site ctx.TraversalVisited.Value
            && (MLIRAccumulator.recallNode contract.Site ctx.Accumulator |> Option.exists (fun (_, ty) ->
                ty = Alex.CodeGeneration.TypeMapping.mapNTUKindToMLIRType NTUKind.NTUunit))
        | _ -> false
    | _ -> true

// ═══════════════════════════════════════════════════════════════════════════
// MATCH WITNESS
// ═══════════════════════════════════════════════════════════════════════════

let private witnessMatchWith (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let combinator = getCombinator()

    match tryMatch pCaseElimination ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((_, arms), _) when arms |> List.exists (fun arm -> not arm.Bindings.IsEmpty || arm.Guard.IsSome) ->
        WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "CaseElimination") (Some "selected scope")
            "Baker must settle pattern bindings and guards inside the selected body before witnessing"
    | Some ((_, arms), _) when not (terminalAdmitted ctx node arms) ->
        WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "CaseElimination") (Some "terminal requirement")
            "The terminal pattern decision is outside its validated requirement frontier"
    | Some ((scrutineeId, arms), _) ->

        // Step 1: Visit scrutinee in CURRENT scope (like ControlFlowWitness condition)
        visitChild scrutineeId ctx combinator

        // Recall scrutinee result
        match MLIRAccumulator.recallNode scrutineeId ctx.Accumulator with
        | None ->
            WitnessOutput.error "CaseElimination: Scrutinee witnessed but no result"
        | Some (scrutineeSSA, scrutineeMLIRType) ->

            // Step 2: Pull the selected bodies through their actual occurrences.
            // Baker owns extraction and guard order inside those bodies.
            let armResults =
                arms |> List.map (fun arm ->
                    let armOps = witnessBranchScope arm.Body ctx combinator
                    let armValueNodeId = findLastValueNode arm.Body ctx.Graph
                    (armOps, armValueNodeId, arm))

            // Step 3: Determine if expression-valued
            // TVar means CCS didn't resolve the match result type — treat as void
            // (if arms are side-effect-only, the match result type stays unresolved)
            let isUnit = Alex.Traversal.Values.isUnitTyped node.Type
            let isExpressionValued =
                not isUnit && (match node.Type with NativeType.TVar _ -> false | _ -> true)

            let result =
                if isExpressionValued then
                    let resultType = mapTypeAt node.Id node.Type ctx |> narrowType ctx.Coeffects ctx.Graph node.Id
                    match tryMatch (getNodeSSAs node.Id) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some (ssas, _) when ssas.Length >= 1 -> Some (ssas.[0], resultType)
                    | _ -> None
                else None

            // Step 4: Delegate to pattern for elision — diagnostic error flow preserved
            let elimination = pBuildMatchElimination scrutineeSSA scrutineeMLIRType scrutineeId armResults result node.Id
            let pattern =
                if isUnit then Alex.Patterns.LiteralPatterns.pWithUnitResult node.Id elimination
                else elimination
            match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
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
