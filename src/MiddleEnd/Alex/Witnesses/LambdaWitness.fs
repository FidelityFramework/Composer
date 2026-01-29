/// LambdaWitness - Witness Lambda operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to ClosurePatterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Lambda nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
///
/// SPECIAL CASE: Entry point Lambdas need to witness function bodies (sub-graphs)
/// that can contain ANY category of nodes. Uses subGraphCombinator to fold over
/// all registered witnesses.
module Alex.Witnesses.LambdaWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.PSGZipper
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ClosurePatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// SUB-GRAPH COMBINATOR
// ═══════════════════════════════════════════════════════════

/// Build sub-graph combinator from nanopass list
/// Folds over all witnesses, returning first non-skip result
let private makeSubGraphCombinator (nanopasses: Nanopass list) : (WitnessContext -> SemanticNode -> WitnessOutput) =
    fun ctx node ->
        let rec tryWitnesses witnesses =
            match witnesses with
            | [] -> WitnessOutput.skip
            | nanopass :: rest ->
                match nanopass.Witness ctx node with
                | output when output = WitnessOutput.skip ->
                    tryWitnesses rest
                | output -> output
        tryWitnesses nanopasses

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness Lambda operations - category-selective (handles only Lambda nodes)
/// Takes nanopass list to build sub-graph combinator for body witnessing
let private witnessLambdaWith (nanopasses: Nanopass list) (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Filter to ONLY ContentPhase witnesses for sub-graph traversal
    // This prevents StructuralPhase witnesses from recursing and causing double-witnessing
    let contentWitnesses = nanopasses |> List.filter (fun np -> np.Phase = ContentPhase)
    let subGraphCombinator = makeSubGraphCombinator contentWitnesses

    match tryMatch pLambdaWithCaptures ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((params', bodyId, captureInfos), _) ->
        // Check if this is an entry point Lambda
        let nodeIdValue = NodeId.value node.Id
        let isEntryPoint = Set.contains nodeIdValue ctx.Coeffects.EntryPointLambdaIds

        if isEntryPoint then
            // Entry point Lambda: generate func.func @main wrapper
            let scopeLabel = sprintf "func_%d" nodeIdValue
            let scopeKind = FunctionScope "main"

            // Mark scope entry
            MLIRAccumulator.addOp (MLIROp.ScopeMarker (ScopeEnter (scopeKind, scopeLabel))) ctx.Accumulator

            // Witness body nodes into shared accumulator
            match SemanticGraph.tryGetNode bodyId ctx.Graph with
            | Some bodyNode ->
                match focusOn bodyId ctx.Zipper with
                | Some bodyZipper ->
                    let bodyCtx = { ctx with Zipper = bodyZipper }
                    let bodyVisited = ref Set.empty  // Fresh visited set for body traversal
                    visitAllNodes subGraphCombinator bodyCtx bodyNode ctx.Accumulator bodyVisited
                | None -> ()
            | None -> ()

            // Mark scope exit
            MLIRAccumulator.addOp (MLIROp.ScopeMarker (ScopeExit (scopeKind, scopeLabel))) ctx.Accumulator

            // Extract operations between markers
            let scopeOps = MLIRAccumulator.extractScope scopeKind scopeLabel ctx.Accumulator

            // Separate body ops from module-level ops
            let bodyOps = scopeOps |> List.filter (fun op -> match op with MLIROp.GlobalString _ -> false | _ -> true)
            let moduleOps = scopeOps |> List.filter (fun op -> match op with MLIROp.GlobalString _ -> true | _ -> false)

            // Get body result for return value
            let bodyResult = MLIRAccumulator.recallNode bodyId ctx.Accumulator

            // Determine return value and type
            let returnSSA, returnType =
                match bodyResult with
                | Some (ssa, ty) -> (Some ssa, ty)
                | None ->
                    match List.tryLast bodyOps with
                    | Some (MLIROp.ArithOp (ArithOp.ConstI (ssa, _, ty))) -> (Some ssa, ty)
                    | _ -> (Some (SSA.V 0), TInt I32)

            let returnOp = MLIROp.FuncOp (FuncOp.Return (returnSSA, Some returnType))
            let completeBody = bodyOps @ [returnOp]

            // Build FuncDef wrapper
            let funcDef = FuncOp.FuncDef("main", [], returnType, completeBody, Public)

            // Replace scope with FuncDef
            MLIRAccumulator.replaceScope scopeKind scopeLabel (MLIROp.FuncOp funcDef) ctx.Accumulator

            // Add module-level ops separately
            MLIRAccumulator.addOps moduleOps ctx.Accumulator

            { InlineOps = []; TopLevelOps = []; Result = TRVoid }
        else
            // Non-entry-point Lambda
            match SemanticGraph.tryGetNode bodyId ctx.Graph with
            | Some bodyNode ->
                match focusOn bodyId ctx.Zipper with
                | Some bodyZipper ->
                    let bodyCtx = { ctx with Zipper = bodyZipper }
                    let bodyVisited = ref Set.empty  // Fresh visited set for body traversal
                    visitAllNodes subGraphCombinator bodyCtx bodyNode ctx.Accumulator bodyVisited
                | None -> ()
            | None -> ()

            { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Create Lambda nanopass with nanopass list for sub-graph traversal (body witnessing)
/// This must be called AFTER all other nanopasses are registered
let createNanopass (nanopasses: Nanopass list) : Nanopass = {
    Name = "Lambda"
    Phase = StructuralPhase
    Witness = witnessLambdaWith nanopasses
}

/// Placeholder nanopass export - will be replaced by createNanopass call in registry
let nanopass : Nanopass = {
    Name = "Lambda"
    Phase = StructuralPhase
    Witness = fun _ _ -> WitnessOutput.error "Lambda nanopass not properly initialized - use createNanopass"
}
