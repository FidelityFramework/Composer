/// VarRefWitness - Witness variable reference nodes
///
/// Variable references forward immutable values or load mutable binding cells.
/// The binding SSA is looked up from the accumulator (bindings witnessed first in post-order).
///
/// FUNCTION REFERENCES: VarRef nodes pointing to function bindings (Lambda nodes) build
/// named functions in value position are elaborated by Baker into lambdas (no thunk here).
/// For direct calls, ApplicationWitness navigates to VarRef for name resolution independently.
///
/// NANOPASS: This witness handles ONLY VarRef nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.VarRefWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemRefPatterns  // pLoadMutableVariable
open Alex.Dialects.Core.Types  // TMemRef
open XParsec
open XParsec.Parsers
open XParsec.Combinators

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness variable reference nodes - forwards binding's SSA
let private witnessVarRef (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pVarRef ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((name, bindingIdOpt), _) ->
        match bindingIdOpt with
        | Some bindingId ->
            // Check if the binding references a function (Lambda node)
            match SemanticGraph.tryGetNode bindingId ctx.Graph with
            | Some bindingNode ->
                // Check binding type
                match bindingNode.Kind with
                | SemanticKind.PatternBinding _ ->
                    // Check accumulator first — match arm Var bindings are bound to
                    // the scrutinee SSA by MatchWitness, not pre-assigned in coeffects.
                    match MLIRAccumulator.recallNode bindingId ctx.Accumulator with
                    | Some (ssa, ty) ->
                        let (ops, readSSA, readTy) = adaptOperand ctx.Coeffects ctx.Graph node.Id node.Id ssa ty
                        { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = readSSA; Type = readTy } }
                    | None ->
                        // Function parameter binding — SSA is in coeffects
                        // Uses platform-aware mapping + per-node width narrowing from coeffects
                        let patternBindingPattern =
                            parser {
                                let! ssa = getNodeSSA bindingId
                                let! state = getUserState
                                let platform = state.Coeffects.TargetPlatform
                                let arch = state.Coeffects.Platform.TargetArch
                                let rawTy = mapTypeAt bindingId bindingNode.Type ctx
                                let ty = Alex.XParsec.PSGCombinators.narrowType state.Coeffects state.Graph bindingId rawTy
                                // the parameter's width, then this read's own (ruling 3)
                                let! (meetOps, readSSA, readTy) = pAdapt node.Id node.Id ssa ty
                                return (meetOps, TRValue { SSA = readSSA; Type = readTy })
                            }

                        match tryMatch patternBindingPattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error $"VarRef '{name}': PatternBinding has no SSA in coeffects"

                | SemanticKind.Binding (_, isMut, _, _) ->
                    // Immutable Lambda bindings forward their function value. A mutable
                    // binding's initializer does not change the cell-load contract.
                    let isFunctionBinding =
                        bindingNode.Children
                        |> List.tryHead
                        |> Option.bind (fun childId -> SemanticGraph.tryGetNode childId ctx.Graph)
                        |> Option.map (fun childNode -> match childNode.Kind with SemanticKind.Lambda _ -> true | _ -> false)
                        |> Option.defaultValue false

                    if not isMut && isFunctionBinding then
                        // Function reference - check if the binding holds a closure value
                        match MLIRAccumulator.recallNode bindingId ctx.Accumulator with
                        | Some (closureSSA, closureTy) ->
                            // Closure-captured function value — return the closure pair
                            { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = closureSSA; Type = closureTy } }
                        | None ->
                            // Call position: ApplicationWitness handles the direct call. A named
                            // function in value position never reaches here: Baker elaborates it
                            // into an eta-expanded Lambda marked for closure pair construction,
                            // witnessed by the closure path like any other lambda.
                            { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                    elif ModuleValues.isSlotBinding ctx.Coeffects.TargetPlatform ctx.Graph bindingNode then
                        // Module-level value: reload from its slot (valid in any function)
                        let bindingName = match bindingNode.Kind with SemanticKind.Binding (n, _, _, _) -> n | _ -> name
                        // the slot's element type at the binding's range width on fabric
                        let valueTy = mapTypeAt bindingId bindingNode.Type ctx |> narrowType ctx.Coeffects ctx.Graph bindingId
                        let globalName = ModuleValues.globalName bindingName bindingId
                        match tryMatchWithDiagnostics (pGlobalSlotLoad bindingId node.Id globalName valueTy)
                                      ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Result.Ok ((ops, TRValue v), _) ->
                            let (meetOps, readSSA, readTy) = adaptOperand ctx.Coeffects ctx.Graph node.Id node.Id v.SSA v.Type
                            { InlineOps = ops @ meetOps; TopLevelOps = []; Result = TRValue { SSA = readSSA; Type = readTy } }
                        | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | Result.Error diagnostic -> WitnessOutput.error $"VarRef '{name}': {diagnostic}"
                    elif Set.contains bindingId ctx.Graph.Codata.Value.Curry.PartialAppBindings then
                        // Partial application binding - no value SSA available
                        // ApplicationWitness handles saturated calls through the coeffect
                        { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                    else
                        // Value binding - post-order: binding already witnessed, recall its SSA
                        match MLIRAccumulator.recallNode bindingId ctx.Accumulator with
                        | Some (ssa, ty) ->
                            // Auto-load ONLY if the Binding is mutable (isMut from PSG).
                            // Mutable bindings hold memref<1xT> cells that need memref.load.
                            // Immutable bindings (including MemRef.alloca results) forward as-is.
                            if isMut then
                                // Mutable cell — extract element type for auto-load
                                let elemTypeOpt =
                                    match ty with
                                    | TMemRef elemType -> Some elemType
                                    | TMemRefStatic (_, elemType) -> Some elemType
                                    | _ -> None
                                match elemTypeOpt with
                                | Some elemType ->
                                    let (NodeId nodeIdInt) = node.Id
                                    match tryMatchWithDiagnostics (pLoadMutableVariable nodeIdInt ssa elemType)
                                                  ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                                    | Result.Ok ((ops, TRValue v), _) ->
                                        // the cell's width, then this read's own (ruling 3: a read
                                        // refined under a guard truncates; at a boundary it extends)
                                        let (meetOps, readSSA, readTy) = adaptOperand ctx.Coeffects ctx.Graph node.Id node.Id v.SSA v.Type
                                        { InlineOps = ops @ meetOps; TopLevelOps = []; Result = TRValue { SSA = readSSA; Type = readTy } }
                                    | Result.Ok ((ops, result), _) ->
                                        { InlineOps = ops; TopLevelOps = []; Result = result }
                                    | Result.Error diagnostic ->
                                        WitnessOutput.error $"VarRef '{name}': {diagnostic}"
                                | None ->
                                    WitnessOutput.error $"VarRef '{name}': Mutable cell has unexpected type {ty}"
                            else
                                // Immutable value (including buffers): forward, at this read's own width
                                let (meetOps, readSSA, readTy) = adaptOperand ctx.Coeffects ctx.Graph node.Id node.Id ssa ty
                                { InlineOps = meetOps; TopLevelOps = []; Result = TRValue { SSA = readSSA; Type = readTy } }
                        | None ->
                            WitnessOutput.error $"VarRef '{name}': Binding not yet witnessed"

                | _ ->
                    WitnessOutput.error $"VarRef '{name}': Unexpected binding kind {bindingNode.Kind}"
            | None ->
                WitnessOutput.error $"VarRef '{name}': Binding node not found"
        | None ->
            WitnessOutput.error $"VarRef '{name}': No binding ID (unresolved reference)"
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// VarRef nanopass - witnesses variable references
let nanopass : Nanopass = {
    Name = "VarRef"
    Witness = witnessVarRef
}
