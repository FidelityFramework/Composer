/// VarRefWitness - Witness variable reference nodes
///
/// Variable references forward immutable values or load mutable binding cells.
/// The binding SSA is looked up from the accumulator (bindings witnessed first in post-order).
///
/// Function values retain separate code/environment operands. Named code in
/// value position receives a func.constant at this actual source occurrence.
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
open Alex.Patterns.CallablePatterns
open Alex.Patterns.MutableCallablePatterns
open Alex.Patterns.SequencePatterns
open Alex.Patterns.LazyPatterns
open Alex.Dialects.Core.Types  // TMemRef
open XParsec
open XParsec.Parsers
open XParsec.Combinators

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness variable reference nodes - forwards binding's SSA
let private witnessVarRef (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let callable pattern =
        match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | Result.Error reason -> WitnessOutput.error $"Callable reference: {reason}"
    let rec directCallee (position: Alex.Traversal.PSGZipper.PSGZipper) =
        match Alex.Traversal.PSGZipper.up position with
        | Some parent ->
            match parent.Focus.Kind with
            | SemanticKind.Application(callee, _) -> callee = position.Focus.Id
            | SemanticKind.TypeAnnotation(inner, _) when inner = position.Focus.Id -> directCallee parent
            | _ -> false
        | None -> false
    let rec assignmentTarget (position: Alex.Traversal.PSGZipper.PSGZipper) =
        match Alex.Traversal.PSGZipper.up position with
        | Some parent ->
            match parent.Focus.Kind with
            | SemanticKind.Set(target, _) -> target = position.Focus.Id
            | SemanticKind.TypeAnnotation(inner, _) when inner = position.Focus.Id -> assignmentTarget parent
            | _ -> false
        | None -> false
    match tryMatch pVarRef ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((name, bindingIdOpt), _) ->
        match bindingIdOpt with
        | Some bindingId ->
            // Check if the binding references a function (Lambda node)
            match SemanticGraph.tryGetNode bindingId ctx.Graph with
            | Some bindingNode when isLazyValue ctx node ->
                match bindingNode.Kind with
                | SemanticKind.Binding(_, true, _, _) ->
                    WitnessOutput.error $"Lazy reference '{name}' requires an admitted pair storage read."
                | _ ->
                    let pattern =
                        if ModuleValues.isSlotBinding ctx.Coeffects.TargetPlatform ctx.Graph bindingNode then pProgramLazyReference ctx bindingId
                        else pLazyForward ctx bindingId
                    match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error reason -> WitnessOutput.error $"Lazy reference '{name}': {reason}"
            | Some bindingNode when isSequenceValue ctx node ->
                match bindingNode.Kind with
                | SemanticKind.Binding(_, true, _, _) ->
                    WitnessOutput.error $"Sequence reference '{name}' requires an admitted pair storage read."
                | _ ->
                    match tryMatchWithDiagnostics (pSequenceForward ctx bindingId) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error reason -> WitnessOutput.error $"Sequence reference '{name}': {reason}"
            | Some bindingNode when (match Clef.Compiler.NativeTypedTree.UnionFind.applySubst node.Type with NativeType.TFun _ -> true | _ -> false) ->
                let declaration =
                    match bindingNode.Kind, bindingNode.Children with
                    | SemanticKind.Binding(_, false, _, _), [implementation] ->
                        match ctx.Graph.Nodes.TryFind implementation with
                        | Some { Kind = SemanticKind.Lambda(_, _, [], _, _); Metadata = metadata } ->
                            [ClosureMetadata.LambdaExpression; ClosureMetadata.RequiresClosurePair]
                            |> List.forall (fun key -> metadata.TryFind key <> Some(MetadataValue.Bool true))
                        | _ -> false
                    | SemanticKind.Lambda(_, _, [], _, LambdaContext.LazyThunk), _ ->
                        (Alex.Traversal.CallableOperands.tryThunkDeclaration ctx bindingId).IsSome
                    | _ -> false
                match bindingNode.Kind with
                | SemanticKind.Binding(_, true, _, _) when assignmentTarget ctx.Zipper ->
                    // Naming the destination is not a value demand. The write
                    // witness recalls its shared cell directly.
                    { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                | SemanticKind.Binding(_, true, _, _) ->
                    match MLIRAccumulator.recallCallableCell bindingId ctx.Accumulator with
                    | None -> WitnessOutput.error $"VarRef '{name}': Binding not yet witnessed"
                    | Some _ -> callable (pReadMutableCallable ctx bindingId node.Id)
                | _ when Set.contains bindingId ctx.Graph.Codata.Value.Curry.PartialAppBindings ->
                    { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                | _ when declaration && directCallee ctx.Zipper ->
                    // The direct invocation consumes the settled declaration
                    // symbol. This actual callee occurrence needs no SSA value.
                    { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                | _ ->
                    match MLIRAccumulator.recallCallable bindingId ctx.Accumulator with
                    | Some _ -> callable (pCallableForward ctx bindingId)
                    | None when declaration -> callable (pNamedCallable ctx)
                    | None -> callable (pCallableForward ctx bindingId)
            | Some bindingNode ->
                // Check binding type
                match bindingNode.Kind with
                | SemanticKind.PatternBinding _ ->
                    // Check accumulator first — match arm Var bindings are bound to
                    // the scrutinee SSA by MatchWitness, not pre-assigned in coeffects.
                    match MLIRAccumulator.recallNode bindingId ctx.Accumulator with
                    | Some (ssa, ty) ->
                        // An occurrence-bound formal still has its parameter
                        // carrier. Transcribe this read's settled refinement,
                        // just as on the coeffect lookup path below.
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
                        if directCallee ctx.Zipper then
                            { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                        else WitnessOutput.error $"Callable reference '{name}' has no concrete settled value carrier."
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
