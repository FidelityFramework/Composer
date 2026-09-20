/// ApplicationWitness - Witness function application nodes (non-intrinsic)
///
/// Accumulator-driven dispatch:
/// - Closure pair in accumulator → pClosureCall (indirect through pair)
/// - No closure pair → pDirectCall (known function name)
///
/// Curry flattening: coeffect-driven direct call (Baker optimization)
/// Intrinsics: delegated to domain-specific witnesses
///
/// NANOPASS: This witness handles ONLY non-intrinsic Application nodes.
module Alex.Witnesses.ApplicationWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ApplicationPatterns
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Helper: Navigate to actual function node (unwrap TypeAnnotation if present)
let private resolveFunctionNode funcId graph =
    match SemanticGraph.tryGetNode funcId graph with
    | Some funcNode ->
        match funcNode.Kind with
        | SemanticKind.TypeAnnotation (innerFuncId, _) ->
            SemanticGraph.tryGetNode innerFuncId graph
        | _ -> Some funcNode
    | None -> None

/// Extract parameter names from a Binding's Lambda child (for hw.instance port names)
/// Navigates: Binding → Lambda chain → parameter name list
let private extractParamNames (bindingId: NodeId) (graph: SemanticGraph) : string list option =
    match SemanticGraph.tryGetNode bindingId graph with
    | Some bindingNode ->
        // Find Lambda child of Binding
        let lambdaChild =
            bindingNode.Children
            |> List.tryPick (fun childId ->
                match SemanticGraph.tryGetNode childId graph with
                | Some child ->
                    match child.Kind with
                    | SemanticKind.Lambda (params', _, _, _, _) -> Some params'
                    | _ -> None
                | None -> None)
        match lambdaChild with
        | Some params' -> Some (params' |> List.map (fun (name, _, _) -> name))
        | None -> None
    | None -> None

/// Each argument of a call adapted to the slot it meets (the parameter node's width for a
/// direct call, the declared Register width through a value): the meets SSAAssignment derived
/// for (call, argument), read and transcribed here.
let private adaptArguments (ctx: WitnessContext) (callId: NodeId) (args: (NodeId * (SSA * MLIRType)) list) : MLIROp list * (SSA * MLIRType) list =
    let adapted =
        args |> List.map (fun (argId, (ssa, ty)) ->
            let (ops, ssa', ty') = adaptOperand ctx.Coeffects ctx.Graph callId argId ssa ty
            (ops, (ssa', ty')))
    (adapted |> List.collect fst, adapted |> List.map snd)

/// The body of the lambda a binding holds (through an annotation): the node whose held width
/// is the width the callee returns at.
let private calleeBody (graph: SemanticGraph) (bindingId: NodeId) : NodeId option =
    match SemanticGraph.tryGetNode bindingId graph with
    | Some ({ Kind = SemanticKind.Binding _ } as binding) ->
        binding.Children
        |> List.tryPick (fun childId ->
            match SemanticGraph.tryGetNode childId graph with
            | Some { Kind = SemanticKind.Lambda (_, bodyId, _, _, _) } -> Some bodyId
            | Some { Kind = SemanticKind.TypeAnnotation (inner, _) } ->
                match SemanticGraph.tryGetNode inner graph with
                | Some { Kind = SemanticKind.Lambda (_, bodyId, _, _, _) } -> Some bodyId
                | _ -> None
            | _ -> None)
    | _ -> None

/// A direct call's result: the callee returns at its body's width; the call node reads it at its
/// own through the meet derived for (call, call).
let private callResult (ctx: WitnessContext) (node: SemanticNode) (ops: MLIROp list) (result: TransferResult) : WitnessOutput =
    match result with
    | TRValue v ->
        let (meetOps, ssa, ty) = adaptOperand ctx.Coeffects ctx.Graph node.Id node.Id v.SSA v.Type
        { InlineOps = ops @ meetOps; TopLevelOps = []; Result = TRValue { SSA = ssa; Type = ty } }
    | other -> { InlineOps = ops; TopLevelOps = []; Result = other }

/// The type a direct call returns: the callee's body at its held width (a scalar), or the
/// mapped result type.
let private calleeReturnType (ctx: WitnessContext) (node: SemanticNode) (bodyId: NodeId option) : MLIRType =
    match bodyId with
    | Some body -> mapTypeAt node.Id node.Type ctx |> narrowType ctx.Coeffects ctx.Graph body
    | None -> mapTypeAt node.Id node.Type ctx |> narrowType ctx.Coeffects ctx.Graph node.Id

/// Witness application nodes - emits function calls (non-intrinsic only)
let private witnessApplication (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // A settled foreign call belongs to PlatformWitness, including saturated
    // calls. Emitting the generated placeholder body would bypass the ABI.
    if Map.containsKey node.Id ctx.Coeffects.Platform.Bindings.Bindings
       || (match node.Kind with
           | SemanticKind.Application (callee, _) -> (Clef.Compiler.PSGSaturation.SemanticGraph.MappedBindings.tryFindCall ctx.Graph callee).IsSome
           | _ -> false) then WitnessOutput.skip
    else
    match tryMatch pApplication ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((funcId, argIds), _) ->
        // ═══════════════════════════════════════════════════════════
        // CURRY FLATTENING: Check for saturated call or partial app
        // ═══════════════════════════════════════════════════════════
        let curryResult = ctx.Graph.Codata.Value.Curry
        match Map.tryFind node.Id curryResult.SaturatedCalls with
        | Some satInfo ->
            // Saturated call: emit direct call to flattened function with ALL args
            let funcName =
                Alex.CodeGeneration.CallableSymbols.tryBinding ctx.Graph satInfo.TargetBindingId
                |> Option.defaultWith (fun () -> sprintf "saturated_%d" (NodeId.value node.Id))

            let argsResult =
                satInfo.AllArgNodes
                |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)
            let allWitnessed = argsResult |> List.forall Option.isSome
            if not allWitnessed then
                let unwitnessedArgs =
                    List.zip satInfo.AllArgNodes argsResult
                    |> List.filter (fun (_, r) -> Option.isNone r)
                    |> List.map fst
                WitnessOutput.error $"Saturated call to {funcName}: some args not witnessed: {unwitnessedArgs}"
            else
                // each argument at its parameter's width: the call's derived meets
                let (meetOps, args) = adaptArguments ctx node.Id (List.zip satInfo.AllArgNodes (argsResult |> List.choose id))
                let retType = calleeReturnType ctx node (calleeBody ctx.Graph satInfo.TargetBindingId)

                // Retrieve deferred InlineOps for partial app arguments
                let deferredOps =
                    satInfo.AllArgNodes
                    |> List.collect (fun argId -> MLIRAccumulator.getDeferredInlineOps argId ctx.Accumulator)

                let paramNames = extractParamNames satInfo.TargetBindingId ctx.Graph
                match tryMatchWithDiagnostics (pDirectCall node.Id funcName args retType paramNames) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Result.Ok ((ops, result), _) -> callResult ctx node (deferredOps @ meetOps @ ops) result
                | Result.Error diagnostic -> WitnessOutput.error $"Saturated call '{funcName}': {diagnostic}"
        | None ->
        match Map.tryFind node.Id curryResult.PartialApplications with
        | Some _ ->
            // Partial application: no MLIR emitted, args captured for saturated call site
            { InlineOps = []; TopLevelOps = []; Result = TRVoid }
        | None ->

        // ═══════════════════════════════════════════════════════════
        // STANDARD APPLICATION HANDLING (non-intrinsic only)
        // ═══════════════════════════════════════════════════════════
        // Uniform calling convention: all function values are closure pairs.
        // VarRefWitness produces pairs for named functions (via thunk) and closures.
        // One path: recall function SSA from accumulator → pClosureCall.

        match resolveFunctionNode funcId ctx.Graph with
        | Some funcNode ->
            match funcNode.Kind with
            | SemanticKind.Intrinsic _ ->
                // Intrinsic operations handled by domain witnesses
                // (MemoryIntrinsicWitness, StringIntrinsicWitness, ArithIntrinsicWitness, PlatformWitness)
                WitnessOutput.skip

            | _ ->
                // Uniform closure call — function node must have been witnessed with a closure pair
                let closureLookup =
                    match funcNode.Kind with
                    | SemanticKind.VarRef (_, Some defId) ->
                        // Check the definition binding first (for Bindings that hold closures),
                        // then the VarRef node itself (function parameters witnessed by VarRefWitness)
                        match MLIRAccumulator.recallNode defId ctx.Accumulator with
                        | Some (ssa, ty) when (match ty with TMemRefStatic _ -> true | _ -> false) -> Some (ssa, ty)
                        | _ ->
                            match MLIRAccumulator.recallNode funcNode.Id ctx.Accumulator with
                            | Some (ssa, ty) when (match ty with TMemRefStatic _ -> true | _ -> false) -> Some (ssa, ty)
                            | _ -> None
                    | _ ->
                        // Non-VarRef function expression (e.g. inline lambda result)
                        match MLIRAccumulator.recallNode funcId ctx.Accumulator with
                        | Some (ssa, ty) when (match ty with TMemRefStatic _ -> true | _ -> false) -> Some (ssa, ty)
                        | _ -> None

                // Recall arguments (shared by both paths)
                let argsResult =
                    argIds
                    |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)
                let allWitnessed = argsResult |> List.forall Option.isSome

                match closureLookup with
                | Some (closureSSA, _closureTy) ->
                    // ═══ CLOSURE CALL: function value is a closure pair ═══
                    if not allWitnessed then
                        let missing = List.zip argIds argsResult |> List.choose (fun (id, r) -> if r.IsNone then Some (NodeId.value id) else None)
                        WitnessOutput.errorCoded AX2001 (Some node.Id) (Some "Application") (Some "ClosureCall")
                            (sprintf "Closure call: arguments not yet witnessed (missing nodes: %A)" missing)
                    else
                        // a call through a value: every argument at the declared Register width
                        // (ruling 1), the call's derived meets
                        let (meetOps, args) = adaptArguments ctx node.Id (List.zip argIds (argsResult |> List.choose id))
                        let retType = mapTypeAt node.Id node.Type ctx |> narrowType ctx.Coeffects ctx.Graph node.Id

                        match tryMatchWithDiagnostics (pClosureCall node.Id closureSSA args retType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Result.Ok ((ops, result), _) -> { InlineOps = meetOps @ ops; TopLevelOps = []; Result = result }
                        | Result.Error diagnostic ->
                            WitnessOutput.errorCoded AX2004 (Some node.Id) (Some "Application") (Some "ClosureCall")
                                (sprintf "Closure call: %s" diagnostic)
                | None ->
                    // ═══ DIRECT CALL: no closure pair (extern functions, non-HOF calls) ═══
                    // Resolve qualified function name from PSG
                    let funcName =
                        match funcNode.Kind with
                        | SemanticKind.VarRef (localName, Some defId) ->
                            Alex.CodeGeneration.CallableSymbols.tryBinding ctx.Graph defId
                            |> Option.defaultValue localName
                        | SemanticKind.VarRef (localName, None) -> localName
                        | _ -> sprintf "func_%d" (NodeId.value funcId)

                    if not allWitnessed then
                        let missing = List.zip argIds argsResult |> List.choose (fun (id, r) -> if r.IsNone then Some (NodeId.value id) else None)
                        WitnessOutput.errorCoded AX2001 (Some node.Id) (Some "Application") (Some "DirectCall")
                            (sprintf "Call to '%s': arguments not yet witnessed (missing nodes: %A)" funcName missing)
                    else
                        // each argument at its parameter's width: the call's derived meets
                        let (meetOps, args) = adaptArguments ctx node.Id (List.zip argIds (argsResult |> List.choose id))
                        let defIdOpt = match funcNode.Kind with SemanticKind.VarRef (_, d) -> d | _ -> None
                        let retType = calleeReturnType ctx node (defIdOpt |> Option.bind (calleeBody ctx.Graph))
                        let paramNames = defIdOpt |> Option.bind (fun d -> extractParamNames d ctx.Graph)
                        match tryMatchWithDiagnostics (pDirectCall node.Id funcName args retType paramNames) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Result.Ok ((ops, result), _) -> callResult ctx node (meetOps @ ops) result
                        | Result.Error diagnostic ->
                            WitnessOutput.errorCoded AX2003 (Some node.Id) (Some "Application") (Some "DirectCall")
                                (sprintf "Call to '%s': %s" funcName diagnostic)
        | None ->
            WitnessOutput.errorCoded AX2002 (Some node.Id) (Some "Application") (Some "ResolveFunctionNode")
                (sprintf "Could not resolve function node %d" (NodeId.value funcId))
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Application nanopass - witnesses function applications (non-intrinsic)
let nanopass : Nanopass = {
    Name = "Application"
    Witness = witnessApplication
}
