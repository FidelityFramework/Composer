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

/// Witness application nodes - emits function calls (non-intrinsic only)
let private witnessApplication (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pApplication ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((funcId, argIds), _) ->
        // ═══════════════════════════════════════════════════════════
        // CURRY FLATTENING: Check for saturated call or partial app
        // ═══════════════════════════════════════════════════════════
        let curryResult = ctx.Coeffects.CurryFlattening
        match Map.tryFind node.Id curryResult.SaturatedCalls with
        | Some satInfo ->
            // Saturated call: emit direct call to flattened function with ALL args
            let funcName =
                match SemanticGraph.tryGetNode satInfo.TargetBindingId ctx.Graph with
                | Some bindingNode ->
                    match bindingNode.Kind with
                    | SemanticKind.Binding (bindName, _, _, _) ->
                        match bindingNode.Parent with
                        | Some parentId ->
                            match SemanticGraph.tryGetNode parentId ctx.Graph with
                            | Some parentNode ->
                                match parentNode.Kind with
                                | SemanticKind.ModuleDef (moduleName, _) ->
                                    sprintf "%s.%s" moduleName bindName
                                | _ -> bindName
                            | None -> bindName
                        | None -> bindName
                    | _ -> sprintf "saturated_%d" (NodeId.value node.Id)
                | None -> sprintf "saturated_%d" (NodeId.value node.Id)

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
                let args = argsResult |> List.choose id
                let retType = mapType node.Type ctx |> narrowType ctx.Coeffects node.Id

                // Retrieve deferred InlineOps for partial app arguments
                let deferredOps =
                    satInfo.AllArgNodes
                    |> List.collect (fun argId -> MLIRAccumulator.getDeferredInlineOps argId ctx.Accumulator)

                let paramNames = extractParamNames satInfo.TargetBindingId ctx.Graph
                match tryMatchWithDiagnostics (pDirectCall node.Id funcName args retType paramNames) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Result.Ok ((ops, result), _) -> { InlineOps = deferredOps @ ops; TopLevelOps = []; Result = result }
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
                        let args = argsResult |> List.choose id
                        let retType = mapType node.Type ctx |> narrowType ctx.Coeffects node.Id

                        match tryMatchWithDiagnostics (pClosureCall node.Id closureSSA args retType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | Result.Error diagnostic ->
                            WitnessOutput.errorCoded AX2004 (Some node.Id) (Some "Application") (Some "ClosureCall")
                                (sprintf "Closure call: %s" diagnostic)
                | None ->
                    // ═══ DIRECT CALL: no closure pair (extern functions, non-HOF calls) ═══
                    // Resolve qualified function name from PSG
                    let funcName =
                        match funcNode.Kind with
                        | SemanticKind.VarRef (localName, Some defId) ->
                            match SemanticGraph.tryGetNode defId ctx.Graph with
                            | Some bindingNode ->
                                match bindingNode.Kind with
                                | SemanticKind.Binding (bindName, _, _, _) ->
                                    match bindingNode.Parent with
                                    | Some parentId ->
                                        match SemanticGraph.tryGetNode parentId ctx.Graph with
                                        | Some parentNode ->
                                            match parentNode.Kind with
                                            | SemanticKind.ModuleDef (moduleName, _) ->
                                                sprintf "%s.%s" moduleName bindName
                                            | _ -> bindName
                                        | None -> bindName
                                    | None -> bindName
                                | _ -> localName
                            | None -> localName
                        | SemanticKind.VarRef (localName, None) -> localName
                        | _ -> sprintf "func_%d" (NodeId.value funcId)

                    if not allWitnessed then
                        let missing = List.zip argIds argsResult |> List.choose (fun (id, r) -> if r.IsNone then Some (NodeId.value id) else None)
                        WitnessOutput.errorCoded AX2001 (Some node.Id) (Some "Application") (Some "DirectCall")
                            (sprintf "Call to '%s': arguments not yet witnessed (missing nodes: %A)" funcName missing)
                    else
                        let args = argsResult |> List.choose id
                        let retType = mapType node.Type ctx |> narrowType ctx.Coeffects node.Id
                        let defIdOpt = match funcNode.Kind with SemanticKind.VarRef (_, d) -> d | _ -> None
                        let paramNames = defIdOpt |> Option.bind (fun d -> extractParamNames d ctx.Graph)
                        match tryMatchWithDiagnostics (pDirectCall node.Id funcName args retType paramNames) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
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
