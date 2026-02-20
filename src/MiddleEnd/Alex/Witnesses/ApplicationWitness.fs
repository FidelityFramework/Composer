/// ApplicationWitness - Witness function application nodes (non-intrinsic)
///
/// Handles: curry flattening, VarRef function calls, indirect calls.
/// Intrinsic operations (MemRef.*, String.*, Sys.*, Operators.*, Convert.*)
/// are handled by domain-specific witnesses (MemoryIntrinsicWitness, etc.)
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
                let retType = mapType node.Type ctx

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

        match resolveFunctionNode funcId ctx.Graph with
        | Some funcNode ->
            match funcNode.Kind with
            | SemanticKind.Intrinsic _ ->
                // Intrinsic operations handled by domain witnesses
                // (MemoryIntrinsicWitness, StringIntrinsicWitness, ArithIntrinsicWitness, PlatformWitness)
                WitnessOutput.skip

            | SemanticKind.VarRef (localName, Some defId) ->
                // Extract qualified function name using PSG resolution
                let funcName =
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

                // Recall argument SSAs with types
                let argsResult =
                    argIds
                    |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)

                let allWitnessed = argsResult |> List.forall Option.isSome
                if not allWitnessed then
                    WitnessOutput.error "Application: Some arguments not yet witnessed"
                else
                    let args = argsResult |> List.choose id
                    let retType = mapType node.Type ctx

                    let paramNames = extractParamNames defId ctx.Graph
                    match tryMatchWithDiagnostics (pDirectCall node.Id funcName args retType paramNames) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"Direct call '{funcName}': {diagnostic}"

            | SemanticKind.VarRef (localName, None) ->
                // Unresolved VarRef — try direct call with local name
                let argsResult =
                    argIds
                    |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)
                let allWitnessed = argsResult |> List.forall Option.isSome
                if not allWitnessed then
                    WitnessOutput.error "Application: Some arguments not yet witnessed"
                else
                    let args = argsResult |> List.choose id
                    let retType = mapType node.Type ctx
                    match tryMatchWithDiagnostics (pDirectCall node.Id localName args retType None) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"Direct call '{localName}': {diagnostic}"

            | _ ->
                // Function is an SSA value (indirect call)
                match MLIRAccumulator.recallNode funcId ctx.Accumulator with
                | None -> WitnessOutput.error "Application: Function not yet witnessed"
                | Some (funcSSA, _) ->
                    let argsResult =
                        argIds
                        |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)

                    let allWitnessed = argsResult |> List.forall Option.isSome
                    if not allWitnessed then
                        WitnessOutput.error "Application: Some arguments not yet witnessed"
                    else
                        let args = argsResult |> List.choose id
                        let retType = mapType node.Type ctx

                        match tryMatchWithDiagnostics (pApplicationCall node.Id funcSSA args retType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | Result.Error diagnostic -> WitnessOutput.error $"Indirect call: {diagnostic}"
        | None ->
            WitnessOutput.error $"Application: Could not resolve function node {funcId}"
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
