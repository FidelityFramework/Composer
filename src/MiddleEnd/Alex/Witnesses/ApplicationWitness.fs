/// ApplicationWitness - Witness function application nodes (non-intrinsic)
///
/// Handles: curry flattening, VarRef function calls, indirect calls.
/// Intrinsic operations (MemRef.*, String.*, Sys.*, Operators.*, Convert.*)
/// are handled by domain-specific witnesses (MemoryIntrinsicWitness, etc.)
///
/// NANOPASS: This witness handles ONLY non-intrinsic Application nodes.
module Alex.Witnesses.ApplicationWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ApplicationPatterns
open Alex.CodeGeneration.TypeMapping

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
                let arch = ctx.Coeffects.Platform.TargetArch
                let retType = mapNativeTypeForArch arch node.Type

                // Retrieve deferred InlineOps for partial app arguments
                let deferredOps =
                    satInfo.AllArgNodes
                    |> List.collect (fun argId -> MLIRAccumulator.getDeferredInlineOps argId ctx.Accumulator)

                match tryMatch (pDirectCall node.Id funcName args retType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some ((ops, result), _) -> { InlineOps = deferredOps @ ops; TopLevelOps = []; Result = result }
                | None -> WitnessOutput.error $"Saturated call to {funcName}: pattern emission failed"
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
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let retType = mapNativeTypeForArch arch node.Type

                    match tryMatch (pDirectCall node.Id funcName args retType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Direct function call pattern emission failed"

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
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let retType = mapNativeTypeForArch arch node.Type
                    match tryMatch (pDirectCall node.Id localName args retType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Direct function call pattern emission failed"

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
                        let arch = ctx.Coeffects.Platform.TargetArch
                        let retType = mapNativeTypeForArch arch node.Type

                        match tryMatch (pApplicationCall node.Id funcSSA args retType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "Application pattern emission failed"
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
