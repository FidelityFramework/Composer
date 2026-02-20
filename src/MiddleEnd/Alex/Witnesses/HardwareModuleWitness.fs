/// HardwareModuleWitness - Witness [<HardwareModule>] bindings for FPGA Mealy machines
///
/// Handles Binding nodes with declRoot = HardwareModule. The child is a RecordExpr
/// (Design<'State, 'Report>) — compile-time metadata that describes the hardware:
///   - InitialState: register reset values
///   - Step: function reference (VarRef → Lambda already emitted by LambdaWitness)
///   - Clock: clock endpoint
///
/// HardwareModule Binding is a scope boundary — the Design RecordExpr children are
/// NOT auto-visited. The witness reads PSG structure directly for metadata extraction,
/// then marks all child nodes as visited to satisfy coverage validation.
///
/// Produces hw.module with seq.compreg registers + hw.instance of step function.
///
/// FPGA-only: registered conditionally in WitnessRegistry.
module Alex.Witnesses.HardwareModuleWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.ScopeContext
open Alex.Patterns.HardwareModulePatterns

// ═══════════════════════════════════════════════════════════
// PSG STRUCTURE EXTRACTION
// ═══════════════════════════════════════════════════════════

/// Unwrap TypeAnnotation nodes to find the underlying node.
/// Design<S,R> bindings often have: Binding → TypeAnnotation → RecordExpr
let rec private unwrapTypeAnnotation (graph: SemanticGraph) (nodeId: NodeId) : NodeId =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.TypeAnnotation (innerNodeId, _) -> unwrapTypeAnnotation graph innerNodeId
        | _ -> nodeId
    | None -> nodeId

/// Extract the RecordExpr fields from the Design child of a HardwareModule Binding.
/// Sees through TypeAnnotation wrappers (Binding → TypeAnnotation → RecordExpr).
let private extractDesignFields (graph: SemanticGraph) (bindingNode: SemanticNode) : (string * NodeId) list option =
    match bindingNode.Children with
    | [valueId] ->
        let unwrappedId = unwrapTypeAnnotation graph valueId
        match SemanticGraph.tryGetNode unwrappedId graph with
        | Some valueNode ->
            match valueNode.Kind with
            | SemanticKind.RecordExpr (fields, _) -> Some fields
            | _ -> None
        | None -> None
    | _ -> None

/// Find a field's NodeId by name from a Design record's field list
let private findDesignField (fieldName: string) (fields: (string * NodeId) list) : NodeId option =
    fields |> List.tryFind (fun (n, _) -> n = fieldName) |> Option.map snd

/// Extract InitialState field values as (fieldName, literalValue) pairs.
/// The InitialState is itself a RecordExpr with literal field values.
let private extractInitialStateValues (graph: SemanticGraph) (initNodeId: NodeId) : (string * int64) list option =
    match SemanticGraph.tryGetNode initNodeId graph with
    | Some initNode ->
        match initNode.Kind with
        | SemanticKind.RecordExpr (fields, _) ->
            let results =
                fields |> List.map (fun (name, valueId) ->
                    match SemanticGraph.tryGetNode valueId graph with
                    | Some valueNode ->
                        match valueNode.Kind with
                        | SemanticKind.Literal (NativeLiteral.Int (v, _)) -> Some (name, v)
                        | SemanticKind.Literal (NativeLiteral.Bool true) -> Some (name, 1L)
                        | SemanticKind.Literal (NativeLiteral.Bool false) -> Some (name, 0L)
                        | _ -> None
                    | None -> None)
            if results |> List.forall Option.isSome then
                Some (results |> List.map Option.get)
            else
                None
        | _ -> None
    | None -> None

/// Determine qualified module name for the HardwareModule Binding
let private qualifiedBindingName (graph: SemanticGraph) (node: SemanticNode) (bindingName: string) : string =
    match node.Parent with
    | Some parentId ->
        match SemanticGraph.tryGetNode parentId graph with
        | Some parentNode ->
            match parentNode.Kind with
            | SemanticKind.ModuleDef (moduleName, _) ->
                sprintf "%s.%s" moduleName bindingName
            | _ -> bindingName
        | None -> bindingName
    | None -> bindingName

/// Mark all nodes in a subtree as visited (for coverage validation).
/// HardwareModule children are compile-time metadata — they don't produce MLIR ops,
/// but they must be marked visited to prevent false coverage errors.
let rec private markSubtreeVisited (graph: SemanticGraph) (nodeId: NodeId) (visited: ref<Set<NodeId>>) : unit =
    if not (Set.contains nodeId !visited) then
        visited := Set.add nodeId !visited
        match SemanticGraph.tryGetNode nodeId graph with
        | Some node ->
            for childId in node.Children do
                markSubtreeVisited graph childId visited
        | None -> ()

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

let private witnessHardwareModule (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.Binding (name, _, _, Some DeclRoot.HardwareModule) ->
        // Mark all child nodes as visited (compile-time metadata, not runtime ops)
        for childId in node.Children do
            markSubtreeVisited ctx.Graph childId ctx.GlobalVisited

        // ── 1. Extract Design<S,R> RecordExpr (through TypeAnnotation) ──
        match extractDesignFields ctx.Graph node with
        | None ->
            WitnessOutput.error $"HardwareModule '{name}': Child is not a RecordExpr (Design<S,R>)"
        | Some designFields ->

        // ── 2. Find Design fields: InitialState, Step ──
        let initNodeOpt = findDesignField "InitialState" designFields
        let stepNodeOpt = findDesignField "Step" designFields

        match initNodeOpt, stepNodeOpt with
        | None, _ ->
            WitnessOutput.error $"HardwareModule '{name}': Missing 'InitialState' field in Design record"
        | _, None ->
            WitnessOutput.error $"HardwareModule '{name}': Missing 'Step' field in Design record"
        | Some initNodeId, Some stepNodeId ->

        // ── 3. Extract InitialState reset values ──
        match extractInitialStateValues ctx.Graph initNodeId with
        | None ->
            WitnessOutput.error $"HardwareModule '{name}': InitialState fields must be integer/bool literals"
        | Some resetValues ->

        // ── 4. Resolve Step function name ──
        match resolveStepFunctionName ctx.Graph stepNodeId with
        | None ->
            WitnessOutput.error $"HardwareModule '{name}': 'Step' field must be a VarRef to a function"
        | Some stepFuncName ->

        // ── 5. Extract State type info ──
        // InitialState's type IS the State type
        match SemanticGraph.tryGetNode initNodeId ctx.Graph with
        | None ->
            WitnessOutput.error $"HardwareModule '{name}': InitialState node not found"
        | Some initNode ->
            let stateType = mapType initNode.Type ctx

            // Verify state type is TStruct
            match stateType with
            | TStruct stateFields ->
                // Match state fields with reset values
                let stateFieldInfo =
                    List.zip stateFields resetValues
                    |> List.map (fun ((fieldName, fieldTy), (_, resetVal)) ->
                        (fieldName, fieldTy, resetVal))

                // ── 6. Build Mealy machine hw.module ──
                let moduleName = qualifiedBindingName ctx.Graph node name
                let info : MealyMachineInfo = {
                    ModuleName = moduleName
                    StepFunctionName = stepFuncName
                    StateType = stateType
                    StateFields = stateFieldInfo
                }

                let hwModuleOp = buildMealyMachineModule info

                // Add hw.module to root scope (top-level declaration)
                let updatedRootScope = ScopeContext.addOp hwModuleOp !ctx.RootScopeContext
                ctx.RootScopeContext := updatedRootScope

                // HardwareModule binding is structural — no inline ops, no value
                { InlineOps = []; TopLevelOps = []; Result = TRVoid }

            | _ ->
                WitnessOutput.error $"HardwareModule '{name}': State type must be a record (TStruct), got {stateType}"

    | _ -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// HardwareModule nanopass - witnesses [<HardwareModule>] bindings
/// FPGA-only: should be conditionally registered in WitnessRegistry
let nanopass : Nanopass = {
    Name = "HardwareModule"
    Witness = witnessHardwareModule
}
