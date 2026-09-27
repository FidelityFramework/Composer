/// HardwareModuleWitness - Witness [<HardwareModule>] bindings for FPGA Mealy machines
///
/// Handles Binding nodes with declRoot = HardwareModule. The child is a RecordExpr
/// (Design<'State, 'Report>) — compile-time metadata that describes the hardware:
///   - InitialState: register reset values
///   - Step: function reference (VarRef → Lambda walked by combinator → LambdaWitness)
///   - Clock: clock endpoint
///
/// HardwareModule Binding is a scope boundary — the Design RecordExpr children are
/// NOT auto-visited. The witness reads PSG structure directly for metadata extraction,
/// marks metadata nodes as visited, then walks the Step function Lambda via combinator
/// so LambdaWitness generates the step's hw.module.
///
/// Produces hw.module with seq.compreg registers + hw.instance of step function.
///
/// FPGA-only: registered conditionally in WitnessRegistry.
/// SCOPE WITNESS: receives combinator for recursive sub-graph traversal.
module Alex.Witnesses.HardwareModuleWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.ScopeContext
open Alex.XParsec.PSGCombinators  // narrowType
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

/// Extract InitialState field values as (fieldName, NativeLiteral) pairs.
/// Preserves the full NativeLiteral (including NTUKind width) through the lowering path.
/// The InitialState is itself a RecordExpr with literal field values.
let private extractInitialStateValues (graph: SemanticGraph) (initNodeId: NodeId) : (string * NativeLiteral) list option =
    // Resolve a field value to a compile-time literal constant.
    // Handles: direct literals, VarRef → Binding → Literal, TypeAnnotation wrappers.
    let rec resolveToLiteral (nodeId: NodeId) : NativeLiteral option =
        match SemanticGraph.tryGetNode nodeId graph with
        | Some node ->
            match node.Kind with
            | SemanticKind.Literal lit ->
                match lit with
                | NativeLiteral.Int _ | NativeLiteral.UInt _ | NativeLiteral.Bool _ -> Some lit
                | _ -> None  // Only integer/bool literals valid for FPGA reset values
            | SemanticKind.VarRef (_, Some bindingId) ->
                // Follow VarRef → Binding → value child
                match SemanticGraph.tryGetNode bindingId graph with
                | Some bindingNode ->
                    match bindingNode.Children with
                    | [valueId] -> resolveToLiteral valueId
                    | _ -> None
                | None -> None
            | SemanticKind.TypeAnnotation (wrappedId, _) ->
                resolveToLiteral wrappedId
            | _ -> None
        | None -> None

    match SemanticGraph.tryGetNode initNodeId graph with
    | Some initNode ->
        match initNode.Kind with
        | SemanticKind.RecordExpr (fields, _) ->
            let results =
                fields |> List.map (fun (name, valueId) ->
                    resolveToLiteral valueId |> Option.map (fun v -> (name, v)))
            if results |> List.forall Option.isSome then
                Some (results |> List.map Option.get)
            else
                None
        | _ -> None
    | None -> None

/// Extract State, Input and Output types from the step function's Lambda.
/// Step : 'S -> 'S (1-param, Design<'S>) or Step : 'S -> I -> 'S * 'R (2-param, Design<'S,'R>)
/// Returns (StateType, InputType option, OutputType option); Input and Output are
/// absent exactly when the step's signature has no input parameter or no output half.
/// StateType comes from the step Lambda's first parameter (authoritative widths from Phase 5C feedback).
/// A Step that does not resolve to a lambda with a state parameter is a settlement defect.
let private extractStepTypes (graph: SemanticGraph) (stepNodeId: NodeId) (ctx: WitnessContext) : Result<MLIRType * MLIRType option * MLIRType option, string> =
    let stepId = NodeId.value stepNodeId
    // Resolve Step VarRef → Binding → Lambda
    match SemanticGraph.tryGetNode stepNodeId graph with
    | Some { Kind = SemanticKind.VarRef (_, Some defId) } ->
        match SemanticGraph.tryGetNode defId graph with
        | Some defNode ->
            match defNode.Children with
            | [lambdaId] ->
                match SemanticGraph.tryGetNode lambdaId graph with
                | Some { Kind = SemanticKind.Lambda (params', bodyId, _, _, _) } ->
                    // Extract InputType from second parameter (if present), narrowed via coeffect
                    let inputType =
                        match params' with
                        | _ :: (_, inputNativeType, inputParamNodeId) :: _ ->
                            let raw = mapType inputNativeType ctx
                            Some (narrowType ctx.Coeffects graph inputParamNodeId raw)
                        | _ -> None
                    // The step body's value type, narrowed at its last value node
                    let rec findLastValue (nid: NodeId) : NodeId =
                        match SemanticGraph.tryGetNode nid graph with
                        | Some n ->
                            match n.Kind with
                            | SemanticKind.Sequential children ->
                                match List.tryLast children with
                                | Some lastId -> findLastValue lastId
                                | None -> nid
                            | _ -> nid
                        | None -> nid
                    match SemanticGraph.tryGetNode bodyId graph, params' with
                    | None, _ ->
                        Result.Error $"PSG settlement did not keep the body {NodeId.value bodyId} of Step lambda {NodeId.value lambdaId} resident in the graph"
                    | Some _, [] ->
                        Result.Error $"CCS source checking did not settle a state parameter for Step lambda {NodeId.value lambdaId}"
                    | Some bodyNode, (_, stateNativeType, stateParamNodeId) :: _ ->
                        // StateType from the first parameter, narrowed via coeffect
                        // (authoritative widths from interval analysis Phase 5C feedback)
                        let stateType = narrowType ctx.Coeffects graph stateParamNodeId (mapType stateNativeType ctx)
                        let narrowedRetType = narrowType ctx.Coeffects graph (findLastValue bodyId) (mapType bodyNode.Type ctx)
                        // If step returns (State, Output), the body type maps to TStruct [Item1; Item2]
                        let outputType =
                            match narrowedRetType with
                            | TStruct (("Item1", _) :: ("Item2", outTy) :: _, _) -> Some outTy
                            | _ -> None  // Single return type — no separate output
                        // ── Port width agreement ──
                        // The step function returns (State, Output). The return State must agree
                        // with the parameter State on field widths so hw.module ports and
                        // hw.instance operands use identical types.
                        match stateType, narrowedRetType with
                        | TStruct (paramFields, paramBytes), TStruct (("Item1", TStruct (retFields, _)) :: _, _) ->
                            if retFields.Length <> paramFields.Length then
                                Result.Error $"CCS source checking did not settle one state shape for Step lambda {NodeId.value lambdaId}: {paramFields.Length} state fields as a parameter, {retFields.Length} as returned"
                            else
                                let disagreement =
                                    List.zip paramFields retFields
                                    |> List.tryPick (fun ((name, paramFty), (_, retFty)) ->
                                        match paramFty, retFty with
                                        | TInt (IntWidth a), TInt (IntWidth b) when a > 0 && b > 0 && a <> b ->
                                            // Both read the state type's FieldRanges CCS settled; a
                                            // disagreement is a defect, never a width chosen here.
                                            Some $"CCS source checking did not settle one width for state field '{name}' of Step lambda {NodeId.value lambdaId}: {a} bits as a parameter, {b} bits as returned; the graph's FieldRanges must give one width"
                                        | _ -> None)
                                match disagreement with
                                | Some reason -> Result.Error reason
                                | None -> Result.Ok (TStruct (paramFields, paramBytes), inputType, outputType)
                        | _ -> Result.Ok (stateType, inputType, outputType)
                | _ -> Result.Error $"PSG settlement did not bind Step {stepId} to a lambda: its declaration's value {NodeId.value lambdaId} is not a Lambda"
            | children -> Result.Error $"PSG settlement did not bind Step {stepId} to a single-valued declaration: declaration {NodeId.value defId} has {children.Length} children"
        | None -> Result.Error $"PSG settlement did not keep the Step declaration {NodeId.value defId} resident in the graph"
    | _ -> Result.Error $"PSG settlement did not resolve Step {stepId} to a reference to its declaration"

/// Determine qualified module name for the HardwareModule Binding
/// A top-level or non-module parent leaves the bare name; a dangling parent is a graph defect.
let private qualifiedBindingName (graph: SemanticGraph) (node: SemanticNode) (bindingName: string) : Result<string, string> =
    match node.Parent with
    | Some parentId ->
        match SemanticGraph.tryGetNode parentId graph with
        | Some parentNode ->
            match parentNode.Kind with
            | SemanticKind.ModuleDef (moduleName, _) ->
                Result.Ok (sprintf "%s.%s" moduleName bindingName)
            | _ -> Result.Ok bindingName
        | None ->
            Result.Error $"PSG settlement did not keep parent {NodeId.value parentId} of HardwareModule binding {NodeId.value node.Id} resident in the graph; its module name has no source"
    | None -> Result.Ok bindingName

/// Mark all nodes in a subtree as visited (child edges only).
/// Used on Design children — safe because Step VarRef's target is NOT a child.
let rec private markSubtreeVisited (graph: SemanticGraph) (nodeId: NodeId) (visited: ref<Set<NodeId>>) : unit =
    if not (Set.contains nodeId !visited) then
        visited := Set.add nodeId !visited
        match SemanticGraph.tryGetNode nodeId graph with
        | Some node ->
            for childId in node.Children do
                markSubtreeVisited graph childId visited
        | None -> ()

/// Mark a metadata chain as visited, following VarRef binding targets transitively.
/// Platform quotations (Clock → Endpoints.clock → Pins.sysClk → ClockEndpoint record)
/// are compile-time data read structurally by the witness — they don't produce MLIR ops,
/// but coverage validation needs to know they were consumed.
let rec private markMetadataChainVisited (graph: SemanticGraph) (nodeId: NodeId) (visited: ref<Set<NodeId>>) : unit =
    if not (Set.contains nodeId !visited) then
        visited := Set.add nodeId !visited
        match SemanticGraph.tryGetNode nodeId graph with
        | Some node ->
            for childId in node.Children do
                markMetadataChainVisited graph childId visited
            // Follow VarRef binding targets (platform tier quotation chain)
            match node.Kind with
            | SemanticKind.VarRef (_, Some bindingId) ->
                markMetadataChainVisited graph bindingId visited
            | _ -> ()
        | None -> ()

/// Resolve Step VarRef to its binding target node (for walking the Lambda body)
let private resolveStepBindingTarget (graph: SemanticGraph) (stepNodeId: NodeId) : SemanticNode option =
    match SemanticGraph.tryGetNode stepNodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.VarRef (_, Some defId) ->
            SemanticGraph.tryGetNode defId graph
        | _ -> None
    | None -> None

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

let private witnessHardwareModule
    (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput))
    (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.Binding (name, _, _, Some DeclRoot.HardwareModule) ->
        // Mark all child nodes as visited (compile-time metadata, not runtime ops)
        for childId in node.Children do
            markSubtreeVisited ctx.Graph childId ctx.GlobalVisited

        // Mark Clock metadata chain as consumed (platform quotation).
        // Clock → Endpoints.clock → Pins.sysClk → ClockEndpoint { Name, FrequencyHz, PackagePin, ... }
        // This chain is compile-time board metadata read structurally, not walked for MLIR ops.
        // Must follow VarRef targets to reach the full platform binding chain.
        // Called on the binding target (not the VarRef node, which markSubtreeVisited already covered).
        match extractDesignFields ctx.Graph node with
        | Some designFields ->
            match findDesignField "Clock" designFields with
            | Some clockNodeId ->
                match SemanticGraph.tryGetNode clockNodeId ctx.Graph with
                | Some { Kind = SemanticKind.VarRef (_, Some clockBindingId) } ->
                    markMetadataChainVisited ctx.Graph clockBindingId ctx.GlobalVisited
                | _ -> ()
            | None -> ()
        | None -> ()

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

        // ── 3. Walk the Step function Lambda via combinator ──
        // This triggers LambdaWitness to generate hw.module for the step function
        // and transitively the function declarations called from the step body.
        let combinator = getCombinator()
        match resolveStepBindingTarget ctx.Graph stepNodeId with
        | None ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "HardwareModule") (Some "Step reference")
                $"PSG settlement did not resolve Step {NodeId.value stepNodeId} of HardwareModule '{name}' to a resident declaration; its step function cannot be witnessed"
        | Some stepBindingNode ->
        // Step resolves a reference to a separate declaration. Enter that
        // reference boundary explicitly; it is not a child of this binding.
        match Alex.Traversal.PSGZipper.create ctx.Graph stepBindingNode.Id with
        | None ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "HardwareModule") (Some "Step reference")
                $"PSG settlement did not place the resolved Step declaration {NodeId.value stepBindingNode.Id} of HardwareModule '{name}' in the current graph"
        | Some stepZipper ->
        visitAllNodes combinator { ctx with Zipper = stepZipper } stepBindingNode ctx.TraversalVisited

        // ── 4. Extract InitialState reset values (preserves NativeLiteral width) ──
        match extractInitialStateValues ctx.Graph initNodeId with
        | None ->
            WitnessOutput.error $"HardwareModule '{name}': InitialState fields must be integer/bool literals"
        | Some resetValues ->

        // ── 5. Resolve Step function name ──
        match resolveStepFunctionName ctx.Graph stepNodeId with
        | None ->
            WitnessOutput.error $"HardwareModule '{name}': 'Step' field must be a VarRef to a function"
        | Some stepFuncName ->

        // ── 6. Extract State/Input/Output types from step Lambda ──
        // State type comes from step Lambda's first parameter (authoritative widths
        // from Phase 5C feedback), never from the init node (which has literal-value widths).
        match extractStepTypes ctx.Graph stepNodeId ctx with
        | Result.Error reason ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "HardwareModule") (Some "Step signature")
                $"HardwareModule '{name}': {reason}"
        | Result.Ok (stateType, inputType, outputType) ->

        // ── 7. Qualified module name ──
        match qualifiedBindingName ctx.Graph node name with
        | Result.Error reason ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "HardwareModule") (Some "module name") reason
        | Result.Ok moduleName ->

            // Reset values pair with state fields by position; the pairing must name the same fields.
            let resetAgrees (stateFields: (string * MLIRType) list) =
                stateFields.Length = resetValues.Length
                && List.forall2 (fun (fieldName: string, _) (resetName: string, _) -> fieldName = resetName) stateFields resetValues

            // Verify state type is TStruct
            match stateType with
            | TStruct (stateFields, _) when not (resetAgrees stateFields) ->
                WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "HardwareModule") (Some "InitialState")
                    (sprintf "CCS source checking did not settle InitialState of HardwareModule '%s' field-for-field with its state type: state fields [%s], reset values [%s]"
                        name (stateFields |> List.map fst |> String.concat "; ") (resetValues |> List.map fst |> String.concat "; "))
            | TStruct (stateFields, _) ->
                // Match state fields with reset values (NativeLiteral preserved)
                let stateFieldInfo =
                    List.zip stateFields resetValues
                    |> List.map (fun ((fieldName, fieldTy), (_, resetLit)) ->
                        (fieldName, fieldTy, resetLit))

                let narrowedInputType = inputType
                // The output type was narrowed through the step body's type: every nested
                // record field at its FieldRanges width (a field nothing constructs: one bit).
                let narrowedOutputType = outputType

                // ── 8. Build Mealy machine hw.module ──
                let info : MealyMachineInfo = {
                    ModuleName = moduleName
                    StepFunctionName = stepFuncName
                    StateType = stateType
                    StateFields = stateFieldInfo
                    InputType = narrowedInputType
                    OutputType = narrowedOutputType
                }

                // The module body's values, named from this binding; the pin facts are the graph's
                let pinMapping = ctx.Graph.Codata.Value.Pins
                let layout = deriveLayout node.Id info pinMapping
                begin
                    let hwModuleOp =
                        match pinMapping with
                        | Some pinMapping ->
                            buildFlatPortMealyModule info pinMapping pinMapping.FieldPinAttrs layout
                        | None ->
                            buildMealyMachineModule info layout

                    // Add hw.module to root scope (top-level declaration)
                    EmissionCorrespondence.record ctx [hwModuleOp]
                    let updatedRootScope = ScopeContext.addOp hwModuleOp !ctx.RootScopeContext
                    ctx.RootScopeContext := updatedRootScope

                    // HardwareModule binding is structural — no inline ops, no value
                    { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                end

            | _ ->
                WitnessOutput.error $"HardwareModule '{name}': State type must be a record (TStruct), got {stateType}"

    | _ -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Create HardwareModule nanopass with combinator for recursive sub-graph traversal.
/// FPGA-only: should be conditionally registered in WitnessRegistry.
let createNanopass (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) : Nanopass =
    {
        Name = "HardwareModule"
        Witness = witnessHardwareModule getCombinator
    }
