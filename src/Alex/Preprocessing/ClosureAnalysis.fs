/// Closure Analysis - Pre-transfer analysis for function references
///
/// ARCHITECTURAL FOUNDATION:
/// This module performs ONCE-per-graph analysis before transfer begins.
/// It determines which VarRefs need closure wrapping (value position) vs
/// which can be direct calls (func position).
///
/// The MLKit-style flat closure model requires:
/// - Functions passed as values: wrap in closure [code_ptr | captures...]
/// - Functions called directly: no closure needed, just emit call
///
/// This eliminates on-demand analysis during transfer, adhering to
/// the photographer principle: observe the structure, don't compute during transfer.
module Alex.Preprocessing.ClosureAnalysis

open FSharp.Native.Compiler.Checking.Native.SemanticGraph

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// Context in which a VarRef appears
type VarRefContext =
    /// VarRef is in function position of an Application - can be direct call
    | FuncPosition of appNodeId: NodeId
    /// VarRef is in value position - needs closure wrapping when referencing a function
    | ValuePosition

/// Information about a VarRef that points to a Lambda/function
type FunctionRef = {
    /// The VarRef NodeId
    VarRefId: int
    /// The name of the referenced variable
    Name: string
    /// The target Lambda's definition NodeId (via Binding)
    TargetDefId: int
    /// The context (func vs value position)
    Context: VarRefContext
}

/// Result of closure analysis for a semantic graph
type ClosureAnalysisResult = {
    /// Set of VarRef NodeIds that need closure wrapping
    /// These are VarRefs in value position that point to Lambdas
    NeedsClosureWrapping: Set<int>

    /// Set of VarRef NodeIds in func position (direct call candidates)
    InFuncPosition: Set<int>

    /// Map from VarRef NodeId to their target Lambda's Binding NodeId
    /// Useful for looking up function names via SSAAssignment.lookupLambdaName
    VarRefToLambdaBinding: Map<int, int>
}

module ClosureAnalysisResult =
    let empty = {
        NeedsClosureWrapping = Set.empty
        InFuncPosition = Set.empty
        VarRefToLambdaBinding = Map.empty
    }

// ═══════════════════════════════════════════════════════════════════════════
// Analysis Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a definition NodeId points to a Binding that contains a Lambda
let private isLambdaBinding (graph: SemanticGraph) (defId: NodeId) : bool =
    match SemanticGraph.tryGetNode defId graph with
    | Some defNode ->
        match defNode.Kind with
        | SemanticKind.Binding _ ->
            // Check if first child is a Lambda
            match defNode.Children with
            | childId :: _ ->
                match SemanticGraph.tryGetNode childId graph with
                | Some childNode ->
                    match childNode.Kind with
                    | SemanticKind.Lambda _ -> true
                    | _ -> false
                | None -> false
            | [] -> false
        | SemanticKind.Lambda _ ->
            // Direct Lambda reference (rare, but possible)
            true
        | _ -> false
    | None -> false

/// Build a set of all NodeIds that are in function position of Applications
/// These are the funcId values from Application (funcId, argIds) nodes
let private findFuncPositionNodes (graph: SemanticGraph) : Set<int> =
    let mutable funcPositions = Set.empty

    for KeyValue(_, node) in graph.Nodes do
        match node.Kind with
        | SemanticKind.Application (funcId, _) ->
            funcPositions <- Set.add (NodeId.value funcId) funcPositions
        | _ -> ()

    funcPositions

/// Find all VarRefs that reference Lambda bindings and classify them
let private analyzeVarRefs (graph: SemanticGraph) (funcPositions: Set<int>) : FunctionRef list =
    let mutable results = []

    for KeyValue(_, node) in graph.Nodes do
        match node.Kind with
        | SemanticKind.VarRef (name, Some defId) ->
            // Check if this VarRef points to a Lambda
            if isLambdaBinding graph defId then
                let nodeIdVal = NodeId.value node.Id
                let context =
                    if Set.contains nodeIdVal funcPositions then
                        FuncPosition node.Id
                    else
                        ValuePosition

                results <- {
                    VarRefId = nodeIdVal
                    Name = name
                    TargetDefId = NodeId.value defId
                    Context = context
                } :: results
        | _ -> ()

    results

// ═══════════════════════════════════════════════════════════════════════════
// Main Analysis Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Perform closure analysis on a semantic graph
/// This should be called ONCE before SSA assignment
let analyze (graph: SemanticGraph) : ClosureAnalysisResult =
    // Step 1: Find all nodes in function position (funcId of Application)
    let funcPositions = findFuncPositionNodes graph

    // Step 2: Find all VarRefs that reference Lambdas and classify them
    let functionRefs = analyzeVarRefs graph funcPositions

    // Step 3: Build the result sets
    let mutable needsWrapping = Set.empty
    let mutable inFuncPos = Set.empty
    let mutable varRefToBinding = Map.empty

    for ref in functionRefs do
        varRefToBinding <- Map.add ref.VarRefId ref.TargetDefId varRefToBinding

        match ref.Context with
        | FuncPosition _ ->
            inFuncPos <- Set.add ref.VarRefId inFuncPos
        | ValuePosition ->
            needsWrapping <- Set.add ref.VarRefId needsWrapping

    {
        NeedsClosureWrapping = needsWrapping
        InFuncPosition = inFuncPos
        VarRefToLambdaBinding = varRefToBinding
    }

// ═══════════════════════════════════════════════════════════════════════════
// Query Functions (for witnesses)
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a VarRef needs closure wrapping
let needsClosureWrapping (varRefNodeId: NodeId) (analysis: ClosureAnalysisResult) : bool =
    Set.contains (NodeId.value varRefNodeId) analysis.NeedsClosureWrapping

/// Check if a VarRef is in function position (can be direct call)
let isInFuncPosition (varRefNodeId: NodeId) (analysis: ClosureAnalysisResult) : bool =
    Set.contains (NodeId.value varRefNodeId) analysis.InFuncPosition

/// Check if a VarRef references a Lambda (either func or value position)
let referencesLambda (varRefNodeId: NodeId) (analysis: ClosureAnalysisResult) : bool =
    Map.containsKey (NodeId.value varRefNodeId) analysis.VarRefToLambdaBinding

/// Get the target Lambda's Binding NodeId for a VarRef
let getTargetBinding (varRefNodeId: NodeId) (analysis: ClosureAnalysisResult) : NodeId option =
    Map.tryFind (NodeId.value varRefNodeId) analysis.VarRefToLambdaBinding
    |> Option.map NodeId
