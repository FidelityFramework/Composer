/// SaturateApplications - Nanopass to reduce pipe operators and flatten curried applications
///
/// ARCHITECTURAL FOUNDATION:
/// F# uses curried application semantics and pipe operators. The FNCS SemanticGraph
/// preserves these structures. This nanopass transforms them to flat, saturated
/// applications that Alex can directly emit as MLIR calls.
///
/// TRANSFORMATIONS:
/// 1. Pipe reduction: App(App(|>, x), f) → App(f, x)
/// 2. Curried flattening: App(App(f, a), b) → App(f, [a, b])
///
/// This runs ONCE before transfer, adhering to the photographer principle:
/// observe and transform structure upfront, don't compute during transfer.
///
/// Reference: Nanopass Framework (Sarkar, Waddell, Dybvig, Keep)
module Alex.Preprocessing.SaturateApplications

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes

// ═══════════════════════════════════════════════════════════════════════════
// Pipe Operator Detection
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a VarRef name is a forward pipe operator (|>)
let private isForwardPipeName (name: string) : bool =
    name = "op_PipeRight" || name.Contains("|>")

/// Check if a VarRef name is a backward pipe operator (<|)
let private isBackwardPipeName (name: string) : bool =
    name = "op_PipeLeft" || name.Contains("<|")

/// Check if a node is a forward pipe operator reference
let private isForwardPipeRef (graph: SemanticGraph) (nodeId: NodeId) : bool =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.VarRef (name, _) -> isForwardPipeName name
        | _ -> false
    | None -> false

/// Check if a node is a backward pipe operator reference
let private isBackwardPipeRef (graph: SemanticGraph) (nodeId: NodeId) : bool =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.VarRef (name, _) -> isBackwardPipeName name
        | _ -> false
    | None -> false

// ═══════════════════════════════════════════════════════════════════════════
// Application Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a node is an Application
let private isApplication (graph: SemanticGraph) (nodeId: NodeId) : bool =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Application _ -> true
        | _ -> false
    | None -> false

/// Get the function and args from an Application node
let private getApplicationParts (graph: SemanticGraph) (nodeId: NodeId) : (NodeId * NodeId list) option =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Application (funcId, argIds) -> Some (funcId, argIds)
        | _ -> None
    | None -> None

// ═══════════════════════════════════════════════════════════════════════════
// Pipe Reduction
// ═══════════════════════════════════════════════════════════════════════════

/// Detect and reduce pipe operator pattern.
/// Pattern: App(App(|>, x), f) → App(f, [x])
/// Pattern: App(App(<|, f), x) → App(f, [x])
///
/// Returns Some (newFuncId, newArgIds) if this is a pipe pattern, None otherwise.
let private tryReducePipe (graph: SemanticGraph) (funcId: NodeId) (argIds: NodeId list) : (NodeId * NodeId list) option =
    // Check if funcId is itself an Application (curried pipe)
    match getApplicationParts graph funcId with
    | Some (innerFuncId, innerArgIds) ->
        // Check if innerFuncId is a pipe operator
        if isForwardPipeRef graph innerFuncId then
            // Forward pipe: App(App(|>, x), f)
            // innerArgIds = [x], argIds = [f]
            match innerArgIds, argIds with
            | [valueId], [funcRefId] ->
                // Transform to App(f, [x])
                Some (funcRefId, [valueId])
            | _ -> None
        elif isBackwardPipeRef graph innerFuncId then
            // Backward pipe: App(App(<|, f), x)
            // innerArgIds = [f], argIds = [x]
            match innerArgIds, argIds with
            | [funcRefId], [valueId] ->
                // Transform to App(f, [x])
                Some (funcRefId, [valueId])
            | _ -> None
        else
            None
    | None -> None

// ═══════════════════════════════════════════════════════════════════════════
// Curried Application Flattening
// ═══════════════════════════════════════════════════════════════════════════

/// Recursively collect all arguments from a chain of curried applications.
/// Returns (rootFuncId, allArgIds) where allArgIds are in application order.
///
/// For ((f a) b) c:
///   - rootFuncId = f
///   - allArgIds = [a, b, c]
let rec private collectCurriedArgs (graph: SemanticGraph) (funcId: NodeId) (argIds: NodeId list) : NodeId * NodeId list =
    match getApplicationParts graph funcId with
    | Some (innerFuncId, innerArgIds) ->
        // funcId is an Application - recurse to collect inner args
        let (rootFunc, innerCollectedArgs) = collectCurriedArgs graph innerFuncId innerArgIds
        (rootFunc, innerCollectedArgs @ argIds)
    | None ->
        // funcId is not an Application - it's the root function
        (funcId, argIds)

// ═══════════════════════════════════════════════════════════════════════════
// Node Transformation
// ═══════════════════════════════════════════════════════════════════════════

/// Transform a single Application node.
/// 1. First try pipe reduction
/// 2. Then flatten any remaining curried structure
let private transformApplication (graph: SemanticGraph) (node: SemanticNode) : SemanticNode =
    match node.Kind with
    | SemanticKind.Application (funcId, argIds) ->
        // Step 1: Try pipe reduction
        let (effectiveFuncId, effectiveArgIds) =
            match tryReducePipe graph funcId argIds with
            | Some (newFuncId, newArgIds) -> (newFuncId, newArgIds)
            | None -> (funcId, argIds)

        // Step 2: Flatten any remaining curried structure
        let (rootFuncId, allArgIds) = collectCurriedArgs graph effectiveFuncId effectiveArgIds

        // Only update if we actually changed something
        if rootFuncId = funcId && allArgIds = argIds then
            node
        else
            { node with
                Kind = SemanticKind.Application(rootFuncId, allArgIds)
                Children = rootFuncId :: allArgIds }
    | _ -> node

// ═══════════════════════════════════════════════════════════════════════════
// Main Nanopass Entry Point
// ═══════════════════════════════════════════════════════════════════════════

/// Saturate all applications in the semantic graph.
/// This transforms pipe operators and curried applications into flat,
/// direct function applications.
///
/// Returns the transformed graph.
let saturateApplications (graph: SemanticGraph) : SemanticGraph =
    // Transform all Application nodes
    let transformedNodes =
        graph.Nodes
        |> Map.map (fun _nodeId node ->
            match node.Kind with
            | SemanticKind.Application _ -> transformApplication graph node
            | _ -> node)

    { graph with Nodes = transformedNodes }
