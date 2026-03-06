/// ValuePositionAnalysis - Pre-compute which VarRef nodes reference functions in value position
///
/// A VarRef to a named function binding is in VALUE position when it appears as an argument
/// to a function application (or any non-call context). In CALL position (the func child of
/// Application), no closure pair is needed — ApplicationWitness handles direct calls.
///
/// This is a COEFFECT: computed once before traversal, observed passively by VarRefWitness.
/// The zipper witnesses, it does not decide. (SpeakEZ: "Learning to Walk")
///
/// The analysis mirrors Baker's isInValuePosition check for Lambdas, extended to VarRefs.
module PSGElaboration.ValuePositionAnalysis

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes

/// Result of value-position analysis
type ValuePositionResult = {
    /// VarRef NodeIds that reference function bindings and appear in value position.
    /// These need closure pair construction (thunk wrapper + memref<2xindex>).
    FunctionVarRefsInValuePosition: Set<NodeId>
}

/// Check if a binding node's first child is a Lambda (function binding)
let private isFunctionBinding (bindingId: NodeId) (graph: SemanticGraph) : bool =
    match SemanticGraph.tryGetNode bindingId graph with
    | Some bindingNode ->
        bindingNode.Children
        |> List.tryHead
        |> Option.bind (fun childId -> SemanticGraph.tryGetNode childId graph)
        |> Option.map (fun childNode -> childNode.Kind.ToString().StartsWith("Lambda"))
        |> Option.defaultValue false
    | None -> false

/// Determine if a node is in call position (the func child of an Application).
/// Looks through TypeAnnotation wrappers (CCS inserts these between Application and VarRef).
let private isInCallPosition (nodeId: NodeId) (node: SemanticNode) (graph: SemanticGraph) : bool =
    match node.Parent with
    | Some parentId ->
        match SemanticGraph.tryGetNode parentId graph with
        | Some parentNode ->
            match parentNode.Kind with
            | SemanticKind.Application (funcId, _) ->
                // This node IS the func child of Application → call position
                funcId = nodeId
            | SemanticKind.TypeAnnotation _ ->
                // VarRef wrapped in TypeAnnotation — check grandparent
                match parentNode.Parent with
                | Some grandParentId ->
                    match SemanticGraph.tryGetNode grandParentId graph with
                    | Some grandParentNode ->
                        match grandParentNode.Kind with
                        | SemanticKind.Application (funcId, _) ->
                            // The TypeAnnotation (our parent) is the func child
                            funcId = parentId
                        | _ -> false
                    | None -> false
                | None -> false
            | _ -> false
        | None -> false
    | None -> false

/// Analyze the PSG to find all VarRef nodes that reference function bindings
/// and appear in value position (need closure pair construction).
let analyze (graph: SemanticGraph) : ValuePositionResult =
    let mutable valuePositionRefs = Set.empty

    for kvp in graph.Nodes do
        let node = kvp.Value
        match node.Kind with
        | SemanticKind.VarRef (_, Some bindingId) ->
            // Only consider VarRefs to Binding nodes (not PatternBinding, etc.)
            match SemanticGraph.tryGetNode bindingId graph with
            | Some bindingNode ->
                match bindingNode.Kind with
                | SemanticKind.Binding _ when isFunctionBinding bindingId graph ->
                    // Function binding VarRef — check if NOT in call position
                    if not (isInCallPosition node.Id node graph) then
                        valuePositionRefs <- Set.add node.Id valuePositionRefs
                | _ -> ()
            | None -> ()
        | _ -> ()

    { FunctionVarRefsInValuePosition = valuePositionRefs }
