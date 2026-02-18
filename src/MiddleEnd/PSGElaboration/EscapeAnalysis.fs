/// Escape Analysis - Value lifetime and allocation strategy determination
///
/// ARCHITECTURAL FOUNDATION:
/// This module performs ONCE-per-graph analysis before transfer begins.
/// It computes:
/// - Which nodes produce values that may need heap allocation (DU constructions, string-returning calls)
/// - Whether those values escape their defining scope (via return, closure, or byref)
/// - The appropriate EscapeKind for each allocating site
///
/// PHOTOGRAPHER PRINCIPLE: Observe the structure, don't compute during transfer.
/// PULL MODEL: Patterns pull allocation decisions from pre-computed coeffects.
///
/// Design authority: "Managed Mutability" blog post — EscapeKind DU, four components.
module PSGElaboration.EscapeAnalysis

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Core

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// How a value escapes its defining scope (from "Managed Mutability" blog)
type EscapeKind =
    /// Value lifetime contained within defining function — safe for stack allocation
    | StackScoped
    /// Value captured by a closure targeting another scope
    | EscapesViaClosure of targetNode: NodeId
    /// Value returned from its defining function — requires heap allocation
    | EscapesViaReturn
    /// Value escapes via byref/address-of — requires pinned or heap allocation
    | EscapesViaByRef

/// What kind of allocating site produced this value
type AllocSiteKind =
    | DUConstruction
    | StringReturningCall

/// Result of escape analysis for a single allocating site
type AllocSiteEscapeInfo = {
    /// The node ID of the allocating site
    NodeId: NodeId
    /// Human-readable name for debugging (case name or function name)
    Name: string
    /// What kind of allocation site this is
    SiteKind: AllocSiteKind
    /// Determined escape kind
    EscapeKind: EscapeKind
    /// Reason for the decision (for debugging / -k output)
    Reason: string
}

/// Result of escape analysis for the entire semantic graph
type EscapeAnalysisResult = {
    /// Map from node ID (int) to escape kind — the primary query interface
    NodeStrategies: Map<int, EscapeKind>
    /// Detailed information for debugging (when -k flag is used)
    Details: AllocSiteEscapeInfo list
}

// Backward compatibility aliases
type AllocationStrategy = EscapeKind
let Stack = StackScoped
let Arena = EscapesViaReturn

type CallSiteEscapeInfo = AllocSiteEscapeInfo

// ═══════════════════════════════════════════════════════════════════════════
// Scope Analysis
// ═══════════════════════════════════════════════════════════════════════════

/// Find the enclosing function scope for a given node
let rec findEnclosingFunction (nodeId: NodeId) (graph: SemanticGraph) : NodeId option =
    match SemanticGraph.tryGetNode nodeId graph with
    | None -> None
    | Some node ->
        match node.Kind with
        | SemanticKind.Lambda _ -> Some nodeId
        | SemanticKind.Binding (_, _, _, _) -> Some nodeId  // Top-level let is a scope boundary
        | _ ->
            // Traverse up to parent
            node.Parent
            |> Option.bind (fun parentId -> findEnclosingFunction parentId graph)

/// Check if a node is used outside of the given scope
let escapesScope (scopeId: NodeId) (valueId: NodeId) (graph: SemanticGraph) : bool =
    // Find all uses of this value via VarRef nodes
    let useSites =
        graph.Nodes
        |> Seq.choose (fun kvp ->
            match kvp.Value.Kind with
            | SemanticKind.VarRef (_, Some defId) when defId = valueId ->
                Some kvp.Key
            | _ -> None)
        |> Seq.toList

    // Check if any use site is outside the scope
    useSites
    |> List.exists (fun useNodeId ->
        let useScope = findEnclosingFunction useNodeId graph
        match useScope with
        | None -> true  // Global scope - escapes
        | Some useScopeId -> useScopeId <> scopeId)  // Different scope - escapes

/// Check if a value is returned from its defining function
let isReturnedFromFunction (valueId: NodeId) (scopeId: NodeId) (graph: SemanticGraph) : bool =
    match SemanticGraph.tryGetNode scopeId graph with
    | None -> false
    | Some scopeNode ->
        // Check if the function body (last child) is or references this value
        match scopeNode.Children |> List.tryLast with
        | None -> false
        | Some lastChildId ->
            // Simple check: is the value the final expression?
            lastChildId = valueId ||
            // Or is it referenced in the final expression?
            (match SemanticGraph.tryGetNode lastChildId graph with
             | Some lastChild ->
                 match lastChild.Kind with
                 | SemanticKind.VarRef (_, Some defId) -> defId = valueId
                 | _ -> false
             | None -> false)

/// Check if a value transitively contributes to the function's return value
/// Traces through IfThenElse branches and Sequential nodes to the function return
let rec isTransitivelyReturned (valueId: NodeId) (scopeId: NodeId) (graph: SemanticGraph) : bool =
    isReturnedFromFunction valueId scopeId graph ||
    // Check if parent node is returned (IfThenElse/Sequential forwarding)
    match SemanticGraph.tryGetNode valueId graph with
    | Some node ->
        node.Parent
        |> Option.map (fun parentId -> isTransitivelyReturned parentId scopeId graph)
        |> Option.defaultValue false
    | None -> false

// ═══════════════════════════════════════════════════════════════════════════
// Allocating Site Detection
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a type is a string type
let isStringType (ty: NativeType) : bool =
    match ty with
    | NativeType.TApp (tycon, _) when tycon.NTUKind = Some NTUKind.NTUstring -> true
    | _ -> false

/// Find all nodes that produce stack-allocatable values
/// This replaces the narrow `findStringReturningCalls` with a generalized version
let findAllocatingSites (graph: SemanticGraph) : (NodeId * string * AllocSiteKind) list =
    graph.Nodes
    |> Seq.choose (fun kvp ->
        let nodeId = kvp.Key
        match kvp.Value.Kind with
        | SemanticKind.DUConstruct (caseName, _, _, _) ->
            Some (nodeId, caseName, DUConstruction)
        | SemanticKind.Application (funcId, _) when isStringType kvp.Value.Type ->
            // Try to get function name from the function node
            let funcName =
                match SemanticGraph.tryGetNode funcId graph with
                | Some funcNode ->
                    match funcNode.Kind with
                    | SemanticKind.VarRef (name, _) -> name
                    | SemanticKind.FieldGet (_, fieldName) -> fieldName
                    | _ -> "<anonymous>"
                | None -> "<unknown>"
            Some (nodeId, funcName, StringReturningCall)
        | _ -> None)
    |> Seq.toList

// ═══════════════════════════════════════════════════════════════════════════
// Escape Analysis Algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Analyze a single allocating site to determine its escape kind
let analyzeAllocSite (siteId: NodeId) (name: string) (siteKind: AllocSiteKind) (graph: SemanticGraph) : AllocSiteEscapeInfo =
    // For DU constructions, the value IS the node itself (no binding indirection needed)
    // For string-returning calls, trace through binding to find scope
    let (valueId, valueName) =
        match siteKind with
        | DUConstruction ->
            // DU construction IS the value — check if THIS node escapes
            (siteId, name)
        | StringReturningCall ->
            // String-returning call — check binding (original behavior)
            let binding =
                graph.Nodes
                |> Seq.tryPick (fun kvp ->
                    match kvp.Value.Kind with
                    | SemanticKind.Binding (bname, _, _, _) ->
                        match kvp.Value.Children with
                        | valueChildId :: _ when valueChildId = siteId ->
                            Some (kvp.Key, bname)
                        | _ -> None
                    | _ -> None)
            match binding with
            | Some (bindingId, bname) -> (bindingId, bname)
            | None -> (siteId, name)  // Not bound — immediate use, stack OK

    // Find the enclosing function scope
    match findEnclosingFunction valueId graph with
    | None ->
        // Global scope — must use heap
        { NodeId = siteId
          Name = name
          SiteKind = siteKind
          EscapeKind = EscapesViaReturn
          Reason = "Global scope - no stack frame to allocate in" }

    | Some scopeId ->
        // Check escape paths in order of specificity
        if escapesScope scopeId valueId graph then
            { NodeId = siteId
              Name = name
              SiteKind = siteKind
              EscapeKind = EscapesViaReturn
              Reason = sprintf "Value '%s' used outside defining function scope" valueName }
        elif isTransitivelyReturned valueId scopeId graph then
            { NodeId = siteId
              Name = name
              SiteKind = siteKind
              EscapeKind = EscapesViaReturn
              Reason = sprintf "Value '%s' returned from function (transitive)" valueName }
        else
            { NodeId = siteId
              Name = name
              SiteKind = siteKind
              EscapeKind = StackScoped
              Reason = sprintf "Value '%s' lifetime contained within function scope" valueName }

/// Perform escape analysis on the entire semantic graph
let analyzeGraph (graph: SemanticGraph) : EscapeAnalysisResult =
    let allocSites = findAllocatingSites graph

    let details =
        allocSites
        |> List.map (fun (siteId, name, siteKind) ->
            analyzeAllocSite siteId name siteKind graph)

    let strategies =
        details
        |> List.map (fun info -> (NodeId.value info.NodeId, info.EscapeKind))
        |> Map.ofList

    { NodeStrategies = strategies
      Details = details }

// ═══════════════════════════════════════════════════════════════════════════
// Query Interface (for Alex witnesses)
// ═══════════════════════════════════════════════════════════════════════════

/// Get the escape kind for a specific node
let getEscapeKind (nodeId: NodeId) (result: EscapeAnalysisResult) : EscapeKind option =
    Map.tryFind (NodeId.value nodeId) result.NodeStrategies

/// Get the escape kind with default fallback (StackScoped)
let getEscapeKindOrDefault (nodeId: NodeId) (result: EscapeAnalysisResult) : EscapeKind =
    getEscapeKind nodeId result
    |> Option.defaultValue StackScoped  // Conservative default: stack allocation

// Backward compatibility
let getAllocationStrategy = getEscapeKind
let getAllocationStrategyOrDefault = getEscapeKindOrDefault
