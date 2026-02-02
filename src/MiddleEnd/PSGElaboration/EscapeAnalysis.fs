/// Escape Analysis - String lifetime and allocation strategy determination
///
/// ARCHITECTURAL FOUNDATION:
/// This module performs ONCE-per-graph analysis before transfer begins.
/// It computes:
/// - Which string-returning function calls have results that escape their defining scope
/// - The appropriate allocation strategy (Stack vs Arena) for each call site
///
/// This eliminates the need for multiple Console APIs (readln vs readlnFrom).
/// The compiler automatically determines allocation strategy based on escape analysis.
///
/// PHOTOGRAPHER PRINCIPLE: Observe the structure, don't compute during transfer.
module PSGElaboration.EscapeAnalysis

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core

// ═══════════════════════════════════════════════════════════════════════════
// Types
// ═══════════════════════════════════════════════════════════════════════════

/// Allocation strategy for string-returning operations
type AllocationStrategy =
    /// Stack allocation via inline - string lifetime contained within function scope
    | Stack
    /// Arena allocation - string escapes defining scope
    | Arena

/// Result of escape analysis for a single call site
type CallSiteEscapeInfo = {
    /// The Application node ID
    CallSiteId: NodeId
    /// The function being called (e.g., "Console.readln")
    FunctionName: string
    /// The result binding ID (where the string is bound)
    ResultBindingId: NodeId option
    /// Determined allocation strategy
    Strategy: AllocationStrategy
    /// Reason for the decision (for debugging)
    Reason: string
}

/// Result of escape analysis for the entire semantic graph
type EscapeAnalysisResult = {
    /// Map from call site NodeId to allocation strategy
    CallSiteStrategies: Map<int, AllocationStrategy>
    /// Detailed information for debugging (when -k flag is used)
    Details: CallSiteEscapeInfo list
}

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

// ═══════════════════════════════════════════════════════════════════════════
// String-Returning Call Detection
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a type is a string type
let isStringType (ty: NativeType) : bool =
    match ty with
    | NativeType.TApp (tycon, _) when tycon.NTUKind = Some NTUKind.NTUstring -> true
    | _ -> false

/// Find all Application nodes that return strings
let findStringReturningCalls (graph: SemanticGraph) : (NodeId * string) list =
    graph.Nodes
    |> Seq.choose (fun kvp ->
        let nodeId = kvp.Key
        let ty = kvp.Value.Type
        match kvp.Value.Kind with
        | SemanticKind.Application (funcId, _) when isStringType ty ->
            // Try to get function name from the function node
            match SemanticGraph.tryGetNode funcId graph with
            | Some funcNode ->
                match funcNode.Kind with
                | SemanticKind.VarRef (name, _) ->
                    Some (nodeId, name)
                | SemanticKind.FieldGet (_, fieldName) ->
                    Some (nodeId, fieldName)
                | _ -> Some (nodeId, "<anonymous>")
            | None -> Some (nodeId, "<unknown>")
        | _ -> None)
    |> Seq.toList

// ═══════════════════════════════════════════════════════════════════════════
// Escape Analysis Algorithm
// ═══════════════════════════════════════════════════════════════════════════

/// Analyze a single call site to determine allocation strategy
let analyzeCallSite (callSiteId: NodeId) (functionName: string) (graph: SemanticGraph) : CallSiteEscapeInfo =
    // Find the binding that captures this call's result
    let resultBinding =
        graph.Nodes
        |> Seq.tryPick (fun kvp ->
            match kvp.Value.Kind with
            | SemanticKind.Binding (name, _, _, _) ->
                // Check if the first child (value) is the call site we're analyzing
                match kvp.Value.Children with
                | valueId :: _ when valueId = callSiteId ->
                    Some (kvp.Key, name)
                | _ -> None
            | _ -> None)

    match resultBinding with
    | None ->
        // Call result not bound to a variable - must be used immediately (stack OK)
        { CallSiteId = callSiteId
          FunctionName = functionName
          ResultBindingId = None
          Strategy = Stack
          Reason = "Result not bound to variable - immediate use only" }

    | Some (bindingId, bindingName) ->
        // Find the enclosing function scope
        match findEnclosingFunction bindingId graph with
        | None ->
            // Global scope - strings must use arena
            { CallSiteId = callSiteId
              FunctionName = functionName
              ResultBindingId = Some bindingId
              Strategy = Arena
              Reason = "Global scope - no stack frame to allocate in" }

        | Some scopeId ->
            // Check if the value escapes this scope
            if escapesScope scopeId bindingId graph then
                { CallSiteId = callSiteId
                  FunctionName = functionName
                  ResultBindingId = Some bindingId
                  Strategy = Arena
                  Reason = sprintf "Value '%s' used outside defining function scope" bindingName }
            elif isReturnedFromFunction bindingId scopeId graph then
                { CallSiteId = callSiteId
                  FunctionName = functionName
                  ResultBindingId = Some bindingId
                  Strategy = Arena
                  Reason = sprintf "Value '%s' returned from function" bindingName }
            else
                { CallSiteId = callSiteId
                  FunctionName = functionName
                  ResultBindingId = Some bindingId
                  Strategy = Stack
                  Reason = sprintf "Value '%s' lifetime contained within function scope" bindingName }

/// Perform escape analysis on the entire semantic graph
let analyzeGraph (graph: SemanticGraph) : EscapeAnalysisResult =
    let stringCalls = findStringReturningCalls graph

    let details =
        stringCalls
        |> List.map (fun (callSiteId, funcName) ->
            analyzeCallSite callSiteId funcName graph)

    let strategies =
        details
        |> List.map (fun info -> (NodeId.value info.CallSiteId, info.Strategy))
        |> Map.ofList

    { CallSiteStrategies = strategies
      Details = details }

// ═══════════════════════════════════════════════════════════════════════════
// Query Interface (for Alex witnesses)
// ═══════════════════════════════════════════════════════════════════════════

/// Get the allocation strategy for a specific call site
let getAllocationStrategy (callSiteId: NodeId) (result: EscapeAnalysisResult) : AllocationStrategy option =
    Map.tryFind (NodeId.value callSiteId) result.CallSiteStrategies

/// Get the allocation strategy with default fallback (Stack)
let getAllocationStrategyOrDefault (callSiteId: NodeId) (result: EscapeAnalysisResult) : AllocationStrategy =
    getAllocationStrategy callSiteId result
    |> Option.defaultValue Stack  // Conservative default: stack allocation
