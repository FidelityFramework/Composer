/// CoverageValidation - Detect unwitnessed reachable PSG nodes
///
/// After two-phase nanopass execution, validates that all reachable nodes were witnessed.
/// Any reachable node that wasn't visited represents a gap in witness coverage - a compiler bug.
///
/// This validation ensures no PSG nodes "fall through" silently without MLIR generation.
module Alex.Traversal.CoverageValidation

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Traversal.TransferTypes

// ═══════════════════════════════════════════════════════════
// COVERAGE VALIDATION
// ═══════════════════════════════════════════════════════════

/// Validate that all reachable nodes were witnessed
/// Returns list of diagnostics for any unwitnessed reachable nodes
let validateCoverage
    (graph: SemanticGraph)
    (allVisited: Set<NodeId>)  // Merged visited set from all nanopasses
    : Diagnostic list =

    // Get all reachable nodes from the graph
    let reachableNodes =
        graph.Nodes
        |> Map.toSeq
        |> Seq.map snd  // Extract SemanticNode from (NodeId, SemanticNode) pairs
        |> Seq.filter (fun node -> node.IsReachable)
        |> Seq.toList

    // Find unwitnessed nodes (reachable but not visited)
    let unwitnessedNodes =
        reachableNodes
        |> List.filter (fun node ->
            not (Set.contains node.Id allVisited))

    // Generate error diagnostics for each unwitnessed node
    unwitnessedNodes
    |> List.map (fun node ->
        // Extract first line of Kind for readable error message
        let kindSummary =
            match node.Kind.ToString().Split('\n') with
            | lines when lines.Length > 0 -> lines.[0]
            | _ -> node.Kind.ToString()

        Diagnostic.error
            (Some node.Id)
            (Some "CoverageValidation")
            (Some "Unwitnessed reachable node")
            (sprintf "PSG node '%s' (ID %d) is reachable but no witness handles it. This is a compiler bug - a witness should be implemented for this node kind." kindSummary (NodeId.value node.Id)))

// ═══════════════════════════════════════════════════════════
// COVERAGE STATISTICS
// ═══════════════════════════════════════════════════════════

/// Compute coverage statistics for reporting
type CoverageStats = {
    TotalNodes: int
    ReachableNodes: int
    WitnessedNodes: int
    UnwitnessedNodes: int
    CoveragePercentage: float
}

/// Calculate coverage statistics from graph and merged visited set
let calculateStats (graph: SemanticGraph) (allVisited: Set<NodeId>) : CoverageStats =
    let totalNodes = Map.count graph.Nodes
    let reachableNodes =
        graph.Nodes
        |> Map.toSeq
        |> Seq.map snd
        |> Seq.filter (fun n -> n.IsReachable)
        |> Seq.length
    let witnessedNodes = Set.count allVisited
    let unwitnessedNodes = reachableNodes - witnessedNodes
    let coveragePercentage =
        if reachableNodes > 0 then
            (float witnessedNodes / float reachableNodes) * 100.0
        else
            0.0

    {
        TotalNodes = totalNodes
        ReachableNodes = reachableNodes
        WitnessedNodes = witnessedNodes
        UnwitnessedNodes = unwitnessedNodes
        CoveragePercentage = coveragePercentage
    }

/// Format coverage stats for logging
let formatStats (stats: CoverageStats) : string =
    sprintf "Coverage: %d/%d witnessed (%.1f%%), %d unwitnessed, %d total nodes"
        stats.WitnessedNodes
        stats.ReachableNodes
        stats.CoveragePercentage
        stats.UnwitnessedNodes
        stats.TotalNodes
