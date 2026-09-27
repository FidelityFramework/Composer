/// CoverageValidation - Detect unwitnessed reachable PSG nodes
///
/// After the shared traversal, validate all source-authorized executable and
/// declaration-scope occurrences. Source-published proof-only nodes do not execute.
///
/// This validation ensures no PSG nodes "fall through" silently without MLIR generation.
module Alex.Traversal.CoverageValidation

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Traversal.TransferTypes

// ═══════════════════════════════════════════════════════════
// COVERAGE VALIDATION
// ═══════════════════════════════════════════════════════════

/// Source publication owns both executable demand and declaration placement.
let private sourceOccurrences (graph: SemanticGraph) =
    Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryRead graph
    |> Result.map (fun projection ->
        let proofOnly = Set.union projection.Ordinary.DeferredOnly projection.Boundary.DeclarationOnly
        graph.Nodes.Values
        |> Seq.filter (fun node ->
            (node.IsReachable || projection.Boundary.ByScope.ContainsKey node.Id)
            && not (proofOnly.Contains node.Id))
        |> Seq.map _.Id
        |> Set.ofSeq)

/// Report any required source occurrence missing from the shared traversal.
let private validateCoverageWith
    (expected: Set<NodeId>)
    (graph: SemanticGraph)
    (allVisited: Set<NodeId>)  // Merged visited set from all nanopasses
    : Diagnostic list =

    let unwitnessedNodes =
        Set.difference expected allVisited
        |> Set.toList
        |> List.map (fun id -> graph.Nodes[id])

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
            (Some "Unwitnessed source occurrence")
            (sprintf "Alex traversal did not witness required PSG occurrence '%s' (ID %d): no declaration root or structural parent placed it, or no witness claims its kind." kindSummary (NodeId.value node.Id)))

let validateCoverage (graph: SemanticGraph) (allVisited: Set<NodeId>) : Diagnostic list =
    match sourceOccurrences graph with
    | Result.Ok expected -> validateCoverageWith expected graph allVisited
    | Result.Error reason ->
        [Diagnostic.error None (Some "CoverageValidation") (Some "source witness projection") reason]

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
    let expected =
        match sourceOccurrences graph with
        | Result.Ok expected -> expected
        | Result.Error reason -> invalidOp ("Coverage requires its source emission seal: " + reason)
    let totalNodes = Map.count graph.Nodes
    let reachableNodes = Set.count expected
    let witnessedNodes = Set.intersect allVisited expected |> Set.count
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
