/// TwoPhaseNanopass - Two-phase execution for content and structural nanopasses
///
/// Phase 1: ContentPhase witnesses (Literal, Arith, Collections) run in parallel
/// Phase 2: StructuralPhase witnesses (Lambda, ControlFlow) wrap content in scope structures
/// Both phases use IcedTasks for parallel execution within each phase
module Alex.Traversal.TwoPhaseNanopass

open System.IO
open System.Text.Json
open IcedTasks
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.PSGZipper
open Alex.Traversal.CoverageValidation

// ═══════════════════════════════════════════════════════════════════════════
// TWO-PHASE EXECUTION (ContentPhase → StructuralPhase)
// ═══════════════════════════════════════════════════════════════════════════

// ARCHITECTURE: Two-phase execution prevents double-witnessing in nested scopes
//
// Phase 1 (ContentPhase): Literal, Arith, Lazy, Collections witnesses
//   - Run in parallel via IcedTasks
//   - Respect scope boundaries (no recursion into Lambda/ControlFlow children)
//   - Return operations for content nodes
//
// Phase 2 (StructuralPhase): Lambda, ControlFlow witnesses
//   - Run after Phase 1 completes
//   - Use witnessSubgraph for scope bodies (reuses Phase 1 witnesses)
//   - Wrap content operations in scope structures (FuncDef, SCFOp)
//
// Rationale: Content lives inside scopes. Scopes must see content before wrapping it.

/// Combine multiple witnesses into a single witness that tries each in sequence
let private combineWitnesses (nanopasses: Nanopass list) : (WitnessContext -> SemanticNode -> WitnessOutput) =
    fun ctx node ->
        let rec tryWitnesses remaining =
            match remaining with
            | [] -> WitnessOutput.skip
            | nanopass :: rest ->
                let result = nanopass.Witness ctx node
                if result = WitnessOutput.skip then
                    tryWitnesses rest
                else
                    result
        tryWitnesses nanopasses

/// Run nanopasses with SHARED accumulator via SINGLE combined traversal
/// This ensures post-order guarantees hold across ALL witnesses, not just within each nanopass
let runNanopassesSequentialShared
    (nanopasses: Nanopass list)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    (sharedAcc: MLIRAccumulator)
    : (Nanopass * Set<FSharp.Native.Compiler.NativeTypedTree.NativeTypes.NodeId>) list =

    // Create combined witness that tries all nanopasses at each node
    let combinedWitness = combineWitnesses nanopasses

    // Single traversal with combined witness (NOT separate traversals per nanopass)
    let visited = ref Set.empty

    // Visit ALL reachable nodes in a SINGLE post-order traversal
    for kvp in graph.Nodes do
        let nodeId, node = kvp.Key, kvp.Value
        if node.IsReachable && not (Set.contains nodeId !visited) then
            match PSGZipper.create graph nodeId with
            | None -> ()
            | Some initialZipper ->
                let nodeCtx = {
                    Graph = graph
                    Coeffects = coeffects
                    Accumulator = sharedAcc
                    Zipper = initialZipper
                }
                visitAllNodes combinedWitness nodeCtx node sharedAcc visited

    // Return visited set for each nanopass (all nanopasses saw the same nodes)
    nanopasses |> List.map (fun np -> (np, !visited))

/// Run nanopasses in parallel with SHARED accumulator (future optimization)
/// Current: Sequential execution (parallel execution requires locking on mutable accumulator)
let runNanopassesParallelShared
    (nanopasses: Nanopass list)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    (sharedAcc: MLIRAccumulator)
    : (Nanopass * Set<FSharp.Native.Compiler.NativeTypedTree.NativeTypes.NodeId>) list =

    // FUTURE: Parallel execution with lock-free data structures
    // Current: Sequential for correctness
    runNanopassesSequentialShared nanopasses graph coeffects sharedAcc

// ═══════════════════════════════════════════════════════════════════════════
// REMOVED: Envelope/Merge Functions
// ═══════════════════════════════════════════════════════════════════════════

/// REMOVED: collectEnvelope and overlayAccumulators
///
/// With shared accumulator architecture, ALL nanopasses write to the SAME accumulator.
/// No merging needed - operations and bindings accumulate directly during traversal.

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ORCHESTRATION
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration for two-phase execution
type TwoPhaseConfig = {
    /// Enable parallel execution within each phase (false for debugging)
    EnableParallel: bool

    /// FUTURE: Enable discovery pass for large projects
    /// Threshold where pre-scanning node types becomes cheaper than empty traversals
    /// Current: Always false (full-fat parallel execution)
    EnableDiscovery: bool

    /// FUTURE: Node count threshold to trigger discovery
    /// Current: Not used (discovery disabled)
    DiscoveryThreshold: int
}

let defaultConfig = {
    EnableParallel = true
    EnableDiscovery = false  // Deferred optimization
    DiscoveryThreshold = 10000  // Placeholder for future tuning
}

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS RESULT SERIALIZATION (for -k flag debugging)
// ═══════════════════════════════════════════════════════════════════════════

/// Serialize a nanopass result to JSON for debugging
let private serializeNanopassResult (intermediatesDir: string) (nanopass: Nanopass) (accumulator: MLIRAccumulator) (visited: Set<NodeId>) : unit =
    let fileName = sprintf "07_%s_witness.json" (nanopass.Name.ToLower())
    let filePath = Path.Combine(intermediatesDir, fileName)

    let summary = {|
        NanopassName = nanopass.Name
        OperationCount = List.length accumulator.AllOps
        ErrorCount = List.length accumulator.Errors
        VisitedNodes = visited |> Set.toList |> List.map (fun nodeId -> NodeId.value nodeId)
        Errors = accumulator.Errors |> List.map Diagnostic.format
        Operations = accumulator.AllOps |> List.map (fun op -> sprintf "%A" op)
    |}

    let json = JsonSerializer.Serialize(summary, JsonSerializerOptions(WriteIndented = true))
    File.WriteAllText(filePath, json)
    printfn "[Alex] Wrote nanopass result: %s" fileName

/// Main entry point: Run all nanopasses in two phases with SHARED accumulator
let executeNanopasses
    (config: TwoPhaseConfig)
    (registry: NanopassRegistry)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    (intermediatesDir: string option)
    : MLIRAccumulator =

    if List.isEmpty registry.Nanopasses then
        MLIRAccumulator.empty()
    else
        // Create SINGLE shared accumulator for ALL nanopasses
        let sharedAcc = MLIRAccumulator.empty()

        // ═════════════════════════════════════════════════════════════════════════
        // PHASE 1: Content Witnesses (Literal, Arith, Lazy, Collections)
        // ═════════════════════════════════════════════════════════════════════════

        let contentPasses =
            registry.Nanopasses
            |> List.filter (fun np -> np.Phase = ContentPhase)

        let contentResults =
            if not (List.isEmpty contentPasses) then
                runNanopassesSequentialShared contentPasses graph coeffects sharedAcc
            else
                []

        // Serialize Phase 1 results (each nanopass gets snapshot of shared accumulator)
        match intermediatesDir with
        | Some dir ->
            contentResults
            |> List.iter (fun (nanopass, visited) ->
                serializeNanopassResult dir nanopass sharedAcc visited)
        | None -> ()

        // ═════════════════════════════════════════════════════════════════════════
        // PHASE 2: Structural Witnesses (Lambda, ControlFlow)
        // ═════════════════════════════════════════════════════════════════════════

        let structuralPasses =
            registry.Nanopasses
            |> List.filter (fun np -> np.Phase = StructuralPhase)

        let structuralResults =
            if not (List.isEmpty structuralPasses) then
                runNanopassesSequentialShared structuralPasses graph coeffects sharedAcc
            else
                []

        // Serialize Phase 2 results
        match intermediatesDir with
        | Some dir ->
            structuralResults
            |> List.iter (fun (nanopass, visited) ->
                serializeNanopassResult dir nanopass sharedAcc visited)
        | None -> ()

        // ═════════════════════════════════════════════════════════════════════════
        // COVERAGE VALIDATION: Detect unwitnessed reachable nodes
        // ═════════════════════════════════════════════════════════════════════════

        // Merge all visited sets from all nanopasses
        let allVisited =
            (contentResults @ structuralResults)
            |> List.fold (fun acc (_, visited) -> Set.union acc visited) Set.empty

        let coverageErrors = CoverageValidation.validateCoverage graph allVisited
        let stats = CoverageValidation.calculateStats graph allVisited

        coverageErrors |> List.iter (fun diag -> MLIRAccumulator.addError diag sharedAcc)

        // Serialize coverage report if intermediates enabled
        match intermediatesDir with
        | Some dir ->
            let coverageReport = {|
                TotalNodes = stats.TotalNodes
                ReachableNodes = stats.ReachableNodes
                WitnessedNodes = stats.WitnessedNodes
                UnwitnessedNodes = stats.UnwitnessedNodes
                CoveragePercentage = stats.CoveragePercentage
                CoverageErrors = coverageErrors |> List.map Diagnostic.format
            |}
            let json = JsonSerializer.Serialize(coverageReport, JsonSerializerOptions(WriteIndented = true))
            let coveragePath = Path.Combine(dir, "08_coverage.json")
            File.WriteAllText(coveragePath, json)

            let totalOps = MLIRAccumulator.totalOperations sharedAcc
            let allOpsCount = List.length sharedAcc.AllOps
            printfn "[Alex] Coverage: %d/%d witnessed (%.1f%%), %d ops in %d stream items"
                stats.WitnessedNodes
                stats.ReachableNodes
                stats.CoveragePercentage
                totalOps
                allOpsCount
        | None -> ()

        sharedAcc
