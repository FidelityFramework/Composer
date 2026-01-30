/// SinglePhaseNanopass - Single-phase post-order execution for all witnesses
///
/// All witnesses run in one post-order traversal (children before parents).
/// Scope-owning witnesses (Lambda, ControlFlow) use markers to extract/wrap operations.
/// 
/// ARCHITECTURAL NOTE: This was previously "TwoPhaseNanopass" with ContentPhase/StructuralPhase.
/// Two-phase execution caused scope tracking bugs (operations emitted before scope markers existed).
/// Single-phase restores correct scoping: operations always emitted after their scope markers.
module Alex.Traversal.SinglePhaseNanopass

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
// SINGLE-PHASE EXECUTION (Post-Order Traversal)
// ═══════════════════════════════════════════════════════════════════════════

// ARCHITECTURE: Single-phase post-order traversal (children before parents)
//
// All witnesses run in one traversal:
//   - Post-order ensures children witnessed before parents
//   - Scope-owning witnesses (Lambda, ControlFlow) insert markers before recursing
//   - Operations accumulate to flat stream between markers
//   - Scope witnesses extract operations between markers and wrap in FuncDef/SCFOp
//
// CORRECTNESS: Operations always emitted AFTER their scope markers exist.
// This fixes the two-phase bug where ContentPhase emitted operations before
// StructuralPhase created scope markers, causing operations to escape to module scope.

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

/// Run nanopasses with SHARED accumulator and GLOBAL visited set via SINGLE combined traversal
/// This ensures post-order guarantees hold across ALL witnesses and phases
let runNanopassesSequentialShared
    (nanopasses: Nanopass list)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    (sharedAcc: MLIRAccumulator)
    (globalVisited: ref<Set<NodeId>>)  // GLOBAL visited set (shared across phases)
    : unit =

    // Create combined witness that tries all nanopasses at each node
    let combinedWitness = combineWitnesses nanopasses

    // Visit ALL reachable nodes in a SINGLE post-order traversal using GLOBAL visited set
    for kvp in graph.Nodes do
        let nodeId, node = kvp.Key, kvp.Value
        if node.IsReachable && not (Set.contains nodeId !globalVisited) then
            match PSGZipper.create graph nodeId with
            | None -> ()
            | Some initialZipper ->
                let nodeCtx = {
                    Graph = graph
                    Coeffects = coeffects
                    Accumulator = sharedAcc
                    Zipper = initialZipper
                    GlobalVisited = globalVisited
                }
                visitAllNodes combinedWitness nodeCtx node sharedAcc globalVisited

/// Run nanopasses in parallel with SHARED accumulator (future optimization)
/// Current: Sequential execution (parallel execution requires locking on mutable accumulator)
let runNanopassesParallelShared
    (nanopasses: Nanopass list)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    (sharedAcc: MLIRAccumulator)
    (globalVisited: ref<Set<NodeId>>)
    : unit =

    // FUTURE: Parallel execution with lock-free data structures
    // Current: Sequential for correctness
    runNanopassesSequentialShared nanopasses graph coeffects sharedAcc globalVisited

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

/// Configuration for single-phase execution
type SinglePhaseConfig = {
    /// FUTURE: Enable parallel witness execution (requires lock-free accumulator)
    /// Current: Always false (sequential execution for correctness)
    EnableParallel: bool

    /// FUTURE: Enable discovery pass for large projects
    /// Threshold where pre-scanning node types becomes cheaper than empty traversals
    /// Current: Always false (full-fat execution)
    EnableDiscovery: bool

    /// FUTURE: Node count threshold to trigger discovery
    /// Current: Not used (discovery disabled)
    DiscoveryThreshold: int
}

let defaultConfig = {
    EnableParallel = false  // Sequential for now
    EnableDiscovery = false  // Deferred optimization
    DiscoveryThreshold = 10000  // Placeholder for future tuning
}

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS RESULT SERIALIZATION (for -k flag debugging)
// ═══════════════════════════════════════════════════════════════════════════

/// Serialize nanopass phase result to JSON for debugging
let private serializePhaseResult (intermediatesDir: string) (phaseName: string) (accumulator: MLIRAccumulator) (globalVisited: ref<Set<NodeId>>) : unit =
    let fileName = sprintf "07_%s_witness.json" (phaseName.ToLower())
    let filePath = Path.Combine(intermediatesDir, fileName)

    let summary = {|
        NanopassName = phaseName
        OperationCount = List.length accumulator.AllOps
        ErrorCount = List.length accumulator.Errors
        VisitedNodes = !globalVisited |> Set.toList |> List.map (fun nodeId -> NodeId.value nodeId)
        Errors = accumulator.Errors |> List.map Diagnostic.format
        Operations = accumulator.AllOps |> List.map (fun op -> sprintf "%A" op)
    |}

    let json = JsonSerializer.Serialize(summary, JsonSerializerOptions(WriteIndented = true))
    File.WriteAllText(filePath, json)
    printfn "[Alex] Wrote nanopass result: %s" fileName

/// Main entry point: Run all nanopasses in single-phase post-order traversal
let executeNanopasses
    (config: SinglePhaseConfig)
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

        // Create SINGLE global visited set for ALL nanopasses
        let globalVisited = ref Set.empty

        // ═════════════════════════════════════════════════════════════════════════
        // SINGLE-PHASE EXECUTION: All witnesses in one post-order traversal
        // ═════════════════════════════════════════════════════════════════════════

        printfn "[Alex] Single-phase execution: %d registered nanopasses" (List.length registry.Nanopasses)

        // Run all nanopasses together in single traversal
        runNanopassesSequentialShared registry.Nanopasses graph coeffects sharedAcc globalVisited

        // Serialize results
        match intermediatesDir with
        | Some dir ->
            serializePhaseResult dir "singlephase" sharedAcc globalVisited
        | None -> ()

        // ═════════════════════════════════════════════════════════════════════════
        // COVERAGE VALIDATION: Detect unwitnessed reachable nodes
        // ═════════════════════════════════════════════════════════════════════════

        let coverageErrors = CoverageValidation.validateCoverage graph !globalVisited
        let stats = CoverageValidation.calculateStats graph !globalVisited

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
