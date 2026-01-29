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

        // Create SINGLE global visited set for ALL nanopasses (across both phases)
        let globalVisited = ref Set.empty

        // ═════════════════════════════════════════════════════════════════════════
        // PHASE 1: Content Witnesses (Literal, Arith, Lazy, Collections)
        // ═════════════════════════════════════════════════════════════════════════

        let contentPasses =
            registry.Nanopasses
            |> List.filter (fun np -> np.Phase = ContentPhase)

        if not (List.isEmpty contentPasses) then
            runNanopassesSequentialShared contentPasses graph coeffects sharedAcc globalVisited

        // Serialize Phase 1 results
        match intermediatesDir with
        | Some dir ->
            serializePhaseResult dir "content" sharedAcc globalVisited
        | None -> ()

        // ═════════════════════════════════════════════════════════════════════════
        // PHASE 2: Structural Witnesses (Lambda, ControlFlow)
        // ═════════════════════════════════════════════════════════════════════════

        let structuralPasses =
            registry.Nanopasses
            |> List.filter (fun np -> np.Phase = StructuralPhase)

        if not (List.isEmpty structuralPasses) then
            runNanopassesSequentialShared structuralPasses graph coeffects sharedAcc globalVisited

        // Serialize Phase 2 results
        match intermediatesDir with
        | Some dir ->
            serializePhaseResult dir "lambda" sharedAcc globalVisited
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
