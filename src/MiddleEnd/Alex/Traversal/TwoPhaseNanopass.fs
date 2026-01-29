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
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
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

/// Run multiple nanopasses in parallel via IcedTasks
/// Results collected AS THEY COMPLETE (order doesn't matter)
let runNanopassesParallel
    (nanopasses: Nanopass list)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    : MLIRAccumulator list =

    // Fan out: Execute all nanopasses in parallel using IcedTasks coldTask
    // ColdTask is unit -> Task<'T>, so we create them, then invoke all with Task.WhenAll
    nanopasses
    |> List.map (fun nanopass ->
        coldTask { return runNanopass nanopass graph coeffects })
    |> List.map (fun ct -> ct())  // Invoke all coldTasks to start them
    |> fun tasks -> System.Threading.Tasks.Task.WhenAll(tasks).GetAwaiter().GetResult()
    |> List.ofArray

/// Run multiple nanopasses sequentially (for debugging/validation)
let runNanopassesSequential
    (nanopasses: Nanopass list)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    : MLIRAccumulator list =

    nanopasses
    |> List.map (fun nanopass -> runNanopass nanopass graph coeffects)

// ═══════════════════════════════════════════════════════════════════════════
// ENVELOPE PASS (Reactive Result Collection)
// ═══════════════════════════════════════════════════════════════════════════

/// Envelope pass: Reactively collect results AS THEY ARRIVE
/// Since nanopasses are referentially transparent and merge is associative,
/// order doesn't matter - merge results as they complete
let collectEnvelopeReactive (nanopassResults: MLIRAccumulator list) : MLIRAccumulator =
    match nanopassResults with
    | [] -> MLIRAccumulator.empty()
    | [single] -> single
    | many ->
        // Overlay all results associatively
        // Order of arrival doesn't matter (associativity + referential transparency)
        many
        |> List.reduce overlayAccumulators

/// Envelope pass (alias for backwards compatibility)
let collectEnvelope = collectEnvelopeReactive

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
let private serializeNanopassResult (intermediatesDir: string) (nanopass: Nanopass) (accumulator: MLIRAccumulator) : unit =
    let fileName = sprintf "07_%s_witness.json" (nanopass.Name.ToLower())
    let filePath = Path.Combine(intermediatesDir, fileName)

    let summary = {|
        NanopassName = nanopass.Name
        OperationCount = List.length accumulator.TopLevelOps
        ErrorCount = List.length accumulator.Errors
        VisitedNodes = accumulator.Visited |> Set.toList |> List.map (fun nodeId -> NodeId.value nodeId)
        Errors = accumulator.Errors |> List.map Diagnostic.format
        Operations = accumulator.TopLevelOps |> List.map (fun op -> sprintf "%A" op)
    |}

    let json = JsonSerializer.Serialize(summary, JsonSerializerOptions(WriteIndented = true))
    File.WriteAllText(filePath, json)
    printfn "[Alex] Wrote nanopass result: %s" fileName

/// Main entry point: Run all nanopasses in two phases and collect results
let executeNanopasses
    (config: TwoPhaseConfig)
    (registry: NanopassRegistry)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    (intermediatesDir: string option)
    : MLIRAccumulator =

    if List.isEmpty registry.Nanopasses then
        // No nanopasses registered - empty result
        MLIRAccumulator.empty()
    else
        // FUTURE: Discovery optimization (currently disabled)
        // if config.EnableDiscovery && graph.Nodes.Count > config.DiscoveryThreshold then
        //     let presentKinds = discoverPresentNodeTypes graph
        //     let filteredRegistry = filterRelevantNanopasses registry presentKinds
        //     registry <- filteredRegistry

        // ═════════════════════════════════════════════════════════════════════════
        // PHASE 1: Content Witnesses (Literal, Arith, Lazy, Collections)
        // ═════════════════════════════════════════════════════════════════════════

        let contentPasses =
            registry.Nanopasses
            |> List.filter (fun np -> np.Phase = ContentPhase)

        let phase1Results =
            if List.isEmpty contentPasses then
                []
            else
                if config.EnableParallel then
                    runNanopassesParallel contentPasses graph coeffects
                else
                    runNanopassesSequential contentPasses graph coeffects

        // Serialize Phase 1 results
        match intermediatesDir with
        | Some dir ->
            List.zip contentPasses phase1Results
            |> List.iter (fun (nanopass, accumulator) ->
                serializeNanopassResult dir nanopass accumulator)
        | None -> ()

        // Overlay Phase 1 results into intermediate accumulator
        let phase1Accumulator = collectEnvelope phase1Results

        // ═════════════════════════════════════════════════════════════════════════
        // PHASE 2: Structural Witnesses (Lambda, ControlFlow)
        // ═════════════════════════════════════════════════════════════════════════

        let structuralPasses =
            registry.Nanopasses
            |> List.filter (fun np -> np.Phase = StructuralPhase)

        let phase2Results =
            if List.isEmpty structuralPasses then
                []
            else
                if config.EnableParallel then
                    runNanopassesParallel structuralPasses graph coeffects
                else
                    runNanopassesSequential structuralPasses graph coeffects

        // Serialize Phase 2 results
        match intermediatesDir with
        | Some dir ->
            List.zip structuralPasses phase2Results
            |> List.iter (fun (nanopass, accumulator) ->
                serializeNanopassResult dir nanopass accumulator)
        | None -> ()

        // Overlay Phase 2 results
        let phase2Accumulator = collectEnvelope phase2Results

        // ═════════════════════════════════════════════════════════════════════════
        // MERGE PHASES: Overlay Phase 1 and Phase 2
        // ═════════════════════════════════════════════════════════════════════════

        let mergedAccumulator = overlayAccumulators phase1Accumulator phase2Accumulator

        // ═════════════════════════════════════════════════════════════════════════
        // COVERAGE VALIDATION: Detect unwitnessed reachable nodes
        // ═════════════════════════════════════════════════════════════════════════

        let coverageErrors = CoverageValidation.validateCoverage graph mergedAccumulator
        let stats = CoverageValidation.calculateStats graph mergedAccumulator

        // Add coverage errors to accumulator (addError mutates in place)
        coverageErrors |> List.iter (fun diag -> MLIRAccumulator.addError diag mergedAccumulator)

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
            printfn "[Alex] Coverage: %d/%d witnessed (%.1f%%)" stats.WitnessedNodes stats.ReachableNodes stats.CoveragePercentage
        | None -> ()

        mergedAccumulator
