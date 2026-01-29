/// NanopassArchitecture - Parallel nanopass framework
///
/// Each witness = one nanopass (complete PSG traversal)
/// Nanopasses run in parallel via IcedTasks
/// Results overlay/fold into cohesive MLIR graph
module Alex.Traversal.NanopassArchitecture

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.PSGZipper

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS TYPE
// ═══════════════════════════════════════════════════════════════════════════

/// Nanopass execution phase
/// ContentPhase: Witnesses run first, respect scope boundaries, don't recurse into scopes
/// StructuralPhase: Witnesses run second, wrap content in scope structures (FuncDef, SCFOp)
type NanopassPhase =
    | ContentPhase      // Literal, Arith, Lazy, Collections - traverse but respect scopes
    | StructuralPhase   // Lambda, ControlFlow - handle scope boundaries

/// A nanopass is a complete PSG traversal that selectively witnesses nodes
type Nanopass = {
    /// Nanopass name (e.g., "Literal", "Arithmetic", "ControlFlow")
    Name: string

    /// Execution phase (ContentPhase runs before StructuralPhase)
    Phase: NanopassPhase

    /// The witnessing function for this nanopass
    /// Returns skip for nodes it doesn't handle
    Witness: WitnessContext -> SemanticNode -> WitnessOutput
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAVERSAL HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Check if this node defines a scope boundary (owns its children)
/// Scope boundaries prevent child recursion during global traversal.
/// Instead, scope-owning witnesses use witnessSubgraph to handle their children.
let private isScopeBoundary (node: SemanticNode) : bool =
    match node.Kind with
    | SemanticKind.Lambda _ -> true
    | SemanticKind.IfThenElse _ -> true
    | SemanticKind.WhileLoop _ -> true
    | SemanticKind.ForLoop _ -> true
    | SemanticKind.ForEach _ -> true
    | SemanticKind.Match _ -> true
    | SemanticKind.TryWith _ -> true
    | _ -> false

/// Visit all nodes in post-order (children before parents)
/// PUBLIC: Used by Lambda/ControlFlow witnesses for sub-graph traversal
/// Post-order ensures children's SSA bindings are available when parent witnesses
let rec visitAllNodes
    (witness: WitnessContext -> SemanticNode -> WitnessOutput)
    (visitedCtx: WitnessContext)
    (currentNode: SemanticNode)
    (accumulator: MLIRAccumulator)
    (visited: ref<Set<NodeId>>)  // GLOBAL visited set (shared across all nanopasses)
    : unit =

    // Check if already visited
    if Set.contains currentNode.Id !visited then
        ()
    else
        // Mark as visited
        visited := Set.add currentNode.Id !visited

        // Focus zipper on this node
        match PSGZipper.focusOn currentNode.Id visitedCtx.Zipper with
        | None -> ()
        | Some focusedZipper ->
            // Shadow context with focused zipper
            let focusedCtx = { visitedCtx with Zipper = focusedZipper }

            // POST-ORDER: Visit children FIRST (before witnessing current node)
            // This ensures children's SSA bindings are available when parent witnesses
            if not (isScopeBoundary currentNode) then
                for childId in currentNode.Children do
                    match SemanticGraph.tryGetNode childId visitedCtx.Graph with
                    | Some childNode -> visitAllNodes witness focusedCtx childNode accumulator visited
                    | None -> ()

            // THEN witness current node (after children are done)
            let output = witness focusedCtx currentNode

            // Add operations to flat accumulator stream
            MLIRAccumulator.addOps output.InlineOps accumulator
            MLIRAccumulator.addOps output.TopLevelOps accumulator

            // Bind result if value (global binding)
            match output.Result with
            | TRValue v ->
                MLIRAccumulator.bindNode currentNode.Id v.SSA v.Type accumulator
            | TRVoid -> ()
            | TRError diag ->
                MLIRAccumulator.addError diag accumulator

/// REMOVED: witnessSubgraph and witnessSubgraphWithResult
///
/// These functions created isolated accumulators which broke SSA binding resolution.
/// With the flat accumulator architecture, scope boundaries are handled via ScopeMarkers
/// instead of isolated accumulators.
///
/// Lambda and ControlFlow witnesses now use:
/// 1. ScopeMarker (ScopeEnter) to mark scope start
/// 2. Witness body nodes into shared accumulator
/// 3. ScopeMarker (ScopeExit) to mark scope end
/// 4. extractScope to get operations between markers
/// 5. Wrap extracted ops in FuncDef/SCFOp
/// 6. replaceScope to substitute wrapped operation for markers+contents

/// Run a single nanopass over entire PSG with SHARED accumulator and GLOBAL visited set
let runNanopass
    (nanopass: Nanopass)
    (graph: SemanticGraph)
    (coeffects: TransferCoeffects)
    (sharedAcc: MLIRAccumulator)  // SHARED accumulator (ops, bindings, errors)
    (globalVisited: ref<Set<NodeId>>)  // GLOBAL visited set (shared across ALL nanopasses)
    : unit =

    // Visit ALL reachable nodes, not just entry-point-reachable nodes
    // This ensures nodes like Console.write/writeln (reachable via VarRef but not via child edges) are witnessed
    for kvp in graph.Nodes do
        let nodeId, node = kvp.Key, kvp.Value
        if node.IsReachable && not (Set.contains nodeId !globalVisited) then
            match PSGZipper.create graph nodeId with
            | None -> ()
            | Some initialZipper ->
                let nodeCtx = {
                    Graph = graph
                    Coeffects = coeffects
                    Accumulator = sharedAcc  // SHARED accumulator
                    Zipper = initialZipper
                    GlobalVisited = globalVisited  // GLOBAL visited set
                }
                // Visit this reachable node (post-order) with GLOBAL visited set
                visitAllNodes nanopass.Witness nodeCtx node sharedAcc globalVisited

// ═══════════════════════════════════════════════════════════════════════════
// REMOVED: overlayAccumulators
// ═══════════════════════════════════════════════════════════════════════════

/// REMOVED: overlayAccumulators
///
/// This function merged separate accumulators from parallel nanopasses.
/// With the flat accumulator architecture, ALL nanopasses share a single accumulator,
/// so no merging is needed. Operations and bindings accumulate directly during traversal.

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRY
// ═══════════════════════════════════════════════════════════════════════════

/// Registry of all nanopasses (populated by witnesses)
type NanopassRegistry = {
    /// All registered nanopasses
    Nanopasses: Nanopass list
}

module NanopassRegistry =
    let empty = { Nanopasses = [] }

    let register (nanopass: Nanopass) (registry: NanopassRegistry) =
        { registry with Nanopasses = nanopass :: registry.Nanopasses }

    let registerAll (nanopasses: Nanopass list) (registry: NanopassRegistry) =
        { registry with Nanopasses = nanopasses @ registry.Nanopasses }
