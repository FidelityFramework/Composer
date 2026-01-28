/// SeqWitness - Witness Seq<'T> operations to MLIR
///
/// PRD-15: Lazy sequence expressions with state machine semantics
///
/// ARCHITECTURAL MODEL (January 28, 2026):
/// This witness follows the compositional architecture:
/// - Elements (atoms) → Patterns (molecules) → Witnesses (observers)
/// - Witnesses are THIN (~40 lines) and DELEGATE to Patterns
/// - NO "chock full of primitives" - use Patterns for composition
module Alex.Witnesses.SeqWitness

open XParsec
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns
open Alex.Traversal.TransferTypes
open Alex.Dialects.Core.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core

// ═══════════════════════════════════════════════════════════════════════════
// PRIVATE HELPERS
// ═══════════════════════════════════════════════════════════════════════════

module SSAAssign = PSGElaboration.SSAAssignment

/// Get all SSAs for a node (multiple for struct construction)
let private getNodeSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA list =
    match SSAAssign.lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> []

/// Convert CaptureInfo list to Val list by looking up SSAs
let private captureInfosToVals (captures: CaptureInfo list) (ssa: SSAAssign.SSAAssignment) (graph: SemanticGraph) (arch: Architecture) : Val list =
    captures
    |> List.choose (fun cap ->
        match cap.SourceNodeId with
        | None -> None  // No source node, can't look up SSA
        | Some sourceNodeId ->
            match SSAAssign.lookupSSA sourceNodeId ssa with
            | Some capSSA ->
                // Use the type from CaptureInfo (already resolved by FNCS)
                let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch cap.Type
                Some { SSA = capSSA; Type = mlirType }
            | None -> None)

/// Extract mutable bindings from a PSG subtree (for seq internal state)
/// PSGElaboration already counted these (via countMutableBindingsInSubtree)
/// This function READS what's in the PSG, it does NOT analyze
let rec private extractMutableBindings (graph: SemanticGraph) (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) (arch: Architecture) : Val list =
    match SemanticGraph.tryGetNode nodeId graph with
    | None -> []
    | Some node ->
        // Check if this node is a mutable binding
        let thisVal =
            match node.Kind with
            | SemanticKind.Binding (name, isMutable, _, _) when isMutable ->
                // Get SSA for this binding (already assigned by PSGElaboration)
                match SSAAssign.lookupSSA nodeId ssa with
                | Some bindSSA ->
                    // Get type (already in PSG from FNCS)
                    let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                    [{ SSA = bindSSA; Type = mlirType }]
                | None -> []
            | _ -> []
        // Recursively extract from children
        let childVals =
            node.Children
            |> List.collect (fun childId -> extractMutableBindings graph childId ssa arch)
        thisVal @ childVals

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS IMPLEMENTATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Witness SeqExpr: Build sequence struct with state machine
let witnessSeqExpr (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Match SeqExpr via XParsec
    match tryMatch pSeqExpr ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | None -> WitnessOutput.skip
    | Some ((bodyId, captures), _) ->
        // Get SSAs for this node
        let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
        if ssas.IsEmpty then
            WitnessOutput.error "SeqExpr: No SSAs allocated"
        else
            // Recall the MoveNext lambda (body) from accumulator
            let bodyIdVal = NodeId.value bodyId
            let bodySSAOpt = MLIRAccumulator.recallNode bodyIdVal ctx.Accumulator
            match bodySSAOpt with
            | None ->
                WitnessOutput.error "SeqExpr: MoveNext lambda not found in accumulator"
            | Some (codePtrSSA, _) ->
                let arch = targetArch ctx

                // Convert captures to Val list (uses types from CaptureInfo, SSAs from PSGElaboration)
                let captureVals = captureInfosToVals captures ctx.Coeffects.SSA ctx.Graph arch

                // Extract internal state from body Lambda
                // PSGElaboration already counted these (countMutableBindingsInSubtree)
                // We just READ the mutable bindings from the PSG
                let internalState = extractMutableBindings ctx.Graph bodyId ctx.Coeffects.SSA arch

                // Delegate to Pattern
                match tryMatch (pBuildSeqStruct codePtrSSA captureVals internalState ssas arch)
                               ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some ((ops, result), _) ->
                    { InlineOps = ops; TopLevelOps = []; Result = result }
                | None ->
                    WitnessOutput.error "SeqExpr: pBuildSeqStruct pattern match failed"

/// Witness ForEach: Iterate over sequence with MoveNext
let witnessForEach (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Match ForEach via XParsec
    match tryMatch pForEach ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | None -> WitnessOutput.skip
    | Some ((var, collectionId, bodyId), _) ->
        // Recall the collection seq from accumulator
        let collectionIdVal = NodeId.value collectionId
        let collectionSSAOpt = MLIRAccumulator.recallNode collectionIdVal ctx.Accumulator
        match collectionSSAOpt with
        | None ->
            WitnessOutput.error "ForEach: Collection not found in accumulator"
        | Some (collectionSSA, _) ->
            let arch = targetArch ctx

            // Body ops: The fold orchestrator handles witnessing bodyId separately
            // ForEach witness just sets up the MoveNext loop structure
            let bodyOps = []  // Gap: pBuildForEachLoop not yet implemented

            // Delegate to Pattern
            match tryMatch (pBuildForEachLoop collectionSSA bodyOps arch)
                           ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some ((ops, result), _) ->
                { InlineOps = ops; TopLevelOps = []; Result = result }
            | None ->
                WitnessOutput.error "ForEach: pBuildForEachLoop pattern match failed"

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// Main witness entry point for Seq operations
let witness (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.SeqExpr _ -> witnessSeqExpr ctx node
    | SemanticKind.ForEach _ -> witnessForEach ctx node
    | _ -> WitnessOutput.skip

// Total: ~117 lines (with comments and structure)
// Core logic: ~60 lines
// Follows LazyWitness canonical pattern
