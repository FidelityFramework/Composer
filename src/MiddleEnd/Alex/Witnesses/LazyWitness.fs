/// LazyWitness - Witness Lazy<'T> operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Lazy-related nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.LazyWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Get all SSAs for a node (for complex nodes with multiple intermediates)
let private getNodeSSAs (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA list =
    match SSAAssign.lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Get captures as Val list from coeffects
/// CaptureInfo contains: Name, Type, IsMutable, SourceNodeId
let private getCaptures (captures: CaptureInfo list) (ctx: WitnessContext) : Val list =
    captures
    |> List.map (fun capture ->
        // Extract NodeId from CaptureInfo and recall the previously-witnessed node
        match capture.SourceNodeId with
        | Some captureId ->
            let nodeIdVal = NodeId.value captureId
            match MLIRAccumulator.recallNode nodeIdVal ctx.Accumulator with
            | Some (ssa, ty) -> { SSA = ssa; Type = ty }
            | None -> failwithf "Capture node %A not yet witnessed" captureId
        | None ->
            failwithf "CaptureInfo for '%s' has no SourceNodeId" capture.Name)

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Lazy operations - category-selective (handles only Lazy nodes)
let private witnessLazy (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Try LazyExpr pattern
    match tryMatch pLazyExpr ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((bodyId, captureInfos), _) ->
        // Get all SSAs from coeffects (5 + N captures per SSAAssignment)
        let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
        let arch = ctx.Coeffects.Platform.TargetArch

        // Get captures as Val list
        let captures = getCaptures captureInfos ctx

        // Get code pointer SSA - this should come from the body function
        // For now, we need to look up the function pointer for the body
        let bodyIdVal = NodeId.value bodyId
        let codePtr =
            match MLIRAccumulator.recallNode bodyIdVal ctx.Accumulator with
            | Some (ssa, _) -> ssa
            | None ->
                // Body not yet witnessed - this is an ordering issue
                // In a proper implementation, we'd ensure the body is witnessed first
                failwithf "LazyExpr body %A not yet witnessed - ordering issue" bodyId

        // Call pattern to build lazy struct
        match tryMatch (pBuildLazyStruct codePtr captures ssas arch) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | None -> WitnessOutput.error "LazyExpr pattern emission failed"

    | None ->
        // Try LazyForce pattern
        match tryMatch pLazyForce ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (lazyNodeId, _) ->
            // LazyForce is a SIMPLE operation (not elaborated by FNCS)
            // SSA cost: Fixed 4 (extract code_ptr, const 1, alloca, call)
            //
            // Calling convention: Thunk receives pointer to lazy struct
            // 1. Extract code_ptr from lazy struct [2]
            // 2. Alloca space for lazy struct on stack
            // 3. Store lazy struct to get pointer
            // 4. Call thunk with pointer -> result
            //
            // The thunk extracts captures internally using LazyLayout coeffect

            let ssas = getNodeSSAs node.Id ctx.Coeffects.SSA
            let arch = ctx.Coeffects.Platform.TargetArch

            // Get the lazy value SSA (the struct we're forcing)
            let lazyIdVal = NodeId.value lazyNodeId
            let lazySSA =
                match MLIRAccumulator.recallNode lazyIdVal ctx.Accumulator with
                | Some (ssa, _) -> ssa
                | None -> failwithf "Lazy value %A not yet witnessed" lazyNodeId

            // Call pattern to emit force operations
            match tryMatch (pBuildLazyForce lazySSA ssas.[3] ssas arch) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | None -> WitnessOutput.error "LazyForce pattern emission failed"

        | None ->
            // Not a lazy node - skip
            WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Lazy nanopass - witnesses LazyExpr and LazyForce nodes
let nanopass : Nanopass = {
    Name = "Lazy"
    Witness = witnessLazy
}
