/// ArithWitness - Witness arithmetic/comparison/bitwise operations to MLIR via XParsec
///
/// Uses XParsec combinators to match PSG structure, delegates to Patterns for MLIR elision.
/// Handles: binary arithmetic, comparisons, bitwise operations, boolean logic, unary operations.
///
/// NANOPASS: This witness handles ONLY arithmetic/comparison/bitwise nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.ArithWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════

/// Get single result SSA for a node
let private getSingleSSA (nodeId: NodeId) (ssa: SSAAssign.SSAAssignment) : SSA =
    match SSAAssign.lookupSSA nodeId ssa with
    | Some s -> s
    | None -> failwithf "No result SSA for node %A" nodeId

/// Get operand SSA by recalling previously-witnessed node
let private recallOperandSSA (nodeId: NodeId) (ctx: WitnessContext) : SSA =
    let nodeIdVal = NodeId.value nodeId
    match MLIRAccumulator.recallNode nodeIdVal ctx.Accumulator with
    | Some (ssa, _) -> ssa
    | None -> failwithf "Operand node %A not yet witnessed" nodeId

/// Try to extract binary operation structure from PSG
/// Returns: (intrinsicNode, lhsNode, rhsNode)
let private tryExtractBinaryOp (graph: SemanticGraph) (node: SemanticNode)
                                : (SemanticNode * NodeId * NodeId) option =
    // Binary operations are typically: App(App(intrinsic, lhs), rhs)
    // But FNCS elaboration may produce: Intrinsic node with Application pointing to operands
    // We need to handle both structures

    match node.Kind with
    | SemanticKind.Intrinsic _ ->
        // For intrinsics, check if there's an Application parent that provides operands
        // This requires looking at the graph structure
        // For now, return None - binary ops might not be represented this way yet
        None

    | SemanticKind.Application (funcId, argIds) ->
        // Check if func is another Application (curried binary op)
        match SemanticGraph.tryGetNode funcId graph with
        | Some funcNode ->
            match funcNode.Kind with
            | SemanticKind.Application (opId, [lhsId]) ->
                // App(App(op, lhs), rhs) structure
                match argIds with
                | [rhsId] ->
                    match SemanticGraph.tryGetNode opId graph with
                    | Some opNode ->
                        match opNode.Kind with
                        | SemanticKind.Intrinsic _ -> Some (opNode, lhsId, rhsId)
                        | _ -> None
                    | _ -> None
                | _ -> None
            | SemanticKind.Intrinsic _ ->
                // App(intrinsic, args) - maybe already saturated
                match argIds with
                | [lhsId; rhsId] -> Some (funcNode, lhsId, rhsId)
                | _ -> None
            | _ -> None
        | None -> None

    | _ -> None

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness arithmetic operations - category-selective (handles only arithmetic/comparison/bitwise nodes)
let private witnessArithmetic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let arch = ctx.Coeffects.Platform.TargetArch

    // Try to match as classified intrinsic
    match tryMatch pClassifiedIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((info, category), _) ->
        // Get result SSA from coeffects
        let resultSSA = getSingleSSA node.Id ctx.Coeffects.SSA

        // Try to extract binary operation structure
        match tryExtractBinaryOp ctx.Graph node with
        | Some (intrinsicNode, lhsId, rhsId) ->
            // Get operand SSAs
            let lhsSSA = recallOperandSSA lhsId ctx
            let rhsSSA = recallOperandSSA rhsId ctx

            // Dispatch based on category
            match category with
            | BinaryArith _ ->
                match tryMatch (pBuildBinaryArith resultSSA lhsSSA rhsSSA arch) ctx.Graph intrinsicNode ctx.Zipper ctx.Coeffects.Platform with
                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | None -> WitnessOutput.error "Binary arithmetic pattern emission failed"

            | Comparison _ ->
                match tryMatch (pBuildComparison resultSSA lhsSSA rhsSSA arch) ctx.Graph intrinsicNode ctx.Zipper ctx.Coeffects.Platform with
                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | None -> WitnessOutput.error "Comparison pattern emission failed"

            | _ ->
                // Not an arithmetic/comparison/bitwise intrinsic - skip
                WitnessOutput.skip

        | None ->
            // Couldn't extract binary structure
            // This might be because the PSG structure is different than expected
            // Or because FNCS hasn't elaborated the operation yet
            WitnessOutput.error $"Could not extract binary operation structure for intrinsic {info.FullName}"

    | None ->
        // Not a classified intrinsic - skip
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Arithmetic nanopass - witnesses binary/unary arithmetic, comparisons, bitwise ops
let nanopass : Nanopass = {
    Name = "Arithmetic"
    Witness = witnessArithmetic
}
