/// MutableAssignmentWitness - Witness mutable variable assignment (x <- value)
///
/// Handles F# mutable assignment via SemanticKind.Set nodes.
/// Emits memref.store to update the mutable variable.
///
/// NANOPASS: This witness handles ONLY Set nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.MutableAssignmentWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core  // SemanticGraph
open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemRefPatterns
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness mutable assignment nodes (x <- value)
let private witnessMutableAssignment (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pSet ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((targetId, valueId), _) ->
        // targetId is a VarRef node. VarRef now auto-loads (returns loaded value).
        // For assignment, we need the MEMREF ADDRESS from the underlying binding.
        // Navigate: targetId (VarRef) → bindingId (Binding) → recall memref from accumulator.
        let memrefResult =
            match SemanticGraph.tryGetNode targetId ctx.Graph with
            | Some { Kind = SemanticKind.VarRef (_, Some bindingId) } ->
                MLIRAccumulator.recallNode bindingId ctx.Accumulator
            | _ -> None

        match memrefResult, MLIRAccumulator.recallNode valueId ctx.Accumulator with
        | Some (memrefSSA, (TMemRef elemType | TMemRefStatic (_, elemType))), Some (valueSSA, _) ->
            // Emit memref.store to update mutable variable
            let (NodeId nodeIdInt) = node.Id
            match tryMatchWithDiagnostics (pStoreMutableVariable nodeIdInt memrefSSA valueSSA elemType)
                          ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((ops, result), _) ->
                { InlineOps = ops; TopLevelOps = []; Result = result }
            | Result.Error diagnostic ->
                WitnessOutput.error $"Mutable assignment: {diagnostic}"
        | Some (_, ty), Some _ ->
            WitnessOutput.error $"Mutable assignment: Target is not a mutable variable (type: {ty})"
        | Some _, None ->
            WitnessOutput.error "Mutable assignment: Value not yet witnessed"
        | None, _ ->
            WitnessOutput.error "Mutable assignment: Target variable not yet witnessed (binding not found)"
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// MutableAssignment nanopass - witnesses mutable variable assignment
let nanopass : Nanopass = {
    Name = "MutableAssignment"
    Witness = witnessMutableAssignment
}
