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
open Alex.Patterns.MutableCallablePatterns
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
        // Module-level mutable value: store into its slot (valid in any function)
        let slotTarget =
            match SemanticGraph.tryGetNode targetId ctx.Graph with
            | Some { Kind = SemanticKind.VarRef (_, Some bindingId) } ->
                match SemanticGraph.tryGetNode bindingId ctx.Graph with
                | Some bindingNode when ModuleValues.isSlotBinding ctx.Coeffects.TargetPlatform ctx.Graph bindingNode -> Some (bindingId, bindingNode)
                | _ -> None
            | _ -> None
        let callableTarget =
            let projection =
                Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryCallable ctx.Graph
                |> Result.defaultWith failwith
            let owners = projection.MutableStorage |> Map.toList |> List.choose (fun (binding, storage) ->
                if storage.Writes |> List.exists (fun write -> write.Site = node.Id) then Some binding else None)
            match owners with
            | [] -> None
            | [binding] -> Some binding
            | _ -> failwith "Source publication assigned a mutable callable write to more than one cell."
        match slotTarget, callableTarget with
        | _, Some binding ->
            match tryMatchWithDiagnostics (pAssignMutableCallable ctx binding node.Id)
                          ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | Result.Error reason -> WitnessOutput.error $"Mutable callable assignment: {reason}"
        | Some (bindingId, bindingNode), None ->
            match bindingNode.Kind, MLIRAccumulator.recallNode valueId ctx.Accumulator with
            | SemanticKind.Binding (bindingName, _, _, _), Some (rawSSA, rawTy) ->
                // the slot at the binding's width; the value adapted to it by its derived meet
                let valueTy = mapTypeAt bindingId ctx
                let (meetOps, valueSSA, _) = adaptOperand ctx.Coeffects ctx.Graph node.Id valueId rawSSA rawTy
                let globalName = ModuleValues.globalName bindingName bindingId
                match tryMatchWithDiagnostics (pGlobalSlotStore bindingId node.Id globalName valueSSA valueTy)
                              ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Result.Ok ((ops, result), _) -> { InlineOps = meetOps @ ops; TopLevelOps = []; Result = result }
                | Result.Error diagnostic -> WitnessOutput.error $"Module value assignment: {diagnostic}"
            | SemanticKind.Binding _, None -> WitnessOutput.error "Module value assignment: Value not yet witnessed"
            | _ ->
                WitnessOutput.errorDiag (
                    Diagnostic.error (Some node.Id) (Some "MutableAssignment") (Some "module value slot")
                        $"PSG settlement (WitnessEmission storage) admitted node {NodeId.value bindingId} as a module value slot, but it is not a named Binding; the slot symbol for assignment node {NodeId.value node.Id} is unsettled")
        | None, None ->

        let memrefResult =
            match SemanticGraph.tryGetNode targetId ctx.Graph with
            | Some { Kind = SemanticKind.VarRef (_, Some bindingId) } ->
                MLIRAccumulator.recallNode bindingId ctx.Accumulator
            | _ -> None

        match memrefResult, MLIRAccumulator.recallNode valueId ctx.Accumulator with
        | Some (memrefSSA, (TMemRef elemType | TMemRefStatic (_, elemType))), Some (rawSSA, rawTy) ->
            // The value at the cell's width (the meet SSAAssignment derived for (set, value)),
            // then memref.store to update the mutable variable
            let (meetOps, valueSSA, _) = adaptOperand ctx.Coeffects ctx.Graph node.Id valueId rawSSA rawTy
            let (NodeId nodeIdInt) = node.Id
            match tryMatchWithDiagnostics (pStoreMutableVariable nodeIdInt memrefSSA valueSSA elemType)
                          ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((ops, result), _) ->
                { InlineOps = meetOps @ ops; TopLevelOps = []; Result = result }
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
