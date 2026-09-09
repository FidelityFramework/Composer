/// BindingWitness - Witness binding nodes (let bindings)
///
/// Bindings don't emit MLIR - they forward the bound value's SSA.
/// The binding name is tracked in the accumulator for VarRef lookup.
///
/// FUNCTION BINDINGS: Bindings whose child is a Lambda (function) are SKIPPED.
/// LambdaWitness handles generating FuncDefs for module-level functions.
///
/// NANOPASS: This witness handles ONLY Binding nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.BindingWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemRefPatterns
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness binding nodes - forwards bound value's SSA (immutable) or emits memref.alloca (mutable)
let private witnessBinding (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pBinding ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((name, isMut, _isRec, isEntry), _) ->
        // Binding has one child - the value being bound
        match node.Children with
        | [valueId] ->
            // Check for underscore discard pattern
            if name = "_" then
                // Discard pattern - value is witnessed for side effects, but no SSA binding created
                // Post-order ensures child is already witnessed
                { InlineOps = []; TopLevelOps = []; Result = TRVoid }
            else
                // Check if the child is a Lambda (function binding)
                match SemanticGraph.tryGetNode valueId ctx.Graph with
                | Some valueNode when valueNode.Kind.ToString().StartsWith("Lambda") ->
                    // Function binding - Lambda child already generated FuncDef (post-order)
                    // Check if Lambda produced a closure value (escaping lambda with captures)
                    match MLIRAccumulator.recallNode valueId ctx.Accumulator with
                    | Some (closureSSA, closureTy) ->
                        // Closure: forward the closure pair as this binding's value
                        { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = closureSSA; Type = closureTy } }
                    | None ->
                        // Named function (no captures) — structural, no SSA value
                        { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                | _ ->
                    // Check if this binding holds a partial application (curry flattening)
                    if Set.contains node.Id ctx.Graph.Codata.Value.Curry.PartialAppBindings then
                        // Partial application binding - no MLIR emitted
                        // Saturated call sites use the coeffect to emit direct calls
                        { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                    // Module-level value: initialize its program-lifetime slot (memref.global).
                    // References reload from the slot in whatever function they occur.
                    elif ModuleValues.isSlotBinding ctx.Coeffects.TargetPlatform ctx.Graph node then
                        // The initializer may be a block (`let a = ... in { ... }`): its value is
                        // the last value node of the block, not the block node itself.
                        let initValueId = findLastValueNode valueId ctx.Graph
                        match MLIRAccumulator.recallNode initValueId ctx.Accumulator with
                        | Some (initialSSA, initialTy) ->
                            let meetOps, valueSSA, _ = adaptOperand ctx.Coeffects ctx.Graph node.Id valueId initialSSA initialTy
                            // the slot's element type at the binding's range width on fabric
                            let valueTy = mapType node.Type ctx |> narrowType ctx.Coeffects ctx.Graph node.Id
                            let globalName = ModuleValues.globalName name node.Id
                            match tryMatchWithDiagnostics (pGlobalSlotInit node.Id globalName valueSSA valueTy)
                                          ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                            | Result.Ok ((ops, result), _) ->
                                { InlineOps = meetOps @ ops; TopLevelOps = []; Result = result }
                            | Result.Error diagnostic ->
                                WitnessOutput.error $"Module value '{name}': {diagnostic}"
                        | None ->
                            WitnessOutput.error $"Module value '{name}': Initial value not yet witnessed"
                    // Check if binding is mutable
                    elif isMut then
                        // MUTABLE BINDING: the cell at the binding node's width (the join of its
                        // value and every assignment); the initial value adapted to it by the
                        // meet SSAAssignment derived for (binding, value)
                        match MLIRAccumulator.recallNode valueId ctx.Accumulator with
                        | Some (rawInitSSA, rawInitTy) ->
                            let (NodeId nodeIdInt) = node.Id
                            let (meetOps, initSSA, initTy) = adaptOperand ctx.Coeffects ctx.Graph node.Id valueId rawInitSSA rawInitTy
                            let elemType =
                                match mapType node.Type ctx with
                                | TInt (IntWidth 0) -> narrowType ctx.Coeffects ctx.Graph node.Id (TInt (IntWidth 0))
                                | _ -> initTy
                            match tryMatchWithDiagnostics (pBuildMutableBinding nodeIdInt elemType initSSA)
                                          ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                            | Result.Ok ((ops, result), _) ->
                                { InlineOps = meetOps @ ops; TopLevelOps = []; Result = result }
                            | Result.Error diagnostic ->
                                WitnessOutput.error $"Mutable binding '{name}': {diagnostic}"
                        | None ->
                            WitnessOutput.error $"Mutable binding '{name}': Initial value not yet witnessed"
                    else
                        // IMMUTABLE BINDING: the value's SSA, brought to the binding's width where a
                        // declaration (a spelled annotation, a descriptor) holds the binding at a
                        // representation the value does not arrive at (the meet SSAAssignment derived
                        // for this binding); the value's own SSA where the widths agree
                        match MLIRAccumulator.recallNode valueId ctx.Accumulator with
                        | Some (rawSSA, rawTy) ->
                            let (meetOps, ssa, ty) = adaptOperand ctx.Coeffects ctx.Graph node.Id valueId rawSSA rawTy
                            { InlineOps = meetOps; TopLevelOps = []; Result = TRValue { SSA = ssa; Type = ty } }
                        | None ->
                            // Check if the child was witnessed but returned TRVoid
                            // This happens for entry point Lambdas (function definitions) and other module-level declarations
                            if isEntry.IsSome then
                                // Entry point binding - child is a function definition, not a value
                                { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                            else
                                WitnessOutput.error $"Binding '{name}': Value not yet witnessed (non-entry binding without SSA)"
        | _ ->
            WitnessOutput.error $"Binding '{name}': Expected 1 child, got {node.Children.Length}"
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Binding nanopass - witnesses let bindings
let nanopass : Nanopass = {
    Name = "Binding"
    Witness = witnessBinding
}
