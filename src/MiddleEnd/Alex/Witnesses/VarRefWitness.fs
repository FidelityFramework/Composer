/// VarRefWitness - Witness variable reference nodes
///
/// Variable references don't emit MLIR - they forward the binding's SSA.
/// The binding SSA is looked up from the accumulator (bindings witnessed first in post-order).
///
/// NANOPASS: This witness handles ONLY VarRef nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.VarRefWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness variable reference nodes - forwards binding's SSA
let private witnessVarRef (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pVarRef ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((name, bindingIdOpt), _) ->
        match bindingIdOpt with
        | Some bindingId ->
            // Post-order: binding already witnessed, recall its SSA
            match MLIRAccumulator.recallNode bindingId ctx.Accumulator with
            | Some (ssa, ty) ->
                // Forward the binding's SSA - VarRef doesn't emit ops
                { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = ssa; Type = ty } }
            | None ->
                WitnessOutput.error $"VarRef '{name}': Binding not yet witnessed"
        | None ->
            WitnessOutput.error $"VarRef '{name}': No binding ID (unresolved reference)"
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// VarRef nanopass - witnesses variable references
let nanopass : Nanopass = {
    Name = "VarRef"
    Phase = ContentPhase
    Witness = witnessVarRef
}
