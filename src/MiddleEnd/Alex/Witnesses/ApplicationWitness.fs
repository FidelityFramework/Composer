/// ApplicationWitness - Witness function application nodes
///
/// Application nodes emit MLIR function calls (direct or indirect).
/// Post-order traversal ensures function and arguments are already witnessed.
///
/// NANOPASS: This witness handles ONLY Application nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.ApplicationWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns
open Alex.CodeGeneration.TypeMapping

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness application nodes - emits function calls
let private witnessApplication (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pApplication ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((funcId, argIds), _) ->
        // Post-order: function and arguments already witnessed
        match MLIRAccumulator.recallNode funcId ctx.Accumulator with
        | None -> WitnessOutput.error "Application: Function not yet witnessed"
        | Some (funcSSA, funcTy) ->
            // Recall argument SSAs
            let argsResult =
                argIds
                |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)

            // Ensure all arguments were witnessed
            let allWitnessed = argsResult |> List.forall Option.isSome
            if not allWitnessed then
                WitnessOutput.error "Application: Some arguments not yet witnessed"
            else
                let argSSAs = argsResult |> List.choose id |> List.map fst

                // Get result SSA and return type
                match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
                | None -> WitnessOutput.error "Application: No SSA assigned to result"
                | Some resultSSA ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let retType = mapNativeTypeForArch arch node.Type

                    // Emit indirect call (future: optimize to direct call for known names)
                    match tryMatch (pApplicationCall resultSSA funcSSA argSSAs retType) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Application pattern emission failed"
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Application nanopass - witnesses function applications
let nanopass : Nanopass = {
    Name = "Application"
    Phase = ContentPhase
    Witness = witnessApplication
}
