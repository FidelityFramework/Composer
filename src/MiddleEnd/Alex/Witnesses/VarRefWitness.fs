/// VarRefWitness - Witness variable reference nodes
///
/// Variable references don't emit MLIR - they forward the binding's SSA.
/// The binding SSA is looked up from the accumulator (bindings witnessed first in post-order).
///
/// FUNCTION REFERENCES: VarRef nodes pointing to function bindings (Lambda nodes) are SKIPPED.
/// ApplicationWitness handles extracting function names directly from VarRef nodes.
///
/// NANOPASS: This witness handles ONLY VarRef nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.VarRefWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types  // TMemRef
open Alex.CodeGeneration.TypeMapping
open XParsec
open XParsec.Parsers
open XParsec.Combinators

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness variable reference nodes - forwards binding's SSA
let private witnessVarRef (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    printfn "[VarRefWitness] Attempting to witness node %A (kind: %s)"
        (NodeId.value node.Id)
        (node.Kind.ToString().Split('\n').[0])
    match tryMatch pVarRef ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((name, bindingIdOpt), _) ->
        printfn "[VarRefWitness] Successfully matched VarRef '%s' with binding %A"
            name
            (bindingIdOpt |> Option.map NodeId.value)
        match bindingIdOpt with
        | Some bindingId ->
            // Check if the binding references a function (Lambda node)
            match SemanticGraph.tryGetNode bindingId ctx.Graph with
            | Some bindingNode ->
                // Check binding type
                match bindingNode.Kind with
                | SemanticKind.PatternBinding _ ->
                    // Parameter binding - SSA is in coeffects, not accumulator
                    // Extract SSA monadically
                    let patternBindingPattern =
                        parser {
                            let! ssa = getNodeSSA bindingId
                            let! state = getUserState
                            let arch = state.Coeffects.Platform.TargetArch
                            let ty = mapNativeTypeForArch arch bindingNode.Type
                            return ([], TRValue { SSA = ssa; Type = ty })
                        }

                    match tryMatch patternBindingPattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error $"VarRef '{name}': PatternBinding has no SSA in coeffects"

                | SemanticKind.Binding _ ->
                    // Check if binding's child is a Lambda (function binding)
                    let isFunctionBinding =
                        bindingNode.Children
                        |> List.tryHead
                        |> Option.bind (fun childId -> SemanticGraph.tryGetNode childId ctx.Graph)
                        |> Option.map (fun childNode -> childNode.Kind.ToString().StartsWith("Lambda"))
                        |> Option.defaultValue false

                    if isFunctionBinding then
                        // Function reference - ApplicationWitness handles this
                        WitnessOutput.skip
                    else
                        // Value binding - post-order: binding already witnessed, recall its SSA
                        match MLIRAccumulator.recallNode bindingId ctx.Accumulator with
                        | Some (ssa, ty) ->
                            // ARCHITECTURAL PRINCIPLE (lvalue vs rvalue):
                            // VarRef ALWAYS forwards the binding's SSA, regardless of mutability.
                            // - For TMemRef: Returns the memref address (consumers decide whether to load)
                            // - For other types: Returns the value (already in SSA)
                            //
                            // This enables:
                            // - Set operations to use memref directly (no load needed)
                            // - Expression contexts to load from memref when needed
                            // - No context-dependent behavior in VarRef itself
                            printfn "[VarRefWitness] Forwarding binding %A with type %A"
                                (NodeId.value bindingId) ty
                            { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = ssa; Type = ty } }
                        | None ->
                            WitnessOutput.error $"VarRef '{name}': Binding not yet witnessed"

                | _ ->
                    WitnessOutput.error $"VarRef '{name}': Unexpected binding kind {bindingNode.Kind}"
            | None ->
                WitnessOutput.error $"VarRef '{name}': Binding node not found"
        | None ->
            WitnessOutput.error $"VarRef '{name}': No binding ID (unresolved reference)"
    | None ->
        printfn "[VarRefWitness] pVarRef FAILED to match node %A (kind: %s)"
            (NodeId.value node.Id)
            (node.Kind.ToString().Split('\n').[0])
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// VarRef nanopass - witnesses variable references
let nanopass : Nanopass = {
    Name = "VarRef"
    Witness = witnessVarRef
}
