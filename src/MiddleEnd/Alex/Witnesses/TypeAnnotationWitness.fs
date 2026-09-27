/// TypeAnnotationWitness - Transparent wrapper witness using XParsec
///
/// TypeAnnotation nodes are transparent wrappers that annotate expressions with types.
/// This witness forwards the wrapped node's result without generating operations.
///
/// ARCHITECTURAL PRINCIPLE: This witness demonstrates proper XParsec monadic structure:
/// - ONLY uses tryMatch on XParsec patterns
/// - NO direct graph API access
/// - NO fallback computation logic
/// - Proper debug tooling for transparency
module Alex.Witnesses.TypeAnnotationWitness

open Clef.Compiler.NativeTypedTree.NativeTypes // NodeId identities
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.CallablePatterns
open Alex.Patterns.SequencePatterns
open Alex.Patterns.LazyPatterns

// ═══════════════════════════════════════════════════════════
// WITNESS IMPLEMENTATION (XParsec patterns only)
// ═══════════════════════════════════════════════════════════

/// Witness TypeAnnotation nodes - transparent wrapper pattern
/// This witness demonstrates proper monadic parser combinator structure
let private witnessTypeAnnotation (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Use XParsec pattern to extract wrapped node ID
    match tryMatch pTypeAnnotation ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((wrappedId, _), _) when isLazyValue ctx node ->
        match tryMatchWithDiagnostics (pLazyForward ctx wrappedId) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | Result.Error reason -> WitnessOutput.error $"Lazy annotation: {reason}"
    | Some ((wrappedId, _), _) when isSequenceValue ctx node ->
        match tryMatchWithDiagnostics (pSequenceForward ctx wrappedId) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | Result.Error reason -> WitnessOutput.error $"Sequence annotation: {reason}"
    | Some ((wrappedId, _), _) ->
        match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryCallable ctx.Graph with
        | Result.Error reason ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "Callable") (Some "annotation")
                $"PSG settlement (WitnessEmission callable) did not publish the value shape for TypeAnnotation node {NodeId.value node.Id}: {reason}"
        | Result.Ok projection ->
        match projection.ValueShapes.TryFind node.Id with
        | None ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "Callable") (Some "annotation")
                $"PSG settlement (WitnessEmission callable) did not settle the value shape for TypeAnnotation node {NodeId.value node.Id}"
        | Some (CallableValueShape.Callable owner) when owner = node.Id ->
            // An applied intrinsic's transparent callee annotation belongs to its own
            // pattern; every other callable annotation forwards its callable operands.
            let pattern =
                if projection.IntrinsicAliases.Contains node.Id then pIntrinsicCalleeAnnotation
                else pCallableForward ctx wrappedId
            match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | Result.Error reason ->
                WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "Callable") (Some "annotation") reason
        | Some (CallableValueShape.Data owner) when owner = node.Id ->
            // The wrapped occurrence already witnessed its own result. Forward it
            // through the common result check; do not search inside its structure.
            match MLIRAccumulator.recallNode wrappedId ctx.Accumulator with
            | Some (wrappedSSA, wrappedType) ->
                { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = wrappedSSA; Type = wrappedType } }
            | None when Alex.Traversal.Values.isUnitTyped ctx.Graph node.Id -> WitnessOutput.empty
            | None ->
                WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "TypeAnnotation") (Some "transparent forward")
                    $"Wrapped occurrence {NodeId.value wrappedId} produced no witnessed value for annotation {NodeId.value node.Id}."
        | Some _ ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "TypeAnnotation") (Some "value shape")
                "The published annotation shape does not identify this occurrence's supported value form."

    | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION
// ═══════════════════════════════════════════════════════════

/// TypeAnnotation nanopass - ContentPhase (runs during body witnessing)
let nanopass : Nanopass = {
    Name = "TypeAnnotation"
    Witness = witnessTypeAnnotation
}
