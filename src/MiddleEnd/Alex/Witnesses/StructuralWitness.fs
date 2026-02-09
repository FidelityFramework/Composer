/// StructuralWitness - Transparent witness for structural container nodes
///
/// Structural nodes (ModuleDef, Sequential, PatternBinding) organize the PSG tree but don't emit MLIR.
/// They need to be "witnessed" per the Domain Responsibility Principle to avoid coverage gaps.
///
/// NANOPASS: This witness handles ONLY structural container nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.StructuralWitness

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness structural nodes transparently - they organize but don't emit ops
let private witnessStructural (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.ModuleDef (moduleName, _) ->
        // Module definition - structural container, children already witnessed
        // Return TRVoid per Domain Responsibility Principle
        { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | SemanticKind.Sequential childIds ->
        // Sequential: value is the LAST child's value (F# semantics: a; b; c evaluates to c)
        // Children already witnessed in post-order. Forward last child's result if it has one.
        // This is critical for match arm conditions (Sequential containing IfThenElse)
        // and expression-valued blocks.
        match childIds with
        | [] -> { InlineOps = []; TopLevelOps = []; Result = TRVoid }
        | _ ->
            let lastChildId = List.last childIds
            match MLIRAccumulator.recallNode lastChildId ctx.Accumulator with
            | Some (ssa, ty) ->
                // Forward last child's result as this Sequential's result
                MLIRAccumulator.bindNode node.Id ssa ty ctx.Accumulator
                { InlineOps = []; TopLevelOps = []; Result = TRVoid }
            | None ->
                // Last child didn't produce a value (e.g., unit expression, void statement)
                { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | SemanticKind.PatternBinding _ ->
        // Function parameter binding - SSA pre-assigned in coeffects (no MLIR generation)
        // This is a structural marker (like .NET/F# idiom for argv, destructured params)
        // VarRef nodes lookup these bindings from coeffects
        { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | SemanticKind.TypeDef _ ->
        // Type definition - structural declaration, no MLIR emission needed
        // DU/record types are used by DUConstruct/DUEliminate/FieldGet, not emitted directly
        { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | SemanticKind.TupleExpr _ ->
        // Tuple expression - structural container for tuple elements
        // Children already witnessed in post-order. Tuple is virtual (not materialized).
        // TupleGet nodes index into the element list to recall individual element SSAs.
        { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | SemanticKind.TupleGet (tupleId, index) ->
        // Tuple element extraction - recall the indexed element's SSA from the accumulator
        // Baker decomposes `match a, b with` into TupleExpr + TupleGet nodes.
        // TupleGet looks up the TupleExpr's element list and recalls the element at `index`.
        match SemanticGraph.tryGetNode tupleId ctx.Graph with
        | Some tupleNode ->
            match tupleNode.Kind with
            | SemanticKind.TupleExpr elements when index < List.length elements ->
                let elementId = elements.[index]
                match MLIRAccumulator.recallNode elementId ctx.Accumulator with
                | Some (ssa, ty) ->
                    // Bind this TupleGet node to the element's SSA (pass-through)
                    MLIRAccumulator.bindNode node.Id ssa ty ctx.Accumulator
                    { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                | None ->
                    WitnessOutput.error (sprintf "TupleGet: element %d (node %d) not in accumulator" index (NodeId.value elementId))
            | _ ->
                WitnessOutput.error (sprintf "TupleGet: tuple node %d is not a TupleExpr or index %d out of range" (NodeId.value tupleId) index)
        | None ->
            WitnessOutput.error (sprintf "TupleGet: tuple node %d not found in graph" (NodeId.value tupleId))

    | _ ->
        // Not a structural node - skip for other witnesses
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Structural nanopass - transparent witness for container nodes
let nanopass : Nanopass = {
    Name = "Structural"
    Witness = witnessStructural
}
