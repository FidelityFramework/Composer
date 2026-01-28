/// LazyWitness - Witness Lazy<'T> operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Lazy-related nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.LazyWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.PSGZipper
open Alex.XParsec.PSGCombinators

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Lazy operations - category-selective (handles only Lazy nodes)
let private witnessLazy (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    // LazyExpr - construct Lazy<'T> value
    | "LazyExpr" ->
        match tryMatch pLazyExpr ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some ((bodyId, captures), _) ->
            // TODO: Call Pattern to build lazy struct
            WitnessOutput.error "LazyExpr matched via XParsec - Pattern integration next"
        | None ->
            WitnessOutput.error "LazyExpr XParsec pattern match failed"

    // LazyForce - force evaluation of Lazy<'T>
    | "LazyForce" ->
        match tryMatch pLazyForce ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (lazyNodeId, _) ->
            // TODO: Call Pattern to force lazy
            WitnessOutput.error "LazyForce matched via XParsec - Pattern integration next"
        | None ->
            WitnessOutput.error "LazyForce XParsec pattern match failed"

    // Skip all other nodes (not Lazy-related)
    | _ -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Lazy nanopass - witnesses LazyExpr and LazyForce nodes
let nanopass : Nanopass = {
    Name = "Lazy"
    Witness = witnessLazy
}
