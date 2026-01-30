/// PlatformWitness - Witness platform I/O operations to MLIR via XParsec
///
/// Uses XParsec combinators to match PSG structure, delegates to PlatformPatterns for syscalls.
/// Handles: Sys.write, Sys.read, and other platform I/O operations.
///
/// NANOPASS: This witness handles ONLY platform syscall nodes (Sys.*).
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.PlatformWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.PlatformPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness platform operations - RETIRED
/// Platform intrinsic applications (Sys.*) are now handled by ApplicationWitness
let private witnessPlatform (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Skip intrinsic nodes - ApplicationWitness handles all intrinsic applications
    match node.Kind with
    | SemanticKind.Intrinsic _ -> WitnessOutput.skip
    | _ -> WitnessOutput.skip  // No non-intrinsic platform operations

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Platform nanopass - witnesses Sys.* syscall operations
let nanopass : Nanopass = {
    Name = "Platform"
    Witness = witnessPlatform
}
