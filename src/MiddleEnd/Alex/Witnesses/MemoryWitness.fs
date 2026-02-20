/// MemoryWitness - Witness memory operations via XParsec
///
/// Pure XParsec monadic observer - ZERO imperative SSA lookups.
/// Witnesses pass NodeIds to Patterns; Patterns extract SSAs via getUserState.
///
/// NANOPASS: This witness handles memory-related operations (FieldGet).
/// DU operations moved to DUWitness (all-platform, codata-dependent).
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.MemoryWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemoryPatterns

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness memory operations - handles FieldGet (struct field access)
/// DU operations (DUConstruct, DUGetTag, DUEliminate) handled by DUWitness
let private witnessMemory (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Skip intrinsic nodes - ApplicationWitness handles intrinsic applications
    match node.Kind with
    | SemanticKind.Intrinsic _ -> WitnessOutput.skip
    | _ ->
        // FieldGet (struct field access like s.Pointer, s.Length)
        match tryMatch pFieldGet ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((structId, fieldName), _) ->
            match MLIRAccumulator.recallNode structId ctx.Accumulator with
            | Some (structSSA, structTy) ->
                let fieldTy = mapType node.Type ctx

                match tryMatchWithDiagnostics (pStructFieldGet node.Id structSSA fieldName structTy fieldTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | Result.Error diagnostic -> WitnessOutput.error $"FieldGet '{fieldName}': {diagnostic}"
            | None -> WitnessOutput.error $"FieldGet: Struct value not yet witnessed (field '{fieldName}')"

        | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Memory nanopass - witnesses memory-related operations
let nanopass : Nanopass = {
    Name = "Memory"
    Witness = witnessMemory
}
