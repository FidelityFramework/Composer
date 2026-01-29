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

/// Witness platform operations - category-selective (handles only Sys.* syscall nodes)
let private witnessPlatform (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pClassifiedAtomicOp ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((info, category), _) ->
        match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
        | None -> WitnessOutput.error "Platform: No SSA assigned"
        | Some resultSSA ->
            match category with
            | PlatformOp op ->
                // Platform operations have specific arity requirements
                // Sys.write(fd: int, buffer: nativeptr<byte>, count: int) -> int
                // Sys.read(fd: int, buffer: nativeptr<byte>, count: int) -> int
                match op, node.Children with
                | "write", [fdId; bufferPtrId; countId] ->
                    match MLIRAccumulator.recallNode fdId ctx.Accumulator,
                          MLIRAccumulator.recallNode bufferPtrId ctx.Accumulator,
                          MLIRAccumulator.recallNode countId ctx.Accumulator with
                    | Some (fdSSA, _), Some (bufferSSA, _), Some (countSSA, _) ->
                        match tryMatch (pSysWrite resultSSA fdSSA bufferSSA countSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "Sys.write pattern emission failed"
                    | _ -> WitnessOutput.error $"Sys.write operands not yet witnessed"

                | "read", [fdId; bufferPtrId; countId] ->
                    match MLIRAccumulator.recallNode fdId ctx.Accumulator,
                          MLIRAccumulator.recallNode bufferPtrId ctx.Accumulator,
                          MLIRAccumulator.recallNode countId ctx.Accumulator with
                    | Some (fdSSA, _), Some (bufferSSA, _), Some (countSSA, _) ->
                        match tryMatch (pSysRead resultSSA fdSSA bufferSSA countSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "Sys.read pattern emission failed"
                    | _ -> WitnessOutput.error $"Sys.read operands not yet witnessed"

                | _ -> WitnessOutput.skip  // Other platform operations not yet implemented

            | _ -> WitnessOutput.skip  // Not a platform operation

    | None -> WitnessOutput.skip  // Not an atomic operation

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Platform nanopass - witnesses Sys.* syscall operations
let nanopass : Nanopass = {
    Name = "Platform"
    Phase = ContentPhase
    Witness = witnessPlatform
}
