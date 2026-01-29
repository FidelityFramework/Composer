/// MemoryWitness - Witness memory and data structure operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to MemoryPatterns for MLIR elision.
///
/// NANOPASS: This witness handles memory-related operations.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.MemoryWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemoryPatterns
open Alex.Patterns.ElisionPatterns

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness memory operations - category-selective (DU operations and MemoryOp atomic operations)
let private witnessMemory (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // First try MemoryOp atomic operations (NativePtr.*)
    match tryMatch pClassifiedAtomicOp ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((info, category), _) ->
        match category with
        | MemoryOp op ->
            match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
            | None -> WitnessOutput.error "MemoryOp: No SSA assigned"
            | Some resultSSA ->
                match op with
                | "stackalloc" ->
                    // NativePtr.stackalloc<'T>() : nativeptr<'T>
                    // No children - just allocates and returns pointer
                    match tryMatch (pNativePtrStackAlloc resultSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "NativePtr.stackalloc pattern emission failed"

                | "write" ->
                    // NativePtr.write (ptr: nativeptr<'T>) (value: 'T) : unit
                    match node.Children with
                    | [ptrId; valueId] ->
                        match MLIRAccumulator.recallNode ptrId ctx.Accumulator,
                              MLIRAccumulator.recallNode valueId ctx.Accumulator with
                        | Some (ptrSSA, _), Some (valueSSA, _) ->
                            match tryMatch (pNativePtrWrite valueSSA ptrSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "NativePtr.write pattern emission failed"
                        | _ -> WitnessOutput.error "NativePtr.write: Operands not yet witnessed"
                    | _ -> WitnessOutput.error $"NativePtr.write: Expected 2 children, got {node.Children.Length}"

                | "read" ->
                    // NativePtr.read (ptr: nativeptr<'T>) : 'T
                    match node.Children with
                    | [ptrId] ->
                        match MLIRAccumulator.recallNode ptrId ctx.Accumulator with
                        | Some (ptrSSA, _) ->
                            match tryMatch (pNativePtrRead resultSSA ptrSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "NativePtr.read pattern emission failed"
                        | None -> WitnessOutput.error "NativePtr.read: Pointer not yet witnessed"
                    | _ -> WitnessOutput.error $"NativePtr.read: Expected 1 child, got {node.Children.Length}"

                | _ -> WitnessOutput.skip  // Other memory operations not yet implemented

        | _ -> WitnessOutput.skip  // Not a MemoryOp

    | None ->
        // Not an atomic operation, try DU operations
        match tryMatch pDUGetTag ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some ((duValueId, _duType), _) ->
            match MLIRAccumulator.recallNode duValueId ctx.Accumulator, SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
            | Some (duSSA, duType), Some tagSSA ->
                match tryMatch (pExtractDUTag duSSA duType tagSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = tagSSA; Type = TInt I8 } }
                | None -> WitnessOutput.error "DUGetTag pattern emission failed"
            | _ -> WitnessOutput.error "DUGetTag: DU value or tag SSA not available"

        | None ->
            match tryMatch pDUEliminate ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
            | Some ((duValueId, caseIndex, _caseName, _payloadType), _) ->
                match MLIRAccumulator.recallNode duValueId ctx.Accumulator, SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
                | Some (duSSA, duType), Some ssas ->
                    match tryMatch (pExtractDUPayload duSSA duType caseIndex duType ssas) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = List.last ssas; Type = duType } }
                    | None -> WitnessOutput.error "DUEliminate pattern emission failed"
                | _ -> WitnessOutput.error "DUEliminate: DU value or SSAs not available"

            | None ->
                match tryMatch pDUConstruct ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                | Some ((_caseName, caseIndex, payloadOpt, _arenaHintOpt), _) ->
                    match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
                    | None -> WitnessOutput.error "DUConstruct: No SSAs assigned"
                    | Some ssas ->
                        let tag = int64 caseIndex
                        let payload =
                            match payloadOpt with
                            | Some payloadId ->
                                match MLIRAccumulator.recallNode payloadId ctx.Accumulator with
                                | Some (ssa, ty) -> [{ SSA = ssa; Type = ty }]
                                | None -> []
                            | None -> []

                        let arch = ctx.Coeffects.Platform.TargetArch
                        let duTy = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                        match tryMatch (pDUCase tag payload ssas duTy) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                        | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRValue { SSA = List.last ssas; Type = duTy } }
                        | None -> WitnessOutput.error "DUConstruct pattern emission failed"

                | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Memory nanopass - witnesses memory-related operations
let nanopass : Nanopass = {
    Name = "Memory"
    Phase = ContentPhase
    Witness = witnessMemory
}
