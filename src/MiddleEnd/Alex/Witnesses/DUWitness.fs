/// DUWitness - All-platform witness for discriminated union operations
///
/// Observes DUConstruct, DUGetTag, DUEliminate nodes.
/// Delegates to DUPatterns for codata-dependent elision (CPU: memref, FPGA: constants/structs).
///
/// NANOPASS: Registered for ALL platforms. Replaces DU handling formerly in MemoryWitness.
module Alex.Witnesses.DUWitness

open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId, NativeType
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core  // SemanticGraph.tryGetNode
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.DUPatterns

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

let private witnessDU (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Skip intrinsic nodes
    match node.Kind with
    | SemanticKind.Intrinsic _ -> WitnessOutput.skip
    | _ ->

    // DUGetTag — extract tag from DU value
    match tryMatch pDUGetTag ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((duValueId, _duType), _) ->
        match MLIRAccumulator.recallNode duValueId ctx.Accumulator with
        | Some (duSSA, duType) ->
            match tryMatchWithDiagnostics (pBuildDUGetTag node.Id duSSA duType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | Result.Error diagnostic -> WitnessOutput.error $"DUGetTag: {diagnostic}"
        | None -> WitnessOutput.error "DUGetTag: DU value not available"

    | None ->

    // DUEliminate — extract payload from DU value
    match tryMatch pDUEliminate ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((duValueId, caseIndex, _caseName, _payloadType), _) ->
        match MLIRAccumulator.recallNode duValueId ctx.Accumulator with
        | Some (duSSA, duType) ->
            // Derive payload type from scrutinee's DU type args, indexed by case.
            // CCS may leave the DUEliminate node's own type unresolved (TVar) when
            // the binding is unused — the scrutinee's TApp carries the concrete types.
            let payloadType =
                match SemanticGraph.tryGetNode duValueId ctx.Graph with
                | Some scrutineeNode ->
                    match scrutineeNode.Type with
                    | NativeType.TApp (tycon, typeArgs) when caseIndex < typeArgs.Length ->
                        mapType typeArgs.[caseIndex] ctx
                    | _ -> mapType node.Type ctx
                | None -> mapType node.Type ctx
            match tryMatchWithDiagnostics (pBuildDUEliminate node.Id duSSA duType caseIndex payloadType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | Result.Error diagnostic ->
                WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "DU") (Some "DUEliminate")
                    (sprintf "DUEliminate case %d: %s" caseIndex diagnostic)
        | None ->
            WitnessOutput.errorCoded AX2001 (Some node.Id) (Some "DU") (Some "DUEliminate")
                (sprintf "DU scrutinee node %d not yet witnessed" (NodeId.value duValueId))

    | None ->

    // DUConstruct — construct DU value
    match tryMatch pDUConstruct ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((_caseName, caseIndex, payloadOpt, _arenaHintOpt), _) ->
        let tag = int64 caseIndex
        let payload =
            match payloadOpt with
            | Some payloadId ->
                match MLIRAccumulator.recallNode payloadId ctx.Accumulator with
                | Some (ssa, ty) -> [{ SSA = ssa; Type = ty }]
                | None -> []
            | None -> []

        let duTy = mapType node.Type ctx

        match tryMatchWithDiagnostics (pBuildDUConstruct node.Id tag payload duTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | Result.Error diagnostic -> WitnessOutput.error $"DUConstruct: {diagnostic}"

    | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION
// ═══════════════════════════════════════════════════════════

let nanopass : Nanopass =
    {
        Name = "DUWitness"
        Witness = witnessDU
    }
