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
open Alex.Patterns.LiteralPatterns

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

let private witnessDU (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.AggregateStorage _ ->
        match tryMatchWithDiagnostics (pBuildAggregateStorage node.Id) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | Result.Error diagnostic -> WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "DU") (Some "aggregate storage") diagnostic
    | SemanticKind.DUInitialize(destination, caseName, caseIndex, payload) ->
        let pattern = pWithUnitResult node.Id (pBuildDUInitialize node.Id destination caseName caseIndex payload)
        match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | Result.Error diagnostic -> WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "DU") (Some "destination initialization") diagnostic
    | _ ->
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
    | Some ((duValueId, caseIndex, _caseName, psgPayloadType), _) ->
        match MLIRAccumulator.recallNode duValueId ctx.Accumulator with
        | Some (duSSA, duType) ->
            let unionNativeType =
                match SemanticGraph.tryGetNode duValueId ctx.Graph with
                | Some scrutinee -> scrutinee.Type
                | None -> node.Type
            // Use the payload type from the PSG node (CCS-resolved).
            // Fall back to the DUEliminate node's own type only if PSG type is TVar
            // (which happens when the binding is unused and CCS didn't resolve it).
            let payloadType =
                let mapped =
                    match psgPayloadType with
                    | NativeType.TVar _ ->
                        // TVar — try to derive from scrutinee's type args
                        match SemanticGraph.tryGetNode duValueId ctx.Graph with
                        | Some scrutineeNode ->
                            match scrutineeNode.Type with
                            | NativeType.TApp (_, typeArgs) when typeArgs.Length > 0 ->
                                // Use min(caseIndex, length-1) since case index doesn't always
                                // equal type arg index (e.g. Option.Some=case1 but typeArgs[0])
                                mapType typeArgs.[min caseIndex (typeArgs.Length - 1)] ctx
                            | _ -> mapType node.Type ctx
                        | None -> mapType node.Type ctx
                    | _ -> mapType psgPayloadType ctx |> narrowType ctx.Coeffects ctx.Graph node.Id
                mapped
            match tryMatchWithDiagnostics (pBuildDUEliminate node.Id duSSA duType unionNativeType payloadType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
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
        // the payload at its slot's width (its derived meet)
        let (meetOps, payload) =
            match payloadOpt with
            | Some payloadId ->
                match MLIRAccumulator.recallNode payloadId ctx.Accumulator with
                | Some (ssa, ty) ->
                    let (ops, adapted, adaptedTy) = adaptOperand ctx.Coeffects ctx.Graph node.Id payloadId ssa ty
                    (ops, [{ SSA = adapted; Type = adaptedTy }])
                | None -> ([], [])
            | None -> ([], [])

        let duTy = mapType node.Type ctx

        match tryMatchWithDiagnostics (pBuildDUConstruct node.Id tag payload duTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((ops, result), _) -> { InlineOps = meetOps @ ops; TopLevelOps = []; Result = result }
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
