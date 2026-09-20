/// DUPatterns - Codata-dependent discriminated union patterns
///
/// PUBLIC: DUWitness calls these to elide DU operations to MLIR.
/// Platform dispatch: CPU → MemoryPatterns (memref-based), FPGA → tag constants (combinational).
module Alex.Patterns.DUPatterns

open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Core.Types.Dialects         // TargetPlatform
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRAtomics  // pConstI
open Alex.Elements.HWElements  // pHWAggregateConstant
open Alex.Elements.MemRefElements
open Alex.CodeGeneration.TypeMapping
open Alex.Patterns.MemoryPatterns  // pDUCase, pExtractDUTag, pExtractDUPayload

/// Allocate only the caller residence and exact union layout settled by Baker.
/// The storage is not a value until an explicit DUInitialize selects its case.
let pBuildAggregateStorage (nodeId: NodeId) : PSGParser<MLIROp list * TransferResult> = parser {
    let! state = getUserState
    do! ensure (state.Graph.Codata.Value.Escapes.TryFind nodeId = Some EscapeKind.StackScoped) $"Aggregate storage {NodeId.value nodeId} has no admitted caller activation"
    let! bytes, alignment =
        match settledLayout state.Graph state.Current.Type with
        | Some(SettledLayout.Union(_, _, Some bytes, Some alignment)) when bytes > 0 && alignment > 0 -> preturn (bytes, alignment)
        | _ -> fail (Message $"Aggregate storage {NodeId.value nodeId} has no complete union layout")
    let result = Alex.Traversal.Values.value nodeId 0
    let! allocation = pAlloca result bytes (TInt(IntWidth 8)) (Some alignment)
    return [allocation], TRValue { SSA = result; Type = TMemRefStatic(bytes, TInt(IntWidth 8)) }
}

/// The graph supplies the selected case and destination. Only the active
/// payload is recalled; semantic branching belongs to Baker's copy recipe.
let pBuildDUInitialize (nodeId: NodeId) (destinationId: NodeId) (caseName: string) (caseIndex: int) (payloadId: NodeId option) : PSGParser<MLIROp list * TransferResult> = parser {
    let! state = getUserState
    let! destinationNode =
        match state.Graph.Nodes.TryFind destinationId with
        | Some node -> preturn node
        | None -> fail (Message "DU initialization has no destination node")
    let! destination, destinationType = pRecallNode destinationId
    let! cases, bytes =
        match settledLayout state.Graph destinationNode.Type with
        | Some(SettledLayout.Union(cases, _, Some bytes, Some alignment)) when bytes > 0 && alignment > 0 -> preturn (cases, bytes)
        | _ -> fail (Message "DU initialization requires a complete destination layout")
    do! ensure (destinationType = TMemRefStatic(bytes, TInt(IntWidth 8))) "DU destination disagrees with its settled byte carrier"
    do! ensure (caseIndex >= 0 && caseIndex < cases.Length) "DU initialization has no declared case"
    let expectedName, payloadSlot = cases[caseIndex]
    do! ensure (caseName = expectedName && payloadSlot.IsSome = payloadId.IsSome) "DU initialization disagrees with its selected case payload"
    let! adaptations, payload =
        match payloadId with
        | None -> preturn ([], [])
        | Some id -> parser {
            let! raw, rawType = pRecallNode id
            let! operations, value, valueType = pAdapt nodeId id raw rawType
            do! ensure (payloadSlot |> Option.bind settledScalarType = Some valueType) "DU initialization payload lacks the selected case's settled scalar carrier"
            return operations, [{ SSA = value; Type = valueType }]
          }
    let! writes, result = pDUCaseAt nodeId { SSA = destination; Type = destinationType } destinationNode.Type (int64 caseIndex) payload
    return adaptations @ writes, result
}

// ═══════════════════════════════════════════════════════════
// DU CONSTRUCT — Codata-dependent elision
// ═══════════════════════════════════════════════════════════

/// Build a DU value. CPU: memref alloc + tag + payload store. FPGA: tag constant (enum) or struct (payload).
let pBuildDUConstruct (nodeId: NodeId) (tag: int64) (payload: Val list) (duTy: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! targetPlatform = getTargetPlatform
        match targetPlatform with
        | FPGA ->
            match payload with
            | [] ->
                let! ssa = getNodeSSA nodeId
                match duTy with
                | TStruct _ ->
                    // Struct DU on FPGA (e.g. ValueNone): zero-initialized aggregate constant.
                    // Its payload fields take the widths of the payload type's FieldRanges
                    // (a field nothing constructs has the empty range, one bit).
                    let! state = getUserState
                    let narrowedTy = narrowType state.Coeffects state.Graph nodeId duTy
                    let! op = pHWAggregateConstant ssa narrowedTy
                    return ([op], TRValue { SSA = ssa; Type = narrowedTy })
                | _ ->
                    // Enum DU on FPGA: just a tag constant. Type is TTag which serializes to correct width.
                    let! op = pConstI ssa tag duTy
                    return ([op], TRValue { SSA = ssa; Type = duTy })
            | _ ->
                return! fail (Message "FPGA DU with payload not yet supported")
        | _ ->
            // CPU/MCU: memory-based DU construction
            return! pDUCase nodeId tag payload duTy
    }

// ═══════════════════════════════════════════════════════════
// DU GET TAG — Codata-dependent elision
// ═══════════════════════════════════════════════════════════

/// Extract DU tag. CPU: memref load at offset 0. FPGA: identity (enum value IS the tag).
let pBuildDUGetTag (nodeId: NodeId) (duSSA: SSA) (duType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! targetPlatform = getTargetPlatform
        match targetPlatform with
        | FPGA ->
            // Enum DU on FPGA: the value IS the tag — pass through
            let! ssa = getNodeSSA nodeId
            return ([], TRValue { SSA = ssa; Type = duType })
        | _ ->
            // CPU/MCU: memory-based tag extraction
            return! pExtractDUTag nodeId duSSA duType
    }

// ═══════════════════════════════════════════════════════════
// DU ELIMINATE — Codata-dependent elision
// ═══════════════════════════════════════════════════════════

/// Extract DU payload. CPU: memref view at the settled payload offset, then the read's own
/// width through its derived meet. FPGA: struct extract (future).
let pBuildDUEliminate (nodeId: NodeId) (duSSA: SSA) (duType: MLIRType) (unionNativeType: NativeType) (payloadType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! targetPlatform = getTargetPlatform
        match targetPlatform with
        | FPGA ->
            return! fail (Message "FPGA DU payload extraction not yet supported")
        | _ ->
            // CPU/MCU: memory-based payload extraction
            let! (ops, result) = pExtractDUPayload nodeId duSSA duType unionNativeType payloadType
            match result with
            | TRValue v ->
                let! (meetOps, readSSA, readTy) = pAdapt nodeId nodeId v.SSA v.Type
                return (ops @ meetOps, TRValue { SSA = readSSA; Type = readTy })
            | other -> return (ops, other)
    }
