/// DUPatterns - Codata-dependent discriminated union patterns
///
/// PUBLIC: DUWitness calls these to elide DU operations to MLIR.
/// Platform dispatch: CPU → MemoryPatterns (memref-based), FPGA → tag constants (combinational).
module Alex.Patterns.DUPatterns

open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Core.Types.Dialects         // TargetPlatform
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRAtomics  // pConstI
open Alex.Elements.HWElements  // pHWAggregateConstant
open Alex.Patterns.MemoryPatterns  // pDUCase, pExtractDUTag, pExtractDUPayload

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
                    // Dead data — tag guarantees value bits are never read.
                    // Clamp any unresolved IntWidth 0 to IntWidth 1 (minimum hw register width).
                    let clampedTy = clampZeroWidths duTy
                    let! op = pHWAggregateConstant ssa clampedTy
                    return ([op], TRValue { SSA = ssa; Type = clampedTy })
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

/// Extract DU payload. CPU: memref view at offset. FPGA: struct extract (future).
let pBuildDUEliminate (nodeId: NodeId) (duSSA: SSA) (duType: MLIRType) (caseIndex: int) (payloadType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! targetPlatform = getTargetPlatform
        match targetPlatform with
        | FPGA ->
            return! fail (Message "FPGA DU payload extraction not yet supported")
        | _ ->
            // CPU/MCU: memory-based payload extraction
            return! pExtractDUPayload nodeId duSSA duType caseIndex payloadType
    }
