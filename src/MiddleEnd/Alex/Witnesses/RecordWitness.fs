/// RecordWitness - Witness record construction and TStruct field access via XParsec
///
/// Leaf witness handling two node kinds:
/// 1. RecordExpr — construct a record from field values
/// 2. FieldGet on TStruct — extract a named field from a record
///
/// Must be registered BEFORE MemoryWitness so it intercepts TStruct FieldGets.
/// Non-TStruct FieldGets (strings, closures, DUs) return skip for MemoryWitness.
///
/// Codata-dependent: CPU → memref byte-offset ops, FPGA → hw.struct_* ops
module Alex.Witnesses.RecordWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.RecordPatterns

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

let private witnessRecord (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Try RecordExpr first
    match tryMatch pRecordExpr ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((fields, copyFrom), _) ->
        // RecordExpr: field value nodes are already walked in post-order.
        // Recall each field value SSA from accumulator.
        let structTy = mapType node.Type ctx

        let fieldValues =
            fields |> List.choose (fun (fieldName, fieldNodeId) ->
                match MLIRAccumulator.recallNode fieldNodeId ctx.Accumulator with
                | Some (fieldSSA, fieldType) -> Some (fieldName, fieldSSA, fieldType)
                | None -> None)

        if fieldValues.Length <> fields.Length then
            WitnessOutput.error $"RecordExpr: Only {fieldValues.Length} of {fields.Length} field values witnessed"
        else
            match copyFrom with
            | Some origId ->
                // Copy-and-update: recall original record SSA, delegate to pBuildRecordCopyWith
                match MLIRAccumulator.recallNode origId ctx.Accumulator with
                | Some (origSSA, _origTy) ->
                    match tryMatch (pBuildRecordCopyWith node.Id structTy origSSA fieldValues) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "RecordExpr copy-with: pBuildRecordCopyWith pattern failed"
                | None ->
                    WitnessOutput.error "RecordExpr copy-with: Original record not in accumulator"
            | None ->
                // Full construction: all field values provided
                match tryMatch (pBuildRecord node.Id structTy fieldValues) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | None -> WitnessOutput.error "RecordExpr: pBuildRecord pattern failed"

    | None ->
        // Try FieldGet on TStruct
        match tryMatch pFieldGet ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((structId, fieldName), _) ->
            match MLIRAccumulator.recallNode structId ctx.Accumulator with
            | Some (structSSA, structTy) ->
                match structTy with
                | TStruct _ ->
                    // TStruct field access — handle here
                    match tryMatch (pRecordFieldGet node.Id structSSA fieldName structTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error $"RecordFieldGet: pRecordFieldGet failed for field '{fieldName}'"
                | _ ->
                    // Not TStruct — skip, let MemoryWitness handle (strings, closures, DUs)
                    WitnessOutput.skip
            | None ->
                // Struct value not yet in accumulator — skip
                WitnessOutput.skip
        | None ->
            WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Record nanopass - witnesses RecordExpr and TStruct FieldGet nodes
let nanopass : Nanopass = {
    Name = "Record"
    Witness = witnessRecord
}
