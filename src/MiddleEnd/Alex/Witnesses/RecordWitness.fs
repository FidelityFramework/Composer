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
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
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
        let structTy = mapType node.Type ctx |> narrowType ctx.Coeffects node.Id

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
                    match tryMatchWithDiagnostics (pBuildRecordCopyWith node.Id structTy origSSA fieldValues) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"RecordExpr copy-with: {diagnostic}"
                | None ->
                    WitnessOutput.error "RecordExpr copy-with: Original record not in accumulator"
            | None ->
                // Full construction: all field values provided
                match tryMatchWithDiagnostics (pBuildRecord node.Id structTy fieldValues) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | Result.Error diagnostic -> WitnessOutput.error $"RecordExpr: {diagnostic}"

    | None ->
        // Try TupleExpr on FPGA — tuples are first-class values (hw.struct_create)
        // StructuralWitness skips TupleExpr on FPGA so we handle it here.
        match node.Kind with
        | SemanticKind.TupleExpr elements when ctx.Coeffects.TargetPlatform = Core.Types.Dialects.FPGA ->
            // Map tuple type to TStruct with Item1, Item2, ... fields
            let structTy =
                (match node.Type with
                 | Clef.Compiler.NativeTypedTree.NativeTypes.NativeType.TTuple(elemTypes, _) ->
                     let fields = elemTypes |> List.mapi (fun i e -> (sprintf "Item%d" (i + 1), mapType e ctx))
                     TStruct fields
                 | _ -> mapType node.Type ctx)
                |> narrowType ctx.Coeffects node.Id

            // Recall each element's SSA from the accumulator (children already walked in post-order)
            let fieldValues =
                elements |> List.mapi (fun i elemId ->
                    let fieldName = sprintf "Item%d" (i + 1)
                    match MLIRAccumulator.recallNode elemId ctx.Accumulator with
                    | Some (ssa, ty) -> Some (fieldName, ssa, ty)
                    | None -> None)

            let resolved = fieldValues |> List.choose id
            if resolved.Length <> elements.Length then
                WitnessOutput.error $"TupleExpr: Only {resolved.Length} of {elements.Length} element values witnessed"
            else
                match tryMatchWithDiagnostics (pBuildRecord node.Id structTy resolved) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | Result.Error diagnostic -> WitnessOutput.error $"TupleExpr: {diagnostic}"
        | _ ->

        // Try FieldGet on TStruct
        match tryMatch pFieldGet ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((structId, fieldName), _) ->
            match MLIRAccumulator.recallNode structId ctx.Accumulator with
            | Some (structSSA, structTy) ->
                match structTy with
                | TStruct _ ->
                    // TStruct field access — handle here
                    match tryMatchWithDiagnostics (pRecordFieldGet node.Id structSSA fieldName structTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"RecordFieldGet: {diagnostic}"
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
