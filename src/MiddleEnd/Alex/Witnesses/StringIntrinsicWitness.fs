/// StringIntrinsicWitness - Witness String intrinsic operations
///
/// A String.* intrinsic application belongs to this witness. The operation
/// selects its own per-operation parser, which either witnesses the settled
/// application or reports the premise it lacks; the refusal is never discarded
/// as a skip.
///
/// NANOPASS: Handles String.* intrinsic applications.
module Alex.Witnesses.StringIntrinsicWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.StringPatterns

let private witnessStringIntrinsic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch (pIntrinsicApplication IntrinsicModule.String) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some ((info, _), _) ->
        let pattern =
            match info.Operation with
            | "length" -> Some pStringLengthIntrinsic
            | "charAt" -> Some pStringCharAtIntrinsic
            | "concat2" -> Some pStringConcat2Intrinsic
            | "contains" -> Some pStringContainsIntrinsic
            | "fromBytes" -> Some pStringFromBytesIntrinsic
            | "toBytes" -> Some pStringToBytesIntrinsic
            | _ -> None
        match pattern with
        | Some pattern ->
            match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | Result.Error message ->
                WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "StringIntrinsic") (Some info.FullName)
                    $"{info.FullName} at node {NodeId.value node.Id}: {message}"
        | None ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "StringIntrinsic") (Some info.FullName)
                $"Baker did not settle {info.FullName} at node {NodeId.value node.Id} into a witnessed string operation: no String witness pattern exists for it"

let nanopass : Nanopass = { Name = "StringIntrinsic"; Witness = witnessStringIntrinsic }
