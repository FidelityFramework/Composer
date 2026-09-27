/// MemoryIntrinsicWitness - Witness Arena and Array intrinsic operations
///
/// An Arena.* or Array.* intrinsic application belongs to this witness. The
/// operation selects its own per-operation parser, which either witnesses the
/// settled application or reports the premise it lacks; the refusal is never
/// discarded as a skip.
///
/// NANOPASS: Handles Arena.* and Array.* intrinsic applications.
module Alex.Witnesses.MemoryIntrinsicWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemoryPatterns
open XParsec.Combinators  // <|>

let private witnessMemoryIntrinsic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let owned = pIntrinsicApplication IntrinsicModule.Arena <|> pIntrinsicApplication IntrinsicModule.Array
    match tryMatch owned ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some ((info, _), _) ->
        let pattern =
            match info.Module, info.Operation with
            | IntrinsicModule.Arena, "create" -> Some pArenaCreateIntrinsic
            | IntrinsicModule.Arena, "alloc" -> Some pArenaAllocIntrinsic
            | IntrinsicModule.Array, "zeroCreate" -> Some pArrayZeroCreateIntrinsic
            | IntrinsicModule.Array, "get" -> Some pArrayGetIntrinsic
            | IntrinsicModule.Array, "set" -> Some pArraySetIntrinsic
            | IntrinsicModule.Array, "sub" -> Some pArraySubIntrinsic
            | IntrinsicModule.Array, "length" -> Some pArrayLengthIntrinsic
            | IntrinsicModule.Array, "blit" -> Some pArrayBlitIntrinsic
            | _ -> None
        match pattern with
        | Some pattern ->
            match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | Result.Error message ->
                WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "MemoryIntrinsic") (Some info.FullName)
                    $"{info.FullName} at node {NodeId.value node.Id}: {message}"
        | None ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "MemoryIntrinsic") (Some info.FullName)
                $"Baker did not settle {info.FullName} at node {NodeId.value node.Id} into a witnessed memory operation: no Arena/Array witness pattern exists for it"

let nanopass : Nanopass = { Name = "MemoryIntrinsic"; Witness = witnessMemoryIntrinsic }
