/// MemoryIntrinsicWitness - Witness Arena and Array intrinsic operations
///
/// Array intrinsics use the same published operation as the Memory witness.
/// No alternate array allocator, bounds construction or copying loop lives here.
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
    | Some ((({ Module = IntrinsicModule.Array } as info), _), _) ->
        match tryMatchWithDiagnostics pPublishedMemoryOperation ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((operations, declarations, result), _) ->
            { InlineOps = operations; TopLevelOps = declarations; Result = result }
        | Result.Error message ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "MemoryIntrinsic") (Some info.FullName)
                $"{info.FullName} at node {NodeId.value node.Id}: {message}"
    | Some ((info, _), _) ->
        let pattern =
            match info.Module, info.Operation with
            | IntrinsicModule.Arena, "create" -> Some pArenaCreateIntrinsic
            | IntrinsicModule.Arena, "alloc" -> Some pArenaAllocIntrinsic
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
