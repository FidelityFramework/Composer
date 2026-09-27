/// String operations observe Baker's complete construction. Descriptor extent
/// uses the shared memory Pattern; compound operations must be ordinary PSG
/// nodes before witnessing, with their storage, guards and encoding evidence.
module Alex.Witnesses.StringIntrinsicWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators

let private witnessStringIntrinsic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch (pIntrinsicApplication IntrinsicModule.String) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some ((info, _), _) ->
        match tryMatchWithDiagnostics Alex.Patterns.MemoryPatterns.pPublishedMemoryOperation ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((operations, declarations, result), _) ->
            { InlineOps = operations; TopLevelOps = declarations; Result = result }
        | Result.Error message ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "StringIntrinsic") (Some info.FullName)
                $"{info.FullName} at node {NodeId.value node.Id} lacks its complete source construction: {message}"

let nanopass : Nanopass = { Name = "StringIntrinsic"; Witness = witnessStringIntrinsic }
