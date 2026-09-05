/// MemoryIntrinsicWitness - Witness Arena and Array intrinsic operations
///
/// Composes per-operation parsers with <|> — no dispatch hub.
/// Each parser self-checks via pIntrinsicApplication + ensure.
///
/// NANOPASS: Handles Arena.* and Array.* intrinsic applications.
module Alex.Witnesses.MemoryIntrinsicWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemoryPatterns
open XParsec.Combinators  // <|>

let private witnessMemoryIntrinsic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let combined =
        pArenaCreateIntrinsic <|> pArenaAllocIntrinsic
        <|> pArrayZeroCreateIntrinsic <|> pArrayGetIntrinsic <|> pArraySetIntrinsic <|> pArraySubIntrinsic
        <|> pArrayLengthIntrinsic <|> pArrayBlitIntrinsic
    match tryMatch combined ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "MemoryIntrinsic"; Witness = witnessMemoryIntrinsic }
