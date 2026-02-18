/// ArithIntrinsicWitness - Witness arithmetic/conversion intrinsic operations
///
/// Composes per-operation parsers with <|> — no dispatch hub.
/// Each parser self-checks via pIntrinsicApplication + ensure.
///
/// NANOPASS: Handles Operators.* and Convert.* intrinsic applications.
module Alex.Witnesses.ArithIntrinsicWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ApplicationPatterns
open XParsec.Combinators  // <|>

let private witnessArithIntrinsic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let combined =
        pBinaryArithIntrinsic <|> pUnaryArithIntrinsic <|> pTypeConversionIntrinsic
    match tryMatch combined ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "ArithIntrinsic"; Witness = witnessArithIntrinsic }
