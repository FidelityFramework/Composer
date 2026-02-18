/// PlatformWitness - Witness platform I/O operations to MLIR via XParsec
///
/// Composes per-operation parsers with <|> — no dispatch hub.
/// Each parser self-checks via pIntrinsicApplication + ensure.
///
/// NANOPASS: Handles Sys.write and Sys.read intrinsic applications.
module Alex.Witnesses.PlatformWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.PlatformPatterns
open XParsec.Combinators  // <|>

let private witnessPlatform (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let combined = pSysWriteIntrinsic <|> pSysReadIntrinsic
    match tryMatch combined ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "Platform"; Witness = witnessPlatform }
