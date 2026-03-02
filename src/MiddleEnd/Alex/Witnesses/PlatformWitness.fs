/// PlatformWitness - Witness platform operations to MLIR via XParsec
///
/// Composes per-operation parsers with <|> — no dispatch hub.
/// Intrinsic parsers self-check via pIntrinsicApplication + ensure.
/// ExternCall parser guards on coeffect presence (Platform.Bindings).
///
/// NANOPASS: Handles Sys.write, Sys.read intrinsic applications
/// and [<FidelityExtern>] resolved calls (ExternCall in coeffects).
module Alex.Witnesses.PlatformWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.PlatformPatterns
open XParsec.Combinators  // <|>

let private witnessPlatform (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let combined = pSysWriteIntrinsic <|> pSysReadIntrinsic <|> pSysReadlineIntrinsic <|> pExternCallResolved
    match tryMatch combined ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "Platform"; Witness = witnessPlatform }
