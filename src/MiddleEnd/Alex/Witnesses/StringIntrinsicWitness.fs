/// StringIntrinsicWitness - Witness String/NativeStr intrinsic operations
///
/// Composes per-operation parsers with <|> — no dispatch hub.
/// Each parser self-checks via pIntrinsicApplication + ensure.
///
/// NANOPASS: Handles String.* and NativeStr.* intrinsic applications.
module Alex.Witnesses.StringIntrinsicWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.StringPatterns
open XParsec.Combinators  // <|>

let private witnessStringIntrinsic (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let combined =
        pStringLengthIntrinsic <|> pStringCharAtIntrinsic
        <|> pStringConcat2Intrinsic <|> pStringContainsIntrinsic
        <|> pNativeStrFromPointerIntrinsic
    match tryMatch combined ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "StringIntrinsic"; Witness = witnessStringIntrinsic }
