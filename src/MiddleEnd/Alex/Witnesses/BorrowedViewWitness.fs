module Alex.Witnesses.BorrowedViewWitness

open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.BorrowedViewPatterns

let private witness (ctx: WitnessContext) (node: Clef.Compiler.PSGSaturation.SemanticGraph.Types.SemanticNode) =
    match Clef.Compiler.PSGSaturation.SemanticGraph.BorrowedViews.operation ctx.Graph node.Id with
    | None -> WitnessOutput.skip
    | Some _ ->
        match tryMatchWithDiagnostics pBorrowedViewIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | Result.Error message -> WitnessOutput.error message

let nanopass : Nanopass = { Name = "BorrowedView"; Witness = witness }
