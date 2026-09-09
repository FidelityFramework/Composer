module Alex.Witnesses.MmioWitness
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MmioPatterns
let private witness (ctx: WitnessContext) (node: Clef.Compiler.PSGSaturation.SemanticGraph.Types.SemanticNode) =
    match Clef.Compiler.PSGSaturation.SemanticGraph.Mmio.operation ctx.Graph node.Id with
    | None -> WitnessOutput.skip
    | Some _ ->
        match tryMatchWithDiagnostics pMmioIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | Result.Error message -> WitnessOutput.error message
let nanopass : Nanopass = { Name = "Mmio"; Witness = witness }
