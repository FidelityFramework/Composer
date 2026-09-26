/// Passive observation of Baker's explicit-demand expression relation.
module Alex.Witnesses.EagerWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.EagerPatterns

let private witness (ctx: WitnessContext) (node: SemanticNode) =
    match node.Kind with
    | SemanticKind.EagerExpr _ ->
        match tryMatchWithDiagnostics (pEagerValue ctx) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((operations, result), _) ->
            { InlineOps = operations; TopLevelOps = []; Result = result }
        | Result.Error message ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "eager") (Some "source demand occurrence") message
    | _ -> WitnessOutput.skip

let nanopass = { Name = "Eager"; Witness = witness }
