/// Passive observation of Baker's ordered, always-active requirement.
module Alex.Witnesses.RequirementWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.RequirementPatterns

let private witness (ctx: WitnessContext) (node: SemanticNode) =
    match node.Kind with
    | SemanticKind.Require _ ->
        match tryMatchWithDiagnostics (pRequirement ctx) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((operations, result), _) ->
            { InlineOps = operations; TopLevelOps = []; Result = result }
        | Result.Error message ->
            WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "Require") (Some "ordered source contract") message
    | _ -> WitnessOutput.skip

let nanopass = { Name = "Require"; Witness = witness }
