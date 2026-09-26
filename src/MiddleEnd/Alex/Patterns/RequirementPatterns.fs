/// Witness an always-active source requirement. Baker supplies the condition,
/// diagnostic and ordered continuation; this pattern supplies no failure policy.
module Alex.Patterns.RequirementPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.CFElements
open Alex.Patterns.LiteralPatterns
module Requirements = Clef.Compiler.PSGSaturation.SemanticGraph.Requirements

let pRequirement (ctx: WitnessContext) = parser {
    let! state = getUserState
    do! ensure (obj.ReferenceEquals(state.Zipper, ctx.Zipper)
                && obj.ReferenceEquals(ctx.Graph, ctx.Zipper.Graph)
                && state.Current.Id = ctx.Zipper.Focus.Id)
            "Requirement must be witnessed at its actual zipper occurrence."
    let! contract =
        match Requirements.tryRequirement ctx.Graph state.Current.Id with
        | Some contract -> preturn contract
        | None -> fail (Message "Requirement has no current ordered source contract.")
    do! ensure (match ctx.Zipper.Path with
                | step :: _ -> step.Parent.Id = contract.Frontier
                               && step.LeftSiblings.IsEmpty
                               && step.RightSiblings = [contract.Continuation]
                | [] -> false)
            "Requirement is outside its admitted source frontier."
    let! condition, conditionType = pRecallNode contract.Condition
    do! ensure (conditionType = TInt(IntWidth 1)) "Requirement condition must retain its Boolean carrier."
    let! operation = pAssert condition contract.Diagnostic
    return! pWithUnitResult contract.Site (preturn ([operation], TRVoid))
}
