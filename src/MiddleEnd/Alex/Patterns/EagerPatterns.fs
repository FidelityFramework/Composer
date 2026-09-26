/// Observe the value of an explicit demand at its already selected source
/// occurrence. The zipper owns traversal; this pattern neither activates a
/// deferred body nor repeats any operand operations.
module Alex.Patterns.EagerPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.NativeTypedTree.UnionFind
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
open Alex.Patterns.CallablePatterns
open Alex.Patterns.SequencePatterns
open Alex.Patterns.LazyPatterns
open Alex.Patterns.LiteralPatterns
module Zipper = Alex.Traversal.PSGZipper
module ExplicitDemand = Clef.Compiler.PSGSaturation.SemanticGraph.ExplicitDemand

let pEagerValue (ctx: WitnessContext) = parser {
    let! state = getUserState
    do! ensure (obj.ReferenceEquals(state.Zipper, ctx.Zipper)
                && obj.ReferenceEquals(ctx.Graph, ctx.Zipper.Graph)
                && state.Current.Id = ctx.Zipper.Focus.Id)
            "Explicit demand must retain its actual zipper occurrence."
    let! operand =
        match ExplicitDemand.operand ctx.Graph state.Current.Id with
        | Some operand -> preturn operand
        | None -> fail (Message "Explicit demand has no current typed source operand.")
    let rows = ctx.Graph.Edges |> List.filter (fun edge ->
        edge.Target = state.Current.Id &&
        match edge.Role with EdgeRole.EagerDemand _ | EdgeRole.EagerDemandPending -> true | _ -> false)
    do! ensure (match rows with
                | [{ Class = EdgeClass.Demand; Role = EdgeRole.EagerDemand EagerFrontier.Expression
                     Ordinal = 0; Sources = [marker; actual] }] -> marker = state.Current.Id && actual = operand
                | _ -> false)
            "Explicit demand has no unique current expression frontier."
    if isLazyValue ctx state.Current then
        return! pLazyForward ctx operand
    elif isSequenceValue ctx state.Current then
        return! pSequenceForward ctx operand
    else
        match applySubst state.Current.Type with
        | NativeType.TFun _ -> return! pCallableForward ctx operand
        | _ ->
            match MLIRAccumulator.recallNode operand state.Accumulator with
            | Some(value, valueType) ->
                let! operations, result, resultType = pAdapt state.Current.Id operand value valueType
                return operations, TRValue { SSA = result; Type = resultType }
            | None when applySubst state.Current.Type = Types.unitType ->
                let completed =
                    Zipper.down 0 ctx.Zipper |> Option.exists (fun child ->
                        child.Focus.Id = operand && MLIRAccumulator.completedVoid child ctx.ScopeContext state.Accumulator)
                do! ensure completed "Explicit unit demand requires its successfully witnessed child result in this exact operation scope."
                return! pWithUnitResult state.Current.Id (preturn ([], TRVoid))
            | None ->
                let! _ = pRecallNode operand
                return! fail (Message "Explicit demand operand has no scalar result.")
}
