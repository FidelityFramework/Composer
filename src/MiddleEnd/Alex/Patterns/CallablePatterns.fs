/// Physical callable values retain distinct code and environment operands.
/// Patterns observe the current occurrence and already settled carrier; they
/// neither traverse implementation bodies nor reconstruct source semantics.
module Alex.Patterns.CallablePatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
open Alex.Elements.FuncElements
module Operands = Alex.Traversal.CallableOperands
module Values = Alex.Traversal.Values

/// The witness supplies the already resolved implementation symbol and the
/// environment value it actually witnessed. No environment is selected by code.
let pCallableValue occurrence (shape: Operands.Shape) symbol (environment: Val option) : PSGParser<MLIROp list * TransferResult> = parser {
    let! state = getUserState
    do! ensure (state.Current.Id = occurrence && state.Zipper.Focus.Id = occurrence)
            "Callable value must be witnessed at its actual Huet occurrence."
    let code = { SSA = Values.callableCode occurrence; Type = Operands.functionType shape }
    let! callable =
        match Operands.create shape code environment with
        | Result.Ok value when (Operands.carrier value).Occurrence = occurrence -> preturn value
        | Result.Ok _ -> fail (Message "Callable shape belongs to a different occurrence.")
        | Result.Error reason -> fail (Message reason)
    let! operation = pFuncConstant code.SSA symbol code.Type
    return [operation], TRCallable callable
}

/// Passive transport reprojects the destination contract but preserves both
/// actual source operands. Binding the result remains the traversal's job.
let pCallableForward (ctx: WitnessContext) source : PSGParser<MLIROp list * TransferResult> = parser {
    let! state = getUserState
    do! ensure (state.Current.Id = ctx.Zipper.Focus.Id && obj.ReferenceEquals(state.Zipper, ctx.Zipper))
            "Callable forwarding requires the current Huet occurrence."
    match Operands.reproject ctx source state.Current.Id with
    | Result.Ok value -> return [], TRCallable value
    | Result.Error reason -> return! fail (Message reason)
}

/// A named code declaration has no runtime environment. Its value occurrence
/// receives an ordinary func.constant with the settled physical signature.
let pNamedCallable (ctx: WitnessContext) : PSGParser<MLIROp list * TransferResult> = parser {
    let! state = getUserState
    do! ensure (state.Current.Id = ctx.Zipper.Focus.Id && obj.ReferenceEquals(state.Zipper, ctx.Zipper))
            "Named callable requires the current Huet occurrence."
    let! shape =
        match Operands.project ctx state.Current.Id with
        | Result.Ok shape when (Operands.environmentType shape).IsNone -> preturn shape
        | Result.Ok _ -> fail (Message "A capturing callable requires its actual witnessed environment.")
        | Result.Error reason -> fail (Message reason)
    let carrier = state.Graph.Codata.Value.CallableCarriers[state.Current.Id]
    let implementation = state.Graph.Nodes[carrier.Implementation]
    let! symbol =
        match implementation.Parent |> Option.bind (fun id -> state.Graph.Nodes.TryFind id) with
        | Some ({ Kind = SemanticKind.Binding(_, false, _, _); Children = [code] } as binding) when code = implementation.Id ->
            match Alex.CodeGeneration.CallableSymbols.tryBinding state.Graph binding.Id with
            | Some symbol -> preturn symbol
            | None -> fail (Message "Callable implementation has no resolved declaration symbol.")
        | _ -> fail (Message "Callable implementation is not an admitted named code declaration.")
    return! pCallableValue state.Current.Id shape symbol None
}
