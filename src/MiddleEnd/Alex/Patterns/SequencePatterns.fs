/// Passive physical transport of an admitted sequence protocol. The source
/// projector determines data versus pair roles; no pattern selects an origin.
module Alex.Patterns.SequencePatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
open Alex.Elements.IndexElements
open Alex.Patterns.ControlFlowPatterns
module Operands = Alex.Traversal.SequenceOperands
module Values = Alex.Traversal.Values

let isSequenceValue (ctx: WitnessContext) (node: SemanticNode) =
    match Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.valueShape ctx.Graph node with
    | CallableValueShape.Sequence _ -> true
    | _ -> false

let private atOccurrence (ctx: WitnessContext) = parser {
    let! state = getUserState
    do! ensure (state.Current.Id = ctx.Zipper.Focus.Id && obj.ReferenceEquals(state.Zipper, ctx.Zipper))
            "Sequence transport requires its actual Huet occurrence."
    return state.Current.Id
}

let pSequenceForward (ctx: WitnessContext) source : PSGParser<MLIROp list * TransferResult> = parser {
    let! occurrence = atOccurrence ctx
    match Operands.reproject ctx source occurrence with
    | Result.Ok value -> return [], TRSequence value
    | Result.Error reason -> return! fail (Message reason)
}

/// One structured switch yields both actual components from its selected arm.
/// The case list and bodies already belong to the settled source control node.
let private pSequenceSwitch (ctx: WitnessContext) selector cases otherwise = parser {
    let! occurrence = atOccurrence ctx
    let! shape =
        match Operands.project ctx occurrence with
        | Result.Ok value -> preturn value
        | Result.Error reason -> fail (Message reason)
    let arm (source, operations) = parser {
        match Operands.reproject ctx source occurrence with
        | Result.Ok value -> return operations, Operands.values value
        | Result.Error reason -> return! fail (Message reason)
    }
    let! branches =
        cases |> List.map (fun (label, source, operations) -> parser {
            let! body = arm (source, operations)
            return label, body
        }) |> Alex.XParsec.Extensions.sequence
    let! fallback = arm otherwise
    let code = { SSA = Values.value occurrence 0; Type = Operands.functionType shape }
    let environment = { SSA = Values.value occurrence 1; Type = Operands.environmentType shape }
    let! value =
        match Operands.create shape code environment with
        | Result.Ok value -> preturn value
        | Result.Error reason -> fail (Message reason)
    let! operations = pBuildIndexSwitch selector branches fallback [code; environment]
    return operations, TRSequence value
}

/// Boolean control is expressed as a portable index switch so its two result
/// components share the same selected region. This does not inspect branch bodies.
let pSequenceConditional (ctx: WitnessContext) (condition: Val)
                         thenId thenOps elseId elseOps : PSGParser<MLIROp list * TransferResult> = parser {
    let! occurrence = atOccurrence ctx
    do! ensure (condition.Type = TInt(IntWidth 1)) "Sequence conditional requires its witnessed Boolean condition."
    let selector = { SSA = Values.value occurrence 2; Type = TIndex }
    let! cast = pIndexCastU selector.SSA condition.SSA condition.Type TIndex
    let! operations, result = pSequenceSwitch ctx selector [1L, thenId, thenOps] (elseId, elseOps)
    return cast :: operations, result
}

let pSequenceDispatch (ctx: WitnessContext) selectorId cases otherwise : PSGParser<MLIROp list * TransferResult> = parser {
    let! occurrence = atOccurrence ctx
    let! state = getUserState
    let! selectorSSA, selectorType = pRecallNode selectorId
    let! prefix, selector =
        match selectorType with
        | TIndex -> preturn ([], { SSA = selectorSSA; Type = TIndex })
        | TInt(IntWidth width) when width > 0 ->
            let result = { SSA = Values.value occurrence 2; Type = TIndex }
            let range = nodeRange state.Graph selectorId |> Option.defaultValue ValueRange.Unbounded
            let operation = Alex.Patterns.MemoryPatterns.indexCastForRange range result.SSA selectorSSA selectorType
            preturn ([operation], result)
        | _ -> fail (Message "Sequence dispatch selector lacks an admitted index carrier.")
    let branches = cases |> List.map (fun (label, source, operations) -> int64 label, source, operations)
    let! operations, result = pSequenceSwitch ctx selector branches otherwise
    return prefix @ operations, result
}
