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
open Alex.Elements.MemRefElements
module Operands = Alex.Traversal.CallableOperands
module Values = Alex.Traversal.Values

/// An annotation around a directly applied intrinsic carries source type
/// information, not a first-class function operand. Follow only transparent
/// annotation positions; the application witness owns the intrinsic operation.
let pIntrinsicCalleeAnnotation : PSGParser<MLIROp list * TransferResult> = parser {
    let! state = getUserState
    let rec callee position =
        match Alex.Traversal.PSGZipper.up position with
        | Some parent ->
            match parent.Focus.Kind with
            | SemanticKind.Application(target, _) -> target = position.Focus.Id
            | SemanticKind.TypeAnnotation(inner, _) when inner = position.Focus.Id -> callee parent
            | _ -> false
        | None -> false
    let rec intrinsic seen id =
        if Set.contains id seen then false
        else
            match state.Graph.Nodes.TryFind id with
            | Some { Kind = SemanticKind.Intrinsic _ } -> true
            | Some { Kind = SemanticKind.TypeAnnotation(inner, _) } -> intrinsic (Set.add id seen) inner
            | _ -> false
    do! ensure (state.Current.Id = state.Zipper.Focus.Id && callee state.Zipper && intrinsic Set.empty state.Current.Id)
            "Annotation does not occupy an applied intrinsic's transparent callee position."
    return [], TRVoid
}

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

let private pProgramInstance (ctx: WitnessContext) binding = parser {
    let! state = getUserState
    do! ensure (obj.ReferenceEquals(state.Graph, ctx.Graph)
                && obj.ReferenceEquals(ctx.Graph, ctx.Zipper.Graph)
                && (ctx.Graph.Nodes.TryFind state.Current.Id
                    |> Option.exists (fun current -> obj.ReferenceEquals(state.Current, current))))
            "Program callable evidence must belong to the current graph occurrence."
    let! instance =
        match Clef.Compiler.Nanopass.ClosureEnvironmentSettlement.programInstance state.Graph binding with
        | Some instance -> preturn instance
        | None -> fail (Message "Program callable lacks its exact initialized instance and static storage authority.")
    let! shape =
        match Operands.project ctx state.Current.Id with
        | Result.Ok shape -> preturn shape
        | Result.Error reason -> fail (Message reason)
    do! ensure (state.Graph.Codata.Value.CallableCarriers.TryFind state.Current.Id
                |> Option.exists (fun carrier ->
                    carrier.Implementation = instance.Carrier.Implementation &&
                    carrier.Environment = instance.Carrier.Environment))
            "Program callable occurrence disagrees with its source-owned implementation and environment."
    return instance, shape
}

/// The initializer has already been witnessed by the zipper. Validate its
/// source-owned program instance, then retain that actual pair unchanged.
let pProgramCallableBinding (ctx: WitnessContext) source = parser {
    let! state = getUserState
    do! ensure (state.Current.Children = [source])
            "Program callable binding lacks its actual initializer occurrence."
    let! _ = pProgramInstance ctx state.Current.Id
    return! pCallableForward ctx source
}

/// A reference names the initialized source-proved allocation and code. The
/// pattern composes only descriptor access and the callable value elements.
let pProgramCallableReference (ctx: WitnessContext) binding = parser {
    let! state = getUserState
    let occurrence = state.Current.Id
    do! ensure (occurrence = ctx.Zipper.Focus.Id && obj.ReferenceEquals(state.Zipper, ctx.Zipper))
            "Program callable reference requires its actual Huet occurrence."
    do! ensure (match state.Current.Kind with SemanticKind.VarRef(_, Some source) -> source = binding | _ -> false)
            "Program callable reference lacks its actual source binding."
    let! instance, shape = pProgramInstance ctx binding
    let! operations, environment =
        match instance.Allocation, Operands.environmentType shape with
        | Some allocation, Some ty -> parser {
            let value = { SSA = Values.value occurrence 0; Type = ty }
            let! load = pMemRefGetGlobal value.SSA (Alex.Patterns.MemoryPatterns.staticValueName allocation) ty
            return [load], Some value
          }
        | None, None -> preturn ([], None)
        | _ -> fail (Message "Program callable instance and physical environment convention disagree.")
    let code = state.Graph.Nodes[instance.Carrier.Implementation]
    let symbol = Alex.CodeGeneration.CallableSymbols.lambda state.Graph code false
    let! callableOps, value = pCallableValue occurrence shape symbol environment
    return operations @ callableOps, value
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
        | _ ->
            match Operands.tryThunkDeclaration ctx implementation.Id with
            | Some (symbol, _, _) -> preturn symbol
            | None -> fail (Message "Callable implementation is not an admitted named code declaration.")
    return! pCallableValue state.Current.Id shape symbol None
}
