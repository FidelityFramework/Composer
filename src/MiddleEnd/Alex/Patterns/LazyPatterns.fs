/// Observe source-settled explicit lazy storage using ordinary typed Elements.
/// The source force graph already contains its guard, computation, cache store
/// and publication; these patterns never reconstruct that algorithm.
module Alex.Patterns.LazyPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ContinuationPatterns
open Alex.Patterns.MemoryPatterns
open Alex.Elements.FuncElements
open Alex.Elements.MemRefElements
open Alex.Elements.IndexElements
open Alex.Patterns.ControlFlowPatterns
module Operands = Alex.Traversal.LazyOperands
module Values = Alex.Traversal.Values

let isLazyValue (_ctx: WitnessContext) (node: SemanticNode) =
    match Clef.Compiler.NativeTypedTree.UnionFind.applySubst node.Type with
    | NativeType.TLazy _ -> true
    | _ -> false

let pRecallLazyEnvironment source (layout: LazyLayout) = parser {
    let! state = getUserState
    do! ensure (state.Graph.Codata.Value.LazyOrigins.TryFind source = Some layout.Owner)
            "Lazy environment operand has no exact source layout origin."
    let expected = TMemRefStatic(layout.Bytes, TInt(IntWidth 8))
    let! value, actual =
        match MLIRAccumulator.recallLazy source state.Accumulator with
        | Some value when value.Occurrence = source && value.Layout = layout -> preturn (value.Environment.SSA, value.Environment.Type)
        | Some _ -> fail (Message "Lazy environment was recalled under another occurrence or layout.")
        | None ->
            match state.Graph.Nodes.TryFind source with
            | Some { Type = NativeType.TLazy _ } -> fail (Message "Lazy value has not been witnessed in this operation scope.")
            | _ -> pRecallNode source
    do! ensure (actual = expected) "Lazy environment does not retain its settled typed descriptor."
    return [], TRValue { SSA = value; Type = expected }
}

let pAllocateLazyEnvironment nodeId (layout: LazyLayout) = parser {
    let! state = getUserState
    let expected = TMemRefStatic(layout.Bytes, TInt(IntWidth 8))
    let ssa = Values.value nodeId 0
    match state.Graph.Codata.Value.Escapes.TryFind nodeId with
    | Some EscapeKind.StackScoped ->
        let! operations, result = pAllocateContinuationStorage nodeId layout.Bytes layout.Alignment
        do! ensure (match result with TRValue value -> value.SSA = ssa && value.Type = expected | _ -> false)
                "Lazy allocation disagrees with the source-settled extent."
        return operations, result
    | Some EscapeKind.StaticLifetime ->
        let! allocation = pAllocValue nodeId ssa expected
        return [allocation], TRValue { SSA = ssa; Type = expected }
    | _ -> return! fail (Message "Lazy instance requires its own admitted storage residence.")
}

let pCreateLazyEnvironment nodeId (layout: LazyLayout) initializers = parser {
    let! state = getUserState
    let expected = TMemRefStatic(layout.Bytes, TInt(IntWidth 8))
    let! allocations, result =
        match state.Graph.Codata.Value.LazyDestinations.TryFind nodeId with
        | Some formal -> pRecallLazyEnvironment formal layout
        | None -> pAllocateLazyEnvironment nodeId layout
    let! environment =
        match result with
        | TRValue value when value.Type = expected -> preturn value.SSA
        | _ -> fail (Message "Lazy formation requires its actual destination descriptor.")
    let initialized = layout.Slots |> List.filter (fun slot -> slot.Source <> layout.Cached)
    do! ensure (initializers |> List.forall (fun (slot, _) -> slot <> layout.Cached))
            "The lazy cache cannot be initialized before its guarded first computation."
    let! stores = pInitializeEnvironmentSlots nodeId environment expected layout.Bytes initialized initializers
    return allocations @ stores, result
}

let pLazyValue occurrence (shape: Operands.Shape) symbol (environment: Val) = parser {
    let! state = getUserState
    do! ensure (state.Current.Id = occurrence && state.Zipper.Focus.Id = occurrence)
            "Lazy formation requires its actual Huet occurrence."
    let code = { SSA = Values.callableCode occurrence; Type = Operands.functionType shape }
    let! value =
        match Operands.create shape code environment with
        | Result.Ok value when value.Occurrence = occurrence -> preturn value
        | Result.Ok _ -> fail (Message "Lazy shape belongs to another occurrence.")
        | Result.Error reason -> fail (Message reason)
    let! operation = pFuncConstant code.SSA symbol code.Type
    return [operation], TRLazy value
}

let pLazyForward (ctx: WitnessContext) source = parser {
    let! state = getUserState
    do! ensure (state.Current.Id = ctx.Zipper.Focus.Id && obj.ReferenceEquals(state.Zipper, ctx.Zipper))
            "Lazy forwarding requires the current Huet occurrence."
    match Operands.reproject ctx source state.Current.Id with
    | Result.Ok value -> return [], TRLazy value
    | Result.Error reason -> return! fail (Message reason)
}

let private programInstance (ctx: WitnessContext) binding shape =
    match Clef.Compiler.Nanopass.LazyRuntime.programInstance ctx.Graph binding with
    | Some (owner, allocation) when owner = (Operands.contract shape).Owner -> preturn allocation
    | _ -> fail (Message "Program lazy reference lacks its exact initialized static instance and storage authority.")

let pProgramLazyBinding (ctx: WitnessContext) source = parser {
    let! shape =
        match Operands.project ctx ctx.Zipper.Focus.Id with
        | Result.Ok shape -> preturn shape
        | Result.Error reason -> fail (Message reason)
    let! _ = programInstance ctx ctx.Zipper.Focus.Id shape
    return! pLazyForward ctx source
}

/// Reload the descriptor for the one source-proved program allocation. This
/// names existing storage and code; it performs no formation or cache writes.
let pProgramLazyReference (ctx: WitnessContext) binding = parser {
    let! state = getUserState
    let occurrence = ctx.Zipper.Focus.Id
    do! ensure (state.Current.Id = occurrence && obj.ReferenceEquals(state.Zipper, ctx.Zipper))
            "Program lazy reference requires its actual Huet occurrence."
    let! shape =
        match Operands.project ctx occurrence with
        | Result.Ok shape -> preturn shape
        | Result.Error reason -> fail (Message reason)
    let! allocation = programInstance ctx binding shape
    let environment = { SSA = Values.value occurrence 0; Type = Operands.environmentType shape }
    let! load = pMemRefGetGlobal environment.SSA (staticValueName allocation) environment.Type
    let layout = Operands.contract shape
    let symbol = Alex.CodeGeneration.CallableSymbols.lambda ctx.Graph ctx.Graph.Nodes[layout.Thunk] false
    let! code, value = pLazyValue occurrence shape symbol environment
    return load :: code, value
}

/// Both operands leave the same selected source region. A common schema does
/// not permit replacing an arm's actual environment with another instance.
let private pLazySwitch (ctx: WitnessContext) selector cases otherwise = parser {
    let! state = getUserState
    let occurrence = state.Current.Id
    do! ensure (occurrence = ctx.Zipper.Focus.Id && obj.ReferenceEquals(state.Zipper, ctx.Zipper))
            "Lazy control transport requires its actual Huet occurrence."
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
    return operations, TRLazy value
}

let pLazyConditional (ctx: WitnessContext) (condition: Val)
                     thenId thenOps elseId elseOps = parser {
    do! ensure (condition.Type = TInt(IntWidth 1)) "Lazy conditional requires its witnessed Boolean condition."
    let selector = { SSA = Values.value ctx.Zipper.Focus.Id 2; Type = TIndex }
    let! cast = pIndexCastU selector.SSA condition.SSA condition.Type TIndex
    let! operations, result = pLazySwitch ctx selector [1L, thenId, thenOps] (elseId, elseOps)
    return cast :: operations, result
}

let pLazyDispatch (ctx: WitnessContext) selectorId cases otherwise = parser {
    let! state = getUserState
    let! selectorSSA, selectorType = pRecallNode selectorId
    let! prefix, selector =
        match selectorType with
        | TIndex -> preturn ([], { SSA = selectorSSA; Type = TIndex })
        | TInt(IntWidth width) when width > 0 ->
            let result = { SSA = Values.value ctx.Zipper.Focus.Id 2; Type = TIndex }
            let range = nodeRange state.Graph selectorId |> Option.defaultValue ValueRange.Unbounded
            let operation = indexCastForRange range result.SSA selectorSSA selectorType
            preturn ([operation], result)
        | _ -> fail (Message "Lazy dispatch selector lacks an admitted index carrier.")
    let branches = cases |> List.map (fun (label, source, operations) -> int64 label, source, operations)
    let! operations, result = pLazySwitch ctx selector branches otherwise
    return prefix @ operations, result
}
