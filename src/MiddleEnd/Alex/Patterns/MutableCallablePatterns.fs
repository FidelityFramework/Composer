/// Witness the source-owned finite mutable callable protocol. Only the
/// discriminator and actual environment descriptor are data. Reads construct a
/// function value through the already settled dispatch; no code address is
/// packed, stored or recovered through a numeric conversion.
module Alex.Patterns.MutableCallablePatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
open Alex.Elements.MemRefElements
open Alex.Elements.IndexElements
open Alex.Elements.FuncElements
open Alex.Patterns.ControlFlowPatterns
module Values = Alex.Traversal.Values
module Operands = Alex.Traversal.CallableOperands
module Storage = Clef.Compiler.PSGSaturation.SemanticGraph.MutableCallableStorage

let private pStorage binding = parser {
    let! state = getUserState
    match state.Graph.Codata.Value.MutableCallableStorage.TryFind binding with
    | Some storage when Storage.validate state.Graph storage -> return storage
    | _ -> return! fail (Message "Mutable callable storage lacks its complete source protocol.")
}

let private pValue (ctx: WitnessContext) (storage: MutableCallableStorage) (write: MutableCallableWrite) = parser {
    let! state = getUserState
    let! value =
        match MLIRAccumulator.recallCallable write.Value state.Accumulator with
        | Some value -> preturn value
        | None -> fail (Message "Initial or assigned callable value has not been witnessed.")
    let! expected =
        match state.Graph.Codata.Value.CallableCarriers.TryFind write.Value, Operands.exactCarrier value with
        | Some expected, Some actual when expected = actual &&
                                        List.tryItem write.Alternative storage.Alternatives = Some write.Value -> preturn expected
        | _ -> fail (Message "Callable write does not match its exact source alternative.")
    let! shape =
        match Operands.project ctx write.Value with
        | Result.Ok shape -> preturn shape
        | Result.Error reason -> fail (Message reason)
    do! ensure ((Operands.code value).Type = Operands.functionType shape &&
                Option.map (fun (value: Val) -> value.Type) (Operands.environment value) = Operands.environmentType shape)
                "Callable write lost its exact physical operands."
    // This first retained-environment admission is program-long storage. Local
    // retained views require the complete covering proof, never a stack label.
    do! ensure (expected.Environment |> Option.forall (fun environment ->
        match state.Graph.Nodes.TryFind environment.Owner with
        | Some { Kind = SemanticKind.ClosureValue(_, allocation) }
            when not (state.Graph.Codata.Value.EnvironmentDestinations.ContainsKey allocation) ->
                state.Graph.Codata.Value.Escapes.TryFind allocation = Some EscapeKind.StaticLifetime
        | _ -> false))
            "Mutable callable environment retention requires a covering source lifetime proof."
    return value
}

let private pWrite nodeId (cell: CallableCellOperand) (write: MutableCallableWrite) (value: CallableOperand) = parser {
    let zero, tag = Values.value nodeId 2, Values.value nodeId 3
    let! zeroOp = pIndexConst zero 0L
    let! tagOp = pIndexConst tag (int64 write.Alternative)
    let! environmentOps =
        match cell.Environment, Operands.environment value with
        | None, None -> preturn []
        | Some destination, Some environment -> parser {
            let! store = pStore environment.SSA destination.SSA [zero] environment.Type destination.Type
            return [store]
          }
        | _ -> fail (Message "Mutable callable write has an unmatched environment component.")
    let! tagStore = pStore tag cell.Discriminator.SSA [zero] TIndex cell.Discriminator.Type
    return [zeroOp; tagOp] @ environmentOps @ [tagStore]
}

let pCreateMutableCallable (ctx: WitnessContext) binding = parser {
    let! state = getUserState
    do! ensure (state.Current.Id = binding && state.Zipper.Focus.Id = binding)
            "Mutable callable allocation requires its actual source occurrence."
    let! storage = pStorage binding
    do! ensure (state.Graph.Codata.Value.Escapes.TryFind binding = Some EscapeKind.StackScoped)
            "Mutable callable cell requires an admitted local allocation lifetime."
    do! ensure storage.Captures.IsEmpty
            "Captured mutable callable storage requires its shared protocol descriptor and covering residence proof."
    let! value = pValue ctx storage storage.Initializer
    let cell =
        { Contract = storage; Discriminator = { SSA = Values.value binding 0; Type = TMemRefStatic(1, TIndex) }
          Environment = storage.EnvironmentBytes |> Option.map (fun bytes ->
              { SSA = Values.value binding 1; Type = TMemRefStatic(1, TMemRefStatic(bytes, TInt(IntWidth 8))) }) }
    let! allocation = pAlloca cell.Discriminator.SSA 1 TIndex None
    let! environmentOps =
        match cell.Environment with
        | Some { SSA = ssa; Type = TMemRefStatic(1, element) } -> parser {
            let! allocation = pAlloca ssa 1 element None
            return [allocation]
          }
        | None -> preturn []
        | _ -> fail (Message "Mutable callable environment cell lost its descriptor representation.")
    let! initialization = pWrite binding cell storage.Initializer value
    return allocation :: environmentOps @ initialization, TRCallableCell cell
}

let pAssignMutableCallable (ctx: WitnessContext) binding assignment = parser {
    let! state = getUserState
    do! ensure (state.Current.Id = assignment && state.Zipper.Focus.Id = assignment)
            "Mutable callable write requires its actual source event."
    let! storage = pStorage binding
    let! write =
        match storage.Writes |> List.filter (fun write -> write.Site = assignment) with
        | [write] -> preturn write
        | _ -> fail (Message "Mutable callable assignment has no unique settled write event.")
    let! cell =
        match MLIRAccumulator.recallCallableCell binding state.Accumulator with
        | Some cell when cell.Contract = storage -> preturn cell
        | _ -> fail (Message "Mutable callable target cell has not been witnessed in this scope.")
    let! value = pValue ctx storage write
    let! operations = pWrite assignment cell write value
    return operations, TRVoid
}

let pReadMutableCallable (ctx: WitnessContext) binding occurrence = parser {
    let! state = getUserState
    do! ensure (state.Current.Id = occurrence && state.Zipper.Focus.Id = occurrence)
            "Mutable callable read requires its actual source snapshot frontier."
    let! storage = pStorage binding
    do! ensure (storage.Reads.Contains occurrence) "This occurrence is not an admitted mutable callable read."
    let! cell =
        match MLIRAccumulator.recallCallableCell binding state.Accumulator with
        | Some cell when cell.Contract = storage -> preturn cell
        | _ -> fail (Message "Mutable callable cell has not been witnessed in this scope.")
    let! shape =
        match Operands.project ctx occurrence with
        | Result.Ok shape -> preturn shape
        | Result.Error reason -> fail (Message reason)
    let zero = Values.value occurrence 0
    let selector = { SSA = Values.value occurrence 1; Type = TIndex }
    let code = { SSA = Values.callableCode occurrence; Type = Operands.functionType shape }
    let environment = Operands.environmentType shape |> Option.map (fun ty -> { SSA = Values.value occurrence 2; Type = ty })
    let! zeroOp = pIndexConst zero 0L
    let! tagLoad = pLoad selector.SSA cell.Discriminator.SSA [zero]
    let! environmentOps =
        match cell.Environment, environment with
        | Some source, Some value -> parser {
            let! load = pLoad value.SSA source.SSA [zero]
            return [load]
          }
        | None, None -> preturn []
        | _ -> fail (Message "Mutable callable read lost its matched environment descriptor.")
    let! alternatives =
        storage.Alternatives |> List.mapi (fun alternative source -> parser {
            let! carrier =
                match state.Graph.Codata.Value.CallableCarriers.TryFind source with
                | Some carrier -> preturn carrier
                | None -> fail (Message "Mutable callable dispatch lost an exact source alternative.")
            let implementation = state.Graph.Nodes[carrier.Implementation]
            let symbol = Alex.CodeGeneration.CallableSymbols.lambda state.Graph implementation false
            let value = { SSA = Values.callableAlternative occurrence alternative; Type = code.Type }
            let! constant = pFuncConstant value.SSA symbol value.Type
            return int64 alternative, ([constant], [value])
        }) |> Alex.XParsec.Extensions.sequence
    // Initialization and every admitted write supply one listed tag. The last
    // alternative is the default of that closed source dispatch, not an unknown
    // callable fallback. Added writes invalidate pStorage before commitment.
    let! dispatch = pBuildIndexSwitch selector (alternatives |> List.take (alternatives.Length - 1))
                                       (alternatives |> List.last |> snd) [code]
    let! value =
        match Operands.create shape code environment with
        | Result.Ok value -> preturn value
        | Result.Error reason -> fail (Message reason)
    return [zeroOp; tagLoad] @ environmentOps @ dispatch, TRCallable value
}
