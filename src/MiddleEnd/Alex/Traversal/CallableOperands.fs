/// Typed transport for already settled callable occurrences. This projection
/// reads the source-owned physical boundary and each participant's carrier. It
/// never synthesizes a symbol, invents an environment, or packs a pair.
module Alex.Traversal.CallableOperands

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.NativeTypedTree.UnionFind
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators

type Shape = private {
    Contract: CallableBoundary
    FunctionType: MLIRType
    EnvironmentType: MLIRType option
    ParameterTypes: MLIRType list list
    ResultTypes: MLIRType list
}

let private collect results =
    match results |> List.tryPick (function Result.Error reason -> Some reason | _ -> None) with
    | Some reason -> Result.Error reason
    | None -> Result.Ok(results |> List.choose (function Result.Ok value -> Some value | _ -> None))

/// Read only source-published signature references. This recursion calculates
/// no expression, discovers no implementation and emits no operations. The
/// Huet traversal and E/P/W composition still witness every function body.
let rec private projectSeen (ctx: WitnessContext) seen occurrence : Result<Shape, string> =
    if Set.contains occurrence seen then Result.Error "Callable signature contains an unresolved recursive component reference."
    else
    let seen = Set.add occurrence seen
    match ctx.Graph.Codata.Value.CallableCarriers.TryFind occurrence,
          ctx.Graph.Codata.Value.CallableJoins.TryFind occurrence,
          ctx.Graph.Codata.Value.CallableFlows.TryFind occurrence with
    | None, None, None -> Result.Error "Callable occurrence has no settled carrier contract."
    | Some _, Some _, _ | Some _, _, Some _ | _, Some _, Some _ -> Result.Error "Callable occurrence has conflicting carrier contracts."
    | None, None, Some flow ->
        if flow.Occurrence <> occurrence || flow.Alternatives.IsEmpty ||
           not (Clef.Compiler.PSGSaturation.SemanticGraph.CallableFlows.validate ctx.Graph flow) then
            Result.Error "Callable flow no longer has its complete source argument, result and alias participants."
        else
            flow.Alternatives |> List.map (projectSeen ctx seen) |> collect |> Result.bind (fun alternatives ->
                let first = List.head alternatives
                if alternatives |> List.exists (fun alternative ->
                    alternative.FunctionType <> first.FunctionType || alternative.EnvironmentType <> first.EnvironmentType ||
                    alternative.ParameterTypes <> first.ParameterTypes || alternative.ResultTypes <> first.ResultTypes) then
                    Result.Error "Callable flow alternatives disagree with their settled common physical convention."
                else Result.Ok { first with Contract = Flow flow })
    | None, Some joined, None ->
        if joined.Occurrence <> occurrence || joined.Alternatives.IsEmpty ||
           not (Clef.Compiler.PSGSaturation.SemanticGraph.MutableCallableStorage.validateJoin ctx.Graph joined) then
            Result.Error "Callable join no longer has its complete source storage and read participants."
        else
            joined.Alternatives |> List.map (projectSeen ctx seen) |> collect |> Result.bind (fun alternatives ->
                let first = List.head alternatives
                if alternatives |> List.exists (fun alternative ->
                    alternative.FunctionType <> first.FunctionType || alternative.EnvironmentType <> first.EnvironmentType ||
                    alternative.ParameterTypes <> first.ParameterTypes || alternative.ResultTypes <> first.ResultTypes) then
                    Result.Error "Callable alternatives lack one settled physical parameter, result and environment convention."
                else
                    Result.Ok { first with Contract = Joined joined })
    | Some carrier, None, None when carrier.Occurrence <> occurrence -> Result.Error "Callable carrier names a different occurrence."
    | Some carrier, None, None ->
        let validContext = function
            | LambdaContext.RegularClosure -> true
            | LambdaContext.LazyThunk ->
                ctx.Graph.Codata.Value.LazyLayouts.Values |> Seq.exists (fun layout ->
                    layout.Thunk = carrier.Implementation &&
                    (LazyOperands.layout ctx layout.Owner |> Option.exists ((=) layout)))
            | _ -> false
        match ctx.Graph.Nodes.TryFind occurrence, ctx.Graph.Nodes.TryFind carrier.Implementation with
        | Some source, Some { Kind = SemanticKind.Lambda(parameters, body, [], _, context) }
            when validContext context &&
                 applySubst (Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.sourceType source) = applySubst carrier.SourceType &&
                 parameters = carrier.Parameters && body = carrier.Result ->
            let sourceShape id =
                ctx.Graph.Nodes.TryFind id |> Option.map (Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.valueShape ctx.Graph)
            let shapesAgree =
                (parameters |> List.map (fun (_, _, id) -> sourceShape id)) = (carrier.ParameterShapes |> List.map Some) &&
                sourceShape body = Some carrier.ResultShape
            if not shapesAgree then Result.Error "Callable component references do not match its exact formal and body participants."
            else
            let permitted = Clef.Compiler.PSGSaturation.SemanticGraph.CallableInstantiations.allowsSignatureData ctx.Graph carrier
            let parameterTypes = carrier.ParameterShapes |> List.map (componentsSeen ctx seen permitted) |> collect
            let resultTypes = componentsSeen ctx seen permitted carrier.ResultShape
            match parameterTypes, resultTypes with
            | Result.Error reason, _ | _, Result.Error reason -> Result.Error reason
            | Result.Ok groups, Result.Ok results ->
                let arguments = List.concat groups
                let environment =
                    match carrier.Environment, groups, parameters with
                    | None, _, _ -> Result.Ok None
                    | Some expected, [actual] :: _, (_, _, formal) :: _ when formal = expected.Formal ->
                        match ctx.Graph.Codata.Value.EnvironmentLayouts.TryFind expected.Owner,
                              ctx.Graph.Codata.Value.EnvironmentOrigins.TryFind occurrence,
                              ctx.Graph.Codata.Value.KnownCallables.TryFind occurrence with
                        | Some layout, Some owner, Some known
                            when owner = expected.Owner && known.EnvironmentOwner = owner &&
                                 known.Implementation = carrier.Implementation && layout.Implementation = carrier.Implementation &&
                                 layout.Formal = formal && not layout.Slots.IsEmpty &&
                                 actual = TMemRefStatic(layout.Bytes, TInt(IntWidth 8)) -> Result.Ok(Some actual)
                        | _ -> Result.Error "Callable occurrence no longer has its settled environment convention."
                    | _ -> Result.Error "Callable environment is not its first physical formal."
                environment |> Result.map (fun environment ->
                    { Contract = Exact carrier; FunctionType = TFunc(arguments, results); EnvironmentType = environment
                      ParameterTypes = groups; ResultTypes = results })
        | _ -> Result.Error "Callable carrier no longer agrees with its source and physical implementation."

and private componentsSeen (ctx: WitnessContext) seen permitted value : Result<MLIRType list, string> =
    match value with
    | CallableValueShape.Lazy occurrence ->
        LazyOperands.project ctx occurrence |> Result.map LazyOperands.componentTypes
    | CallableValueShape.Sequence occurrence ->
        SequenceOperands.project ctx occurrence |> Result.map SequenceOperands.componentTypes
    | CallableValueShape.Callable occurrence ->
        projectSeen ctx seen occurrence |> Result.map (fun shape -> shape.FunctionType :: Option.toList shape.EnvironmentType)
    | CallableValueShape.Data id ->
        match ctx.Graph.Nodes.TryFind id with
        | Some node ->
            match applySubst node.Type with
            | NativeType.TFun _ | NativeType.TForall _ -> Result.Error "Callable component cannot be read as a scalar data operand."
            | ty when hasUnboundVars ty ||
                      (not (List.isEmpty (freeMeasureVars ty)) &&
                       not (permitted id)) ->
                Result.Error "Callable signature data participant still has unresolved type or dimension variables."
            | _ ->
                try
                    let ty = mapTypeAt id node.Type ctx |> narrowType ctx.Coeffects ctx.Graph id
                    Result.Ok(if ty = TVoid then [] else [ty])
                with ex -> Result.Error ex.Message
        | None -> Result.Error "Callable signature data participant is absent."

let project ctx occurrence = projectSeen ctx Set.empty occurrence
let components ctx value = componentsSeen ctx Set.empty (fun _ -> false) value

/// Shared physical parameters retain their symbolic measure variables. Only
/// the current source-owned call instance can authorize that signature here.
let parametersAtCall (ctx: WitnessContext) site implementation parameters =
    if ctx.Zipper.Focus.Id <> site || not (obj.ReferenceEquals(ctx.Zipper.Focus, ctx.Graph.Nodes[site])) then
        Result.Error "Direct call signature projection requires its current Huet occurrence."
    else
    let shapes = parameters |> List.map (fun (_, ty, id) ->
        match ctx.Graph.Nodes.TryFind id with
        | Some node when applySubst node.Type = applySubst ty ->
            Result.Ok(Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.valueShape ctx.Graph node)
        | _ -> Result.Error "Direct call formal no longer has its declared source type.")
    shapes |> collect |> Result.bind (fun shapes ->
        let needsInstance = parameters |> List.exists (fun (_, ty, _) -> not (List.isEmpty (freeMeasureVars ty)))
        if not needsInstance then shapes |> List.map (components ctx) |> collect else
        match Clef.Compiler.PSGSaturation.SemanticGraph.CallableInstantiations.callReader ctx.Graph site implementation with
        | Some proof when proof.Parameters = parameters ->
            shapes |> List.map (componentsSeen ctx Set.empty proof.SignatureData.Contains) |> collect
        | _ -> Result.Error "Direct call lacks its exact source scheme instance and actual environment correspondence.")
let functionType shape = shape.FunctionType
let environmentType shape = shape.EnvironmentType
let parameterTypes shape = shape.ParameterTypes
let resultTypes shape = shape.ResultTypes

/// Baker's lazy code declaration is an actual Lambda, not a synthetic source
/// binding. Projection validates its current lazy layout and physical formals
/// before either a direct call or code-value occurrence may name its symbol.
let tryThunkDeclaration (ctx: WitnessContext) implementation =
    match ctx.Graph.Nodes.TryFind implementation with
    | Some ({ Kind = SemanticKind.Lambda(parameters, body, [], _, LambdaContext.LazyThunk) } as node) ->
        match project ctx implementation with
        | Result.Ok shape when (environmentType shape).IsNone ->
            Some(Alex.CodeGeneration.CallableSymbols.lambda ctx.Graph node false, parameters, body)
        | _ -> None
    | _ -> None

/// The caller supplies values it actually witnessed at this occurrence. Equal
/// implementation identities never justify replacing the environment operand.
let create (shape: Shape) (code: Val) (environment: Val option) : Result<CallableOperand, string> =
    if code.Type <> shape.FunctionType then Result.Error "Callable code does not match its settled physical function type."
    elif Option.map (fun (value: Val) -> value.Type) environment <> shape.EnvironmentType then
        Result.Error "Callable environment does not match its settled physical carrier."
    elif environment |> Option.exists (fun value -> value.SSA = code.SSA) then
        Result.Error "Callable code and environment must be distinct SSA operands."
    else Result.Ok { Carrier = shape.Contract; Code = code; Environment = environment }

let bind (ctx: WitnessContext) occurrence code environment =
    project ctx occurrence
    |> Result.bind (fun shape -> create shape code environment)
    |> Result.bind (fun value -> MLIRAccumulator.bindCallable occurrence value ctx.Accumulator)

/// Copy transport into another settled source occurrence. Both components come
/// from the original value; this performs no code-identity environment lookup.
let reproject (ctx: WitnessContext) source destination =
    match MLIRAccumulator.recallCallable source ctx.Accumulator with
    | None -> Result.Error "Source callable has not been witnessed in this operation scope."
    | Some value when value.Carrier.Occurrence <> source -> Result.Error "Source callable was recalled under another occurrence's identity."
    | Some value ->
        project ctx destination |> Result.bind (fun shape ->
            let sameExact (left: CallableCarrier) (right: CallableCarrier) =
                left.Implementation = right.Implementation && left.Environment = right.Environment
            let exactAlternatives = function
                | Exact carrier -> [carrier]
                | Flow flow -> flow.Alternatives |> List.choose ctx.Graph.Codata.Value.CallableCarriers.TryFind
                | Joined _ -> []
            let sameOrigin =
                match value.Carrier, shape.Contract with
                | Exact source, Exact destination ->
                    sameExact source destination
                | Joined source, Joined destination ->
                    source.Storage = destination.Storage && source.Read = destination.Read && source.Alternatives = destination.Alternatives
                | (Exact _ | Flow _), (Exact _ | Flow _) ->
                    let sources, destinations = exactAlternatives value.Carrier, exactAlternatives shape.Contract
                    not sources.IsEmpty && not destinations.IsEmpty &&
                    (sources |> List.forall (fun source -> destinations |> List.exists (sameExact source)))
                | _ -> false
            let sameSourceType =
                applySubst value.Carrier.SourceType = applySubst shape.Contract.SourceType ||
                Clef.Compiler.PSGSaturation.SemanticGraph.CallableInstantiations.canTransport ctx.Graph source destination
            if not sameOrigin || not sameSourceType then
                Result.Error "Callable copy does not preserve its settled code, environment owner, and source type."
            else create shape value.Code value.Environment)

let copy (ctx: WitnessContext) source destination =
    reproject ctx source destination
    |> Result.bind (fun value -> MLIRAccumulator.bindCallable destination value ctx.Accumulator)

let values (value: CallableOperand) = value.Code :: Option.toList value.Environment
let code (value: CallableOperand) = value.Code
let environment (value: CallableOperand) = value.Environment
let carrier (value: CallableOperand) = value.Carrier
let exactCarrier (value: CallableOperand) = match value.Carrier with Exact carrier -> Some carrier | Joined _ | Flow _ -> None
let cellDiscriminator (value: CallableCellOperand) = value.Discriminator
let cellEnvironment (value: CallableCellOperand) = value.Environment
let cellContract (value: CallableCellOperand) = value.Contract
