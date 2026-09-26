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
    Contract: CallableCarrier
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
    match ctx.Graph.Codata.Value.CallableCarriers.TryFind occurrence with
    | None -> Result.Error "Callable occurrence has no settled carrier contract."
    | Some carrier when carrier.Occurrence <> occurrence -> Result.Error "Callable carrier names a different occurrence."
    | Some carrier ->
        match ctx.Graph.Nodes.TryFind occurrence, ctx.Graph.Nodes.TryFind carrier.Implementation with
        | Some source, Some { Kind = SemanticKind.Lambda(parameters, body, [], _, LambdaContext.RegularClosure) }
            when applySubst (Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.sourceType source) = applySubst carrier.SourceType &&
                 parameters = carrier.Parameters && body = carrier.Result ->
            let sourceShape id =
                ctx.Graph.Nodes.TryFind id |> Option.map Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.valueShape
            let shapesAgree =
                (parameters |> List.map (fun (_, _, id) -> sourceShape id)) = (carrier.ParameterShapes |> List.map Some) &&
                sourceShape body = Some carrier.ResultShape
            if not shapesAgree then Result.Error "Callable component references do not match its exact formal and body participants."
            else
            let parameterTypes = carrier.ParameterShapes |> List.map (componentsSeen ctx seen) |> collect
            let resultTypes = componentsSeen ctx seen carrier.ResultShape
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
                    { Contract = carrier; FunctionType = TFunc(arguments, results); EnvironmentType = environment
                      ParameterTypes = groups; ResultTypes = results })
        | _ -> Result.Error "Callable carrier no longer agrees with its source and physical implementation."

and private componentsSeen (ctx: WitnessContext) seen value : Result<MLIRType list, string> =
    match value with
    | CallableValueShape.Callable occurrence ->
        projectSeen ctx seen occurrence |> Result.map (fun shape -> shape.FunctionType :: Option.toList shape.EnvironmentType)
    | CallableValueShape.Data id ->
        match ctx.Graph.Nodes.TryFind id with
        | Some node ->
            match applySubst node.Type with
            | NativeType.TFun _ | NativeType.TForall _ -> Result.Error "Callable component cannot be read as a scalar data operand."
            | ty when hasUnboundVars ty || not (List.isEmpty (freeMeasureVars ty)) ->
                Result.Error "Callable signature data participant still has unresolved type or dimension variables."
            | _ ->
                try
                    let ty = mapTypeAt id node.Type ctx |> narrowType ctx.Coeffects ctx.Graph id
                    Result.Ok(if ty = TVoid then [] else [ty])
                with ex -> Result.Error ex.Message
        | None -> Result.Error "Callable signature data participant is absent."

let project ctx occurrence = projectSeen ctx Set.empty occurrence
let components ctx value = componentsSeen ctx Set.empty value
let functionType shape = shape.FunctionType
let environmentType shape = shape.EnvironmentType
let parameterTypes shape = shape.ParameterTypes
let resultTypes shape = shape.ResultTypes

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
            if value.Carrier.Implementation <> shape.Contract.Implementation ||
               value.Carrier.Environment <> shape.Contract.Environment ||
               applySubst value.Carrier.SourceType <> applySubst shape.Contract.SourceType then
                Result.Error "Callable copy does not preserve its settled code, environment owner, and source type."
            else create shape value.Code value.Environment)

let copy (ctx: WitnessContext) source destination =
    reproject ctx source destination
    |> Result.bind (fun value -> MLIRAccumulator.bindCallable destination value ctx.Accumulator)

let values (value: CallableOperand) = value.Code :: Option.toList value.Environment
let code (value: CallableOperand) = value.Code
let environment (value: CallableOperand) = value.Environment
let carrier (value: CallableOperand) = value.Carrier
