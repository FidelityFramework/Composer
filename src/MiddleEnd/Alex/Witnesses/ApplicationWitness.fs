/// Application witnesses read settled call boundaries and already witnessed
/// operands at the current Huet occurrence. Function bodies remain the common
/// traversal's responsibility; Patterns compose the physical invocation.
module Alex.Witnesses.ApplicationWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.NativeTypedTree.UnionFind
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ApplicationPatterns
open Alex.Dialects.Core.Types
module Operands = Alex.Traversal.CallableOperands

/// Read the declaration's actual lambda boundary, without visiting its body.
let private declaration (ctx: WitnessContext) binding =
    Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryCallable ctx.Graph
    |> Result.toOption |> Option.bind (fun projection -> projection.Declarations.TryFind binding)
    |> Option.filter (fun declaration -> declaration.Captures.IsEmpty)
    |> Option.map (fun declaration -> declaration.Parameters, declaration.Result, declaration.Implementation)

let private failure (node: SemanticNode) phase message =
    WitnessOutput.errorCoded AX2001 (Some node.Id) (Some "Application") (Some phase) message

/// Logical arguments retain their actual operands. Callable values expand from
/// their witnessed carrier and never enter the scalar type mapping.
let private arguments (ctx: WitnessContext) call sources =
    let readings =
        sources |> List.map (fun source ->
            match MLIRAccumulator.recallCallable source ctx.Accumulator with
            | Some callable -> Result.Ok([], Operands.values callable)
            | None ->
                match MLIRAccumulator.recallSequence source ctx.Accumulator with
                | Some sequence -> Result.Ok([], Alex.Traversal.SequenceOperands.values sequence)
                | None ->
                match MLIRAccumulator.recallLazy source ctx.Accumulator with
                | Some lazyValue -> Result.Ok([], Alex.Traversal.LazyOperands.values lazyValue)
                | None ->
                match Operands.valueShape ctx source with
                | Result.Error reason -> Result.Error reason
                | Result.Ok(CallableValueShape.Callable _ | CallableValueShape.Lazy _ | CallableValueShape.Sequence _) ->
                    Result.Error $"Callable argument {NodeId.value source} has no witnessed operands"
                | _ ->
                    match MLIRAccumulator.recallNode source ctx.Accumulator with
                    | Some (ssa, ty) ->
                        let operations, value, actual = adaptOperand ctx.Coeffects ctx.Graph call source ssa ty
                        Result.Ok(operations, [{ SSA = value; Type = actual }])
                    | None -> Result.Error $"Argument {NodeId.value source} has not been witnessed")
    match readings |> List.tryPick (function Result.Error reason -> Some reason | _ -> None) with
    | Some reason -> Result.Error reason
    | None ->
        let values = readings |> List.choose (function Result.Ok value -> Some value | _ -> None)
        Result.Ok(List.collect fst values, List.collect snd values)

let private observe (ctx: WitnessContext) (node: SemanticNode) prefix pattern =
    match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Result.Error reason -> failure node "physical invocation" reason
    | Result.Ok ((operations, result), _) ->
        match result with
        | TRValue value ->
            let meets, ssa, ty = adaptOperand ctx.Coeffects ctx.Graph node.Id node.Id value.SSA value.Type
            { InlineOps = prefix @ operations @ meets; TopLevelOps = []; Result = TRValue { SSA = ssa; Type = ty } }
        | _ -> { InlineOps = prefix @ operations; TopLevelOps = []; Result = result }

let private invoke (ctx: WitnessContext) (node: SemanticNode) invocation sources body names environment expected =
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryOrdinary ctx.Graph with
    | Result.Error reason -> failure node "source demand projection" reason
    | Result.Ok demand ->
        let projection = demand.Calls.TryFind node.Id
        let transported =
            match projection with
            | Some proof when proof.Actuals = sources ->
                sources |> List.indexed |> List.choose (fun (ordinal, source) -> if proof.Omitted.Contains ordinal then None else Some source) |> Result.Ok
            | Some _ -> Result.Error "Call transport disagrees with its exact source demand projection"
            | None -> Result.Ok sources
        match transported |> Result.bind (arguments ctx node.Id) with
        | Result.Error reason -> failure node "argument operands" reason
        | Result.Ok (meets, operands) ->
            let actuals = Option.toList environment @ operands
            let agrees = expected |> Option.forall (fun types -> types = List.map (fun (value: Val) -> value.Type) actuals)
            if not agrees then failure node "parameter boundary" "Direct invocation operands disagree with its settled physical parameter components"
            else
            let prefix = meets
            match Operands.valueShape ctx node.Id with
            | Result.Error reason -> failure node "result boundary" reason
            | Result.Ok(CallableValueShape.Callable _) ->
                match Operands.project ctx node.Id with
                | Result.Error reason -> failure node "callable result" reason
                | Result.Ok shape -> observe ctx node prefix (pCallableApplication node.Id invocation actuals shape)
            | Result.Ok(CallableValueShape.Sequence _) ->
                match Alex.Traversal.SequenceOperands.project ctx node.Id with
                | Result.Error reason -> failure node "sequence result" reason
                | Result.Ok shape -> observe ctx node prefix (pSequenceApplication node.Id invocation actuals shape)
            | Result.Ok(CallableValueShape.Lazy _) ->
                match Alex.Traversal.LazyOperands.project ctx node.Id with
                | Result.Error reason -> failure node "lazy result" reason
                | Result.Ok shape -> observe ctx node prefix (pLazyApplication node.Id invocation actuals shape)
            | _ ->
                match invocation, body with
                | Direct symbol, Some body ->
                    // The call's result is held at its declaration body's settled width.
                    let resultType = mapTypeAt node.Id node.Type ctx |> narrowType ctx.Coeffects ctx.Graph body
                    observe ctx node prefix (pDirectCall node.Id symbol (actuals |> List.map (fun value -> value.SSA, value.Type)) resultType names)
                | Direct _, None ->
                    failure node "result boundary"
                        $"Baker callable projection did not settle the declaration body for direct call {NodeId.value node.Id}: its result width has no settled source"
                | Indirect code, _ ->
                    match code.Type with
                    | TFunc(_, [actualResult]) -> observe ctx node prefix (pIndirectApplication node.Id code actuals actualResult)
                    | _ -> failure node "result boundary" "Scalar application requires exactly one result in its witnessed function signature"

let private direct (ctx: WitnessContext) (node: SemanticNode) binding (sources: NodeId list) =
    let declared =
        match Alex.CodeGeneration.CallableSymbols.tryBinding ctx.Graph binding, declaration ctx binding with
        | Some symbol, Some(parameters, body, implementation) -> Some(symbol, parameters, body, implementation)
        | _ -> Operands.tryThunkDeclaration ctx binding |> Option.map (fun (symbol, parameters, body) -> symbol, parameters, body, binding)
    match declared, Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryOrdinary ctx.Graph with
    | _, Result.Error reason -> failure node "source demand projection" reason
    | Some(symbol, parameters, body, implementation), Result.Ok demand ->
        if parameters.Length <> sources.Length then
            failure node "declared boundary" "Settled direct call does not supply its actual declared parameters"
        else
            let names =
                parameters |> List.collect (fun (name, _, formal) ->
                    if (demand.Parameters.TryFind implementation |> Option.defaultValue Set.empty).Contains formal then [] else
                    match Operands.valueShape ctx formal with
                    | Result.Ok(CallableValueShape.Callable _) ->
                        match Operands.project ctx formal with
                        | Result.Ok shape -> (name + "_code") :: (if (Operands.environmentType shape).IsSome then [name + "_environment"] else [])
                        | Result.Error _ -> [name] // the component reading below reports the absent boundary
                    | Result.Ok(CallableValueShape.Sequence _) -> [name + "_pull"; name + "_environment"]
                    | Result.Ok(CallableValueShape.Lazy _) -> [name + "_thunk"; name + "_environment"]
                    | _ -> [name])
            match Operands.parametersAtCall ctx node.Id implementation parameters with
            | Result.Error reason -> failure node "parameter boundary" reason
            | Result.Ok components ->
                let expected = List.concat components
                invoke ctx node (Direct symbol) sources (Some body) (Some names) None (Some expected)
    | _ -> failure node "declaration identity" "Direct call lacks its settled declaration symbol and actual lambda boundary"

let private witnessApplication (ctx: WitnessContext) (node: SemanticNode) =
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryCallable ctx.Graph with
    | Result.Error reason -> failure node "source callable projection" reason
    | Result.Ok projection ->
    // Foreign admission and its explicit ABI remain PlatformWitness's concern.
    if Map.containsKey node.Id ctx.Coeffects.Platform.Bindings.Bindings
       || projection.ForeignCalls.Contains node.Id then WitnessOutput.skip
    else
    match tryMatch pApplication ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some ((callee, supplied), _) ->
        let curry = ctx.Graph.Codata.Value.Curry
        match curry.SaturatedCalls.TryFind node.Id with
        | Some call -> direct ctx node call.TargetBindingId call.AllArgNodes
        | None when curry.PartialApplications.ContainsKey node.Id ->
            { InlineOps = []; TopLevelOps = []; Result = TRVoid }
        | None ->
            match MLIRAccumulator.recallCallable callee ctx.Accumulator with
            | Some callable ->
                invoke ctx node (Indirect(Operands.code callable)) supplied None None (Operands.environment callable) None
            | None ->
                match projection.IntrinsicAliases.Contains callee, projection.DirectCallees.TryFind callee with
                | true, _ -> WitnessOutput.skip
                | _, Some binding -> direct ctx node binding supplied
                | _ -> failure node "callable occurrence" "Application callee has no witnessed callable operands or settled direct declaration"

let nanopass : Nanopass = { Name = "Application"; Witness = witnessApplication }
