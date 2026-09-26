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
    match ctx.Graph.Nodes.TryFind binding with
    | Some { Kind = SemanticKind.Binding(_, false, _, _); Children = [child] } ->
        let child =
            match ctx.Graph.Nodes[child].Kind with
            | SemanticKind.TypeAnnotation(inner, _) -> ctx.Graph.Nodes[inner]
            | _ -> ctx.Graph.Nodes[child]
        match child.Kind with
        | SemanticKind.Lambda(parameters, body, [], _, _) -> Some(parameters, body)
        | _ -> None
    | _ -> None

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
                match ctx.Graph.Nodes.TryFind source with
                | Some node when (match applySubst node.Type with NativeType.TFun _ | NativeType.TLazy _ -> true | _ -> false) ->
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
    match arguments ctx node.Id sources with
    | Result.Error reason -> failure node "argument operands" reason
    | Result.Ok (meets, operands) ->
        let actuals = Option.toList environment @ operands
        let agrees = expected |> Option.forall (fun types -> types = List.map (fun (value: Val) -> value.Type) actuals)
        if not agrees then failure node "parameter boundary" "Direct invocation operands disagree with its settled physical parameter components"
        else
        let deferred = sources |> List.collect (fun source -> MLIRAccumulator.getDeferredInlineOps source ctx.Accumulator)
        let prefix = deferred @ meets
        match applySubst node.Type with
        | NativeType.TFun _ ->
            match Operands.project ctx node.Id with
            | Result.Error reason -> failure node "callable result" reason
            | Result.Ok shape -> observe ctx node prefix (pCallableApplication node.Id invocation actuals shape)
        | NativeType.TSeq _ | NativeType.TSeqEnumerator _ ->
            match Alex.Traversal.SequenceOperands.project ctx node.Id with
            | Result.Error reason -> failure node "sequence result" reason
            | Result.Ok shape -> observe ctx node prefix (pSequenceApplication node.Id invocation actuals shape)
        | NativeType.TLazy _ ->
            match Alex.Traversal.LazyOperands.project ctx node.Id with
            | Result.Error reason -> failure node "lazy result" reason
            | Result.Ok shape -> observe ctx node prefix (pLazyApplication node.Id invocation actuals shape)
        | _ ->
            let resultType =
                mapTypeAt node.Id node.Type ctx
                |> narrowType ctx.Coeffects ctx.Graph (Option.defaultValue node.Id body)
            match invocation with
            | Direct symbol ->
                observe ctx node prefix (pDirectCall node.Id symbol (actuals |> List.map (fun value -> value.SSA, value.Type)) resultType names)
            | Indirect code ->
                match code.Type with
                | TFunc(_, [actualResult]) -> observe ctx node prefix (pIndirectApplication node.Id code actuals actualResult)
                | _ -> failure node "result boundary" "Scalar application requires exactly one result in its witnessed function signature"

let private direct (ctx: WitnessContext) (node: SemanticNode) binding (sources: NodeId list) =
    let declared =
        match Alex.CodeGeneration.CallableSymbols.tryBinding ctx.Graph binding, declaration ctx binding with
        | Some symbol, Some(parameters, body) -> Some(symbol, parameters, body)
        | _ -> Operands.tryThunkDeclaration ctx binding
    match declared with
    | Some(symbol, parameters, body) ->
        if parameters.Length <> sources.Length then
            failure node "declared boundary" "Settled direct call does not supply its actual declared parameters"
        else
            let names =
                parameters |> List.collect (fun (name, _, formal) ->
                    match Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.valueShape ctx.Graph ctx.Graph.Nodes[formal] with
                    | CallableValueShape.Callable _ ->
                        match Operands.project ctx formal with
                        | Result.Ok shape -> (name + "_code") :: (if (Operands.environmentType shape).IsSome then [name + "_environment"] else [])
                        | Result.Error _ -> [name] // the component reading below reports the absent boundary
                    | CallableValueShape.Sequence _ -> [name + "_pull"; name + "_environment"]
                    | CallableValueShape.Lazy _ -> [name + "_thunk"; name + "_environment"]
                    | _ -> [name])
            let components =
                parameters |> List.map (fun (_, _, formal) ->
                    Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.valueShape ctx.Graph ctx.Graph.Nodes[formal]
                    |> Operands.components ctx)
            match components |> List.tryPick (function Result.Error reason -> Some reason | _ -> None) with
            | Some reason -> failure node "parameter boundary" reason
            | None ->
                let expected = components |> List.collect (function Result.Ok types -> types | _ -> [])
                invoke ctx node (Direct symbol) sources (Some body) (Some names) None (Some expected)
    | _ -> failure node "declaration identity" "Direct call lacks its settled declaration symbol and actual lambda boundary"

let private witnessApplication (ctx: WitnessContext) (node: SemanticNode) =
    // Foreign admission and its explicit ABI remain PlatformWitness's concern.
    if Map.containsKey node.Id ctx.Coeffects.Platform.Bindings.Bindings
       || (match node.Kind with
           | SemanticKind.Application(callee, _) -> (Clef.Compiler.PSGSaturation.SemanticGraph.MappedBindings.tryFindCall ctx.Graph callee).IsSome
           | _ -> false) then WitnessOutput.skip
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
                let calleeNode =
                    match ctx.Graph.Nodes.TryFind callee with
                    | Some { Kind = SemanticKind.TypeAnnotation(inner, _) } -> ctx.Graph.Nodes.TryFind inner
                    | other -> other
                match calleeNode with
                | Some { Kind = SemanticKind.Intrinsic _ } -> WitnessOutput.skip
                | Some { Kind = SemanticKind.VarRef(_, Some binding) } -> direct ctx node binding supplied
                | _ -> failure node "callable occurrence" "Application callee has no witnessed callable operands or settled direct declaration"

let nanopass : Nanopass = { Name = "Application"; Witness = witnessApplication }
