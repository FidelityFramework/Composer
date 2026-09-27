/// LambdaWitness - Witness Lambda operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to ClosurePatterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Lambda nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
///
/// Every function body, including Baker's startup activation, is pulled through
/// the same registered witness fixed point. Startup order is resident in the PSG.
module Alex.Witnesses.LambdaWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.PSGZipper
open Alex.Traversal.ScopeContext
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ClosurePatterns
open Alex.XParsec.PSGCombinators  // For findLastValueNode
open Alex.CodeGeneration.TypeMapping
open Alex.Elements.MLIRAtomics  // For pUndef, pInsertValue, pExtractValue
open Alex.Elements.FuncElements  // For pFuncConstant
module Values = Alex.Traversal.Values
module CallableOperands = Alex.Traversal.CallableOperands
open XParsec
open XParsec.Parsers
open XParsec.Combinators

// ═══════════════════════════════════════════════════════════
// Y-COMBINATOR PATTERN
// ═══════════════════════════════════════════════════════════
//
// Lambda witnesses need to handle nested lambdas (closures, higher-order functions).
// This requires recursive self-reference: the combinator must include itself.
//
// Solution: Y-combinator fixed point via thunk (unit -> Combinator)
// The combinator getter is passed from WitnessRegistry, allowing deferred evaluation
// and creating a proper fixed point where witnesses can recursively invoke themselves.

// ═══════════════════════════════════════════════════════════
// CURRY FLATTENING SUPPORT
// ═══════════════════════════════════════════════════════════

/// Structural membership of this occurrence, without following binding references.
/// Already witnessed dependencies outside the body remain available; a shared
/// body must still be observed in each function's operation/operand scope.
let private structuralMembers (position: PSGZipper) =
    let rec collect seen position =
        if Set.contains position.Focus.Id seen then seen
        else
            let seen = Set.add position.Focus.Id seen
            position.Focus.Children
            |> List.indexed
            |> List.fold (fun found (index, _) ->
                match down index position with
                | Some child -> collect found child
                | None -> found) seen
    collect Set.empty position

/// Each real formal owns one scalar or a settled callable's separate operand
/// components. This reads a signature; it neither visits nor emits a body.
type private ParameterShape = Scalar | Omitted | Callable of CallableOperands.Shape | Sequence of Alex.Traversal.SequenceOperands.Shape | Lazy of Alex.Traversal.LazyOperands.Shape

let private parameterComponents (ctx: WitnessContext) parameters =
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryOrdinary ctx.Graph with
    | Result.Error reason -> Result.Error reason
    | Result.Ok projection ->
        let omitted = projection.Parameters.TryFind ctx.Zipper.Focus.Id |> Option.defaultValue Set.empty
        let groups = parameters |> List.map (fun (_, ty, id) ->
            if omitted.Contains id then Result.Ok([], Omitted) else
            match CallableOperands.valueShape ctx id with
            | Result.Error reason -> Result.Error reason
            | Result.Ok(CallableValueShape.Callable _) ->
                CallableOperands.project ctx id |> Result.map (fun shape ->
                    (CallableOperands.functionType shape :: Option.toList (CallableOperands.environmentType shape)), Callable shape)
            | Result.Ok(CallableValueShape.Sequence _) ->
                Alex.Traversal.SequenceOperands.project ctx id |> Result.map (fun shape ->
                    Alex.Traversal.SequenceOperands.componentTypes shape, Sequence shape)
            | Result.Ok(CallableValueShape.Lazy _) ->
                Alex.Traversal.LazyOperands.project ctx id |> Result.map (fun shape ->
                    Alex.Traversal.LazyOperands.componentTypes shape, Lazy shape)
            | Result.Ok(CallableValueShape.Data _) ->
                try Result.Ok([mapTypeAt id ctx], Scalar)
                with ex -> Result.Error ex.Message)
        match groups |> List.tryPick (function Result.Error reason -> Some reason | _ -> None) with
        | Some reason -> Result.Error reason
        | None -> Result.Ok(groups |> List.choose (function Result.Ok group -> Some group | _ -> None))

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness Lambda operations - category-selective (handles only Lambda nodes)
/// Takes combinator getter (Y-combinator thunk) for recursive self-reference
let private witnessLambdaWith (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Get the full combinator (including ourselves) via Y-combinator fixed point
    let combinator = getCombinator()

    let reading =
        tryMatch pLambdaWithCaptures ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator
        |> Option.map (fun (((parameters, _, _), _) as matched) -> matched, parameterComponents ctx parameters)
    match reading with
    | Some (((_, _, captures), _), _) when not captures.IsEmpty || ctx.Graph.Codata.Value.Closures.ContainsKey node.Id ->
        WitnessOutput.error "Lambda requires Baker's materialized code/environment contract; packed closure placement is retired."
    | Some (_, Result.Error reason) -> WitnessOutput.error $"Callable formal components: {reason}"
    | Some (((params', bodyId, _), _), Result.Ok parameterTypes) ->
        // Check if this is a declaration root Lambda
        let declRootOpt = Map.tryFind node.Id ctx.Graph.Codata.Value.DeclarationRootLambdas

        match declRootOpt with
        | Some DeclRoot.EntryPoint when
            Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryStorage ctx.Graph
            |> Result.toOption |> Option.bind _.Startup
            |> Option.forall (fun plan -> plan.EntryLambda <> node.Id) ->
            WitnessOutput.error "Entry lambda lacks Baker's settled program-initialization relation"

        | Some DeclRoot.HardwareModule ->
            // HardwareModule Lambda — future: hw.module with Design<S,R> extraction
            // For now, HardwareModule bindings are NOT Lambdas (they're RecordExprs),
            // so this branch should not be reached. If it is, return error.
            WitnessOutput.error "HardwareModule Lambda not yet supported"

        | Some DeclRoot.KernelModule ->
            // KernelModule Lambda — kernel bindings are RecordExprs (ElementKernel<'T>),
            // not Lambdas. If a Lambda is tagged as KernelModule, it is an error.
            WitnessOutput.error "KernelModule Lambda not yet supported"

        | None | Some DeclRoot.EntryPoint ->
            match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryStorage ctx.Graph with
            | Result.Error reason ->
                WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "Lambda") (Some "declaration symbol")
                    $"Baker storage projection did not settle the program-initialization relation that names lambda {NodeId.value node.Id}: {reason}"
            | Result.Ok storage ->
            // Startup is an ordinary graph body with the same passive function
            // witness. Its settled declaration root supplies export visibility.
            let own = Values.values node.Id

            // Definitions, direct calls and closure code addresses share the same
            // resolved binding identity; equal local source names remain distinct.
            let funcName =
                match storage.Startup with
                | Some plan when plan.EntryLambda = node.Id -> plan.Symbol
                | _ -> Alex.CodeGeneration.CallableSymbols.lambda ctx.Graph node false

            // The actual lambda occurrence supplies ordered formals. A callable
            // formal expands into code and its actual environment; subsequent
            // formals start after all preceding components, not their type arrows.
            let parameterValues, _ =
                parameterTypes |> List.mapFold (fun ordinal (types, _) ->
                    let values = types |> List.mapi (fun offset ty -> { SSA = SSA.Arg(ordinal + offset); Type = ty })
                    values, ordinal + types.Length) 0
            let mlirParams = parameterValues |> List.concat |> List.map (fun value -> value.SSA, value.Type)

            let funcParams = mlirParams

            // ═══ SSATypes SCOPING ═══
            // SSA values (V n, Arg n) are per-function — different functions reuse the same SSA names.
            // SSATypes is a global map, so we save/restore to isolate each function's type registrations.
            let savedOperands = MLIRAccumulator.snapshotOperands ctx.Accumulator
            let savedSSATypes = savedOperands.Types
            ctx.Accumulator.SSATypes <- Map.empty

            // Register parameter SSA types for this function scope
            for (paramSSA, mlirType) in funcParams do
                MLIRAccumulator.registerSSAType paramSSA mlirType ctx.Accumulator

            // Operand recall is scoped independently of pure SSA naming. Bind
            // the actual formals here, shadowing any parent-scope occurrence;
            // The complete operand scope is restored after the body is witnessed.
            for ((_, _, paramId), ((_, shape), values)) in List.zip params' (List.zip parameterTypes parameterValues) do
                match shape, values with
                | Omitted, [] -> ()
                | Scalar, [value] -> MLIRAccumulator.bindNode paramId value.SSA value.Type ctx.Accumulator
                | Callable shape, code :: environment ->
                    match CallableOperands.create shape code (List.tryHead environment)
                          |> Result.bind (fun value -> MLIRAccumulator.bindCallable paramId value ctx.Accumulator) with
                    | Result.Ok () -> ()
                    | Result.Error reason ->
                        MLIRAccumulator.addError (Diagnostic.error (Some paramId) (Some "Lambda") (Some "Callable formal") reason) ctx.Accumulator
                | Sequence shape, [code; environment] ->
                    match Alex.Traversal.SequenceOperands.create shape code environment
                          |> Result.bind (fun value -> MLIRAccumulator.bindSequence paramId value ctx.Accumulator) with
                    | Result.Ok () -> ()
                    | Result.Error reason ->
                        MLIRAccumulator.addError (Diagnostic.error (Some paramId) (Some "Lambda") (Some "Sequence formal") reason) ctx.Accumulator
                | Lazy shape, [code; environment] ->
                    match Alex.Traversal.LazyOperands.create shape code environment
                          |> Result.bind (fun value -> MLIRAccumulator.bindLazy paramId value ctx.Accumulator) with
                    | Result.Ok () -> ()
                    | Result.Error reason ->
                        MLIRAccumulator.addError (Diagnostic.error (Some paramId) (Some "Lambda") (Some "Lazy formal") reason) ctx.Accumulator
                | _ ->
                    MLIRAccumulator.addError (Diagnostic.error (Some paramId) (Some "Lambda") (Some "Formal components")
                        "Settled formal components do not match their witnessed argument values") ctx.Accumulator

            // Create child scope for function body (principled accumulation)
            let bodyScope = ScopeContext.createChild !ctx.ScopeContext FunctionLevel
            let bodyScopeRef = ref bodyScope

            // Descend from the actual lambda occurrence. Re-rooting at bodyId
            // would discard the enclosing function and its Huet breadcrumbs.
            let childPosition id =
                ctx.Zipper.Focus.Children |> List.tryFindIndex ((=) id)
                |> Option.bind (fun index -> down index ctx.Zipper)
            match childPosition bodyId with
            | Some bodyZipper ->
                let paramIds = params' |> List.map (fun (_, _, id) -> id) |> Set.ofList
                let inherited =
                    if ctx.Coeffects.TargetPlatform = Core.Types.Dialects.FPGA then Set.empty
                    else Set.difference !ctx.TraversalVisited (Set.union paramIds (structuralMembers bodyZipper))
                let bodyVisited = ref inherited
                let functionCtx = { ctx with ScopeContext = bodyScopeRef; TraversalVisited = bodyVisited }
                for (_, _, paramId) in params' do
                    match childPosition paramId with
                    | Some parameter ->
                        visitAllNodes combinator { functionCtx with Zipper = parameter } parameter.Focus bodyVisited
                    | None ->
                        MLIRAccumulator.addError (Diagnostic.error (Some paramId) (Some "Lambda") (Some "Parameter occurrence")
                            "Lambda parameter is absent from its structural occurrence") ctx.Accumulator
                let bodyCtx = { functionCtx with Zipper = bodyZipper }
                visitAllNodes combinator bodyCtx bodyZipper.Focus bodyVisited
            | None ->
                MLIRAccumulator.addError (Diagnostic.error (Some bodyId) (Some "Lambda") (Some "Body occurrence")
                    "Lambda body is absent from its structural occurrence") ctx.Accumulator

            // Restore parent's SSATypes (isolate this function's registrations)
            ctx.Accumulator.SSATypes <- savedSSATypes

            // Extract operations from child scope ref (NOT from parent!)
            let bodyOps = ScopeContext.getOps !bodyScopeRef

            // Get body result for return value
            let actualValueNode = findLastValueNode bodyId ctx.Graph
            let bodyResult = MLIRAccumulator.recallNode actualValueNode ctx.Accumulator
            let bodyCallable =
                match MLIRAccumulator.recallCallable actualValueNode ctx.Accumulator with
                | Some _ ->
                    match CallableOperands.reproject ctx actualValueNode bodyId with
                    | Result.Ok value -> Some value
                    | Result.Error reason ->
                        MLIRAccumulator.addError (Diagnostic.error (Some bodyId) (Some "Lambda") (Some "Callable result") reason) ctx.Accumulator
                        None
                | None -> None
            let bodySequence =
                match MLIRAccumulator.recallSequence actualValueNode ctx.Accumulator with
                | Some _ ->
                    match Alex.Traversal.SequenceOperands.reproject ctx actualValueNode bodyId with
                    | Result.Ok value -> Some value
                    | Result.Error reason ->
                        MLIRAccumulator.addError (Diagnostic.error (Some bodyId) (Some "Lambda") (Some "Sequence result") reason) ctx.Accumulator
                        None
                | None -> None
            let bodyLazy =
                match MLIRAccumulator.recallLazy actualValueNode ctx.Accumulator with
                | Some _ ->
                    match Alex.Traversal.LazyOperands.reproject ctx actualValueNode bodyId with
                    | Result.Ok value -> Some value
                    | Result.Error reason ->
                        MLIRAccumulator.addError (Diagnostic.error (Some bodyId) (Some "Lambda") (Some "Lazy result") reason) ctx.Accumulator
                        None
                | None -> None
            let bodyComponents =
                match bodyCallable, bodySequence, bodyLazy with
                | Some value, _, _ -> Some(CallableOperands.values value)
                | _, Some value, _ -> Some(Alex.Traversal.SequenceOperands.values value)
                | _, _, Some value -> Some(Alex.Traversal.LazyOperands.values value)
                | _ -> None
            MLIRAccumulator.restoreOperands savedOperands ctx.Accumulator

            // The result occurrence and return conversion are source facts.
            match ctx.Graph.Nodes.TryFind bodyId with
            | None ->
                WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "Lambda") (Some "return type")
                    $"PSG settlement did not keep the body {NodeId.value bodyId} of lambda {NodeId.value node.Id} resident in the graph; its return type has no source"
            | Some _ ->
            if System.Environment.GetEnvironmentVariable("COMPOSER_TRACE_TRAVERSAL") = "1" then
                printfn "[LambdaWitness] %s: body=%d valueNode=%d bodyResult=%A"
                    funcName (NodeId.value bodyId) (NodeId.value actualValueNode) bodyResult
            let rawReturnType =
                match bodyComponents with
                | Some values -> (List.head values).Type
                | None -> mapTypeAt bodyId ctx
            let returnMeet = Map.tryFind node.Id ctx.Graph.Codata.Value.ReturnMeets |> Option.map (fun m -> m, Values.returnMeetValue node.Id)
            let returnType =
                match bodyComponents, returnMeet, bodyResult with
                | Some values, _, _ -> (List.head values).Type
                | None, Some (meet, _), Some _ -> TInt (IntWidth meet.To)
                | None, _, Some (_, actualTy) -> actualTy
                | None, _, None -> requireValueType ctx.Graph bodyId rawReturnType
            let returnMeetOps =
                match returnMeet, bodyResult with
                | Some (meet, result), Some (ssa, _) -> [ meetOp meet result ssa ]
                | _ -> []
            let bodyOps = bodyOps @ returnMeetOps

            // Handle bodyResult based on return type
            let returnSSA =
                match bodyComponents, returnMeet, bodyResult with
                | Some values, _, _ -> Some (List.head values).SSA
                | None, Some (_, result), Some _ -> Some result
                | None, _, Some (ssa, _) -> Some ssa
                | None, _, None ->
                    if Values.isUnitTyped ctx.Graph bodyId then None
                    else
                        let bodyNodeKindStr =
                            match SemanticGraph.tryGetNode actualValueNode ctx.Graph with
                            | Some bodyNode ->
                                let kindStr = sprintf "%A" bodyNode.Kind |> fun s -> s.Split('\n').[0]
                                sprintf "Body node %d is %s" (NodeId.value actualValueNode) kindStr
                            | None ->
                                sprintf "Body node %d not found in graph" (NodeId.value actualValueNode)
                        let hint =
                            match SemanticGraph.tryGetNode actualValueNode ctx.Graph with
                            | Some bodyNode when bodyNode.Kind.ToString().StartsWith("Lambda") ->
                                " [Nested callable return requires witnessed operands from its settled source carrier.]"
                            | _ -> ""
                        let err = Diagnostic.error (Some node.Id) (Some "Lambda") (Some (sprintf "%s return" funcName))
                                    (sprintf "%s — produced no result.%s" bodyNodeKindStr hint)
                        MLIRAccumulator.addError err ctx.Accumulator
                        None

            // Delegate function wrapping to Pattern — coeffect determines func.func vs hw.module
            let expandedNames =
                List.zip params' parameterValues |> List.collect (fun ((name, _, _), values) ->
                    if values.Length = 1 then [name]
                    else values |> List.mapi (fun index _ -> sprintf "%s_%d" name index))
            let paramNames = expandedNames
            let visibility = if declRootOpt = Some DeclRoot.EntryPoint then FuncVisibility.Public else FuncVisibility.Private
            let resultTypes =
                match bodyComponents with
                | Some values -> values |> List.map _.Type
                | None -> [returnType]
            let definition =
                match bodyComponents, CallableOperands.valueShape ctx bodyId with
                | Some values, _ -> pFunctionDefResults visibility funcName funcParams (Some paramNames) resultTypes bodyOps values
                | None, Result.Error reason -> fail (Message reason)
                | None, Result.Ok(CallableValueShape.Callable _ | CallableValueShape.Sequence _ | CallableValueShape.Lazy _) -> fail (Message "Body result has no witnessed canonical function/environment operands")
                | None, _ ->
                    pFunctionDef visibility funcName funcParams (Some paramNames) returnType bodyOps returnSSA
                        (match SemanticGraph.tryGetNode bodyId ctx.Graph with Some b when Values.isUnitTyped ctx.Graph b.Id -> Some (Values.unitReturnValue node.Id) | _ -> None)
            match tryMatchWithDiagnostics definition ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok (funcDefOp, _) ->
                EmissionCorrespondence.record ctx [funcDefOp]
                let updatedRootScope = ScopeContext.addOp funcDefOp !ctx.RootScopeContext
                ctx.RootScopeContext := updatedRootScope

                let nativeCallback phase message =
                    MLIRAccumulator.addError (Diagnostic.error (Some node.Id) (Some "Lambda") (Some phase) message) ctx.Accumulator
                match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryCallable ctx.Graph with
                | Result.Error reason ->
                    nativeCallback "Native callback membership"
                        $"Baker callable projection did not settle void-callback membership for lambda {NodeId.value node.Id}: {reason}"
                | Result.Ok projection when projection.VoidCallbacks.Contains node.Id ->
                    match projection.NativeEntries.TryFind node.Id with
                    | None ->
                        nativeCallback "Native callback entry"
                            $"Baker callable projection did not settle the native entry symbol for void callback lambda {NodeId.value node.Id}"
                    | Some entry ->
                        let arguments = funcParams |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
                        let call = MLIROp.FuncOp (FuncOp.FuncCall ([{ SSA = own.[0]; Type = returnType }], funcName, arguments))
                        let body = [call; MLIROp.FuncOp (FuncOp.Return [])]
                        match tryMatchWithDiagnostics (pFuncDef entry funcParams TVoid body FuncVisibility.Private) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Result.Ok (thunk, _) ->
                            EmissionCorrespondence.record ctx [thunk]
                            ctx.RootScopeContext := ScopeContext.addOp thunk !ctx.RootScopeContext
                        | Result.Error message -> nativeCallback "Native callback thunk" message
                | Result.Ok _ -> ()

                { InlineOps = []; TopLevelOps = []; Result = TRVoid }

            | Result.Error diagnostic ->
                WitnessOutput.error $"Function '{funcName}': {diagnostic}"

    | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Create Lambda nanopass with Y-combinator thunk for recursive self-reference
/// The combinator getter allows deferred evaluation, creating a fixed point where
/// this witness can handle nested lambdas (closures, higher-order functions)
let createNanopass (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) : Nanopass = {
    Name = "Lambda"
    Witness = witnessLambdaWith getCombinator
}
