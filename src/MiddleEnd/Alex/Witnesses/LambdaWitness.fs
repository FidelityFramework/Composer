/// LambdaWitness - Witness Lambda operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to ClosurePatterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Lambda nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
///
/// SPECIAL CASE: Entry point Lambdas need to witness function bodies (sub-graphs)
/// that can contain ANY category of nodes. Uses combinator to fold over
/// all registered witnesses.
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

/// Unroll through N levels of TFun to find the innermost return type.
/// For a flattened Lambda with N params: TFun(t1, TFun(t2, ... TFun(tN, retType)...)) → retType
let rec private unrollReturnType (nParams: int) (ty: NativeType) : NativeType =
    if nParams <= 0 then ty
    else
        match ty with
        | NativeType.TFun (_, inner) -> unrollReturnType (nParams - 1) inner
        | _ -> ty

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness Lambda operations - category-selective (handles only Lambda nodes)
/// Takes combinator getter (Y-combinator thunk) for recursive self-reference
let private witnessLambdaWith (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Get the full combinator (including ourselves) via Y-combinator fixed point
    let combinator = getCombinator()

    match tryMatch pLambdaWithCaptures ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((params', bodyId, captureInfos), _) ->
        // FIRST: Visit parameter nodes (PatternBindings) to mark them as witnessed
        // ALL Lambdas must visit their parameters for coverage validation
        // Parameters are structural (SSA comes from coeffects), but must be visited
        for (_, _, paramNodeId) in params' do
            match SemanticGraph.tryGetNode paramNodeId ctx.Graph with
            | Some paramNode ->
                // Visit parameter with sub-graph combinator (will hit StructuralWitness)
                visitAllNodes combinator ctx paramNode ctx.TraversalVisited
            | None -> ()

        // Check if this is a declaration root Lambda
        let nodeIdValue = NodeId.value node.Id
        let declRootOpt = Map.tryFind node.Id ctx.Graph.Codata.Value.DeclarationRootLambdas

        match declRootOpt with
        | Some DeclRoot.EntryPoint ->
            // Entry point Lambda: generate func.func @main wrapper

            // ═══ SSATypes SCOPING ═══
            // SSA values (V n, Arg n) are per-function — different functions reuse the same SSA names.
            // SSATypes is a global map, so we save/restore to isolate each function's type registrations.
            let savedSSATypes = ctx.Accumulator.SSATypes
            let savedNodeAssoc = ctx.Accumulator.NodeAssoc
            ctx.Accumulator.SSATypes <- Map.empty

            // Register parameter SSA types for this function scope
            let argvType = TMemRef (TInt (IntWidth 8))
            MLIRAccumulator.registerSSAType (SSA.Arg 0) argvType ctx.Accumulator

            // Create child scope for function body (principled accumulation)
            let bodyScope = ScopeContext.createChild !ctx.ScopeContext FunctionLevel
            let bodyScopeRef = ref bodyScope

            // PROLOGUE: module-level value bindings of every module (in module order) are
            // initialized here, inside the entry point's body scope, into their program-
            // lifetime slots. Their SSAs were assigned in main's namespace (SSAAssignment
            // Pass 1). Dependencies between them resolve through the VarRef cross-reference
            // visit in visitAllNodes, in this same scope.
            if ctx.Coeffects.TargetPlatform <> Core.Types.Dialects.FPGA && ctx.Coeffects.TargetPlatform <> Core.Types.Dialects.NPU then
                for kvp in ctx.Graph.ModuleClassifications.Value do
                    for initId in kvp.Value.ModuleInit do
                        match SemanticGraph.tryGetNode initId ctx.Graph with
                        | Some initNode when initNode.IsReachable && not (Set.contains initId !ctx.TraversalVisited) ->
                            match focusOn initId ctx.Zipper with
                            | Some initZipper ->
                                let initCtx = { ctx with Zipper = initZipper; ScopeContext = bodyScopeRef }
                                visitAllNodes combinator initCtx initNode ctx.TraversalVisited
                            | None -> ()
                        | _ -> ()

            // Witness body nodes with child scope context
            match SemanticGraph.tryGetNode bodyId ctx.Graph with
            | Some bodyNode ->
                match focusOn bodyId ctx.Zipper with
                | Some bodyZipper ->
                    let bodyCtx = { ctx with Zipper = bodyZipper; ScopeContext = bodyScopeRef }
                    visitAllNodes combinator bodyCtx bodyNode ctx.TraversalVisited
                | None -> ()
            | None -> ()

            // Restore parent's SSATypes (isolate this function's registrations)
            ctx.Accumulator.SSATypes <- savedSSATypes

            // Extract operations from child scope ref (NOT from parent!)
            let bodyOps = ScopeContext.getOps !bodyScopeRef

            // Get body result for return value
            // Traverse Sequential structure to find actual value-producing node
            let actualValueNode = findLastValueNode bodyId ctx.Graph
            let bodyResult = MLIRAccumulator.recallNode actualValueNode ctx.Accumulator
            // A capture's prologue SSA belongs to this function. Restore parent
            // associations before constructing this closure or its siblings.
            ctx.Accumulator.NodeAssoc <- savedNodeAssoc

            // Determine return type from Lambda type signature
            // For flattened Lambdas with N params, unroll N levels of TFun. The entry point's
            // result is ABI-governed (§4.1): the body node sits at the declared Register width,
            // and the last value is brought to it by the return meet SSAAssignment derived.
            let innerReturnNativeType = unrollReturnType (List.length params') node.Type
            let expectedReturnType = mapTypeAt bodyId innerReturnNativeType ctx |> narrowType ctx.Coeffects ctx.Graph bodyId
            let returnMeet = Map.tryFind node.Id ctx.Graph.Codata.Value.ReturnMeets |> Option.map (fun m -> m, Values.returnMeetValue node.Id)
            let returnMeetOps =
                match returnMeet, bodyResult with
                | Some (meet, result), Some (ssa, _) -> [ meetOp meet result ssa ]
                | _ -> []

            // Handle bodyResult based on return type
            let returnSSA, returnType =
                match returnMeet, bodyResult with
                | Some (meet, result), Some _ -> (Some result, TInt (IntWidth meet.To))
                | _, Some (ssa, ty) -> (Some ssa, ty)
                | _, None ->
                    // Check if Lambda returns unit - if so, None is expected (TRVoid)
                    match innerReturnNativeType with
                    | NativeType.TApp ({ NTUKind = Some NTUKind.NTUunit }, []) ->
                        // Unit-returning function - no result SSA is expected
                        (None, expectedReturnType)
                    | _ ->
                        // Non-unit function should have produced a result
                        let bodyNodeKindStr =
                            match SemanticGraph.tryGetNode actualValueNode ctx.Graph with
                            | Some bodyNode ->
                                let kindStr = sprintf "%A" bodyNode.Kind |> fun s -> s.Split('\n').[0]
                                let typeStr = sprintf "%A" bodyNode.Type
                                sprintf "Body node %d is %s (type: %s)" (NodeId.value actualValueNode) kindStr typeStr
                            | None ->
                                sprintf "Body node %d not found in graph" (NodeId.value actualValueNode)
                        let hint =
                            match SemanticGraph.tryGetNode actualValueNode ctx.Graph with
                            | Some bodyNode when bodyNode.Kind.ToString().StartsWith("Lambda") ->
                                " [HINT: Body is a nested Lambda — Lambda produces TRVoid (emits FuncDef as side-effect). " +
                                "Returning a function value (currying/thunk) is not yet implemented]"
                            | _ -> ""
                        let err = Diagnostic.error (Some node.Id) (Some "Lambda") (Some "Entry point return")
                                    (sprintf "%s — produced no result.%s" bodyNodeKindStr hint)
                        MLIRAccumulator.addError err ctx.Accumulator
                        (None, expectedReturnType)

            let returnOp = MLIROp.FuncOp (FuncOp.Return (returnSSA, Some returnType))
            let completeBody = bodyOps @ returnMeetOps @ [returnOp]

            // Build func.func @main wrapper (portable MLIR)
            // Parameters: argv as memref<?xi8> (dynamic-sized buffer)
            let argvType = TMemRef (TInt (IntWidth 8))
            let funcParams = [(SSA.Arg 0, argvType)]
            let funcDef = FuncOp.FuncDef("main", funcParams, returnType, completeBody, Public)

            // Add FuncDef to parent scope (ctx.ScopeContext, which is root for entry points)
            let updatedParentScope = ScopeContext.addOp (MLIROp.FuncOp funcDef) !ctx.ScopeContext
            ctx.ScopeContext := updatedParentScope

            // Return empty - FuncDef already added to parent scope
            { InlineOps = []; TopLevelOps = []; Result = TRVoid }

        | Some DeclRoot.HardwareModule ->
            // HardwareModule Lambda — future: hw.module with Design<S,R> extraction
            // For now, HardwareModule bindings are NOT Lambdas (they're RecordExprs),
            // so this branch should not be reached. If it is, return error.
            WitnessOutput.error "HardwareModule Lambda not yet supported"

        | Some DeclRoot.KernelModule ->
            // KernelModule Lambda — kernel bindings are RecordExprs (ElementKernel<'T>),
            // not Lambdas. If a Lambda is tagged as KernelModule, it is an error.
            WitnessOutput.error "KernelModule Lambda not yet supported"

        | None ->
            // Non-root Lambda: Generate FuncDef for module-level function
            // Check for ClosureLayout — determines if this is a closure (escaping lambda with captures)
            let closureLayoutOpt = Map.tryFind node.Id ctx.Graph.Codata.Value.Closures

            // The closure's environment as the graph placed it (Codata.Closures), read into the
            // shapes this witness emits: each slot's MLIR type and its byte offset in the struct,
            // the struct type, and the names emission gives the callee prologue (per capture its
            // work values then its result, then the env reconstruction pair) and the construction.
            // the Pointer width is read only where a slot needs it: the fabric declares none and places no closure
            let ptrBytes () = declaredPointerBytes ctx.Coeffects.Platform.TargetArch
            let slotType (slot: CaptureSlot) : MLIRType =
                match slot.Holds with
                | CaptureSlotKind.CellView _ | CaptureSlotKind.ValueView _ ->
                    failwithf "LambdaWitness: closure %d has a continuation descriptor slot; its dedicated frame witness is required" nodeIdValue
                | CaptureSlotKind.Address | CaptureSlotKind.Handle -> TIndex
                | CaptureSlotKind.Decomposed ->
                    let ptr = ptrBytes ()
                    TStruct ([ ("ptr", TIndex); ("len", TIndex) ], Some { Offsets = [ 0; ptr ]; Size = 2 * ptr; Align = ptr })
                | CaptureSlotKind.Scalar settled ->
                    match settled with
                    | SettledSlot.Integer (bits, _) -> TInt (IntWidth bits)
                    | SettledSlot.Bool -> TInt (IntWidth 1)
                    | SettledSlot.Char | SettledSlot.Unit -> TInt (IntWidth 32)
                    | SettledSlot.Real 32 -> TFloat F32
                    | SettledSlot.Real _ -> TFloat F64
                    | SettledSlot.Pointer _ -> TIndex
                    | SettledSlot.Opaque what -> failwithf "LambdaWitness: closure %d holds %s, which has no MLIR type" nodeIdValue what
            let closureStructType (cl: ClosurePlacement) =
                match cl.Prefix with
                | ClosurePrefix.RegularClosure -> TMemRefStatic (cl.WithPrefixBytes, TInt (IntWidth 8))
                | prefix -> failwithf "LambdaWitness: %A requires its dedicated environment witness" prefix
            let absoluteOffset (cl: ClosurePlacement) (slot: CaptureSlot) = cl.PrefixBytes + slot.ByteOffset
            let prologue (cl: ClosurePlacement) : SSA list list * (SSA * SSA) =
                let perCapture =
                    cl.Captures |> List.map (fun slot ->
                        let work =
                            match slot.Holds with
                            | CaptureSlotKind.Decomposed -> 7
                            | CaptureSlotKind.Address -> 4
                            | _ -> 2
                        List.init work (fun k -> Values.prologueValue node.Id (100 + 10 * slot.Index + k)) @ [ Values.prologueValue node.Id slot.Index ])
                perCapture, (Values.prologueValue node.Id 900, Values.prologueValue node.Id 901)
            let own = Values.values node.Id
            let constructionValues (slot: CaptureSlot) : SSA list =
                let count =
                    match slot.Holds with
                    | CaptureSlotKind.CellView _ | CaptureSlotKind.ValueView _ ->
                        failwithf "LambdaWitness: closure %d has a continuation descriptor slot; its dedicated frame witness is required" nodeIdValue
                    | CaptureSlotKind.Decomposed -> 5
                    | CaptureSlotKind.Address -> 2
                    | CaptureSlotKind.Handle | CaptureSlotKind.Scalar _ -> 1
                List.init count (fun k -> own.[16 + 8 * slot.Index + k])

            // Definitions, direct calls and closure code addresses share the same
            // resolved binding identity; equal local source names remain distinct.
            let funcName = Alex.CodeGeneration.CallableSymbols.lambda ctx.Graph node closureLayoutOpt.IsSome

            // Map parameters to MLIR types and build parameter list with SSAs
            // For FPGA, parameter types are abstract (IntWidth 0) and must be narrowed
            // using the width inference coeffect before they become hw.module port declarations.
            let extractParamSSAs =
                parser {
                    let rec extractParams ps =
                        parser {
                            match ps with
                            | [] -> return []
                            | (_paramName, paramType, paramNodeId) :: rest ->
                                let rawType = mapTypeAt paramNodeId paramType ctx
                                let mlirType = narrowType ctx.Coeffects ctx.Graph paramNodeId rawType
                                let! paramSSA = getNodeSSA paramNodeId
                                let! restParams = extractParams rest
                                return (paramSSA, mlirType) :: restParams
                        }
                    return! extractParams params'
                }

            let mlirParams =
                match tryMatch extractParamSSAs ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some (paramList, _) -> paramList
                | None ->
                    printfn "[ERROR] LambdaWitness: Parameter SSAs not found in coeffects for Lambda node %A" (NodeId.value node.Id)
                    printfn "[ERROR] This indicates SSAAssignment nanopass failed to pre-allocate parameter SSAs"
                    printfn "[ERROR] Parameters: %A" params'
                    []  // Return empty list - will cause compilation to fail with proper error

            // For closures: prepend env parameter (Arg 0 = raw pointer as index)
            // The call site passes the env as an index (raw pointer from the uniform pair).
            // The lambda body reconstructs the typed memref<Nxi8> from this index before
            // extracting captures. This ensures calling convention agreement.
            let funcParams =
                match closureLayoutOpt with
                | Some _layout ->
                    // Closure: env param prepended as TIndex (raw pointer).
                    // Shift user params to Arg 1, Arg 2, etc.
                    (SSA.Arg 0, TIndex) :: mlirParams
                | None ->
                    mlirParams

            // ═══ SSATypes SCOPING ═══
            // SSA values (V n, Arg n) are per-function — different functions reuse the same SSA names.
            // SSATypes is a global map, so we save/restore to isolate each function's type registrations.
            let savedSSATypes = ctx.Accumulator.SSATypes
            let savedNodeAssoc = ctx.Accumulator.NodeAssoc
            ctx.Accumulator.SSATypes <- Map.empty

            // Register parameter SSA types for this function scope
            for (paramSSA, mlirType) in funcParams do
                MLIRAccumulator.registerSSAType paramSSA mlirType ctx.Accumulator

            // ═══ SAVE CAPTURE SOURCE SSAs BEFORE EXTRACTION REGISTRATION ═══
            // Capture extraction (below) registers inner-function SSAs in NodeAssoc via bindNode,
            // overwriting the parent-scope SSAs. Snapshot parent-scope SSAs NOW so closure
            // construction can reference the actually-emitted values.
            let savedCaptureSSAs =
                match closureLayoutOpt with
                | Some layout ->
                    layout.Captures
                    |> List.map (fun cap ->
                        match cap.SourceNode with
                        | Some sourceId ->
                            let source =
                                MLIRAccumulator.recallNode sourceId ctx.Accumulator
                                |> Option.orElseWith (fun () ->
                                    // Parameters need not have been read in the parent
                                    // body yet. Their argument SSA and physical view are
                                    // already assigned and registered by that function.
                                    let assigned = Values.resultOf ctx.Coeffects.TargetPlatform ctx.Graph sourceId
                                    match assigned with
                                    | SSA.Arg _ -> Map.tryFind assigned savedSSATypes |> Option.map (fun ty -> assigned, ty)
                                    | _ -> None)
                            source
                            |> Option.map (fun (ssa, ty) ->
                                // Mutable cells carry a dynamic semantic view, but their
                                // actual allocation retains the single-cell extent.
                                ssa, (Map.tryFind ssa savedSSATypes |> Option.defaultValue ty))
                        | None -> None)
                | None -> []

            // For closures: build capture extraction prologue
            // Extract captures from Arg 0 (env/closure struct) at function entry
            // Uses memref.reinterpret_cast to create typed views at byte offsets
            let captureExtractionOps =
                match closureLayoutOpt with
                | Some layout when not layout.Captures.IsEmpty ->
                    // Each slot's type and settled byte offset, read from the layout SSAAssignment
                    // derived (the header before the first capture is the layout's)
                    let captureSlots =
                        layout.Captures |> List.mapi (fun i cap ->
                            let valueTy =
                                match cap.Holds, savedCaptureSSAs.[i] |> Option.map snd with
                                | CaptureSlotKind.Decomposed, Some (TMemRef element | TMemRefStatic (_, element)) -> TMemRef element
                                | CaptureSlotKind.Address, Some (TMemRefStatic _ as ty) -> ty
                                | CaptureSlotKind.Address, Some (TStruct (_, Some _) as ty) -> ty
                                | (CaptureSlotKind.Decomposed | CaptureSlotKind.Address), other ->
                                    failwithf "LambdaWitness: capture '%s' has no retained bounded view: %A" cap.Capture other
                                | _ -> slotType cap
                            (slotType cap, absoluteOffset layout cap, valueTy))
                    // The callee prologue's values are the closure layout's, derived by SSAAssignment
                    // in the order pExtractCaptures consumes them (per capture: its work values, its
                    // result), then the env reconstruction pair. Nothing is numbered here.
                    let extractionSSAs, (rawEnvSSA, envMemrefSSA) = prologue layout
                    let captureResultTypes =  // per capture: the type body code sees
                        captureSlots |> List.map (fun (_, _, valueTy) -> valueTy)

                    // ═══ ENV RECONSTRUCTION PROLOGUE ═══
                    // Arg 0 arrives as index (raw pointer from uniform pair).
                    // Reconstruct memref<Nxi8> so capture extraction can use typed views.
                    let dynMemrefTy = TMemRef(TInt (IntWidth 8))
                    let envReconstructionOps = [
                        MLIROp.MemRefOp(MemRefOp.IndexToMemRef(rawEnvSSA, SSA.Arg 0, dynMemrefTy))
                        MLIROp.MemRefOp(MemRefOp.ReinterpretCast(envMemrefSSA, rawEnvSSA, 0, (match (closureStructType layout) with TMemRefStatic(sz, _) -> sz | _ -> 0), dynMemrefTy, (closureStructType layout)))
                    ]

                    match tryMatch (pExtractCaptures captureSlots (closureStructType layout) envMemrefSSA extractionSSAs) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some (ops, _) ->
                        // Register capture SSAs in accumulator so body references can find them.
                        // For decomposed memref captures, bind with the RECONSTRUCTED type (memref<?>),
                        // not the slot type (TStruct). The extraction reconstructs the memref.
                        for i in 0 .. layout.Captures.Length - 1 do
                            let cap = layout.Captures.[i]
                            match cap.SourceNode with
                            | Some sourceId ->
                                let captureSSA = Values.prologueValue node.Id cap.Index
                                let bindType = captureResultTypes.[i]
                                MLIRAccumulator.bindNode sourceId captureSSA bindType ctx.Accumulator
                            | None -> ()
                        // Prepend env reconstruction (index → memref<Nxi8>) before extraction
                        envReconstructionOps @ ops
                    | None ->
                        printfn "[ERROR] LambdaWitness: Capture extraction failed for closure Lambda %d" nodeIdValue
                        []
                | _ -> []

            // Create child scope for function body (principled accumulation)
            let bodyScope = ScopeContext.createChild !ctx.ScopeContext FunctionLevel
            let bodyScopeRef = ref bodyScope

            // Add capture extraction ops to body scope FIRST (prologue)
            for op in captureExtractionOps do
                let updated = ScopeContext.addOp op !bodyScopeRef
                bodyScopeRef := updated

            // FPGA: Per-function visited set for hw.module scope isolation.
            let bodyVisited =
                if ctx.Coeffects.TargetPlatform = Core.Types.Dialects.FPGA then
                    let paramIds = params' |> List.fold (fun s (_, _, pid) -> Set.add pid s) Set.empty
                    ref paramIds
                else
                    ctx.GlobalVisited

            // Witness body nodes with child scope context
            match SemanticGraph.tryGetNode bodyId ctx.Graph with
            | Some bodyNode ->
                match focusOn bodyId ctx.Zipper with
                | Some bodyZipper ->
                    let bodyCtx = { ctx with Zipper = bodyZipper; ScopeContext = bodyScopeRef; TraversalVisited = bodyVisited }
                    visitAllNodes combinator bodyCtx bodyNode bodyVisited
                | None -> ()
            | None -> ()

            // Restore parent's SSATypes (isolate this function's registrations)
            ctx.Accumulator.SSATypes <- savedSSATypes

            // Extract operations from child scope ref (NOT from parent!)
            let bodyOps = ScopeContext.getOps !bodyScopeRef

            // Get body result for return value
            let actualValueNode = findLastValueNode bodyId ctx.Graph
            let bodyResult = MLIRAccumulator.recallNode actualValueNode ctx.Accumulator
            ctx.Accumulator.NodeAssoc <- savedNodeAssoc

            // Determine return type from Lambda type signature
            // For flattened Lambdas with N params, unroll N levels of TFun. The result is held at
            // the body node's width (the width every caller reads for the call); the last value
            // is brought to it by the return meet SSAAssignment derived, the last value of this
            // scope (an escaping lambda's body sits at the declared Register width, ruling 1).
            let innerReturnNativeType2 = unrollReturnType (List.length params') node.Type
            if System.Environment.GetEnvironmentVariable("COMPOSER_TRACE_TRAVERSAL") = "1" then
                printfn "[LambdaWitness] %s: body=%d valueNode=%d bodyResult=%A returnNative=%A"
                    funcName (NodeId.value bodyId) (NodeId.value actualValueNode) bodyResult innerReturnNativeType2
            let rawReturnType = mapTypeAt bodyId innerReturnNativeType2 ctx
            let nativeVoid =
                Clef.Compiler.PSGSaturation.SemanticGraph.CallbackDeclarations.forLambda ctx.Graph node.Id
                |> Option.exists (fun callback -> callback.ReturnsVoid)
            let returnMeet = Map.tryFind node.Id ctx.Graph.Codata.Value.ReturnMeets |> Option.map (fun m -> m, Values.returnMeetValue node.Id)
            let returnType =
                match returnMeet, bodyResult with
                | Some (meet, _), Some _ -> TInt (IntWidth meet.To)
                | _, Some (_, actualTy) -> actualTy
                | _, None -> narrowType ctx.Coeffects ctx.Graph bodyId rawReturnType
            let returnMeetOps =
                match returnMeet, bodyResult with
                | Some (meet, result), Some (ssa, _) -> [ meetOp meet result ssa ]
                | _ -> []
            let bodyOps = bodyOps @ returnMeetOps

            // Handle bodyResult based on return type
            let returnSSA =
                match returnMeet, bodyResult with
                | Some (_, result), Some _ -> Some result
                | _, Some (ssa, _) -> Some ssa
                | _, None ->
                    match innerReturnNativeType2 with
                    | NativeType.TApp ({ NTUKind = Some NTUKind.NTUunit }, []) ->
                        None
                    | _ ->
                        let bodyNodeKindStr =
                            match SemanticGraph.tryGetNode actualValueNode ctx.Graph with
                            | Some bodyNode ->
                                let kindStr = sprintf "%A" bodyNode.Kind |> fun s -> s.Split('\n').[0]
                                let typeStr = sprintf "%A" bodyNode.Type
                                sprintf "Body node %d is %s (type: %s)" (NodeId.value actualValueNode) kindStr typeStr
                            | None ->
                                sprintf "Body node %d not found in graph" (NodeId.value actualValueNode)
                        let hint =
                            match SemanticGraph.tryGetNode actualValueNode ctx.Graph with
                            | Some bodyNode when bodyNode.Kind.ToString().StartsWith("Lambda") ->
                                " [HINT: Body is a nested Lambda — Lambda produces TRVoid (emits FuncDef as side-effect). " +
                                "Returning a function value (currying/thunk) is not yet implemented]"
                            | _ -> ""
                        let err = Diagnostic.error (Some node.Id) (Some "Lambda") (Some (sprintf "%s return" funcName))
                                    (sprintf "%s — produced no result.%s" bodyNodeKindStr hint)
                        MLIRAccumulator.addError err ctx.Accumulator
                        None

            // Delegate function wrapping to Pattern — coeffect determines func.func vs hw.module
            let paramNames =
                match closureLayoutOpt with
                | Some _ -> "env" :: (params' |> List.map (fun (name, _, _) -> name))
                | None -> params' |> List.map (fun (name, _, _) -> name)
            match tryMatchWithDiagnostics (pFunctionDef funcName funcParams (Some paramNames) returnType bodyOps returnSSA (match SemanticGraph.tryGetNode bodyId ctx.Graph with Some b when Values.isUnitTyped b.Type -> Some (Values.unitReturnValue node.Id) | _ -> None)) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok (funcDefOp, _) ->
                let updatedRootScope = ScopeContext.addOp funcDefOp !ctx.RootScopeContext
                ctx.RootScopeContext := updatedRootScope

                if nativeVoid then
                    let entry = Clef.Compiler.PSGSaturation.SemanticGraph.FunctionPointers.nativeEntrySymbol node.Id
                    let arguments = funcParams |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
                    let call = MLIROp.FuncOp (FuncOp.FuncCall (Some own.[0], funcName, arguments, returnType))
                    let body = [call; MLIROp.FuncOp (FuncOp.Return (None, None))]
                    match tryMatchWithDiagnostics (pFuncDef entry funcParams TVoid body FuncVisibility.Private) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok (thunk, _) -> ctx.RootScopeContext := ScopeContext.addOp thunk !ctx.RootScopeContext
                    | Result.Error message -> MLIRAccumulator.addError (Diagnostic.error (Some node.Id) (Some "Lambda") (Some "Native callback thunk") message) ctx.Accumulator

                // ═══ CLOSURE CONSTRUCTION (parent scope) ═══
                // If this Lambda has captures, build closure struct + uniform pair in parent scope
                // All stores use memref.reinterpret_cast to create typed views at byte offsets
                match closureLayoutOpt with
                | Some layout ->
                    // 1. Get code pointer (func.constant @funcName → real function type → index)
                    // func.constant must use actual function type; we cast to index for storage
                    let innerFuncParamTypes = funcParams |> List.map snd
                    let funcRefSSA = own.[0]
                    let funcTy = TFunc (innerFuncParamTypes, returnType)
                    let funcConstOp = MLIROp.FuncOp (FuncOp.FuncConstant (funcRefSSA, funcName, funcTy))
                    ctx.ScopeContext := ScopeContext.addOp funcConstOp !ctx.ScopeContext
                    // Cast function reference → index for storage in closure struct/pair
                    let codePtrTy = TIndex
                    let castOp = MLIROp.FuncOp (FuncOp.FuncToIndex (own.[1], funcRefSSA, innerFuncParamTypes, returnType))
                    ctx.ScopeContext := ScopeContext.addOp castOp !ctx.ScopeContext

                    // 2. Allocate closure struct — PULL allocation strategy from escape analysis coeffect.
                    // Four-point lifetime lattice (closure-representation.md §3.3):
                    //   StackScoped    → alloca              (scope-bounded closure, stack)
                    //   StaticLifetime → get_global          (program-lifetime closure, static storage, no heap)
                    //   EscapesVia*    → alloc               (escaping closure, heap)
                    // StaticLifetime references a module-level memref.global instead of allocating; the
                    // GlobalMemref decl is emitted as a TopLevelOp (collected in staticGlobalDecls).
                    let closureTy = (closureStructType layout)
                    let escapeKind = escapeOf ctx.Graph node.Id
                    // Per-closure static-storage symbol names (unique by node id). Only used on the
                    // StaticLifetime path; the decls are threaded to the module root via TopLevelOps.
                    let closureGlobalName = sprintf "__clef_closure_env_%d" nodeIdValue
                    let pairGlobalName = sprintf "__clef_closure_pair_%d" nodeIdValue
                    let mutable staticGlobalDecls : MLIROp list = []
                    let envAllocOp =
                        match escapeKind with
                        | EscapeKind.StackScoped ->
                            MLIROp.MemRefOp (MemRefOp.Alloca (own.[2], closureTy, None))
                        | EscapeKind.StaticLifetime ->
                            staticGlobalDecls <- staticGlobalDecls @ [ MLIROp.GlobalMemref (closureGlobalName, closureTy) ]
                            MLIROp.MemRefOp (MemRefOp.GetGlobal (own.[2], closureGlobalName, closureTy))
                        | EscapeKind.EscapesViaReturn | EscapeKind.EscapesViaClosure _ | EscapeKind.EscapesViaByRef ->
                            MLIROp.MemRefOp (MemRefOp.AllocStatic (own.[2], closureTy, None))
                    let parentScope2 = ScopeContext.addOp envAllocOp !ctx.ScopeContext
                    ctx.ScopeContext := parentScope2

                    // Shared zero constant for all store indices
                    let zeroSSA = own.[3]
                    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, TIndex))
                    let parentScope2a = ScopeContext.addOp zeroOp !ctx.ScopeContext
                    ctx.ScopeContext := parentScope2a

                    // 3. Insert code_ptr at byte offset 0 via reinterpret_cast
                    let codeViewSSA = own.[4]
                    let codeViewTy = TMemRefStatic (1, codePtrTy)
                    let codeCastOp = MLIROp.MemRefOp (MemRefOp.ReinterpretCast (codeViewSSA, own.[2], 0, 1, closureTy, codeViewTy))
                    let parentScope3 = ScopeContext.addOp codeCastOp !ctx.ScopeContext
                    ctx.ScopeContext := parentScope3
                    let codeStoreOp = MLIROp.MemRefOp (MemRefOp.Store (own.[1], codeViewSSA, [zeroSSA], codePtrTy, codeViewTy))
                    let parentScope4 = ScopeContext.addOp codeStoreOp !ctx.ScopeContext
                    ctx.ScopeContext := parentScope4

                    // 4. Insert captures at their settled byte offsets via reinterpret_cast, each
                    // with the construction values the layout derived for it (five for a
                    // decomposed memref, two for a base-pointer slot, one for a scalar)
                    for i in 0 .. layout.Captures.Length - 1 do
                        let cap = layout.Captures.[i]
                        let captureByteOffset = absoluteOffset layout cap
                        let captureSSAs = constructionValues cap
                        // Resolve capture source SSA from the SAVED parent-scope snapshot.
                        // We snapshot NodeAssoc BEFORE body emission because body emission
                        // overwrites parent-scope entries with inner function extraction SSAs.
                        // The saved SSAs reflect the actually-emitted values in the parent scope,
                        // unlike coeffects ResultSSA which may not match emission (over-allocated SSAs).
                        let captureSSAOpt =
                            if i < savedCaptureSSAs.Length then
                                savedCaptureSSAs.[i] |> Option.map fst
                            else
                                None
                        match captureSSAOpt with
                        | Some capSSA ->
                            match (slotType cap), captureSSAs with
                            | TStruct ([("ptr", TIndex); ("len", TIndex)], slotBytes), [ ptrSSA; dimZeroSSA; lenSSA; ptrViewSSA; lenViewSSA ] ->
                                // Decomposed memref: extract ptr + len from source memref, store separately

                                // Extract base pointer from source memref
                                let srcMemrefTy =
                                    match savedCaptureSSAs.[i] |> Option.map snd with
                                    | Some (TMemRef _ | TMemRefStatic _ as ty) -> ty
                                    | other -> failwithf "LambdaWitness: decomposed capture '%s' has no buffer: %A" cap.Capture other
                                let extractPtrOp = MLIROp.MemRefOp(MemRefOp.ExtractBasePtr(ptrSSA, capSSA, srcMemrefTy))
                                ctx.ScopeContext := ScopeContext.addOp extractPtrOp !ctx.ScopeContext

                                // Extract length (dim 0) from source memref
                                let dimZeroOp = MLIROp.ArithOp(ArithOp.ConstI(dimZeroSSA, 0L, TIndex))
                                ctx.ScopeContext := ScopeContext.addOp dimZeroOp !ctx.ScopeContext
                                let dimOp = MLIROp.MemRefOp(MemRefOp.Dim(lenSSA, capSSA, dimZeroSSA, srcMemrefTy))
                                ctx.ScopeContext := ScopeContext.addOp dimOp !ctx.ScopeContext

                                // Store ptr at current byte offset
                                let ptrViewTy = TMemRefStatic(1, TIndex)
                                let ptrCastOp = MLIROp.MemRefOp(MemRefOp.ReinterpretCast(ptrViewSSA, own.[2], captureByteOffset, 1, closureTy, ptrViewTy))
                                ctx.ScopeContext := ScopeContext.addOp ptrCastOp !ctx.ScopeContext
                                let ptrStoreOp = MLIROp.MemRefOp(MemRefOp.Store(ptrSSA, ptrViewSSA, [zeroSSA], TIndex, ptrViewTy))
                                ctx.ScopeContext := ScopeContext.addOp ptrStoreOp !ctx.ScopeContext

                                // Store len at its offset within the decomposed slot
                                let lenByteOffset =
                                    match slotBytes with
                                    | Some b -> captureByteOffset + b.Offsets.[1]
                                    | None -> failwith "LambdaWitness: a decomposed string slot with no derived layout"
                                let lenViewTy = TMemRefStatic(1, TIndex)
                                let lenCastOp = MLIROp.MemRefOp(MemRefOp.ReinterpretCast(lenViewSSA, own.[2], lenByteOffset, 1, closureTy, lenViewTy))
                                ctx.ScopeContext := ScopeContext.addOp lenCastOp !ctx.ScopeContext
                                let lenStoreOp = MLIROp.MemRefOp(MemRefOp.Store(lenSSA, lenViewSSA, [zeroSSA], TIndex, lenViewTy))
                                ctx.ScopeContext := ScopeContext.addOp lenStoreOp !ctx.ScopeContext

                            | _, (viewSSA :: rest) ->
                                // Scalar: single reinterpret_cast + store
                                let extractSSAOpt =
                                    match (cap.Holds = CaptureSlotKind.Address), rest with
                                    | true, [ extractSSA ] -> Some extractSSA
                                    | true, _ -> failwithf "LambdaWitness: closure %d capture '%s' extracts a base pointer but was derived %d values" nodeIdValue cap.Capture captureSSAs.Length
                                    | false, _ -> None
                                let viewTy = TMemRefStatic (1, (slotType cap))
                                let castOp = MLIROp.MemRefOp (MemRefOp.ReinterpretCast (viewSSA, own.[2], captureByteOffset, 1, closureTy, viewTy))
                                ctx.ScopeContext := ScopeContext.addOp castOp !ctx.ScopeContext
                                // A slot holding a memref value (a record, an array, a mutable cell) as its
                                // base index: the extraction value is the layout's, derived from the capture's
                                // type. The accumulator's type must agree, or the derivation and the emission
                                // have diverged, which is reported, never patched here.
                                let sourceType =
                                    if i < savedCaptureSSAs.Length then savedCaptureSSAs.[i] |> Option.map snd else None
                                let sourceIsMemRef =
                                    match sourceType with
                                    | Some (TMemRefStatic _ | TMemRef _ | TStruct (_, Some _)) -> true
                                    | _ -> false
                                let actualCapSSA =
                                    match extractSSAOpt, sourceIsMemRef with
                                    | Some extractSSA, true ->
                                        let extractOp = MLIROp.MemRefOp(MemRefOp.ExtractBasePtr(extractSSA, capSSA, sourceType.Value))
                                        ctx.ScopeContext := ScopeContext.addOp extractOp !ctx.ScopeContext
                                        extractSSA
                                    | None, false -> capSSA
                                    | Some _, false ->
                                        failwithf "LambdaWitness: closure %d capture '%s' was derived to extract a base pointer (index slot, memref source) but its source is bound as %A" nodeIdValue cap.Capture sourceType
                                    | None, true ->
                                        failwithf "LambdaWitness: closure %d capture '%s' binds a memref source to an index slot with no extraction value derived; SSAAssignment.captureExtractsBasePointer does not cover its type" nodeIdValue cap.Capture
                                let storeOp = MLIROp.MemRefOp (MemRefOp.Store (actualCapSSA, viewSSA, [zeroSSA], (slotType cap), viewTy))
                                ctx.ScopeContext := ScopeContext.addOp storeOp !ctx.ScopeContext
                            | _, values ->
                                failwithf "LambdaWitness: closure %d capture '%s' of slot type %A was derived %d values; the derivation and the emission disagree" nodeIdValue cap.Capture (slotType cap) values.Length
                        | None ->
                            failwithf "LambdaWitness: capture '%s' (source %A) not found in accumulator for closure %d" cap.Capture cap.SourceNode nodeIdValue

                    // 5. Build uniform pair {code_ptr, env_ptr}
                    // Pair type matches TypeMapping: TMemRefStatic(2, TIndex) = memref<2xindex>
                    // Same escape analysis governs the pair — it shares the closure's lifetime.
                    let pairTy = TMemRefStatic(2, TIndex)

                    let pairAllocOp =
                        match escapeKind with
                        | EscapeKind.StackScoped ->
                            MLIROp.MemRefOp (MemRefOp.Alloca (own.[5], pairTy, None))
                        | EscapeKind.StaticLifetime ->
                            staticGlobalDecls <- staticGlobalDecls @ [ MLIROp.GlobalMemref (pairGlobalName, pairTy) ]
                            MLIROp.MemRefOp (MemRefOp.GetGlobal (own.[5], pairGlobalName, pairTy))
                        | EscapeKind.EscapesViaReturn | EscapeKind.EscapesViaClosure _ | EscapeKind.EscapesViaByRef ->
                            MLIROp.MemRefOp (MemRefOp.AllocStatic (own.[5], pairTy, None))
                    ctx.ScopeContext := ScopeContext.addOp pairAllocOp !ctx.ScopeContext

                    // Store code_ptr at element [0] — direct store, same element type
                    let pairCodeStoreOp = MLIROp.MemRefOp (MemRefOp.Store (own.[1], own.[5], [zeroSSA], TIndex, pairTy))
                    ctx.ScopeContext := ScopeContext.addOp pairCodeStoreOp !ctx.ScopeContext

                    // Extract env_ptr = base pointer of closure struct
                    let envPtrSSA = own.[6]
                    let envExtractOp = MLIROp.MemRefOp (MemRefOp.ExtractBasePtr (envPtrSSA, own.[2], closureTy))
                    ctx.ScopeContext := ScopeContext.addOp envExtractOp !ctx.ScopeContext

                    // Store env_ptr at element [1]
                    let oneSSA = own.[7]
                    let oneOp = MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, TIndex))
                    ctx.ScopeContext := ScopeContext.addOp oneOp !ctx.ScopeContext
                    let pairEnvStoreOp = MLIROp.MemRefOp (MemRefOp.Store (envPtrSSA, own.[5], [oneSSA], TIndex, pairTy))
                    ctx.ScopeContext := ScopeContext.addOp pairEnvStoreOp !ctx.ScopeContext

                    // Register closure pair in accumulator for BindingWitness/VarRefWitness to find
                    MLIRAccumulator.bindNode node.Id own.[5] pairTy ctx.Accumulator

                    // On the StaticLifetime path, staticGlobalDecls carries the module-level
                    // memref.global declarations backing this closure's env and pair; they ride
                    // out as TopLevelOps to the module root. Empty on every other lifetime path.
                    { InlineOps = []; TopLevelOps = staticGlobalDecls; Result = TRValue { SSA = own.[5]; Type = pairTy } }

                | None ->
                    // No captures — plain named function, no closure construction needed
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
