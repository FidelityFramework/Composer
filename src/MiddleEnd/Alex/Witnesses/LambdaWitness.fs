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
open Alex.CodeGeneration.TypeMapping  // For resolveTypeParams, mlirTypeSizeForArch
open Alex.Elements.MLIRAtomics  // For pUndef, pInsertValue, pExtractValue
open Alex.Elements.FuncElements  // For pFuncConstant
open PSGElaboration.SSAAssignment  // For lookupClosureLayout
open PSGElaboration.Coeffects  // For ClosureLayout, closureExtractionBaseIndex
open PSGElaboration.EscapeAnalysis  // For getEscapeKindOrDefault — PULL allocation strategy
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
        let declRootOpt = Map.tryFind nodeIdValue ctx.Coeffects.DeclarationRootLambdas

        match declRootOpt with
        | Some DeclRoot.EntryPoint ->
            // Entry point Lambda: generate func.func @main wrapper

            // ═══ SSATypes SCOPING ═══
            // SSA values (V n, Arg n) are per-function — different functions reuse the same SSA names.
            // SSATypes is a global map, so we save/restore to isolate each function's type registrations.
            let savedSSATypes = ctx.Accumulator.SSATypes
            ctx.Accumulator.SSATypes <- Map.empty

            // Register parameter SSA types for this function scope
            let argvType = TMemRef (TInt (IntWidth 8))
            MLIRAccumulator.registerSSAType (SSA.Arg 0) argvType ctx.Accumulator

            // Eagerly resolve type parameters from the Lambda's type signature
            // so inner expression types resolve correctly through Union-Find
            resolveTypeParams ctx.Graph node.Type

            // Create child scope for function body (principled accumulation)
            let bodyScope = ScopeContext.createChild !ctx.ScopeContext FunctionLevel
            let bodyScopeRef = ref bodyScope

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

            // Determine return type from Lambda type signature
            // For flattened Lambdas with N params, unroll N levels of TFun
            let innerReturnNativeType = unrollReturnType (List.length params') node.Type
            let expectedReturnType = mapType innerReturnNativeType ctx

            // Handle bodyResult based on return type
            let returnSSA, returnType =
                match bodyResult with
                | Some (ssa, ty) -> (Some ssa, ty)
                | None ->
                    // Check if Lambda returns unit - if so, None is expected (TRVoid)
                    match innerReturnNativeType with
                    | NativeType.TApp ({ NTUKind = Some NTUunit }, []) ->
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
            let completeBody = bodyOps @ [returnOp]

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
            let closureLayoutOpt = lookupClosureLayout node.Id ctx.Coeffects.SSA

            // Extract QUALIFIED function name from parent Binding + ModuleDef (if present)
            // Same logic as ApplicationWitness for qualified name resolution
            let funcName =
                // Lambda's parent should be Binding node - extract name directly from PSG
                match node.Parent with
                | Some bindingId ->
                    match SemanticGraph.tryGetNode bindingId ctx.Graph with
                    | Some bindingNode ->
                        match bindingNode.Kind with
                        | SemanticKind.Binding (bindingName, _, _, _) ->
                            // Got binding name - check if Binding has ModuleDef parent for qualification
                            match bindingNode.Parent with
                            | Some moduleParentId ->
                                match SemanticGraph.tryGetNode moduleParentId ctx.Graph with
                                | Some moduleParent ->
                                    match moduleParent.Kind with
                                    | SemanticKind.ModuleDef (moduleName, _) ->
                                        // Qualified name: Module.Function
                                        sprintf "%s.%s" moduleName bindingName
                                    | _ -> bindingName  // No ModuleDef parent, use binding name
                                | None -> bindingName
                            | None -> bindingName
                        | _ ->
                            // Parent is not a Binding - use generic lambda name
                            sprintf "lambda_%d" nodeIdValue
                    | None -> sprintf "lambda_%d" nodeIdValue
                | None -> sprintf "lambda_%d" nodeIdValue

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
                                let rawType = mapType paramType ctx
                                let mlirType = narrowType ctx.Coeffects paramNodeId rawType
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
            ctx.Accumulator.SSATypes <- Map.empty

            // Register parameter SSA types for this function scope
            for (paramSSA, mlirType) in funcParams do
                MLIRAccumulator.registerSSAType paramSSA mlirType ctx.Accumulator

            // Eagerly resolve type parameters from the Lambda's type signature
            resolveTypeParams ctx.Graph node.Type

            // ═══ SAVE CAPTURE SOURCE SSAs BEFORE EXTRACTION REGISTRATION ═══
            // Capture extraction (below) registers inner-function SSAs in NodeAssoc via bindNode,
            // overwriting the parent-scope SSAs. Snapshot parent-scope SSAs NOW so closure
            // construction can reference the actually-emitted values.
            let savedCaptureSSAs =
                match closureLayoutOpt with
                | Some layout ->
                    layout.Captures
                    |> List.map (fun cap ->
                        match cap.SourceNodeId with
                        | Some sourceId -> MLIRAccumulator.recallNode sourceId ctx.Accumulator
                        | None -> None)
                | None -> []

            // For closures: build capture extraction prologue
            // Extract captures from Arg 0 (env/closure struct) at function entry
            // Uses memref.reinterpret_cast to create typed views at byte offsets
            let captureExtractionOps =
                match closureLayoutOpt with
                | Some layout when not layout.Captures.IsEmpty ->
                    let captureTypes = layout.Captures |> List.map (fun cap -> cap.SlotType)
                    // Compute prefix byte offset (bytes before first capture in struct)
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let sizeOf = mlirTypeSizeForArch arch
                    let prefixByteOffset =
                        match layout.Context with
                        | LambdaContext.RegularClosure -> sizeOf TIndex  // code_ptr = one platform word
                        | LambdaContext.LazyThunk ->
                            // {computed: i1, value: T, code_ptr: ptr} — need value type from lazy struct
                            sizeOf (TInt (IntWidth 1)) + sizeOf TIndex + sizeOf TIndex  // Approximate
                        | LambdaContext.SeqGenerator ->
                            // {state: i32, current: T, code_ptr: ptr}
                            sizeOf (TInt (IntWidth 32)) + sizeOf TIndex + sizeOf TIndex  // Approximate
                    // Build SSAs for extraction: variable-width per capture type.
                    // Result SSAs are V(0)..V(n-1), work SSAs start at V(n).
                    // pExtractCaptures consumes SSAs in order per capture:
                    //   decomposed memref: 8 SSAs [ptrView,ptrZero,ptr, lenView,lenZero,len, rawMemref, result]
                    //   scalar: 3 SSAs [view, zero, result]
                    let n = captureTypes.Length
                    let extractionSSAs, captureResultTypes, totalWorkSSAs =
                        let mutable ssaList = []
                        let mutable resultTypes = []  // per-capture: the type body code sees
                        let mutable workIdx = 0  // work SSA counter (offset from n)
                        for i in 0 .. n - 1 do
                            let capTy = captureTypes.[i]
                            match capTy with
                            | TStruct [("ptr", TIndex); ("len", TIndex)] ->
                                // Decomposed memref: 7 work SSAs + 1 result SSA
                                let ptrViewSSA  = V (n + workIdx)
                                let ptrZeroSSA  = V (n + workIdx + 1)
                                let ptrSSA      = V (n + workIdx + 2)
                                let lenViewSSA  = V (n + workIdx + 3)
                                let lenZeroSSA  = V (n + workIdx + 4)
                                let lenSSA      = V (n + workIdx + 5)
                                let rawMemrefSSA = V (n + workIdx + 6)
                                let resultSSA   = V i
                                ssaList <- ssaList @ [ptrViewSSA; ptrZeroSSA; ptrSSA; lenViewSSA; lenZeroSSA; lenSSA; rawMemrefSSA; resultSSA]
                                resultTypes <- resultTypes @ [TMemRef(TInt (IntWidth 8))]
                                workIdx <- workIdx + 7
                            | _ ->
                                // Scalar: 2 work SSAs + 1 result SSA
                                let viewSSA   = V (n + workIdx)
                                let zeroSSA   = V (n + workIdx + 1)
                                let resultSSA = V i
                                ssaList <- ssaList @ [viewSSA; zeroSSA; resultSSA]
                                resultTypes <- resultTypes @ [capTy]
                                workIdx <- workIdx + 2
                        ssaList, resultTypes, workIdx

                    // ═══ ENV RECONSTRUCTION PROLOGUE ═══
                    // Arg 0 arrives as index (raw pointer from uniform pair).
                    // Reconstruct memref<Nxi8> so capture extraction can use typed views.
                    // Uses 2 SSAs at end of work range: rawEnvSSA, envMemrefSSA
                    let rawEnvSSA = V (n + totalWorkSSAs)       // memref<?xi8> from IndexToMemRef
                    let envMemrefSSA = V (n + totalWorkSSAs + 1) // memref<Nxi8> from ReinterpretCast
                    let dynMemrefTy = TMemRef(TInt (IntWidth 8))
                    let envReconstructionOps = [
                        MLIROp.MemRefOp(MemRefOp.IndexToMemRef(rawEnvSSA, SSA.Arg 0, dynMemrefTy))
                        MLIROp.MemRefOp(MemRefOp.ReinterpretCast(envMemrefSSA, rawEnvSSA, 0, (match layout.ClosureStructType with TMemRefStatic(sz, _) -> sz | _ -> 0), dynMemrefTy, layout.ClosureStructType))
                    ]

                    match tryMatch (pExtractCaptures prefixByteOffset captureTypes layout.ClosureStructType envMemrefSSA extractionSSAs) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Some (ops, _) ->
                        // Register capture SSAs in accumulator so body references can find them.
                        // For decomposed memref captures, bind with the RECONSTRUCTED type (memref<?>),
                        // not the slot type (TStruct). The extraction reconstructs the memref.
                        for i in 0 .. layout.Captures.Length - 1 do
                            let cap = layout.Captures.[i]
                            match cap.SourceNodeId with
                            | Some sourceId ->
                                let captureSSA = V cap.SlotIndex
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

            // Determine return type from Lambda type signature
            // For flattened Lambdas with N params, unroll N levels of TFun
            let innerReturnNativeType2 = unrollReturnType (List.length params') node.Type
            let rawReturnType = mapType innerReturnNativeType2 ctx
            let returnType =
                match bodyResult with
                | Some (_, actualTy) -> actualTy
                | None -> narrowType ctx.Coeffects actualValueNode rawReturnType

            // Handle bodyResult based on return type
            let returnSSA =
                match bodyResult with
                | Some (ssa, _) -> Some ssa
                | None ->
                    match innerReturnNativeType2 with
                    | NativeType.TApp ({ NTUKind = Some NTUunit }, []) ->
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
            match tryMatchWithDiagnostics (pFunctionDef funcName funcParams (Some paramNames) returnType bodyOps returnSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok (funcDefOp, _) ->
                let updatedRootScope = ScopeContext.addOp funcDefOp !ctx.RootScopeContext
                ctx.RootScopeContext := updatedRootScope

                // ═══ CLOSURE CONSTRUCTION (parent scope) ═══
                // If this Lambda has captures, build closure struct + uniform pair in parent scope
                // All stores use memref.reinterpret_cast to create typed views at byte offsets
                match closureLayoutOpt with
                | Some layout ->
                    // 1. Get code pointer (func.constant @funcName → real function type → index)
                    // func.constant must use actual function type; we cast to index for storage
                    let innerFuncParamTypes = funcParams |> List.map snd
                    let funcRefSSA = layout.SizeGepSSA  // Repurpose unused heap SSA for func ref
                    let funcTy = TFunc (innerFuncParamTypes, returnType)
                    let funcConstOp = MLIROp.FuncOp (FuncOp.FuncConstant (funcRefSSA, funcName, funcTy))
                    ctx.ScopeContext := ScopeContext.addOp funcConstOp !ctx.ScopeContext
                    // Cast function reference → index for storage in closure struct/pair
                    let codePtrTy = TIndex
                    let castOp = MLIROp.FuncOp (FuncOp.FuncToIndex (layout.CodeAddrSSA, funcRefSSA, innerFuncParamTypes, returnType))
                    ctx.ScopeContext := ScopeContext.addOp castOp !ctx.ScopeContext

                    // 2. Allocate closure struct — PULL allocation strategy from escape analysis coeffect.
                    // Four-point lifetime lattice (closure-representation.md §3.3):
                    //   StackScoped    → alloca              (scope-bounded closure, stack)
                    //   StaticLifetime → get_global          (program-lifetime closure, static storage, no heap)
                    //   EscapesVia*    → alloc               (escaping closure, heap)
                    // StaticLifetime references a module-level memref.global instead of allocating; the
                    // GlobalMemref decl is emitted as a TopLevelOp (collected in staticGlobalDecls).
                    let closureTy = layout.ClosureStructType
                    let escapeKind = getEscapeKindOrDefault node.Id ctx.Coeffects.EscapeAnalysis
                    // Per-closure static-storage symbol names (unique by node id). Only used on the
                    // StaticLifetime path; the decls are threaded to the module root via TopLevelOps.
                    let closureGlobalName = sprintf "__clef_closure_env_%d" nodeIdValue
                    let pairGlobalName = sprintf "__clef_closure_pair_%d" nodeIdValue
                    let mutable staticGlobalDecls : MLIROp list = []
                    let envAllocOp =
                        match escapeKind with
                        | StackScoped ->
                            MLIROp.MemRefOp (MemRefOp.Alloca (layout.ClosureUndefSSA, closureTy, None))
                        | StaticLifetime ->
                            staticGlobalDecls <- staticGlobalDecls @ [ MLIROp.GlobalMemref (closureGlobalName, closureTy) ]
                            MLIROp.MemRefOp (MemRefOp.GetGlobal (layout.ClosureUndefSSA, closureGlobalName, closureTy))
                        | EscapesViaReturn | EscapesViaClosure _ | EscapesViaByRef ->
                            MLIROp.MemRefOp (MemRefOp.AllocStatic (layout.ClosureUndefSSA, closureTy, None))
                    let parentScope2 = ScopeContext.addOp envAllocOp !ctx.ScopeContext
                    ctx.ScopeContext := parentScope2

                    // Shared zero constant for all store indices
                    let zeroSSA = layout.HeapPosSSA  // Repurpose unused heap SSA
                    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, TIndex))
                    let parentScope2a = ScopeContext.addOp zeroOp !ctx.ScopeContext
                    ctx.ScopeContext := parentScope2a

                    // 3. Insert code_ptr at byte offset 0 via reinterpret_cast
                    let codeViewSSA = layout.ClosureWithCodeSSA  // Repurpose as view SSA
                    let codeViewTy = TMemRefStatic (1, codePtrTy)
                    let codeCastOp = MLIROp.MemRefOp (MemRefOp.ReinterpretCast (codeViewSSA, layout.ClosureUndefSSA, 0, 1, closureTy, codeViewTy))
                    let parentScope3 = ScopeContext.addOp codeCastOp !ctx.ScopeContext
                    ctx.ScopeContext := parentScope3
                    let codeStoreOp = MLIROp.MemRefOp (MemRefOp.Store (layout.CodeAddrSSA, codeViewSSA, [zeroSSA], codePtrTy, codeViewTy))
                    let parentScope4 = ScopeContext.addOp codeStoreOp !ctx.ScopeContext
                    ctx.ScopeContext := parentScope4

                    // 4. Insert captures at computed byte offsets via reinterpret_cast
                    // Variable-width: decomposed memref captures use 5 SSAs from CaptureInsertSSAs,
                    // scalar captures use 1 SSA. CaptureInsertSSAs is a flat list sized by
                    // captureConstructionSSACount per capture.
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let sizeOf = mlirTypeSizeForArch arch
                    let mutable captureByteOffset = sizeOf codePtrTy  // Start after code_ptr
                    let mutable ssaIdx = 0  // Index into flat CaptureInsertSSAs list
                    for i in 0 .. layout.Captures.Length - 1 do
                        let cap = layout.Captures.[i]
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
                            match cap.SlotType with
                            | TStruct [("ptr", TIndex); ("len", TIndex)] ->
                                // Decomposed memref: extract ptr + len from source memref, store separately
                                let ptrSSA     = layout.CaptureInsertSSAs.[ssaIdx]
                                let dimZeroSSA = layout.CaptureInsertSSAs.[ssaIdx + 1]
                                let lenSSA     = layout.CaptureInsertSSAs.[ssaIdx + 2]
                                let ptrViewSSA = layout.CaptureInsertSSAs.[ssaIdx + 3]
                                let lenViewSSA = layout.CaptureInsertSSAs.[ssaIdx + 4]
                                ssaIdx <- ssaIdx + 5

                                // Extract base pointer from source memref
                                let srcMemrefTy = TMemRef(TInt (IntWidth 8))
                                let extractPtrOp = MLIROp.MemRefOp(MemRefOp.ExtractBasePtr(ptrSSA, capSSA, srcMemrefTy))
                                ctx.ScopeContext := ScopeContext.addOp extractPtrOp !ctx.ScopeContext

                                // Extract length (dim 0) from source memref
                                let dimZeroOp = MLIROp.ArithOp(ArithOp.ConstI(dimZeroSSA, 0L, TIndex))
                                ctx.ScopeContext := ScopeContext.addOp dimZeroOp !ctx.ScopeContext
                                let dimOp = MLIROp.MemRefOp(MemRefOp.Dim(lenSSA, capSSA, dimZeroSSA, srcMemrefTy))
                                ctx.ScopeContext := ScopeContext.addOp dimOp !ctx.ScopeContext

                                // Store ptr at current byte offset
                                let ptrViewTy = TMemRefStatic(1, TIndex)
                                let ptrCastOp = MLIROp.MemRefOp(MemRefOp.ReinterpretCast(ptrViewSSA, layout.ClosureUndefSSA, captureByteOffset, 1, closureTy, ptrViewTy))
                                ctx.ScopeContext := ScopeContext.addOp ptrCastOp !ctx.ScopeContext
                                let ptrStoreOp = MLIROp.MemRefOp(MemRefOp.Store(ptrSSA, ptrViewSSA, [zeroSSA], TIndex, ptrViewTy))
                                ctx.ScopeContext := ScopeContext.addOp ptrStoreOp !ctx.ScopeContext

                                // Store len at next word offset
                                let lenByteOffset = captureByteOffset + sizeOf TIndex
                                let lenViewTy = TMemRefStatic(1, TIndex)
                                let lenCastOp = MLIROp.MemRefOp(MemRefOp.ReinterpretCast(lenViewSSA, layout.ClosureUndefSSA, lenByteOffset, 1, closureTy, lenViewTy))
                                ctx.ScopeContext := ScopeContext.addOp lenCastOp !ctx.ScopeContext
                                let lenStoreOp = MLIROp.MemRefOp(MemRefOp.Store(lenSSA, lenViewSSA, [zeroSSA], TIndex, lenViewTy))
                                ctx.ScopeContext := ScopeContext.addOp lenStoreOp !ctx.ScopeContext

                            | _ ->
                                // Scalar: single reinterpret_cast + store
                                let viewSSA = layout.CaptureInsertSSAs.[ssaIdx]
                                ssaIdx <- ssaIdx + 1
                                let viewTy = TMemRefStatic (1, cap.SlotType)
                                let castOp = MLIROp.MemRefOp (MemRefOp.ReinterpretCast (viewSSA, layout.ClosureUndefSSA, captureByteOffset, 1, closureTy, viewTy))
                                ctx.ScopeContext := ScopeContext.addOp castOp !ctx.ScopeContext
                                // If capture slot expects index but source is a memref, extract base pointer.
                                // Records, arrays, and other heap-allocated types are memref in MLIR but
                                // the closure slot stores them as index (raw pointer).
                                let actualCapSSA =
                                    if i < savedCaptureSSAs.Length then
                                        match savedCaptureSSAs.[i] with
                                        | Some (_, srcTy) when cap.SlotType = TIndex && (match srcTy with TMemRefStatic _ | TMemRef _ -> true | _ -> false) ->
                                            // Source is memref but slot is index — extract base pointer
                                            let tempIdx = ctx.Accumulator.MLIRTempCounter
                                            ctx.Accumulator.MLIRTempCounter <- tempIdx + 1
                                            let extractSSA = V (10000 + tempIdx)  // High range to avoid collision with pre-computed SSAs
                                            let extractOp = MLIROp.MemRefOp(MemRefOp.ExtractBasePtr(extractSSA, capSSA, srcTy))
                                            ctx.ScopeContext := ScopeContext.addOp extractOp !ctx.ScopeContext
                                            extractSSA
                                        | _ -> capSSA
                                    else capSSA
                                let storeOp = MLIROp.MemRefOp (MemRefOp.Store (actualCapSSA, viewSSA, [zeroSSA], cap.SlotType, viewTy))
                                ctx.ScopeContext := ScopeContext.addOp storeOp !ctx.ScopeContext
                        | None ->
                            printfn "[WARN] LambdaWitness: Capture '%s' (source %A) not found in accumulator for closure %d"
                                cap.Name cap.SourceNodeId nodeIdValue
                            // Still advance ssaIdx for consistency
                            match cap.SlotType with
                            | TStruct [("ptr", TIndex); ("len", TIndex)] -> ssaIdx <- ssaIdx + 5
                            | _ -> ssaIdx <- ssaIdx + 1
                        captureByteOffset <- captureByteOffset + sizeOf cap.SlotType

                    // 5. Build uniform pair {code_ptr, env_ptr}
                    // Pair type matches TypeMapping: TMemRefStatic(2, TIndex) = memref<2xindex>
                    // Same escape analysis governs the pair — it shares the closure's lifetime.
                    let pairTy = TMemRefStatic(2, TIndex)

                    let pairAllocOp =
                        match escapeKind with
                        | StackScoped ->
                            MLIROp.MemRefOp (MemRefOp.Alloca (layout.PairUndefSSA, pairTy, None))
                        | StaticLifetime ->
                            staticGlobalDecls <- staticGlobalDecls @ [ MLIROp.GlobalMemref (pairGlobalName, pairTy) ]
                            MLIROp.MemRefOp (MemRefOp.GetGlobal (layout.PairUndefSSA, pairGlobalName, pairTy))
                        | EscapesViaReturn | EscapesViaClosure _ | EscapesViaByRef ->
                            MLIROp.MemRefOp (MemRefOp.AllocStatic (layout.PairUndefSSA, pairTy, None))
                    ctx.ScopeContext := ScopeContext.addOp pairAllocOp !ctx.ScopeContext

                    // Store code_ptr at element [0] — direct store, same element type
                    let pairCodeStoreOp = MLIROp.MemRefOp (MemRefOp.Store (layout.CodeAddrSSA, layout.PairUndefSSA, [zeroSSA], TIndex, pairTy))
                    ctx.ScopeContext := ScopeContext.addOp pairCodeStoreOp !ctx.ScopeContext

                    // Extract env_ptr = base pointer of closure struct
                    let envPtrSSA = layout.HeapPosPtrSSA  // Reuse a pre-allocated SSA for env_ptr extraction
                    let envExtractOp = MLIROp.MemRefOp (MemRefOp.ExtractBasePtr (envPtrSSA, layout.ClosureUndefSSA, closureTy))
                    ctx.ScopeContext := ScopeContext.addOp envExtractOp !ctx.ScopeContext

                    // Store env_ptr at element [1]
                    let oneSSA = layout.PairWithCodeSSA  // Repurpose as constant 1 SSA
                    let oneOp = MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, TIndex))
                    ctx.ScopeContext := ScopeContext.addOp oneOp !ctx.ScopeContext
                    let pairEnvStoreOp = MLIROp.MemRefOp (MemRefOp.Store (envPtrSSA, layout.PairUndefSSA, [oneSSA], TIndex, pairTy))
                    ctx.ScopeContext := ScopeContext.addOp pairEnvStoreOp !ctx.ScopeContext

                    // Register closure pair in accumulator for BindingWitness/VarRefWitness to find
                    MLIRAccumulator.bindNode node.Id layout.PairUndefSSA pairTy ctx.Accumulator

                    // On the StaticLifetime path, staticGlobalDecls carries the module-level
                    // memref.global declarations backing this closure's env and pair; they ride
                    // out as TopLevelOps to the module root. Empty on every other lifetime path.
                    { InlineOps = []; TopLevelOps = staticGlobalDecls; Result = TRValue { SSA = layout.PairUndefSSA; Type = pairTy } }

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

