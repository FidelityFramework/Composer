/// MLIRTransfer - Canonical Transfer from SemanticGraph to MLIR
///
/// CANONICAL ARCHITECTURE (January 2026):
/// See: mlir_transfer_canonical_architecture memory
///
/// The Three Concerns:
/// - PSGZipper: Pure navigation (Focus, Path, Graph)
/// - TransferCoeffects: Pre-computed, immutable coeffects
/// - MLIRAccumulator: Mutable fold state
///
/// Transfer is a FOLD: witnesses RETURN codata, the fold accumulates.
/// NO push-model emission. NO mutable coeffects. NO state in zipper.
module Alex.Traversal.MLIRTransfer

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Dialects.LLVM.Templates
open Alex.Dialects.Arith.Templates
open Alex.Traversal.PSGZipper
open Alex.Traversal.TransferTypes
open Alex.Bindings.PlatformTypes
open Alex.Patterns.SemanticPatterns
open Alex.CodeGeneration.TypeMapping
open PSGElaboration.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// MODULE ALIASES
// ═══════════════════════════════════════════════════════════════════════════

module MutAnalysis = PSGElaboration.MutabilityAnalysis
module SSAAssign = PSGElaboration.SSAAssignment
module StringCollect = PSGElaboration.StringCollection
module PatternAnalysis = PSGElaboration.PatternBindingAnalysis
module YieldStateIndices = PSGElaboration.YieldStateIndices
module PlatformRes = PSGElaboration.PlatformBindingResolution

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT ACCESSORS (Local versions for backward compatibility)
// ═══════════════════════════════════════════════════════════════════════════

/// Get single pre-assigned SSA for a node (coeffects version)
let private requireSSAFromCoeffs (nodeId: NodeId) (coeffs: TransferCoeffects) : SSA =
    match SSAAssign.lookupSSA nodeId coeffs.SSA with
    | Some ssa -> ssa
    | None -> failwithf "No SSA for node %A" nodeId

/// Get all pre-assigned SSAs for a node (coeffects version)
let private requireSSAsFromCoeffs (nodeId: NodeId) (coeffs: TransferCoeffects) : SSA list =
    match SSAAssign.lookupSSAs nodeId coeffs.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Derive byte length of string in UTF-8 encoding
let deriveStringByteLength (s: string) : int =
    System.Text.Encoding.UTF8.GetByteCount(s)

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT COMPUTATION
// ═══════════════════════════════════════════════════════════════════════════

/// Compute all coeffects from graph (called ONCE at start)
let computeCoeffects (graph: SemanticGraph) (isFreestanding: bool) : TransferCoeffects =
    let hostPlatform = TargetPlatform.detectHost()
    let runtimeMode = if isFreestanding then Freestanding else Console
    {
        SSA = SSAAssign.assignSSA hostPlatform.Arch graph
        Platform = PlatformRes.analyze graph runtimeMode hostPlatform.OS hostPlatform.Arch
        Mutability = MutAnalysis.analyze graph
        PatternBindings = PatternAnalysis.analyze graph
        Strings = StringCollect.collect graph
        YieldStates = YieldStateIndices.run graph
        EntryPointLambdaIds = MutAnalysis.findEntryPointLambdaIds graph
    }

// ═══════════════════════════════════════════════════════════════════════════
// RECURSIVE TRAVERSAL WITH WITNESSING
// ═══════════════════════════════════════════════════════════════════════════

/// Visit a node and its children, returning comprehensive WitnessOutput
///
/// ARCHITECTURAL PRINCIPLE: This is the FOLD. It:
/// 1. Navigates via zipper (Huet-style, purely positional)
/// 2. Classifies via SemanticKind match (semantic lens)
/// 3. Calls witnesses which RETURN WitnessOutput (codata)
/// 4. ACCUMULATES the returned codata (single point of accumulation)
///
/// The fold is the only place that adds to TopLevelOps.
/// Witnesses return; the fold accumulates.
let rec private visitNode
    (ctx: WitnessContext)
    (z: PSGZipper)
    (nodeId: NodeId)
    : WitnessOutput =

    let nodeIdVal = NodeId.value nodeId
    let acc = ctx.Accumulator

    // DAG handling: if already visited, recall the cached result
    if MLIRAccumulator.isVisited nodeIdVal acc then
        match MLIRAccumulator.recallNode nodeIdVal acc with
        | Some (ssa, ty) -> WitnessOutput.value { SSA = ssa; Type = ty }
        | None -> WitnessOutput.empty
    else
        MLIRAccumulator.markVisited nodeIdVal acc

        match SemanticGraph.tryGetNode nodeId ctx.Graph with
        | None ->
            WitnessOutput.error (sprintf "Node %d not found" nodeIdVal)
        | Some node ->
            // Classify and witness the node
            let output = classifyAndWitness ctx z node

            // ACCUMULATE: The fold adds top-level ops (witnesses just return them)
            MLIRAccumulator.addTopLevelOps output.TopLevelOps acc

            // Record the result for DAG handling
            recordResult nodeIdVal output.Result acc

            // Return with TopLevelOps cleared (they've been accumulated)
            { output with TopLevelOps = [] }

/// Classify node by SemanticKind and witness it
/// Returns WitnessOutput with all codata (inline ops, top-level ops, result)
///
/// NOTE: Witnesses currently return (MLIROp list * TransferResult).
/// This function bridges to WitnessOutput. As witnesses are updated,
/// they will return WitnessOutput directly.
and private classifyAndWitness
    (ctx: WitnessContext)
    (z: PSGZipper)
    (node: SemanticNode)
    : WitnessOutput =

    let acc = ctx.Accumulator

    match node.Kind with
    // ─────────────────────────────────────────────────────────────────────
    // LITERALS
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Literal lit ->
        let ops, result = Alex.Witnesses.LiteralWitness.witness ctx.Coeffects.SSA ctx.Coeffects.Platform.TargetArch node.Id lit
        WitnessOutput.inline' ops result

    // ─────────────────────────────────────────────────────────────────────
    // VARIABLE REFERENCES - Direct pattern + template handling
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.VarRef (name, defIdOpt) ->
        let mutability = ctx.Coeffects.Mutability
        let arch = ctx.Coeffects.Platform.TargetArch
        
        // Helper: get type from definition node
        let getDefType nodeId =
            match SemanticGraph.tryGetNode nodeId ctx.Graph with
            | Some n -> mapNativeTypeWithGraphForArch arch ctx.Graph n.Type
            | None -> MLIRTypes.i32
        
        // 1. Module-level mutable: addressof + load
        match mutability with
        | ModuleLevelMutableRef name globalName ->
            let ssas = requireSSAs node.Id ctx
            let ptrSSA, loadSSA = ssas.[0], ssas.[1]
            let elemType = defIdOpt |> Option.map getDefType |> Option.defaultValue MLIRTypes.i32
            let addrOp = MLIROp.LLVMOp (AddressOf (ptrSSA, GFunc globalName))
            let loadOp = MLIROp.LLVMOp (Load (loadSSA, ptrSSA, elemType, NotAtomic))
            WitnessOutput.inline' [addrOp; loadOp] (TRValue { SSA = loadSSA; Type = elemType })
        | _ ->
        
        match defIdOpt with
        | None ->
            // No defId - lookup by name in accumulator
            match MLIRAccumulator.recallVar name acc with
            | Some (ssa, ty) -> WitnessOutput.value { SSA = ssa; Type = ty }
            | None -> WitnessOutput.error (sprintf "Variable '%s' has no definition" name)
        | Some defId ->
            let defIdVal = NodeId.value defId
            
            // 2. Addressed mutable: load from alloca
            match mutability with
            | AddressedMutableRef defIdVal () ->
                match SSAAssign.lookupSSAs defId ctx.Coeffects.SSA with
                | Some ssas when ssas.Length >= 2 ->
                    let allocaSSA = ssas.[1]
                    let loadSSA = requireSSA node.Id ctx
                    let elemType = getDefType defId
                    let loadOp = MLIROp.LLVMOp (Load (loadSSA, allocaSSA, elemType, NotAtomic))
                    WitnessOutput.inline' [loadOp] (TRValue { SSA = loadSSA; Type = elemType })
                | _ -> WitnessOutput.error (sprintf "No alloca SSA for addressed mutable '%s'" name)
            | _ ->
            
            // 3. Captured variable
            if MLIRAccumulator.isCapturedVariable name acc then
                match MLIRAccumulator.recallVar name acc with
                | Some (ptrSSA, _) when MLIRAccumulator.isCapturedMutable name acc ->
                    let elemType = getDefType defId
                    let loadSSA = requireSSA node.Id ctx
                    let loadOp = MLIROp.LLVMOp (Load (loadSSA, ptrSSA, elemType, NotAtomic))
                    WitnessOutput.inline' [loadOp] (TRValue { SSA = loadSSA; Type = elemType })
                | Some (ssa, ty) ->
                    WitnessOutput.value { SSA = ssa; Type = ty }
                | None ->
                    WitnessOutput.error (sprintf "Captured variable '%s' not in scope" name)
            else
            
            // 4. Function parameter or let-bound value: lookup in accumulator/coeffects
            match MLIRAccumulator.recallVar name acc with
            | Some (ssa, ty) -> WitnessOutput.value { SSA = ssa; Type = ty }
            | None ->
            match MLIRAccumulator.recallNode defIdVal acc with
            | Some (ssa, ty) -> WitnessOutput.value { SSA = ssa; Type = ty }
            | None ->
            match SSAAssign.lookupSSA defId ctx.Coeffects.SSA with
            | Some ssa -> WitnessOutput.value { SSA = ssa; Type = getDefType defId }
            | None ->
            // Function reference marker
            WitnessOutput.inline' [] (TRBuiltin (name, []))

    // ─────────────────────────────────────────────────────────────────────
    // MODULE - Visit children, no ops produced
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.ModuleDef (_name, members) ->
        visitChildren ctx z members

    // ─────────────────────────────────────────────────────────────────────
    // SEQUENTIAL - Visit children in order, return last result
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Sequential exprs ->
        visitChildren ctx z exprs

    // ─────────────────────────────────────────────────────────────────────
    // LET EXPRESSION - Direct pattern + template handling
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Binding (name, isMut, _isRec, _isEntry) ->
        match node.Children with
        | [valueId] ->
            let valueOutput = visitNode ctx z valueId
            let nodeIdVal = NodeId.value node.Id
            let mutability = ctx.Coeffects.Mutability
            let arch = ctx.Coeffects.Platform.TargetArch
            
            // 1. Module-level mutable: emit global definition
            match mutability with
            | ModuleLevelMutable nodeIdVal globalName when isMut ->
                let mlirType = mapNativeTypeWithGraphForArch arch ctx.Graph node.Type
                let globalOp = MLIROp.LLVMOp (GlobalDef (globalName, "zeroinitializer", mlirType, false))
                { InlineOps = valueOutput.InlineOps
                  TopLevelOps = [globalOp] @ valueOutput.TopLevelOps
                  Result = TRVoid }
            | _ ->
            
            // 2. Addressed mutable: alloca + store
            match mutability with
            | AddressedMutable nodeIdVal () when isMut ->
                match valueOutput.Result with
                | TRValue valueVal ->
                    match SSAAssign.lookupSSAs node.Id ctx.Coeffects.SSA with
                    | Some ssas when ssas.Length >= 2 ->
                        let oneSSA = ssas.[0]
                        let allocaSSA = ssas.[1]
                        let elemType = valueVal.Type
                        let oneOp = MLIROp.ArithOp (ArithOp.ConstI (oneSSA, 1L, MLIRTypes.i64))
                        let allocaOp = MLIROp.LLVMOp (Alloca (allocaSSA, oneSSA, elemType, None))
                        let storeOp = MLIROp.LLVMOp (Store (valueVal.SSA, allocaSSA, elemType, NotAtomic))
                        // Record the alloca pointer for later VarRef lookups
                        MLIRAccumulator.bindVar name allocaSSA MLIRTypes.ptr acc
                        MLIRAccumulator.bindNode nodeIdVal allocaSSA MLIRTypes.ptr acc
                        { InlineOps = valueOutput.InlineOps @ [oneOp; allocaOp; storeOp]
                          TopLevelOps = valueOutput.TopLevelOps
                          Result = TRValue { SSA = allocaSSA; Type = MLIRTypes.ptr } }
                    | _ ->
                        WitnessOutput.error (sprintf "No SSAs for mutable '%s'" name)
                | TRError msg ->
                    { valueOutput with Result = TRError msg }
                | _ ->
                    WitnessOutput.error (sprintf "Mutable '%s' has void value" name)
            | _ ->
            
            // 3. Regular binding: value flows through, record association
            match valueOutput.Result with
            | TRValue v ->
                MLIRAccumulator.bindVar name v.SSA v.Type acc
                MLIRAccumulator.bindNode nodeIdVal v.SSA v.Type acc
                valueOutput
            | TRVoid ->
                valueOutput
            | TRError _ ->
                valueOutput
            | TRBuiltin _ ->
                valueOutput
        | _ ->
            WitnessOutput.error (sprintf "Let expression '%s' has wrong number of children" name)

    // ─────────────────────────────────────────────────────────────────────
    // SET (Mutable assignment) - Coeffect-driven classification
    // The target is a NodeId - we look at the target node to classify it
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Set (targetId, valueId) ->
        let valueOutput = visitNode ctx z valueId
        let mutability = ctx.Coeffects.Mutability
        
        // Look up the target node to understand what we're assigning to
        match SemanticGraph.tryGetNode targetId ctx.Graph with
        | None ->
            WitnessOutput.error (sprintf "Set target node %d not found" (NodeId.value targetId))
        | Some targetNode ->
            match valueOutput.Result with
            | TRValue v ->
                // Classify the target via coeffects
                match targetNode.Kind with
                | SemanticKind.VarRef (name, defIdOpt) ->
                    // 1. Module-level mutable: addressof + store
                    match mutability with
                    | ModuleLevelMutableRef name globalName ->
                        let addrSSA = requireSSA node.Id ctx
                        let addrOp = MLIROp.LLVMOp (AddressOf (addrSSA, GFunc globalName))
                        let storeOp = MLIROp.LLVMOp (Store (v.SSA, addrSSA, v.Type, NotAtomic))
                        { valueOutput with
                            InlineOps = valueOutput.InlineOps @ [addrOp; storeOp]
                            Result = TRVoid }
                    | _ ->
                    
                    // 2. Captured mutable: store through captured pointer
                    if MLIRAccumulator.isCapturedMutable name acc then
                        match MLIRAccumulator.recallVar name acc with
                        | Some (ptrSSA, _) ->
                            let storeOp = MLIROp.LLVMOp (Store (v.SSA, ptrSSA, v.Type, NotAtomic))
                            { valueOutput with
                                InlineOps = valueOutput.InlineOps @ [storeOp]
                                Result = TRVoid }
                        | None ->
                            { valueOutput with Result = TRError (sprintf "Captured mutable '%s' not in scope" name) }
                    else
                    
                    // 3. Addressed mutable: store to alloca
                    match defIdOpt with
                    | Some defId ->
                        let defIdVal = NodeId.value defId
                        match mutability with
                        | AddressedMutableRef defIdVal () ->
                            match SSAAssign.lookupSSAs defId ctx.Coeffects.SSA with
                            | Some ssas when ssas.Length >= 2 ->
                                let allocaSSA = ssas.[1]
                                let storeOp = MLIROp.LLVMOp (Store (v.SSA, allocaSSA, v.Type, NotAtomic))
                                { valueOutput with
                                    InlineOps = valueOutput.InlineOps @ [storeOp]
                                    Result = TRVoid }
                            | _ ->
                                { valueOutput with Result = TRError (sprintf "No alloca for '%s'" name) }
                        | _ ->
                            { valueOutput with Result = TRError (sprintf "Set target '%s' is not mutable" name) }
                    | None ->
                        { valueOutput with Result = TRError (sprintf "Set target '%s' has no definition" name) }
                | _ ->
                    WitnessOutput.error (sprintf "Set target is not a VarRef: %A" targetNode.Kind)
            | TRError msg ->
                { valueOutput with Result = TRError msg }
            | _ ->
                { valueOutput with Result = TRVoid }

    // ─────────────────────────────────────────────────────────────────────
    // APPLICATION
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Application (funcId, argIds) ->
        // Visit function and arguments
        let funcOutput = visitNode ctx z funcId
        let argOutputs = argIds |> List.map (visitNode ctx z)
        let combinedArgs = WitnessOutput.combineAll argOutputs

        // Witness the application
        let appOps, appResult = Alex.Witnesses.Application.Witness.witness ctx node

        { InlineOps = funcOutput.InlineOps @ combinedArgs.InlineOps @ appOps
          TopLevelOps = funcOutput.TopLevelOps @ combinedArgs.TopLevelOps
          Result = appResult }

    // ─────────────────────────────────────────────────────────────────────
    // LAMBDA - Pre-bind params, visit body, witness
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Lambda (params', bodyId, _captures, _enclosing, _context) ->
        // Pre-bind parameters (pushes scope, returns MLIR params)
        let funcParams = Alex.Witnesses.LambdaWitness.preBindParams ctx node

        // Visit body
        let bodyOutput = visitNode ctx z bodyId

        // Create the body callback for witness (pull model)
        let witnessBody = fun (_ctx: WitnessContext) -> (bodyOutput.InlineOps, bodyOutput.Result)

        // Witness the lambda - returns (funcDefOpt, closureOps, result)
        let funcDefOpt, closureOps, result =
            Alex.Witnesses.LambdaWitness.witness params' bodyId node funcParams witnessBody ctx

        // Function definition goes to TopLevelOps (returned, not mutated)
        let topLevelOps =
            match funcDefOpt with
            | Some funcDef -> funcDef :: bodyOutput.TopLevelOps
            | None -> bodyOutput.TopLevelOps

        { InlineOps = closureOps
          TopLevelOps = topLevelOps
          Result = result }

    // ─────────────────────────────────────────────────────────────────────
    // IF-THEN-ELSE
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.IfThenElse (condId, thenId, elseIdOpt) ->
        let condOutput = visitNode ctx z condId
        let condSSA = match condOutput.Result with TRValue v -> v.SSA | _ -> V 0

        let thenOutput = visitNode ctx z thenId
        let thenSSA = match thenOutput.Result with TRValue v -> Some v.SSA | _ -> None

        let elseOutput, elseSSA =
            match elseIdOpt with
            | Some elseId ->
                let o = visitNode ctx z elseId
                let ssa = match o.Result with TRValue v -> Some v.SSA | _ -> None
                Some o, ssa
            | None -> None, None

        let resultType = match thenOutput.Result with TRValue v -> Some v.Type | _ -> None

        let elseOpsOpt = elseOutput |> Option.map (fun o -> o.InlineOps)
        let ifOps, ifResult =
            Alex.Witnesses.ControlFlowWitness.witnessIfThenElse node.Id ctx condSSA thenOutput.InlineOps thenSSA elseOpsOpt elseSSA resultType

        let allTopLevel =
            condOutput.TopLevelOps @ thenOutput.TopLevelOps @
            (elseOutput |> Option.map (fun o -> o.TopLevelOps) |> Option.defaultValue [])

        { InlineOps = condOutput.InlineOps @ ifOps
          TopLevelOps = allTopLevel
          Result = ifResult }

    // ─────────────────────────────────────────────────────────────────────
    // WHILE LOOP
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.WhileLoop (condId, bodyId) ->
        let condOutput = visitNode ctx z condId
        let condSSA = match condOutput.Result with TRValue v -> v.SSA | _ -> V 0
        let bodyOutput = visitNode ctx z bodyId

        let whileOps, whileResult =
            Alex.Witnesses.ControlFlowWitness.witnessWhileLoop node.Id ctx condOutput.InlineOps condSSA bodyOutput.InlineOps []

        { InlineOps = whileOps
          TopLevelOps = condOutput.TopLevelOps @ bodyOutput.TopLevelOps
          Result = whileResult }

    // ─────────────────────────────────────────────────────────────────────
    // TYPE ANNOTATION - Transparent, pass through
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.TypeAnnotation (innerExpr, _) ->
        visitNode ctx z innerExpr

    // ─────────────────────────────────────────────────────────────────────
    // INTRINSIC / PLATFORM - Handled via Application when called
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.Intrinsic _
    | SemanticKind.PlatformBinding _ ->
        WitnessOutput.empty

    // ─────────────────────────────────────────────────────────────────────
    // PATTERN (Parameter) - handled by Lambda preBindParams
    // ─────────────────────────────────────────────────────────────────────
    | SemanticKind.PatternBinding _ ->
        WitnessOutput.empty

    // ─────────────────────────────────────────────────────────────────────
    // UNHANDLED
    // ─────────────────────────────────────────────────────────────────────
    | _ ->
        let msg = sprintf "MLIRTransfer: Unhandled node kind %A at node %d" node.Kind (NodeId.value node.Id)
        WitnessOutput.error msg

/// Record a result in the accumulator (for DAG handling)
and private recordResult (nodeIdVal: int) (result: TransferResult) (acc: MLIRAccumulator) : unit =
    match result with
    | TRValue v -> MLIRAccumulator.bindNode nodeIdVal v.SSA v.Type acc
    | _ -> ()

/// Visit children and combine their outputs
and private visitChildren
    (ctx: WitnessContext)
    (z: PSGZipper)
    (childIds: NodeId list)
    : WitnessOutput =
    childIds
    |> List.map (visitNode ctx z)
    |> WitnessOutput.combineAll

// ═══════════════════════════════════════════════════════════════════════════
// MLIR GENERATION
// ═══════════════════════════════════════════════════════════════════════════

/// Generate MLIR module wrapper with string table and operations
let private wrapInModule (ops: MLIROp list) (stringTable: StringCollect.StringTable) : string =
    let sb = System.Text.StringBuilder()

    // Module header
    sb.AppendLine("module {") |> ignore

    // String constants from StringTable coeffect
    for KeyValue(_hash, entry) in stringTable do
        let escaped = entry.Content.Replace("\\", "\\\\").Replace("\"", "\\\"").Replace("\n", "\\n")
        sb.AppendLine(sprintf "  llvm.mlir.global private constant %s(\"%s\\00\") : !llvm.array<%d x i8>"
            entry.GlobalName escaped (entry.ByteLength + 1)) |> ignore

    // Operations (reversed since we accumulated in reverse order)
    for op in List.rev ops do
        sb.AppendLine(sprintf "  %s" (Alex.Dialects.Core.Serialize.opToString op)) |> ignore

    // Module footer
    sb.AppendLine("}") |> ignore

    sb.ToString()

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// Transfer graph to MLIR with diagnostics
/// This is the main entry point called by CompilationOrchestrator
let transferGraphWithDiagnostics
    (graph: SemanticGraph)
    (isFreestanding: bool)
    (intermediatesDir: string option)
    : string * string list =

    // Compute all coeffects ONCE before traversal
    let coeffects = computeCoeffects graph isFreestanding

    // Serialize coeffects if requested (debugging)
    match intermediatesDir with
    | Some dir ->
        PSGElaboration.PreprocessingSerializer.serialize dir coeffects.SSA coeffects.EntryPointLambdaIds graph
    | None -> ()

    // Create accumulator
    let acc = MLIRAccumulator.empty()

    // Create witness context
    let ctx: WitnessContext = {
        Coeffects = coeffects
        Accumulator = acc
        Graph = graph
    }

    // Visit entry points - the fold accumulates TopLevelOps from returned codata
    for entryId in graph.EntryPoints do
        match PSGZipper.create graph entryId with
        | Some z ->
            // The zipper provides positional attention
            // visitNode returns WitnessOutput with all codata
            // TopLevelOps are accumulated by the fold (in visitNode)
            let _output = visitNode ctx z entryId
            // Note: TopLevelOps already accumulated by visitNode
            // InlineOps from entry point are typically empty (module-level)
            ()
        | None ->
            MLIRAccumulator.addError (sprintf "Could not create zipper at entry point %A" entryId) acc

    // Generate MLIR from accumulated top-level ops
    let mlir = wrapInModule acc.TopLevelOps coeffects.Strings

    (mlir, List.rev acc.Errors)

/// Transfer graph to MLIR (no diagnostics)
let transferGraph (graph: SemanticGraph) (isFreestanding: bool) : string =
    fst (transferGraphWithDiagnostics graph isFreestanding None)
