/// KernelModuleWitness - Witness [<KernelModule>] bindings for NPU compute kernels
///
/// Handles Binding nodes with declRoot = KernelModule. The child is a RecordExpr
/// (ElementKernel<'T>) containing:
///   - Compute: VarRef to a pure binary function ('T -> 'T -> 'T)
///   - Shape: RecordExpr { Elements: int; Grain: int }
///
/// Generates complete MLIR-AIE module text as a RawMLIR op:
///   aie.device(npu2) with tiles, object FIFOs, compute cores, and runtime sequence.
///
/// The Compute function body is lowered to arith ops inside each aie.core.
/// For hello world, only arith.muli is generated (from `fun a b -> a * b`).
///
/// Architecture:
///   - RecordExpr metadata extraction follows HardwareModuleWitness scope boundary pattern
///     (direct PSG traversal for compile-time structure; metadata nodes marked visited)
///   - Element type resolution uses mapType → Serialize.typeToString (full type pipeline)
///   - Arith op resolution uses classifyAtomicOp via tryMatch + pIntrinsicApplication
///     (monadic catamorphism through XParsec combinator layer)
///
/// NPU-only: registered conditionally in WitnessRegistry.
/// SCOPE WITNESS: receives combinator for recursive sub-graph traversal.
module Alex.Witnesses.KernelModuleWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.ScopeContext
open Alex.XParsec.PSGCombinators

// ═══════════════════════════════════════════════════════════
// PSG STRUCTURE EXTRACTION (Scope Boundary Pattern)
//
// Same pattern as HardwareModuleWitness: direct PSG traversal
// for compile-time metadata in RecordExpr fields. These nodes
// are structural (not runtime operations) and are marked visited
// after extraction.
// ═══════════════════════════════════════════════════════════

/// Unwrap TypeAnnotation nodes to find the underlying node.
let rec private unwrapTypeAnnotation (graph: SemanticGraph) (nodeId: NodeId) : NodeId =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.TypeAnnotation (innerNodeId, _) -> unwrapTypeAnnotation graph innerNodeId
        | _ -> nodeId
    | None -> nodeId

/// Extract the RecordExpr fields from the ElementKernel child of a KernelModule Binding.
let private extractKernelFields (graph: SemanticGraph) (bindingNode: SemanticNode) : (string * NodeId) list option =
    match bindingNode.Children with
    | [valueId] ->
        let unwrappedId = unwrapTypeAnnotation graph valueId
        match SemanticGraph.tryGetNode unwrappedId graph with
        | Some valueNode ->
            match valueNode.Kind with
            | SemanticKind.RecordExpr (fields, _) -> Some fields
            | _ -> None
        | None -> None
    | _ -> None

/// Find a field's NodeId by name
let private findField (fieldName: string) (fields: (string * NodeId) list) : NodeId option =
    fields |> List.tryFind (fun (n, _) -> n = fieldName) |> Option.map snd

/// Resolve an integer literal from a PSG node (following VarRef → Binding → Literal chains).
/// Compile-time constants in Shape { Elements; Grain } are evaluated structurally.
let rec private resolveIntLiteral (graph: SemanticGraph) (nodeId: NodeId) : int option =
    match SemanticGraph.tryGetNode nodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.Literal (NativeLiteral.Int (value, _)) -> Some (int value)
        | SemanticKind.VarRef (_, Some bindingId) ->
            match SemanticGraph.tryGetNode bindingId graph with
            | Some bindingNode ->
                match bindingNode.Children with
                | [valueId] -> resolveIntLiteral graph valueId
                | _ -> None
            | None -> None
        | SemanticKind.TypeAnnotation (wrappedId, _) ->
            resolveIntLiteral graph wrappedId
        | _ -> None
    | None -> None

/// Extract Shape record fields (Elements, Grain) from a nested RecordExpr
let private extractShape (graph: SemanticGraph) (shapeNodeId: NodeId) : (int * int) option =
    let unwrappedId = unwrapTypeAnnotation graph shapeNodeId
    match SemanticGraph.tryGetNode unwrappedId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.RecordExpr (fields, _) ->
            let elements = findField "Elements" fields |> Option.bind (resolveIntLiteral graph)
            let grain = findField "Grain" fields |> Option.bind (resolveIntLiteral graph)
            match elements, grain with
            | Some e, Some g when g > 0 && e > 0 && e % g = 0 -> Some (e, g)
            | _ -> None
        | _ -> None
    | None -> None

/// Mark all nodes in a subtree as visited (child edges only).
/// Used on metadata children; safe because VarRef targets are followed separately.
let rec private markSubtreeVisited (graph: SemanticGraph) (nodeId: NodeId) (visited: ref<Set<NodeId>>) : unit =
    if not (Set.contains nodeId !visited) then
        visited := Set.add nodeId !visited
        match SemanticGraph.tryGetNode nodeId graph with
        | Some node ->
            for childId in node.Children do
                markSubtreeVisited graph childId visited
        | None -> ()

/// Mark metadata chain as visited, following VarRef binding targets transitively.
/// Compute function and its body are compile-time metadata read structurally;
/// they don't produce MLIR ops, but coverage validation needs to know they were consumed.
let rec private markMetadataChainVisited (graph: SemanticGraph) (nodeId: NodeId) (visited: ref<Set<NodeId>>) : unit =
    if not (Set.contains nodeId !visited) then
        visited := Set.add nodeId !visited
        match SemanticGraph.tryGetNode nodeId graph with
        | Some node ->
            for childId in node.Children do
                markMetadataChainVisited graph childId visited
            match node.Kind with
            | SemanticKind.VarRef (_, Some bindingId) ->
                markMetadataChainVisited graph bindingId visited
            | _ -> ()
        | None -> ()

// ═══════════════════════════════════════════════════════════
// TYPE AND OPERATION RESOLUTION (via Coeffects + Combinators)
// ═══════════════════════════════════════════════════════════

/// Resolve the element type ('T) from the Compute function's NativeType.
/// Compute: 'T -> 'T -> 'T — we walk through TFun to the innermost return type,
/// then use mapType (TransferTypes) to convert NativeType → MLIRType,
/// and typeToString (Serialize) to produce the MLIR text.
///
/// This flows through the full type resolution pipeline:
///   NativeType → mapNativeTypeForTarget → MLIRType → typeToString → string
let private resolveElementType (ctx: WitnessContext) (computeNodeId: NodeId) : string option =
    match SemanticGraph.tryGetNode computeNodeId ctx.Graph with
    | Some node ->
        // Walk through TFun chain: 'T -> 'T -> 'T  →  TFun(T, TFun(T, T))
        let rec innerReturnType (ty: NativeType) =
            match ty with
            | NativeType.TFun (_, range) -> innerReturnType range
            | other -> other
        let retNativeType = innerReturnType node.Type
        let mlirType = mapType retNativeType ctx
        let typeStr = typeToString mlirType
        Some typeStr
    | None -> None

/// Resolve the Compute function name from a VarRef
let private resolveComputeFunctionName (graph: SemanticGraph) (computeNodeId: NodeId) : string option =
    match SemanticGraph.tryGetNode computeNodeId graph with
    | Some node ->
        match node.Kind with
        | SemanticKind.VarRef (name, _) -> Some name
        | _ -> None
    | None -> None

/// Resolve the arith operation from the Compute function body via classifyAtomicOp.
///
/// Walks VarRef → Binding → Lambda → body, then uses tryMatch with
/// pIntrinsicApplication to identify the intrinsic monadically through
/// the XParsec combinator layer. classifyAtomicOp maps the IntrinsicInfo
/// to the MLIR arith operation string.
///
/// Returns (mlirOpName, isFloat) or None.
let private resolveComputeOp (ctx: WitnessContext) (computeNodeId: NodeId) : (string * bool) option =
    let graph = ctx.Graph

    // Walk VarRef → Binding target → Lambda → body → find Application with Intrinsic
    let rec findIntrinsicInBody (nodeId: NodeId) : IntrinsicInfo option =
        match SemanticGraph.tryGetNode nodeId graph with
        | Some node ->
            match node.Kind with
            | SemanticKind.Application (funcId, _argIds) ->
                // Resolve function node through TypeAnnotation
                let funcNodeOpt =
                    match SemanticGraph.tryGetNode funcId graph with
                    | Some funcNode ->
                        match funcNode.Kind with
                        | SemanticKind.TypeAnnotation (innerFuncId, _) ->
                            SemanticGraph.tryGetNode innerFuncId graph
                        | _ -> Some funcNode
                    | None -> None
                match funcNodeOpt with
                | Some funcNode ->
                    match funcNode.Kind with
                    | SemanticKind.Intrinsic info -> Some info
                    | _ -> None
                | None -> None
            | SemanticKind.Sequential children ->
                match List.tryLast children with
                | Some lastId -> findIntrinsicInBody lastId
                | None -> None
            | SemanticKind.Lambda (_, bodyId, _, _, _) ->
                findIntrinsicInBody bodyId
            | SemanticKind.Binding (_, _, _, _) ->
                match node.Children with
                | [childId] -> findIntrinsicInBody childId
                | _ -> None
            | _ -> None
        | None -> None

    // Follow VarRef → Binding → Lambda → body
    match SemanticGraph.tryGetNode computeNodeId graph with
    | Some { Kind = SemanticKind.VarRef (_, Some defId) } ->
        match SemanticGraph.tryGetNode defId graph with
        | Some defNode ->
            match defNode.Children with
            | [lambdaId] ->
                match findIntrinsicInBody lambdaId with
                | Some info ->
                    // Use classifyAtomicOp to map to MLIR arith operation
                    match classifyAtomicOp info with
                    | BinaryArith op ->
                        // Determine int/float suffix from element type
                        let isFloat =
                            match resolveElementType ctx computeNodeId with
                            | Some s when s.StartsWith("f") -> true
                            | _ -> false
                        let suffixed =
                            if isFloat then
                                match op with
                                | "mul" -> "mulf"
                                | "add" -> "addf"
                                | "sub" -> "subf"
                                | "div" -> "divf"
                                | other -> other
                            else
                                match op with
                                | "mul" -> "muli"
                                | "add" -> "addi"
                                | "sub" -> "subi"
                                | "div" -> "divi"
                                | "rem" -> "remi"
                                | other -> other + "i"
                        Some (suffixed, isFloat)
                    | _ -> None
                | None -> None
            | _ -> None
        | None -> None
    | _ -> None

// ═══════════════════════════════════════════════════════════
// MLIR-AIE TEXT GENERATION
// ═══════════════════════════════════════════════════════════

/// Generate complete MLIR-AIE module text for an element-wise kernel.
///
/// Parameters:
///   tileCount  - number of compute tiles (Elements / Grain)
///   grain      - elements per tile
///   elemType   - MLIR element type (e.g., "i32")
///   arithOp    - MLIR arith operation (e.g., "muli")
///   totalElems - total elements
let private generateAIEModule (tileCount: int) (grain: int) (elemType: string) (arithOp: string) (totalElems: int) : string =
    let sb = System.Text.StringBuilder()
    let emit (s: string) = sb.AppendLine(s) |> ignore
    let emitf fmt = Printf.kprintf emit fmt

    emit "aie.device(npu2) {"

    // Shim tiles (row 0, one per column)
    for i in 0 .. tileCount - 1 do
        emitf "  %%shim_%d = aie.tile(%d, 0)" i i

    emit ""

    // Compute tiles (row 2, one per column)
    for i in 0 .. tileCount - 1 do
        emitf "  %%tile_%d = aie.tile(%d, 2)" i i

    emit ""

    // Object FIFOs per tile: in1, in2, out (depth 2 for double buffering)
    for i in 0 .. tileCount - 1 do
        emitf "  aie.objectfifo @in1_%d(%%shim_%d, {%%tile_%d}, 2 : i32) : !aie.objectfifo<memref<%d x %s>>" i i i grain elemType
        emitf "  aie.objectfifo @in2_%d(%%shim_%d, {%%tile_%d}, 2 : i32) : !aie.objectfifo<memref<%d x %s>>" i i i grain elemType
        emitf "  aie.objectfifo @out_%d(%%tile_%d, {%%shim_%d}, 2 : i32) : !aie.objectfifo<memref<%d x %s>>" i i i grain elemType
        emit ""

    // Compute cores: one per tile
    for i in 0 .. tileCount - 1 do
        emitf "  %%core_%d = aie.core(%%tile_%d) {" i i
        emit  "    %c0 = arith.constant 0 : index"
        emit  "    %c1 = arith.constant 1 : index"
        emitf "    %%c_grain = arith.constant %d : index" grain
        emit  "    %c_inf = arith.constant 4294967295 : index"
        emit  ""
        emit  "    scf.for %iter = %c0 to %c_inf step %c1 {"
        emitf "      %%sub_a = aie.objectfifo.acquire @in1_%d(Consume, 1) : !aie.objectfifosubview<memref<%dx%s>>" i grain elemType
        emitf "      %%buf_a = aie.objectfifo.subview.access %%sub_a[0] : !aie.objectfifosubview<memref<%dx%s>> -> memref<%dx%s>" grain elemType grain elemType
        emit  ""
        emitf "      %%sub_b = aie.objectfifo.acquire @in2_%d(Consume, 1) : !aie.objectfifosubview<memref<%dx%s>>" i grain elemType
        emitf "      %%buf_b = aie.objectfifo.subview.access %%sub_b[0] : !aie.objectfifosubview<memref<%dx%s>> -> memref<%dx%s>" grain elemType grain elemType
        emit  ""
        emitf "      %%sub_c = aie.objectfifo.acquire @out_%d(Produce, 1) : !aie.objectfifosubview<memref<%dx%s>>" i grain elemType
        emitf "      %%buf_c = aie.objectfifo.subview.access %%sub_c[0] : !aie.objectfifosubview<memref<%dx%s>> -> memref<%dx%s>" grain elemType grain elemType
        emit  ""
        emitf "      scf.for %%i = %%c0 to %%c_grain step %%c1 {"
        emitf "        %%a = memref.load %%buf_a[%%i] : memref<%dx%s>" grain elemType
        emitf "        %%b = memref.load %%buf_b[%%i] : memref<%dx%s>" grain elemType
        emitf "        %%r = arith.%s %%a, %%b : %s" arithOp elemType
        emitf "        memref.store %%r, %%buf_c[%%i] : memref<%dx%s>" grain elemType
        emit  "      }"
        emit  ""
        emitf "      aie.objectfifo.release @in1_%d(Consume, 1)" i
        emitf "      aie.objectfifo.release @in2_%d(Consume, 1)" i
        emitf "      aie.objectfifo.release @out_%d(Produce, 1)" i
        emit  "    }"
        emit  "    aie.end"
        emit  "  }"
        emit  ""

    // Runtime sequence: host-side DMA configuration
    emitf "  aie.runtime_sequence(%%A: memref<%dx%s>, %%B: memref<%dx%s>, %%C: memref<%dx%s>) {" totalElems elemType totalElems elemType totalElems elemType

    // Input DMA tasks per tile
    for i in 0 .. tileCount - 1 do
        let offset = i * grain
        emitf "    %%t%d_a = aiex.dma_configure_task_for @in1_%d {" i i
        emitf "      aie.dma_bd(%%A : memref<%dx%s>, %d, %d," totalElems elemType offset grain
        emit  "        [<size = 1, stride = 0>, <size = 1, stride = 0>,"
        emitf "         <size = 1, stride = 0>, <size = %d, stride = 1>])" grain
        emit  "        {burst_length = 0 : i32}"
        emit  "      aie.end"
        emit  "    }"
        emitf "    aiex.dma_start_task(%%t%d_a)" i
        emit  ""
        emitf "    %%t%d_b = aiex.dma_configure_task_for @in2_%d {" i i
        emitf "      aie.dma_bd(%%B : memref<%dx%s>, %d, %d," totalElems elemType offset grain
        emit  "        [<size = 1, stride = 0>, <size = 1, stride = 0>,"
        emitf "         <size = 1, stride = 0>, <size = %d, stride = 1>])" grain
        emit  "        {burst_length = 0 : i32}"
        emit  "      aie.end"
        emit  "    }"
        emitf "    aiex.dma_start_task(%%t%d_b)" i
        emit  ""

    // Output DMA tasks per tile (with issue_token for synchronization)
    for i in 0 .. tileCount - 1 do
        let offset = i * grain
        emitf "    %%t%d_c = aiex.dma_configure_task_for @out_%d {" i i
        emitf "      aie.dma_bd(%%C : memref<%dx%s>, %d, %d," totalElems elemType offset grain
        emit  "        [<size = 1, stride = 0>, <size = 1, stride = 0>,"
        emitf "         <size = 1, stride = 0>, <size = %d, stride = 1>])" grain
        emit  "        {burst_length = 0 : i32}"
        emit  "      aie.end"
        emit  "    } {issue_token = true}"
        emitf "    aiex.dma_start_task(%%t%d_c)" i
        emit  ""

    // Await output completions
    for i in 0 .. tileCount - 1 do
        emitf "    aiex.dma_await_task(%%t%d_c)" i

    // Free input tasks
    for i in 0 .. tileCount - 1 do
        emitf "    aiex.dma_free_task(%%t%d_a)" i
        emitf "    aiex.dma_free_task(%%t%d_b)" i

    emit "  }"
    emit "}"

    sb.ToString()

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

let private witnessKernelModule
    (_getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput))
    (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Kind with
    | SemanticKind.Binding (name, _, _, Some DeclRoot.KernelModule) ->
        // Mark all child nodes as visited (compile-time metadata, not runtime ops)
        for childId in node.Children do
            markSubtreeVisited ctx.Graph childId ctx.GlobalVisited

        // Mark the Compute function's binding target and its full body as visited.
        // The Compute function is compile-time metadata read structurally by this witness;
        // its body (Lambda → arith Application) does not produce runtime MLIR ops on the host.
        match extractKernelFields ctx.Graph node with
        | Some fields ->
            match findField "Compute" fields with
            | Some computeNodeId ->
                match SemanticGraph.tryGetNode computeNodeId ctx.Graph with
                | Some { Kind = SemanticKind.VarRef (_, Some defId) } ->
                    markMetadataChainVisited ctx.Graph defId ctx.GlobalVisited
                | _ -> ()
            | None -> ()
        | None -> ()

        // ── 1. Extract ElementKernel<'T> RecordExpr ──
        match extractKernelFields ctx.Graph node with
        | None ->
            WitnessOutput.error $"KernelModule '{name}': Child is not a RecordExpr (ElementKernel<'T>)"
        | Some fields ->

        // ── 2. Find required fields: Compute, Shape ──
        let computeNodeOpt = findField "Compute" fields
        let shapeNodeOpt = findField "Shape" fields

        match computeNodeOpt, shapeNodeOpt with
        | None, _ ->
            WitnessOutput.error $"KernelModule '{name}': Missing 'Compute' field in ElementKernel record"
        | _, None ->
            WitnessOutput.error $"KernelModule '{name}': Missing 'Shape' field in ElementKernel record"
        | Some computeNodeId, Some shapeNodeId ->

        // ── 3. Extract Shape { Elements, Grain } ──
        match extractShape ctx.Graph shapeNodeId with
        | None ->
            WitnessOutput.error $"KernelModule '{name}': Shape must have positive Elements and Grain where Elements mod Grain = 0"
        | Some (totalElems, grain) ->

        let tileCount = totalElems / grain

        // ── 4. Resolve element type via mapType → typeToString ──
        match resolveElementType ctx computeNodeId with
        | None ->
            WitnessOutput.error $"KernelModule '{name}': Cannot resolve element type from Compute function"
        | Some elemType ->

        // ── 5. Resolve arithmetic operation via classifyAtomicOp ──
        let (arithOp, _isFloat) =
            match resolveComputeOp ctx computeNodeId with
            | Some op -> op
            | None -> ("muli", false)  // Default to multiply for hello world

        // ── 6. Generate MLIR-AIE module ──
        let aieModule = generateAIEModule tileCount grain elemType arithOp totalElems

        // Add as RawMLIR op to root scope
        let rawOp = MLIROp.RawMLIR aieModule
        let updatedRootScope = ScopeContext.addOp rawOp !ctx.RootScopeContext
        ctx.RootScopeContext := updatedRootScope

        // KernelModule binding is structural; no inline ops, no value
        { InlineOps = []; TopLevelOps = []; Result = TRVoid }

    | _ -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Create KernelModule nanopass with combinator for recursive sub-graph traversal.
/// NPU-only: should be conditionally registered in WitnessRegistry.
let createNanopass (getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) : Nanopass =
    {
        Name = "KernelModule"
        Witness = witnessKernelModule getCombinator
    }
