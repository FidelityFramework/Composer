/// MLIRTransfer - Thin fold from SemanticGraph to MLIR via XParsec + Witnesses
///
/// ARCHITECTURE (January 2026 Greenfield):
/// - XParsec: Pattern match PSG node structure
/// - Witnesses: Extract data, delegate to Patterns
/// - Patterns: Compose Elements into MLIR ops
/// - Elements: Atomic MLIR operation emission
///
/// This file does ONE thing: fold PSG structure via XParsec, dispatch to witnesses.
/// NO transform logic. NO inline MLIR construction. ONLY orchestration.
module Alex.Traversal.MLIRTransfer

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Traversal.PSGZipper
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
open Alex.CodeGeneration.TypeMapping
open PSGElaboration.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS OUTPUT (Codata Return Type)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness output - witnesses RETURN this, fold accumulates
type WitnessOutput = {
    InlineOps: MLIROp list      // Operations to emit inline
    TopLevelOps: MLIROp list    // Top-level declarations (functions, globals)
    Result: TransferResult      // Resulting value/block/error
}

module WitnessOutput =
    let inline value (ops: MLIROp list) (val_: Val) : WitnessOutput =
        { InlineOps = ops; TopLevelOps = []; Result = TRValue val_ }

    let inline valueWithTopLevel (inlineOps: MLIROp list) (topOps: MLIROp list) (val_: Val) : WitnessOutput =
        { InlineOps = inlineOps; TopLevelOps = topOps; Result = TRValue val_ }

    let inline block (ops: MLIROp list) (blockId: BlockId) : WitnessOutput =
        { InlineOps = ops; TopLevelOps = []; Result = TRBlock blockId }

    let inline error (msg: string) : WitnessOutput =
        { InlineOps = []; TopLevelOps = []; Result = TRError msg }

    let inline unit_ (ops: MLIROp list) : WitnessOutput =
        { InlineOps = ops; TopLevelOps = []; Result = TRUnit }

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS CONTEXT (Immutable Context for Witnesses)
// ═══════════════════════════════════════════════════════════════════════════

/// Context passed to witnesses - immutable, contains coeffects and platform info
type WitnessContext = {
    Graph: SemanticGraph
    Coeffects: TransferCoeffects
    Platform: PlatformResolutionResult
    Zipper: PSGZipper
}

// ═══════════════════════════════════════════════════════════════════════════
// MLIR ACCUMULATOR (Mutable Fold State)
// ═══════════════════════════════════════════════════════════════════════════

/// Mutable accumulator for MLIR generation
/// The fold is pure (witnesses return codata), but we accumulate into this structure
type MLIRAccumulator = {
    mutable InlineOps: MLIROp list
    mutable TopLevelOps: MLIROp list
    mutable NodeBindings: Map<int, Val>  // NodeId.value -> Val
}

module MLIRAccumulator =
    let create () : MLIRAccumulator = {
        InlineOps = []
        TopLevelOps = []
        NodeBindings = Map.empty
    }

    let accumulate (output: WitnessOutput) (acc: MLIRAccumulator) : unit =
        acc.InlineOps <- acc.InlineOps @ output.InlineOps
        acc.TopLevelOps <- acc.TopLevelOps @ output.TopLevelOps

    let bindNode (nodeId: int) (val_: Val) (acc: MLIRAccumulator) : unit =
        acc.NodeBindings <- Map.add nodeId val_ acc.NodeBindings

    let tryGetBinding (nodeId: int) (acc: MLIRAccumulator) : Val option =
        Map.tryFind nodeId acc.NodeBindings

// ═══════════════════════════════════════════════════════════════════════════
// THE FOLD (Core Transfer Logic)
// ═══════════════════════════════════════════════════════════════════════════

/// Visit a single PSG node - dispatch to appropriate witness via XParsec
let rec visitNode (ctx: WitnessContext) (z: PSGZipper) (nodeId: NodeId) (acc: MLIRAccumulator) : WitnessOutput =

    // Check if already processed
    match MLIRAccumulator.tryGetBinding (NodeId.value nodeId) acc with
    | Some existingVal ->
        // Already processed - return cached value
        WitnessOutput.value [] existingVal
    | None ->
        // Get node from graph
        match SemanticGraph.tryGetNode nodeId ctx.Graph with
        | None ->
            WitnessOutput.error (sprintf "Node %d not found in graph" (NodeId.value nodeId))
        | Some node ->
            // Dispatch based on SemanticKind
            match node.Kind with

            // ═══════════════════════════════════════════════════════════
            // LITERALS
            // ═══════════════════════════════════════════════════════════

            | SemanticKind.Const (ConstKind.Bool b) ->
                WitnessOutput.error "Bool literal not implemented - needs witness"

            | SemanticKind.Const (ConstKind.Int32 i) ->
                WitnessOutput.error "Int32 literal not implemented - needs witness"

            | SemanticKind.Const (ConstKind.Int64 i) ->
                WitnessOutput.error "Int64 literal not implemented - needs witness"

            | SemanticKind.Const (ConstKind.String s) ->
                WitnessOutput.error "String literal not implemented - needs witness"

            | SemanticKind.Const kind ->
                WitnessOutput.error (sprintf "Const kind %A not implemented" kind)

            // ═══════════════════════════════════════════════════════════
            // BINDINGS
            // ═══════════════════════════════════════════════════════════

            | SemanticKind.Let (bindingId, bodyId) ->
                // Process binding, then body
                let bindingOutput = visitNode ctx z bindingId acc
                MLIRAccumulator.accumulate bindingOutput acc
                visitNode ctx z bodyId acc

            | SemanticKind.LetRec (bindings, bodyId) ->
                // Process all bindings, then body
                for bindingId in bindings do
                    let bindingOutput = visitNode ctx z bindingId acc
                    MLIRAccumulator.accumulate bindingOutput acc
                visitNode ctx z bodyId acc

            | SemanticKind.Lambda _ ->
                WitnessOutput.error "Lambda not implemented - needs witness"

            // ═══════════════════════════════════════════════════════════
            // APPLICATION
            // ═══════════════════════════════════════════════════════════

            | SemanticKind.Application (funcNodeId, argNodeIds) ->
                WitnessOutput.error "Application not implemented - needs witness dispatcher"

            // ═══════════════════════════════════════════════════════════
            // CONTROL FLOW
            // ═══════════════════════════════════════════════════════════

            | SemanticKind.IfThenElse (condId, thenId, elseId) ->
                WitnessOutput.error "IfThenElse not implemented - needs witness"

            | SemanticKind.WhileLoop _ ->
                WitnessOutput.error "WhileLoop not implemented - needs witness"

            | SemanticKind.Match _ ->
                WitnessOutput.error "Match not implemented - needs witness"

            // ═══════════════════════════════════════════════════════════
            // LAZY (PRD-14)
            // ═══════════════════════════════════════════════════════════

            | SemanticKind.LazyExpr _ ->
                // Use XParsec to match LazyExpr structure
                match tryMatch pLazyExpr ctx.Graph node z ctx.Platform with
                | Some ((bodyId, captures), _) ->
                    WitnessOutput.error "LazyExpr matched via XParsec - needs LazyWitness.witnessLazyCreate"
                | None ->
                    WitnessOutput.error "LazyExpr XParsec pattern match failed"

            | SemanticKind.LazyForce _ ->
                // Use XParsec to match LazyForce structure
                match tryMatch pLazyForce ctx.Graph node z ctx.Platform with
                | Some (lazyNodeId, _) ->
                    WitnessOutput.error "LazyForce matched via XParsec - needs LazyWitness.witnessLazyForce"
                | None ->
                    WitnessOutput.error "LazyForce XParsec pattern match failed"

            // ═══════════════════════════════════════════════════════════
            // SEQUENCES (PRD-15)
            // ═══════════════════════════════════════════════════════════

            | SemanticKind.SeqExpr _ ->
                WitnessOutput.error "SeqExpr not implemented - needs witness"

            | SemanticKind.Yield _ ->
                WitnessOutput.error "Yield not implemented - needs witness"

            | SemanticKind.YieldBang _ ->
                WitnessOutput.error "YieldBang not implemented - needs witness"

            // ═══════════════════════════════════════════════════════════
            // REFERENCES
            // ═══════════════════════════════════════════════════════════

            | SemanticKind.VarRef _ ->
                WitnessOutput.error "VarRef not implemented - needs witness"

            | SemanticKind.FuncRef _ ->
                WitnessOutput.error "FuncRef not implemented - needs witness"

            // ═══════════════════════════════════════════════════════════
            // TUPLES
            // ═══════════════════════════════════════════════════════════

            | SemanticKind.Tuple elemIds ->
                WitnessOutput.error "Tuple not implemented - needs witness"

            | SemanticKind.TupleGet (tupleId, index) ->
                WitnessOutput.error "TupleGet not implemented - needs witness"

            // ═══════════════════════════════════════════════════════════
            // RECORDS
            // ═══════════════════════════════════════════════════════════

            | SemanticKind.RecordExpr _ ->
                WitnessOutput.error "RecordExpr not implemented - needs witness"

            | SemanticKind.FieldGet _ ->
                WitnessOutput.error "FieldGet not implemented - needs witness"

            | SemanticKind.FieldSet _ ->
                WitnessOutput.error "FieldSet not implemented - needs witness"

            // ═══════════════════════════════════════════════════════════
            // UNIONS
            // ═══════════════════════════════════════════════════════════

            | SemanticKind.UnionCase _ ->
                WitnessOutput.error "UnionCase not implemented - needs witness"

            // ═══════════════════════════════════════════════════════════
            // INTRINSICS
            // ═══════════════════════════════════════════════════════════

            | SemanticKind.Intrinsic info ->
                WitnessOutput.error (sprintf "Intrinsic %A not implemented - needs witness" info)

            // ═══════════════════════════════════════════════════════════
            // UNHANDLED
            // ═══════════════════════════════════════════════════════════

            | kind ->
                WitnessOutput.error (sprintf "Unhandled SemanticKind: %A" kind)

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// Transfer PSG to MLIR starting from entry point
let transfer
    (graph: SemanticGraph)
    (entryNodeId: NodeId)
    (coeffects: TransferCoeffects)
    (platform: PlatformResolutionResult)
    : Result<MLIROp list * MLIROp list, string> =

    // Create zipper at entry point
    match SemanticGraph.tryGetNode entryNodeId graph with
    | None ->
        Error (sprintf "Entry node %d not found" (NodeId.value entryNodeId))
    | Some entryNode ->
        let zipper = PSGZipper.create graph entryNode
        let ctx = {
            Graph = graph
            Coeffects = coeffects
            Platform = platform
            Zipper = zipper
        }
        let acc = MLIRAccumulator.create ()

        // Fold over entry point
        let output = visitNode ctx zipper entryNodeId acc
        MLIRAccumulator.accumulate output acc

        // Check for errors
        match output.Result with
        | TRError msg ->
            Error msg
        | _ ->
            Ok (acc.InlineOps, acc.TopLevelOps)
