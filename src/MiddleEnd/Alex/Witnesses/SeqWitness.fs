/// SeqWitness - Witness Seq<'T> operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Seq-related nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.SeqWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ClosurePatterns
open Alex.Patterns.ControlFlowPatterns
open XParsec
open XParsec.Parsers
open XParsec.Combinators

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Seq operations - category-selective (handles only Seq nodes)
let private witnessSeq (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pSeqExpr ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((bodyId, captures), _) ->
        // Extract SSAs monadically using XParsec state threading
        let seqPattern =
            parser {
                let! state = getUserState
                let arch = state.Coeffects.Platform.TargetArch

                // Extract result SSAs for SeqExpr (monadic)
                let! ssas = getNodeSSAs node.Id

                // Extract capture SSAs (monadic)
                let captureNodeIds =
                    captures
                    |> List.choose (fun cap ->
                        cap.SourceNodeId
                        |> Option.map (fun id -> (id, cap.Type)))

                let! captureVals =
                    let rec extractCaptures caps =
                        parser {
                            match caps with
                            | [] -> return []
                            | (id, capType) :: rest ->
                                let! ssa = getNodeSSA id
                                let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch capType
                                let! restVals = extractCaptures rest
                                return { SSA = ssa; Type = mlirType } :: restVals
                        }
                    extractCaptures captureNodeIds

                // Extract mutable binding SSAs from body (monadic)
                let rec extractMutableBindings nodeId =
                    parser {
                        match SemanticGraph.tryGetNode nodeId state.Graph with
                        | None -> return []
                        | Some n ->
                            let! thisVal =
                                match n.Kind with
                                | SemanticKind.Binding (_, true, _, _) ->
                                    parser {
                                        let! ssa = getNodeSSA nodeId
                                        let mlirType = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch n.Type
                                        return [{ SSA = ssa; Type = mlirType }]
                                    }
                                | _ -> preturn []

                            // Recursively extract from children
                            let rec extractChildren children =
                                parser {
                                    match children with
                                    | [] -> return []
                                    | child :: rest ->
                                        let! childVals = extractMutableBindings child
                                        let! restVals = extractChildren rest
                                        return childVals @ restVals
                                }
                            let! childVals = extractChildren n.Children
                            return thisVal @ childVals
                    }

                let! internalState = extractMutableBindings bodyId

                // Get code pointer from accumulator
                match MLIRAccumulator.recallNode bodyId state.Accumulator with
                | None -> return! fail (Message "SeqExpr: Body not yet witnessed")
                | Some (codePtr, codePtrTy) ->
                    // Get Seq<T> type from node
                    // With TMemRefStatic, we can't extract current type from structure - use fallback
                    let currentTy = TIndex  // fallback (may need type tracking refactor)
                    return! pBuildSeqStruct currentTy codePtrTy codePtr captureVals internalState ssas arch
            }

        match tryMatch seqPattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | None -> WitnessOutput.error "SeqExpr pattern emission failed"

    | None ->
        match tryMatch pForEach ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((_, collectionId, _), _) ->
            match MLIRAccumulator.recallNode collectionId ctx.Accumulator with
            | None -> WitnessOutput.error "ForEach: Collection not yet witnessed"
            | Some (collectionSSA, _) ->
                let arch = ctx.Coeffects.Platform.TargetArch
                let bodyOps = []
                match tryMatch (pBuildForEachLoop collectionSSA bodyOps arch) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                | None -> WitnessOutput.error "ForEach pattern emission failed"

        | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Seq nanopass - witnesses SeqExpr and ForEach nodes
let nanopass : Nanopass = {
    Name = "Seq"
    Witness = witnessSeq
}
