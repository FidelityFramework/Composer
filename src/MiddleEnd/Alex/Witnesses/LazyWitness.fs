/// LazyWitness - Witness Lazy<'T> operations via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Lazy-related nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.LazyWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ClosurePatterns
open XParsec
open XParsec.Parsers
open XParsec.Combinators

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Lazy operations - category-selective (handles only Lazy nodes)
let private witnessLazy (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Try LazyExpr pattern
    match tryMatch pLazyExpr ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((bodyId, captureInfos), _) ->
        // Extract SSAs monadically
        let lazyExprPattern =
            parser {
                let! state = getUserState
                let arch = state.Coeffects.Platform.TargetArch

                // Extract result SSAs for LazyExpr (monadic)
                let! ssas = getNodeSSAs node.Id

                // Captures come from accumulator (already witnessed nodes)
                let captures =
                    captureInfos
                    |> List.choose (fun capture ->
                        capture.SourceNodeId
                        |> Option.bind (fun id -> MLIRAccumulator.recallNode id state.Accumulator)
                        |> Option.map (fun (ssa, ty) -> { SSA = ssa; Type = ty }))

                match MLIRAccumulator.recallNode bodyId state.Accumulator with
                | None -> return! fail (Message "LazyExpr: Body not yet witnessed")
                | Some (codePtr, codePtrTy) ->
                    // Get Lazy<T> type from node
                    // With TMemRefStatic, we can't extract value type from structure - use fallback
                    let valueTy = TIndex  // fallback (may need type tracking refactor)
                    return! pBuildLazyStruct valueTy codePtrTy codePtr captures ssas arch
            }

        match tryMatch lazyExprPattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | None -> WitnessOutput.error "LazyExpr pattern emission failed"

    | None ->
        // Try LazyForce pattern
        match tryMatch pLazyForce ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some (lazyNodeId, _) ->
            // Extract SSAs monadically
            let lazyForcePattern =
                parser {
                    let! state = getUserState
                    let arch = state.Coeffects.Platform.TargetArch

                    // Extract result SSAs for LazyForce (monadic)
                    let! ssas = getNodeSSAs node.Id

                    if ssas.Length < 4 then
                        return! fail (Message $"LazyForce: Expected 4 SSAs, got {ssas.Length}")
                    else
                        match MLIRAccumulator.recallNode lazyNodeId state.Accumulator with
                        | None -> return! fail (Message "LazyForce: Lazy value not yet witnessed")
                        | Some (lazySSA, lazyTy) ->
                            // LazyForce SSAs: [0]=code_ptr, [1]=const1, [2]=alloca, [3]=result
                            let resultSSA = ssas.[3]
                            let intermediateSsas = [ssas.[0]; ssas.[1]; ssas.[2]]
                            let resultTy = Alex.CodeGeneration.TypeMapping.mapNativeTypeForArch arch node.Type
                            return! pBuildLazyForce lazySSA lazyTy resultSSA resultTy intermediateSsas arch
                }

            match tryMatch lazyForcePattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | None -> WitnessOutput.error "LazyForce pattern emission failed"

        | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Lazy nanopass - witnesses LazyExpr and LazyForce nodes
let nanopass : Nanopass = {
    Name = "Lazy"
    Witness = witnessLazy
}
