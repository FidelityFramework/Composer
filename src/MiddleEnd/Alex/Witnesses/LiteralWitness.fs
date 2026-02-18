/// Literal Witness - Witness literal values to MLIR via XParsec
///
/// Uses XParsec combinators from PSGCombinators to match PSG structure,
/// then delegates to Patterns for MLIR elision.
///
/// NANOPASS: This witness handles ONLY Literal nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.LiteralWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.LiteralPatterns
open XParsec
open XParsec.Parsers
open XParsec.Combinators

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness Literal nodes - category-selective (handles only Literal nodes)
let private witnessLiteralNode (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pLiteral ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some (lit, _) ->
        let arch = ctx.Coeffects.Platform.TargetArch

        // String literals: 2 SSAs — memref.get_global (static) + memref.cast (static → dynamic)
        match lit with
        | NativeLiteral.String content ->
            // Extract SSAs monadically
            let stringPattern =
                parser {
                    // Extract result SSAs for string literal (monadic)
                    let! ssas = getNodeSSAs node.Id

                    if ssas.Length < 2 then
                        return! fail (Message $"String literal: Expected 2 SSAs, got {ssas.Length}")
                    else
                        return! pBuildStringLiteral content ssas arch
                }

            // Use trace-enabled variant to capture full execution path
            match tryMatchWithTrace stringPattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok (((inlineOps, globalName, strContent, byteLength), result), _, _trace) ->
                // Success - emit GlobalString via coordination (dependent transparency)
                let topLevelOps =
                    match MLIRAccumulator.tryEmitGlobal globalName strContent byteLength ctx.Accumulator with
                    | Some globalOp -> [globalOp]
                    | None -> []  // Already emitted by another witness
                { InlineOps = inlineOps; TopLevelOps = topLevelOps; Result = result }
            | Result.Error (err, trace) ->
                // Failure - serialize trace for debugging
                // TODO: Serialize trace to intermediates/07_literal_witness_nodeXXX_trace.json
                let traceMsg = trace |> List.map ExecutionTrace.format |> String.concat "\n"
                let diag = Diagnostic.error (Some node.Id) (Some "Literal") (Some "pBuildStringLiteral")
                                (sprintf "String literal pattern emission failed:\nXParsec Error: %A\nExecution Trace:\n%s" err traceMsg)
                WitnessOutput.errorDiag diag

        | _ ->
            // Other literals (int, bool, float, char, etc.) use single SSA
            // Extract SSA monadically
            let literalPattern =
                parser {
                    let! ssa = getNodeSSA node.Id
                    return! pBuildLiteral lit ssa arch
                }

            match tryMatch literalPattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | None ->
                let diag = Diagnostic.error (Some node.Id) (Some "Literal") (Some "pBuildLiteral") "Literal pattern emission failed"
                WitnessOutput.errorDiag diag

    | None -> WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Literal nanopass - witnesses Literal nodes (int, bool, char, float, etc.)
let nanopass : Nanopass = {
    Name = "Literal"
    Witness = witnessLiteralNode
}
