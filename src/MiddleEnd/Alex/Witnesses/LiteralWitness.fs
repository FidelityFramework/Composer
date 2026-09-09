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

        match lit with
        | NativeLiteral.String content ->
            match ctx.Graph.StaticStringPool with
            | None ->
                WitnessOutput.errorDiag (Diagnostic.error (Some node.Id) (Some "Literal") (Some "StaticStringPool")
                    "Source string storage has no settled BAREWire pool; refusing independent allocation.")
            | Some pool ->
                match pool.Entries |> List.tryFind (fun entry -> List.contains node.Id entry.NodeIds && entry.Content = content) with
                | None ->
                    WitnessOutput.errorDiag (Diagnostic.error (Some node.Id) (Some "Literal") (Some "StaticStringPool")
                        "Source string is absent from the settled BAREWire pool.")
                | Some entry ->
                    let ops, result = stringPoolView pool entry (Alex.Traversal.Values.values node.Id)
                    // All pool obligations travel on the single allocation, including those
                    // for duplicate literals; no witness ordering can drop an anchor.
                    let anchors =
                        pool.Entries
                        |> List.collect (fun entry -> entry.NodeIds)
                        |> List.collect (fun id ->
                            match ctx.Graph.Nodes.TryFind id with
                            | Some literal ->
                                match literal.Metadata.TryFind ObligationMetadata.Anchors with
                                | Some (MetadataValue.StringList names) -> names
                                | _ -> []
                            | None -> [])
                        |> List.distinct
                    let globals =
                        if Set.contains pool.Symbol ctx.Accumulator.EmittedGlobals then []
                        else
                            ctx.Accumulator.EmittedGlobals <- Set.add pool.Symbol ctx.Accumulator.EmittedGlobals
                            [Alex.Dialects.Core.Types.MLIROp.GlobalBytePool (pool.Symbol, pool.Bytes, pool.Alignment, anchors)]
                    { InlineOps = ops; TopLevelOps = globals; Result = result }

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
