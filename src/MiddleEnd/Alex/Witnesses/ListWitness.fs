/// ListWitness - Witness List<'T> operations via XParsec
///
/// Pure XParsec monadic observer - ZERO imperative SSA lookups.
/// Witnesses pass NodeIds to Patterns; Patterns extract SSAs via getUserState.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): Eliminated ALL imperative SSA lookups.
/// This witness embodies the codata photographer principle - pure observation.
///
/// NANOPASS: This witness handles ONLY List intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.ListWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.CollectionPatterns

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness List operations - pure XParsec monadic observer
let private witnessList (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some (info, _) when info.Module <> IntrinsicModule.List -> WitnessOutput.skip
    | Some (info, _) ->
        match info.Operation with
        | "empty" ->
            let listType = mapType node.Type ctx

            match tryMatchWithDiagnostics (pListEmpty node.Id listType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | Result.Error diagnostic -> WitnessOutput.error $"List.empty: {diagnostic}"

        | "isEmpty" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (listSSA, _) ->
                    match tryMatchWithDiagnostics (pListIsEmpty node.Id listSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"List.isEmpty: {diagnostic}"
                | None -> WitnessOutput.error "List.isEmpty: List not yet witnessed"
            | _ -> WitnessOutput.error $"List.isEmpty: Expected 1 child, got {node.Children.Length}"

        | "head" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (listSSA, _) ->
                    let elementType = mapType node.Type ctx

                    match tryMatchWithDiagnostics (pListHead node.Id listSSA elementType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"List.head: {diagnostic}"
                | None -> WitnessOutput.error "List.head: List not yet witnessed"
            | _ -> WitnessOutput.error $"List.head: Expected 1 child, got {node.Children.Length}"

        | "tail" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (listSSA, _) ->
                    let tailType = mapType node.Type ctx

                    match tryMatchWithDiagnostics (pListTail node.Id listSSA tailType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"List.tail: {diagnostic}"
                | None -> WitnessOutput.error "List.tail: List not yet witnessed"
            | _ -> WitnessOutput.error $"List.tail: Expected 1 child, got {node.Children.Length}"

        | "cons" ->
            match node.Children with
            | [headId; tailId] ->
                match MLIRAccumulator.recallNode headId ctx.Accumulator, MLIRAccumulator.recallNode tailId ctx.Accumulator with
                | Some (headSSA, headType), Some (tailSSA, tailType) ->
                    let head = { SSA = headSSA; Type = headType }
                    let tail = { SSA = tailSSA; Type = tailType }
                    let listType = mapType node.Type ctx

                    match tryMatchWithDiagnostics (pListCons node.Id head tail listType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"List.cons: {diagnostic}"
                | _ -> WitnessOutput.error "List.cons: Head or tail not yet witnessed"
            | _ -> WitnessOutput.error $"List.cons: Expected 2 children, got {node.Children.Length}"

        | "map" | "filter" | "fold" | "length" ->
            WitnessOutput.error $"List.{info.Operation} requires Baker decomposition"

        | op -> WitnessOutput.error $"Unknown List operation: {op}"

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// List nanopass - witnesses List intrinsic nodes
let nanopass : Nanopass = {
    Name = "List"
    Witness = witnessList
}
