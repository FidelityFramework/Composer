/// OptionWitness - Witness Option<'T> operations via XParsec
///
/// Pure XParsec monadic observer - ZERO imperative SSA lookups.
/// Witnesses pass NodeIds to Patterns; Patterns extract SSAs via getUserState.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): Eliminated ALL imperative SSA lookups.
/// This witness embodies the codata photographer principle - pure observation.
///
/// NANOPASS: This witness handles ONLY Option intrinsic nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.OptionWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.CollectionPatterns
open Alex.CodeGeneration.TypeMapping  // mlirTypeSize

// ═══════════════════════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Option operations - pure XParsec monadic observer
let private witnessOption (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pIntrinsic ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | None -> WitnessOutput.skip
    | Some (info, _) when info.Module <> IntrinsicModule.Option -> WitnessOutput.skip
    | Some (info, _) ->
        match info.Operation with
        | "Some" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (valSSA, valType) ->
                    let value = { SSA = valSSA; Type = valType }
                    let totalBytes = 1 + mlirTypeSize valType
                    let optionTy = TMemRefStatic(totalBytes, TInt (IntWidth 8))

                    match tryMatchWithDiagnostics (pOptionSome node.Id value optionTy) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"Option.Some: {diagnostic}"
                | None -> WitnessOutput.error "Option.Some: Value not yet witnessed"
            | _ -> WitnessOutput.error $"Option.Some: Expected 1 child, got {node.Children.Length}"

        | "None" ->
            let optionType = mapType node.Type ctx

            match tryMatchWithDiagnostics (pOptionNone node.Id optionType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | Result.Error diagnostic -> WitnessOutput.error $"Option.None: {diagnostic}"

        | "isSome" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (optSSA, _) ->
                    match tryMatchWithDiagnostics (pOptionIsSome node.Id optSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"Option.isSome: {diagnostic}"
                | None -> WitnessOutput.error "Option.isSome: Option not yet witnessed"
            | _ -> WitnessOutput.error $"Option.isSome: Expected 1 child, got {node.Children.Length}"

        | "isNone" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (optSSA, _) ->
                    match tryMatchWithDiagnostics (pOptionIsNone node.Id optSSA) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"Option.isNone: {diagnostic}"
                | None -> WitnessOutput.error "Option.isNone: Option not yet witnessed"
            | _ -> WitnessOutput.error $"Option.isNone: Expected 1 child, got {node.Children.Length}"

        | "get" ->
            match node.Children with
            | [childId] ->
                match MLIRAccumulator.recallNode childId ctx.Accumulator with
                | Some (optSSA, _) ->
                    let valueType = TIndex  // Fallback type

                    match tryMatchWithDiagnostics (pOptionGet node.Id optSSA valueType) ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                    | Result.Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | Result.Error diagnostic -> WitnessOutput.error $"Option.get: {diagnostic}"
                | None -> WitnessOutput.error "Option.get: Option not yet witnessed"
            | _ -> WitnessOutput.error $"Option.get: Expected 1 child, got {node.Children.Length}"

        | "map" | "bind" | "defaultValue" | "defaultWith" ->
            WitnessOutput.error $"Option.{info.Operation} requires Baker decomposition"

        | op -> WitnessOutput.error $"Unknown Option operation: {op}"

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════════════════════

/// Option nanopass - witnesses Option intrinsic nodes
let nanopass : Nanopass = {
    Name = "Option"
    Witness = witnessOption
}
