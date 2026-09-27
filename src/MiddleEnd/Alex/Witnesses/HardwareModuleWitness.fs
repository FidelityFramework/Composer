/// Observe one source-settled hardware declaration. Ordinary shared traversal
/// witnesses its published code roots and excludes its proven metadata census.
module Alex.Witnesses.HardwareModuleWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.SpatialPatterns

let private witness (ctx: WitnessContext) (node: SemanticNode) =
    match node.Kind with
    | SemanticKind.Binding(_, _, _, Some DeclRoot.HardwareModule) ->
        match tryMatchWithDiagnostics pPublishedSpatialModule ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((declarations, result), _) -> { InlineOps = []; TopLevelOps = declarations; Result = result }
        | Result.Error reason -> WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "HardwareModule") (Some "source publication") reason
    | _ -> WitnessOutput.skip

let createNanopass (_getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput)) : Nanopass =
    { Name = "HardwareModule"; Witness = witness }
