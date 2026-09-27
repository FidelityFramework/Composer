/// Witness source-published boundary calls through Patterns and Elements.
/// Host/library selection, declarations, ABI and adaptation are source facts.
module Alex.Witnesses.PlatformWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.PlatformPatterns

module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission

let private witnessPlatform (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match Publication.tryBoundary ctx.Graph with
    | Result.Error reason -> WitnessOutput.error reason
    | Result.Ok boundary ->
        let pattern =
            if boundary.Calls.ContainsKey node.Id then Some pBoundaryCall
            elif boundary.DeclarationLeaves.Contains node.Id then Some pBoundaryDeclaration
            else None
        match pattern with
        | Some pattern ->
            match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Result.Ok ((operations, result), _) ->
                { InlineOps = operations; TopLevelOps = []; Result = result }
            | Result.Error reason -> WitnessOutput.error reason
        | None when ctx.Coeffects.Platform.Bindings.Bindings.ContainsKey node.Id ->
            WitnessOutput.error $"Boundary call {NodeId.value node.Id} lacks its source-published declaration and ABI contract."
        | None -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "Platform"; Witness = witnessPlatform }
