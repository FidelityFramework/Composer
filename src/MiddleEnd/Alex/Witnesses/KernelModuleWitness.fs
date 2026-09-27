/// Passive kernel declaration witnessing through the shared spatial Pattern.
module Alex.Witnesses.KernelModuleWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.SpatialPatterns

let private witnessKernel (_getCombinator: unit -> (WitnessContext -> SemanticNode -> WitnessOutput))
                          (ctx: WitnessContext) (node: SemanticNode) =
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.trySpatial ctx.Graph with
    | Result.Error reason -> WitnessOutput.error reason
    | Result.Ok projection when projection.Kernels.ContainsKey node.Id ->
        match tryMatchWithDiagnostics pPublishedSpatialModule ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((operations,result),_) -> { InlineOps=[]; TopLevelOps=operations; Result=result }
        | Result.Error reason -> WitnessOutput.error reason
    | Result.Ok _ ->
        match node.Kind with
        | SemanticKind.Binding(_,_,_,Some DeclRoot.KernelModule) -> WitnessOutput.error "Kernel declaration lacks its complete source spatial publication."
        | _ -> WitnessOutput.skip

let createNanopass getCombinator : Nanopass = { Name="KernelModule"; Witness=witnessKernel getCombinator }
