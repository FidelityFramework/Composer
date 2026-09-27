/// Witness the memory operation published for the current Huet occurrence.
/// Baker owns its operands, representation, access and storage authority.
module Alex.Witnesses.MemoryWitness

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemoryPatterns

module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission

let private witnessMemory (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match Publication.tryMemory ctx.Graph with
    | Result.Error reason -> WitnessOutput.error reason
    | Result.Ok memory when memory.Operations.ContainsKey node.Id ->
        match tryMatchWithDiagnostics pPublishedMemoryOperation ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Result.Ok ((operations, declarations, result), _) ->
            { InlineOps = operations; TopLevelOps = declarations; Result = result }
        | Result.Error reason -> WitnessOutput.error reason
    | Result.Ok memory when memory.Required.Contains node.Id ->
        WitnessOutput.error $"Memory operation {NodeId.value node.Id} lacks its source-published access and representation contract."
    | Result.Ok _ -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "Memory"; Witness = witnessMemory }
