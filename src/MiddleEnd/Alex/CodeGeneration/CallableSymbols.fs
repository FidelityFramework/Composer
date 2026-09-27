/// Project resolved PSG callable identity into an emission symbol.
/// Local source names are scoped; module and external names retain their ABI spelling.
module Alex.CodeGeneration.CallableSymbols

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

let private render = function
    | CallableSymbolName.ModuleBinding(moduleName, name) -> moduleName + "." + name
    | CallableSymbolName.LocalBinding(id, name) -> sprintf "__clef_local_%d_%s" (NodeId.value id) name
    | CallableSymbolName.RootBinding name -> name
    | CallableSymbolName.Anonymous id -> sprintf "lambda_%d" (NodeId.value id)

/// The published symbol of a callable binding; None when the projection publishes none for it.
/// A graph with no callable projection is a stop naming the projection's refusal, never None.
let tryBinding (graph: SemanticGraph) (id: NodeId) : string option =
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryCallable graph with
    | Result.Ok projection -> projection.Symbols.TryFind id |> Option.map render
    | Result.Error reason ->
        invalidOp (sprintf "PSG settlement (WitnessEmission) did not settle the callable projection for node %d: %s" (NodeId.value id) reason)

let lambda (graph: SemanticGraph) (node: SemanticNode) _hasClosureLayout : string =
    match tryBinding graph node.Id with
    | Some symbol -> symbol
    | None ->
        invalidOp (sprintf "PSG settlement (WitnessEmission) did not settle a declaration symbol for the callable code at node %d" (NodeId.value node.Id))
