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

let tryBinding (graph: SemanticGraph) (id: NodeId) : string option =
    Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryCallable graph
    |> Result.toOption |> Option.bind (fun projection -> projection.Symbols.TryFind id) |> Option.map render

let lambda (graph: SemanticGraph) (node: SemanticNode) _hasClosureLayout : string =
    match tryBinding graph node.Id with
    | Some symbol -> symbol
    | None -> invalidOp "Callable code has no source-published declaration identity."
