/// Project resolved PSG callable identity into an emission symbol.
/// Local source names are scoped; module and external names retain their ABI spelling.
module Alex.CodeGeneration.CallableSymbols

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core

let tryBinding (graph: SemanticGraph) (id: NodeId) : string option =
    match SemanticGraph.tryGetNode id graph with
    | Some { Kind = SemanticKind.Binding (name, _, _, _); Parent = parent } ->
        match parent |> Option.bind (fun id -> SemanticGraph.tryGetNode id graph) with
        | Some { Kind = SemanticKind.ModuleDef (moduleName, _) } -> Some (moduleName + "." + name)
        | _ when parent.IsSome -> Some (sprintf "__clef_local_%d_%s" (NodeId.value id) name)
        | _ -> Some name
    | _ -> None

let lambda (graph: SemanticGraph) (node: SemanticNode) hasClosureLayout : string =
    let anonymous () = sprintf "lambda_%d" (NodeId.value node.Id)
    if hasClosureLayout
       && ([ClosureMetadata.LambdaExpression; ClosureMetadata.RequiresClosurePair]
           |> List.exists (fun key -> node.Metadata.TryFind key = Some (MetadataValue.Bool true))) then
        anonymous ()
    else
        node.Parent |> Option.bind (tryBinding graph) |> Option.defaultWith anonymous
