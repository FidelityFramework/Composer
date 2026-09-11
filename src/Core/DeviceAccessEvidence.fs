/// Serialize CCS's settled access evidence without reinterpreting declarations.
module Core.DeviceAccessEvidence

open System.IO
open System.Text.Json
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

let write path (graph: SemanticGraph) =
    let nodeId (NodeId n) = n
    let source id =
        let range = graph.Nodes.[id].Range
        {| node = nodeId id; file = range.File; line = range.Start.Line; column = range.Start.Column |}
    let sites =
        graph.Codata.Value.Mmio |> Map.toArray |> Array.map (fun (id, site) ->
            let binding =
                match site.Binding with
                | None -> null
                | Some binding ->
                    box {| plan = binding.Plan; grant = binding.Grant; register = binding.Register
                           region = binding.Region; mapping = binding.Mapping; addressSpace = binding.AddressSpace
                           declarations = binding.DeclarationNodes |> List.map source |> List.toArray
                           externalPremises = List.toArray binding.Premises
                           predicates = binding.Predicates |> List.map (fun p ->
                               {| name = p.Name; status = string p.Status; source = p.Source; message = p.Message
                                  declaration = source p.Declaration; expression = source p.Expression
                                  dependencies = p.Dependencies |> List.map source |> List.toArray |}) |> List.toArray |}
            {| site = source id; operation = site.Operation; address = string site.Address
               transactionBits = site.Bits; binding = binding |})
    let evidence = {| schemaVersion = 1; authority = "CCS settled device-access evidence"
                      scope = "Static declaration checks; external mapping/hardware premises are not proved. No runtime or solver certificate."
                      sites = sites |}
    File.WriteAllText(path, JsonSerializer.Serialize(evidence, JsonSerializerOptions(WriteIndented = true)))
