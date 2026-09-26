/// Validate the correspondence received from Alex before target realization.
module Core.WitnessArtifacts

open System
open System.Security.Cryptography
open System.Text
open System.IO
open System.Text.Json
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Core.Types.WitnessArtifacts

let witnessRun (WholeCheckedGraph(id, _)) = id
let graph (WholeCheckedGraph(_, graph)) = graph
let beginWholeGraphWitness graph = WholeCheckedGraph(Guid.NewGuid(), graph)

let sameScope left right =
    witnessRun left = witnessRun right && Object.ReferenceEquals(graph left, graph right)

let validateOccurrence scope (occurrence: Occurrence) =
    let nodes = (graph scope).Nodes
    let current (node: SemanticNode) =
        nodes.TryFind node.Id |> Option.exists (fun actual -> Object.ReferenceEquals(actual, node))
    let rec path (child: NodeId) (steps: (SemanticNode * NodeId list * NodeId list) list) =
        match steps with
        | [] -> child = occurrence.Anchor.Id
        | (parent, left, right) :: rest ->
            current parent && parent.Children = left @ [child] @ right && path parent.Id rest
    if not (sameScope scope occurrence.Scope) then Error "Witness occurrence belongs to a different checked graph snapshot or witness run"
    elif not (current occurrence.Anchor) || not (current occurrence.Focus) || not (path occurrence.Focus.Id occurrence.Path) then
        Error "Witness occurrence does not retain the actual current PSG focus and Huet path"
    else Ok ()

let definitionSymbol = function
    | MLIROp.FuncOp(FuncDef(name, _, _, _, _))
    | MLIROp.NoUnwindFunction(FuncDef(name, _, _, _, _))
    | MLIROp.GlobalString(name, _, _, _)
    | MLIROp.GlobalBytePool(name, _, _, _)
    | MLIROp.GlobalMemref(name, _, _)
    | MLIROp.HWOp(HWModule(name, _, _, _)) -> Some name
    | _ -> None

let isDefinition op = definitionSymbol op |> Option.isSome
let isOwnedOperation = function MLIROp.RawMLIR _ -> true | op -> isDefinition op

/// Structural operation walking is only physical inventory validation; it does
/// not traverse or reconstruct any source computation.
let rec flatten operation = seq {
    yield operation
    match operation with
    | MLIROp.FuncOp(FuncDef(_, _, _, body, _))
    | MLIROp.NoUnwindFunction(FuncDef(_, _, _, body, _))
    | MLIROp.HWOp(HWModule(_, _, _, body))
    | MLIROp.Block(_, body) | MLIROp.Region body -> yield! Seq.collect flatten body
    | MLIROp.SCFOp(SCFOp.If(_, yes, no, _)) ->
        yield! Seq.collect flatten yes
        yield! Seq.collect flatten (Option.defaultValue [] no)
    | MLIROp.SCFOp(SCFOp.While(condition, body)) -> yield! Seq.collect flatten (condition @ body)
    | MLIROp.SCFOp(SCFOp.For(_, _, _, body)) -> yield! Seq.collect flatten body
    | MLIROp.SCFOp(SCFOp.IndexSwitch(_, cases, fallback, _)) ->
        yield! Seq.collect flatten ((List.collect snd cases) @ fallback)
    | _ -> ()
}

let private hash text = SHA256.HashData(Encoding.UTF8.GetBytes(text: string)) |> Convert.ToHexString

let private imports operations =
    operations |> List.choose (function
        | MLIROp.FuncOp(FuncDecl(name, arguments, results, visibility, byval)) ->
            Some { Symbol = name; Arguments = arguments; Results = results; Visibility = visibility; Byval = byval }
        | _ -> None)

let private startup scope activation =
    match activation with
    | TargetModuleActivation -> None
    | CheckedProgramStartup ->
        Clef.Compiler.PSGSaturation.SemanticGraph.ProgramInitialization.read (graph scope)
        |> Option.map (fun plan -> plan.EntryLambda, plan.Symbol)

let validate scope operations text writable (catalog: Catalog) =
    let fail message = Error ("Witness artifact catalog: " + message)
    if not (sameScope scope catalog.Scope) then fail "different graph snapshot or witness run"
    else
        match catalog.Units with
        | [unit] when unit.Id = "whole-module" && sameScope scope unit.Scope ->
            let definitions = operations |> List.filter isOwnedOperation
            let recorded = unit.Definitions |> List.map _.Operation
            let symbols = definitions |> List.choose definitionSymbol
            let occurrenceFailure =
                unit.Definitions |> List.tryPick (fun row ->
                    match validateOccurrence scope row.Occurrence with Error message -> Some message | Ok () -> None)
            let all = operations |> List.collect (flatten >> Seq.toList)
            let signatures =
                all |> List.choose (function
                    | MLIROp.FuncOp(FuncDef(name, args, results, _, _))
                    | MLIROp.NoUnwindFunction(FuncDef(name, args, results, _, _)) -> Some(name, (List.map snd args, results))
                    | MLIROp.FuncOp(FuncDecl(name, args, results, _, _)) -> Some(name, (args, results))
                    | _ -> None)
                |> List.groupBy fst
            let conflicting = signatures |> List.exists (fun (_, rows) -> rows |> List.map snd |> List.distinct |> List.length <> 1)
            let signatureMap = signatures |> List.map (fun (name, rows) -> name, snd rows.Head) |> Map.ofList
            let globalTypes =
                definitions |> List.choose (function
                    | MLIROp.GlobalMemref(name, ty, _) -> Some(name, ty)
                    | MLIROp.GlobalString(name, _, length, _) -> Some(name, TMemRefStatic(length, TInt(IntWidth 8)))
                    | MLIROp.GlobalBytePool(name, bytes, _, _) -> Some(name, TMemRefStatic(bytes.Length, TInt(IntWidth 8)))
                    | _ -> None) |> Map.ofList
            let referenceFailure =
                all |> List.tryPick (function
                    | MLIROp.FuncOp(FuncCall(results, name, args)) ->
                        if signatureMap.TryFind name = Some(List.map (fun (v: Val) -> v.Type) args, List.map (fun (v: Val) -> v.Type) results) then None
                        else Some (sprintf "unowned or mistyped function call '%s'" name)
                    | MLIROp.FuncOp(FuncConstant(_, name, ty)) ->
                        match signatureMap.TryFind name with
                        | Some(args, results) when ty = TFunc(args, results) -> None
                        | _ -> Some(sprintf "unowned or mistyped function address '%s'" name)
                    | MLIROp.MemRefOp(MemRefOp.GetGlobal(_, name, ty)) when globalTypes.TryFind name <> Some ty ->
                        Some(sprintf "unowned or mistyped global reference '%s'" name)
                    | _ -> None)
            let duplicateRecords = recorded |> List.distinct |> List.length <> recorded.Length
            let ownedWritable = definitions |> List.choose (function MLIROp.GlobalMemref(name, _, Some entry) -> Some(name, entry) | _ -> None)
            let unownedWritable = definitions |> List.exists (function MLIROp.GlobalMemref(_, _, None) -> true | _ -> false)
            let opaque = definitions |> List.exists (function MLIROp.RawMLIR _ -> true | _ -> false)
            let expectedStartup = startup scope catalog.Activation
            let startupOwners =
                match expectedStartup with
                | None -> 0
                | Some(node, symbol) ->
                    unit.Definitions |> List.filter (fun row -> row.Occurrence.Focus.Id = node && definitionSymbol row.Operation = Some symbol) |> List.length
            if occurrenceFailure.IsSome then fail occurrenceFailure.Value
            elif symbols.Length <> (Set.ofList symbols).Count then fail "duplicate symbol definition ownership"
            elif duplicateRecords || definitions.Length <> recorded.Length || definitions |> List.exists (fun op -> not (List.contains op recorded)) then
                fail "missing, duplicate or changed emitted definition correspondence"
            elif all |> List.filter isOwnedOperation |> List.length <> definitions.Length then fail "nested definition has no module-unit ownership"
            elif all |> List.choose (function MLIROp.FuncOp(FuncDecl _) as op -> Some op | _ -> None) |> List.length <> (imports operations).Length then
                fail "nested external declaration has no module import ownership"
            elif conflicting then fail "conflicting typed import/definition signatures"
            elif unit.Imports <> imports operations || unit.Imports.Length <> (unit.Imports |> List.map _.Symbol |> Set.ofList).Count then
                fail "missing, changed or duplicate typed import inventory"
            elif referenceFailure.IsSome then fail referenceFailure.Value
            elif unownedWritable || unit.WritableStorage <> writable || ownedWritable <> writable then fail "writable storage ownership differs from admitted source inventory"
            elif unit.Startup <> expectedStartup || (expectedStartup.IsSome && startupOwners <> 1) then
                fail "planned startup requires exactly one corresponding emitted definition"
            elif unit.HasOpaqueTargetContent <> opaque then fail "opaque target-content coverage differs"
            elif unit.ContentHash <> hash text then fail "portable artifact content changed after catalog construction"
            else Ok ()
        | _ -> fail "current whole-graph witness run requires exactly one whole-module unit"

let create scope activation definitions operations text writable =
    let unit =
        { Id = "whole-module"; Scope = scope; Definitions = definitions
          Imports = imports operations; WritableStorage = writable
          Startup = startup scope activation
          HasOpaqueTargetContent = operations |> List.exists (function MLIROp.RawMLIR _ -> true | _ -> false)
          ContentHash = hash text }
    let catalog = { Scope = scope; Activation = activation; Units = [unit] }
    validate scope operations text writable catalog |> Result.map (fun () -> catalog)

/// A diagnostic manifest of validated current correspondence. It is not a
/// persistent cache key or permission to reuse another generation's objects.
let write path (catalog: Catalog) =
    let units =
        catalog.Units |> List.map (fun unit ->
            {| id = unit.Id; contentSha256 = unit.ContentHash
               opaqueTargetContent = unit.HasOpaqueTargetContent
               definitions = unit.Definitions |> List.map (fun row ->
                   {| symbol = definitionSymbol row.Operation |> Option.defaultValue "<opaque-target-unit>"
                      sourceNode = NodeId.value row.Occurrence.Focus.Id
                      traversalRoot = NodeId.value row.Occurrence.Anchor.Id
                      path = row.Occurrence.Path |> List.map (fun (parent, left, right) ->
                          {| parent = NodeId.value parent.Id; left = List.map NodeId.value left; right = List.map NodeId.value right |}) |})
               imports = unit.Imports |> List.map (fun row ->
                   {| symbol = row.Symbol; arguments = List.map string row.Arguments; results = List.map string row.Results |})
               writableSymbols = List.map fst unit.WritableStorage
               startup = unit.Startup |> Option.map (fun (node, symbol) -> {| node = NodeId.value node; symbol = symbol |}) |})
    let manifest = {| schema = 1; witnessRun = witnessRun catalog.Scope; semanticScope = "whole-checked-graph"; activation = string catalog.Activation; units = units |}
    File.WriteAllText(path, JsonSerializer.Serialize(manifest, JsonSerializerOptions(WriteIndented = true)))
