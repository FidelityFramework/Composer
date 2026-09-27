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
    | MLIROp.SpatialModule(SpatialModuleWitness.Hardware plan) -> Some plan.Name
    | MLIROp.SpatialModule(SpatialModuleWitness.Kernel plan) -> Some plan.Name
    | MLIROp.FuncOp(FuncDef(name, _, _, _, _))
    | MLIROp.NoUnwindFunction(FuncDef(name, _, _, _, _))
    | MLIROp.GlobalString(name, _, _, _)
    | MLIROp.GlobalBytePool(name, _, _, _)
    | MLIROp.GlobalMemref(name, _, _)
    | MLIROp.GlobalArray(name, _, _)
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

let private expectedOutcomes obligations =
    obligations |> List.map (fun (ob: ObligationInfo) -> { Anchor = ob.Id; Source = ob.Source; Verdict = "unsat" })

/// Exact current-source and physical-transcription correspondence. This checks
/// evidence identity; it does not decide an obligation or invoke a solver.
let validateProof scope (proof: ProofEnvelope) =
    let fail message = Error ("Witness proof envelope: " + message)
    let current = Clef.Compiler.Nanopass.ObligationDischarge.ofGraph (graph scope)
    let sourceQuery = Clef.Compiler.Nanopass.ObligationDischarge.smtLib current
    let sourceFailure =
        proof.Source |> Option.bind (fun receipt ->
            if not (Object.ReferenceEquals(receipt.Graph, graph scope)) || receipt.Obligations <> current then
                Some "PSG discharge belongs to a different graph snapshot or obligation inventory"
            elif receipt.Query <> sourceQuery || receipt.Evidence.InputSha256 <> hash sourceQuery
                 || receipt.Evidence.QuerySha256 <> hash sourceQuery
                 || receipt.Evidence.Stage <> "psg" || receipt.Evidence.Invocation = Guid.Empty
                 || receipt.Evidence.Outcomes <> expectedOutcomes current then
                Some "PSG discharge evidence differs from the exact required queries and outcomes"
            else None)
    let mlirFailure =
        proof.Mlir |> Option.bind (fun receipt ->
            if not (sameScope scope receipt.Scope) || receipt.Operations <> proof.Operations || receipt.Text <> proof.Text then
                Some "MLIR discharge belongs to different witnessed operations or scope"
            else
                match proof.Source with
                | Some source when Object.ReferenceEquals(source, receipt.Source)
                                   && receipt.Evidence.Invocation = source.Evidence.Invocation
                                   && receipt.Evidence.Stage = "mlir"
                                   && receipt.Evidence.InputSha256 = hash proof.Text
                                   && receipt.Evidence.QuerySha256 = hash receipt.Query
                                   && receipt.Evidence.Outcomes = expectedOutcomes current -> None
                | _ -> Some "MLIR discharge is not paired with the exact PSG discharge" )
    if not (sameScope scope proof.Scope) then fail "different graph snapshot or witness run"
    elif proof.Obligations <> current then fail "missing, changed or reordered source obligation inventory"
    elif (current |> List.map _.Id |> Set.ofList).Count <> current.Length then fail "duplicate source obligation anchors"
    elif proof.Operations <> Alex.Traversal.SMTTransfer.operations current then fail "typed SMT differs from the source obligation transcription"
    elif proof.Text <> Alex.Traversal.SMTTransfer.transfer current then fail "SMT text differs from the exact typed obligation operations"
    elif sourceFailure.IsSome then fail sourceFailure.Value
    elif mlirFailure.IsSome then fail mlirFailure.Value
    else Ok ()

/// Backend production admission requires a current source-stage receipt before
/// native realization. Fixture catalog construction alone does not grant it.
let validateRequiredProof (catalog: Catalog) =
    let required = Clef.Compiler.Nanopass.ObligationDischarge.ofGraph (graph catalog.Scope)
    match catalog.Proof with
    | None when not required.IsEmpty -> Error "Required proof envelope is missing for the current source obligations"
    | None -> Ok ()
    | Some proof ->
        validateProof catalog.Scope proof
        |> Result.bind (fun () ->
            if not required.IsEmpty && proof.Source.IsNone then Error "Required PSG obligations have no current compilation solver discharge"
            else Ok ())

let private imports operations =
    operations |> List.choose (function
        | MLIROp.FuncOp(FuncDecl(name, arguments, results, visibility, byval)) ->
            Some { Symbol = name; Arguments = arguments; Results = results; Visibility = visibility; Byval = byval; Boundary = None; IntrinsicWrite = None }
        | MLIROp.FuncOp(BoundaryFuncDecl declaration) ->
            Some { Symbol = declaration.Symbol; Arguments = BoundaryAbi.parameters declaration
                   Results = BoundaryAbi.results declaration; Visibility = FuncVisibility.Private
                   Byval = []; Boundary = Some declaration; IntrinsicWrite = None }
        | MLIROp.FuncOp(IntrinsicWriteDecl declaration) ->
            Some { Symbol = declaration.Symbol; Arguments = IntrinsicWriteAbi.parameters declaration
                   Results = IntrinsicWriteAbi.results declaration; Visibility = FuncVisibility.Private
                   Byval = []; Boundary = None; IntrinsicWrite = Some declaration }
        | _ -> None)

let private spatialModules operations =
    operations |> List.choose (function MLIROp.SpatialModule declaration -> Some declaration | _ -> None)

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
                    | MLIROp.FuncOp(BoundaryFuncDecl declaration) ->
                        Some(declaration.Symbol, (BoundaryAbi.parameters declaration, BoundaryAbi.results declaration))
                    | MLIROp.FuncOp(IntrinsicWriteDecl declaration) ->
                        Some(declaration.Symbol, (IntrinsicWriteAbi.parameters declaration, IntrinsicWriteAbi.results declaration))
                    | _ -> None)
                |> List.groupBy fst
            let conflicting = signatures |> List.exists (fun (_, rows) -> rows |> List.map snd |> List.distinct |> List.length <> 1)
            let signatureMap = signatures |> List.map (fun (name, rows) -> name, snd rows.Head) |> Map.ofList
            let globalTypes =
                definitions |> List.choose (function
                    | MLIROp.GlobalMemref(name, ty, _) -> Some(name, ty)
                    | MLIROp.GlobalArray(name, ty, _) -> Some(name, ty)
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
            let boundaryFailure =
                let retained = unit.Imports |> List.choose _.Boundary
                let writes = unit.Imports |> List.choose _.IntrinsicWrite
                match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryBoundary (graph scope) with
                | Error reason when not retained.IsEmpty || not writes.IsEmpty -> Some reason
                | Error _ -> None // Unpublished isolated physical correspondence fixtures.
                | Ok publication ->
                    let retainedMap = retained |> List.map (fun declaration -> declaration.Identity, declaration) |> Map.ofList
                    let writeMap = writes |> List.map (fun declaration -> declaration.Identity, declaration) |> Map.ofList
                    if retainedMap = publication.Imports && retained.Length = retainedMap.Count
                       && writeMap = publication.IntrinsicWriteImports && writes.Length = writeMap.Count then None
                    else Some "boundary declaration differs from its source-published ABI"
            let startupOwners =
                match expectedStartup with
                | None -> 0
                | Some(node, symbol) ->
                    unit.Definitions |> List.filter (fun row -> row.Occurrence.Focus.Id = node && definitionSymbol row.Operation = Some symbol) |> List.length
            let memoryFailure =
                let arrays = definitions |> List.choose (function MLIROp.GlobalArray(_, ty, authority) -> Some(ty, authority) | _ -> None)
                match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryMemory (graph scope) with
                | Error reason when not arrays.IsEmpty -> Some reason
                | Error _ -> None // Isolated physical fixtures have no source memory declarations.
                | Ok memory ->
                    let expected = memory.Operations |> Map.toList |> List.choose (function
                        | site, MemoryWitnessOperation.ArrayLiteral ({ Residence = MemoryResidence.ImmutableProgram _ } as authority) -> Some(site, authority)
                        | _ -> None) |> Map.ofList
                    let retained = arrays |> List.map (fun (_, authority) -> authority.Site, authority) |> Map.ofList
                    if retained <> expected || retained.Count <> arrays.Length then
                        Some "immutable array inventory differs from its source-published storage identities"
                    elif arrays |> List.exists (fun (ty, authority) ->
                        let expectedType = SettledScalar.tryType authority.Element |> Option.map (fun element -> TMemRefStatic(authority.Length, element))
                        expectedType <> Some ty || authority.Initializers.IsNone) then
                        Some "immutable array differs from its source-published initializer, representation or residence"
                    else None
            let spatialFailure =
                let retained = spatialModules operations
                let hardware = retained |> List.choose (function SpatialModuleWitness.Hardware plan -> Some(plan.Site,plan) | _ -> None)
                let kernels = retained |> List.choose (function SpatialModuleWitness.Kernel plan -> Some(plan.Site,plan) | _ -> None)
                let occurrenceAgrees = unit.Definitions |> List.forall (fun row ->
                    match row.Operation with
                    | MLIROp.SpatialModule(SpatialModuleWitness.Hardware plan) -> row.Occurrence.Focus.Id=plan.Site
                    | MLIROp.SpatialModule(SpatialModuleWitness.Kernel plan) -> row.Occurrence.Focus.Id=plan.Site
                    | _ -> true)
                match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.trySpatial (graph scope) with
                | Error reason when not retained.IsEmpty -> Some reason
                | Error _ -> None
                | Ok publication ->
                    if unit.SpatialModules<>retained || not occurrenceAgrees ||
                       Map.ofList hardware<>publication.Hardware || Map.ofList kernels<>publication.Kernels ||
                       hardware.Length<>publication.Hardware.Count || kernels.Length<>publication.Kernels.Count then
                        Some "spatial module differs from its complete source-published plan or declaration occurrence"
                    else None
            let proofFailure =
                catalog.Proof |> Option.bind (fun proof ->
                    match validateProof scope proof with Error reason -> Some reason | Ok () -> None)
            if occurrenceFailure.IsSome then fail occurrenceFailure.Value
            elif proofFailure.IsSome then fail proofFailure.Value
            elif boundaryFailure.IsSome then fail boundaryFailure.Value
            elif memoryFailure.IsSome then fail memoryFailure.Value
            elif spatialFailure.IsSome then fail spatialFailure.Value
            elif symbols.Length <> (Set.ofList symbols).Count then fail "duplicate symbol definition ownership"
            elif duplicateRecords || definitions.Length <> recorded.Length || definitions |> List.exists (fun op -> not (List.contains op recorded)) then
                fail "missing, duplicate or changed emitted definition correspondence"
            elif all |> List.filter isOwnedOperation |> List.length <> definitions.Length then fail "nested definition has no module-unit ownership"
            elif all |> List.choose (function MLIROp.FuncOp(FuncDecl _ | BoundaryFuncDecl _ | IntrinsicWriteDecl _) as op -> Some op | _ -> None) |> List.length <> (imports operations).Length then
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

let private createCore proof scope activation definitions operations text writable =
    let unit =
        { Id = "whole-module"; Scope = scope; Definitions = definitions
          Imports = imports operations; SpatialModules = spatialModules operations; WritableStorage = writable
          Startup = startup scope activation
          HasOpaqueTargetContent = operations |> List.exists (function MLIROp.RawMLIR _ -> true | _ -> false)
          ContentHash = hash text }
    let catalog = { Scope = scope; Activation = activation; Units = [unit]; Proof = proof }
    validate scope operations text writable catalog |> Result.map (fun () -> catalog)

let create scope activation definitions operations text writable =
    createCore None scope activation definitions operations text writable

let createWithProof proof scope activation definitions operations text writable =
    createCore (Some proof) scope activation definitions operations text writable

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
                   {| symbol = row.Symbol; arguments = List.map string row.Arguments; results = List.map string row.Results
                      sourceBoundary = row.Boundary |> Option.map (fun declaration ->
                          {| declaration = NodeId.value declaration.Identity; library = declaration.Library
                             convention = string declaration.CallingConvention
                             parameters = declaration.Parameters |> List.map (snd >> string)
                             result = declaration.Result |> Option.map string |})
                      intrinsicWrite = row.IntrinsicWrite |> Option.map (fun declaration ->
                          {| declaration = NodeId.value declaration.Identity; core = NodeId.value declaration.Core
                             endpoint = NodeId.value declaration.Endpoint; surface = NodeId.value declaration.Surface
                             syscallNumber = string declaration.SyscallNumber
                             returnContract = NodeId.value declaration.ReturnContract
                             fd = string declaration.Fd; count = string declaration.Count
                             result = string declaration.Result; byteRepresentation = declaration.ByteRepresentation.Name |}) |})
               writableSymbols = List.map fst unit.WritableStorage
               startup = unit.Startup |> Option.map (fun (node, symbol) -> {| node = NodeId.value node; symbol = symbol |}) |})
    let proof =
        catalog.Proof |> Option.map (fun envelope ->
            {| obligations = envelope.Obligations |> List.map (fun ob -> {| anchor = ob.Id; source = ob.Source; kind = ob.Kind |})
               mlirSha256 = hash envelope.Text
               source = envelope.Source |> Option.map _.Evidence
               mlir = envelope.Mlir |> Option.map _.Evidence |})
    let manifest = {| schema = 2; witnessRun = witnessRun catalog.Scope; semanticScope = "whole-checked-graph"; activation = string catalog.Activation; units = units; proof = proof |}
    File.WriteAllText(path, JsonSerializer.Serialize(manifest, JsonSerializerOptions(WriteIndented = true)))
