namespace Clef.Compiler.Editor

open System
open System.IO
open System.Security.Cryptography
open System.Text
open System.Threading
open System.Threading.Tasks
open Microsoft.FSharp.Reflection
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.Project
module CompilerDiagnostics = Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics
module Discharge = Clef.Compiler.Nanopass.ObligationDischarge
module PhaseConfig = Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig

module private Projection =
    // Main CCS has process-global node counters, measure cells and phase config.
    // All editor sessions in this process must share the same check lock.
    let checkGate = new SemaphoreSlim(1, 1)
    let path (value: string) = Path.GetFullPath(value).Replace('\\', '/')
    let hash (bytes: byte array) = Convert.ToHexStringLower(SHA256.HashData bytes)
    let compilerIdentity =
        lazy (typeof<SemanticNode>.Assembly.Location |> File.ReadAllBytes |> hash)

    let inputFiles projectPath =
        let visited = System.Collections.Generic.HashSet<string>(StringComparer.Ordinal)
        let inputs = System.Collections.Generic.SortedSet<string>(StringComparer.Ordinal)
        let rec visit manifest =
            let manifest = path manifest
            inputs.Add manifest |> ignore
            if visited.Add manifest then
                // Use the compiler loader's exact dependency paths and declared
                // source base. There is no manifest-name or directory heuristic.
                match FidprojLoader.load manifest with
                | Error _ -> () // Keep its known path for repair; the check reports the failure.
                | Ok options ->
                    for source in options.SourceFiles do
                        inputs.Add(path (Path.Combine(options.ProjectDirectory, source))) |> ignore
                    for dependency in options.Dependencies do
                        // This matches SourceResolver: dependencies without a
                        // local Path are not source inputs in current CCS.
                        dependency.Path |> Option.iter visit
        visit projectPath
        inputs |> Seq.toList

    let span (range: SourceRange) : SourceSpan option =
        if String.IsNullOrWhiteSpace range.File || range.Start.Line < 1 || range.End.Line < 1 then None
        else
            Some {
                FilePath = path range.File
                StartLine = range.Start.Line - 1; StartCharacter = range.Start.Column
                EndLine = range.End.Line - 1; EndCharacter = range.End.Column
            }

    let kindName kind =
        let case, _ = FSharpValue.GetUnionFields(kind, typeof<SemanticKind>)
        case.Name

    let node (value: SemanticNode) : NodeView = {
        NodeId = NodeId.value value.Id
        Name =
            match value.Kind with
            | SemanticKind.Binding(name, _, _, _) | SemanticKind.VarRef(name, _)
            | SemanticKind.PatternBinding name -> Some name
            | _ -> None
        Kind = kindName value.Kind
        // Reading substitutions and formatting remain compiler operations.
        Type = value.Type |> Clef.Compiler.NativeTypedTree.UnionFind.applySubst |> formatType
        Range = span value.Range
        IsReachable = value.IsReachable
        ValueRange = value.ValueRange |> Option.map ValueRange.render
    }

    let diagnostic (value: CompilerDiagnostics.Diagnostic) : DiagnosticView = {
        Severity = string value.Severity
        EffectiveSeverity = string (CompilerDiagnostics.Diagnostic.effectiveSeverity value)
        Code = value.Code; Message = value.Message; Range = span value.Range
        RelatedNodeIds = value.RelatedNodes |> List.map NodeId.value
        Reachability = string value.Reachability
        IsUnnecessary = CompilerDiagnostics.Diagnostic.isUnnecessary value
    }

    let freeze revision projectPath (result: ProjectCheckResult) =
        let graph = result.CheckResult.Graph
        let nodes = graph.Nodes |> Map.map (fun _ value -> node value)
        let obligations =
            SemanticGraph.obligations graph
            |> List.map (fun (value, obligation) ->
                let query = Discharge.smtLib [obligation]
                let premises =
                    SemanticGraph.edgesInto value.Id graph
                    |> List.filter (fun edge -> edge.Class = EdgeClass.Obligation)
                    |> List.collect (fun edge -> edge.Sources)
                    |> List.distinct
                    |> List.choose (fun id -> Map.tryFind id nodes)
                {
                    Id = obligation.Id; NodeId = NodeId.value value.Id
                    Kind = obligation.Kind; Logic = obligation.Logic
                    Statement = obligation.Statement; Source = obligation.Source; Refs = obligation.Refs
                    Range = span value.Range; Premises = premises
                    SmtLib = query; QueryHash = query |> Encoding.UTF8.GetBytes |> hash
                })
        let hovers =
            graph.Nodes
            |> Map.toList
            |> List.choose (fun (id, value) ->
                let view = nodes[id]
                match view.Range, value.Kind with
                | None, _ | _, SemanticKind.Obligation _ -> None
                | Some range, _ ->
                    let definition =
                        match value.Kind with
                        | SemanticKind.VarRef(_, Some definition) -> nodes |> Map.tryFind definition |> Option.bind (fun n -> n.Range)
                        | _ -> None
                    let anchors =
                        match Map.tryFind ObligationMetadata.Anchors value.Metadata with
                        | Some (MetadataValue.StringList values) -> values
                        | _ -> []
                    Some {
                        NodeId = view.NodeId; Name = view.Name; Kind = view.Kind; Type = view.Type
                        Range = range; Definition = definition; IsReachable = view.IsReachable
                        ValueRange = view.ValueRange; ObligationIds = anchors
                    })
        let snapshot = {
            Revision = revision; ProjectPath = projectPath
            CompilerIdentity = compilerIdentity.Value
            Sources = result.SourceFiles |> List.map (fun (file, content) -> { FilePath = path file; Content = content })
            InputFiles = inputFiles projectPath @ (result.SourceFiles |> List.map (fst >> path)) |> List.distinct |> List.sort
            Diagnostics = result.CheckResult.Diagnostics |> List.map diagnostic
            ParseFailures = result.ParseErrors |> Map.toList |> List.map (fun (file, messages) -> { FilePath = path file; Messages = messages })
            Obligations = obligations; Failure = None
        }
        snapshot, hovers

/// A serialized editor read service over the current main CCS project reference.
/// It does not merge or substitute the separate fidelity worktree's semantics.
/// Checks follow CCS's authoritative .fidproj source ordering and platform facts.
type EditorSession(projectPath: string) =
    let projectPath = Projection.path projectPath
    let stateGate = obj()
    let mutable revision = 0L
    let mutable current: (EditorSnapshot * Map<string, HoverView list>) option = None

    let reserve () =
        lock stateGate (fun () ->
            revision <- revision + 1L
            current <- None
            revision)

    member _.Revision = lock stateGate (fun () -> revision)
    member _.Current = lock stateGate (fun () -> current |> Option.map fst)
    member _.Invalidate() = reserve ()

    /// Reserving the revision and hiding prior reads happens synchronously at
    /// invocation. Concurrent requests queue behind the process-wide CCS lock;
    /// a superseded result is never published or returned as current.
    member _.CheckAsync(volatileContent: Map<string, string>) : Task<EditorSnapshot option> =
        let requested = reserve ()
        let inputs = volatileContent |> Map.toList |> List.map (fun (file, text) -> Projection.path file, text) |> Map.ofList
        Task.Run<EditorSnapshot option>(fun () ->
            Projection.checkGate.Wait()
            try
                if lock stateGate (fun () -> requested <> revision) then None
                else
                    let config = PhaseConfig.getConfig()
                    let snapshot, hovers =
                        try
                            // Stdio belongs to LSP; this service requests no intermediate output.
                            PhaseConfig.setConfig PhaseConfig.defaultConfig
                            match ProjectChecker.checkProjectWithVolatile projectPath inputs with
                            | Ok result -> Projection.freeze requested projectPath result
                            | Error failure ->
                                { Revision = requested; ProjectPath = projectPath
                                  CompilerIdentity = Projection.compilerIdentity.Value
                                  Sources = []; InputFiles = Projection.inputFiles projectPath
                                  Diagnostics = []; ParseFailures = []; Obligations = []
                                  Failure = Some failure }, []
                        finally PhaseConfig.setConfig config
                    let index = hovers |> List.groupBy (fun view -> view.Range.FilePath) |> Map.ofList
                    lock stateGate (fun () ->
                        if requested <> revision then None
                        else
                            current <- Some (snapshot, index)
                            Some snapshot)
            finally Projection.checkGate.Release() |> ignore)

    member _.TryHover(expectedRevision: int64, filePath: string, line: int, character: int) : HoverView option =
        let file = Projection.path filePath
        lock stateGate (fun () ->
            match current with
            | Some (snapshot, index) when snapshot.Revision = expectedRevision ->
                index
                |> Map.tryFind file
                |> Option.bind (fun candidates ->
                    candidates
                    |> List.filter (fun view ->
                        let range = view.Range
                        (line, character) >= (range.StartLine, range.StartCharacter)
                        && (line, character) < (range.EndLine, range.EndCharacter))
                    // Select by the compiler's source intervals. References carry
                    // their resolved definition identity; names are never searched.
                    |> List.sortBy (fun view ->
                        let r = view.Range
                        r.EndLine - r.StartLine, r.EndCharacter - r.StartCharacter,
                        (match view.Kind with "VarRef" -> 0 | "Binding" | "PatternBinding" -> 1 | _ -> 2), view.NodeId)
                    |> List.tryHead
                    // Keep the error node in interval selection so an unresolved name
                    // cannot fall back to a containing expression's unrelated type.
                    |> Option.filter (fun view -> view.Kind <> "Error"))
            | _ -> None)
