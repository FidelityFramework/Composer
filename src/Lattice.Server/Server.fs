namespace Lattice.Server

open System
open System.IO
open System.Text.Json
open System.Threading
open System.Threading.Tasks
open Clef.Compiler.Editor
open StreamJsonRpc

module private Wire =
    let field (name: string) (value: JsonElement) = value.GetProperty(name)
    let text name value = (field name value).GetString()
    let number name value = (field name value).GetInt32()
    let path (uri: string) =
        let value = Uri(uri)
        if not value.IsFile then invalidArg "uri" "Only local file URIs are supported."
        Path.GetFullPath(value.LocalPath).Replace('\\', '/')
    let uri file = Uri(Path.GetFullPath file).AbsoluteUri
    let range (span: SourceSpan) =
        {| start = {| line = span.StartLine; character = span.StartCharacter |}
           ``end`` = {| line = span.EndLine; character = span.EndCharacter |} |}
    let location (span: SourceSpan) = box {| uri = uri span.FilePath; range = range span |}
    let node (value: NodeView) =
        {| nodeId = value.NodeId; name = value.Name |> Option.toObj; kind = value.Kind; ``type`` = value.Type
           location = value.Range |> Option.map location |> Option.defaultValue null |}
    let error code message = LocalRpcException(message, ErrorCode = code)
    let changed () = error -32801 "The checked project changed; request current results."

/// Standard LSP transport over the compiler-owned editing session. No inference
/// or obligation construction is performed by this adapter.
type Server(rpc: JsonRpc, configuredProject: string, solver: string) =
    let gate = obj()
    let dispatchSlots = new SemaphoreSlim(2)
    let mutable session: EditorSession option = None
    let mutable sources: Map<string, string> = Map.empty
    let mutable versions: Map<string, int> = Map.empty
    let mutable generation = 0L
    let mutable pending: Task<EditorSnapshot option> = Task.FromResult None
    let mutable proofCancellation = new CancellationTokenSource()
    // In-flight and completed work share one entry within the current generation.
    let mutable proofCache: Map<string, Task<ProofResult>> = Map.empty
    let mutable stopping = false
    let mutable published: Set<string> = Set.empty
    let mutable inputFiles: Set<string> = Set.empty
    let mutable watchers: FileSystemWatcher list = []
    let mutable inputChanged: unit -> unit = ignore

    let watchInputs files =
        let next = files |> Set.ofList
        let changed = next <> inputFiles
        if changed then
            for watcher in watchers do watcher.Dispose()
            inputFiles <- next
            let directories = next |> Seq.map Path.GetDirectoryName |> Seq.distinct |> Seq.filter Directory.Exists
            watchers <-
                [ for directory in directories do
                    let watcher = new FileSystemWatcher(directory)
                    watcher.NotifyFilter <- NotifyFilters.FileName ||| NotifyFilters.LastWrite ||| NotifyFilters.Size
                    let changed path =
                        let relevant = lock gate (fun () -> not stopping && Set.contains (Path.GetFullPath(path).Replace('\\', '/')) inputFiles)
                        if relevant then inputChanged()
                    watcher.Changed.Add(fun e -> changed e.FullPath)
                    watcher.Created.Add(fun e -> changed e.FullPath)
                    watcher.Deleted.Add(fun e -> changed e.FullPath)
                    watcher.Renamed.Add(fun e -> changed e.OldFullPath; changed e.FullPath)
                    // An overflow means the event stream cannot establish freshness.
                    watcher.Error.Add(fun _ -> if not stopping then inputChanged())
                    watcher.EnableRaisingEvents <- true
                    yield watcher ]
        changed

    let notify methodName value = rpc.NotifyWithParameterObjectAsync(methodName, box value)
    let log message = notify "window/logMessage" {| ``type`` = 1; message = message |}
    let invalidate () =
        generation <- generation + 1L
        proofCancellation.Cancel()
        proofCancellation.Dispose()
        proofCancellation <- new CancellationTokenSource()
        proofCache <- Map.empty
        session |> Option.iter (fun s -> s.Invalidate() |> ignore)

    let publish (stamp: int64) (snapshot: EditorSnapshot) = task {
        let sends = lock gate (fun () ->
            if generation <> stamp || stopping then []
            elif watchInputs snapshot.InputFiles then
                // A new dependency may have changed before its watcher existed.
                // Settle one check with the complete watch set installed.
                inputChanged()
                []
            else
                let byFile = snapshot.Diagnostics |> List.choose (fun d -> d.Range |> Option.map (fun r -> r.FilePath, (d, r))) |> List.groupBy fst |> Map.ofList
                let files = Set.union published (snapshot.Sources |> Seq.map (fun s -> s.FilePath) |> Set.ofSeq)
                published <- files
                [ for file in files do
                    let diagnostics =
                        Map.tryFind file byFile |> Option.defaultValue []
                        |> List.map (fun (_, (d, span)) ->
                            {| range = Wire.range span
                               severity = match d.EffectiveSeverity with "Error" -> 1 | "Warning" -> 2 | _ -> 3
                               tags = if d.IsUnnecessary then [1] else [] // LSP DiagnosticTag.Unnecessary enables normal editor fading.
                               code = d.Code; source = "CCS"; message = d.Message |})
                    // Omit version for a file that has not been opened by this client.
                    let payload: obj =
                        match Map.tryFind file versions with
                        | Some version -> box {| uri = Wire.uri file; version = version; diagnostics = diagnostics |}
                        | None -> box {| uri = Wire.uri file; diagnostics = diagnostics |}
                    yield rpc.NotifyWithParameterObjectAsync("textDocument/publishDiagnostics", payload)
                  yield notify "clef/proofsChanged" {| checkGeneration = string stamp |} ])
        do! Task.WhenAll(sends)
        if lock gate (fun () -> generation = stamp) then
            for failure in snapshot.ParseFailures do
                do! log (sprintf "%s: %s" failure.FilePath (String.concat "\n" failure.Messages))
            for diagnostic in snapshot.Diagnostics do
                if diagnostic.Range.IsNone then do! log (sprintf "%s: %s" diagnostic.Code diagnostic.Message)
            match snapshot.Failure with
            | Some message -> do! log message
            | None -> ()
    }

    let startCheck () =
        let scheduled = lock gate (fun () ->
            // A queued watcher callback may arrive after shutdown or disposal.
            if stopping then None
            else
                invalidate()
                match session with
                | Some s ->
                    let requested, inputs, token = generation, sources, proofCancellation.Token
                    pending <- task {
                        try
                            // Invalidate immediately, but let a burst of edits or file
                            // notifications settle before entering the compiler.
                            do! Task.Delay(150, token)
                            let work = lock gate (fun () ->
                                if stopping || requested <> generation then Task.FromResult None
                                else s.CheckAsync inputs)
                            return! work.WaitAsync(token)
                        with :? OperationCanceledException -> return None
                    }
                    Some (generation, pending)
                | None -> None)
        match scheduled with
        | None -> ()
        | Some (stamp, work) ->
            let completion = task {
                do! notify "clef/proofsChanged" {| checkGeneration = string stamp |}
                let! result = work
                match result with
                | Some snapshot -> do! publish stamp snapshot
                | None -> ()
            }
            completion.ContinueWith((fun (t: Task) -> if t.IsFaulted then Console.Error.WriteLine(t.Exception.GetBaseException().Message)), TaskScheduler.Default) |> ignore

    do inputChanged <- startCheck

    let current (file: string) (expectedVersion: int option) = task {
        let stamp, work = lock gate (fun () -> generation, pending)
        let! snapshot = work
        return lock gate (fun () ->
            if stopping || stamp <> generation then raise (Wire.changed())
            match expectedVersion with
            | Some expected when Map.tryFind file versions <> Some expected -> raise (Wire.changed())
            | _ -> ()
            match snapshot with
            | Some value when watchInputs value.InputFiles ->
                inputChanged()
                raise (Wire.changed())
            | Some value when not value.ParseFailures.IsEmpty ->
                let files = value.ParseFailures |> List.map (fun f -> Path.GetFileName f.FilePath) |> String.concat ", "
                raise (Wire.error -32002 ("CCS could not parse " + files + ". See Lattice output for parser messages."))
            | Some value when value.Failure.IsNone -> stamp, value
            | Some value -> raise (Wire.error -32002 (value.Failure |> Option.defaultValue "Project checking failed."))
            | None -> raise (Wire.error -32002 "No checked project is available."))
    }

    [<JsonRpcMethod("initialize", UseSingleObjectParameterDeserialization = true)>]
    member _.Initialize(parameters: JsonElement) =
        let root =
            let mutable value = Unchecked.defaultof<JsonElement>
            if parameters.TryGetProperty("rootUri", &value) && value.ValueKind = JsonValueKind.String then Wire.path (value.GetString())
            else Directory.GetCurrentDirectory()
        let project =
            if configuredProject <> "" then Path.GetFullPath configuredProject
            else
                match Directory.GetFiles(root, "*.fidproj") with
                | [| project |] -> project
                | [||] -> raise (Wire.error -32602 "Open a folder containing one .fidproj, or configure --project with an explicit path.")
                | _ -> raise (Wire.error -32602 "Multiple .fidproj files: select one with --project.")
        lock gate (fun () -> session <- Some (EditorSession project))
        box {| capabilities =
                   {| positionEncoding = "utf-16"
                      textDocumentSync = {| openClose = true; change = 1 |}
                      hoverProvider = true
                      definitionProvider = true
                      experimental = {| clefProofs = {| version = 1 |} |} |}
               serverInfo = {| name = "Lattice (CCS)"; version = "0.1.0" |} |}

    [<JsonRpcMethod("initialized", UseSingleObjectParameterDeserialization = true)>]
    member _.Initialized(_parameters: JsonElement) = startCheck()

    [<JsonRpcMethod("textDocument/didOpen", UseSingleObjectParameterDeserialization = true)>]
    member _.Open(parameters: JsonElement) =
        let document = Wire.field "textDocument" parameters
        let file = Wire.path (Wire.text "uri" document)
        if Wire.text "languageId" document = "clef" && file.EndsWith(".clef", StringComparison.Ordinal) then
            lock gate (fun () ->
                sources <- Map.add file (Wire.text "text" document) sources
                versions <- Map.add file (Wire.number "version" document) versions
                startCheck())

    [<JsonRpcMethod("textDocument/didChange", UseSingleObjectParameterDeserialization = true)>]
    member _.Change(parameters: JsonElement) =
        let document = Wire.field "textDocument" parameters
        let file = Wire.path (Wire.text "uri" document)
        let version = Wire.number "version" document
        let changes = Wire.field "contentChanges" parameters |> fun v -> v.EnumerateArray() |> Seq.toArray
        if changes.Length <> 1 then raise (Wire.error -32602 "Lattice currently requires full-document synchronization.")
        let mutable ranged = Unchecked.defaultof<JsonElement>
        if changes[0].TryGetProperty("range", &ranged) then raise (Wire.error -32602 "Ranged edits were not negotiated.")
        lock gate (fun () ->
            match Map.tryFind file versions with
            | Some previous when version > previous ->
                versions <- Map.add file version versions
                sources <- Map.add file (Wire.text "text" changes[0]) sources
                startCheck()
            | _ -> ())

    [<JsonRpcMethod("textDocument/didClose", UseSingleObjectParameterDeserialization = true)>]
    member _.Close(parameters: JsonElement) =
        let file = Wire.field "textDocument" parameters |> Wire.text "uri" |> Wire.path
        lock gate (fun () ->
            sources <- Map.remove file sources
            versions <- Map.remove file versions
            startCheck())

    [<JsonRpcMethod("workspace/didChangeWatchedFiles", UseSingleObjectParameterDeserialization = true)>]
    member _.FilesChanged(_parameters: JsonElement) = startCheck()

    [<JsonRpcMethod("$/setTrace", UseSingleObjectParameterDeserialization = true)>]
    member _.Trace(_parameters: JsonElement) = ()

    [<JsonRpcMethod("textDocument/hover", UseSingleObjectParameterDeserialization = true)>]
    member _.Hover(parameters: JsonElement) = task {
        let file = Wire.field "textDocument" parameters |> Wire.text "uri" |> Wire.path
        let position = Wire.field "position" parameters
        let! stamp, snapshot = current file None
        return lock gate (fun () ->
            if stamp <> generation then raise (Wire.changed())
            match session |> Option.bind (fun s -> s.TryHover(snapshot.Revision, file, Wire.number "line" position, Wire.number "character" position)) with
            | None -> null
            | Some hover ->
                let title = match hover.Name with Some name -> name + ": " + hover.Type | None -> hover.Type
                let facts = hover.ValueRange |> Option.map (fun r -> "\nRange: " + r) |> Option.defaultValue ""
                box {| contents = {| kind = "plaintext"; value = title + facts |}; range = Wire.range hover.Range |})
    }

    [<JsonRpcMethod("textDocument/definition", UseSingleObjectParameterDeserialization = true)>]
    member _.Definition(parameters: JsonElement) = task {
        let file = Wire.field "textDocument" parameters |> Wire.text "uri" |> Wire.path
        let position = Wire.field "position" parameters
        let! stamp, snapshot = current file None
        return lock gate (fun () ->
            if stamp <> generation then raise (Wire.changed())
            session |> Option.bind (fun s -> s.TryHover(snapshot.Revision, file, Wire.number "line" position, Wire.number "character" position))
            |> Option.bind (fun h -> h.Definition) |> Option.map Wire.location |> Option.defaultValue null)
    }

    /// Read-only startup projection; all order, source identity and pending
    /// facts were constructed in CCS. The transport performs no selection.
    [<JsonRpcMethod("clef/programInitialization", UseSingleObjectParameterDeserialization = true)>]
    member _.ProgramInitialization(parameters: JsonElement) = task {
        let document = Wire.field "textDocument" parameters
        let uri = Wire.text "uri" document
        let file = Wire.path uri
        let version = Wire.number "version" document
        let! stamp, snapshot = current file (Some version)
        return lock gate (fun () ->
            if generation <> stamp || Map.tryFind file versions <> Some version then raise (Wire.changed())
            let plan = snapshot.ProgramInitialization |> Option.map (fun value ->
                box {| entry = Wire.node value.Entry; sourceEntry = Wire.node value.SourceEntry
                       spineNodeId = value.SpineNodeId; entryCallNodeId = value.EntryCallNodeId
                       initializers = value.Initializers |> List.map (fun row ->
                           {| ordinal = row.Ordinal; ``module`` = Wire.node row.Module
                              binding = Wire.node row.Binding; value = Wire.node row.Value
                              requiresProgramStorage = row.RequiresProgramStorage; hasProgramAuthority = row.HasProgramAuthority |}) |})
            box {| textDocument = {| uri = uri; version = version |}; checkGeneration = string stamp
                   compilerIdentity = snapshot.CompilerIdentity; plan = plan |> Option.defaultValue null
                   pending = snapshot.ProgramInitializationPending |> List.map (fun value ->
                       {| site = Wire.node value.Site; sources = value.Sources |> List.map Wire.node; reason = value.Reason |}) |})
    }

    [<JsonRpcMethod("clef/proofs", UseSingleObjectParameterDeserialization = true)>]
    member _.Proofs(parameters: JsonElement, cancellation: CancellationToken) = task {
        let document = Wire.field "textDocument" parameters
        let uri = Wire.text "uri" document
        let file = Wire.path uri
        let version = Wire.number "version" document
        let! stamp, snapshot = current file (Some version)
        let token = lock gate (fun () ->
            if stamp <> generation then raise (Wire.changed())
            proofCancellation.Token)
        use linked = CancellationTokenSource.CreateLinkedTokenSource(token, cancellation)
        let selected = snapshot.Obligations |> List.filter (fun o ->
            o.Range |> Option.exists (fun r -> r.FilePath = file)
            || o.Premises |> List.exists (fun p -> p.Range |> Option.exists (fun r -> r.FilePath = file)))
        let check (obligation: ObligationView) = task {
            let work = lock gate (fun () ->
                if generation <> stamp then raise (Wire.changed())
                match Map.tryFind obligation.QueryHash proofCache with
                | Some work -> work
                | None ->
                    let work = task {
                        if not snapshot.ParseFailures.IsEmpty || snapshot.Diagnostics |> List.exists (fun d -> d.EffectiveSeverity = "Error") then
                            return { State = "not-dispatched"; Detail = "Resolve compiler errors before dispatching this snapshot."; Solver = solver; QueryHash = obligation.QueryHash }
                        else
                            do! dispatchSlots.WaitAsync(token)
                            try return! ProofDispatch.checkAsync solver obligation token
                            finally dispatchSlots.Release() |> ignore
                    }
                    proofCache <- Map.add obligation.QueryHash work proofCache
                    work)
            // A caller may stop waiting without cancelling another document's
            // identical query. An edit still cancels all work for this generation.
            let! result = work.WaitAsync(linked.Token)
            return box {| id = obligation.Id; kind = obligation.Kind; logic = obligation.Logic
                          statement = obligation.Statement; source = obligation.Source; refs = obligation.Refs
                          premises = obligation.Premises |> List.map (fun p -> sprintf "%s: %s (%s, node %d)" (p.Name |> Option.defaultValue p.Kind) p.Type p.Kind p.NodeId)
                          location = obligation.Range |> Option.map Wire.location |> Option.defaultValue null
                          smtLib = obligation.SmtLib; queryHash = obligation.QueryHash
                          status = {| phase = "source"; state = result.State; detail = result.Detail; solver = result.Solver |} |}
        }
        try
            let! obligations = selected |> List.map check |> Task.WhenAll
            return lock gate (fun () ->
                if generation <> stamp || Map.tryFind file versions <> Some version then raise (Wire.changed())
                box {| textDocument = {| uri = uri; version = version |}
                       checkGeneration = string stamp
                       compilerIdentity = snapshot.CompilerIdentity
                       obligations = obligations |})
        with :? OperationCanceledException when not cancellation.IsCancellationRequested ->
            return raise (Wire.changed())
    }

    [<JsonRpcMethod("shutdown")>]
    member _.Shutdown() =
        lock gate (fun () -> stopping <- true; invalidate())
        null : obj

    [<JsonRpcMethod("exit")>]
    member _.Exit() = rpc.Dispose()

    interface IDisposable with
        member _.Dispose() =
            lock gate (fun () -> stopping <- true; proofCancellation.Cancel())
            for watcher in watchers do watcher.Dispose()
            proofCancellation.Dispose()
