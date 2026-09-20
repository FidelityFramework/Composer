/// Shared project/editor contract gate. CCS owns selection, diagnostics and
/// provenance; this module checks only their immutable public projections.
module ProgramLifetimeProjection

open System
open System.IO
open System.Text.Json
open Clef.Compiler.Editor

type Case = { name: string; before: string; after: string; code: string; message: string }

let run fixtureDirectory evidenceDirectory =
    let require condition message = if not condition then failwith message
    let current value = value |> Option.defaultWith (fun () -> failwith "Platform check was superseded")
    Directory.CreateDirectory evidenceDirectory |> ignore
    for name in ["Names.clef"; "Platform.clef"; "Main.clef"; "Platform.fidproj"; "App.fidproj"; "Cases.json"; "Startup.clef"; "Startup.fidproj"] do
        File.Copy(Path.Combine(fixtureDirectory, name), Path.Combine(evidenceDirectory, name), true)
    let path name = Path.GetFullPath(Path.Combine(evidenceDirectory, name)).Replace('\\', '/')
    let namesFile, platformFile = path "Names.clef", path "Platform.clef"
    let namesText, platformText = File.ReadAllText namesFile, File.ReadAllText platformFile
    let session = EditorSession(path "App.fidproj")
    let check changes = session.CheckAsync(changes).Result |> current
    let errors (snapshot: EditorSnapshot) =
        require snapshot.Failure.IsNone $"Project selection failed: {snapshot.Failure}"
        require snapshot.ParseFailures.IsEmpty $"Project source did not parse: {snapshot.ParseFailures}"
        snapshot.Diagnostics |> List.filter (fun diagnostic -> diagnostic.EffectiveSeverity = "Error")
    let admitted snapshot = require (errors snapshot |> List.isEmpty) $"Unexpected declaration errors: {snapshot.Diagnostics}"
    let position (text: string) marker =
        let index = text.LastIndexOf(marker, StringComparison.Ordinal)
        require (index >= 0) $"Missing source marker: {marker}"
        let prefix = text.Substring(0, index)
        let line = prefix |> Seq.filter ((=) '\n') |> Seq.length
        line, prefix.Length - (prefix.LastIndexOf('\n') + 1)
    let provenance (snapshot: EditorSnapshot) =
        for name, definitionLine in ["immutableName", 1; "mutableName", 2] do
            let line, column = position platformText name
            let hover = session.TryHover(snapshot.Revision, platformFile, line, column) |> current
            require (hover.Name = Some name && hover.Type = "string") $"Role reference lost its compiler type/name: {hover}"
            let declaration = session.TryHover(snapshot.Revision, namesFile, definitionLine, 4) |> current
            let definition = hover.Definition |> current
            require (declaration.Name = Some name && declaration.Type = "string"
                     && definition = declaration.Range && definition.FilePath = namesFile && definition.StartLine = definitionLine)
                $"Role reference lost its exact source definition: {definition}"
    let first = check Map.empty
    admitted first
    provenance first
    require (Set.isSubset (Set.ofList [path "App.fidproj"; path "Platform.fidproj"; namesFile; platformFile; path "Main.clef"]) (Set.ofList first.InputFiles))
        "Selected platform sources/manifests were omitted from editor inputs"
    let retained = JsonSerializer.Serialize first
    let renamed = namesText.Replace("constant-vault", "renamed-image").Replace("state-vault", "renamed-state")
    let renamedSnapshot = check (Map.ofList [namesFile, renamed])
    admitted renamedSnapshot
    provenance renamedSnapshot
    require (renamedSnapshot.Revision > first.Revision) "Unsaved designation rename did not advance revision"
    require ((renamedSnapshot.Sources |> List.find (fun source -> source.FilePath = namesFile)).Content = renamed)
        "Selected platform checking ignored unsaved dependency text"
    let cases = JsonSerializer.Deserialize<Case array>(File.ReadAllText(path "Cases.json"))
    let mutable revision = renamedSnapshot.Revision
    let evidence =
        cases |> Array.map (fun case ->
            let marked = platformText.Replace(case.before, case.after)
            let start = marked.IndexOf('«')
            let finish = marked.IndexOf('»')
            require (start >= 0 && finish > start) "Diagnostic fixture lost its marked span"
            let prefix = marked.Substring(0, start)
            let line = prefix |> Seq.filter ((=) '\n') |> Seq.length
            let column = prefix.Length - (prefix.LastIndexOf('\n') + 1)
            let source = marked.Remove(finish, 1).Remove(start, 1)
            let bad = check (Map.ofList [platformFile, source])
            let diagnostics = errors bad
            require (diagnostics.Length = 1) $"Expected one declaration error: {diagnostics}"
            let diagnostic = diagnostics.Head
            let message = case.message.Replace("{platform}", Path.GetFileName evidenceDirectory)
            require (diagnostic.Code = case.code && diagnostic.Message = message)
                $"Declaration diagnostic changed: {diagnostic}"
            let expected = { FilePath = platformFile; StartLine = line; StartCharacter = column
                             EndLine = line; EndCharacter = column + finish - start - 1 }
            require (diagnostic.Range = Some expected) $"Declaration diagnostic lost its exact source span: {diagnostic.Range}"
            require (bad.Revision > revision) "Error edit did not advance revision"
            let repaired = check Map.empty
            admitted repaired
            provenance repaired
            require (repaired.Revision > bad.Revision) "Repair did not advance revision"
            revision <- repaired.Revision
            {| name = case.name; diagnostic = diagnostic; errorRevision = bad.Revision; repairRevision = repaired.Revision |})
    require (File.ReadAllText namesFile = namesText && File.ReadAllText platformFile = platformText) "Unsaved edits changed disk files"
    require (JsonSerializer.Serialize first = retained) "A later platform check mutated an earlier public snapshot"
    let startupFile = path "Startup.clef"
    let startupText = File.ReadAllText startupFile
    let startupSession = EditorSession(path "Startup.fidproj")
    let startup = startupSession.CheckAsync(Map.empty).Result |> current
    admitted startup
    let plan = startup.ProgramInitialization |> current
    require (plan.SourceEntry.Name = Some "main") "Startup lost its preserved source entry"
    require ((plan.Initializers |> List.map (fun row -> row.Binding.Name)) = [Some "state"; Some "unused"; Some "stored"])
        "Startup order differs from the compiler's selected source occurrences"
    for ordinal, row in List.indexed plan.Initializers do
        require (row.Ordinal = ordinal && row.Binding.NodeId <> row.Value.NodeId && row.Binding.Range.IsSome && row.Value.Range.IsSome)
            "Startup order/provenance is incomplete"
        require (not row.HasProgramAuthority) "A source-only check invented native program-space authority"
    require ((plan.Initializers |> List.filter _.RequiresProgramStorage |> List.map (fun row -> row.Binding.Name)) = [Some "state"; Some "stored"])
        "Startup storage intent lost its source values"
    let startupRetained = JsonSerializer.Serialize startup
    let opaque = startupText.Replace("let initialize () = state <- true", "let mutable initialize = fun () -> state <- true")
    let pending = startupSession.CheckAsync(Map.ofList [startupFile, opaque]).Result |> current
    admitted pending
    require pending.ProgramInitialization.IsNone "An opaque initializer was given a settled startup plan"
    let fact = pending.ProgramInitializationPending |> List.exactlyOne
    require (fact.Site.Name = Some "unused" && fact.Site.Range.IsSome && fact.Sources.Length = 2
             && (fact.Sources |> List.forall (fun source -> source.Range.IsSome))) "Pending startup lost its exact source incidence"
    require (fact.Reason = "An indirect initializer call lacks a proved pre-entry dependency boundary.") "Pending startup reason changed"
    let repairedStartup = startupSession.CheckAsync(Map.empty).Result |> current
    admitted repairedStartup
    require (repairedStartup.ProgramInitialization.IsSome && repairedStartup.ProgramInitializationPending.IsEmpty)
        "Unsaved repair did not restore the compiler startup plan"
    require (JsonSerializer.Serialize startup = startupRetained && File.ReadAllText startupFile = startupText)
        "Startup query edited source or mutated an earlier snapshot"
    let result = {| compilerIdentity = first.CompilerIdentity; firstRevision = first.Revision; finalRevision = revision
                    renamedImmutable = "renamed-image"; renamedMutable = "renamed-state"; cases = evidence
                    startup = plan; startupPending = fact; startupRepairRevision = repairedStartup.Revision |}
    File.WriteAllText(path "evidence.json", JsonSerializer.Serialize(result, JsonSerializerOptions(WriteIndented = true)))
    printfn "PASS selected program-lifetime roles, exact declaration errors, source definitions and unsaved repairs"
    printfn "Evidence: %s" evidenceDirectory
