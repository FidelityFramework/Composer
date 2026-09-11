module PlatformCatalogTests

open System
open System.IO
open Clef.Compiler.Project

/// Resolve the real catalogue with the compiler's manifest and dependency loader.
/// Optional arguments add consumer manifests from sibling repositories.
[<EntryPoint>]
let main args =
    let platform = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../../../Fidelity.Platform"))
    let manifests =
        Seq.append (Directory.EnumerateFiles(platform, "*.fidproj", SearchOption.AllDirectories)) args
        |> Seq.map Path.GetFullPath
        |> Seq.filter (fun path ->
            [ "/targets/"; "/obj/"; "/bin/"; "/worktrees/"; "/.git/"; "/.claude/"; "/.codex/" ]
            |> List.exists path.Contains |> not)
        |> Seq.distinct
        |> Seq.sort
        |> Seq.toList
    let failures = ResizeArray<string>()
    let mutable sourceCount = 0
    for manifest in manifests do
        match FidprojLoader.load manifest with
        | Error error -> failures.Add(sprintf "%s: %s" manifest error)
        | Ok options ->
            match SourceResolver.getSourcesAndLibraries options with
            | Error error -> failures.Add(sprintf "%s: %s" manifest (SourceResolutionError.format error))
            | Ok resolved ->
                sourceCount <- sourceCount + resolved.SourcePaths.Length
                if resolved.SourcePaths.Length <> (resolved.SourcePaths |> List.distinct |> List.length) then
                    failures.Add(sprintf "%s: repeated source declaration identity" manifest)
                for source in resolved.SourcePaths do
                    if not (File.Exists source) then failures.Add(sprintf "%s: missing source %s" manifest source)
                    if source.StartsWith(platform + "/", StringComparison.Ordinal) && Path.GetExtension source <> ".clef" then
                        failures.Add(sprintf "%s: platform production source is not Clef: %s" manifest source)
    if failures.Count > 0 then
        failures |> Seq.iter (eprintfn "%s")
        eprintfn "%d of %d catalogue/consumer manifests failed resolution" failures.Count manifests.Length
        1
    else
        printfn "%d catalogue/consumer manifests resolved; %d source references retain unique identities within each closure" manifests.Length sourceCount
        0
