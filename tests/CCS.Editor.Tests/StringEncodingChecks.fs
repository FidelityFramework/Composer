module StringEncodingChecks

open System
open System.IO
open System.Text.Json
open Clef.Compiler.Editor

let run root =
    let require condition message = if not condition then failwith message
    let current value = value |> Option.defaultWith (fun () -> failwith "Encoding check was superseded")
    let directory = Path.Combine(root, "string-encoding")
    Directory.CreateDirectory directory |> ignore
    let write (name: string) (source: string) =
        let path = Path.Combine(directory, name)
        File.WriteAllText(path, source)
        path
    let platform = """module EncodingPlatform
type WidthDeclaration = { Name: string; Bits: int }
type Representation = { Name: string; Capability: string; Family: string; Bits: int; MinMagnitude: string; MaxMagnitude: string; Boundary: string }
type TargetCore = { Runtime: string; Widths: WidthDeclaration array; Representations: Representation array }
type MemorySpace = { Name: string; Kind: string; Capacity: int; Alignment: int; Granularity: int; Growth: string; Access: string; Base: int option }
type ProgramLifetimeSpaces = { Immutable: string; Mutable: string option }
type PlatformDescription = { Id: string; Core: TargetCore option; Spaces: MemorySpace array; ProgramLifetime: ProgramLifetimeSpaces option }
let image = { Name="encoding-image"; Kind="rodata"; Capacity=4096; Alignment=16; Granularity=16; Growth="fixed"; Access="r"; Base=None }
let core: TargetCore = {
    Runtime = "libc"
    Widths = [| { Name="Pointer"; Bits=64 }; { Name="Register"; Bits=64 } |]
    Representations = [|
        { Name="uint8"; Capability="native"; Family="uint"; Bits=8; MinMagnitude="0"; MaxMagnitude="255"; Boundary="wrap" }
        { Name="int64"; Capability="native"; Family="int"; Bits=64; MinMagnitude="-9223372036854775808"; MaxMagnitude="9223372036854775807"; Boundary="wrap" }
        { Name="uint64"; Capability="native"; Family="uint"; Bits=64; MinMagnitude="0"; MaxMagnitude="18446744073709551615"; Boundary="wrap" } |] }
let description = { Id="encoding-editor"; Core=Some core; Spaces=[|image|]; ProgramLifetime=Some { Immutable="encoding-image"; Mutable=None } }
"""
    write "Platform.clef" platform |> ignore
    write "Platform.fidproj" "[package]\nname=\"EncodingPlatform\"\n[compilation]\ntarget=\"cpu\"\n[platform]\ndescription=\"EncodingPlatform.description\"\nruntime_model=\"libc\"\n[build]\nsources=[\"Platform.clef\"]\noutput_kind=\"library\"\n" |> ignore
    let project = write "App.fidproj" "[package]\nname=\"EncodingEditor\"\n[compilation]\ntarget=\"cpu\"\n[dependencies]\nplatform={path=\"Platform.fidproj\"}\n[build]\nsources=[\"Main.clef\"]\noutput_kind=\"library\"\n"
    let source = "module EncodingInput\n[<EntryPoint>]\nlet main _ =\n    let bytes: int array = [| 65 |]\n    let text = String.fromBytes bytes\n    String.length text\n"
    let file = write "Main.clef" source
    let session = EditorSession(project)
    let check changes = session.CheckAsync(changes).Result |> current
    let errors (snapshot: EditorSnapshot) =
        require snapshot.Failure.IsNone $"Encoding project failed: {snapshot.Failure}"
        require snapshot.ParseFailures.IsEmpty $"Encoding source did not parse: {snapshot.ParseFailures}"
        snapshot.Diagnostics |> List.filter (fun diagnostic -> diagnostic.EffectiveSeverity = "Error")
    let first = check Map.empty
    require (errors first |> List.isEmpty) $"Valid encoding failed: {first.Diagnostics}"
    require (first.Obligations |> List.exists (fun obligation -> obligation.Kind = "string-byte-storage")) "Selected encoding lost its compiler-owned storage obligation"
    let retained = JsonSerializer.Serialize first
    let invalid = source.Replace("[| 65 |]", "[| 256 |]")
    let failed = check (Map.ofList [file, invalid])
    let diagnostic = errors failed |> List.exactlyOne
    require (diagnostic.Code = "CCS8404" && diagnostic.Message = "String.fromBytes requires proved byte storage: every stored integer must be proved within 0..255.")
        $"Encoding diagnostic changed: {diagnostic}"
    let expected = { FilePath = file; StartLine = 4; StartCharacter = 15; EndLine = 4; EndCharacter = 37 }
    require (diagnostic.Range = Some expected && not diagnostic.RelatedNodeIds.IsEmpty) "Encoding diagnostic lost its full source span or graph participants"
    let repaired = check Map.empty
    require (errors repaired |> List.isEmpty) $"Unsaved repair did not clear encoding error: {repaired.Diagnostics}"
    require (first.Revision < failed.Revision && failed.Revision < repaired.Revision) "Encoding edit and repair lost revision identity"
    require (File.ReadAllText file = source && JsonSerializer.Serialize first = retained) "Encoding edit mutated disk or an earlier snapshot"
    let logicalProject = write "SourceOnly.fidproj" "[package]\nname=\"LogicalEncoding\"\n[compilation]\ntarget=\"library\"\n[build]\nsources=[\"Main.clef\"]\noutput_kind=\"library\"\n"
    let logical = EditorSession(logicalProject).CheckAsync(Map.empty).Result |> current
    require (errors logical |> List.isEmpty) $"Source-only intent acquired target errors: {logical.Diagnostics}"
    require (logical.Obligations |> List.forall (fun obligation -> obligation.Kind <> "string-byte-storage")) "Source-only checking invented physical byte-storage proof"
    printfn "PASS encoding platform selection, exact CCS8404 span/provenance, unsaved repair, immutable snapshot and source-only intent"
    printfn "Encoding compiler: %s" first.CompilerIdentity
