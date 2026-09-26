// A bounded .NET stdio client for the actual Lattice server. Build the server
// and its aligned CCS.Editor/CCS dependencies before invoking this check.
open System
open System.IO
open System.Diagnostics
open System.Text
open System.Text.Json
open System.Threading

let args = fsi.CommandLineArgs |> Array.skip 1 |> Array.filter ((<>) "--")
if args.Length <> 2 then failwith "Usage: dotnet fsi LiveCaptureChecks.fsx -- <Lattice.Server.dll> <new-evidence-directory>"
let server, root = Path.GetFullPath args[0], Path.GetFullPath args[1]
if not (File.Exists server) || Directory.Exists root then failwith "Require a built server and a new evidence directory."
Directory.CreateDirectory root |> ignore
let source = """module Capture
[<Measure>] type m
[<Measure>] type s
[<EntryPoint>]
let main _ =
    let offset = 7<m>
    let make () = fun (value: int<m>) -> offset + value
    let shift = make ()
    let answer = shift 3<m>
    if answer = 10<m> then 0 else 1
"""
let file = Path.Combine(root, "Capture.clef")
let project = Path.Combine(root, "Capture.fidproj")
File.WriteAllText(file, source)
File.WriteAllText(project, "[package]\nname = \"live-capture-check\"\n[compilation]\ntarget = \"library\"\n[build]\nsources = [\"Capture.clef\"]\noutput_kind = \"library\"\n")
let uri = Uri(file).AbsoluteUri
let options = ProcessStartInfo("dotnet", UseShellExecute = false, RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true)
options.ArgumentList.Add server
options.ArgumentList.Add "--project"
options.ArgumentList.Add project
let child = new Process(StartInfo = options)
if not (child.Start()) then failwith "Could not start Lattice."
let stderr = child.StandardError.ReadToEndAsync()
let input, output = child.StandardInput.BaseStream, child.StandardOutput.BaseStream
let transcript = new StreamWriter(Path.Combine(root, "protocol.jsonl"))
let json value = JsonSerializer.Serialize value
let send value =
    let text = json value
    transcript.WriteLine(json {| direction = "send"; message = text |})
    transcript.Flush()
    let body = Encoding.UTF8.GetBytes text
    let header = Encoding.ASCII.GetBytes(sprintf "Content-Length: %d\r\n\r\n" body.Length)
    input.Write header
    input.Write body
    input.Flush()
let receive (token: CancellationToken) =
    let header = StringBuilder()
    let one = Array.zeroCreate<byte> 1
    while not (header.ToString().EndsWith("\r\n\r\n", StringComparison.Ordinal)) do
        output.ReadExactlyAsync(one.AsMemory(), token).AsTask().GetAwaiter().GetResult()
        header.Append(char one[0]) |> ignore
        if header.Length > 8192 then failwith "Oversized LSP header."
    let length =
        header.ToString().Split("\r\n", StringSplitOptions.RemoveEmptyEntries)
        |> Array.pick (fun line ->
            if line.StartsWith("Content-Length:", StringComparison.OrdinalIgnoreCase) then Some (Int32.Parse(line.Substring(15).Trim())) else None)
    if length < 0 || length > 16 * 1024 * 1024 then failwith "Invalid LSP frame length."
    let body = Array.zeroCreate<byte> length
    output.ReadExactlyAsync(body.AsMemory(), token).AsTask().GetAwaiter().GetResult()
    let text = Encoding.UTF8.GetString body
    transcript.WriteLine(json {| direction = "receive"; message = text |})
    transcript.Flush()
    use document = JsonDocument.Parse text
    document.RootElement.Clone()
let property name (value: JsonElement) = value.GetProperty(name: string)
let mutable requestId = 0
let request methodName parameters =
    requestId <- requestId + 1
    if obj.ReferenceEquals(parameters, null) then
        send {| jsonrpc = "2.0"; id = requestId; method = methodName |}
    else
        send {| jsonrpc = "2.0"; id = requestId; method = methodName; ``params`` = parameters |}
    use timeout = new CancellationTokenSource(TimeSpan.FromSeconds 30.0)
    let mutable reply = None
    while reply.IsNone do
        let message = receive timeout.Token
        let mutable id = Unchecked.defaultof<JsonElement>
        if message.TryGetProperty("id", &id) && id.ValueKind = JsonValueKind.Number && id.GetInt32() = requestId then reply <- Some message
    reply.Value
let result (reply: JsonElement) =
    let mutable error = Unchecked.defaultof<JsonElement>
    if reply.TryGetProperty("error", &error) then failwithf "LSP request failed: %s" (error.GetRawText())
    property "result" reply
let notify methodName parameters =
    if obj.ReferenceEquals(parameters, null) then send {| jsonrpc = "2.0"; method = methodName |}
    else send {| jsonrpc = "2.0"; method = methodName; ``params`` = parameters |}
let check condition message = if not condition then failwith message
let diagnostics version =
    use timeout = new CancellationTokenSource(TimeSpan.FromSeconds 30.0)
    let mutable found = None
    while found.IsNone do
        let message = receive timeout.Token
        let mutable methodName = Unchecked.defaultof<JsonElement>
        if message.TryGetProperty("method", &methodName) && methodName.GetString() = "textDocument/publishDiagnostics" then
            let payload = property "params" message
            let mutable actual = Unchecked.defaultof<JsonElement>
            if (property "uri" payload).GetString() = uri && payload.TryGetProperty("version", &actual) && actual.GetInt32() = version then
                found <- Some (property "diagnostics" payload |> fun items -> items.EnumerateArray() |> Seq.toArray)
    found.Value
let position line character = {| textDocument = {| uri = uri |}; position = {| line = line; character = character |} |}
let proofs version = request "clef/proofs" {| textDocument = {| uri = uri; version = version |} |}
let hasErrors items = items |> Array.exists (fun item -> (property "severity" item).GetInt32() = 1)
try
    request "initialize" {| rootUri = Uri(root).AbsoluteUri; capabilities = {| |} |} |> result |> ignore
    notify "initialized" {| |}
    notify "textDocument/didOpen" {| textDocument = {| uri = uri; languageId = "clef"; version = 1; text = source |} |}
    check (diagnostics 1 |> hasErrors |> not) "Initial source has effective errors."
    let lines = source.Split('\n')
    let captureColumn = lines[6].IndexOf("offset", StringComparison.Ordinal)
    let hover () = request "textDocument/hover" (position 6 captureColumn) |> result
    let firstHover = hover ()
    check ((property "contents" firstHover |> property "value").GetString().StartsWith("offset: int<m>", StringComparison.Ordinal)) "Capture hover lost source name/type."
    let definition = request "textDocument/definition" (position 6 captureColumn) |> result
    check ((property "uri" definition).GetString() = uri) "Capture navigates outside its source."
    let start = property "range" definition |> property "start"
    check ((property "line" start).GetInt32() = 5 && (property "character" start).GetInt32() = lines[5].IndexOf("offset", StringComparison.Ordinal)) "Capture navigates to an internal formal instead of the exact declaration."
    let firstProofs = proofs 1 |> result
    let compilerIdentity = (property "compilerIdentity" firstProofs).GetString()
    check (compilerIdentity.Length = 64) "Missing compiler identity."
    printfn "PASS live capture source hover and exact definition on revision 1"
    let broken = source.Replace("shift 3<m>", "shift 3<s>")
    notify "textDocument/didChange" {| textDocument = {| uri = uri; version = 2 |}; contentChanges = [| {| text = broken |} |] |}
    let errors = diagnostics 2
    check (errors |> Array.exists (fun error ->
        (property "code" error).GetString() = "CCS8040" && (property "severity" error).GetInt32() = 1 &&
        (property "range" error |> property "start" |> property "line").GetInt32() = 8)) "Unsaved dimensional edit did not produce its located compiler error."
    let obsolete = proofs 1
    check ((property "error" obsolete |> property "code").GetInt32() = -32801) "Obsolete document version was accepted."
    printfn "PASS unsaved dimensional error and rejection of obsolete revision 1"
    notify "textDocument/didChange" {| textDocument = {| uri = uri; version = 3 |}; contentChanges = [| {| text = source |} |] |}
    check (diagnostics 3 |> hasErrors |> not) "Repair did not clear effective errors."
    check ((hover ()).GetRawText() = firstHover.GetRawText()) "Repair changed the source capture hover."
    check ((request "textDocument/definition" (position 6 captureColumn) |> result).GetRawText() = definition.GetRawText()) "Repair changed the source declaration identity."
    let repaired = proofs 3 |> result
    check ((property "compilerIdentity" repaired).GetString() = compilerIdentity) "Compiler changed during gate."
    check ((property "checkGeneration" repaired).GetString() <> (property "checkGeneration" firstProofs).GetString()) "Repair reused the obsolete generation."
    check (File.ReadAllText file = source) "Unsaved edits changed disk source."
    printfn "PASS live repair, fresh generation, stable compiler identity and unchanged disk source"
    request "shutdown" null |> result |> ignore
    notify "exit" null
    // No further client messages follow exit; close the owned transport too.
    child.StandardInput.Close()
    check (child.WaitForExit(5000)) "Lattice did not exit after shutdown."
    check (child.ExitCode = 0) "Lattice exited unsuccessfully."
    File.WriteAllText(Path.Combine(root, "evidence.json"), json {| CompilerIdentity = compilerIdentity; Server = server; Versions = [1; 2; 3]; ExactSourceHoverAndDefinition = true; LocatedError = "CCS8040"; ObsoleteVersionError = -32801; UnsavedRepair = true; CleanShutdown = true |})
finally
    if not child.HasExited then child.Kill(true)
    child.WaitForExit()
    File.WriteAllText(Path.Combine(root, "server.stderr.log"), stderr.GetAwaiter().GetResult())
    child.Dispose()
    transcript.Dispose()
