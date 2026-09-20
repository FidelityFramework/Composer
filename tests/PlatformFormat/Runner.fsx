#load "../regression/RunnerCore.fsx"

open System
open System.IO
open System.Security.Cryptography
open System.Text.Json

let root = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../.."))
let compiler =
    match fsi.CommandLineArgs.[1..] with
    | [||] -> Path.Combine(root, "src/bin/Debug/net10.0/Composer")
    | [|path|] -> Path.GetFullPath path
    | _ -> failwith "Usage: dotnet fsi tests/PlatformFormat/Runner.fsx [Composer executable]"
let platform = Path.GetFullPath(Path.Combine(root, "../Fidelity.Platform"))
let work = Path.Combine(Path.GetTempPath(), "composer-platform-format-" + Guid.NewGuid().ToString("N"))
Directory.CreateDirectory work |> ignore
printfn "Native artifacts: %s" work

let source = Path.Combine(platform, "Environments/Linux/x86_64/Format.clef")
File.Copy(source, Path.Combine(work, "Format.clef"))
for name in ["Boundaries.clef"; "ExpectedOutput.txt"] do
    File.Copy(Path.Combine(__SOURCE_DIRECTORY__, name), Path.Combine(work, name))
let surface = Path.Combine(platform, "Environments/Linux/x86_64/Fidelity.Platform.CompilerSurface.fidproj")
let project = Path.Combine(work, "Boundaries.fidproj")
File.WriteAllText(project,
    "[package]\nname = \"FormatBoundaries\"\nversion = \"0.1.0\"\n" +
    "[compilation]\ntarget = \"cpu\"\n[dependencies]\nplatform = { path = " +
    JsonSerializer.Serialize(surface) + " }\n[build]\nsources = [\"Format.clef\", \"Boundaries.clef\"]\n" +
    "output = \"format-boundaries\"\noutput_kind = \"console\"\n")
let inputs =
    [ compiler
      Path.Combine(Path.GetDirectoryName compiler, "Composer.dll")
      Path.Combine(Path.GetDirectoryName compiler, "Clef.Compiler.Service.dll")
      source
      Path.Combine(work, "Boundaries.clef")
      Path.Combine(work, "ExpectedOutput.txt") ]
    |> List.map (fun path ->
        use stream = File.OpenRead path
        {| path = path; sha256 = Convert.ToHexString(SHA256.HashData stream) |})
File.WriteAllText(Path.Combine(work, "inputs.json"),
    JsonSerializer.Serialize(inputs, JsonSerializerOptions(WriteIndented = true)))

let runStage name executable arguments timeout =
    let result, _ = RunnerCore.runProcess executable arguments work None timeout
    match result with
    | RunnerCore.Completed(code, stdout, stderr) ->
        File.WriteAllText(Path.Combine(work, name + ".stdout.txt"), stdout)
        File.WriteAllText(Path.Combine(work, name + ".stderr.txt"), stderr)
        File.WriteAllText(Path.Combine(work, name + ".log"), stdout + stderr)
        if code <> 0 then failwithf "%s exited %d; see %s.log" name code name
        stdout, stderr
    | RunnerCore.Timeout timeout -> failwithf "%s timed out after %d ms" name timeout
    | RunnerCore.Failed error -> failwithf "%s could not run: %s" name error.Message

let mutable stage = "compile"
let failure =
    try
        let binary = Path.Combine(work, "format-boundaries")
        runStage "compile" compiler ["compile"; project; "-o"; binary; "-k"; "--no-color"] 240000 |> ignore
        stage <- "verify"
        let mlir = Path.Combine(work, "targets/intermediates/10_output.mlir")
        runStage "verify" "mlir-opt" [mlir; "--verify-each"; "-o"; Path.Combine(work, "verified.mlir")] 60000 |> ignore
        stage <- "native"
        let stdout, stderr = runStage "native" binary [] 30000
        let expected = File.ReadAllText(Path.Combine(work, "ExpectedOutput.txt"))
        if stdout.Replace("\r\n", "\n") <> expected.Replace("\r\n", "\n") || stderr <> "" then
            failwith "Native output differs from the 22 expected boundary lines or stderr is nonempty"
        ""
    with error -> error.Message
File.WriteAllText(Path.Combine(work, "evidence.json"),
    JsonSerializer.Serialize({| passed = failure = ""; stage = stage; failure = failure |},
                             JsonSerializerOptions(WriteIndented = true)))
if failure <> "" then
    eprintfn "FAIL: %s" failure
    exit 1
printfn "PASS: UTF-8 storage and snapshot isolation; 22 signed integer/decimal boundaries"
