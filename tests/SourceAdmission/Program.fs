module SourceAdmissionTests

open System
open System.IO
open System.Security.Cryptography
open System.Text.Json

type Case = { Name: string; Source: string; ErrorLine: int; ErrorCode: string; ErrorMessage: string }
type Evidence = {
    Name: string
    Passed: bool
    ExpectedDiagnostic: string
    CompileExit: int
    VerifyExit: int
    NativeExit: int
    Failure: string
}

let private source lines = String.concat "\n" lines + "\n"
let private builderCase name expression = {
    Name = name
    Source = source (["module Admission"; "let builder value = value"; "[<EntryPoint>]"; "let main _ ="]
                     @ expression @ ["    if answer = 7 then 0 else 1"])
    ErrorLine = 5
    ErrorCode = "CCS8401"
    ErrorMessage = "This computation expression has no admitted native builder semantics"
}

let private sequenceCase name code line message expressions = {
    Name = name
    Source = source (["module Admission"; "[<Measure>] type m"; "[<Measure>] type s";
                      "[<EntryPoint>]"; "let main _ ="]
                     @ expressions @ ["    ignore wrong"; "    0"])
    ErrorLine = line
    ErrorCode = code
    ErrorMessage = message
}

let cases = [
    builderCase "function-return" ["    let answer = builder { return 7 }"]
    builderCase "function-let-bang" [
        "    let answer = builder {"
        "        let! value = 7"
        "        return value"
        "    }"
    ]
    builderCase "function-do-bang" [
        "    let answer = builder {"
        "        do! ()"
        "        return 7"
        "    }"
    ]
    { Name = "shadowed-seq"; ErrorLine = 5; ErrorCode = "CCS8401"
      ErrorMessage = "This computation expression has no admitted native builder semantics"; Source = source [
        "module Admission"
        "let seq value = value"
        "[<EntryPoint>]"
        "let main _ ="
        "    let answer = seq { yield 7 }"
        "    ignore answer"
        "    0"
    ] }
    { Name = "unowned-return"; ErrorLine = 4; ErrorCode = "CCS8401"
      ErrorMessage = "The 'return' form has no admitted native computation owner"; Source = source [
        "module Admission"; "[<EntryPoint>]"; "let main _ ="; "    return 7"
    ] }
    { Name = "unowned-yield"; ErrorLine = 4; ErrorCode = "CCS8401"
      ErrorMessage = "The 'yield' form requires an enclosing native seq expression"; Source = source [
        "module Admission"; "[<EntryPoint>]"; "let main _ ="; "    yield 7"
    ] }
    { Name = "unowned-resource-use"; ErrorLine = 4; ErrorCode = "CCS8401"
      ErrorMessage = "Resource-use bindings require an admitted native resource lifecycle"; Source = source [
        "module Admission"; "[<EntryPoint>]"; "let main _ ="
        "    use value = 7"
        "    if value = 7 then 0 else 1"
    ] }
    sequenceCase "sequence-mixed-dimensions" "CCS8040" 6
        "Measure mismatch: 'm' vs 's'; the residual 'm / s' is not 1"
        ["    let wrong = seq { yield 1<m>; yield 2<s> }"]
    sequenceCase "sequence-scalar-delegation" "CCS8003" 6
        "Type mismatch at {source}(6,31): expected 'seq<int>', got 'int'"
        ["    let wrong = seq { yield 1; yield! 42 }"]
    sequenceCase "sequence-delegation-annotation" "CCS8003" 7
        "Type mismatch at {source}(7,27): expected 'int', got 'bool'"
        ["    let ints = seq { yield 1 }"; "    let wrong: seq<bool> = seq { yield! ints }"]
    { Name = "ordinary-control"; ErrorLine = 0; ErrorCode = ""; ErrorMessage = ""; Source = source [
        "module Admission"
        "let increment value = value + 1"
        "[<EntryPoint>]"
        "let main _ ="
        "    let mutable observed: int = 0"
        "    Result.iter (fun value -> observed <- value) (Ok 4: Result<int, int>)"
        "    for index = 1 to 2 do"
        "        observed <- increment observed"
        "    if observed = 6 then"
        "        Console.writeln \"ordinary calls, Result.iter and loops: pass\""
        "        0"
        "    else 1"
    ] }
]

let private check compiler platform work (test: Case) =
    let directory = Path.Combine(work, test.Name)
    Directory.CreateDirectory directory |> ignore
    let sourceFile = Path.Combine(directory, "Main.clef")
    File.WriteAllText(sourceFile, test.Source)
    let project = Path.Combine(directory, "Admission.fidproj")
    File.WriteAllText(project, source [
        "[package]"; "name = \"Admission\""; "version = \"0.1.0\""
        "[compilation]"; "target = \"cpu\""
        "[dependencies]"; "platform = { path = " + JsonSerializer.Serialize(platform: string) + " }"
        "[build]"; "sources = [\"Main.clef\"]"; "output = \"admission\""; "output_kind = \"console\""
    ])
    // The CLI prefix is project-relative; a type mismatch's embedded source
    // range retains the absolute filename passed to CCS by the project loader.
    let message = test.ErrorMessage.Replace("{source}", sourceFile)
    let expected = if test.ErrorLine = 0 then "" else sprintf "Main.clef:%d: error %s: %s" test.ErrorLine test.ErrorCode message
    let mutable compileExit, verifyExit, nativeExit = -1, -1, -1
    let mutable failure = ""
    try
        let compiled = Tests.Process.run compiler ["compile"; project; "-k"; "--no-color"] 600000 (Some (Path.Combine(directory, "compile.log")))
        compileExit <- compiled.ExitCode
        let output = Path.Combine(directory, "targets/admission")
        let intermediates = Path.Combine(directory, "targets/intermediates")
        if test.ErrorLine > 0 then
            if compileExit <> 1 then failwithf "Expected source rejection exit 1, got %d" compileExit
            let matching =
                compiled.Output.Replace("\r\n", "\n").Split('\n')
                |> Array.filter (fun line -> line = expected)
            if matching.Length <> 1 then
                failwithf "Expected exactly one diagnostic '%s'; got %d" expected matching.Length
            if matching.[0].Contains "[unreachable]" then failwith "Expected an effective source error, not an unreachable finding"
            if not (compiled.Output.Contains "Compilation failed with 1 error(s)") then
                failwith "Expected the source diagnostic gate to reject this single invalid form"
            if File.Exists output then failwith "Rejected source produced a native executable"
            if Directory.Exists intermediates && not (Seq.isEmpty (Directory.EnumerateFiles(intermediates, "*.mlir", SearchOption.AllDirectories))) then
                failwith "Rejected source reached witnessed MLIR output"
        else
            if compileExit <> 0 then failwithf "Ordinary source control failed compilation (%d)" compileExit
            let mlir = Path.Combine(intermediates, "10_output.mlir")
            if not (File.Exists mlir) then failwith "Positive control has no retained MLIR module"
            let verified = Tests.Process.run "mlir-opt" [mlir; "--verify-each"; "-o"; Path.Combine(directory, "verified.mlir")] 60000 (Some (Path.Combine(directory, "verify.log")))
            verifyExit <- verified.ExitCode
            if verifyExit <> 0 then failwithf "Positive control failed stock MLIR verification (%d)" verifyExit
            let executed = Tests.Process.run output [] 10000 (Some (Path.Combine(directory, "run.log")))
            nativeExit <- executed.ExitCode
            if nativeExit <> 0 then failwithf "Positive control failed native execution (%d)" nativeExit
            if executed.Output.Replace("\r\n", "\n").TrimEnd('\n', '\r') <> "ordinary calls, Result.iter and loops: pass" then
                failwith "Positive control output changed"
    with error -> failure <- error.Message
    let passed = failure = ""
    printfn "%s %s%s" (if passed then "PASS" else "FAIL") test.Name (if passed then "" else ": " + failure)
    { Name = test.Name; Passed = passed; ExpectedDiagnostic = expected
      CompileExit = compileExit; VerifyExit = verifyExit; NativeExit = nativeExit; Failure = failure }

[<EntryPoint>]
let main args =
    let root = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../.."))
    let compiler = if args.Length > 0 then Path.GetFullPath args.[0] else Path.Combine(root, "src/bin/Debug/net10.0/Composer")
    let requested = args |> Array.skip (min 1 args.Length) |> Array.toList
    for name in requested do
        if not (cases |> List.exists (fun test -> test.Name = name)) then failwithf "Unknown source admission case: %s" name
    let selected = if requested.IsEmpty then cases else cases |> List.filter (fun test -> List.contains test.Name requested)
    let platform = Path.GetFullPath(Path.Combine(root, "../Fidelity.Platform/Environments/Linux/x86_64/Fidelity.Platform.CompilerSurface.fidproj"))
    let work = Path.Combine(Path.GetTempPath(), "composer-source-admission-" + Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory work |> ignore
    let compilerFiles =
        [compiler; Path.Combine(Path.GetDirectoryName compiler, "Composer.dll"); Path.Combine(Path.GetDirectoryName compiler, "Clef.Compiler.Service.dll")]
        |> List.map (fun path -> {| path = path; sha256 = Convert.ToHexString(SHA256.HashData(File.ReadAllBytes path)) |})
    let results = selected |> List.map (check compiler platform work)
    let passed = results |> List.filter (fun result -> result.Passed) |> List.length
    File.WriteAllText(Path.Combine(work, "evidence.json"), JsonSerializer.Serialize({| compiler = compilerFiles; cases = results; passed = passed; total = results.Length |}, JsonSerializerOptions(WriteIndented = true)))
    printfn "%d/%d source admission cases passed. Evidence: %s" passed results.Length work
    if passed = results.Length then 0 else 1
