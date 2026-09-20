module NativeSequenceTests

open System
open System.IO
open System.Security.Cryptography
open System.Text.Json

type Evidence = {
    Passed: bool
    CompileExit: int
    VerifyExit: int
    NativeExit: int
    Failure: string
    ExpectedOutput: string
    ActualOutput: string
}

let private normalized (value: string) = value.Replace("\r\n", "\n")

[<EntryPoint>]
let main args =
    let compilerArgument, sampleName =
        match args with
        | [||] -> None, "15a_SequenceSemantics"
        | [|compiler|] when compiler <> "--sample" -> Some compiler, "15a_SequenceSemantics"
        | [|"--sample"; sample|] -> None, sample
        | [|compiler; "--sample"; sample|] -> Some compiler, sample
        | _ -> failwith "Usage: NativeSequences.Tests [Composer executable] [--sample 15a_SequenceSemantics|15b_SequenceElements|15c_SequenceTemplateBorrows|15d_SequenceAdditive|16_SeqOperations|16a_SequenceOperations|16b_SequenceCallbacks|16c_SequenceComposition|16d_SequenceCollect|16e_SequenceSearch|16f_SequenceOptions|16g_SequenceStartup|16h_SequenceApplications]"
    let sourceName, outputName =
        match sampleName with
        | "15a_SequenceSemantics" -> "SequenceSemantics", "sequence-semantics"
        | "15b_SequenceElements" -> "SequenceElements", "sequence-elements"
        | "15c_SequenceTemplateBorrows" -> "SequenceTemplateBorrows", "sequence-template-borrows"
        | "15d_SequenceAdditive" -> "SequenceAdditive", "sequence-additive"
        | "16_SeqOperations" -> "SeqOperations", "SeqOperations"
        | "16a_SequenceOperations" -> "SequenceOperations", "sequence-operations"
        | "16b_SequenceCallbacks" -> "SequenceCallbacks", "sequence-callbacks"
        | "16c_SequenceComposition" -> "SequenceComposition", "sequence-composition"
        | "16d_SequenceCollect" -> "SequenceCollect", "sequence-collect"
        | "16e_SequenceSearch" -> "SequenceSearch", "sequence-search"
        | "16f_SequenceOptions" -> "SequenceOptions", "sequence-options"
        | "16g_SequenceStartup" -> "SequenceStartup", "sequence-startup"
        | "16h_SequenceApplications" -> "SequenceApplications", "sequence-applications"
        | other -> failwithf "Unknown sequence sample: %s" other
    let root = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../.."))
    let compiler =
        compilerArgument |> Option.map Path.GetFullPath
        |> Option.defaultValue (Path.Combine(root, "src/bin/Debug/net10.0/Composer"))
    let sample = Path.Combine(root, "samples/console/FidelityHelloWorld", sampleName)
    let work = Path.Combine(Path.GetTempPath(), "composer-native-sequences-" + Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory work |> ignore
    File.Copy(Path.Combine(sample, sourceName + ".clef"), Path.Combine(work, sourceName + ".clef"))
    let additionalSources = if sampleName = "16g_SequenceStartup" then ["StartupFirst.clef"] else []
    for source in additionalSources do
        File.Copy(Path.Combine(sample, source), Path.Combine(work, source))
    File.Copy(Path.Combine(sample, "ExpectedOutput.txt"), Path.Combine(work, "ExpectedOutput.txt"))
    let project = File.ReadAllText(Path.Combine(sample, sourceName + ".fidproj"))
    let copiedProject =
        [ "../../../../../Fidelity.Platform/Environments/Linux/x86_64/Fidelity.Platform.CompilerSurface.fidproj"
          "../../../../../Fidelity.Platform/Profiles/Linux_x86_64_Default/Fidelity.Platform.fidproj" ]
        |> List.fold (fun (text: string) relative ->
            let absolute = Path.GetFullPath(Path.Combine(sample, relative))
            text.Replace(JsonSerializer.Serialize relative, JsonSerializer.Serialize absolute)) project
    let projectPath = Path.Combine(work, sourceName + ".fidproj")
    File.WriteAllText(projectPath, copiedProject)
    let expected = File.ReadAllText(Path.Combine(work, "ExpectedOutput.txt")) |> normalized
    let compilerFiles =
        [compiler; Path.Combine(Path.GetDirectoryName compiler, "Composer.dll"); Path.Combine(Path.GetDirectoryName compiler, "Clef.Compiler.Service.dll")]
        |> List.map (fun path ->
            use stream = File.OpenRead path
            {| path = path; sha256 = Convert.ToHexString(SHA256.HashData stream) |})
    let mutable compileExit = -1
    let mutable verifyExit = -1
    let mutable nativeExit = -1
    let mutable actual = ""
    let mutable failure = ""
    try
        let compiled = Tests.Process.run compiler ["compile"; projectPath; "-k"; "--no-color"] 600000 (Some (Path.Combine(work, "compile.log")))
        compileExit <- compiled.ExitCode
        if compileExit <> 0 then failwithf "Sequence compilation failed (%d)" compileExit
        let mlir = Path.Combine(work, "targets/intermediates/10_output.mlir")
        if not (File.Exists mlir) then failwith "Successful compilation omitted retained MLIR"
        let verified = Tests.Process.run "mlir-opt" [mlir; "--verify-each"; "-o"; Path.Combine(work, "verified.mlir")] 60000 (Some (Path.Combine(work, "verify.log")))
        verifyExit <- verified.ExitCode
        if verifyExit <> 0 then failwithf "Stock MLIR verification failed (%d)" verifyExit
        let executable = Path.Combine(work, "targets", outputName)
        let executed = Tests.Process.run executable [] 10000 (Some (Path.Combine(work, "run.log")))
        nativeExit <- executed.ExitCode
        actual <- normalized executed.Output
        if nativeExit <> 0 then failwithf "Sequence semantics failed at native exit %d" nativeExit
        if actual <> expected then failwith "Native sequence output differs from ExpectedOutput.txt"
    with error -> failure <- error.Message
    let result = {
        Passed = failure = ""; CompileExit = compileExit; VerifyExit = verifyExit
        NativeExit = nativeExit; Failure = failure; ExpectedOutput = expected; ActualOutput = actual }
    File.WriteAllText(Path.Combine(work, "evidence.json"),
        JsonSerializer.Serialize({| compiler = compilerFiles; sample = sampleName; result = result |},
                                 JsonSerializerOptions(WriteIndented = true)))
    printfn "%s %s%s" (if result.Passed then "PASS" else "FAIL") sampleName (if result.Passed then "" else ": " + failure)
    printfn "Evidence: %s" work
    if result.Passed then 0 else 1
