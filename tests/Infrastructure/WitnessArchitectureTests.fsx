open System
open System.IO
open System.Diagnostics
open System.Xml.Linq

let root = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../.."))
let targets = Path.Combine(root, "build/WitnessArchitecture.targets")
let work = Path.Combine(Path.GetTempPath(), "composer-witness-architecture-" + Guid.NewGuid().ToString("N"))
Directory.CreateDirectory work |> ignore
let name value = XName.Get value
let attribute key value = XAttribute(name key, value)
let check (label: string) (relative: string) (source: string) (expected: string option) =
    let directory = Path.Combine(work, label)
    let path = Path.Combine(directory, relative)
    Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore
    File.WriteAllText(path, source)
    let project = Path.Combine(directory, "Probe.proj")
    XDocument(
        XElement(name "Project",
            XElement(name "Import", attribute "Project" targets),
            XElement(name "Target", attribute "Name" "Probe",
                XElement(name "ValidateWitnessArchitecture",
                    attribute "Sources" path, attribute "SourceRoot" directory))))
        .Save(project)
    let start = ProcessStartInfo("dotnet", UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true)
    for arg in ["msbuild"; project; "-nologo"; "-t:Probe"] do start.ArgumentList.Add arg
    use child = new Process(StartInfo = start)
    if not (child.Start()) then failwith "Could not start architecture check"
    let stdout, stderr = child.StandardOutput.ReadToEndAsync(), child.StandardError.ReadToEndAsync()
    if not (child.WaitForExit 30000) then
        child.Kill true
        child.WaitForExit()
        failwith (label + " timed out")
    let output = stdout.GetAwaiter().GetResult() + stderr.GetAwaiter().GetResult()
    match expected with
    | None when child.ExitCode = 0 -> ()
    | Some code when child.ExitCode <> 0 && output.Contains(code: string) -> ()
    | _ -> failwithf "%s failed its expected build outcome (%d):\n%s" label child.ExitCode output
    printfn "PASS %s" label

try
    check "direct-witness-accepted" "MiddleEnd/Alex/Direct.fs" "let declaration = FuncDecl(name, args, results)" None
    check "stock-backend-accepted" "BackEnd/Lowering.fs" "let args = [\"--pass-pipeline=builtin.module(convert-func-to-llvm)\"]" None
    check "plugin-load-rejected" "BackEnd/Lowering.fs" "let args = [\"--load-pass-plugin=retired.so\"]" (Some "ARCH001")
    check "plugin-discovery-rejected" "BackEnd/Lowering.fs" "let path = Environment.GetEnvironmentVariable(\"FIDELITY_MLIR_PLUGINS\")" (Some "ARCH001")
    check "middle-end-pass-rejected" "MiddleEnd/Generate.fs" "MLIRNanopass.applyPasses operations" (Some "ARCH002")
    check "middle-end-tool-rejected" "MiddleEnd/Generate.fs" "ProcessStartInfo(\"mlir-opt\")" (Some "ARCH002")
    check "retired-pipeline-path-rejected" "MiddleEnd/Alex/Pipeline/Repair.fs" "let repair operations = operations" (Some "ARCH002")
    printfn "All seven witness architecture build checks passed."
finally
    Directory.Delete(work, true)
