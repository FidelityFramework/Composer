// BCL-only host for the existing native scalar-reference regression.
// This file neither builds Composer nor substitutes a host implementation for Clef.
open System
open System.Diagnostics
open System.IO
open System.Runtime.InteropServices
open System.Security.Cryptography
open System.Text
open System.Text.Json
open System.Text.RegularExpressions
open System.Threading
open System.Threading.Tasks

type Outcome = { Status: string; ExitCode: int option; ElapsedMs: int64; Detail: string }
type Options = { Compiler: string; Artifacts: string }

module private Linux =
    [<Struct; StructLayout(LayoutKind.Sequential)>]
    type Limit =
        val mutable Current: uint64
        val mutable Maximum: uint64
        new(current, maximum) = { Current = current; Maximum = maximum }

    [<DllImport("libc", SetLastError = true)>]
    extern int setrlimit(int resource, Limit& limit)

    [<DllImport("libc", SetLastError = true)>]
    extern int kill(int pid, int signal)

let usage = "Usage: dotnet fsi tests/ForeignScalarArrays/Run.fsx -- <private-Composer-path> [new-artifact-directory]"

let parseArguments arguments =
    match arguments with
    | ["--help"] | ["-h"] -> Ok None
    | [compiler] when not (compiler.StartsWith("-", StringComparison.Ordinal)) ->
        Ok(Some { Compiler = Path.GetFullPath compiler
                  Artifacts = Path.Combine(Path.GetTempPath(), "clef-scalar-arrays-" + Guid.NewGuid().ToString("N")) })
    | [compiler; artifacts] when not (compiler.StartsWith("-", StringComparison.Ordinal)) && not (artifacts.StartsWith("-", StringComparison.Ordinal)) ->
        Ok(Some { Compiler = Path.GetFullPath compiler; Artifacts = Path.GetFullPath artifacts })
    | _ -> Error usage

let private within (parent: string) (path: string) =
    let relative = Path.GetRelativePath(parent, path)
    relative = "." || (not (Path.IsPathRooted relative) && relative <> ".." && not (relative.StartsWith(".." + string Path.DirectorySeparatorChar, StringComparison.Ordinal)))

let validateOptions options =
    if not (OperatingSystem.IsLinux()) || RuntimeInformation.ProcessArchitecture <> Architecture.X64 then
        Error "This gate requires the declared Linux x86_64 native execution environment."
    elif not (File.Exists options.Compiler) then Error ("Missing compiler snapshot: " + options.Compiler)
    elif Directory.Exists options.Artifacts || File.Exists options.Artifacts then Error ("Artifact directory must be new: " + options.Artifacts)
    elif within (Path.GetDirectoryName options.Compiler) options.Artifacts || within __SOURCE_DIRECTORY__ options.Artifacts then
        Error "Artifacts must be outside the compiler snapshot and checked-in fixture directory."
    else Ok ()

let disableCoreFiles () =
    let mutable limit = Linux.Limit(0UL, 0UL)
    if Linux.setrlimit(4, &limit) <> 0 then
        failwithf "Cannot disable core dumps: errno %d" (Marshal.GetLastPInvokeError())

let private writeJson path value =
    File.WriteAllText(path, JsonSerializer.Serialize(value, JsonSerializerOptions(WriteIndented = true)))

/// Direct arguments, independent raw streams, EOF input and one deadline for
/// process exit plus both pipes. ProcessHost gives every job a Linux process group.
let runProcess host directory stage command (arguments: string list) (environment: (string * string) list) timeoutMs = task {
    Directory.CreateDirectory directory |> ignore
    let ready = Path.Combine(directory, stage + ".ready")
    let executable, args =
        match host with
        | Some assembly -> "dotnet", [assembly; "--ready"; ready; "--"; command] @ arguments
        | None -> command, arguments
    let start = ProcessStartInfo(executable, UseShellExecute = false, WorkingDirectory = directory,
                                RedirectStandardInput = true, RedirectStandardOutput = true, RedirectStandardError = true)
    for argument in args do start.ArgumentList.Add argument
    for name, value in environment do start.Environment[name] <- value
    use process = new Process(StartInfo = start)
    use cancellation = new CancellationTokenSource()
    use stdout = File.Create(Path.Combine(directory, stage + ".stdout.log"))
    use stderr = File.Create(Path.Combine(directory, stage + ".stderr.log"))
    let watch = Stopwatch.StartNew()
    let terminate () =
        if host.IsSome && OperatingSystem.IsLinux() && File.Exists ready then
            match Int32.TryParse(File.ReadAllText ready) with
            | true, pid when pid = process.Id -> Linux.kill(-pid, 9) |> ignore
            | _ -> ()
        try if not process.HasExited then process.Kill(true) with :? InvalidOperationException -> ()
    let! outcome = task {
        try
            if not (process.Start()) then failwith ("Could not start " + executable)
            process.StandardInput.Close()
            let finished = Task.WhenAll [|
                process.StandardOutput.BaseStream.CopyToAsync(stdout, cancellation.Token)
                process.StandardError.BaseStream.CopyToAsync(stderr, cancellation.Token)
                process.WaitForExitAsync(cancellation.Token) |]
            try
                do! finished.WaitAsync(TimeSpan.FromMilliseconds(float timeoutMs))
                terminate ()
                if host.IsSome && File.Exists(ready + ".error") then
                    return { Status = "failed"; ExitCode = None; ElapsedMs = watch.ElapsedMilliseconds; Detail = File.ReadAllText(ready + ".error") }
                else
                    return { Status = "completed"; ExitCode = Some process.ExitCode; ElapsedMs = watch.ElapsedMilliseconds; Detail = "" }
            with ex ->
                terminate ()
                cancellation.Cancel()
                try do! finished.WaitAsync(TimeSpan.FromSeconds 5.) with _ -> ()
                try do! process.WaitForExitAsync().WaitAsync(TimeSpan.FromSeconds 5.) with _ -> ()
                return { Status = (if ex :? TimeoutException then "timeout" else "failed")
                         ExitCode = None; ElapsedMs = watch.ElapsedMilliseconds; Detail = ex.Message }
        with ex ->
            return { Status = "failed"; ExitCode = None; ElapsedMs = watch.ElapsedMilliseconds; Detail = ex.Message }
    }
    stdout.Flush()
    stderr.Flush()
    writeJson (Path.Combine(directory, stage + ".json"))
        {| Command = command; Arguments = arguments; Environment = environment; TimeoutMs = timeoutMs
           Status = outcome.Status; ExitCode = outcome.ExitCode |> Option.toNullable
           ElapsedMs = outcome.ElapsedMs; Detail = outcome.Detail |}
    return outcome
}

let prepareHost root = task {
    let project = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../Infrastructure/ProcessHost/ProcessHost.fsproj"))
    let output = Path.Combine(root, "process-host")
    let! result = runProcess None root "host-build" "dotnet"
                      ["build"; project; "--artifacts-path"; Path.Combine(root, "host-build");
                       "--output"; output; "--disable-build-servers"] [] 120000
    if result.Status <> "completed" || result.ExitCode <> Some 0 then
        failwithf "Private ProcessHost build failed (%s); see %s" result.Status root
    return Path.Combine(output, "ProcessHost.dll")
}

let positiveRunErrors outcome =
    if outcome.Status = "completed" && outcome.ExitCode = Some 0 then []
    else [sprintf "Positive executable did not exit zero: %A" outcome]

let negativeRunErrors outcome (stdout: byte array) =
    // Python returns -SIGABRT/-SIGILL; .NET Unix ExitCode encodes 128+signal.
    let exitAccepted = outcome.Status = "completed" && List.contains outcome.ExitCode [Some 134; Some 132]
    [ if not exitAccepted then yield sprintf "Invalid byte did not terminate with SIGABRT/SIGILL (134/132): %A" outcome
      if stdout.Length <> 0 then yield "Invalid input reached write before rejection (stdout is nonempty)." ]

let positiveIrErrors (ir: string) =
    [ for required in ["memref<?xi64>"; "memref<?xi8>"; "memref<?xi32>"; "arith.extui"; "arith.extsi"; "memref.dealloc"] do
          if not (ir.Contains(required, StringComparison.Ordinal)) then
              yield "Missing native scalar projection evidence: " + required
      if not (Regex.IsMatch(ir, @"func.call @write\([^\n]*\n\s+memref.dealloc")) then
          yield "Readonly input unexpectedly has copyback operations." ]

let negativeIrErrors (ir: string) =
    if ir.Contains("Foreign reference bytes element is outside its declared range", StringComparison.Ordinal) then []
    else ["Missing declared byte-range rejection diagnostic in native IR."]

let private hash path =
    use input = File.OpenRead path
    Convert.ToHexString(SHA256.HashData input)

let private inventory paths = paths |> List.distinct |> List.sort |> List.map (fun path -> path, hash path)

let compilerInventory compiler =
    Directory.EnumerateFiles(Path.GetDirectoryName compiler, "*", SearchOption.AllDirectories)
    |> Seq.toList |> inventory

let sourceInventory () =
    let repos = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../../.."))
    let dependencyFiles =
        ["Fidelity.Platform"; "Fidelity.Pthread"; "BAREWire"]
        |> List.collect (fun name ->
            let directory = Path.Combine(repos, name)
            if Directory.Exists directory then
                Directory.EnumerateFiles(directory, "*", SearchOption.AllDirectories)
                |> Seq.filter (fun path -> [".clef"; ".fidproj"] |> List.contains (Path.GetExtension path))
                |> Seq.filter (fun path ->
                    let parts = Path.GetRelativePath(directory, path).Split(Path.DirectorySeparatorChar)
                    not (parts |> Array.exists (fun part -> ["bin"; "obj"; "targets"; ".git"] |> List.contains part)))
                |> Seq.toList
            else [])
    dependencyFiles @ [ for name in ["Bindings.clef"; "Main.clef"; "InvalidByte.clef"; "ForeignScalarArrays.fidproj"] -> Path.Combine(__SOURCE_DIRECTORY__, name) ]
    |> inventory

let stageCase root name sourceFile =
    let directory = Path.Combine(root, name)
    Directory.CreateDirectory directory |> ignore
    File.Copy(Path.Combine(__SOURCE_DIRECTORY__, "Bindings.clef"), Path.Combine(directory, "Bindings.clef"))
    File.Copy(Path.Combine(__SOURCE_DIRECTORY__, sourceFile), Path.Combine(directory, "Main.clef"))
    let repos = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../../.."))
    let project =
        ["Fidelity.Platform.CompilerSurface"; "Fidelity.Pthread"]
        |> List.fold (fun (text: string) dependency ->
            let relative = "Fidelity.Platform/Environments/Linux/x86_64/" + dependency + ".fidproj"
            let original = JsonSerializer.Serialize("../../../" + relative)
            if not (text.Contains(original, StringComparison.Ordinal)) then
                failwith ("Declared fixture dependency was not found: " + relative)
            text.Replace(original, JsonSerializer.Serialize(Path.Combine(repos, relative)), StringComparison.Ordinal))
            (File.ReadAllText(Path.Combine(__SOURCE_DIRECTORY__, "ForeignScalarArrays.fidproj")))
    let path = Path.Combine(directory, "ForeignScalarArrays.fidproj")
    File.WriteAllText(path, project)
    directory, path

let run options = task {
    match validateOptions options with
    | Error message -> return failwith message
    | Ok () -> ()
    Directory.CreateDirectory options.Artifacts |> ignore
    printfn "Artifacts: %s" options.Artifacts
    disableCoreFiles ()
    let compilerBefore, sourceBefore = compilerInventory options.Compiler, sourceInventory ()
    writeJson (Path.Combine(options.Artifacts, "compiler.before.json")) compilerBefore
    writeJson (Path.Combine(options.Artifacts, "inputs.before.json")) sourceBefore
    writeJson (Path.Combine(options.Artifacts, "run.json"))
        {| Compiler = options.Compiler; CompilerRebuiltByThisRun = false; SourceDirectory = __SOURCE_DIRECTORY__
           CompileTimeoutMs = 600000; RuntimeTimeoutMs = 10000; Perturbations = ["0"; "165"]
           ExpectedNegativeExitCodes = [134; 132]; CoreFilesDisabled = true |}
    let failures = ResizeArray<string>()
    try
        let! host = prepareHost options.Artifacts
        for name, sourceFile in ["positive", "Main.clef"; "invalid-byte", "InvalidByte.clef"] do
            let directory, project = stageCase options.Artifacts name sourceFile
            let binary = Path.Combine(directory, "scalar-arrays")
            let artifacts = Path.Combine(directory, "artifacts")
            let! compiled = runProcess (Some host) directory "compile" options.Compiler
                                ["compile"; project; "-o"; binary; "-k"; "--no-color"; "--artifacts-dir"; artifacts] [] 600000
            if compiled.Status <> "completed" || compiled.ExitCode <> Some 0 then
                failures.Add(sprintf "%s compile failed (%s/%A); see %s" name compiled.Status compiled.ExitCode directory)
            else
                let ir = File.ReadAllText(Path.Combine(artifacts, "intermediates/10_output.mlir"))
                let irErrors = if name = "positive" then positiveIrErrors ir else negativeIrErrors ir
                failures.AddRange(irErrors |> List.map (fun message -> name + ": " + message))
                let environments = if name = "positive" then ["0"; "165"] else [""]
                for perturb in environments do
                    let stage = if perturb = "" then "run" else "run-" + perturb
                    let environment = if perturb = "" then [] else ["MALLOC_PERTURB_", perturb]
                    let! outcome = runProcess (Some host) directory stage binary [] environment 10000
                    let errors =
                        if name = "positive" then positiveRunErrors outcome
                        else negativeRunErrors outcome (File.ReadAllBytes(Path.Combine(directory, stage + ".stdout.log")))
                    failures.AddRange(errors |> List.map (fun message -> name + "/" + stage + ": " + message))
                    printfn "%s/%s %s exit=%A elapsed=%dms" name stage (if errors.IsEmpty then "PASS" else "FAIL") outcome.ExitCode outcome.ElapsedMs
    with ex -> failures.Add(ex.ToString())
    let compilerAfter, sourceAfter = compilerInventory options.Compiler, sourceInventory ()
    writeJson (Path.Combine(options.Artifacts, "compiler.after.json")) compilerAfter
    writeJson (Path.Combine(options.Artifacts, "inputs.after.json")) sourceAfter
    if compilerAfter <> compilerBefore then failures.Add "Compiler snapshot changed during the run."
    if sourceAfter <> sourceBefore then failures.Add "Fixture or dependency inputs changed during the run."
    writeJson (Path.Combine(options.Artifacts, "result.json")) {| Passed = failures.Count = 0; Failures = failures.ToArray() |}
    if failures.Count <> 0 then
        for failure in failures do eprintfn "%s" failure
        return 1
    else
        printfn "PASS both scalar-reference gates; artifacts: %s" options.Artifacts
        return 0
}
