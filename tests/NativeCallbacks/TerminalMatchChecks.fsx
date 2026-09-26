// Compile and execute the terminal match contract using an already built,
// isolated Composer snapshot. No shared compiler build is performed here.
#load "../regression/RunnerCore.fsx"

open System
open System.IO
open System.Runtime.InteropServices
open System.Text.Json
open System.Threading.Tasks
open RunnerCore

module CoreFiles =
    [<Struct; StructLayout(LayoutKind.Sequential)>]
    type Limit =
        val mutable Current: uint64
        val mutable Maximum: uint64
        new(current, maximum) = { Current = current; Maximum = maximum }

    [<DllImport("libc", SetLastError = true)>]
    extern int setrlimit(int resource, Limit& limit)

    let disable () =
        if OperatingSystem.IsLinux() then
            let mutable limit = Limit(0UL, 0UL)
            if setrlimit(4, &limit) <> 0 then
                failwithf "Cannot disable core dumps for intentional assertion failures: errno %d" (Marshal.GetLastPInvokeError())

let arguments = fsi.CommandLineArgs |> Array.skip 1 |> Array.filter ((<>) "--")
if arguments.Length <> 2 && not (arguments.Length = 3 && List.contains arguments[2] ["--positive"; "--all"]) then
    failwith "Usage: dotnet fsi TerminalMatchChecks.fsx -- <private-Composer-path> <new-artifact-directory> [--positive|--all]"
let mode = if arguments.Length = 3 then arguments[2] else "--terminal"
let compiler, root = Path.GetFullPath arguments[0], Path.GetFullPath arguments[1]
if not (File.Exists compiler) then failwithf "Missing compiler snapshot: %s" compiler
if Directory.Exists root then failwithf "Artifact directory must be new: %s" root
Directory.CreateDirectory root |> ignore
CoreFiles.disable ()

let config =
    { SamplesRoot = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, ".."))
      CompilerPath = compiler; DefaultTimeoutSeconds = 180; PruneIntermediates = false }
let cases =
    let terminal =
        [ "TerminalMatchSuccess", "terminal-match-success", false
          "TerminalMatchPatternFailure", "terminal-match-pattern-failure", true
          "TerminalMatchGuardFailure", "terminal-match-guard-failure", true ]
    if mode = "--positive" then
        [ "GuardedMatch", "guarded-match", false
          "TerminalMatchSuccess", "terminal-match-success", false ]
    elif mode = "--all" then
        [ "GuardedMatch", "guarded-match", false
          "LiteralMatch", "literal-match", false ] @ terminal
    else
        terminal
let samples =
    cases |> List.map (fun (name, binary, _) ->
        { Name = "NativeCallbacks"; ProjectFile = name + ".fidproj"; BinaryName = binary
          StdinFile = None; ExpectedOutput = ""; TimeoutSeconds = 180; Skip = false; SkipReason = None })
File.WriteAllText(Path.Combine(root, "run.json"), JsonSerializer.Serialize(
    {| Compiler = compiler; SourceDirectory = __SOURCE_DIRECTORY__; CompilerRebuiltByThisRun = false
       FullIntermediates = true; CoreFilesDisabled = OperatingSystem.IsLinux(); Selection = cases |},
    JsonSerializerOptions(WriteIndented = true)))

// Keep a compile/run phase barrier. Every invocation has distinct artifacts and
// the native failure controls use the existing bounded .NET ProcessHost.
let compiled =
    samples |> List.mapi (fun index sample ->
        let directory = Path.Combine(root, sprintf "%04d" (index + 1))
        Directory.CreateDirectory directory |> ignore
        compileSamplePhaseAsync config (Some directory) sample)
    |> Task.WhenAll |> fun task -> task.GetAwaiter().GetResult()

let mutable failures = 0
for index in 0 .. compiled.Length - 1 do
    let name, _, expectsFailure = cases[index]
    let directory = Path.Combine(root, sprintf "%04d" (index + 1))
    let sample, compile, binary = compiled[index]
    match compile, binary with
    | CompileSuccess compileMs, Some binary ->
        let outcome, runMs = runProcess binary [] __SOURCE_DIRECTORY__ None 10000
        let status =
            match outcome with
            | Completed(code, stdout, stderr) ->
                File.WriteAllText(Path.Combine(directory, "run.stdout.log"), stdout)
                File.WriteAllText(Path.Combine(directory, "run.stderr.log"), stderr)
                let expectedDiagnostic = sprintf "Pattern match failed at %s:4:10" (Path.Combine(__SOURCE_DIRECTORY__, name + ".clef"))
                let passed =
                    if expectsFailure then
                        code <> 0 && code < 200 && stdout = "" && stderr.TrimEnd('\r', '\n') = expectedDiagnostic
                    else code = 0 && stdout = "" && stderr = ""
                sprintf "%s exit=%d compile_ms=%d run_ms=%d" (if passed then "PASS" else "FAIL") code compileMs runMs, passed
            | Timeout timeout -> sprintf "FAIL timeout_ms=%d" timeout, false
            | Failed error -> sprintf "FAIL %s" (error.ToString()), false
        File.WriteAllText(Path.Combine(directory, "run.status"), fst status + "\n")
        printfn "%s %s" name (fst status)
        if not (snd status) then failures <- failures + 1
    | _ ->
        failures <- failures + 1
        printfn "%s FAIL compilation; see %s" sample.ProjectFile (Path.Combine(directory, "compile.stdout.log"))
if failures <> 0 then failwithf "%d terminal match control(s) failed; artifacts: %s" failures root
printfn "All %d selected match controls passed; artifacts: %s" cases.Length root
