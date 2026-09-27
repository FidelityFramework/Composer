/// Fresh, bounded external proof checks for the current compile invocation.
/// Source and MLIR checks remain distinct solver-dependent stages. Neither
/// receipt substitutes for the generated-artifact Rocq check.
module Core.ProofDispatch

open System
open System.Diagnostics
open System.IO
open System.Security.Cryptography
open System.Text
open System.Text.Json
open System.Text.RegularExpressions
open System.Threading.Tasks
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Core.Types.WitnessArtifacts

let private hash (text: string) = SHA256.HashData(Encoding.UTF8.GetBytes text) |> Convert.ToHexString

let private fileHash path =
    use stream = File.OpenRead path
    SHA256.HashData stream |> Convert.ToHexString

let private anchors (obligations: ObligationInfo list) =
    obligations |> List.map (fun ob -> sprintf "'%s' (%s)" ob.Id ob.Source) |> String.concat ", "

let private resolveTool name =
    let paths = Environment.GetEnvironmentVariable "PATH" |> Option.ofObj |> Option.defaultValue ""
    paths.Split(Path.PathSeparator)
    |> Array.map (fun directory -> Path.Combine(directory, name))
    |> Array.tryFind File.Exists
    |> function
       | None -> Error (sprintf "Required proof tool '%s' was not found on PATH" name)
       | Some path ->
           let file = FileInfo path
           let target = file.ResolveLinkTarget true
           Ok (if isNull target then file.FullName else target.FullName)

/// stdin, stdout and stderr flow concurrently. One process has at most thirty
/// seconds, including query input; a timeout kills its entire process tree.
let private run executable arguments input =
    try
        let start = ProcessStartInfo(executable)
        start.UseShellExecute <- false
        start.RedirectStandardInput <- true
        start.RedirectStandardOutput <- true
        start.RedirectStandardError <- true
        for argument in arguments do start.ArgumentList.Add argument
        use child = new Process(StartInfo = start)
        if not (child.Start()) then Error (sprintf "Could not start proof tool %s" executable)
        else
            let output = child.StandardOutput.ReadToEndAsync()
            let errors = child.StandardError.ReadToEndAsync()
            let writer = task {
                do! child.StandardInput.WriteAsync(input: string)
                child.StandardInput.Close()
            }
            if not (child.WaitForExit 30000) then
                child.Kill true
                child.WaitForExit 5000 |> ignore
                Error (sprintf "Proof tool %s timed out after 30 seconds" executable)
            elif not (Task.WhenAll([| output :> Task; errors :> Task; writer :> Task |]).Wait 5000) then
                Error (sprintf "Proof tool %s did not complete redirected I/O" executable)
            else
                let stdout, stderr = output.Result, errors.Result
                if child.ExitCode <> 0 then
                    Error (sprintf "Proof tool %s exited %d: %s%s" executable child.ExitCode stderr stdout)
                else Ok (stdout, stderr)
    with error -> Error (sprintf "Proof tool %s failed: %s" executable error.Message)

let private identify name arguments =
    resolveTool name
    |> Result.bind (fun executable ->
        let before = fileHash executable
        run executable ["--version"] ""
        |> Result.bind (fun (version, _) ->
            if String.IsNullOrWhiteSpace version || fileHash executable <> before then
                Error (sprintf "Proof tool '%s' has no stable executable/version identity" name)
            else Ok { Executable = executable; Sha256 = before; Version = version.Trim(); Arguments = arguments }))

let private runIdentified (tool: ProofTool) input =
    if fileHash tool.Executable <> tool.Sha256 then Error (sprintf "Proof tool changed before execution: %s" tool.Executable)
    else
        run tool.Executable tool.Arguments input
        |> Result.bind (fun result ->
            if fileHash tool.Executable <> tool.Sha256 then Error (sprintf "Proof tool changed during execution: %s" tool.Executable)
            else Ok result)

let private write artifacts name (text: string) =
    match artifacts with
    | None -> ()
    | Some directory ->
        Directory.CreateDirectory directory |> ignore
        File.WriteAllText(Path.Combine(directory, name), text)

let private writeEvidence artifacts name (evidence: ProofStageEvidence) =
    write artifacts name (JsonSerializer.Serialize(evidence, JsonSerializerOptions(WriteIndented = true)))

/// Reject missing/duplicate/reordered query anchors and check commands before
/// dispatch. This validates physical correspondence, not the predicate itself.
let private validateQuery (obligations: ObligationInfo list) (query: string) =
    let withoutComments = Regex.Replace(query, @";[^\r\n]*", "")
    let checks = Regex.Matches(withoutComments, @"\(\s*check-sat\s*\)").Count
    let matches =
        obligations |> List.map (fun ob ->
            let symbol = Regex.Escape ob.Id
            ob, Regex.Matches(withoutComments, @"\(\s*declare-(?:fun|const)\s+(?:\|" + symbol + @"\||" + symbol + @")(?=\s|\))"))
    match matches |> List.tryFind (fun (_, found) -> found.Count <> 1) with
    | Some(ob, _) -> Error (sprintf "Proof query does not retain exactly one declaration of obligation '%s' (%s)" ob.Id ob.Source)
    | None when checks <> obligations.Length -> Error (sprintf "Proof query contains %d checks for %d required obligations: %s" checks obligations.Length (anchors obligations))
    | None ->
        let positions = matches |> List.map (fun (_, found) -> found[0].Index)
        if positions <> List.sort positions then Error "Proof query changed the ordered obligation anchor inventory"
        else Ok ()

let private outcomes (obligations: ObligationInfo list) (stdout: string) =
    let verdicts = stdout.Split([|'\r'; '\n'|], StringSplitOptions.RemoveEmptyEntries) |> Array.map _.Trim() |> Array.toList
    if verdicts.Length <> obligations.Length then
        Error (sprintf "Proof solver returned %d outcomes for %d obligations (%s): %s" verdicts.Length obligations.Length (anchors obligations) stdout)
    else
        List.zip obligations verdicts
        |> List.tryPick (fun (ob, verdict) ->
            if verdict = "unsat" then None
            else Some (sprintf "Required obligation '%s' (%s) was not discharged: %s" ob.Id ob.Source verdict))
        |> function
           | Some reason -> Error reason
           | None -> Ok (List.zip obligations verdicts |> List.map (fun (ob, verdict) -> { Anchor = ob.Id; Source = ob.Source; Verdict = verdict }))

let private solve invocation stage inputHash tools obligations query artifacts prefix =
    validateQuery obligations query
    |> Result.bind (fun () ->
        let check =
            if List.isEmpty obligations then Ok ([], "", "", tools)
            else
                identify "cvc5" ["--lang=smt2"; "--tlimit=25000"]
                |> Result.bind (fun solver ->
                    runIdentified solver query
                    |> Result.bind (fun (stdout, stderr) ->
                        write artifacts (prefix + ".stdout") stdout
                        write artifacts (prefix + ".stderr") stderr
                        outcomes obligations stdout |> Result.map (fun results -> results, stdout, stderr, tools @ [solver])))
        check
        |> Result.mapError (fun reason -> sprintf "%s proof stage (%s): %s" stage (anchors obligations) reason)
        |> Result.map (fun (results, stdout, stderr, checkedTools) ->
            let evidence =
                { Invocation = invocation; Stage = stage; InputSha256 = inputHash; QuerySha256 = hash query
                  Tools = checkedTools; Outcomes = results; StandardOutput = stdout; StandardError = stderr }
            writeEvidence artifacts (prefix + ".json") evidence
            evidence))

/// Called by core orchestration after source admission and before Alex. This
/// dispatches the source-owned SMT-LIB projection; it performs no settlement.
let dischargeSource (graph: SemanticGraph) artifacts : Result<SourceProofReceipt, string> =
    try
        let obligations = Clef.Compiler.Nanopass.ObligationDischarge.ofGraph graph
        let query = Clef.Compiler.Nanopass.ObligationDischarge.smtLib obligations
        let invocation = Guid.NewGuid()
        write artifacts "06b_obligations.smt2" query
        if (obligations |> List.map _.Id |> Set.ofList).Count <> obligations.Length then Error "PSG proof stage has duplicate obligation anchors"
        else
            solve invocation "psg" (hash query) [] obligations query artifacts "06c_psg_discharge"
            |> Result.map (fun evidence -> SourceProofReceipt(graph, obligations, query, evidence))
    with error -> Error ("PSG proof dispatch failed: " + error.Message)

/// Called by common backend admission before any target realization. Every
/// invocation exports and solves the exact witnessed module again; no reuse.
let dischargeMlir (proof: ProofEnvelope) artifacts : Result<MlirProofReceipt, string> =
    try
        Core.WitnessArtifacts.validateProof proof.Scope proof
        |> Result.bind (fun () ->
            match proof.Source with
            | None -> Error "MLIR proof dispatch requires the current invocation's PSG discharge receipt"
            | Some source ->
                write artifacts "09_obligations.mlir" proof.Text
                let exported =
                    if proof.Obligations.IsEmpty then Ok ("", [])
                    else
                        identify "mlir-translate" ["--export-smtlib"]
                        |> Result.bind (fun translator -> runIdentified translator proof.Text |> Result.map (fun (query, _) -> query, [translator]))
                exported
                |> Result.mapError (fun reason -> sprintf "MLIR proof export (%s): %s" (anchors proof.Obligations) reason)
                |> Result.bind (fun (query, translators) ->
                    write artifacts "09_obligations.smt2" query
                    solve source.Evidence.Invocation "mlir" (hash proof.Text) translators proof.Obligations query artifacts "09_mlir_discharge"
                    |> Result.map (fun evidence -> MlirProofReceipt(proof.Scope, source, proof.Operations, proof.Text, query, evidence))))
    with error -> Error ("MLIR proof dispatch failed: " + error.Message)
