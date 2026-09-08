namespace Clef.Compiler.Editor

open System
open System.Diagnostics
open System.Threading
open System.Threading.Tasks

/// Evidence for the exact compiler-generated query, separate from its graph obligation.
type ProofResult = {
    State: string
    Detail: string
    Solver: string
    QueryHash: string
}

module ProofDispatch =
    /// Refutation-style source obligations only. This does not attest to lowering.
    let checkAsync (solver: string) (obligation: ObligationView) (cancellation: CancellationToken) = task {
        use solverProcess = new Process()
        solverProcess.StartInfo <- ProcessStartInfo(solver)
        solverProcess.StartInfo.UseShellExecute <- false
        solverProcess.StartInfo.RedirectStandardInput <- true
        solverProcess.StartInfo.RedirectStandardOutput <- true
        solverProcess.StartInfo.RedirectStandardError <- true
        solverProcess.StartInfo.ArgumentList.Add("--lang=smt2")
        solverProcess.StartInfo.ArgumentList.Add("--tlimit-per=2000")
        use limit = CancellationTokenSource.CreateLinkedTokenSource(cancellation)
        limit.CancelAfter(TimeSpan.FromSeconds 5.)
        let result state detail = { State = state; Detail = detail; Solver = solver; QueryHash = obligation.QueryHash }
        try
            cancellation.ThrowIfCancellationRequested()
            if not (solverProcess.Start()) then return result "error" "Solver process did not start."
            else
                let output = solverProcess.StandardOutput.ReadToEndAsync(limit.Token)
                let errors = solverProcess.StandardError.ReadToEndAsync(limit.Token)
                do! solverProcess.StandardInput.WriteLineAsync(obligation.SmtLib.AsMemory(), limit.Token)
                solverProcess.StandardInput.Close()
                do! solverProcess.WaitForExitAsync(limit.Token)
                let! stdout = output
                let! stderr = errors
                let answer = stdout.Trim()
                if solverProcess.ExitCode <> 0 then
                    return result "error" (sprintf "Solver exited %d: %s" solverProcess.ExitCode (stderr.Trim()))
                else
                    match answer with
                    | "unsat" -> return result "proved" "cvc5 returned unsat for the negated source obligation."
                    | "sat" -> return result "counterexample" "cvc5 returned sat: the source obligation does not hold under these premises."
                    | "unknown" -> return result "unknown" "cvc5 could not decide this query within the configured limit."
                    | _ -> return result "error" (sprintf "Unexpected solver response: %s %s" answer (stderr.Trim()))
        with
        | :? OperationCanceledException ->
            try if not solverProcess.HasExited then solverProcess.Kill(true) with _ -> ()
            cancellation.ThrowIfCancellationRequested()
            return result "unknown" "Solver execution exceeded the wall-clock limit."
        | ex ->
            try if not solverProcess.HasExited then solverProcess.Kill(true) with _ -> ()
            return result "error" ex.Message
    }
