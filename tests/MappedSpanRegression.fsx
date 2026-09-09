// Build Composer, then: dotnet fsi tests/MappedSpanRegression.fsx
// Both solver dispatches and deliberately weakened runtime premises. The
// theorem starts AFTER checked byte-extent construction; it is not a proof of
// driver allocation provenance, multiplication lowering, or carrier lifetime.
#I "../src/bin/Debug/net10.0"
#r "Fidelity.Data.dll"
#r "Clef.Compiler.Service.dll"
#r "Composer.dll"
open System.Diagnostics
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

let run (tool:string) (args:string) (input:string) =
    let info = ProcessStartInfo(tool, args)
    info.RedirectStandardInput <- true
    info.RedirectStandardOutput <- true
    info.RedirectStandardError <- true
    use child = Process.Start info
    child.StandardInput.Write input
    child.StandardInput.Close()
    let output = child.StandardOutput.ReadToEnd()
    let errors = child.StandardError.ReadToEnd()
    child.WaitForExit()
    if child.ExitCode <> 0 then failwithf "%s failed: %s" tool errors
    output.Trim()
let solver input = run "cvc5" "--lang=smt2 --tlimit=5000" input
let check expected input =
    let actual = solver input
    if actual <> expected then failwithf "Expected %s; got %s\n%s" expected actual input

for bits, bytes, alignment in [32,1,1; 32,4,4; 64,4,4; 64,4,32; 64,8,8] do
    let model = {PointerBits=bits;MaximumExtent=(1I <<< (bits-1))-1I;ElementBytes=bytes;BaseAlignment=alignment
                 ElementAlignment=int(System.Numerics.BigInteger.GreatestCommonDivisor(bigint bytes,bigint alignment))}
    let ob = {Id="mapped_span_check";Kind="mapped-element-span";Logic="QF_LIA";Statement="Conditional mapped index span";Source="regression";Refs=[];Body=ObligationBody.MappedElementSpan model}
    let source = Clef.Compiler.Nanopass.ObligationDischarge.smtLib [ob]
    let mlir = Alex.Traversal.SMTTransfer.transfer [ob]
    let native = run "mlir-translate" "--export-smtlib" mlir
    if not (native.Contains ob.Id) then failwith "Mapped proof anchor lost in native dispatch."
    check "unsat" source
    check "unsat" native
    let weakIndex = source.Replace(sprintf "(assert (< mapped_index (div mapped_bytes %d)))" bytes, "")
    let weakEndpoint = source.Replace(sprintf "(assert (<= mapped_base (- %A mapped_bytes)))" model.MaximumExtent, "")
    check "sat" weakIndex
    check "sat" weakEndpoint
    if model.ElementAlignment > 1 then
        source.Replace(sprintf "(assert (= (mod mapped_base %d) 0))" alignment, "") |> check "sat"
printfn "PASS mapped span: 5 source/native UNSAT pairs and 14 weakened-guard SAT counterexamples (5-second solver limit)."
