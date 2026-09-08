// Build src/Composer.fsproj, then run: dotnet fsi tests/SMTTransferRegression.fsx
// Requires mlir-translate with SMT export support and cvc5 on PATH.
// Exercises actual source and native solver dispatch, including deliberately false claims.
#I "../src/bin/Debug/net10.0"
#r "Fidelity.Data.dll"
#r "Clef.Compiler.Service.dll"
#r "Composer.dll"
open System
open System.IO
open System.Diagnostics
open Clef.Compiler.NativeTypedTree.DimensionAlgebra
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
let m = Dimension.ofBase { Name = "m"; Module = ["Units"] }
let s = Dimension.ofBase { Name = "s"; Module = ["Units"] }
let u = Dimension.ofVar { Id = 501; Name = Some "u" }
let v = Dimension.ofVar { Id = 502; Name = Some "v" }
let quotient a b = Dimension.mul a (Dimension.inv b)
let cases = [
    DimensionalRule.Quotient, m, s, Some (quotient m s), "unsat"
    DimensionalRule.Quotient, m, s, Some m, "sat"
    DimensionalRule.Product, m, s, Some (Dimension.mul m s), "unsat"
    DimensionalRule.SameDimension, m, m, Some m, "unsat"
    DimensionalRule.SameDimension, m, s, Some m, "sat"
    DimensionalRule.Comparison, m, m, None, "unsat"
    DimensionalRule.Comparison, m, s, None, "sat"
    DimensionalRule.Quotient, u, v, Some (quotient u v), "unsat"
    DimensionalRule.Quotient, u, v, Some (quotient v u), "sat"
    DimensionalRule.Quotient, Dimension.one, Dimension.one, Some Dimension.one, "unsat"
    DimensionalRule.Quotient, Dimension.one, Dimension.one, None, "sat"
]
let run (tool: string) (args: string) input =
    let info = ProcessStartInfo(tool, args)
    info.RedirectStandardInput <- true
    info.RedirectStandardOutput <- true
    info.RedirectStandardError <- true
    use child = Process.Start info
    child.StandardInput.Write(input: string)
    child.StandardInput.Close()
    let output = child.StandardOutput.ReadToEnd()
    let errors = child.StandardError.ReadToEnd()
    child.WaitForExit()
    if child.ExitCode <> 0 then failwithf "%s failed: %s" tool errors
    output.Trim()
for rule, left, right, result, expected in cases do
    let ob = { Id = "dimension_check"; Kind = "dimension-regression"; Logic = "QF_LIA"; Statement = "transfer regression"
               Source = "test"; Refs = []; Body = ObligationBody.DimensionalRelation(rule,left,right,result) }
    let mlir = Alex.Traversal.SMTTransfer.transfer [ob]
    let smt = run "mlir-translate" "--export-smtlib" mlir
    let actual = run "cvc5" "--lang=smt2" smt
    if actual <> expected then failwithf "%A: expected %s got %s" ob.Body expected actual
printfn "PASS %d native SMT transfer cases through mlir-translate and cvc5" cases.Length
let rational text =
    match Clef.Compiler.NativeTypedTree.NativeTypes.ExactRational.tryParseDecimal text with
    | Ok value -> value
    | Error message -> failwith message
let q, zero, one = rational "0.1", rational "0", rational "1"
let realCases = [
    ObligationBody.RealLiteralRange(q,q,q), "unsat"
    ObligationBody.RealLiteralRange(q,one,one), "sat"
    ObligationBody.RealRepresentationCoverage(q,q,zero,one), "unsat"
    ObligationBody.RealRepresentationCoverage(one,one,zero,q), "sat"
    ObligationBody.RealLiteralRange(rational "1e-400", rational "1e-400", rational "1e-400"), "unsat"
    ObligationBody.RealRepresentationCoverage(rational "1e400", rational "1e400", rational "-1e308", rational "1e308"), "sat"
]
for body, expected in realCases do
    let ob = { Id = "real_check"; Kind = "real-regression"; Logic = "QF_LRA"; Statement = "transfer regression"
               Source = "test"; Refs = []; Body = body }
    let source = run "cvc5" "--lang=smt2" (Clef.Compiler.Nanopass.ObligationDischarge.smtLib [ob])
    let smt = run "mlir-translate" "--export-smtlib" (Alex.Traversal.SMTTransfer.transfer [ob])
    let native = run "cvc5" "--lang=smt2" smt
    if source <> expected || native <> expected then failwithf "Expected %s, source=%s native=%s" expected source native
printfn "PASS %d real-bound source/native parity cases" realCases.Length
let unsigned64Max = 18446744073709551615I
let signed64Min = -9223372036854775808I
let signed64Max = 9223372036854775807I
let integerCases = [
    ObligationBody.IntegerLiteralRange(12I,12I,12I), "unsat"
    ObligationBody.IntegerLiteralRange(-12I,-12I,-12I), "unsat"
    ObligationBody.IntegerLiteralRange(signed64Min - 1I,signed64Min - 1I,signed64Min - 1I), "unsat"
    ObligationBody.IntegerLiteralRange(unsigned64Max,unsigned64Max,unsigned64Max), "unsat"
    ObligationBody.IntegerLiteralRange(unsigned64Max + 1I,0I,0I), "sat" // Detect an accidental UInt64 wrap.
    ObligationBody.IntegerLiteralRange(0I,1I,-1I), "sat"
    ObligationBody.IntegerRepresentationCoverage(-32I,-1I,-128I,127I), "unsat"
    ObligationBody.IntegerRepresentationCoverage(unsigned64Max,unsigned64Max,0I,unsigned64Max), "unsat"
    ObligationBody.IntegerRepresentationCoverage(unsigned64Max,unsigned64Max,signed64Min,signed64Max), "sat"
    ObligationBody.IntegerRepresentationCoverage(signed64Min - 1I,signed64Min - 1I,signed64Min,signed64Max), "sat"
    ObligationBody.IntegerRepresentationCoverage(1I,0I,0I,1I), "sat"
]
for index, (body, expected) in List.indexed integerCases do
    let ob = { Id = sprintf "integer_check_%d" index; Kind = "integer-regression"; Logic = "QF_LIA"
               Statement = "integer transfer regression"; Source = "test"; Refs = []; Body = body }
    let source = run "cvc5" "--lang=smt2" (Clef.Compiler.Nanopass.ObligationDischarge.smtLib [ob])
    let smt = run "mlir-translate" "--export-smtlib" (Alex.Traversal.SMTTransfer.transfer [ob])
    if not (smt.Contains(ob.Id)) then failwithf "Integer anchor %s did not survive transfer" ob.Id
    let native = run "cvc5" "--lang=smt2" smt
    if source <> expected || native <> expected then
        failwithf "Integer %d: expected %s, source=%s native=%s" index expected source native
printfn "PASS %d integer source/native parity cases" integerCases.Length
let applicationCases = [
    ["argument", Some m, Some m; "result", Some (quotient m s), Some (quotient m s)], "unsat"
    [("argument", Some m, Some s)], "sat"
    [("argument", Some (quotient u v), Some (quotient u v))], "unsat"
    [("argument", Some (quotient u v), Some (quotient v u))], "sat"
    [("argument", Some Dimension.one, Some Dimension.one)], "unsat"
    [("argument", None, Some Dimension.one)], "sat"
    [("argument", Some Dimension.one, None)], "sat"
    [("argument", None, None)], "sat"
    ["argument", Some m, Some m; "result", Some m, Some s], "sat"
    [], "sat"
]
for index, (comparisons, expected) in List.indexed applicationCases do
    let ob = { Id = sprintf "application_check_%d" index; Kind = "application-dimensions"; Logic = "QF_LIA"
               Statement = "application transfer regression"; Source = "test"; Refs = []
               Body = ObligationBody.ApplicationDimensions comparisons }
    let source = run "cvc5" "--lang=smt2" (Clef.Compiler.Nanopass.ObligationDischarge.smtLib [ob])
    let smt = run "mlir-translate" "--export-smtlib" (Alex.Traversal.SMTTransfer.transfer [ob])
    if not (smt.Contains(ob.Id)) then failwithf "Application anchor %s did not survive transfer" ob.Id
    let native = run "cvc5" "--lang=smt2" smt
    if source <> expected || native <> expected then
        failwithf "Application %d: expected %s, source=%s native=%s" index expected source native
printfn "PASS %d application dimension source/native parity cases" applicationCases.Length
