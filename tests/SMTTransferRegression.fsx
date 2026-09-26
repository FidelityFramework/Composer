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
// This file tests source/build dispatch parity, including exact frame layouts;
// the separate continuation placement tests check the graph facts' provenance.
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
let trip : FiniteLoopTripModel =
    { InitialLower = 1I; LimitUpper = 6I; MinimumStep = 1I; Inclusive = true; MaximumIterations = 6I }
let recurrence : AdditiveLoopInvariantModel =
    { MaximumIterations = 6I; InitialLower = 0I; InitialUpper = 0I
      DeltaLower = 1I; DeltaUpper = 6I; Lower = 0I; Upper = 36I }
let loopCases = [
    ObligationBody.FiniteLoopTrip trip, "unsat"
    ObligationBody.FiniteLoopTrip { trip with Inclusive = false; MaximumIterations = 5I }, "unsat"
    ObligationBody.FiniteLoopTrip { trip with InitialLower = 7I; MaximumIterations = 0I }, "unsat"
    ObligationBody.FiniteLoopTrip { trip with InitialLower = -6I; LimitUpper = -1I }, "unsat"
    ObligationBody.FiniteLoopTrip { trip with LimitUpper = unsigned64Max; MaximumIterations = unsigned64Max }, "unsat"
    ObligationBody.FiniteLoopTrip { trip with MaximumIterations = 5I }, "sat"
    ObligationBody.FiniteLoopTrip { trip with MinimumStep = 0I }, "sat"
    ObligationBody.FiniteLoopTrip { trip with InitialLower = 100I; MaximumIterations = -1I }, "sat"
    ObligationBody.AdditiveLoopInvariant recurrence, "unsat"
    ObligationBody.AdditiveLoopInvariant { recurrence with DeltaLower = -6I; DeltaUpper = -1I; Lower = -36I; Upper = 0I }, "unsat"
    ObligationBody.AdditiveLoopInvariant { recurrence with DeltaLower = -2I; DeltaUpper = 3I; Lower = -12I; Upper = 18I }, "unsat"
    ObligationBody.AdditiveLoopInvariant { recurrence with MaximumIterations = 0I; InitialLower = 4I; InitialUpper = 7I; Lower = 4I; Upper = 7I }, "unsat"
    ObligationBody.AdditiveLoopInvariant { recurrence with MaximumIterations = unsigned64Max; DeltaUpper = 1I; Upper = unsigned64Max }, "unsat"
    ObligationBody.AdditiveLoopInvariant { recurrence with MaximumIterations = -1I }, "sat"
    ObligationBody.AdditiveLoopInvariant { recurrence with InitialLower = 1I; InitialUpper = 0I }, "sat"
    ObligationBody.AdditiveLoopInvariant { recurrence with DeltaLower = 7I; DeltaUpper = 6I }, "sat"
    ObligationBody.AdditiveLoopInvariant { recurrence with Lower = 37I; Upper = 36I }, "sat"
    ObligationBody.AdditiveLoopInvariant { recurrence with Lower = 1I }, "sat" // Initial value excluded.
    ObligationBody.AdditiveLoopInvariant { recurrence with Upper = 35I }, "sat" // Final prefix excluded.
    ObligationBody.AdditiveLoopInvariant { recurrence with DeltaLower = -2I; DeltaUpper = 3I; Lower = -11I; Upper = 18I }, "sat"
]
for index, (body, expected) in List.indexed loopCases do
    let ob = { Id = sprintf "loop_check_%d" index; Kind = "loop-range-regression"; Logic = "QF_LIA"
               Statement = "finite additive enclosure"; Source = "test"; Refs = []; Body = body }
    let source = run "cvc5" "--lang=smt2" (Clef.Compiler.Nanopass.ObligationDischarge.smtLib [ob])
    let smt = run "mlir-translate" "--export-smtlib" (Alex.Traversal.SMTTransfer.transfer [ob])
    if not (smt.Contains(ob.Id)) then failwithf "Loop anchor %s did not survive transfer" ob.Id
    let native = run "cvc5" "--lang=smt2" smt
    if source <> expected || native <> expected then
        failwithf "Loop %d: expected %s, source=%s native=%s for %A" index expected source native body
printfn "PASS %d loop-range source/native parity cases" loopCases.Length
// Certificates are input data: both dispatches must check every provided power,
// the exponent chain, monotonic upper trajectory, and the final enclosure.
// Ground bigint equations avoid selecting any machine-width representation.
let recurrencePowers (coefficients: bigint list list) (count: bigint) =
    let dimension = coefficients.Length
    let multiply (left: bigint list list) (right: bigint list list) =
        List.init dimension (fun row ->
            List.init dimension (fun column ->
                [0 .. dimension-1] |> List.sumBy(fun k -> left[row][k] * right[k][column])))
    let rec bits count found =
        if count = 0I then found else bits (count / 2I) ((count % 2I = 1I) :: found)
    let identity = List.init dimension (fun row -> List.init dimension (fun column -> if row = column then 1I else 0I))
    let _, _, certificate =
        bits count [] |> List.fold(fun (exponent, power, steps) odd ->
            let nextExponent = 2I * exponent + (if odd then 1I else 0I)
            let squared = multiply power power
            let nextPower = if odd then multiply squared coefficients else squared
            let step: RecurrencePowerStep = {Odd=odd; Exponent=nextExponent; Matrix=nextPower}
            nextExponent, nextPower, step :: steps) (0I, identity, [])
    List.rev certificate
let finiteLinear: FiniteLinearRecurrenceModel =
    { MaximumIterations=3I; InitialLower=[1I]; InitialUpper=[3I]
      CoefficientLower=[[2I]]; CoefficientUpper=[[2I]]
      Powers=recurrencePowers [[2I]] 3I; Lower=[0I]; Upper=[24I] }
let fibonacci: FiniteLinearRecurrenceModel =
    { MaximumIterations=6I; InitialLower=[0I; 1I]; InitialUpper=[0I; 1I]
      CoefficientLower=[[0I; 1I]; [1I; 1I]]; CoefficientUpper=[[0I; 1I]; [1I; 1I]]
      Powers=recurrencePowers [[0I; 1I]; [1I; 1I]] 6I; Lower=[0I; 0I]; Upper=[8I; 13I] }
let beyondWord = 1I <<< 160
let hugeCount = 1I <<< 80
let invalidFirst alter = {finiteLinear with Powers=alter finiteLinear.Powers.Head :: finiteLinear.Powers.Tail}
let linearCases = [
    "scalar", finiteLinear, "unsat"
    "coupled", fibonacci, "unsat"
    "asymmetric-coupling", {fibonacci with MaximumIterations=3I; InitialLower=[1I; 2I]; InitialUpper=[1I; 2I]; CoefficientLower=[[1I; 2I]; [0I; 1I]]; CoefficientUpper=[[1I; 2I]; [0I; 1I]]; Powers=recurrencePowers [[1I; 2I]; [0I; 1I]] 3I; Upper=[13I; 2I]}, "unsat"
    "coefficient-interval", {fibonacci with CoefficientLower=[[0I; 0I]; [0I; 0I]]}, "unsat"
    "zero-iterations", {finiteLinear with MaximumIterations=0I; Powers=[]; Upper=[3I]}, "unsat"
    "big-values", {finiteLinear with MaximumIterations=7I; InitialLower=[beyondWord]; InitialUpper=[beyondWord]; Powers=recurrencePowers [[2I]] 7I; Upper=[128I*beyondWord]}, "unsat"
    "big-count", {finiteLinear with MaximumIterations=hugeCount; CoefficientLower=[[1I]]; CoefficientUpper=[[1I]]; Powers=recurrencePowers [[1I]] hugeCount; Upper=[3I]}, "unsat"
    "negative-count", {finiteLinear with MaximumIterations= -1I}, "sat"
    "negative-seed", {finiteLinear with InitialLower=[-1I]}, "sat"
    "inverted-seed", {finiteLinear with InitialLower=[4I]}, "sat"
    "negative-coefficient", {finiteLinear with CoefficientLower=[[-1I]]}, "sat"
    "inverted-coefficient", {finiteLinear with CoefficientLower=[[3I]]}, "sat"
    "unproved-positive-lower", {finiteLinear with Lower=[1I]}, "sat"
    "final-store-excluded", {finiteLinear with Upper=[23I]}, "sat"
    "zero-iteration-seed-excluded", {finiteLinear with MaximumIterations=0I; Powers=[]; Upper=[2I]}, "sat"
    // M*x < x would let a final-only check omit the larger initial/prefix state.
    "nonmonotone-initial", {finiteLinear with CoefficientLower=[[0I]]; CoefficientUpper=[[0I]]; Powers=recurrencePowers [[0I]] 3I; Upper=[0I]}, "sat"
    "changed-power", invalidFirst (fun step -> {step with Matrix=[[3I]]}), "sat"
    "forged-final-power", {finiteLinear with Powers=[finiteLinear.Powers.Head; {finiteLinear.Powers[1] with Matrix=[[0I]]}]; Upper=[0I]}, "sat"
    "changed-exponent", invalidFirst (fun step -> {step with Exponent=2I}), "sat"
    "changed-bit", invalidFirst (fun step -> {step with Odd=false}), "sat"
    "missing-power", {finiteLinear with Powers=[]}, "sat"
    "count-disagrees", {finiteLinear with MaximumIterations=4I}, "sat"
    "coupled-last-component", {fibonacci with Upper=[8I; 12I]}, "sat"
    "coupled-negative-entry", {fibonacci with CoefficientLower=[[0I; 1I]; [-1I; 1I]]}, "sat"
    "empty-vector", {finiteLinear with InitialLower=[]}, "sat"
    "zero-state", {finiteLinear with MaximumIterations=0I; InitialLower=[]; InitialUpper=[]; Lower=[]; Upper=[]; CoefficientLower=[]; CoefficientUpper=[]; Powers=[]}, "sat"
    "wrong-vector-size", {finiteLinear with Upper=[24I; 24I]}, "sat"
    "ragged-coefficient", {fibonacci with CoefficientUpper=[[0I; 1I]; [1I]]}, "sat"
    "wrong-power-shape", invalidFirst (fun step -> {step with Matrix=[]}), "sat"
    "extra-power-column", invalidFirst (fun step -> {step with Matrix=[[2I; 0I]]}), "sat"
    "unsupported-dimension", {finiteLinear with MaximumIterations=0I; InitialLower=[0I; 0I; 0I]; InitialUpper=[0I; 0I; 0I]; Lower=[0I; 0I; 0I]; Upper=[0I; 0I; 0I]; CoefficientLower=List.replicate 3 [0I; 0I; 0I]; CoefficientUpper=List.replicate 3 [0I; 0I; 0I]; Powers=[]}, "sat"
]
for name, model, expected in linearCases do
    let ob =
        { Id="linear_" + name.Replace('-', '_'); Kind="finite-linear-recurrence"; Logic="QF_LIA"
          Statement="nonnegative recurrence prefixes"; Source="test"; Refs=[]
          Body=ObligationBody.FiniteLinearRecurrence model }
    let source = run "cvc5" "--lang=smt2" (Clef.Compiler.Nanopass.ObligationDischarge.smtLib [ob])
    let smt = run "mlir-translate" "--export-smtlib" (Alex.Traversal.SMTTransfer.transfer [ob])
    if not (smt.Contains ob.Id) then failwithf "Recurrence anchor %s did not survive transfer" ob.Id
    let native = run "cvc5" "--lang=smt2" smt
    if source <> expected || native <> expected then
        failwithf "Recurrence %s: expected %s, source=%s native=%s" name expected source native
printfn "PASS %d finite-linear recurrence source/native parity cases" linearCases.Length
// Each prefix may contain any count of each admitted store up to its proved
// activation bound. Mixed signs must enclose intermediate values, not only a net sum.
let effectCases = [
    "two-instances", 0I, [2I,1I], 0I, 2I, "unsat"
    "mixed-sign-prefixes", 10I, [1I,5I; 1I,-3I; 1I,2I; 1I,-7I], 0I, 17I, "unsat"
    "initial-only", 7I, [], 7I, 7I, "unsat"
    "zero-activations", 7I, [0I,beyondWord], 7I, 7I, "unsat"
    "big-positive", beyondWord, [hugeCount,beyondWord], beyondWord, beyondWord*(hugeCount+1I), "unsat"
    "big-negative", -beyondWord, [hugeCount,-beyondWord], -beyondWord*(hugeCount+1I), -beyondWord, "unsat"
    "negative-count", 0I, [-1I,1I], 0I, 0I, "sat"
    "final-excluded", 0I, [2I,1I], 0I, 1I, "sat"
    "negative-prefix-excluded", 10I, [1I,5I; 1I,-7I], 4I, 15I, "sat"
    "net-sum-is-insufficient", 10I, [1I,5I; 1I,-3I; 1I,2I; 1I,-7I], 7I, 10I, "sat"
    "reversed-bounds", 0I, [1I,0I], 1I, -1I, "sat"
    "initial-excluded", 7I, [], 8I, 8I, "sat"
]
for index, (name, initial, contributions, lower, upper, expected) in List.indexed effectCases do
    let ob = { Id = sprintf "effect_check_%d" index; Kind = "finite-additive-effects"; Logic = "QF_LIA"
               Statement = name; Source = "test"; Refs = []
               Body = ObligationBody.FiniteAdditiveEffects(initial, contributions, lower, upper) }
    let source = run "cvc5" "--lang=smt2" (Clef.Compiler.Nanopass.ObligationDischarge.smtLib [ob])
    let smt = run "mlir-translate" "--export-smtlib" (Alex.Traversal.SMTTransfer.transfer [ob])
    if not (smt.Contains(ob.Id)) then failwithf "Effect anchor %s did not survive transfer" ob.Id
    let native = run "cvc5" "--lang=smt2" smt
    if source <> expected || native <> expected then
        failwithf "Effect %s: expected %s, source=%s native=%s" name expected source native
printfn "PASS %d finite-effect source/native parity cases" effectCases.Length
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
let staticLayoutCases = [
    ObligationBody.StaticStorageLayout ([(0,3,1);(4,5,4)],9,16,8,16L,8,8), "unsat"
    ObligationBody.StaticStorageLayout ([(0,3,1);(2,5,1)],7,16,8,16L,8,8), "sat" // Overlap.
    ObligationBody.StaticStorageLayout ([(0,3,1);(4,5,4)],9,16,8,8L,8,8), "sat" // Capacity.
    ObligationBody.StaticStorageLayout ([(0,3,1);(4,5,4)],9,16,4,16L,8,8), "sat" // Pool alignment.
    ObligationBody.StaticStorageLayout ([(0,3,1);(3,5,4)],8,16,8,16L,8,8), "sat" // Slot alignment.
    ObligationBody.StaticStorageLayout ([(0,3,1)],3,15,8,16L,8,8), "sat" // Granularity.
    ObligationBody.StaticStorageLayout ([(0,3,1)],3,16,8,16L,0,8), "sat" // Invalid declaration.
    ObligationBody.StaticStorageLayout ([(0,3,0)],3,16,8,16L,8,8), "sat" // Invalid slot.
    ObligationBody.StaticStorageLayout ([(0,3,16)],3,16,8,16L,8,8), "sat" // Slot needs stronger base alignment.
    ObligationBody.StaticStorageLayout ([(0,3,1)],4,16,8,16L,8,8), "sat" // Incorrect used size.
    ObligationBody.StaticStorageLayout ([(2147483647,2147483647,1)],0,16,8,16L,8,8), "sat" // No host int wrap.
    ObligationBody.StaticStorageLayout ([],0,0,8,16L,8,8), "unsat"
]
for index, (body, expected) in List.indexed staticLayoutCases do
    let ob = { Id = sprintf "static_layout_%d" index; Kind = "static-layout-regression"; Logic = "QF_LIA"
               Statement = "settled layout"; Source = "test"; Refs = []; Body = body }
    let source = run "cvc5" "--lang=smt2" (Clef.Compiler.Nanopass.ObligationDischarge.smtLib [ob])
    let smt = run "mlir-translate" "--export-smtlib" (Alex.Traversal.SMTTransfer.transfer [ob])
    let native = run "cvc5" "--lang=smt2" smt
    if source <> expected || native <> expected then failwithf "Layout expected %s, source=%s native=%s for %A" expected source native body
printfn "PASS %d concrete-layout source/native parity cases" staticLayoutCases.Length
let continuationLayoutCases = [
    [0,1,1; 1,1,1; 2,2,2], 4, 2, "unsat"
    [0,1,1; 1,1,1; 8,40,8], 48, 8, "unsat"
    [], 0, 1, "unsat"
    [0,1,1; 3,1,1], 4, 1, "sat" // Misplaced slot/excess interior padding.
    [0,2,2; 0,2,2], 2, 2, "sat" // Overlap.
    [0,1,1; 2,2,2], 3, 2, "sat" // Truncated extent.
    [0,1,1; 2,2,2], 6, 2, "sat" // Excess terminal padding.
    [0,1,1; 2,2,2], 4, 1, "sat" // Insufficient frame alignment.
    [0,3,3], 3, 3, "sat"       // Unsupported non-power-of-two alignment.
    [0,1,0], 1, 1, "sat"
    [-1,1,1], 0, 1, "sat"
    [0,0,1], 0, 1, "sat"
    [0,Int32.MaxValue,1; Int32.MaxValue,1,1], Int32.MaxValue, 1, "sat"
    [], 1, 1, "sat"
    [], 0, 2, "sat"
]
for index, (slots, extent, alignment, expected) in List.indexed continuationLayoutCases do
    let ob = { Id = sprintf "continuation_layout_%d" index; Kind = "continuation-layout-regression"; Logic = "QF_LIA"
               Statement = "exact continuation placement"; Source = "test"; Refs = []
               Body = ObligationBody.ContinuationLayout(slots, extent, alignment) }
    let source = run "cvc5" "--lang=smt2" (Clef.Compiler.Nanopass.ObligationDischarge.smtLib [ob])
    let smt = run "mlir-translate" "--export-smtlib" (Alex.Traversal.SMTTransfer.transfer [ob])
    if not (smt.Contains(ob.Id)) then failwithf "Continuation anchor %s did not survive transfer" ob.Id
    let native = run "cvc5" "--lang=smt2" smt
    if source <> expected || native <> expected then
        failwithf "Continuation %d: expected %s, source=%s native=%s for %A" index expected source native ob.Body
printfn "PASS %d continuation-layout source/native parity cases" continuationLayoutCases.Length
