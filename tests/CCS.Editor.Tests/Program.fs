module Clef.Compiler.Editor.Tests

open System
open System.IO
open System.Security.Cryptography
open System.Text
open System.Threading
open Clef.Compiler.Editor
open Clef.Compiler.NativeTypedTree.DimensionAlgebra
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

let check condition message = if not condition then failwith message
let equal expected actual = check (expected = actual) $"Expected {expected}; got {actual}"
let get (value: 'a option) = value |> Option.defaultWith (fun () -> failwith "Expected a current result")

let root = Path.Combine(Path.GetTempPath(), "ccs-editor-tests-" + Guid.NewGuid().ToString("N"))
let write (name: string) (text: string) =
    let file = Path.Combine(root, name)
    Directory.CreateDirectory(Path.GetDirectoryName file) |> ignore
    File.WriteAllText(file, text)
    file

let programLifetimeChecks () =
    ProgramLifetimeProjection.run (Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../Fixtures/ProgramLifetime")))
        (Path.Combine(root, "program-lifetime"))

let inputFileChecks () =
    let commonSources = [ "declarations/sources/Common.clef"; "declarations/sources/Alternate.clef" ]
    let common = write commonSources[0] "module Common\nlet value = 1\n"
    let alternate = write commonSources[1] "module Common\nlet value = 2\n"
    let left = write "code/Left.clef" "module Left\nlet value = Common.value\n"
    let right = write "code/Right.clef" "module Right\nlet value = Common.value\n"
    let main = write "app/Main.clef" "module Main\nlet value = Left.value + Right.value\n"
    let manifest dependencies sources =
        $"[package]\nname = \"input-files\"\n[compilation]\ntarget = \"library\"\n[dependencies]\n{dependencies}\n[build]\nsources = [{sources}]\noutput_kind = \"library\"\n"
    let commonManifest = write "declarations/deep/Common.fidproj" (manifest "" "\"../sources/Common.clef\"")
    let commonDependency = "common = { path = \"../declarations/deep/Common.fidproj\" }"
    let leftManifest = write "manifests/Left.fidproj" (manifest commonDependency "\"../code/Left.clef\"")
    let rightManifest = write "manifests/Right.fidproj" (manifest commonDependency "\"../code/Right.clef\"")
    let mainManifest = write "app/App.fidproj" (manifest
        "left = { path = \"../manifests/Left.fidproj\" }\nright = { path = \"../manifests/Right.fidproj\" }" "\"Main.clef\"")
    let session = EditorSession(mainManifest)
    let first = session.CheckAsync(Map.empty).Result |> get
    check first.Failure.IsNone $"Dependency fixture failed: {first.Failure}"
    equal [common; left; right; main] (first.Sources |> List.map (fun source -> source.FilePath))
    equal (Set.ofList [mainManifest; leftManifest; rightManifest; commonManifest; common; left; right; main]) (Set.ofList first.InputFiles)
    equal first.InputFiles.Length (Set.count (Set.ofList first.InputFiles))
    File.WriteAllText(commonManifest, manifest "" "\"../sources/Alternate.clef\"")
    let next = session.CheckAsync(Map.empty).Result |> get
    check (List.contains alternate next.InputFiles) "Manifest edit did not replace the watched source."
    check (not (List.contains common next.InputFiles)) "Removed source remained in the current input set."
    equal [alternate; left; right; main] (next.Sources |> List.map (fun source -> source.FilePath))
    File.Delete commonManifest
    let failed = session.CheckAsync(Map.empty).Result |> get
    check failed.Failure.IsSome "Deleted dependency manifest must fail the check."
    check (List.contains commonManifest failed.InputFiles) "Missing referenced manifest must remain observable for repair."
    printfn "PASS exact external manifest/source inputs, diamond dependencies, manifest edit and repair paths"

let proofChecks (obligation: ObligationView) =
    let run candidate = ProofDispatch.checkAsync "cvc5" candidate CancellationToken.None |> fun work -> work.GetAwaiter().GetResult()
    let query text =
        { obligation with SmtLib = text; QueryHash = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes text)) }
    let proved = run obligation
    equal "proved" proved.State
    equal obligation.QueryHash proved.QueryHash
    equal "counterexample" (run (query "(set-logic QF_LIA)\n(assert true)\n(check-sat)")).State
    equal "unknown" (run (query "(set-logic QF_LIA)\n(set-option :rlimit-per 1)\n(declare-const x Int)\n(assert (> x 0))\n(check-sat)")).State
    equal "error" (run (query "(this-is-not-smt)")).State
    let missing = ProofDispatch.checkAsync (Path.Combine(root, "missing-solver")) obligation CancellationToken.None |> fun work -> work.GetAwaiter().GetResult()
    equal "error" missing.State
    use cancelled = new CancellationTokenSource()
    cancelled.Cancel()
    let mutable observed = false
    try ProofDispatch.checkAsync "cvc5" obligation cancelled.Token |> fun work -> work.GetAwaiter().GetResult() |> ignore
    with :? OperationCanceledException -> observed <- true
    check observed "Caller cancellation must not be reported as a solver verdict."
    printfn "PASS actual cvc5 proved/counterexample/unknown/error and cancellation boundaries"

let dimensionalSolverChecks (template: ObligationView) =
    let basis owner name =
        Dimension.ofBase { Name = name; Module = [owner] }
    let m, s = basis "Units" "m", basis "Units" "s"
    let u = Dimension.ofVar { Id = 10001; Name = Some "u" }
    let v = Dimension.ofVar { Id = 10002; Name = Some "v" }
    let quotient left right = Dimension.mul left (Dimension.inv right)
    let cases = [
        "quotient", DimensionalRule.Quotient, m, s, Some (quotient m s), "proved"
        "corrupt quotient result", DimensionalRule.Quotient, m, s, Some m, "counterexample"
        "product", DimensionalRule.Product, m, s, Some (Dimension.mul m s), "proved"
        "same dimension", DimensionalRule.SameDimension, m, m, Some m, "proved"
        "incompatible operands", DimensionalRule.SameDimension, m, s, Some m, "counterexample"
        "corrupt sum result", DimensionalRule.SameDimension, m, m, Some s, "counterexample"
        "comparison", DimensionalRule.Comparison, m, m, None, "proved"
        "incompatible comparison", DimensionalRule.Comparison, m, s, None, "counterexample"
        "formal measure variables", DimensionalRule.Quotient, u, v, Some (quotient u v), "proved"
        "exchanged formal variables", DimensionalRule.Quotient, u, v, Some (quotient v u), "counterexample"
        "module identity", DimensionalRule.Comparison, m, basis "Other" "m", None, "counterexample"
        "cancellation", DimensionalRule.Quotient, m, m, Some (quotient m m), "proved"
        "missing result", DimensionalRule.Quotient, m, s, None, "counterexample"
    ]
    for name, rule, left, right, result, expected in cases do
        let obligation: ObligationInfo =
            { Id = "dimensional_regression"; Kind = "dimension-regression"; Logic = "QF_LIA"
              Statement = name; Source = "dimensional solver regression"; Refs = []
              Body = ObligationBody.DimensionalRelation(rule, left, right, result) }
        let query = Clef.Compiler.Nanopass.ObligationDischarge.smtLib [obligation]
        let candidate = { template with SmtLib = query; QueryHash = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes query)) }
        let actual = ProofDispatch.checkAsync "cvc5" candidate CancellationToken.None |> fun work -> work.GetAwaiter().GetResult()
        check (actual.State = expected) $"{name}: expected {expected}, got {actual.State}"
    printfn "PASS dimensional cvc5 laws, corrupt results, formal variables and module identity"

let realSolverChecks (template: ObligationView) =
    let rational text =
        match Clef.Compiler.NativeTypedTree.NativeTypes.ExactRational.tryParseDecimal text with
        | Ok value -> value
        | Error message -> failwith message
    let tenth, zero, one = rational "0.1", rational "0", rational "1"
    let roundedTenth = rational "0.1000000000000000055511151231257827021181583404541015625"
    for name, body, expected in [
        "exact decimal point", ObligationBody.RealLiteralRange(tenth, tenth, tenth), "proved"
        "host rounding cannot replace source range", ObligationBody.RealLiteralRange(tenth, roundedTenth, roundedTenth), "counterexample"
        "declared bounds cover point", ObligationBody.RealRepresentationCoverage(tenth, tenth, zero, one), "proved"
        "declared capacity does not imply value fits", ObligationBody.RealRepresentationCoverage(rational "12", rational "12", zero, one), "counterexample"
        "empty interval", ObligationBody.RealRepresentationCoverage(one, zero, zero, one), "counterexample"
    ] do
        let obligation: ObligationInfo =
            { Id = "real_regression"; Kind = "real-regression"; Logic = "QF_LRA"
              Statement = name; Source = "real solver regression"; Refs = []; Body = body }
        let query = Clef.Compiler.Nanopass.ObligationDischarge.smtLib [obligation]
        let candidate = { template with SmtLib = query; QueryHash = Convert.ToHexStringLower(SHA256.HashData(Encoding.UTF8.GetBytes query)) }
        let actual = ProofDispatch.checkAsync "cvc5" candidate CancellationToken.None |> fun work -> work.GetAwaiter().GetResult()
        check (actual.State = expected) $"{name}: expected {expected}, got {actual.State}"
    printfn "PASS exact source-real QF_LRA ranges and representation coverage, including false claims"

let importVisibilityChecks () =
    let project = write "import-scope/Scope.fidproj" """[package]
name = "import-scope"
[compilation]
target = "library"
[build]
sources = ["Units.clef", "Main.clef"]
output_kind = "library"
"""
    let units = write "import-scope/Units.clef" """module Demo.Units
[<Measure>] type m
[<Measure>] type s
let speed (distance: float<m>) (elapsed: float<s>) = distance / elapsed
"""
    let source = """module Demo.Main
open Demo.Units
[<EntryPoint>]
let main argv =
    let velocity = speed 12.0<m> 3.0<s>
    if velocity > 0.0<m/s> then 0 else 1
"""
    let mainFile = write "import-scope/Main.clef" source
    let line = 4
    let column = source.Split('\n').[line].IndexOf("speed", StringComparison.Ordinal)
    let session = EditorSession(project)
    let successful (snapshot: EditorSnapshot) =
        check snapshot.Failure.IsNone $"Import fixture failed: {snapshot.Failure}"
        check snapshot.ParseFailures.IsEmpty $"Import fixture failed to parse: {snapshot.ParseFailures}"
        check (snapshot.Diagnostics |> List.forall (fun d -> d.EffectiveSeverity <> "Error"))
            $"Imported binding has errors: {snapshot.Diagnostics}"
        let dimensionProofs = snapshot.Obligations |> List.filter (fun obligation ->
            obligation.Kind.StartsWith("dimension-", StringComparison.Ordinal))
        let division = dimensionProofs |> List.tryFind (fun obligation ->
            obligation.Range |> Option.exists (fun range -> range.FilePath = units && range.StartLine = 3))
        check division.IsSome "The speed division has no compiler-generated dimensional obligation."
        let division = get division
        equal "QF_LIA" division.Logic
        check (division.Premises.Length >= 3) "Dimensional evidence must cite the operation and both operands."
        let application = dimensionProofs |> List.tryFind (fun obligation ->
            obligation.Kind = "dimension-application" &&
            (obligation.Range |> Option.exists (fun range -> range.FilePath = mainFile && range.StartLine = line)))
        check application.IsSome "The velocity call has no compiler-generated application obligation."
        let application = get application
        equal "QF_LIA" application.Logic
        check (application.Premises.Length >= 4) "Application evidence must connect the call, callee and actual arguments."
        check (application.Premises |> List.exists (fun premise ->
            premise.Range |> Option.exists (fun range -> range.FilePath = units && range.StartLine = 3)))
            "Application evidence lost the resolved speed definition in Units.clef."
        for obligation in dimensionProofs do
            let result = ProofDispatch.checkAsync "cvc5" obligation CancellationToken.None |> fun work -> work.GetAwaiter().GetResult()
            equal "proved" result.State
        let speed = session.TryHover(snapshot.Revision, mainFile, line, column) |> get
        equal "float<m> -> float<s> -> float<m / s>" speed.Type
        let definition = speed.Definition |> get
        equal units definition.FilePath
        equal 3 definition.StartLine
        let declaration =
            (snapshot.Sources |> List.find (fun file -> file.FilePath = units)).Content.Split('\n').[3]
        for name, expectedType in
            [ "speed", "float<m> -> float<s> -> float<m / s>"
              "distance", "float<m>"
              "elapsed", "float<s>" ] do
            let start = declaration.IndexOf(name, StringComparison.Ordinal)
            for offset in 0 .. name.Length - 1 do
                let declared = session.TryHover(snapshot.Revision, units, 3, start + offset) |> get
                check (declared.Type = expectedType)
                    $"Declaration hover for {name} at column {start + offset}: expected {expectedType}; got {declared}"
        speed
    let first = session.CheckAsync(Map.empty).Result |> get
    equal [units; mainFile] (first.Sources |> List.map (fun file -> file.FilePath))
    let imported = successful first
    dimensionalSolverChecks (first.Obligations |> List.find (fun ob -> ob.Kind = "dimension-quotient"))
    let realProofs = first.Obligations |> List.filter (fun ob -> ob.Kind = "real-literal-range")
    equal 3 realProofs.Length
    for obligation in realProofs do
        equal "QF_LRA" obligation.Logic
        let actual = ProofDispatch.checkAsync "cvc5" obligation CancellationToken.None |> fun work -> work.GetAwaiter().GetResult()
        equal "proved" actual.State
    realSolverChecks realProofs.Head
    let retained = sprintf "%A" first
    let withoutOpen = source.Replace("open Demo.Units", "// open Demo.Units")
    let pending = session.CheckAsync(Map.ofList [mainFile, withoutOpen])
    check session.Current.IsNone "Import edit retained the previous current snapshot."
    check (session.TryHover(first.Revision, mainFile, line, column).IsNone)
        "Pending import edit served the old speed reference."
    let removed = pending.Result |> get
    check removed.Failure.IsNone $"Import removal failed to check: {removed.Failure}"
    check removed.ParseFailures.IsEmpty $"Commenting an import failed to parse: {removed.ParseFailures}"
    check (removed.Revision > first.Revision) "Import edit did not advance the revision."
    check (removed.Diagnostics |> List.exists (fun diagnostic ->
        diagnostic.Code = "CCS8009"
        && diagnostic.Severity = "Error" && diagnostic.EffectiveSeverity = "Error"
        && diagnostic.Message.Contains("speed", StringComparison.Ordinal)
        && (diagnostic.Range |> Option.exists (fun range ->
            range.FilePath = mainFile && range.StartLine = line && range.EndLine = line
            && range.StartCharacter <= column && range.EndCharacter >= column + "speed".Length))))
        $"Missing located, effective unknown-speed error after import removal: {removed.Diagnostics}"
    check (session.TryHover(first.Revision, mainFile, line, column).IsNone)
        "Completed import edit still accepted the old snapshot revision."
    let unresolved = session.TryHover(removed.Revision, mainFile, line, column)
    check unresolved.IsNone $"Unavailable speed must have no semantic hover or definition: {unresolved}"
    equal withoutOpen (removed.Sources |> List.find (fun file -> file.FilePath = mainFile)).Content
    equal source (File.ReadAllText mainFile)
    check (removed.Obligations |> List.forall (fun obligation ->
        obligation.Kind <> "dimension-application" ||
        (obligation.Range |> Option.forall (fun range -> range.FilePath <> mainFile || range.StartLine <> line))))
        "An unresolved speed reference retained a previous application obligation."
    equal retained (sprintf "%A" first)
    let restored = session.CheckAsync(Map.ofList [mainFile, source]).Result |> get
    check (restored.Revision > removed.Revision) "Restoring the import did not advance the revision."
    let recovered = successful restored
    equal imported.Definition recovered.Definition
    check (session.TryHover(removed.Revision, mainFile, line, column).IsNone)
        "Restoring the import retained a read from the failed revision."
    printfn "PASS unsaved import removal/restoration, located name error and fresh dimensional definition"

let integerLiteralChecks () =
    let project = write "integer-proofs/Integers.fidproj" """[package]
name = "integer-proofs"
[compilation]
target = "library"
[build]
sources = ["Main.clef"]
output_kind = "library"
"""
    let source = """module IntegerProofs
[<Measure>] type m
[<Measure>] type s
[<EntryPoint>]
let main _ =
    let distance = 12<m>
    let elapsed = 3<s>
    if distance > 0<m> && elapsed > 0<s> then 0 else 1
"""
    let file = write "integer-proofs/Main.clef" source
    let session = EditorSession(project)
    let checkedSnapshot overrides =
        let snapshot = session.CheckAsync(overrides).Result |> get
        check snapshot.Failure.IsNone $"Integer fixture failed: {snapshot.Failure}"
        check snapshot.ParseFailures.IsEmpty $"Integer fixture did not parse: {snapshot.ParseFailures}"
        check (snapshot.Diagnostics |> List.forall (fun d -> d.EffectiveSeverity <> "Error"))
            $"Integer fixture has errors: {snapshot.Diagnostics}"
        check (snapshot.Obligations |> List.forall (fun ob -> ob.Kind <> "integer-representation-coverage"))
            "A platform-free project fabricated a selected integer representation."
        snapshot
    let atLine line (snapshot: EditorSnapshot) =
        snapshot.Obligations |> List.filter (fun ob ->
            ob.Kind = "integer-literal-range" &&
            (ob.Range |> Option.exists (fun range -> range.FilePath = file && range.StartLine = line)))
    let prove (obligation: ObligationView) =
        equal "QF_LIA" obligation.Logic
        check (not obligation.Premises.IsEmpty) "Integer evidence lost its graph premises."
        let verdict = ProofDispatch.checkAsync "cvc5" obligation CancellationToken.None |> fun work -> work.GetAwaiter().GetResult()
        equal "proved" verdict.State
    let first = checkedSnapshot Map.empty
    for line in [5; 6] do
        let proofs = atLine line first
        equal 1 proofs.Length
        prove proofs.Head
    let oldDistance = (atLine 5 first).Head
    let changedSource = source.Replace("12<m>", "24<m>")
    let changed = checkedSnapshot (Map.ofList [file, changedSource])
    let newDistance = (atLine 5 changed).Head
    check (changed.Revision > first.Revision && newDistance.QueryHash <> oldDistance.QueryHash)
        "Changing an integer literal retained its previous proof query."
    prove newDistance
    let removedSource = changedSource.Replace("    let distance = 24<m>", "    // distance removed").Replace("distance > 0<m>", "0<m> > 0<m>")
    let removed = checkedSnapshot (Map.ofList [file, removedSource])
    check ((atLine 5 removed).IsEmpty) "A deleted integer binding retained its source obligation."
    equal source (File.ReadAllText file)
    printfn "PASS measured integer source obligations, cvc5 dispatch and fresh edit/removal queries"

let directCaptureChecks () =
    let project = write "direct-captures/Editor.fidproj" """[package]
name = "editor-direct-captures"
[compilation]
target = "library"
[build]
sources = ["Main.clef"]
output_kind = "library"
"""
    let source = """module DirectCaptures
[<Measure>] type m
[<EntryPoint>]
let main _ =
    let offset = 7<m>
    let shift (value: int<m>) = offset + value
    let plain (value: int<m>) = value
    let make () = fun (value: int<m>) -> offset + value
    let shifted = shift 3<m>
    let unchanged = plain 10<m>
    let produced = make () 3<m>
    if shifted = unchanged && produced = 10<m> then 0 else 1
"""
    let file = write "direct-captures/Main.clef" source
    let session = EditorSession(project)
    let snapshot = session.CheckAsync(Map.empty).Result |> get
    check snapshot.Failure.IsNone $"Direct capture fixture failed: {snapshot.Failure}"
    check snapshot.ParseFailures.IsEmpty $"Direct capture fixture did not parse: {snapshot.ParseFailures}"
    check (snapshot.Diagnostics |> List.forall (fun diagnostic -> diagnostic.EffectiveSeverity <> "Error"))
        $"Direct capture fixture has errors: {snapshot.Diagnostics}"
    let lines = source.Split('\n')
    let at (marker: string) (name: string) =
        let line = lines |> Array.findIndex (fun text -> text.Contains(marker, StringComparison.Ordinal))
        let column = lines[line].IndexOf(name, StringComparison.Ordinal)
        session.TryHover(snapshot.Revision, file, line, column) |> get
    for name, signature, declaration, reference in
        [ "shift", "int<m> -> int<m>", "let shift", "let shifted = shift"
          "plain", "int<m> -> int<m>", "let plain", "let unchanged = plain"
          "make", "unit -> int<m> -> int<m>", "let make", "let produced = make" ] do
        let definition = at declaration name
        let referenceLine = lines |> Array.findIndex (fun text -> text.Contains(reference, StringComparison.Ordinal))
        let referenceColumn = lines[referenceLine].LastIndexOf(name, StringComparison.Ordinal)
        let useSite = session.TryHover(snapshot.Revision, file, referenceLine, referenceColumn) |> get
        equal signature definition.Type
        equal signature useSite.Type
        equal "VarRef" useSite.Kind
        equal definition.Range (useSite.Definition |> get)
    equal "int<m>" (at "let produced" "produced").Type
    let capture = at "let offset" "offset"
    for marker in ["let shift"; "let make"] do
        let reference = at marker "offset"
        equal "VarRef" reference.Kind
        equal "int<m>" reference.Type
        equal capture.Range (reference.Definition |> get)
    printfn "PASS source callable signatures and capture origins for direct captures, captureless control and returned functions"

let closureEnvironmentChecks () =
    let project = write "closure-environments/Editor.fidproj" """[package]
name = "editor-closure-environments"
[compilation]
target = "library"
[build]
sources = ["Main.clef"]
output_kind = "library"
"""
    let source = """module ClosureEnvironments
[<Measure>] type m
[<Measure>] type s
[<EntryPoint>]
let main _ =
    let bias = 2<m>
    let mutable offset = 7<m>
    let mapper = fun (value: int<m>) -> value + offset + bias
    let mapped = Seq.map mapper (seq { yield 1<m> })
    let mutable total = 0<m>
    for value in mapped do total <- total + value
    if total = 10<m> then 0 else 1
"""
    let file = write "closure-environments/Main.clef" source
    let session = EditorSession(project)
    let lines = source.Split('\n')
    let at (snapshot: EditorSnapshot) (marker: string) (name: string) =
        let line = lines |> Array.findIndex (fun text -> text.Contains(marker, StringComparison.Ordinal))
        let column = lines[line].LastIndexOf(name, StringComparison.Ordinal)
        session.TryHover(snapshot.Revision, file, line, column) |> get
    let checkedSnapshot inputs =
        let snapshot = session.CheckAsync(inputs).Result |> get
        check snapshot.Failure.IsNone $"Closure environment fixture failed: {snapshot.Failure}"
        check snapshot.ParseFailures.IsEmpty $"Closure environment fixture did not parse: {snapshot.ParseFailures}"
        check (snapshot.Diagnostics |> List.forall (fun diagnostic -> diagnostic.EffectiveSeverity <> "Error"))
            $"Closure environment fixture has errors: {snapshot.Diagnostics}"
        snapshot
    let checkProjection snapshot =
        let mapper = at snapshot "let mapper" "mapper"
        let reference = at snapshot "let mapped" "mapper"
        equal "int<m> -> int<m>" mapper.Type
        equal mapper.Type reference.Type
        equal mapper.Range (reference.Definition |> get)
        equal "seq<int<m>>" (at snapshot "let mapped" "mapped").Type
        for name in ["offset"; "bias"] do
            let declaration = at snapshot (if name = "offset" then "let mutable offset" else "let bias") name
            let captured = at snapshot "let mapper" name
            equal "EnvironmentRead" captured.Kind
            equal (Some name) captured.Name
            equal "int<m>" captured.Type
            equal declaration.Range (captured.Definition |> get)
    let first = checkedSnapshot Map.empty
    checkProjection first
    let retained = sprintf "%A" first
    let changed = source.Replace("offset = 7<m>", "offset = 7<s>")
    let invalid = session.CheckAsync(Map.ofList [file, changed]).Result |> get
    check invalid.Failure.IsNone $"Dimensional edit failed to produce a source snapshot: {invalid.Failure}"
    check (invalid.Diagnostics |> List.exists (fun diagnostic ->
        diagnostic.Code = "CCS8040" && diagnostic.EffectiveSeverity = "Error" &&
        (diagnostic.Range |> Option.exists (fun range -> range.FilePath = file && range.StartLine = 7))))
        $"Captured dimension mismatch lost its source diagnostic: {invalid.Diagnostics}"
    let repaired = checkedSnapshot Map.empty
    checkProjection repaired
    check (repaired.Revision > invalid.Revision) "Repair did not produce a fresh closure projection."
    equal retained (sprintf "%A" first)
    equal source (File.ReadAllText file)
    printfn "PASS materialized callback signatures, captured declaration identities and dimensional edit/repair"

let sequenceApplicationChecks () =
    let project = write "sequence-applications/Editor.fidproj" """[package]
name = "editor-sequence-applications"
[compilation]
target = "library"
[build]
sources = ["Main.clef"]
output_kind = "library"
"""
    let source = """module SequenceApplications
[<Measure>] type m
[<Measure>] type s
[<EntryPoint>]
let main _ =
    let offset = 7<m>
    let mapper = fun (value: int<m>) -> value + offset
    let mapOperation: (int<m> -> int<m>) -> seq<int<m>> -> seq<int<m>> = Seq.map
    let storedMap = mapOperation mapper
    let mapped = storedMap (seq { yield 1<m> })
    let folder = fun (state: float<s>) (_: int<m>) -> state + 0.25<s>
    let afterFolder = Seq.fold folder
    let afterState = afterFolder 1.0<s>
    let observed = afterState mapped
    if observed = 1.25<s> then 0 else 1
"""
    let file = write "sequence-applications/Main.clef" source
    let session = EditorSession(project)
    let lines = source.Split('\n')
    let location (marker: string) (name: string) =
        let line = lines |> Array.findIndex (fun text -> text.Contains(marker, StringComparison.Ordinal))
        line, lines[line].LastIndexOf(name, StringComparison.Ordinal)
    let at (snapshot: EditorSnapshot) marker name =
        let line, column = location marker name
        session.TryHover(snapshot.Revision, file, line, column) |> get
    let checkedSnapshot inputs =
        let snapshot = session.CheckAsync(inputs).Result |> get
        check snapshot.Failure.IsNone $"Sequence application fixture failed: {snapshot.Failure}"
        check snapshot.ParseFailures.IsEmpty $"Sequence application fixture did not parse: {snapshot.ParseFailures}"
        check (snapshot.Diagnostics |> List.forall (fun diagnostic -> diagnostic.EffectiveSeverity <> "Error"))
            $"Sequence application fixture has errors: {snapshot.Diagnostics}"
        snapshot
    let checkProjection snapshot =
        let references =
            [ "mapper", "int<m> -> int<m>", "let mapper", "let storedMap"
              "mapOperation", "(int<m> -> int<m>) -> seq<int<m>> -> seq<int<m>>", "let mapOperation", "let storedMap"
              "storedMap", "seq<int<m>> -> seq<int<m>>", "let storedMap", "let mapped"
              "folder", "float<s> -> int<m> -> float<s>", "let folder", "let afterFolder"
              "afterFolder", "float<s> -> seq<int<m>> -> float<s>", "let afterFolder", "let afterState"
              "afterState", "seq<int<m>> -> float<s>", "let afterState", "let observed" ]
            |> List.map (fun (name, signature, declaration, reference) ->
                let declared = at snapshot declaration name
                let used = at snapshot reference name
                equal signature declared.Type
                equal signature used.Type
                equal (Some name) used.Name
                equal declared.Range (used.Definition |> get)
                declared, used)
        equal "seq<int<m>>" (at snapshot "let mapped" "mapped").Type
        equal "float<s>" (at snapshot "let observed" "observed").Type
        let offset = at snapshot "let offset" "offset"
        let captured = at snapshot "let mapper" "offset"
        equal "int<m>" captured.Type
        equal (Some "offset") captured.Name
        equal offset.Range (captured.Definition |> get)
        references, captured
    let first = checkedSnapshot Map.empty
    let originalProjection = checkProjection first
    let retained = sprintf "%A" (first, originalProjection)
    let changed = source.Replace("offset = 7<m>", "offset = 7<s>")
    let pending = session.CheckAsync(Map.ofList [file, changed])
    let line, column = location "let observed" "afterState"
    check (session.TryHover(first.Revision, file, line, column).IsNone)
        "A pending staged-callable edit served a stale source signature."
    let invalid = pending.Result |> get
    check invalid.Failure.IsNone $"Measured sequence edit failed to produce a source snapshot: {invalid.Failure}"
    check invalid.ParseFailures.IsEmpty $"Measured sequence edit did not parse: {invalid.ParseFailures}"
    let mapperLine, _ = location "let mapper" "offset"
    check (invalid.Diagnostics |> List.exists (fun diagnostic ->
        diagnostic.Code = "CCS8040" && diagnostic.EffectiveSeverity = "Error" &&
        (diagnostic.Range |> Option.exists (fun range -> range.FilePath = file && range.StartLine = mapperLine))))
        $"Staged sequence capture mismatch lost its source diagnostic: {invalid.Diagnostics}"
    check (invalid.Revision > first.Revision) "Measured edit did not advance the sequence snapshot revision."
    let repaired = checkedSnapshot Map.empty
    checkProjection repaired |> ignore
    check (repaired.Revision > invalid.Revision) "Repair did not produce a fresh staged sequence projection."
    check (session.TryHover(invalid.Revision, file, line, column).IsNone)
        "Repair still served a hover from the invalid source revision."
    equal retained (sprintf "%A" (first, originalProjection))
    equal source (File.ReadAllText file)
    printfn "PASS staged Seq source signatures, definition identities and measured edit/repair with stale-read rejection"

let loopObligationChecks () =
    let project = write "loop-obligations/Editor.fidproj" """[package]
name = "editor-loop-obligations"
[compilation]
target = "library"
[build]
sources = ["Main.clef"]
output_kind = "library"
"""
    let source = """module LoopObligations
let observed = seq {
    let mutable total = 0
    let mutable i = 1
    while i <= 6 do
        total <- total + i
        yield total
        i <- i + 1
}
[<EntryPoint>]
let main _ = ignore observed; 0
"""
    let file = write "loop-obligations/Main.clef" source
    let session = EditorSession(project)
    let checkedSnapshot text =
        let snapshot = session.CheckAsync(Map.ofList [file, text]).Result |> get
        check snapshot.Failure.IsNone $"Loop projection failed: {snapshot.Failure}"
        check snapshot.ParseFailures.IsEmpty $"Loop projection did not parse: {snapshot.ParseFailures}"
        check (snapshot.Diagnostics |> List.forall (fun diagnostic -> diagnostic.EffectiveSeverity <> "Error"))
            $"Loop projection has errors: {snapshot.Diagnostics}"
        snapshot
    let loops (snapshot: EditorSnapshot) =
        snapshot.Obligations |> List.filter (fun obligation ->
            obligation.Kind = "finite-loop-trip" || obligation.Kind = "additive-loop-invariant")
    let checkEvidence upper snapshot =
        let obligations = loops snapshot
        equal 2 obligations.Length
        let total = session.TryHover(snapshot.Revision, file, 2, 16) |> get
        equal (Some $"[0, {upper}]") total.ValueRange
        for obligation in obligations do
            equal (Some total.Range) obligation.Range
            check (List.contains obligation.Id total.ObligationIds) "The cell hover lost its obligation anchor."
            let hasPremise kind line =
                obligation.Premises |> List.exists (fun premise ->
                    premise.Kind = kind && premise.Range |> Option.exists (fun span -> span.FilePath = file && span.StartLine = line))
            check (hasPremise "Binding" 2 && hasPremise "Binding" 3 && hasPremise "WhileLoop" 4 && hasPremise "Set" 5 && hasPremise "Set" 7)
                $"The recurrence lost source premise navigation: {obligation.Premises}"
            let verdict = ProofDispatch.checkAsync "cvc5" obligation CancellationToken.None |> fun task -> task.GetAwaiter().GetResult()
            equal "proved" verdict.State
        obligations
    let first = checkedSnapshot source
    let originalEvidence = checkEvidence 36 first
    let retained = sprintf "%A" first
    let narrowed = checkedSnapshot (source.Replace("i <= 6", "i <= 3"))
    let changedEvidence = checkEvidence 9 narrowed
    check (narrowed.Revision > first.Revision) "The bound edit retained the old revision."
    check ((originalEvidence |> List.find (fun item -> item.Kind = "additive-loop-invariant")).QueryHash <>
           (changedEvidence |> List.find (fun item -> item.Kind = "additive-loop-invariant")).QueryHash)
        "The bound edit retained the old recurrence query."
    let retracted = checkedSnapshot (source.Replace("total <- total + i", "total <- total * i"))
    check (List.isEmpty (loops retracted)) "A non-additive edit retained an obsolete recurrence obligation."
    check (session.TryHover(first.Revision, file, 2, 16).IsNone) "A stale recurrence hover remained current."
    let repaired = checkedSnapshot source
    checkEvidence 36 repaired |> ignore
    equal retained (sprintf "%A" first)
    equal source (File.ReadAllText file)
    printfn "PASS loop proof premises, source navigation, solver dispatch and unsaved bound/retraction/repair"

let callEffectRangeChecks () =
    let project = write "call-effects/Editor.fidproj" """[package]
name = "editor-call-effects"
[compilation]
target = "library"
[build]
sources = ["Main.clef"]
output_kind = "library"
"""
    let source = """module CallEffects
let callChangesState () =
    let mutable state: int = 1
    let change = fun () -> state <- 300
    if state < 10 then
        change ()
        let observedAfterCall = state
        observedAfterCall
    else state
let storedPredicate () =
    let mutable state: int = 1
    let wasSmall = state < 10
    state <- 300
    if wasSmall then
        let observedAfterPredicate = state
        observedAfterPredicate
    else state
[<EntryPoint>]
let main _ =
    if callChangesState () = 300 && storedPredicate () = 300 then 0 else 1
"""
    let file = write "call-effects/Main.clef" source
    let session = EditorSession(project)
    let validate (snapshot: EditorSnapshot) =
        check snapshot.Failure.IsNone $"Call effect fixture failed: {snapshot.Failure}"
        check snapshot.ParseFailures.IsEmpty $"Call effect fixture did not parse: {snapshot.ParseFailures}"
        check (snapshot.Diagnostics |> List.forall (fun diagnostic -> diagnostic.EffectiveSeverity <> "Error"))
            $"Call effect fixture has errors: {snapshot.Diagnostics}"
    let at (text: string) (marker: string) =
        let lines = text.Split('\n')
        let line = lines |> Array.findIndex (fun value -> value.Contains(marker, StringComparison.Ordinal))
        line, lines[line].LastIndexOf("state", StringComparison.Ordinal)
    let read (snapshot: EditorSnapshot) text marker =
        let line, column = at text marker
        let hover = session.TryHover(snapshot.Revision, file, line, column) |> get
        equal "VarRef" hover.Kind
        equal (Some "state") hover.Name
        equal "int" hover.Type
        hover
    let markers = ["let observedAfterCall"; "let observedAfterPredicate"]
    let first = session.CheckAsync(Map.empty).Result |> get
    validate first
    let reads = markers |> List.map (read first source)
    for hover in reads do equal (Some "[1, 300]") hover.ValueRange
    let retained = System.Text.Json.JsonSerializer.Serialize {| snapshot = first; reads = reads |}
    let changedSource = source.Replace("300", "700")
    let changedTask = session.CheckAsync(Map.ofList [file, changedSource])
    let line, column = at source markers.Head
    check (session.TryHover(first.Revision, file, line, column).IsNone)
        "A pending effect edit served a stale range."
    let changed = changedTask.Result |> get
    validate changed
    check (changed.Revision > first.Revision) "Effect edit did not advance the snapshot revision."
    for marker in markers do
        equal (Some "[1, 700]") (read changed changedSource marker).ValueRange
    equal retained (System.Text.Json.JsonSerializer.Serialize {| snapshot = first; reads = reads |})
    equal source (File.ReadAllText file)
    printfn "PASS call-write and stored-predicate range invalidation, fresh hover and retained immutable snapshot"

let checks () =
    Directory.CreateDirectory(root) |> ignore
    let project = write "Editor.fidproj" """[package]
name = "editor-check"
version = "0.1.0"
[compilation]
target = "library"
[build]
sources = ["Measures.clef", "Use.clef"]
output = "editor-check"
output_kind = "library"
"""
    let measures = write "Measures.clef" """module Measures
[<Measure>] type m
[<Measure>] type s
let distance = 4.0<m>
let duration = 2.0<s>
"""
    let source = """module Use
open Measures
let speed = distance / duration
let shadow = 1.0<m>
let inner () =
    let shadow = 2.0<s>
    shadow
let outer = shadow
let message = "proof fixture"
[<EntryPoint>]
let main _ = if message = "proof fixture" then 0 else 1
"""
    let useFile = write "Use.clef" source
    let session = EditorSession(project)
    let first = session.CheckAsync(Map.empty).Result |> get
    check first.Failure.IsNone $"Project failed: {first.Failure}"
    check first.ParseFailures.IsEmpty $"Unexpected parse failures: {first.ParseFailures}"
    equal [measures; useFile] (first.Sources |> List.map (fun file -> file.FilePath))
    let hover line column = session.TryHover(first.Revision, useFile, line, column) |> get
    let speed = hover 2 5
    check (speed.Type.Contains("m") && speed.Type.Contains("s")) $"Missing inferred speed dimension: {speed.Type}"
    let inner = hover 6 6
    let outer = hover 7 13
    check (inner.Type.Contains("<s>")) $"Inner shadow lost seconds: {inner}"
    check (outer.Type.Contains("<m>")) $"Outer shadow lost metres: {outer}"
    equal 5 (inner.Definition |> get).StartLine
    equal 3 (outer.Definition |> get).StartLine
    check (inner.NodeId <> outer.NodeId) "Shadowed references must preserve separate identities."
    check (session.TryHover(first.Revision, useFile, 99, 0).IsNone) "No invented hover beyond source."
    printfn "PASS ordered project, inferred dimensions and shadowed reference lookup"

    check (not first.Obligations.IsEmpty) "Expected compiler-generated string obligations."
    for obligation in first.Obligations do
        check (obligation.SmtLib.Contains("(check-sat)")) "Compiler obligation query is missing."
        check (obligation.SmtLib.Contains(obligation.Id)) "Query lost its compiler anchor."
        equal 64 obligation.QueryHash.Length
        check (not obligation.Premises.IsEmpty) "Obligation lost its graph dependencies."
    printfn "PASS compiler-authored obligations and premise projections"
    proofChecks first.Obligations.Head

    let retained = sprintf "%A" first
    let bad = source.Replace("let speed = distance / duration", "let speed = distance + duration")
    let secondTask = session.CheckAsync(Map.ofList [useFile, bad])
    check session.Current.IsNone "Pending check must immediately hide the previous snapshot."
    check (session.TryHover(first.Revision, useFile, 2, 5).IsNone) "Pending check served stale hover."
    let second = secondTask.Result |> get
    check (second.Revision > first.Revision) "Revision must advance."
    check (second.Diagnostics |> List.exists (fun d -> d.Severity = "Error")) $"Missing dimensional error: {second.Diagnostics}"
    check (second.Diagnostics |> List.exists (fun d -> d.Range |> Option.exists (fun r -> r.FilePath = useFile && r.StartLine = 2))) "Dimensional error lost its compiler source range."
    equal source (File.ReadAllText useFile)
    equal retained (sprintf "%A" first)
    let repaired = session.CheckAsync(Map.empty).Result |> get
    check (repaired.Diagnostics |> List.forall (fun d -> d.Severity <> "Error")) $"Errors did not clear: {repaired.Diagnostics}"
    printfn "PASS unsaved dimensional error, clearing and retained immutable snapshot"

    let unicodeLine = "let unicode () = let text = \"😀\" in text"
    let unicodeSource = source.Replace("\n", "\r\n") + unicodeLine + "\r\n"
    let unicodeSnapshot = session.CheckAsync(Map.ofList [useFile, unicodeSource]).Result |> get
    let line = unicodeSource.Split('\n').Length - 2
    let column = unicodeLine.LastIndexOf("text", StringComparison.Ordinal)
    let unicodeHover = session.TryHover(unicodeSnapshot.Revision, useFile, line, column) |> get
    equal (Some "text") unicodeHover.Name
    equal column unicodeHover.Range.StartCharacter
    equal (column + 4) unicodeHover.Range.EndCharacter
    equal unicodeSource (unicodeSnapshot.Sources |> List.find (fun file -> file.FilePath = useFile)).Content
    check (session.TryHover(unicodeSnapshot.Revision, useFile, line, column + 4)
        |> Option.forall (fun view -> view.NodeId <> unicodeHover.NodeId)) "Reference end must be exclusive."
    printfn "PASS UTF-16 columns, CRLF text preservation and exclusive ranges"

    let pending = session.CheckAsync(Map.empty)
    let invalidated = session.Invalidate()
    let _ = pending.Result
    check session.Current.IsNone "Invalidated check republished stale data."
    equal invalidated session.Revision
    printfn "PASS explicit invalidation rejects delayed reads"

    let parse = session.CheckAsync(Map.ofList [useFile, "module Use\nlet broken = (\n"]).Result |> get
    check (not parse.ParseFailures.IsEmpty) "Expected preserved parser failure messages."
    check (parse.ParseFailures |> List.exists (fun value -> value.FilePath = useFile)) "Parser failure lost file identity."
    printfn "PASS parser failures preserved without fabricated locations"

    let absent = EditorSession(Path.Combine(root, "Missing.fidproj")).CheckAsync(Map.empty).Result |> get
    check absent.Failure.IsSome "Missing project must be an explicit failure."
    equal [Path.Combine(root, "Missing.fidproj")] absent.InputFiles
    printfn "PASS project loading failure is explicit"
    inputFileChecks ()
    importVisibilityChecks ()
    integerLiteralChecks ()
    directCaptureChecks ()
    closureEnvironmentChecks ()
    sequenceApplicationChecks ()
    loopObligationChecks ()
    callEffectRangeChecks ()
    programLifetimeChecks ()
    StringEncodingChecks.run root

let inspectSample project =
    let session = EditorSession(project)
    let snapshot = session.CheckAsync(Map.empty).Result |> get
    printfn "Sample revision %d: %d sources, %d diagnostics, %d parse failures, %d obligations; failure=%A"
        snapshot.Revision snapshot.Sources.Length snapshot.Diagnostics.Length snapshot.ParseFailures.Length snapshot.Obligations.Length snapshot.Failure
    for diagnostic in snapshot.Diagnostics do printfn "%A" diagnostic
    for failure in snapshot.ParseFailures do printfn "%A" failure
    for source in snapshot.Sources do
        source.Content.Split('\n') |> Array.iteri (fun line text ->
            let column = text.IndexOf("velocity", StringComparison.Ordinal)
            if column >= 0 then printfn "Velocity at %s:%d: %A" source.FilePath (line + 1) (session.TryHover(snapshot.Revision, source.FilePath, line, column)))
    for obligation in snapshot.Obligations do
        let result = ProofDispatch.checkAsync "cvc5" obligation CancellationToken.None |> fun work -> work.GetAwaiter().GetResult()
        printfn "Obligation %s: %s (%d premises); solver=%s" obligation.Id obligation.Statement obligation.Premises.Length result.State
        equal "proved" result.State
    check snapshot.Failure.IsNone "Sample project load failed."
    check snapshot.ParseFailures.IsEmpty "Sample parsing failed."
    check (snapshot.Diagnostics |> List.forall (fun d -> d.EffectiveSeverity <> "Error")) "Sample has compiler errors."
    check (not snapshot.Obligations.IsEmpty) "Sample has no compiler obligations."

[<EntryPoint>]
let main args =
    try
        try
            match args with
            | [| "--sample"; project |] -> inspectSample project
            | [| "--direct-captures" |] -> directCaptureChecks ()
            | [| "--closure-environments" |] -> closureEnvironmentChecks ()
            | [| "--sequence-applications" |] -> sequenceApplicationChecks ()
            | [| "--loop-obligations" |] -> loopObligationChecks ()
            | [| "--call-effects" |] -> callEffectRangeChecks ()
            | [| "--program-lifetime" |] -> programLifetimeChecks ()
            | [| "--string-encoding" |] -> StringEncodingChecks.run root
            | [||] -> checks ()
            | _ -> failwith "Usage: CCS.Editor.Tests [--sample path.fidproj | --direct-captures | --closure-environments | --sequence-applications | --loop-obligations | --call-effects | --program-lifetime | --string-encoding]"
            0
        with error -> eprintfn "%O" error; 1
    finally if Directory.Exists root then Directory.Delete(root, true)
