# Language coverage waypoints

This record connects the language/compiler and tooling revisions for bounded
implementation steps. A passing editor projection, solver query, MLIR verifier
or executable establishes its own boundary; none substitutes for the others.
The [review](Clef_Language_Completion_Review_2026-09-19.md) and
[incremental contract direction](Nanopass_Incremental_Contract_Direction.md)
retain the wider roadmap and unresolved contracts.

## C-04 Option defaults and C-01 callable prerequisites — 2026-09-19

`Option.defaultValue` selects an eagerly evaluated fallback. `Option.defaultWith`
evaluates its thunk expression eagerly but invokes the thunk only for `None`.
Stored partial applications snapshot the thunk value; mutable storage referenced
inside that thunk remains shared. All supplied operands evaluate before
invocation, including operands applied to a returned function. Fresh type schemes
preserve independent NTU specialization and dimensional payload identity.

The implementation uses the existing Baker Option recipes and closure
ingredients, through nanopass fan-out/fold-in. Tests inspect the returned graph's
selection, extraction, unit application, partial snapshot and application
obligation participants. They do not regenerate evidence inside the assertion.
No Option-specific witness or alternate traversal was added to Alex.

The full pipeline exposed two distinct callable defects:

1. Source elaboration assigned `fun () -> body` a function type without creating
   its logical unit formal. The fix retains that formal in the initial graph,
   like named unit functions and Baker-created closures. Raw and saturated tests
   check resident parameter nodes, their types and parents, body/result agreement,
   captured mutable storage and successive returned function boundaries.
2. Alex's binding/reference witnesses forwarded a lambda initializer before
   observing the binding's mutability. The fix routes the already settled mutable
   binding through the existing cell patterns. The cell contains the closure
   carrier; loading it supplies the immutable snapshot. Assignment replaces the
   cell's value, preserving the earlier snapshot. No new cast, closure packing
   convention, inferred lifetime or graph repair is introduced in Alex.

### Companion revisions

Use these revisions together. Composer `252f8d9` is the compiler/tooling
integration checkpoint. The earlier `defaultValue` compiler work
was incorporated in clef `94e28c7ba`; `f3bea0377` expanded its admission tests.

| Repository | Revision | Scope |
|---|---|---|
| clef | `259d4786c` | Deferred defaults, explicit unit formals, graph and negative tests |
| clef-lang-spec | `1d909905f` | Native bounds and incremental cutoff contracts |
| ClefAutoComplete (`fidelity`) | `cb46b5259` | Compiler-owned Option/query handoff; retired bridge remains reference material |
| lattice-analyzers | `b998a87b7` | Inherited rule corpus repairs and separate CCS.Editor projection gate |
| lattice-vscode (`fidelity`) | `2fc268a88` | Reviewed thin client baseline, real server/editor gates and Option regression |
| lattice-vim (`master`) | `4d7a947e2` | Reviewed Clef client registration and Neovim transport gate |
| clef-grammar | `f59fe1235` | Measured lexical grammar and TextMate regression gate |
| BAREWire | `33364d4a7` | Unchanged dependency/review baseline |
| Fidelity.Platform | `dcd3424ed` | Unchanged dependency; native Option oracles use CompilerSurface |
| Fidelity.UI | `b7ef6f90e` | Unchanged design triangulation baseline |
| lattice-vscode-helpers (`master`) | `821e1e72b` | Reviewed inherited Fable helper library; absent from the active thin client's dependencies |

Existing untracked grammar/client work was reviewed and tested as the necessary
self-contained tooling baseline before committing it. Unrelated working-tree
roadmap, platform and branding edits were excluded from these checkpoints.

### Validation

| Gate | Fresh result and retained local evidence |
|---|---|
| CCS | **311/311** service tests, including 35 deferred-default cases, 33 eager-default cases and 5 unit-formal cases; reachable negative cases require the exact code, effective severity and source range |
| Alex | **18/18**, including actual MLIR verification and standard lowering at 32/64-bit index widths, positional graph preservation, mutable closure cells and exact missing-input diagnostics; `/tmp/composer-alex-mutable-closure-tests.log` |
| Native callbacks | **9/9** fresh executables: both defaults, OptionPartials/Callbacks/Evaluation, GenericRecords, FunctionSnapshots, ListenerEntry and IgnoreValues; `/tmp/composer-callbacks-fsharp-6642dcd466ca4d2b9bbf822b9037dac0/`. The final strengthened `defaultWith` check also requires the replaced thunk to return its new value: `/tmp/composer-callbacks-fsharp-e11180f44cf84896986e226614cc8168/` |
| FidelityHello | **08a and 08b pass** compilation, native zero exit and exact manifest output; `/tmp/clef-option-waypoint-oracles/`. The final 08b source adds the same replaced-thunk assertion and passed again |
| CCS.Editor | **13 reported groups**, including actual cvc5 outcomes, source proofs and snapshot/edit invalidation; `/tmp/clef-option-waypoint-editor.log` |
| Analyzer corpus | **90/90**, all 12 inherited analyzers registered by the aligned CLI; `/tmp/lattice-analyzers-option-waypoint.log` and `/tmp/lattice-analyzers-cli.log` |
| Analyzer CCS projection | **6 accepted + 6 exact rejections**, dimensional hover, stale revision rejection and unsaved repair; `/tmp/lattice-ccs-options-59dd91027036454f80007af24d2270b8/evidence.json` |
| Client Option stdio | **10 negative edits with correction + 4 measured hovers**, exact codes/spans, versioned publication and clean server exit; `/tmp/lattice-option-waypoint-2XqsCu/result.json` |
| VS Code | **41/41** Node tests; real transport, TOML, F5 and CCS/cvc5 proof-view gates pass. Proof view: `/tmp/lattice-ccs-host-h6y00N/result.json`; F5: `/tmp/lattice-f5-host-TpHBU8/result.json` |
| Grammar / Neovim | **8/8** actual TextMate tests; headless Neovim registration/transport fixture passes. These do not establish compiler semantics |
| Regression harness | **8 groups**: native exit, launch errors, concurrent streams, process-tree timeout, blocked input, literal arguments, unmatched filters and CLI exit propagation; `/tmp/clef-option-waypoint-runner.log` |
| Proof/artifact regression | **50 SMT transfer cases** and **10 static-storage correspondence cases**, including false claims and artifact mutations; `/tmp/clef-option-waypoint-smt.log` and `/tmp/clef-option-waypoint-storage.log` |

The new [08a](../samples/console/FidelityHelloWorld/08a_OptionDefaults/) and
[08b](../samples/console/FidelityHelloWorld/08b_OptionDefaultWith/) FidelityHello
variants retain fixed expected output in the regression manifest. Native callback
fixtures check distinct exit codes for evaluation order, captures, payloads and
unit effects. Their harness also verifies every retained MLIR module with the
real `mlir-opt` and records compiler assembly hashes.

The full VS Code proof-view run preceded the final unit-formal fix. Its evidence
records the compiler it actually loaded. The focused Option stdio and analyzer
projection gates were rerun afterward; those final runs loaded CCS SHA-256
`a030ba651ec48a5c38a6fa22036df959e7c9f5c764d5d9da5a2073b6b2d17991`.
Temporary evidence directories may expire; the committed fixtures and commands
are the repeatable acceptance contract.

### Remaining boundaries

- Closure recipe migration remains C-01 work. Generic fold-in does not yet
  remap every capture source, structural lambda edges do not encode the complete
  capture relation, and current closure metadata is not a complete provenance
  or release proof. The existing `OptionFunctionPayloads` named-capture failure
  remains a separate regression: placement omits an environment for a nested
  named function without the required upstream capture-parameter rewrite.
- Passing native tests still contain interim closure casts and informational
  range findings. They do not establish the final two-value closure convention,
  complete closure obligations, or new dialect-family coverage. Each additional
  MLIR dialect needs its own admitted graph forms and preservation gates.
- The older `08_Option`, `11_Closures`, `12_HigherOrderFunctions` and
  `18_Generalization` baseline failed on reachable legacy formatter types/literal
  suffixes before these changes. The new variants do not hide or close that
  Platform migration. See the review's retained failure record.
- Lattice still needs a compiler-owned scope/completion query. No client-side
  intrinsic catalogue was added. The inherited analyzers remain a rule corpus;
  the new consumer gate does not turn them into Clef semantic authorities.
- Fractional numeric values and inverse dimensions have positive controls;
  written fractional dimensional exponents currently require the existing
  admission diagnostic. NFT admission, general CE/actor scheduling and selective
  incremental graph repair remain separate roadmap work.

### Reproduce

Coordinate shared compiler outputs and run from Composer unless another
directory is specified:

```sh
dotnet test ../clef/tests/Clef.Compiler.Service.Tests/Clef.Compiler.Service.Tests.fsproj
dotnet build src/Composer.fsproj
dotnet build src/Lattice.Server/Lattice.Server.fsproj
dotnet test tests/Alex.Tests/Alex.Tests.fsproj
dotnet run --project tests/CCS.Editor.Tests/CCS.Editor.Tests.fsproj
dotnet run --project tests/NativeCallbacks/NativeCallbacks.Tests.fsproj -- src/bin/Debug/net10.0/Composer OptionDefaultWith OptionDefaults OptionPartials OptionCallbacks OptionEvaluation GenericRecords FunctionSnapshots ListenerEntry IgnoreValues
dotnet fsi tests/regression/Runner.fsx -- --sample 08a_OptionDefaults --sample 08b_OptionDefaultWith
dotnet fsi tests/regression/RunnerTests.fsx
dotnet run --project ../lattice-analyzers/tests/Lattice.CCS.Integration/Lattice.CCS.Integration.fsproj
dotnet test ../lattice-analyzers/tests/Lattice.Analyzers.Tests/Lattice.Analyzers.Tests.fsproj -c Release
```

With Node.js 22, run `npm test` in clef-grammar and lattice-vscode/client;
`npm run test:options` and the documented editor-host gates belong to the latter.
Run `bash tests/run.sh` in lattice-vim for its transport fixture. The clients'
READMEs identify host/editor prerequisites and the scope of each gate.
