# Language coverage waypoints

This record connects the language/compiler and tooling revisions for bounded
implementation steps. A passing editor projection, solver query, MLIR verifier
or executable establishes its own boundary; none substitutes for the others.
The [review](Clef_Language_Completion_Review_2026-09-19.md) and
[incremental contract direction](Nanopass_Incremental_Contract_Direction.md)
retain the wider roadmap and unresolved contracts.

## C-01/C-04 immutable iteration bindings — 2026-09-20

Specification `25e2954` defines a fresh immutable source binding for every
integer loop iteration. The previous elaboration exposed its mutable counter
directly: source assignment could alter induction, captures shared later counter
updates, and a direct local function could fall outside immutable capture
admission. `ControlFlow.checkFor` now establishes a distinct immutable binding
from the hidden counter at each body entry. Source reads and capture origins
resolve to that binding; guard and step operations resolve to the counter.
Nested same-name loops retain separate identities. Source assignments receive
the existing CCS8009 diagnostic at the assigned value's exact span.

Eight regressions failed against the preceding compiler and now pass. They
inspect the actual initial graph artifact and returned saturated graph, including
counter/source separation, captures, nested identities and four located assignment
rejections. The native baseline stopped at the local function's mutable capture
(`/tmp/composer-loop-captures-before.log`); its unchanged fixture now passes.
Alex has no new witness, pattern, layout or semantic repair.

Companion revisions are clef `62f7eb9ce`, lattice-analyzers `c9b6cfb`,
lattice-vscode `753cdb1` and CAC `5c8e6120`, paired with this Composer checkpoint.

| Gate | Result |
|---|---|
| CCS | **544/544**, including eight new binding/capture/negative cases; `/tmp/clef-loop-bindings-full.log` |
| Native / MLIR | **3/3** LoopCaptures, RangeLoops and CountedLoops, with six/four/four groups; stock verification and native exit; `/tmp/composer-callbacks-fsharp-0ab5921cbece4b1788a239b1087f6d06/` |
| FidelityHello | **11b_LoopCaptures**, exact five-line output and zero exit; `/tmp/composer-loop-binding-fidelityhello.log` |
| Analyzer projection | **25 accepted / 30 exact rejections**; `/tmp/lattice-ccs-surface-dbffb4810abc4370a1342b28740bde91/evidence.json` |
| LSP | **34 diagnostic edits and repairs**, captured integer/source signature and definition at the loop identifier; `/tmp/lattice-surface-waypoint-SXt3em/result.json` |

Both final tooling gates loaded CCS SHA-256
`05caf164966447c51fb56f5c7fd4bf2b27e31e07c2415cb9bc16df869b26f5f9`.
CAC records the same source/counter distinction. Existing closure residence and
representation boundaries remain: native execution does not establish their
complete proof discharge or the final two-value closure representation.

## F-09 Result callbacks and integer range loops — 2026-09-20

Specification `808cb1a` records the native `Result.map`, `mapError` and `bind`
contracts in [Error Handling](../../clef-lang-spec/spec/error-handling.md#native-result-operations).
Use clef `4f1dea0b4`, lattice-analyzers `8051e73`, lattice-vscode `3d196f8`
and CAC `06ef01f2` with this Composer checkpoint.
The source implementation uses fresh quantified schemes and Baker case recipes.
`map` and `bind` select Ok; `mapError` selects Error. Supplied operands remain
eager, with the callback invoked once only in its selected case. Untouched
payloads retain their types, dimensions and resource identities; a changed
Result type may require reconstruction of its enclosing case. Bind returns the
callback's Result with its case unchanged. Placement retains the existing DU
lifetime contract rather than F-09's historical byte-size assumptions.

Stored partials retain their callback values and shared captured storage. Bare
aliases instantiate independently. Explicit type arguments are ordered
`map<'a,'b,'e>`, `bind<'a,'b,'e>` and `mapError<'a,'e,'f>`. The
[12-group native fixture](../tests/NativeCallbacks/ResultCallbacks.clef) and
[09a sample](../samples/console/FidelityHelloWorld/09a_ResultCallbacks/README.md)
cover both cases, callback factories and pipes, snapshots, independent measured
success/error types, record/function/unit payloads and propagation pipelines.
These are native acceptance oracles; source admission alone does not pass them.

The first native Result run stopped at placement: a reachable
`Result<'?467,'?469>` retained unresolved case payload types and had no settled
size. The diagnostic is retained in `/tmp/composer-result-native.log`, with the
project under `/tmp/composer-callbacks-fsharp-aaef35089c134ae0b08b52b1b7001b6b/`.
Monomorphization's bare-alias classifier recognized only resolved Option
intrinsics. It now admits resolved Result intrinsics through the same existing
specialization rules. Four regressions fail before the correction and pass
afterward; all Result source cases also reject open types in reachable closure,
formal and DU nodes. The original native expectations pass unchanged. No Alex
representation fallback or witness change was needed.

A separate source normalization handles named, closed, unstepped integer
`for value in first .. last` loops, including whole-range and endpoint
parentheses. It reuses the existing counted-loop elaboration, preserving
first-before-last evaluation and resolved induction references. A lexical
`op_Range` binding excludes this normalization, as do stepped ranges; their
existing ForEach path is not newly admitted. This is not general iterable or
range-operator support. Source induction mutability and per-iteration captures
remain separate contracts.

| Gate | Current checkpoint |
|---|---|
| CCS | **536/536**, including 46 Result cases with 19 exact negatives and eight range-loop cases; `/tmp/clef-result-alias-full.log` |
| Native / MLIR | **2/2 fresh executables**, 12 Result and four range-loop groups; retained modules pass stock verification; `/tmp/composer-callbacks-fsharp-f66bfe0235064c1ba147fb9eab8719de/` |
| FidelityHello | **09a_ResultCallbacks passes** fresh compilation, zero exit and exact six-line output; `/tmp/composer-result-alias-fidelityhello.log` |
| Analyzer projection | **24 accepted / 28 exact rejections**; `/tmp/lattice-ccs-surface-513dd2dde02d4c98ad5294c8aa22169a/evidence.json` |
| LSP | **32 diagnostic edits and repairs**, four Result hovers plus unit loop result and integer induction hovers; `/tmp/lattice-surface-waypoint-6aFHHu/result.json` |

Both final tooling gates loaded CCS SHA-256
`c711fc455867ae963984f4388a4a7776cb109a13d7505a4277772e09d8816659`.
Alex implementation, solver transfer and transport interfaces are unchanged;
their preceding gates remain applicable. No new dialect-family coverage is
claimed. CAC's handoff now includes the independent Result payload contract.

## C-04 optional folds and counted-bound order — 2026-09-20

Clef `a8c5229df` implements the native fold contracts adopted in specification
`97a8e08`. `fold` takes folder/state/option; `foldBack` takes folder/option/state.
State and payload have independent NTU types. Both retain None state, invoke the
folder once for Some, preserve eager operand order and snapshot both partial
frontiers. Bare aliases specialize independently. A function-valued state remains
separate from the operation's three-argument boundary.

The first native fold gate found a missing dominance relationship in completed
residuals: their shared state reference was first realized inside Some and then
recalled from None. Four graph tests reproduced the defect. Baker now places the
residual's local operands before the conditional, preserving their identities.
The original native expectations pass unchanged; Alex needed no adjustment.

A separate counted-loop oracle found finish-before-start evaluation. The graph
now orders the start initializer before the finish initializer, as the language
specifies. Four native groups cover ascending, descending, zero-trip and consumed
unit results. Induction-variable mutability and per-iteration closure identity
remain separate work; this correction establishes bound ordering only.

| Gate | Fresh result |
|---|---|
| CCS | **482/482**: 44 fold cases (16 exact negatives, four residual-dominance regressions) and two counted-bound graph cases; `/tmp/clef-option-fold-dominance-full.log` |
| Fold native / MLIR | **15 groups**, fresh executable and stock retained-module verification; `/tmp/composer-callbacks-fsharp-1862255ad1844f5f8e7d3d5eb365401c/` |
| Counted-loop native / MLIR | **4 groups** passed before the isolated fold-residual correction; `/tmp/composer-callbacks-fsharp-d057c5ef1de041e7a051a6701d25ae67/CountedLoops/`. Its pre-fix reversed-order observations remain in `/tmp/clef-counted-loops-2irghu7t/` |
| FidelityHello | **08e_OptionFolds** compilation, native exit and exact six-line output pass; `/tmp/composer-option-folds-final-fidelityhello.log` |
| Analyzer-facing projection | **20 accepted / 22 exact rejections**; `/tmp/lattice-ccs-surface-880ec51c241f4fd2b2207023917230bb/evidence.json` |
| LSP | **26 diagnostic edits and repairs**, six fold hovers; `/tmp/lattice-surface-waypoint-F342xg/result.json` |

Companion revisions: lattice-vscode `90f1412`, lattice-analyzers `82703c7`,
ClefAutoComplete `b16bdde8`. Final projection gates loaded CCS SHA-256
`2e733809e905e6bdb8e0a4a8872709a665a37fb858d10151477a8831b3d04207`.
The existing Alex, solver-transfer and transport implementations are unchanged;
their preceding component gates remain applicable.

## Structured unit results and lexical math identities — 2026-09-20

Alex now preserves unit results for matches and while loops as well as
conditionals. Each witness reads the settled unit type and composes the existing
unit-result pattern after its control-flow operations. While-region terminators
now belong to the Pattern layer. The native fixture first reproduced missing
unit arguments and stored bindings, then passed unchanged after the fix.

Clef `40cabe767` separately preserves explicit module members and function-valued
fields named `Math.sin` ahead of the intrinsic fallback. The math source gate
checks dimensionless intrinsic admission, nine exact negative cases, lexical
identity, source-level higher-order forms and existing literal evidence.

| Gate | Fresh result |
|---|---|
| CCS | **436/436**, including 17 math source cases; `/tmp/clef-math-sine-full.log` |
| Alex | **25/25**; `/tmp/alex-unit-expressions-tests.log` |
| Native / MLIR | **2/2** fresh executables: UnitExpressions (six groups, direct/stored/nested match and loop values) and OptionIteration. Both retained modules pass stock verification; `/tmp/composer-callbacks-fsharp-d84826f373ca4f6d8b79c8b548c88ab5/` |
| Analyzer-facing projection | **17 accepted / 19 exact rejections**; `/tmp/lattice-ccs-surface-3f212d31214543a8a30946acd65e66e8/evidence.json` |
| LSP | **23 diagnostic edits and repairs**, including two lexical Math measured-result repairs; `/tmp/lattice-surface-waypoint-gzPtBT/result.json` |

Companion revisions: lattice-vscode `c4a6e37`, lattice-analyzers `577ed02`.
Both projections loaded CCS SHA-256
`f4b6115f8ae6b4e9f3aaa2f6e558650b5192091a921c8341ff8ab89f981f1e86`.
The active CAC handoff and unchanged transport/grammar revisions still apply.

The [prospective math oracle and prerequisite record](../tests/NativeMath/README.md)
is explicitly unregistered and has not passed native compilation. There is no
new math witness in this checkpoint. Scalar real selection and a typed target
provider must be settled upstream; existing `Fixed64`, record `SettledSlot.Real`
and a link declaration alone do not constitute that call contract. The record
pins the relevant roadmap and concrete missing facts before implementation.

## C-04 optional iteration and unit-valued conditionals — 2026-09-20

Clef `1c20fad11` adds `Option.iter` through a fresh native scheme and the existing
Baker recipe path. Both operands are eager; the action consumes the payload once
only for Some, and both branches return unit. Direct applications, pipes, stored
partials and independently specialized bare aliases retain that contract.

The native gate exposed an Alex gap: a unit-typed conditional retained its effects
but returned `TRVoid`, so its result could not be passed directly to another
function. Baker's unit node and incidence were correct. The conditional witness
now observes the settled unit type and composes a pattern that preserves the
control-flow operations followed by the existing unit literal representation.
Missing operands and failed patterns remain errors; no graph repair or traversal
change is involved.

| Gate | Fresh result |
|---|---|
| CCS | **419/419**, including 24 iteration cases (12 exact negatives); `/tmp/clef-option-iteration-full.log` |
| Alex | **25/25**, including four unit-result component cases: effect order, missing-operand failure, rejection of a preexisting value, stock MLIR verification and LLVM lowering; `/tmp/alex-option-iteration-unit-tests.log` |
| Native / MLIR | **2/2** fresh executables: OptionIteration (12 groups) and IgnoreValues; both retained modules pass stock `mlir-opt --verify-each`. `/tmp/composer-callbacks-fsharp-c4721c33cc594916a29fd001f98472ee/` |
| FidelityHello | **08d_OptionIteration** passes compilation, native exit and exact six-line output; `/tmp/composer-option-iteration-unit-fidelityhello.log` |
| Analyzer-facing projection | **15 accepted / 18 exact rejections**, plus retained capture projections; `/tmp/lattice-ccs-surface-757480766e2945459fd2527a491adb50/evidence.json` |
| LSP | **22 diagnostic edits and repairs**, including five iteration hovers; `/tmp/lattice-surface-waypoint-dcmG9g/result.json` |

Companion revisions: lattice-vscode `8a57357`, lattice-analyzers `c91f270`, and
ClefAutoComplete `38dceb27`. Both external projections loaded CCS SHA-256
`e3b450ba770c8bbb523e51af47fd161631561999f7b536282c7e68076a7cda91`.
CCS.Editor, server transport, grammar and Neovim interfaces are unchanged.
This increment establishes unit-valued conditionals; other structured unit-result
witnesses and the remaining collection surface retain their own gates.

## C-04 optional alternatives and temporal range facts — 2026-09-20

Clef `f5fdbc966` adds `Option.orElse` and `Option.orElseWith` through fresh native
schemes and the existing Baker Option recipes. They preserve the selected option,
including None. Both operands are evaluated eagerly; the deferred producer runs
only when the input is None. Stored partials preserve their initial fallback or
producer value while retaining shared captured storage. Bare aliases specialize
independently, including measured payloads. No new Alex intrinsic path is needed.

Native testing exposed two existing range defects that this increment also fixes:

- A closure wrote `300` into a cell initially containing `1`, but a subsequent
  read retained the pre-call `state < 10` refinement. Its `[1,9]` range caused an
  incorrect sixteen-to-eight-bit truncation. CCS now computes finite may-write
  summaries across direct, transitive, recursive and value calls, invalidating
  affected facts in operand evaluation order. Earlier snapshots, pure calls and
  unrelated bindings retain their valid refinements.
- Combining saved Boolean checks replayed their earlier observations as facts
  about current mutable storage. The Option oracle observed the correct trace
  `1246` but incorrectly narrowed it to eight bits in the final conjunction.
  Saved predicates no longer reinstate those mutable-definition bounds. Effectful
  comparison operands and predicate calls also preserve observation timing.

The original native expectations were retained. Alex continues to consume settled
ranges; neither defect was repaired by changing its casts or traversal. The
may-write calculation is currently an internal finite analysis, not a claim of
incremental effect-ledger support or completion of the mutable-cell representation.

| Gate | Fresh result |
|---|---|
| CCS | **395/395**, including 43 Option-alternative cases (22 exact negatives) and 18 temporal range cases |
| Native / MLIR | **4/4** fresh executables: CallEffects (13 groups), OptionAlternatives (25 groups), DirectCaptures and OptionDefaultWith; all retained modules pass `mlir-opt --verify-each`. `/tmp/composer-callbacks-fsharp-60dbfbb289b9450b8da165fc2575a883/` |
| FidelityHello | **08c_OptionAlternatives** passes compilation, native exit and exact output; `/tmp/composer-option-effects-fidelityhello.log` |
| CCS.Editor | **15 groups**, including `[1,300]` post-write ranges, unsaved `[1,700]` updates, stale-hover rejection and immutable earlier snapshots; `/tmp/ccs-editor-final-effects-full.log` |
| Analyzer-facing projection | **12 accepted / 14 exact rejections**, plus direct-capture signatures and origins; `/tmp/lattice-ccs-surface-b8c386aa62b54f63acef5d128860d8f0/evidence.json` |
| LSP | **18 diagnostic edits and repairs**, eight optional-result/partial hovers and retained capture projections; `/tmp/lattice-surface-waypoint-hI8Wxm/result.json` |
| Proof/artifact controls | **50 SMT transfer / 10 static-storage correspondence** cases; `/tmp/composer-option-effects-smt.log`, `/tmp/composer-option-effects-storage.log` |

Companion revisions: lattice-vscode `68d3cc1`, lattice-analyzers `83256f6`, and
ClefAutoComplete `971e5e93`. The two external projection gates independently load
CCS SHA-256 `06eb3c1b7d6a3ecc8f9e1692e299ff6492e9d925c1d9a4be7bca08798bb2acd0`.
Alex source and client protocol/grammar are unchanged in this increment; the
earlier component and transport revisions remain applicable. Source and native
fixtures retain the failed behaviors as regressions. Complete closure residence,
fractional dimensional-exponent admission and remaining collection families are
still separate work.

## C-01 immutable direct captures — 2026-09-19

Clef `cdbaf8636` moves eligible named-function capture passing into Baker
ingredients, a recipe and nanopass fan-out/fold-in. Complete-use admission permits
direct calls and recursive forwarding; named function value uses, partial uses,
opaque references and mutable capture frontiers remain unconverted. Capture
formals and operands retain NTU types, source identity, structural/reference
incidence and explicit capture-origin provenance. Independent resident graph
relations survive the transformation. Returned anonymous closures capture the
new formal without collapsing their own callable boundary.

This Composer companion projects resolved local binding identity into a shared
target symbol for definitions, ordinary/saturated calls and hardware step
references. The native oracle exposed the old collision between independent
local functions with the same name; those duplicate names remain in the fixture.
Module/external symbols and settled native callback address plans retain their
existing spelling. Alex's traversal and witness responsibilities are unchanged.

CCS.Editor retains the source callable signature and follows the explicit capture
origin to the source declaration for navigation. Hidden formals remain visible in
the semantic graph. Companion peering is pinned by lattice-vscode `f839e28` and
lattice-analyzers `9a762be`; both test the actual compiler projection. CAC's active
handoff, grammar and Neovim transport interfaces are unchanged from the preceding
waypoint, so their recorded revisions remain applicable.

| Gate | Result |
|---|---|
| CCS | **334/334**, including 17 direct-capture cases: recursion, nested capture identity, shadowing, source signatures/origins, exact incidence, independent-edge preservation, idempotence and located dimensional rejection |
| Alex | **21/21**, including three new callable-symbol cases, ordinary/saturated witness calls, hardware reference projection and preserved native address plans |
| Native | **3/3** fresh executables: DirectCaptures, OptionDefaultWith and ListenerEntry; each retained module passes stock `mlir-opt --verify-each`. `/tmp/composer-callbacks-fsharp-a97319380d584c4981172fac9e8b30d9/` |
| FidelityHello | **11a_DirectCaptures** compiles, exits zero and matches exact manifest output; `/tmp/composer-direct-captures-fidelityhello.log` |
| CCS.Editor | **14 groups**, including source callable signatures, measured results and captured-variable definition origins; `/tmp/clef-direct-captures-editor.log` |
| LSP | Existing 10 Option negative cases plus exact CCS8040 direct-call rejection, source signatures, measured result and real go-to-definition; unsaved repair restores the views. `/tmp/lattice-surface-waypoint-5ky2GK/result.json` |
| Analyzer-facing projection | Existing 6 accepted/6 rejected Option cases plus direct-capture views/origins and exact dimensional rejection; `/tmp/lattice-ccs-surface-298468aa4259456da14fd8ddfc487015/evidence.json` |
| Proof/artifact controls | **50 SMT transfer** and **10 static-storage correspondence** cases pass; `/tmp/composer-direct-captures-smt.log`, `/tmp/composer-direct-captures-storage.log` |

The two external tooling gates independently loaded CCS SHA-256
`9d946c2a330b819bff364f36e541f1a80f161b6545c411d2b28781a4c59044fb`.
Temporary evidence can expire; the committed tests and manifests remain the
repeatable contract. This is the immutable direct form, not completion of C-01's
materialized environments, lifetime/release obligations or final two-value closure
representation. The [mutable cell direction](Direct_Capture_Cell_Contract.md)
records the next storage, call-effect and residence requirements. The existing
mutable `OptionFunctionPayloads` failure remains open, as does Platform formatter
migration needed by the older `11_Closures` oracle.

## C-01 fold-in reference identity — 2026-09-19

Clef `70f233fcf` redirects resolved variable definitions and Lambda/Lazy/Seq
capture sources through the same replacement map as structural references and
hyperedges. Capture mode, type, source range and unresolved references are
preserved; shadowed names remain distinguished by definition identity. Both
surviving nodes and nodes introduced by another recipe follow the replacements.
The shared `remapKindReferences` operation also supports a recipe's explicitly
scoped substitutions.

Six focused replacement cases and the full **317/317 CCS** suite pass
(`/tmp/clef-foldin-references.log`). **13 CCS.Editor groups** pass, including
resolved definitions and immutable snapshots (`/tmp/clef-foldin-editor.log`).
The protocol and client interfaces are unchanged; their companion revisions
remain those recorded below. This is a reference-preservation prerequisite,
not completion of closure layout/lifetime obligations or a new native gate.
The earlier note that generic fold-in omits capture-source remapping is closed
by this waypoint; other C-01 boundaries remain.

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
