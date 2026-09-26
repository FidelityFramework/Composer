# Language coverage waypoints

This record connects the language/compiler and tooling revisions for bounded
implementation steps. A passing editor projection, solver query, MLIR verifier
or executable establishes its own boundary; none substitutes for the others.
The [review](Clef_Language_Completion_Review_2026-09-19.md) and
[incremental contract direction](Nanopass_Incremental_Contract_Direction.md)
retain the wider roadmap and unresolved contracts.

The [C/F completion ledger](C_F_Completion_Ledger.md) turns the current PRD
criteria into an ordered implementation and acceptance inventory. It records
remaining work and executable gates; it does not add a passing result.

## Main integration and selected-match checkpoint — 2026-09-26

The [C/F checkpoint report](C_F_Checkpoint_2026-09-26.md) maps delivered behavior
and the next required gates to every C-01–C-07 and affected F-01–F-10 PRD. All
Composer changes now reside in the main worktree. The explicit-demand source
foundation from `d8effc6` has been integrated with the current Lazy, callable,
sequence and proof protocols; it is no longer only an isolated worktree change.
Final main source v20 passes **1,355/1,355**, and Alex v20 passes **223/223**,
including the raw constant CPU/FPGA component controls. The eager-callee factory
regression's unnecessary Baker eta wrapper is corrected. The default editor
suite, analyzer integration and actual live LSP gate pass, including clean server
shutdown. Exact closure/Lazy source-provenance relations preserve public capture
references while generated storage/cache reads remain internal.

The private v19 compiler is `/tmp/composer-checkpoint-v19/compiler`; source
compiler SHA-256 is
`fd0d1f18c43a54a83eab92daa171d645d9ecddf604bb4892e5271cc1aadbeac0`.
Specification commit `586010e` and site commit `8b6bd57` record the supporting
contract corrections. Clef `14fb7c7` and the Composer commit containing this
waypoint close the main-branch checkpoint. The final v20 snapshot and companion
revisions are in the report. The v20 projection correction passes all source,
editor, analyzer and live LSP gates, plus fresh native14/16h confirmation; the
broader v19 executions below retain their actual cohort identity.

Recorded results retain these cohort boundaries:

| Gate | Recorded result and artifacts |
|---|---|
| Final integrated source v20 | **1,355/1,355 passed**, `/tmp/clef-checkpoint-full-v20.log`. |
| Final physical composition v20 | Alex **223/223 passed**, including raw constant CPU/FPGA and backend requirement controls. This is component evidence, not execution on an FPGA device. |
| Final editor/analyzer/live LSP v20 | Default editor21 groups and analyzer141 reported checks pass; live LSP passes with clean shutdown. `/tmp/composer-checkpoint-live-v20/evidence.json`. |
| Final native confirmation v20 | **14 and16h: 2/2 compiled and executed**, unchanged oracles. `/tmp/composer-checkpoint-native-v20.log`. |
| Native v19 Lazy | **14,14a,14b: 3/3 compiled and 3/3 executed**. `/tmp/composer-lazy-main-final.log`. These cover memoization/instance/cell identity, typed scalar results and declared static-string backing. |
| Native/artifact differential v19 | Full and pruned **01,16a,16h: 6/6 compiled and 6/6 executed**. MLIR, LLVM IR, stdout and stderr are byte-identical between modes; all six outputs match byte-exact manifest oracles. All **nine** PSG pairs preserve exact included nodes and complete joint evidence. `/tmp/composer-pruned-differential-main-v19b.log`, `/tmp/composer-pruned-differential-main-v19b-validate.log`. |
| Native v19 selected matches | **5/5 passed**: GuardedMatch, LiteralMatch and TerminalMatchSuccess exit 0 with exact empty streams; terminal pattern/guard failures exit 1 with their exact source-located diagnostic. `/tmp/composer-checkpoint-matches-v19.log`. |
| Proof/process controls v16 | All eight runner gate groups plus parallel-runner checks; StaticStorage **10**; SMT **85** across native, real, integer, loop, dimension, layout and continuation cohorts. These retain their v16 cohort and are not relabeled as v19/source20 executions. |

Selected-match elaboration now places payload reads, original source bindings
and guards inside their selected source scope. Baker preserves rule order and
false-guard fallthrough; Alex observes guard-free, binding-free arm metadata
through the actual Huet occurrence. Terminal `Require` checks retain exact
condition/input/type/body/frontier participants. The selected-body result repair
and backend diagnostic realization are verified by the native success and
failure controls, replacing the earlier pending failure-path result.

Literal patterns use ordinary typed source equality before boolean selection.
The native control distinguishes Unicode, fractional floating values, equal
separately formed strings, same-length unequal and empty strings, signed zero,
unit guard order and runtime NaN. Source retraction tests preserve dimensional
and selected-input authority. These results do not implement every source
pattern form: the report assigns the remaining Or/As/And/list/array representation
and selection work to the source checker/Baker owners and their storage contracts.

Source static-pool proofs cover the declared pool's actual backing, layout and
capacity. Diagnostic globals and helper code introduced by the backend are
additional whole-image resource contributions. Their accounting belongs to the
Platform/backend contract; neither the source pool proof nor F-07 bitwise
acceptance supplies it implicitly.

The integrated eager syntax and conditional source-demand relations remain
distinct from ordinary call-by-need storage and native demand delivery. Continue
implementing the specification's [ordinary call-by-need](../../clef-lang-spec/spec/expressions.md#default-demand-and-sharing)
and [explicit eager](../../clef-lang-spec/spec/expressions.md#eager-expressions)
requirements within the existing C/F PRDs, alongside every other C/F acceptance
criterion recorded in the ledger. None of these bounded results marks a C PRD
Complete.

## Pruned artifacts and complete callable consumption — 2026-09-26

`--prune-intermediates` is available on the compiler and bounded regression
runner. It selects nodes before kind/type formatting, starting with reachability
and declared roots and preserving the complete transitive participant set of
joint evidence. It leaves the compiler graph and its reachability flags intact.
Full dumps remain the default; the view records source/emitted counts and
external or missing references. See the [artifact policy](Regression_Check_Policy.md#pruned-diagnostic-artifacts).

Complete callee consumption now has its own joint residence evidence. It cites
the actual allocation, covering activation, callee and every target's real
formals, actuals and body. `CallableOrigins` distinguishes a complete invocation
from a known partial alternative without conflating partial formation with an
opaque origin. The mixed captured/plain higher-order regression proves bounded
consumption and retracts it after an opaque alternative or missing argument.
This resolves the actual11 lifetime failure; establishing its mixed calling
convention remains part of the active source dispatcher work with actual
environment identity and complete ingress proof.

Verification against the coordinated CCS source-v3 / test-v4 snapshot and
Composer build in `/tmp/composer-callable-pruned-build-v1.log`:

| Gate | Evidence |
|---|---|
| Focused source cohort | **370 passed**, including projection, complete callable flow/consumption, Lazy layout/residence and corrected canonical closure/Option/Result/Seq assertions. `/tmp/clef-lazy-runtime-flow-focused-v4.log`. Original sample sources and native oracles were unchanged. |
| Full main CCS | **1,141 passed**, `/tmp/clef-main-full-after-carrier-flow-v4.log`. This is separate from the isolated explicit-demand worktree below. |
| Full Alex | **149 passed**, zero skips, including the three new common callable-flow cases. `/tmp/composer-callable-pruned-alex-v1.log`. |
| Bounded .NET runner | All parallel-runner checks passed, including limits, actual overlap, ordered results, barriers, streams, input and descendant/orphan cleanup. `/tmp/composer-pruned-runner-tests-v1.log`. |
| Source-to-native differential | **6/6 compilation and 6/6 execution passed**, zero skips: full and pruned **01**, **16a**, **16h**, using one private compiler snapshot with three worker processes. MLIR, LLVM IR, stdout and stderr are byte-identical between modes. All six stdout streams match the byte-exact manifest oracles. `/tmp/composer-pruned-differential-v1/run.log`; snapshot hashes, selection and per-job artifacts are retained there. |
| Artifact integrity | All **nine** paired PSG checkpoints preserve each emitted node exactly and retain every participant of included joint evidence. `/tmp/composer-pruned-differential-v1/validate.log`. |

Final PSG serialization sizes from that differential:

| Sample | Full nodes → pruned nodes | Full bytes → pruned bytes |
|---|---:|---:|
| 01 | 31,910 → 68 | 28,610,573 → 134,431 |
| 16a | 13,850 → 8,835 | 12,176,045 → 7,896,156 |
| 16h | 10,036 → 5,177 | 8,890,675 → 4,843,084 |

Recorded compilation times came from contending independent jobs; they are not
a controlled speedup benchmark. The result establishes reduced serialization
and unchanged compiled behavior for the selected controls. It does not establish
an incremental partition or complete the C-series language gates.

A subsequent serialization probe found that the structural graph was formatted
into `01_psg0.json` before reachability and then overwritten by the intended PSG0
checkpoint. Removing that first write preserves the retained phase artifact.
The private compiler in `/tmp/composer-pruned-differential-v2/compiler`, built
against the later Lazy/callable source cohort, recorded these serial Hello runs
in `/tmp/composer-pruned-differential-v2/timing.log`:

| Artifact mode | Wall time |
|---|---:|
| No intermediates | 6,242 ms |
| Pruned intermediates | 7,898 ms |
| Full intermediates | 69,499 ms |

The earlier serial probe recorded 6,338 / 27,813 / 87,887 ms respectively.
These are individual local measurements, not a controlled repeated benchmark.
The later cohort preserves native01/16a in both artifact modes but exposes a
16h frame-read callable-provenance regression in both modes. Its run must not
replace the passing six-job differential above; the complete source access
relation and its regression checks are being corrected together.

## Explicit demand source foundation — 2026-09-26

The initially isolated Clef worktree `codex/evaluation-foundation-20260926`, at local commit
`d8effc6e6e10` based on `f898c97e11c5`, carries the source foundation for the
[evaluation contract](Evaluation_Strategy_Contract.md). The `eager` keyword is
type-transparent syntax with the same prefix extent as `lazy`. Parser, checker,
typed-tree projection, remapping and monomorphization preserve the actual marker
and operand. Value-origin readers preserve callable and sequence identity through
the marker without treating formation as invocation or enumeration.

Baker projects local conditional demand relations at bindings, actuals,
components and expression occurrences. An annotation may expose a direct marker;
an alias, unselected branch or nested deferred computation may not. Projection
retracts changed or retired premises, records incomplete premises explicitly and
runs again after continuation rewriting. A source node's reachability is not an
execution decision. Explicit eager actuals at a partial formation are excluded
from the later-call deferred-argument set; their original identities remain in
the saturation contract.

The application staging recipe now preserves each actual's owning call. Its old
all-argument sequential prelude moved later supplied expressions ahead of the
body producing the returned callable. The replacement creates the distinct calls
and leaves their demand/order obligations at those boundaries. The old
`Supplied expressions are evaluated in order before either staged call` assertion
has been replaced with source ownership and explicit eager-stage tests, according
to the specification's [eager expression rules](../../clef-lang-spec/spec/expressions.md#eager-expressions).

Reifying a named function value likewise retains its declaration's real arity;
the wrapper cannot absorb a function-valued result by counting all type arrows.
First-boundary evidence includes every known alternative and its actual formals,
including partial origins. An opaque alternative or an earlier returned-callable
boundary keeps later actuals pending instead of assigning them to an unproved
common frontier.

Source verification on this isolated worktree:

| Gate | Evidence |
|---|---|
| Parser, source demand and callable cohort | **82 passed**: parser extent/offside/ranges, type/range identity, specialization, direct/annotated frontiers, aggregate order, callable/sequence identity, distinct application stages, partial formation, mixed/opaque alternatives, reification and edit retraction. `/tmp/clef-eager-reification-focused-v1.log`. |
| Actual continuation rewrite | Included in that cohort at declared **32-bit and 64-bit** profiles. Both cloned markers cite their actual frame operands; retired markers retain no demand authority. |
| Full foundation assembly | **1,111 passed**, `/tmp/clef-eager-foundation-full-v3.log`. |

These are source-graph gates. Delivery of the specification's
[ordinary call-by-need](../../clef-lang-spec/spec/expressions.md#default-demand-and-sharing)
and [explicit eager](../../clef-lang-spec/spec/expressions.md#eager-expressions)
requirements continues through shared deferred storage, operation-specific Baker
demand saturation, passive Alex witnessing, native effect traces and editor
invalidation. These obligations belong to the existing C/F PRDs, as mapped in the
[evaluation contract](Evaluation_Strategy_Contract.md#evaluation-requirements-through-the-owning-stages)
and recorded in the [completion ledger](C_F_Completion_Ledger.md). The source cohort does not stand in
for those executable acceptance gates. The source foundation is now integrated
with main's callable/sequence/Lazy protocols as recorded in the checkpoint above;
source/Alex/editor/live-LSP and the selected native gates pass on v19. The later
source20 public-projection change retains its own verification boundary. The
counts in this subsection remain results from the earlier isolated foundation,
not final main counts.

## Canonical sequence carriers through native 16h — 2026-09-26

The coupled carrier migration in `Composer-callable-transport` now compiles and
executes the unchanged `16h_SequenceApplications` sample. All seven semantic
groups exit successfully, with byte-for-byte `ExpectedOutput.txt` agreement and
empty stderr. This supersedes the pending native-16h result in the previous
checkpoint below; it is a bounded C-01/C-06/C-07 gate, not completion of the full
C-series or the separate evaluation-strategy contract.

Baker publishes exact sequence alternatives and their connected physical
families. Each member retains its actual generator, formal, complete slot
representations, source values and proof participants. Common state/current
storage is source-placed; an empty producer does not acquire a current value.
Alex transports separate generator-code and actual-environment operands through
calls, results and value occurrences, including joins with multiple admitted
origins. It checks source family authority and does not choose an origin by
frame size, element type or a representative member. Raw generator/result
storage remains distinct from a source sequence value.

Fresh enumeration has an explicit source-owned representation-copy contract:
exact template and destination storage, extent, alignment, address space,
lifetime, nonoverlap and capture identity. The portable opaque `memref.copy`
lowers through the pinned stock pipeline to LLVM memcpy. Copying representation
bytes grants no typed initialization or successful-current-read evidence;
fresh-entry reset establishes only control state. Immutable captured descriptors
retain their original shared values/cells. Formation and copying do not invoke a
producer or force a deferred payload.

The last integration defect was duplicate emission of the same materialized
code Lambda through both its source `ClosureValue` and canonical named binding.
Traversal now reuses global coverage only for that exact definition-only Lambda
identity: a capture-free Lambda with its actual immutable sole-child binding and
no legacy closure-value construction. Local closure/environment formation and
shared body values still follow their actual Huet occurrence. The new regression
failed against the preceding binary, then passed alongside the existing
shared-body/block-argument test and stock MLIR verification.

| Gate | Evidence and limit |
|---|---|
| Focused CCS source cohort | **82 passed**, `/tmp/clef-sequence-boundary-v10.log`: exact origins/families, template copy/storage rejection, continuation meets, mutable callable storage, and additive Lazy source/range contracts. This is not a full CCS-suite result. |
| Composer build | Passed, `/tmp/composer-sequence-carrier-v6.log`, against the coordinated CCS DLL with project-reference rebuilding disabled. |
| Full Alex assembly | **146 passed**, `/tmp/composer-sequence-alex-full-v1.log`, including family/operand retraction, scoped transport, 32/64-bit stock memcpy lowering, mutable callable controls and the duplicate-definition regression. |
| Actual native 16h | Stock MLIR verification, LLVM lowering and native link passed; native exit **0**, exact existing output and empty stderr. Artifacts and compile/runtime logs: `/tmp/composer-sequence-native-v3/`. Source and expected output were unchanged. |
| Four-worker native controls from the same compiler snapshot | **01 and 16a compiled and executed with exact output**. **11 failed source environment admission** (`CCS8403`, `UnsupportedConsumer` at `Closures.fs:86`); **12 failed Alex callable-boundary admission** (missing settled carrier contracts). No samples were skipped. The coupled migration must restore both controls before its checkpoint gate closes. `/tmp/composer-sequence-controls-v1/runner.log`, 98.4 seconds. |

This is a working-tree snapshot, not a committed release: Composer branch
`codex/callable-transport-20260926` is based on
`db4bc8b01bb647c251815d5a08ac54889c7a2788`; the coordinated Clef working tree is
based on `f898c97e11c5a45da91899901bac29250a8d96a2`. Companion revisions remain
Fidelity.Platform `d42c9988f7fd`, BAREWire `6e212487aef5`, and clef-lang-spec
`280efcc551a4`, with the specification's current working-copy clarifications.
The tested binaries precede subsequent Lazy-factory source additions and the
isolated explicit-demand foundation above. Those changes require their own
coordinated build and combined gates; this sample's existing formation assertions
do not establish ordinary call-by-need behavior. A private copy of the verified
compiler cohort, its file hashes and control-run records are retained under
`/tmp/composer-sequence-controls-v1/` so later source changes cannot alter that
cohort's evidence.

**Subsequent demand-contract correction.** The older source fixture requiring
`Seq.map (effect; fun value -> value)` to have no retained environment assumed
the supplied computation had already executed at formation. The governing
[default demand and sharing contract](../../clef-lang-spec/spec/expressions.md#default-demand-and-sharing)
leaves an unused ordinary argument, including its effects, unevaluated. Code
identity alone cannot erase that computation. The revised source regression
requires its original immutable binding, effect and capture initializer to
remain one shared identity, and requires body reads to retain that exact slot.
A separate positive case still requires a genuinely pure stateless callable
to use plain code without an invented environment. Sequence-result destination
preparation likewise retains each original ordinary actual, introducing only
destination storage before the actual invocation. Native event-order and
unused-argument observations must verify the specification's
[ordinary call-by-need](../../clef-lang-spec/spec/expressions.md#default-demand-and-sharing)
and [explicit eager](../../clef-lang-spec/spec/expressions.md#eager-expressions)
rules through the application and sequence contracts in
[C-02](PRDs/C-02-HigherOrderFunctions.md), [F-04](PRDs/F-04-CurryingLambdas.md)
and [C-07](PRDs/C-07-SeqOperations.md). The earlier native sample pass above does
not prove these corrected demand rules.

## Callable settlement and coordinated verification — 2026-09-26

**Intermediate implementation checkpoint under the C-series delivery contract.**
This entry supersedes the source-settlement diagnosis in the earlier staged-Seq
entry below. The actual 16h source now reaches settled graph construction without
effective source errors. Its native acceptance remains the next integration
gate, with its source and expected output preserved.

Clef now shares exact callable-origin and formal/actual relationships across
range, activation, environment and sequence analyses. Factory results retain
caller-owned destinations, eager capture snapshots and the actual source and
destination lifetime participants. Residence settlement validates the joint
relationships, including unknown uses and retraction, rather than treating a
finite capture list or an equal-sized frame as a lifetime proof. Stateless
callables receive code declarations without an invented empty environment.
Immutable operation aliases retain their polymorphic source declarations and
specialize concrete uses through the existing monomorphization path.
The editor gate also exposed source-name/definition loss during stateless code
promotion. Each rewritten reference now retains its own source declaration in
validated graph provenance, even when several aliases share one code binding.
Editor projection consumes that relationship; native invocation still uses the
promoted code target. The original hover/navigation assertions are preserved.

Composer preserves actual Huet occurrence and scoped operand recall while
adding typed callable operand composition and portable function operations with
ordered multiple results. These foundations carry code and environment as
distinct operands and validate the CCS-published carrier. Their component tests
include real MLIR verification and LLVM lowering for 32-bit and 64-bit profiles.
They do not reconstruct capture, layout, lifetime or source semantics in Alex.

The coupled switch of the remaining witnesses to that callable convention is
preserved on `codex/callable-transport-20260926` in the separate
`Composer-callable-transport` worktree. Its integration must settle multiple
sequence origins and mutable callable storage together through the existing
Element/Pattern/Witness pipeline. The checkpoint retains the verified witness
integration and its native controls. The [checkpoint policy](Regression_Check_Policy.md#checkpoint-cadence)
requires this separation when a coherent cross-cutting change is ready before
the next coupled feature gate; it does not waive that gate.

The [closure settlement contract](Closure_Settlement_Contract.md), closure
architecture documents and C-series criteria were reconciled against the Clef
specification: immutable formation snapshots, mutable-cell sharing, public source
signatures, actual native formals, environment-first invocation and caller-owned
result destinations each retain their own authority. Baker owns elaboration and
saturation; Alex observes those results through its zipper. Incremental work must
retain the supporting and invalidated relationships as well as positive facts.

Validation:

| Gate | Result and limit |
|---|---|
| Full CCS test assembly | **1,058 passed**, including exact origin, result-destination, residence, alias-specialization and source-navigation regressions. |
| Full Alex test assembly | **124 passed**, including actual Huet occurrence, typed callable operands and portable multiple-result verification/lowering. |
| Full CCS.Editor default suite | Passed source signatures/definitions, unsaved edit/repair, stale snapshot rejection, capture/environment/sequence projections, and its actual cvc5 dispatch checks. |
| .NET parallel-runner tests | Passed bounded overlap, phase barrier, ordered results, independent artifacts, streams/input, failure reporting and process cleanup including orphaned descendants. |
| Actual 16h source settlement | Unchanged sample passed with no parse failures or effective source errors and 10,073 graph nodes. Native acceptance belongs to the coupled integration gate above. |
| Four-worker native controls | **4/4 compiled and 4/4 executed**, with exact manifest output for 01, 11, 12 and 16a; no skipped entries. |

The consumer assemblies were rebuilt with project references enabled after the
shared graph union changed. An earlier stale editor assembly misread the new
union case; its initial cast failure is retained in the investigation log. The
subsequent source-navigation failure drove the provenance correction above.

Reproduction from the Composer root:

```bash
dotnet test ../clef/tests/Clef.Compiler.Service.Tests/Clef.Compiler.Service.Tests.fsproj --no-restore --disable-build-servers -v:minimal
dotnet test tests/Alex.Tests/Alex.Tests.fsproj --no-restore --disable-build-servers -v:minimal
dotnet run --project tests/CCS.Editor.Tests/CCS.Editor.Tests.fsproj --no-restore --disable-build-servers
dotnet fsi tests/regression/ParallelRunnerTests.fsx
dotnet fsi tests/regression/Runner.fsx -- --sample 01_HelloWorldDirect --sample 11_Closures --sample 12_HigherOrderFunctions --sample 16a_SequenceOperations --jobs 4 --timeout 180
```

The native cohort retains full `-k` artifacts and uses an explicit 180-second
per-job limit. It is a functional control gate; it does not establish the
30-second default-timeout suite or full-manifest acceptance. Artifact formatting
and the reachable-only serialization view retain their separate follow-on gate.
The final run took 94.7 seconds; its compiler snapshot, hashes, per-job artifacts
and results are under
`/tmp/composer-checks/20260926T113957-878016bda9a844dc8b5d3e632f38a605/`,
with console log `/tmp/composer-checkpoint-native-final.log`.

Coordinated source input: Clef `f898c97e11c5a45da91899901bac29250a8d96a2` and
the Composer commit containing this entry (parent `c81d16b760d7`). Companion
revisions: Fidelity.Platform `d42c9988f7fd`, BAREWire `6e212487aef5`, and
clef-lang-spec `280efcc551a4`. The separate spec/site/whitepaper documentation
sweep remains in those repositories' working trees; these two authorized commits
cover Clef and Composer. Local Markdown targets/anchors and whitespace were
checked before committing.

Final suite logs: `/tmp/clef-checkpoint-final-tests.log`,
`/tmp/composer-checkpoint-alex-coordinated.log`,
`/tmp/composer-checkpoint-editor-coordinated.log` and
`/tmp/composer-checkpoint-runner.log`; actual 16h source admission is recorded in
`/tmp/clef-checkpoint-16h-final.log`. The earlier entries remain dated
investigation evidence; their failing runs and smaller suite counts are not this
checkpoint's results.

## Staged Seq contracts and bounded process checks — 2026-09-26

**Bounded implementation checkpoint; C-01–C-07 remain In-Progress.** Core Seq
operations now use Baker's existing recipe/fold-in path for stored partial
applications and bare operation values. Supplied operands are snapshotted at
their written formation frontiers; fold keeps independently typed state and
payloads. Seq occurrences participate in the existing generic instantiation and
immutable-alias specialization paths. The new source tests preserve dimensional
types, effects, binding identity and located rejections. An unsupplied polymorphic
`tryHead` alias retains its declaration scheme and has a concrete call-site
instance; the source declaration is not forced monomorphic to satisfy a test.

The full CCS suite exposed recipe creation for historical, unreachable partial
call nodes retained by application flattening. Those obsolete Seq occurrences no
longer create another deferred producer. The retained graph and owning
reachability/sequence-ownership analyses remain intact; live staged frontiers
still elaborate. The existing ownership tests were preserved.

Alex lambda witnessing now descends from the actual occurrence and uses scoped
formal operands and body visitation. Native controls exposed two integration
defects in that change: a nested function rediscovered a globally emitted
definition, and a recalled formal bypassed its settled read meet. The fixes
distinguish definition-only dependencies from local closure construction and
transcribe the existing read refinement. Shared-body, nested-definition and
formal-read component tests exercise real witness output and MLIR verification.
No capture, layout or source-operation reconstruction was added to Alex.

The [regression runner](../tests/regression/README.md) now supports `--jobs N`.
It builds a private .NET process host, builds Composer once under a cooperative
build/copy lease, snapshots the compiler, and runs bounded compilation jobs
followed by bounded native jobs. Each run and ordinal has separate artifacts,
streams and status files; `--artifacts-dir` isolates direct CLI invocations too.
Linux process-group cleanup covers a parent that exits while a descendant holds
the pipes. Native stdin/exit/output contracts remain checked, all `-k` artifacts
are retained, and failed checks keep the run red. This implements independent
check concurrency; compiler-internal parallel saturation and serialization remain
separate work.

The timeout investigation used the unchanged baseline binary. Closures with
`-k -T` reported **83.663 s** across timed phases, including **83.211 s** in the
front end. Without `-k`, the same source/profile reported **6.829 s**. A .NET trace
placed most sampled main-thread work in `emitPhaseIfEnabled` and F# reflective
object formatting. These are individual observations, not a benchmark series.
The runner's 30-second default bounds an example compilation, independently of
the compiler's .NET build. Manifest limits were not increased. Full versus
reachable-only serialization and bounded artifact formatting remain prospective
optimizations; no graph pruning or reduced evidence was used to obtain a pass.

Validation:

| Gate | Result and limit |
|---|---|
| Full CCS test assembly | **1,023 passed**, including 17 staged-Seq cases and the unchanged sequence-ownership tests. |
| Full Alex test assembly | **97 passed**, including three actual-occurrence/dependency/read-meet regressions. |
| CCS.Editor `--sequence-applications` | Passed public signatures, source definitions, measured unsaved edit/repair and stale-read rejection. This targeted check does not claim a full LSP run. |
| .NET parallel-runner tests | Passed worker bounds/overlap, ordered results, phase barrier, process streams/input, launch failures, nonzero exits, live-parent and orphan timeout cleanup, and directory/symlink isolation. |
| Four-worker native control run | 01, 02, 11, 12 and 16a compiled and ran with exact expected output; 16h failed source admission, so the run returned failure. This diagnostic cohort used `--timeout 180`; it does not establish that the default timeout suite passes. |
| Same-project concurrency | Two independent 16a entries compiled and ran together with separate artifacts and exact output, while the control runner was also active. Both runner invocations used private compiler snapshots. |
| Final Baker-guard native replay | Two-worker 16a/16h run used their unchanged manifest limits: 16a passed compilation/execution; 16h retained 44 CCS8403 continuation/environment settlement errors and produced no accepted executable. |

This is not completion of native stored-Seq support. Resume at 16h's exact
parent/child frame identities, retained ordinary callable/environment formation,
range/representation settlement and factory/consumer boundaries. Preserve its
source and expected output. Existing original16, original13/original15,
collection storage and canonical lazy gates remain open as recorded in the
C-series contract; this checkpoint did not run the complete native manifest or
new proof, circuit or device acceptance gates.

The [incremental direction](Nanopass_Incremental_Contract_Direction.md#21-reachability-is-a-supported-conclusion)
now makes reachability support/retraction explicit and distinguishes serialized
views from semantic dependency regions. The proposed joint boundary judgments
and segmented MLIR → objects → LLD/ELF path retain actual Huet context, semantic
versus realization dependency ownership, content/configuration identities and
compatible artifact/evidence publication. Their edit/retraction/interface tests
are requirements, not implemented incremental-compilation claims.

Coordinated inputs: Composer `c81d16b760d7` and clef `15a688a74948`, with the
working-tree implementation/tests described above; Fidelity.Platform
`d42c9988f7fd`, BAREWire `6e212487aef5`, clef-lang-spec `280efcc551a4`.
No commits were created. Existing documentation and companion-repository work
were preserved. Each native runner artifact root retains compiler hashes,
manifest/selection/configuration and per-job output:

- Four-worker controls: `/tmp/composer-checks/20260926T100405-14afa6823c774c5080ebaca1178115e8/`.
- Concurrent duplicate project: `/tmp/composer-checks/20260926T100405-d2f60f536f89463fb1f18f840a009a8b/`.
- Final Baker replay: `/tmp/composer-checks/20260926T101020-fe28051188724dc79d7dceafdf903a5e/`.
- Source/editor/component logs: `/tmp/c-series-20260926-ccs-full.log`, `/tmp/c-series-20260926-editor-seq.log`, `/tmp/composer-parallel-alex-tests.log`.
- Harness log: `/tmp/composer-parallel-runner-final-tests.log`.
- Baseline timing/trace: `/tmp/c-series-20260926-11-timing.log`, `/tmp/c-series-20260926-11-no-k.log`, `/tmp/c-series-20260926-11-profile.nettrace`.

Whitespace and local documentation links were checked. The earlier corpus-drift
baseline remains the pre-existing specification diagnostic-code finding; no
specification source was changed in this implementation checkpoint.

## C-series nanopass boundaries and functional showcases — 2026-09-25

**Documentation checkpoint; implementation statuses unchanged.** The shared
[ownership boundary](PRDs/C-Series-Acceptance.md#11-baker-construction-alex-witnessing-and-backend-realization)
now explicitly preserves Baker nanopass elaboration, fan-out/fold-in and
saturation, and Alex's actual Huet occurrence with Element/Pattern/Witness
composition into admitted portable MLIR. Zipper navigation laws do not establish
dependency freshness or correct occurrence preservation through every production
scope traversal. The acceptance criteria cover both distinctions.

[Design-time continuation](PRDs/C-Series-Acceptance.md#42-scope-aware-design-time-nanopass-continuation)
follows the existing incremental direction: affected semantic regions extend
beyond lexical subtrees; crossing participants, alternative supports, cycles,
retractions and stale completions retain their own obligations. Future selective
reuse is compared with a fresh check before artifact/behavior acceptance. This
does not freeze a new pass protocol or claim an incremental engine is implemented.

FPGA guidance now explicitly retains target commitment in the backend. The
planned Dynamatic handoff previously placed Handshake directly in Alex; it now
receives portable MLIR plus correlated graph evidence and performs the conversion
in the FPGA leg. Existing CIRCT-emitting hardware witnesses remain identified
reconciliation work. Scheduling, buffering, component selection and HDL generation
do not become middle-end algorithms. The eBPF overview likewise separates Alex
witnessing from backend encoding/register allocation and final-host admission.

The [functional showcase plan](Functional_Surface_Showcases.md) connects C-series
delivery to HelloWayland frame planning, WrenHello's versioned signal facade,
HelloDISCO's bounded model/view and the planned
[Arty continuation case](fpga-targeting/08_functional_continuation_case_study.md).
The Arty case distinguishes captured producer identity, suspension/join semantics,
backpressure and reset from actual circuit overlap and FPGA artifact evidence.
UI function/CE equivalence preserves cold activation, keyed payload updates and
scope-owned disposal under Fidelity.UI's contracts; Ripple is ergonomic prior
art. Async, threading, reactive, UI and target successors keep their own gates.

Source/document inspection references:

| Repository | Revision |
|---|---|
| Composer | `a2a02be` starting criteria; boundary/case additions recorded at `c81d16b` during this review |
| clef / clef-lang-spec | `15a688a74948` / `280efcc551a4` |
| BAREWire / Fidelity.Platform | `6e212487aef5` / `d42c9988f7fd` |
| Fidelity.UI / Fable.Ripple | `a1c280bbc78f` / `8e4d31273ce6` |
| HelloArty / HelloWayland | `e3c0b158de6b` / `adc4cf6bde11` |
| WrenHello / HelloDISCO | `cd6f307cb8f0` / `94dddd49d237` plus existing HelloDISCO working-tree changes |

HelloDISCO's uncommitted settings record, behavior projections and related
project/tests were inspected and preserved. Their presence does not establish
native or board acceptance of those changes. This checkpoint edits Composer
documentation only; no application, compiler, fixture, image or device changes
were made.

Validation covers changed and incoming local links/anchors, architecture and
showcase cross-review, and whitespace. The corpus drift gate retains its one
pre-existing specification diagnostic-code finding at `error-handling.md:303`;
the before/after logs are `/tmp/clef-c-series-boundary-drift-before.log` and
`/tmp/clef-c-series-boundary-drift-after.log`. No compiler, native, UI, solver,
circuit or board tests were run. Resume implementation with the fresh coordinated
C baseline and unchanged 16h; the showcase slices supplement those gates as
their actual prerequisites become available.

## C-series criteria and realization integrity — 2026-09-25

**Documentation checkpoint; C-01 through C-07 remain In-Progress.** The
[shared acceptance contract](PRDs/C-Series-Acceptance.md) and all seven C PRDs
now distinguish existing implementation, dated bounded evidence and remaining
acceptance. F-xx and C-xx have equal delivery standing: dependency order does
not imply different standards of semantic or artifact integrity. An accurate
diagnostic for a promised conforming use remains an open positive gate.

The revisions retire obsolete capture/layout work in Alex, fixed SSA budgets,
packed interior callables, default global arenas, cold-start checklists and the
old BAREWire blocker inventory. Current recipes, witnesses and native oracles
remain the implementation assets. C-01 retains its boundary contract and §14
form-family anchors; the direct mutable-cell note now uses stable section links.
C-03 retains implemented recursive identity/capture work. C-04 retains its full
promised List/Map/Set/Option/range/support surface and identifies the missing 13a
manifest gates. C-05 requires normative caching and records the necessary future
sample14/manifest correction; those fixtures are unchanged here. C-06/C-07 now
separate their recorded native successes from recurrence, retained environment,
aggregate and staged-operation gaps. C-07 core and successor rows remain explicit.

The [Colibri component basis](fpga-targeting/02_colibri_circuit_basis.md),
[FPGA artifact verification](fpga-targeting/05_artifact_verification.md) and
[eBPF admission boundary](ebpf-targeting/01_verifier_as_design_time_contract.md)
inform the common criteria directly. Their lesson is delivery integrity through
the selected realization: exact participant and component correspondence,
preserved observable traces, checked target assumptions, and evidence renewed
when lowering changes a premise. Artifact checks consume actual emitted or
reconstructed outputs; a second model serialized from the same upstream graph
does not validate the emitter. Circuit-to-silicon and verifier/host gates retain
their distinct evidence levels. This alignment neither completes M-01's general
admission/evidence transport nor claims new FPGA/eBPF integration or deployment.

Inspected source references were Composer `a9583674c34c`, clef `15a688a74948`,
BAREWire `6e212487aef5`, Fidelity.Platform `d42c9988f7fd`, clef-lang-spec
`280efcc551a4`, and Colibri `3fa784121cce`. Current site and Markdown-paper working
copies informed the review; their existing modifications were preserved. These
are inspection references, not a freshly tested combined toolchain. Only
Composer documentation is changed in this checkpoint; no source, sample,
manifest, compiler output or dependency was modified.

Validation covers changed local links/anchors, incoming links to the revised
documents, operation-inventory retention, semantic cross-review and
`git diff --check`. No compiler, solver, native, formal-circuit or hardware tests
were run. The corpus drift gate retains its one pre-existing finding in the
specification diagnostic-registry paragraph (`error-handling.md:303`, an inherited
diagnostic-code example); no new finding is introduced. Scheduled migration
vocabulary is reported separately from that failure. Before/after logs are
`/tmp/clef-c-series-drift-before.log` and `/tmp/clef-c-series-drift-after.log`.

Resume with the [dependency-led continuation](PRDs/C-Series-Acceptance.md#5-dependency-led-continuation):
establish a fresh coordinated baseline, then use unchanged 16h to complete
staged application and retained callable/environment contracts. Original16,
original13/original15, collection storage and canonical lazy acceptance remain
explicit subsequent/shared gates. Existing valid expectations stay fixed;
normatively obsolete expectations require a documented semantic correction.

## Interactive compiler workbench planning — 2026-09-25

**Documentation checkpoint; WB-01 through WB-04 remain Planned.** The new
[workbench plan](Interactive_Compiler_Workbench.md) coordinates a bounded SageFS
bootstrap-host pilot alongside C-01–C-07, a shared versioned service for Lattice
and MCP, responsive cvc5 dispatch, and native ORC execution. Composer owns the
workstream; CCS/Baker own semantics, Alex retains passive ctx pull witnessing,
and the existing proof service owns dispatch and evidence. No language PRD
acceptance status changes.

The plan requires compiler-generation identity for hot patches, isolated or
serialized compiler state, current PSG/query correspondence, explicit rejection
controls and fresh persisted-source replay. It records the current per-query
solver process, generation-local cache and batch-publication limits; warm workers
and dependency-directed reuse remain measured implementation work. The earlier
Transcribe sequence now distinguishes this enabling work from later native
self-hosting and notebook milestones.

The selected interactive CLI name is `clefx`, matching `.clefx` script files.
The [interactive specification](../../clef-lang-spec/spec/interactive-development.md)
now separates native execution from FSI bootstrap hosting, removes unsupported
interpreter/hybrid modes and invented working command surfaces, and identifies
the pending binding, initialization, lifetime and invocation contracts. Clef
does not carry F#'s separate `.fsi` signature files. Legitimate F# implementation
scripts and interfaces retain their names.

The [Alex overview](Alex_Architecture_Overview.md),
[pipeline guide](Architecture_Canonical.md), README and agent guidance now follow
current source: Baker-owned semantic settlement, positional ctx pull, derived
SSA names, one combined witness traversal and explicit mutable emission
bookkeeping outside the zipper. Parallel-zipper implementation/performance claims
and migration handoffs are superseded historical records. Related Clef, Lattice,
site, Atelier, Farscape and migration-planning documents use the same ownership
and implementation status. Atelier's reflection proposal no longer equates
attention with freshness or places range/escape analysis in Composer.

Inspection used Composer `9d5c85266a5392da345d69ee9297a3536282b4cc` and Clef
`e94fa905f7f13c2b8216b355e9d84b17dceef5c8`, including pre-existing Clef working-tree
changes, which were preserved. These are source references, not a tested combined
toolchain. This checkpoint adds no package, host installation or executable
behavior; two source comments were clarified. No compiler, solver, native
execution or performance tests were run. Documentation validation passed:
changed local links/anchors, affected incoming references, and whitespace checks
across the edited repositories. The corpus drift gate is clean; its six baseline
findings were corrected or given accurate historical/FFI context without changing
the gate. The non-Git `Clef_migration` originals are preserved under
`/tmp/clef-migration-doc-baseline-4d3xju_i` for comparison.

Resume with [WB-01's pilot acceptance](Interactive_Compiler_Workbench.md#pilot-acceptance):
choose the coverage defect and ordinary baseline, pin the host, then exercise the
same pipeline through the smallest adapter. Other milestones and existing
language feature statuses remain unchanged.

## FPGA circuit and artifact verification planning — 2026-09-25

**Documentation checkpoint; FPGA-01 through FPGA-07 remain Planned.** The new
[FPGA series](fpga-targeting/README.md) is the restart point for the Clef-fed
Dynamatic fork, canonical Colibri component realization, direct VHDL-2008 emitter
and independently implemented VHDL-to-Rocq adapter. Its
[roadmap](fpga-targeting/07_roadmap.md) separates source/component correspondence,
mapped-netlist evidence and final-bitstream reconstruction.

Shared edges remain owned by M-01, numeric/width selection, graph-resident
obligations, the existing proof-composition/toolchain design, Fidelity.Platform,
BAREWire and Lattice. The series records the current HelloArty/HelloProof limits,
Colibri's per-component formal coverage and licenses, and VHDL2Rocq's proprietary
combinational implementation. No Lean dependency or second general proof service
is introduced. An admitted FPGA profile cannot silently bypass its component and
evidence policy.

Source inspection used Composer `c46485096c38dc2c9fef9db4ae60caa1905ca1a3`,
Dynamatic `2fac2911faf35b84156e08782a9fcfe8fc6174d8` and Colibri
`3fa784121ccea86d9e65b2e0dc08d2a3327f5f2f`; these are research references, not
a tested combined toolchain. Pre-existing documentation working-tree edits were
retained. Validation covers the new/changed local links and anchors and
`git diff --check`. No compiler, solver, Rocq, synthesis, native or hardware tests
were rerun; no implementation or PRD acceptance status is advanced here.

## Bidirectional composition design accounting - 2026-09-21

**Documentation checkpoint; native NFT work remains Planned.** The
[coordination and acceptance plan](Bidirectional_Composition_Plan.md) is the
single restart point for this case study. It links each source edit and its
validation, including the NFT/PHG/FPS manuscripts, specification capsules,
engineering design and the *A Path Less Traveled* blog article. Paper changes
are Markdown only; generated TeX/PDF review and publication remain separate.

The [NFT architecture](Negative_Fractional_Types_Architecture.md) now separates
two-way dependency from inverse execution, identifies value-indexed resource
and dynamic-instance obligations, and gives a bounded-storage exact inverse
example. The PHG account distinguishes executable structure from local coeffects
and ordered joint relations. Chen and Sabry's Theorem 25 erratum is recorded
without withdrawing their compact-closed result or attributing it to Tardis.

The plan retains independent executable gates for bidirectional demand semantics,
finite indexed resources, and a target-checked reversible region. Neither this
checkpoint nor the source-comment corrections in Clef promote a language PRD.
Validation and the separate *Cold Half of Concurrency* publication result are
recorded in the coordinator, avoiding another change ledger here.

## Clef repository history migration — 2026-09-20

Clef's repository maintenance establishes a new root at the February 18 CCS
rename, retaining the subsequent Clef development sequence and removing upstream
refs and unused files. The cleaned compiler at `c1491aa`, with completion evidence
recorded and pushed in `e94fa90`, has a
[migration record](../../clef/docs/handoffs/Repository_History.md) and an
[old-to-new commit map](../../clef/docs/handoffs/commit-map.tsv). Earlier Clef
hashes below are historical evidence references: resolve them through that map;
do not merge old ancestry back into the maintained repository.

The source cleanup passed **1,006/1,006 CCS tests**, a Composer build and the
native formatter/UTF-8 snapshot gate with stock MLIR verification and exact
22-line output. Native artifacts are retained at
`/tmp/composer-platform-format-0ba710b6f8914b81b3a0e657fe850bcf`.
The filtered tip tree matched the tested cleanup byte-for-byte before adding
the commit map. This maintenance changes no feature acceptance status or
source-language contract. Post-rename authorship, dates and merge structure are
retained; removed paths are also filtered from historical snapshots.

Remote publication atomically replaced `main` and deleted 209 obsolete branches
and 143 inherited tags. A fresh SSH clone exposed only `main`, passed Git
integrity checks and a fresh CCS build, and held 2.21 MiB of packed Git objects
versus 415.46 MiB before maintenance. The 101 unused files removed from the
current tree totaled 13.33 MiB. External recovery bundles and inventories live
under `/home/hhh/repo-archives/clef-thinning-2026-09-20/`; no old-history backup
ref remains in the maintained repository. Both existing local worktrees were
reconciled, with the former dimensions branch now a clean detached checkout.

## Target-aware dialect planning and next-session handoff — 2026-09-20

**Planning synchronized; M-01 remains Planned.** Start the next session with
[M-01](PRDs/M-01-DialectAdmission.md), especially its
[contract map](PRDs/M-01-DialectAdmission.md#5-numeric-selection-parallelism-and-design-time-projection)
and [implementation handoff](PRDs/M-01-DialectAdmission.md#9-resuming-implementation-across-repositories),
then the owning language PRD and the linked clef-lang-spec chapters. The standard
governs semantics; the [PRD index](PRDs/README.md) records scope and status.
Historical blog/PRD pseudocode does not override the current contracts.

Baker owns semantic construction, joint constraints, elaboration and saturation.
Alex's passive Huet zipper observes the full settled expression and platform
facts to select appropriate Elements/Patterns/Witnesses for the backend. The
admission key is expression family × platform/backend profile × witness form.
An explicit-block `cf` form can suit one profile while another requires `scf`.
Numeric selection and arithmetic construction govern `arith`/`math` forms;
operation-specific capabilities, rounding, capacity and allowed decomposition
survive the handoff. RPC wait relationships and scheduler manifests carry the
separate progress, resource and supervision requirements. Tooling receives these
target-specific facts and diagnostics through the shared CCS projection.

M-01 links Numeric Selection §§10.3–10.5/11/14, Synchronous RPC and Wait
Classification, Scheduler Contract and Platform Bindings. The reviewed
*Pondering Fearless Parallelism*, *Fearless Concurrency Gets Real* and
*Carrying Proofs into JavaScript* articles supply motivation and oracle cases.
The plan distinguishes specified obligations from open fact schemas,
construction-selection mechanisms and unimplemented target support.

Candidate math/affine/vector/tensor/async/cf families are demand-driven;
index is already baseline. CIRCT, existing GPU/ROCDL, AIE, proposed Triton and
JSIR pathways retain distinct acceptance scopes. No new dialect is enabled by
this record. The source inventory at Composer `1fccb02` identifies incomplete
index serialization/vector sizing, existing direct cf/CIRCT/AIE forms needing
boundary reconciliation, and a backend interface accepting text/configuration
without a general graph/proof-correspondence input. M-01.a/b pair that inventory
with one demanded expression/profile and its complete information transport
before expanding vocabulary. Preserve working target oracles during this work.

| Repository | Revision | Scope |
|---|---|---|
| clef-lang-spec | `2813371` | Backend lowering §2.1.1: target-aware operation/profile admission, complete information handoff and existing numeric/concurrency contract dependencies |
| clef | `97dc5e478` | Baker retooling and Lattice consumer plans; retire the stale claim that current specs require deferred closure casts |
| Fidelity.Platform | `d42c998` | Operation-specific numeric/scheduler capability plan and documentation index |
| BAREWire | `6e21248` | Layout, publication, partial-state fidelity and lifecycle handoff acceptance plan |
| Fidelity.UI | `a1c280b` | Piped FP/CE, numeric and display/parallel execution triangulation |
| ClefAutoComplete | `c55d25e3` | Shared projection and protocol acceptance responsibilities; reference fork remains reference-only |
| lattice-analyzers | `239d983` | Numeric/concurrency regression purposes and required CCS checks |
| lattice-vscode | `9388a6b` | Target evidence presentation and invalidation acceptance plan |
| lattice-vim | `9e99649` | Target-aware real-compiler gate planning, separate from existing transport fixtures |
| Composer | This coordinated commit | M-01, PRD index/handoff, Lattice integration, thin-middle/completion analysis clarification, dated audit follow-up and this waypoint |

Validation: changed local documentation links/anchors and PRD status columns
checked; `git diff --check` passes. The repository-wide vocabulary drift gate
still reports **six existing findings**, all verified unchanged by this batch;
`/tmp/clef-m01-planning-drift.log` records them. Its initial seventh finding
exposed a stale five-dialect/direct-cf exclusion in the completion analysis,
now reconciled to M-01. This is not a clean global drift-gate claim.
No compiler code changed and no compiler,
solver, native or device tests were rerun. The F-05 evidence below remains the
latest implementation tranche. F-06 and C-01–C-07 retain their recorded open
gates. This planning commit does not close them. Companion library/tooling
implementation duties are recorded in M-01 §9 and in the companion plans at the
revisions above; no executable source edits are part of this synchronization.

## F-05 character storage and native formatting — 2026-09-20

**F-05 restored to Complete at its sample scope.** The original AddNumbers sample
again passes compilation, stock MLIR verification and exact native output. The
character-buffer regression is resolved in Baker's graph; Alex consumes settled
carriers and rejects mismatches. F-06's parsing regression and the open C-series
exit gates remain separate work.

Both `String.fromBytes` and `String.toBytes` have logical `int[]` signatures.
Baker retains the exact allocation, aliases, writes, dependent reads and selected
platform representation declaration in resident storage/proof incidence.
`fromBytes` establishes an immutable snapshot; `toBytes` composes an ordinary
allocation and copy loop from an internal byte view into independently inferred
integer storage. A later write of 4096 remains an integer write and cannot mutate
the source string. Source element types and unrelated array storage are preserved.
Range and loop saturation run after these recipes, followed by placement and
settled obligations. Canonical reachability refresh retires replaced intrinsic
callees without deleting proof provenance or relaxing witness coverage.

The [standard conversion contract](../../clef-lang-spec/spec/native-type-mappings.md#integer-byte-unit-conversions)
and NTU representation tables are reconciled. Current `fromBytes` admission covers
closed ASCII buffers and immutable constant valid UTF-8 sequences; unknown or
escaping storage and unproved text validity are explicit errors. This does not
claim general dynamic UTF-8 validation or arbitrary floating-point formatting.

| Repository | Revision | Scope |
|---|---|---|
| clef | `30f64b061` | Baker storage/snapshot recipes, range and meet integration, exact diagnostics and 15 encoding cases |
| Fidelity.Platform | `92752a7` | Plain NTU formatter buffers, bounded digit construction and minimum signed integer handling |
| clef-lang-spec | `5dace04` | Standard conversion signatures, snapshot laws and NTU representation authority |
| Composer | This coordinated commit | Passive carrier witnessing, native and editor gates, F-05 status, Fidelity.UI/LVGL roadmap correction |

Final CCS SHA-256 is `2581ee6415188161541896fced63d6ed992ebabd193ddbd9d45c22698907e33c`;
Composer is `dfb0d7c02fd450441e277bafa4759f841a5866240c86de963b37eab49931ae7b`.

| Gate | Result and retained evidence |
|---|---|
| CCS | **1006/1006** on preceding `563959af…af1065`, `/tmp/clef-string-storage-full-tests.log`; after the focused reachability correction, **15/15** encoding cases pass on the final artifact, `/tmp/clef-string-storage-reachability-focused.log` |
| Alex | **94/94**, `/tmp/composer-string-storage-alex-full.log`, before that core-only reachability correction; no Alex change followed |
| SMT transfer | **85/85** through `mlir-translate`/cvc5, `/tmp/composer-string-storage-smt.log`, on the preceding core artifact |
| Native formatter | **PASS**: UTF-8 bytes, empty/nonzero slices, both snapshot directions, a subsequent 4096 write and 22 numeric outputs; `/tmp/composer-platform-format-c854c08cf4ff4ad8875fad9821954a8e/` |
| Original F-05 | **PASS**, `/tmp/composer-foundation-native-0c1fc64d1b0a449cbbac16143006d9bf/` |
| Full-profile 16g | **PASS**, `/tmp/composer-foundation-native-355ce0de5afe49499751980e252c6047/` |
| Editor projection | **PASS** on the final artifact: exact `CCS8404` span/message/participants, unsaved repair, immutable prior snapshot and source-only absence of physical proof; `/tmp/composer-string-encoding-reachability-editor-tests.log` |

All three native gates require stock MLIR verification, native exit zero, empty
stderr and exact output; their directories retain input hashes and evidence.
The formatter runner now lives in Composer's F# bootstrap tests; the earlier
Python draft was removed from Fidelity.Platform. BAREWire descriptors and the
public editor/diagnostic transport contracts are unchanged. CAC, Lattice and the
analyzers continue through that shared CCS projection; no new live LSP run is
claimed. The PRD index now identifies Fidelity.UI native rendering as the primary
direction, with an optional Farscape/LVGL adapter and HelloWayland as a working
experimental oracle.

## PRD status and repository synchronization — 2026-09-20

The [PRD index](PRDs/README.md) now uses only Planned, In-Progress and Complete,
with a separate Note in every category. Foundation completion follows the
recorded native evidence: F-00–F-04 and F-07–F-10 are Complete at their stated
scope; F-05/F-06 were reopened for the recorded formatter/parsing regressions.
The later character-storage entry above closes F-05.
C-01–C-07 remain In-Progress. Later sequence composition has exposed additional
closure, callable storage and residence requirements in the earlier areas.

Numbering guides the work; current dependencies determine the sequence. Async
and Threading precede Reactive. Within Reactive, Incremental leads from the
closure/lazy/thunk basis while Observable is developed alongside it, including
shared invalidation, cutoff, effect order and ownership gates. HelloWayland's
recorded Ariel CPU and typed-carrier acceptance supplies working-sketch evidence
from before full actor-model expression. The project author reports saturation
of all 32 CPU threads. This is useful workload evidence, not an authoritative
actor/scheduler design or closure of the A/T PRDs. Its September 9 acceptance
record and the author's observation are not fresh execution measurements from
this synchronization batch.

This batch publishes the previously pending design records, corrects stale
server/tooling status, aligns sample/library manifests with the existing
platform profile/environment taxonomy, and retires the misidentified
Allwinner-H6 Sweet Potato sample. It adds no compiler implementation. The C-07
compiler and proof results below retain their original revisions and limits.

| Repository | Revision | Synchronization scope |
|---|---|---|
| Composer | `be0f14cd9666` | PRD status, incremental and JavaScript/WebView design, credential roadmap, sample catalog and obsolete SBC retirement |
| clef-lang-spec | `d04574a` | Credential custody, authenticated sealing, durable commit and recovery requirements |
| ClefAutoComplete (`fidelity`) | `1f347d78` | Reference fork and active CCS/Lattice boundary reconciled |
| lattice-vscode (`fidelity`) | `390d5b2` | Active client status and root development launch configuration |
| lattice-vscode-helpers (`master`) | `3cb2979` | Inherited bindings solution path, branding and scope; not an active-client dependency |
| Fidelity.Data | `213918a` | Previously committed hosted-source filename/project alignment |
| Fidelity.Desktop | `1e36556` | Wayland platform environment dependency |
| Fidelity.Image | `7ef6765` | Image binding platform environment dependencies |
| HelloArty | `e3c0b15` | Platform profile/product distinction |
| HelloNappy | `6e90f59` | Linux profile and environment bindings |
| HelloWayland (`pre-light-mode`) | `adc4cf6` | Platform profiles/environment paths, including Ariel gates; also publishes prior acceptance record `81fa3c0` |
| WrenHello | `cd6f307` | Existing Fable/JSX/Solid and embedded WebView build documentation |
| Fidelity.CloudEdge (`agents`) | `340d9b7` | Xantham pathway and current ambient-module evidence; Fable reference work, not Clef semantic widening |

The unchanged clef, BAREWire, Fidelity.Platform, Fidelity.UI and analyzer
implementation revisions remain those recorded in the C-07 table below.
Validation here consists of PRD status/Note checks, local documentation targets,
JSON/TOML and project dependency paths, solution/project source paths, and diff
checks. Root VSCode launch configuration received static checks; the prior real
F5 gate belongs to the client workspace. The documentation drift script still
reports six findings outside the edited tooling READMEs
(`/tmp/lattice-readme-sync-drift.log`); no clean global drift gate is claimed.

Explicitly pending local work:

- At that synchronization, Fidelity.Platform's `Format.clef` migration and
  `tests/Format/native/` remained unpublished. Its 22-case native gate failed stock
  MLIR verification: the first `Format.int` call expects `memref<?xi8>` while the
  callee returns `memref<?xi64>`. No native boundary assertion ran.
  `/tmp/platform-format-native-xv9_k_me/compile.log`; Composer assembly SHA-256
  `0121c21407e8a15fb8e9fd3803c1384f82e56e04d55c0262afa1a61d5a81fb43`, CCS
  `9fe00834eda63cc9a4b91253b983d5e2aaf2e49fb4ba536fb3b271ae1f709e0f`.
  That compiler/storage-contract blocker is now resolved by the coordinated
  character-storage entry above; the runner was moved to Composer.
- Fidelity.Font's matching FreeType dependency-path change is committed locally
  as `79f022a`; the repository has no configured remote or upstream. Its SSH
  publication destination has been requested.
- Fidelity.Signal retains an older, unvalidated runtime-table/native-pointer
  prototype and generated `target/` artifacts. It has no remote. Its README
  distinguishes that prototype from the planned Incremental/Observable contract;
  these edits are not admitted reactive implementation or part of this push.
- Historical tooling state: Fidelity.WebView's untracked `.serena/` and the retired FsNativeAutoComplete repository's modified
  `.serena/project.yml` are local tooling configuration, outside this sync.

## C-07 sequence operations — implementation waypoint, acceptance open, 2026-09-20

**Coordinated implementation waypoint; C-07 exit gates remain open.** This entry
records the assembled work and actual gates so continuation does not rely on
conversation history. The [C-07 PRD](PRDs/C-07-SeqOperations.md) retains the full
exit criteria; companion revisions are recorded below.

Latest consolidated gates on CCS
`da1f5790bd5d1aaec12a28fb09401907704bc157951957ca616dd52a10300c8f`:

| Gate | Result |
|---|---|
| CCS | **987/987**, `/tmp/clef-c07-unit-activation-full-tests.log`. Module tests now assert ordered reference membership and lexical identity without executable children; the empty-sequence oracle explicitly demands both owners. |
| Alex | **90/90**, `/tmp/composer-c07-final-alex-tests.log` |
| SMT transfer | **85/85**, including real `mlir-translate`/cvc5 dispatch and source/native loop parity: `/tmp/composer-c07-final-smt-transfer.log` |
| Native 16a–f | All six pass fresh compilation, stock MLIR verification, exact output and exit 0. Evidence suffixes under `/tmp/composer-native-sequences-…`: `5d753990fc6b45ff9fc900708b8c1f5f`, `e8a5cb6c935b4bd397d78ef815d9f738`, `ed5d00fc292e4f7cb816f7c7ddfce666`, `be58181ceea640049a85a96e02504c58`, `15acc441f7ac4264a0a8558dde2ebdad`, `0df583f172dd4e568981fd03806a36e0` |
| Native controls | 15a and 15d pass the same gates: `/tmp/composer-native-sequences-fa046022410b4b15b8764ecb5e9a1d3d`, `/tmp/composer-native-sequences-c872553ccb2545729ecdf8498d466a1a` |
| Editor / analyzer projection | Shared final projection passes: `/tmp/composer-program-lifetime-owned-editor-tests.log`. Analyzer executable passed the identical linked gate on preceding `344c4583`: `/tmp/lattice-program-lifetime-99a71908a26e48f395b53a37ad27281d/evidence.json`; its duplicate run was not repeated after unit activation. |
| Live LSP | Final selected-platform error/repair and source-only startup/pending projections pass: `/tmp/lattice-program-lifetime-lsp-9jk9Hi/evidence.json`, `/tmp/lattice-program-lifetime-lsp-chLebG/evidence.json` |
| BAREWire | **580/580**, `/tmp/barewire-program-lifetime-tests.log`; descriptor implementation unchanged since this run |

The full-profile compilation-unit boundary is now explicit: executable-owned
files seed eager startup; executable value/function demand activates dependency
implementation units and all their observable eager initializers to a fixed
point. Type/declaration imports alone do not activate runtime effects.
`ProgramUnitActivation` retains exact activation participants in F. Startup runs
before source main, while lexical module membership remains a reference relation.
This removes Alex's initializer scanning and preserves design-time visibility.

The final focused correction uses CCS
`9fe00834eda63cc9a4b91253b983d5e2aaf2e49fb4ba536fb3b271ae1f709e0f`.
Suspension liveness now consumes the exact writable program-cell authority for
scalar external cells, retaining its participants in the suspension relation.
It does not establish the backing lifetime of a descriptor or aggregate.
Continuation evidence passes **18/18**, including four new authority cases
(`/tmp/clef-c07-program-cell-liveness-tests.log`); related Alex components pass
**13/13** (`/tmp/composer-c07-program-cell-alex-tests.log`). Native 16a and 16f
remain green on this artifact:
`/tmp/composer-native-sequences-05728d81b98f4605a5ff9195e0aed2ec`,
`/tmp/composer-native-sequences-5286f211d8b0432d812bb60cbd726125`.
The unchanged full-profile **16g passes** stock MLIR, exact output and exit 0:
`/tmp/composer-native-sequences-177d299383c0462c8cfbb2925dfa8b8b`.
That run's full-profile input closure included the then-uncommitted Format
migration, although 16g did not call it; 16a–f used CompilerSurface without Format.
The character-storage entry above records a fresh 16g run with the now-published
formatter source and its hash.
It exercises two files, multiple modules, unused observable initialization,
once-only formation, deferred pulls, and repeated main/named-function use.
The broader suite and tooling records above retain their actual preceding hash;
they were not all repeated for this scalar residence correction.

**Acceptance remains open:**

- Original16 now reaches its actual factory/capture boundaries and reports 68
  CCS8403 diagnostics, chiefly returned captured environments, forwarded factory
  origins and their dependent residence failures. The original executable source
  and output are retained. `/tmp/composer-native-sequences-5d26f74e5bb74498bf3f0527060ca5b3`.
- New16h exposes missing staged elaboration for stored/bare Seq operation values.
  Their retained sequence and callable captures need C-01/C-02 admission; alias
  substitution must not replay already supplied effects. This also prevents
  complete-use classification of callbacks shared with those unresolved uses.
  `/tmp/composer-native-sequences-5de3f2e3cb2d49d2b7aae19b560f9cd6` records the
  rejection on `344c4583`; subsequent changes concern unit activation, not partial
  application. The oracle is unchanged and has not passed.

The PRD index distinguishes tested core implementation from full acceptance and
records the next dependency work. No hardware deployment, complete argv adapter,
new dialect family, or incremental graph repair is claimed by this waypoint.

### Companion revisions

Use Composer `d3365e1e7566` with these peer revisions for this implementation
waypoint. That commit excludes the then-pending roadmap, platform-format,
branding and sample-removal edits; the synchronization entry above tracks their
subsequent disposition.

| Repository | Revision | Scope |
|---|---|---|
| Composer | `d3365e1e7566` | Alex integration, native oracles, Editor/LSP query and initial PRD status index |
| BAREWire | `7b9de43700b7` | Explicit immutable/mutable program-space designation |
| Fidelity.Platform | `5a6c2860e826` | Existing profile spaces named without invented fallback |
| clef | `bf9632a05062` | Sequence, closure, startup and proof-incidence implementation |
| clef-lang-spec | `18e889419830` | Native composition, startup and successor contracts |
| lattice-analyzers | `2d476000275b` | Shared CCS projection and exact rejection/repair gates |
| lattice-vscode | `6c3fad5a75f0` | Live sequence/platform/startup protocol gates |
| ClefAutoComplete (`fidelity`) | `21930abd9a98` (unchanged) | Active semantic authority remains CCS; no separate platform/intrinsic catalogue was introduced |
| Fidelity.UI | `b7ef6f90e86e` (unchanged) | Design triangulation reference; no new UI syntax conformance claimed |

The original16 expected-output file intentionally retains trailing spaces from
the source oracle. They are observable output, not whitespace cleanup targets.

### Implementation trail

The following records earlier artifacts and the defects they exposed. The gates
and unresolved acceptance conditions above supersede their interim status.

Baker's producer and consumer recipes compose the shared iterator ingredient.
`take` checks demand before pulling; `iter`, `fold`, `exists` and `forall` retain
ordered eager operands and exact current-read certificates. Fold state uses its
own checked NTU type. Pull-effect relations retain exact possible generator
bodies for range invalidation; missing or mixed origins stay unresolved.
Generated integer literals use the NTU kind of their declared type, allowing
platform and range settlement to determine their storage width.

Captured callbacks now have explicit callable, environment and formal identities.
Baker materializes capture reads/writes and complete applications; Alex reads
placed environment slots and the actual environment occurrence. Mutable captures
retain the original cell. The bounded admission covers scalar immutable captures
and scalar cells within a proven covering activation. Returned child sequences
now retain eager immutable capture initializers and explicit borrowed-cell
incidence, with the full constructor path and covering allocation as premises.
Opaque and aggregate environments still need their residence/representation
contracts. [Closure values as data](Closure_As_Data.md) records the detailed seam.

| Gate | Evidence so far |
|---|---|
| CCS | 877/878 in the full run; the final Collect assertion was updated for the explicit environment argument and then passed unchanged in production: all 878 have passing evidence. `/tmp/clef-c07-full-tests.log`, `/tmp/clef-c07-collect-test.log` |
| Alex | 77/77, including six environment construction/recall and missing-premise cases; `/tmp/composer-c07-alex-tests.log`, before the two latest native corrections |
| Native 16a | Fresh compile, stock MLIR verification, exact output and exit 0: `/tmp/composer-native-sequences-2001ea0935c74f32b670175e8dfe5084` |
| Native 16b | The same gates for aliases, composed demand, repeated formation and shared mutable cells: `/tmp/composer-native-sequences-5cbcfcc2aa0c4859be0455b3e4cde5b1` |
| Native 16c | The same gates for both map/filter orders, multiple captures, empty/singleton inputs, deep composition and repeated enumeration: `/tmp/composer-native-sequences-187a041210b94fb9b89851a1bee6704a` |
| Native 16d | The same gates for captured returned children, shared mutable effects, variable/empty inner sequences, repeated enumeration and taking within a child: `/tmp/composer-native-sequences-f75e09e7b1bb4509a29ad2e4b6d3cf6c` |
| CCS child/match relations | Closure environments 11/11, factory results 9/9 and nested matches 9/9 on the 16d artifact; `/tmp/clef-c07-child-environment-final-tests.log` and `/tmp/clef-c07-child-environment-nested-match-tests.log` retain the runs and corrected concrete-type fixtures |
| CCS.Editor | Focused source signature/capture definition, dimensional error/repair and retained snapshot gate passes: `/tmp/composer-c07-closure-editor-tests.log`, on the earlier artifact |

16a, 16b and 16c used CCS SHA-256
`feb4d22032d202f2d161e76eb294387f95a2531c1dc49a03a33366139b912562`.
The earlier Alex and Editor gates used
`f554c883dda000e597d51b962adfee988cacbf4904041016321aa1b2e8d85d62`.
Later uncommitted source edits are not covered by those results.

Preparing caller-owned sequence storage prepends a destination parameter.
Environment identity now follows its retained `EnvironmentFormal` relation
through that transformation, including exact actual arity and environment owner,
instead of assuming argument zero. The isolated Collect probe passes fresh
compilation, stock MLIR verification and native exit 0 on CCS
`769c53b3737254bff8e4106c3bbf4ca88bef8e10234a6a588734e813b85732b0`:
`/tmp/clef-c07-collect-probe/compile-formal.log` and `verified.mlir`.

16d and the focused child/match cases used CCS SHA-256
`1c71c4455a551205244c312bf364dd8bdfeaa5f1a1e3f11a1fdc993181ff5b85`.
The earlier returned-child failure is resolved by the resident capture,
initializer, formal and allocation relationships; no Alex source scan supplies
them. Nested constructor matches retain branch-local extraction and exact
fallback paths. Tuple/record payload patterns remain a separate admission gap.

Search consumers share guarded iteration; the earlier 12 source/graph cases
passed in `/tmp/clef-c07-search-tests.log` and
`/tmp/clef-c07-search-protocol-tests.log`. Native investigation exposed generated
Option intrinsics left unsaturated. A shared native Option ingredient now
constructs their complete DU structure; its stronger 29-case Option/search gate
is recorded below. The permanent 16e fixture retains its Option-valued element
case. The temporary scalar-only probe is diagnostic evidence, not a replacement
acceptance fixture.

The original 16 fixture now uses the full default profile and its original
executable statements and expected output. `/tmp/composer-native-sequences-78aaf168d4e74080abffbb45cec8ed40`
records 172 CCS8403 diagnostics, principally missing module-initialization
activation and returned storage. Ordered startup ownership must move into Baker
before reachability, with complete initialization/use coverage and explicit
storage authority. Alex's former module-initializer prologue scan supplied no
such proof; the startup graph replacement and its gates are recorded below.

Finite additive loop work constructs joint induction/accumulation relations and
source/build obligation projections. It binds the admitted guard, initial values,
step, stores and exact update expressions; arbitrary-width range settlement
precedes physical carrier selection. On CCS SHA-256
`90f079076d5878fad73d534f2878eb3bd1b1024925e453c89be8659747579d87`:

| Gate | Evidence |
|---|---|
| CCS loop relations | 15/15, including control/effect/privacy negatives and evidence replacement: `/tmp/clef-c07-loop-range-tests.log` |
| Shared Option/search | 29/29 (17 Option and 12 search), requiring complete typed DU structure rather than dormant Option intrinsics: `/tmp/clef-c07-options-selection-tests.log` |
| SMT transfer | 85/85, including 20 new source/native loop parity cases and false numeric claims: `/tmp/composer-c07-loop-smt-parity.log` |
| Native 15d | Fresh compile, stock MLIR verification, exact eight output lines and exit 0: `/tmp/composer-native-sequences-fb0caeada794416a8c9ee17572ead115`. Triangular/repeated enumeration, negative/mixed deltas, zero trips and non-unit ascending/descending steps |
| Editor loop proof | Source-cell/store navigation, bound edit `[0,36]` → `[0,9]`, actual cvc5 dispatch, retraction, repair and retained snapshots: `/tmp/composer-c07-loop-editor-tests.log` |
| Analyzer projection | 52 accepted groups and 57 exact diagnostic rejections, capture identity and snapshot repair: `/tmp/lattice-ccs-surface-3bd0baf316b54ff984553df8f04326c1/evidence.json`. The captured producer read now expects `EnvironmentRead`, preserving its exact original declaration span |
| Actual LSP | 61 diagnostic edits/repairs plus sequence signatures and capture definitions: `/tmp/lattice-surface-waypoint-KZhjb8/result.json`, Node 22.23.2, server using the same CCS artifact |

The original 15 compile at `/tmp/composer-original15-additive-lw25qz_7/compile.log`
has eight errors. Its triangular recurrence no longer has a settlement error;
Fibonacci and power fields remain unresolved, alongside six module-template
activation failures. Coupled/multiplicative recurrence evidence must include
ordered intermediate values and final stores, not just yielded values; the
additive certificate does not establish a matrix/power enclosure.

Continuation scratch now distinguishes actual control-owned declarations and
captures from external assignment targets. A `Set` no longer silently creates a
private copy of an external cell. All 14 continuation-evidence cases pass, and
the scalar search probe passes fresh compilation, stock MLIR verification,
exact output and native exit 0 on CCS `309bd46f…add64de`:
`/tmp/clef-c07-search-scalar-8ef53a111bc2453bb658c89dfc5d76ee`.

Scalar-payload Option sequence values now have owned byte regions, explicit
selected-case initialization, and consumer snapshots. Seven core aggregate
cases pass on CCS `993291d5…172511`, including retained-snapshot evaluation and
rejecting missing current/use/residence evidence. Ten new Alex component cases
pass in `/tmp/composer-c07-aggregate-pattern-tests.log`; Some and None operations
also pass stock MLIR verification and LLVM lowering. The preceding 77 Alex
cases passed on the same production artifact. This does not admit aggregate
captures or infer lifetime from a descriptor copy.

Permanent 16f adds retained Option values, repeated constructor sites, independent
enumerators, delegation and Boolean/measured-real payloads. Both 16e and 16f now
reach Alex. Native gates exposed missing facts in the copy operations constructed
after range analysis: Option tag bounds are corrected at ingredient construction;
selected payload range/carrier transport remains open. Latest retained failures:
`/tmp/composer-native-sequences-df0716b307ce456f87d08c8484172431`
and `/tmp/composer-native-sequences-238ee1853ed84e90b51b7cb3c7d6bab7`.
Component success is not recorded as native conformance. On CCS
`74269f55…ab31b16`, both native fixtures reach stock MLIR verification, which
rejects a copy destination established only inside the Some arm. 16e also exposes
a nested-match fallback shared across exclusive regions. The subsequent
batch sequences the destination before case selection and creates distinct
branch occurrences with explicit source incidence. Recipe fold-in now carries
new hyperedges through the same simultaneous replacement map as the nodes.
On CCS `feed086b…103b`, the startup structure cohort passes 9/9, nested matches
and branch occurrences 12/12, and aggregate/FoldIn cases 14/14. The final batch
also guards cyclic declaration aliases and projects executable entry bindings
before reachability, keeping lexical descriptor membership out of startup demand.

Program initialization is moving into Baker before first reachability. The
working implementation separates a generated startup activation from callable
source `main`, preserves lexical module membership without treating it as
execution containment, and records ordered initialization, slot intent, writable
authority and complete-call lifetime dependencies separately. The two-module-per-
file order reversal is corrected. New native 16g retains unused observable
initializers and checks repeated sequence use from main and a named function.
Its native gate and the final combined graph cohort remain pending. Composer
builds against CCS `bedea1f6…b7f26`; Alex's module-initializer scan is removed and
the generated entry uses the ordinary function witness. Three component cases
on the preceding `feed086b…103b` artifact pass real MLIR verification and refuse
missing/mismatched writable authority before constructing a slot.

Startup must also be inspectable during source editing. Its initializer order,
source identities, slot intent, storage authority and pending prerequisites belong
to the same PSG consumed by CAC and Lattice. The peered gate is adding a direct
projection of those relations; it does not infer execution from module order in
the client. Hardware bring-up and numeric selection remain governed by the
selected platform's declarations and graph obligations. The hosted entry's
pre-existing source `argv` conversion gap is not closed by the startup wrapper;
these native oracles ignore that argument.

The companion storage declaration is explicit across BAREWire, Fidelity.Platform,
CCS and Composer target projections. `ProgramLifetime` names existing immutable
and optional mutable spaces; absent authority supplies no placement fallback.
Read-only image authority does not permit runtime initialization writes. BAREWire
passes 580 checks (`/tmp/barewire-program-lifetime-tests.log`), and 16 new CCS
authority cases plus four existing string-layout cases pass on `74269f55…ab31b16`
(`/tmp/clef-program-lifetime-tests.log`). Named-space and proof-incidence tests
include exact declaration errors, aliases, missing/ambiguous spaces and rejected
runtime descriptor factories. Slot layout/capacity is distinct from role authority.
Affected Editor/analyzer/LSP projection and repair gates are being added; earlier
sequence tooling passes do not establish these newer contracts.

Cleanup removes duplicated recipe node constructors in favor of Ingredients and
three unused Alex prototypes: code-address-in-environment construction,
capture-prepending invocation and fixed-width global arena allocation. The active
legacy lazy representation and mutable traversal/scope driver remain separate
architectural work; removing unused prototypes does not establish their migration.
Search consumer retooling and peered analyzer/LSP validation are still in progress.

## C-06 native continuation settlement — 2026-09-20

**Native continuation implementation waypoint; aggregate C-06 regression gate
remains open.** This combines the sequence feature's source, graph, placement,
proof, middle-end, native-oracle and tooling work into one coordinated revision
set. It does not mark the five compilation failures below as successful gates.

Baker now composes its local evaluation relationships into control occurrences,
definite value availability, cuts, resume entries and live-across sets. Recipes
construct Boolean MoveNext bodies as ordinary PSG frame accesses, conditionals,
loops and literal-case dispatch. A persistent discriminant names the resume
cut; a local dispatch position sequences work within one pull. Alex follows
those regions through its existing Huet zipper and passive witness function,
using standard `scf.index_switch`, `func`, `memref` and arithmetic operations.
The standard MLIR pipeline lowers structured control to `cf` and LLVM.

Source `for ... in` declarations retain real identities. Consumption and
`yield!` share the guarded iterator ingredient. Current-read certificates name
the exact iterator, successful guard and consuming loop. Element-range evidence
retains that prerequisite together with all possible owners and yielded payloads;
the range fixed point joins their bounds. Missing incidence or an unknown
alternative prevents a narrow result. A zero-cut body still runs its effects on
the final pull; its current identity requires no physical payload storage.

Placement separates persistent values from scratch needed only within one pull.
A mutable cell retained by a child extends storage liveness even after the outer
body stops reading its scalar value. Layout obligations retain actual slot
participants, offsets, extents and alignments. Source cut/resume identities and
numeric state obligations survive machine elaboration. These relations do not
claim that layout checking alone proves lifetime, target capacity or the full
linear-continuation contract.

Named, fully applied factories can receive explicit caller-owned destination
storage. Generator-local children occupy distinct, bounded regions of the parent
frame. Captured mutable cells retain their original identity. Physical frame
storage is unboxed: retained descriptors carry addresses, offsets, extents and
strides; source NTU types and proof information guide compilation rather than
becoming runtime type objects. Generated capture reads/borrows occur explicitly
before constructor initialization. Layout-owner incidence is provenance, so it
does not reactivate a retired source initializer on the emission spine.

Native frame construction requires successful source admission. Invalid source
types keep their precise source diagnostic without additional errors from
attempting to settle a frame from the rejected premise. Valid source whose
control, current-read, layout or residence prerequisites remain unresolved still
receives an explicit settlement diagnostic. No heap/static residence fallback
is inferred for an escaping template or captured cell.

| Gate | Current evidence |
|---|---|
| CCS | **848/848**, zero skipped, on final CCS `f4bbc287…432c1a`; `/tmp/clef-c06-captured-templates-full-tests.log` |
| Alex | **71/71** on CCS `08d54752…84482`, before the native-only captured-template extension; `/tmp/alex-c06-final-admission-tests.log` |
| SMT transfer | **65/65**, including 15 continuation-layout source/native/cvc5 parity and false-claim cases; `/tmp/composer-smt-c06-final-admission.log` |
| Native 15a | Fresh compilation, stock MLIR verification, exact output and exit 0; `/tmp/composer-native-sequences-d001fa03098148f8a94fb2fdf37454e2` on final CCS `f4bbc287…432c1a` |
| Native 15b | Boolean, unit, real, measured integer and inverse measured real payloads; same required native stages; `/tmp/composer-native-sequences-1d8223ce127843b7884aa296956c0e2f` on earlier CCS `5dad941b…40f44` |
| Native 15c | Scoped captured templates, repeated deferred delegation, shared cells and nested capture levels; fresh compile, stock MLIR verification and exact native output; `/tmp/composer-native-sequences-6f767e0ab8374941a1d041b0b9789985` on final CCS `f4bbc287…432c1a` |
| Public CLI | 12/16 cases passed before the source-admission correction; the four original type/dimension rejections pass unchanged on the corrected artifact. `/tmp/composer-source-admission-000edec5a3b74f4286abaac3d3dba11c`, `/tmp/composer-source-admission-15540117b5aa4ec4928ea13d70ce4f38` |
| FidelityHello 01–15 plus variants | **23/28 compiled, 23/23 executed successfully**, no skips; final CCS `08d54752…84482`. `/tmp/composer-c06-final-regression.log` includes the separate compilation failures below. Both 15a and 15b pass this final-artifact run. |
| Analyzer projection | **44 accepted / 46 exact rejected** on CCS `08d54752…84482`; `/tmp/lattice-ccs-surface-a5571e6451fd49eeb0018150240a0396/evidence.json` |
| LSP | **50 diagnostic edits and repairs**, source capture/induction definitions and normal server exit; same artifact; `/tmp/lattice-surface-waypoint-gTl1Dg/result.json` |

15a checks literal and repeated enumeration, delayed pre/post-yield effects,
conditional and counted loops, empty effects, caller-owned factories, delegation,
nested independent iteration, retained mutable child captures, factories inside
generators and chained empty inputs. 15b adds scalar/NTU element representation
coverage; fractional numeric values do not claim admission of fractional measure
exponents. 15c passes scoped captured-template delegation. Its finite borrow relation retains
the source allocation, covering activation, captured declaration, generator and
constructor. Complete use must stay within the covering lifetime; lexical nesting
alone cannot authorize it. The dedicated residence/factory/region cohort passes
**30/30**, including return/store/opaque/unknown and ambiguous-owner negatives.
The original 15 remains a separate oracle; neither a 15a nor a 15b pass
substitutes for it. Its accumulating generators require observable frame ranges
that the current recurrence analysis does not establish. Other recorded failures
are 05's `Format.float` result-carrier mismatch, 06's legacy `int` conversion
names in Parse, 13's unresolved generic integer width, and 14's lazy width/extent
read. These are observed failures, not skipped or successful gates. The ordinary
Option/Result/control/capture oracle families pass. Board deploy pipelines have
not been exercised.

The existing CAC drift gate reports the same five findings outside these changes;
CAC remains a retired bridge, with CCS.Editor as semantic authority. The source
projection fixtures and tracking docs in lattice-analyzers, lattice-vscode and
CAC move with this feature. The coordinated revisions below identify this implementation waypoint; the
aggregate regression gate remains open.

The next feature is C-07. Its revised PRD follows these graph contracts rather
than historical wrapper emission: captured-template residence feeds append;
callback producers need the full callable/environment contract; take requires
a count guard before input demand; consumers must use certified iteration, and
fold must retain independent state and element types. Unknown/mixed callable
origins, escaping/reference-capturing factory results and aggregate storage
budgets remain explicit contracts, not inferred successes.

Final CCS artifact SHA-256:
`f4bbc2879280b8252e3c7424a1b399e981eb492e49b07fb1def617d45f432c1a`.
Composer artifact SHA-256:
`6074e2fbe0d7f4339b3302ebb50673f53c06c69e6d352ba60c24711391258580`.
The projection gates identify their earlier source-equivalent CCS artifact
explicitly; Editor and Server copies were refreshed to the final artifact after
the native-only residence extension. No repeated full tooling run is implied.

Companion revisions: clef `12aa78d2b`, clef-lang-spec `1b1ab6b`,
lattice-analyzers `24e4a667`, lattice-vscode `32dbe19e`, CAC `21930abd`.
The Composer commit containing this entry is the coordinating anchor.

## C-06 local evaluation relations before suspension segments — 2026-09-20

Baker records local evaluation demands and entry/completion ports on the graph
after curry normalization, so those relations reference the final operand
identities. Ordered operands remain distinct from references to values already
produced. A binding's initializer precedes its result; a definition reference
does not cause that initializer to execute again.

Conditional paths retain their guard and selected branch, including the nested
conditionals that implement short-circuit expressions. Loop relations distinguish
the initial guard, successful body path, body-to-guard backedge and exhausted
continuation. Creating a lambda, lazy value or nested sequence retains its
capture-formation boundary without entering the deferred body. At a yield, its
payload demand precedes the suspension's `Resume` continuation. These facts are
compositional local relations; they do not constitute a complete global control-
flow graph or prove dominance, path feasibility or segment liveness.

`EvaluationOperand` distinguishes value demands from assignment storage;
`EvaluationFlow` connects ports of its target node. Operand identities named by
those ports are explicit edge participants. Root and capture relations retain
the owner/generator and captured declaration identities. Unsupported local forms
carry `EvaluationPending` residuals; no fall-through execution path is invented.

Alex production code remains unchanged. The added component case parses and
checks a guarded loop sequence, requires the final graph's evaluation facts,
then observes its unresolved yield at a Huet focus. The expected result remains
the explicit missing-frame diagnostic, no emitted operations, and no graph,
zipper or accumulator mutation. Local evaluation evidence does not authorize
native suspension witnessing.

| Gate | Result |
|---|---|
| CCS | **740/740**, zero skipped; focused sequence cohort **89/89**, including 17 new evaluation cases; `/tmp/clef-sequence-evaluation-full.log`, `/tmp/clef-sequence-evaluation-tests.log` |
| Alex | **9/9** sequence boundary cases, including parsed guarded-loop evaluation facts at a real Huet focus; `/tmp/composer-sequence-evaluation-alex.log`. Tests built with `BuildProjectReferences=false`; no compiler references rebuilt |
| Public Composer | **3/3** selected SourceAdmission cases: exact map/append dimension rejections before artifacts and ordinary FP control with stock MLIR verification/native execution; `/tmp/composer-source-admission-5c758e6b5ab44809b762bec0555916f2/` |
| FidelityHello | **11b_LoopCaptures passes**, fresh compilation, exact output and native exit; `/tmp/composer-sequence-evaluation-fidelityhello.log` |
| Analyzer projection | **41 accepted / 45 exact rejections**, revisions 1–106; `/tmp/lattice-ccs-surface-f378d9c5505f463aa6c2f621ff83023c/evidence.json` |
| LSP | **49 diagnostic edits and repairs**, guarded-loop types, lambda signature and four original capture definitions; `/tmp/lattice-surface-waypoint-BkOf1a/result.json` |

Both tooling gates loaded CCS SHA-256
`5e228d39c0d0134229834bb9b0df818fa6b3bd02aea00f48f9144b7a16aa302f`.
The separate CAC documentation drift gate's five pre-existing findings remain
unchanged, with evidence in `/tmp/lattice-sequence-ownership-doc-drift.log`;
that gate was not rerun for this checkpoint.

Suspension segments, live-across storage, frame extent and placement, Boolean
resumption and their proof obligations remain upstream work. The aggregate
storage-budget follow-up recorded below also remains open. The passing source,
tooling and witness checks are separate from any eventual native sequence gate;
this checkpoint establishes no native sequence execution result.

Companion revisions: clef `945c3e9b9`, clef-lang-spec `0d5db4e`,
lattice-analyzers `aea8a54`, lattice-vscode `72df049`, CAC `fed959a6`.

## C-06 delegation iteration before suspension segments — 2026-09-20

Baker now expands admitted `yield! input` into iteration within the delegating
owner. The shared `Ingredients.Sequences.iterate` ingredient is also used by
sequence producers: it initializes one enumerator, checks `moveNext`, binds
`current` once on success, and runs the supplied unit action. Delegation supplies
an ordinary yield as that action. Exhaustion follows the while loop's false
path; an empty delegated input does not introduce a yield before the surrounding
computation continues. The original input is evaluated once when execution
reaches that delegation, not when its enclosing sequence value is created.

The source site's identity and range remain on a unit `Sequential` wrapper.
Generated protocol nodes have point source anchors. A `DelegationOrigin`
provenance relation joins the original site and input to the generated yield;
the owner/generator delimiter relation transfers to that yield and ownership
is checked again after fold-in. A supplied sequence's own suspension sites
retain their separate owner. Existing enclosing branches, loops, operand
identities and proof incidence remain attached to the source wrapper.

Alex production code is unchanged. An additional boundary test uses public
`parseAndCheck` on Boolean delegation, finds the actual Baker-generated yield
with both provenance and delimiter relations, and observes it at its Huet focus.
The required result remains an explicit missing-frame error with no operations
or graph/accumulator changes. These facts do not authorize native suspension.

| Gate | Result |
|---|---|
| CCS | **723/723**, including 11 delegation cases; focused delegation/ownership/producer/element cohort **64/64**; `/tmp/clef-sequence-delegation-full.log`, `/tmp/clef-sequence-delegation-tests.log` |
| Alex | **8/8** sequence boundary cases, including the source-derived delegated yield with both resident relations; `/tmp/composer-sequence-delegation-alex.log` |
| Public Composer | **3/3** selected SourceAdmission cases: exact map/append dimension rejections before artifacts, ordinary FP control with stock MLIR verification and native execution; `/tmp/composer-source-admission-10e7973256424b53afc7dec8f30a8725/` |
| FidelityHello | **11b_LoopCaptures passes**, fresh compilation, exact output and native exit; `/tmp/composer-sequence-delegation-fidelityhello.log` |
| Analyzer projection | **39 accepted / 45 exact rejections**, revisions 1–104; `/tmp/lattice-ccs-surface-b7f9259f2ca54e88afb7bdbbc82052d8/evidence.json` |
| LSP | **49 diagnostic edits and repairs**, original `yield!` span/unit result, nested append/collect types and captured-storage definitions; `/tmp/lattice-surface-waypoint-eHlq9I/result.json` |

Both tooling gates loaded CCS SHA-256
`167ef2f9d2124344499ff9bf900961410fd10b2a022203cd2fbb1a8f6527e1a6`.
The source tests also preserve unrelated resident proof relations, the source
wrapper's emission boundary and the original input's range, and require a
second delegation pass to be a no-op. Unowned or scalar sites are left intact
for their admission diagnostics; no owner or sequence element type is invented.
Specification `2d5a85b` records delegation timing and ownership.

The next step is graph-resident evaluation order, preserving conditional choices,
joins, loop backedges and deferred boundaries before segment liveness is
computed. That relation must distinguish reuse of an already evaluated value
from a new evaluation; a definition reference does not re-run its initializer,
and a global visited set does not determine evaluation multiplicity across loops.
This checkpoint does not supply a complete control-flow graph,
suspension segments, frame layout, Boolean resumption or lifetime proofs.
The aggregate storage-budget follow-up recorded below remains open.

Companion revisions: clef `a3c43be9b`, lattice-analyzers `d605d4d`,
lattice-vscode `a35bf3f`, CAC `de169cb4`.

## C-06 delimiter ownership and passive witness boundary — 2026-09-20

Baker's `Suspensions` ingredient constructs a `Suspension/Delimiter` hyperedge
whose ordered sources are the sequence owner and its generator, and whose target
is that owner's `Yield` or `YieldBang` site. `SequenceOwnershipRecipes` visits the
canonical structural relation beneath each reachable owner's generator body.
Nested sequence expressions establish their own ownership; ordinary lambda,
lazy and quotation bodies are separate deferred boundaries. Definition references
do not cause a traversal into a called function's body.

The `SequenceOwnership` nanopass runs after producer and capture elaboration.
It diagnoses malformed or multiply owned reachable suspension sites and folds
only delimiter relations into the graph. Repeating the pass replaces that
projection while retaining unrelated hyperedges. Conditional branches and loop
bodies retain their original structure: lexical ownership does not establish
whether a guarded site executes or where evaluation resumes. A false guard is
not permission to discard its surrounding effects.

The obsolete `SeqSaturation` coeffect and its body-shape classifier are removed
from graph construction and fold-in. That scan crossed deferred owners, flattened
body structure and guessed internal frame indices. Its removal establishes one
authority for ownership; it does not replace the missing suspension recipe.
Alex's independent mutable-binding scan and placeholder current-value type are
also removed. The public Seq nanopass returns an explicit error at unelaborated
`SeqExpr`, `Yield` and `YieldBang` focuses, including graphs that already carry a
delimiter relation. Focused component cases require zero emitted operations and
unchanged graph, zipper and accumulator state; unrelated nodes remain available
to other witnesses. The existing `ForEach` path is unchanged.

| Gate | Result |
|---|---|
| CCS | **712/712**, including 15 ownership cases for source/generated/nested/deferred/delegated sites, guarded and effect-only bodies, malformed ownership (`CCS8402`) and relation replacement/retraction; `/tmp/clef-sequence-ownership-full.log` |
| Alex | **7/7** public witness cases at explicit Huet focuses: unsettled owner/yield/delegation rejected with and without delimiter evidence, unrelated focus skipped, no operations or graph/accumulator changes; `/tmp/composer-sequence-ownership-alex.log` |
| Public Composer | **3/3** selected SourceAdmission cases: exact map/append dimension rejections before target artifacts, ordinary FP control with stock MLIR verification and native execution; `/tmp/composer-source-admission-4403d1740ef5475a92a573188006743f/` |
| FidelityHello | **11b_LoopCaptures passes**, fresh compilation, exact output and native exit; `/tmp/composer-sequence-ownership-fidelityhello.log` |
| Analyzer projection | **37 accepted / 45 exact rejections**, revisions 1–102; `/tmp/lattice-ccs-surface-e3301ee0a7b94705b7ededffe037e9c7/evidence.json` |
| LSP | **49 diagnostic edits and repairs**, nested/delegated/effect-only sequence hovers and original captured-storage definitions; `/tmp/lattice-surface-waypoint-gvFlXL/result.json` |

Both tooling gates loaded CCS SHA-256
`4b82f075ecfae0504e75bccafdd5453c9c4078bfdc25dee4470bbc046d179c8d`.
Specification `fd25a36` states the ownership law separately from suspension
execution. This is a whole-projection re-fire today, not a claim that incremental
dependency-directed invalidation is implemented.

Evaluation segments, post-yield continuation, short-circuit and loop behavior,
delegation, live-across storage, Boolean resumption and frame extent/lifetime
obligations remain subsequent Baker work. Empty effectful bodies also require
correct enumeration behavior. This waypoint establishes neither native sequence
execution nor aggregate storage bounds. Public SourceAdmission remains a source
rejection gate with an ordinary FP native control; valid sequence source and
editor projections do not substitute for the missing native frame contract.

The separate CAC documentation drift gate currently fails on **five pre-existing
retired-vocabulary lines**, recorded in
`/tmp/lattice-sequence-ownership-doc-drift.log`. They are outside this checkpoint's
modified files. Its scheduled-code and inherited FCS-surface counts are inventory,
not additional failures; no broad documentation cleanup is included here.

Companion revisions: clef `57dccfcfa`, lattice-analyzers `9fc09b2`,
lattice-vscode `4eb588e`, CAC `7153f1f2`.

## C-06/C-07 producer graph and timing contracts — 2026-09-20

`Seq.map`, `filter`, `collect` and `append` now form immutable snapshots of their
supplied operands in source argument order. Generator-local references resolve
to those snapshots. Enumeration and callback invocation remain inside the
deferred generator: the enumerator is bound at entry, and each current element
is bound before callback execution. Filter's predicate and yielding branch use
that same current value. Append captures both inputs eagerly and delegates in
order within its generator.

The shared sequence ingredient creates the same owner/generator/formal structure
as source elaboration, preserving parent and canonical parameter relationships.
Yield and delegation operations have unit type; their payload nodes retain their
own element/sequence types. The three Map traversal helpers using this ingredient
now have unit generator bodies, local tree capture references, explicit outer
formals and tuple children. These helper checks do not establish native Map or
sentinel representation conformance.

Generated producer nodes carry point source anchors; the replacement expression
retains the source call's full range. Existing operand ranges and captured storage
identities remain intact. Normal fan-out/fold-in preserves both the affected
application obligation's incidence and unrelated proof relationships.

| Gate | Result |
|---|---|
| CCS | **697/697**, including eight direct recipe/fold-in cases, four source producer cases, four exact negative cases and three shared tree-helper cases; `/tmp/clef-seq-producers-full.log` |
| Baseline | **12 intended failures / four passing negative controls** against the previous compiler; `/tmp/clef-seq-producers-before.log`, `/tmp/clef-seq-producers-source-before.log` |
| Public Composer | **3/3** selected SourceAdmission cases: exact map/append dimension rejections before target artifacts, plus ordinary FP control with stock MLIR verification/native execution; `/tmp/composer-source-admission-332c69578adb4dc89a03767a5fd050c9/` |
| FidelityHello | **11b_LoopCaptures passes**, exact output and native exit; `/tmp/composer-seq-producers-fidelityhello.log` |
| Analyzer projection | **35 accepted / 44 exact rejections**, revisions 1–98; `/tmp/lattice-ccs-surface-0379d88e676540d8b63ed2c42d593af2/evidence.json` |
| LSP | **48 diagnostic edits and repairs**, four producer application-result hovers and original captured-threshold definitions; `/tmp/lattice-surface-waypoint-cdMfSm/result.json` |

Both tooling gates loaded CCS SHA-256
`49a1003ff9a3606844b2f03d63f5ec6240f7ce9b9ceef3071faa40fe41b52c3a`.
Specification `3b64906` clarifies that independent iteration state preserves
sharing of storage captured by supplied function values. Site revisions
`dec4d4e` and `52ca6de` remove the superseded closure-dialect section from
"Seq'ing Simplicity" and explain liveness and initialization boundaries.

This checkpoint establishes producer graph contracts. Native suspension cuts,
live-across slot assignment, Boolean resumption construction, frame extent and
lifetime obligations remain pending; Alex is unchanged. Next, establish cut
ownership and graph evaluation order before frame settlement. Conditional and
nested yields, pre/post-yield effects, delegation and empty-but-effectful bodies
must preserve their source behavior.

**Prospero/Ariel follow-up, independent of actor topology:** growth in retained
sequence state through nesting, composition or consumption can create memory
pressure even before a full actor topology exists. A bounded individual frame
does not establish a bound on total live sequence storage. Memory accounting,
monitoring and target-budget policy for that growth need consideration alongside
later suspension work; this checkpoint introduces no accounting or scheduling
policy and keeps the initial implementation focused on correct semantics.

Stack-only working-memory profiles for small-device unikernels are a concrete
case for that follow-up, potentially including the post-quantum credential.
The [sequence lifetime contract](../../clef-lang-spec/spec/seq-representation.md)
and [suspension placement contract](../../clef-lang-spec/spec/dcont-representation.md)
already require storage whose lifetime covers every use. For a stack-backed
suspension, resumption must remain within the lifetime of its backing storage.
The further budget question concerns simultaneously retained sequence frames,
captured storage and delegated/nested state alongside the target's other stack
requirements. A literal extent for one frame does not answer that question;
unbounded iteration alone does not imply growing retained state either. Keep
placement/lifetime admission and aggregate memory-budget evidence explicit in
Baker's graph contracts, with Alex passively witnessing the settled result.
Prospero/Ariel accounting and monitoring are follow-up work, including before
actor topology; this checkpoint establishes neither aggregate bounds nor a
runtime monitor.

Companion revisions: clef `a06997a82`, lattice-analyzers `56159c6`,
lattice-vscode `ddda782`, CAC `a95e0313`.

## C-06 resident sequence generator formal — 2026-09-20

The source sequence generator previously named `NodeId -1` as its formal. It now
owns a real typed `PatternBinding`, ordered before its body, with the canonical
parameter relation and matching parent. The formal's type agrees with its tuple
and generator domain, retaining the current internal sequence-pointer type.
Its zero-width source anchor preserves file/point provenance through the owner
without occupying a source token or entering the user's lexical environment.

All six new cases failed on the old missing formal. They now pair actual
`01_psg0` artifacts with the saturated graph: identity, ordered children, parent,
types and canonical `kindEdges` relations agree. Nested owners have distinct
formals. Mutable captures and an immutable source binding also named `_seq_ptr`
retain their original definition identities. Structural relations are projected
by `kindEdges`, as documented by the graph contract; these tests do not require
duplicating them in the explicit n-ary enrichment edge collection.

| Gate | Result |
|---|---|
| CCS | **678/678**, including six formal cases; `/tmp/clef-sequence-formals-full.log`; baseline `/tmp/clef-sequence-formals-before.log` |
| Public Composer | **4/4** selected SourceAdmission cases: three exact sequence rejections with no target artifacts and ordinary FP control with stock MLIR/native execution; `/tmp/composer-source-admission-f806f10b6e444b2c83141d0c359fcbdc/` |
| FidelityHello | **11b_LoopCaptures passes**, exact output and native exit; `/tmp/composer-sequence-formals-fidelityhello.log` |
| Analyzer projection | **33 accepted / 42 exact rejections**, revisions 1–92; `/tmp/lattice-ccs-surface-072679c2d2824ef480d70b06efb284ca/evidence.json` |
| LSP | **46 diagnostic edits and repairs**, source `SeqExpr` hover, captured factory/result/seed hovers and exact seed definition; `/tmp/lattice-surface-waypoint-dCjrnm/result.json` |

Both tooling gates loaded CCS SHA-256
`b936e97e2cd68776a52dd72cc51368cc92856c10e0d37cf4f21080843ef4269d`.
This repairs source graph identity. The generator still contains an unelaborated
unit yield body; constructing its Boolean resumption result requires the
suspension recipe. Frame representation, cut segmentation, live-across placement
and lifetime obligations remain pending. Alex is unchanged.

The next upstream dependency is consistency with recipe-produced sequences:
`clef/src/Compiler/Baker/Ingredients/Primitives.fs` still creates raw-body
`SeqExpr` nodes and gives `yield'`/`yieldBang` element types instead of unit.
`SeqRecipes` expands HOFs into these structures; `BakerSaturation` does not yet
select source `SeqExpr`/`Yield` for suspension construction. Align those producer
contracts before implementing owner-scoped cuts, segments and live-across facts
through Baker's existing recipe fan-out/fold-in seam. The canonical
[suspension contract](../../clef-lang-spec/spec/dcont-representation.md) requires
VC-EXT, VC-STATE, VC-ACC, VC-DOM and VC-ONE; existing obligation ingredients are
integration mechanisms, not evidence that these proofs already exist.

Companion revisions: clef `62a0d38dc`, lattice-analyzers `3afbad7`,
lattice-vscode `a83fd82`, CAC `9d09823d`.

## F-09 Result case predicates — 2026-09-20

Specification `5f49a02` defines `Result.isOk` and `Result.isError` as unary
predicates with two independent payload parameters. Baker composes the existing
typed tag read and comparison ingredients, preceded by the original input.
Predicates never extract or invoke a payload. Bare values become ordinary unary
closures; their uses retain resolved parameter types and application obligation
participants. No Alex witness or target layout rule was added.

| Gate | Result |
|---|---|
| CCS | **672/672**, including 20 predicate cases with eight exact negative cases; `/tmp/clef-result-predicates-full.log` |
| Native / MLIR | **2/2** ResultCases (eight groups, 141–148) and ResultElimination (15 groups), stock verification and native exit zero; `/tmp/composer-callbacks-fsharp-555a1a0ad1fb4a7293ad8f1eb33d0b8c/` |
| FidelityHello | New **09c_ResultCases passes** five groups with exact six-line output; `/tmp/composer-result-predicates-fidelityhello.log` |
| Analyzer projection | **32 accepted / 42 exact rejections**, revisions 1–91; `/tmp/lattice-ccs-surface-81efcb271ba34a32b0ad229a1120da7c/evidence.json` |
| LSP | **46 diagnostic edits and repairs**, 12 Result hovers; `/tmp/lattice-surface-waypoint-fI5CJr/result.json` |

Native cases include eager factories and pipes, stored predicate identity,
independently measured and inverse-dimensional payloads, callable construction
without invocation, unit and record payloads, lexical shadowing, and Boolean
short-circuit composition. Every native Result fixes both payload types; tag-only
use does not authorize inventing a representation for an unresolved payload.
Source tests additionally retain captured storage identity and exact application
obligation relationships. Both tooling gates loaded CCS SHA-256
`da5931aca2af35313163c8e444a8339d725e37db591d40bcf4125154438a3809`.

Companion revisions: clef `eba9b3349`, lattice-analyzers `aa22093`,
lattice-vscode `df638c1`, CAC `d6fdb7f3`.

## C-06 sequence owner and element constraints — 2026-09-20

Sequence elaboration creates the actual owner before checking its body. Each
`yield` constrains that owner's element type; `yield!` constrains its operand to
the owner's sequence type. Nested sequences create independent owners. Completing
the owner preserves its identity and replaces its temporary body reference and
children together. This replaces inference from the first descendant yield.

The baseline accepted all 14 invalid element/delegation cases and inferred four
outer sequence types from nested sequences. All 18 regressions now pass, alongside
accepted dimensional, nested and typed-empty controls and an exact annotation
conflict. The source fixture checks both the shared constraints and completed
owner/child/parent relationships.

| Gate | Result |
|---|---|
| CCS | **652/652**, including 30 sequence cases; `/tmp/clef-sequence-elements-full.log` |
| Public Composer | Three new exact rejections with no MLIR/executable, plus ordinary FP control with stock MLIR verification and native execution. Mixed dimensions/control: `/tmp/composer-source-admission-c078a06c55c14a478711e78e15e78a59/`; scalar delegation/annotation: `/tmp/composer-source-admission-324e9f34511f4b799df3dd97b86fd9fa/` |
| FidelityHello | **11b_LoopCaptures passes** compilation, exact output and native exit; `/tmp/composer-sequence-elements-fidelityhello.log` |
| Analyzer projection | **30 accepted / 40 exact rejections**, revisions 1–85; `/tmp/lattice-ccs-surface-02670ff7e1ac411cb37190521d74fb74/evidence.json` |
| LSP | **44 diagnostic edits and repairs**, including three sequence repairs and independent outer `seq<int<m>>` / inner `seq<bool>` hovers; `/tmp/lattice-surface-waypoint-KmwzZQ/result.json` |

Two initial CLI expectations used a short embedded filename; project diagnostics
carry the absolute source path. Correcting those exact expectations passed on
the same binaries. Both tooling gates loaded CCS SHA-256
`44f1fb2af812e9f7ea81e0ea1708198e57b8b54a14a41c5488338e8c03db3aa5`.

This implements the element constraint portion of C-06. The existing MoveNext
formal placeholder, interim frame representation, residence/lifetime obligations
and native sequence execution remain separate work. Alex is unchanged.

Companion revisions: clef `934c36365`, lattice-analyzers `3814567`,
lattice-vscode `b4cc636`, CAC `a4da7f23`.

## C-02/C-06 computation admission — 2026-09-20

The checker previously erased unsupported computation syntax into ordinary
applications, sequences, matches, loops or payloads. All 28 new negative cases
were accepted before this correction. The public compiler also compiled
`builder { return 7 }` with a plain identity function into MLIR and a native
executable; `/tmp/composer-source-admission-c8d40278c30a44f39a48e9a4ea4b8c36/`
retains that failing rejection gate.

Unsupported builder bodies, bind/bang/return forms, unowned yields and resource
use now produce located CCS8401 diagnostics and Error/TError graph nodes before
their meaning can be erased. Ordinary function, lambda and lazy bodies clear
inherited sequence context; a nested sequence establishes its own context.
Lexically bound `seq` values resolve normally. The obsolete `match!` erasure
helper was removed. This is an admission correction; general native builder
dispatch and resource lifecycle elaboration remain implementation work.

| Gate | Result |
|---|---|
| CCS | **622/622**: 28 exact negative cases and eight preserved native-seq/ordinary controls; `/tmp/clef-computation-admission-full.log` |
| Public Composer | **8/8 SourceAdmission cases**: seven exact file/line/code/message rejections, each with no MLIR or native executable; ordinary FP control verifies with stock MLIR and executes with exact output. `/tmp/composer-source-admission-5be5c33c867b4e34adcbef88a548ae99/` |
| FidelityHello | **11b_LoopCaptures passes** compilation, exact output and native exit; `/tmp/composer-computation-admission-fidelityhello.log` |
| Analyzer projection | **29 accepted / 37 exact rejections**; `/tmp/lattice-ccs-surface-1d2c1bece70b432fac24e684bc8c34de/evidence.json` |
| LSP | **41 diagnostic edits and repairs**, including four exact CE errors and repaired `seq<int<m>>`/unit views; `/tmp/lattice-surface-waypoint-zWfGY8/result.json` |

CLI output exposes the diagnostic start line; CCS and LSP additionally check
the full source span. Both final tooling gates loaded CCS SHA-256
`57442252f6c09df482ae88ecc3341cd1832af2645e148229cc67da6820950953`.
Alex, solver transfer and target representation are unchanged. Existing native
sequence source admission does not establish complete element/delegation typing,
sequence owner/frame/formal settlement or native sequence conformance. Those
remain distinct C-06 work; no new frame or lifetime proof is claimed here.

Companion revisions are clef `558f2a2ab`, lattice-analyzers `fdc7b3d`,
lattice-vscode `75288af` and CAC `958c5176`, paired with this Composer checkpoint.

## F-09 Result defaults and iteration — 2026-09-20

Specification `82cd330` adds `Result.defaultValue`, `Result.defaultWith` and
`Result.iter`. Defaults select the Ok payload or the supplied fallback; deferred
recovery invokes its handler only on Error, passing that actual payload. Iteration
invokes its action only on Ok and returns unit. Independent success/error types,
dimensions and payload identities remain in the graph.

The shared Result recipe composes typed case elimination and ordinary application.
Its direct prefix evaluates all supplied operands before selection; its residual
prefix places local operand values before either case arm. Partials snapshot the
supplied value while retaining shared captured storage. Defaults consume exactly
two operands even when their result is a function: later arguments evaluate
before selection and then apply the selected function. The 15-group native
fixture checks this timing alongside factories/pipes, independent aliases,
explicit type arguments, callable/record/unit payloads and failure propagation.
The existing 12-group Result callback fixture passes after the shared recipe
adjustment. Alex, target layout and solver-transfer implementation are unchanged.

| Gate | Result |
|---|---|
| CCS | **586/586**, including 42 elimination cases with 17 exact negatives; both Result suites pass **88/88**; `/tmp/clef-result-elimination-full.log` |
| Native / MLIR | **2/2** ResultElimination and ResultCallbacks; stock verification and zero exit; `/tmp/composer-callbacks-fsharp-97ac1b5c8396498b98e255d9de209abf/` |
| FidelityHello | **09b_ResultElimination passes**, exact six-line output and zero exit; `/tmp/composer-result-elimination-fidelityhello.log` |
| Analyzer projection | **28 accepted / 33 exact rejections**; `/tmp/lattice-ccs-surface-ea25ff3133854d0fb69813a1678c8f33/evidence.json` |
| LSP | **37 diagnostic edits and repairs**, eight Result hovers including Error-handler partial and unit action; `/tmp/lattice-surface-waypoint-Ar2IZF/result.json` |

Both final tooling gates loaded CCS SHA-256
`0f2dda52a3dcd86dea0744ddb4a734ebfa0a1f08ce1e9146afe2fb44da5743f1`.
CAC records the same two payload parameters and callable result boundary.
Existing closure/DU placement and proof obligations remain applicable; no new
allocation rule or general computation-expression support is implied.

Use clef `a03222c24`, lattice-analyzers `dd7a4f1`, lattice-vscode `842bd35`
and CAC `554e0c8f` with this Composer checkpoint.

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
