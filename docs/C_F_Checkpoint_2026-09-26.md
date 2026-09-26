# Clef / Composer checkpoint: C-series coverage and affected F-series gates

**Scoped integration checkpoint, 2026-09-26.** This
records delivered changes and their actual source, tooling and native cohorts.
**It does not mark any C PRD Complete.**
The subsequent full-manifest audit below found 27 compilation failures. This
checkpoint is not evidence of a clean F/C regression handoff.
C and F have equal delivery standing; the order below expresses dependencies.
The current implementation is substantially beyond the initial PRD review, but
specification agreement, source proof, physical composition and native behavior
remain separate acceptance observations.

Documentation correction: assistant-created dependency numbering has been
removed. Work is owned by the existing C/F PRDs and their specification clauses,
linked from the [master PRD index](PRDs/README.md) and the
[shared evaluation traceability](PRDs/C-Series-Acceptance.md#12-shared-evaluation-requirements-within-existing-prds).
The ledger is an implementation inventory, not an additional source of language
requirements or a separately authorized prerequisite project. The correction
changes no compiler behavior, acceptance result or feature status.

## Subsequent dimensional, trace and bounded-storage checkpoint

September 26, source commit Clef `d5ae0d9` on `main`; corresponding Composer
source is committed with this entry. This cohort extends the earlier evidence
below. It advances no whole-feature completion status and is not a rerun of the
complete native manifest.

Delivered source changes preserve quantified dimensional schemes through aliases,
specialization and recursive instances; infer actual tuple formals and contextual
record owners; retain immutable specialization history through successive
fold-ins; and settle finite recurrence and actual Lazy-formation effects without
invented widths. Historical specialization parents remain inspectable without
owning current lexical children. Full/pruned serialization retains the derivation.
Program storage is tied to declared authority, exact backing and actual native
placement. Alex consumes those facts through its existing Patterns/Witnesses;
LLVM checks the actual ABI extents and writable, non-executable ELF coverage.

| Observed gate | Result and evidence |
|---|---|
| Full CCS test assembly, including registered inference, specialization tape and finite-effect cases | **1,499/1,499**, zero skips; `/tmp/clef-full-lazy-effect-v13.log` |
| Full Alex tests | **274/274**, zero skips; `/tmp/composer-lazy-effect-alex-v13.log` |
| Source/native SMT transfer parity | **128 cases**, including signed all-prefix contributions, refusals and arbitrary-precision finite-effect counts; `/tmp/composer-smt-effect-parity-v13.log` |
| Unchanged native14,14a,14b | **3/3**, exact stdout and empty stderr; `/tmp/composer-lazy-effect-native-v13.log`; source/dependency hash comparison empty |
| Prior v11 native cohort | 04,04b,13,14,15,16b passed exact output; `/tmp/composer-writable-native-v11.log`. These retain their v11 identity. |

The new native cohort uses `/tmp/composer-lazy-effect-v13/compiler`, with inventory
`/tmp/composer-lazy-effect-v13/compiler.sha256`. SHA-256 identities are:

- Composer.dll: `0F2FA9EC9463C8D3CD88A203721E91DE5815F14933031AADFCCB7FD18B3C3687`
- CCS: `2FE2AE680266F5754784C7D2340571FCAF22776955FC5BF14D174479777CB571`
- BAREWire: `FCC595BC8B38127E0930209CCF6B91CB73E1EDD81F55FD5696CC991D26E3FBB1`

Per-job streams, expected results, `inputs.json`, `results.json` and the empty
`changed-inputs.json` are in
`/tmp/composer-lazy-effect-v13/native/20260926T193409-eff2a75484744c918bafd40bfa50d5ec`.
Observed compilation times were 47.16s, 48.69s and 33.60s respectively; execution
was 31ms, 31ms and 30ms. These are end-to-end sample compilation observations,
not Composer's own build time or a proven latency budget.

Reproduction uses the ordinary projects and .NET harness:

```sh
dotnet test tests/Clef.Compiler.Service.Tests/Clef.Compiler.Service.Tests.fsproj --no-build --no-restore
dotnet build src/Composer.fsproj -p:BuildProjectReferences=false --no-restore
dotnet test tests/Alex.Tests/Alex.Tests.fsproj -p:BuildProjectReferences=false --no-restore
dotnet fsi /tmp/SMTTransferRegression-v13.fsx
dotnet fsi /tmp/run-lazy-effect-native-v13.fsx 14_Lazy 14a_LazyScalarResults 14b_LazyStringViews
```

The first command runs in Clef against the freshly built test assembly; the next
two run in Composer. The private FSI drivers pin the recorded snapshot; the
source parity cases live in `tests/SMTTransferRegression.fsx`. Before independent
jobs start, build the coordinated outputs once and freeze their dependency set.

Explicit next defects retain their owning implementation tasks: native04a still
evaluates an unused ordinary argument, and native04c exposes unused-binding
demand. The newly isolated unresolved-record-member scheme case must retain its
constraint rather than generalizing an unconstrained function. These are
C-01/C-02/F-04 and source-inference obligations, respectively. Passing explicit
Lazy memoization does not establish ordinary call-by-need. The accepted F-11/C-08
numeric expansion has its own registered validation inventory.

## Earlier checkpoint and audit evidence

Clef: `14fb7c7`; Composer: `f6d391d` on `main`.
Companion revisions: specification `586010e`, site `8b6bd57`, BAREWire `f7d4693`,
analyzer projection `b987570`; unchanged Platform baseline `d42c9988f7fd`.
Both compiler integrations are in their main worktrees.

The final v20 private compiler is `/tmp/composer-checkpoint-v20/compiler`.
Its source compiler SHA-256 is
`b95b41bf5f23e7ac7ed31309276b94811f252579f51452136e5b8975df637874`;
Composer SHA-256 is
`1390e2dfb2c0483de6c80bcbf5c02e677407679235f0878c6da262bd01eb6c84`.
The full snapshot manifest is `/tmp/composer-checkpoint-v20/compiler.sha256`.
These assemblies were built before committing the verified source, so their
embedded version labels retain the earlier Git base.

The broader native v19 cohort predates only the final Lazy source-provenance
projection correction. Its separate snapshot is
`/tmp/composer-checkpoint-v19/compiler`, source SHA-256
`fd0d1f18c43a54a83eab92daa171d645d9ecddf604bb4892e5271cc1aadbeac0`.
The final v20 native confirmation repeats Lazy memoization and 16h transport;
earlier executions below retain their actual cohort labels.

## Subsequent full-manifest audit — 2026-09-26

An independent agent ran the complete 48-entry native manifest after the
checkpoint. Inspection of its saved `results.json` confirms **21 compiled and
ran successfully, 27 failed compilation, zero skipped**. The 27 failures did
not proceed to native execution. The runner checked normalized stdout and zero
exit status for the 21 successes; this is not a byte-exact stderr assertion.
The [repository outcome summary](evidence/2026-09-26-manifest-audit.json) retains
all 48 sample names/results, compiler hashes and a hash of the original result
file. It is an extraction of that run, not a second execution.

The evidence root is
`/tmp/claude-1000/-home-hhh-repos-clef/402241eb-a4c4-43b3-9a24-eeb5e3905ddb/scratchpad/baseline/20260926T170046-6106d4f275d44281b1e7ce131b4a130a`.
It retains the manifest, selected expectations, run settings, compiler hashes,
all job logs and results. It used six independent jobs, full intermediates and
a 300-second per-job timeout. Its compiler snapshot hashes are:

- CCS: `24d09cca20f19dd9caddae088fb7359e32d68f65fdf4cca05c2a276979ce126e`.
- Composer: `6dce64dc8019a4d5929d27611fb9b41f140d25af9ac1f87c313b80f15af43da4`.

These are a separate build cohort from the v19/v20 binaries below. The saved
compiler banner identifies Composer `6d54764`; the aggregate failure count
does not by itself identify which earlier change introduced each defect.

| Failed samples | Existing acceptance owners |
|---|---|
| 04 | F-04, C-01/C-02 callable application and results |
| 06 | F-06 interactive parsing; the legacy platform conversion rejection was already recorded |
| 08a–e | F-08 and C-04 Option operations, with C-01/C-02 callback and environment dependencies |
| 09a–c | F-09 Result operations, with C-01/C-02 callback and selected-branch dependencies |
| 11, 11a, 11b | C-01 closures, direct captures and loop captures |
| 13 | C-03 recursion |
| 15 | C-06 sequence recurrence and frame settlement |
| 16b, 16c, 16e, 16g, original16 | C-06/C-07 sequence composition, callbacks, demand and startup; C-01/C-02 callable dependencies |
| 18 Generalization, 19 ModuleValues | F-04/C-02 generalization and F-01/C-01 program initialization and captured storage |
| 20 ArraySurface | C-04 Array and supporting operations |
| 21 RecordsAndTags, 22 UnionPayloads, 23 RecordSurface | F-05/F-08/F-09/F-10 aggregate and selected payload behavior, with C-04 supporting operations |
| 17 ExternCall | Foreign binding/ABI fixture reconciliation, with C-01 foreign-boundary and F-08 optional-result overlap; retired integer-as-pointer types must not be re-admitted to make this sample pass |

Thus **26 failures directly exercise C/F contracts**; 17 needs its distinct
foreign-boundary contract. The actual samples 18–23 are not the planned A-series
examples which reuse those numbers in the PRD index. Scope must follow the
source program and owning contract, not the numeric prefix alone.

The earlier native successes recorded below did not include any of these 27
failed samples. Several have older passing evidence, so their present failures
reopen the affected regression gates; others were already recorded acceptance
gaps. Neither category can be called a newly introduced defect solely from this
one run. F-04, F-08 and F-09 must not retain an unqualified current passing claim.

One concrete cause is confirmed by code inspection: direct-capture elaboration
prepends physical capture formals and preserves the source signature, while
`CallableCarriers.settle` excludes only environment/result formals when comparing
that signature. It omits proved direct-capture formals. The same omission exists
in Clef `14fb7c7` and its parent `f898c97`; attributing its introduction to the
last commit alone is unsupported. The correction belongs in CCS settlement,
with typed provenance and invalidation checks, followed by unchanged native
direct-capture and neighboring controls. No Alex semantic workaround is licensed.

The handoff wording overstated the evidence. A successful compiler build, clean
worktree and selected passing tests establish those observations only. Shared
callable/closure changes require the affected F/C native baseline to be run and
reported before describing the handoff as regression-clean.

### Direct-capture correction and neighboring checks

Clef `f65907e` corrects the direct-capture public/physical signature distinction.
The reader validates the actual capture origin, immutable source, typed formal,
parameter incidence and leading position. The callable retains every physical
parameter. Removed, duplicated or altered premises retract admission; Alex does
not infer the missing source contract.

The new source regression failed before the repair, including both declared
32-bit and 64-bit platform cases. After the repair, **85 focused tests and all
1,371 CCS tests pass**, with zero skips. The rebuilt Alex suite passes **223/223**.
Logs: `/tmp/clef-direct-capture-carrier-red.log`,
`/tmp/clef-direct-capture-platform-red.log`,
`/tmp/clef-direct-capture-carrier-green.log`,
`/tmp/clef-direct-capture-full.log`, `/tmp/composer-direct-capture-alex.log`.

The fresh [seven-sample native result](evidence/2026-09-26-direct-capture-repair.json)
is **3 compiled and ran successfully, 4 failed compilation, zero skipped**.
Original 11a now passes unchanged, alongside 12 and 16h. Their stdout additionally
matches the manifest byte-for-byte, with empty stderr. 08e, 11b, 13 and 16b still
fail; this run is not reported as a clean cohort. It used three jobs, pruned
intermediates and a 180-second per-job limit. The JSON records exact compiler
hashes and every selected outcome; full logs remain at its evidence root.

The remaining 13/16b signature failures concern generic and measure instantiation,
not missing direct-capture formals. Correct their owning source contracts rather
than widening the hidden-formal exception. 11b retains a storage residence failure;
08e retains returned-environment and callable-instantiation failures. These remain
within their existing C/F acceptance obligations.

## Planning estimates and their limits

The owner has requested completion and architectural verification of all F/C
acceptance areas, rather than revised percentages. The conversational estimates
below are withdrawn as a planning basis: they did not systematically account
for the failed regressions now observed. They are retained solely as the record
of what was previously claimed, not replaced with new guesses.

The following numbers preserve the assistant's conversational estimates of
remaining effort, made without a new code review. They were not calculated from
a weighted acceptance inventory, and they are not measured coverage, PRD status,
token forecasts or spending commitments. Individual uncertainty was estimated at
±10 percentage points, ±15 for C-03/C-04.

| PRD | Previously stated effort estimate remaining |
|---|---:|
| C-01 | 45% |
| C-02 | 40% |
| C-03 | 55% |
| C-04 | 65% |
| C-05 | 35% |
| C-06 | 35% |
| C-07 | 45% |
| F-01 | 10% |
| F-02 | 30% |
| F-03 | 15% |
| F-04 | 30% |
| F-05 | 40% |
| F-06 | 20% |
| F-07 | 20% |
| F-08 | 25% |
| F-09 | 25% |
| F-10 | 30% |

The earlier wording overstated what these numbers establish. In particular,
the F estimates mixed anticipated C-driven extensions and regression work with
the historical F scope. They do **not** reopen the completed F baselines in the
master index or transfer A-04's arena extension into F-02. The rough group figures
of 50% C and 25% F were not weighted calculations and must not be treated as firm
remaining-budget percentages. Shared work must be budgeted once under its named
C/F owner, with dependent regressions identified; there is no extra numbered
workstream to add to the bill. The PRD and evidence tables below remain the
auditable scope record.

## Recorded evidence and current integration

| Observation | Established result | Boundary of that evidence |
|---|---|---|
| Final integrated source v20 | **1,355/1,355 CCS tests passed**, zero failures/skips. `/tmp/clef-checkpoint-full-v20.log`. | Includes eager factory/staged-call corrections and exact closure/Lazy source-provenance retraction. Source tests do not substitute for runtime demand traces. |
| Final physical composition v20 | **223/223 Alex tests passed**, zero failures/skips, including raw constant CPU/FPGA and backend requirement controls. `/tmp/composer-checkpoint-alex-v20.log`. | Component/stock-MLIR evidence is not a native FPGA device result or full C-series acceptance. |
| Final editor/analyzer/live LSP v20 | Default editor **21 groups** and analyzer integration **141 reported checks** pass. Live LSP capture hover, exact definition, located unsaved error, obsolete-version rejection, repair and clean shutdown pass. `/tmp/composer-checkpoint-editor-v20.log`, `/tmp/composer-checkpoint-analyzer-v20.log`, `/tmp/composer-checkpoint-live-v20/evidence.json`. | Public source types/identity remain distinct from internal environment/cache reads. This proves the listed batch/editing boundaries, not general incremental recompilation. |
| Final native confirmation v20 | **14_Lazy and 16h_SequenceApplications: 2/2 compilation and 2/2 execution**, unchanged exact oracles, zero failures/skips. `/tmp/composer-checkpoint-native-v20.log`. | Confirms memoization and callable/sequence transport after the last source-provenance change. |
| Native v19 selected-match suite | **5/5 passed**: `GuardedMatch`, `LiteralMatch`, `TerminalMatchSuccess`, `TerminalMatchPatternFailure`, `TerminalMatchGuardFailure`. Three success exits were 0 with exact empty streams; the two intended failures exited 1 with their exact source-located diagnostics. `/tmp/composer-checkpoint-matches-v19.log`. | Includes stock MLIR verification, lowering, linking and execution. It proves the listed match cases, not the whole source pattern chapter or ordinary call-by-need runtime. |
| C-05 native v19 cohort | **14/14a/14b all compiled and ran: 3/3**. Sample 14 proves successful memoization shared through aliases, independent returned instances and original mutable cells; 14a adds demanded unit/bool/integer/real/measured results; 14b proves retained descriptors backed by the actual declared immutable string pool. `/tmp/composer-lazy-main-final.log`. | It does not establish arbitrary retained views, all aggregate/callable caches or the ordinary shared-demand runtime. |
| Native/artifact differential v19 | **01/16a/16h all compiled and ran in both full and pruned modes: 6/6 compilation and 6/6 execution**, byte-identical MLIR, LLVM IR and runtime streams. All six outputs match byte-exact manifest oracles; all nine PSG view pairs preserve exact included nodes and complete joint evidence. `/tmp/composer-pruned-differential-main-v19b.log`, `/tmp/composer-pruned-differential-main-v19b-validate.log`. | This gate succeeds after the earlier frame-provenance regression and source access/all-writer correction. It does not establish full C-01/C-06/C-07 acceptance or ordinary call-by-need semantics. |
| Proof and runner controls v16 | All eight runner gate groups plus parallel-runner checks, 10 StaticStorage checks and 85 SMT checks passed (11 native, 6 real, 11 integer, 20 loop, 10 dimensional, 12 layout, 15 continuation). | These retain their recorded v16 cohort; they are not relabeled as v19 or source20 executions. SMT/component outcomes do not substitute for native observations. |
| BAREWire .NET gate driver | **12/12 driver checks pass**, including byte-exact output, missing/stale artifacts, error exits, timeouts and orphan cleanup. `/tmp/barewire-native-gate-checkpoint.log`. | This validates the replacement of Python gate drivers; it does not claim a fresh native RoundTrip run. |
| Explicit-demand foundation and main integration | The foundation from `d8effc6e6e10` is integrated: eager syntax/frontiers, real callable arity, staged actual ownership, remapping and retraction coexist with current Lazy/callable/sequence protocols. The unnecessary eager-callee eta wrapper is corrected in the passing source v19 cohort. | These results do not establish ordinary shared-demand storage or native acceptance of ordinary call-by-need semantics. |

The new native literal control distinguishes Unicode characters, fractional
floating values, separately formed equal strings, same-length unequal strings,
empty strings, signed zero, unit guard order and runtime NaN. GuardedMatch checks
wrong-tag suppression, selected payload scope, same-tag false-guard fallthrough,
stopping after success and tuple/constant guard ordering. Failure controls reject
timeouts and missing/mismatched diagnostics; a nonzero exit alone is insufficient.

Reserving `eager` required renaming three sample-local identifiers in 09b, 16a
and 16e. All three parse with the new grammar; the 16a differential preserves its
unchanged oracle. Existing eager-oriented trace fixtures still require the owning
PRDs' reconciliation with specified call-by-need semantics below; their current
passes do not establish lazy-default behavior. No full sample-manifest or whole
C-feature completion is claimed here.

## C-01 through C-07: acceptance and next delivery work

The authoritative criteria are the individual PRDs and
[C-Series-Acceptance.md](PRDs/C-Series-Acceptance.md).
Each row below retains positive source/native behavior, exact negative admission,
proof retraction, physical artifact correspondence and actual tooling projection.
A responsible refusal protects the compiler; it does not complete a promised
positive language form.

| PRD | Required surface and contracts | Concrete delivery/evidence now | Next acceptance work |
|---|---|---|---|
| [C-01 Closures](PRDs/C-01-Closures.md) | No/one/multiple captures; deferred immutable binding identity and shared mutable cells; direct, materialized, nested, returned, stored and recursive callables; correct flat environment layout/residence; public source identity; distinct foreign entry/registration/descriptor boundaries. | Actual code/environment operands replace packed or invented callable representations. Capture-free code keeps no dummy environment. Actual environment formals and caller-owned destinations preserve independent formations. Complete ingress/consumption and exact continuation-slot access include source identity, all writers and edit retraction. Recorded native16h and existing direct/loop capture gates exercise bounded slices. | Run original11, 11a/b, 12 and callback corpus on the integrated cohort. Complete direct mutable-cell signature authority and recursive forwarding; general record/tuple/DU/collection callable storage, mixed callable alternatives and retained/nested recapture. Add independent factory/cell overwrite/return-lifetime native traces and navigation repair. Keep C-01 §6.7 registration invocation/release and §14 descriptor relocation/code-identity obligations explicit with their FFI owners. |
| [C-02 Higher-order functions](PRDs/C-02-HigherOrderFunctions.md) | Callback arguments/results with independent dimensions; direct/pipe/bare/stored/partial forms; actual declared application boundaries and function-valued results; aliases, joins and aggregate storage. | Eleven Seq operation values now have staged/bare reification; generic immutable operation aliases retain source provenance. Carrier projection and passive transport preserve the real function/environment pair, not one convenient known origin. Factory preparation inserts destinations while retaining original ordinary actuals. The source foundation corrects all-arrow arity and all-actuals staging. | Reconcile all Option/Result/Seq callback frontiers with the specified ordinary call-by-need rules, then execute unused/eager actual and returned-callable traces. Restore all original12/16/16h controls together; cover mixed captured/plain callbacks and general stored results. Run source, component, native and editor/analyzer/LSP gates for each newly admitted form. |
| [C-03 Recursion](PRDs/C-03-Recursion.md) | Self/nested/mutual groups, captures, type generalization, effects and mutable/partial/returned uses; numeric recurrence; specified tail-call/stack behavior and recursive initialization. | Existing recursive/direct capture and range/effect mechanisms are retained; newer callable/formal/cell contracts strengthen their premises. This campaign's match/callable passes do not constitute a new original13 or deep-tail result. | Resolve original13's integer-width boundary from real arithmetic/guards; run factorial120 and sum55 unchanged. Implement the complete recursive-group matrix and actual recurrence/effect convergence. Prove each claimed tail form structurally and with deep bounded-stack execution. Reconcile recursive value initialization with native failure semantics and add its positive/negative cases. |
| [C-04 Core collections](PRDs/C-04-CoreCollections.md) | Full List/Map/Set/Option inventory, Array/support/range/tuple promises, listed normative extensions; sentinel/arena-floor authority, relative links, guarded access, persistence, exact bounds/alignment/capacity/lifetime. | Existing recipes and operation/source tests are implementation, not absent work. Generic DU payload alignment now uses real maximum alignment/extent rather than offset1 assumptions. Selected-match payload/type identity and ordinary typed equality provide stronger shared prerequisites. BAREWire has a .NET exact-byte gate driver and its harness tests. | Establish shared collection representation and negative storage gates; then verify every operation below. Replace the simplified Set two-child removal with persistent rebalance and gate all AVL rotations/deletions/old-root preservation. Audit/register both13a fixtures with independent exact oracles. Run Array bounds/overlap/HOF and all stepped range forms. Execute fresh BAREWire RoundTrip and admitted Platform consumers; build success alone does not discharge them. |
| [C-05 Lazy](PRDs/C-05-Lazy.md) | Explicit Lazy first-force memoization, aliases and independent instances, canonical thunk/environment, computed/cache publication, exact typed results/captures/residence; owned single-forcer publication and specified failure/reentry behavior. | Canonical Baker memoization and joint all-access proofs replace placeholder result carriers and Alex-owned force logic. Exact ranges come from validated thunk/result/effect premises. Source residence and passive pair transport preserve actual instances;14/14a/14b have the bounded native results above. | Retain the passing v19 14/14a/14b controls while extending the capture/result families, and keep their closure/sequence controls green. Deliver shared deferred immutable capture behavior, nested/forwarded retained views and aggregate/callable results with actual backing authority. Settle and test reentry/nonreturning force behavior. Implement concurrent admission only with its single-forcer ownership and target publication evidence; it is not implied by a CAS or single-thread result. |
| [C-06 SimpleSeq](PRDs/C-06-SimpleSeq.md) | Delimited owner/control, suspend/resume, definite initialization and successful Current; real generator/frame/slot families; independent enumerators and shared external cells; returned/parent/program residence; arbitrary admitted element representations. | Multiple-origin sequence families retain complete member slots and representation facts. Separate generator/environment operands preserve the actual pair. Fresh enumeration has explicit source-owned copy/reset authority. Continuation callable accesses now prove source slot, real storage/formal/generator, initializer and all writer premises; retired logical participants are allowed only through this exact relation. Recorded15a–d/16h gates retain their bounded evidence. | Run original15 plus15a–d and exact NativeSequences gates. Resolve coupled/multiplicative updates from intermediate/store bounds. Gate interleaved enumerators, parent/child return residence, capacity and retained aggregate/callable Current across later pulls/exhaustion. Add retraction after changed guards, storage, writers or origins and actual client repair. |
| [C-07 Seq operations](PRDs/C-07-SeqOperations.md) | All eleven core operations in every admitted direct/pipe/bare/stored/partial form, repeated enumeration, independent callback/state types, demand/short circuit and factory residence; every listed successor operation. | Core recipes,17 staged/bare source controls and canonical sequence/callable transport are implemented. Native16h passes its unchanged seven-group oracle again on main v19 in both artifact modes. This does not establish current ordinary lazy-default demand, original16's full factory/recurrence cases or successor support. | Run original16,16a–h and exact source/negative/tooling gates on one cohort after reconciliation with the specified ordinary call-by-need rules. Count actual pulls/callbacks, including nonpositive take, empty effects and no post-decision work. Implement and gate `empty`, `length`, `isEmpty`, `head`, `min`, `max`, `minBy`, `maxBy`, `toList`, `toArray`, including source registration, selection semantics and materialized output residence/capacity. |

### Preserve the entire collection/operation scope

This is a coverage inventory, not a claim each listed operation has passed.
For each operation record its admitted source scheme, owning Baker contract,
direct/pipe/bare/partial forms where meaningful, positive/negative source cases
and native values/effects/storage oracle. The full tables remain in C-04 §3 and
[C_F_Completion_Ledger.md §2](C_F_Completion_Ledger.md#2-operation-inventory-that-must-survive-the-campaign).

| Family | Operations retained in the acceptance campaign |
|---|---|
| List | `empty`, literals, `cons`/`::` and cons patterns, `isEmpty`, `head`, `tail`, `length`, `rev`, `append`/`@`, `map`, `filter`, `fold`, `foldBack`, `tryHead`, `tryFind`, `forall`, `exists`; then `collect`, `reduce`, `contains`, `tryPick`, `minBy`, `maxBy`, `min`, `max`, `last`, `forall2`, `sum`, `sumBy`, `average`, iteration, `toSeq`, `ofSeq`. |
| Map | `empty`, `isEmpty`, `add`, `remove`, `tryFind`, `find`, `containsKey`, `count`, `keys`, `values`, `toList`, `ofList`, `map`, `filter`, `fold`; then `toSeq`, `iter`, `forall`, `exists`, `ofSeq`, `ofArray`. |
| Set | `empty`, `isEmpty`, `add`, `remove`, `contains`, `count`, `union`, `intersect`, `difference`, `isSubset`, `toList`, `ofList`, `map`, `filter`, `fold`; then `isSuperset`, `forall`, `exists`, `iter`, `toSeq`, `toArray`, `ofSeq`, `ofArray`, `singleton`. |
| Option / F-08 | Constructors/matching; `map`, `bind`, `defaultValue`, `defaultWith`, `orElse`, `orElseWith`, `iter`, `fold`, `foldBack`, `filter`, `exists`, `forall`, `isSome`, `isNone`, `get`, `toList`; then `map2`, `map3`, `flatten`, `toArray`. Nullable conversions require an explicit boundary/profile and cannot introduce interior null. |
| Result / F-09 | `Ok`, `Error`, matching, `map`, `mapError`, `bind`, `defaultValue`, `defaultWith`, `iter`, `isOk`, `isError`; independent success/error types, inactive callbacks/payloads and actual function-valued results. |
| Array/support | `blit`, `map`, `fold`, `init`, `sum`, `sumBy`; preserve literals, `zeroCreate`, get/set/indexing/length and20_ArraySurface. Include overlapping blits, both bounds, empty/count/extent, accumulator dimensions and retained element storage. |
| Ranges/tuples/helpers | Inclusive List/Array/Seq ranges with implicit/explicit step; nested tuple binding, wildcards, `fst`, `snd`, `min`, `max`, list-based `String.concat`. Preserve one shared input, deferred unused payloads, endpoint/step contracts and exact capacity. |
| Seq core | `map`, `filter`, `collect`, `append`, `take`, `fold`, `iter`, `exists`, `forall`, `tryHead`, `tryPick`; successors as listed in C-07 above. |

## F-01 through F-10: reciprocal acceptance

The historical F statuses describe earlier delivery; they are not fresh
verification of shared owners changed by this campaign. Reengineering these
owners obliges the affected F gates as well as the C gates.

| PRD and actual owning contract | Impact and evidence now | Fresh gate to close the impact |
|---|---|---|
| [F-01](PRDs/F-01-HelloWorldDirect.md): entry/startup, static strings, console and platform calls | Full/pruned01 passes the fresh main-v19 differential. New failure realization preserves actual source diagnostic text through native lowering. | Retain01 in the final integrated cohort; run19_ModuleValues and full-profile startup16g, verifying exact streams, entry order and no accepted artifact after effective source errors. |
| [F-02](PRDs/F-02-ArenaAllocation.md): string/memref representation, concat/length, allocation and backing lifetime | Environment/sequence/Lazy placement and DU alignment change shared storage premises. Existing02 has earlier control evidence. F-02 itself distinguishes its historical heap bridge from A-04 true arena work. | Re-run02 and retained buffer/string/foreign-array cases; inspect actual writable authority, extent/alignment/capacity, covering lifetime and release. Do not report general arena conformance from02 output. |
| [F-03](PRDs/F-03-PipeOperators.md): both pipe rewrites and application boundaries | Real declared arity and staged Seq values replace assumptions that all type arrows or supplied arguments share one call frontier. Recorded16h includes pipelines. | Run03 plus direct/forward/back-pipe equivalents under the specified ordinary call-by-need rules; distinguish unused ordinary actuals, explicit eager actuals and later returned-callable invocation. |
| [F-04](PRDs/F-04-CurryingLambdas.md): lambda types, curry/partial formation, thunks | Canonical code/environment operands, typed residual signatures and explicit-demand source tests materially strengthen this owner. Old packed closure/SSA sketches are not normative. | Re-run04/12/18_Generalization and FunctionSnapshots/UnitExpressions/OptionPartials; add captured partials and distinct dynamic factories with exact demand/identity traces. |
| [F-05](PRDs/F-05-DiscriminatedUnions.md): typed tags/payloads, pattern selection and expression-valued results | New selected-scope Baker decisions preserve guard fallthrough and source payload bindings; Require preserves terminal failure. Generic DU alignment and v19 all-five native match controls pass their stated scope. Original05's PRD pass is dated2026-09-20. | Run original05/21_RecordsAndTags/22_UnionPayloads and heterogeneous/measured/nested-tag cases on this cohort. Implement the specified Or/As/And/list/array pattern owners below. Gate retained aggregate/callable/view payloads and raw-decision source/physical refusal, including actual FPGA discriminator values. |
| [F-06](PRDs/F-06-InteractiveParsing.md): parse, numeric promotion, ordered console input and tuple patterns | Selected tuple payload/guard scope is now tested; no fresh full interactive06 result follows from those controls. | Run06 with exact manifest stdin plus invalid-input/branch cases, preserving demanded reads and native Result behavior. Check nested tuple binding identity and inactive branch input suppression. |
| [F-07](PRDs/F-07-BitwiseOperators.md): integer bit contracts, shifts/casts and physical widths | Common range/layout and typed discriminator code affects the same carrier infrastructure. This campaign makes no new blanket bitwise support claim. | Run07 and bit/cast component/source gates at admitted widths, boundaries and signedness; retain operation-defined bit patterns instead of choosing a convenient callback carrier. |
| [F-08](PRDs/F-08-OptionType.md): Option cases, typed payload and callback recipes | v19 verifies actual Some/None selection, guarded payload access and fallthrough. Canonical callable transport and matched-case type identity strengthen Option paths. | Run08/08a–e and all Option native callback/elimination controls. Reconcile old eager default/fallback traces with the specified ordinary call-by-need rules; prove tag observation leaves unused payloads deferred, `Some None` stays nested and retained function payloads preserve actual environments. |
| [F-09](PRDs/F-09-ResultType.md): independent success/error types, heterogeneous storage and native recovery | Canonical source fixtures and aligned payload layouts are covered in recorded source/component cohorts. Require termination is explicitly separate from recoverable Result and no exception engine is manufactured. | Run09/09a–c, ResultCallbacks/ResultElimination/ResultCases under the specified ordinary call-by-need rules; preserve inactive callbacks, independent dimensions, nested tags, retained error/value payload lifetime and real function-valued results. |
| [F-10](PRDs/F-10-RecordTypes.md): construction/access/copy-update, nested records and record patterns | Source selected-match cases cover record binding/guard order; callable/Lazy/sequence storage changes affect record fields. No fresh original10 native result is claimed here. | Run10/20/21/23_RecordSurface and GenericRecords/CapturedRecords/FunctionFields. Verify old record versions, nested selected payloads, copies and retained callable/view fields across subsequent calls/allocations. |

### Pattern work belongs to the source owner

The following are concrete implementation tasks found by comparing
[specified pattern laws](../../clef-lang-spec/spec/patterns.md#as-patterns)
with [the current checker](../../clef/src/Compiler/NativeTypedTree/Expressions/Patterns.fs).
They belong to F-05's shared pattern foundation, C-04 list/array storage and F-10
record consumers; Alex must not acquire new pattern-decision algorithms.

| Specified form | Current source defect | Owning change and decisive oracle |
|---|---|---|
| Or | Checker examines both operands but retains only the left pattern/bindings. | Retain ordered alternatives and equal name/type binding obligations; Baker selects either branch and joins the same source variables. Native second-alternative-only success, both failures, incompatible binding diagnostics and guard-on-selected-result cases. |
| As | Checker appends alias bindings while discarding the alias pattern structure. | Retain exact whole-input alias alongside inner projections; preserve one shared input identity. Native whole/part simultaneous use and source navigation/retraction; no duplicated evaluation. |
| List/cons versus Array | List and array literals both become `Pattern.Array`; cons becomes a tuple despite requiring guarded list structure. | Distinguish typed list sentinel/cons links from array length/index storage; perform length/tag guards before payload reads. Native empty/exact/short/long inputs, nested/cons patterns and wrong-shape guard suppression under actual capacity/lifetime contracts. |
| And | Multiple same-input conjuncts are represented using a tuple of remaining patterns. | Retain a same-input conjunction with correct binding compatibility and source order; Baker composes selected tests/projections. Native first-only/second-only/both matches and dimensional/binding negatives. |

The new selected-scope recipe does not repair information already lost by these
inherited checker paths; Or in particular can still arrive as only its left
alternative. Fix the owning representation and conforming positive implementation
together with the gate; a refusal alone is not completion. Broader active/type-test
pattern promises need the same specification and owning-admission audit, without importing a managed exception/null model.

## Architecture and correctness changes

These are measurable changes to ownership and correctness, beyond additional
sample totals:

1. **Source algorithms moved to their proper owner.** Baker now settles pattern
   selection, selected payload/guard scope, memoization and staged application
   boundaries. Alex consumes admitted structure through Huet occurrences and
   Element/Pattern/Witness composition; it does not reconstruct the algorithm.
2. **Identity is carried through the full physical boundary.** Code and real
   environment operands, sequence family membership, actual factory destinations,
   original source aliases and dynamic storage remain distinct. Equal layouts,
   one observed caller or a known symbol cannot stand in for a complete proof.
3. **Proof premises now cover the work actually consumed.** Complete ingress,
   continuation accesses/all writers, Lazy all-access/cache ordering and terminal
   match requirements include their joint participants and retract after edits.
   Scoped fact reuse cannot retain authority after changed storage, type, path or
   initialization facts.
4. **Actual occurrence and emitted artifact defects were repaired.** Shared code
   declaration coverage avoids duplicate MLIR functions without global symbol
   deduplication of values. Selected match results use their actual body SSA.
   Backend failure realization now preserves the diagnostic that stock assertion
   lowering lost; native failure oracles caught this.
5. **Portable physical operations retain language distinctions.** DU payload
   alignment derives from actual members. Typed literal equality handles strings,
   floating values, characters and unit through their existing operation owners.
   FPGA discrimination must preserve actual constants, not positional arm numbers;
   target scheduling/encoding remains backend work.
6. **Performance evidence is separated from semantics.** Diagnostic formatting
   dominated the recorded full-dump slowdown; bounded .NET process jobs and
   artifact projection address measured work without inventing compiler thread
   parallelism. Pruned views preserve the original graph and complete joint
   evidence, verified against emitted/lowered artifacts and exact native outputs.

This supports continued scope-aware design-time incrementality: local ownership,
actual zipper occurrence and complete invalidation premises are useful progress.
It is not a claim that the entire compiler already performs incremental
recompilation, nor that runtime Incremental/Observable computation shares one
activation, scheduling or cache policy.

### Source storage and whole-image resource accounting

Baker's static-pool proof covers the source-declared pool, its literal backing,
authority, layout and capacity participants. It does not automatically account
for diagnostic globals or helper code introduced by backend realization. A
whole-image resource claim must include those target additions and their actual
placement, extent, lifetime and image/code cost in the Platform/backend accounting
contract. This is a cross-cutting resource and artifact obligation; F-07 owns
bitwise semantics, not this accounting. The passing static-storage and native
failure controls establish their respective boundaries without filling this gap
by inference.

## Next coordinated acceptance order

1. **C-01/C-02 and F-03/F-04 — shared binding and argument demand:** reuse proven memo-storage
   ingredients through a distinct ordinary demand contract; do not change source
   values into public `Lazy<'T>`. Native first slice: unused effectful actual gives
   zero traces; two demands of one actual give one; distinct dynamic actuals give
   distinct traces; direct `eager` runs at its reached actual boundary before the
   callee body. Then cover partial/result-call stages, inactive branches, selected
   payloads, shared captures and operation-specific demand.
2. **Complete general callable/storage and selected-pattern routes:** finish the
   C-01/C-02 direct-cell/retained-field gates and F-05 source forms above, with
   real layout, scope and failure authority. Extend Lazy/Seq payload families from
   their admitted scalar/static-string slices using the same exact backing rules.
3. **Close recurrence and full collections:** original13/15 bounds and tail
   gates; common sentinel/arena/persistence contract; all List/Array/Option/Result
   and Map/Set operations; all Seq successor and materializer gates. Register
   previously unregistered13a cases and keep an individual operation evidence row.
4. **Deliver the reciprocal F and Framework gates:** expanded manifest, exact
   sequence/callback/admission/format controls, actual BAREWire RoundTrip and
   Ariel release/mapped-carrier checks through .NET drivers, analyzer/live-LSP
   editing and source-to-transformed-artifact integrity. Use the WrenHello,
   HelloDISCO and Elmish/CE UI plans as concrete consumer slices with their own
   reactive/UI/async owners, not claims already established by these compiler tests.

Existing executable entry points are listed in the
[completion ledger §5](C_F_Completion_Ledger.md#5-executable-checks-and-evidence-record).
The new match suite is directly reproducible with an existing private snapshot:

```sh
dotnet fsi tests/NativeCallbacks/TerminalMatchChecks.fsx -- /tmp/composer-checkpoint-v19/compiler/Composer /tmp/composer-matches-fresh --all
```

Use a fresh output root, preserve each harness's stream/timeout contract and
retain full artifacts and snapshot hashes. Report a gate as passed only after
its actual command and independent oracle have run against the stated cohort.
