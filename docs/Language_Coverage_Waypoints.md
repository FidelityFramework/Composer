# Language coverage waypoints

This record connects the language/compiler and tooling revisions for bounded
implementation steps. A passing editor projection, solver query, MLIR verifier
or executable establishes its own boundary; none substitutes for the others.
The [review](Clef_Language_Completion_Review_2026-09-19.md) and
[incremental contract direction](Nanopass_Incremental_Contract_Direction.md)
retain the wider roadmap and unresolved contracts.

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
- Fidelity.WebView's untracked `.serena/` and FsNativeAutoComplete's modified
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
