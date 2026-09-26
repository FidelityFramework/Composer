# C-series and affected F-series completion ledger

**Execution inventory, 2026-09-26. All gates below remain open unless their
required evidence is recorded in the coverage waypoints.** This is a document
inspection, not a new compiler run. It makes the existing
[C-series acceptance contract](PRDs/C-Series-Acceptance.md) actionable without
replacing any PRD criterion or advancing a status.

The earlier committed checkpoint, Composer `db4bc8b01bb6` and Clef
`f898c97e11c5`, passed 1,058 CCS tests, 124 Alex tests, the default editor suite
and four native controls; native16h had not yet passed. Subsequent main-v19
verification now includes native14/14a/14b and the full/pruned01/16a/16h
differential, plus selected-match success and exact diagnostic failures. The
[current checkpoint report](C_F_Checkpoint_2026-09-26.md) separates these
recorded cohorts: final source v20 passes 1,355/1,355, Alex v20 passes 223/223,
and default editor, analyzer and live LSP gates pass. The eager-callee eta-wrapper
defect and closure/Lazy source-projection defects are resolved. Final v20 native
14 and 16h both pass; the broader v19 native results retain their actual labels.
Clef `14fb7c7` and the Composer commit containing this report are the coordinated
main-branch checkpoint; exact companion revisions and hashes are in the report.

**Governing correction, 2026-09-26:** Clef is lazy by default with call-by-need
sharing. The owner clarified that an unused ordinary argument retains its
effects deferred; repeated demands share one computation. Earlier passes that
assert blanket eager argument/initializer evaluation remain historical evidence,
not authority for current semantics. The [evaluation strategy contract](Evaluation_Strategy_Contract.md)
and [core expression rules](../../clef-lang-spec/spec/expressions.md#default-demand-and-sharing)
govern reconciliation. Preserve source fixtures where possible; when an expected
trace encodes the superseded eager rule, record the old expectation, governing
clause and new independent oracle with the owning implementation change.

An authorized intermediate commit or push does not end this campaign. Continue
through every required C gate and affected F gate. A core-operation pass does not
erase successor operations, a source rejection does not implement a conforming
positive case, and a historical F status does not excuse a new regression.
Explicit foreign/target horizons retain their owning contracts; they are not
silently counted as implemented or used to defer required interior behavior.

## 1. Dependency order and closure conditions

The order expresses prerequisites, not relative importance. Independent source
fixtures, contract reconciliation and artifact checks can proceed alongside the
current implementation. Coordinate builds sharing CCS/Composer outputs.

| Gate | Required work and completion observation | Owners, existing checks and next decisive oracle |
|---|---|---|
| **G0 — default demand and shared computation** | Establish call-by-need for ordinary bindings, arguments, captures and payloads. Unused effectful operands remain deferred; repeated demands share; selected branches and operation-specific strictness determine what executes. Preserve explicit sequential, entry/startup and foreign/resource activation contracts. Prove any earlier evaluation or thunk elimination preserves effects, termination, identity and lifetime. | All C/F gates use these semantics. Source demand/effect owners, Baker nanopasses and admission settle the plan; Alex witnesses it without choosing eagerness. Add source, graph, native and editor cases for unused effectful argument/binding (zero traces), two uses of one binding (one trace), two separately created computations (two traces when demanded), branch/short circuit, unused Option/Result fallback, bare/stored partials, ignored fold state, zero-take input and retained deferred payloads. Audit existing `OptionEvaluation`, `Result*`, `SequenceApplicationCases`, `FunctionSnapshots` and native16 traces before treating them as current oracles. |
| **G1 — canonical callable transport** | Finish separate code/environment operands through parameters, results, aliases, annotations, sequential results, branches and stored values. Physical arity comes from actual formals, not all arrows in a source type. Preserve source signatures and actual environments; no packed function address, invented empty environment or witness inference. | [C-01 §8](PRDs/C-01-Closures.md#8-validation), [C-02 §6](PRDs/C-02-HigherOrderFunctions.md#6-validation); Clef `CallableCarriers`, `CallableOrigins`, `CallableApplications`, environment recipes/result destinations; Alex `CallableOperands`, function operations and owning Patterns/Witnesses. Existing `SequenceApplicationCases`, `EnvironmentFactoryResultsCases`, `CallableOperandTests`, `FunctionResultTests`, `CallableTransportTests`. First native gate: unchanged **16h**, with **16a**, **11**, **12** controls. |
| **G2 — mutable identity and general storage** | Complete mutable callable cells, selection snapshots, direct mutable capture signatures, joins and function payloads in records/tuples/DUs/collections. Preserve the actual activation's cell and both callable components across overwrite, recursive forwarding and return. Distinct formations sharing code remain distinct; equal layouts do not imply equal implementations. | G1; [direct-cell contract](Direct_Capture_Cell_Contract.md), [closure settlement](Closure_Settlement_Contract.md). Existing `ClosureEnvironmentCases`, `ClosureEnvironmentRangeCases`, `CallEffectRangeCases`, `DirectCaptureCases`, `MutableClosureTests`; native `FunctionSnapshots`, `FunctionFields`, `CapturedRecords`, `CapturedBuffers`, `OptionFunctionPayloads`, `DirectCaptures`, `CallEffects`. Add returned mutable-counter, conditional/DU join and independent-factory tests where existing cases lack actual retained-lifetime evidence. |
| **G3 — complete sequence identity and residence** | Transport multiple legitimate sequence origins without selecting one code/frame arbitrarily. Admit closed formal uses and returned templates with caller/parent/program storage, exact child destinations and retained backing allocations. Independent/interleaved enumerators share external cells but not progress. Aggregate/callable current values remain valid after later pulls, exhaustion and child return. | G1/G2 for callable values; owning `SequenceOrigins`, `SequenceResidence`, `SequenceFactoryResults`, `ProgramActivation`, evaluation/control and continuation settlement. Existing source `Sequence*Cases`, especially origins, residence, aggregate/current, region and factory cases. Native **15a–d**, **16a–h**, then unchanged **16_SeqOperations**; **16g** preserves full-profile startup. A scalar Option-current pass is not general aggregate transport. |
| **G4 — recursion and numeric recurrence** | Close self/nested/mutual function groups, captures, generalization, recursive effects and returned/partial uses. Settle original13's numeric boundary and original15's coupled/multiplicative recurrences from actual guards, intermediates and stores. Establish bounded stack behavior for every claimed tail form with structural lowering evidence and deep execution. | [C-03 §§5–7](PRDs/C-03-Recursion.md#5-recursive-groups-and-unsettled-source-contracts); binding/group checking, direct captures, `RangeAnalysis`, loop/recurrence and selected backend owners. Existing direct-capture, call-effect, counted/range-loop and sequence-range cases. Native **13_Recursion** (including factorial 120 and sum 55), **15_SimpleSeq**, `DirectCaptures`, `CallEffects`, `CountedLoops`, `RangeLoops`. Add explicit mutual-tail, cleanup/effect and recursive-cell oracles; existing small outputs do not prove bounded stack. |
| **G5 — common collection storage** | Settle immutable sentinel authority, hosting-arena zero slot/floor, arena-relative links, exact same-node payload guards, extent/alignment/capacity and covering lifetimes. Persistent versions share only admitted backing storage. Preserve/reset the floor and reject cross-arena links, sentinel writes, insufficient capacity and absent authority. | [C-04 §5](PRDs/C-04-CoreCollections.md#5-representation-and-storage-acceptance), List/Map/Set representation chapters, placement/obligations and declared BAREWire/Platform storage. Existing generic DU construction in `CollectionPatterns` is an implementation starting point, not sentinel conformance. Add source, graph, actual-artifact and native storage oracles before operation-family acceptance. G5 can develop independently of G4, then join it for traversals. |
| **G6 — Lists, ranges, Array and support operations** | Complete the exact inventories in §2, shared deferred operands, stable demanded traversal, guarded extraction, fold direction, short circuit and resource-safe conversion. Collection and sequence ranges include steps, endpoint demand/order, exact counts and final-step overflow. Tuple destructuring shares its RHS and leaves unused payloads deferred while preserving nested names/types. | G0 for demand; G1/G2 for callbacks and payloads, G4/G5 for recursion/storage, G3 for sequence ranges/conversions. Existing `ListRecipes`, `OptionRecipes`, loop recipes, Array/source primitives and `20_ArraySurface`; original **13a** fixtures need audit, exact oracles and manifest registration. Add stepped/materialized range, overlap/bounds, string capacity, persistence and deep traversal tests. |
| **G7 — persistent Map/Set** | Complete insertion/replacement, all AVL rotations, every deletion shape, ordering/height consistency, retained earlier roots, transforms, folds, set algebra and conversions in §2. Re-establish Set-map uniqueness after collisions. | G4–G6, `MapRecipes`, `SetRecipes`, reusable tree ingredients and admitted collection Patterns/Witnesses. `SetRecipes` still contains a simplified two-child removal merge: replace it with correct persistent deletion and rebalance. Existing `TreeSequenceRecipeCases` establishes recipe shape, not AVL behavior. Add independent reference-result and structural invariant oracles; retain old roots during native checks. |
| **G8 — canonical lazy values** | Complete `(thunk, env)` acceptance across computed/result/capture storage, once-only normal force and the remaining result/residence families. Aliases share a cache; separate factories do not. Preserve shared deferred immutable bindings, shared mutable cells and retained aggregate/callable values. | G1/G2 and applicable G5 storage; [C-05 §§8–9](PRDs/C-05-Lazy.md#8-validation), [lazy representation §§9/11](../../clef-lang-spec/spec/lazy-representation.md#9-memoization-strategy). Canonical typed storage and passive witnessing now replace inherited placeholder/code-field/force-SSA assumptions. Corrected **14**, scalar **14a** and static-string **14b** pass together on main v19. Preserve those oracles while delivering deferred capture, nested/forwarded view and aggregate/callable cache gates. |
| **G9 — full Seq operation surface** | After unchanged original16/16h and the eleven core operations pass, implement and gate every successor in §2. Materializers preserve output order, independent types, capacity and lifetime after input exhaustion. Consumers count actual demand and key/callback effects. | G3/G4, G5–G7 for materialization; `SeqRecipes` and owning iterator ingredients. Add native successor cases and source admission for dormant/unregistered names. A recipe branch alone is not public operation support. |
| **G10 — cross-family delivery** | Reconcile every row, affected F behavior, actual tool projections, graph-to-artifact preservation and reachable Framework consumers on one coordinated cohort. Run the expanded full manifest and specialized gates; report every failed/skipped/unregistered promised case. | All applicable gates above; §3–§5 below and the shared acceptance contract. Close a PRD only when its entire required inventory is accounted for. Required work discovered during these gates becomes an owning-stage implementation task and is rerun; it is not moved to a permanent bookmark. |

G0 is a semantic prerequisite, not a later optimization.
Its graph and native acceptance must distinguish construction from demand,
yield/union-tag observation from payload demand, a mutable cell from an immutable
binding that shares a read, and call-by-need from explicit Lazy/Incremental cache
policies. Existing scalar-only physical layouts do not prove strictness. Required
deferred representations and their lifetime evidence are implementation work.

The subsequently authorized [explicit `eager` expression](../../clef-lang-spec/spec/expressions.md#eager-expressions)
is an additional G0 surface gate: implement lexical/parser admission, source
demand facts, Baker elaboration and native behavior together. Test reached unused
eager bindings, eager actuals at activated complete/partial boundaries, source
order, later function-result application, transparent grouping/annotations,
direct eager aggregate components and sharing without replay. Negative-demand
oracles keep markers inside unused nested arguments, unselected branches and
uncalled/deferred bodies inactive. Outer-value demand must not silently deep-force
ordinary payloads, force an explicit Lazy or enumerate a sequence. Correct eager
idioms are not automatic warnings; optional cost advice needs established facts
or clearly identified target/profile assumptions.

### G0 foundation integration inventory — 2026-09-26

Clef foundation commit `d8effc6` added eager syntax, local demand relations,
real declared callable boundaries and source tests. These changes are now
integrated into main with the current Lazy, continuation, callable and provenance
protocols. Final source v20 passes 1,355/1,355 (`/tmp/clef-checkpoint-full-v20.log`)
and Alex v20 passes 223/223. The eager-callee factory regression's unnecessary
Baker eta wrapper is corrected. Default editor, analyzer and actual live LSP
gates pass, including closure/Lazy source identity and clean shutdown; live
evidence is `/tmp/composer-checkpoint-live-v20/evidence.json`.
Native G0 acceptance is the next required delivery gate, including ordinary
call-by-need storage and its executable traces. The earlier read-only inventory identified seven textual merge
sites, retained here as a verification checklist:
`Nanopass/FoldIn.fs`, `Nanopass/Monomorphization.fs`,
`NativeTypedTree/ClefExpr.fs`, `NativeTypedTree/NativeService.fs`,
`PSGSaturation/SemanticGraph/CallableOrigins.fs`,
`PSGSaturation/SemanticGraph/Types.fs`, and the CCS test project file.
The integrated switches must preserve current Lazy, continuation, choice and
provenance cases, and retain `ResolvedCall.Complete` alongside `FirstBoundaries`.

The semantic verification is larger than those textual conflicts. Current
`CallableIngress`, `CallableFlows`, `CallableCarriers`, Lazy origin/residence,
sequence residence and the environment/sequence/lazy result-destination readers
must recognize an eager expression's value identity while retaining its actual
activation frontier. Alias promotion must not erase a marker or move its work.
Refresh owned demand relations after recipes and final continuation realization;
do not reuse relations whose source participants have changed. Verify the
foundation's removal of the all-actuals prefix from `ApplicationRecipes.stage`
together with current destination-only factory preparation. Option/Result/Seq recipe
prefixes still require an operation-specific demand audit.

The first executable ordinary-demand slice needs a distinct source contract for
one shared computation per dynamic ordinary binding/actual, preserving its public
type and source identity. Reuse the explicit Lazy path's guarded cache read,
result-before-publication order, field tiling and covering storage proofs through
shared lower-level ingredients; do not relabel ordinary values as public
`Lazy<'T>` or inherit cold/reactive invalidation policies. Baker must elaborate
the memoization algorithm and settle all demand sites and complete uses. Alex
must receive admitted operations and separate callable/environment operands.
The first native scalar cases must distinguish an unused effectful actual (zero
traces), repeated demand of one actual (one trace), separate dynamic actuals
(distinct traces), and a direct explicit eager actual (trace before callee body).
Add branch/fallthrough and partial/returned-callable frontiers before broadening
the same contract to retained captures and structured cached values. Source
demand-edge tests alone do not discharge these gates.

### Selected match scope evidence — 2026-09-26

The selected-scope Baker recipe now owns constructor/constant selection, nested
tuple/record projections, original source bindings and ordered guard
fallthrough. Witness-facing arms have no binding or guard metadata; payload
reads and guards occur in the selected body. The v10 source gate passed all
nine initial `MatchDecisionCases`, and the seven Alex component cases passed.
The native [GuardedMatch](../tests/NativeCallbacks/GuardedMatch.clef) control
compiled and ran with exit 0 and exact empty output using the private v10
compiler (`/tmp/composer-guarded-match-v10`, compile 4.85 s, run 32 ms). It
distinguishes wrong-tag guard suppression, same-tag false-guard fallthrough,
stopping after the first successful guard, tuple payload scope and constant-arm
guard order. This evidence covers those cases, not all source pattern forms.

Terminal refutable patterns and guards now have an always-active source
`Require` contract: a false condition terminates with a source
diagnostic; successful continuation does not manufacture a result. The reader
checks the complete pattern-test, selected body and ordered-frontier
participants, and Alex additionally checks the actual Huet occurrence.
[TerminalMatchChecks.fsx](../tests/NativeCallbacks/TerminalMatchChecks.fsx)
compiles independent success/pattern-failure/guard-failure controls from an
existing private compiler snapshot; `--all` also runs GuardedMatch and
LiteralMatch. It requires exact diagnostics, rejects timeouts and suppresses
core files locally for intentional failure controls.
All 12 initial source requirement checks and 13 selected-match checks passed in
the v12 focused cohort. The first native terminal run
(`/tmp/composer-terminal-match-v12`) reached stock MLIR verification for all
three controls and found a real single-arm result correspondence defect: the
emitter returned an undefined case SSA value instead of the actual selected-body
value. The singleton pattern now returns its actual adapted body carrier; both
constant and union component controls pass stock MLIR verification. The v13
native `TerminalMatchSuccess` control passes with exit 0 and exact empty output
(compile 11.983 s, run 31 ms); unchanged `GuardedMatch` also passes (14.038 s,
33 ms). Artifacts: `/tmp/composer-terminal-match-v13-positive`.

The v13 requirement reader retains the checked input type in an ordinary
transparent annotation, including its dimensions. It validates the original
input, annotation, pattern category and exact instantiated union type together.
All 20 reader controls pass, including changed dimensional premises. The focused
source gate passes **164/164**, and the full coherent source gate passes
**1,263/1,263** (`/tmp/clef-requirements-focused-v13.log`,
`/tmp/clef-requirements-full-v13.log`). The private compiler and hashes are in
`/tmp/composer-requirements-v13`; source and Composer builds are recorded in its
`cohort.json`.

The later main-v19 native suite passes all five controls, including both
terminal failures with **exit 1 and their exact source-located diagnostics**.
The three successful controls exit 0 with exact empty stdout/stderr. The log is
`/tmp/composer-checkpoint-matches-v19.log`, using
`/tmp/composer-checkpoint-v19/compiler`. Backend realization now preserves
the diagnostic that stock assertion lowering previously dropped. LiteralMatch
checks Unicode characters, fractional values, independently formed equal strings,
same-length unequal/empty strings, signed zero, unit and runtime NaN. All 20
Requirement cases and 19 MatchDecision cases first passed isolated FSI validation
against the v14 DLL (`/tmp/match-literal-fixtures-v14.log`); the integrated v19
full-assembly gate now passes 1,354/1,354. The report retains the exact remaining
Or/As/And/list/array source-pattern work; these controls do not cover the full
pattern chapter.

Baker's static-pool capacity proof covers its declared source pool and exact
backing participants. Backend-generated diagnostic globals and helper code need
their own contribution to whole-image resource accounting through the owning
Platform/backend contract. Passing source storage proofs and native diagnostics
does not establish that accounting; it is not an F-07 bitwise responsibility.

G2 direct mutable-cell signatures and G8 reentrancy/failure semantics have explicit
contract decisions below. G1/G3 must continue with already settled semantics while
those decisions are resolved. G5 storage, G4 recurrence and missing oracle work are
parallel work packages once their edited-file and build ownership is assigned.

## 2. Operation inventory that must survive the campaign

Each operation needs a source scheme/admission row, an owning implementation,
discriminating positive/negative cases and fresh native evidence. Grouped rows
below are work packages, not permission to mark several operations passed from
one representative example. Expand a package into individual evidence rows when
implementing it. Check direct, pipe, bare alias and each partial frontier where
the operation's type permits them, with independent payload/state dimensions.

| Family | Required operations and forms | Distinguishing oracle / dependency |
|---|---|---|
| **List — C-04 §3.1** | `empty`, literals, `cons`/`::`, cons patterns, `isEmpty`, `head`, `tail`, `length`, `rev`, `append`/`@`, `map`, `filter`, `fold`, `foldBack`, `tryHead`, `tryFind`, `forall`, `exists` | Empty/singleton/many; repeated empty tail; same-node guarded head; both fold orders; stable transforms; short circuit; sharing and stack/resource behavior. G5/G6. |
| **List — C-04 §3.8** | `collect`, `reduce`, `contains`, `tryPick`, `minBy`, `maxBy`, `min`, `max`, `last`, `forall2`, `sum`, `sumBy`, `average`, iteration, `toSeq`, `ofSeq` | Add explicit empty/nonempty, comparison/tie, mismatched-length and numeric contracts where needed; observe key/callback count, ordered results and conversion residence. These rows remain in this execution inventory, not silently outside full-family acceptance. |
| **Map — C-04 §3.2** | `empty`, `isEmpty`, `add`, `remove`, `tryFind`, `find`, `containsKey`, `count`, `keys`, `values`, `toList`, `ofList`, `map`, `filter`, `fold` | Duplicate-key replacement; sorted and reverse insertion; four rotations; absent/leaf/one-child/two-child/root/last deletion; compare contents/order/heights and preserved old roots. G7. |
| **Map — C-04 §3.8** | `toSeq`, `iter`, `forall`, `exists`, `ofSeq`, `ofArray` | Comparison-order traversal, decisive stopping and conversions retaining key/value types and storage. |
| **Set — C-04 §3.3** | `empty`, `isEmpty`, `add`, `remove`, `contains`, `count`, `union`, `intersect`, `difference`, `isSubset`, `toList`, `ofList`, `map`, `filter`, `fold` | All empty algebra combinations, duplicate insertion, mapping collisions, ordered traversal and persistent AVL deletion. G7. |
| **Set — C-04 §3.8** | `isSuperset`, `forall`, `exists`, `iter`, `toSeq`, `toArray`, `ofSeq`, `ofArray`, `singleton` | Empty truth laws, stopping and ordered/deduplicated conversions with exact capacity. |
| **Option — C-04 §3.4, F-08** | `None`, `Some`, matching; `map`, `bind`, `defaultValue`, `defaultWith`, `orElse`, `orElseWith`, `iter`, `fold`, `foldBack`, `filter`, `exists`, `forall`, `isSome`, `isNone`, `get`, `toList` | Reconcile existing `Option*Cases` and NativeCallbacks with G0. Unselected fallback/callback computations stay deferred; tag checks do not force payloads; function-valued results apply only after the operation boundary. Preserve valid identity/typing/storage oracles. `get` requires the admission reconciliation below; `toList` requires G5/G6. |
| **Option — C-04 §3.8** | `map2`, `map3`, `flatten`, `toArray`; account explicitly for `toNullable`/`ofNullable` at an admitted boundary/profile | Nested tags cannot erase `Some None`; independent payload dimensions and argument effects survive multiple-input selection. Nullable conversion cannot introduce interior null; settle and test its actual boundary before advertising it. |
| **Result — F-09 and C-02 transport** | `Ok`, `Error`, matching; `map`, `mapError`, `bind`, `defaultValue`, `defaultWith`, `iter`, `isOk`, `isError` | [Native Result operations](../../clef-lang-spec/spec/error-handling.md#native-result-operations) fix independent success/error types. Preserve `ResultOperationCases`, `ResultEliminationCases`, `ResultPredicateCases`, native `ResultCallbacks`, `ResultElimination`, `ResultCases` and **09/09a–c** under canonical transport. Tag tests must not extract/invoke payloads. |
| **Array — C-04 §3.5 and supporting consumers** | `blit`, `map`, `fold`, `init`, `sum`, `sumBy`; preserve literals, `zeroCreate`, indexing/get/set/length and existing **20_ArraySurface** behavior | No dedicated array-operation representation chapter is assumed: reconcile actual source schemes, selected NTU/storage rules and each promised operation. `blit` needs overlap and both bounds; callback/order, accumulator widths, empty input, count/extent and retained array-of-record/function behavior need explicit cases. **20** alone does not cover these HOFs or overlap. |
| **Ranges, tuples, helpers — C-04 §§3.5–3.7** | Inclusive List/Array/Seq ranges with implicit/explicit step; nested tuple lets, wildcards, `fst`, `snd`, `min`, `max`, list-based `String.concat` | First/step/last evaluation once; positive/negative/zero-trip/zero-step and final-step limits; independent component types; RHS once; string separator/order/empty/capacity. Existing counted/unstepped loop tests do not establish all six range intentions. |
| **Seq core — C-07 §1** | `map`, `filter`, `collect`, `append`, `take`, `fold`, `iter`, `exists`, `forall`, `tryHead`, `tryPick` | Both fold frontiers; callback factories/snapshots; repeated enumeration; shared cells; short inputs/nonpositive take; no post-decision pull/callback; empty effects; independent Option result types. Preserve **16a–h** and **original16**, not just totals. G1–G4. |
| **Seq successors — C-07 §8** | `empty`, `length`, `isEmpty`, `head`, `min`, `max`, `minBy`, `maxBy`, `toList`, `toArray` | `isEmpty` demands one pull including empty-body effects; `length` exhausts exactly; extrema require nonempty/comparison/tie/key-effect contracts; materializers require G5–G7. Establish source admission for `maxBy` before claiming the dormant recipe. |

The [C-04 inventory](PRDs/C-04-CoreCollections.md#3-ccs-intrinsics-and-promised-operation-inventory)
and [C-07 successor table](PRDs/C-07-SeqOperations.md#8-completion-record) remain
authoritative. Additional registered names discovered during the scheme/recipe
audit must have a stated contract and status; silently accepting an unimplemented
name is not an acceptable completion state.

## 3. Reciprocal F-series gates

The F-series has equal delivery standing. Its retrospective implementation
sketches do not override current specification or justify packed callable fields,
fixed widths or unproved allocation. Update affected descriptions when the owning
implementation changes; keep valid source/native expectations.

| Affected PRD | C change that can break it | Existing controls and required extension |
|---|---|---|
| **F-01**, **F-06** | Entry/startup, source-gate ordering, unit/native ABI, format/parse and effectful full-platform dependencies | **01**, **06** with manifest stdin; **19_ModuleValues**, **16g** and `IgnoreValues`. Retain exact startup order and prevent accepted artifacts after effective source errors. |
| **F-02** | Environment/collection/cache allocation, writable authority, bounds and lifetime | **02** plus program-lifetime, foreign-reference/array, capture-buffer and storage negatives. Its historical heap bridge is not evidence of current arena conformance. Actual declared authority/capacity and backing lifetime are needed in G2/G5/G8. |
| **F-03**, **F-04** | Both pipe directions, partial formation, actual versus curried arity, function results and generic aliases | **03**, **04**, **18_Generalization**, **12**, native `OptionEvaluation`, `OptionPartials`, `FunctionSnapshots`, `UnitExpressions`. Preserve order/once-only effects and subsequent application of callable results. Reconcile F-04's historical packed closure/thunk account with the delivered canonical form. |
| **F-05**, **F-08**, **F-09** | Case tags, typed payloads, selected-arm guards, callable/aggregate payload copying and branch results | **05**, **08/08a–e**, **09/09a–c**, **21_RecordsAndTags**, **22_UnionPayloads** and all Option/Result native callback cases. Preserve inactive cases and nested tags, heterogeneous/measured/unit payloads and actual retained function environments. F-08's old unchecked/undefined wording does not settle current absent-payload admission. |
| **F-07** | Range-selected widths, bit representation, shifts and byte-order behavior changed by common scalar/layout work | **07_BitsTest** and relevant bit/cast source and component tests; preserve the operation's declared bit contract rather than substituting a convenient callback carrier. |
| **F-10** | Record construction/copy/update, field layouts, nested patterns and fields containing callable/sequence/Option/Result values | **10**, **20**, **21**, **23_RecordSurface**; native `GenericRecords`, `CapturedRecords`, `FunctionFields`. Retained fields survive subsequent calls/allocations and preserve original record versions. Remove historical “code pointer as extra field” guidance when canonical function storage is delivered. |

## 4. Missing or weak gates to implement

These are implementation tasks in the owning tranche, not exemptions:

1. **Register collections.** The inspected [manifest](../tests/regression/Manifest.toml)
   has **46 entries** and no 13a entries. Both
   [SimpleCollections](../samples/console/FidelityHelloWorld/13a_SimpleCollections/SimpleCollections.fs)
   and [BAREWireCollections](../samples/console/FidelityHelloWorld/13a_BAREWireCollections/BAREWireCollections.fs)
   exist. Audit source against normative guards/operations, make their platform
   inputs reproducible, derive independent exact output and source assertions,
   and register both. Their comments saying “full coverage” do not supply it.
2. **Add resource and persistent-structure oracles.** Native totals alone miss
   dropped AVL subtrees, mutated old roots, a copied shared cell, expired views,
   excess allocation or unbounded stack. Add the G2/G4–G8 discriminating cases
   and inspect actual storage/guard/call artifacts against settled participants.
   Keep altered offset, signature, guard, capacity and origin controls red.
3. **Preserve the corrected lazy oracle.** Sample14 and its manifest now require
   one computation effect across repeated force, as specified by first-force
   memoization. Its native control passes; registered14a adds unit, boolean,
   signed, floating and measured results plus distinct factory instances.
   Registered14b verifies repeated cached strings retain their actual immutable
   backing, one captured alias executes once and two returned instances execute
   twice. Native14b passes with exact stdout and exit0 using the v13 private
   compiler (compile13.76 s, run34 ms;
   `/tmp/composer-lazy-strings-v1`). Retain the independent backing, authority
   and actual-instance retraction controls as capture/storage support grows.
4. **Retain two kinds of sequence output checks.** The general runner trims
   trailing spaces/newlines; the [NativeSequences runner](../tests/NativeSequences/Program.fs)
   normalizes only CRLF and also verifies retained MLIR. Run the exact sequence
   harness where whitespace/demand assertions are part of the oracle. Both must
   execute fresh artifacts; printed pass labels alone are insufficient.
5. **Do not freeze implementation limitations as language negatives.** The
   [SourceAdmission harness](../tests/SourceAdmission/README.md) includes unknown
   sequence-input and escaping factory-cell cases. Keep unsafe/missing-premise
   negatives. When G2/G3 establishes a formerly missing valid home or complete
   use, add its positive counterpart and migrate only the obsolete limitation
   expectation, with the changed proof recorded. Never accept an unrelated error.
6. **Extend real tooling clients.** The default CCS.Editor suite now includes
   staged sequence projections, but analyzer/live-LSP coverage must be inventoried
   for each newly admitted form. Exercise measured hovers, exact source alias and
   capture definitions, malformed edits, repair and stale-revision rejection in
   the actual client path. The existing analyzer and LSP suites do not inherit a
   new editor test automatically. Rebuild all consumers after shared DU changes.
7. **Run actual consumers with .NET drivers.** BAREWire
   [RoundTrip](../../BAREWire/samples/RoundTrip/RoundTrip.fidproj) has an independent
   byte oracle; [Ariel](../../Fidelity.Platform/Environments/Linux/x86_64/Ariel/README.md)
   has explicit scoped callback, retirement and real mapped-carrier checks.
   RoundTrip now has a [.NET driver](../../BAREWire/tests/NativeGate.fsx) and
   [driver tests](../../BAREWire/tests/NativeGateTests.fsx), preserving exact bytes,
   fresh-artifact checks, failures and timeout cleanup. A fresh coordinated
   RoundTrip run remains required. Ariel's process-driver layer still needs the
   .NET migration, preserving fixtures, byte comparisons and release observations.
   A library build, protocol model or prior
   successful hash does not close these consumer gates. Hardware-dependent
   evidence retains the actual device prerequisite and result, not a simulated pass.
8. **Check the transformed artifact.** Existing Alex component tests and native
   operation-presence checks are bounded evidence. Add each changed property's
   admitted preservation result or actual emitted/lowered-artifact check, as
   required by [shared acceptance §4.1](PRDs/C-Series-Acceptance.md#41-integrity-through-realization-colibri-fpga-and-ebpf).
   A model regenerated from the pre-emission graph cannot detect incorrect wiring
   in the emitted program. FPGA/Colibri and eBPF guide integrity here; their
   scheduling/encoding algorithms remain backend work, not Alex semantics.

### Contract decisions with immediate work attached

| Seam | Required resolution before the corresponding positive claim |
|---|---|
| Direct mutable capture signature | Select the single typed cell-domain authority described in [Direct Capture Cell Contract](Direct_Capture_Cell_Contract.md), propagate it through type/application/range/layout consumers, then implement and test direct and recursive forwarding. Existing `EnvironmentBorrow` alone does not settle that signature. |
| Recursive value initialization; lazy reentry/nonreturning force | Reconcile native initialization/failure with the current error model; specify and test the admitted behavior and located refusal of inadmissible forms. Continue settled function recursion and ordinary successful memoization meanwhile. |
| Partial selectors and successor extrema | Settle `Option.get`, `Map.find`, `Seq.head`, extrema/selection ties and zero-step range admission where the PRDs identify ambiguity. Implement same-subject guards or the adopted native outcome; no imported managed exception or fabricated default payload. |
| Concurrent lazy forcing | Apply the normative single-forcer ownership and target publication requirements before advertising it. Single-thread memoization is required now. Cross-actor/thread use joins the owning A/T contracts; a CAS instruction alone is not that proof. |
| Foreign registration and descriptor transport | Keep [C-01 §6.7](PRDs/C-01-Closures.md#67-the-boundary-contract-as-a-joint-constraint) A/B/C and [§14.2](PRDs/C-01-Closures.md#142-the-form-family-and-its-selection-at-saturation) explicit. Preserve named typed entries now. Capturing registration needs real adapter/invocation/release evidence with Farscape/IO/Desktop owners; memory-fabric descriptor transfer additionally needs relocation, sharing and destination-code contracts. These explicit broader integrations do not waive any required interior capture/return/storage gate. |

## 5. Executable checks and evidence record

Run from the active Composer worktree. Commands below exist now; they do not
claim passing results. Coordinate builds; project references must be enabled
when compiler contracts changed. Confirm nonzero test discovery and inspect the
runner's selected entries. Add missing cases before treating a full run as full
coverage. New 13a selectors are executable only after manifest registration.

```sh
# G1–G3 focused source/graph and physical composition.
dotnet test ../clef/tests/Clef.Compiler.Service.Tests/Clef.Compiler.Service.Tests.fsproj --filter 'FullyQualifiedName~SequenceApplicationCases|FullyQualifiedName~ClosureEnvironmentCases|FullyQualifiedName~EnvironmentFactoryResultsCases|FullyQualifiedName~SequenceOriginCases|FullyQualifiedName~SequenceResidenceCases'
dotnet test tests/Alex.Tests/Alex.Tests.fsproj --filter 'FullyQualifiedName~Callable|FullyQualifiedName~FunctionResult|FullyQualifiedName~LambdaOccurrence|FullyQualifiedName~MutableClosure|FullyQualifiedName~SequenceBoundary'
dotnet run --project tests/CCS.Editor.Tests/CCS.Editor.Tests.fsproj -- --sequence-applications
dotnet fsi tests/regression/Runner.fsx -- --sample 16h_SequenceApplications --sample 16a_SequenceOperations --jobs 2 --results /tmp/composer-callable-gates
dotnet run --project tests/NativeSequences/NativeSequences.Tests.fsproj -- src/bin/Debug/net10.0/Composer --sample 16h_SequenceApplications

# Discriminating canonical-storage/callback controls; expand to all cases at G10.
dotnet run --project tests/NativeCallbacks/NativeCallbacks.Tests.fsproj -- src/bin/Debug/net10.0/Composer FunctionSnapshots FunctionFields CapturedRecords CapturedBuffers OptionFunctionPayloads DirectCaptures CallEffects

# G4 original recursive and recurrence cases, without replacing their sources.
dotnet fsi tests/regression/Runner.fsx -- --sample 13_Recursion --sample 15_SimpleSeq --sample 15d_SequenceAdditive --jobs 3 --results /tmp/composer-recurrence-gates

# G10 coordinated broad baseline, after missing native entries are registered.
dotnet test ../clef/tests/Clef.Compiler.Service.Tests/Clef.Compiler.Service.Tests.fsproj
dotnet test tests/Alex.Tests/Alex.Tests.fsproj
dotnet run --project tests/CCS.Editor.Tests/CCS.Editor.Tests.fsproj
dotnet run --project ../lattice-analyzers/tests/Lattice.CCS.Integration/Lattice.CCS.Integration.fsproj -p:CCSEditorProject="$PWD/src/CCS.Editor/CCS.Editor.fsproj"
dotnet run --project tests/NativeCallbacks/NativeCallbacks.Tests.fsproj -- src/bin/Debug/net10.0/Composer
dotnet run --project tests/SourceAdmission/SourceAdmission.Tests.fsproj -- src/bin/Debug/net10.0/Composer
dotnet fsi tests/PlatformFormat/Runner.fsx
dotnet fsi tests/regression/Runner.fsx -- --jobs 4 --results /tmp/composer-c-f-delivery

# Actual BAREWire consumer, after selecting the coordinated built compiler.
dotnet fsi ../BAREWire/tests/NativeGateTests.fsx
dotnet fsi ../BAREWire/tests/NativeGate.fsx -- "$PWD/src/bin/Debug/net10.0/Composer" --process-host-project "$PWD/tests/Infrastructure/ProcessHost/ProcessHost.fsproj" --results /tmp/barewire-native
```

The full default editor invocation already includes its specialized projection
groups; do not duplicate them all at the same gate without a new failure. The
analyzer command explicitly selects this worktree's editor. Live LSP requires
its separate actual-server/client gate and recorded loaded compiler identity;
the existing client registrations are in
[package.json](../../lattice-vscode/client/package.json). Keep .NET-only new
process orchestration; no Python check commands are introduced here.

NativeSequences, callback, source-admission, formatter and regression harnesses
have different timeout/output contracts. Preserve each recorded contract. The
last four-control checkpoint used an explicit 180-second override and did not
establish default-timeout acceptance. Artifact-formatting performance remains
work to close when it prevents required manifest runs; do not suppress `-k`,
prune semantic facts or weaken expected output to obtain a green result.

For every completed increment, append the gate/operation IDs, exact cases/counts,
source and compiler hashes, selected full/CompilerSurface platform, source/proof/
editor/Alex/artifact/native results, remaining failures, and retained evidence
paths to the [waypoints](Language_Coverage_Waypoints.md). A recorded source pass
cannot replace its native result, and 32/64-bit MLIR component verification cannot
be reported as cross-device execution. Final PRD reconciliation compares this
inventory with the original criteria and the actual expanded tests, including
all affected F controls and required consumer/tooling gates.
