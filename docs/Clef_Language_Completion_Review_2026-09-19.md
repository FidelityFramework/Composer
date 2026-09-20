# Clef language completion: September 19 review

This is an evidence and supersession record for the existing
[completion roadmap](Clef_Language_Completion_Analysis.md) and
[workload frame](Clef_Language_Completion_Workload_Frame.md). Composer is the
canonical home of the implementation roadmap. Completion §14 supplies the
dependency baseline; ordering and designs remain subject to correction from
coverage and implementation evidence. This review does not introduce a parallel
plan in Clef.

The current scope includes FP combinators such as `Option.map`, NTU dimensional
types, negative/fractional admission, computation expressions, Olivier/Prospero/
Ariel, and additional standard MLIR dialects. The earlier analysis's exclusion
of negative/fractional types and its historical HelloDISCO work window do not
define this review's scope. Clef semantics govern throughout: no null, object
widening, CLR reification, class-based actor model or source escape context.

## 1. Baker nanopasses are the implementation structure

The existing [Baker design](../../clef/docs/fidelity/Baker_Saturation_Architecture.md)
defines ingredients, structural patterns and operation recipes. Semantic
extensions use these through the established nanopass fan-out/fold-in structure.
Recipes introduce the required graph structure and return its codata; fold-in
integrates them at the established accumulation point. A missing ingredient is
work in that layer, not a reason for a recipe or witness to construct a second
semantic path. Zippers carry positional navigation, not semantic analysis or
accumulation state.

Alex's witness architecture is pull-based through the positional Huet zipper,
with compositional elements and patterns. Semantic changes must not introduce
imperative emission drivers, push traversal, recursive subtree emitters or mutable semantic
accumulators. A fact missing at a witness is supplied by its owning Baker
nanopass, not inferred by emission code.

Each feature must identify the nodes and relationships its recipe introduces,
their source origins, the complete participant set and ordered roles, the facts
required for settlement, and the obligations attached to those participants.
The applicable checking procedures and their readiness belong to elaboration
and saturation. Alex consumes settled consequences projected onto nodes or
deliberately reified attributes; it does not query hyperedges to reconstruct
meaning. [PSG nanopasses](PSG_Nanopass_Architecture.md),
[thin middle end](Thin_Middle_End_Design.md).

Dimensions, ranges, access, lifetime, layout and proof facts can participate in
one joint hyperedge. Their solver-family projections remain distinct and retain
the same program identities and premises. The correction ledger at the start
of [Horizon Requirements](../../clef/docs/fidelity/phg/Horizon_Requirements.md)
explicitly supersedes older table entries claiming that hyperedges cannot span
families. Dimension group equations, propagated range least fixed points,
bitvector-mask obligations and region/access enum equality must not be collapsed
into one constraint procedure. Sharing an arithmetic solver theory does not
identify the group and range disciplines; enum equality has its own QF_UF
projection.

## 2. Existing work items and the language needs they cover

These are prior requirements, not claims that every listed operation is already
implemented. PRD capability lists survive where their old realization sketches
have been superseded.

| Existing work | Clef surface and graph contract | Acceptance pressure |
|---|---|---|
| [C-01](PRDs/C-01-Closures.md), [C-02](PRDs/C-02-HigherOrderFunctions.md), [C-03](PRDs/C-03-Recursion.md) | Generic application, partial/residual functions, stored/returned/nested functions, recursion, ordered evaluation, capture identity and shared environment forms | BAREWire generic codecs; typed actor behaviors; retained callbacks |
| [C-04](PRDs/C-04-CoreCollections.md) | List construction/query/map/filter/fold/foldBack/rev/append/search; Map lookup/update/remove/traversal/transformation; Set operations; Option map/bind/defaults/tests/get/toList; ranges and tuple decomposition | BAREWire schema/layout tables and distinct typed outcomes; UI children |
| C-04 §13.2 | Primitive constructors/eliminators versus decomposable HOFs; type-only signatures do not supply Baker elaboration | Combinators retain actual argument/result dimensions, storage identity and callback behavior |
| [C-05](PRDs/C-05-Lazy.md), [C-06](PRDs/C-06-SimpleSeq.md), [C-07](PRDs/C-07-SeqOperations.md) | Deferred construction, force, yield/delegation, map/filter/take/fold/collect, nested composition and independent iteration | Deferred traversal, mailbox selection, incremental computation |
| [A-01](PRDs/A-01-BasicAsync.md), [A-02](PRDs/A-02-AsyncAwait.md), [suspension design](Delimited_Continuations_Architecture.md) | Return/bind/delay/composition, evaluation-ordered segments, cuts, live-across slots, delimiter association, resumption and retirement | Actual delayed I/O, timer/RPC completion and actor receive |
| [T-03](PRDs/T-03-BasicActor.md), [T-04](PRDs/T-04-ActorReply.md), [T-05](PRDs/T-05-ParallelActors.md) | Typed posting, receive loops, state, replies and actor composition under the current scheduler contract | Olivier message execution, Prospero supervision/placement, Ariel scheduling |
| [R-01](PRDs/R-01-ObservableFoundations.md), [R-02](PRDs/R-02-ObservableOperators.md), [R-03](PRDs/R-03-ObservableIntegration.md) | Typed emission, owned subscription/release and planned operator composition | Reactive delivery; the normative core is `Observable<'T>`, not the PRDs' inherited observer-object model |
| [R-04](PRDs/R-04-IncrementalFoundations.md), [R-05](PRDs/R-05-IncrementalDynamism.md), [R-06](PRDs/R-06-IncrementalIntegration.md) | Return/map/map2, variables, cutoff, stabilization, bind and active dependencies; local graphs and inter-actor delivery | Fidelity.UI and Prospero demand management |
| [UI component model](../../Fidelity.UI/docs/02_component_model.md), [reactive signals](../../clef-lang-spec/spec/reactive-signals.md) | Ordinary typed functions/lists/modifiers and equivalent optional CEs; cold construction, Signal/Memo/Effect/Batch, mounted ownership | Retained UI composition, keyed identity and coherent stabilization |

BAREWire's [intersection subset](../../BAREWire/docs/12%20Intersection%20Subset.md)
requires independent generic instantiation, recursive aggregates, nested
patterns and qualified record identity. Its restrictions are compiler debt.
For layout analysis, `Ok (Some offsets)`, `Ok None` and `Error findings` have
distinct meanings; FP composition must preserve them before allocation or proof
construction. Its September 6 native failures supersede the document's earlier
September 3 success as a current acceptance claim.

The latest R-04/R-06 and [UI reconsideration](../../Fidelity.UI/docs/08_ui_model_reconsideration.md)
already correct several important assumptions: dependencies follow actual reads
and effects, including callees and aliases; a cutoff on one join input cannot
erase independent invalidation on another; one actor can own many local
incremental nodes; and demand withdrawal does not establish reclamation while
uses remain outstanding. Their existing conformance cases should be reused.

## 3. Shared environment, CEs and scheduling

The review found an admission defect: non-seq computation bodies became
`Sequential`, return forms became their payload, and `do!` became a unit-typed
sequence. A plain identity function could therefore pass as a builder. The
2026-09-20 [computation admission waypoint](Language_Coverage_Waypoints.md)
closes that false acceptance with located CCS8401 errors before semantic
erasure, including unsupported resource use and inherited yield context across
ordinary deferred boundaries. Existing native `seq` source handling remains
separate. General builder resolution, bind/delay semantics and suspension still
require the functional and shared-environment prerequisites in completion §14.
See [checker handling](../../clef/src/Compiler/NativeTypedTree/NativeService.fs).

C-01 §14 and the [closure contract](../../clef-lang-spec/spec/closure-representation.md)
already define closure environment, continuation frame and actor state as
instances of the shared environment construction. The interior function value
is `(fn, env)`; the function is not stored as a data word in the environment.
The [layout design](../../clef/docs/fidelity/phg/Layout_As_Joint_Constraint.md)
settles placement and its joint proposition together. The
[retooling plan](../../clef/docs/fidelity/phg/Closure_Retooling_Plan.md) supplies
the nanopasses and gates; older sample counts in it remain dated evidence.

General CE support must preserve the admitted operations and their evaluation,
scope and lifecycle laws. Delay/Run distinguish construction from activation;
bind retains continuation identity; loops and yield retain live state; resource
operations distinguish lexical setup from mounted ownership and pending uses.
`and!` expresses independent inputs without itself promising threads. A
synchronously completed bind does not establish suspension support.

The suspension design already specifies VC-EXT/STATE/ACC/DOM/ONE over graph
segments, frames and delimiter/delivery relationships. CPU state-machine
realization is first, with monitor-map IPC/timers named as the initial consumer.
The same construction serves actor receive with its declared delivery edge.

The [scheduler contract](../../clef-lang-spec/spec/scheduler-contract.md) assigns
Olivier actor/message execution, Prospero supervision/restart/placement, and
Ariel suspension/resumption scheduling. It requires exclusive actor turns,
finite-event fairness, bounded refusal, independent control-plane capacity,
reproducibility under substituted sources and declared target assumptions.
Clients are the compiler and supervisor. Its first simulated actor profile
still has no conforming implementation recorded.

The [existing Ariel package](../../Fidelity.Platform/Environments/Linux/x86_64/Ariel/README.md)
provides narrower, concrete synchronous-region execution: persistent bounded
carriers, exact claims, generation acknowledgments and callback retirement.
[Dispatch regions](../../BAREWire/docs/13%20Dispatch%20Regions.md) already
distinguish output completion from retirement and require backing identity,
complete indirect capture footprints and exact partition coverage. This
substrate remains a regression baseline, not proof of general actor scheduling.

## 4. NTU and negative/fractional admission

The [dimensional design](../../clef/docs/fidelity/phg/Dimensional_Range_Design.md)
and [status ledger](../../clef/docs/fidelity/phg/Dimensional_Steps_1_2_Sequence.md)
already settle source kind/dimension, range provenance and target representation
selection. Their later corrections govern older sections: program facts reside
in CCS codata; emission derives value names from node identity and ordinal.
Region/access equality is separate from measure algebra. Dimensions persist for
their compilation/proof roles; final BARE bytes do not imply early erasure.

The [negative/fractional draft](Negative_Fractional_Types_Architecture.md)
specifies direction and pairing judgments on the PSG: `Neg` preserves physical
dimension while reversing judgment direction; `Recip` inverts dimension.
VC-PAIR/DIR/CANCEL/OBS compose with environment and lifetime obligations over
the same actual participants. The one-shot-frame and fractional-demand-cell
realizations are hypotheses awaiting their exercise.

Negative integer measure exponents, rational measure exponents, exact rational
numeric literals and dual type constructors are distinct. Current dimension
carriers use integer exponents; the documented initial increment rejects
fractional exponent syntax. Exact rational literals do not close that extension.
The draft's rational cancellation uses QF_LRA and its second horizon presupposes
closure recipe migration and CPU suspension. It explicitly does not schedule a
new sequence. Multi-shot duals and grade-axis composition remain open there.

## 5. Alex's standard dialect expansion

The requested `affine`, `tensor`, `async`, `math`, `index` and `cf`, alongside `memref`,
`vector`, `func`, `scf` and `arith`, extend the historical five-dialect baseline.
The existing retooling plan provides the admission register and consumer-gate
mechanism. The five-only tables require coordinated revision as support lands;
they do not overrule this requested scope. Continuation semantics still settle
in Baker/PSG before witnessing standard operations.

| Dialect | Current source evidence | Completion connection |
|---|---|---|
| `arith`, `memref` | Typed operation families, serialization and consumers exist; current memref constructors cover scalar and rank-one forms | Aggregate/buffer operations must use settled extent, placement and numeric facts |
| `func` | Calls, constants, declarations and definitions exist; the current single-result type model does not establish the settled multi-value closure convention | Shared environment migration and higher-order application |
| `scf` | If/while/for/yield/condition exist; no `scf.index_switch` representation/emission found | The suspension design's discriminant dispatch remains a concrete missing operation |
| `index` | 26 `IndexOp` cases; six serialize, including two emitted as `arith.constant`; remaining cases become TODO comments | Complete required dimensional-index operations and target realization; source physical dimensions remain NTU facts |
| `vector` | Vector type/printing and backend conversion exist; no operation family/witness found; current size helper ignores lane count | Connect admitted lane/shape/operation facts and their layout obligations |
| `affine` | Upstream tooling provides the dialect and lowering, but no Alex operation/element/pattern family was found | Settled iteration domains, bounds and maps must arrive from the owning graph analysis; verify their witness and target realization |
| `tensor` | No type/operation family or emission found; no bufferization phase in the inspected LLVM pipeline | Pure multidimensional values before the declared layout/ownership commitment |
| `async` | No token/value types, operation family or emission found | Settled dependency/task/suspension forms, preserving scheduler and BAREWire contracts |
| `math` | No operation family or emission found | Recognizable nonlinear operations with declared dimensional and numeric semantics |
| `cf` | Direct `cf.assert` exists; no branch/switch operation family found; backend SCF-to-CF lowering exists | Explicit block-control forms where the settled continuation/pathway contract requires them |

Evidence: [operation/type model](../src/MiddleEnd/Alex/Dialects/Core/Types.fs),
[serialization](../src/MiddleEnd/Alex/Dialects/Core/Serialize.fs),
[bounded-view assertions](../src/MiddleEnd/Alex/Patterns/BorrowedViewPatterns.fs),
[LLVM pipeline](../src/BackEnd/LLVM/Lowering.fs). This audit did not implement
new dialect support. Standard meanings were checked against the official
[MLIR dialect reference](https://mlir.llvm.org/docs/Dialects/).

## 6. Supersession findings

| Retained material | Governing correction |
|---|---|
| C/A/T PRD layout and coroutine recipes in Alex, direct LLVM emission, one OS thread per actor | Keep their capability cases; use C-01 §14, Baker suspension, the scheduler contract and the thin witness boundary for implementation |
| C-01's older unchecked status boxes | Later closure architecture and fresh native cases establish narrower actual support; individual family gaps remain |
| R-01–R-03 observer objects, disposal interfaces and runtime queue sketches | The current Observable core uses one payload parameter and owner-scoped subscription release; the full operator/CE surface remains specification work |
| R-05 blanket elimination of runtime cost/cycles/reclamation work | Current R-04/R-06 retain instance state, independent invalidation and outstanding-use lifetime obligations |
| Blanket claims that all hypergraph/proof work is future | PSG Nanopass's September addendum records graph obligations, residence, consecutive-string layout, discharge and artifact twins as landed |
| Early dimensional tables, source seals and preassigned SSA plans | Later owner rulings/as-built corrections in the dimensional design and handoff govern; old results are not fresh gates |
| February SMT strategy | Retired after full review; [Obligation Residency](Obligation_Residency_Design.md), [Proof Composition](Proof_Composition_Architecture.md), dimensional work and [Lattice contract](../../clef/docs/fidelity/phg/Lattice_Consumer_Contract.md) own its relevant subjects |
| Clef migration guide's invented permission/escape context, binding stubs and coeffect-arrow syntax | Removed; platform calls retain the same memory, access, lifetime and proof judgments. Bounds obligations have no source exemption |
| Clef Vision and lazy/seq/coroutine strategy sketches | Historical orientation/inventory; Composer's current roadmap and governing semantic contracts determine work |
| Incremental specification's inherited `Equals` override/comparer wording and syntax-only dependency account | Replaced with its already specified ordinary typed cutoff function and actual read/effect dependency requirements; native Incremental and bootstrap reuse are distinguished in the [contract direction](Nanopass_Incremental_Contract_Direction.md#9-native-incremental-and-bootstrap-investigations) |

Remaining recorded intersections include initial non-memoizing versus later
memoizing Lazy behavior, stored-function placement wording, dynamic child-scope
reclamation, pending subscription delivery and failure behavior, and the NFT
draft's acknowledged pre-print conflicts. T-05's fixed-sleep sample can observe
partial totals and its printed worker totals are inconsistent; it is not yet a
decisive composition gate. These need resolution at their owning work items.

## 7. Dependency order and evidence collected

Completion §14 identifies the dependencies: contract inventory; functional expression
and inference; shared environment; collection/deferred families; generalized
suspension; actor coordination/reactivity; admitted net pathway; additional
target/proof profiles. Numeric, lifetime, boundary and proof work supplies
prerequisites throughout. The earlier dimensional sequence is CS9 operators →
CS10 range → CS11 CPU/boundaries → region/access → CS12 source migration → CS13
reals; its dated checkpoints are not a new current backlog or acceptance count.
The owner's September 19 clarification explicitly prioritizes completion and
coverage over a fixed order or an outdated design. New lettered FidelityHello
variants are authorized wherever distinct cases need an oracle. Architectural
and semantic prerequisites still have to be established by the owning passes.

Each slice owes the roadmap's four artifacts: source cases; inspectable graph
with origins and obligations; realized artifact with correspondence; and actual
gate results for that revision. Graph-born obligations and artifact twins are
specific implemented families, not evidence that all proof composition is done.

Reviewed heads: clef `83d01dd1`, Composer `a979f64c`, BAREWire `33364d4a`,
Fidelity.Platform `dcd3424e`, Fidelity.UI `b7ef6f90`. Current working-tree
revisions, including R-04/R-06, were read and preserved; HEAD alone does not
identify that snapshot.

| Fresh check | Result and scope |
|---|---|
| Composer Debug build | Passed; zero errors, three existing dependency/source-link warnings |
| CCS.Editor.Tests | Passed 13 reported groups, including actual cvc5 outcomes, dimensions, source identity and invalidation |
| StaticStorageRegression | Passed 10 correspondence cases, including deliberately changed artifacts |
| NativeCallbacks | `OptionPartials` compiled and executed; `OptionFunctionPayloads` failed on capture `directTrace` having no retained bounded view; remaining ten cases were not run |
| Ariel lifecycle model | Two tests passed, including 1,747 bounded states; model evidence, not native actor conformance |
| [Source inventory](../../clef/tests/LanguageSurface/Probe.fsx) | 11/20 expectations met; nine gaps; [baseline](../../clef/tests/LanguageSurface/baseline-2026-09-19.txt) includes the loaded CCS assembly hash |

The source inventory checks ordinary generic functions, captures, generic record
fields, Option map/bind/filter and measured mapping, plus rejection of null,
object widening and boxing. Missing Option.fold, Result.map/bind and List.map/fold
resolution account for five probe gaps; four non-builder CE forms are incorrectly
accepted. These are source-checking observations without a platform or a native/
proof claim. Positive verdicts do not assert inferred result types or graph
obligations. Dimension/null/object/boxing rejection requires the recorded located
diagnostic identity; non-builder CE rejection currently requires a located
effective error, pending its admission diagnostic. Reasons are printed. The
captured-view native failure independently identifies a shared
environment case whose provenance must be traced upstream, not guessed in Alex.

At this initial review checkpoint, no compiler semantic change had been made.
`Option.fold` was an exploratory probe: the reviewed normative Option chapter
and C-04 do not specify it, so its absence does not establish a missing adopted
contract. Native BAREWire, UI rendering, hardware targets and the full regression
suite were not freshly run. Subsequent implementation evidence is recorded below.

Post-edit validation: the source inventory still reports 11/20 met and nine
gaps against the same CCS hash, now retaining rejection diagnostics and checking
their intended codes for dimension/null/object/boxing cases. Added local links
resolve and diff whitespace checks pass. The corpus drift gate still reports
the same four unscheduled findings as before this work: the dimensional handoff's
historical pointer wording, two completion-analysis vocabulary lines and a C
boundary comment in PlatformPatterns. It separately reports 4,161 scheduled
migration lines. This is an unchanged failing gate, not a clean baseline.

## 8. Full-read inventory and limits

The following documents were read in full, including corrections, historical
appendices and current working-tree revisions. Bounded rereads covered portions
omitted by truncated tool output. This is not a claim that every repository file
or external research paper was read.

| Repository/group | Full reads |
|---|---|
| Composer roadmap | Completion Analysis; Completion Workload Frame; PRDs README |
| Composer PRDs | C-01 through C-07; A-01/A-02; T-03/T-04/T-05; R-01 through R-06 |
| Composer architecture | Thin Middle End; Single Flattening; Obligation Residency; Delimited Continuations; Negative/Fractional Types; Proof Composition; Closure Nanopass Architecture |
| Clef PHG | README; PSG-to-PHG Plan; Layout as Joint Constraint; Closure Retooling; Design Supersession Register; Dimensional Handoff; Dimensional Range Design; Dimensional Steps 1–2 Sequence; Dimensional Step 1–2 Design; Dimensional Vetting; Types as Ranges; Horizon Requirements; Lattice Consumer Contract |
| Clef other | Baker Saturation Architecture; CCS Lazy/Seq/Coroutine Intrinsics; From F# to Clef; Clef Language Vision; SMT Integration Strategy before deletion |
| BAREWire | docs README; Readiness Audit; Implementation Status; Intersection Subset; Substrate Formalism; Arena Design; Platform Description; Dispatch Regions |
| Fidelity.Platform | docs README; Canonical Platform Spec; BAREWire Rebase Plan; Admission and Sidecars; Platform Composition; D4b Platform Carrier Location; STM32H7 Synth Design; Ariel Linux/x86_64 README; Ariel native STATUS |
| Fidelity.UI | README; documentation chapters 00 through 08 |
| Specification | closure-representation; seq-representation; observable-computation; reactive-signals; scheduler-contract; platform-bindings; ffi-boundary; special-attributes-and-types; behavior-classification; memory-regions; access-kinds; program-hypergraph; conformance |

Targeted references, not full reads in this review: Coeffect Analysis opening
and projection provisions; PSG Nanopass's complete PHG addendum; Lattice
Integration status/correspondence sections; lazy-representation §§9/11/12;
seq-operations-representation overview and §§3–5; incremental-computation
§§1–6/11/14–15 and §8.1 boundary note; program-semantic-graph §14; relevant
width-inference and array-bounds provisions; Platform DISPLAY_MODEL; source
paths supporting the measured gaps and dialect table. External papers/preprints
were not independently audited; Horizon was read as their locally vetted
synthesis, with its correction ledger taking precedence over retained tables.

## 9. Implementation evidence

The first implemented surface addition is `Option.defaultValue`, with the source
scheme `'a -> 'a option -> 'a`. Its fallback is evaluated eagerly, including when
the option is `Some`. A stored partial evaluates and snapshots the fallback when
formed. The existing lookup rules preserve lexical declarations and function
fields named `Option.defaultValue`. The
[typed intrinsic scheme](../../clef/src/Compiler/NativeTypedTree/Expressions/Intrinsics.fs),
[Baker recipe](../../clef/src/Compiler/Baker/Recipes/OptionRecipes.fs) and
[saturation dispatch](../../clef/src/Compiler/Nanopass/BakerSaturation.fs)
implement this through ordinary graph structures. The operation consumes two
arguments; further arguments invoke a selected function payload. An option-valued
fallback remains the payload and is not mistaken for the option being consumed.

The [CCS cases](../../clef/tests/Clef.Compiler.Service.Tests/OptionDefaultsCases.fs)
check source types and dimensions, independent alias specializations, explicit
type application, nested options, callable payload boundaries and lexical
precedence. Graph checks require eager argument order, a guard and extraction
referring to the same option node, an immutable partial-formation snapshot with
its original source identity, and no remaining reachable `defaultValue`
intrinsic. Stored measured applications retain the ordinary application
obligation's call, callee, arguments and referenced definition as participants.
This is existing application evidence, not a new Option-specific theorem or
complete closure proof discharge.

Admitted cases require `parseAndCheck` success and no graph error nodes. Every
negative case reaches an entrypoint and requires `CheckFailure`, effective error
severity, the expected diagnostic code and its exact marked source span.
Rejections cover incompatible dimensions through direct, stored, nested and
callable payloads; callable arity and scalar overapplication; type-argument arity;
and numeric-kind or non-option argument mismatches. Written positive and negative
rational measure exponents currently require CCS8048; a square root requiring a
nonintegral dimensional exponent requires CCS8041. Positive controls preserve
inverse measures and equal dimensions from different producers, and retain the
exact numeric literal `-0.25` as rational metadata `-1/4` without manufacturing an
integer range. These distinguish numeric fractional values, dimensional
exponents and the separate NFT dual-type admission work.

| Fresh gate | Result and scope |
|---|---|
| CCS service tests | **271/271 passed**, including **33 OptionDefaults cases**. This supersedes the earlier implementation checkpoint of 252 service tests and 14 OptionDefaults cases; the final expansion changed tests only. |
| CCS.Editor.Tests after the implementation | **13 reported groups passed**, including actual cvc5 outcomes, immutable snapshots and edit invalidation. Existing package-downgrade/dependency warnings remain. Log: `/tmp/clef-option-defaults-editor.log`. |
| [Alex component tests](../tests/Alex.Tests/README.md) | **12/12 passed**. Huet navigation preserves graph facts and obligation edges; public pattern tests check operand recall, index-range consumption and specific missing-input diagnostics. Serialized functions pass the real MLIR verifier and standard lowering at 32- and 64-bit index widths. |
| [NativeCallbacks](../tests/NativeCallbacks/README.md) selected cases | Four fresh executables passed: [OptionDefaults](../tests/NativeCallbacks/OptionDefaults.clef), OptionPartials, OptionCallbacks and OptionEvaluation. Coverage includes eager argument effects, Some/None callback behavior, stored snapshots, aggregate/function payloads and measured specialization. Evidence: `/tmp/composer-callbacks-fsharp-d513833ce41e466590d1a38124f1698d/`. |
| [FidelityHello 08a_OptionDefaults](../samples/console/FidelityHelloWorld/08a_OptionDefaults/OptionDefaults.clef) | Fresh compile and native execution both exited zero; all six fixed lines matched its [manifest entry](../tests/regression/Manifest.toml). It covers eager fallback ordering, stored defaults, nested options, function selection and measured payloads. Evidence: `/tmp/clef-option-defaults-oracle-2rbbthft/`, including source-to-native logs and loaded compiler assembly hashes. |
| [Regression harness tests](../tests/regression/RunnerTests.fsx) | Eight focused groups passed: matching stdout with nonzero native exit; launch failure; concurrent stdout/stderr; timeout and descendant termination; blocked input; literal argument boundaries; unmatched filters; real CLI exit propagation. |
| [SMT transfer regression](../tests/SMTTransferRegression.fsx) | **50 cases passed**: 11 dimensional, 6 real-bound, 11 integer, 10 application-dimension and 12 concrete-layout cases, including deliberately false claims through the actual solver/export paths. Log: `/tmp/clef-option-defaults-smt.log`. |
| [Static-storage correspondence](../tests/StaticStorageRegression.fsx) | **10 cases passed**, including changed offsets, bytes, alignment, symbols, anchors and plans. Log: `/tmp/clef-option-defaults-storage.log`. |

The new Alex project exercises the existing public observation boundary. Its
fixtures supply settled graph facts; obligation edges are checked for
preservation, not discharge. It introduces no semantic analysis in Alex, no new
dialect support and no complete array-bounds or closure-proof claim. The current
index pattern's missing-range fallback is explicitly recorded in that suite's
README rather than being presented as a rejection gate.

The FidelityHello variant selects the
[CompilerSurface description](../../Fidelity.Platform/Environments/Linux/x86_64/Fidelity.Platform.CompilerSurface.fidproj)
and prints fixed strings through Console, without pulling the legacy numeric
formatter into its reachable source. `08_Option` and its expectation are
preserved. The earlier four-sample baseline (`08_Option`, `11_Closures`,
`12_HigherOrderFunctions`, `18_Generalization`) failed before this addition on
the existing formatter's `byte` and suffixed literals at
[Format.clef](../../Fidelity.Platform/Environments/Linux/x86_64/Format.clef),
lines 86, 92, 97 and 104. Those failures precede native execution; they do not
establish regressions in the new operation. That baseline was not repeated
without a change to its failing dependency.

The successful 08a build still reports 106 informational findings: 87 from
unreachable dependency declarations and 19 range findings in the sample.
Its retained witnessed MLIR contains 39 `scf.if`, 17 `func.call_indirect` and
40 interim `unrealized_conversion_cast` operations. This native behavioral
result does not complete range migration or the closure representation change.
The initial `OptionFunctionPayloads` failure on capture `directTrace` lacking a
retained bounded view remains unclosed; it was not included in the four-case
passing selection. General CE, actor scheduling and NFT admission are not
established by these gates.

During this implementation the shared clef checkout advanced externally to
`94e28c7ba`, incorporating the compiler edits. The subsequent authorized
waypoint commits and companion toolchain revisions are recorded in
[Language Coverage Waypoints](Language_Coverage_Waypoints.md). The earlier
review HEADs alone do not identify this implementation checkpoint.

Reproduction commands, from Composer, run sequentially after coordinating shared
compiler outputs:

```sh
dotnet build src/Composer.fsproj -c Debug
dotnet test ../clef/tests/Clef.Compiler.Service.Tests/Clef.Compiler.Service.Tests.fsproj
dotnet test tests/Alex.Tests/Alex.Tests.fsproj
dotnet run --project tests/CCS.Editor.Tests/CCS.Editor.Tests.fsproj
dotnet run --project tests/NativeCallbacks/NativeCallbacks.Tests.fsproj -- /home/hhh/repos/Composer/src/bin/Debug/net10.0/Composer OptionDefaults OptionPartials OptionCallbacks OptionEvaluation
dotnet fsi tests/regression/Runner.fsx -- --sample 08a_OptionDefaults
dotnet fsi tests/regression/RunnerTests.fsx
dotnet fsi tests/SMTTransferRegression.fsx
dotnet fsi tests/StaticStorageRegression.fsx
```

The normal FidelityHello runner rebuilds Composer before compiling its selected
samples. The recorded 08a check instead called the same manifest and
[RunnerCore process/output functions](../tests/regression/RunnerCore.fsx)
against the already-built executable, without an implicit rebuild. Its temporary
`Run.fsx` is retained beside the evidence. The final service-test expansion was
built with `-p:BuildProjectReferences=false` and then run with
`--no-build --no-restore`; that follow-up changed no compiler implementation.

## 10. Negative coverage follows the full pipeline

The owner's coverage requirement extends from NTU/CCS admission through graph
settlement, proof dispatch, witnessing and artifact correspondence. Test totals
do not establish equivalence with F#'s coverage. A systematic category comparison
with that baseline remains work; Clef's additional graph and proof boundaries
need their own cases.

| Boundary | Existing foundation | Required expansion |
|---|---|---|
| NTU/source admission | Dimensional cases and the reachable, located OptionDefaults rejection cases | Apply public `CheckFailure` assertions consistently; distinguish raw diagnostics on unreachable declarations from rejected executable programs; extend each new construct's positive/negative composition matrix |
| Baker construction and callable structure | Option, callable-application, closed-callback and monomorphization suites | Missing/different participants, alias and capture composition, origins and replacement correspondence; verify refusal at the owning public boundary as well as local analysis findings |
| Range, access, lifetime and layout | Integer-obligation, borrowed-view, foreign-reference and static-string-layout cases | Cross-application contradictions and exact responsible participants; missing facts versus contradicted facts; source-to-witness refusal for each admitted family |
| Proof dispatch and evidence | CCS.Editor outcomes/invalidation, application-obligation tests and 50 SMT transfer cases | Altered premises, unsupported rule/profile, missing or stale evidence and composed cross-family dependencies; a solver counterexample alone does not demonstrate every downstream refusal gate |
| Alex and MLIR | Specific missing-operand/carrier diagnostics, verifier/lowering tests and static-storage artifact mutations | Missing witness coverage, prerequisite failures and malformed operations for each dialect family; preserve source origins and provider identity as external components become possible |
| Incremental/segmented work | Current immutable snapshots and coarse stale-result exclusion | The topology, retraction, join, cycle and migration rejection cases in the [contract direction](Nanopass_Incremental_Contract_Direction.md); selective reuse is not yet an implemented gate |

Each rejection case should identify its input, expected phase, diagnostic or
failed obligation, relevant source/graph identities, and the downstream action
that must be refused. Unexpected exceptions, parser failures in a checker test,
empty test selection and unrelated diagnostic codes cannot satisfy that case.
Pair refusals with nearby admitted cases so a test does not reward rejecting the
whole language feature. Source checking, local analysis findings, solver answers,
MLIR verification and native behavior remain distinct observations of one
connected contract.
