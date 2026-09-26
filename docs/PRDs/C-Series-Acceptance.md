# C-series acceptance and continuation contract

**Criteria reconciliation: September 25, 2026. Documentation only.** C-01 through
C-07 remain **In-Progress**. Each has existing implementation and bounded
acceptance evidence. This document aligns their remaining gates; it neither
resets that evidence nor reports a new compiler run.

## 1. Authority and completion scope

The current [Clef specification](../../../clef-lang-spec/spec/conformance.md)
governs semantics. The [Baker contract](../../../clef/docs/fidelity/Baker_Saturation_Architecture.md),
[Alex architecture](../Alex_Architecture_Overview.md) and
[M-01 admission contract](M-01-DialectAdmission.md) govern implementation
ownership and operation/profile realization. The
[coverage waypoints](../Language_Coverage_Waypoints.md) identify demonstrated
behavior, actual revisions and open gates. Historical implementation sketches
and sample expectations yield to those contracts.

The C-series delivers the functional capabilities enumerated in its seven PRDs,
with source, graph, artifact, native and tooling evidence on named platform
selections. Completing it is not a claim of complete Clef conformance or of every
target pathway. Conversely, a currently unsupported conforming use remains an
implementation gap; diagnosing it accurately does not fulfill its positive gate.
Target-capability rejection and a compiler implementation limitation must be
distinguished in the acceptance record.

F-xx and C-xx carry equal importance as delivery contracts for Clef and Composer.
Their category names organize the roadmap; they do not give computation features
a weaker completion standard. Dependency order identifies which facts must be
available first, not which PRDs deserve rigor. Both groups require faithful
source behavior, located diagnostics, justified representation and storage,
preservation through the selected realization, and evidence for the delivered
artifact. A passing demonstration cannot close an advertised capability whose
composition or integrity obligations remain open.

Changes across the groups carry reciprocal regression responsibility: C work
preserves affected F behavior, and subsequent F corrections preserve accepted C
behavior. The final cohort reports both groups' relevant results and remaining
failures without presenting a narrower green selection as complete delivery.

- C-04 retains its promised List, Map, Set, Option, range and supporting operation
  surface. Existing recipes and source schemes are starting points, not native
  acceptance. Broader library additions require their own enumerated contracts.
- C-07 records its eleven-operation core separately from its listed successor
  consumers and materializers. A core checkpoint does not establish those
  successors or close the full PRD while its required gates remain open.
- Async, actor scheduling, reactive/incremental execution, general foreign
  callback retention, memory fabrics and additional target pathways retain their
  owning PRDs. Shared prerequisites needed by a C feature are implemented and
  tested here without claiming the whole dependent workstream.
- Numeric and lifetime obligations remain applicable throughout. Neither an
  absent allocator nor an unknown range authorizes a fallback representation.

The reference pathway for the next implementation cohort is the existing Linux
native compiler path. Record the exact full or CompilerSurface platform selection
per case; success without reachable platform formatting or startup does not
establish full-profile behavior. Cross-width component checks are valuable but
do not establish execution on another hardware target.

### 1.1 Baker construction, Alex witnessing and backend realization

Every C-series slice preserves the nanopass architecture and the boundary between
front-end semantic construction and middle-end witnessing. The
[incremental contract direction](../Nanopass_Incremental_Contract_Direction.md#8-baker-settlement-and-extensible-alex-witnessing)
governs their evolution; extending the language surface cannot collapse them
into a single traversal or move unfinished settlement into emission.

| Owner | Required responsibility |
|---|---|
| CCS/Baker front end | Elaborate through reusable ingredients and operation recipes; fan out applicable work and fold in graph changes with origins, ordered occurrences and joint participants intact. Owning nanopasses propagate and saturate semantic, evaluation, capture, layout and proof facts under the selected declarations. |
| Alex middle end | Navigate the settled graph at its actual Huet occurrence. Witnesses pull facts through context and invoke Patterns, which compose Elements into admitted portable physical MLIR forms. Preserve this composition and graph-to-operation correspondence as coverage grows. |
| Selected backend | Perform target realization and its preservation checks. On the FPGA path this includes circuit transformations, scheduling, buffering, Colibri selection/composition, HDL emission and technology mapping. These mechanisms do not become Alex algorithms. |

Baker's graph construction and Alex's observation have different topologies.
Preserve zipper navigation and reconstruction laws, including focus, path,
snapshot and occurrence-dependent scope, without requiring both phases to share
one concrete zipper type. A structural path does not enumerate all participants
of a joint constraint. Dependency discovery, pass scheduling, proof support and
solver state retain their owning infrastructure; they are not semantic state
hidden inside a zipper.

Baker saturation must distinguish settled work from missing prerequisites and
contradictions. Quiescence, cancellation or an exhausted budget cannot establish
readiness. Each affected analysis owns its refinement and convergence rules;
Alex consumes the resulting admitted facts. Flattened MLIR means the semantic
decomposition has already occurred above Alex's witness boundary. Structured
operations, regions, results and block arguments remain valid physical forms.
Alex's emission bookkeeping does not become a second elaboration or saturation
engine.

Alex remains target-aware through selected declarations and admitted portable
physical forms. The [target-commitment boundary](../../../clef-lang-spec/spec/backend-lowering-architecture.md#2-portable-middle-end-target-committing-backend)
places target-specific dialects and encoding in the backend. That awareness
neither imports backend circuit algorithms nor permits
reconstructing missing source semantics. A new form needs its owning Baker
contract where semantics change, Element/Pattern/Witness coverage where physical
expression changes, and backend admission where realization changes.

## 2. Established work and the next unsettled contract

These are dated recorded observations, not a fresh baseline. Consult the linked
waypoints for compiler hashes, companion revisions, scope and partial reruns.

| PRD | Established work to preserve | Remaining acceptance focus |
|---|---|---|
| [C-01](C-01-Closures.md) | Direct immutable capture passing and bounded known-callee environments, native capture/callback controls | General callable transport, actual environment instances, mutable direct-call signature authority, aggregate/callable captures, residence and release |
| [C-02](C-02-HigherOrderFunctions.md) | Native callback applications and typed Option/Result/Seq recipe families | Stored/bare operation values, actual callable boundaries with shared deferred operands, retained values and full application forms; unchanged 16h |
| [C-03](C-03-Recursion.md) | Recursive binding identity, nested capture discovery and existing native recursive paths | Original13 generic-width failure, recursive-group effects/captures and admitted numeric recurrence rules |
| [C-04](C-04-CoreCollections.md) | Tested Option operations, range-loop work, collection schemes/recipes and existing array consumers | Canonical collection storage, sentinel/link/guard/capacity evidence, persistent operations and registered 13a native gates |
| [C-05](C-05-Lazy.md) | Source admission and inherited lazy realization | Canonical environment/result representation and memoization; explicit correction of sample14's obsolete recomputation expectation |
| [C-06](C-06-SimpleSeq.md) | Native continuation construction, 15a–d bounded sequence/element/borrow/additive cases | Original15 coupled/multiplicative recurrences, aggregate retention, full callable origins and wider residence |
| [C-07](C-07-SeqOperations.md) | 16a–g producer/consumer, callback, demand, Option and startup cases | Original16 factory/capture residence, 16h staged callable forms, remaining successor operation contracts |

The [C-06 checkpoint](../Language_Coverage_Waypoints.md#c-06-native-continuation-settlement--2026-09-20)
and [C-07 checkpoint](../Language_Coverage_Waypoints.md#c-07-sequence-operations--implementation-waypoint-acceptance-open-2026-09-20)
retain their own cohorts. Later [formatter acceptance](../Language_Coverage_Waypoints.md#f-05-character-storage-and-native-formatting--2026-09-20)
closes the recorded F-05 regression. Earlier failing totals must not be repeated
as a newly measured state; later focused passes do not constitute a full rerun.

## 3. Acceptance matrix for each capability

Apply the [evaluation strategy contract](../Evaluation_Strategy_Contract.md)
when shared callable, capture, storage or continuation machinery serves different
computation forms. Formation, activation, demand, delivery, memoization and
dependency invalidation have distinct oracles. Runtime reactive stabilization
and scope-aware incremental compilation also retain separate contracts.

Before implementation, give each promised operation/form a row naming its
specification clause, owning passes, existing evidence, open cases and target
selection. Expand the following dimensions where applicable; use representative
interactions with explicit coverage rationale rather than claiming all
combinations from one example.

| Dimension | Required distinctions |
|---|---|
| Application | Direct, pipeline, alias, bare operation value, each partial-application frontier, returned and stored use; lexical shadowing and explicit specialization where admitted |
| Values and types | Scalar and measured values, independent callback/accumulator/result types, nested Option/Result, tuples/records and function-valued payloads supported by the contract |
| Formation and effects | Ordinary bindings and arguments preserve shared deferred identity; unused arguments and their effects remain deferred. Demands and explicit sequencing establish required effect order; short circuit prevents later demand; aliases do not replay initializers |
| Capture identity | Immutable captures retain original shared deferred binding identities without forcing formation snapshots; referenced mutable storage remains shared; separate formations of one implementation retain distinct environment instances |
| Storage | Scope, caller/region and program lifetimes; actual backing allocations, views and aliases; initialization, overwrite, release and peak live capacity; declared immutable/mutable authority |
| Numeric meaning | Dimensions survive specialization and layout; ranges cover intermediate computations and stores; representation and adaptation come from the selected declarations |
| Data/control boundaries | Empty/singleton/multiple values, branch joins, nested/recursive uses, successful-current/nonempty guards, exhaustion and independent enumeration |
| Refusal | Contradicted facts, missing premises and unsupported forms have distinct responsible diagnostics; neighboring valid cases remain accepted |

Mutable capture tests distinguish a scalar value from a reference to its cell.
An origin identifies an allocation site; forwarding must preserve the actual
runtime activation's storage. A retained descriptor does not extend its backing
lifetime. An immutable capture can still refer to mutable storage.

## 4. Evidence required to close a row

| Boundary | Acceptance evidence |
|---|---|
| CCS source admission | Positive cases assert the inferred source types/dimensions and contain no reachable error nodes. Negative cases require the owning phase, effective error severity, expected diagnostic identity and exact source span. A parser crash, unrelated error or unreachable finding cannot satisfy a checker rejection. |
| Baker graph construction | Inspect types, ordered operand occurrences, binding/capture identities, generated formals, evaluation/control relations, placement and applicable obligations. Nanopass fold-in preserves origins and all proof participants; affected facts are rebuilt or invalidated before renewed settlement. |
| Settlement and proofs | Required properties are discharged by their owning rules before commitment. Record actual checking outcomes and premises. Changed/missing participants, contradicted bounds and stale evidence must not be reported as valid. Finite incidence alone is not a lifetime or termination proof. |
| Alex witnessing | Witnesses consume settled facts through actual ctx/Huet positions and compose Patterns/Elements. Exercise shared occurrences, lambda/match/control-region traversal and scoped operand/block-argument recall, including the distinction between a deliberate graph-root entry and loss of structural ancestry. Component tests reject missing prerequisites; no semantic repair or backend circuit algorithm is introduced. The real serializer and MLIR verifier accept the resulting portable physical forms. |
| Backend realization | Admitted lowering accepts those forms and preserves their claims under the selected target contracts. Altered layout/operand/representation artifacts exercise affected correspondence checks at their owning boundary. |
| Native behavior | Fresh source compilation produces the executed artifact; require successful exit and exact values/ordered effects/demand. Retained aggregate and environment tests observe values after subsequent calls, pulls or returns that could invalidate storage. |
| Editor and clients | CCS.Editor, analyzer projection and actual LSP retain public types, original definitions, located errors, invalidation and unsaved repair for affected constructs. Record unaffected/inactive clients explicitly. |
| Integration | Relevant foundation/C controls and selected BAREWire/Platform consumers pass on the final coordinated source/target cohort. Record known unrelated failures without treating the overall run as green. |

Each capability leaves four reviewable artifacts: source cases, an inspectable
graph with obligations/provenance, the realized artifact with correspondence,
and actual gate results tied to compiler/dependency inputs. Supported obligations
are derived automatically; optional proof presentation does not control checking.
The [proof-composition architecture](../Proof_Composition_Architecture.md) owns
the wider evidence interfaces; new theorem-library integrations are separate
admissions, not implicit prerequisites for every C change.

Existing valid oracles remain unchanged. If an oracle contradicts the normative
contract, document the old expectation, governing clause and replacement before
changing it. C-05's second-force output is a known such migration. An implementation
failure alone never justifies weakening an expected value or skipping a case.

Keep failure attribution distinct: an unsettled Baker prerequisite, absent Alex
coverage for a settled form, and failed component/artifact correspondence are
different conditions. Preserve their source/graph origins and responsible
boundary rather than reporting each as a source type error.

### 4.1 Integrity through realization: Colibri, FPGA and eBPF

The [FPGA workstream](../fpga-targeting/README.md) and
[eBPF workstream](../ebpf-targeting/README.md) are design inputs to these C-series
criteria now. Their realization work makes the same integrity requirement
concrete at circuit and verified-machine boundaries: preserve the program's
meaning and its justification through the delivered implementation. Additional
target deployment remains separately gated; the preservation discipline applies
to Baker and Alex on every admitted path.

This is a directional guide to integrity at each boundary, under
[the ownership contract](#11-baker-construction-alex-witnessing-and-backend-realization).
The C-series adopts the discipline of justified composition and checked
correspondence. FPGA scheduling, circuit selection, handshake/buffer insertion,
HDL generation and physical mapping remain backend work. Their existence does
not authorize a circuit compiler inside Alex or circuit-specific elaboration in
the general C-series witness vocabulary.

Colibri contributes a maintained circuit basis with implementation sources,
self-checking simulations, properties and formal tasks. The
[component admission contract](../fpga-targeting/02_colibri_circuit_basis.md)
connects the selected revision and parameters to semantics, initial state,
transitions, observations and environmental premises. This is instructive for
the C-series' reusable ingredients, recipes and witness forms: a component's
local correctness must compose with the actual operands, storage, control and
target facts at its use site.

| Target lesson | C-series requirement at Baker/Alex and later lowering |
|---|---|
| A controlled component basis prevents an unreviewed circuit fallback | Name the admitted graph protocol, physical form, implementation/profile identity and prerequisites. Missing coverage remains a located failure; a convenient alternate emitter cannot supply missing semantics. |
| Correct components can still be wired incorrectly | Check exact callee/environment, iterator/current, capture/cell, allocation/view and argument/result incidence. Independently correct recipes are insufficient if composition substitutes an environment or drops a participant. |
| Elastic circuits preserve accepted transaction identity, order and multiplicity | Preserve actual formation, demand, callback and yield observations. Tests detect duplicated arguments, extra pulls, lost values and effects moved across deferred boundaries. Progress requires its own premises; trace safety alone does not establish it. |
| Width, reset, depth, buffering and clock changes can invalidate circuit proofs | Width/representation, initialization, frame extent, storage authority, call structure and backend transformation changes invalidate affected evidence. Preserve the original source and selected declaration identities, not only their printed values. |
| Physical memory inference and mapped resources need evidence beyond behavioral similarity | Validate actual allocation, layout, bounds, capacity and release realization. Equal output from one run does not justify a changed storage budget or lifetime policy. |
| eBPF separates source safety, bytecode legibility and host admission | Distinguish Baker settlement, Alex/witness coverage, structural MLIR validity, final-artifact preservation and execution/admission. Success at one boundary cannot waive another. |

The [FPGA artifact-verification contract](../fpga-targeting/05_artifact_verification.md)
requires independently recovering the emitted or reconstructed implementation.
Apply that rule to each C property claimed as preserved: check the actual
serialized/lowered artifact or use an admitted preservation result for the
transform. Generating a program and a second model from the same pre-emission
graph does not test whether the program was emitted correctly. Matching proof
identifiers and hashes bind evidence to inputs; they do not prove the relation.

Record a preservation entry for each affected lowering edge: source/target
subjects, observable relation, premises, admitted transform or artifact check,
checker/trust dependencies, artifact identity and invalidation keys. Mutation
controls should change a real relevant fact: a closure field offset, call
signature, lazy initialization/store, successful-current guard, recurrence
adaptation or selected storage declaration. The dependent claim must fail or
become unresolved at the responsible boundary. A blanket source solver rerun
against unchanged premises cannot substitute for this check.

On the [eBPF path](../ebpf-targeting/01_verifier_as_design_time_contract.md),
register allocation, spills and relocation can affect final guards, stack use
and admission even after source settlement. The pinned host verifier remains an
independent gate; its acceptance does not prove application payload semantics.
On the FPGA path the chain continues through emitted HDL, mapped primitives,
configured routing/bitstream and board observation, with a separately checked
relation and explicit physical assumptions at each claimed level. This is how
the integrity requirement reaches the silicon realization rather than stopping
at an attractive intermediate representation.

Colibri's component/configuration-specific evidence is retained at its actual
strength. Simulation, bounded checking, inductive proof, certificate replay and
kernel checking are distinct results. The existing local Colibri assets and
target plans do not establish Composer's planned complete circuit proof chain;
the C-series imports their concrete admission and correspondence discipline.
Planned integrations must use the shared proof service and source-linked CCS
projection; this requirement does not claim completed target integration.

### 4.2 Scope-aware design-time nanopass continuation

C-series work must preserve a path toward incremental Baker elaboration and
saturation. Follow the
[nanopass incremental direction](../Nanopass_Incremental_Contract_Direction.md)
without prematurely fixing its pass protocol or claiming selective recompilation
already exists. Current fan-out uses a shared fresh-ID allocator; fold-in rebuilds
the graph and invalidates analyses. Those mechanisms are the baseline to evolve.

The scope of a design-time change is its affected semantic dependency region.
Lexical scope and the zipper path contribute context; cross-scope bindings,
captures, aliases, calls, storage, effects and joint proof participants can carry
the change farther. A subtree boundary alone cannot justify retaining a fact.
Each new recipe or owning analysis must expose the identities, origins and
premises needed to determine that region rather than hiding them in an emitter.

As selective reuse enters a C-series slice, its acceptance includes these cases:

| Change or boundary | Required observation |
|---|---|
| Binding, argument, capture or rule changes, including newly present alternatives | Retain the actual dependency footprint, including lookup scope and relevant absence/selection facts. Invalidate affected elaborations, facts and proof supports. |
| A fact crosses a region boundary or has several independent supports | Preserve ordered joint participants and alternative derivations. Stop propagation only when the exported interface remains equivalent for its consumers and their supporting evidence remains valid. |
| Recursive dependencies or a withdrawn premise | Re-establish the owning domains' admitted fixed points across cycles and joins; retract unsupported conclusions. An unchanged output value alone is insufficient. |
| Work resumes after a newer edit or changed target/rule selection | Validate snapshot, occurrence/path, query and premise identities. Stale completion cannot publish into the current graph; zipper navigation alone does not establish freshness. |
| Incremental result reaches the witness boundary | Compare its settled graph, readiness and located diagnostics with a fresh check of the same inputs, allowing valid identity renaming with retained correspondence. Then check realized artifact/behavior equivalence separately. |
| Reachability or segmented-artifact reuse changes | Exercise supported live/dead transitions, last-root deletion and detached cycles. Revalidate segment interfaces and realization dependencies; retire stale symbols/objects and publish a compatible artifact/evidence manifest. |

The [segmented MLIR/object/ELF direction](../Nanopass_Incremental_Contract_Direction.md#22-settled-regions-through-segmented-mlir-objects-and-elf)
keeps dependency regions, Alex occurrence contexts and backend object boundaries
distinct. A compact reachable-only diagnostic dump is a serialization view; it
does not establish a reusable semantic region. Fresh-check equivalence supplements
the independent specification, admission and native/artifact gates above.

Use the C forms themselves as acceptance pressure: shadowed or shared callable
occurrences, changed staged operands and captures, a recursive-group edit, changed
collection storage authority, a lazy capture edit, and changed sequence demand or
current-read premises. Observe both the affected dependents and the justification
for retained unaffected work. Measure recomputation and latency after correctness;
a full-check fallback remains valid while selective reuse is unimplemented.
This compiler infrastructure direction is distinct from implementing the Clef
runtime Incremental library and does not move Baker work into Alex or the editor.

## 5. Dependency-led continuation

The next slices extend existing implementations. This order expresses shared
prerequisites; it does not block an independent correction whose premises exist.

| Slice | Work and first gate |
|---|---|
| Current baseline | Build the coordinated compiler; resolve project/platform inputs; run the relevant original and lettered samples and owning suites. Register missing collection oracles. Preserve failures and classify their first incorrect fact. |
| C-01/C-02 callable composition | Begin with unchanged 16h: retain supplied operands and environments at each partial frontier. Complete returned/retained residence and general function/environment transport, including signatures, calls, returns and joins. Preserve original11/12 and bounded callbacks. |
| C-03 and numeric prerequisites | Close original13's numeric/application gap; validate recursive identity, captures and effects. Add explicitly admitted coupled/multiplicative recurrence evidence for original15, including intermediate values and final stores. |
| C-04 storage and collections | Establish sentinel, arena-relative link, access, extent/capacity and lifetime contracts. Complete List/ranges/supporting array paths, then persistent Map/Set operations and real collection consumers. |
| C-05 lazy | Reuse the shared environment contract, settle cached result storage and force transitions, correct sample14's obsolete oracle and prove once-only normal-force effects. |
| C-06/C-07 remaining composition | Close original15, original16 and 16h; extend aggregate/current retention and child residence, then the explicitly listed successor consumers/materializers. |
| Final acceptance cohort | Run the promised surface and relevant controls against one recorded compiler/dependency/target snapshot; reconcile each PRD, specification discrepancies and waypoint. Only closed gates advance PRD status. |

Trace failures from source checking through Baker settlement, Alex realization
and backend preservation. Fix the first incorrect fact at its owner. Existing
recipes and physical patterns should be reused when their contracts match;
missing semantics are not reconstructed during emission.

The internal mutable-cell signature choice remains in
[Direct Capture Cell Contract](../Direct_Capture_Cell_Contract.md).
Recursive value initialization, lazy reentry/failure and any other genuinely
unspecified behavior need their owning contract resolved before claiming those
forms. They are distinct from already-settled normal lazy memoization or ordinary
recursive function behavior.

## 6. Framework consumers and target boundaries

The [functional surface showcases](../Functional_Surface_Showcases.md) connect
these contracts to HelloWayland's pure frame plan, WrenHello's signal-based
bridge facade, HelloDISCO's bounded model/view, and HelloArty's planned parallel
continuation case. They supply application composition pressure alongside the
enumerated PRD gates. Their UI, reactive, async and hardware extensions retain
separate acceptance; richer source notation alone does not establish them.

[BAREWire's intersection subset](../../../BAREWire/docs/12%20Intersection%20Subset.md)
records accommodations of particular compiler builds. Current array/loop-based
library code supersedes C-04's old Map/Set blocker inventory. Use generic codecs,
schema traversal and selected documented accommodations as consumers; preserve
their bytes, bounded failures and source identities when introducing richer forms.
A library-only check does not establish execution of reachable native paths.
Refresh the native RoundTrip gate rather than copying an older success claim
past the later documented integrity-recheck failures.

[Ariel scoped callbacks](../../../Fidelity.Platform/Environments/Linux/x86_64/Ariel/README.md)
provide a lifetime/callable regression seam. Preserve the declared synchronous
borrow, callback result, retirement and release relationships, with rejection of
escaping borrowed storage. Its native/hardware tests retain their actual
availability and evidence scope; this use does not complete general scheduling.

[Fidelity.Platform's canonical contract](../../../Fidelity.Platform/docs/CANONICAL_PLATFORM_SPEC.md)
and BAREWire declarations supply target capabilities and storage authority.
`ProgramLifetime` names actual immutable and optional mutable spaces. Runtime
initialization requires write authority; a read-only image is not writable
because a source binding is immutable. No compiler-local storage fallback is
introduced by this plan.

## 7. Superseded criteria and retained history

| Earlier framing | Current disposition |
|---|---|
| C features described as absent because an old checklist was unchecked | Preserve source inventory and dated acceptance; repair the specific missing composition or representation contract. |
| Packed callable addresses and cast repair as the target closure form | Retired interior design. Complete canonical separate function/environment transport; explicit foreign ABI adaptation remains at its declared boundary. |
| Capture/layout/yield reconstruction and operation algorithms in Alex | Superseded by Baker ingredients/recipes, owning analysis and graph settlement; Alex realizes admitted settled forms. |
| Fixed SSA budgets, universal integer carriers or layout from capture counts | Superseded by graph-derived value roles, range/target representation and exact field placement. |
| A default global arena before region support | No implicit authority. Admit storage from actual target declarations and lifetime/capacity evidence. |
| Non-memoizing Lazy described as correct, or all memoization waiting for A-04–A-06 | Superseded by the normative first-force/cache contract; stable scoped/program storage and shared environment work provide the relevant dependencies. |
| The old BAREWire Map/Set blocker list or mandatory vectorization before collections | Historical motivation, not the current critical path. Establish native scalar semantics and storage; admit optimizations through M-01 as demanded. |
| A passing sample, source scheme or solver answer completing a family | Bounded evidence only. Close the enumerated source/representation/effect/residence and preservation gates. |

Removed implementation sketches remain available in repository history. Current
PRDs retain useful contract links and identify what still exists in code; removing
a superseded sketch does not claim its implementation has been migrated.

## 8. Checkpoint record

Every implementation checkpoint records scope, actual source/build inputs,
platform selection, artifact identities, commands/results, changed obligations,
tooling coverage, unresolved failures and the next unsettled contract in
[Language Coverage Waypoints](../Language_Coverage_Waypoints.md). Coordinate shared
compiler outputs; independent investigations and fixtures can proceed in parallel.

This reconciliation changes documentation only. No acceptance count, runtime
result or PRD completion status is advanced by editing these criteria. The next
implementation starts with a fresh baseline and the C-01/C-02 callable gates.
