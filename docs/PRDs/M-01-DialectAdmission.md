# M-01: Dialect Admission and Target Realization

> **Status**: Planned | **Category**: Middle End | **Scope**: Operation families and their named target pathways
> **Planning record**: 2026-09-20. This document adds no executable dialect support.

## 1. Alex's receiving and expression contract

**Alex must receive and faithfully express the full Baker-settled computation
and the information required by its selected backend.** Baker owns the semantic
construction and decisions. Alex's Elements/Patterns/Witnesses provide the
complete receiving vocabulary and its faithful realization through the Huet
zipper. Dialect planning is therefore a question of expressive sufficiency and
information preservation across this handoff.

The handoff comprises executable expression **and** its associated information:
NTU dimensions, numeric representation, shape, layout, ownership, effects,
continuation/scheduling relationships, target constraints, proof identities and
provenance. Each fact needed downstream must have an explicit carrier: a typed
operation/type/attribute, or a correlated graph projection available to that
consumer. An operation stream with disconnected evidence is incomplete.

```text
Baker: settled PSG/hypergraph, selected realization and evidence
    -> Alex Elements / Patterns / Witnesses
    -> admitted dialect expression + correlated graph facts/proof identities
    -> selected backend's realization and preservation checks
```

An MLIR dialect is useful when it preserves an operation or relationship that
the selected pathway needs before a later realization step. Alex must be able to
express the complete admitted form, including nested regions, results, block
arguments and dependencies. A missing receiving form is concrete Elements,
Patterns or Witnesses work; a missing semantic fact belongs to Baker.

This PRD gives the existing [retooling admission discipline](../../../clef/docs/fidelity/phg/Closure_Retooling_Plan.md#7-the-drift-gates--checked-in-ci-not-remembered)
a cross-cutting Composer waypoint. The [language standard](../../../clef-lang-spec/spec/backend-lowering-architecture.md#21-portable-dialects)
governs admission; [Thin Middle End](../Thin_Middle_End_Design.md) governs the
boundary. C/A/T/R and target PRDs supply the actual source cases. This is not a
requirement to implement every candidate dialect before continuing those areas.

## 2. Admission is per expression, platform and witness form

A register entry names the exact operations, types, attributes and forms covered
for a selected platform/backend profile. CPU, GPU, NPU, FPGA, JavaScript, Triton
pathways and the MCU families need their own capability records. The information
is available through declarations, quotations and expressions supplied by
BAREWire, Fidelity.Platform, Fidelity.UI where applicable, and the program,
elaborated into Baker's graph.

**Alex is target-aware.** A Pattern/Witness can match the selected profile and
the graph's settled facts to choose the appropriate MLIR form. This is faithful
observation of the available information. It does not require all targets to
receive the same dialect subset, control-flow form or operation sequence.
For example, an explicit-block `cf` form can suit one profile while another
requires a structured `scf` form. The capability record and graph relationships
make that choice determinate. The selected profile is a proof dependency, not
an incidental string or an architecture-name heuristic.

The admission key is **expression family × platform/backend profile × witness
form**. Record the form's prerequisites, suitability/selection rule, accepted
input/output vocabulary and evidence. Where several forms are valid, a declared
selection policy resolves the choice without inventing new semantic facts.
Having a dialect installed, a type printer, a conversion pass, or one working
operation is insufficient evidence for the rest of that dialect.

| Required field | Admission record |
|---|---|
| Demand | Source expression, owning language/target PRD, and a permanent oracle that needs the form; explain the benefit over the already admitted decomposition |
| Standard | Governing Clef semantics and explicit operation-level extension to the witnessed vocabulary, where needed |
| Baker owner | Ingredients/Recipes and nanopasses that construct and saturate the semantic relationships, target facts and proof requirements used by witness selection |
| Graph contract | Exact nodes, ordered participant roles, joint constraints, declarations, coeffects, source origins and graph revision |
| Evidence | Required proofs or admitted runtime checks, their assumptions, applicable solver/certificate route, and unresolved/failure behavior |
| Alex contract | Typed Elements, composed Patterns and passive Witnesses covering the full admitted expression; exact operands, results, regions, successors, attributes and observed prerequisites |
| Information transport | For every backend-required fact: its graph identity, concrete IR or retained-graph carrier, reader, and preservation through subsequent rewrites |
| Pathway contract | Selected platform/profile, its accepted forms and target-aware witness selection rule, declared pass sequence, output forms and runtime/ABI dependencies |
| Preservation | Correspondence from graph identities and obligations through each rewrite, including splits, fusion, elimination and bufferization |
| Acceptance | Compiler/tool versions and hashes, positive/negative CCS and Alex tests, verifier results, target artifacts and execution/simulation evidence |
| Status | Planned, In-Progress or Complete, with a separate Note describing the exact operation/pathway boundary |

Every newly enabled form needs these fields. The initial register can remain
reviewable documentation and typed existing compiler contracts; this PRD does
not prescribe a new central witness dispatcher or a general plugin framework.

## 3. Candidate intermediate vocabulary

These are demand-driven candidates, not blanket admissions. `index` is already
in the standard baseline. The other rows require an operation-level admission
decision even where implementation fragments already exist.

| Vocabulary | Useful intermediate form | Required Baker-settled information | Admission condition |
|---|---|---|---|
| [`math`](https://mlir.llvm.org/docs/Dialects/MathOps/) | Recognizable nonlinear/transcendental operations over admitted scalar or shaped values | Dimensions, domain/range, numeric representation, accuracy, rounding and allowed algebraic transformations | A numeric/geometric oracle benefits from retaining the operation before target realization. |
| [`affine`](https://mlir.llvm.org/docs/Dialects/Affine/) | Affine iteration domains and index maps | Legal dimensions/symbols, bounds, dependences, effects and the selected iteration transformation | The particular loop/map is proved admissible; an arbitrary FP traversal is not presumed affine. |
| [`vector`](https://mlir.llvm.org/docs/Dialects/Vector/) | Explicit lane operations, transfers and reductions | Lane shape, element representation, masks/tails, access bounds, reduction order and selected realization | A lane-level source/target oracle needs it. A vector type alone establishes neither operation support nor correct storage extent. |
| [`tensor`](https://mlir.llvm.org/docs/Dialects/TensorOps/) | Immutable multidimensional values and shape operations before storage materialization | Rank/extents, element dimensions, value semantics and the ownership/layout/copy requirements of the selected bufferization plan | The shaped-value contract requires this intermediate form; mutable arrays do not automatically become tensors. |
| [`async`](https://mlir.llvm.org/docs/Dialects/AsyncDialect/) | Execution dependencies, tokens and task results for already settled segments | Actor/mailbox behavior, cuts, frame residence, completion, cancellation, scheduler relationships and BAREWire obligations | An A/T oracle requires these operations and their realization preserves the established contract. A runtime lowering must not invent another continuation/frame model or import an undeclared runtime. |
| [`index`](https://mlir.llvm.org/docs/Dialects/IndexOps/) | Index/extent arithmetic without an early machine-word commitment in the operation vocabulary | Coordinate meanings, bounds, conversions and the selected platform's eventual index representation | Complete the exact operations demanded. MLIR dimensions/symbols and `index` values do not carry Clef's NTU physical-dimension algebra. |
| [`cf`](https://mlir.llvm.org/docs/Dialects/ControlFlowDialect/) | Explicit blocks, branches and block arguments for an already settled control graph | Successor relations, values crossing edges, resume order, dominance requirements, failure behavior and target form requirements | Admit direct `cf` for profiles that require or benefit from it; retain/select `scf` for profiles that require structured form. Record when a later SCF-to-CF conversion is the appropriate route instead. |

For `math`, identities over mathematical reals do not alone authorize floating
point reassociation, contraction or approximation. Baker must establish the
applicable numerical contract before a fusion/approximation pass may use it.
Transcendental obligations do not become discharged merely because dimensional
equations or integer bounds are discharged; retain the distinct proof families
and any unresolved obligations.

For `tensor`, [bufferization](https://mlir.llvm.org/docs/Bufferization/) is an
explicit preservation boundary. Allocation, aliasing, copies, residence and
release must realize Baker's settled plan. The pass cannot discover a different
ownership policy and silently substitute it. Deferring physical materialization
does not defer semantic ownership or permission to mutate.

For `async`, later realization into scheduler/BAREWire operations implements
contracts already established in Baker. BAREWire contracts are not created by
an async lowering. The same rule applies to Olivier, Prospero and Ariel.

## 4. Further candidates and distinct target pathways

The [upstream dialect catalog](https://mlir.llvm.org/docs/Dialects/) is a source
of possible vocabulary, not a completion checklist. Additional candidates are
`linalg` for structured contractions/reductions; `sparse_tensor` for a proved
sparse representation; `complex` for admitted complex arithmetic; and `quant`
for quantization contracts. Each needs its own demand, graph facts and gates.
Shape utilities must not become a second NTU/shape solver. `transform`/PDL can
describe compiler rewrites if needed; they are not executable Clef semantics.
`smt` belongs to verification artifacts, separately from executable admission.

Each pathway below needs the complete relevant expression and graph information.
The operation may have a useful intermediate lifetime before its final target
form; admission records where that lifetime starts and ends, and how its evidence
remains available throughout.

| Pathway | Possible realization vocabulary | Required correspondence and acceptance |
|---|---|---|
| CPU/MCU and other LLVM pathways | Admitted portable forms into LLVM/target operations | Integer/index widths, ABI, memory spaces, numerical contracts and native behavior under the selected profile |
| FPGA / [CIRCT](https://circt.llvm.org/docs/Dialects/) | `hw`, `comb`, `seq`, then applicable SystemVerilog/export flow; other CIRCT families only on demonstrated demand | Clock/reset, state transitions, latency/throughput, arithmetic behavior, storage/resource bounds and protocol backpressure from the graph; simulation/equivalence before board deployment |
| Existing GPU pathway | `gpu` and ROCDL for the current AMD path; other GPU families only as separately admitted pathways | Device entry, launch/work partition, address spaces, synchronization, transfer/lifetime and numerical behavior; preserve existing GPU evidence during migration |
| Proposed Triton pathway | An explicitly pinned subset of [`tt`, `ttg` and applicable target dialects](https://triton-lang.org/main/dialects/dialects.html) | Tile/layout, masks, reductions, memory spaces, precision/accumulation and execution contracts from Baker; a verified conversion route and CPU/device comparison |
| NPU / MLIR-AIE | Applicable AIE target operations | Tile placement, routes, bounded FIFO/DMA relationships, synchronization and numerical contracts already present in the graph |
| Other pathways, including SPIR-V, WebAssembly and JSIR | Their declared realization vocabulary | Individual capability and preservation records; a portable operation is not a claim that every pathway currently supports it |

Triton is a proposed kernel realization route, not an implemented Composer path
and not the existing ROCDL path under another name. AI model semantics, automatic
differentiation and training orchestration are separate source/graph work; a
Triton dialect entry does not establish them. There is no assumed universal
`tensor -> linalg -> vector -> Triton` conversion chain. FPGA realization likewise
needs a proved spatial/temporal interpretation, not just a different serializer.

## 5. Numeric selection, parallelism and design-time projection

The governing contracts are already in clef-lang-spec. This PRD connects them
to operation/profile admission; the blog articles supply motivation and oracle
cases, not replacement semantics.

| Concern | Governing contract | Required Alex/backend correspondence |
|---|---|---|
| Representation and arithmetic construction | [Numeric Selection §§9–10.5](../../../clef-lang-spec/spec/numeric-selection.md#9-carriage-and-the-real-interval-domain), [Width Inference](../../../clef-lang-spec/spec/width-inference.md) | Selected representation and its range evidence; primitive and intermediate precision, scale, rounding, exceptional behavior, permitted transformations, construction and capacity evidence |
| Blocking dependencies | [Synchronous RPC and Wait Classification](../../../clef-lang-spec/spec/synchronous-rpc-liveness.md) | Blocking-wait hyperedges, suspended continuation/reply identities, checked ordering, feasibility status and the specified supervised path for unresolved sites |
| Progress and resources | [Scheduler Contract §§3–7](../../../clef-lang-spec/spec/scheduler-contract.md#3-fairness) | Target assumption manifest, fairness and turn discipline, admission capacities, control-plane availability, cancellation and cleanup prerequisites |
| Platform and host realization | [Platform Bindings](../../../clef-lang-spec/spec/platform-bindings.md), [Backend Lowering §4.5](../../../clef-lang-spec/spec/backend-lowering-architecture.md#45-carrier-realization-on-pathways-without-linear-memory) | Selected platform declarations and quotation provenance, memory/transfer capabilities and host value-model requirements, retained through the declared pathway |
| Authoring feedback | [Numeric Selection §11](../../../clef-lang-spec/spec/numeric-selection.md#11-design-time-surfacing-and-accuracy-preservation), RPC diagnostics and scheduler assumption manifest | Source-related, target-specific evidence and unresolved obligations through the shared CCS projection to CAC, Lattice and lattice-analyzers |

### 5.1 From numeric selection to arith/math witnessing

Record the complete chain: **source operation → selected representation →
admitted arithmetic construction → target-eligible witness form → backend
realization**. `arith` and `math` are complementary operation vocabularies;
there is no rule that an NTU real automatically becomes an IEEE operation, or
that every mathematical operation becomes a `math` op. A selected fixed-point,
posit or IEEE construction can require different primitives and intermediate
state. Preserve a recognizable `math` operation where its admitted semantics
and eventual realization satisfy that construction. Otherwise witness its
Baker-settled decomposition using the admitted vocabulary. Never select an
IEEE-shaped op first and retrofit the representation afterward.

Operation eligibility includes intermediate precision, rounding, subnormal and
exceptional behavior, FMA/contraction, vector/matrix semantics and any software
realization allowed by capability policy. The RA6M5 case must use the selected
board's actual available and enabled capabilities, ABI and runtime provisions;
neither an MCU label nor a floating-point-format flag supplies that evidence.
Hardware support and a permitted software construction are distinct facts.
`math` admission must also identify the operation's domain and accuracy evidence;
representation error is not a bound on the composed calculation's error.

The accuracy-only representation objective and capability filters remain those
of Numeric Selection. Cost can rank eligible realizations preserving the chosen
representation and required arithmetic contract (§10.4). An estimated speedup
cannot justify weaker accuracy, changed rounding or a different reduction order.
Fast-math flags and downstream rewrite options are part of preservation review.
The platform fact schema, construction registry and selection procedure that
the standard explicitly leaves open (§14) remain implementation/design work.

### 5.2 Parallel and host execution

[Pondering Fearless Parallelism](https://clef-lang.com/blog/pondering-fearless-parallelism/)
supplies useful paired oracles: retain a specified reduction tree under changing
worker completion order; preserve the initialization, term multiplicity, merge
and finalization laws of an exact construction; and reject insufficient capacity
for any admitted partial or merge. A rounded partial sent over BAREWire cannot
stand in for an exact accumulator unless that conversion is proved exact.
Publication, residence, transfer and progress obligations accompany the numeric
ones. Vectorization must preserve required arithmetic dependencies and any
cross-lane carry semantics.

[Fearless Concurrency Gets Real](https://clef-lang.com/blog/fearless-concurrency-gets-real/)
motivates the wait graph; the current RPC specification governs its precise
classification. A candidate cycle in a may-wait graph is distinct from a proved
feasible blocking cycle. Acyclicity does not establish fairness, turn progress,
external completion or executable timeout recovery. Every proposed async/control
form must preserve those distinctions under its actual target profile, including
bounded mailboxes and control-plane capacity on freestanding devices. Graph
classification occurs in Baker; Alex observes the resulting relationships.

[Carrying Proofs into JavaScript](https://clef-lang.com/blog/carrying-proofs-into-javascript/)
adds host-specific oracle requirements: facts surviving suspension need their
validity conditions; joins need the correct job/input/partition identities and
multiplicities; serialized partials must preserve the arithmetic contract.
Single-threaded callback execution proves neither whole-handler atomicity across
suspension nor eventual completion. Host dispatch guarantees, generated boundary
checks and discharged graph obligations remain separately identified, as the
scheduler's isolate manifest requires. Durable recovery cases must retain the
accepted-contribution state together with the numerical state. These are planned
JSIR acceptance cases, not a claim of implemented distributed proof transport.

### 5.3 Tooling and discriminating acceptance cases

The shared CCS design-time projection must identify the selected profile,
representation, range justification, construction eligibility and missing facts,
with source spans and participating graph identities. Where candidate comparison
is provided, distinguish representation error, computation error, reproducibility
and cost evidence. Retain established/refuted/unresolved status and the target's
assumption manifest. A target or capability change invalidates dependent results
in the editor as well as the compiler. CAC, Lattice and lattice-analyzers consume
this projection; they do not each implement numeric selection or wait analysis.

Add focused positive/negative oracles for a changed numeric capability, forbidden
contraction/reassociation, partial-sum overflow despite a fitting final sum,
unsupported timeout recovery, and a stale or duplicate partition contribution.
Pair CPU/MCU, accelerator and JSIR cases where their contracts differ. Require
matching compiler and unsaved-buffer diagnostics before closing the affected
operation/profile scope. Existing Fidelity.UI and HelloWayland expressions
provide additional display and scheduling triangulation; their working CPU
behavior alone cannot close other target gates.

## 6. Elements / Patterns / Witnesses and preservation gates

Elements represent admitted MLIR operations with typed operands, results and
attributes. Patterns compose those operations from already projected graph facts.
Witnesses remain passive, focused observations through the Huet zipper. The
selected platform/backend is part of that observation: Patterns/Witnesses match
the graph's target coeffects and the profile's form requirements. A target-aware
choice between admitted realizations is part of Alex's job. The graph determines
the semantic decomposition; witnesses do not scan subtrees to discover
semantics or recursively construct an alternative emission plan.

Admission tests must include the same semantic expression under different
profiles, checking the selected form and equivalent required behavior. A `cf`
case must not silently enter a profile requiring `scf`, or vice versa. Missing,
inconsistent or stale target facts produce a precise prerequisite diagnostic;
they do not trigger a guessed default. Target changes invalidate dependent form
selection and preservation evidence.

Consumer gates must inspect typed operations and nested regions, including their
type/attribute restrictions. An operation-prefix check is only one drift check.
Unsupported forms must not disappear into TODO comments, raw text or an
unrealized cast. A missing fact reports the source site and participating graph
identities, with the Baker owner responsible for settlement.

Each rewrite consumes an identified, Baker-admitted transformation and preserves
its premises and conclusions. Changed numerical order, aliasing, memory usage or
execution order requires the corresponding Baker evidence. A verifier-success
flag is not a semantic-preservation proof. The temporary reconciliation ledger
may check correspondence while graph-carried evidence matures; it does not
replace that evidence. Retain graph revision, target declarations, provider and
toolchain versions so a premise change invalidates the dependent evidence.

The gate sequence is source acceptance/rejection, graph settlement and negative
incidence checks, typed Alex observation, stock dialect verification, declared
pass-pipeline verification and target execution/simulation. Compare numerical
results under the stated error contract, not an arbitrary tolerance. Include
wrong dimensions, invalid domains, bad extents/masks, non-affine dependence,
unavailable representations, lost ownership, stale proofs, unsupported profiles,
and missing witness coverage where applicable. Source diagnostics and unresolved
prerequisites must reach CCS editor projections, CAC, Lattice and the analyzers.

FidelityHello variants may provide small permanent oracles. HelloWayland and
Fidelity.UI supply workload triangulation; FPGA/embedded board examples add
deployment evidence when the hardware is available. A hardware-unavailable gate
is recorded as pending. No repeated board deployment is required for a docs-only
planning change.

## 7. Current evidence and reconciliation work

At Composer `1fccb02`, the operation model contains `scf.index_switch` and its
serializer (the September 19 audit predates that addition). `index` has declared
cases that still serialize as TODO comments. `vector` has a type/printing path,
but its size function currently ignores lane count. Neither establishes full
operation-family admission. `math`, `tensor`, `affine` and `async` do not have
complete Alex operation/Element/Pattern/Witness families in this inspection.

`cf.assert` is actually serialized despite the older standard's blanket `cf`
exclusion. FPGA witnesses currently construct CIRCT forms; the NPU kernel
witness constructs raw AIE text. These need explicit stage and information
contracts reconciled with the standard before their implementation patterns are
extended. Preserve their working oracles and target-aware witness selection;
move any semantic reconstruction into Baker and account for target operations
at the declared realization boundary.
The existing GPU backend is an operational ROCDL path; its historical design's
array-admission limitations require fresh checking against current CCS.

The current [backend interface](../../src/Core/Types/Pipeline.fs) accepts MLIR
text and `BackEndContext`; that context has target/link/deployment configuration
but no general graph-fact or obligation-correspondence input. Extending this
handoff is explicit M-01.b work. Its contract must expose the admitted expression,
selected target facts and immutable semantic/proof projections needed by each
consumer, retaining source and graph identities. Merely adding dialect serializers
would leave this part of Alex's receiving/forwarding contract incomplete.

Source evidence: [operation/types](../../src/MiddleEnd/Alex/Dialects/Core/Types.fs),
[serialization](../../src/MiddleEnd/Alex/Dialects/Core/Serialize.fs),
[hardware witness](../../src/MiddleEnd/Alex/Witnesses/HardwareModuleWitness.fs),
[kernel witness](../../src/MiddleEnd/Alex/Witnesses/KernelModuleWitness.fs),
[CIRCT backend](../../src/BackEnd/CIRCT/Pipeline.fs),
[GPU backend](../../src/BackEnd/GPU/Lowering.fs), and
[prior audit](../Clef_Language_Completion_Review_2026-09-19.md#5-alexs-standard-dialect-expansion).
This is a source/document review, not a fresh target run.

## 8. Work packages and exit criteria

| Step | Deliverable | Status | Note |
|---|---|---|---|
| M-01.a | Reconciled baseline operation register and boundary inventory | Planned | Resolve partial index/vector support, direct `cf.assert`, and current FPGA/AIE construction against the standard; retain operational oracles. |
| M-01.b | Complete Alex/backend information transport, typed consumer gates and rewrite correspondence | Planned | Extend the current text/configuration handoff with required immutable Baker fact/proof projections and graph identity; use existing graph/read/verification mechanisms. |
| M-01.c | First demanded mathematical operation family | Planned | Trace numeric selection and arithmetic construction into arith/math forms under §5.1; include target capability negatives and matching design-time projection. |
| M-01.d | Demanded shape/iteration/lane families | Planned | Admit tensor, affine, vector or linalg only where the oracle and Baker contracts warrant them; gate bufferization separately. |
| M-01.e | Demanded asynchronous/block-control forms | Planned | Follow A/T, RPC wait and scheduler contracts; validate progress/resource prerequisites per target and decide where async/direct cf are needed. |
| M-01.f | Target realization reconciliation and extensions | Planned | Preserve FPGA/AIE/GPU behavior while restoring the boundary; Triton and further routes need independent operation/profile gates. |

The letters are traceable work packages, not a mandatory serial climb. A family
closes only with its named operation/profile scope, matching standard, graph
contracts, tooling projections and passing gates in the language coverage
waypoints. Deferring a candidate with a recorded reason is a valid admission
decision. It does not claim implementation or permanently exclude that dialect.

## 9. Resuming implementation across repositories

Read the [coverage waypoints](../Language_Coverage_Waypoints.md), this PRD's
§5 contract map, the linked standard chapters and the owning feature PRD before
selecting a batch. Earlier PRD pseudocode and blog illustrations do not override
those contracts. In particular, inherited LLVM-coroutine or subtree-emission
sketches are not an implementation template for Baker/PSG and passive Alex.

The first M-01 batch should pair **M-01.a and the demanded part of M-01.b** with
one concrete expression/profile oracle. Inspect its graph facts, typed operation
model, witness selection and backend inputs together. Record which facts already
exist and which require Baker or platform declaration work. Establish the carrier
and negative prerequisite gate before expanding its operation vocabulary. The
source findings in §7 identify starting points, not a mandate to enable every
candidate or repair every target in one batch. This planning record does not
close or reorder the outstanding C-series/F-06 acceptance gates.

| Repository/component | Coordinated implementation responsibility |
|---|---|
| clef-lang-spec | Governing semantics, capability and preservation contracts; resolve explicitly open mechanisms as evidence warrants, without declaring implementation complete |
| clef / CCS / Baker | Ingredients/Recipes, elaboration and saturation nanopasses; joint constraint construction, proof dispatch, target facts and shared editor projection |
| Composer / Alex / backends | Typed Elements/Patterns/Witnesses, target-aware form selection, complete backend handoff, rewrite correspondence and source-to-artifact oracles |
| Fidelity.Platform and BAREWire | Operation-specific platform declarations, runtime/host prerequisites, storage/publication/transfer contracts consumed in Baker; update providers when a demanded fact is missing |
| Fidelity.UI and HelloWayland | Shared display expressions and target requirements; piped FP/CE and serial/parallel rendering comparisons, preserving the distinction between design contract and working sketch |
| CAC, Lattice and lattice-analyzers | Consume versioned CCS projections for the selected target; keep source diagnostics, related proof participants and unsaved-buffer invalidation aligned |

For each feature-sized batch, record the admitted operation/profile scope,
positive and discriminating negative results, remaining gates, and coordinated
repository revisions in the waypoints. Run the tests affected by that batch;
do not repeat unchanged full suites for each internal step. Hardware-dependent
acceptance remains explicitly pending when no device is available. Commit and
push the completed batch across affected repositories, retaining existing
operational oracles. No companion source changes are implied by this docs-only
planning synchronization.
