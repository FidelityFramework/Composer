# Composer PRDs - Master Index

> **Purpose**: Category-prefixed PRD organization enabling modular growth across platform targets

Status reconciliation, 2026-09-20: operational capability, PRD completion and
regression evidence are recorded separately. The foundation is implemented and
operational within the scope below; **Retrospective** describes how its documents
were written, not an unfinished implementation. The
[language coverage record](../Language_Coverage_Waypoints.md) carries the tested
compiler/tooling revisions and unresolved gates. An absent rerun at the latest
checkpoint does not revoke an established operational baseline; an observed
regression is identified explicitly. Every roadmap row uses **Planned**,
**In-Progress** or **Complete**, with scope and evidence in its Note column.
The original numbering guides progression; current dependency evidence determines
the work order. Later composition can reopen an earlier area: a foundation sample
may be complete while returned closures, collection storage or continuation
lifetimes still require work in the computation PRDs.

---

## Category Overview

| Prefix | Category | Description | PRD Range |
|--------|----------|-------------|-----------|
| **F-xx** | Foundation | Core compilation, Samples 01-10 | F-00 to F-10 |
| **C-xx** | Computation | Closures, HOFs, Lazy, Seq | C-01 to C-07 |
| **A-xx** | Async | Async, Await, Regions | A-01 to A-06 |
| **I-xx** | IO | Sockets, WebSocket | I-01 to I-02 |
| **D-xx** | Desktop | GTK, WebView | D-01 to D-02 |
| **T-xx** | Threading | Threads, Mutex, Actors | T-01 to T-05 |
| **R-xx** | Reactive | Observable, Rx operators, Incremental | R-01 to R-06 |
| **E-xx** | Embedded | USB, RTOS, bounded native UI | Future |
| **M-xx** | Middle End | Alex expression coverage, dialect admission and backend information transport | M-01 |

---

## Target Requirements Matrix

Not all PRDs apply to all targets. This matrix clarifies which features are needed for which platform configurations.

| PRD Category | WREN Stack | QuantumCredential | Embedded UI | Unikernel |
|--------------|------------|-------------------|----------|-----------|
| **Foundation (F)** | Required | Required | Required | Required |
| **Computation (C)** | Required | Required | Required | Required |
| **Async (A)** | Required | Partial (A-01,02) | Partial | Required |
| **IO (I)** | Required | Partial | N/A | Required |
| **Desktop (D)** | Required | N/A | N/A | N/A |
| **Threading (T)** | Required | Partial | N/A | Required |
| **Reactive (R)** | Required | Partial | Partial | Optional |
| **Embedded (E)** | N/A | Required | Required | N/A |
| **Middle End (M)** | Required | Required | Required | Required |

Embedded UI is a deployment profile of the portable Fidelity.UI contract. The
primary native direction is a Clef reactive-area engine, with shared component
semantics across native and DOM/WebView realizations. An optional Farscape binding
can adapt LVGL to that interface; LVGL does not define a compiler target. The
[Fidelity.UI architecture](../../../Fidelity.UI/docs/00_architecture.md) records
the design and current implementation boundary.

---

## Complete PRD List

### Foundation (F-xx) - Core Compilation

The latest recorded foundation-wide native run is the
[C-06 regression checkpoint](../Language_Coverage_Waypoints.md#c-06-native-continuation-settlement--2026-09-20)
on CCS `08d54752…84482`: 01–04, 07–10 and every 08a–e / 09a–c variant compiled,
ran and matched expected output. Sample 05's formatter regression is now closed
by the [character-storage acceptance](../Language_Coverage_Waypoints.md#f-05-character-storage-and-native-formatting--2026-09-20);
06 retains the specific regression below. The [regression manifest](../../tests/regression/Manifest.toml) retains
their original acceptance programs and output; `/tmp/composer-c06-final-regression.log`
records that run. C-07's later focused gates do not constitute a new all-foundation
run.

| PRD | Title | Sample | Status | Note |
|-----|-------|--------|--------------------|--------------------------------------------|
| [F-00](F-00-Synopsis.md) | Foundation Series Synopsis | 01–10 | Complete | Retrospective overview of the implemented foundation. Its old pending entry for Records is superseded by the passing 10 native gate; F-06 retains the reopened parsing regression below. |
| [F-01](F-01-HelloWorldDirect.md) | HelloWorldDirect | 01 | Complete | Native pipeline, static strings and console output are operational; 01 passes. |
| [F-02](F-02-ArenaAllocation.md) | Arena Allocation | 02 | Complete | The PRD's input/string-allocation sample passes. Its explicit future arena/region extension belongs to A-04; this status does not close that contract. |
| [F-03](F-03-PipeOperators.md) | Pipe Operators | 03 | Complete | Pipe normalization and function application are operational; 03 passes. |
| [F-04](F-04-CurryingLambdas.md) | Currying & Lambdas | 04 | Complete | Curried calls, lambdas and the sample's partial application pass. Broader returned/retained callable environments remain C-01/C-02 work. |
| [F-05](F-05-DiscriminatedUnions.md) | Discriminated Unions | 05 | Complete | Original 05 again passes stock MLIR and exact native output after Baker settles integer/UTF-8 storage and snapshot ownership; the separate 22-case formatter and encoding oracle also passes. |
| [F-06](F-06-InteractiveParsing.md) | Interactive Parsing | 06 | In-Progress | Implemented interactive parsing/mixed numeric DU baseline; original 06 has a recorded source-admission regression on three legacy `int` conversion calls in platform Parse (`CCS8009`). |
| [F-07](F-07-BitwiseOperators.md) | Bitwise Operators | 07 | Complete | AND/OR/XOR/complement/shifts, comparisons and Boolean composition pass. The surface is native operators, not the retired `Bits.*` byte-order/bitcast API. |
| [F-08](F-08-OptionType.md) | Option Type | 08, 08a–e | Complete | Some/None and matching, plus tested defaults, alternatives, iteration and folds; all six samples pass. Wider collection/callable contracts remain separately scoped. |
| [F-09](F-09-ResultType.md) | Result Type | 09, 09a–c | Complete | Ok/Error and matching, map/mapError/bind, defaults, iteration and predicates; all four samples pass. Historical unchecked `get`/`getError` sketches are not admitted native operations. |
| [F-10](F-10-RecordTypes.md) | Record Types | 10 | Complete | Construction, field access, copy/update, nested records and guarded/nested/wildcard record patterns are operational; 10 passes with exact output. |

The foundation PRDs mostly document completed work retrospectively. Their old
fixed-width layouts, allocation sketches and intermediate closure representations
do not override current Clef specifications or Baker's settled graph contracts.
F-06 also retains the documented platform input-buffering limitation: the
regression harness supplies separate lines with a pause; buffered multi-line
input is a separate `Console.readln` behavior issue
([recorded detail](../Surface_Gaps_2026-09.md#samples-06-11-12-and-13)).

### Computation (C-xx) - Functional Abstractions

The foundation has enabled substantial computation support. C-01 and C-02 have
expanded/reopened as C-06/C-07 demand retained environments and stored operation
values. Residence and aggregate composition also cross these boundaries. Each
computation PRD remains In-Progress until its own acceptance gates are satisfied;
newer passing samples establish their bounded paths.

| PRD | Title | Sample | Status | Note |
|-----|-------|--------|--------|------|
| [C-01](C-01-Closures.md) | MLKit-Style Flat Closures | 11 | In-Progress | Bounded native environments tested; returned/retained callable storage and broader residence remain open. |
| [C-02](C-02-HigherOrderFunctions.md) | Higher-Order Functions | 12 | In-Progress | Native callback paths tested; stored/bare Seq operation partials expose remaining callable admission work. |
| [C-03](C-03-Recursion.md) | Recursion & Tail Calls | 13 | In-Progress | Implementation exists; original 13 has a recorded generic integer-width failure. Full PRD acceptance is not established by C-07. |
| [C-04](C-04-CoreCollections.md) | Core Collections | 13a | In-Progress | Option operations are tested; general collection storage, bounded extent and native gates remain open. |
| [C-05](C-05-Lazy.md) | Lazy Evaluation | 14 | In-Progress | Implementation exists; original 14 has a recorded width/extent failure and canonical lazy acceptance remains open. |
| [C-06](C-06-SimpleSeq.md) | Simple Sequences | 15, 15a–d | In-Progress | Native core and bounded scalar/Option transport tested; original recurrence/aggregate and broader residence gates remain separate. |
| [C-07](C-07-SeqOperations.md) | Sequence Operations | 16, 16a–h | In-Progress | 16a–g pass; original 16 and 16h retain factory/capture and staged callable failures. Full operation coverage remains open; see waypoint. |

### Async (A-xx) - Asynchronous Programming

Before advancing this family, reconcile the inherited implementation sketches
with the Clef CE/delimited-continuation contracts. C-07 does not establish actor
scheduling, async lifetime admission, or a new MLIR dialect's preservation gates.

| PRD | Title | Sample | Status | Note |
|-----|-------|--------|--------|------|
| [A-01](A-01-BasicAsync.md) | Basic Async | 17 | Planned | Reconcile CE admission and deferred execution before the native gate. |
| [A-02](A-02-AsyncAwait.md) | Async/Await | 18 | Planned | Requires settled suspension/resumption and lifetime contracts. |
| [A-03](A-03-AsyncParallel.md) | Async Parallel | 19 | Planned | Parallel orchestration and its acceptance gate remain ahead. |
| [A-04](A-04-BasicRegion.md) | Basic Regions | 20 | Planned | General explicit region/arena contract; F-02's input sample does not close it. |
| [A-05](A-05-RegionPassing.md) | Region Passing | 21 | Planned | Interprocedural region ownership and passing gates remain ahead. |
| [A-06](A-06-RegionEscape.md) | Region Escape Analysis | 22 | Planned | General region escape acceptance extends beyond bounded current residence proofs. |

### Next implementation handoff

Read the current [coverage waypoints](../Language_Coverage_Waypoints.md) and
[M-01 contract map](M-01-DialectAdmission.md#5-numeric-selection-parallelism-and-design-time-projection)
with the linked clef-lang-spec chapters before using inherited implementation
sketches. The standard governs semantics; this index records acceptance status;
M-01 records target-aware expression and information-preservation gates.

| Area | Concrete entry condition and first gate |
|------|-----------------------------------------|
| C-01 / C-02 | Reuse staged operand snapshots for Seq partial/bare values; admit retained sequence and callable environments with exact residence/use evidence. `16h_SequenceApplications` is the unchanged native gate. |
| C-04 with remaining C-07 consumers | Establish collection storage, bounded links, extent and current/nonempty contracts before claiming `toList`, `toArray` or extrema support. Use the existing BAREWire collection contracts and native oracles. |
| C-05 | Reconcile the inherited lazy implementation with the canonical closure/thunk contract before treating it as a foundation for native `Incremental<'T>`. |
| A / T families and added dialects | Follow [M-01](M-01-DialectAdmission.md): numeric selection and construction govern arith/math forms; RPC wait relationships and scheduler manifests govern async/control forms. Alex selects witnesses from settled facts for the actual target. Require graph, backend and design-time gates for each admitted operation/profile. |
| R-04 first within Reactive, with R-01/R-02 | After the preceding Async/Threading work, lead with the static incremental core on C-01/C-05: tracked inputs, cached `return`/`map`/`map2`, demand and cutoff. Develop typed Observable delivery and matched operators alongside it, with shared versioned invalidation and an event-to-cache native gate. |
| R-05 with R-03/R-06 | Add dynamic dependency replacement and child lifetimes alongside the corresponding Observable bridges. Gate independent invalidations, ordered effects, stale work, demand withdrawal and disposal before extending actor/target integration. |

### IO (I-xx) - Network & File I/O

| PRD | Title | Sample | Status | Note |
|-----|-------|--------|--------|------|
| [I-01](I-01-SocketBasics.md) | Socket Basics | 23 | Planned | PRD socket/resource acceptance remains ahead of this language checkpoint. |
| [I-02](I-02-WebSocketEcho.md) | WebSocket Echo | 24 | Planned | Protocol composition follows the required IO and lifetime contracts. |

### Desktop (D-xx) - Desktop Applications

| PRD | Title | Sample | Status | Note |
|-----|-------|--------|--------|------|
| [D-01](D-01-GTKWindow.md) | GTK Window | 25 | Planned | This GTK PRD's callback/resource gates remain ahead. |
| [D-02](D-02-WebViewBasic.md) | WebView Basic | 26 | Planned | This WebView integration PRD remains roadmap work. |

### Threading (T-xx) - Concurrency

Async and Threading precede the Reactive family in the intended progression.
Existing [HelloWayland Ariel CPU acceptance](../../../HelloWayland/docs/multi-core-cpu.md)
and [typed carrier gates](../../../HelloWayland/tests/ariel-typed/README.md)
describe a working implementation sketch that predates full actor-model
expression. Native serial/parallel rendering, active worker threads, resize and
normal close/join are recorded as passing; the project author also reports
saturation of all 32 CPU threads. These results supply workload and regression
evidence. The sketch is not authoritative for the actor or scheduler contracts
and does not close the A/T PRDs or establish automatic capture/access extraction
for arbitrary dispatch.

| PRD | Title | Sample | Status | Note |
|-----|-------|--------|--------|------|
| [T-01](T-01-BasicThread.md) | Basic Threading | 27 | Planned | PRD-level thread ownership and acceptance; existing platform parallelism is a separate baseline. |
| [T-02](T-02-MutexSync.md) | Mutex Synchronization | 28 | Planned | Scoped synchronization and resource gates remain ahead. |
| [T-03](T-03-BasicActor.md) | Basic Actor | 29 | Planned | Actor ownership, scheduling and protocol contracts need reconciliation. |
| [T-04](T-04-ActorReply.md) | Actor Reply | 30 | Planned | Request/reply lifetime and protocol acceptance remain ahead. |
| [T-05](T-05-ParallelActors.md) | Parallel Actors | 31 | Planned | Builds on admitted actor and synchronization contracts. |

### Reactive (R-xx) - Reactive Extensions

**Within Reactive, the priority is Incremental-first, developed together with
Observable.** This does not move Reactive ahead of the preceding Async/Threading
work. Begin R-04's static core from the C-01 closure and C-05 lazy/thunk basis,
with tracked reads, cached values, equality/cutoff and proven storage lifetimes.
R-01 typed delivery and subscription ownership, then matched R-02 operators,
develop alongside that core. Numeric order does not require completing the
Observable family before starting Incremental. R-05 adds dynamic dependency
replacement and child lifetimes; R-03/R-06 develop the corresponding bridges
and demand/disposal contracts together.

The shared acceptance work must retain source and version identities for
invalidation, including all relevant read/effect dependencies. At a join, cutoff
on one path must preserve another path's independent invalidation. Event delivery
and required effects retain their order; fusion cannot silently discard emissions.
Demand withdrawal, subscription disposal, detached subgraphs and outstanding or
stale work need explicit ownership rules before reclamation. These are semantic
requirements, not a mandated common runtime representation or an actor per node.
The current [Incremental](../../../clef-lang-spec/spec/incremental-computation.md)
and [Observable](../../../clef-lang-spec/spec/observable-computation.md) contracts
govern reconciliation of the inherited PRD sketches. All six rows remain Planned.

| PRD | Title | Sample | Status | Note |
|-----|-------|--------|--------|------|
| [R-01](R-01-ObservableFoundations.md) | Observable Foundations | 32 | Planned | Develop typed delivery/subscription ownership alongside the R-04 core. |
| [R-02](R-02-ObservableOperators.md) | Observable Operators | 33 | Planned | Develop matched operators with incremental use cases; preserve event order and effects through composition. |
| [R-03](R-03-ObservableIntegration.md) | Observable Integration | 34 | Planned | Pair bridges with R-05/R-06 demand and lifetime contracts; actor/IO protocols remain additional prerequisites. |
| [R-04](R-04-IncrementalFoundations.md) | Incremental Foundations | 35 | Planned | Lead the family: C-01/C-05-based static core, tracked invalidation, cached values and independent-cause cutoff. |
| [R-05](R-05-IncrementalDynamism.md) | Incremental Dynamism | 36 | Planned | Extend R-04 with dynamic dependencies, replacement and child lifetimes; no reclamation implied by demand loss alone. |
| [R-06](R-06-IncrementalIntegration.md) | Incremental Integration | 37 | Planned | Co-develop Observable invalidation/post-cutoff bridges and disposal; actor and cross-target consistency need explicit later gates. |

### Embedded (E-xx) - MCU & Unikernel

The UI work follows [Fidelity.UI's native portable model](../../../Fidelity.UI/README.md):
owned reactive areas, component composition and explicit capabilities, with
bounded storage, work queues and display completion on embedded profiles.
[HelloWayland](../../../HelloWayland/README.md) supplies a working experimental
renderer oracle, including native presentation, parallel raster work, resize and
teardown. Its ad hoc rendering path is useful acceptance evidence; the general
Fidelity.UI engine and portable component gates remain planned. LVGL remains a
component/resource reference and an optional Farscape interoperability adapter.

| PRD | Title | Sample | Status | Note |
|-----|-------|--------|--------|------|
| E-01 | USB Device Stack | Future | Planned | Future scope; PRD not yet specified. |
| E-02 | RTOS Integration | Future | Planned | Future scope; PRD not yet specified. |
| E-03 | Native Embedded UI | Future | Planned | Fidelity.UI shared semantics and native reactive-area engine under a bounded device profile; optional Farscape/LVGL adapter. PRD and engine acceptance are not yet complete. |

### Middle End (M-xx) - Expression and Information Preservation

Alex must receive the complete Baker-settled expression and carry every fact
needed by the selected backend. Dialect admission is per expression family,
selected platform/backend profile and witness form, with explicit target-aware
Elements/Patterns/Witnesses coverage and information transport. Numeric selection,
arithmetic construction, wait-for relationships and scheduler capabilities are
governing inputs, including in design-time tooling. Candidate dialects are evaluated as language and target work demands
them; they are not a mandatory list or a new serial phase before C/A/T/R work.

| PRD | Title | Sample | Status | Note |
|-----|-------|--------|--------|------|
| [M-01](M-01-DialectAdmission.md) | Dialect Admission and Target Realization | Per operation/pathway oracle | Planned | Complete Alex's receiving/forwarding contract for Baker-settled expression, graph facts and proofs; consider math, affine, vector, tensor, async and cf, complete demanded index operations, and distinguish CIRCT, existing GPU/AIE and proposed Triton pathways. |

### Interactive compiler workbench

The [workbench and native REPL bridge](../Interactive_Compiler_Workbench.md)
is a Composer-owned development workstream alongside language completion.
SageFS is the first bootstrap-host candidate. The initial pilot can run alongside
C-01–C-07; it does not wait for full self-hosting or advance those PRDs by itself.
CCS/Baker retain semantic authority, Alex retains ctx pull witnessing, and
Lattice/MCP clients share the existing proof and session service. WB identifiers
are workstream milestones, not another language PRD category.

| Milestone | Scope | Status | Note |
|---|---|---|---|
| [WB-01](../Interactive_Compiler_Workbench.md#pilot-acceptance) | SageFS compiler-workbench pilot | Planned | Evaluate a pinned host on a real coverage case; require rejection controls, saved-source replay, actual owning gates and measured feedback costs. No adapter is implemented by this plan. |
| [WB-02](../Interactive_Compiler_Workbench.md#proof-responsiveness) | Shared session and responsive proofs | Planned | Reuse the CCS editor/proof service; establish compiler-generation identity, shared-client correctness and measured cvc5 scheduling/reuse. |
| [WB-03](../Interactive_Compiler_Workbench.md#milestones) | Native interactive execution | Planned | Connect the ordinary lowering to ORC/JITLink with explicit invocation/lifetime contracts and JIT/AOT acceptance for a named slice. |
| [WB-04](../Interactive_Compiler_Workbench.md#milestones) | Clef-facing clients and host transition | Planned | Extend Lattice clients and preserve the session contract when bootstrap-specific hosting is replaced; notebooks remain separately scoped. |

### FPGA targeting and artifact verification

The dedicated [FPGA series](../fpga-targeting/README.md) owns this target's
engineering roadmap. It shares M-01's operation/profile admission, graph-resident
obligations, numeric selection, Fidelity.Platform/BAREWire contracts and the
[managed Rocq proof service](../Proof_Composition_Architecture.md). It does not
change the acceptance status or priority of the language families above.

| Area | Scope | Status | Note |
|---|---|---|---|
| [Dynamatic fork](../fpga-targeting/01_dynamatic_fork.md) | Clef-derived input, admitted dataflow transformations and preserved width/proof identities | Planned | Reuse the DHLS engine independently of C application ingestion; no fork integration is established by this plan. |
| [Colibri realization](../fpga-targeting/02_colibri_circuit_basis.md) | Canonical component set, checked contracts and direct VHDL-2008 lowering | Planned | Reuse original implementations and verification assets; extensions and any Clef ports need explicit correspondence. |
| [Circuit and artifact proofs](../fpga-targeting/04_vhdl_to_rocq.md) | Our VHDL-to-Rocq adapter, sequential composition and extracted netlist/configuration evidence | Planned | VHDL2Rocq supplies a published method, not an available dependency; proof coverage is declared per artifact and property. |

[FPGA-01 through FPGA-07](../fpga-targeting/07_roadmap.md) record dependencies,
positive/rejection gates and the progression from HelloArty and a bounded elastic
oracle to mapped-circuit and bitstream evidence. They are workstream milestones,
not a new source-language PRD hierarchy.

### Second Horizon - Admitted Papers

Three working papers are admitted to the future reach, at the second horizon or beyond. The work each sets out is primarily PSG and hypergraph engineering, carried through the Alex coeffect and codata architecture.

| Paper | Named Reach | Depends On | Status | Note |
|-------|-------------|------------|--------|------|
| FPS | "Fixed-Point Scaffolding": three axes meeting at a node (compilation, joint-constraint, verification-strength) | C-01, C-02, C-05, R-04 to R-06 | Planned | Admitted second-horizon research; implementation not started. |
| NFT | Negative and fractional resources, with native bidirectional composition as a reference case; [architecture](../Negative_Fractional_Types_Architecture.md) and [coordination/acceptance plan](../Bidirectional_Composition_Plan.md) | Applicable C-01/C-02/C-05 contracts; M-01 for native witnesses; R-04 to R-06 for reactive extensions | Planned | Design and source corrections tracked separately from implementation. Finite bidirectional, indexed-resource and exact-reconstruction exercises remain Planned. |
| ADM | "Adaptive Domain Models": the geometric product as a joint constraint, with grade inference deriving the non-zero Cayley table entries at design time and eliminating the structurally zero entries from the compiled computation | C-01, C-02, C-05, R-04 to R-06 | Planned | Admitted second-horizon research; implementation not started. |

The applicable closure and lazy contracts expose captures and delayed dependencies; reactive extensions additionally need incremental invalidation and cutoff. These prerequisites do not establish the papers' new semantic or decidability claims. Each admitted fragment needs its own inference, lifecycle and preservation argument, with target realization subject to M-01. The [bidirectional composition plan](../Bidirectional_Composition_Plan.md#executable-work) separates finite pure exercises from their later reactive extensions. The geometric-algebra reach adds its grade and blade-support obligations.

---

## Dependency Graph

M-01 spans the language families and target pathways below. Its operation-level
gates accompany the feature that requires them; target availability and
information-preservation evidence determine which dialect forms are admitted.

```
Foundation (F-01 to F-10)
    └── Implemented core baseline; specific regressions/extensions tracked above
            │
            ├── Computation (C-01 to C-07)
            │       │
            │       ├── C-01 Closures ← F-04 Lambdas, F-10 Records
            │       ├── C-02 HOFs ← C-01 Closures
            │       ├── C-03 Recursion ← C-01 Closures
            │       ├── C-04 Collections ← C-02, C-03
            │       ├── C-05 Lazy ← C-01 Closures
            │       ├── C-06 SimpleSeq ← C-05 Lazy
            │       └── C-07 SeqOps ← C-06 SimpleSeq
            │
            ├── Async (A-01 to A-06)
            │       │
            │       ├── A-01 BasicAsync ← C-01, C-05
            │       ├── A-02 AsyncAwait ← A-01
            │       ├── A-03 AsyncParallel ← A-02
            │       ├── A-04 BasicRegion ← A-01
            │       ├── A-05 RegionPassing ← A-04
            │       └── A-06 RegionEscape ← A-05
            │
            ├── IO (I-01 to I-02)
            │       │
            │       ├── I-01 SocketBasics ← A-02
            │       └── I-02 WebSocketEcho ← I-01
            │
            ├── Desktop (D-01 to D-02)
            │       │
            │       ├── D-01 GTKWindow ← A-02
            │       └── D-02 WebViewBasic ← D-01
            │
            ├── Threading (T-01 to T-05)
            │       │
            │       ├── T-01 BasicThread ← A-02
            │       ├── T-02 MutexSync ← T-01
            │       ├── T-03 BasicActor ← T-02, C-07
            │       ├── T-04 ActorReply ← T-03
            │       └── T-05 ParallelActors ← T-04
            │
            └── Reactive (R-01 to R-06)
                    │
                    ├── R-04 IncrementalFoundations ← C-01, C-05, F-05/F-10, owned cache storage
                    ├── R-05 IncrementalDynamism ← R-04
                    ├── R-01 ObservableFoundations ↔ R-04 shared contracts; C-01, admitted delivery/lifetimes
                    ├── R-02 ObservableOperators ← R-01, C-07; paired incremental use cases
                    ├── R-03 ObservableIntegration ↔ R-05/R-06; admitted async/actor boundaries
                    └── R-06 IncrementalIntegration ← R-04/R-05 + R-01/R-03; T-03 for actor hosting
```

---

## Status Legend

| Status | Meaning |
|--------|---------|
| Planned | Roadmap scope not yet at an implemented acceptance waypoint; the Note identifies specification or prerequisite work |
| In-Progress | Implementation or remediation is underway, with acceptance gates still open; existing operational paths are identified in the Note |
| Complete | The stated scope has an established operational/acceptance baseline; the Note records its boundary and tested revision, without implying a rerun at every later checkpoint |

Retrospective describes document provenance only. A subsequently observed
acceptance regression reopens the affected row as In-Progress; absence of a new
run alone does not. Planned future research and unspecified PRDs are distinguished
in their Notes rather than by additional status values.

---

## Migration Notes

This index replaces the previous sequential `WREN_Stack_PRDs/00_Index.md (now PRDs/INDEX.md)` with category-prefixed organization.

**Mapping from old to new:**
- PRD-00 → F-00 (Foundation Synopsis)
- PRD-11 to PRD-16 → C-01 to C-07 (Computation)
- PRD-17 to PRD-22 → A-01 to A-06 (Async)
- PRD-23 to PRD-24 → I-01 to I-02 (IO)
- PRD-25 to PRD-26 → D-01 to D-02 (Desktop)
- PRD-27 to PRD-31 → T-01 to T-05 (Threading)

Cross-references in existing PRDs have been updated to use the new naming scheme.
