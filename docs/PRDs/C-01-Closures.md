# C-01: MLKit-Style Flat Closures

> **Status:** In-Progress. Closure implementation and bounded native acceptance
> exist; general callable transport and residence gates remain open.
> **Samples:** `11_Closures`, `11a_DirectCaptures`, `11b_LoopCaptures`, with C-02
> and C-06/C-07 composition oracles.
> **Criteria realignment:** 2026-09-25; documentation only, with no new runtime results.

## 1. Executive Summary

C-01 completes closure formation, capture identity, invocation and storage under
the current Clef specification. It builds on working compiler paths. The remaining
work makes those paths compose across returned, stored and deferred function
values while preserving type, evaluation, lifetime and proof contracts.

The [closure specification](../../../clef-lang-spec/spec/closure-representation.md),
[foreign boundary](../../../clef-lang-spec/spec/ffi-boundary.md) and
[backend contract](../../../clef-lang-spec/spec/backend-lowering-architecture.md)
govern semantics. The [shared C-series criteria](C-Series-Acceptance.md) govern
completion accounting; [coverage waypoints](../Language_Coverage_Waypoints.md)
record dated results and companion revisions. Earlier implementation sketches in
this PRD are retired by §9, not alternative compiler instructions.

C-01 has the same delivery standing as the F-series. Dependencies determine work
order; they do not reduce the obligation to deliver its promised surface intact.

## 2. Language Feature Specification

### 2.1 Clef Closure Semantics

A function value retains its implementation and the environment formed at that
source occurrence. Repeated factory calls can share a code identity while having
distinct environments. An alias retains the function value observed at binding;
it does not become a forwarding call that rereads a mutable binding later.
Partial applications preserve already supplied operands and their effects under
[C-02](C-02-HigherOrderFunctions.md).

### 2.2 Capture Modes

| Binding | Required observation | Acceptance pressure |
|---|---|---|
| Immutable | Retain the value at formation with its source type and dimensions | Later formation, updates and shadowing cannot change an earlier capture |
| Mutable | Retain the original storage cell | Closures share updates; independent enclosing activations retain different cells |
| Aggregate or view | Preserve value semantics and referenced allocations' required lifetimes | Copying a descriptor does not prove backing-storage residence |
| Callable | Retain implementation and actual environment | Equal layouts or one code symbol cannot substitute for the captured value |

Every field is initialized. Absence uses an admitted optional type, not a null
capture. Source identity, allocation occurrence and physical carrier are distinct;
a node ID naming an allocation site is not one global runtime cell.

### 2.3 Flat vs Linked Closures

The environment is one flat block of capture slots; access does not walk a chain
of enclosing environments. A finite capture frontier supplies finite direct layout
obligations. It does not prove arbitrary transitive reachability, captured-storage
validity or safe transfer by itself. See
[closure proof extraction](../../../clef-lang-spec/spec/closure-representation.md#11-proof-extraction-at-closure-sites).

### 2.4 Memory Layout

The canonical middle-end function value is the two-value pair `(fn, env)`. The
function half is never stored as a numeric address inside the environment. A
proved known callee can elide that half; an admitted nonescaping named function
passes captures directly. Neither optimization supplies missing environment
identity or residence evidence.

Widths, offsets, alignment and extent come from settled NTU representations and
the selected platform. No field-count formula, host word size or default cache
line determines layout. Section 14 records the form family and obligations.

### 2.5 Arena-Based Lifetime Management

The historical arena-only sketch is replaced by the specification's lifetime
classification: scope-bounded, region-bounded, program-lifetime or dynamic.
Placement requires an available home on the selected target. Escaping a defining
scope does not automatically imply heap allocation; missing storage authority or
capacity is not permission for a fallback allocation.

Retained mutable cells and aggregate backing storage need their own covering
lifetimes. Program-lifetime initialization additionally requires writable platform
authority. General region APIs remain coordinated with A-04–A-06; their roadmap
position does not waive C-01's current residence gates.

## 3. CCS Layer Implementation

CCS constructs source types, declaration identities and capture relationships.
Baker ingredients, recipes and owning saturation passes establish application
staging, chosen forms, initialization/access/call relationships, layout, residence
and resident obligations before Alex observes them.

Existing immutable direct-capture elaboration proves complete eligible use and
adds truthful leading formals and operands, including recursive forwarding.
Existing bounded materialized environments retain `ClosureValue`,
`EnvironmentCreate`, capture reads/writes/borrows, implementation formals and
actual environment operands. [Closure values as data](../Closure_As_Data.md)
describes those identities; later waypoints supersede its interim native-status prose.

Direct mutable capture passing still needs one authoritative typed application
contract. [The cell contract](../Direct_Capture_Cell_Contract.md) records the
internal storage-type versus typed-signature-fact seam. Existing `CellView` and
environment-borrow machinery can be reused, but does not settle hidden cell
formals, recursive forwarding or their effect summaries. This PRD selects neither
a new source type nor competing implementation signatures.

## 4. Composer/Alex Layer Implementation

Alex pulls settled graph/codata, platform facts and the actual Huet position.
Environment witnesses/patterns express admitted operations; ordinary application
and lambda witnesses consume Baker's explicit operands and formals. Missing
capture, layout, availability or residence premises remain specific failures in
the owning stage.

General pair transport still needs multi-value recall, signatures, call/return
results, joins and stored-callable handling. The legacy packed path in
`ApplicationPatterns.pClosureCall` is implementation debt, not the canonical
contract. The bounded known-callee path does not establish arbitrary function
storage or invocation. `Values.fs` derives SSA names from graph/role ordinals and
block arguments; no capture reconstruction or SSA-preassignment pass is added.

## 5. MLIR Output Specification

| Settled form | Admitted physical expression | Required correspondence |
|---|---|---|
| Known direct function | `func.call` with graph-established operands | Exact implementation, signature and actual captures |
| Materialized function value | Function-typed SSA value and separate environment | Both survive parameters, returns, branches and eliminators |
| Unknown-callee invocation | `func.call_indirect` with actual environment first | Signature and environment instance agree at that occurrence |
| Environment slots | Typed views and loads/stores over placed storage | Exact offsets, extents, alignment, capture modes and backing allocation |
| Target boundary | Selected pathway's admitted realization | ABI, layout and semantic obligations survive commitment |

Stock verification does not prove capture timing, lifetime or native correctness.
Interior function addresses are not converted to data. Canonical closure output
SHALL NOT rely on `unrealized_conversion_cast` or a cast-resolution plugin.
[M-01](M-01-DialectAdmission.md) governs operation/profile admission and downstream
information preservation, including transformations of the witnessed artifact.

## 6. FFI Boundary Marshaling

### 6.1 The Boundary Problem

| Crossing | Clef contract | Additional evidence |
|---|---|---|
| Named native callback entry | `FnPtr<'F>` with declared signature | Entry identity, scalar ABI and explicit opaque context where supplied |
| Capturing callback registration | Closure and admitted registration adapter | Trampoline, actual environment, registration/invocation/release lifetime |
| Foreign function value | Typed `FnPtr<'F>`, optional where required | Symbol/declaration provenance and full signature through invocation |
| Foreign data handle/reference | `CHandle<'T>` or admitted bounded reference | Layout, extent, access, residence and the API's retention contract |

### 6.2 Clef → C Callback (Pattern A: Registration)

The current native entry surface uses a named module function without captures.
`FnPtr.ofFunction` rejects arbitrary lambdas, including capture-free anonymous
lambdas. Foreign context remains explicit under the
[FFI specification](../../../clef-lang-spec/spec/ffi-boundary.md#34-fnptroffunction).
General capturing registration requires an adapter and §6.7's joint contract;
a working named listener entry does not establish that support.

### 6.3 C Function Pointer → Clef Callable (Pattern B: dlsym)

The historical integer-pointer sketch does not define the source surface. Use
typed `FnPtr<'F>` operations and the declared foreign signature. `FnPtr.fromSymbol`
declares a link-time symbol; runtime lookup, when supplied by a platform binding,
needs its own typed/nullability contract. A foreign function pointer is not an
interior closure with an invented environment.

### 6.4 Clef Struct → C Raw Pointer

Address extraction realizes an admitted typed boundary reference while retaining
actual storage, representation, extent, access and lifetime. A temporary copy
cannot establish retained-registration lifetime or preserve shared mutation.
`nativeptr`, `NativePtr`, `voidptr` and integer-as-pointer source examples from
the original PRD are retired.

### 6.5 FFI Argument Marshaling at ExternCall Boundaries

Generated declarations govern scalar representation, logical unit versus C
`void`, optional handle/function-pointer conversion and bounded reference
arguments. The selected backend commits addresses and ABI layout. Generic memref
extraction or equal byte size alone is insufficient evidence for a foreign signature.

### 6.6 Design Principle: C Does Not Leak In

`CHandle<'T>` remains opaque; `FnPtr<'F>` retains its function type. Foreign
absence sentinels are converted at the boundary. Raw-pointer source operations,
interior C calling conventions and numeric function-address carriers are not
workarounds for a missing callback contract.

### 6.7 The Boundary Contract as a Joint Constraint

A crossing joins the Clef value/source identity, declared foreign ABI and
storage/lifetime owner. Its graph relation retains declaration, entry/adapter,
actual arguments/environment, registration, invocation and release as applicable.
Independent pairwise checks cannot replace the joint lifetime claim.

| Tier | Foreign surface | Required registration contract |
|---|---|---|
| A | Userdata and destroy hook | Environment survives every admitted invocation; the declared destroy protocol releases it exactly once |
| B | Userdata without destroy hook | An owned registration handle/protocol connects installation, all uses and disconnect/destruction before release |
| C | No userdata | Admitted closed native entry satisfying `FnPtr.ofFunction` source restrictions, not merely an empty capture count |

A/B adapter mechanisms and their release/provenance gates remain explicit work
with the owning Farscape and IO/Desktop callback integrations. This table defines
their acceptance contract; it does not claim every tier is implemented or select
a new source linear-handle syntax. Existing typed entry/reference gates remain.

Acceptance requires exact graph-to-artifact correspondence for the crossing and
its obligations. Historical deferred-cast-count reconciliation is retired under
the current closure/backend specifications. Missing, mismatched or stale
participant evidence must refuse the crossing. Recording a proposition or checking
ABI shape is not discharge. Finite direct fields do not establish a retained
reference's separate validity or an external API's completion/release premise.

## 7. Lazy and Seq Extensions

### 7.1 Lazy Thunk (C-05)

Lazy adds memoization state and a cached-result slot, retaining the separate thunk
half. [C-05](C-05-Lazy.md) owns deferred execution, first-force publication and
subsequent cached observations. Repeated recomputation is not alternative lazy
semantics or a prerequisite staging concession.

### 7.2 Seq Generator (C-06/C-07)

Sequences add continuation state, current availability and persistent slots under
[C-06](C-06-SimpleSeq.md)/[C-07](C-07-SeqOperations.md). Captured callbacks retain
actual environments across suspension. Returned child sequences and callback
environments require covering storage independently of descriptor copies.

## 8. Validation

Apply the [shared gates](C-Series-Acceptance.md) to each admitted row. These are
completion requirements, not new pass claims.

A precise diagnostic for a currently unsupported conforming use does not satisfy
its positive gate. Such gaps keep C-01 In-Progress; genuine target-capability
refusals and separately scoped future integrations are recorded distinctly.

| Contract | Positive source/native cases | Discriminating negative or preservation cases |
|---|---|---|
| Formation | Zero/one/multiple captures, mixed types, shadowing, immutable loop snapshots, repeated formation | Missing identity/capture cannot be omitted; later updates cannot alter saved immutable values |
| Mutable cells | Shared updates, independent factory activations, recursive forwarding | Copied cells, stale range guards and uncovered residence are rejected |
| Direct form | Fully accounted named direct calls, recursion, measured signatures | Returned/stored/partial/opaque uses do not receive unproved direct-only admission |
| Full callable values | Returned/nested functions, parameters/results, conditional selection, record/tuple/DU/collection storage | Different implementations with equal layouts and different formations of one implementation remain distinct |
| Deferred composition | Callable/sequence captures, nested recapture, factories and `collect` children | Missing backing lifetime, ambiguous origins or incomplete use coverage remain located residuals |
| Placement | Scoped, caller/region-owned and program-lifetime cases under declared authority | Wrong extent/alignment, unavailable storage, invalid sharing or absent writable authority cannot fall back to guessed allocation |
| Tooling | Source signatures, capture navigation, dimensions and unsaved repair | Hidden parameters do not replace public types; stale evidence is invalidated |
| Foreign crossing | Existing named entries and admitted reference/registration adapters | Wrong signature/nullability, inadmissible `FnPtr.ofFunction`, absent ABI provenance and premature release are refused |

Retain original11, 11a/b, original12 and the unchanged original16/16h composition
oracles. Add absent cases instead of substituting a no-capture callback for a
retained environment. Check effects and allocation-instance identity, not only totals.

## 9. Implementation Status

| Dated evidence | Established boundary | Remaining qualification |
|---|---|---|
| [2026-09-19 direct captures](../Language_Coverage_Waypoints.md#c-01-immutable-direct-captures--2026-09-19) | Immutable direct/recursive elaboration, 11a, native callback and peered projection gates | Mutable direct form and general materialized-pair conformance remain open |
| [2026-09-20 iteration bindings](../Language_Coverage_Waypoints.md#c-01c-04-immutable-iteration-bindings--2026-09-20) | Fresh immutable iteration captures and 11b native behavior | Broader capture residence and canonical transport remain separate |
| [C-06 regression checkpoint](../Language_Coverage_Waypoints.md#c-06-native-continuation-settlement--2026-09-20) | 23/28 compiled and 23/23 executed successfully; original11/12 are in the passing cohort | Recorded recursion/lazy/sequence and foundation failures remain separately accounted |
| [C-07 waypoint](../Language_Coverage_Waypoints.md#c-07-sequence-operations--implementation-waypoint-acceptance-open-2026-09-20) | Bounded materialized scalar callbacks; native16a–g with graph/proof/tooling evidence at the recorded artifacts | Original16/16h retain factory/capture and staged-application failures |
| [Native callback corpus](../../tests/NativeCallbacks/README.md) | Named listener entry, scalar ABI and ordinary calls have recorded native evidence | Unannotated higher-order ABI provenance and general capturing registration remain separate |

Exact hashes, cohort boundaries and focused reruns remain in those records. This
realignment does not claim repetition on the current tree.

Retired material includes code-address-in-environment examples, numeric packed
pairs, fixed-width/global-arena defaults, SSA-preassignment, raw-pointer source
examples and unchecked historical phase boxes. Executable vestiges are migration
work; replacement must preserve established behavior under current contracts.
Existing Baker construction/native success is not relabeled as absent implementation.

## 10. Expanded Sample 11 — Boundary Marshaling Coverage

Coverage extends beyond the current original fixture: mutable counters and
accumulators; immutable strings/mixed captures; zero-capture values; nested returned
closures; HOF invocation; bounded aggregate captures; and §6's distinct foreign
crossings. Each needs its own applicable gate. Full registration adapters remain
tracked with §6.7's integrations, and descriptor transfer with §14.2's horizon;
neither receives credit from an ordinary interior closure test.

## 11. Files to Create/Modify

| Owner | Relevant current code/contracts |
|---|---|
| CCS source | `NativeTypedTree/Expressions/Bindings.fs`, `Applications.fs`; signatures and generalization |
| Baker construction | `Nanopass/CallableApplications.fs`, `ClosureElaboration.fs`, `ClosureEnvironmentElaboration.fs`; corresponding Application/Closure/ClosureEnvironment recipes |
| Settlement | `PSGSaturation/SemanticGraph/DirectCaptures.fs`, `ClosureEnvironments.fs`, placement/range/residence owners; `Nanopass/ClosureEnvironmentSettlement.fs` |
| Alex | `Traversal/Values.fs`, `Dialects/Core/Types.fs`, environment/application/lambda witnesses and patterns |
| Evidence/clients | Source/graph, Alex, native callback/sequence, proof correspondence and shared CCS.Editor projection gates |

Use these owners rather than recreating retired source paths or a second emitter.

## 12. Academic References

Region soundness and safe-for-space conversion motivate the architecture: Tofte
and Talpin, *Region-Based Memory Management* (1997), and Shao and Appel,
*Space-Efficient Closure Representations* (1994). The Clef specification states
the adopted obligations; literature does not establish implementation discharge.

## 13. Related PRDs

- [C-02](C-02-HigherOrderFunctions.md): application, staging and transport.
- [C-03](C-03-Recursion.md): recursive identity, forwarding and numeric evidence.
- [C-04](C-04-CoreCollections.md): stored functions and collection backing storage.
- [C-05](C-05-Lazy.md), [C-06](C-06-SimpleSeq.md), [C-07](C-07-SeqOperations.md): deferred environments and transition/lifetime contracts.
- [M-01](M-01-DialectAdmission.md): operation/profile admission and information preservation.

## 14. The Closure Saturation Form Family

This retains the architectural reference point for closures, continuation frames
and later actor state. Form names organize settlement; they are not source syntax
or evidence that every form is implemented.

### 14.1 The Environment as the General Object

The graph retains implementation identity, ordered captured declarations and
formation values, modes, allocation, applications, lifetime classification and
retirement. Slot-layout and captured-storage evidence concern those same
participants. Extending families add their own slot classes and transition
disciplines; shared vocabulary does not close their separate semantic gates.

### 14.2 The Form Family and Its Selection at Saturation

| Form | Selection premise | Realization and current boundary |
|---|---|---|
| Vacant | No captures | Elide an unnecessary environment where the admitted convention permits; no null stand-in |
| Unmaterialized | Complete use proves the nonescaping named direct form | Leading captures; immutable form has native evidence, mutable signature/effect work remains |
| Stack flat | Value and backing captures fit a covering activation | Typed placed environment; bounded known-callee scalar callbacks are implemented |
| Mixed ByRef | Mutable captures retain shared cell identity | Exact cell descriptors/residence; bounded cases do not admit arbitrary forwarding/recapture |
| Region escaping | Returned/stored environment needs a longer-lived home | Caller/region/program-lifetime placement as classified; dynamic storage only under the required target contract |
| Large capture | Target-specific field policy changes physical copy/view choice | Preserve value semantics/sharing and redo layout/backing-lifetime obligations; no guessed threshold |
| Descriptor shared | Transfer across a memory fabric | Explicit later horizon: representation compatibility, reference validity/relocation, sharing and destination code availability under BAREWire |

Materializing refinements can compose. Flatness does not make captured references
self-contained, and region-scoped descriptor fields alone do not establish
transport. Foreign fence packing belongs to §6.7, not an interior form.

### 14.3 Full Expression in Standard MLIR Primitives

Function constants, direct/indirect calls, typed memref views and ordinary
structured control express admitted layout-realizing forms. The full pair travels
through multi-value operands/results without a numeric function-address field.
Elision requires settled premises. Additional forms and target commitment follow
the backend specification and M-01.

Verification checks symbols, signatures and operation structure, not view bounds,
captured-cell lifetime or external release completion.

### 14.4 Verification Conditions per Form

| Obligation | Required evidence |
|---|---|
| VC-EXT | Fields, padding, alignment and complete extent agree with selected representation |
| VC-DIS | Containment and required disjointness over actual offsets/sizes |
| VC-REG | Backing allocations cover admitted uses; placement has authority and capacity |
| VC-REL | Owning activation/region/registration has applicable retirement and single-release evidence, without outstanding use |
| VC-APP | Actual implementation, formals, environment and operands satisfy one truthful typed application contract |

Literal layout arithmetic uses the standing admitted fragment; incidence,
declaration, application and lifetime judgments retain their owning mechanisms.
Missing premises remain explicit through commitment. A finite frontier is not
transitive storage-lifetime proof. Slot extensions and transfers add obligations.

### 14.5 The Two-Sided Check: PSG Dispatch and the MLIR smt Dialect

Graph-resident obligations and discharge remain connected to actual witnessed
artifacts. Recorded bounded solver/transfer/correspondence gates do not establish
general closure artifact correspondence or every form's proof transport.

Retain owner, slot, allocation, application and source participants through
lowering; preserve or recheck properties affected by each transformation.
Re-solving an upstream formula without checking the artifact is insufficient.
Altered offsets, carriers, origins, signatures and stale evidence must fail the
responsible gate. Emitting an SMT module is not a discharge result. Broader
transport follows [obligation residency](../Obligation_Residency_Design.md),
[proof composition](../Proof_Composition_Architecture.md) and M-01.

The [FPGA/Colibri plan](../fpga-targeting/README.md) and
[eBPF admission plan](../ebpf-targeting/README.md) sharpen this requirement now:
source contracts must survive settled graph, admitted witness and transformed
artifact. Circuit component correspondence and pinned-host bytecode admission
are different downstream checks of that integrity. Their planned integrations
do not certify every target or follow from a host-native closure pass.

### 14.6 Worked Example: The Counter End to End

`makeCounter start` creates a new mutable cell and a closure retaining it. Two
calls on one counter observe successive updates; a second formation has an
independent cell. Returning it requires a proved home for environment and cell.

Baker establishes initialization, actual cell forwarding, range/effect updates,
layout and residence. Alex consumes those facts and the actual callable pair.
A static singleton cell, hand-selected carrier or departed stack view cannot
substitute for distinct formations. This remains a required returned mutable
counter gate, not a new implementation result.

### 14.7 Status and Sequencing

1. Preserve existing direct/bounded materialized paths while completing C-02's staged operation-value gates.
2. Settle direct mutable-cell signatures and general pair transport, including joins, aggregate/DU storage and call/return agreement.
3. Extend returned/nested environment and backing-storage residence; validate original16/16h and C-04/C-05/C-06 composition.
4. Close graph, proof/artifact, native and tooling gates on a recorded final cohort. Keep registration adapters and descriptor transfer identified until their own gates pass.

Historical phase boxes and isolated pattern fixtures cannot establish Complete.
The final record states passed forms/profiles, specific remaining refusals and
the broader boundary/horizon claims still open.
