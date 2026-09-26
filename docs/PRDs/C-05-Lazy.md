# C-05: Lazy Values and Memoization

> **Sample**: `14_Lazy` | **Status**: In-Progress | **Depends On**: [C-01](C-01-Closures.md)
>
> **Criteria realignment, 2026-09-25.** Existing lazy checking, capture analysis
> and physical implementation are retained as the migration starting point.
> This revision changes acceptance criteria only; it reports no new execution.

## 1. Executive Summary

A lazy value defers its body until forced. The first force evaluates and stores
the result; subsequent forces of that same value return the stored result without
evaluating the body again. Memoization is required language behavior, including
when the body has effects. It is not an optional optimization or a milestone
deferred until every arena PRD is complete.

The [C-series acceptance contract](C-Series-Acceptance.md) governs status and
evidence. The normative authority is
[Lazy Value Representation, especially §§3, 4, 9 and 11](../../../clef-lang-spec/spec/lazy-representation.md),
with [delayed expressions](../../../clef-lang-spec/spec/expressions.md),
[closure representation](../../../clef-lang-spec/spec/closure-representation.md)
and [backend lowering](../../../clef-lang-spec/spec/backend-lowering-architecture.md).

Sequences share closure/capture machinery but maintain independent enumeration
state; they do not memoize one result. C-06/C-07 progress does not establish C-05
acceptance, and C-05 does not require a new sequence protocol.

## 2. Current State and Remaining Boundary

The [language coverage waypoints](../Language_Coverage_Waypoints.md) record
sample 14's unresolved lazy width/extent read. That native gate remains open.
Current source contains `TLazy`, `LazyExpr`, `LazyForce`, shared capture discovery
and a lazy witness/pattern path. These are existing work to realign, not missing
types and modules to create again.

| Existing component | Current inspection / continuation point |
|---|---|
| [NativeTypes.fs](../../../clef/src/Compiler/NativeTypedTree/NativeTypes.fs) and native unification | Lazy element identity already exists; preserve its NTU type and dimensional constraints |
| [Collections.fs](../../../clef/src/Compiler/NativeTypedTree/Expressions/Collections.fs) | `checkLazy` builds the thunk and shares capture analysis with functions |
| [LazyWitness.fs](../../src/MiddleEnd/Alex/Witnesses/LazyWitness.fs) | Existing physical path still contains a placeholder `TIndex` cached-value type and fixed force-SSA assumptions |
| [ClosurePatterns.fs](../../src/MiddleEnd/Alex/Patterns/ClosurePatterns.fs) | Existing lazy struct/force patterns still reflect code-address-in-environment assumptions |
| [Sample 14](../../samples/console/FidelityHelloWorld/14_Lazy/Lazy.fs) and [manifest](../../tests/regression/Manifest.toml) | Currently expect the effectful thunk to recompute on the second force; this is a stale oracle, not the accepted language behavior |

These source observations explain pending work; they are not fresh build or
runtime results. The [closure retooling plan](../../../clef/docs/fidelity/phg/Closure_Retooling_Plan.md)
and C-01 own the shared callable migration. Preserve already working checking,
capture-origin and reference-remapping behavior while replacing obsolete forms.

## 3. Language Feature Specification

For source such as:

```fsharp
let delayed = lazy (Console.writeln "Computing"; 42)
let first = Lazy.force delayed
let second = Lazy.force delayed
```

construction does not execute the body, `Computing` occurs once at the first
force, and both results are 42. An alias of `delayed` names the same memoized
instance. Two factory calls creating lazy values retain distinct instances and
each may execute its own body once.

Immutable captures retain their formation-time values, including any references
inside those values. Mutable captures retain the original shared cell; a write
before first force affects the computation, while a write after successful force
does not cause a second evaluation. The cached result preserves its own reference
sharing and lifetime. A cached `Result.Error` is an ordinary result value, not a
reason to retry the thunk.

Retain the `Lazy<'T>` element type through checking, capture, storage and force.
Test unit, Boolean, integer, real, measured and admitted aggregate/callable
results; do not represent all cached values as `index`. Any claimed alternative
surface such as `.Value` must share this contract and have its own source gate.

## 4. CCS Layer Implementation

Baker and the owning CCS nanopasses settle the lazy callable, capture operations,
typed memoization slots, force control and storage obligations. The canonical
portable form is:

```text
Lazy<T> = (thunk, env)
env logical fields: [0] computed, [1] cached T, [2..] captures
thunk: receives its environment and reads the settled captures
```

The function value is separate from the flat environment. No code address is an
environment data field. Exact callable identity can permit code-component
elision; it never permits losing the actual environment instance or its formation
effects. Flat means one capture environment, not the absence of a typed view or
the prohibition of the environment parameter.

The owning passes must establish:

- Capture source identity, mode and formation order, excluding module bindings
  from the capture list and retaining mutable cell identity.
- Complete types, representation, offsets, extent and alignment for the computed
  state, cached result and captures under the selected platform facts.
- Definite initialization and the guard permitting a cached-result read. Before
  successful computation, the result slot supplies no value of `T`; do not invent
  a source default, null value or initialized result to fill that gap.
- One memoization identity across aliases, captures, stores and returned values.
  Copies of a carrier do not create a fresh cache for the same lazy instance.
- Source, call, layout, residence and proof participants through recipe fan-out
  and fold-in. An attached layout obligation does not prove lifetime or once-only
  evaluation.

## 5. Composer Implementation

Alex pulls the settled callable, child regions, typed storage and force decisions
through its context and Huet position, then composes the admitted physical form.
It must not reconstruct the thunk algorithm, choose capture slots, infer storage
lifetimes or repair unknown cached-result types during emission.

Use portable `func`, control-flow and `memref` forms admitted for the selected
pathway. Target ABI realization belongs below the backend boundary. The old
direct LLVM struct/address recipes, code-pointer environment field, separate
capture-parameter thunk convention and preassigned/fixed SSA budgets are retired
implementation sketches. They are not alternatives to the current contract.

Target-aware realization must retain correspondence between the lazy instance,
its storage, cached-value accesses and the actual emitted force/publication
operations. A downstream rewrite that changes an obligation's premises requires
preserved or renewed evidence for that artifact. Source memoization evidence,
layout evidence and target acceptance establish different boundaries. Apply the
shared acceptance contract to every claimed pathway; target deployment retains
its own gate.

The [Alex overview](../Alex_Architecture_Overview.md) describes derived SSA names
and the actual traversal. Physical identity/count accounting follows witnessed
operations and their settled types; a count of four or `5 + captures` does not
establish the semantics or storage of a force operation.

## 6. Storage and Lifetime

Memoization needs stable storage for the same lazy instance. Placement follows
the closure lifetime lattice: scope-bounded stack storage, a covering region,
program-lifetime static storage, or genuinely dynamic storage where the target
admits it. Required capture backing storage and cached-result storage must outlive
every admitted use. Escape alone does not justify heap allocation or static
placement; multiple factory instances cannot be collapsed into one global cache.

Program-lifetime caching requires the selected platform's writable storage
authority for the memoization transition. Immutable image authority does not
permit runtime cache writes. The shared [storage contract](C-Series-Acceptance.md#6-framework-consumers-and-target-boundaries)
connects those declarations to the actual allocation and force operations.

Scope-bounded and program-lifetime forms can support memoization without making
completion of all region APIs a prerequisite. Region/dynamic forms retain their
own availability, capacity, release and no-heap-target diagnostics. Reject missing
residence premises explicitly rather than returning a view of a dead frame or
silently allocating a GC-managed carrier.

## 7. Memoization, Concurrency and Open Semantics

Single-threaded memoization is a settled requirement and the first completion
slice. Concurrent or cross-actor force is a separately identified admission:
[Lazy Representation §11](../../../clef-lang-spec/spec/lazy-representation.md#11-normative-requirements)
requires ownership evidence establishing one semantic forcer. Its target
publication/visibility mechanism must preserve the stored result and computed
state. A CAS instruction alone does not discharge that ownership or ordering
contract. Until those premises and their realization are supported, such uses
must receive an explicit located diagnostic and remain outside the accepted
scope; a single-thread pass cannot establish concurrent force support.

Two cases require an explicit native contract before admission: forcing the same
unevaluated lazy value from within its own evaluation, and evaluation that does
not produce a value because of termination, cancellation or an admitted foreign
failure. The current once-successful-force rule does not specify a general
reentrancy/retry/cached-failure protocol. Do not invent one or import managed
exception caching. Coordinate recursive initialization with [C-03](C-03-Recursion.md)
and the [native error model](../../../clef-lang-spec/spec/error-handling.md), then
record the selected behavior and negative cases. Ordinary returned `Result`
values remain covered by normal memoization.

## 8. Validation

Apply the [shared gates](C-Series-Acceptance.md) to these observations:

| Contract | Required oracle |
|---|---|
| Deferred body | Constructing an unforced lazy value runs none of its body effects |
| Once-only force | Two forces, including through aliases, run the body once and return the same cached value |
| Capture semantics | Immutable snapshots, writes before first force, shared mutable captures and cached-result sharing survive the whole pipeline |
| Instance identity | Multiple factory results each memoize independently; no accidental cross-instance cache or expired capture |
| Typed results | Scalar, measured and each admitted aggregate/callable result retain exact types and representation premises |
| Storage | Scope/static and every additionally claimed placement have complete-use residence and release evidence |
| Negative admission | Wrong types, unresolved range/layout/residence and unsupported concurrency/reentrancy forms produce the responsible diagnostic, effective severity and source range |
| Physical/native | Correct graph and portable witness, stock MLIR verification, backend acceptance, exact native effect/output trace and exit status |

**Sample 14 and manifest correction is future implementation acceptance work.**
When the compiler implements canonical memoization, update the sample's old
`no memoization` commentary/label and the manifest together: the second force must
not print `Computing expensive value...` again. Preserve its capture/factory
checks and require the corrected exact-output gate. Changing the expectation
alone cannot establish completion, and retaining the current repeated-effect
expectation cannot certify the specified behavior. This documentation revision
does not modify either fixture.

## 9. Implementation Checklist

- [ ] Preserve existing lazy source typing/capture work and settle the canonical
      `(thunk, env)` form through the owning Baker mechanisms.
- [ ] Resolve sample 14's width/extent boundary and remove placeholder cached
      types through real type/layout settlement.
- [ ] Establish single-thread memoization, alias identity and independent factory
      instances with typed cached results and complete-use lifetimes.
- [ ] Realize the settled form through current Alex patterns and backend contracts;
      retire code-address fields and fixed SSA-budget assumptions.
- [ ] Correct sample 14 and its manifest with the compiler change, then pass the
      exact effect/output gate and applicable closure/sequence regressions.
- [ ] Keep concurrent force explicitly unadmitted until ownership, publication and
      target evidence pass; reconcile reentrant/non-returning force semantics
      before claiming those cases.
- [ ] Record actual source, graph, witness, proof and native results in the
      waypoint; close only the scope those gates establish.

## 10. Related PRDs

- [C-01: Closures](C-01-Closures.md): shared callable representation and residence.
- [C-02: Higher-Order Functions](C-02-HigherOrderFunctions.md): callable cached
  results and lazy values retained by callbacks.
- [C-03: Recursion](C-03-Recursion.md): recursive initialization and force cycles.
- [C-06: SimpleSeq](C-06-SimpleSeq.md), [C-07: SeqOperations](C-07-SeqOperations.md):
  shared capture/layout mechanisms with distinct demand and iteration semantics.
