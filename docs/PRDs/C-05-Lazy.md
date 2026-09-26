# C-05: Lazy Values and Memoization

> **Samples**: `14_Lazy`, `14a_LazyScalarResults`, `14b_LazyStringViews` | **Status**: In-Progress | **Depends On**: [C-01](C-01-Closures.md)
>
> **Criteria realignment, 2026-09-25.** Existing lazy checking, capture analysis
> and physical implementation are retained as the migration starting point.
> This revision changes acceptance criteria only; it reports no new execution.
>
> **Implementation checkpoint, 2026-09-26.** Sample 14 now passes the corrected
> memoization oracle, including factory aliases and original mutable cells.
> Sample 14a also passes the demanded unit, Boolean, integer, real and measured
> result gate. Sample 14b passes retained string results and immutable captures
> backed by the current declared static string pool. These are bounded,
> single-threaded storage gates; the remaining capture,
> result and execution families below are still required acceptance work.

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

The [evaluation strategy contract](../Evaluation_Strategy_Contract.md) separates
lazy memoization from dependency-invalidated incremental caches and observable
delivery. Mutating a capture after successful force does not invalidate a lazy
result. Cold activation and push/pull delivery are independent questions.

## 2. Current State and Remaining Boundary

The earlier sample 14 width/extent failure is superseded by the 2026-09-26 native
checkpoint. Baker now makes the guard, cold computation, cache store, publication
and hot read explicit in the source graph. Alex observes that graph through the
existing traversal and portable operations. The original source `TLazy`,
`LazyExpr`, `LazyForce` and capture discovery remain the entry to this settlement.

| Existing component | Current inspection / continuation point |
|---|---|
| [NativeTypes.fs](../../../clef/src/Compiler/NativeTypedTree/NativeTypes.fs) and native unification | Lazy element identity already exists; preserve its NTU type and dimensional constraints |
| [Collections.fs](../../../clef/src/Compiler/NativeTypedTree/Expressions/Collections.fs) | `checkLazy` builds the thunk and shares capture analysis with functions |
| [LazyRecipes.fs](../../../clef/src/Compiler/Baker/Recipes/LazyRecipes.fs) and [LazyValues.fs](../../../clef/src/Compiler/PSGSaturation/SemanticGraph/LazyValues.fs) | Source formation and exact guarded memoization protocol; the cache has a typed declaration and no initial value |
| [LazyFactoryResults.fs](../../../clef/src/Compiler/Nanopass/LazyFactoryResults.fs), [LazyResidence.fs](../../../clef/src/Compiler/PSGSaturation/SemanticGraph/LazyResidence.fs) and [LazyRuntime.fs](../../../clef/src/Compiler/Nanopass/LazyRuntime.fs) | Distinct caller destinations, complete-use residence, typed placement and proof retraction; unchanged settlement retains proof identities |
| [LazyOperands.fs](../../src/MiddleEnd/Alex/Traversal/LazyOperands.fs), [LazyWitness.fs](../../src/MiddleEnd/Alex/Witnesses/LazyWitness.fs) and [LazyPatterns.fs](../../src/MiddleEnd/Alex/Patterns/LazyPatterns.fs) | Separate actual thunk/environment operands, typed field operations and passive forwarding; no force algorithm, code-address field or placeholder cache type in Alex |
| [Sample 14](../../samples/console/FidelityHelloWorld/14_Lazy/Lazy.fs) and [manifest](../../tests/regression/Manifest.toml) | Corrected together after a native run established memoization; demanded output checks now cover repeated global force, factory-instance independence, factory aliasing and mutation before/after first force |

The source and component gates also check exact program storage authority,
uninitialized cache declarations, original instance operands, typed thunk formals
and rejection after removal of owning evidence. Declaration reachability and
physical observation do not manufacture an initialized cache value. Program
references name the actual admitted static allocation; two factory instances
cannot substitute for one another because they share a schema.

The [closure retooling plan](../../../clef/docs/fidelity/phg/Closure_Retooling_Plan.md)
and [C-01](C-01-Closures.md) retain the shared callable obligations. Required continuation work
includes deferred immutable capture identities under the specification's
[ordinary call-by-need rules](../../../clef-lang-spec/spec/expressions.md#default-demand-and-sharing), nested/forwarded views,
the remaining result families, and additional storage/control-flow cases. The
passing scalar gate does not establish these cases by analogy.

## 3. Language Feature Specification

For source such as:

```fsharp
let delayed = lazy (Console.writeln "Computing"; 42)
let first = Lazy.force delayed
let second = Lazy.force delayed
Console.writeln (string first)
Console.writeln (string second)
```

construction does not execute the body. The two ordinary result bindings are
also deferred; the sequential writes demand them. `Computing` occurs once when
the first demanded force executes, and both results are 42. An alias of `delayed` names the same memoized
instance. Two factory calls creating lazy values retain distinct instances and
each may execute its own body once.

Immutable captures retain the original shared binding, including its deferred
identity and any references in an already computed value. Capture formation
does not force the binding. Mutable captures retain the original shared cell; a write
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
| Capture semantics | Shared deferred immutable bindings, already computed values, writes before first force, original mutable cells and cached-result sharing survive the whole pipeline |
| Instance identity | Multiple factory results each memoize independently; no accidental cross-instance cache or expired capture |
| Typed results | Scalar, measured and each admitted aggregate/callable result retain exact types and representation premises |
| Storage | Scope/static and every additionally claimed placement have complete-use residence and release evidence |
| Negative admission | Wrong types, unresolved range/layout/residence and unsupported concurrency/reentrancy forms produce the responsible diagnostic, effective severity and source range |
| Physical/native | Correct graph and portable witness, stock MLIR verification, backend acceptance, exact native effect/output trace and exit status |

**2026-09-26 sample 14 checkpoint.** The original sample first compiled and ran
with only the stale repeated-effect expectation differing: its second force
returned 42 without printing `Computing expensive value...` again. The sample
and manifest were then corrected together and extended with two demanded checks:
a returned factory alias yields its existing cached 7, and a lazy value sharing
mutable cells yields 20 on both forces despite an intervening write to 30, with
a printed thunk execution count of 1. The extended sample passed stock MLIR
verification, linking, exact stdout and exit 0 (17.41 s compilation, 31 ms run).

Evidence is retained in `/tmp/composer-lazy-memoization-v1.log` and
`/tmp/composer-lazy-memoization-v1/0001`, using the private verified compiler
`/tmp/composer-lazy-native-v4/compiler`. The preceding base observation is in
`/tmp/composer-lazy-native-v4.log`. The C-05 source cases passed within
`/tmp/clef-lazy-boundary-focused-v5.log` (114/116 overall; the two failures concern
the separate callable-promotion change), and the physical cohort passed 12/12
in `/tmp/composer-lazy-components-v6.log`. The subsequent full Alex gate passed
167/167 with zero skips in `/tmp/composer-lazy-passive-alex-v6.log`.

The source fixtures containing `ignore (Lazy.force ...)` are graph/proof tests,
not runtime effect-demand oracles: `ignore` does not demand its argument under
the [ordinary call-by-need rules](../../../clef-lang-spec/spec/expressions.md#default-demand-and-sharing).
Sample 14 demands results through output operations. This checkpoint does
not close the broader C-05 acceptance table.

**2026-09-26 typed scalar checkpoint.** Registered
[sample 14a](../../samples/console/FidelityHelloWorld/14a_LazyScalarResults/LazyScalarResults.clef)
passed stock verification, linking and its exact native stdout/exit-0 oracle
(19.47 s compilation, 30 ms run). It demands unit, Boolean, negative integer,
real, measured integer and measured real results; checks the effect count is
zero before demand; forces one instance through two aliases with count one;
and forces two factory instances with total count two. Unit forces are explicit
sequential expressions. The other checks demand results through printed
conditions, so none relies on `ignore` forcing an ordinary argument.

Evidence is `/tmp/composer-lazy-scalars-v4.log` and
`/tmp/composer-lazy-scalars-v4/0001`, with compiler hashes and build identities
retained beside `/tmp/composer-lazy-scalars-v4/compiler`. This gate includes
the actual-condition Huet descent repair and the Lazy rewrite's own liveness
refresh for replaced assignment targets. It preserves the original shared
mutable cells; it does not exempt unwitnessed reachable nodes from coverage.
Retained aggregate, view and callable results still require their own source
storage proofs and physical/native gates.

**2026-09-26 retained string checkpoint.** Registered
[sample 14b](../../samples/console/FidelityHelloWorld/14b_LazyStringViews/LazyStringViews.clef)
passed stock verification, linking and exact native stdout/exit 0 (13.76 s
compilation, 34 ms run), using `/tmp/composer-requirements-v13/compiler`.
Evidence is `/tmp/composer-lazy-strings-v1.log` and
`/tmp/composer-lazy-strings-v1/0001`. An immutable captured string is forced
through two aliases with one execution; two returned instances produce the
expected strings with two executions despite another alias force.

Admission depends on the actual final result and immutable capture paths
reaching literals in the current BAREWire string pool, with the selected
immutable program-lifetime authority and exact bytes, sentinel, capacity and
layout premises. The cache holds the complete view descriptor; no string body
is replayed to recover a value. The source cohort passed 164/164 in
`/tmp/clef-requirements-focused-v13.log`, including retraction after backing,
authority, literal or provenance changes. This checkpoint covers immutable
image-backed strings; other retained views still need their own coverage.

**2026-09-26 main-repository cohort.** After integrating the compiler work into
main, all three registered samples (14, 14a and 14b) passed together against
`/tmp/composer-requirements-v14/compiler`: 3/3 compilations and 3/3 exact native
output/exit oracles, with zero skips and at most three independent compiler
processes. Evidence is `/tmp/composer-lazy-main-v14.log` and the per-job artifacts
under `/tmp/composer-lazy-main-v14`. This snapshot also includes the exact thunk
formal, immutable alias, declared annotation type and all structural occurrence
checks for capture access. These results precede the subsequent
[explicit eager](../../../clef-lang-spec/spec/expressions.md#eager-expressions)
source integration; they do not claim execution of that later combined state.

## 9. Implementation Checklist

- [x] Preserve existing lazy source typing/capture work and settle the canonical
      `(thunk, env)` form through the owning Baker mechanisms for the current gate.
- [x] Resolve sample 14's width/extent boundary and remove placeholder cached
      types through real type/layout settlement.
- [x] Establish the sample's single-thread memoization, alias identity and independent
      factory instances with typed cached results and complete-use lifetimes.
- [x] Realize the settled form through current Alex patterns and backend contracts;
      retire code-address fields and fixed SSA-budget assumptions.
- [x] Correct sample 14 and its manifest with the compiler change and pass its
      extended exact effect/output gate.
- [x] Pass demanded unit, Boolean, integer, real and measured scalar results,
      with before-demand, alias and independent-instance execution counts.
- [x] Pass cached string results and immutable captures with current declared
      image backing, complete descriptor transport, aliases and distinct factories.
- [ ] Complete the remaining capture/result/storage families in §§3, 6 and 8,
      including shared deferred immutable bindings, nested/forwarded views and
      source-admitted aggregate/callable caches; preserve every positive oracle.
- [ ] Pass the applicable closure/sequence controls and the whole C-05 gate
      against one coherent compiler checkpoint.
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
