# Closure settlement contract

This is the shared implementation and acceptance contract for closure work across
C-01, C-02 and C-05–C-07. The [language specification](../../clef-lang-spec/spec/closure-representation.md)
governs it. Delivery includes the required representation family and source
behaviors; a successful subset does not close a C-series gate.

## Source semantics that every form preserves

| Rule | Required behavior | Governing specification |
| --- | --- | --- |
| Immutable capture | Evaluate and retain the value at formation; references contained in it preserve sharing. | Closure §2.2 |
| Mutable capture | Retain the original storage instance. All reads and writes share it, including across nested and returned functions. | Closure §2.2, §3.3 |
| Partial application | Evaluate supplied operands once, in source order, at each application frontier. A later call cannot replay them or reread a reassigned source binding. | [Expressions](../../clef-lang-spec/spec/expressions.md), [sequence operations](../../clef-lang-spec/spec/seq-operations-representation.md) |
| Actual callable boundary | Distinguish declaration currying from a function returned as a value. A `TFun` spine alone does not establish native arity. | Expressions, closure §6 |
| No captures | Use direct code with no environment. | Closure §3.1, §9 |
| Named nonescaping function | Use leading capture parameters when complete use proves that form. Syntax and binding parentage are insufficient. | Closure §8, §10.11–12 |
| Materialized closure | Keep function and flat environment separate; elide the function half only for a proved known callee. | Closure §2.1, §6.3 |
| Invocation | Environment first, then other admitted internal operands and explicit source arguments; all physical formals and actuals have one truthful signature. | Closure §6.1 |
| Storage | Prove a covering lifetime for the environment and every retained reference, under declared target storage facts. | Closure §2.3, §3.3, §11 |

A code symbol or source allocation site identifies neither a unique runtime
formation nor a unique mutable cell. Repeated calls can create different instances
with the same layout. Known-code elision must retain the actual environment value
at each occurrence. A mutable alias and an opaque alternative cannot be treated
as an immutable value path.

## Owning stages and dependency transport

CCS establishes lexical identities, source types and capture modes. Baker recipes
construct formation, invocation, capture access and destination protocols through
fan-out/fold-in. Owning nanopasses settle effects, numeric ranges, complete uses,
lifetimes, layout and obligations. Alex reads the settled facts at the actual
Huet position; it does not infer captures, reconstruct a missing environment,
select a lifetime or repair a call's arity.

The zipper remains structural and immutable. Derived SSA names, operand recall,
visited state and scope associations remain emission bookkeeping. Shared bodies
are observed through the invoking path; re-rooting a shared lambda does not
preserve its occurrence-dependent parameter scope.

The shared callable-origin analysis follows concrete declarations, immutable
aliases, actual/formal supplies, returned bodies and supported aggregate paths.
It preserves every known alternative and explicit unknown participants. Consumers
use its actual call boundaries. Type-compatible but unrelated functions must not
pollute a proven call's range or effects. An unknown participant must prevent a
consumer from treating the known subset as complete.

## Formation and storage protocol

The source callable retains its type, identity and source range. Its lifted
implementation and hidden formals have truthful physical signatures, with source
signature provenance for editor projection. A captured value's slot identity is
separate from the initializer value at a particular formation.

| Relationship | Required participants and meaning |
| --- | --- |
| Capture | Layout owner, original source slot, actual initializer, capture mode and ordinal. |
| Environment formal | Layout owner, lifted implementation and the actual environment parameter. |
| Nested formation | Parent implementation/formal, child constructor, original slots and explicit read/borrow initializers. |
| Result destination | Exact returning implementation, constructor, layout owner and destination formal. |
| Prepared result call | Actual invocation, supplied values, destination operand and distinct caller allocation. |
| Input borrow | Source allocation, covering activation, actual argument, formal and callee at the exact call. |
| Retained result view | Captured slot/initializer, factory/formal, exact call/actual, destination and allocation. |
| Residence proof | Complete uses and all crossing activations, retaining the obligations of captured views. |
| Layout proof | Ordered slot extents, alignment, containment, disjointness and all construction participants. |

These are joint relationships with ordered participants. For example, changing
the initializer while retaining the slot key changes the lifetime premise;
changing an actual argument while retaining the factory and formal changes the
particular source storage. A code identity, type match or allocation site alone
cannot replace either relationship. The concrete graph-role shapes are recorded
in [Closure as data](Closure_As_Data.md#3-published-facts-and-resident-incidence).

Preparation and proof have different meanings. Creating a caller destination
makes storage explicit; it does not establish that a captured reference covers
that destination's complete use. A pending retained-view relationship must be
discharged before native admission. Moving a result's descriptor cannot extend
storage in a finished callee activation.

A returned environment is initialized at its original formation frontier. A
caller allocation supplies backing storage without running the initializer there
or copying an already-invalid callee stack reference. A second call receives a
distinct destination. Destination insertion preserves the environment-first
convention and the order/evaluation count of all original operands.

The same storage reasoning applies jointly to a callback retained in a sequence,
a sequence retained in a partial application, and a child sequence retained by
an enumerator. Their descriptor layouts are different; their shared lifetime
premises cannot be proved independently by ignoring each other's uses.

Before final residence admission, revalidate each destination against its actual
constructor, returning implementation, physical signature and complete call set.
For every retained view, resolve the exact call's supplied value through
immutable snapshots and explicit environment reads, then prove that its backing
storage covers all uses of each caller destination. Scalar copying and retaining
a descriptor have different lifetime premises. No pending capture requirement,
preexisting unrelated witness or successful placement substitutes for the joint
storage proof.

## Saturation and commitment

Representation recipes expose relationships before their consumers commit.
Capture forwarding precedes range/effect settlement. Result-destination
preparation precedes final residence checks for both environment and sequence
families. Layout and continuation machinery consume the resulting evidence;
preliminary missing-premise diagnostics cannot survive as stale final facts after
settlement has supplied and proved those premises.

Complete-use reasoning includes ordinary calls, retained values, deferred
constructors and every relevant reference edge. It must reject missing or
contradictory participants, unknown consumers and cycles with no established
covering root. A recursive call graph is not a lifetime proof merely because a
fixed point stopped changing.

Coverage is directional and scoped: a particular ordinary activation may cover
its complete ordinary callees without giving their values program lifetime.
Every incoming call participates, including calls through aliases or formals;
known and unknown alternatives cannot be reduced to the known subset. Deferred
calls retain their exact constructor and captured-source obligations. A returned
constructor transfers storage responsibility only through the validated caller
destination protocol. It cannot inherit the lifetime of a departed factory.

Environment-first argument projections can pass through immutable snapshots
introduced by operand preparation. Their aliases, actual argument positions and
all independent reference uses remain proof participants. Replacing such a path
with mutable storage, adding an opaque use, detaching an incoming call from its
activation or changing a capture/destination row requires renewed settlement;
the old conclusion cannot be kept merely because its original successful path
still exists.

The finite capture list bounds direct slot obligations. It does not bound
arbitrary transitive storage, prove that a result is nonescaping, or discharge
release and sharing obligations. Target realization preserves or rechecks each
affected property. A successful MLIR verifier does not establish source semantics
or residence.

## Incremental contract

Every conclusion needs its actual read dependencies and complete alternative
support. These include negative observations: no additional consumer, no opaque
callee alternative, no conflicting capture row and no write invalidating a guard.
Recording only successful structural reachability cannot preserve that conclusion
when an edit introduces a new use.

The dependency region therefore includes both source and destination storage,
the owning constructor and implementation, actual/formal supplies, immutable
alias paths, captured slot/initializer/mode/ordinal, complete consumers and
crossed activations. Missing, duplicated or contradictory destination/capture
rows invalidate the affected relation. New incoming calls and opaque references
invalidate the earlier absence premises even when no existing node changes.

A scope-changing edit must retract affected facts, invalidate dependent layout
and code, resaturate through the owning nanopasses, and agree with a fresh check.
Alternative support and recursive groups must be handled explicitly. A diagnostic
serialization view is not a semantic compilation slice. Semantic dependency
regions, portable MLIR segments and linked object partitions have separate
contracts, as specified by the [incremental direction](Nanopass_Incremental_Contract_Direction.md).

## C-series delivery gates

The closure family is delivered when all required source forms preserve the
above contract through the supported compiler pathways. Acceptance includes:

- Canonical direct and materialized forms, with full function/environment
  transport for unknown callees and no packed interior or cast-repair route.
- Direct, pipeline, aliased, stored, partially applied, returned, recursively
  invoked and aggregate-held function values, with truthful call/return types.
- Immutable snapshots, shared mutable cells and independent runtime formations;
  nested capture forwarding retains actual values and cell identity.
- Sequence restart, exhaustion, short circuit, empty inputs, both fold frontiers,
  ordered append/collect and independently measured callback/result domains.
- Proven stack, region, program and dynamic lifetime choices where admitted by
  the platform; invalid source receives the specified located diagnosis.
- Source/graph and rejection checks, actual-occurrence Alex checks, declared
  proof gates, editor invalidation/projection, and unchanged native oracles.
- Missing-edge, contradictory-edge, added-consumer, new-unknown-alternative,
  mutable-write and scoped-edit cases that invalidate the relevant conclusion.

Use the [regression policy](Regression_Check_Policy.md) during implementation.
Shared recipe/analysis changes expand the affected set across their consumers.
A full C-feature closure includes its complete delivery gate; a focused passing
cohort never substitutes for that gate.

Required source forms that the current proof or representation cannot yet admit
remain explicit acceptance work. They are not permanent exclusions, and an
accurate rejection does not satisfy their positive gate. Keep test additions,
actual focused execution, broad integration and native/artifact realization as
distinct evidence in the [waypoint record](Language_Coverage_Waypoints.md).
