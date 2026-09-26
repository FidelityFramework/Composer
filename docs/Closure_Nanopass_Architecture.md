# Closure Architecture

**Design synchronization: 2026-09-26.** The [closure specification](../../clef-lang-spec/spec/closure-representation.md)
governs semantics and representation. [C-01 §14](PRDs/C-01-Closures.md#14-the-closure-saturation-form-family)
and the [settlement contract](Closure_Settlement_Contract.md) describe delivery
obligations. [Closure values as data](Closure_As_Data.md) records the bounded
implementation; [waypoints](Language_Coverage_Waypoints.md) record validation.
This document does not declare the full closure family complete.

## 1. Executive Summary

A capturing function carries a function value and one flat environment. Each
capture is reached directly through its own slot. Capturing another function
value preserves that value's environment; this does not permit replacing lexical
capture slots with a chain of parent environments. An immutable captured value
may contain references whose sharing and lifetime still require proof.

A capture-free function needs no environment. A named nonescaping function uses
leading capture parameters when its complete uses establish that form. A known
materialized callee can elide the function half while retaining the actual
environment occurrence. Unknown alternatives prevent this elision.

## 2. Layer Responsibilities

| Layer | Responsibility |
| --- | --- |
| CCS checking | Resolve lexical bindings, types, capture sets and mutability. |
| Baker / owning CCS nanopasses | Elaborate calls and formation frontiers; settle capture access, effects, ranges, layout, lifetime and obligations through recipes and fan-out/fold-in. |
| Alex | Pull settled facts at the actual Huet occurrence and compose Elements through Patterns and Witnesses. |
| Backend | Realize the admitted portable form for the target and preserve or recheck affected properties. |

CCS does not preassign Alex SSA identifiers. `Alex.Traversal.Values` derives
names; the accumulator records emitted operands and scope associations separately
from the immutable zipper. Historical `PSGElaboration/SSAAssignment` diagrams and
code-pointer-in-environment sketches do not describe the current architecture.

## 3. Memory Layout

### 3.1 Flat Closure Structure

```text
(fn, env)
fn  = function value, or elided for a proved known callee
env = [capture slot 0 | padding | capture slot 1 | ... | tail padding]
```

The function value is never a field of `env`. Extent includes target-declared
alignment and padding; it is not merely the sum of payload sizes. Captured
scalar widths come from settled ranges and declared representations. The
portable materialized environment is a byte `memref` with a settled extent.
The full unknown-callee form requires separate function/environment transport;
a packed pair or an unrealized cast cannot stand in for that work.

### 3.2 Capture Modes

| Binding | Capture | Required behavior |
| --- | --- | --- |
| Immutable | Value | Snapshot once at formation; references inside the value retain sharing. |
| Mutable | Typed cell view | Read and write the same storage instance as every other capture of the binding. |
| Ref cell held immutably | Ref-cell value | Copy the reference, preserving the referenced cell. |

### 3.3 Extended Struct Layouts

Lazy values and sequences add their specified memoization or suspension slot
classes to the environment. Each slot class has an initialization and transition
discipline. State/current slots do not authorize eager evaluation of a sequence
body, and a finite frame does not establish its backing storage's residence.
See the [lazy](../../clef-lang-spec/spec/lazy-representation.md) and
[sequence](../../clef-lang-spec/spec/seq-representation.md) specifications.

## 4. Why Flat: the Finiteness Lemma

The finite capture set supplies an enumerated **direct obligation frontier**.
Once types and platform representations settle, its offsets, extents, alignment,
initialization and direct lifetime constraints can be expressed over those
participants. Unresolved premises remain explicit until commitment.

The field list does not bound arbitrary transitive storage reachable through a
capture. It does not alone prove liveness, reference validity, a unique release,
complete use, or decidability of every program property. An emitted proposition
is not a discharged proof. These qualifications follow
[closure representation §11](../../clef-lang-spec/spec/closure-representation.md#11-proof-extraction-at-closure-sites).

Foreign crossings additionally require declared ABI, registration, invocation,
quiescence and release contracts as applicable. Enumerated participants and a
flat layout make those obligations expressible; they do not establish them.
Source `nativeptr` is not an escape hatch. Internal membrane plumbing and typed
`FnPtr`/`CHandle` boundaries remain governed by the
[foreign boundary specification](../../clef-lang-spec/spec/ffi-boundary.md).

## 5. Coeffect — ClosureLayout

Current materialized facts are `EnvironmentLayout`, `EnvironmentOrigins`,
`KnownCallables`, explicit capture/formal incidence and admitted residence. The
layout contains slots, extent, alignment and resident obligations, not SSA names.
Legacy `Codata.Closures` remains an implementation boundary to reconcile; its
presence does not establish full canonical closure support.

Knowing the implementation and layout owner does not select a runtime instance.
Two formations of one lambda can have different environments. Witnesses recall
the actual occurrence and its settled carrier. Shared graph bodies must retain
their actual Huet path and local parameter associations.

## 6. Pipeline Flow

```text
source checking and lexical capture facts
  -> Baker callable/formation recipes and fan-out/fold-in
  -> effect, range, capture and result-destination relationships
  -> complete-use residence + layout + source admission
  -> Alex actual-occurrence ctx pull
  -> portable operations and regions
  -> target realization and native/source-independent validation
```

Representation preparation supplies a requirement. A result destination is not
by itself a proof that every retained reference outlives the result. An allocation
in a returning activation is not repaired by labeling it nonescaping. The
owning lifetime analysis must prove covering storage before Alex can allocate or
initialize it. Missing evidence prevents commitment.

## 7. FFI Boundary

A Clef closure is not interchangeable with a native callback pointer. The foreign
signature determines explicit `FnPtr` and context handling. Boundary adapters
retain actual source state and declared storage/representation facts; native
addresses are not recovered from numeric values or a packed closure interior.

## 8. Function values and C-series delivery

`let saved = selected` snapshots the selected function value at that point,
including when `selected` is mutable. It must not become a forwarding function
that rereads `selected` at invocation. Partial applications likewise evaluate
supplied operands once, in source order, and each later application has its own
frontier. Returned function expressions are separate callable boundaries;
syntactic declaration currying does not erase them.

The `16h` work exercises stored sequence applications and retained environments.
C-series acceptance also requires arbitrary specified returned, unknown,
recursive, aggregate-held and foreign-held function forms under their governing
contracts. Tests cover source semantics, graph/evidence rejection, actual-position
Alex behavior, editor projection and unchanged native oracles. Every discovered
blind spot is implementation work included in that delivery gate.

## 9. References

- [Closure settlement contract](Closure_Settlement_Contract.md)
- [Closure values as data](Closure_As_Data.md)
- [C-01 closure PRD](PRDs/C-01-Closures.md)
- [Baker architecture](../../clef/docs/fidelity/Baker_Saturation_Architecture.md)
- [Scope-aware incremental direction](Nanopass_Incremental_Contract_Direction.md)
- [Regression check policy](Regression_Check_Policy.md)
