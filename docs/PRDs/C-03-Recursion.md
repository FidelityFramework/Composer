# C-03: Recursive Bindings and Tail Calls

> **Sample**: `13_Recursion` | **Status**: In-Progress | **Depends On**: [C-01](C-01-Closures.md), [C-02](C-02-HigherOrderFunctions.md)
>
> **Criteria realignment, 2026-09-25.** Existing recursive binding and capture
> work is retained. This revision replaces the January implementation sketches
> with current acceptance criteria; it records no new compiler execution or pass.

## 1. Executive Summary

Complete source recursion through the existing CCS/Baker/Alex pipeline: self
recursion, nested recursive functions, mutually recursive function groups and
their admitted callable uses. Recursive references must retain binding identity,
types, capture identity and evaluation effects through lowering. Tail-recursive
forms must receive the stack behavior their admitted contract requires.

The [C-series acceptance contract](C-Series-Acceptance.md) governs evidence and
status. The [recursive expression specification](../../../clef-lang-spec/spec/expressions.md#recursive-definition-expressions),
[recursive inference rules](../../../clef-lang-spec/spec/inference-constraint-solving.md)
and [closure representation](../../../clef-lang-spec/spec/closure-representation.md)
govern semantics. A passing recursive example is bounded evidence, not completion
of the whole source-recursion surface.

## 2. Current State and Remaining Boundary

The [language coverage waypoints](../Language_Coverage_Waypoints.md) are the
execution record. They establish the September 19 immutable direct-capture
increment, including recursive forwarding, shadowing, nested capture identity
and collision-free target symbols. They also record the original sample 13's
unresolved generic integer-width failure. That failure remains an open native
gate; the direct-capture fixture does not replace it.

Current source already contains the following foundations:

| Foundation to retain | Owning implementation / evidence |
|---|---|
| Recursive binding identities available before body checking | [Bindings.fs](../../../clef/src/Compiler/NativeTypedTree/Expressions/Bindings.fs), with module-level group handling in [NativeService.fs](../../../clef/src/Compiler/NativeTypedTree/NativeService.fs) |
| Nested named-function capture discovery, excluding own parameters and self | Existing `computeCaptures` use in `Bindings.fs` |
| Baker-owned immutable direct capture form and recursive forwarding | [ClosureRecipes.fs](../../../clef/src/Compiler/Baker/Recipes/ClosureRecipes.fs), [DirectCaptures.fs](../../../clef/src/Compiler/PSGSaturation/SemanticGraph/DirectCaptures.fs), September 19 waypoint |
| Target symbols derived from resolved local binding identity | The same waypoint's callable-symbol and native gates |
| Source-reference/capture remapping through fold-in | September 19 fold-in reference identity waypoint |

This inventory does not establish acceptance of all recursive groups, mutable
captures, escaping recursive functions or width/range recurrences. Resume from
the first unsettled relationship in a failing case. Do not recreate the old
missing-NodeId or missing-capture fixes as if they were unimplemented.

## 3. CCS Implementation

CCS and Baker own the following source and graph contracts:

1. Every member of a recursive function group has a stable identity available
   while checking every body in that group. Self and peer references resolve to
   that identity; shadowed names and independent local functions stay distinct.
2. Recursive uses impose the current type and measure constraints. Group
   generalization follows the specification; independent uses cannot accidentally
   share a solved type variable, and captures do not erase generalizable measures.
3. Capture discovery uses resolved definitions. An outer variable referenced by
   a nested recursive function is retained; a same-named local parameter is not
   mistaken for that capture. Module bindings remain direct references.
4. Baker establishes the complete-use callable form. Eligible nonescaping named
   functions pass settled captures as additional parameters; recursive calls
   forward those same participants. Escaping, returned, stored, partially applied
   or higher-order uses retain the C-01/C-02 callable and lifetime obligations.
5. Mutable capture groups preserve one shared cell where the source shares one.
   The [direct capture cell contract](../Direct_Capture_Cell_Contract.md) owns the
   remaining representation, residence and call-effect requirements; copying a
   mutable value into recursive arguments changes its semantics.
6. Recursive effect/range analysis accounts for the whole call group and all
   relevant writes. It must reach a sound fixed point or retain a located
   unresolved obligation. A caller's earlier bound must not survive a recursive
   mutation without justification. A source `int` never acquires a convenient
   hardcoded width to make sample 13 compile.

Keep source ranges, graph/reference incidence, capture origins and obligation
participants through elaboration and fold-in. Missing required facts are diagnosed
by their owning stage before an unsupported witness is attempted.

## 4. Composer Implementation

Alex observes the settled function, callable identity, operands and child regions
through its existing context and Huet position. It composes the admitted physical
form with Elements/Patterns/Witnesses; it does not discover captures or a recursive
group by walking names or rebuilding the source algorithm.

The [current architecture](../Architecture_Canonical.md) and
[Alex overview](../Alex_Architecture_Overview.md) replace the old
`SSAAssignment`, parent-chain naming and direct `llvm.func` recipes. SSA identities
derive from witnessed graph roles and block arguments. Target realization belongs
to the selected backend, under the
[operation/pathway contract](../../../clef-lang-spec/spec/backend-lowering-architecture.md).

The selected target can change the admitted call or tail form, but cannot change
the settled source semantics. Trace the actual emitted calls, arguments, storage
and cleanup to the graph participants and declared target facts. Recheck affected
obligations when a downstream transformation changes their premises; a source
proof or successful verifier result does not certify the final realization. The
shared acceptance contract applies this correspondence discipline to each claimed
pathway without making every target's deployment a C-03 completion prerequisite.

For each admitted tail form, identify the source tail position, cleanup/resource
requirements and selected lowering. Check that argument evaluation, capture
forwarding and return behavior survive the transformation. The
[stack-overflow contract](../../../clef-lang-spec/spec/special-attributes-and-types.md#stack-overflow)
and [list-operation requirements](../../../clef-lang-spec/spec/list-operations-representation.md)
must not be reduced to an assumption that LLVM happens to optimize the call.
Provide structural evidence and a sufficiently deep native oracle for the claimed
bounded-stack behavior. Non-tail recursion retains its declared stack/resource
requirements; compiler saturation termination does not establish program
termination or a bound on runtime recursion.

## 5. Recursive Groups and Unsettled Source Contracts

Mutually recursive functions belong to this PRD's acceptance surface. Test peer
calls, different logical argument lists, shared outer captures and independent
group instances. Direct-capture conversion must consider the group's complete
uses; one escaping or opaque use cannot be ignored when selecting a form.

Recursive **value initialization** is a separate source case. The inherited
[recursive safety analysis](../../../clef-lang-spec/spec/inference-constraint-solving.md#recursive-safety-analysis)
still describes lazy initialization, runtime self-reference checks and an F#
exception outcome, while Clef's [error model](../../../clef-lang-spec/spec/error-handling.md)
uses native values and explicit diagnostics. Record and reconcile the native
initialization/failure contract before claiming that case. Do not import the
managed exception mechanism or count function-recursion tests as value-initialization
coverage. Unsupported initialization forms need explicit, located rejection.

This is a specific specification seam, not a reason to postpone the settled
function-group, capture, range or tail-call work. C-03's status must keep the
unadmitted initialization surface visible when reporting its accepted scope.

## 6. Verification

Apply the [shared gates](C-Series-Acceptance.md) to the following matrix:

| Boundary | Required evidence |
|---|---|
| Binding and group identity | Self/peer references resolve before body completion; shadowing and duplicate local names retain distinct definitions through fold-in and target symbol emission |
| Source schemes | Independent ordinary/measured uses; exact type/dimension rejection with diagnostic code, effective severity and source range |
| Captures and callables | Nested immutable captures, recursive forwarding, mutually recursive shared captures, mutable cell identity, and each claimed returned/stored/partial/HOF use |
| Effects and ranges | Recursive writes invalidate stale observations; sound recurrence settlement or the responsible unresolved-range diagnostic |
| Physical realization | Settled call operands/signatures and regions survive portable witnessing, MLIR verification and the selected backend |
| Tail behavior | Eligible self and mutual tail forms preserve effects and cleanup; claimed bounded-stack behavior has structural and native evidence |
| Native regression | Original `13_Recursion` compiles and executes with its intended results, including factorial 5 = 120 and sumTo 10 = 55; focused fixtures supplement it |
| Unsupported boundaries | Invalid cycles/initialization and unavailable callable, lifetime or representation forms fail at the responsible source/settlement boundary |

Run the applicable surrounding closure, HOF, collection and sequence regression
gates because they reuse recursive construction and capture facts. Record the
actual compiler/dependency revisions and the admitted profile. No editor, graph,
verifier or native gate substitutes for another.

## 7. Implementation Checklist

Existing foundations above are retained; the unchecked items below are acceptance
work, not a declaration that every underlying mechanism is absent.

- [ ] Reproduce and resolve sample 13's generic integer-width boundary in the
      owning inference/range/representation stage, retaining the source contract.
- [ ] Establish the self/nested/mutual function-group matrix, preserving existing
      recursive identities, capture recipes and target symbols.
- [ ] Admit remaining mutable, returned/stored and partial recursive callable
      forms with C-01/C-02, or retain their explicit residual scope.
- [ ] Establish recursive effect/range convergence and exact negative diagnostics.
- [ ] Establish the required tail-call/stack behavior for each claimed form.
- [ ] Resolve or explicitly delimit recursive value initialization against the
      native specification, without hiding its unsupported cases.
- [ ] Run and record the full applicable gates; update the waypoint and PRD status
      only for the scope the evidence establishes.

## 8. Related PRDs

- [C-01: Closures](C-01-Closures.md): capture identity, residence and callable forms.
- [C-02: Higher-Order Functions](C-02-HigherOrderFunctions.md): recursive functions
  passed, stored, returned or partially applied.
- [C-04: Core Collections](C-04-CoreCollections.md): recursive recipes and tail folds.
- [C-05: Lazy](C-05-Lazy.md): memoization and the recursive-initialization seam.
- [C-06: SimpleSeq](C-06-SimpleSeq.md): recursive calls crossing suspension/lifetime boundaries.
