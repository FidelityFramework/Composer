# C-04: Core Collections and Range Expressions

> **Status**: In-Progress | **Depends On**: C-01 (Closures), C-02 (Higher-Order Functions), C-03 (Recursion)
> **Existing samples**: `13a_SimpleCollections`, `13a_BAREWireCollections`; bounded Option oracles `08a`–`08e`.
> **Criteria realignment — 2026-09-25:** this document preserves the promised collection surface and the dated implementation evidence. It changes no compiler behavior and records no fresh test execution.

The [shared C-series acceptance contract](C-Series-Acceptance.md) applies to every
increment. Language semantics come from the current Clef specification, especially
[List](../../../clef-lang-spec/spec/list-operations-representation.md),
[Map](../../../clef-lang-spec/spec/map-representation.md),
[Set](../../../clef-lang-spec/spec/set-representation.md),
[Option](../../../clef-lang-spec/spec/option-operations-representation.md),
[memory regions](../../../clef-lang-spec/spec/memory-regions.md) and
[numeric selection](../../../clef-lang-spec/spec/numeric-selection.md).
[Language Coverage Waypoints](../Language_Coverage_Waypoints.md) records the
bounded native, graph and tooling results and their coordinated revisions.

This replaces the January implementation sketch and its conflicting checklists.
The old collection-as-closure layouts, source pointer representations, assumed
global arena, missing-file claims and vector-first plan are superseded. Their
operation promises remain in §3. Existing recipes and witnesses are implementation
assets to reconcile with the specification, not evidence of complete support.

## 1. Executive Summary

C-04 covers persistent Lists, Maps and Sets, Option operations, collection
range expressions, tuple binding/projection and the supporting operations in §3.
Map and Set completion is part of this PRD. Completing Option and a minimal List
sample does not complete C-04.

All forms obey the [evaluation strategy contract](../Evaluation_Strategy_Contract.md):
Clef is lazy by default, and unused ordinary operands retain their effects
deferred. Collection structure, payload demand and complete traversal are
separate obligations. A materialized native layout does not by itself authorize
eager evaluation. Historical strict-operand traces below require reconciliation
against the clarified source contract, not preservation as current semantics.

Acceptance requires source admission, semantic graph construction, settled
representation and storage obligations, physical witnessing, and native behavior
for the promised operations. The selected profile must supply the required
storage and numeric capabilities. Unsupported uses need located diagnostics in
their owning stage. A declared intrinsic signature or a passing component
fixture alone establishes neither usable source syntax nor native conformance.
An accurate rejection of a valid, promised operation is still an implementation
gap: it cannot close that operation's positive gate. Capability limits must be
explicit and justified by the selected contract, not introduced to remove a
difficult case from the promised surface. C and F are peer delivery obligations;
the dependency order here is an engineering sequence, not a priority ranking.

### 1.1 Why This PRD?

These operations let applications express immutable state, searches, transforms
and folds without hand-written storage loops. BAREWire's current shared source
uses arrays and explicit loops; its former Map/Set call counts and boxing
refactoring notes describe a replaced implementation. Its codec, schema and
layout workloads remain useful acceptance consumers, with independent byte and
failure oracles. See §12 for the present integration boundary.

### 1.2 Design Philosophy

- Preserve concrete NTU element, key, value and state types, dimensions and
  source identities through specialization and recipe expansion.
- Settle algorithm, evaluation order, capture, comparison, storage and lifetime
  relationships in CCS/Baker. Alex observes those relationships at the actual
  Huet position and composes admitted physical operations.
- Preserve persistent versions through same-arena structural sharing and
  lifetime joins. Node construction writes fresh storage; it does not mutate
  an earlier version or the sentinel.
- Derive widths, extents, alignments and capacity requirements from the graph
  and selected platform. Optimization requires its own semantic justification
  and target admission.

## 2. Type Definitions and Semantic Contracts

### 2.1 List<'T>

A list node is the `Empty | Cons` aggregate specified by the List representation
chapter: tag, head payload and arena-relative tail index. A List value selects a
node in its settled arena. `List.empty` denotes index 0, a valid sentinel;
`List.isEmpty` tests its Empty tag. `List.tail []` is `[]`, since the sentinel's
tail indexes itself. This corrects the former runtime-failure description.

`List.head` is admitted only when the `Cons` test on the same node dominates the
application on the saturated graph. The compiler must not invent a test with a
runtime failure arm to discharge that requirement. A literal construction may
settle the case. Traversal recipes establish their own case guards before
reading payloads. The same nonempty requirement applies to the partial List
selectors named in the specification.

### 2.2 Map<'K, 'V>

A Map is a persistent AVL tree with key, value, left/right arena-relative indices
and height. The sentinel has height 0, self-indexing child links and unread
payload slots. Every modification preserves earlier roots, shares unchanged
subtrees, maintains AVL balance, and uses the admitted comparison for `'K`.
Traversal yields keys in comparison order. Replacement of an existing key keeps
one entry; deletion must handle leaf, one-child and two-child cases without
losing unrelated entries.

### 2.3 Set<'T>

A Set uses the corresponding persistent AVL representation without a separate
Map value slot. It preserves uniqueness, comparison order and prior versions.
Mapped values may collide: `Set.map` must restore uniqueness and ordering under
the result element comparison. Union, intersection and difference must respect
the same storage and lifetime requirements as single-tree operations.

### 2.4 NTUKind Extensions

The current type algebra already contains collection constructors. Its
[source](../../../clef/src/Compiler/NativeTypedTree/NativeTypes.fs) and the
[Native Type Universe](../../../clef-lang-spec/spec/native-type-universe.md)
are the starting points; this PRD does not instruct implementers to add a
second List/Map/Set universe or equate those types with an untyped pointer.

### 2.5 Type Constructor Arity Principle

List, Set and Option have one payload parameter; Map has independent key and
value parameters. Fold state is independent of collection payload. Instantiation
must remain fresh at each polymorphic use, including bare aliases and stored
partials. Tests must mix distinct types and dimensions in one reachable program
and reject mismatched callback/state/result types at the source location.

### 2.6 Option

Option retains the specification's native tagged-value contract: None and Some
are distinct cases, with a platform-selected addressable tag and a concrete
payload representation. None is not an absent interior pointer. Its HOFs expand
through Baker case recipes; callable payloads and state retain their separate
function boundaries and C-01 lifetime obligations.

Demand follows the operation and selected case. An unselected fallback or
callback expression remains deferred, including its initializer effects.
`defaultWith` and `orElseWith` demand their producer only for a demanded None
branch. Case tests do not force unused payloads; mapping a payload retains its
shared deferred computation until that payload is needed. A demanded `iter`
performs its selected action. Stored partials retain established values or shared
deferred identities without replay; mutable captured storage retains its original
cell. `forall None` is true; `exists None` is false. Both folds retain independent
state/payload types and leave an unused folder or state undemanded.

`Option.get` remains in scope as the specified payload extraction primitive.
Its absent-case admission must be recorded explicitly before claiming complete
source support; this PRD does not retain the old invented exception behavior or
authorize reading an unselected payload. The same requirement applies to the
success precondition of `Map.find`, whose operation inventory alone does not
settle missing-key behavior. Any unresolved contract belongs in the specification
and CCS admission work before a native implementation is accepted.

### 2.7 Range Expressions

Promised syntax includes `[first .. last]`, `[first .. step .. last]`, the
corresponding array forms, and the range producer consumed by `seq { ... }`.
List and array ranges retain their respective collection contracts and demand
behavior; sequence realization belongs jointly with
[C-06](C-06-SimpleSeq.md) and [C-07](C-07-SeqOperations.md).

Acceptance covers ascending, descending, singleton, empty and stepped cases,
endpoint evaluation order, lexical range-operator identity, and exact element
count and byte extent. Zero step and unsupported numeric forms must receive the
owning specified outcome or admission diagnostic before division, allocation or
iteration. Count arithmetic and the final induction step cannot overflow simply
because the mathematical range is valid. Element dimensions, progression and
representation must agree under the current numeric contract; there is no
source-level fixed machine-width default in this plan.

The existing named, closed, unstepped integer `for value in first .. last`
normalization is useful implementation, but its September 20 gate does not
establish materialized ranges, stepped ranges, general enumerable loops or
per-iteration closure identity. See §8.

## 3. CCS Intrinsics and Promised Operation Inventory

The tables preserve the PRD's operation scope. They are acceptance obligations,
not a claim that every signature, recipe or executable already works. Syntax and
resolved operation identity govern elaboration; an old proposed internal
`Range.*` helper name does not require a new public API.

### 3.1 List Intrinsics

| Operation | Signature / result | Required behavior |
|---|---|---|
| `List.empty` | `'T list` | Valid sentinel; no per-use allocation |
| `List.isEmpty` | `'T list -> bool` | Empty case test |
| `List.head` | `'T list -> 'T` | Existing same-node Cons guard required |
| `List.tail` | `'T list -> 'T list` | Tail link; empty tail is empty |
| `List.cons`, `::` | `'T -> 'T list -> 'T list` | Fresh node sharing the tail in its arena |
| `List.length` | `'T list -> int` | Exact cardinality |
| `List.rev` | `'T list -> 'T list` | Reversed order |
| `List.append`, `@` | `'T list -> 'T list -> 'T list` | Left elements followed by right; sharing/lifetime contract retained |
| `List.map` | `('T -> 'U) -> 'T list -> 'U list` | One result per input, correct order and callback effects |
| `List.filter` | `('T -> bool) -> 'T list -> 'T list` | Stable retained subsequence |
| `List.fold` | `('S -> 'T -> 'S) -> 'S -> 'T list -> 'S` | Left fold, independent state type |
| `List.foldBack` | `('T -> 'S -> 'S) -> 'T list -> 'S -> 'S` | Right fold with its distinct argument order |
| `List.tryHead` | `'T list -> 'T option` | None for empty; Some head otherwise |
| `List.tryFind` | `('T -> bool) -> 'T list -> 'T option` | First match with short-circuit behavior |
| `List.forall` | `('T -> bool) -> 'T list -> bool` | True for empty; stop at first false |
| `List.exists` | `('T -> bool) -> 'T list -> bool` | False for empty; stop at first true |

List literals and cons patterns are part of this inventory. They must elaborate
to the same constructors and guards as explicit operations.

### 3.2 Map Intrinsics

| Operation | Signature / result | Required behavior |
|---|---|---|
| `Map.empty` | `Map<'K,'V>` | Valid sentinel |
| `Map.isEmpty` | `Map<'K,'V> -> bool` | Height-zero test |
| `Map.add` | `'K -> 'V -> Map<'K,'V> -> Map<'K,'V>` | Insert or replace, preserve other versions and AVL balance |
| `Map.remove` | `'K -> Map<'K,'V> -> Map<'K,'V>` | Remove present key; preserve absent-key contents and rebalance |
| `Map.tryFind` | `'K -> Map<'K,'V> -> 'V option` | Typed hit or miss |
| `Map.find` | `'K -> Map<'K,'V> -> 'V` | Success contract resolved as required by §2.6 |
| `Map.containsKey` | `'K -> Map<'K,'V> -> bool` | Membership under the same comparison |
| `Map.count` | `Map<'K,'V> -> int` | Distinct-key cardinality |
| `Map.keys` | `Map<'K,'V> -> 'K list` | Keys in comparison order |
| `Map.values` | `Map<'K,'V> -> 'V list` | Values in matching key order |
| `Map.toList` | `Map<'K,'V> -> ('K * 'V) list` | Ordered key/value pairs |
| `Map.ofList` | `('K * 'V) list -> Map<'K,'V>` | Repeated insertion with specified duplicate-key replacement |
| `Map.map` | `('K -> 'V -> 'U) -> Map<'K,'V> -> Map<'K,'U>` | Preserve keys; transform values |
| `Map.filter` | `('K -> 'V -> bool) -> Map<'K,'V> -> Map<'K,'V>` | Retain matching entries with a valid persistent tree |
| `Map.fold` | `('S -> 'K -> 'V -> 'S) -> 'S -> Map<'K,'V> -> 'S` | Fold in comparison order |

### 3.3 Set Intrinsics

| Operation | Signature / result | Required behavior |
|---|---|---|
| `Set.empty` | `Set<'T>` | Valid sentinel |
| `Set.isEmpty` | `Set<'T> -> bool` | Height-zero test |
| `Set.add` | `'T -> Set<'T> -> Set<'T>` | Insert with uniqueness and persistence |
| `Set.remove` | `'T -> Set<'T> -> Set<'T>` | Correct removal/rebalance for every child shape |
| `Set.contains` | `'T -> Set<'T> -> bool` | Membership |
| `Set.count` | `Set<'T> -> int` | Distinct-element cardinality |
| `Set.union` | `Set<'T> -> Set<'T> -> Set<'T>` | Elements in either operand |
| `Set.intersect` | `Set<'T> -> Set<'T> -> Set<'T>` | Elements in both operands |
| `Set.difference` | `Set<'T> -> Set<'T> -> Set<'T>` | Elements in the first operand only |
| `Set.isSubset` | `Set<'T> -> Set<'T> -> bool` | Set containment, including empty operands |
| `Set.toList` | `Set<'T> -> 'T list` | Comparison-ordered elements |
| `Set.ofList` | `'T list -> Set<'T>` | Construct with duplicates removed |
| `Set.map` | `('T -> 'U) -> Set<'T> -> Set<'U>` | Re-establish result ordering and uniqueness |
| `Set.filter` | `('T -> bool) -> Set<'T> -> Set<'T>` | Persistent subset |
| `Set.fold` | `('S -> 'T -> 'S) -> 'S -> Set<'T> -> 'S` | Fold in comparison order |

### 3.4 Option Intrinsics (Enhancement)

| Operation | Signature / result | Required behavior |
|---|---|---|
| `None`, `Some` | `'T option`, `'T -> 'T option` | Concrete case construction and matching |
| `Option.map` | `('T -> 'U) -> 'T option -> 'U option` | Transform Some payload |
| `Option.bind` | `('T -> 'U option) -> 'T option -> 'U option` | Return selected callback result |
| `Option.defaultValue` | `'T -> 'T option -> 'T` | Demand fallback only for selected None result |
| `Option.defaultWith` | `(unit -> 'T) -> 'T option -> 'T` | Invoke producer only for None |
| `Option.orElse` | `'T option -> 'T option -> 'T option` | Demand fallback option only for selected None result |
| `Option.orElseWith` | `(unit -> 'T option) -> 'T option -> 'T option` | Invoke optional producer only for None |
| `Option.iter` | `('T -> unit) -> 'T option -> unit` | Invoke action once for Some; retain unit result |
| `Option.fold` | `('S -> 'T -> 'S) -> 'S -> 'T option -> 'S` | None retains state; Some applies folder to state then payload |
| `Option.foldBack` | `('T -> 'S -> 'S) -> 'T option -> 'S -> 'S` | None retains state; Some applies folder to payload then state |
| `Option.filter` | `('T -> bool) -> 'T option -> 'T option` | Retain Some when predicate holds |
| `Option.exists` | `('T -> bool) -> 'T option -> bool` | False for None |
| `Option.forall` | `('T -> bool) -> 'T option -> bool` | True for None |
| `Option.isSome`, `Option.isNone` | `'T option -> bool` | Case tests |
| `Option.get` | `'T option -> 'T` | Payload extraction; admission boundary in §2.6 |
| `Option.toList` | `'T option -> 'T list` | Zero or one elements under the List storage contract |

### 3.5 Additional Small Intrinsics

The original supporting promises remain: comparison-based `min` and `max`, pair
projections `fst` and `snd`, `Array.blit`, and separator-based `String.concat`
over a string list. Acceptance includes type/dimension preservation, evaluation
order, array extent and overlap behavior, empty inputs and exact output capacity.
Their owning standard contracts govern details; no BAREWire-specific emitter is
required. `Array.map`, `Array.fold`, `Array.init`, `Array.sum` and `Array.sumBy`, present in the former
optimization plan, also need semantic coverage when used as collection consumers;
SIMD selection is separately governed by §11.

### 3.6 Tuple Destructuring in Let Bindings

Pairs, larger tuples, nested tuple patterns, parentheses and wildcards must bind
the correct components while evaluating the right-hand side once. Bindings retain
their individual types, dimensions, source identities and lexical scope. Shape
or type mismatches require located diagnostics; an unsupported pattern must not
silently bind `_` or disappear.

The old `Bindings.fs` pseudocode is not a current defect diagnosis. Inspect the
existing [binding checker](../../../clef/src/Compiler/NativeTypedTree/Expressions/Bindings.fs)
and [pattern checker](../../../clef/src/Compiler/NativeTypedTree/Expressions/Patterns.fs)
before changing their owning graph construction. BAREWire RoundTrip already
contains reachable tuple-return/destructuring consumers; broader nested-pattern
coverage still needs its own evidence.

### 3.7 Range Surface

| Historical helper family | Required source capability | Acceptance owner |
|---|---|---|
| `Range.toList`, `Range.toListStep` | Inclusive list ranges, with implicit or explicit step | C-04 |
| `Range.toArray`, `Range.toArrayStep` | Inclusive array ranges, with implicit or explicit step | C-04 |
| `Range.toSeq`, `Range.toSeqStep` | Corresponding lazy range producer | C-04 source semantics; C-06/C-07 generator, traversal and lifetime |

The inventory retains all six intentions without requiring those historical
internal helper spellings. Range operators must retain lexical identity and
the progression rules of the language.

### 3.8 Additional Normative Families and Roadmap Boundaries

The standard specifies more than the original minimum inventory. These are
explicit follow-on acceptance rows, not implied by a family name being present:

| Family | Additional operations to account for |
|---|---|
| List | `List.collect`, `List.reduce`, `List.contains`, `List.tryPick`, `List.minBy`, `List.maxBy`, `List.min`, `List.max`, `List.last`, `List.forall2`, `List.sum`, `List.sumBy`, `List.average`; iteration and `List.toSeq`/`List.ofSeq` integration retain their own acceptance |
| Map | `Map.toSeq`, `Map.iter`, `Map.forall`, `Map.exists`, `Map.ofSeq`, `Map.ofArray` |
| Set | `Set.isSuperset`, `Set.forall`, `Set.exists`, `Set.iter`, `Set.toSeq`, `Set.toArray`, `Set.ofSeq`, `Set.ofArray`, `Set.singleton` |
| Option | `Option.map2`, `Option.map3`, `Option.flatten`, `Option.toArray`; `Option.toNullable`/`Option.ofNullable` require an explicit boundary/profile contract |

Record each row's intended increment and evidence before making a full-family
conformance claim. Sequence conversions share C-06/C-07 acceptance; nullable
boundary conversion does not introduce interior null. These extensions do not
remove Map, Set or any §3.1–§3.7 promise from C-04 completion.

## 4. Implementation Strategy

The [Baker contract](../../../clef/docs/fidelity/Baker_Saturation_Architecture.md)
and [Alex overview](../Alex_Architecture_Overview.md) govern the implementation.
Much of the machinery exists:

| Current owner | Existing implementation | Remaining acceptance boundary |
|---|---|---|
| Source typing and specialization | [Intrinsics](../../../clef/src/Compiler/NativeTypedTree/Expressions/Intrinsics.fs), [NativeTypes](../../../clef/src/Compiler/NativeTypedTree/NativeTypes.fs), [Monomorphization](../../../clef/src/Compiler/Nanopass/Monomorphization.fs) | Reachable source forms, independent schemes, complete types and located rejection for every claimed operation |
| Baker decomposition | [ListRecipes](../../../clef/src/Compiler/Baker/Recipes/ListRecipes.fs), [MapRecipes](../../../clef/src/Compiler/Baker/Recipes/MapRecipes.fs), [SetRecipes](../../../clef/src/Compiler/Baker/Recipes/SetRecipes.fs), [OptionRecipes](../../../clef/src/Compiler/Baker/Recipes/OptionRecipes.fs) | Correct algorithms, operation-specific demand and sharing, guarded payload reads, graph/reference incidence, layout/lifetime/obligations and complete residuals |
| Reusable ingredients | [Primitives](../../../clef/src/Compiler/Baker/Ingredients/Primitives.fs), [Patterns](../../../clef/src/Compiler/Baker/Ingredients/Patterns.fs), [Options](../../../clef/src/Compiler/Baker/Ingredients/Options.fs) | Sentinel/node/traversal forms and recipe laws agree with the current standard |
| Alex physical expression | [CollectionPatterns](../../src/MiddleEnd/Alex/Patterns/CollectionPatterns.fs), [ListWitness](../../src/MiddleEnd/Alex/Witnesses/ListWitness.fs), [MapWitness](../../src/MiddleEnd/Alex/Witnesses/MapWitness.fs), [SetWitness](../../src/MiddleEnd/Alex/Witnesses/SetWitness.fs), [OptionWitness](../../src/MiddleEnd/Alex/Witnesses/OptionWitness.fs) | Consume settled facts and emit the admitted sentinel/arena form; no source-name algorithm selection or invented storage premise |
| Integer range normalization | [LoopRanges](../../../clef/src/Compiler/Nanopass/LoopRanges.fs), [LoopRangeRecipes](../../../clef/src/Compiler/Baker/Recipes/LoopRangeRecipes.fs) | Preserve the existing bounded loop subset; extend materialized and stepped forms only with their own contracts |

The List witness already handles primitives and rejects surviving HOF operations
that require Baker decomposition. Map/Set recipes and witnesses also exist.
Current CollectionPatterns still constructs several empty/node values through
generic DU construction, and Set deletion contains a simplified merge path.
Those source observations identify work to inspect and replace; they are not a
fresh execution result or proof of the specified sentinel representation and AVL
deletion. Recipe-file existence must never substitute for algorithm acceptance.

For each increment, fix the first missing or incorrect fact in the pipeline.
CCS/Baker owns new algorithm structure and obligation participants. Alex can
extend physical expression for an admitted form but cannot reconstruct a tree
algorithm, allocate an unexplained arena, or repair a missing guard while
emitting. SSA names are derived by current Alex machinery; the January
preassignment and alternate-transfer sketches are not implementation directions.

## 5. Representation and Storage Acceptance

| Obligation | Required criterion |
|---|---|
| Sentinel residence (VC-RES) | One immutable program-lifetime image per concrete collection instantiation, citing the selected platform's named authority through graph evidence |
| Hosting-arena floor | Initialize the zero sentinel slot at arena creation; its size covers the largest hosted node. Bump allocation begins at the floor; reset returns to the floor |
| Sentinel immutability (VC-RO) | No subsequent node write can target the sentinel; prohibited writes produce the owning diagnostic, including CCS8020 where specified |
| Arena links (VC-LINK) | Links are bounded arena-relative indices into the value's settled arena. Store-site capacity/range obligations hold; loads inherit them; index 0 is valid |
| Payload guards (VC-GUARD) | Reads are dominated by the List Cons or tree nonzero-height fact for the same node; rotation guards satisfy the specified height implications |
| Persistent sharing | Fresh path copies preserve prior versions and share unchanged nodes within one arena. Derivation edges settle the covering lifetime; no cross-arena link or implicit copying substitutes for an unavailable lifetime home |
| Extent and capacity | Node layouts, alignments, padding, counts, floor and total live allocation agree with the selected target and actual emitted storage; no wrapped arithmetic or guessed capacity |
| Callback boundary | Traversal callbacks preserve C-01/C-02 evaluation, partial-application, capture and lifetime contracts; shared mutable state invalidates stale range facts when required |

BAREWire's `ProgramLifetime` designation names existing immutable and optional
mutable spaces; it does not grant a generic storage fallback. Static sentinel
images use immutable authority. Runtime writes, initialization of a hosting
arena and any mutable program-lifetime storage need their own admitted authority
and realization. A source `let` alone does not authorize image writes. See
[Platform integration](../../../Fidelity.Platform/docs/CANONICAL_PLATFORM_SPEC.md)
and [BAREWire's storage planner](../../../BAREWire/src/Platform/StaticStorage.fs).
The emitter must use the actual settled placements against which evidence was
produced. A missing space, insufficient capacity or unplaceable lifetime remains
an explicit rejection.

### 5.1 Preservation Through Lowering

Collection allocation, access and representation contracts remain obligations
through backend transformations. The handoff must retain the relationship
between source/graph nodes, admitted operations, concrete storage and generated
artifacts. A transform that changes placement, addressing, widths, access order
or allocation must preserve the affected claim or produce new evidence for it.
Graph settlement or successful MLIR verification alone does not establish that
the final artifact implements those facts.

The [FPGA verification workstream](../fpga-targeting/README.md) and
[M-01](M-01-DialectAdmission.md) provide the same integrity discipline for
Colibri circuit realization, hardware artifacts and target admission. C-04 uses
that discipline now: inspect actual emitted allocations, offsets, loads/stores,
branch guards and backend output, then test correspondence with the retained
graph evidence. On a hardware or verifier-gated profile, add the selected
target's artifact/gate obligations rather than promoting a CPU result to
cross-target support. These target plans do not claim completed hardware or BPF
collection support.

## 6. Completion Sequence

1. Preserve the accepted Option/default/iteration/fold behavior and its graph,
   tooling and native oracles. Resolve the remaining partial-operation admission
   contracts and inventory gaps without widening claims from those successes.
2. Establish the common sentinel image, hosting-arena floor, link, guard,
   capacity and lifetime contracts in the owning graph and selected profile.
   Exercise multiple collection types in one arena and independent arenas.
3. Bring List construction, matching and traversals through that representation;
   then complete the List inventory and Option-to-List integration.
4. Complete persistent Map/Set primitives, comparisons, all rotations, updates,
   deletion and the promised traversals/transforms/set algebra. Keep earlier
   versions live in tests so mutation or dropped subtrees cannot pass unnoticed.
5. Complete collection ranges, tuple/supporting operations and bounded real-library
   consumers under the default demand contract. Join sequence range/conversion
   acceptance with C-06/C-07.
6. Reconcile every promised operation with exact accepted forms, negative cases,
   supported profiles and retained evidence. Update the standard only where an
   actual unresolved contract was settled, then the PRD and waypoints together.

These are implementation slices, not permission to leave Map/Set or difficult
negative cases outside the final C-04 gate.

## 7. Required Acceptance Cases

| Area | Positive and adversarial cases |
|---|---|
| Source and specialization | Direct calls, pipes, aliases, stored partials, records/functions/unit/measured payloads; wrong arity/type/dimension, shadowed operation names and unresolved reachable types |
| Evaluation | Unused effectful arguments/fallbacks/callbacks produce no trace; repeated demand of one binding produces one trace; demanded mutable reads preserve observation order; case-tag tests leave payloads deferred; ignored fold state is not forced; distinct fold/foldBack order and state types |
| Lists | Empty/singleton/many, repeated tail of empty, guarded head, unguarded-head rejection, stable map/filter/append, short-circuit search and prior-version sharing |
| Maps/Sets | Sorted/reverse insertion, duplicate insert/replace, all four AVL rotations, deletion of absent/leaf/one-child/two-child/root/last entry, order and height consistency, retained earlier roots, Set-map collisions and all empty set-algebra combinations |
| Storage | Missing immutable/mutable authority, exact capacity and one-node excess, alignment, floor-preserving reset, attempted sentinel mutation, cross-arena sharing and escaping lifetimes; actual placement correspondence |
| Ranges | List/Array materialization and sequence demand boundaries, positive/negative steps, singleton/zero-trip, zero step, unsupported types, shared endpoint/step computations with correct demand, count/byte overflow and final-step limits |
| Tuples/supporting operations | One shared RHS computation, demanded projections and undemanded wildcard payloads, nested bindings, lexical shadowing, arity/type failure, array-copy overlap/bounds, empty and capacity-limited string concatenation |
| Graph and emission | Resident guard/derivation/residence identities survive recipe replacement; no unresolved HOF residual reaches a primitive witness; stock verification and native results agree with independent oracles |
| Backend correspondence | Actual artifact storage/access agrees with graph evidence after transforms; changed widths, placement, guards or artifacts cannot reuse an unrelated earlier verdict |

Use the shared acceptance contract for compiler hashes, coordinated revisions,
native exit/output checks, diagnostic location/severity, editor projections and
artifact correspondence. Graph proof, verifier success, hosted reference output
and native execution establish different boundaries; retain each required gate.

## 8. Validation

The following are **recorded September 2026 results**, not tests rerun during
this document realignment. Their detailed logs, revisions and limitations remain
in [Language Coverage Waypoints](../Language_Coverage_Waypoints.md). Older Clef
hashes follow that document's repository-history mapping.

The September 26 owner clarification establishes lazy-by-default ordinary
arguments and bindings. Any historical test requiring eager fallback/operand
effects is evidence of the old implementation only. Migrate its expectation with
the governing demand clause and owning implementation change; retain all valid
typing, identity, lifetime, guard and selected-effect assertions.

| Dated increment | Recorded implementation and evidence | Limit |
|---|---|---|
| September 19 Option defaults | 311/311 CCS tests; nine native callback executables; `08a_OptionDefaults` and `08b_OptionDefaultWith` exact-output gates; graph, Alex, editor/analyzer and LSP checks | Historical eager-fallback traces are superseded by the clarified demand contract; partial identity and unit-formal coverage remain bounded evidence, not full closure residence or collection completion |
| September 20 Option alternatives | 395/395 CCS tests including alternatives and temporal range regressions; four native executables; `08c_OptionAlternatives` | May-write invalidation and saved-predicate observation timing matter to callback correctness; no complete mutable-cell claim |
| September 20 Option iteration | 419/419 CCS tests; 25/25 Alex component cases; two native executables; `08d_OptionIteration` | Includes unit-valued conditional results; does not establish all collection callbacks |
| September 20 Option folds | 482/482 CCS tests; 15 native fold groups; `08e_OptionFolds`; editor/analyzer/LSP projection | Independent state/payload, both partial frontiers and graph dominance for shared state; other families remain open |
| September 20 counted bounds and integer range loops | Four native counted-loop groups; later 536/536 CCS checkpoint and four native unstepped range-loop groups | Bound ordering and named closed unstepped integer-loop normalization only; not general materialized/stepped ranges |

The existing [13a SimpleCollections sample](../../samples/console/FidelityHelloWorld/13a_SimpleCollections/)
and [13a BAREWireCollections sample](../../samples/console/FidelityHelloWorld/13a_BAREWireCollections/)
remain useful source fixtures, but their presence is not a passing native gate.
Preserve independent expected output while adapting reachable source to the
current standard. Do not shrink a regression selection or rewrite an oracle to
make an unsupported collection family appear complete.

No current full List/Map/Set sentinel, lifetime and native acceptance is claimed
by this PRD. The preserved Option results are substantial delivered behavior;
the common representation and remaining operation inventory are still work.

## 9. Implementation Checklist

- [ ] Every §3.1–§3.7 promise has a source form, owning semantic contract,
  implementation location, native oracle and rejection cases; additional §3.8
  families are explicitly accounted for. A diagnostic for a valid promised use
  remains an open positive gate.
- [ ] Common sentinel residence, initialization, floor, read-only protection,
  links, guards, capacity and lifetime joins are graph-resident and preserved.
- [ ] List traversal and complete persistent AVL behavior pass the §7 cases,
  including old-root preservation and two-child deletion.
- [ ] Remaining Option and partial-operation admission contracts are settled;
  the dated defaults, alternatives, iteration and fold regressions still pass.
- [ ] Ranges and tuple/supporting operations retain exact types, dimensions,
  operand order and bounded storage on each claimed profile.
- [ ] Reachable library consumers pass fresh native and applicable differential
  gates; declaration-only/library compilation is never substituted.
- [ ] Backend artifact correspondence preserves the actual allocation, access
  and representation obligations after each relevant lowering transform.
- [ ] The shared acceptance gates and affected existing regressions pass with
  coordinated compiler/platform/library identities and retained evidence.
- [ ] Documentation reports accepted scope and remaining boundaries accurately;
  C-04 is marked complete only after all promised rows meet their criteria.

## 10. Relationship to Other PRDs

| Owner | Relationship |
|---|---|
| C-01 / C-02 | Callable identity, captures, partial application, environment representation and callback lifetime |
| C-03 | Recursive traversal, mutual tree helpers and justified stack/tail behavior |
| C-05 | Default call-by-need and explicit Lazy payloads share representation obligations; explicit memoization, persistent collection structure and fresh sequence enumeration retain distinct contracts |
| C-06 / C-07 | Range generators, collection/sequence conversion, repeat traversal and consumer lifetime |
| A-04 | Wider region surface; required collection lifetime/capacity facts cannot be postponed behind a global arena |
| M-01 | Target-specific physical-form admission and preservation, including any demanded vector realization |
| F-09 | Result callbacks share case-recipe machinery; their successful gates do not establish untested Option/List/Map/Set operations |

## 11. Optimization and Anti-Patterns

Persistent structure permits sharing; it does not alone prove callback purity,
parallel safety or associative arithmetic. A Map/filter/fold callback may have
effects. Numeric reductions retain the specified arithmetic construction,
rounding and result semantics. Vectorization, parallel scheduling and suspension
are separate admitted transformations with their own evidence under
[M-01](M-01-DialectAdmission.md) and the owning contracts.

Existing vector templates may be reused when a demanded expression/profile
supports them. An architecture name is not evidence of an instruction-set
extension, a fixed lane count or admissible reassociation. Scalar semantic
acceptance does not require a new vector witness, and a vector-looking MLIR
module does not establish collection semantics.

Prohibited shortcuts include using source addresses as collection links,
allocating an undeclared process-wide arena, runtime absence checks in place of
the sentinel, semantic decomposition in Alex, mutation of persistent nodes,
host boxed/BCL collections as the native realization, and integer-width guesses
that bypass NTU settlement. Missing graph facts remain failures in their owning
stage; a second emitter is not a remedy.

## 12. Notes on BAREWire Integration

Use the current [intersection-subset record](../../../BAREWire/docs/12%20Intersection%20Subset.md)
and [RoundTrip workload](../../../BAREWire/samples/RoundTrip/Main.clef), not the
January list of blockers. Generic codecs, typed option outcomes, tuple returns,
schema traversals and layout/extent calculations exercise useful composition
boundaries. Migrate a selected `SUBSET` workaround only after its preferred form
passes the supported source and execution gates; preserve the library's shared
.NET/Fable/Composer behavior and wire bytes.

The native gate compiles a fresh reachable executable, requires successful exit
and compares an independent transcript. BAREWire's historical September 3 native
success and later September 6 failure refer to their recorded compiler snapshots;
neither is a new verdict on the current checkout. Hosted checks and metadata-only
compilation do not establish native acceptance. There is no requirement to change
BAREWire's public array-based declaration schemas solely to exercise List or Map.

Selected-platform declaration identity, capacities and storage authority must
survive integration. A proved declaration model is not evidence that independently
placed generated objects obey it. Keep native correspondence and failure behavior
in the acceptance gate alongside successful library output.
