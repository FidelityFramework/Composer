# C-07: Sequence Operations

> **Status:** In-Progress. Core implementation tested; full acceptance remains open. Native
> 16a–f cover producer/consumer composition, callbacks, demand and scalar-Option
> transport; full-profile 16g validates program startup and repeated sequence use.
> Original16 and retained operation partials still have open acceptance gates. C-07 is not complete; see the dated waypoint for
> exact artifacts and current failures.
> **Criteria reconciled:** September 25, 2026; no new execution claim.
> **Sample:** [16_SeqOperations](../../samples/console/FidelityHelloWorld/16_SeqOperations).
> **Dependencies:** [C-06](C-06-SimpleSeq.md), [C-01](C-01-Closures.md),
> [C-02](C-02-HigherOrderFunctions.md); [C-05](C-05-Lazy.md) supplies related
> deferred-formation contracts.
>
> **Authority:** [Sequence representation](../../../clef-lang-spec/spec/seq-representation.md),
> [closure representation](../../../clef-lang-spec/spec/closure-representation.md),
> [sequence operations](../../../clef-lang-spec/spec/seq-operations-representation.md)
> and [delimited continuations](../../../clef-lang-spec/spec/dcont-representation.md).
> [Language coverage waypoints](../Language_Coverage_Waypoints.md) record tested
> implementation evidence and the C-06 dependency handoff. Historical Alex wrapper
> layout passes, stored function-address fields, fixed SSA formulas and imperative
> operation emitters in this PRD are superseded by the direction below.

The [shared C-series acceptance contract](C-Series-Acceptance.md) governs evidence,
profile scope and continuation, including the FPGA/Colibri and eBPF lessons for
preservation through actual emitted artifacts. Existing bounded implementation
is the starting point for the remaining composition gates.

## 1. Feature and semantic commitments

Sequence operations compose deferred producers and eager consumers over Clef
`seq<'T>`. Source NTU element, callback, accumulator and dimensional constraints
survive graph construction and proof settlement. Physical placement does not
replace those types with a universal integer width or an erased object carrier.

| Operation | Source contract | Required evaluation behavior |
|-----------|-----------------|------------------------------|
| `map` | `('T -> 'U) -> seq<'T> -> seq<'U>` | Lazily invoke the mapper once per pulled input element |
| `filter` | `('T -> bool) -> seq<'T> -> seq<'T>` | Pull until a predicate accepts an element or the input exhausts; preserve order |
| `append` | `seq<'T> -> seq<'T> -> seq<'T>` | Enumerate the first input to exhaustion, then the second |
| `collect` | `('T -> seq<'U>) -> seq<'T> -> seq<'U>` | Invoke the mapper once per outer element and fully enumerate that inner result before advancing the outer input |
| `take` | `int -> seq<'T> -> seq<'T>` | Yield at most the requested count; stop on earlier input exhaustion; never pull after the limit |
| `fold` | `('S -> 'T -> 'S) -> 'S -> seq<'T> -> 'S` | Consume immediately in order, preserving independent accumulator and element types |
| `iter` | `('T -> unit) -> seq<'T> -> unit` | Consume immediately and perform one ordered action per element |
| `exists` | `('T -> bool) -> seq<'T> -> bool` | Stop on the first true predicate; return false on exhaustion |
| `forall` | `('T -> bool) -> seq<'T> -> bool` | Stop on the first false predicate; return true on exhaustion |
| `tryHead` | `seq<'T> -> option<'T>` | Return the first successfully pulled value, or None on exhaustion, with no later pull |
| `tryPick` | `('T -> option<'U>) -> seq<'T> -> option<'U>` | Retain each chooser result once; stop on the first Some, or return None after exhaustion |

Constructing a producer evaluates its supplied expressions in source argument
order and preserves their values/captured storage. It does not run its deferred
iterator body or invoke callbacks prematurely. A consumer likewise evaluates its
supplied expressions before iteration; moving an argument into a loop must not
repeat its formation effects.

Each enumeration owns independent iteration state. Copies of sequence or callback
values preserve the identities of referenced mutable cells. Re-enumeration must
not copy another iterator's progress, deep-copy external mutable captures or
repeat an operand initializer that already ran at producer formation.

`take` checks its remaining count before asking the input for an element. Zero or
negative remaining count produces no pull. A shorter input completes normally;
this PRD does not import F#/CLR exception behavior. Filtering can require several
upstream pulls to produce one output, and a downstream limit must stop the entire
upstream demand chain at the correct point.

Empty inputs invoke no per-element callback. An input that performs effects and
then exhausts still performs those effects when a consumer actually pulls it.
No current value exists merely because an iterator was constructed; every read
requires the successful-pull premise for that exact iterator.

`exists` and `forall` return false and true respectively on empty input, without
calling the predicate. A decisive predicate result prevents every subsequent
pull and callback; computing the same final Boolean after extra effects is not
conforming behavior.

## 2. Canonical representation and current limits

Every sequence follows the full `(moveNext, env)` callable contract. Callback
values likewise retain their callable identity and environment. Function values
are not stored as untyped addresses inside sequence environments. A direct call
is available only when the graph establishes its exact callable identity; known
code identity alone does not supply a captured activation environment.

C-06 supplies typed frame slots, independent iteration state, explicit caller
storage and bounded parent-owned child regions. It also supplies exact sequence
origins for supported callable-half elision. C-07 must compose these contracts;
it must not create an independent `MapSeqLayout`/`FilterSeqLayout` subsystem in
Alex or infer layout by traversing source callback bodies.

A retained view descriptor is an unboxed physical storage value containing
address, offset, extent and stride information. Its backing storage needs an
admitted lifetime independently of the descriptor. Capturing an input template
or a callback does not extend either backing allocation's lifetime automatically.
Target placement determines field widths, alignments, offsets and complete
extents; no capture-count formula determines the resulting SSA count or layout.

The sequence-operations specification now derives physical fields and admitted
copies or borrows from the shared sequence and closure contracts. Operation
semantics and ownership requirements remain independent of a particular layout.
The canonical full callable remains the contract for function-valued parameters
and multiple possible origins. [Closure values captured by other computations](../Closure_As_Data.md)
sets out the graph identities and environment work for the known-callee case.
The first bounded materialized form is now present in source as described below;
neither the design document nor that source inventory is a native acceptance result.

Aggregate-current preservation also intersects with BAREWire zero-copy access.
The current scalar-Option work uses explicit owned snapshots and constructs
directly into established destinations. That realization must not become a
language rule requiring every value observation to copy. Borrow or transfer
elision needs joint allocation, lifetime, use and overwrite evidence in Baker;
Alex consumes the chosen realization. BAREWire's
[substrate contract](../../../BAREWire/docs/Substrate_Formalism.md) and
[dispatch temporal obligations](../../../BAREWire/docs/13%20Dispatch%20Regions.md#temporal-obligations)
supply the shared vocabulary. Repeated snapshot allocations also need future
space/capacity and reuse analysis for constrained targets; a proved activation
lifetime alone is not that bound.

## 3. Existing implementation inventory

This table distinguishes registration and graph construction from native support.
The [September 20 C-07 waypoint](../Language_Coverage_Waypoints.md#c-07-sequence-operations--implementation-waypoint-acceptance-open-2026-09-20)
records native 16a–g and their actual compiler artifacts. Its bounded successes
supersede the earlier blanket statements that native producer, consumer and
search conformance are all pending. Wider forms remain acceptance work.

| Area | Present in source | Remaining work |
|------|-------------------|----------------|
| `map`, `filter`, `collect`, `append` | Baker producer recipes snapshot eager arguments, construct typed generator-local capture references and use the shared iterator/delegation ingredient; 16a–d exercise bounded native composition | Extend retained/returned input and callback residence, full callable forms and aggregate captures; preserve the tested child-storage cases while extending `collect` |
| `take` | Public scheme and producer dispatch; fresh remaining count, positive-demand guard and decrement after success; bounded demand/native composition passes | Extend stored partials and wider source/storage forms without regressing nonpositive/exact/short-input or upstream post-yield traces |
| `iter` | Public scheme and consumer dispatch; callback/input snapshots precede shared ordered iteration; bounded native effects pass | Stored actions and wider callable/payload residence; preserve unit results, empty-input behavior and ordered effects |
| `fold` | Independent `<'S,'T>` source scheme, checked state type and eager operand snapshots; bounded native independent-state and empty cases pass | Both retained partial frontiers, aggregate/function state where admitted, general callback environments and source application forms |
| `exists`, `forall` | Predicate schemes and `iterateWhile` dispatch; native short circuit and empty results are recorded | Stored predicate operations and broader callable/storage composition; preserve both stopping polarities and exact callback counts |
| `length`, `toList`, `toArray` | Their accumulator construction now uses the same `seqFoldLeft` binding/while path as `fold`, replacing its former recursion; list/array conversion still goes through reversed-list construction and List operations | Own input effects, count/extent/range and materialization contracts plus native gates; the shared iteration change does not establish List allocation or array conversion conformance |
| `isEmpty` | Recipe makes one pull and negates its result | Establish native source/effect behavior; do not turn a zero-cut effectful input into a skipped pull |
| `head` | Public scheme and single-pull recipe, whose current read does not retain the successful-pull condition | Establish the source nonempty/result contract and exact current premise before native use |
| `min`, `max`, `minBy` | Public schemes and older recursive recipes with an initial current read | Retool exact iterator/current identity, initial nonempty admission, comparison/key typing and repeated callback evaluation |
| `tryHead`, `tryPick` | Public schemes and optional-selection recipes snapshot supplied operands, retain candidates once and use guarded `iterateWhile`; native 16e covers stopping, formation, independent dimensions and nested Option selection | Bare/stored operation values and broader aggregate/callable results; preserve 16e and 16f retention rather than treating search as unimplemented |
| `maxBy` | Dormant recursive recipe branch; the public Seq resolver does not register the name | Resolve its source contract and current/selection premises before advertising admission |
| Materialized callbacks | Known regular lambdas retain explicit `ClosureValue`/environment formation, scalar slots, an environment formal and actual-environment call operands; bounded native 16b composition is recorded | Returned/opaque/aggregate callables, mutable callable aliases, nested recaptures and full function/environment transport require further representation/residence gates |
| `Seq.empty` | Public polymorphic value and primitive designation | Keep its concrete lowering/admission distinct from the tested C-06 no-yield source generator |

The producer foundation is in
[SeqRecipes.fs](../../../clef/src/Compiler/Baker/Recipes/SeqRecipes.fs) and
[Sequences.fs](../../../clef/src/Compiler/Baker/Ingredients/Sequences.fs).
[Intrinsics.fs](../../../clef/src/Compiler/NativeTypedTree/Expressions/Intrinsics.fs)
owns source schemes;
[BakerSaturation.fs](../../../clef/src/Compiler/Nanopass/BakerSaturation.fs)
owns recipe dispatch and supplies the checked operand/result types.

The shared iterator ingredient already establishes an enumerator binding before
its loop, a guarded pull, and one current binding before the supplied unit action.
Its source/provenance relationships are reused by C-06 consumption and delegation.
`iterateWhile` adds a decision before that pull while retaining its exact iterator
and current prefix. The remaining recursive extremum recipes are
not equivalent evidence merely because their comments describe an iterator loop.
They must establish the same allocation, value-availability, current and capture
contracts through the shared ingredient or a separately admitted graph protocol.

## 4. Missing contracts and owning layers

### 4.1 Captured input template and backing storage

Existing producer snapshots capture the input sequence value. Ordinary scoped
delegation of a captured input is also a C-06 requirement. Its bounded
captured-template contract is implemented there through
[SequenceResidence](../../../clef/src/Compiler/PSGSaturation/SemanticGraph/SequenceResidence.fs):
proven activation coverage and finite borrow incidence must establish the
backing storage's availability. C-07 consumes that contract for callback-free
`append` and transformers; it must not build a competing lifetime analysis.

The resulting contract must connect the exact input allocation/template, producer
capture, fresh enumerator and their uses. An owned copy or child region needs
its placed extent and lifetime evidence; a retained view needs evidence that its
backing storage survives every admitted use. Caller-destination and parent-region
facts must retain allocation-occurrence identity, including factory-created
inputs and multiple enumerations. A permissive escape default is not a substitute
for that relationship.

### 4.2 Callback callable and environment

The current [closure environment admission](../../../clef/src/Compiler/PSGSaturation/SemanticGraph/ClosureEnvironments.fs)
and [formation recipe](../../../clef/src/Compiler/Baker/Recipes/ClosureEnvironmentRecipes.fs)
materialize a bounded known-callee form. A regular lambda captured by a sequence
must have resolved scalar capture declarations and complete admitted uses. Its
immutable aliases retain exact callable identity; mutable captured scalar cells
retain their storage identity. The current selection excludes missing/aggregate
capture sources, mutable callable aliases, opaque or returned callable uses and
nested recaptures of the same source frontier. This is an implementation boundary,
not a restriction on the normative closure model.

The source callable remains `TFun` with its original source identity and signature.
Its environment formation, capture accesses, lifted implementation formal and
actual-environment call operands are explicit graph relationships. Placement and
residence must establish that environment's backing storage at each use. The
generated code declaration is identified by those relationships; treating all
function-typed bindings as code would lose materialized environment values.

This does not supply an arbitrary `TFun` storage field or the complete unknown
callable pair. A function invoked with several possible callback or sequence
origins still requires the full callable contract when exact-origin elision is
unavailable. Literal and named callbacks, immutable aliases, callbacks returned
from factories and function-valued parameters need their own applicable evidence;
the first selected form does not establish every path.

For `collect`, the callback returns a sequence whose backing storage must survive
its delegated iteration and the outer suspension. Reusing C-06 caller destinations
and owned regions requires exact returned-origin and residence premises; merely
storing the returned descriptor is insufficient.

### 4.3 Demand and consumer state

`take` now constructs explicit short-circuit control in the graph: inspect remaining count,
pull only on the positive path, bind current after success, then update count and
yield. Retain the original count type/range premises and source formation effects.
This law must hold through a composed filter/map pipeline, including count zero
and an input that can continue indefinitely.

`fold` and `iter` now compose ordered consumer actions with the shared iterator
protocol. Folder state and element types remain independent; callback and input
snapshots precede enumeration. `exists`/`forall` retain their last Boolean decision
and admit another pull only while it has not become decisive. Native conformance
for bounded paths is recorded in 16a–g; retained operations and wider callable
forms remain open. `length`, list/array materialization and extremum operations
have additional accumulator, extent, nonempty or comparison contracts.
These are upstream graph and proof obligations, not new Alex loop emitters.

Search/current admission must cite the actual successful guard and iterator.
The current certification pass supports the shared while protocol. A recursive
or single-pull consumer requires its own equally explicit admission or a recipe
that produces the supported protocol. Empty/nonempty behavior must be decided
before allowing a current read; no default element can fill the gap.

## 5. Implementation direction

Use the following dependency order to guide work. It is not a rigid gate against
an independent consumer improvement whose premises are already settled.

1. **Stored and bare operation values through C-01/C-02.** Use unchanged
   `16h_SequenceApplications` as the first application gate. Elaborate each
   partial frontier with retained supplied values and formation effects; preserve
   public schemes, lexical binding and dimensions. Chasing an alias must not
   replay its initializer. Full callable use classification must account for
   every reachable use, including unresolved partials.
2. **Returned and retained environments.** Extend existing exact-origin and
   bounded residence paths to the factory/capture uses in unchanged original16.
   Carry actual environment instances through calls, returns and joins using the
   canonical callable contract. Callback-produced children and captured inputs
   need backing-storage evidence across their complete use, including outer
   suspension. Preserve 16a–g throughout this shared C-01/C-06 work.
3. **Aggregate results and storage budgets.** Generalize the established
   scalar-Option retention path to admitted aggregate/callable compositions with
   explicit copy/borrow/transfer choices, overwrite constraints, peak capacity
   and reclamation. Share C-04 storage contracts for materialization.
4. **Successor consumers and final cohort.** Complete the explicit successor
   rows in §8 after their source contracts and prerequisites are established.
   Run original16, 16a–h, affected C-06 inputs and ordinary controls on the final
   coordinated compiler/dependency/target cohort, with proof and tooling gates.

Each slice begins from a source/native oracle with fixed expected values and
effects, plus exact rejected premises at their owning layer. Complete the related
source, graph, Alex and peered-tooling work together. Do not use a standalone MLIR
module or a hand-emitted special case as source conformance.

## 6. Baker to Alex integration

Producer recipes create ordinary `SeqExpr`, generator, binding, application,
conditional, loop and yield nodes with exact source/provenance relationships.
Consumption and delegation use their shared ingredients. The existing Baker
sequence passes then settle ownership, evaluation, composed control, cut/resume
incidence, liveness, placement, residence and current-read admission. Recipe
fan-out/fold-in retains every semantic/reference participant and obligation.

The final graph publishes typed frames, origins, initializer/destination rows,
owned regions, representation meets and resident evidence. A layout proof does
not establish captured-storage lifetime, and known callable identity does not
prove the callback's environment available. Missing premises remain explicit
residuals at their owning layer.

Alex consumes these facts at Huet zipper positions. Existing witnesses pull the
declared child regions and existing patterns compose typed memory accesses,
function calls, `scf.if`, `scf.while` and `scf.index_switch`. Standard backend
lowering handles the resulting `func`/`memref`/`arith`/`index`/`scf` operations.
There is no source-shape recognizer, recursive subtree emitter, mutable semantic
state in the zipper or operation-specific imperative MoveNext builder.

These are the recorded native forms. Their reconciliation and any additional
operation/profile forms follow [M-01](M-01-DialectAdmission.md); its general
admission and evidence transport remain planned. The shared
[realization integrity criteria](C-Series-Acceptance.md#41-integrity-through-realization-colibri-fpga-and-ebpf)
apply to transformed outputs as well: demand guards, operand order, environment
identity and actual storage must correspond to Baker's claims after lowering.

## 7. Coverage commitments

The existing sample's feature coverage remains the acceptance target. Its
[original source fixture](../../samples/console/FidelityHelloWorld/16_SeqOperations/README.md)
is now packaged in the Clef native harness with executable semantics unchanged;
its factories, module-level formation and while recurrences remain an unpassed
gate. Keep expected behavior fixed and record actual artifacts in the waypoint.

| Coverage group | Required cases and representative expected results |
|----------------|----------------------------------------------------|
| C-06 inputs | Literal, loop, conditional, effectful empty and supported factory sequences |
| Basic map | Double `1..5` → `2,4,6,8,10`; square and offset transforms |
| Basic filter | Even/odd/threshold predicates; preserve accepted order; all-rejected input exhausts |
| Take boundaries | `take 3` of `1..100` → `1,2,3`; exact length; shorter input completes; nonpositive count does not pull |
| Fold | Sum `1..10` → `55`, product `1..5` → `120`, maximum and count; empty input returns initial state |
| Predicate consumers | Exists stops on first true and forall on first false; empty false/true, all-exhausted outcomes and no pull/callback after a decision |
| Independent NTU state | Measured input and differently measured fold state; inverse dimensions, real/scalar carriers and logical unit values where the source contract admits them |
| Composed operations | Filter/map in either order; first three doubled evens → `4,8,12`; sum of even squares through ten → `220` |
| Manual equivalence | Compare fold/iter consumers with ordinary `for ... in` for values and ordered effects |
| Captured callbacks | Scale, threshold, multiple immutable captures and mutable-cell identity; offset-ten fold of `1..5` → `65` |
| Collect | Two/three yields per element, captured multiplier, variable-length inner loops, empty inner sequences and independent outer/inner state |
| Captured composition | Each stage retains its own callback environment; filter above five, double, take five → `12,14,16,18,20` |
| Empty and singleton | No callback on empty input; singleton mapping; empty delegation effects; repeated enumeration |
| Deep composition | Four/five stages; filter multiples of five, double, filter above fifty, take five, fold → `400` |

Demand-sensitive tests must count upstream pulls and callback invocations, not
only compare final totals. Exercise the same producer twice and wrappers created
from one source, preserving shared external cell identity while separating
iteration state. Captured and parameterized functions must not receive credit
from tests that only use no-capture literals.

Native variants must include direct applications, pipeline applications and
admitted stored/partial operation values, as well as distinct formations sharing
one code identity. Track unsupported callable/value forms explicitly; a source
scheme or peered hover does not pass their native representation gate.

Negative cases retain exact compiler codes, severity and source ranges for wrong
callback argument/result dimensions, non-Boolean predicates, non-unit actions,
wrong sequence elements and invalid accumulator types. Missing lifetime,
nonempty/current or callable premises require the existing responsible diagnostic,
with source/node provenance where available; accepting any error is insufficient.

## 8. Completion record

The eleven operations in §1 define the core implementation. Full acceptance of
that core includes original16 and 16h, not only the recorded 16a–g successes.
The September 20 waypoint reports original16 factory/capture and residence
failures and a still-unpassed 16h staged-callable gate; these remain explicit
implementation work. No new native run is claimed here.

Other registered Seq names do not inherit core acceptance. Preserve the
following successor work under C-07 with explicit source contracts and gates;
a core checkpoint must not silently mark the full PRD Complete or waive rows.

| Successor surface | Prerequisites and native acceptance |
|---|---|
| `Seq.empty` | Its normative no-element behavior, independent instantiation and admitted value/application forms; distinguish the intrinsic from a tested source no-yield generator |
| `length` | Exact count and count range, effectful empty input, full exhaustion and no duplicate pull/callback |
| `isEmpty` | One demanded pull, including empty-body effects, and no subsequent pull after the answer |
| `head` | Establish the source nonempty/failure contract and exact successful-current premise before admitting payload extraction; no fabricated default element |
| `min`, `max`, `minBy` | Establish nonempty/current, comparison/key typing, evaluation/tie behavior and numeric/resource premises; exact effectful key evaluation and ordered traversal |
| `maxBy` | Public source admission and settled selection contract before native advertising; an existing dormant recipe is not a supported operation |
| `toList`, `toArray` | C-04 construction/storage, target authority, length/extent and peak capacity; preserve order, types, effects and retained output lifetime after input exhaustion |

Where a successor's observable behavior is not settled in the normative
operation chapters, reconcile that contract before implementation rather than
importing exception or tie behavior from another language. These rows expose
the work; they do not introduce new semantics through a test expectation.
C-05 lazy conformance and C-06's original coupled/multiplicative recurrence
gates remain independently tracked.

- [ ] Registered source schemes, recipes and supported operations agree; dormant
  recipe branches are not advertised as implemented primitives.
- [ ] Captured input templates, callbacks and callback-produced sequences have
  admitted backing storage, full callable representation where needed, and
  retained type/provenance/obligation participants.
- [ ] Native source oracles cover the groups above with exact output and demand
  effects, including composition and captures; expected results remain unchanged.
- [ ] CCS source/graph negatives and Alex component prerequisites pass, alongside
  real MLIR verification and standard target lowering.
- [ ] CCS.Editor, analyzer-facing and actual LSP gates preserve dimensional public
  signatures, original capture definitions, precise errors and unsaved repairs.
- [ ] Relevant earlier FidelityHello controls remain correct on the final compiler.
- [ ] Listed successor rows have their own settled contracts and recorded native
  gates; core and successor results remain distinguishable.
- [ ] The operations specification, this PRD and waypoint agree on actual support,
  open representation decisions and final artifact evidence.

[Closure nanopass architecture](../Closure_Nanopass_Architecture.md) and
[delimited continuations architecture](../Delimited_Continuations_Architecture.md)
govern this work alongside the normative chapters. Broader C-07 completion must
not be inferred from the earlier producer graph checkpoint or C-06 native pass.

## 9. Criteria supersession

The earlier native append/map/filter/consumer bring-up sequence is superseded by
the remaining-work order in §5. Its bounded implementations already have native
evidence. The earlier blanket pending-native statements are replaced by §3's
source inventory and the dated waypoint; the accepted producer/consumer laws and
original source oracles remain intact. This reconciliation changes no compiler
code, sample expectation or PRD completion status.
