# C-07: Sequence Operations

> **Status:** Planned native completion; existing source schemes and Baker
> recipes provide part of the implementation. C-07 is not complete.
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

The sequence-operations specification still contains historical requirements
for ordinal wrapper fields, whole-environment inline copies and fixed SSA costs.
Those sections require explicit reconciliation with the current base sequence
and closure contracts as C-07 representation support is implemented. This PRD
preserves their operation semantics and ownership intent; it does not treat an
unsettled storage choice as already admitted. The canonical full callable remains
the contract for function-valued parameters and multiple possible origins.

## 3. Existing implementation inventory

This table distinguishes registration and graph construction from native support.
The C-06 native gate alone does not complete these operation paths.

| Area | Present in source | Remaining work |
|------|-------------------|----------------|
| `map`, `filter`, `collect`, `append` | Baker producer recipes snapshot eager arguments, construct typed generator-local capture references and use the shared iterator/delegation ingredient | Consume the bounded captured-template contract implemented in C-06; settle remaining escaping/opaque input and callback environments; validate native composition |
| `take`, `iter` | Public NTU schemes | No corresponding Seq recipe dispatch; add graph recipes with the required demand/effect order |
| `fold`, `length`, `toList`, `toArray` | Older recursive consumer recipes | Retain one initialized iterator, exact current admission and eager inputs through the settled consumer protocol |
| `fold` state | Public scheme quantifies independent `<'S,'T>` | Baker dispatch currently passes no state type and the recipe falls back to element type; preserve the actual accumulator type |
| `isEmpty` | Recipe makes one pull and negates its result | Establish native source/effect behavior; do not turn a zero-cut effectful input into a skipped pull |
| `head`, `min`, `max`, `minBy` | Consumer recipes | Initial successful-pull evidence is absent where current is read; settle nonempty and comparison/key premises before native use |
| `exists`, `forall`, `tryHead`, `maxBy` | Recipe branches | Public Seq resolver does not register these names; admission and implementation remain separate work |
| `forall` | Older Boolean-fold recipe | Its predicate polarity is wrong for the helper's true-branch short circuit; correct before exposing the operation |
| `tryPick` | Public scheme and recursive recipe | Bind each chooser result once and preserve short-circuit/current admission |
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
The older recursive consumers are not equivalent evidence merely because their
comments describe an iterator loop. They must establish the same allocation,
value-availability, current and capture contracts, through that ingredient or a
separately admitted graph protocol.

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

Continuation placement currently supports scalar fields and typed sequence,
array and record views; it does not supply a general `TFun` frame field.
[SequenceMachineRecipes](../../../clef/src/Compiler/Baker/Recipes/SequenceMachineRecipes.fs)
recognizes some declaration bindings as symbolic code. That distinction must not
elide the environment of a callback with captures.

C-01/C-02 closure ingredients must establish callback formation, source capture
identities, parameter/result types, environment storage and call representation.
Literal and named callbacks, aliases, callbacks returned from factories and
function-valued parameters can have different admission requirements. A helper
called with several possible callback or sequence origins still requires the
full callable contract when exact-origin elision is unavailable.

For `collect`, the callback returns a sequence whose backing storage must survive
its delegated iteration and the outer suspension. Reusing C-06 caller destinations
and owned regions requires exact returned-origin and residence premises; merely
storing the returned descriptor is insufficient.

### 4.3 Demand and consumer state

`take` needs explicit short-circuit control in the graph: inspect remaining count,
pull only on the positive path, bind current after success, then update count and
yield. Retain the original count type/range premises and source formation effects.
This law must hold through a composed filter/map pipeline, including count zero
and an input that can continue indefinitely.

`fold` and `iter` should compose ordered consumer actions with the shared iterator
protocol. Folder state and element types remain independent; callback and input
snapshots precede enumeration. `length`, list/array materialization and search
operations have additional accumulator, extent, nonempty or stopping contracts.
These are upstream graph and proof obligations, not new Alex loop emitters.

Search/current admission must cite the actual successful guard and iterator.
The current certification pass supports the shared while protocol. A recursive
or single-pull consumer requires its own equally explicit admission or a recipe
that produces the supported protocol. Empty/nonempty behavior must be decided
before allowing a current read; no default element can fill the gap.

## 5. Implementation direction

Use the following dependency order to guide work. It is not a rigid gate against
an independent consumer improvement whose premises are already settled.

1. **Native append and captured input residence.** Exercise the existing producer
   and delegation recipes without adding callback storage. Apply the captured
   template/backing relationship and fresh iterator ownership, then validate
   delayed input effects, empty prefixes/suffixes, repeated enumeration and
   supported factory inputs. Preserve the original left/right eager argument
   order and separate suspension owners. This slice applies C-06's bounded
   contract for captured templates; escaping and opaque input uses remain explicit
   residuals until their additional lifetime premises are admitted.
2. **Native map/filter and callback composition.** Add the missing callable/
   environment contract through C-01/C-02 ingredients, retaining immutable and
   mutable captures. Cover no-capture and captured callbacks, aliases and admitted
   parameter/factory paths; then extend `collect` with returned child residence.
3. **Counted demand and eager consumers.** Add `take` through the shared producer
   structure and its guarded demand relation. Retool `fold`/`iter` around the
   settled iterator/action protocol, fixing independent state typing and eager
   input order. Extend short-circuit and materializing consumers only with their
   own type, storage and proof premises.

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

## 7. Coverage commitments

The existing sample's feature coverage remains the acceptance target. Its legacy
source fixture must be brought into the current Clef native harness; its presence
in the repository is not a passing result. Keep expected behavior fixed and
record actual artifacts in the waypoint.

| Coverage group | Required cases and representative expected results |
|----------------|----------------------------------------------------|
| C-06 inputs | Literal, loop, conditional, effectful empty and supported factory sequences |
| Basic map | Double `1..5` → `2,4,6,8,10`; square and offset transforms |
| Basic filter | Even/odd/threshold predicates; preserve accepted order; all-rejected input exhausts |
| Take boundaries | `take 3` of `1..100` → `1,2,3`; exact length; shorter input completes; nonpositive count does not pull |
| Fold | Sum `1..10` → `55`, product `1..5` → `120`, maximum and count; empty input returns initial state |
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

Negative cases retain exact compiler codes, severity and source ranges for wrong
callback argument/result dimensions, non-Boolean predicates, non-unit actions,
wrong sequence elements and invalid accumulator types. Missing lifetime,
nonempty/current or callable premises require the existing responsible diagnostic,
with source/node provenance where available; accepting any error is insufficient.

## 8. Completion record

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
- [ ] The operations specification, this PRD and waypoint agree on actual support,
  open representation decisions and final artifact evidence.

[Closure nanopass architecture](../Closure_Nanopass_Architecture.md) and
[delimited continuations architecture](../Delimited_Continuations_Architecture.md)
govern this work alongside the normative chapters. Broader C-07 completion must
not be inferred from the earlier producer graph checkpoint or C-06 native pass.
