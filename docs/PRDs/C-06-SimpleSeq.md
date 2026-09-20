# C-06: Simple Sequence Expressions

> **Status:** Native continuation core implemented; aggregate regression gate remains open.
> **Samples:** FidelityHello `15a_SequenceSemantics`, `15b_SequenceElements`,
> `15c_SequenceTemplateBorrows`, NativeSequences.
> **Dependencies:** C-01 closure/capture contracts, C-05 deferred-value context;
> shared producer ingredients also serve C-07.
>
> **Authority:** [Sequence representation](../../../clef-lang-spec/spec/seq-representation.md),
> [closure representation](../../../clef-lang-spec/spec/closure-representation.md)
> and [delimited continuation representation](../../../clef-lang-spec/spec/dcont-representation.md)
> govern representation. This PRD describes implementation and its remaining
> acceptance work. [Language coverage waypoints](../Language_Coverage_Waypoints.md)
> record actual gate evidence. Historical inline function-address layouts,
> Alex yield scans and imperative state-machine emitters are superseded.

## 1. Feature and source laws

`seq { }` describes a deferred, resumable computation. Each successful pull
produces one value; exhaustion returns false. `for ... in` consumes these pulls.

```fsharp
let multiplesOf factor count = seq {
    let mutable i = 1
    while i <= count do
        yield factor * i
        i <- i + 1
}
```

The source contracts are:

- Creating the sequence preserves capture formation and eager operand evaluation;
  it does not execute the deferred generator body.
- A pull follows source evaluation order until a yield or exhaustion. Resuming
  continues after that yield, including the remainder of the loop body before
  the next guard evaluation.
- Every yield in one sequence constrains that owner's element type. Nested
  sequences have independent owners. `yield!` supplies a sequence of the same
  element type, retaining NTU dimensions and type constraints.
- Each enumeration starts with independent iteration state. Immutable captures
  retain their creation-time values; mutable captures retain the original cell
  identity. Re-enumeration does not clone those external cells or undo their
  effects.
- Current is valid only after a successful pull of the exact iterator. A sequence
  with no yields may still execute effects when pulled; it does not fabricate
  a current element.
- Source `seq` recognition honors lexical binding. An ordinary function, lambda
  or lazy body does not inherit an enclosing sequence's delimiter. Unsupported
  computation-expression forms are diagnosed at source admission.

These are Clef semantics, with no CLR enumerable object, interface dispatch,
`obj` widening, null sentinel or denotable pointer plumbing.

## 2. Representation and ownership boundary

The canonical sequence value is `(moveNext, env)`: a function value and its typed
storage environment. The function address is not a data field in the environment.
The full callable contract remains applicable when a consumer needs both values.
Current native lowering elides the callable half only when Baker supplies the
exact generator origin for that use. An unresolved or mixed origin is not license
for Alex to invent a symbol, layout or erased carrier.

The environment contains the settled state discriminant, a current slot when
needed, captured values/cell views and values whose storage must survive a cut.
The source `TSeq<'T>` retains its NTU element type. It does not itself prescribe
an environment extent or the width of every stored integer. Baker's target
placement, range facts and adaptation meets determine the physical fields.

Persistent frame storage and activation scratch are distinct. A value needed
between generated dispatch regions during one pull can have a scratch slot
without surviving suspension. Immutable as well as mutable values can require
persistent storage when live across a cut. A retained mutable-cell borrow can
extend storage lifetime beyond the outer body's last scalar read.

A stored view descriptor retains physical address, offset, extent and stride.
It is an unboxed storage value, not a runtime type object, tag or boxed scalar.
Its layout and cost are target facts. Neither a fixed five-word footprint nor
an assumption that optimization erases descriptor fields is a language rule.

## 3. Baker pipeline and retained contracts

The implementation composes existing ingredients and recipes through nanopass
fan-out/fold-in. It preserves source identities, types, ranges, captures,
reference participants and provenance when it introduces graph nodes. The driver
orders these passes and publishes their results; semantic construction belongs
to their Baker recipes.

| Stage | Current owner | Output and ordering requirement |
|-------|---------------|---------------------------------|
| Source checking | CCS computation checking | Actual sequence owner before body checking; one fresh element constraint per owner; real typed generator formal with a source-point anchor |
| Producer formation | Shared sequence ingredient and producer recipes | Eager operand snapshots, deferred body references and actual `SeqGenerator` nodes; ordinary producer calls retain their public source ranges |
| Consumption | `SequenceConsumption` / `SequenceConsumptionRecipes` | Iterator creation, guarded pulls and an immutable source loop declaration with real identity |
| Delimiter ownership | `SequenceOwnership` / `SequenceOwnershipRecipes` | Joint owner/generator relation to each owned suspension site; rerun after delegation |
| Delegation | `SequenceDelegation` / `SequenceDelegationRecipes` | Owner-local iterator binding, while/moveNext, current binding and yield; original `yield!` identity/range retained as a unit wrapper |
| Element ranges | `SequenceElements` / `SequenceElementRecipes` | Exact successful-pull, owner and payload incidence for the range fixed point; unknown alternatives prevent narrowing |
| Evaluation | `SequenceEvaluation` / `SequenceEvaluationRecipes` | Ordered operand demands, entry/completion ports, conditional branches, backedges and deferred formation boundaries, after final curry normalization |
| Control and liveness | `SequenceControlRecipes` | Composed control occurrences, uses/definitions, successor transfers, cuts, resume entries and live-across sets |
| Storage | `Placement`, `SequenceResidence`, `SequenceRegions` | Exact persistent/scratch slots, extents and alignments; allocation residence; bounded child regions in owning frames |
| Machine construction | `SequenceMachineRecipes` | Ordinary typed graph for Boolean MoveNext: state/frame accesses, local control dispatch, yield stores and exhaustion |
| Resident evidence | `SequenceContinuationEvidence`, `ContinuationObligationRecipes` | Checked cut/resume/liveness incidence, bounded discriminant obligations and layout obligations, retaining their graph participants |
| Final publication | `SequenceRuntime`, CCS codata construction | Settled maps, current-read admission, representation meets and diagnostics before Alex |

Local evaluation ports alone do not establish global dominance or liveness.
Control composition must establish value availability and definite assignment
before synthesis. The persisted resume discriminant names cuts; the generator's
local dispatch position can name additional control occurrences within one pull.
These are separate identities, not syntactic yield numbering performed by Alex.

Machine elaboration replaces the generator body through the existing recipe
contract. The real environment formal remains attached to its generator. Internal
point anchors must not displace the user's `seq<'T>` hover or the original
capture declaration's navigation target.

Relevant implementation entry points:

- [SequenceRuntime.fs](../../../clef/src/Compiler/Nanopass/SequenceRuntime.fs)
- [SequenceControlRecipes.fs](../../../clef/src/Compiler/Baker/Recipes/SequenceControlRecipes.fs)
- [SequenceMachineRecipes.fs](../../../clef/src/Compiler/Baker/Recipes/SequenceMachineRecipes.fs)
- [SequenceContinuationEvidence.fs](../../../clef/src/Compiler/Baker/Recipes/SequenceContinuationEvidence.fs)
- [SequenceRegions.fs](../../../clef/src/Compiler/Nanopass/SequenceRegions.fs)
- [Meets.fs](../../../clef/src/Compiler/PSGSaturation/SemanticGraph/Meets.fs)

## 4. Settlement published to Alex

The following are current compiler-internal contracts, not new source syntax.

| Fact | Meaning |
|------|---------|
| `ContinuationFrames` | Owner, generator and formal identities; state/current identities; typed placed persistent/scratch slots; extents/alignment; resume states and resident obligation references |
| `SequenceOrigins` | Exact generator owner for a value/use/formal, supporting typed carriers and proven callable-half elision |
| `ContinuationStorage` | Exact scratch allocation/reference to its owner |
| `SequenceInitializers` | Constructor-occurrence-specific capture slot/value pairs in established evaluation order |
| `SequenceDestinations` | Constructor occurrence to an explicitly supplied caller-owned destination |
| `ContinuationRegions` | Allocation occurrence to parent owner/formal, child owner and exact byte offset/extent/alignment |
| `SequenceCurrentReads` | Reads admitted by the successful-pull premise for the corresponding iterator |
| `Escapes`, `Meets` | Settled allocation residence and source/slot representation adaptations |

`FrameRead`, `FrameWrite` and `FrameBorrow` name actual storage and slot identities.
A mutable capture reads or writes through its retained cell descriptor. Borrowing
an inline scalar slot returns its typed cell view; borrowing a captured cell
returns the stored descriptor. A buffer value does not acquire mutation rights
through a witness conversion.

Baker's layout obligation records actual slot participants and finite placement.
Cut/resume evidence records source owner/generator, yield payload, resume target,
live values and generated machine participants. The existence of these rows is
not a universal proof of lifetime, effects or complete continuation correctness.
Each obligation retains its owning analysis/discharge boundary. Unsettled control,
origin, residence or current-read prerequisites produce compiler diagnostics;
Alex cannot discharge them by guessing a representation.

`Meets.continuations` derives width adaptations from the supplied frame/origin/
storage maps and graph ranges without forcing codata during its construction.
Reads, writes, current extraction and dispatch results consume those meets.
Source integer widths, dimensional constraints and signedness are not replaced
with a blanket `i32` or target-word convention.

## 5. Construction, enumeration and bounded residence

A constructor initializes state and its exact capture set. It does not initialize
current to a default element or eagerly execute internal binding initializers.
A fresh enumerator copies capture values/descriptors from the template and starts
at the initial state. It does not copy the template's current or iteration state.

For a zero-cut generator, current retains its logical element identity but needs
no physical slot. A certified consumer still performs the final pull, preserving
body effects, then skips its unattainable successful-current branch. Scratch
storage may have explicit zero extent; no slot access is valid within it.

Storage is available through three settled paths:

1. An ordinary allocating site has an explicit residence and admitted frame
   extent. Alex uses the corresponding existing allocation pattern.
2. A supported factory call supplies caller-owned storage through an explicit
   hidden destination. `ContinuationAllocate` allocates raw storage; the
   constructor initializes the supplied descriptor without allocating again.
3. A bounded child allocation inside a generator uses a distinct region appended
   to that parent's persistent frame. Its exact parent formal and coordinates
   produce a typed byte view, not a child allocation during MoveNext.

The factory and region paths are scoped implementation support. They require
settled per-occurrence origins, a finite nonrecursive region dependency and
storage that outlives every admitted use. Recursive frame growth, unresolved
escapes and arbitrary callable/aggregate transport require their own admitted
contracts; this PRD does not mark them implemented. Returning a descriptor never
extends the backing storage lifetime by itself.

Scoped captured templates are also admitted when a complete-use analysis proves
that their source allocation's activation covers every use of the capturing
sequence. The finite `SequenceTemplateBorrow` relation retains the allocation,
covering activation, captured declaration, generator and constructor. Nested
and repeated local uses pass; return, store, opaque use, unknown input and
missing/shared constructor ownership remain residuals. 15c tests this path with
shared mutable source cells and independent enumeration.

## 6. Passive Alex composition and standard MLIR

[SeqWitness.fs](../../src/MiddleEnd/Alex/Witnesses/SeqWitness.fs) consumes the
settled construction/access/call facts. [ContinuationPatterns.fs](../../src/MiddleEnd/Alex/Patterns/ContinuationPatterns.fs)
composes typed view, load/store, allocation and call Elements. It checks supplied
identities and carriers; it does not build a frame or rediscover a body shape.

`ContinuationDispatch` supplies a selector, literal case labels and explicit
child graph regions. The control witness pulls those children through the
existing witness function at their Huet zipper positions. Pattern composition
produces `scf.index_switch` with branch effects and settled result carriers.
Ordinary conditionals and loops use existing structured control patterns.
There is no source subtree emitter, recursive yield collection, imperative
MoveNext builder or mutable semantic environment in Alex.

The witnessed operations use standard `func`, `memref`, `arith`, `index` and `scf`.
Frame extent and field offsets are literals from placement. The state carrier is
converted to `index` through the existing typed/range-aware operation when
required by `scf.index_switch`; frame stores retain their settled carrier.

The backend's [LLVM lowering pipeline](../../src/BackEnd/LLVM/Lowering.fs)
expands memory metadata, lowers memrefs and vectors, converts structured control
to `cf`, and lowers control, index, function and arithmetic operations before
reconciling conversions. Target index width comes from the platform contract.
No private continuation MLIR dialect or local witness lowering pass is needed.
A stock verifier accepting this output does not establish source semantics,
upstream proof discharge or target lifetime admission.

## 7. Completion gates

Implementation exists for the source/graph contracts, frame and machine
construction, resident evidence and passive pattern paths described above.
That statement is distinct from completion of all feature gates. The
`15a_SequenceSemantics` oracle now passes stock MLIR verification and exact native
output for literals, repeated enumeration, delayed effects, conditional and
counted loops, empty sequences, supported factories, delegation, independent
nested iteration, retained mutable child captures, factories within generators
and chained empty effects. Final compiler, Alex, proof-transfer, tooling and
ordinary native-control gates remain to be recorded together in the waypoint.

Before marking C-06 complete, record final results against the same compiler
revision in the waypoint. Do not substitute generated-MLIR-only acceptance for
source execution or change expected native values to accommodate a failure.

- [x] Full relevant CCS source/graph suite: accepted NTU element/ownership/control
  cases and exact rejected source/settlement premises, including participant and
  provenance retention.
- [x] Alex component suite: actual Huet child pulls, immutable graph facts,
  typed slots/descriptors, dispatch, missing prerequisite diagnostics, fresh
  enumeration, caller destinations and distinct owned regions; real MLIR
  verification and standard lowering.
- [x] Native sequence oracle and FidelityHello `15a_SequenceSemantics`: literal and empty
  sequences, effect order before/after yield, guarded and repeated loops,
  captures, repeated enumeration, supported factories, delegation and composition
  within the admitted contract.
- [ ] Existing ordinary native controls remain correct on the final compiler.
- [x] CCS.Editor, analyzer-facing projections and actual LSP gates retain public
  `seq<'T>` types, dimensional errors, source capture definitions and unsaved
  repairs. Internal generated declarations do not replace source projections.
- [x] Normative spec, PRD and waypoint describe the final supported boundary and
  any remaining residuals consistently; final evidence identifies compiler
  artifacts rather than stale builds.

The implementation waypoint records 848/848 CCS tests, 71/71 Alex cases,
65/65 SMT-transfer cases, the three native variants and peered projection gates.
The broad FidelityHello gate remains 23/28: formatter/parsing issues, unresolved
recursion/lazy widths, and accumulating sequence ranges in the original 15 remain
open. These failures are not waived or represented as a complete aggregate gate.
See the waypoint for exact artifact hashes, logs and companion revisions.

## 8. Related work

- [C-01: Closures](C-01-Closures.md): callable values, capture identity and residence.
- [C-05: Lazy](C-05-Lazy.md): deferred formation and memoized values; sequences do
  not inherit a memoized current-value default.
- [C-07: Sequence operations](C-07-SeqOperations.md): producers/consumers compose
  the same sequence ingredients; no competing wrapper-emission architecture.
- [Closure nanopass architecture](../Closure_Nanopass_Architecture.md) and
  [delimited continuations](../Delimited_Continuations_Architecture.md):
  upstream recipe ownership and proof-bearing graph construction.
