# C-06: Simple Sequence Expressions

> **Status:** In-Progress. Native continuation core implemented; full acceptance remains open.
> **Criteria reconciled:** September 25, 2026; no new execution claim.
> **Samples:** Original `15_SimpleSeq`, FidelityHello `15a_SequenceSemantics`,
> `15b_SequenceElements`, `15c_SequenceTemplateBorrows`, `15d_SequenceAdditive`,
> and the [NativeSequences harness](../../tests/NativeSequences/README.md).
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

The [shared C-series acceptance contract](C-Series-Acceptance.md) governs evidence,
profile scope and continuation. This PRD preserves the implemented continuation
core and identifies the composition work still owed.

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

This illustrates source formation and suspension. Native admission also needs
the actual call-site numeric ranges, storage and selected-platform premises;
the generic example alone is not an accepted executable or a fabricated bound.

The source contracts are:

- Creating the sequence preserves established capture values and shared deferred
  identities under the [evaluation strategy contract](../Evaluation_Strategy_Contract.md);
  it forces neither unused operand initializers nor the deferred generator body.
- A pull follows source evaluation order until a yield or exhaustion. Resuming
  continues after that yield, including the remainder of the loop body before
  the next guard evaluation.
- A yielded payload can retain a shared deferred computation. Successful-pull
  observation alone does not force an unused payload; demand for current's value
  shares the computation and requires storage valid for every retained use.
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
| Producer formation | Shared sequence ingredient and producer recipes | Established values or shared deferred operands, body references and actual `SeqGenerator` nodes; preserve original captures, demand boundaries and public source ranges; inherited eager snapshots require reconciliation |
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

The later C-07 startup work additionally retains explicit writable program-cell
authority for scalar external captures. That authority is separate from slot
layout/capacity and from a descriptor's backing lifetime. Its full-profile 16g
gate does not establish unrestricted program-lifetime aggregate or sequence
storage. Read the [current coverage record](../Language_Coverage_Waypoints.md)
alongside these bounded paths.

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

This describes the recorded native form. Additional expression/profile forms
follow [M-01](M-01-DialectAdmission.md), with explicit prerequisites and evidence;
it is not a claim that every operation in those dialects or every target works.

The backend's [LLVM lowering pipeline](../../src/BackEnd/LLVM/Lowering.fs)
expands memory metadata, lowers memrefs and vectors, converts structured control
to `cf`, and lowers control, index, function and arithmetic operations before
reconciling conversions. Target index width comes from the platform contract.
No private continuation MLIR dialect or local witness lowering pass is needed.
A stock verifier accepting this output does not establish source semantics,
upstream proof discharge or target lifetime admission.

## 7. Completion gates

### 7.1 Recorded implementation evidence

The [September 20 C-06 checkpoint](../Language_Coverage_Waypoints.md#c-06-native-continuation-settlement--2026-09-20)
records source/graph, Alex, SMT-transfer, native and editor/client results with
their artifact hashes. Its historical 23/28 broad regression result is qualified
by later checkpoints: the formatter regression subsequently closed, and C-07
added additive-range and scalar-Option transport evidence. No new aggregate
regression run is implied by this criteria reconciliation.

| Oracle | Recorded bounded behavior to retain |
|---|---|
| 15a | Literal/repeated enumeration, delayed pre/post-yield effects, conditionals and counted loops, effectful empty generators, supported factories, delegation, nested iteration and child captures |
| 15b | Boolean, observable unit, real and measured element carriers; numeric fractions do not establish fractional measure-exponent admission |
| 15c | Scoped captured templates with proved covering activations, shared mutable cells and independent enumeration |
| 15d | Finite additive recurrence evidence, triangular/negative/mixed state, zero trips and non-unit steps in both directions |
| C-07 16f / 16g | Scalar-payload Option values retained across pulls/exhaustion; program startup and scalar external-cell authority respectively, within their recorded scope |

The [later C-07 checkpoint](../Language_Coverage_Waypoints.md#c-07-sequence-operations--implementation-waypoint-acceptance-open-2026-09-20)
owns those subsequent results. The original15 still has separate coupled and
multiplicative recurrence obligations. Its Fibonacci and power cases require
bounds for ordered intermediate updates and stores, not only yielded values.

### 7.2 Remaining acceptance matrix

| Area | Positive gate | Refusal/preservation gate |
|---|---|---|
| Original source coverage | Original15 and 15a–d compile and run with their expected values/effects on a coordinated compiler cohort | Keep unsupported recurrence, platform and residence facts explicit; preserve valid neighboring cases and original source spans |
| Callable transport | Direct, returned, retained and multiple-origin sequences use C-01/C-02's actual function/environment values and truthful signatures | Exact-origin elision requires evidence; no guessed generator or backing lifetime |
| Element/current transport | Admitted nested aggregate and callable payloads retain types and values after another pull, exhaustion and child return; ownership may select copy, borrow or transfer | Successful-current guard names the exact iterator; reject missing initialization, invalidated backing storage, wrong dimensions or unavailable representation |
| Residence and capacity | Factory destinations, captured templates, parent-owned children and declared program storage cover every admitted use; repeated formations have distinct dynamic instances | Test absent/ambiguous authority, escape beyond covering activation, alias overwrite, recursive frame growth and insufficient peak capacity at their owning boundaries |
| Control and recurrence | Ordered evaluation and control occurrences establish cut/resume transfers, definite assignment and live-across storage; C-03/numeric rules establish applicable recurrence bounds | Changed guards, updates, participants or control paths invalidate the corresponding evidence; compiler convergence is separate from runtime termination |
| Observable enumeration | Repeated/interleaved enumerators have fresh progress and shared external capture identity; no-yield bodies retain demanded effects | No body execution at formation, no default current, no replayed operand initializer and no pull after a decisive downstream stop |
| Tooling and realization | Public element types, source capture navigation and unsaved repair agree with graph facts; serialized MLIR and native behavior realize those facts | Missing prerequisites fail in the responsible stage; artifact changes and stale editor/proof results cannot retain valid status |

Apply the [common evidence gates](C-Series-Acceptance.md#4-evidence-required-to-close-a-row)
to every supported form. Full C-06 completion requires the original and lettered
native gates, relevant ordinary controls and the owning proof/tooling checks on
the final recorded source/dependency/target cohort. Existing passing component
and bounded native evidence remains valuable while those gates are open.

## 8. Related work

- [C-01: Closures](C-01-Closures.md): callable values, capture identity and residence.
- [C-05: Lazy](C-05-Lazy.md): deferred formation and memoized values; sequences do
  not inherit a memoized current-value default.
- [C-07: Sequence operations](C-07-SeqOperations.md): producers/consumers compose
  the same sequence ingredients; no competing wrapper-emission architecture.
- [Closure nanopass architecture](../Closure_Nanopass_Architecture.md) and
  [delimited continuations](../Delimited_Continuations_Architecture.md):
  upstream recipe ownership and proof-bearing graph construction.

## 9. Criteria supersession

The older completion checklist mixed bounded passes with a full-family exit and
repeated a historical broad-regression failure set as current. Sections 7.1–7.2
replace that account with dated evidence and explicit remaining gates. The
implemented frame, control, residence and continuation protocols remain the
starting point. No compiler code, source fixture, expectation or feature status
changes in this documentation reconciliation.
