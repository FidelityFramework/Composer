# Closure values captured by other computations

> **Governing contract:** [Closure settlement](Closure_Settlement_Contract.md),
> [closure representation](../../clef-lang-spec/spec/closure-representation.md),
> [C-01 §14](PRDs/C-01-Closures.md#14-the-closure-saturation-form-family), and the
> [Baker construction contract](../../clef/docs/fidelity/Baker_Saturation_Architecture.md).
> This is the materialized known-callee mechanism within the complete C-series
> delivery obligation. Validation is recorded with the actual source, graph,
> editor, proof, Alex and native results in the [waypoints](Language_Coverage_Waypoints.md).

## 1. Materialized callable contract

A computation capturing a callback retains its actual formation environment.
Knowing the implementation does not identify an environment instance. Baker
settles implementation identity separately from the actual environment descriptor;
calls pass the recalled environment as their first argument.

Stored producer/consumer partials, returned frontiers and their nested captures
participate in one lifetime contract. Immutable values retain formation-time
sharing. Mutable captures retain the same cell. A sequence or callable descriptor
requires a covering lifetime for the backing storage it references.

The source graph must carry actual/formal supplies, nested initializer forwarding,
result destinations and complete-use evidence before native admission. Capturing
a value or providing a destination grants no lifetime by itself. The required
C-series forms include ordinary, returned, stored and deferred function values;
these requirements are exercised across the `16` lettered sequence cases and the
C-01/C-02 closure/application gates.

## 2. Source identity, formation and calls

The [closure environment recipe](../../clef/src/Compiler/Baker/Recipes/ClosureEnvironmentRecipes.fs)
uses ordinary Baker ingredients and fan-out/fold-in. Its identities are:

| Identity | Meaning |
|----------|---------|
| `F` | Original callable expression and environment-layout owner; retains source function type, ID and full range |
| `L` | New implementation lambda with a truthful environment-first type |
| `B` | Real immutable declaration binding containing `L`, used by ordinary direct-call references |
| `Q` | Real environment formal at ordinal zero; an internal result destination, when present, follows it before source parameters |
| `E` | Environment construction at `F`; repeated runtime formation retains distinct allocation instances |
| `Cᵢ` | Original captured declaration, also the slot key |
| `Iᵢ` | Already evaluated value or original cell descriptor recalled at this formation |
| `U` | Actual callable occurrence recalled by an alias or frame read |

Implemented internal graph kinds are:

```text
ClosureValue(implementation: L, environment: E)       : original source function type
EnvironmentAllocate(owner: F)                        : internal byte array
EnvironmentCreate(owner: F, initializers: (Cᵢ * Iᵢ) list)
                                                     : internal byte array
EnvironmentReference(callable: U)                    : internal byte array
EnvironmentRead(environment, slot: Cᵢ)               : captured source type
EnvironmentBorrow(environment, slot: Cᵢ)             : internal typed cell view
EnvironmentWrite(environment, slot: Cᵢ, value)        : unit
```

The internal carrier is `Types.mkArrayType Types.uint8Type`; its physical extent
comes from the settled environment layout. Source `TFun` types remain unchanged.
Explicit reads and borrows forward nested formation initializers. The child retains its own flat slots; an immutable capture copies its actual value, while a mutable capture forwards the original cell view. Residence is proved separately.

`F` structurally retains `L` and `E`. The function body stays deferred; formation
requires `E`, not execution of `L`. Initializers name already evaluated captured
values. Their reference incidence preserves availability without structurally
reevaluating the declaration's initializer. The current lexical formation uses
`(Cᵢ, Cᵢ)` for lexical formation. A nested formation can use `(Cᵢ, Iᵢ)` where `Iᵢ` is an explicit parent-environment read or borrow, retaining the original slot identity.

A call is rewritten to an ordinary `Application` whose callee references `B` and whose arguments start with `EnvironmentReference(U)`. A hidden result destination follows it when required, then the source arguments. Destination preparation snapshots the original operands once and in order.
Alex does not perform this call rewrite. `L` has no implicit capture list. Its
captured references keep their original node IDs/ranges and become explicit
accesses through `Q` and the original `Cᵢ`.

Generated formals and code declarations use point source anchors.
`Closure.SourceSignature` records the original type on changed implementation
and declaration nodes. Editor projection resolves an environment read/borrow
through its exact source slot, then any existing `CaptureOrigin` relation.

## 3. Published facts and resident incidence

[ClosureEnvironments](../../clef/src/Compiler/PSGSaturation/SemanticGraph/ClosureEnvironments.fs)
reads exact kinds, aliases and frame-read origins. It does not reconstruct an
environment from a source name or code symbol.

| Fact | Implemented shape |
|------|-------------------|
| `EnvironmentLayout` | `Owner`, `Implementation`, `Formal`, ordered `Slots: ContinuationSlot list`, `Bytes`, `Alignment`, resident `Obligations` |
| `EnvironmentLayouts` | Layout owner `F` → exact layout |
| `EnvironmentOrigins` | Actual environment/callable/formal/reference/frame-read occurrence → `F` |
| `KnownCallables` | Callable occurrence → `{Implementation: L; EnvironmentOwner: F}` |
| `EnvironmentView F` | `CaptureSlotKind` for the actual descriptor of a separately known callable environment |
| `Escapes[E]` | Explicit admitted residence of the actual allocation, never a fallback inferred from capture or code identity |
| `EnvironmentDestinations` | Original constructor → caller-supplied formal used for initialization |

`KnownCallables` deliberately does not supply a replacement environment value.
The value remains the actual occurrence `U`, including a frame read. Two calls
with the same implementation may carry distinct environment instances.

The recipe also retains typed graph incidence:

- `EnvironmentCapture mutableCell`: ordered `[F; Cᵢ; Iᵢ]` → `E`, retaining mode
  and initializer order even when two participants have the same ID.
- `EnvironmentInitializer`: individual reference edges to initializer values.
- `EnvironmentFormal`: `[F; L]` → `Q`, distinguishing the internal formal from
  a source captured declaration.
- `FrameSlot`: original slot provenance for explicit reads, writes and borrows.
- `EnvironmentResidence`: allocation, covering activation, captured sources and
  complete-use participants retained by the residence proof.
- `EnvironmentResultDestination`: `[factory implementation; F; destination
  formal]` → original `E`.
- `EnvironmentResultCall`: `[factory implementation; E; destination formal; caller allocation;
  actual destination]` → exact complete call. These two relationships describe
  the prepared storage protocol; neither grants residence.
- `SequenceInputBorrow`: `[source allocation; covering activation; actual
  argument; formal; callee implementation]` → exact complete call. The same
  complete-formal-use proof covers admitted sequence and known callable inputs.
- `SequenceResultCapture`: `[slot; initializer; factory implementation; source
  formal; call; actual source argument; actual destination; caller allocation]`
  → returned sequence constructor. Its ordinal is the source formal's actual
  position after internal operands have been inserted. This is a pending
  retained-view requirement, not a borrow proof.

Generated `B` declarations are compile-time code identities. Continuation definite
initialization includes only those bindings proved by the `ClosureValue`, real
lambda/formal and `EnvironmentFormal` relationship. `F`, `E` and ordinary callable
aliases still require actual initialization/capture; they are not made initially
available by this rule.

A capture-free function is plain named code with no environment, as required by
closure representation §3.1 and §9. A stateless callable retained by another
computation needs no runtime capture slot when exact immutable code identity is
established. Removing that slot must preserve all earlier operand/effect evaluation.
An empty byte allocation is not the canonical substitute for this direct form.

## 4. Layout, residence and range obligations

[Placement](../../clef/src/Compiler/PSGSaturation/SemanticGraph/Placement.fs)
uses the same slot selection and tiling for closure environments and continuation
frames. Closure environments have no state/current prefix and no code field.
Widths and alignment follow source NTU ranges and the declared target. The shared
exact-layout obligation retains the owner, implementation, formal, construction
and slot participants.

A mutable capture holds its original typed cell descriptor. Reads/writes access
that cell; creation does not copy its scalar into a substitute cell. The descriptor
is unboxed address/offset/extent/stride data, not a runtime type object. Immutable
scalar captures are copied at formation in their settled representation.

[SequenceResidence](../../clef/src/Compiler/PSGSaturation/SemanticGraph/SequenceResidence.fs)
reuses the finite complete-use covering proof for environments. A
mutable cell must cover all uses of every environment retaining it, including
complete ordinary call paths and returned destinations. Captures crossing a sequence generator must have
its exact constructor/capture relationship and bounded use. Unknown consumers and opaque independent references invalidate complete-use conclusions. A returned value requires validated caller-destination and retained-view relationships. Parent
pointers alone do not prove this property.

[ProgramActivation](../../clef/src/Compiler/PSGSaturation/SemanticGraph/ProgramActivation.fs)
establishes directional coverage from an ordinary activation with explicit
program-entry provenance. It follows all incoming complete calls and retains
deferred-owner prerequisites; lexical enclosure and an unrooted call cycle are
insufficient. Coverage from a local caller is not a claim of program lifetime.
Immutable snapshots of an environment projection preserve its exact argument
position; every alias and independent reference remains part of the complete-use
check. New mutable, escaping or opaque uses retract that conclusion.

Caller-destination preparation is revalidated against the actual final
constructor, physical formal/argument position, allocation owner/type and
complete factory call set. A returned descriptor is admitted only when each
actual captured view's backing allocation covers every destination use. The
proof follows nested explicit reads and immutable snapshots to their actual
initializers, preserving the slot identity separately. It checks the source
storage's uses and the destination's uses together. A destination row, an
unrelated borrow edge or a matching layout alone cannot discharge this proof.

These facts implement bounded parts of the C-01 obligations:

- **VC-EXT / VC-DIS:** concrete finite extent, containment, alignment and disjoint
  field placement through the shared layout obligation.
- **VC-REG / VC-REL:** bounded backing storage through complete use and its
  covering lifetime across actual calls, returned destinations and deferred uses. A descriptor does not extend a departed activation.
- **VC-APP:** actual env-first implementation/formal/argument relationships in
  the ordinary typed call graph. Unknown-callee transport must preserve the full function/environment correspondence.

Range/effect analysis follows the original source cell and the typed capture
mode. Immutable reads use their formation initializer facts; mutable writes and
transitive calls invalidate guards on that same source cell. Shared slot-meet
logic settles scalar width adaptations before Alex consumes them. Layout proof
never substitutes for residence or availability proof.

## 5. Implemented phase order

The [driver](../../clef/src/Compiler/NativeTypedTree/NativeService.fs) orders the
new work with the existing passes:

1. Callable staging and direct immutable-capture elaboration precede
   `ClosureEnvironmentElaboration`. Its Baker recipe introduces formation,
   code, formals, accesses and ordinary env-first applications.
2. Sequence consumption/ownership/delegation, element/effect relations and range
   analysis see those explicit operations. Source evaluation contracts retain
   formation boundaries and source-cell identities.
3. Aggregate placement and curry normalization precede result-destination preparation. Both closure and sequence result relationships are exposed before final residence checks. `ClosureEnvironmentSettlement` proves common storage uses, places slots and adds resident layout obligations before continuation placement.
4. Sequence control preserves implementation-code availability separately from
   runtime environment initialization. Frame placement selects `EnvironmentView`
   from exact environment origins. The machine retains actual descriptor values.
5. Final codata publishes environment layouts/origins/callables, explicit residence
   and shared slot meets. These readings pass maps explicitly and never force
   `graph.Codata` while constructing it.

Native environment settlement is skipped for rejected source and for checking
without a declared platform. Source graph/projection support therefore does not
claim physical layout or introduce native-layout diagnostics into invalid source.

## 6. Alex consumption

[EnvironmentPatterns](../src/MiddleEnd/Alex/Patterns/EnvironmentPatterns.fs) and
[EnvironmentWitness](../src/MiddleEnd/Alex/Witnesses/EnvironmentWitness.fs) consume
the supplied slots, exact initializer rows and admitted residence. Common
[continuation patterns](../src/MiddleEnd/Alex/Patterns/ContinuationPatterns.fs)
provide typed descriptor initialization and slot access through stock MLIR.
Missing facts or mismatched carriers produce specific diagnostics.

`ClosureValue` recalls `E`; `EnvironmentReference` recalls the actual `U`.
Site-aware carrier mapping reads the environment layout. ApplicationWitness
handles the ordinary graph call already rewritten by Baker. LambdaWitness
receives real parameters and performs no legacy closure construction for `L`.
No environment pattern discovers captures, chooses layout, or rebuilds a view
from a legacy packed pair.

The governing architecture is positional Huet witness pull. Accumulators and
visited sets are emission bookkeeping separate from the zipper. Every shared
body is witnessed through its actual occurrence, with local parameter associations
restored at scope exit. All semantic construction belongs in Baker.

## 7. Delivery and regression obligations

The [settlement contract](Closure_Settlement_Contract.md#c-series-delivery-gates)
defines the complete C-series gate. Full separate function/environment transport,
all specified capture and lifetime forms, and correct stored/returned/deferred
composition are delivery work. Canonical layout and identity must hold through
source checking, graph elaboration, actual-position witnessing and backend realization.

Acceptance checks retain exact source/native oracles, shared mutation, independent
formations, eager snapshots, sequence restart and short circuit. Negative checks
remove or contradict proof participants and add previously absent consumers.
Editor checks must observe the same source signatures, definitions, capture
origins and invalidated dependencies. Follow the [regression policy](Regression_Check_Policy.md)
for focused changes and the complete gate when closing a C-series area.

The focused [activation](../../clef/tests/Clef.Compiler.Service.Tests/ProgramActivationCases.fs)
and [residence](../../clef/tests/Clef.Compiler.Service.Tests/SequenceResidenceCases.fs)
cases exercise complete incoming calls, immutable environment snapshots,
sequence/callable formal uses, caller destinations and damaged retained-view
requirements. Test presence describes the intended regression obligation; the
waypoint records the actual executed cohort and result. Full returned, stored,
aggregate-held and unknown-callee forms remain required acceptance work wherever
their complete contract has not yet passed. A conservative rejection preserves
integrity but does not close the positive gate for a conforming source program.
