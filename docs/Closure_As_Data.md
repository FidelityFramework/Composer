# Closure values captured by other computations

> **Status:** The bounded known-callee implementation is on disk. Source graph,
> range/effect and Editor projection checks have passed. Native callback execution
> is still being verified; this document makes no native-success or C-07 completion
> claim. The latest control-availability regression accompanies an upstream fix
> for generated implementation declarations.
>
> **Governing decisions:** [C-01 §14](PRDs/C-01-Closures.md#14-the-closure-saturation-form-family),
> [closure representation](../../clef-lang-spec/spec/closure-representation.md),
> [Closure Retooling Plan §3/§6](../../clef/docs/fidelity/phg/Closure_Retooling_Plan.md)
> and [closure nanopasses](Closure_Nanopass_Architecture.md). The full `(fn, env)`
> contract remains the language architecture; the current increment implements
> its known-callee storage case.

## 1. Implemented boundary

A sequence capturing a callback captures its actual formation environment.
Knowing the implementing function does not identify an environment instance.
Baker retains a proved implementation identity separately and stores the actual
known callback environment as a typed descriptor in the sequence frame. Calls
receive that recalled environment as an ordinary leading argument.

The current admission covers concrete source callback lambdas, immutable aliases
and their complete direct uses in sequence producers. Captures are immutable
scalars or shared mutable scalar cells. The recipe rejects outer captures used
inside an additional deferred lambda, lazy value or sequence in the callback.
Returned/opaque callables and aggregate or callable captures require further
contracts. Capturing a callback does not make its allocation nonescaping by fiat.

Environment storage is admitted only when complete graph use proves a covering
ordinary activation. Generator-local formation needs an owned environment region;
returned environments need a proved destination or other residence. Neither is
implemented by this increment. Missing residence is a native settlement residual,
not an allocation fallback.

The admitted path never stores code in the environment, uses `ValueView TFun`,
or reads the old packed callable representation. Existing legacy closure paths
remain elsewhere; their existence does not extend this admission boundary.

## 2. Source identity, formation and calls

The [closure environment recipe](../../clef/src/Compiler/Baker/Recipes/ClosureEnvironmentRecipes.fs)
uses ordinary Baker ingredients and fan-out/fold-in. Its identities are:

| Identity | Meaning |
|----------|---------|
| `F` | Original callable expression and environment-layout owner; retains source function type, ID and full range |
| `L` | New implementation lambda with a truthful environment-first type |
| `B` | Real immutable declaration binding containing `L`, used by ordinary direct-call references |
| `Q` | Real environment formal preceding the original explicit parameters |
| `E` | Environment construction at `F`; repeated runtime formation retains distinct allocation instances |
| `Cᵢ` | Original captured declaration, also the slot key |
| `Iᵢ` | Already evaluated value or original cell descriptor recalled at this formation |
| `U` | Actual callable occurrence recalled by an alias or frame read |

Implemented internal graph kinds are:

```text
ClosureValue(implementation: L, environment: E)       : original source function type
EnvironmentCreate(owner: F, initializers: (Cᵢ * Iᵢ) list)
                                                     : internal byte array
EnvironmentReference(callable: U)                    : internal byte array
EnvironmentRead(environment, slot: Cᵢ)               : captured source type
EnvironmentBorrow(environment, slot: Cᵢ)             : internal typed cell view
EnvironmentWrite(environment, slot: Cᵢ, value)        : unit
```

The internal carrier is `Types.mkArrayType Types.uint8Type`; its physical extent
comes from the settled environment layout. Source `TFun` types remain unchanged.
The current recipe emits reads and writes; the explicit borrow operation supports
typed slot access but does not by itself admit new source escape/forwarding cases.

`F` structurally retains `L` and `E`. The function body stays deferred; formation
requires `E`, not execution of `L`. Initializers name already evaluated captured
values. Their reference incidence preserves availability without structurally
reevaluating the declaration's initializer. The current lexical formation uses
`(Cᵢ, Cᵢ)`, loading an immutable value or retaining the actual mutable cell.

A call is rewritten to an ordinary `Application` whose callee references `B` and
whose arguments are `EnvironmentReference(U)` followed by the original arguments.
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
| `Escapes[E]` | Explicit admitted allocation residence; current Baker path proves `StackScoped` |

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

Generated `B` declarations are compile-time code identities. Continuation definite
initialization includes only those bindings proved by the `ClosureValue`, real
lambda/formal and `EnvironmentFormal` relationship. `F`, `E` and ordinary callable
aliases still require actual initialization/capture; they are not made initially
available by this rule.

The materialized empty-capture case has an explicit zero-byte environment,
alignment one, and a real environment formal. It has no null or zero-address
placeholder and admits no slot access. Eliding that empty environment/formal is
a separate optimization of the canonical vacant form.

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
reuses the finite complete-use covering proof for environments. The current
mutable-cell case requires that cell and environment to belong to the same
covering ordinary activation. Captures crossing a sequence generator must have
its exact constructor/capture relationship and bounded use. Unknown consumers,
returns and opaque independent references keep residence unresolved. Parent
pointers alone do not prove this property.

These facts implement bounded parts of the C-01 obligations:

- **VC-EXT / VC-DIS:** concrete finite extent, containment, alignment and disjoint
  field placement through the shared layout obligation.
- **VC-REG / VC-REL:** bounded backing storage through complete use and its
  ordinary activation lifetime. Returned/generator-local environment placement
  remains unresolved; a descriptor does not extend a departed activation.
- **VC-APP:** actual env-first implementation/formal/argument relationships in
  the ordinary typed call graph. Full unknown-callee correspondence remains open.

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
3. Aggregate placement and curry normalization precede final environment
   settlement. `ClosureEnvironmentSettlement` places common slots, records
   residence and adds resident layout obligations before continuation placement.
4. Sequence control preserves implementation-code availability separately from
   runtime environment initialization. Frame placement selects `EnvironmentView`
   from exact environment origins. The machine retains actual descriptor values.
5. Final codata publishes environment layouts/origins/callables, explicit residence
   and shared slot meets. These readings pass maps explicitly and never force
   `graph.Codata` while constructing it.

Native environment settlement is skipped for rejected source and for checking
without a declared platform. Source graph/projection support therefore does not
claim physical layout or introduce native-layout diagnostics into invalid source.

## 6. Alex consumption and remaining architecture debt

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

The governing architecture remains positional Huet witness pull. Existing
`NanopassArchitecture.visitAllNodes` and branch-scope collection still contain
recursive traversal and mutable emission collection. That driver migration is
separate debt; these remaining mechanisms are not evidence that recursive source
emission is the intended architecture. New semantic construction belongs in Baker.

## 7. Work still required

Canonical full `(fn, env)` support includes unknown function halves, multi-value
operand recall, call/return signatures, joins and stored-callable placement.
Those contracts remain open. A descriptor cannot be relabeled as a function or
carry a code address in a numeric field.

Other remaining cases include returned callback environments with exact caller
destinations, generator-owned environment regions, additional captured aggregate
or callable views, and callbacks whose outer captures cross another deferred
scope. Collect must prove the lifetime of a returned child sequence and any
retained callback environment. Existing sequence destination/region machinery
is a basis for this work, not automatic admission.

The source graph/range and Editor gates cover actual types, aliases, capture
identities, exact definition navigation, effects and invalid-reference rejection.
Alex component cases check typed descriptor operations and missing prerequisites.
Native map/filter cases must still establish exact formation/call effects,
repeated enumeration, changed shared cells and distinct runtime formations of
one code identity. Pending native results and the complete C-07 operation gates
are recorded in the owning roadmap and regression evidence, not inferred here
from isolated MLIR verification.
