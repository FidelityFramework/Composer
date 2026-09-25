# Negative and Fractional Types Architecture

> **Status: Proposed companion design; native NFT implementation Planned.**
> Revised 2026-09-21. The [coordination record](Bidirectional_Composition_Plan.md)
> owns the cross-repository accounting and executable gates. This document owns
> the engineering design. The specification's proposed glossary entries do not
> confer normative standing on these constructs.

## 1. Scope and reference semantics

The intended developer benefit is ordinary domain composition with checked
resource and reconstruction contracts. Applications should not thread a history
object or expose compiler proof objects merely to make the composition legal.
The compiler must still account for every runtime dependency required by the
selected realization. Hiding an argument is not a storage optimization.

Chen and Sabry's [2021 calculus](https://homes.luddy.indiana.edu/sabry/files/2021/02/popl.pdf)
is the operational reference for directed negative types and value-indexed
fractional resources. James and Sabry's earlier logic-variable/unification
interpretation is a separate research direction. It is not the operational
semantics of the 2021 fractional calculus. Kennedy's dimensional algebra remains
an independent check: an inverse unit and a rational measure exponent do not
identify a fractional value resource.

The [February 7, 2021 erratum](https://homes.luddy.indiana.edu/sabry/files/2021/02/errata.pdf)
withdraws Theorem 25's inverse-category claim. Its proof omitted uniqueness of
generalized inverses, and the erratum supplies a counterexample. The compact
closed result in Theorem 24 is not withdrawn. This design must name a selected
reversal and prove its laws on the admitted fragment; compact closure alone
cannot supply a unique inverse for arbitrary Clef programs.

The [NFT working manuscript](../../arxiv-papers/negative-fractional-types-in-fidelity.md)
owns the fuller theoretical case. [FPS](../../arxiv-papers/fixed-point-scaffolding.md)
owns the proof-preservation account and
[PHG](../../arxiv-papers/program-hypergraph-paper.md) the joint-relation model.
Changes to these Markdown sources do not update their generated publications.

## 2. Ordinary composition and two directions

The proposed interface can remain ordinary Clef. For example, a numerical
library could define the following functions under an admitted recovery contract:

```fsharp
let later = initial |> evolve model steps
let earlier = later |> retrace model steps
```

These are illustrative domain functions, not approved intrinsics or syntax.
A downstream `retrace` use would contribute a reconstruction requirement to the
earlier evolution's joint constraints. Baker would select an admissible
realization or report the unmet requirement at the relevant boundary. Possible
realizations include a proved exact inverse, replay under a declared input
contract, or retained residual information within an explicit storage budget.

The Haskell [Tardis implementation](https://raw.githubusercontent.com/DanBurton/tardis/master/src/Control/Monad/Trans/Tardis.hs)
is a reference for a different, composable property: information flowing both
ways between stages. It does not supply Clef's feature name. For consecutive
pure computations `p` and `q`, its bind has the equations

```text
(a, (b0, f1)) = p   (b1, f0)
(r, (b1, f2)) = q a (b2, f1)
```

Forward state `f1` comes from the earlier stage, while backward state `b1`
comes from the later stage. Lazy recursive bindings can make the feedback
productive. Forcing a cyclic dependency with no productive definition can
instead diverge. A native interpretation needs its own demand rules; a graph
of equations alone does not choose Haskell's evaluation semantics.

Neither state channel is inherently a dual resource. `f` is not a `Neg` trace,
and `b` is not a `Recip` token. Bidirectional dependency can describe an ordinary
noninvertible computation. Actual inverse execution additionally requires an
inverse law or enough information to reconstruct the predecessor.

| Library mechanism | Proposed Clef location | Obligation retained |
|---|---|---|
| `return` and bind | Typed composition and elaboration | Unit/associativity under the chosen effect and observation model |
| Two state channels | Ordered input/output ports in the PHG | The equations above and port identity |
| Lazy recursive binding / `MonadFix` | Demand and feedback semantics; admitted continuation realization | Productivity, effects, sharing and failure behavior |
| Transformer stack | Explicit effect/coeffect composition | Ordering, scope and interactions; no automatic commutativity |
| User-threaded recovery data | Region contract and selected implementation | Exact inverse, reproducible replay, or declared retained information |
| Resource wrappers | Indexed resource typing and dynamic protocol where needed | Matching value, scoped instance, lifetime and permitted use |
| Monad laws | Generic library or elaboration proofs | Lawful composition is separate from solving one program's constraints |

This mapping removes repetitive source bookkeeping only where a compiler
contract replaces it. General monads do not become solvable merely by giving
their operations graph nodes. A compiler analysis fixed point, a program's
recursive value, and a constraint solver's model have different meanings.

## 3. Type and instance identity

A proposed fractional resource needs more than `Recip<'T>`. Its internal
judgment must distinguish at least the following information:

| Component | Meaning |
|---|---|
| Carrier `A` | The ordinary type and its relevant refinements/measure |
| Value index `v : A` | The immutable value expected at discharge |
| Pairing identity `κ` | A generative identity scoped to this dynamic pairing |
| Usage and lifetime | Who may use the resource, how often, and within which region |
| Equality evidence | A proof of the admitted match or a retained runtime comparison |

`Recip<A; v; κ>` is explanatory judgment notation here, not a proposed parser
spelling. Nominal resolution identifies the carrier and declarations. Ordinary
unification can align carrier types and index variables within an admitted
fragment. It does not establish arbitrary value equality or identify two
dynamic resources created by the same source expression.

For the value-indexed reference core, the selected value supplies a pair
`η_v : 1 → A × 1/v`; its corresponding discharge checks that the supplied value
is `v`. The pair's precise typing and failure behavior must follow the chosen
calculus. An implementation may prove the match statically or preserve the
check. No `.UnifyAndLookup` operation is assumed. An index over mutable data
needs a stable snapshot or a maintained invariant before it can serve as this
immutable identity.

Negative types belong to the directed additive calculus. The additive cup/cap
uses the additive unit `0`, not ML's inhabited `unit` (`1` for products).
Treating an additive cup as an ordinary total function returning a sum from
`unit` loses that distinction. Introduction, elimination, and control transfer
need a directed judgment before surface forms such as `Neg<A>` are admitted.
A negative value can retain its carrier's physical dimension without being a
saved execution history.

## 4. One semantic graph

The PHG contains an executable computation spine. Local requirements appear
as coeffects. Ordered relations record constraints involving several uses or
values together, and codata retains derived facts and their evidence. These
roles coexist in one graph; proof relations do not create a second running
computation. See the [PHG specification](../../clef-lang-spec/spec/program-hypergraph.md)
and [Baker architecture](../../clef/docs/fidelity/Baker_Saturation_Architecture.md).

For the proposed resource fragment, a relation would connect the creation
site, indexed value, relevant continuation/region, and permitted consumption
sites. Roles and occurrence order must survive even when two ports refer to
the same node. Splitting that relation into unrelated type-equality edges can
lose its joint condition.

The lattice intuition belongs to an analysis domain's ordering of information.
It does not assert that graph topology forms a lattice. A finite graph supports
finite incidence enumeration; saturation still needs a convergence argument
for its transfer functions and fact domain. A numerical proof may also require
an external lemma with discharged premises. Flat captures expose dependencies
but do not make all semantic properties decidable.

A source creation node can execute repeatedly. One source elimination node
can run zero times or many times. The lifecycle proof must therefore range
over reachable dynamic instances and paths, including cancellation and failure.
A linearity result cannot follow from counting graph endpoints alone. For an
exceptional path the policy must specify cleanup, transfer, or rejection.

## 5. Delay, reuse, and effects

The resource discipline must compose with C-01 closure captures and C-05 lazy
slots. Memoizing a result containing a consumable resource does not authorize
multiple consumers of that resource. Likewise, a multi-shot continuation needs
an explicit copy/transfer rule for its captured resources. Pure ordinary values
can remain shareable without making every dual resource duplicable.

Pending proof obligations may survive delayed execution. Their premises must
still hold when demanded. A check involving an external response or mutable
contents is not discharged merely because its carrier and dimensional metadata
are static. Effects in either direction require an explicit order and an
observable failure contract. An implementation must not perform external I/O
again merely to reconstruct an earlier internal state.

For the first fragment, reject unsupported open dual resources at an untyped
FFI boundary. A later typed adapter could carry a verified protocol and runtime
checks; that would be a separately admitted boundary contract. Solver timeout
or unknown is unresolved evidence, not a successful discharge. A dynamic value
mismatch follows the admitted runtime failure path. Impossible matches proved
statically can receive a source diagnostic.

## 6. Storage and reconstruction

An inverse recipe can be compiled once per region. Its runtime state need not
contain a predecessor for every executed step. For existing state `(q,p)` over
a declared modular integer ring, consider

```text
forward: p := p + F(q); q := q + G(p)
reverse: q := q - G(p); p := p - F(q)
```

With total pure `F` and `G` under unchanged model parameters, reversal restores
`q` first and then `p`.
The full update is bijective even when either helper is many-to-one. Exact
bounded arithmetic without overflow is another possible contract. Rounded
floating-point addition cannot silently replace the declared arithmetic.
Recomputed helper values require workspace, but the construction does not
require a growing history or a second trajectory. The finite ring and its
physical interpretation remain distinct models.

Under such a bijection, arbitrarily many reverse steps can use bounded live
state and workspace. A finite state space eventually cycles. Identifying an
unbounded absolute step count itself needs unbounded information, so the claim
concerns repeated reversal, not an unlimited timestamp in a fixed machine word.
A lossy map cannot recover discarded distinctions from its output alone. A
replay design can trade computation for storage if it retains an origin and can
reproduce inputs, with its step/address bookkeeping included in the budget.

For a streaming state of size `n` and `k` forward tangent directions, the live
primal/tangent storage can be `O(n(1+k))`, independent of elapsed steps `T`.
Requested outputs, parameters, residuals and workspaces add their own costs.
A differential tangent is neither a negative type nor a fractional token.
Differentiating a continuous model also does not establish that its quantized
machine update is invertible.

A bound such as `E_(i+1) ≤ A_i E_i + ρ_i` can determine an approximate
reconstruction window when the amplification and local-error bounds are
justified for the chosen model and precision. A Lyapunov estimate alone does
not recover discarded bits or prove an exact historical trajectory. Checkpoints
can limit a window's error; a fixed collection of checkpoints does not retain
arbitrary history unless an exact inverse, reproducible replay, or external
storage supplies the missing information. See the
[Lyapunov design](../../clef-lang-site/hugo/content/docs/design/types/lyapunov-window.md).

## 7. Saturation and target preservation

Baker would settle the interpretation and required constraints. Alex's selected
witness must receive every fact needed to preserve that interpretation. M-01
admits expression/profile/witness combinations individually, so this design
makes no fixed dialect list a prerequisite.

Static evidence may be erased after its operational consequences have been
realized and its preservation justified. A required runtime equality check,
failure branch, retained residual, or one-shot protocol remains executable
behavior. A standard dialect can represent such behavior without becoming the
source of NFT semantics. Each lowering must carry a checked correspondence or
re-establish the affected property. Merely passing an MLIR verifier does not
establish that correspondence.

The initial witness should expose diagnostics for the required and selected
recovery strategy, including its storage bound and remaining premises. The
source interface can be quiet while the compiler's explanation remains
inspectable.

## 8. Executable gates and dependencies

The [coordination record](Bidirectional_Composition_Plan.md#executable-work)
tracks three separately admissible exercises. All remain **Planned**:

1. A finite pure two-channel interpreter with productive and unproductive
   feedback examples. Establish the bind equations and demand behavior before
   attaching a resource interpretation.
2. A finite value-indexed core with positive match, wrong value, missing use,
   duplicate use, repeated force, and copied-continuation cases. State the
   equality theory and dynamic identity/lifetime rules explicitly.
3. One exact reversible region on one native target. Prove its discrete inverse,
   retain the source-to-graph-to-target correspondence, and measure peak live
   memory against increasing step counts without accumulating output history.

Closure representation and HOF composition depend on the relevant C-01/C-02
contracts. Delayed resource use additionally depends on C-05 and applicable
continuation rules. Reactive invalidation extends the work through R-04 to
R-06; it is not a prerequisite for every finite pure inverse example. Native
realization needs its M-01 operation/profile gate. No amount of completing
these prerequisites substitutes for the NFT-specific proofs and oracles.

## 9. Open choices and supersession

Surface spelling, index equality beyond the finite fragment, and the bridge
between ordinary bidirectional state and directed resources remain open.
So do resource-bearing multi-shot continuations and the precise numeric proof
library for approximate reconstruction. Quantum and negative-grade extensions
require separate models and acceptance evidence.

This revision supersedes the earlier June companion treatment's identification
of fractional resources with inverse dimensions, additive `unit` signature,
type-only cancellation check, one-site/one-use argument, obligatory negative
frames, and universal static discharge. It also replaces the obsolete fixed
dialect boundary with M-01's target-aware admission. Those corrections preserve
the intended ordinary API while making its implementation obligations explicit.
