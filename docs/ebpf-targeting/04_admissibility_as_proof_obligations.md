# Admissibility as Proof Obligations

**SpeakEZ Technologies | Fidelity Framework**
**July 2026 — exploratory design note**

This document places eBPF admissibility inside the graduated verification model:
which obligations arise, which tier and fragment each falls in, how they ride the
existing coeffect machinery, and how a bounded eBPF subset can exercise the
proposed proof infrastructure against an independent admission gate. The
composed path remains implementation work. This note uses the framework's proof-layer
design notes and the [Decidable By Construction](../../../arxiv-papers/decidable-by-construction.md)
capstone; it does not assume that layer is built, and it is explicit about what it
waits on.

> **Solver naming.** This note writes obligations in **SMT-LIB2** and names
> fragments (`QF_LIA`, `QF_BV`), not solvers. Each obligation must identify its
> supported fragment and premises. Z3, cvc5 or a specialized decision procedure
> may discharge it; solver choice does not expand the accepted contract.
> Decidability is not a cost bound. Timeout, unknown and missing-premise results
> remain unresolved, with a stable diagnostic when commitment requires discharge.

## The state of the proof layer (read this first)

The PSG, elaboration and saturation pipeline provide the foundation. **Complete
eBPF obligation discharge, preservation through lowering and agreement with the
pinned kernel still need composed implementation evidence.** The graduated model
organizes that work; the actual obligation, not its tier label, determines which
decision procedure applies:

| Tier | Fragment | Discharge | Cost |
|---|---|---|---|
| **Tier 1** | dimensional equations and structural classifications | the decision procedure for that classification | analysis-dependent |
| **Tier 2** | supported `QF_LIA`, `QF_LRA`, `QF_BV` obligations | SMT or a specialized decision procedure | query-dependent, budgeted |
| **Tier 3** | restricted probabilistic / nonlinear obligations | justified enclosures or library lemmas with checked premises | obligation-dependent |
| **Tier 4** | relational / probabilistic contracts | the designated proof system | obligation-dependent |

The proposed proof dispatch is **staged** over the same obligation graph: obligations are discharged at design time
(weakest-precondition reading, surfaced live through the Lattice language server)
and **re-validated at each MLIR lowering pass** through the SMT-dialect
translation-validation mechanism (the consequence-rule reading — every lowering
must preserve what was proven). This coupling of a *front-end (design-time)*
and *middle-end (build-time)* proof dispatch is the subject of a pending patent;
its stated payoff is not only safer computation graphs but graphs **better
optimized for "the braid"** — the structure that carries delimited continuations
and interaction-net crossings intact through lowering. eBPF is a clean early
exercise of the *front-end/build-time* coupling on a target whose gate is external
and unforgiving.

**Why eBPF is a useful first bite.** The initial subset can concentrate on
structural checks and bounded arithmetic, with an independent oracle for
admission ([01](01_verifier_as_design_time_contract.md)). That oracle judges the
delivered bytecode under the host's rules. It does not validate every compiler
proof, payload preservation, application policy or numerical accuracy. Those
remain separate obligations; kernel, JIT and helper contracts remain explicit
external assumptions.

## Obligations, tier by tier

### Tier 1 — structural facts carried as coeffects

These classifications are inputs to later obligations. Dimensional unification
has its own algebra; it does not by itself establish pointer bounds, numeric
coverage or the cost of an escape analysis.

- **Target capabilities.** The program may use only representations and
  operations the target declares. Removing unsupported candidates does not prove
  that a remaining representation covers the inferred range; coverage is its own
  obligation. The initial integer-only subset need not introduce numerical
  selection merely to classify and route an opaque packet payload.
- **Context typing.** The `BpfProgram` root's signature `XdpMd -> XdpAction` is a
  type fact. Proving that a computed access stays inside that context also needs
  the range and dominance obligations below.
- **Declared pointer-free layout.** Map-value descriptors can exclude pointer
  fields structurally. That check requires a resolved layout; it does not prove
  that arbitrary integer bytes never encode a pointer. Information-flow claims
  need their own premises and evidence.
- **Stack-scoped classification.** Escape analysis assigns each value a class in
  the `stack < arena < heap < static` lattice; for BPF, "heap" is simply not in
  the capability set, so any value classified there is a witnessed failure before
  any byte-budget arithmetic runs.

The load-bearing precedent: **escape classification is already a coeffect
discharged this way.** The memory-coeffect design treats DMM as a coeffect
discipline on the PSG, one tier above dimensional consistency, with lifetime
promotion checked as a `QF_LIA` inequality. The BPF stack obligation is the same
machinery with the lattice ceiling lowered.

### Tier 2 — `QF_LIA` / `QF_BV`, the verifier's arithmetic

For this subset, bounded arithmetic obligations can use the following fragments
once their input facts have been established. The compiler must select the
fragment that models the actual operation, including machine wrap where relevant.

Range evidence combines dataflow and guards with checked library laws and their
premises; boundary declarations constrain what must fit. These are named facts
with different roles, not competing authority tiers. A declared capacity limits
what an endpoint admits; it does not prove the program's value lies within that
capacity. Conversely, a sound inferred interval extending beyond a boundary is
not itself a reachable counterexample: its excess may come from approximation.
Refine the enclosure or establish the missing relation before committing an
access. An unresolved containment obligation must prevent that commitment, but
the diagnostic must distinguish missing proof from an exhibited violating value.
See [range evidence and boundary constraints](../../../clef-lang-spec/spec/numeric-selection.md#3-range-evidence-and-boundary-constraints).

- **Loop termination → justified premises and a bound obligation.** The initial
  subset requires bounds established by dataflow, a library law with checked
  premises, or a declared input/helper contract whose assumptions remain visible.
  Once the trip-count relation is linear, its containment in a declared limit is
  a `QF_LIA` query. Deriving that relation is not automatically linear or free.
  Current D10 introduces no source seal or width-named numeric type as a remedy.
  An unresolved bound stays pending during elaboration and becomes a located,
  stable diagnostic when admission requires it. Legible emission — including
  `bpf_loop` or supported iterators — belongs to [03](03_lowering_and_artifacts.md).

  General termination remains undecidable. The proposed subset accepts only the
  loop forms and premises its analyses support; it does not turn arbitrary
  termination into a decidable question or authorize an unchecked assertion.

- **Pointer bounds → a range containment obligation.** Every packet/map access
  must be dominated by a guard proving the offset in range. Interval analysis (the
  same image computation that drives width inference) produces the range; the
  obligation "0 ≤ off ∧ off + len ≤ end" is `QF_LIA` when the arithmetic is
  justified over integers; wrap-sensitive operations need a suitable bit-vector
  model or established no-overflow premises. The *emission* of the
  dominating guard in a verifier-legible shape is the legibility contract; the
  *proof that a guard suffices* is this obligation.

- **Register ranges → range facts, threaded.** The verifier tracks per-register
  ranges through control flow. Compiler range facts must be sound for the chosen
  operations and legible in the emitted guards; the two analyses need not have
  identical internal states. Downstream containment chooses `QF_LIA` or `QF_BV`
  according to those semantics.

- **Stack byte budget → a linear sum.** Given stack-scoped classification (Tier 1),
  the obligation "Σ frame_bytes ≤ 512" is a single linear inequality — the
  memory-coeffect pattern, one axis over.

- **Helper capability → candidate-set filter, not a score.** Helper/kfunc
  availability for the program type on the pinned host is a set-membership check
  against the descriptor's version-ranged matrix ([02](02_platform_shape.md)). A
  missing capability is a witnessed failure naming the version that satisfies it —
  the capability-gate discipline verbatim, never a silent substitution.

### Advisory — the analysis-budget estimate

The verifier has host-dependent analysis limits and can reject programs that
exhaust them. A **complexity-estimate coeffect**
(path-state growth as a function of branch structure) lets the language server
warn "this program's estimated verifier cost approaches the budget; consider
splitting via tail calls" before a load fails. This one is honestly heuristic at
first — an estimate, not a proof — and should be labeled as such; it sharpens as
it is calibrated against the CI oracle.

## Why this rides the existing coeffect frame, not a new one

The through-line the capstone and the framework's verification design both insist
on: **these obligations are not a bolt-on proof pass; they are coeffects on the
PSG, read (not recomputed) by later stages.** The preservation chain the framework
already defines —

```
Dimension --range--> Representation --width--> Footprint --escape--> Allocation
```

— is a chain of coeffects whose required facts are established before commitment
and carried forward, with unresolved premises kept visible during elaboration.
eBPF adds obligations that hang off the *same arrows*: representation selection
(no FP) hangs off `range→Representation`; the stack budget hangs off
`escape→Allocation`; loop bounds and pointer ranges are the interval-analysis
inputs to `Dimension→range` reused. The eBPF admissibility bundle is a *reading*
of coeffects the pipeline already computes, plus a small number of new Tier-2
obligations discharged by the same SMT-dialect mechanism as deadlock-freedom and
lifetime promotion. That is why the proof half composes from standing art rather
than requiring new theory: the target is adversarial, but the analysis is the one
the framework was built to perform.

## The staged discharge, made concrete for a BPF program

1. **Design time (front end).** As the developer writes an XDP filter, the PSG
   accrues coeffects and obligations. Structural capability checks and supported
   arithmetic queries use their respective procedures. Diagnostics identify the
   source span, boundary declaration and failed or unresolved premise. A solver
   timeout or unknown result is not discharge. The resulting claim is scoped to
   the supported subset, pinned host and recorded assumptions.
2. **Build time (middle end).** Each MLIR lowering pass toward the BPF object is
   translation-validated by the SMT-dialect: a transformation that would move an
   access out from under its dominating guard, or unroll a loop past its certified
   bound, or otherwise deform a certified property, must be rejected when
   preservation cannot be established. Evidence is tied to the actual lowered
   artifact, not just the source-level claim.
3. **External grade.** The CI oracle loads the object on the pinned kernel matrix
   (and separately checks supported objects with PREVAIL). Agreement is evidence
   for admission on those hosts, not a general validation of the proof engine.
   Unexpected rejection is a reproducible contract, analysis or emission bug —
   see [01](01_verifier_as_design_time_contract.md). Payload and numerical
   preservation need their own checks, as [05](05_threebody_integration.md) explains.

## The braid connection, briefly

The pending-patent framing is that hard-coupling design-time and build-time proof
dispatch yields graphs **better optimized for the braid** — the non-separable
crossing of sequential control with spawned parallel width, carried with its
crossings intact through lowering. eBPF does not itself stress the braid (an XDP
filter is largely straight-line with bounded loops), which is *why it is a good
first target for the coupling*: it exercises the front-end/build-time proof
handshake on obligations that are almost all Tier 1/2, without simultaneously
demanding the non-abelian braid sheaf that remains open research. The braid rides
on delimited continuations and interaction nets preserved through MLIR; eBPF rides
on the *same preservation discipline* applied to verifier guards. Proving the
coupling on a bounded eBPF obligation set is a rehearsal for the harder braid
demonstration later; tractability must be measured on the actual queries.

## What this waits on, honestly

- **Tier-2 SMT integration.** Solvers are available; the eBPF path needs
  supported obligations connected to the PSG, source diagnostics and target
  declarations. It does not depend on inventing source seal attributes.
- **The SMT-dialect obligation as an in-IR operation.** Named as current focus in
  the framework's own status notes; the build-time half of the staged discharge depends
  on it. A smaller front-end-plus-load-test slice can supply bounded evidence
  before complete lowering preservation is implemented.
- **A bounded initial subset.** The first admission examples can use structural
  checks and linear/bit-vector bounds without requiring the full relational or
  nonlinear machinery. A program using a library lemma must still justify that
  lemma's premises; broader application claims do not inherit admission's scope.

The deliverable is a bounded compiler-to-kernel agreement result, with located
diagnostics and explicit assumptions. It supports the later composition without
claiming that admission proves the whole application correct.
