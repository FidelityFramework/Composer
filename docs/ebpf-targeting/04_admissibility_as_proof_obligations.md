# Admissibility as Proof Obligations

**SpeakEZ Technologies | Fidelity Framework**
**July 2026 — exploratory design note**

This document places eBPF admissibility inside the graduated verification model:
which obligations arise, which tier and fragment each falls in, how they ride the
existing coeffect machinery, and why eBPF is the smallest complete, externally
graded deployment of a proof infrastructure that is — today — design-mature but
implementation-early. It is written against the framework's own proof-layer
design notes and the [Decidable By Construction](../../../arxiv-papers/decidable-by-construction.md)
capstone; it does not assume that layer is built, and it is explicit about what it
waits on.

> **Solver naming.** This note writes obligations in **SMT-LIB2** and names
> fragments (`QF_LIA`, `QF_BV`), not solvers. Z3 is the first and easiest reach
> for demonstrations, but the fragments here are deliberately narrow so that
> CVC5 or a specialized decision procedure can discharge the hot paths where that
> preserves performance. The architecture commits to the *fragment*, not the
> engine.

## The state of the proof layer (read this first)

The framework's verification design is unusually complete on paper and explicitly
pre-implementation in the compiler. The PSG, elaboration, and saturation pipeline
exist; **the proof layer that reads obligations off the graph and discharges them
is upcoming work.** The graduated model is designed as:

| Tier | Fragment | Discharge | Cost |
|---|---|---|---|
| **Tier 1** | `ℤⁿ` (dimensions, grades, escape lattice) | abelian-group unification; graph-integrity via SMT | free by parametricity |
| **Tier 2** | `QF_LIA` (+ `QF_LRA`, `QF_BV`) — bounds, ranges, lifetimes, acyclicity | SMT, sound and complete | moderate |
| **Tier 3** | restricted probabilistic / nonlinear | SMT + library lemmas proved once | higher |
| **Tier 4** | relational / probabilistic (constant-time, crypto) | pRHL checker + proof-assistant kernel | highest |

The proof dispatch is **dual-pass**: obligations are discharged at design time
(weakest-precondition reading, surfaced live through the Lattice language server)
and **re-validated at each MLIR lowering pass** through the SMT-dialect
translation-validation mechanism (the consequence-rule reading — every lowering
must preserve what was proven). This dual coupling of a *front-end (design-time)*
and *middle-end (build-time)* proof dispatch is the subject of a pending patent;
its stated payoff is not only safer computation graphs but graphs **better
optimized for "the braid"** — the structure that carries delimited continuations
and interaction-net crossings intact through lowering. eBPF is a clean early
exercise of the *front-end/build-time* coupling on a target whose gate is external
and unforgiving.

**Why eBPF is the right first bite.** Its obligations are almost entirely Tier 1
and Tier 2 — no Tier 4 relational reasoning, a bounded and enumerable obligation
set, a fixed target subset, and (uniquely) an external oracle that grades the
result ([01](01_verifier_as_design_time_contract.md)). It is the minimal complete
deployment of the design: small enough to build against an early proof layer, real
enough that the kernel checks our work.

## Obligations, tier by tier

### Tier 1 — free, carried as coeffects, no solver query

These fall out of elaboration; the capstone's whole argument is that this class is
decidable, polynomial, and principal over `ℤⁿ`, at negligible marginal cost.

- **No floating point.** Representation selection over the target's capability set;
  with the FP capability absent, IEEE and posit fall out of the candidate set by
  the coverage filter, leaving fixed-point or a witnessed `coverage-empty` error.
  A representation-selection coeffect, not a proof query.
- **Context typing.** The `BpfProgram` root's signature `XdpMd -> XdpAction` is a
  type fact; accessing outside the typed context window is a type error, not a
  verifier surprise.
- **No pointer leaks.** Map value types are pointer-free records by construction —
  a "cannot form" property in the same sense the capstone gives buffer overflows
  at Tier 1.
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

This is the bulk of what the kernel verifier itself computes, and it is exactly
the fragment the graduated model designates for Z3-class discharge.

- **Loop termination → a bound obligation.** eBPF forbids loops without a provable
  bound. The three-tier *range authority* model (intrinsic dataflow / library law
  / developer seal) supplies iteration bounds the same way it supplies value
  ranges: inferred where dataflow fixes them, supplied by a library law, or
  developer-sealed — with lower tiers becoming diagnostic obligations. The
  obligation "this loop's trip count ≤ N" is `QF_LIA`. Where no authority
  establishes a bound, the program *correctly falls through* to a witnessed
  failure — the same loud fall-through the ThreeBody close-encounter node relies
  on, here meaning "seal this loop or it cannot be admitted." (Legible emission —
  `bpf_loop`, open-coded iterators — is the [03](03_lowering_and_artifacts.md)
  half; the *bound proof* is here.)

  A subtlety worth stating: general loop termination is undecidable (it is Tier 3+
  in the model, and outside all tiers in the limit). eBPF sidesteps this exactly
  as the capstone's title move prescribes — by **construction**: the source
  language admits only loops whose bound is establishable by an authority tier, so
  the admissible subset is decidable not because termination became decidable but
  because the un-boundable programs cannot be expressed as admissible.

- **Pointer bounds → a range containment obligation.** Every packet/map access
  must be dominated by a guard proving the offset in range. Interval analysis (the
  same image computation that drives width inference) produces the range; the
  obligation "0 ≤ off ∧ off + len ≤ end" is `QF_LIA`. The *emission* of the
  dominating guard in a verifier-legible shape is the legibility contract; the
  *proof that a guard suffices* is this obligation.

- **Register ranges → range facts, threaded.** The verifier tracks per-register
  ranges through every path. Interval analysis carries the identical information
  as a coeffect; downstream containment obligations are `QF_LIA`/`QF_BV`.

- **Stack byte budget → a linear sum.** Given stack-scoped classification (Tier 1),
  the obligation "Σ frame_bytes ≤ 512" is a single linear inequality — the
  memory-coeffect pattern, one axis over.

- **Helper capability → candidate-set filter, not a score.** Helper/kfunc
  availability for the program type on the pinned host is a set-membership check
  against the descriptor's version-ranged matrix ([02](02_platform_shape.md)). A
  missing capability is a witnessed failure naming the version that satisfies it —
  the capability-gate discipline verbatim, never a silent substitution.

### Advisory — the analysis-budget estimate

The verifier abandons analysis after ~1M instructions of path state, and can
reject *correct* programs for exhausting it. A **complexity-estimate coeffect**
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

— is a chain of coeffects, every arrow settled at design time and carried forward.
eBPF adds obligations that hang off the *same arrows*: representation selection
(no FP) hangs off `range→Representation`; the stack budget hangs off
`escape→Allocation`; loop bounds and pointer ranges are the interval-analysis
inputs to `Dimension→range` reused. The eBPF admissibility bundle is a *reading*
of coeffects the pipeline already computes, plus a small number of new Tier-2
obligations discharged by the same SMT-dialect mechanism as deadlock-freedom and
lifetime promotion. That is why the proof half composes from standing art rather
than requiring new theory: the target is adversarial, but the analysis is the one
the framework was built to perform.

## The dual-pass, made concrete for a BPF program

1. **Design time (front end).** As the developer writes an XDP filter, the PSG
   accrues coeffects; the language server discharges the Tier-1 classifications
   for free and the Tier-2 obligations (loop bound, pointer range, stack sum,
   helper availability) as `QF_LIA`/`QF_BV` queries, surfacing any failure at the
   offending span. The program is *known admissible before it is compiled.*
2. **Build time (middle end).** Each MLIR lowering pass toward the BPF object is
   translation-validated by the SMT-dialect: a transformation that would move an
   access out from under its dominating guard, or unroll a loop past its certified
   bound, or otherwise deform a certified property, is rejected. This is the
   contractual answer to the clang-vs-verifier fight — the optimizer *cannot*
   optimize a proof away.
3. **External grade.** The CI oracle loads the object on the pinned kernel matrix
   (and PREVAIL). Agreement is the expected case; disagreement is a reproducible
   bug against either the proof layer (unsound) or the emission vocabulary
   (illegible) — see [01](01_verifier_as_design_time_contract.md).

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
coupling on eBPF's tractable obligation set is a rehearsal that de-risks the
harder braid demonstration later — same handshake, decidable payload.

## What this waits on, honestly

- **Tier-2 SMT integration.** The solver is available; the integration with Clef
  attribute syntax and the PSG coeffect infrastructure is in design. eBPF's
  obligations are squarely in the fragment that integration targets first, so eBPF
  is a *driver* for it, not blocked behind something more distant.
- **The SMT-dialect obligation as an in-IR operation.** Named as current focus in
  the framework's own status notes; the build-time half of the dual-pass depends
  on it. Until it lands, the design-time half plus the external oracle already
  gives a working (if less airtight) admissibility story.
- **Nothing at Tier 3/4.** eBPF admissibility deliberately requires none of the
  probabilistic/relational machinery, the lemma library, or the pRHL checker.
  That is the point: it is the deployment you can build while those mature.

The honest summary: the proof *design* is ready and the proof *engine* is early,
and eBPF is precisely the target whose demands sit inside the ready part while
supplying an external oracle that grades the early engine as it comes up.
