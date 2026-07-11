# The Verifier as a Design-Time Contract

**SpeakEZ Technologies | Fidelity Framework**
**July 2026 — exploratory design note**

The Linux kernel will run code it has never seen, under full privilege, on
production machines — on one condition: the code must arrive *provably* unable to
harm the kernel, and the proof must be one the kernel's own checker can
reconstruct. This document catalogs what that checker demands, shows that the
demands are the same *family* of analysis Fidelity already performs at design
time, and states the claim precisely: admissibility is a compile-time property.

## What the verifier actually demands

Before a BPF program may attach to anything, the in-kernel verifier statically
analyzes its bytecode. The obligations, organized by kind:

| Obligation | Mechanism | Notes |
|---|---|---|
| **Termination** | CFG walk; every loop must have a provable bound | Bounded loops verified by simulation since kernel 5.3; `bpf_loop` helper (5.17) and open-coded iterators (6.4) give verifier-friendly forms |
| **Register safety** | Per-register state tracking through every reachable path — not just types, *ranges* | "This register holds a packet pointer that has been bounds-checked; that one holds a value in [0, 255]" |
| **Memory safety** | Pointer provenance + explicit bounds guards dominating every access | Direct packet access requires a `data + off ≤ data_end` comparison the verifier can see |
| **Stack bound** | ≤ 512 bytes combined across bpf2bpf frames | No recursion; call depth limited |
| **Call discipline** | Only numbered helpers and BTF-exported kfuncs valid for the program type | No libc, no arbitrary kernel calls; some helpers gated GPL-only by the program's declared license |
| **Context typing** | Each program type has a fixed context struct (`xdp_md`, `__sk_buff`, `pt_regs`…) and return convention | Access outside the typed window is rejected |
| **Information flow** | No kernel pointers leaked to user-readable storage | A type-level property in a language with typed map values |
| **Analysis budget** | The verifier explores up to ~1M instructions of path state before giving up | Complex branching can exhaust the budget on *correct* programs |
| **No floating point** | The ISA has none | Integer and pointer arithmetic only |

Two structural facts about this gate matter for everything downstream.

**First, it is post-hoc enforcement with recurring cost.** The verifier re-derives
safety from scratch on every load, bounded by its instruction budget, and its
acceptance set is ultimately *its implementation*, not a specification. This is
precisely the cost structure the
[Decidable By Construction](../../../arxiv-papers/decidable-by-construction.md)
capstone critiques in AI reliability infrastructure: the corrective mechanism runs
after the artifact exists, and the price recurs per deployment. The design-time
alternative pays once, at elaboration, and carries the result forward as a
certificate.

**Second, it checks legibility, not just truth.** The verifier accepts programs
whose safety it can *reconstruct within its own idiom vocabulary*. A bounds check
that clang's optimizer has strength-reduced into an induction-variable form the
range tracker cannot follow is rejected even though the program is sound. This is
the notorious failure mode of the C-to-BPF path: developers fight the optimizer,
sprinkle `volatile`, and pin compiler versions, because the toolchain between
their source and the bytecode does not *know* the verifier's idiom set exists.

## The two-halves claim

Admissibility therefore decomposes:

> **Admissibility = semantic safety ∧ syntactic legibility.**

A compiler that guarantees admission must do both: *prove* the safety properties,
and *emit* them in shapes the target checker recognizes. These are different
kinds of work, discharged at different stages:

- **The proof half** is front-end and middle-end analysis: per-site obligations
  read off the PSG and discharged at design time
  ([04](04_admissibility_as_proof_obligations.md)).
- **The legibility half** is emission discipline: the witness/pattern layer emits
  only idioms from a curated, verifier-recognizable vocabulary, and the MLIR
  SMT-dialect translation-validation pass (the dual-pass architecture's build-time
  re-discharge — see `clef-lang-site: docs/internals/verification/proofs-to-silicon.md`)
  rejects any transformation that would deform a certified guard. Where clang
  users fight their optimizer, Fidelity's optimizer is contractually bound to
  preserve what was proven.

Neither half is novel machinery invented for eBPF. The proof half is the
graduated verification model's Tier 1/Tier 2 fragment doing exactly what it was
designed to do. The legibility half is the proof-before-lowering / re-discharge
discipline that already defines the compilation pipeline's shape. eBPF is an
*instance*, not an extension — which is the strongest position a new target can
occupy.

## The isomorphism, obligation by obligation

| Verifier obligation | Fidelity machinery | Where it lives |
|---|---|---|
| Loop termination | Integer interval analysis + the three-tier range-authority model (intrinsic dataflow / library law / developer seal) from the numeric-selection spec, applied to iteration bounds | Tier 1/2; by-construction subset restriction ([04](04_admissibility_as_proof_obligations.md)) |
| Register ranges | Interval analysis — the same image computation that drives width inference and representation selection | Feeds Tier 2 obligations |
| Pointer bounds | Interval analysis on offsets; emission of dominating guards in recognized shapes | Tier 2 (QF_LIA) + legibility contract |
| Stack ≤ 512B | Escape classification (stack-scoped lattice) + a linear byte-sum obligation | Tier 1 classification, Tier 2 sum |
| Helper discipline | The capability gate: a candidate-set filter over the platform descriptor's helper matrix; a missing capability is a witnessed failure, never a silent swap | Descriptor-driven ([02](02_platform_shape.md)) |
| Context typing | Typed program roots — `XdpMd -> XdpAction` as a `DeclRoot` case, exactly as `Design<'S,'R>` roots hardware modules | PSG structure ([03](03_lowering_and_artifacts.md)) |
| No pointer leaks | Map value types are pointer-free records by construction | Type system |
| Analysis budget | A complexity-estimate coeffect (advisory at first — honest about its heuristic nature) | Coeffect carriage |
| No floating point | The numeric-selection coverage filter with an empty FP capability: IEEE and posit representations fall out of the candidate set, leaving fixed-point or a witnessed `coverage-empty` error | Already specified — eBPF is the first target where the FP capability is *entirely absent* |

The last row deserves emphasis. The numeric-selection spec models "the target
lacks a representation" as a first-class, hard-error-producing condition. That
was designed for FPGA and MCU realities; eBPF exercises it in its most extreme
form without requiring a single new rule.

## The free adversarial oracle

Every claim the proof engine makes about a BPF program is independently checked
by machinery we do not control: the Linux verifier at load time, and PREVAIL —
the published, abstract-interpretation-based verifier used by ebpf-for-windows —
on the other OS. This creates something no other Fidelity target offers: an
**external grading harness for the prover itself**. A CI stage that compiles the
probe corpus and loads every object on a matrix of pinned kernels (plus a PREVAIL
run) converts any disagreement into a reproducible bug against either the proof
layer or the emission vocabulary.

Two asymmetries to respect:

- **Their acceptance is not our soundness.** The Linux verifier is conservative
  in ways that shift between kernel versions; it may reject what we correctly
  proved (a legibility gap — our emission vocabulary needs a new certified idiom)
  and it will accept plenty we would not emit. The relationship is: our
  certificate must *imply* their acceptance, never the reverse.
- **PREVAIL is the better-specified partner.** Its zone-domain abstract
  interpretation is described in the literature (Gershuni et al., PLDI 2019),
  which makes the Windows gate — ironically — the cleaner formal contract to
  target. This is why Windows support is carried as a *design forcing function*
  even while day-to-day development happens on a Linux kernel: it keeps the
  design honest that this is an OS-level concern, not a hardware concern, and it
  prevents Linux-verifier folklore from ossifying into the architecture.

## What this buys, concretely

For the developer: admissibility failures appear in the editor, through the
Lattice language server, at the offending expression — "this loop's bound is not
established by any authority tier; seal it or supply a range law" — instead of as
a truncated verifier log after a failed load on a production host.

For the pipeline: a BPF object that leaves Composer carries its discharge record.
The kernel's verifier becomes what a type checker's runtime is to a well-typed
program — a gate you pass by construction, not a boss fight.

For the platform: the same contract vocabulary (obligation catalogs, capability
matrices, emission idiom sets) is exactly what a WASM validator target needs
next. The class is designed once; eBPF is merely its most adversarial member.
