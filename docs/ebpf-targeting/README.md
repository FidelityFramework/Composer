# eBPF Targeting: The Kernel as a Verified Substrate

**SpeakEZ Technologies | Fidelity Framework**
**July 2026 — exploratory design series; nothing here is etched in stone**

This series captures the shape of a problem space: compiling Clef to eBPF such that
admission by the kernel's verifier is a *compile-time guarantee* rather than a
load-time gamble. It exists to sound out where eBPF targeting should reside in the
Clef/CCS/Composer maturation pipeline — what it forces, what it waits on, and what
it uniquely validates.

## The thesis in one sentence

**If it compiles, it loads.**

The Linux kernel verifier is an abstract interpreter run at load time: it tracks
per-register value ranges, proves loop termination, bounds stack depth, and checks
pointer provenance before a program is allowed to exist in the kernel. Fidelity
already runs the same *family* of analysis at design time — integer interval
analysis, escape classification, the numeric-selection machinery's range-driven
representation choice. The [Decidable By Construction](../../../arxiv-papers/decidable-by-construction.md)
capstone gives this a precise home: the verifier's obligations are almost entirely
expressible in QF_LIA — Tier 2 of the graduated verification model — and the one
obligation that is undecidable in general (termination) becomes decidable *by
construction* when the source language admits only loops with provable bounds.
That is the capstone's titular move, applied to a new gatekeeper.

The claim has two halves, and both are required:

1. **The proof half.** Every obligation the verifier checks is discharged at
   design time, per site, through the SMT-LIB2-mediated proof dispatch that the
   graduated verification model already specifies.
2. **The legibility half.** Passing the verifier is not just about *being* safe —
   it is about being *visibly* safe in the specific idioms the checker recognizes.
   The notorious pain of the clang→BPF path is the optimizer transforming a sound
   bounds check into a shape the verifier cannot track. Fidelity owns every emitted
   shape: witnesses observe coeffects, patterns elide known-good idioms, and
   nothing between the PSG and the bytecode is permitted to optimize a guard out
   of legibility. Admissibility = semantic safety ∧ syntactic legibility.

## Why this target is worth the trouble

eBPF is not one more ISA. It is the second member (after WASM) of a class the
platform taxonomy does not yet name: **hosted, verified ISAs** — targets where
admission is gated by a checker rather than by physics (FPGA) or an ABI (CPU).
Designing for the class rather than the member pays twice.

Three properties make it strategically valuable *now*:

- **It is a forcing function for the proof infrastructure.** Clef/CCS does not yet
  have proof-discharge capacity; that design is early. eBPF admissibility is the
  smallest complete, externally-validated deployment of the graduated verification
  model: a bounded set of QF_LIA obligations, a fixed target subset, and a
  measurable success criterion. The way MCU work forces the reactive intrinsics,
  eBPF forces the SMT-LIB2 adaptation.
- **It comes with a free, adversarial oracle.** The kernel verifier (and PREVAIL
  on Windows — a published, principled abstract interpreter) independently checks
  every claim our proof engine makes. A CI gate that loads every compiled probe is
  an empirical validation harness *for the prover itself*. No other planned target
  grades our proof work for us.
- **The cost-structure argument applies verbatim.** The verifier is post-hoc
  enforcement: every load pays the analysis cost again, bounded by a one-million
  instruction budget, and programs are routinely rejected for being illegible
  rather than unsafe. Design-time discharge pays once, at elaboration, and
  surfaces failures in the editor (through the Lattice language server) with
  source spans — not in `dmesg` after deployment.

## What eBPF forces, waits on, and validates

| Relationship | Item |
|---|---|
| **Forces** | SMT-LIB2 proof dispatch (Tier 2 / QF_LIA) in its first bounded deployment; capability-keyed witness gating (retiring the binary `isCPULike` predicate); a versioned capability matrix in `Fidelity.Platform/Contracts`; BTF emission from Clef types |
| **Waits on** | Nothing hard — the LLVM BPF backend exists today, and the admissibility coeffect layer composes from standing analysis art (interval analysis, escape classification, coeffect plumbing). Proof *certificates* deepen as the proof infrastructure matures |
| **Validates** | The proof engine (via the external oracle); the hooks-as-pins generalization of platform descriptors; the hosted-verified-ISA class design that WASM will reuse |

## The io_uring companion

Two contemporary kernel developments redraw the same boundary in opposite
directions: eBPF pushes verified compute *down* into the kernel; io_uring pulls
the syscall *out* of the I/O hot path via shared-memory rings. io_uring is not a
compilation target — it is a userspace runtime concern, the natural Linux
substrate for Clef's runtime-free coroutine async, and its SQ/CQ rings are
BAREWire's zero-copy philosophy meeting the kernel's. It is treated here only
where the two meet (AF_XDP's shared UMEM rings in the ThreeBody data plane,
[05](05_threebody_integration.md)); its own design belongs to the
concurrency/async track.

## The series

| Doc | Question it answers |
|---|---|
| [01 — The Verifier as a Design-Time Contract](01_verifier_as_design_time_contract.md) | What does the kernel actually demand, and why is Fidelity's existing machinery the right shape for it? |
| [02 — Platform Shape](02_platform_shape.md) | Where does eBPF live in Fidelity.Platform — and what do hosted verified ISAs do to the descriptor schema? |
| [03 — Lowering and Artifacts](03_lowering_and_artifacts.md) | How does a `.clef` source become a loadable BPF ELF object — and when do we stop using LLVM for it? |
| [04 — Admissibility as Proof Obligations](04_admissibility_as_proof_obligations.md) | How do verifier checks become per-site SMT-LIB2 obligations in the graduated verification model? |
| [05 — ThreeBody Integration](05_threebody_integration.md) | What do the data plane and observation plane look like in the heterogeneous-compute demo? |

## Relationship to sibling series

- [wasm-targeting/](../wasm-targeting/) — the other hosted verified ISA; the
  class-level constructs proposed here (versioned capability matrix,
  capability-keyed witness gating) are designed for both.
- [javascript-targeting/](../javascript-targeting/) — the type-carrying
  discussion that motivates BTF-from-Clef-types in [03](03_lowering_and_artifacts.md).
- [Numeric_Selection_Implementation.md](../Numeric_Selection_Implementation.md)
  and the normative spec it points to — the ratified vocabulary (coverage
  filters, capability gates, witnessed failures, three-tier range authority)
  that the admissibility story reuses rather than reinvents.
