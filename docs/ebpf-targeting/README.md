# eBPF Targeting: The Kernel as a Verified Substrate

**SpeakEZ Technologies | Fidelity Framework**
**July 2026 — exploratory design series; nothing here is etched in stone**

This series designs a bounded Clef-to-eBPF path whose compilation results agree
with admission on a pinned kernel profile. It places the required obligations in
the Clef/CCS/Composer pipeline, so failures can be explained while editing and the
delivered bytecode can be checked independently. The composed implementation and
its agreement evidence remain work to establish.

## The thesis in one sentence

**Within the supported subset and pinned host contract, compilation should agree with kernel admission.**

The Linux kernel verifier is an abstract interpreter run at load time: it tracks
per-register value ranges, checks supported loop bounds, bounds stack depth, and checks
pointer provenance before a program is allowed to exist in the kernel. Fidelity
already runs the same *family* of analysis at design time — integer interval
analysis, escape classification, the numeric-selection machinery's range-driven
representation choice. The [Decidable By Construction](../../../arxiv-papers/decidable-by-construction.md)
capstone motivates restricting the accepted questions. Linear containment can
use QF_LIA; wrap-sensitive arithmetic can use QF_BV; capability membership is a
structural check. Establishing the premises is a separate task. Loop bounds come
from supported inference, checked library laws or declared contracts, with no
source-seal escape hatch. Decidability does not bound solver cost: timeout,
unknown or missing premises remain unresolved, and yield a stable, located
diagnostic when admission requires discharge.

The claim has two halves, and both are required:

1. **The proof half.** The supported subset's obligations are discharged per
   site by their designated procedures, with premises and external assumptions
   retained in the joint obligation graph through lowering.
2. **The legibility half.** Passing the verifier is not just about *being* safe —
   it is about being *visibly* safe in the specific idioms the checker recognizes.
   The notorious pain of the clang→BPF path is the optimizer transforming a sound
   bounds check into a shape the verifier cannot track. Fidelity owns every emitted
   shape: witnesses observe coeffects, patterns elide known-good idioms, and
   transformations must establish preservation before proceeding. CI then checks
   the actual artifact on the pinned hosts. This is evidence about admission;
   payload preservation, application policy and numerical accuracy have separate
   contracts. The verifier, JIT and helpers remain part of the trusted boundary.

The [platform admission and handoff reference](../../../Fidelity.Platform/docs/ADMISSION_AND_SIDECARS.md)
now supplies checked `.clef` source packages and an F# test runner. Linux,
Windows and macOS classic BPF have separate host contracts; Strix Halo/Arty is
the ThreeBody physical reference, with Metal as another UMA realization. These
checks do not yet derive compiler proof obligations or emit/load BPF artifacts.

## Why this target is worth the trouble

eBPF is not one more ISA. It is the second member (after WASM) of a class the
platform taxonomy now separates under `AbstractMachines/`: **hosted, verified ISAs** — targets where
admission is gated by a checker rather than by physics (FPGA) or an ABI (CPU).
Designing for the class rather than the member pays twice.

Three properties make it strategically valuable *now*:

- **It exercises the proof infrastructure.** A bounded target subset gives
  the joint obligation graph, diagnostics and preservation checks a concrete
  compiler-to-kernel agreement criterion.
- **It comes with an independent admission oracle.** The kernel verifier and
  separate PREVAIL checks judge delivered objects under their supported rules.
  Their agreement is useful artifact evidence, not validation of every claim
  the compiler or application makes.
- **It makes failures actionable earlier.** Host-dependent verifier budgets
  and bytecode legibility remain deployment constraints. Design-time analysis
  can surface those concerns in the editor; it does not eliminate load-time
  checking or guarantee that a solver query is cheap.

## What eBPF forces, waits on, and validates

| Relationship | Item |
|---|---|
| **Requires implementation** | Source obligation extraction and proof dispatch; capability-driven witness selection; exact host fact resolution over BAREWire availability metadata; final-artifact inspection and BTF emission |
| **Waits on** | A composed path connecting supported obligations, diagnostics, preservation through lowering and pinned-host artifact checks |
| **Validates** | Admission agreement for the tested subset and hosts; use of shared platform declarations for hooks, helpers and limits |

## The io_uring companion

Two contemporary kernel developments redraw the same boundary in opposite
directions: eBPF pushes verified compute *down* into the kernel; io_uring pulls
the syscall *out* of the I/O hot path via shared-memory rings. io_uring is not a
compilation target — it is a userspace I/O concern, a candidate Linux
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

- [C-series acceptance](../PRDs/C-Series-Acceptance.md#41-integrity-through-realization-colibri-fpga-and-ebpf)
  applies the distinction between semantic settlement, faithful realization and
  independent artifact admission to current Baker/Alex work. This shared
  integrity requirement does not depend on completing an eBPF deployment first.
- [wasm-targeting/](../wasm-targeting/) — the other hosted verified ISA; the
  class-level constructs proposed here (versioned capability matrix,
  capability-keyed witness gating) are designed for both.
- [javascript-targeting/](../javascript-targeting/) — the type-carrying
  discussion that motivates BTF-from-Clef-types in [03](03_lowering_and_artifacts.md).
- [Numeric_Selection_Implementation.md](../Numeric_Selection_Implementation.md)
  and the normative spec it points to — the ratified vocabulary (coverage
  filters, capability gates, located failures, and range evidence checked against
  boundary constraints)
  that the admissibility story reuses rather than reinvents.
