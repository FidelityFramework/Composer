# ThreeBody Integration: The Kernel in the Heterogeneous Weave

**SpeakEZ Technologies | Fidelity Framework**
**July 2026 — exploratory design note**

ThreeBody routes each regime of a gravitational simulation to the substrate that
suits it — close encounters to an FPGA (b-posit + quire, where IEEE FP64
catastrophically cancels), the bulk to a GPU, the far field to an NPU,
orchestration to the CPU. This document adds the substrate ThreeBody is currently
missing: the **kernel**, in two roles. The headline role is the one that turns
heads — the **CPU↔FPGA weld over a Layer-2 link**, where eBPF/XDP routes the
FPGA's frames *without traversing the full network stack*, making the sidecar a
genuine two-way kernel/userspace participant whose computational integrity is
preserved across the boundary. The second role is observation: kernel-verified
instrumentation that lets the demo watch itself.

> **Status honesty.** ThreeBody is documentation-only today — empty `src/`, and
> Composer's numeric-selection guide §13 labels it "a design proposal to rebuild"
> whose repo docs carry drift the spec now polices (quire size, `es`, the
> four-architecture framing). The eBPF track must **not** hang its MVP on
> ThreeBody's rebuild. eBPF gets its own hello-world progression
> ([below](#the-ebpf-hello-progression-comes-first)); ThreeBody integration is
> the *later synthesis*, honoring §13's corrections.

## The headline: the FPGA weld as a proof of fusion

ThreeBody reaches the Arty A7 sidecar over a **Layer-2 network link** — not a
local bus. That choice is not incidental; it is the demo's most interesting claim.
A user-space-only heterogeneous demo (marshal, `send()`, `recv()`, unmarshal
through the full TCP/IP stack) is *satisfying* but unremarkable. A demo where the
close-encounter frames reach the FPGA and return **through kernel-level routing
that skips the stack** is a different statement: that the CPU and FPGA are welded
into one computation whose integrity survives the substrate crossing, at a latency
the full stack cannot offer.

The physics makes this a hard requirement, not a flourish. Close encounters are
~0.1% of interactions but they are the *timestep-critical* ones — the integrator
cannot advance until the exact b-posit force comes back from the sidecar. Every
microsecond of stack traversal on that path is integrator stall. So the transport
for the FPGA weld wants to be:

- **On the fast path in:** an **XDP** program at the driver, before the `sk_buff`
  exists, recognizing the sidecar's return frames by their L2 signature and
  `XDP_REDIRECT`-ing them straight to the consumer — or into an **AF_XDP** socket
  whose UMEM rings the orchestrator reads with zero copy. The frame never climbs
  the IP/TCP layers it does not need.
- **On the fast path out:** frames to the sidecar emitted through the same
  AF_XDP path, bypassing the stack in the other direction.
- **Bounded and legible:** the XDP classifier is exactly the kind of program
  [01](01_verifier_as_design_time_contract.md)–[04](04_admissibility_as_proof_obligations.md)
  describe — straight-line parse, one bounded loop at most, a map lookup, a verdict
  — admissible *by construction* from Clef source.

This is where the two kernel transcripts fuse. **eBPF/XDP** is the verified
routing that keeps the FPGA frames off the slow path; **io_uring / AF_XDP shared
rings** are the zero-copy hand-off between kernel and orchestrator — BAREWire's
zero-copy philosophy meeting the kernel's own ring buffers. (io_uring itself is a
userspace-runtime concern of the async track, not a compile target; it appears
here only at this seam.)

### Why the weld preserves computational integrity

The claim that turns heads is not "it's fast." It is that **the fusion does not
deform the computation.** Three properties, each traceable to standing framework
art:

1. **The number type crosses intact.** The close-encounter contract is b-posit32 +
   800-bit quire, sealed at the force site (Tier 3 seal, per numeric-selection
   §13). BAREWire carries that value across the L2 link as schema-verified,
   zero-copy bytes — the *same* representation the FPGA computed and the CPU
   integrator consumes. No re-encoding, no silent widening, no JSON-lossy
   round-trip. The dimensional metadata (`float<N>`, the force law's units) rides
   with it. The weld is a data-plane crossing that a *type* survives, which is the
   javascript-targeting / BTF thesis ([03](03_lowering_and_artifacts.md)) applied
   to a wire.

2. **The routing is proven not to corrupt.** The XDP classifier that redirects the
   frames is itself admissible-by-construction: its bounds are proven, its map
   accesses guarded, its verdict typed. The frame that reaches the orchestrator
   was routed by a program the kernel certified cannot have mangled it. Contrast a
   hand-written C XDP program whose correctness is a load-time gamble — here the
   routing layer inherits the same design-time guarantee as the physics.

3. **The crossing is a supervised, two-way process.** The FPGA actor
   (`FpgaBPosit`) already lives under Prospero's OneForOne supervision, and the
   demo's fault story is "USB-C disconnect → actor dies → restart on reconnection."
   Over an L2 link, the *link itself* becomes the observable: the XDP path can
   feed real frame-arrival timing and loss into the supervisor, so a degraded or
   silent sidecar is detected from kernel-level evidence rather than a userspace
   timeout. The weld is genuinely bidirectional — commands and frames out,
   forces and link-health in — and the health signal is itself kernel-verified.

The heterogeneous-compute thesis was: *place each regime on the substrate whose
structure fits it.* The FPGA weld extends the thesis one level — **place the
transport of a regime on the substrate whose structure fits it too.** The
timestep-critical path gets kernel-verified, stack-bypassing routing because that
is what its latency and integrity demand, and that placement is a design-time
decision exactly like the regime routing above it.

## The second role: the observation plane

The demo already reserves a `Telemetry` actor for "hardware counter collection +
display." eBPF completes the thesis by making the *observation regime* another
placed substrate: kernel-verified instrumentation, compiled from the same language
as the physics, streaming into that actor.

- **USB URB / link latency** to the sidecar (a kprobe or the XDP path's own
  timestamps) — the ground truth behind Prospero's restart decisions.
- **GPU submission latency** (ioctl / fence-wait kprobes on the DRM path) — is the
  medium-distance regime keeping up?
- **Scheduler latency** across the actor threads (`sched` tracepoints) — are the
  four regime actors getting their cores?

Each stream is integers — histograms and counters, which is all eBPF can compute
(no FP) and exactly what telemetry wants — framed as BAREWire records through a
ring-buffer map into the Telemetry actor, rendered in the demo's own Wayland
panels (`Platform.Display`, no WebView). **The demo watches itself with
kernel-verified instrumentation.** Every regime of the problem — including the
meta-regime of observing the problem, and the transport of the regimes — is placed
by design-time analysis.

## The full picture

```
                         ┌─────────────────────────────────────────┐
                         │  Orchestrator (CPU actor, Clef→native)   │
                         │  timestep · regime classify · Prospero   │
                         └───▲───────────────▲──────────────▲───────┘
  forces (b-posit+quire) │   telemetry    │   verdicts   │
        BAREWire, zero-copy  │  (ring buffer)  │             │
                    ┌────────┴───────┐  ┌──────┴──────┐  ┌───┴────────┐
                    │  AF_XDP UMEM   │  │ ringbuf map │  │  GPU / NPU │
                    │  rings (kernel)│  │  (kernel)   │  │  actors    │
                    └────────▲───────┘  └──────▲──────┘  └────────────┘
     ══ L2 link ══▶ ┌────────┴───────┐         │
     Arty A7 frames │  XDP program   │─────────┘  observation-plane probes
     (close-encntr) │  (Clef→BPF .o) │            (kprobe/tracepoint → ringbuf)
                    │  parse·guard·  │            all Clef→BPF, admissible
                    │  REDIRECT      │            by construction
                    └────────────────┘
```

Two kernel-resident Clef→BPF artifacts (the XDP router; the probe set), both
admissible by construction; two kernel/userspace zero-copy seams (AF_XDP rings for
the data plane, a ring-buffer map for telemetry); one L2 weld that a sealed
b-posit+quire value crosses without deformation.

## The eBPF hello progression comes first

Before any of the above, eBPF earns its place with a standalone progression
modeled on FidelityHelloWorld / HelloArty — each step a compile→load→run proof
that the pipeline works, graded by the external oracle:

| Step | Program | Proves |
|---|---|---|
| **B-01** | XDP packet counter, per-CPU array map | The whole pipeline: `DeclRoot.BpfProgram`, the LLVM BPF artifact tail, a map, load + attach + run |
| **B-02** | XDP drop-by-blocklist, `.rodata`/array-map config from userspace | The capability gate + static-placement of pushed-down config; a bounded map-lookup verdict |
| **B-03** | kprobe latency histogram → ring buffer | The observation-plane primitive; ring-buffer streaming to userspace |
| **B-04** | AF_XDP redirect to a userspace ring | The data-plane hand-off the FPGA weld needs |
| **B-05 (synthesis)** | ThreeBody FPGA-weld router + telemetry probes | The full circle, on a rebuilt ThreeBody honoring §13 |

B-01 through B-04 are independent of ThreeBody and prove the target on their own.
B-05 is the synthesis, gated on ThreeBody's rebuild — not on eBPF.

## Corrections inherited from ThreeBody §13

When B-05 is built, it inherits the numeric-selection guide's required
corrections so the demo argues *for* the framework rather than against it:
normalize to natural units (G = 1) so constants do not land in the wide-dynamic
band that routes to IEEE; 800-bit quire (fixed 25×32-bit vector for b-posit, independent of precision), not 512-bit;
`eS = 5`; a symplectic/time-reversible integrator with an independent
high-precision reference; conserved-quantity drift (energy, angular momentum, and
the currently-omitted linear momentum) as the witnessed evidence. The eBPF layer
does not touch the arithmetic; it routes and observes. But the synthesis demo is
only credible on a corrected simulation, so B-05 waits for that rebuild by design.

## Why this is the full-circle moment

HelloArty proved width inference to real silicon: dimensional and coeffect
guarantees carried into an Artix-7 bitstream through place-and-route. The eBPF
weld proves the *complementary* half — that a design-time proof obligation
(admissibility) can be discharged such that a **verified artifact loads into a
production kernel by construction**, and that the resulting kernel/userspace
fusion carries a computation across substrates without deforming it. One end of
the story is "our proofs reach the fabric." The other is "our proofs reach the
kernel, and weld two substrates into one integer-honest, posit-exact computation
that skips the stack." ThreeBody is where both ends meet in a single running
demo — the heterogeneous weave with the kernel finally in it.
