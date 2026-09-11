# ThreeBody Integration: The Kernel in the Heterogeneous Weave

**SpeakEZ Technologies | Fidelity Framework**
**July 2026 — exploratory design note; aligned with the current contract model September 2026**

ThreeBody aims to demonstrate how posit/quire precision can extend useful
prediction and tape-free numerical reversal in a chaotic gravitational
simulation. It compares trajectories with an independent reference and IEEE
controls, then recomputes the return trajectory without a saved tape. The useful
prediction horizon and reversal residual are the measurements that establish
what each representation achieves.

The proposed placement puts close encounters on an FPGA, bulk work on a GPU,
far-field work on an NPU and orchestration on the CPU. This note adds the kernel's
data and observation paths. Those paths, BAREWire's shared contracts and the
compiler's proof machinery support the numerical experiment; they are not its
primary purpose. BAREWire is the glue between these substrates. The same glue
also serves Conclave, a platform for intelligent distributed systems on Cloudflare.

The physical reference is **AMD Strix Halo plus Digilent Arty A7**. The
[platform handoff declarations](../../../Fidelity.Platform/Profiles/StrixHalo_ArtyLab/Handoffs.clef)
now separate Strix CPU/GPU shared backing, host/NIC ownership and the FPGA
request/reply boundary. The [admission design](../../../Fidelity.Platform/docs/ADMISSION_AND_SIDECARS.md)
records the remaining mapping, coherence, wire-layout and timing obligations.
These are reference checks, not a demonstrated ThreeBody hardware path.

> **Status honesty.** ThreeBody is documentation-only today — empty `src/`;
> the placements and supervisor described here are proposals, not an executed
> CPU→kernel→FPGA round trip. The current [numeric-selection specification](../../../clef-lang-spec/spec/numeric-selection.md)
> governs representation and quire adequacy. The eBPF track must **not** hang its
> MVP on ThreeBody's rebuild. eBPF gets its own hello-world progression
> ([below](#the-ebpf-hello-progression-comes-first)); ThreeBody integration is
> the later synthesis.

## The data plane supports the numerical experiment

The proposed Arty A7 sidecar uses a **Layer-2 network link**. Its transport must
preserve the agreed representation and keep the experiment's latency measurable.
Stack bypass is a candidate mechanism, not evidence by itself that the numerical
result is preserved or that a deadline is met.

Close-encounter results can lie on the integrator's critical path. The workload
must establish their frequency, payload and deadline; the transport design is:

- **In:** native-driver **XDP** classifies received frames and redirects matching
  traffic through an XSKMAP to the AF_XDP RX ring.
- **Out:** userspace submits frames through the **AF_XDP TX ring**. This does not
  require outgoing traffic to traverse an XDP receive program.
- **Ownership:** FILL/RX and TX/COMPLETION transfer UMEM ownership. Completion
  returns a buffer for reuse; it is not an application reply or proof of delivery.
- **Bounded and legible:** the classifier is the kind of program
  [01](01_verifier_as_design_time_contract.md)–[04](04_admissibility_as_proof_obligations.md)
  describe — bounded parse, guarded lookup, typed verdict — whose emitted
  bytecode must pass the pinned kernel's verifier.

AF_XDP zero-copy depends on the driver, device and bind mode; copy mode remains
possible. A benchmark must record the selected mode, and a profile requiring
zero-copy must fail when unavailable. Ring producers and consumers must obey
their ownership and synchronization rules. See the [Linux AF_XDP documentation](https://docs.kernel.org/networking/af_xdp.html).
io_uring is a separate userspace I/O mechanism; it is not the AF_XDP transport.

### The joint contract, and what each check establishes

Integrity needs several obligations to agree on the same declarations. Passing
one does not discharge the others:

1. **Numerical meaning and representation.** The application fixes the force
   law, dimensions, normalization and error experiment. A boundary descriptor
   fixes the selected format, widths, byte/limb order and payload layout; the
   compiler checks coverage at that boundary. Under current D10 there are no
   source seals or width-named numeric source types. Names such as b-posit32 and
   800-bit quire describe a boundary format, not an annotation the developer puts
   on a value. Both endpoints must agree before representation erasure: the
   untagged bytes do not carry dimensions or reconstruct the contract themselves.

2. **Kernel admission and payload preservation.** The verifier checks the
   delivered program against its safety rules. A legal program can still modify
   or drop a payload. A read-only classifier therefore needs a separate effect
   or equivalence argument that it preserves the payload bytes it redirects,
   alongside bounds, helper availability and map-layout checks. The kernel,
   JIT, helpers, driver and FPGA implementation remain explicit external
   assumptions unless independently justified. Admission is not a proof of
   force accuracy, packet authenticity or end-to-end delivery.

3. **Framing, resources and session state.** BAREWire supplies the envelope and
   correlation field, bounded decoding, layout validation and declared buffer
   bounds. Endpoint logic must enforce schema agreement, request/timestep
   matching, length checks, ownership, duplicate/stale-result policy and timeout.
   Timestamps and counters can inform a proposed Prospero supervisor; silence
   still requires a deadline and a decision in the host. Restart policy must
   not allow a result from an earlier session to advance the current simulation.

The existing BAREWire ThreeBody fixture derives a 124-byte layout: three U32
identifiers, three U32 force components and 25 U32 quire words. It checks layout
and BTF offsets and asks cvc5 whether that declared buffer fits a 1500-byte unit.
This is useful bounded evidence, not a numerical codec or a hardware round trip
([fixture](../../../BAREWire/tests/PlatformTests.fs)). The real transport must
account for its envelope and framing overhead; a throughput obligation also needs
the simulation's period and traffic volume.

Keep these checks located and diagnosable: a stable diagnostic identifies the
obligation, source or boundary declaration and failed premise. Linear bounds fit
QF_LIA; finite bit-layout checks can use QF_BV. Decidability does not promise a
small solving cost, and nonlinear numerical bounds need their own justified
enclosures. A timeout, unknown result or missing premise remains unresolved;
it cannot silently become proof. The proposed joint proof graph must retain
both discharged obligations and external assumptions through lowering.

## The second role: the observation plane

The proposed `Telemetry` actor can receive instrumentation compiled from the same
language as the physics. This helps explain the numerical and latency results.

- **Frame arrival, loss counters and link latency** from the receive path —
  observations for the host's deadline and restart decisions.
- **GPU submission latency** (ioctl / fence-wait probes on the DRM path) —
  is the medium-distance regime keeping up?
- **Scheduler latency** across the actor threads (`sched` tracepoints) —
  are the regime actors getting their cores?

Each stream uses integer histograms and counters rather than floating-point
arithmetic, framed as BAREWire records through a ring-buffer map into the
Telemetry actor and rendered in the proposed Wayland panels (`Platform.Display`,
no WebView). Verifier admission protects the probe's permitted operations;
timestamp interpretation, dropped events and the observer's effect on timing
still belong in the experiment's evidence.

## The full picture

```text
CPU orchestrator ── AF_XDP TX / NIC ── L2 request ──▶ FPGA
CPU orchestrator ◀─ AF_XDP RX / XDP ◀─ L2 reply ───── FPGA
       ▲                 │
       └── telemetry ring / probes

Shared boundary contract: format · layout · bounds · session/timestep
Separate evidence: admission · payload preservation · numerical accuracy
```

The intended artifacts are an XDP classifier and a probe set, each admitted on
the pinned kernel. Their evidence supports the shared computation; neither
replaces evidence about the numerical kernel or its transport.

## The eBPF hello progression comes first

Before this synthesis, eBPF earns its place with a standalone progression
modeled on FidelityHelloWorld / HelloArty — each step a compile→load→run check
graded by the external oracle:

| Step | Program | Establishes |
|---|---|---|
| **B-01** | XDP packet counter, per-CPU array map | The pipeline: `DeclRoot.BpfProgram`, LLVM BPF artifact, map, load + attach + run |
| **B-02** | XDP drop-by-blocklist, `.rodata`/array-map config from userspace | Capability and placement gates; bounded map-lookup verdict |
| **B-03** | kprobe latency histogram → ring buffer | Observation-plane streaming to userspace |
| **B-04** | AF_XDP redirect to a userspace ring | Data-plane hand-off, with the selected driver/bind mode recorded |
| **B-05 (synthesis)** | ThreeBody FPGA router + telemetry probes | Composition evidence supporting the rebuilt numerical experiment |

B-01 through B-04 are independent of ThreeBody and validate the target on their
own. B-05 is gated on both that progression and the numerical rebuild.

## Corrections inherited from ThreeBody §13

When B-05 is built, the numerical experiment needs normalized, justified input
ranges; units alone do not establish coverage. The selected b-posit32 boundary
uses the documented `eS = 5` and 800-bit quire layout (25×32-bit words). A finite
quire requires exact product representation and bounds on every reachable
partial sum in evaluation order. Its fixed allocation does not permit unlimited
exact accumulation; final rounding and prior input error remain. The
[normative adequacy contract](../../../clef-lang-spec/spec/numeric-selection.md#1021-the-quire-adequacy-invariant)
governs this independently of the kernel's admission rules.

Use a symplectic, time-reversible integrator and an independent high-precision
reference. Compare energy, angular and linear momentum drift, useful trajectory
horizon and reversal residual against the number of steps, with FP64 and
compensated-summation controls. Reversal means negating momenta and recomputing
with the same forward operator, without replaying saved states. It does not mean
bit-exact recovery. Measure whether posit/quire extends the useful horizon; it
does not change the physical system's Lyapunov exponent. A proposed structural
reversibility type would be representation-agnostic and is not a numerical
accuracy certificate.

## What a successful synthesis would show

HelloArty supplies precedent for carrying width information into a hardware
artifact. The eBPF progression adds an independent kernel admission gate.
ThreeBody then asks the application question: can the composed system preserve
enough numerical information to extend useful prediction and tape-free reversal,
within its measured latency budget? Answer that with the numerical curves and
the declared assumptions alongside the layout, transport and compiler evidence.
