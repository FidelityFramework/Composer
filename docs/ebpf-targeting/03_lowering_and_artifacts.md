# Lowering and Artifacts: From `.clef` to a Loadable BPF Object

**SpeakEZ Technologies | Fidelity Framework**
**July 2026 — exploratory design note**

This document walks the pipeline from Clef source to a relocatable BPF ELF object
the kernel loader accepts: the program-root model (already built, minus one DU
case), the witness-gating refactor eBPF forces, the LLVM lowering path and its
BTF/maps/license sub-problems, the no-link artifact tail, and the decision gate
for when to stop using LLVM for this target. Nothing here is committed; the point
is to locate the work in the pipeline and size it.

## The program root is already built — BPF is the fourth case

The best structural news from the pipeline audit: the "a declaration that is an
entry the *host environment* calls, with its own extraction and scope boundary"
pattern is already exercised three times. `DeclRoot` is not the two-case DU an
earlier note assumed — it is three:

```fsharp
[<RequireQualifiedAccess>]
type DeclRoot =
    | EntryPoint      // CPU: [<EntryPoint>] / main — the OS calls this
    | HardwareModule  // FPGA: [<HardwareModule>] — this IS the circuit
    | KernelModule    // NPU: [<KernelModule>] — dispatched to AIE tiles
```

An eBPF program section is the fourth member of exactly this family — an entry the
kernel calls when a hook fires:

```fsharp
    | BpfProgram      // eBPF: [<BpfProgram("xdp")>] — the kernel calls this on the hook
```

Everything the FPGA path does for `HardwareModule`, the eBPF path mirrors for
`BpfProgram`:

- **Marking:** an attribute (`[<BpfProgram("xdp")>]`, paralleling
  `[<HardwareModule>]`) tags the root; the attach role string selects the
  `AttachEndpoint` from the descriptor ([02](02_platform_shape.md)).
- **Typed signature:** the program is `XdpMd -> XdpAction` — a typed root exactly
  as the Mealy machine is `Design<'S,'R>`. The context struct and return
  convention come from the `AttachEndpoint`, so the typed window is
  descriptor-driven, not hardcoded.
- **A `BpfProgramWitness`** mirroring `HardwareModuleWitness`: scope boundary,
  children owned, signature extracted. The witness observes; it does not compute.
- **The MiddleEnd op-filter gains a BPF arm.** The exit filter already drops
  `func.func` for FPGA and keeps only the `aie.device` block for NPU; the BPF arm
  keeps program-rooted functions plus map globals and drops CPU entry points.
  This is precedent-for-precedent, not new architecture.

## Maps are the typed boundary — and they are BAREWire-shaped

A BPF program is short-lived: it runs on an event, returns a verdict, and its
stack vanishes. Persistent state lives in **maps** — key/value stores shared
between the kernel program and userspace. This is the same producer/consumer,
shared-memory boundary BAREWire exists to serialize, with one property the type
system supplies for free: **map values are pointer-free records by construction**,
which is precisely the "no kernel pointer leaks to userspace" obligation
([01](01_verifier_as_design_time_contract.md)) discharged at the type level rather
than by the verifier.

The record-layout coeffects Composer already computes for codegen are the same
information the kernel needs as **BTF** — which is the crux of the lowering story
below.

Map *kinds* (hash, array, per-CPU array, ring buffer, LRU…) come from the
descriptor's `MapKind` records; a per-CPU array for a lock-free hot-path counter
and a ring buffer for streaming events to userspace are the two the observation
plane leans on ([05](05_threebody_integration.md)).

## The witness-gating refactor eBPF forces

The pipeline audit surfaced the one genuinely invasive change. Witness
registration is gated by a binary predicate:

```fsharp
let isCPULike = targetPlatform <> FPGA && targetPlatform <> NPU
```

A new `BPF` platform would be *silently* classified `isCPULike = true` and handed
the full CPU witness set — including the `PlatformWitness`, whose libc /
`dlopen` / arbitrary-syscall model is categorically wrong for a target that calls
numbered helpers from a fixed table and has no libc at all.

But BPF's profile is genuinely *mixed*, which is why a third boolean is the wrong
fix:

| Witness class | CPU | FPGA | NPU | **BPF** |
|---|---|---|---|---|
| Arithmetic, DU, Record, Structural | ✓ | ✓ | ✓ | ✓ |
| Mutable assignment | ✓ | — | — | ✓ (stack + maps) |
| Memory / heap collections (List, Map, Set) | ✓ | — | — | **maps only, no heap** |
| Syscall / extern / dlopen (`PlatformWitness`) | ✓ | — | — | **✗ — helpers only** |
| Lazy, Seq | ✓ | — | — | ✗ |

The right fix — and the code comments already gesture at a "Three-Category Model"
— is to **key witness registration on declared platform capabilities** (has-heap,
has-syscalls, has-fp, has-unbounded-loops, has-helper-table) drawn from the
descriptor, rather than on platform identity. This is the same inversion the
numeric-selection spec made for representations: capability *sets* filter
candidates; identity match-statements do not scale past three platforms. eBPF is
the forcing function; WASM collects the payoff. Because this refactors standing
machinery rather than adding to it, it is the item to sequence and review most
carefully.

## The LLVM path (the MVP), and its three real sub-problems

The pipeline audit produced a touchpoint map; most of it is plumbing forced to
completeness by exhaustive DU matches:

- Add `BPF` to the Composer and clef `TargetPlatform` DUs; the compiler flags
  every non-exhaustive match, which *is* the change list.
- Route `BPF -> BackEnd.LLVM` in `resolveBackEnd` (LLVM has a first-class BPF
  target).
- Add a platform→triple resolver. Today the triple is hardcoded x86_64 with only
  a CLI override; a `bpfel-unknown-none` (little-endian) / `bpfeb` entry needs an
  actual table — which the canonical platform spec wants regardless.
- Parse `target = "bpf"` and the BPF `output_kind` in the fidproj loader (both
  currently hard-error on unknown strings).

The MLIR→LLVM lowering itself is largely target-neutral and reusable. Three
sub-problems inside the LLVM path are real work, not plumbing:

### 1. BTF emission — emit it from Clef types, not via DWARF

Modern kernels want BTF: function info for bpf2bpf calls, map type descriptions,
and CO-RE relocation anchors. The clang route generates BTF from debug metadata —
a path Composer's MLIR pipeline does not feed. But BTF is *a type serialization
format*, and Fidelity owns complete type information at the PSG level, richer than
what survives to LLVM metadata. The recommendation is to **emit BTF directly from
Clef types** as a section writer over the record-layout coeffects already
computed — avoiding a lossy DWARF round-trip and reusing the BAREWire-adjacent
schema discipline. This is also the cleanest line in the eventual write-up: the
kernel already speaks types; BTF is its schema language, and Clef can speak it
natively. CO-RE relocation (the mechanism that makes one object load across kernel
layouts) layers on top and pairs with the versioned capability matrix
([02](02_platform_shape.md)).

### 2. Maps and global data as sections

BTF-defined maps are globals in the `.maps` section; `.rodata`/`.bss` global data
become array maps under the hood. The recent escape-analysis static-storage work
(program-lifetime values placed via `memref.global`) composes directly here — the
same placement decision, a different ELF section. Config pushed down from
userspace (a blocklist the XDP program consults per packet) is a `.rodata` /
array-map read, exactly the numeric-selection static-placement art in a new home.

### 3. The license gate

A BPF object declares a license, and GPL-only helpers are gated on it. Mechanically
this is one more capability input to the same gate ([04](04_admissibility_as_proof_obligations.md)):
the `HelperEndpoint.GplOnly` flag crossed with the fidproj `license` field. It is
also a *business* seam worth flagging explicitly — the source-available posture of
the FidelityFramework repos intersects with "your probe must declare `GPL` to call
`bpf_probe_read` and the tracing helpers." Not a compiler problem, but a decision
the compiler surfaces.

## The artifact tail: no link step

This is where eBPF departs hardest from the existing LLVM backend. The current
backend runs `llc` then `clang` to link an executable — three `DeploymentMode`
branches (console `-lc`, freestanding `-nostdlib -Wl,-e,_start`, shared
`-shared`). **None of them fit.** eBPF wants:

```
llc -march=bpf -filetype=obj  →  program.o        (the FINAL artifact)
```

A relocatable ELF `.o` loaded by libbpf / the `bpf()` syscall. No `clang` link, no
`_start`, no `-lc`, no `-static`. So the artifact tail is a **new
`DeploymentMode`** (e.g. `BpfObject`) that stops after `llc` and emits the `.o` as
the `BackEndArtifact` — plus the BTF/maps sections from the sub-problems above.
The loader/skeleton (libbpf, or a generated skeleton header) is a separate
userspace concern, matching how the FPGA path emits a bitstream whose *loading*
is out of band.

## Project shape and the single-source future

MVP project shape mirrors HelloArty exactly: a standalone fidproj
(`target = "bpf"`) producing the `.o`, with a **peer CPU-target project** as the
loader/consumer, the two sharing a map-schema module so the kernel and userspace
sides agree on layout by construction (one BAREWire schema, both sides).

Single-project **dual-artifact** emission — one source producing both the
userspace binary and the BPF `.o` from a shared map schema — is a genuinely
attractive future feature and genuinely new machinery (two backends, two op
filters, one PSG). Do not gate the MVP on it; note it as the natural maturation
step once the split-project pipeline is proven.

## The decision gate: LLVM, until it lies

The standing framework stance is LLVM for commodity cores, clean direct emission
for novel substrates. eBPF sits on the line, and the case for **direct emission**
is unusually strong: the ISA is tiny (~10 registers, ~100 instructions), there is
no register-allocation pressure worth the name, and **shape control is the entire
game** — the clang-vs-verifier fight is precisely an optimizer destroying
legibility, which owning emission ends permanently. It would also be an ideal,
low-surface-area rehearsal for the hand-rolled RISC-V backend thesis (encode leaf
+ semantic contract), at a fraction of the instruction count.

The case for **LLVM-first** is pragmatic: the backend is mature, `-mcpu=v1..v4` is
handled, and — decisively — the admissibility coeffect layer
([04](04_admissibility_as_proof_obligations.md)) is backend-independent, so
nothing built for the LLVM path is wasted if emission moves in-house later.

**Recommendation: LLVM first, with direct emission as an explicit, documented
tripwire** — triggered the first time the SMT-dialect re-discharge pass catches
LLVM transforming a certified guard out of verifier legibility. That event is the
concrete signal that LLVM's optimizer and the verifier's idiom set have diverged
beyond what emission-vocabulary curation can bridge, and it converts the
LLVM-vs-direct question from a philosophical one into a triggered engineering
decision. Not dogma either way — a tripwire.

## Touchpoint summary (work order)

| Concern | Nature | Notes |
|---|---|---|
| `TargetPlatform` DU (Composer + clef) | plumbing | exhaustive matches enforce the change list |
| `resolveBackEnd` BPF arm → LLVM | plumbing | single dispatch seam |
| clef→Composer bridge (2 matches) | plumbing | |
| fidproj `target` / `output_kind` parse | plumbing | avoid the `"kernel"` string collision |
| platform→triple resolver | small new | `bpfel`/`bpfeb`; wanted anyway |
| MiddleEnd op-filter BPF arm | small new | precedented by FPGA/NPU arms |
| `DeclRoot.BpfProgram` + `BpfProgramWitness` | moderate | mirrors `HardwareModule` |
| Artifact tail (`llc -march=bpf`, no link) | moderate new | new `DeploymentMode` |
| BTF-from-Clef-types section writer | **real work** | over existing layout coeffects |
| Maps/globals as sections | moderate | composes with escape static-placement |
| License gate | small | capability input + business seam |
| **Capability-keyed witness gating** | **invasive refactor** | retires binary `isCPULike`; the item to review hardest |
