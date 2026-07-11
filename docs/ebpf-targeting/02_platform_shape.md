# Platform Shape: eBPF as a Hosted Verified ISA

**SpeakEZ Technologies | Fidelity Framework**
**July 2026 — exploratory design note**

Where does eBPF live in `Fidelity.Platform`? Answering it well requires naming a
class the taxonomy does not yet have, and confronting a fact eBPF makes
unavoidable: **for this target, the operating system is the device.** This
document proposes the descriptor shape, the class-level constructs eBPF forces
(a versioned capability matrix; composition across platform definitions), and why
the ISA is fixed while the *host* is the axis of variation.

## The taxonomy gap: hosted verified ISAs

`Fidelity.Platform` is substrate-first, and nesting depth is substrate-dependent
by design (`CPU/OS/ISA`, `MCU/Vendor/Family/Board`, `FPGA/Vendor/Family/Board`).
The substrate answers three questions: which pipeline, which backend, which walk
strategy. eBPF answers all three differently from CPU, so it is not a CPU
sub-case:

- **Walk:** a restricted witness profile (no heap, no syscalls, numbered helpers).
- **Pipeline tail:** the artifact is a relocatable ELF object consumed by a
  *loader*, not a linked executable ([03](03_lowering_and_artifacts.md)).
- **Admission gate:** proof obligations no other substrate imposes
  ([01](01_verifier_as_design_time_contract.md)).

eBPF and WASM are the two members of a class the taxonomy should name:
**hosted verified ISAs** — targets where admission is gated by a *checker* rather
than by physics (FPGA), an ABI (CPU), or a fabric (NPU). The class is defined by
what its descriptors must carry:

1. an **ISA level** (BPF v1–v4; WASM MVP + proposals),
2. a **verifier/validator contract** (limits, recognized idioms, acceptance
   profile),
3. a **host-versioned capability surface** (which helpers/hooks/instructions
   exist, as a function of host version),
4. a **license/permission gate** (GPL-only helpers; WASI capability grants).

Proposing `BPF/` as a top-level family (sibling to a future `WASM/`) captures the
class while keeping concrete-ISA naming consistent with `CPU`/`MCU`/`FPGA`.

## The inversion: OS as device

CPU in the platform tree is deliberately descriptor-less — `CPU/Linux/x86_64` has
no device leaf, and its facts flow through the fidproj `[platform]` tuple. eBPF
inverts this. The ISA is *fixed* (the BPF instruction set is stable and
architecture-independent; the JIT maps it to the host CPU after admission), so the
CPU is not the interesting axis. What varies — what determines whether a program
is admissible at all — is **the host and its version**: which hooks exist, which
helpers are callable, which map types are available, what the verifier will
accept. So the natural shape is:

```
BPF/Linux/<series>          e.g. BPF/Linux/6.x
BPF/Windows/<efw-version>   ebpf-for-windows, PREVAIL-gated
```

The leaf is not a board; it is **an OS at a capability level**. This is the sense
in which, for eBPF, the OS *is* the device — the descriptor enumerates the
kernel's programmable surface the way the ArtyA7 descriptor enumerates the FPGA's
pins.

## Hooks are pins

The FPGA descriptor pattern transfers almost directly. The ArtyA7 descriptor is a
set of typed endpoint records (`PinEndpoint`, `ClockEndpoint`, `ResetEndpoint`)
that Composer consumes *structurally* — `PlatformPinResolution` matches on the
record's type name and extracts fields, producing a coeffect that codegen reads.
An eBPF descriptor is the same shape of thing, one conceptual level up:

```
AttachEndpoint  { HookName; ContextType; ReturnConvention; SinceVersion; ... }
HelperEndpoint  { Id; Name; Signature; GplOnly; SinceVersion; ProgramTypes; ... }
MapKind         { Name; KeyConstraint; ValueConstraint; SinceVersion; ... }
VerifierLimits  { StackBytes; InsnBudget; MaxLoopKind; ... }
IsaLevel        { Version; }   // v1..v4
```

- `AttachEndpoint` is the "pin": it binds a logical program role (`xdp`,
  `kprobe/…`, `tracepoint/…`, `cgroup/skb`) to its context struct and return
  convention — the typed window a `BpfProgram` root plugs into
  ([03](03_lowering_and_artifacts.md)).
- `HelperEndpoint` populates the **capability gate**'s candidate set. A program
  calling an operation with no available helper on the pinned host produces a
  witnessed failure carrying the version that would satisfy it — the exact
  `coverage-empty` discipline the numeric-selection spec already defines.
- `VerifierLimits` feeds the stack-budget and complexity-estimate coeffects.

The consumption mechanism already exists and is proven on FPGA:
structural field-name extraction into a coeffect, read by later passes. No new
descriptor-loading machinery is required — only new endpoint record types in
`Contracts/` and a resolver keyed on their type names.

## What eBPF forces: a versioned capability matrix

Here is the genuinely new construct. `Fidelity.Platform` today models capability
by **presence**: a peripheral "exists" iff its endpoint record is in the
descriptor's lists; "not yet available" is an empty list plus a
`plannedEndpointFamilies` note. There is no notion of "available from version N
onward" anywhere in the repository.

eBPF cannot use presence-only, because helper and hook availability is
intrinsically a function of kernel version — `bpf_loop` from 5.17, ring buffers
from 5.8, open-coded iterators from 6.4, and so on. So eBPF forces a
**version-ranged capability** into the schema:

- Each endpoint record gains a `SinceVersion` (and optionally `UntilVersion` /
  deprecation).
- A project pins a **minimum host version** (and optionally a range) in its
  fidproj.
- The capability gate filters the candidate set against the pinned range;
  below-minimum use is a witnessed failure naming the version that would satisfy
  it.

**Design this at the `Contracts/` level, not in the BPF leaf.** Three consumers
want the identical construct:

- **WASM** — SIMD, threads, GC, exception-handling per runtime-and-version.
- **MCU errata** — "peripheral X works from silicon revision Y" is the same
  version-ranged-availability shape (the platform docs already flag revision
  handling as absent).
- **CPU ISA extensions** — AVX-512 / SVE presence per microarchitecture.

The versioned capability matrix is a class-level asset that eBPF happens to force
first because it *cannot function without it*. That makes eBPF the right forcing
function, and the wrong place to scope the solution.

## Composition across platform definitions

A note prompted by a real design worry: a kernel target legitimately draws on
facts that live in more than one platform definition. The host CPU's word size
and endianness (a `CPU/Linux/x86_64` fact) co-determine map layout and BTF
emission alongside the kernel's programmable surface (a `BPF/Linux/6.x` fact). We
do **not** want those CPU facts duplicated into every BPF leaf — that is exactly
the scattered-notation problem to avoid — but we also should not chase a DRY
purity spiral over it.

The platform tree already has the right instrument: `Profiles/` composites, which
`StrixHalo_ArtyLab` uses to bind multiple substrates into one deployment. A
kernel eBPF target is naturally a **profile that composes an OS-capability
descriptor with a host-ISA descriptor**:

```
Profiles/BPF_Linux_x86_64 = compose(
    BPF/Linux/6.x,            // hooks, helpers, verifier limits, ISA level
    CPU/Linux/x86_64          // word size, endianness — referenced, not copied
)
```

The composition *references* the CPU facts (single source of truth) and *layers*
the kernel surface on top. The consuming coeffect is assembled from both
contributions — the same "assembled, not copied" posture the braid design takes
toward per-site proofs. This is the principled answer to "multiple quotations
from multiple platform definitions": composition at the profile layer, reference
not duplication, and no obligation to over-normalize beyond what a real target
actually reads.

## Portability as an intersection profile

Windows eBPF (ebpf-for-windows) is real, and its validator — PREVAIL — is a
published abstract interpreter, arguably a *cleaner* formal contract than the
Linux in-kernel verifier whose acceptance set is its implementation. This makes
Windows a first-class **design forcing function** even though day-to-day
development runs on a Linux kernel (Omarchy): it keeps the architecture honest
that eBPF admissibility is an **OS concern, not a hardware concern**, and prevents
Linux-verifier folklore from ossifying into the design.

Portable programs target the **intersection** of the admissible subsets, and the
tree already has the construct: a `Profiles/BPF_Portable` composite pinning the
common hooks, the shared helper subset, and the tighter of the two verifiers'
limits. A program built against the portable profile is one both gates admit by
construction. Profiles thus do double duty — composing OS-with-host for a single
target, and intersecting OS-with-OS for a portable one.

## fidproj surface (sketch)

```toml
[compilation]
target = "bpf"            # new TargetPlatform case

[build]
output_kind = "bpf"      # NOT "kernel" — that string already means NPU compute
                         # kernels in the loader; a distinct word avoids collision

[platform]
profile = "BPF_Linux_x86_64"   # or "BPF_Portable"
min_host = "6.1"               # pins the capability-gate version floor
program = "xdp"                # selects the AttachEndpoint / context type
license = "GPL"                # feeds the GPL-only helper gate
```

The `output_kind` collision is a real landmine surfaced in the pipeline audit:
the loader already maps the string `"kernel"` to NPU-sense compute kernels, so
the BPF deployment mode must use a different word (`"bpf"` or `"attached"`).

## Open questions for this layer

- **Leaf granularity on Linux.** `BPF/Linux/6.x` versus finer
  `BPF/Linux/6.6-lts` — how much does LTS-vs-mainline capability drift justify?
  Leaning coarse (a series) with `SinceVersion` on endpoints carrying the
  precision, rather than a proliferation of near-identical leaves.
- **Where `VerifierLimits` for Linux come from.** The Linux verifier's limits are
  partly documented, partly empirical. The honest first cut encodes the
  documented floor and treats the CI oracle ([01](01_verifier_as_design_time_contract.md))
  as the source of truth for the rest.
- **Whether the class gets its own top-level doc.** If WASM targeting adopts the
  versioned capability matrix and the validator-contract records, the
  hosted-verified-ISA class may deserve a `Contracts/`-level design note of its
  own that both series reference.
