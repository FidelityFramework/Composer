# Platform shape: instruction machine, host and workload

Updated 2026-09-10. The former `BPF/Linux/6.x` sketch is superseded by the
implemented [Fidelity.Platform taxonomy](../../../Fidelity.Platform/PLATFORM_STRUCTURE.md)
and [admission reference model](../../../Fidelity.Platform/docs/ADMISSION_AND_SIDECARS.md).

## Source owners

- `Contracts/Admission.clef`: common host facts, budget scopes, report evidence
  and executable consistency checks.
- `Contracts/Sidecar.clef`: BAREWire-backed producer/consumer handoffs.
- `AbstractMachines/eBPF/` and `AbstractMachines/cBPF/`: instruction widths.
- `Environments/Linux/eBPF/`: Linux admission and XDP reference subset.
- `Environments/Windows/eBPF/`: Windows-specific verification, native artifact
  and XDP extension requirements.
- `Environments/macOS/BPF/`: classic packet capture/filter/transmit contract.
- `Environments/Linux/x86_64/ROCm/` and `Environments/macOS/Metal/`:
  environment-specific CPU/GPU allocation and visibility obligations.
- `Profiles/`: independent admission references, Strix Halo/Arty ThreeBody
  handoffs and the Apple Silicon comparison.

No `SubstrateKind.BPF` hardware category is required. An instruction machine
does not declare a native host ABI; an OS admission surface is not a physical
product. Host architecture still matters for loader structures, mappings,
endianness and native conversion.

## Reuse BAREWire

BAREWire already declares `BoundarySurface`, `Endpoint`, `MemorySpace`,
`BufferSchema`, `Transport`, `Limit` and version availability fields. The
earlier proposal to introduce a second versioned endpoint vocabulary here was
superseded by that implementation.

Availability by version is only a filter. Exact kernel build/configuration,
backports, privilege, license, program type, installed extensions and the loader
toolchain must narrow the usable surface. A minimum version alone is insufficient.

The reference subsets intentionally expose no map/helper inventory. Extending
them requires actual signatures, context and invalidation effects, map schema
constraints and target evidence. Native `bpf()` syscall numbers and Windows
structures must come from architecture-specific bindings.

## Selection and artifacts

Current `BPF_Reference` values remain unresolved: no exact host facts, no profile
digest and no assumed numeric budgets. The library manifest exposes reference
data and does not select a deployable `[platform] description`.

The existing Strix Halo/Arty package remains a physical component catalogue with
handoff reference data. It does not resolve several compilation targets into one
executable or automatically merge memory domains. CPU/GPU views must not count
the shared RAM twice. Arty USB bring-up and the intended Ethernet workload path
have separate roles.

A future portable source subset would select the intersection of supported
operations, while compiling and checking each host's ABI, artifact and gate
separately. There is no implemented portable profile or single cross-OS binary.

## Remaining compiler integration

The `target = "bpf"`, `output_kind = "bpf"` and profile-selection examples from
the older sketch were proposals, not supported fidproj fields. This structure
does not add them. Existing `output_kind = "kernel"` has another compiler meaning
and must not be reused accidentally.

Implementation still needs a typed program-root contract, capability-driven
lowering, source obligation extraction, final-artifact checks, Linux/Windows
artifact packaging and Composer-directed verifier/load/attach orchestration.
The platform reference model provides a testable boundary for that work; it
does not make the backend plumbing disappear.
