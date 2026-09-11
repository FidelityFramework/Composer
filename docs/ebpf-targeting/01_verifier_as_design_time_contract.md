# The verifier as a design-time contract

Updated 2026-09-10. This is the compiler integration design; the implemented
[platform reference model](../../../Fidelity.Platform/docs/ADMISSION_AND_SIDECARS.md)
has narrower, tested responsibilities.

The objective is early admission feedback for a bounded program subset and an
exact host profile. Source safety, recognizable final bytecode and the target's
analysis budget are separate obligations. A successful source proof does not
automatically establish all three.

## Obligations and evidence

| Concern | Design-time analysis | Final boundary |
| --- | --- | --- |
| Control flow | Establish supported loop/path bounds and call discipline | Target-specific CFG and verifier recognition |
| Initialization and ranges | Track initialized values and scalar bounds | Register allocation, spills and bytecode dataflow |
| Memory | Provenance, nullable references, bounds, alignment and lifetime | Actual guards, helper effects and context access rules |
| Calls and maps | Selected program type, helper/map inventory, permissions and layout | Host implementation, config, privilege and license policy |
| Stack | Every reachable call chain, backend spills and special call restrictions | Pinned verifier accounting rules |
| Analysis work | Estimate path/state exploration, initially advisory | Work actually performed by the target verifier |
| Execution cost | Bound program paths; retain helper cost assumptions | JIT/native code, host scheduling and measured conditions |

The often-quoted Linux 512-byte stack ceiling is not a universal BPF constant or
a sufficient call-chain analysis. Tail-call/subprogram combinations and other
host rules must be pinned. Verifier instruction visits, bytecode instruction
slots and executed instructions are different quantities. None by itself proves
a time bound.

Linux documents register state, provenance, context-specific operations and
range tracking in its [verifier reference](https://docs.kernel.org/bpf/verifier.html).
Exact acceptance remains a property of the selected implementation; parts of
that overview describe historical restrictions. The ISA specification does not
enumerate every OS verifier rule.

## Deferred inference and predicates

Existing typed Clef predicates express source requirements. The proposed
compiler consumer derives source-located obligations and keeps unresolved
premises pending while editing. Concrete deployment requires discharge or an
explicitly permitted external premise. Missing information, solver timeout and
unknown results never become successful proofs.

Lowering must preserve those properties and their recognizable bytecode forms.
The intended checks run again after transformations that affect admission,
including register allocation, relocations and Windows native conversion.
Evidence must bind to the final artifact and resolved environment, not merely
to the source text or a minimum kernel version.

The new reference checker validates supplied report identities, budget units
and scopes, evidence standing and subset membership. It does not derive
obligations from a PSG, inspect bytecode, authenticate proofs or call a verifier.
Its evidence strings are references to results a future adapter must check.

## Independent admission gates

Linux kernel admission and Windows verification/loading require distinct
adapters. A standalone PREVAIL result does not establish that a Windows driver
loads or attaches. macOS classic BPF has its own instruction machine and filter
validator; it is not a third eBPF host.

An eventual agreement corpus should include admitted programs, deliberate
rejections, unknown premises, boundary limits and transformations that invalidate
earlier evidence. Record exact artifact/profile digests and gate diagnostics.
Unexpected rejection within the supported subset is a compiler/contract
investigation, with the target gate retained.

Admission does not prove application policy, payload preservation, network
delivery or ThreeBody numerical accuracy. Those obligations share source and
layout facts but require their own evidence.
