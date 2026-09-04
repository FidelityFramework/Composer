# MLIR Dialect Strategy: Standard Dialects Only

> **Path**: Standard MLIR dialects (scf, func, arith, memref, index) witnessed from the saturated PSG.
>
> **Supersession note.** This document was written as "demo path vs. full vision", where the full vision was a pair of custom DCont/Inet dialects selected by purity analysis. That vision is retired: delimited continuations and interaction-net structure are settled on the PSG as hypergraph structure and witnessed as standard dialects (Thin_Middle_End_Design.md; Delimited_Continuations_Architecture.md; clef-lang-spec `dcont-representation.md`). The "demo path" below is therefore the design, not a stopgap. The sections that costed the custom dialects have been removed; the decision record is kept as history.

---

## Executive Summary

The Fidelity framework's compilation model distinguishes between delimited continuations (DCont) for sequential effects and interaction nets (Inet) for pure parallelism. Both are hypergraph structure on the PSG: the continuation as a saturated aggregate (segments, frame, delimiter edge), the net as hyperedge structure over enumerated source sets. Standard MLIR dialects are the witnessed form of both; no custom dialect is built.

This document captures the dialect strategy for the demo, which is the standing strategy.

---

## The DCont/Inet Duality (Conceptual Model)

The Composer compiler analyzes referential transparency to determine compilation strategy:

| Pattern | Compilation Target | Characteristics |
|---------|-------------------|-----------------|
| **Pure operations** | net structure on the PSG → data flow (`scf.parallel`) | No effects, no dependencies, parallel by construction |
| **Effectful operations** | suspension recipe on the PSG → `scf.index_switch` state machine | I/O, state, sequential with cuts |
| **Hybrid** | Both, split at the effect boundary by saturation | Net structure wraps suspension at effect boundaries |

For quad-channel ADC sampling:

```
Inet (parallel across 4 cores)
  └── DCont (sequential within each core)
        └── shift at each ADC read (natural preemption point)
```

This model is the design. It is realized as graph structure at saturation and witnessed as standard dialects; there is no alternative implementation approach to choose between.

---

## Custom Dialects (Retired)

An earlier revision of this document estimated the infrastructure for custom DCont/Inet dialects — TableGen definitions, C++ implementations, lowering passes, purity-driven selection. That work is not planned. The duality it was meant to carry lives in the PSG as hypergraph structure and is witnessed as standard dialects; a dialect above the witness boundary would re-introduce the middle-end decisions the thin-middle-end design removes.

## Standard Dialect Approach (Demo Path)

For the demo, standard MLIR dialects provide equivalent runtime behavior:

### Dialect Stack

```
func + scf + arith + memref
        ↓
      llvm
        ↓
   LLVM IR → native
```

### Dialect Roles

| Dialect | Role in Demo | Standard? |
|---------|--------------|-----------|
| **func** | Function definitions, syscall wrappers | Yes |
| **scf** | `scf.parallel` for multi-channel sampling | Yes |
| **arith** | Bit manipulation, interleaving | Yes |
| **memref** | Buffer management, ADC sample storage | Yes |
| **llvm** | Final lowering to LLVM IR | Yes |

All dialects have mature implementations, comprehensive documentation, and battle-tested lowering paths.

### Parallel Sampling via scf.parallel

```mlir
// Quad-channel parallel ADC sampling
func.func @sampleAllChannels(%buffer: memref<4xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index

  // scf.parallel provides Inet-like "run these simultaneously"
  scf.parallel (%ch) = (%c0) to (%c4) step (%c1) {
    %sample = func.call @readAdcChannel(%ch) : (index) -> i32
    memref.store %sample, %buffer[%ch] : memref<4xi32>
    scf.yield
  }

  return
}

// ADC read via Platform.Bindings (syscall)
func.func private @readAdcChannel(index) -> i32
```

### Continuation Points (Implicit)

The DCont-like suspension points come "for free" from the OS:

1. `@readAdcChannel` calls into Platform.Bindings
2. Platform.Bindings emits a syscall (sysfs read or ioctl)
3. The syscall traps to kernel mode
4. The kernel handles I/O, scheduler can preempt
5. When I/O completes, execution resumes

No explicit continuation capture needed. Linux provides preemptible I/O at the syscall boundary.

### Lowering Path

```
scf.parallel
    ↓ (scf-to-openmp or scf-parallel-loop-tiling)
omp.parallel / omp.wsloop
    ↓ (convert-openmp-to-llvm)
llvm.call @__kmpc_fork_call (OpenMP runtime)
    ↓ (llvm translation)
LLVM IR with pthread/OpenMP calls
    ↓ (llc)
Native ARM64 binary
```

Alternative path without OpenMP:

```
scf.parallel
    ↓ (scf-to-cf)
cf.br / cf.cond_br (unrolled or serialized)
    ↓ (convert-cf-to-llvm)
llvm dialect
    ↓
LLVM IR
```

For true parallelism, the OpenMP path is preferred.

---

## What the Demo Validates

Using standard dialects, the demo still validates:

| Concept | How Validated |
|---------|---------------|
| **Parallel entropy sampling** | scf.parallel executes on 4 cores |
| **Natural suspension at I/O** | Syscalls yield to OS scheduler |
| **Interleaved entropy** | arith operations combine channels |
| **Platform.Bindings pattern** | func.call to Alex-emitted syscalls |
| **Quotation-based constraints** | clef nanopasses attach metadata |

What the demo did not exercise, and where the standing design settles each:

| Concept | Settled by |
|---------|-----------|
| **Automatic purity analysis** | the escape/effect class read at fold-in; no dialect selection exists |
| **Zero-allocation continuations** | the frame placed by the lifetime lattice at saturation (Delimited_Continuations_Architecture.md §3) |
| **Formal parallelism verification** | hyperedge obligations over enumerated source sets, discharged at saturation |
| **Interaction net reduction** | hyperedge rule structure on the PSG; annihilation as a monotone fold-in consequence, never node deletion (waypoint in `clef/docs/fidelity/phg/Design_Supersession_Register.md`) |

---

## Progression Path

### Phase 1: Demo (January)

- Standard MLIR dialects only
- Manual "this is parallel" decisions in Alex
- scf.parallel for multi-channel sampling
- Syscall-based I/O with OS-provided preemption

### Phase 2: Purity Analysis (Post-Demo)

- Extend Alex with referential transparency detection
- Annotate PSG nodes with purity information
- Generate scf.parallel automatically for pure regions
- Still using standard dialects

### Phases 3–5: Retired

The remaining phases of the earlier progression (a DCont dialect, an Inet dialect, unified dialect selection) are superseded. Their objectives are met on the PSG instead: zero-allocation continuation capture is the frame placed by the lifetime lattice; formal reduction semantics are the hyperedge rules discharged at saturation; purity-driven selection is the escape/effect class read at fold-in. See Delimited_Continuations_Architecture.md and PSG_Nanopass_Architecture.md.

---

## File References

### Demo Implementation

| File | Purpose |
|------|---------|
| `Alex/CodeGeneration/MLIRBuilder.fs` | MLIR emission infrastructure |
| `Alex/Bindings/Linux/` | Platform-specific syscall emission |
| `Fidelity.Platform` | Platform.Bindings signatures |

### Architecture Documents

| Document | Relevance |
|----------|-----------|
| [02_YoshiPi_Architecture.md](./02_YoshiPi_Architecture.md) | Quad-channel hardware design |
| [05_PostQuantum_Architecture.md](./05_PostQuantum_Architecture.md) | Parallel entropy pipeline |
| [README.md](./README.md) | DCont/Inet duality overview |

### SpeakEZ Articles

| Article | Relevance |
|---------|-----------|
| Seeking Referential Transparency | Purity analysis and dialect selection |
| The DCont/Inet Duality | Computation expression decomposition |
| Delimited Continuations: Fidelity's Turning Point | Continuation preservation |

---

## Decision Record

**Decision**: Use standard MLIR dialects for the QuantumCredential demo.

**Rationale**:
1. Custom dialects require 14-23 weeks of infrastructure work
2. Standard dialects provide equivalent runtime behavior for demo scenarios
3. Demo timeline requires pragmatic risk management
4. Conceptual validation does not require formal dialect semantics

**Consequences**:
- Parallel execution validated via scf.parallel
- Continuation points implicit at syscall boundaries
- No formal purity-driven dialect selection
- Architecture documents describe vision; demo validates concept

**Status**: Not revisited. The conditions the earlier revision listed for revisiting (formal verification, zero-allocation capture, automatic purity analysis, heterogeneous Inet targets) are met by hypergraph structure on the PSG, not by custom dialects.
