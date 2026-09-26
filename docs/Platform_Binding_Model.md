# Platform Binding Model

## Overview

The Fidelity platform binding model provides substrate-aware type resolution and MLIR generation across the full hardware spectrum. Platform bindings are organized substrate-first, reflecting the reality that CPU, FPGA, GPU, NPU, and MCU targets have fundamentally different memory models, syscall conventions, and numeric formats.

## Repository Structure

```
~/repos/Fidelity.Platform/
├── PLATFORM_STRUCTURE.md
├── Profiles/                       # Multi-substrate system profiles
│   └── StrixHalo_ArtyLab/          # CPU+FPGA+GPU+NPU profile
├── CPU/
│   └── Linux/
│       └── X86_64/
│           └── StrixHalo/          # Zen5-specific bindings
│               ├── Types.fs
│               ├── Platform.fs
│               ├── Capabilities.fs
│               ├── MemoryRegions.fs
│               ├── CacheCharacteristics.fs
│               ├── Syscalls.fs
│               └── Fidelity.Platform.CPU.Linux.X86_64.StrixHalo.fsproj
├── FPGA/
│   └── Xilinx/
│       └── Artix7/
│           └── ArtyA7_100T/        # Artix-7 100T binding
│               ├── Types.fs        # NTUposit resolution
│               ├── Platform.fs
│               ├── Capabilities.fs
│               └── ...
├── GPU/
│   └── AMD/
│       └── RDNA3_5/
│           └── StrixHalo_iGPU/
├── NPU/
│   └── AMD/
│       └── XDNA2/
│           └── StrixHalo_NPU/
├── MCU/
│   └── ST/
│       └── STM32F7/
│           └── MeadowF7/
└── CGRA/
    └── ...
```

The substrate-first layout makes cross-substrate profiling natural: a `Profiles/StrixHalo_ArtyLab` profile composes CPU + FPGA + GPU + NPU bindings for a single physical system.

## Platform Descriptor

> **Membrane note.** The `nativeint` and `nativeptr` appearances in this document are membrane plumbing recorded point-in-time, internal `TNativePtr` surface governed by the exit in `Closure_Nanopass_Architecture.md` Section 4 ("Why Flat: the Finiteness Lemma") and the boundary contract of `PRDs/C-01-Closures.md` Section 6.7.

The core platform quotation (CPU/Linux/X86_64 example):

```fsharp
// Platform.fs
let platform: Expr<PlatformDescriptor> = <@
    { Architecture = X86_64
      OperatingSystem = Linux
      WordSize = 64
      Endianness = LittleEndian
      TypeLayouts = Map.ofList [
          "int", { Size = 8; Alignment = 8 }
          "int32", { Size = 4; Alignment = 4 }
          "nativeint", { Size = 8; Alignment = 8 }
          "nativeptr", { Size = 8; Alignment = 8 }
      ]
      SyscallConvention = sysV_AMD64_syscall }
@>
```

## Platform Predicates (F*-Inspired)

Abstract propositions for conditional compilation:

```fsharp
// Capabilities.fs
module Capabilities =
    let fits_u64: Expr<bool> = <@ true @>
    let has_avx2: Expr<bool> = <@ true @>
    let has_avx512: Expr<bool> = <@ false @>  // CPU-dependent
    let has_posit_hw: Expr<bool> = <@ false @>  // FPGA only
    let vector_width_max: Expr<int> = <@ 256 @>  // AVX2
```

### Using Predicates for Conditional Compilation

```fsharp
// In Clef application code
let vectorAdd (a: array<float>) (b: array<float>) =
    if Platform.has_avx512 then
        vectorAdd_avx512 a b
    elif Platform.has_avx2 then
        vectorAdd_avx2 a b
    else
        vectorAdd_scalar a b
```

CCS/Baker resolves declared predicates, settles the selected branch and retains
the declaration and selection premises in the PSG. Alex passively witnesses the
published structure; it does not evaluate platform predicates or infer dead code.

## Memory Regions

For DMM (Deterministic Memory Management) integration:

```fsharp
// MemoryRegions.fs
module MemoryRegions =
    let stackRegion: Expr<MemoryRegion> = <@
        { Name = "Stack"
          MaxSize = 8388608      // 8 MB typical
          Alignment = 16
          GrowthDirection = Down
          ThreadLocal = true }
    @>

    let arenaRegion: Expr<MemoryRegion> = <@
        { Name = "Arena"
          Strategy = BumpAllocator
          DefaultSize = 1048576  // 1 MB
          Alignment = 16 }
    @>
```

CCS/Baker settles escape, covering lifetime, placement, access and capacity under
the selected storage declarations. These are distinct premises; an escape cause
alone neither selects an allocation nor supplies a missing lifetime. Alex reads
the published storage facts and actual allocation identities.

## Cache Characteristics

```fsharp
// CacheCharacteristics.fs
module CacheInfo =
    let l1_line_size: Expr<int> = <@ 64 @>
    let l1_size: Expr<int> = <@ 32768 @>  // 32 KB
    let l2_size: Expr<int> = <@ 262144 @>  // 256 KB
    let l3_size: Expr<int> = <@ 8388608 @>  // 8 MB
    let prefetch_distance: Expr<int> = <@ 256 @>
```

Lattice surfaces cache locality estimates for hot loops based on DMM allocation size + these characteristics.

## Syscall Conventions (CPU targets)

```fsharp
// Syscalls.fs
module Syscalls =
    let convention: Expr<SyscallConvention> = <@
        { CallingConvention = SysV_AMD64
          ArgRegisters = [| RDI; RSI; RDX; R10; R8; R9 |]
          ReturnRegister = RAX
          ErrorReturn = NegativeErrno
          SyscallInstruction = Syscall }
    @>

    let sys_write: Expr<int> = <@ 1 @>
    let sys_read: Expr<int> = <@ 0 @>
    let sys_nanosleep: Expr<int> = <@ 35 @>
    let sys_exit_group: Expr<int> = <@ 231 @>
```

## Integration with fidproj

Projects reference the substrate-appropriate binding:

```toml
# HelloWorld.fidproj (CPU/Linux/X86_64)
[package]
name = "HelloWorld"

[dependencies]
platform = { path = "/home/hhh/repos/Fidelity.Platform/CPU/Linux/X86_64/StrixHalo" }

[build]
sources = ["HelloWorld.fs"]
output = "helloworld"
output_kind = "freestanding"
```

For multi-substrate systems, reference a profile:

```toml
[dependencies]
platform = { path = "/home/hhh/repos/Fidelity.Platform/Profiles/StrixHalo_ArtyLab" }
```

## Pipeline Integration

| Owner | Responsibility |
|---|---|
| Composer CLI | Supply the project and selected declaration inputs to CCS |
| CCS/Baker | Read and validate declarations; infer source types/dimensions; settle predicates, ranges, numeric representations, layout, storage and declaration/ABI facts through nanopass ingredients and recipes |
| Source publication | Retain immutable settled codata, exact declaration and occurrence identities, ordered joint proof premises and the intermediate rewrite record |
| Alex | Read the published facts through passive Huet Element/Pattern/Witness composition; emit admitted portable physical forms |
| Composer backend | Realize the selected target operations and ABI, preserving the published contracts in the actual artifact |

Missing platform or semantic facts fail at their CCS/Baker owner. Alex does not
analyze declarations, choose numeric formats, settle layouts, query hyperedges
for new facts or eliminate source branches. All custom MLIR plugins and their
compatibility dependencies are retired.

## Native Representation Settlement

Clef's integer and real kinds retain their native dimensional identity. CCS/Baker
selects representations from justified ranges and the actual offered platform
set under the numeric-selection contract. Boundary declarations retain their
coverage and transfer obligations. Neither an architecture name nor a host word
size supplies a default integer width or permission to substitute numeric forms.

Alex reads settled representation and adaptation facts. The backend realizes
the selected operations and checks their actual modes, layout and correspondence.

## Adding New Substrates

1. **Create substrate directory following the hierarchy:**
   ```
   CPU/Linux/X86_64/NewSoC/
   FPGA/Xilinx/Artix7/NewBoard/
   MCU/Nordic/nRF52840/NewModule/
   ```

2. **Create the standard module set:** Types.fs, Platform.fs, Capabilities.fs, MemoryRegions.fs

3. **For CPU targets:** add Syscalls.fs and CacheCharacteristics.fs

4. **For each target:** declare offered representations, capabilities and operation modes for CCS/Baker settlement; target-specific realization belongs to the backend

5. **Create .fsproj and reference in fidproj**

6. **For multi-substrate systems:** create a Profile that composes the relevant bindings

## Related Documentation

- `NTU_Architecture.md` - NTU type system design and DTS integration
- `CCS_Architecture.md` - DTS/DMM coeffect analysis
- `Architecture_Canonical.md` - Composer pipeline overview
