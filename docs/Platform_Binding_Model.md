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

CCS sees the predicates as abstract. Alex witnesses them to eliminate dead branches at compile time.

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

DMM escape classifications (StackScoped, ClosureCapture, ReturnEscape, ByRefEscape) map to these regions via the `NTUMemorySpace` qualifier attached to PSG nodes during coeffect analysis.

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

```
┌─────────────────────────────────────────────────────────┐
│  Composer CLI                                            │
│  1. Parse fidproj with Fidelity.Toml                    │
│  2. Load Fidelity.Platform binding(s)                   │
│  3. Extract quotations (platform, capabilities, etc.)   │
│  4. Pass to CCS as PlatformContext                      │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  CCS                                                    │
│  1. Receive PlatformContext                             │
│  2. Attach platform metadata to PSG nodes               │
│  3. Validate NTU type identity (not width)              │
│  4. Resolve DTS dimensions, DMM escape classifications  │
│  5. Return PSG with quotations attached                 │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Alex                                                    │
│  1. Read platform quotations from PSG                   │
│  2. Witness NTU types → concrete MLIR types             │
│  3. Eliminate dead branches via predicates              │
│  4. Select numeric format per DTS dimensional domain    │
│  5. Generate substrate-optimized MLIR                   │
└─────────────────────────────────────────────────────────┘
```

## NTU Type Resolution by Substrate

| NTU Type | CPU/Linux/X86_64 | CPU/Linux/ARM32 | FPGA/Xilinx/Artix7 |
|----------|------------------|-----------------|---------------------|
| NTUint | i64 | i32 | i32 (softcore) |
| NTUuint | i64 | i32 | i32 |
| NTUptr<'T> | ptr (8B) | ptr (4B) | ptr (4B) |
| NTUsize | u64 | u32 | u32 |
| NTUfloat64 | f64 | f64 | posit32 (if DTS selects) |
| NTUposit(32,2) | emulated | emulated | hardened IP |

FPGA bindings may map `NTUfloat64` to `posit32` when the DTS dimensional domain and target capabilities both confirm the substitution is safe. This is the representation selection rule described in `CCS_Architecture.md`.

## Adding New Substrates

1. **Create substrate directory following the hierarchy:**
   ```
   CPU/Linux/X86_64/NewSoC/
   FPGA/Xilinx/Artix7/NewBoard/
   MCU/Nordic/nRF52840/NewModule/
   ```

2. **Create the standard module set:** Types.fs, Platform.fs, Capabilities.fs, MemoryRegions.fs

3. **For CPU targets:** add Syscalls.fs and CacheCharacteristics.fs

4. **For FPGA targets:** add posit resolution rules to Types.fs

5. **Create .fsproj and reference in fidproj**

6. **For multi-substrate systems:** create a Profile that composes the relevant bindings

## Related Documentation

- `NTU_Architecture.md` - NTU type system design and DTS integration
- `CCS_Architecture.md` - DTS/DMM coeffect analysis
- `Architecture_Canonical.md` - Composer pipeline overview
