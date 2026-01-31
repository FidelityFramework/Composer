# Firefly: F# Native Compiler

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-orange.svg)](Commercial.md)
[![Pipeline](https://img.shields.io/badge/Pipeline-25%20nanopasses-blue)]()

<p align="center">
🚧 <strong>Under Active Development</strong> 🚧<br>
<em>Early development. Not production-ready.</em>
</p>

Ahead-of-time F# compiler producing native executables without managed runtime or garbage collection. Leverages [F# Native Compiler Services (FNCS)](https://github.com/FidelityFramework/fsnative) for type checking and semantic analysis, generates MLIR through Alex multi-targeting layer, produces native binaries via LLVM.

## Architecture

Firefly implements a true nanopass compiler architecture with ~25 distinct passes from F# source to native binary. Each pass performs a single, well-defined transformation on an intermediate representation.

### Nanopass Pipeline

```
F# Source
    ↓
┌─────────────────────────────────────────────────────────────┐
│ FNCS (6 phases)                                             │
│ Phase 0: FCS parse and type check                           │
│ Phase 1: Structural construction (SynExpr → PSG)            │
│ Phase 2: Symbol correlation (attach FSharpSymbol)           │
│ Phase 3: Soft-delete reachability (mark unreachable)        │
│ Phase 4: Typed tree overlay (type resolution via zipper)    │
│ Phase 5+: Enrichment (def-use, operations, saturation)      │
└─────────────────────────────────────────────────────────────┘
    ↓ PSG (Program Semantic Graph)
┌─────────────────────────────────────────────────────────────┐
│ Alex Witnesses (16 category-selective generators)           │
│ - ApplicationWitness: function calls                         │
│ - LambdaWitness: function definitions                        │
│ - ControlFlowWitness: if/while/for                           │
│ - MemoryWitness: allocations                                 │
│ - OptionWitness, SeqWitness, LazyWitness: type constructors  │
│ - 10 additional witnesses for complete F# coverage          │
└─────────────────────────────────────────────────────────────┘
    ↓ Portable MLIR (memref, arith, func, index, scf)
┌─────────────────────────────────────────────────────────────┐
│ MLIR Structural Passes (4 passes)                           │
│ 1. Structural folding (deduplicate function bodies)          │
│ 2. Declaration collection (external function declarations)   │
│ 3. Type normalization (insert memref.cast at call sites)     │
│ 4. FFI conversion (delegated to mlir-opt)                    │
└─────────────────────────────────────────────────────────────┘
    ↓ MLIR (portable dialects)
┌─────────────────────────────────────────────────────────────┐
│ mlir-opt Dialect Lowering                                    │
│ - memref → LLVM struct                                       │
│ - arith → LLVM arithmetic                                    │
│ - scf → cf → LLVM control flow                               │
│ - index → platform word size                                 │
└─────────────────────────────────────────────────────────────┘
    ↓ LLVM IR
┌─────────────────────────────────────────────────────────────┐
│ LLVM + Clang                                                 │
│ - Optimization passes                                        │
│ - Code generation                                            │
│ - Linking                                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
Native Binary (zero runtime dependencies)
```

## Architectural Principles

**1. Nanopass Throughout**
Unlike traditional compilers with monolithic passes, Firefly uses single-purpose transformations at every tier. Each pass is independently testable and inspectable with `-k` flag.

**2. Coeffects Over Runtime**
Pre-computed analysis (SSA assignment, platform resolution, mutability tracking) guides code generation. No runtime discovery.

**3. Codata Witnesses**
Witnesses observe PSG structure and return MLIR operations. They do not build or transform—observation only. This preserves PSG immutability.

**4. Quotations as Semantic Carriers**
F# quotations (`Expr<'T>`) carry platform constraints and peripheral descriptors through compilation as inspectable data structures. No runtime evaluation.

**5. Zipper + XParsec**
Bidirectional PSG traversal with composable pattern matching. Enables local reasoning without global context threading.

**6. Portable Until Proven Backend-Specific**
MiddleEnd emits only portable MLIR dialects (memref, arith, func, index, scf). Target-specific lowering delegated to mlir-opt and LLVM.

## Native Type System

FNCS provides native type universe (`NTUKind`) at compile time. Types are compiler intrinsics, not runtime constructs:

- Primitives: `i8`, `i16`, `i32`, `i64`, `f32`, `f64` → MLIR integer/float types
- Pointers: `nativeptr<'T>` → opaque pointers
- Strings: Fat pointers `{ptr: memref<?xi8>, len: index}` → memref operations
- Structures: Records/unions → MLIR struct types with precise layout

### Intrinsic Operations

Platform operations defined in FNCS as compiler intrinsics:

**System (`Sys` module):**
- `Sys.write(fd: i64, buf: nativeptr<i8>, count: i64): i64` — syscall
- `Sys.read(fd: i64, buf: nativeptr<i8>, count: i64): i64` — syscall
- `Sys.exit(code: i32): unit` — process termination

**Memory (`NativePtr` module):**
- `NativePtr.read(ptr: nativeptr<'T>): 'T` — load
- `NativePtr.write(ptr: nativeptr<'T>, value: 'T): unit` — store
- `NativePtr.stackalloc(count: i64): nativeptr<'T>` — stack allocation

All intrinsics resolve to platform-specific MLIR during Alex traversal.

## Minimal Example

```fsharp
module HelloWorld

[<EntryPoint>]
let main argv =
    Console.write "Hello, World!"
    0
```

Compiles to native binary with:
- Zero .NET runtime dependencies
- Direct syscalls for I/O
- Stack-only allocation (no heap)
- MLIR → LLVM optimization

```bash
firefly compile HelloWorld.fidproj
./target/helloworld  # Freestanding native binary
```

See `/samples/console/FidelityHelloWorld/` for progressive examples demonstrating pipes, currying, pattern matching, closures, sequences.

## Project Configuration

`.fidproj` files use TOML:

```toml
[package]
name = "HelloWorld"

[compilation]
memory_model = "stack_only"
target = "native"

[build]
sources = ["HelloWorld.fs"]
output = "helloworld"
output_kind = "console"  # or "freestanding"
```

## Build Workflow

```bash
# Build compiler
cd src && dotnet build

# Compile project
firefly compile MyProject.fidproj

# Keep intermediates for inspection
firefly compile MyProject.fidproj -k
```

### Intermediate Artifacts

With `-k` flag, inspect each nanopass output in `target/intermediates/`:

| File | Nanopass Output |
|------|----------------|
| `01_psg0.json` | Initial PSG with reachability |
| `02_intrinsic_recipes.json` | Intrinsic elaboration recipes |
| `03_psg1.json` | PSG after intrinsic fold-in |
| `04_saturation_recipes.json` | Baker saturation recipes |
| `05_psg2.json` | Final saturated PSG to Alex |
| `06_coeffects.json` | SSA, platform, mutability analysis |
| `07_output.mlir` | Alex-generated portable MLIR |
| `08_after_structural_folding.mlir` | Deduplicated function bodies |
| `09_after_ffi_conversion.mlir` | FFI boundary preparation |
| `10_after_declaration_collection.mlir` | External function declarations |
| `11_after_type_normalization.mlir` | Call site type casts |
| `12_output.ll` | LLVM IR after mlir-opt lowering |

### Regression Testing

```bash
cd tests/regression
dotnet fsi Runner.fsx                    # All samples
dotnet fsi Runner.fsx -- --parallel      # Parallel execution
dotnet fsi Runner.fsx -- --sample 01_HelloWorldDirect
```

## Directory Structure

```
src/
├── CLI/                    Command-line interface
├── Core/                   Configuration, timing, diagnostics
├── FrontEnd/               FNCS integration
├── MiddleEnd/
│   ├── PSGElaboration/     Coeffect analysis (SSA, platform, etc.)
│   └── Alex/               MLIR generation layer
│       ├── Dialects/       MLIR type system
│       ├── CodeGeneration/ Type mapping, sizing
│       ├── Traversal/      PSGZipper, XParsec combinators
│       ├── Witnesses/      16 category-selective generators
│       ├── Patterns/       Composable MLIR templates
│       └── Pipeline/       Orchestration, MLIR passes
└── BackEnd/                LLVM compilation, linking
```

## Multi-Stack Targeting

Portable MLIR enables diverse hardware targets:

| Target | Status | Lowering Path |
|--------|--------|---------------|
| x86-64 CPU | ✅ Working | memref → LLVM struct |
| ARM Cortex-M | 🚧 Planned | memref → custom embedded lowering |
| CUDA GPU | 🚧 Planned | memref → SPIR-V/PTX lowering |
| AMD ROCm | 🚧 Planned | memref → SPIR-V lowering |
| Xilinx FPGA | 🚧 Planned | memref → HDL stream buffer |
| CGRA | 🚧 Planned | memref → dataflow lowering |
| NPU | 🚧 Planned | memref → tensor descriptor |
| WebAssembly | 🚧 Planned | memref → WASM linear memory |

Previously blocked by hard-coded LLVM types. Now possible via target-specific mlir-opt lowering.

## Documentation

| Document | Content |
|----------|---------|
| `docs/Architecture_Canonical.md` | FNCS-first architecture, intrinsic modules |
| `docs/PSG_Nanopass_Architecture.md` | Phase 0-5+ detailed design |
| `docs/TypedTree_Zipper_Design.md` | Zipper traversal, XParsec integration |
| `docs/XParsec_PSG_Architecture.md` | Pattern combinators, codata witnesses |
| `docs/Baker_Architecture.md` | Phase 4 type resolution |
| `docs/PRDs/INDEX.md` | Product requirement documents by category |

## Roadmap

Development organized by category-prefixed PRDs. See [docs/PRDs/INDEX.md](docs/PRDs/INDEX.md).

**Completed:**
- F-01 through F-10: Foundation (samples 01-10)
- C-01 through C-07: Closures, higher-order functions, recursion, sequences

**In Progress:**
- A-01 through A-06: Async workflows, region-based memory
- Multi-stack targeting (ARM Cortex-M, GPU, FPGA)

**Planned:**
- I-01, I-02: Socket I/O, WebSocket
- T-01 through T-05: Threads, actors, parallel execution
- E-01 through E-03: Embedded MCU support

## Contributing

Areas of interest:
- MLIR dialect design for novel hardware targets
- Memory optimization patterns
- Nanopass transformations for advanced F# features
- F* integration for proof-carrying code

## License

Dual-licensed under Apache License 2.0 and Commercial License. See [Commercial.md](Commercial.md) for commercial use. Patent notice: U.S. Patent Application No. 63/786,247 "System and Method for Zero-Copy Inter-Process Communication Using BARE Protocol". See [PATENTS.md](PATENTS.md).

## Acknowledgments

- **Don Syme and F# Contributors**: Quotations, active patterns, computation expressions enable self-hosting
- **MLIR Community**: Multi-level IR infrastructure
- **LLVM Project**: Robust code generation
- **Nanopass Framework**: Compiler architecture principles
- **Triton-CPU**: MLIR-based compilation patterns
