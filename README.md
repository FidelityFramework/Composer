# Composer: Clef Compiler

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-orange.svg)](Commercial.md)
[![Architecture](https://img.shields.io/badge/Architecture-CCS%20%2F%20Baker%20%2F%20Alex-blue)](docs/Architecture_Canonical.md)

<p align="center">
🚧 <strong>Under Active Development</strong> 🚧<br>
<em>Coverage is tracked by the owning PRDs and dated validation waypoints. Not production-ready.</em>
</p>

Ahead-of-time Clef compiler producing native executables without managed runtime or garbage collection. Uses [Clef Compiler Services (CCS)](https://github.com/FidelityFramework/clef) for type checking and semantic analysis, generates MLIR through Alex multi-targeting layer, produces native binaries via LLVM.

## Compiler and editor integration

[Lattice integration](docs/Lattice_Integration.md) coordinates work across CCS, Composer, the VSCode and Neovim/Vim clients, grammar and helper repositories. This solution now includes [CCS.Editor](src/CCS.Editor/README.md) and the [Lattice server](src/Lattice.Server/README.md), with a local [HelloDimensionsProof](samples/lattice/HelloDimensionsProof/README.md) editor demo. It shows dimensional hover, compiler diagnostics and expandable source obligations dispatched to cvc5. Compiler-branch reconciliation and the broader editor gates remain explicit in the integration design.

[CCS architecture](docs/CCS_Architecture.md) describes the compiler-owned facts that both lowering and editor queries consume. [BAREWire](https://github.com/FidelityFramework/BAREWire) and [Fidelity.Platform](https://github.com/FidelityFramework/Fidelity.Platform) supply contracts and target declarations used in that reasoning. Language requirements remain in the [Clef specification](https://github.com/FidelityFramework/clef-lang-spec).

[Interactive compiler workbench and native REPL bridge](docs/Interactive_Compiler_Workbench.md)
is planned work alongside language completion. It starts with a bounded SageFS
hosting evaluation, preserves Baker/Alex authority, and coordinates shared
Lattice/MCP sessions, responsive design-time proof dispatch and native ORC
execution. Its roadmap milestones do not assert an implemented adapter or JIT.

The sample counts and recent-change lists below retain their February 2026 dates; they are historical measurements, not results from the current tooling integration gates.

[Proof composition and the Rocq toolchain](docs/Proof_Composition_Architecture.md) records the design for automatically composing local, concurrent, distributed and device-level evidence. It identifies reusable Iris/Verdi-family foundations, their semantic integration requirements, the managed toolchain and the gates separating proposed coverage from demonstrated verification.

[FPGA targeting and artifact verification](docs/fpga-targeting/README.md) is a
dedicated roadmap for a Clef-fed Dynamatic fork, Colibri-centered circuit
realization, direct VHDL-2008 output and an independently implemented
VHDL-to-Rocq adapter. It shares admission and proof infrastructure with the
existing compiler; its [milestones](docs/fpga-targeting/07_roadmap.md) distinguish
planned circuit, mapped-netlist and bitstream verification from current evidence.

## Historical validation snapshot (February 2026)

**Working Samples**: 3 of 16 console samples compile and execute correctly:
- ✅ 01_HelloWorldDirect (static strings, basic Console)
- ✅ 02_HelloWorldSaturated (mutable variables in loops, string interpolation)
- ✅ 03_HelloWorldHalfCurried (pipe operators, function values)

**Recent Achievements**:
- **VarRef SSA Auto-Loading**: Mutable variables used as memref indices now auto-load values compositionally
- **CCS Contract Compliance**: NativeStr.fromPointer honors substring extraction via allocate + memcpy
- **Compositional Patterns**: Element/Pattern/Witness stratification validated with cross-discipline composition

**Known Limitations**:
- 13 of 16 samples fail compilation (closure capture, higher-order functions, complex control flow)
- Managed mutability limited to local variables in simple loops
- Partial escape analysis (closure capture detection works, mutable lifetime integration pending)
- Generic instantiation and SRTP resolution issues remain

See: `docs/PRDs/README.md` for full feature roadmap and status.

## Architecture

CCS constructs the typed graph and Baker elaborates/saturates its computation and
relationships. Composer consumes that graph through Alex and realizes the selected
target. The [pipeline overview](docs/Architecture_Canonical.md) is the current
source map; older pass counts and FCS typed-tree-overlay diagrams are historical.

```text
Clef source + project/library/platform inputs
  -> CCS checking and typed PSG construction
  -> Baker recipes / saturation / owning admission and obligation passes
  -> Composer source-diagnostic and target gates
  -> Alex ctx pull through graph/codata, coeffects and the Huet zipper
       Elements -> Patterns -> Witnesses
       admitted physical operations + required graph correspondence
  -> declaration collection and bounded correspondence checks
  -> selected backend
       ELF: mlir-opt -> mlir-translate -> opt (target bitcode)
            -> ld.lld (LLVM code generation and linking)
  -> native artifact and its execution/verification gates
```

The [direct LLVM/LLD backend](docs/LLVM_Backend.md) invokes neither Clang nor a
separate `llc`. Native console deployment can use libc, startup objects and a
loader; other deployment modes retain their own runtime requirements. Avoiding
the .NET runtime does not imply zero native runtime dependencies.

### Architectural principles

- **Baker settles; Alex witnesses.** Source algorithms, evaluation relationships,
  captures, residence, layout and proof premises belong in their owning CCS/Baker
  stages. Missing semantics cannot be supplied by a late emitter or F#/C surrogate.
- **Context pull preserves position.** The Huet zipper holds focus, path and graph.
  Program facts come from graph nodes/codata; current `TransferCoeffects` holds
  platform reads and target selection. Emission accumulators, scopes and visited
  sets are separate bookkeeping, not a semantic reconstruction layer.
- **Compose the physical vocabulary.** Witnesses observe through `ctx` and invoke
  Patterns, which compose Elements. `module internal` restricts assembly visibility;
  it does not prohibit Witness-to-Element access within the Composer assembly.
  There is no correctness-bearing line-count limit for a Witness or Pattern.
- **Describe actual traversal.** Registered witnesses are combined in one
  post-order traversal with scope-owned callbacks. This is not independent parallel
  traversal per witness. `Values.fs` derives SSA names from node/role ordinals and
  block arguments; CCS does not run an SSA-preassignment pass.
- **Thin emission retains admitted structure.** Structured operations and regions
  are compatible with flat/thin witnessing. Alex is target-aware; the admission
  key is expression family × platform/backend profile × witness form.
- **Evidence has a scope.** Graph tests, MLIR verification, solver answers and
  native oracles establish different boundaries. [M-01](docs/PRDs/M-01-DialectAdmission.md)
  still owns planned general admission, target-path reconciliation and correlated
  fact/proof transport. Existing bounded checks do not establish complete coverage.

See [Alex Architecture](docs/Alex_Architecture_Overview.md) for the current
implementation and [Thin Middle End](docs/Thin_Middle_End_Design.md) for its
semantic boundary.

## Native types and representations

CCS's native type algebra retains Clef dimensions and declared representation
requirements. Integer widths, layouts, callable environments and lifetimes must
come from their owning language/platform contracts, not a host-language type or a
convenient machine-width default. Alex reads those facts when selecting physical
carriers.

Current CPU string patterns use byte memrefs, while settled static strings can
share a BAREWire pool with exact offsets, bytes and obligation anchors. A memref
is an MLIR carrier, not the source-language definition of a string or a universal
promise about descriptor size. Closure, sequence and aggregate carriers likewise
have separately admitted layout and residence requirements. The
[Alex component suite](tests/Alex.Tests/README.md) records the boundaries tested.

Internal `TNativePtr` plumbing is not a user-denotable general pointer API. The
[FFI contract](../clef-lang-spec/spec/ffi-boundary.md) and
[C-01](docs/PRDs/C-01-Closures.md) govern typed foreign boundaries and remaining
representation work. Earlier `NativePtr` examples or MLIR-shaped intrinsic
signatures are not Clef surface declarations.

## Minimal example

The [direct HelloWorld sample](samples/console/FidelityHelloWorld/01_HelloWorldDirect/HelloWorld.fs)
exercises static output through the ordinary pipeline:

```fsharp
module Examples.HelloWorldDirect

[<EntryPoint>]
let main argv =
    Console.write "Hello, World!"
    Console.writeln ""
    0
```

Its [project](samples/console/FidelityHelloWorld/01_HelloWorldDirect/HelloWorld.fidproj)
selects a Fidelity.Platform dependency and CPU target. Use that declared profile
when reproducing the sample; source syntax alone does not establish a target.

```bash
dotnet build src/Composer.fsproj
src/bin/Debug/net10.0/Composer compile samples/console/FidelityHelloWorld/01_HelloWorldDirect/HelloWorld.fidproj -k
samples/console/FidelityHelloWorld/01_HelloWorldDirect/targets/helloworld
```

For a new `.fidproj`, copy an existing project for the intended platform and
update its inputs. CPU projects use `target = "cpu"`; `output_kind` selects the
deployment/runtime contract rather than the compiler's semantic model. Cross
runtime/link inputs are documented in [LLVM Backend](docs/LLVM_Backend.md).

## Build and validation

Coordinate builds when Composer and CCS are shared with another task. The
[regression runner](tests/regression/README.md) builds the compiler and runs
sample/native-output cases sequentially:

```bash
cd tests/regression
dotnet fsi Runner.fsx
dotnet fsi Runner.fsx -- --verbose
dotnet fsi Runner.fsx -- --sample 02_HelloWorldSaturated
```

`--parallel` is unsupported. Owning [source admission](tests/SourceAdmission/README.md),
[Alex component](tests/Alex.Tests/README.md), proof and native gates supplement
these regressions. A subset or a historical sample count is not a fresh full-suite
result.

With `-k`, retained artifacts in the sample's `targets/intermediates/` include:

| Artifact | Boundary |
|---|---|
| `01_psg0.json` | Initial typed PSG/reachability view. |
| `02_intrinsic_recipes.json`, `03_psg1.json` | Intrinsic elaboration and fold-in. |
| `04_saturation_recipes.json`, `05_psg2.json` | Baker recipes and final graph view. |
| `07_output.mlir` | MLIR retained by the orchestrator. |
| `08_after_declaration_collection.mlir` | Post-witness declaration collection. |
| `09_obligations.mlir` | Optional emitted SMT module; not a discharge verdict. |
| `10_output.mlir` | Final middle-end serialization. |
| `08_output.ll` and associated bitcode | LLVM backend handoff. |

The old `06_coeffects.json` identifier remains reserved in PhaseConfig; its name
does not establish an active separate Composer analysis pass. The obsolete four
middle-end pass sequence and its artifact names are not current output contracts.

## Directory structure

```text
src/
├── CLI/                    Command-line interface
├── Core/                   Pipeline/target orchestration and backend contracts
├── FrontEnd/               Calls CCS project checking
├── CCS.Editor/             Versioned compiler projection and proof dispatch
├── Lattice.Server/         Editor transport and scheduling
├── MiddleEnd/
│   ├── MLIRGeneration.fs   Alex ingress, validation and serialization
│   └── Alex/
│       ├── Dialects/       Physical operations/types and serialization
│       ├── CodeGeneration/ Type and callable-symbol mapping
│       ├── Traversal/      Huet context, derived values, traversal and coverage
│       ├── XParsec/        Graph observation combinators
│       ├── Elements/       Atomic physical operations
│       ├── Patterns/       Composed admitted forms
│       ├── Witnesses/      Context-pulled graph observation
│       └── Pipeline/       Post-witness declaration collection
└── BackEnd/                Selected target realization and artifacts
```

Baker, graph construction and owning semantic analyses are in the companion
[Clef repository](https://github.com/FidelityFramework/clef), not an additional
Composer `PSGElaboration` pipeline.

## Targets and roadmap

The [PRD index](docs/PRDs/README.md) records current feature/target scope and
acceptance evidence. [Language Coverage Waypoints](docs/Language_Coverage_Waypoints.md)
records coordinated revisions, remaining failures and native oracles. Existing
CPU, MCU and other backend implementations must be distinguished from complete
language/target admission; a portable MLIR vocabulary alone does not implement
a new target.

Relevant workstreams include [Cortex-M](docs/MCU_Backend.md),
[FPGA](docs/fpga-targeting/README.md),
[WebAssembly](docs/wasm-targeting/README.md),
[JavaScript](docs/javascript-targeting/README.md),
[M-01 dialect admission](docs/PRDs/M-01-DialectAdmission.md), and the
[interactive workbench](docs/Interactive_Compiler_Workbench.md). Follow their own
status records; the February snapshot above is preserved as history.

## Documentation

| Document | Content |
|---|---|
| [Pipeline overview](docs/Architecture_Canonical.md) | Current CCS/Baker/Alex/backend ownership and source map. |
| [Baker contract](../clef/docs/fidelity/Baker_Saturation_Architecture.md) | Construction, recipes, saturation and graph relationships. |
| [Alex overview](docs/Alex_Architecture_Overview.md) | Context pull, positional traversal, physical expression and evidence limits. |
| [CCS architecture](docs/CCS_Architecture.md) | Semantic service and graph facts. |
| [Lattice integration](docs/Lattice_Integration.md) | Repository map, editor transport and proof-view gates. |
| [Workbench](docs/Interactive_Compiler_Workbench.md) | Planned resident compiler and native REPL bridge. |
| [LLVM backend](docs/LLVM_Backend.md) | Direct LLVM/LLD realization and native runtime inputs. |
| [PRD index](docs/PRDs/README.md) | Feature statuses with scoped evidence. |
| [C/F checkpoint](docs/C_F_Checkpoint_2026-09-26.md) | Verified September 26 scope, remaining work by C/F owner, and estimate provenance. |

## Recent Changes (February 2026)

### Managed Mutability Milestone

**Achievement**: Local mutable variables in simple loops now work via TMemRef auto-loading.

**What Works**:
- `let mutable pos = 0` → `memref.alloca() : memref<1xindex>`
- Mutable variables as memref indices (auto-load value before use)
- Mutable variables in loop conditions (while, for)
- String operations honoring CCS contracts (substring extraction)

**What Doesn't Work**:
- Mutable variables captured in closures (closure detection exists, allocation strategy integration pending)
- Mutable variables passed across function boundaries (return/byref escape detection needed)
- Higher-order functions with mutable state
- Complex control flow with escaping mutables

**Architectural Pattern Established**: Compositional auto-loading via type-driven discrimination (Rule 9 in managed mutability architecture principles).

See: Serena memory `managed_mutability_feb2026_milestone` for complete details.

## Contributing

Areas of interest:
- MLIR dialect design for novel hardware targets
- Memory optimization patterns (escape analysis, loop unrolling)
- Nanopass transformations for advanced Clef features
- Closure capture and higher-order function support
- Graph-resident obligations, cvc5 dispatch and the planned proof-composition service

## License

Dual-licensed under Apache License 2.0 and Commercial License. See [Commercial.md](Commercial.md) for commercial use. Patent notice: U.S. Patent Application No. 63/786,247 "System and Method for Zero-Copy Inter-Process Communication Using BARE Protocol". See [PATENTS.md](PATENTS.md).

## Acknowledgments

- **Don Syme and F# contributors**: Language and compiler heritage used by the bootstrap implementation
- **Clef contributors**: Native language, graph and compiler development
- **MLIR Community**: Multi-level IR infrastructure
- **LLVM Project**: Robust code generation
- **Nanopass Framework**: Compiler architecture principles
- **Triton-CPU**: MLIR-based compilation patterns
- **MLKit**: Flat closure representation patterns
