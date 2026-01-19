# Firefly Sample Suite

This directory contains sample projects demonstrating Firefly's capabilities for compiling F# to native code. The samples serve as both documentation and regression tests for the compiler.

## FidelityHelloWorld: The Canonical Progression

The `console/FidelityHelloWorld/` directory contains a carefully designed progression of samples that exercise increasingly sophisticated F# features. Each sample builds on the previous, proving specific compiler capabilities needed for native compilation.

> **Design Philosophy**: These samples follow the principle that the same output can require dramatically different compilation complexity. Sample 01 and Sample 04 both print "Hello, World!" but Sample 04's full currying requires closure creation, capture analysis, and escape analysis that Sample 01 doesn't need.

### Sample Progression

| # | Sample | F# Features | Compiler Capabilities | Status |
|---|--------|-------------|----------------------|--------|
| **01** | HelloWorldDirect | Static strings, basic calls | Linear emission, FNCS intrinsics | ✓ |
| **02** | HelloWorldSaturated | Arena allocation, byref parameters | Lifetime tracking, saturated calls | ✓ |
| **03** | HelloWorldHalfCurried | Pipe operator (`\|>`), function values | Pipe desugaring, forward references | ✓ |
| **04** | HelloWorldFullCurried | Full currying, partial application | Closure creation, capture analysis | ✓ |
| **05** | AddNumbers | Discriminated unions, pattern matching | DU representation, multi-case dispatch | ✓ |
| **06** | AddNumbersInteractive | String parsing, arithmetic on DU | Parse intrinsics, type coercion | ✓ |
| **07** | BitsTest | Byte order, bit casting intrinsics | Platform-specific bit operations | ✓ |
| **08** | Option | Option type, pattern matching | Built-in Option compilation | ✓ |
| **09** | Result | Result type, error handling | Built-in Result compilation | ✓ |
| **10** | Records | Record types, copy-update, nesting | Struct layout, field access | ✓ |
| **11** | Closures | Lambdas, captured variables, mutable state | Flat closure analysis, arena capture | ✓ |
| **12** | HigherOrderFunctions | Functions as values, composition | Function pointers, HOF patterns | ✓ |
| **13** | Recursion | Tail recursion, mutual recursion | TCO detection, forward declarations | ✓ |
| **14** | Lazy | `lazy { }`, `Lazy.force` | Lazy struct, flat closures (PRD-14) | ✓ |
| **15** | SimpleSeq | `seq { }`, `yield`, while loops | State machine compilation (PRD-15) | ✓ |
| **16** | SeqOperations | Seq.map, filter, take, fold, collect | Pipeline composition (PRD-16) | ✓ |

### Feature Categories

The progression divides into four phases:

**Phase 1: Function Calling Patterns (01-04)**
Demonstrates the spectrum from direct calls to full currying, each requiring different calling conventions and memory management.

**Phase 2: Core Data Types (05-10)**
Exercises discriminated unions, Option, Result, and Records, the foundational F# types that must compile to efficient native representations.

**Phase 3: Functional Patterns (11-13)**
Proves closure capture analysis, higher-order function support, and recursion optimization, the patterns that distinguish functional from imperative compilation.

**Phase 4: Lazy and Sequences (14-16)**
Implements deferred computation and sequence expressions, requiring state machines and composed pipeline structs.

## Building and Running Samples

### Single Sample

```bash
# Navigate to sample directory
cd samples/console/FidelityHelloWorld/01_HelloWorldDirect

# Compile
firefly compile HelloWorld.fidproj

# Run
./target/helloworld
```

### With Intermediate Files

The `-k` flag preserves intermediate artifacts for debugging:

```bash
firefly compile HelloWorld.fidproj -k

# Examine intermediates
ls target/intermediates/
# fncs_phase_1_structural.json  - PSG after structural construction
# fncs_phase_4_reachability.json - PSG after reachability analysis
# fncs_phase_5_final.json       - Final PSG with all coeffects
# alex_coeffects.json           - Coeffects computed for emission
# HelloWorld.mlir               - Generated MLIR
# HelloWorld.ll                 - LLVM IR
```

### Interactive Samples

Samples 02, 03, 04, and 06 require user input. Each has a `.stdin` file for automated testing:

```bash
# Interactive mode
cd samples/console/FidelityHelloWorld/03_HelloWorldHalfCurried
firefly compile HelloWorld.fidproj
./target/helloworld
# Enter your name: Alice
# Hello, Alice!

# Automated with stdin file
./target/helloworld < HelloWorld.stdin
```

## Regression Test Suite

The samples serve as regression tests via the test harness in `tests/regression/`.

### Running All Tests

```bash
cd tests/regression

# Run full suite (sequential)
dotnet fsi Runner.fsx

# Run in parallel (faster)
dotnet fsi Runner.fsx -- --parallel

# Verbose output with detailed errors
dotnet fsi Runner.fsx -- --verbose
```

### Running Specific Samples

```bash
# Single sample
dotnet fsi Runner.fsx -- --sample 01_HelloWorldDirect

# Multiple samples
dotnet fsi Runner.fsx -- --sample 07_BitsTest --sample 14_Lazy

# Samples by prefix (all closure-related)
dotnet fsi Runner.fsx -- --sample 11_Closures --sample 12_HigherOrderFunctions
```

### Test Output

The test runner reports compilation and execution status for each sample:

```
=== Firefly Regression Test ===
Run ID: 2026-01-19T10:30:00

[Compilation Phase]
  01_HelloWorldDirect     PASS (0.8s)
  02_HelloWorldSaturated  PASS (1.2s)
  ...

[Execution Phase]
  01_HelloWorldDirect     PASS
  02_HelloWorldSaturated  PASS
  ...

[Summary]
  Total: 16  Passed: 16  Failed: 0  Skipped: 0
  Duration: 45.3s
  Status: ALL PASS
```

### Test Configuration

Test definitions live in `tests/regression/Manifest.toml`:

```toml
[[samples]]
name = "01_HelloWorldDirect"
project = "HelloWorld.fidproj"
binary = "helloworld"
timeout = 30

[[samples]]
name = "15_SimpleSeq"
project = "SimpleSeq.fidproj"
binary = "SimpleSeq"
timeout = 60  # Longer timeout for complex sample
```

## Directory Structure

```
samples/
├── console/
│   ├── FidelityHelloWorld/     # Canonical sample progression
│   │   ├── 01_HelloWorldDirect/
│   │   ├── 02_HelloWorldSaturated/
│   │   ├── ...
│   │   └── 16_SeqOperations/
│   ├── TimeLoop/               # Platform time operations
│   └── SignalTest/             # Reactive signals (experimental)
├── embedded/                   # ARM microcontroller targets
│   ├── common/                 # Shared startup and linker scripts
│   ├── stm32l5-blinky/         # LED blink on NUCLEO-L552ZE-Q
│   └── stm32l5-uart/           # Serial communication
├── sbc/                        # Single-board computer targets
│   └── sweet-potato-blinky/    # LED blink on Libre Sweet Potato
├── templates/                  # Platform configuration templates
├── samples.json                # Sample catalog metadata
└── README.md                   # This file
```

## Sample Details

### Samples 01-04: The Hello World Progression

These four samples all produce a greeting but demonstrate fundamentally different compilation paths:

| Sample | Calling Convention | Memory Model | Compilation Complexity |
|--------|-------------------|--------------|----------------------|
| **01_Direct** | Direct intrinsic calls | Static strings only | ~20 MLIR lines |
| **02_Saturated** | All arguments provided | Arena allocation | ~40 MLIR lines |
| **03_HalfCurried** | Pipe operator | Stack frame for readln | ~60 MLIR lines |
| **04_FullCurried** | Partial application | Closure allocation | ~100 MLIR lines |

See the blog post [Learning to Walk](https://speakez.com/blog/learning-to-walk/) for a detailed explanation of how these samples reveal the compiler's traversal strategy.

### Samples 05-10: Core Data Types

| Sample | Type Tested | Key Feature |
|--------|------------|-------------|
| **05_AddNumbers** | Discriminated Union | Multi-case pattern matching |
| **06_AddNumbersInteractive** | DU + Parsing | String.contains, Parse intrinsics |
| **07_BitsTest** | Intrinsics | Byte order, bit casting |
| **08_Option** | Option<'T> | Some/None dispatch |
| **09_Result** | Result<'T,'E> | Ok/Error handling |
| **10_Records** | Record types | Field access, copy-update, nesting |

### Samples 11-13: Functional Patterns

| Sample | Pattern | Key Compiler Feature |
|--------|---------|---------------------|
| **11_Closures** | Closure capture | Flat closure analysis, arena mutable state |
| **12_HigherOrderFunctions** | HOF patterns | Function pointers, composition |
| **13_Recursion** | Recursive functions | Tail call optimization, mutual recursion |

See [Gaining Closure](https://speakez.com/blog/gaining-closure/) for the flat closure architecture these samples exercise.

### Samples 14-16: Lazy and Sequences

| Sample | Feature | Architecture |
|--------|---------|--------------|
| **14_Lazy** | `lazy { }`, `Lazy.force` | Extended flat closure with memoization state |
| **15_SimpleSeq** | `seq { }`, `yield`, while loops | State machine compilation |
| **16_SeqOperations** | Seq.map, filter, fold, collect | Composed pipeline structs |

See [Why Lazy Is Hard](https://speakez.com/blog/why-lazy-is-hard/) and [Seq'ing Simplicity](https://speakez.com/blog/seqing-simplicity/) for the design decisions behind these samples.

## Embedded and SBC Samples

These samples target bare-metal ARM platforms without an operating system.

### STM32L5 Blinky

LED blink demo for the NUCLEO-L552ZE-Q development board.

```bash
cd samples/embedded/stm32l5-blinky
firefly compile Blinky.fidproj --target thumbv8m.main-none-eabihf

# Flash via OpenOCD
openocd -f interface/stlink.cfg -f target/stm32l5x.cfg \
  -c "program blinky.elf verify reset exit"
```

### Sweet Potato Blinky

LED blink demo for the Libre Sweet Potato (Allwinner H6).

```bash
cd samples/sbc/sweet-potato-blinky
firefly compile Blinky.fidproj --target aarch64-unknown-none
```

## Platform Templates

Template files in `templates/` define platform-specific configurations:

| Template | Target | Use Case |
|----------|--------|----------|
| `desktop-x64.toml` | x86-64 Linux/macOS/Windows | Console applications |
| `stm32l5.toml` | STM32L5 Cortex-M33 | Microcontroller |
| `allwinner-h6.toml` | Allwinner H6 Cortex-A53 | Single-board computer |
| `nucleo-l552ze-q.toml` | NUCLEO board specifics | Development board |

## The Road to WREN Stack

The FidelityHelloWorld samples are the proving ground for **WREN Stack** (WebView + Reactive + Embedded + Native), the paved path for building native F# desktop applications. Each sample proves compiler capabilities required for the capstone: a full MailboxProcessor-based actor system that powers native desktop applications with WebView frontends.

### Complete Sample Roadmap

The full progression spans 31 samples across nine phases, culminating in the WREN Stack Alpha:

#### Implemented Samples (01-16)

| # | Sample | PRD | Status |
|---|--------|-----|--------|
| 01-04 | HelloWorld Progression | — | ✓ Complete |
| 05-06 | AddNumbers | — | ✓ Complete |
| 07 | BitsTest | — | ✓ Complete |
| 08-09 | Option / Result | — | ✓ Complete |
| 10 | Records | — | ✓ Complete |
| 11 | Closures | [PRD-11](docs/WREN_Stack_PRDs/PRD-11-Closures.md) | ✓ Complete |
| 12 | HigherOrderFunctions | [PRD-12](docs/WREN_Stack_PRDs/PRD-12-HigherOrderFunctions.md) | ✓ Complete |
| 13 | Recursion | [PRD-13](docs/WREN_Stack_PRDs/PRD-13-Recursion.md) | ✓ Complete |
| 14 | Lazy | [PRD-14](docs/WREN_Stack_PRDs/PRD-14-Lazy.md) | ✓ Complete |
| 15 | SimpleSeq | [PRD-15](docs/WREN_Stack_PRDs/PRD-15-SimpleSeq.md) | ✓ Complete |
| 16 | SeqOperations | [PRD-16](docs/WREN_Stack_PRDs/PRD-16-SeqOperations.md) | ✓ Complete |

#### Phase D: Async via LLVM Coroutines (17-19)

Async is implemented using LLVM coroutine intrinsics (`llvm.coro.*`), not the MLIR async dialect. This compiles to state machines at compile time with no runtime library needed.

| # | Sample | PRD | Features |
|---|--------|-----|----------|
| 17 | BasicAsync | [PRD-17](docs/WREN_Stack_PRDs/PRD-17-BasicAsync.md) | `async { return value }` - trivial coroutine |
| 18 | AsyncAwait | [PRD-18](docs/WREN_Stack_PRDs/PRD-18-AsyncAwait.md) | `async { let! x = ... }` - suspension coeffects |
| 19 | AsyncParallel | [PRD-19](docs/WREN_Stack_PRDs/PRD-19-AsyncParallel.md) | `Async.Parallel` - sequential composition |

#### Phase E: Scoped Regions (20-22)

Compiler-inferred deterministic memory regions with automatic disposal at scope exit. No `IDisposable`, no `use` keyword. Region is a coeffect.

| # | Sample | PRD | Features |
|---|--------|-----|----------|
| 20 | BasicRegion | [PRD-20](docs/WREN_Stack_PRDs/PRD-20-BasicRegion.md) | Region allocation/disposal, `NeedsCleanup` coeffect |
| 21 | RegionPassing | [PRD-21](docs/WREN_Stack_PRDs/PRD-21-RegionPassing.md) | Region parameters, `BorrowedRegion` coeffect |
| 22 | RegionEscape | [PRD-22](docs/WREN_Stack_PRDs/PRD-22-RegionEscape.md) | Escape prevention, `CopyOut` analysis |

#### Phase F: Networking (23-24)

Region-backed I/O buffers for socket communication.

| # | Sample | PRD | Features |
|---|--------|-----|----------|
| 23 | SocketBasics | [PRD-23](docs/WREN_Stack_PRDs/PRD-23-SocketBasics.md) | TCP sockets via `Sys.*` intrinsics |
| 24 | WebSocketEcho | [PRD-24](docs/WREN_Stack_PRDs/PRD-24-WebSocketEcho.md) | WebSocket echo server, Layer 3 library |

#### Phase G: Desktop Scaffold (25-26)

FFI bindings for GTK and WebKitGTK, the foundation of WREN Stack's UI layer.

| # | Sample | PRD | Features |
|---|--------|-----|----------|
| 25 | GTKWindow | [PRD-25](docs/WREN_Stack_PRDs/PRD-25-GTKWindow.md) | GTK window, `FFICall` pattern, `ExternCall` template |
| 26 | WebViewBasic | [PRD-26](docs/WREN_Stack_PRDs/PRD-26-WebViewBasic.md) | WebKitGTK WebView with HTML content |

#### Phase H: Threading Primitives (27-28)

OS thread primitives for true parallelism.

| # | Sample | PRD | Features |
|---|--------|-----|----------|
| 27 | BasicThread | [PRD-27](docs/WREN_Stack_PRDs/PRD-27-BasicThread.md) | `Thread.create` / `Thread.join`, capture coeffects |
| 28 | MutexSync | [PRD-28](docs/WREN_Stack_PRDs/PRD-28-MutexSync.md) | Mutex synchronization, `SyncPrimitive` coeffect |

#### Phase I: MailboxProcessor — The Capstone (29-31)

MailboxProcessor is the **capstone feature** that synthesizes all prior capabilities: async (for message loop), closures (for behavior functions), threading (for true parallelism), scoped regions (for worker memory), and records/DUs (for message types).

| # | Sample | PRD | Features |
|---|--------|-----|----------|
| 29 | BasicActor | [PRD-29](docs/WREN_Stack_PRDs/PRD-29-BasicActor.md) | `MailboxProcessor.Start`, `Post`/`Receive` |
| 30 | ActorReply | [PRD-30](docs/WREN_Stack_PRDs/PRD-30-ActorReply.md) | `PostAndReply` request-response pattern |
| 31 | ParallelActors | [PRD-31](docs/WREN_Stack_PRDs/PRD-31-ParallelActors.md) | Multiple actors with region-based worker memory |

### WREN Stack Alpha

With Sample 31 complete, the **WREN Stack Alpha** becomes possible:

```
┌─────────────────────────────────────────────────────────────────────┐
│                        WREN Stack Application                        │
│  ┌──────────────────┐         ┌──────────────────┐                  │
│  │  Frontend (Fable) │         │  Backend (Firefly)│                 │
│  │  - Partas.Solid   │         │  - Native binary  │                 │
│  │  - SolidJS UI     │         │  - GTK/WebKitGTK  │                 │
│  └────────┬─────────┘         └────────┬─────────┘                  │
│           │    ┌────────────────┐      │                             │
│           └───►│  BAREWire IPC  │◄─────┘                             │
│                │  (Shared Types)│                                    │
│                └────────────────┘                                    │
└─────────────────────────────────────────────────────────────────────┘
```

**WREN** = **W**ebView + **R**eactive + **E**mbedded + **N**ative

The goal: `dotnet new wrenstack` creates a working native desktop application with:
- WebView frontend (Partas.Solid → SolidJS)
- Native backend (Firefly → MLIR → native binary)  
- Bidirectional IPC via BAREWire
- Actor-based concurrency via MailboxProcessor

See [WRENStack_Roadmap.md](docs/WRENStack_Roadmap.md) for the complete architecture and milestone checklist.

## Adding New Samples

When contributing new samples:

1. **Follow the numbering convention**: Next sample continues the progression (17, 18, etc.)
2. **Create a PRD first**: Document the feature in `docs/WREN_Stack_PRDs/PRD-##-Name.md`
3. **Include a `.fidproj` file**: Use existing samples as templates
4. **Add expected output**: For regression testing
5. **Update `Manifest.toml`**: Register with the test harness
6. **Document the feature tested**: What compiler capability does this prove?

## Related Documentation

### Architecture
- [WRENStack_Roadmap.md](docs/WRENStack_Roadmap.md) - Complete roadmap and milestone checklist
- [WREN_STACK.md](docs/WREN_STACK.md) - WREN Stack design philosophy
- [FidelityHelloWorld_Progression.md](docs/FidelityHelloWorld_Progression.md) - Detailed sample design rationale

### Blog Posts
- [Learning to Walk](https://speakez.com/blog/learning-to-walk/) - PSG traversal explained via samples
- [Gaining Closure](https://speakez.com/blog/gaining-closure/) - Flat closure architecture
- [Why Lazy Is Hard](https://speakez.com/blog/why-lazy-is-hard/) - Lazy evaluation design
- [Seq'ing Simplicity](https://speakez.com/blog/seqing-simplicity/) - Sequence expression compilation

### PRD Index
All Product Requirements Documents live in `docs/WREN_Stack_PRDs/`:
- PRD-11 through PRD-16: Implemented (Closures → SeqOperations)
- PRD-17 through PRD-19: Async via LLVM Coroutines
- PRD-20 through PRD-22: Scoped Regions
- PRD-23 through PRD-24: Networking
- PRD-25 through PRD-26: Desktop/WebView
- PRD-27 through PRD-28: Threading
- PRD-29 through PRD-31: MailboxProcessor (Capstone)

## License

MIT License - See [LICENSE](../LICENSE) for details.
