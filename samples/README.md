# Composer Sample Suite

This directory contains compiler examples and regression inputs. The
[regression manifest](../tests/regression/Manifest.toml) identifies the selected
console cases and expected output. Hardware acceptance belongs with each
platform and application's evidence.

> **WREN** = **W**ebView + **R**eactive + **E**mbedded + **N**ative

## The Sample Progression

The [console progression](console/FidelityHelloWorld/) exercises language and
compiler features. The table links those topics to their current PRD locations.
It is a design index. Sample availability and execution results come from the
manifest and the corresponding run report.

### Sample topics and design references

| Phase | # | Sample | Key Features | PRD |
|-------|---|--------|--------------|-----|
| **A: Foundations** | 01 | HelloWorldDirect | Static strings, direct calls | [F-01](../docs/PRDs/F-01-HelloWorldDirect.md) |
| | 02 | HelloWorldSaturated | Saturated calls, console input and greeting | [F-02](../docs/PRDs/F-02-ArenaAllocation.md) |
| | 03 | HelloWorldHalfCurried | Forward pipe, named greeting function | [F-03](../docs/PRDs/F-03-PipeOperators.md) |
| | 04 | HelloWorldFullCurried | Full currying, partial application | [F-04](../docs/PRDs/F-04-CurryingLambdas.md) |
| | 05 | AddNumbers | Discriminated unions, pattern matching | [F-05](../docs/PRDs/F-05-DiscriminatedUnions.md) |
| | 06 | AddNumbersInteractive | String parsing, arithmetic | [F-06](../docs/PRDs/F-06-InteractiveParsing.md) |
| | 07 | BitsTest | Bitwise operations, shifts, comparisons and Boolean operators | [F-07](../docs/PRDs/F-07-BitwiseOperators.md) |
| | 08 | Option | Option type, Some/None | [F-08](../docs/PRDs/F-08-OptionType.md) |
| | 08a | [OptionDefaults](console/FidelityHelloWorld/08a_OptionDefaults/) | Eager fallback, stored defaults, nested/function/measured payloads | [C-04](../docs/PRDs/C-04-CoreCollections.md) |
| | 08b | [OptionDefaultWith](console/FidelityHelloWorld/08b_OptionDefaultWith/) | Deferred defaults, thunk snapshots, mutable captures and unit effects | [C-04](../docs/PRDs/C-04-CoreCollections.md) |
| | 09 | Result | Result type, Ok/Error | [F-09](../docs/PRDs/F-09-ResultType.md) |
| | 10 | Records | Record types, copy-update, nesting | [F-10](../docs/PRDs/F-10-RecordTypes.md) |
| **B: Functional** | 11 | Closures | Lambdas, capture analysis, mutable state | [C-01](../docs/PRDs/C-01-Closures.md) |
| | 12 | HigherOrderFunctions | Functions as values, composition | [C-02](../docs/PRDs/C-02-HigherOrderFunctions.md) |
| | 13 | Recursion | Tail recursion, mutual recursion | [C-03](../docs/PRDs/C-03-Recursion.md) |
| **C: Lazy/Seq** | 14 | Lazy | `lazy { }`, `Lazy.force`, flat closures | [C-05](../docs/PRDs/C-05-Lazy.md) |
| | 15 | SimpleSeq | `seq { }`, `yield`, state machines | [C-06](../docs/PRDs/C-06-SimpleSeq.md) |
| | 16 | SeqOperations | Seq.map, filter, fold, collect | [C-07](../docs/PRDs/C-07-SeqOperations.md) |
| **D: Async** | 17 | BasicAsync | Deferred execution and completion | [A-01](../docs/PRDs/A-01-BasicAsync.md) |
| | 18 | AsyncAwait | `let!`, suspension coeffects | [A-02](../docs/PRDs/A-02-AsyncAwait.md) |
| | 19 | AsyncParallel | `Async.Parallel` composition | [A-03](../docs/PRDs/A-03-AsyncParallel.md) |
| **E: Regions** | 20 | BasicRegion | Region alloc/dispose, `NeedsCleanup` | [A-04](../docs/PRDs/A-04-BasicRegion.md) |
| | 21 | RegionPassing | Region parameters, `BorrowedRegion` | [A-05](../docs/PRDs/A-05-RegionPassing.md) |
| | 22 | RegionEscape | Escape analysis, `CopyOut` | [A-06](../docs/PRDs/A-06-RegionEscape.md) |
| **F: Networking** | 23 | SocketBasics | TCP via `Sys.*` intrinsics | [I-01](../docs/PRDs/I-01-SocketBasics.md) |
| | 24 | WebSocketEcho | WebSocket protocol | [I-02](../docs/PRDs/I-02-WebSocketEcho.md) |
| **G: Desktop** | 25 | GTKWindow | GTK FFI, `ExternCall` | [D-01](../docs/PRDs/D-01-GTKWindow.md) |
| | 26 | WebViewBasic | WebKitGTK WebView | [D-02](../docs/PRDs/D-02-WebViewBasic.md) |
| **H: Threading** | 27 | BasicThread | `Thread.create`/`join` | [T-01](../docs/PRDs/T-01-BasicThread.md) |
| | 28 | MutexSync | Mutex, `SyncPrimitive` | [T-02](../docs/PRDs/T-02-MutexSync.md) |
| **I: Capstone** | 29 | BasicActor | Actor creation and message delivery | [T-03](../docs/PRDs/T-03-BasicActor.md) |
| | 30 | ActorReply | Request/reply behavior | [T-04](../docs/PRDs/T-04-ActorReply.md) |
| | 31 | ParallelActors | Multi-actor with regions | [T-05](../docs/PRDs/T-05-ParallelActors.md) |

The [threading and actor PRDs](../docs/PRDs/README.md#threading-t-xx---concurrency)
record the corresponding design work. Their presence does not establish that
an application or target implements every operation.

## Building and Running

### Single Sample

```bash
cd samples/console/FidelityHelloWorld/01_HelloWorldDirect
composer compile HelloWorld.fidproj
./targets/helloworld
```

### With Intermediate Files

```bash
composer compile HelloWorld.fidproj -k
ls targets/intermediates/
# Inspect the artifacts produced by this compiler build
```

### Interactive Samples

Samples 02-04 and 06 require input. Each has a `.stdin` file:

```bash
./targets/helloworld < HelloWorld.stdin
```

## Regression Test Suite

The test harness is [Runner.fsx](../tests/regression/Runner.fsx).

```bash
cd tests/regression

# Full suite
dotnet fsi Runner.fsx

# Specific samples
dotnet fsi Runner.fsx -- --sample 11_Closures --sample 14_Lazy
```

Test definitions are in [Manifest.toml](../tests/regression/Manifest.toml). The runner reports compilation and execution status for each sample.

## Directory Structure

```
samples/
├── console/
│   ├── FidelityHelloWorld/     # Console progression and feature cases
│   ├── TimeLoop/               # Platform time operations
│   └── SignalTest/             # Reactive signals
├── embedded/                   # ARM microcontroller targets
│   ├── common/                 # Startup and linker scripts
│   ├── stm32l5-blinky/         # NUCLEO-L552ZE-Q
│   └── stm32l5-uart/           # Serial communication
├── sbc/                        # SBC port plan; no executable sample
│   └── README.md               # AML-S905X-CC-V2 KeyStation references
├── templates/                  # Platform configurations
└── samples.json                # Sample catalog
```

## Related Documentation

- [WRENStack_Roadmap.md](../docs/WRENStack_Roadmap.md) - Architecture and milestones
- [Compiler PRDs](../docs/PRDs/README.md) - Feature design index
- [Learning to Walk](https://speakez.com/blog/learning-to-walk/) - PSG traversal via samples
- [Gaining Closure](https://speakez.com/blog/gaining-closure/) - Flat closure architecture
- [Why Lazy Is Hard](https://speakez.com/blog/why-lazy-is-hard/) - Lazy evaluation
- [Seq'ing Simplicity](https://speakez.com/blog/seqing-simplicity/) - Sequence expressions

## License

MIT License - See [LICENSE](../LICENSE) for details.
