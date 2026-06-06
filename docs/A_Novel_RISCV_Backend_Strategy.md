# A Novel RISC-V Backend for the Fidelity Framework

**SpeakEZ Technologies | Fidelity Framework**
**May 2026 | Forward-Looking Strategy & Design Note**

## 1. Purpose and Framing

This document records the strategic position of the Fidelity Framework toward
RISC-V as a compilation target, and the role the external `riscv-fs` project — an
executable RISC-V ISA specification written in F# — plays in reaching that
position. It is the durable, externalized form of an extended design analysis:
an adversarial assessment of `riscv-fs` itself, a reading of Composer's existing
non-LLVM backends, and a synthesis of how both relate to the framework's
hardware-software co-design thesis.

It is deliberately a **strategy and design note, not an implementation
specification.** There is no Composer RISC-V backend today, and none is scheduled
here. The claim is narrower and more durable: `riscv-fs` is a *pointer to a future
state*. Its first and nearest value is **conceptual reach** — a faithful,
executable model of RISC-V mechanics lets the front and middle ends of the
compiler learn to reason in RISC-V terms long before any backend exists. Its
later value is as a **reference** for how a dedicated backend would turn
fully-lowered MLIR into native code for hardware that uses RISC-V in a targeted,
orchestration capacity.

The through-line, stated once so the rest can be read against it:

> The front end already computes the hardware intent — layouts, working sets,
> access patterns, parallelism strategy — as semantic knowledge (§2). The only
> question any backend answers is whether that intent survives to the silicon.
> LLVM is a permanent fixture wherever it adequately addresses a triple, and is
> displaced only on **material advantage** (§5). That advantage is concentrated
> on substrates whose on-chip memory is **explicitly software-managed**, where
> the data movement *is* the program and LLVM's transparent-cache model is
> categorically inapplicable (§6–§7). On those substrates a hand-rolled backend
> that elides front-end coeffects directly to instructions is the only coherent
> path — and `riscv-fs` supplies the faithful instruction vocabulary and the
> differential oracle to do it (§3), composing from the non-LLVM backend pattern
> Composer already runs in production for the AMD NPU (§4).

The framing is intended to be even-handed throughout: where a capability is
shipped it is described as shipped, and where it is design intent it is marked as
such. The RISC-V vehicle and the co-design frame are inseparable, which is why
the document opens with the predicate before it reaches the backend.

## 2. The Co-Design Predicate

### 2.1 The Inversion

Most compiler backends are predicated on a quiet assumption: that the compiler — and the developer above it — is unaware of, or uninterested in, the direct expression of computation on hardware. A program is lowered to a generic intermediate representation, handed to a general-purpose backend, and that backend re-derives whatever hardware-relevant facts it can recover from the generic form. Cache behavior, alignment, parallelism strategy, false sharing, working-set residence — none of these are *stated*; they are *guessed at*, downstream, from an IR that has already thrown the relevant knowledge away.

The Fidelity Framework inverts this. **Hardware-software co-design is the predicate, not an afterthought, and the compiler supplies the hardware opportunities at the front end.** The front and middle ends do not produce a generic artifact and hope a backend recovers intent. They compute the hardware-relevant facts as *semantic knowledge* — working sets, access strides, alignment, isolation guarantees, parallelism strategy — and they hold that knowledge as first-class data attached to the program graph.

Two pieces of existing front-and-middle-end machinery already embody this predicate: **deterministic layout** (BAREWire) and **coeffect analysis**. Neither is a backend. Both produce hardware knowledge *before* any target is chosen. This is what makes the co-design predicate real rather than aspirational, and it is the foundation on which the backend thesis (§5) rests.

### 2.2 Deterministic Layout: From Speculation to Calculation

Traditional systems face a fundamental obstacle to any compile-time hardware reasoning: **memory layout uncertainty.** A managed runtime reorders fields for packing, inserts metadata headers of varying size, relocates objects during collection. Even C++ offers limited guarantees about where an allocator places an object. One cannot predict cache-line usage without knowing where data resides — so cache analysis at compile time is effectively impossible.

BAREWire eliminates this uncertainty. Every field offset, structure size, and alignment requirement is statically known and guaranteed:

```fsharp
[<BAREStruct>]
type OrderBook = {
    [<BAREField(0, Offset=0)>]   BidCount: uint32      // Bytes 0-3, always
    [<BAREField(1, Offset=4)>]   AskCount: uint32      // Bytes 4-7, guaranteed
    [<BAREField(2, Offset=8)>]   Bids: BAREArray<Order> // Starting byte 8, predictable
    [<BAREField(3, Offset=256)>] Asks: BAREArray<Order> // Cache line aligned
}
```

This determinism "transforms cache analysis from speculation to calculation." When the compiler encounters an actor processing `OrderBook` messages, it knows *precisely* which cache lines each field access will touch — not as an estimate, but as a fact derived from the layout.

The worked example that makes this concrete is the market-tick analysis. Given:

```fsharp
type MarketTick = {
    [<BAREField(0)>] Symbol: uint64     // Bytes 0-7
    [<BAREField(1)>] Price: float32     // Bytes 8-11
    [<BAREField(2)>] Volume: uint32     // Bytes 12-15
    [<BAREField(3)>] Timestamp: uint64  // Bytes 16-23
}

let processTickBatch (ticks: BAREArray<MarketTick>) =
    for tick in ticks do
        updatePrice tick.Symbol tick.Price  // Compiler knows: touches bytes 0-11
        if tick.Volume > threshold then      // Adds bytes 12-15 to working set
            recordHighVolume tick            // Full structure access
```

the semantic graph captures the access shape exactly: the common path touches **12 bytes per tick** (a partial cache line), the high-volume path touches all **24 bytes**. From this, three categories of knowledge become *exact* rather than estimated:

- **Working-set size becomes a calculation.** Processing 1000 ticks requires exactly 12 KB on the common path — fitting within L1 — versus 24 KB for high-volume batches, suggesting an L2 strategy. No profiling required; the number is computed.
- **Access-pattern classification leverages known offsets.** Iterating a `BAREArray<MarketTick>` has an exact stride of 24 bytes; the compiler can decide whether that is cache-friendly sequential access or a cache-line-splitting hazard.
- **Temporal locality is tracked per field.** Because field positions are guaranteed, the compiler can prefetch the frequently accessed `Symbol` and `Price` while leaving `Volume` and `Timestamp` to demand-loading.

The same determinism resolves architecture-specific facts from the target triple rather than from manual annotation. The `[<CacheLineAligned>]` attribute pads to the line size — and the line size itself is resolved from the triple:

```fsharp
let getCacheLineSize (target: TargetTriple) =
    match target.Architecture, target.Vendor with
    | "aarch64", "apple" -> 128   // Apple Silicon
    | "aarch64", _       -> 64    // Other ARM64
    | "x86_64", _        -> 64    // x86-64
    | "arm", _           -> 32    // ARM Cortex-M (embedded)
    | _                  -> 64    // Default
```

So `[<CacheLineAligned>] type WorkerCounter = { mutable value: int64 }` pads to 64 bytes on x86-64 and 128 bytes on Apple Silicon — *correctly*, without the developer adjusting anything. (The cache-aware-layouts specification, BAREWire peer repo, marks this triple-driven line-size resolution, the `[<CacheLineAligned>]` attribute, false-sharing analysis, and arena cache-line alignment all as **Planned** — see §2.7.)

Field-level offset control lets the compiler segregate cache lines surgically, placing read-shared data, isolated writes, and private buffers on distinct lines by construction:

```fsharp
[<BAREStruct>]
type ActorState = {
    [<BAREField(0, Offset=0)>]   ReadOnlyConfig: Config     // Shared read, cache line 0
    [<BAREField(1, Offset=64)>]  MutableCounter: uint64     // Isolated write, cache line 1
    [<BAREField(2, Offset=128)>] LocalBuffer: BAREArray      // Private data, cache line 2+
}
```

**Arena coloring**, inspired by page coloring, goes one step further: because layouts are predictable, the placement engine can compute *which cache sets* each structure maps to and deliberately position arenas at offsets that avoid conflict misses between frequently accessed actors. This is design intent (Prospero, §2.7), not shipped — but it is *enabled* by the determinism, which is the load-bearing point.

### 2.3 Structural Elimination of False Sharing

The most consequential property of this model is not an optimization but an *elimination*. Per-actor arenas remove false sharing **by construction, not by discipline.** When each actor's mutable state lives in its own arena, no other actor can address that memory; the problem of independent data sharing a cache line *cannot arise across actor boundaries.* This follows from Fidelity's capability-based ownership: each actor holds exclusive capability to its arena, messages transfer ownership through typed channels, and the sender relinquishes access on send.

The contrast with the prevailing systems languages is the entire argument, and it is worth stating precisely:

| Approach | Mechanism | Failure mode |
|---|---|---|
| **C++** | `alignas(64)`, `std::atomic`, `std::hardware_destructive_interference_size` | Nothing in the type system catches a *missing* `alignas`. Discoverable only by profiling after the fact. |
| **Rust** | `#[repr(align(64))]`, `CachePadded<T>` | Compiler cannot detect the *absence* of padding; the borrow checker reasons about lifetimes, not cache behavior. Left to programmer discipline. |
| **Fidelity** | Per-actor arenas under capability ownership | Actors that do not share memory *cannot* share cache lines. Isolation is a structural consequence of the ownership rules. |

Rust's borrow checker cannot reason about false sharing at all; it tracks lifetimes through lexical scope, not cache lines. C++ leaves the developer to "remember to pad, remember to align, remember to profile, and remember to fix what profiling reveals." Fidelity makes cache isolation a *type-system consequence*: "the right abstractions eliminate entire categories of bugs rather than detecting them."

Crucially, the claim is **verifiable**. The relevant hardware counter is **HITM (Hit Modified)** — fired when a core reads a cache line another core has modified. On Linux, `perf c2c` exposes it directly:

```bash
perf c2c record -o perf.data ./my_program   # record cache-to-cache events
perf c2c report --stdio                       # map contention back to source
```

When Fidelity claims two actors have isolated arenas, `perf c2c` should report **zero HITM events** between them. A true report substantiates the compile-time guarantee; a violation indicates either a compiler bug or incorrect usage. This is the verification loop that distinguishes the co-design predicate from a marketing claim: *the compiler produces a falsifiable guarantee, and a standard tool confirms it.*

### 2.4 Coeffect Analysis: Hardware Intent as Data Beside the Graph

Where BAREWire supplies *spatial* knowledge (where data lives), coeffect analysis supplies *strategic* knowledge (what a computation needs from its environment, and therefore how it should be parallelized and placed). Coeffects are the dual of effects: effects describe what a computation does *to* its environment; coeffects describe what it requires *from* it.

The framing case is the choice between two compilation strategies that determine what optimizations are even possible:

- **Interaction nets** — for pure computation with no sequential dependencies or external effects. Computation is a graph of local rewrite rules; everything that can happen at once does. Maps naturally to GPU thread blocks, systolic arrays, dataflow fabrics.
- **Delimited continuations** — for computation where order matters: external resources, temporal state, controlled concurrency. Captures "the rest of the computation" at well-defined points.

The selection is driven by inferred context requirements:

```fsharp
type ContextRequirement =
    | Pure                           // No external dependencies → Interaction nets
    | MemoryAccess of AccessPattern  // Data access pattern → Guides parallelization
    | ResourceAccess of Resource Set // External resources → Delimited continuations
    | Temporal of HistoryDepth       // Needs past values → Streaming architecture
```

The architecturally decisive property is *where this knowledge lives*. Coeffects are **not** woven into the AST; they are held in **external analysis maps keyed by node id, beside the graph:**

```fsharp
// The graph node stays clean — structure only
type PSGNode = {
    Id: NodeId
    Kind: PSGNodeKind
    Symbol: FSharpSymbol option
    SourceRange: range
}

// Coeffects live in external maps
type CoeffectAnalysis = {
    ComputationPatterns:  Map<NodeId, ComputationPattern>
    MemoryAccessPatterns: Map<NodeId, AccessPattern>
    ResourceRequirements: Map<NodeId, Set<Resource>>
    TemporalDependencies: Map<NodeId, HistoryDepth>
}

// Strategy selection consults the maps, not the node
let selectBackend (node: PSGNode) (coeffects: CoeffectAnalysis) =
    match Map.tryFind node.Id coeffects.ComputationPatterns with
    | Some PureDataParallel     -> InteractionNetBackend
    | Some ResourceDependent    -> DelimitedContinuationBackend
    | Some StreamingComputation -> PipelineParallelBackend
    | _                         -> DefaultSequentialBackend
```

This "maps beside the graph" shape is **real in the Composer codebase today**, not aspirational. `PSGElaboration.Coeffects` and its siblings ([src/MiddleEnd/PSGElaboration/Coeffects.fs](../src/MiddleEnd/PSGElaboration/Coeffects.fs), [CoeffectValidation.fs](../src/MiddleEnd/PSGElaboration/CoeffectValidation.fs)) compute exactly such `Map<NodeId, _>` / `Set<NodeId>` structures — for example `AddressedMutableBindings: Set<NodeId>` and `ModifiedVarsInLoopBodies` — and they are serialized for inspection as `06_coeffects.json` in every sample's intermediates. A fragment from `10a_ImperativeControl`:

```json
{
  "mutability": {
    "AddressedMutableBindings": [706, 708, 765, 767, 769, 862, ...],
    "ModifiedVarsInLoopBodies": [
      { "Item1": 854, "Item2": ["intPos", "isDone", "iv"] },
      { "Item1": 900, "Item2": ["ci", "pos"] }
    ]
  }
}
```

The conceptual `CoeffectAnalysis` of the design note and the shipped `PSGElaboration.Coeffects` are the same idea at two altitudes: *metadata about the graph, computed before lowering, never mutating the graph.* (See [Coeffect_Analysis_Architecture.md](./Coeffect_Analysis_Architecture.md): "Coeffect analysis never modifies the PSG. It only computes metadata.")

### 2.5 Reconfiguration as the Reward: Per-Layer and Hybrid Placement

The payoff of holding strategy as data is that **the strategy can differ per node**, which is precisely what heterogeneous and reconfigurable hardware demands. The canonical illustration is a neural-net forward pass on a reconfigurable substrate (NextSilicon's Maverick):

```fsharp
let rec forward (input: Tensor) (layers: Layer list) =
    match layers with
    | [] -> input
    | Linear weights :: rest ->
        let output = input |> matmul weights |> activate     // Coeffect: matrix-multiply
        forward output rest
    | Attention heads :: rest ->
        let output = multiHeadAttention input heads          // Coeffect: all-to-all
        forward output rest
    | Convolution kernel :: rest ->
        let output = convolveSpatial input kernel            // Coeffect: local spatial
        forward output rest
```

Coeffect analysis classifies each layer type independently, and the substrate reconfigures *between* layers:

- `Linear` → pure matrix ops → configure as a **systolic array**
- `Attention` → all-to-all communication → configure as a **crossbar**
- `Convolution` → local spatial pattern → configure as a **2D mesh**

The same machinery drives a **hybrid CPU/GPU split** for a BitNet + compressed-KV model, where coeffects discover the natural device boundary: BitNet attention is *add/subtract only, sequential streaming, low arithmetic intensity* → CPU with AVX-512; the compressed KV cache is *parallel decompression, random access, high parallelism, memory-intensive* → GPU with warp-level primitives — with BAREWire's unified memory eliminating the CPU↔GPU transfer entirely (`AccessPattern = CPUWrite_GPURead`). Different layers have vastly different computational characteristics; coeffects let the compiler put the right computation on the right processor instead of running a monolithic workload on one device.

Notably, the *same* analysis adapts per architecture. On GPU the relevant pathologies are entirely different from CPU — there is **no false sharing** under SIMT (a warp of 32 threads executes the same instruction in lockstep and cannot concurrently write the same location), so the analysis instead targets memory **coalescing**, shared-memory bank conflicts, warp divergence, and occupancy. The principle transfers ("compile-time analysis of access patterns produces runtime performance"); the specific patterns differ entirely. This is the co-design predicate generalizing: the front end computes the *target-relevant* facts, whatever the target is.

### 2.6 The Critical Observation: A Lossy Handoff

Here is the observation that motivates the entire backend thesis, and it must be stated plainly because it is a defect in the *current* design, not a strength.

The 2025-era CPU cache post does all of the rich front-end reasoning described above — working sets, strides, prefetch distances, non-temporal candidacy, cache-management instruction selection — and then **hands portable MLIR to `mlir-opt` and the LLVM backend to act on it.** Composer is described as emitting only generic `memref` / `arith` / `func`:

```mlir
// Composer emits portable MLIR
%data  = memref.alloca() : memref<1024xf32>
%c0    = arith.constant 0 : index
%index = arith.addi %c0, %offset : index
%value = memref.load %data[%index] : memref<1024xf32>
```

and the architecture-specific decisions are *delegated downstream*: "For Intel targets with aggressive hardware prefetchers, `mlir-opt` can introduce prefetch hints during lowering." Non-temporal moves (`movnti`, `movntq`), cache-management ops (`clzero` on AMD, `cldemote` on Intel), prefetch-distance calibration — all of these are described as things the **backend selects**, guided by Composer's processor-knowledge repository (the "Library of Alexandria") that *informs `mlir-opt`* rather than emitting the decision directly.

This is a **lossy handoff.** The front end computed that a given store streams once and should bypass cache; it computed the exact stride and the prefetch distance; it computed that two arenas are isolated. But the MLIR that crosses into LLVM carries *none of this* — it is generic `memref`/`arith`, indistinguishable from IR produced by a compiler that knew nothing. LLVM never sees the coeffects. It sees only the generic IR and must *re-derive*, through its own heuristics and opinions, facts the front end already held as certainties. The "Library of Alexandria guides `mlir-opt`" framing is exactly the symptom: the knowledge is used to *nudge an optimizer's guesses* rather than to *determine the emitted form*.

This contradicts the elision principle already established for Composer's own code generation: **"PSG → MLIR is the residue of observation, not active emission."** The fix is to extend that principle one stage further:

> **Coeffect → machine instruction is also elision.**

Hardware intent computed at the front end should be *carried* to the emitted form as a residue of the knowledge already in hand — not surrendered to a downstream optimizer that re-derives it through guesses. A store known to be non-temporal should elide *to* the non-temporal instruction because the coeffect says so, not because `mlir-opt` independently rediscovered the streaming pattern. This is the gap a dedicated backend exists to close (§5), and it is the precise sense in which a generic LLVM handoff becomes a *liability* rather than a convenience for the narrow class of targets where the front-end knowledge is the whole point.

### 2.7 What Is Real Versus What Is Envisioned

Intellectual honesty requires a clean line between shipped mechanism and design intent, because the co-design predicate is credible *only* on the strength of what already exists.

**Real today:**

- **The PSG** as the single semantic source of truth, with coeffect analysis running after enrichment and before lowering ([Coeffect_Analysis_Architecture.md](./Coeffect_Analysis_Architecture.md)).
- **Coeffects as maps beside the graph** — `Map<NodeId, _>` / `Set<NodeId>` structures in [src/MiddleEnd/PSGElaboration/Coeffects.fs](../src/MiddleEnd/PSGElaboration/Coeffects.fs) and siblings, computed without mutating the graph, serialized as `06_coeffects.json`.
- **BAREWire layout determinism** — guaranteed offsets, sizes, alignment; the property from which all compile-time cache reasoning derives.
- **The elision principle** for PSG → MLIR, already operative in Alex's Witness/Pattern model.

**Envisioned / roadmap (stated as design intent in the source posts, and largely marked *Planned* in the BAREWire spec):**

- **Prospero runtime placement** — L3-domain grouping, hyperthread-aware placement, cache-warming-cost-weighted migration. Described throughout the CPU post in the future tense ("we envision," "our planned implementation").
- **Arena coloring** to avoid conflict misses — enabled by determinism, not yet built.
- **Performance-counter-driven adaptation** — `adaptCacheStrategy` reacting to HITM / L1-L2 hit rates at migration points; the verification *loop* (`perf c2c`) is real tooling, but the *adaptive feedback* is design intent.
- **Triple-driven cache-line resolution, `[<CacheLineAligned>]`, false-sharing analysis, arena cache-line alignment** — all listed as **Planned** in the BAREWire *Cache-Aware Layouts* specification (peer repo).
- **GPU coalescing transforms, shared-memory staging, warp-shuffle reductions, automatic fence insertion** — described as "Composer would" / "our vision," not as shipped passes.

The distinction matters for the thesis that follows. The argument for a dedicated backend (§5) does **not** rest on the envisioned runtime; it rests on what is already computed at the front end — deterministic layout and coeffects-as-data — and on the single structural defect of surrendering that knowledge to a generic IR. The predicate is real; the lossy handoff is real; closing it is the work.

## 3. What riscv-fs Is — Provenance, Fidelity, and Structure

Before any strategy can rest on `riscv-fs`, one question has to be answered without hedging: is this a serious artifact, or is it the kind of plausible-looking machine-generated code that compiles, passes a smoke test, and silently hallucinates the load-bearing bit math? We ran a deep adversarial assessment — fifteen verification passes, with empirical `dotnet fsi` checks against the actual code rather than inspection alone — and the answer is settled. `riscv-fs` is the real thing: a hand-written, multi-year, executable RISC-V ISA specification in F#, in the Bluespec/Forvis functional-spec lineage. It is decisively **not AI slop**. But its value to Composer is bounded sharply by *what it fundamentally is*, and that boundary is the most important thing this section establishes.

### Provenance: settling the slop question

The upstream is [github.com/mrLSD/riscv-fs](https://github.com/mrLSD/riscv-fs), a single-author project by Evgeny Ukhanov (mrLSD). The provenance signature is everything generative slop is not:

- **191 commits spanning 2019-06 to 2026-05** — a genuine multi-year maintenance arc (heavy development through 2020, deliberate upkeep touches in 2023/2024/2026), not a one-shot dump.
- **MIT licensed** (Copyright 2019), with semantic version tags `v0.1.0` through `v0.4.1`.
- **18 numbered GitHub PRs with paired implementation+test PRs** (e.g. PR #8 feature / PR #9 tests for an extension), feature branches (`rv64i`, `rv-extention-M`, `rv-extension-A`, `feat/update-dotnet10`, `fix/rv32-shift-mask`) each merged via real PR hygiene.
- **Real, current CI**: GitHub Actions (`actions/checkout@v6`, `setup-dotnet@v5`, `10.0.x`) running restore/build/test on push and PR to master.
- An **active, substantive audit branch** `origin/auido-2026-05-31` (+1038/−150) adding LR/SC reservation semantics, 32-bit AMO.W width handling, misaligned-atomic traps, full PT_LOAD ELF segment loading, RV32 DIV/REM overflow checks, and a 100%-line-coverage push — ongoing rigor, not abandonment.

A factual correction to a premise we have been carrying: **the `houstonhaynes` fork is byte-for-byte identical to upstream master** (`git rev-list --left-right --count` = 0/0, clean working tree, zero divergent commits). The .NET 10 / FSharp.Core 10.0.107 bump (commit `1929976`, PR #17) and the RV32 shift-mask conformance fix (`f9dfe06`, PR #18) are **both authored and committed upstream** by the original maintainer (`evgeny@aurora.dev`), merged on mrLSD. The fork carries no independent .NET 10 work — that framing is simply wrong; the work originated upstream.

The strongest reason this is the *opposite* of slop is internal, not circumstantial: **the hard parts are correct and tested, and the one real defect cluster is a human regression, not a generative failure.** Generative slop hallucinates plausible-but-wrong masks while getting the boilerplate right. Here the inverse holds — the genuinely tricky bit math (B/J immediate scatter-pack, MULH signedness, the 64-bit high-multiply negation correction, DIV-by-zero and `INT_MIN`/−1 overflow special cases) is all *right* and empirically verified, while the single broken area (RV64A in `DecodeA64.fs`) is a classic copy-paste regression in an *untested sibling file* — exactly consistent with the README openly admitting A-extension tests do not exist. A code generator does not produce that bug while nailing the much harder RV32A path beside it.

### Fidelity gradient by extension

`riscv-fs` is high-fidelity on what it covers, with a sharp quality gradient. The grades below come from the adversarial passes, each confirmed empirically.

| Slice | Grade | Verdict |
|-------|-------|---------|
| RV32I encode/decode | **A** | Real and faithful |
| RV64I `*W` semantics | **A** | Real and faithful |
| Bits/state foundation | **A** | Real and faithful (two latent engineering hazards) |
| M extension | **B** | Real but two genuine spec-divergence bugs |
| A extension | **D** | RV32A sound; RV64A non-functional |

**RV32I (A) — the immediates are right.** The fiddly part of RISC-V decode is the scattered B-type and J-type immediates, where the spec scrambles immediate bits across non-contiguous instruction-word positions. `riscv-fs` reassembles them exactly. From `DecodeI.fs` lines 93–107:

```fsharp
let imm12_B =
        (   ((instr.bitSlice 31  31) <<< 12) |||
            ((instr.bitSlice 30  25) <<< 5 ) |||
            ((instr.bitSlice 11   8) <<< 1 ) |||
            ((instr.bitSlice  7   7) <<< 11)
        ).signExtend 13
let imm20_J =
        (   ((instr.bitSlice  31  31) <<< 20) |||
            ((instr.bitSlice  30  21) <<<  1) |||
            ((instr.bitSlice  20  20) <<< 11) |||
            ((instr.bitSlice  19  12) <<< 12)
        ).signExtend 21
```

This matches the RISC-V manual's immediate layout bit-for-bit. Opcode and funct constants are likewise exact (`0b0110111` LUI, `0b0010111` AUIPC, `0b1100111` JALR, at `DecodeI.fs:116–120`).

**RV64I (A) — the `*W` ops sign-extend correctly.** The RV64 word instructions must compute in 32 bits then sign-extend the low 32 bits back to 64, and `LWU` must zero-extend. Both are right. From `ExecuteI64.fs`: `ADDIW` (line 42) computes `int32(...) + int32 imm12` then writes `int64 rdVal`; `LWU` (lines 10–18) takes `uint32 memResult.Value` then `int64 memVal` (zero-extend); `LD` (line 22) sign-extends the doubleword load. Every `*W` op is consistently `int64(int32 ...)` — compute narrow, sign-extend wide.

**Bits/state foundation (A) — verified empirically, not by inspection.** The primitive layer in `Bits.fs` and `MachineState.fs` is faithful on every load-bearing detail:

- **x0 hardwired on both read and write**, so no instruction needs a special case. `MachineState.fs:21–32`: `getRegister` returns `0L` for `reg = 0`; `setRegister` coerces `value` to `0L` when `reg = 0`.
- **Sign extension with no off-by-one.** `Bits.fs:31–33`: `signExtend n = (x <<< (32-n)) >>> (32-n)` — shift-left-then-arithmetic-shift-right on a signed `Int32`, the canonical idiom.
- **SRL-vs-SRA logical/arithmetic distinction** and **correct `0x1f`/`0x3f` shift masking by XLEN.** In `ExecuteI.fs`: `SRL` (lines 332–340) masks `0x1f` on RV32 / `0x3f` on RV64 and casts through `uint32`/`uint64` for *logical* fill; `SRA` (lines 349–357) uses the same masking but a *signed* `>>>` for *arithmetic* (sign-bit) fill. These are the exact spots where a from-scratch backend silently diverges, and they are correct here.

**M extension (B) — hard parts right, two real bugs.** The high-multiply variants are correct and BigInteger-checked, including the negation-correction trick for negative 64×64 products. From `ExecuteM.fs:23–29`:

```fsharp
let mulh (x : int64, y : int64) : int64 =
    let neg = (x < 0L) <> (y < 0L)
    let x1 = uint64(if x < 0L then -x else x)
    let y1 = uint64(if y < 0L then -y else y)
    let res = mulhu(x1, y1)
    let resd = if (x * y) = 0L then 1L else 0L
    if neg then int64(~~~res) + resd else int64 res
```

That `~~~res + resd` is the subtle two's-complement high-half correction — exactly what slop gets wrong, and it is right. `MULHSU` (lines 31–36) handles the mixed signed/unsigned case correctly.

But there are **two genuine spec divergences** in `ExecuteM64.fs`: `DIVUW` (lines 33–43) and `REMUW` (lines 63–72) divide the **full 64-bit register contents** instead of truncating operands to the low 32 bits as the spec mandates:

```fsharp
let execDIVUW (rd : Register) (rs1 : Register) (rs2 : Register) (mstate : MachineState) =
    let rs1Val = mstate.getRegister rs1          // full 64-bit value, NOT uint32-truncated
    let rs2Val = mstate.getRegister rs2
    ...
        int32(uint64 rs1Val / uint64 rs2Val)      // should be uint32 rs1Val / uint32 rs2Val
```

These diverge from hardware whenever the upper register bits are nonzero. The signed word ops (`MULW`/`DIVW`/`REMW`) correctly truncate-then-sign-extend; only the unsigned word divides are wrong.

**A extension (D) — RV32A sound, RV64A non-functional.** `DecodeA.fs` (RV32A) is faithful: funct7 case literals carry the `0b` prefix and bind the real `rs2` (lines 43–53). `DecodeA64.fs` (RV64A) is broken by **two stacked copy-paste defects**:

1. **Decimal literals where binary was intended.** The funct7 match arms lost their `0b` prefix (`DecodeA64.fs:43–53`): `| 00010 ->`, `| 00011 ->`, etc. These are read as *decimal* 10, 11, 1, 0, 100, 1100… so 9 of 11 double-word atomics can never match the 5-bit field. Compare the *correct* RV32A sibling — `DecodeA.fs:43` reads `| 0b00010 ->`.
2. **Dropped `rs2` operand.** Every AMO/SC case binds `rs2 = rs1` (`DecodeA64.fs:44–53`: `{| ... rs2 = rs1 ... |}`), copying `rs1` instead of decoding the real `rs2` field. The RV32A sibling correctly uses `rs2 = rs2` (`DecodeA.fs:44`).

On top of that, the **LR/SC model tracks no reservation set** — `SC` always succeeds — and the `aq`/`rl` ordering bits are decoded then ignored. That is an acceptable simplification for a single-hart sequential simulator, but it is *not* faithful for any concurrency or memory-ordering claim.

**Net trust boundary:** treat `riscv-fs` as a trustworthy reference for the integer base (I, and M minus `DIVUW`/`REMUW`). Do **not** trust the RV64A path, atomicity, or memory-ordering.

### Two engineering hazards in the foundation

Both are latent — currently masked, not currently wrong — but worth flagging because they are *deceptive*, not obvious.

**Hazard 1: `setRegister` mutates a shared array behind a record-copy disguise.** `MachineState.fs:27–32`:

```fsharp
member x.setRegister (reg: Register) (value: MachineInt) : MachineState =
    let registers = x.Registers
    let value = if reg = 0 then 0L else value
    Array.set registers reg (x.alignByArch value)   // in-place mutation of the SHARED array
    { x with Registers = registers }                 // looks pure; aliases the same array
```

The `{ x with Registers = registers }` *looks* like value-semantics state threading, but `registers` is the same `.NET` array object — `Array.set` mutates it in place. Value semantics are an illusion; any code that retains a prior `MachineState` and expects it unchanged would observe corruption. It is masked today only because the run loop discards old state each step.

**Hazard 2: `bitSlice` degenerates to 0 for a full-width slice.** `Bits.fs:4–5`:

```fsharp
member x.bitSlice endBit startBit =
    (x >>> startBit) &&& ~~~(-1L <<< (endBit - startBit + 1))
```

For a full 64-bit slice (`endBit - startBit + 1 = 64`), `.NET` masks the shift count modulo 64, so `-1L <<< 64` becomes `-1L <<< 0 = -1L`, `~~~(-1L) = 0L`, and the whole slice returns `0`. Untriggered in current use (no caller requests a 64-wide slice) but a real trap for anyone reusing the primitive.

### Structure

The architecture is a textbook fetch-decode-execute interpreter, decomposed by extension — the canonical Forvis/Bluespec functional-spec shape transliterated into deliberately elementary F#:

- **Decode\*/Execute\* per extension.** Six decode files (`DecodeI`, `DecodeI64`, `DecodeM`, `DecodeM64`, `DecodeA`, `DecodeA64`) and mirrored execute files. Decode is separated from execute throughout.
- **Decoded instructions as DUs with anonymous-record payloads.** Each extension defines a discriminated union (`InstructionI`, `InstructionM`, …) ending in a `None` "not found" case, with cases carrying F# anonymous records — `{| rd: Register; rs1: Register; imm12: InstrField |}` (`DecodeI.fs:14`) — giving named-field clarity without a proliferation of nominal types.
- **Immutable `MachineState` record** (`MachineState.fs:13–20`): `PC: MachineInt`; `Registers: RegisterVal array`; `Memory: Map<int64, byte>` (sparse, byte-addressable, little-endian via `combineBytes` in `Bits.fs:35–37`); `Arch`; `RunState`. `RunMachineState` is a DU (`NotRun | Run | Stopped | Trap of TrapErrors`) — errors are *values*, not exceptions, making the executor referentially transparent.
- **Type abbreviations as a thin semantic layer** (`Arch.fs:4–10`): `MachineInt = int64` is the single universal word; `Register = int32`; `InstrField = int32` (the raw instruction word and immediate fields). RV32 is *narrowed dynamically* via `alignByArch`/`alignByArchUnsign` (`MachineState.fs:74–86`: `int64(int32 v)` for signed, `int64(uint32 v)` for unsigned), not by a separate type — so RV32 correctness does not depend on each op masking.
- **Bit primitives via `.NET` type augmentation** on `System.Int64` and `System.Int32` (`Bits.fs:3,28`) — `bitSlice`, `isSet`, `signExtend`. An F#-specific idiom.
- **`Decoder.fs` is speculative-decode-then-gate.** `Decoder.fs:14–63` runs all six per-extension decoders, gates each result against the active `mstate.Arch` (testing `decX <> Ext.None` structurally), and returns `execFunc option = (MachineState -> MachineState) option` — a *partially-applied executor closure*, not the decoded value. Currying is the dispatch mechanism.

### The category point

This is the single most important framing in the entire assessment, and the strategy hinges on it:

> **`riscv-fs` is an ISA-semantics model and decoder — a fetch-decode-execute interpreter — that contains bit-knowledge *only*.**

It models instruction encodings, field-slice formulas, per-extension dispatch tables, and per-instruction state-transformer semantics. It contains **none of**: instruction selection, register allocation, ABI / calling-convention lowering, scheduling, or relocations / linking. Those are absent not by oversight but by *category*: an interpreter reads bits that are *already* a chosen instruction over concrete `x0`–`x31` operands, so it never needs to *choose* instructions, *allocate* registers, *lower* a calling convention, *schedule*, or *link*. Those five absent pieces are the structural reasoning a code generator performs — but in Fidelity that reasoning is not a monolithic backend to be built; it is extraction carried out upstream (§5), which is exactly what leaves the backend itself thin.

What `riscv-fs` gives Composer, therefore, is **verified integer-ISA knowledge** — the vocabulary the upstream extraction elides *into* — and the shape of a **lightweight in-process oracle**. What it does not supply is the structural reasoning; but that reasoning lives upstream, in the PSG's deterministic enumeration and the middle-end's coeffect passes (§5), not in a monolithic backend riscv-fs would be expected to seed. The distinction this document holds is between the *thin emission tail* and the *upstream extraction* — not between an "easy 20%" and a "hard 80%" of one backend.

### Orthogonal to the lowering, not a pathway through it

One corollary of the category point is worth stating outright, because it is the most natural place to misread `riscv-fs`'s relevance. `riscv-fs` is itself a **.NET console application** — `Program.fs` is an ordinary `[<EntryPoint>]`, and `Run.fs` loads a RISC-V ELF into a `Map<int64, byte>` via `ELFSharp` and steps a fetch-decode-execute machine (both in the `riscv-fs` peer repo). RISC-V is therefore its **guest** — the input it reads and *interprets* — never its **target**. Its own build path (`dotnet build`, JIT or Native AOT) compiles the *interpreter* for whatever **host** it runs on (x86-64, arm64); it never emits a single RISC-V instruction. The `.fsproj` configures no `RuntimeIdentifier` and no `PublishAot`, but the point holds even if it did: AOT-publishing the simulator to `linux-riscv64` would only make the *emulator* run natively on RISC-V hardware — RISC-V as host, still not as target. No AOT trick turns an interpreter into a code generator.

It follows that **`riscv-fs` supplies no pathway to a RISC-V target, and is not meant to.** Its contribution is **orthogonal** to the lowering problem: everything Composer draws from it is *extracted from the scaffold* and applied elsewhere — the opcode/funct constants and bit-field formulas are read off the source as static data and *inverted* into Composer's own encoder, and the executable semantics are run host-side as a differential oracle that *grades* Composer's output. Both serve the real work — *abstracting a proper lowering to a real target* (§5–§7) — but neither **is** that lowering, and neither shortens the path to it. The RISC-V that Fidelity ultimately emits comes from the hand-rolled Composer backend; `riscv-fs` informs and verifies that backend without ever standing on the path between MLIR and the silicon. When this document says `riscv-fs` is a "pointer to a future state," it means a *source of extractable knowledge and a checker* — emphatically not a route the build can travel.

## 4. How Composer Already Compiles MLIR to Novel Hardware — the CIRCT and AIE Backends

The RISC-V thesis does not propose a new compiler architecture. It proposes a new *backend* slotted into a contract that already carries two non-trivial, non-LLVM hardware targets to silicon. This section documents that standing art at the code level: the backend contract, the routing that selects a backend with zero dispatch, and the two existing backends — CIRCT (FPGA) and AIE (NPU) — that compile MLIR text into hardware artifacts. Both are *thin orchestrators of external dialect tools*, never self-contained codegen. That fact is the precedent the RISC-V thesis must consciously decide whether to honor or break.

### The Backend Contract: MLIR text in, artifact out

A backend in Composer is not a class hierarchy, a plugin registry, or a visitor. It is a two-field record — a name and a function value — defined in [Pipeline.fs](../src/Core/Types/Pipeline.fs):

```fsharp
// src/Core/Types/Pipeline.fs:35-40
type BackEnd = {
    /// Human-readable name for logging
    Name: string
    /// Compile MLIR text to target artifact
    Compile: string -> BackEndContext -> Result<BackEndArtifact, string>
}
```

The signature is the whole contract. `Compile` takes `string` (MLIR text — *already-elided* dialect IR, not an in-memory AST) and a `BackEndContext`, and returns `Result<BackEndArtifact, string>`. Everything a backend needs to know about *the world* arrives in the context record ([Pipeline.fs:18-30](../src/Core/Types/Pipeline.fs)); everything it produces is one of four artifact shapes ([Pipeline.fs:9-13](../src/Core/Types/Pipeline.fs)):

```fsharp
type BackEndArtifact =
    | NativeBinary of path: string
    | Verilog of path: string
    | Xclbin of xclbinPath: string * instsPath: string
    | IntermediateOnly of format: string
```

The artifact DU is the load-bearing detail for the RISC-V discussion: the existing targets do not all emit a "binary." FPGA emits `Verilog`; NPU emits an `Xclbin` *pair* (the bitstream plus a separate DPU instruction stream). Only LLVM emits `NativeBinary`. The contract was deliberately widened beyond "produce an executable" because the hardware targets do not, themselves, terminate in an ELF.

The `BackEndContext` is the orchestrator's promise to stay backend-agnostic. It carries `OutputPath`, `IntermediatesDir`, an optional `TargetTripleOverride`, the `DeploymentMode`, an `EmitIntermediateOnly` stop-flag, and the accumulated `ExternLibraries` set — and the type comments make the agnosticism explicit: *"the orchestrator assembles this but doesn't interpret it"* and, on the triple override, *"Backend-specific: LLVM uses it, CIRCT ignores it"* ([Pipeline.fs:21-23](../src/Core/Types/Pipeline.fs)). The orchestrator hands the same context to every backend; each backend reads only the fields it cares about.

### Backend-as-value routing: one match, no dispatch

There is exactly one place in the compiler where a target platform becomes a backend, and it is configuration, not control flow. [PlatformPipeline.fs](../src/Core/PlatformPipeline.fs) is twelve lines of substance:

```fsharp
// src/Core/PlatformPipeline.fs:13-18
let resolveBackEnd (targetPlatform: TargetPlatform) : BackEnd =
    match targetPlatform with
    | FPGA -> BackEnd.CIRCT.Pipeline.backend
    | CPU | MCU | TargetPlatform.Library -> BackEnd.LLVM.Pipeline.backend
    | GPU -> { Name = "GPU"; Compile = fun _ _ -> Error "GPU backend not yet implemented." }
    | NPU -> BackEnd.AIE.Pipeline.backend
```

The module header states the architectural intent directly: *"This is the ONE place where TargetPlatform maps to a BackEnd value. It runs at pipeline assembly time, not during compilation. The orchestrator never sees this match — it receives the assembled BackEnd"* ([PlatformPipeline.fs:1-6](../src/Core/PlatformPipeline.fs)). Each arm is a *value* — `BackEnd.AIE.Pipeline.backend` is a record literal, not a call. The orchestrator receives a `BackEnd` and applies its `Compile` field; it has no `match` on platform, no registry lookup, no virtual dispatch. The GPU arm even shows the cost of being unimplemented: a one-line backend whose `Compile` returns `Error`, fully type-checked and routable, just not yet real.

**This is the slot the RISC-V backend fills.** Adding a RISC-V target is, structurally, one new arm in `resolveBackEnd` pointing at one new `backend` value of type `BackEnd`. The thesis inherits the entire contract for free; the only open question is what that backend's `Compile` *does* inside.

### The key architectural fact: existing non-LLVM backends are thin orchestrators

CIRCT and AIE prove Composer can target hardware that LLVM cannot describe. But neither of them is a code generator in the sense the RISC-V thesis contemplates. **Each emits or transforms a target MLIR dialect and then delegates the heavy lifting to external dialect tools — `circt-opt` for FPGA; `aie-opt`, `aie-translate`, Peano, `bootgen`, and `xclbinutil` for NPU. Neither backend emits a single instruction byte itself.** Both are coordinators of out-of-process toolchains, marshalling files between stages.

That is the precedent a hand-rolled RISC-V backend would *break*. A self-contained RISC-V codegen — one that emits machine code or assembly without delegating to an external assembler/linker dialect tool — would be the first backend in Composer to *generate* a binary rather than *orchestrate* one out of upstream-supplied tools. Whether that is the right move is the thesis's central design tension; the value of this section is to make the contrast concrete.

### The CIRCT backend (FPGA): the simplest non-LLVM precedent

The FPGA path is the minimal demonstration that "non-LLVM, dialect-tool-delegating" is a real, working shape. Alex elides directly to the `hw`/`comb`/`seq` dialects upstream, so by the time MLIR text reaches the backend, the hardware is already described — the backend only optimizes and exports. [CIRCT/Pipeline.fs](../src/BackEnd/CIRCT/Pipeline.fs) is a two-step `backend` value:

```fsharp
// src/BackEnd/CIRCT/Pipeline.fs:16-45 (condensed)
let backend : BackEnd = {
    Name = "CIRCT"
    Compile = fun mlirText ctx ->
        // write mlirText to output.mlir
        // Step 1: optimize hw/comb/seq
        Lowering.optimizeHW mlirPath optimizedPath
        |> Result.bind (fun () ->
            if ctx.EmitIntermediateOnly then Ok (IntermediateOnly "CIRCT hw/comb/seq optimized")
            else
                // Step 2: export to SystemVerilog
                Lowering.exportToVerilog optimizedPath svPath
                |> Result.map (fun () -> Verilog svPath)) }
```

Both steps are single `circt-opt` invocations ([CIRCT/Lowering.fs](../src/BackEnd/CIRCT/Lowering.fs)):

```fsharp
// src/BackEnd/CIRCT/Lowering.fs:44-46 — optimize
let args = sprintf "%s --map-arith-to-comb --canonicalize --cse -o %s" mlirPath optimizedPath
runTool circtOptPath args

// src/BackEnd/CIRCT/Lowering.fs:75-79 — export
let args = sprintf "%s --lower-seq-to-sv --lower-hw-to-sv --export-verilog -o /dev/null" hwPath
runToolToFile circtOptPath args svPath
```

The module header is unambiguous about the delegation boundary: *"Alex elides directly to hw/comb/seq dialects, so the backend only needs: 1. circt-opt: canonicalize + CSE; 2. circt-opt: lower-seq-to-sv + lower-hw-to-sv + export-verilog. LLVM is never involved in this path"* ([CIRCT/Pipeline.fs:1-9](../src/BackEnd/CIRCT/Pipeline.fs)). The backend itself contributes no transform logic — it shells `circt-opt` twice and names the resulting `.sv` file. `circt-opt` does every dialect lowering; the F# code is plumbing. The tool binary is resolved from the `CIRCT_OPT_PATH` env var with a bare-name PATH fallback ([CIRCT/Lowering.fs:13-16](../src/BackEnd/CIRCT/Lowering.fs)). This is the template at its thinnest: emit a dialect, call the dialect's optimizer, capture the output.

### The AIE backend (NPU): the full seven-step orchestration

The NPU path is the deep precedent — and the one most directly relevant to RISC-V, because it physically embodies the "LLVM-fork for dense compute; direct-emit for orchestration" split. [AIE/Pipeline.fs](../src/BackEnd/AIE/Pipeline.fs) is again a thin `backend` value: it writes the incoming MLIR-AIE text, derives the `.xclbin` and `_insts.bin` paths from `ctx.OutputPath`, and calls one function ([AIE/Pipeline.fs:12-36](../src/BackEnd/AIE/Pipeline.fs)):

```fsharp
// src/BackEnd/AIE/Pipeline.fs:33-35
timePhase "BackEnd.AIECompile" "Compiling MLIR-AIE to xclbin" (fun () ->
    Lowering.lowerToXclbin mlirPath xclbinPath instsPath)
|> Result.map (fun () -> Xclbin (xclbinPath, instsPath))
```

Note the artifact: `Xclbin (xclbinPath, instsPath)` — the *pair*. The bitstream and the instruction stream are co-equal outputs, and the contract carries both because the host runtime needs both files staged together (the HelloNappy README confirms this: *"Both must be in the host's working directory when the executable runs"*).

All of the real work lives in [AIE/Lowering.fs](../src/BackEnd/AIE/Lowering.fs) `lowerToXclbin` ([Lowering.fs:314-489](../src/BackEnd/AIE/Lowering.fs)). The module header summarizes the seven steps and the *deliberate toolchain choice* up front ([Lowering.fs:1-14](../src/BackEnd/AIE/Lowering.fs)), and the pipeline is documented as mirroring `aiecc.py` minus the proprietary tools: *"The pipeline mirrors aiecc.py's --no-xchesscc --no-xbridge flow but implemented as direct tool invocations"* ([Lowering.fs:309-313](../src/BackEnd/AIE/Lowering.fs)). The `--no-xchesscc --no-xbridge` flags are the point: this is the **open `llvm-aie` (Peano) fork path, not the Vitis/xchesscc proprietary path.** AIE compute kernels go through an open-source LLVM fork, exactly the model RISC-V dense compute could follow.

**Tool resolution** ([Lowering.fs:25-60](../src/BackEnd/AIE/Lowering.fs)) walks a layered search: `AIE_TOOLCHAIN` env (falling back to `~/aie-toolchain`) for the root; per-tool env overrides (`AIE_OPT_PATH`, `AIE_TRANSLATE_PATH`, `BOOTGEN_PATH`) then `bin/` then PATH; and crucially, Peano is resolved via `PEANO_INSTALL_DIR` or, by default, the pip layout `lib/python3.12/site-packages/llvm-aie/bin/` ([Lowering.fs:42-50](../src/BackEnd/AIE/Lowering.fs)). Peano's `opt` and `llc` are the AIE2 code generators; everything else (`aie-opt`, `aie-translate`, `bootgen`, `xclbinutil`) is orchestration tooling.

The seven steps, with citations:

1. **`aie-opt` phase 1 — resource allocation** ([Lowering.fs:349-352](../src/BackEnd/AIE/Lowering.fs)). The pass pipeline ([Lowering.fs:203-218](../src/BackEnd/AIE/Lowering.fs)) runs `aie-register-objectFifos`, `aie-objectFifo-stateful-transform`, lock/BD-id assignment, cascade/broadcast/multicast lowering, and `aie-assign-buffer-addresses{alloc-scheme=basic-sequential}`. Produces `input_with_addresses.mlir`.
2. **`aie-opt` phase 2 — Pathfinder routing** ([Lowering.fs:355-357](../src/BackEnd/AIE/Lowering.fs)): `aie.device(aie-create-pathfinder-flows)` ([Lowering.fs:221-222](../src/BackEnd/AIE/Lowering.fs)). Produces `input_physical.mlir`.
3. **Per-core compilation via Peano** ([Lowering.fs:361-435](../src/BackEnd/AIE/Lowering.fs)) — the heart of the dense-compute track. Sub-step 3a lowers core ops to the LLVM dialect with the `coreLoweringPipeline` ([Lowering.fs:239-241](../src/BackEnd/AIE/Lowering.fs)), producing `input_lowered.mlir`. Then `extractCoreTiles` scrapes `%core_C_R = aie.core` ops out of the physical MLIR by regex ([Lowering.fs:246-249](../src/BackEnd/AIE/Lowering.fs)). For *each* tile, `compileCore` ([Lowering.fs:385-405](../src/BackEnd/AIE/Lowering.fs)) runs the three-stage per-tile chain:
   - **3b** `aie-translate --mlir-to-llvmir --tilecol=C --tilerow=R` → `core_C_R.ll`
   - **3c** Peano `opt --passes=default<O2> -inline-threshold=10 -S` → `core_C_R.opt.ll`, then Peano `llc -O2 --march=aie2 --function-sections --filetype=obj` → `core_C_R.o`
   
   Finally 3d ([Lowering.fs:422-435](../src/BackEnd/AIE/Lowering.fs)) re-emits the physical MLIR in generic form (`--mlir-print-op-generic`) and `patchPhysicalMlirWithElfs` ([Lowering.fs:254-307](../src/BackEnd/AIE/Lowering.fs)) rewrites each `aie.core` body to an empty `aie.end` plus an `elf_file = "core_C_R.o"` attribute — because CDO generation requires empty bodies and ELF references.
4. **NPU instruction lowering + binary** ([Lowering.fs:437-446](../src/BackEnd/AIE/Lowering.fs)): `aie-opt` phase 3 ([Lowering.fs:225-232](../src/BackEnd/AIE/Lowering.fs)) runs `aie-dma-to-npu` etc. into `npu_insts.mlir`, then `aie-translate --aie-npu-to-binary --aie-output-binary` emits the `insts.bin` DPU instruction stream.
5. **CDO generation** ([Lowering.fs:448-450](../src/BackEnd/AIE/Lowering.fs)): `aie-translate --aie-generate-cdo` over the *patched* physical MLIR (with the ELF refs) emits the `*_aie_cdo_*.bin` configuration-data-object files.
6. **`bootgen` — CDO → PDI** ([Lowering.fs:453-460](../src/BackEnd/AIE/Lowering.fs)): writes a `design.bif` ([Lowering.fs:181-194](../src/BackEnd/AIE/Lowering.fs)) listing the three CDO files, then `bootgen -arch versal -image design.bif -o design.pdi -w`.
7. **`xclbinutil` — package xclbin** ([Lowering.fs:462-481](../src/BackEnd/AIE/Lowering.fs)): emits `mem_topology.json`, `kernels.json` (kernel `MLIR_AIE`, id `0x901`), and `aie_partition.json` ([Lowering.fs:119-178](../src/BackEnd/AIE/Lowering.fs)), then `xclbinutil --add-replace-section MEM_TOPOLOGY:JSON ... --add-kernel ... --add-replace-section AIE_PARTITION:JSON ... --output X.xclbin`.

Across all seven steps the F# contributes regex extraction, MLIR text patching, JSON/BIF emission, and process plumbing (`runTool`/`runToolUnit`/`runToolToFile`, [Lowering.fs:68-111](../src/BackEnd/AIE/Lowering.fs), augmenting `PATH` and `PEANO_INSTALL_DIR` per [Lowering.fs:336-345](../src/BackEnd/AIE/Lowering.fs)). **It generates no machine code.** Peano's `llc` is the only thing that lowers to AIE2 object code, and it is an external LLVM fork.

### The two-track split made physical

The HelloNappy proof-of-concept (peer repo `HelloNappy`) makes the two-track architecture visible as files on disk. Its `kernel/targets/intermediates/aie_prj/` directory contains the exact artifacts the seven steps produce. The kernel is a four-tile element-wise multiply (`Elements = 64`, `Grain = 16` ⇒ 64/16 = 4 tiles), so the per-tile compute track appears *four times over*, one chain per tile:

```
core_0_2.ll   core_0_2.opt.ll   core_0_2.o     ← tile (col 0, row 2)
core_1_2.ll   core_1_2.opt.ll   core_1_2.o     ← tile (col 1, row 2)
core_2_2.ll   core_2_2.opt.ll   core_2_2.o     ← tile (col 2, row 2)
core_3_2.ll   core_3_2.opt.ll   core_3_2.o     ← tile (col 3, row 2)
```

Each `core_N_2.ll` is genuine LLVM IR targeting the AIE engine — `target triple = "aie2p"`, with the AIE2 intrinsics declared (`llvm.aie2p.acquire`, `llvm.aie2p.release`, `llvm.aie2p.put.ms`, `llvm.aie2p.get.ss`, the vector `mcd.write.vec`/`scd.read.vec`) and the per-tile object-FIFO buffers as external globals (`@in1_0_cons_buff_0 = external global [16 x i32]`, etc.). The `.opt.ll` carries the Peano-O2 datalayout (`target datalayout = "e-m:e-p:20:32-..."`). The `.o` is the Peano-`llc`-produced AIE2 ELF (356 bytes per tile). **This is the dense-compute track: real LLVM IR through a real LLVM fork to a real object file.**

The orchestration track sits right beside it in the *same directory*, produced with **no LLVM at all**:

```
npu_insts.mlir              ← aie-opt phase 3 output (38 KB)
main_aie_cdo_elfs.bin       ← aie-translate --aie-generate-cdo
main_aie_cdo_init.bin
main_aie_cdo_enable.bin
design.bif  →  design.pdi   ← bootgen
mem_topology.json  kernels.json  aie_partition.json   ← xclbinutil inputs
input_physical.mlir / input_physical_generic.mlir / input_physical_patched.mlir
input_with_addresses.mlir / input_lowered.mlir
```

The DPU instruction stream (`npu_insts.mlir` → the kernel's `_insts.bin`), the CDO configuration objects, the PDI, and the xclbin packaging are *emitted directly from MLIR dialect tools and templates* — they never touch a compiler back-end in the LLVM sense. The `design.bif` confirms the hand-off: it is a generated text file naming the three CDO `.bin` files for `bootgen` to fold into a Versal PDI image.

This is the exact template the RISC-V thesis cites: **per-tile / per-core dense compute goes through an LLVM fork (Peano here; an LLVM RISC-V target there); the surrounding orchestration — instruction sequencing, memory topology, packaging — is emitted directly without LLVM.** Composer has already shipped this split to working silicon. The HelloNappy README's own pipeline diagram draws the same boundary — the kernel path reads *"-> Composer (AIE backend) -> aie_prj / MLIR_AIE -> Vitis / Peano -> .xclbin + insts.bin"* — and the target hardware is concrete: AMD Strix Halo with the XDNA2 NPU under the `amdxdna` driver, XRT 2.21.0.

### What this means for the RISC-V thesis

The standing art establishes three things the RISC-V backend inherits or must consciously reckon with:

- **The contract is free.** A RISC-V target is one `resolveBackEnd` arm ([PlatformPipeline.fs:13-18](../src/Core/PlatformPipeline.fs)) and one `BackEnd` record value. MLIR text arrives; an artifact leaves. No orchestrator change is required — the orchestrator already never dispatches on platform.
- **"Non-LLVM hardware target" is solved twice over.** CIRCT (FPGA → SystemVerilog via `circt-opt`) and AIE (NPU → xclbin+insts via the seven-step Peano/`aie-*`/`bootgen`/`xclbinutil` flow) both demonstrate that Composer reaches novel silicon by *emitting a dialect and delegating to that dialect's tools*. Neither emits a byte of machine code in F#.
- **The break, if taken, is deliberate.** Every existing backend is a thin orchestrator of upstream tools. A hand-rolled, fully self-contained RISC-V codegen — emitting machine code or assembly without delegating to an external dialect/assembler tool — would be the *first* backend to generate a binary itself rather than coordinate one. The AIE two-track split (Peano for dense compute, direct emit for orchestration) is the available middle path that keeps RISC-V inside the established precedent rather than outside it.

## 5. The Dedicated RISC-V Backend Thesis

A dedicated RISC-V backend is not a foregone conclusion. It is a claim that must be *earned*, target by target, against a high bar. This section states the claim precisely, establishes the burden of proof, and resolves the one boundary question that has historically caused drift: *where does the backend attach to the pipeline, and what does it consume?*

### The claim, stated narrowly

> RISC-V in the Fidelity/Composer architecture is a **companion orchestration** target. Its role is to *configure and sequence* an accelerator fabric — never to *be* the fabric. The dedicated backend that targets it generates an **instruction stream directly from a fully-lowered MLIR dialect**, bypassing LLVM, and it does so *only* on substrates where the lossy generic-IR handoff would discard semantic context that LLVM is structurally incapable of using.

Every qualifier in that sentence is load-bearing. "Companion orchestration" excludes the general-purpose CPU role. "Directly from MLIR" excludes the LLVM toolchain path. "Only on substrates where..." is the material-advantage gate that keeps this from being a land-grab against a mature, well-funded backend.

### The mental model: MLIR-AIE's host/runtime side

The clearest existing analogue lives in this very repository. The AMD NPU backend ([Lowering.fs](../src/BackEnd/AIE/Lowering.fs)) already splits its work along exactly the seam we are arguing for. Its `aieOptPassPipeline` ([Lowering.fs:203-234](../src/BackEnd/AIE/Lowering.fs)) lowers the *orchestration* — locks, buffer descriptors, DMA routing, the runtime sequence — directly through MLIR passes:

```
// Phase 3: NPU instruction lowering — the orchestration stream
"builtin.module(aie.device(" +
"aie-materialize-bd-chains," +          // buffer-descriptor chains
"aie-substitute-shim-dma-allocations," + // shim-DMA staging
"aie-assign-runtime-sequence-bd-ids," + // the aiex.runtime_sequence
"aie-dma-tasks-to-npu," +               // dma_memcpy_nd → NPU instrs
"aie-dma-to-npu," +
"aie-lower-set-lock))"                  // locks / semaphores
```

This phase emits the *launch sequence*: the shim-DMA buffer descriptors, the locks and semaphores that gate inter-tile data movement, the `dma_memcpy_nd` staging, and the pre-load tasks. None of it goes through LLVM. It is *direct-emit from the dialect* because the orchestration is the part of the program that the dialect already describes losslessly.

But the same file is scrupulously honest about the *other* half. Per-core compute kernels do **not** get this treatment — they take a backend path. The `coreLoweringPipeline` ([Lowering.fs:236-241](../src/BackEnd/AIE/Lowering.fs)) lowers each `aie.core` body all the way to LLVM (`convert-func-to-llvm`, `convert-to-llvm`), to be finished by a per-core compiler (historically xchesscc, now Peano / llvm-aie). MLIR-AIE emits orchestration *directly*, but it still compiles per-core compute via a backend.

That asymmetry is the thesis in miniature, and it generalizes:

| Workload class | Lowering strategy | Why |
|---|---|---|
| **Orchestration** (DMA, locks, launch sequence, descriptor staging) | **Direct-emit from dialect** | The dialect already *is* the orchestration; there is no generic-IR layer that adds value |
| **Per-core compute kernel** | **May warrant a backend** | Register-rich arithmetic over transparent local memory is exactly what LLVM is good at |

The RISC-V thesis is: *the RISC-V part of these systems is almost always the orchestration part.* On the substrates that matter most, there is no per-core compute kernel for RISC-V to run at all — the compute lives in a fabric the RISC-V core merely configures.

### LLVM is a permanent fixture; the burden of proof is on the dedicated backend

The Composer backend contract is deliberately neutral. A backend is a function value ([Pipeline.fs:35-40](../src/Core/Types/Pipeline.fs)):

```fsharp
type BackEnd = {
    Name: string
    Compile: string -> BackEndContext -> Result<BackEndArtifact, string>
}
```

and the platform router assigns LLVM to every general-purpose target ([PlatformPipeline.fs:13-19](../src/Core/PlatformPipeline.fs)):

```fsharp
| CPU | MCU | TargetPlatform.Library -> BackEnd.LLVM.Pipeline.backend
```

This is not an accident pending replacement. **LLVM is a permanent fixture.** For the commodity-CPU case it is the mainline RISC-V target, the home of decades of instruction-scheduling and register-allocation engineering, and the path of least surprise. A dedicated RISC-V backend does not *displace* LLVM in general; it *coexists* with it and is selected only where it can prove material advantage. The burden of proof sits on the new backend, never on LLVM.

There are two honest roads to "a RISC-V backend," and they should be framed even-handedly:

- **Road A — fork/specialize LLVM (Peano-style).** Take llvm-aie / a vendor RISC-V target, teach it the accelerator's quirks, and ride the existing backend. Mature codegen for free; pays the integration and maintenance tax of a fork.
- **Road B — hand-roll, sidestep LLVM.** Emit the instruction stream directly from a fully-lowered MLIR dialect. No fork, no generic-IR layer; the cost is authoring the upstream coeffect passes (instruction-selection tiling, register binding, ABI) and the thin encoder ourselves, rather than inheriting them.

The argument *for Road B is specific to Fidelity* and turns on one fact: **the MLIR that Alex produces is already the residue of Baker/PSG/coeffect observation.** By the codata principle at the heart of Composer's Alex layer — witnesses *observe and return*; they never build or compute — PSG→MLIR is *elision*: the residual of that observation, not active emission. By the time we hold the lowered MLIR, it encodes NTU dimensional intent, coeffect-derived data-movement structure, and codata-shaped control. Road A throws all of that away at the generic-IR handoff: LLVM ignores NTU intent, ignores coeffects, ignores codata, and reconstructs a target machine model from scratch under assumptions (transparent cache hierarchy, conventional stack, inherited C ABI) that may be *categorically* wrong for the substrate.

The prize, then, is **avoiding the lossy handoff** — not "escaping opinionated infrastructure" in the abstract. LLVM's opinions are excellent when its machine model holds. The case for Road B exists precisely where its model does *not* hold, and where the discarded semantic context is the only thing that makes correct code generation tractable.

### The fully-lowered-MLIR boundary (resolved)

A recurring source of drift was the temptation to let the RISC-V backend "reach back" into the PSG for context it felt was missing in the MLIR. **This is resolved, and the resolution is firm: it does not.** The dedicated RISC-V backend embraces the *fully-lowered MLIR* — the identical contract honored by the CIRCT and AIE backends, both of which consume only the `Compile : string -> BackEndContext -> ...` text-in interface ([Pipeline.fs:38](../src/Core/Types/Pipeline.fs)). It never opens the PSG, never imports a nanopass module, never inspects Baker state. (Reaching back would also violate layer separation outright: code generation must not import nanopass modules or inspect PSG-construction state — the nanopasses run first, and the backend consumes only their fully-lowered residue.)

Coherence with Baker/PSG/coeffects is therefore *not* bought by the backend reaching upstream. It is bought by a three-beat discipline:

1. **The front- and middle-end *express*** RISC-V and co-design intent — register-as-port mappings, atomics for inter-core sync, data-movement structure — into the MLIR.
2. **The MLIR *carries*** that intent, richly enough and in a dialect-stable enough form that nothing is lost in transit.
3. **The backend *embraces*** the MLIR exactly as written, reading only what is on the page.

Under this discipline, **"fully-lowered MLIR" is a contract on middle-end expressiveness**, not a limitation on the backend. If the backend finds itself wishing it could see a coeffect, the fix is upstream — the middle-end must have *expressed* that coeffect into the MLIR — never a peephole into the PSG at codegen time. This is the framework's cardinal rule — *fix upstream*: never patch where the symptom appears, but at the earliest pipeline stage where the defect actually exists — applied to the backend boundary. That earliest stage is the middle-end's expressiveness, not the backend's vision.

### Decisions as coeffects, emission as elision

A dedicated RISC-V backend is conventionally described as a pile of *algorithms*: an instruction selector, a register allocator, a scheduler. In the Fidelity architecture these are reframed to obey the codata model — **decisions are made upstream and recorded as coeffects; the backend witnesses and elides, it does not decide.** The zipper witnesses; it does not decide — and that discipline holds all the way down to the bit stream.

- **Instruction selection → a tiling nanopass.** Instead of a backend-resident matcher, ISel becomes a middle-end pass that tiles MLIR ops onto RISC-V instruction templates and records the tiling *as a coeffect*. The backend reads the chosen template; it does not search. This is "decisions upstream" applied to ISel — the selection is mise-en-place, pre-computed and observed.

- **Register binding → the determined image of SSA the PSG already carries.** This is the place a document like this most easily drifts, so it must be stated in the framework's own terms. SSA in Fidelity is **deterministically enumerated as part of elaboration and saturation in the PSG**, by the strict nanopass infrastructure in CCS, and it arrives already settled. It is emphatically *not* assigned piecemeal by a counter threaded through a recursive descent at codegen time — that conventional, Triton-style enumeration is precisely the antipattern the nanopass discipline was built to exclude, and reproducing it anywhere is a recipe for drift. The RISC-V target therefore adds no allocator and no enumeration. Its one structural addition is *physical binding*: the determined image of the already-enumerated SSA onto `x`-registers and spill slots, recorded — like every other structural decision — as a coeffect the thin emission tail reads. (The SSA machinery exists in CCS; a strategy note does not restate it.)

- **ABI / calling convention → a coeffect, and the deliberate place to shed LLVM's opinions.** On an orchestration substrate there may be *no conventional stack* and *no inherited lp64/ilp32 register-passing convention*: arguments may map to **configuration ports** on the fabric, results to status registers, with no spill area at all. Making the ABI a coeffect is precisely where the architecture *chooses* to discard the C calling convention that Road A would silently inherit. (Where the substrate genuinely is a conventional core, the coeffect simply records `lp64`/`ilp32` — the mechanism is the same; only the recorded decision differs. This honors the framework's filter against *premature concretization* — never bake CPU-specific widths or calling conventions into generated code ahead of the platform context that should resolve them.)

- **Scheduling → skipped.** For in-order orchestration cores, instruction scheduling buys little and risks reordering the data-movement sequence that *is* the program. It is deliberately out of scope for the first backend.

### The emission tail is thin — no LLVM, no Peano fork, no Farscape

Because the structural reasoning is extracted upstream, what the RISC-V backend itself does collapses to two mechanical steps: **encode** each selected instruction into its machine-code word — the inverse of `riscv-fs`'s decode tables — and **write the instruction blob** into whatever the loader expects. Neither step touches LLVM, and the point worth dwelling on is how *little* is left. CIRCT and AIE are non-LLVM but still delegate the heavy lifting to external dialect tools (`circt-opt`; `aie-opt`/Peano/`bootgen`/`xclbinutil`, [Lowering.fs:240](../src/BackEnd/AIE/Lowering.fs)); a pure-orchestration RISC-V backend has no such tool in the loop, yet it is *thinner* than either of them, not heavier — because the work the external tools were doing has been moved upstream into the PSG, not reimplemented here.

This is worth stating against the two reflexes that pull a backend back toward heavyweight infrastructure:

- **No LLVM, and no Peano-style fork of it.** A stripped LLVM fork (AMD's Peano for AIE2) earns its keep only when the target needs LLVM's instruction selection and scheduling for an exotic ISA. RISC-V on an orchestration core needs none of it: selection and register binding are upstream coeffects, encoding is the inverted `riscv-fs` table, and there is nothing for an LLVM fork to contribute. The fork is precisely the thing this approach *avoids*.
- **No off-the-shelf ELF library bound through Farscape.** Writing an ELF — or, for a bare orchestration core, a flat instruction blob or a vendor container — is a self-contained, fully-specified task with no relationship to LLVM, small enough to hand-roll: an ELF header, a couple of `PT_LOAD` segments, the `.text` bytes, an entry point, and for a single self-contained image no link step at all. Reaching for **Farscape** to bind a C/C++ ELF writer (LIEF, libelf, ELFIO) is a *distraction* here, and naming why matters: Farscape's substantial value is generating bindings so *Clef programs* can call C/C++ libraries — but the emission tail is *compiler* code, the blob writer is trivial, and binding a heavy external library would re-introduce a dependency at the one place whose entire virtue is that it has none. The binding mechanism is a non-question; the blob is just bytes.

So "fully self-contained" does not mean a bigger lift. It is the opposite: with the decisions carried upstream and emission reduced to encode-and-write, the backend for a pure-orchestration substrate can be **strikingly thin**, and that thinness — not heft — is the result. The one cost that is genuinely real is correctness without LLVM's tested codegen, and that is exactly what the `riscv-fs` oracle (§7) buys back. The thinness is also *why* the approach is gated: it is only this thin where the memory is software-managed and the structure is carried, never on a commodity core where LLVM's machine model genuinely applies.

---

## 6. Target Substrates

The material-advantage gate sorts candidate RISC-V substrates along a single gradient: **how much of LLVM's machine model still applies.** At one end sits a commodity core where LLVM is simply the right answer; at the other, a control plane for a fabric where LLVM's model is not merely suboptimal but *categorically inapplicable*.

### Vendor multiplicity: exemplars, not a catalog

RISC-V is an ISA, not a vendor, so "a RISC-V target" is always *some vendor's* part — SiFive, StarFive, T-Head/XuanTie, Nuclei, and more arriving steadily. The substrates below are **exemplars of the two axes, not an enumeration**: a part this document never names is placed the same way, by asking *does LLVM's machine model fit the core?* (the codegen axis) and *what C/C++ does the vendor ship to bind?* (peripheral access). There will always be another vendor; the axes, not the list, are what generalize.

Vendor multiplicity reaches into the toolchain itself, and it is worth stating plainly so the keep-LLVM branch is not mistaken for "mainline LLVM only." **"Keep LLVM" sometimes means keep a *vendor's* LLVM/GCC fork.** T-Head ships the XuanTie GCC fork for the C910's pre-ratification **RVV 0.7.1** vector — binary- and source-incompatible with the ratified RVV 1.0, so mainline LLVM/GCC cannot target it — just as AMD ships Peano for AIE2. Those forks are *still* the keep-LLVM branch: they exist because the core needs LLVM's instruction-selection and scheduling machinery for **dense compute** (vectors, VLIW). They are precisely *not* Fidelity's thin-direct path, which stays reserved for the software-managed-memory orchestration sliver regardless of vendor. The rule holds across the whole vendor field: dense compute on any vendor's application or accelerator core → LLVM, possibly a vendor fork of it; the orchestration/control sliver on a software-managed substrate → thin-direct.

### The gradient

| Substrate | RISC-V role | Memory model | LLVM's model | Advantage of a dedicated backend |
|---|---|---|---|---|
| **SiFive standard cores** | The application CPU itself | Transparent / hardware-managed cache | Mainline, well-tuned target | **Least** — keep LLVM |
| **Tenstorrent** | "Baby" RV32IM cores orchestrating Tensix engines | Explicitly software-managed L1 SRAM; NoC | Inapplicable to the data movement | **Moderate** — orchestration warrants direct-emit; compute is custom (not RISC-V) |
| **NextSilicon Maverick-2** | RISC-V control plane configuring a dataflow CGRA | Dataflow buffers; no transparent hierarchy | Categorically inapplicable | **Most** — pure orchestration, no compute ISA to share |

**SiFive standard cores — least advantage.** A SiFive core *is* the application processor. It has a transparent, hardware-managed cache; a conventional stack; an inherited ABI. This is LLVM's mainline RISC-V target, tuned over years — and SiFive's cores are about as first-class as RISC-V LLVM support gets: the company upstreams aggressively (notably the vector backend), and its compiler pedigree runs deep, with Chris Lattner (who created LLVM and later MLIR) having led platform engineering there ~2020–2022. Going around LLVM on a SiFive part would mean fighting the grain of the strongest toolchain support the ISA has — with the added irony that Composer reaches that triple *through* MLIR, the same lineage. Two further practical facts reinforce keeping LLVM here: such a target wants a managed runtime with **GC** and the **V (vector) extension**, neither of which the `riscv-fs` oracle currently models; and on a transparent-cache core there is no semantic context being thrown away at the handoff that LLVM could not also have used. The correct move is to **keep the triple** and route through `BackEnd.LLVM.Pipeline.backend` exactly as `CPU` does today ([PlatformPipeline.fs:16](../src/Core/PlatformPipeline.fs)) — and this holds *even when the deployment is a freestanding unikernel* rather than hosted Linux, a distinction the *Two orthogonal axes* subsection below makes explicit.

**Tenstorrent — moderate advantage.** Here the RISC-V cores are small RV32IM "baby" cores whose *job* is to orchestrate the surrounding Tensix matrix/vector engines: stage data through software-managed L1 SRAM, move it across the NoC, and launch compute. The orchestration half is a strong direct-emit candidate. But the compute half — the Tensix engine — is a *custom* ISA, not RISC-V, and certainly not something the `riscv-fs` oracle describes. So Tenstorrent splits exactly like MLIR-AIE: orchestration (RISC-V, direct-emit) is in scope; per-engine compute is out of scope for the RISC-V backend.

**NextSilicon Maverick-2 — most advantage.** The Maverick-2 RISC-V core is a *pure control plane*: it configures and steers a reconfigurable dataflow CGRA. There is **no compute-kernel ISA to share work with** — the compute is the fabric's dataflow, not instructions the RISC-V core executes. This is orchestration in its purest form, and it is where the dedicated-backend thesis is strongest: the entire program the RISC-V core runs *is* the configuration and launch sequence. There is nothing for LLVM's codegen strengths to contribute and everything in its machine model to get in the way.

### The material-advantage criterion, made precise

The gradient is governed by one criterion, and it is sharper than "is LLVM suboptimal here":

- **Hardware-managed / transparent cache (commodity CPU).** Cache-aware compilation on such a target means *hinting a mechanism the hardware runs anyway* — prefetch hints, alignment, blocking. LLVM is at worst **suboptimal**; the cache still works if you ignore it. **Keep LLVM.** A dedicated backend would be re-deriving, at great cost, decisions LLVM already makes well.

- **Explicitly software-managed memory (Tensix L1 SRAM, AIE memory tiles, dataflow buffers, NoC).** There *is no transparent cache*. **The data movement is the program.** Every byte's journey — which buffer, which lock gates it, which DMA descriptor stages it, which hop across the NoC — must be expressed explicitly. LLVM's transparent-hierarchy/stack/ABI model is **categorically inapplicable**, not merely suboptimal: it assumes a memory system that does not exist on this substrate. On these targets, **coeffect-to-data-movement elision is the only coherent path**, because the data-movement schedule is precisely the coeffect-shaped structure the middle-end observed and that LLVM would discard.

This collapses three apparently separate claims into one identity. On a software-managed-memory substrate:

> **cache-aware compilation = the orchestration thesis = the hand-rolled-backend thesis.**

They are the same statement viewed from three angles. "Cache-aware" is meaningless where there is no cache — what you actually need is *data-movement-aware*, which is orchestration, which the dialect already encodes, which is exactly why you emit directly and skip the LLVM handoff.

### Two orthogonal axes: what the core is, and what the deployment is

It is tempting to read "unikernel" as a synonym for "thin, LLVM-free backend." It is not, and conflating the two is a real source of drift. There are two independent axes here, and only one of them selects the backend.

- **The codegen axis** — LLVM vs. a thin direct emitter — is decided by the *core's machine model*, which is exactly the material-advantage gate above. Transparent cache + conventional ABI → keep LLVM. Software-managed memory + orchestration role → thin-direct.
- **The deployment axis** — hosted under an OS vs. a freestanding unikernel — is decided by *what you link and how you boot*: a unikernel is `-ffreestanding`, no libc, your own `_start` and linker script, and peripheral access via memory-mapped loads and stores. This is a source-and-link property. It does **not**, by itself, select a backend.

"Address the peripherals directly" and "emit the binary without LLVM" are therefore *different kinds of directness*. The first is a unikernel virtue achieved at the source and link layer; the second is the thin-backend thesis, justified only where the machine model breaks. A program can have the first and still use LLVM for the second.

**Worked example — a unikernel that keeps LLVM (StarFive VisionFive 2).** The VisionFive 2 carries a SiFive U74 quad-core, **RV64GC** (hardware F/D float, C compression, atomics), an MMU, a 2 MB L2 cache, and rich peripherals (USB 3.0, dual GbE, HDMI, MIPI CSI/DSI, 40-pin GPIO, M.2, eMMC). A unikernel for it should route through the LLVM triple in freestanding mode, for two decisive reasons. First, it is a transparent-cache application core with a standard ABI — LLVM's mainline RISC-V target, where its machine model is *correct*, not a liability, so the material-advantage bar is not met. Second, RV64GC needs F/D and C, which `riscv-fs` neither encodes nor can verify; a thin path would reimplement what LLVM does well and forfeit the oracle. The directness the board invites is delivered *above* LLVM: model each peripheral's register block as a BAREWire layout with exact offsets, so MMIO access is typed and deterministic rather than hand-counted; carry the freestanding/no-alloc discipline as coeffects; lower **PSG → MLIR → LLVM (freestanding RV64GC) → freestanding ELF**. Direct hardware access and tested codegen, together — the co-design predicate's strongest CPU-side showing, where the front end does the work and the LLVM triple does the lowering.

**Addressing those peripherals — bind the vendor's C/C++.** This need not go through a *formal* HAL. Farscape generates bindings for any C/C++ a vendor ships — SDK, driver library, BSP, or reference register-poke code — so the binding target is simply "whatever C/C++ the vendor provides," exactly as with ST's HAL/LL or Renesas's FSP; the only board-specific variables are *whose* code it is and *what shape* it takes. Two facts settle that for the VF2. First, its peripherals are **StarFive's, not SiFive's**: SiFive supplies the CPU complex (U74/S7) and the standard privileged peripherals (the CLINT timer and PLIC interrupt controller), while StarFive integrates everything else — the UART, for instance, is a Synopsys DesignWare DW8250 — documented in the JH7110 Technical Reference Manual and exposed through the device tree. Second, StarFive ships a **substantial C/C++ stack** for it (the VisionFive 2 SDK, U-Boot and OpenSBI platform code, the JH7110 driver tree) — all bindable; SiFive's own Freedom Metal is just one more such C SDK, scoped to its MCU-class parts. So peripheral access on the VF2 is the *same vendor-C/C++ binding move* you already use for ST and Renesas — it binds a StarFive SDK rather than a CMSIS-style HAL, and that distinction does not matter to Farscape. (One honest nuance: bind the bare-metal-appropriate pieces — register-level drivers and init sequences — rather than OS-coupled code that assumes a kernel runtime. And where you would rather not carry C into a freestanding image at all, the register *description* — the JH7110 TRM/device tree, or a CMSIS-SVD file where published — can instead be ingested into BAREWire layouts for typed MMIO; that is a stylistic choice and the peripheral-layer expression of the co-design predicate, not a necessity forced by any "gap.")

**Worked example — a unikernel that drops LLVM (bare MCU / orchestration core).** A microcontroller-class RISC-V (RV32IMC, no FPU, no MMU, software-managed or no cache, fixed memory) is the opposite case, and here the deployment and the machine model point the same way. Emission is genuinely encode-and-write — no linker, a single flat image — and the FPU-less core is exactly where the front end's representation discipline pays off: because the NTU/dimensional type system fixes a value's representation upstream — a fixed-point form, a posit, a specific integer width, from the range its type admits — and carries it through lowering as preserved structure (the type-preserving fixed-point scaffold; see the *fixed-point-scaffolding* and *negative-and-fractional-types* papers in the `arxiv-papers` peer repo), the thin backend emits correct fixed-point arithmetic directly, with no soft-float lowering and no `libgcc`-style runtime. The numeric correctness was settled at elaboration; the backend only spells it out.

The rule that disentangles the two: **the core decides the backend, not the OS-lessness.** A U74 unikernel is LLVM-freestanding; a Maverick-2 control core or a bare RV32IMC unikernel is thin-direct. Both are unikernels — the memory model, not the absence of an operating system, is what places them.

---

## 7. RISC-V as Orchestration, Not Fabric

This section makes explicit the inversion that anchors everything above: in the systems we care about, the RISC-V core is **the conductor, not the orchestra.**

### The inversion

The instinctive reading of "RISC-V backend" is "compile arithmetic to RISC-V instructions." On a Maverick-2 or a Tenstorrent or an NPU shim, that reading is *backwards*. The RISC-V core does not crunch the numbers; it tells the fabric where to find the numbers, when to start, which buffer to fill, which lock to release. Its instruction stream is overwhelmingly **integer control, memory addressing, branches, and synchronization** — the verbs of orchestration — and almost never the dense floating-point or vector arithmetic that a compute kernel comprises.

This is exactly the AIE Phase-3 picture ([Lowering.fs:224-232](../src/BackEnd/AIE/Lowering.fs)) generalized: buffer descriptors, DMA staging, locks/semaphores, a runtime launch sequence. Emit *that* directly from the dialect, because *that* is what the dialect natively describes. The fabric's compute is somebody else's problem (a per-core backend, a CGRA bitstream, a Tensix kernel) — and on the purest substrates, there is no separable compute step at all because the fabric *is* the computation.

### Three roles for riscv-fs: vocabulary, oracle — under coeffect/elision

The peer repository **riscv-fs** (an F# RISC-V ISA model and simulator; *not* a Composer-repo-relative dependency) plays a precise, bounded part in this architecture, framed through the pillars:

- **Coeffects are the *intent*.** The tiling nanopass, the physical binding of the PSG's deterministically-enumerated SSA, and the ABI coeffect together record *what the program means to do* on the RISC-V core.
- **Elision is the *grammar*.** PSG→MLIR→stream is elision — the residual of observation — so the backend *spells out* the recorded intent rather than inventing it.
- **riscv-fs is the *vocabulary*** — the concrete instruction leaf that the intent elides *into*. When a tile says "this is an `add`," riscv-fs supplies the canonical RV instruction that word denotes.
- **riscv-fs is also the *oracle*** — it *proves the bits mean what the coeffects intended*. We encode the stream, run it on the riscv-fs simulator, and check that the resulting `MachineState` is what the coeffects said it should be. The oracle closes the loop with zero external tools.

Crucially, riscv-fs's coverage is **well-matched to orchestration**. RV32/64 **IMA** gives integer arithmetic, control, memory addressing, branches, and multiply/divide — the integer-control-and-memory core of every orchestration program. The one caveat is sharp and must be tracked: riscv-fs's **A-extension (atomics) is its weakest area** — and atomics are *exactly* what orchestration needs for inter-core synchronization (the lock/semaphore role that AIE's `aie-lower-set-lock` fills). The A-extension support in the oracle must be **completed and verified** before it can certify any multi-core orchestration stream. Everything else in IMA is ready for the role.

### What riscv-fs provides vs. what must be built

| Concern | riscv-fs provides | Must be built (Composer side) |
|---|---|---|
| **Instruction vocabulary (RV32/64 IM)** | Canonical instruction definitions and semantics | Inverted encoding tables (semantics → bit pattern) for the emitter |
| **Atomics (A-extension)** | Partial / weakest area | **Complete + verify** A-extension; it is the inter-core sync path |
| **Execution oracle** | Cycle/step simulator, observable `MachineState` | Test harness that diffs emitted-stream state against coeffect expectation |
| **Instruction selection** | — | Tiling nanopass: MLIR op → RISC-V template, recorded as coeffect |
| **Register binding** | — | Physical binding of the PSG's already-enumerated SSA onto `x`-registers/spill slots, recorded as a coeffect — no allocator, no re-enumeration |
| **ABI / calling convention** | — | ABI coeffect: registers→config-ports, or recorded lp64/ilp32 |
| **Binary encoding / packaging** | — | Self-contained encoder + flat-binary writer (first backend with no external tool) |
| **Scheduling** | — | Deliberately skipped for in-order orchestration |

The pattern is consistent: riscv-fs supplies the *meaning of the leaf* and an *oracle to check it*; Composer supplies the *path from coeffect to leaf* and the *self-contained encoder*. riscv-fs is never asked to do code generation; it is asked to define vocabulary and to render judgment.

### The first milestone

The thesis becomes testable with the smallest possible end-to-end slice, deliberately chosen to exercise *every* novel element while excluding everything that can be deferred:

1. **One tiny integer kernel** — a handful of integer ops, no floats, no vectors, no GC, no strings. Pure orchestration-shaped arithmetic.
2. **Elide it to MLIR** through the existing Alex pipeline — proving the front/middle expresses RISC-V intent into the dialect.
3. **Integer instruction selection** — the tiling nanopass produces RISC-V templates as a coeffect.
4. **Register binding as coeffect** — bind the PSG's already-enumerated SSA onto `x`-registers and spill slots; the one RISC-V-specific structural step, recorded as a coeffect rather than computed by a codegen-time allocator.
5. **Flat-binary encode** from **inverted riscv-fs tables** — the self-contained encoder, with zero external tools in the loop.
6. **Run on the riscv-fs oracle** and **diff the resulting `MachineState`** against the coeffect-derived expectation.

This milestone targets a **Maverick-2-style pure-orchestration profile** — integer control plane, no compute-kernel ISA — so that the *first* thing built is also the *strongest-advantage* case, where the dedicated-backend thesis is least contestable. It uses **no external tools whatsoever**: the entire path, from fully-lowered MLIR to a verified `MachineState`, lives inside Composer and riscv-fs. If this slice closes — bits encoded, executed, and proven to mean what the coeffects intended — the architecture is validated end to end and the only remaining work is breadth (atomics completion, ABI variants, additional substrates), not feasibility.

## 8. Relationship to Existing Documents

This note sits within an established documentation set. The most relevant
neighbours:

- [Architecture_Canonical.md](./Architecture_Canonical.md) — the authoritative
  two-layer model, platform-binding model, and nanopass pipeline this strategy
  operates within. Read first for the surrounding architecture.
- [Coeffect_Analysis_Architecture.md](./Coeffect_Analysis_Architecture.md) and
  [PSG_Nanopass_Architecture.md](./PSG_Nanopass_Architecture.md) — the coeffect
  and nanopass machinery in which instruction selection, register allocation, and
  ABI would live as passes (§5). The "decisions as coeffects, emission as
  elision" argument is an application of these.
- [AIE_Backend_Design.md](./AIE_Backend_Design.md) — the structural precedent
  documented in §4: a non-LLVM backend that splits a specialized-LLVM compute
  path (Peano) from a directly-emitted orchestration path. The closest existing
  analogue to the proposed RISC-V backend.
- [Alex_Architecture_Overview.md](./Alex_Architecture_Overview.md) — the
  Zipper/XParsec/Bindings elision model that produces the fully-lowered MLIR the
  backend would consume.
- [Platform_Binding_Model.md](./Platform_Binding_Model.md) — the
  Bindings-as-data model onto which a per-extension RISC-V encoder maps.
- [NTU_Architecture.md](./NTU_Architecture.md) — the dimensional type system that
  instruction selection reads when resolving operand widths.
- [LLVM_Dialect_Reference.md](./LLVM_Dialect_Reference.md) and
  [CCS_Architecture.md](./CCS_Architecture.md) — the existing LLVM lowering path
  that remains the default for commodity triples, and the Clef Compiler Services
  front end that originates the intent.

External and peer sources cited in this document:

- *Cache-Conscious Memory Management: CPU Edition*
  (speakez.tech/blog/cache-aware-compilation-cpu/) and *Context-Aware
  Compilation* (speakez.tech/blog/context-aware-compilation/) — the co-design
  predicate that frames §2; published in the `clef-lang-site` peer repository.
- `riscv-fs` (github.com/mrLSD/riscv-fs; forked at houstonhaynes/riscv-fs) — the
  executable RISC-V ISA reference assessed in §3. A peer repository.
- HelloNappy (peer repository) — the working AMD XDNA2 NPU proof-of-concept whose
  `aie_prj` intermediates make the two-track split in §4 concrete.
- BAREWire (peer repository), *Cache-Aware Layouts* — the deterministic-layout
  foundation referenced in §2.
