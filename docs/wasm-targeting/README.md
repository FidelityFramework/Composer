# WASM Targeting

**SpeakEZ Technologies | Fidelity Framework**
**April 2026**

## What's Here

- **README.md** (this file) — the triangulation of pathways (LLVM WASM / WAMI), continuation strategies (Coroutines / Stack Switching / JSPI), and Cloudflare's step-graded compute (Tier 1 pure JS / Tier 2 WASM-in-Worker / Tier 3 Container).
- [01_type_carrying_and_workloads.md](./01_type_carrying_and_workloads.md) — the type-carrying delta between JSON (the default Worker vocabulary), JavaScript's TypedArray/DataView surface, and WASM linear memory with WIT-typed boundaries. Provides the decision rubric for which workloads actually belong in Tier 2 and which do not. Answers "my Worker does X — should I reach for WASM?" with specifics rather than general handwaving.
- [02_deployment_and_lifecycle.md](./02_deployment_and_lifecycle.md) — the Cloudflare-specific deployment model: WASM as a library bundled inside a Worker (not a separate service), the containment hierarchy, cold-start sequence, and the JS shim's irreducible role. Answers "how does a WASM instance actually get deployed and invoked on Cloudflare?"
- [03_build_bundle_and_execution_models.md](./03_build_bundle_and_execution_models.md) — zooms out beyond Cloudflare: how Composer builds the target-agnostic `.wasm`, how substrate-specific bundlers wrap it for different hosts (Cloudflare Workers, browser, Wasmtime, embedded, Component Model), the "WASM as unikernel" execution model that the DCont scheduler enables, and the break point where single-instance cooperative concurrency gives way to multi-instance fan-out with BAREWire coordination.


WebAssembly is not JavaScript. The two targets share deployment contexts — browsers run both, Cloudflare Workers accept both, Node.js and standalone runtimes support both — but they are architecturally distinct compilation targets with different runtime models, different type-carrying properties, and different ways of handling continuation-bearing code. This subfolder is the companion to [javascript-targeting/](../javascript-targeting/) and covers the WASM story specifically.

The WASM story in Fidelity triangulates across three axes that are worth keeping separate even though they interact constantly:

1. **Pathway**: the compilation path inside Composer that produces WASM. Two coexisting pathways exist: LLVM WASM and MLIR WAMI. They differ in what they preserve from the source and in what runtime features they require.
2. **Continuation handling**: how DCont semantics (async, actors, computation expressions, all unified as delimited continuations in Clef) survive the lowering to WASM. Two coexisting strategies exist: DCont-via-Coroutines (state machine lowering) and DCont-Native (via WebAssembly's Stack Switching proposal). Pathway and strategy are related but not identical; both pathways can use either strategy depending on what the target runtime supports.
3. **Deployment context**: where the WASM artifact actually runs. Cloudflare Workers is the focus here because it sits in an interesting architectural position between pure JavaScript Workers and full Containers — WASM is the middle tier of a step-graded compute model that the framework can target deliberately. Browsers and standalone WASM runtimes are adjacent deployment contexts that the same artifacts can reach.

This document walks each axis in turn, then shows the matrix.

## Axis 1: The Two Pathways

### Pathway A — LLVM WASM

Composer's LLVM backend can emit WebAssembly via LLVM's native WASM target. This is the straightforward path: Alex emits `func`/`memref`/`arith`/`scf` ops, the MLIR lowering produces LLVM IR, LLVM's WASM backend produces `.wasm` modules. The pathway shares most of its infrastructure with native ELF compilation, diverging only at the final code generation step.

```
Clef source → CCS → PSG → Baker → Alex (MLIR witnesses)
           → func/memref/arith/scf ops
           → mlir-opt → LLVM IR
           → llc --target=wasm32-wasi → .wasm
```

Properties:

- **Broad runtime support.** Anything that runs WASM runs LLVM-emitted WASM: browsers, Node.js, Wasmtime, WasmEdge, Cloudflare Workers, embedded WASM runtimes. LLVM's WASM target is mature and well-understood.
- **Standard WASM MVP + extensions.** Uses the stable WASM feature set plus widely-supported proposals (multi-value, reference types, bulk memory). Does not require experimental runtime features.
- **State machine lowering for async.** DCont semantics compile through LLVM's coroutine infrastructure, which lowers to explicit state machines. The continuation structure is preserved in the source and mid-pipeline but reified as control-flow state before reaching LLVM IR. On the wire, async functions look like state machines that step through discrete suspension points.
- **Memory model is linear memory.** WASM's linear memory is what `memref` operations lower to. Typed access is provided by the compiler's static knowledge of memory layout; the runtime sees a single byte array.
- **No stack switching required.** LLVM emits state-machine code; the runtime does not need native coroutine or stack-switching support.

This is the pathway that works today for any WASM target Composer might reach. It is the equivalent of the standard LLVM native path translated from ELF to `.wasm`, and it carries the same tradeoffs: broad support, predictable behavior, continuation structure reified into state machines before emission.

### Pathway B — WAMI (MLIR WASM Dialects)

WAMI is the research project that defined WebAssembly MLIR dialects (`SsaWasm`, `Wasm`) and demonstrated compilation to WebAssembly *without going through LLVM IR*. The WAMI paper (arxiv.org/html/2506.16048v1) describes the pipeline: Clef-style source compiles through MLIR dialects, lowers through MLIR's WASM dialects directly, and emits `.wasm` modules. LLVM is out of the loop entirely.

```
Clef source → CCS → PSG → Baker → Alex (MLIR witnesses)
           → func/memref/arith/scf ops
           → lowering pass → wasm-ssa ops (WAMI dialect)
           → wasm-ssa → wasm dialect → .wasm
```

Properties:

- **Stack switching preservation.** The WAMI path is designed around the WebAssembly Stack Switching proposal, which adds native suspend/resume primitives to WASM. DCont semantics can lower to stack-switching ops directly, without reification into state machines. The continuation structure survives intact through the backend.
- **MLIR-native end-to-end.** The pipeline never leaves MLIR's infrastructure. Optimization passes, dataflow analyses, and dialect conversions all operate in MLIR's framework. This parallels CIRCT's approach (MLIR all the way to Verilog without ever touching LLVM) and MLIR-AIE's (MLIR to xclbin through Peano).
- **Research-grade, not production.** WAMI is currently a research project. The WebAssembly Stack Switching proposal (phase 3 as of early 2026) is supported by V8 behind flags but is not universally available. Wasmtime has experimental support. Cloudflare Workers does not currently expose stack switching at the isolate boundary.
- **Runtime gating required.** A WAMI-produced `.wasm` artifact targeting stack switching requires the runtime to support it. If stack switching is not available, the artifact will fail to instantiate. Composer needs to know which continuation strategy the target runtime supports and gate the pathway choice accordingly.

WAMI is the forward-looking pathway. It is how Fidelity preserves the full expressive power of DCont all the way to WASM emission, on the runtimes that can receive it. Over the horizon in which WASM stack switching becomes broadly available, WAMI is the pathway that makes WASM an equal citizen with native for delimited-continuation-heavy code.

### Why Both Pathways

Neither pathway dominates. LLVM WASM is deployable today to any WASM runtime; WAMI preserves semantic structure LLVM's lowering destroys. The choice is made per-project and per-deployment-target based on:

- **Does the target runtime support stack switching?** If yes, WAMI gives better code. If no, LLVM WASM is required.
- **Does the code use DCont-heavy patterns?** Actor supervision trees, long-running async with many suspension points, effect-handler-style code — these benefit from stack switching when available.
- **Is the target a runtime that Composer itself controls?** For a Wasmtime embedding Composer configures, enabling stack switching is a deployment-time choice. For a third-party runtime (a browser, Cloudflare Workers today), the feature support is fixed and the pathway must adapt.

The architecture encodes this as a target-profile decision, the same mechanism that picks between native LLVM and CIRCT for different hardware classes. Composer's `.fidproj` target declaration names both the runtime (Wasmtime, Workers, browser) and the required feature set (stack-switching required, state-machine acceptable); Alex's witness registration and the backend pathway choice follow.

## Axis 2: Continuation Handling

DCont (delimited continuations) is the unifying abstraction for async, actors, and computation expressions in Clef. Every continuation-bearing construct compiles through the DCont MLIR dialect. What differs by target is how DCont ops are lowered into the target's execution model.

### Strategy 1: DCont-via-Coroutines (State Machine Reification)

On targets that lack native stack-switching primitives, DCont ops lower through LLVM coroutine infrastructure. Each `dcont.shift` becomes a coroutine suspension point; the coroutine frame becomes the reified continuation. At the WASM emission layer, the coroutine frame is an explicit struct living in linear memory, and suspension is explicit state assignment followed by return-to-caller.

This is the strategy the LLVM WASM pathway uses. It is also the strategy WAMI uses when targeting a runtime without stack switching — WAMI can fall back to state-machine lowering if the target profile demands it. The work on LLVM coroutine emission is not throwaway; it is a production-tested implementation substrate for DCont on any target that lacks native support.

Tradeoffs:
- Works on every WASM runtime.
- Reifies the coroutine frame into memory; suspension touches memory on every step.
- Loses some optimization opportunities that true stack switching would preserve (inlining across suspension points, stack-allocated continuation state).

### Strategy 2: DCont-Native (WebAssembly Stack Switching)

On targets with stack switching, DCont ops lower directly to the stack-switching primitives. Suspension is a native operation; resumption is native; the continuation is a first-class runtime value handled by the WASM engine rather than by explicit user-code state machines.

The WebAssembly Stack Switching proposal introduces:
- `cont.new` — create a continuation
- `cont.bind` — pre-bind arguments
- `cont.suspend` — suspend and yield a tag value
- `cont.resume` — resume a continuation

These are the WASM-native equivalents of the `shift`/`reset`/`resume` primitives Clef's DCont dialect exposes at the source level. The mapping is direct; no reification needed.

Tradeoffs:
- Preserves continuation structure at the runtime level.
- Enables optimization patterns LLVM coroutines cannot express (zero-cost suspension when no state needs to be saved, engine-managed continuation pool).
- Requires runtime support (V8 behind flags, Wasmtime experimental, not universal).

### Strategy Selection Is Orthogonal to Pathway

A common confusion: pathway (LLVM vs. WAMI) and strategy (coroutines vs. stack switching) seem like they should be bundled. They are not.

- LLVM WASM can, in principle, emit WASM that uses stack switching through LLVM's WASM backend — LLVM has partial support for the proposal. In practice, coroutines are simpler and more mature.
- WAMI can lower DCont to either stack switching (preferred, when available) or coroutine-style state machines (fallback, for runtimes without stack switching).

The matrix is:

| Pathway | Default strategy | Fallback strategy |
|:--------|:----------------|:------------------|
| LLVM WASM | Coroutines (state machines) | — |
| WAMI | Stack switching (when target supports) | Coroutines |

Alex's target-profile logic determines which combination is active. The PSG codata carries the runtime capability flags; the backend choice follows from those flags.

## Axis 3: Cloudflare's Step-Graded Compute

Cloudflare Workers' architecture has changed over time. What started as a single compute model (JavaScript in V8 isolates) now spans a graded set of compute tiers, each with distinct isolation, memory, and cold-start properties. WASM fits in the middle of this grade.

### Tier 1 — Pure JavaScript Workers

The original Workers model. JavaScript code runs in a V8 isolate. Each request gets an isolate (possibly shared across requests); cold start is ~5 milliseconds typical; memory is limited to 128 MB; compute time is measured in milliseconds per request. No filesystem, no traditional processes, no SharedArrayBuffer across isolates.

Strengths: fastest cold start, highest density, cheapest per-request.
Constraints: JavaScript's type system (or lack thereof); no type-carrying beyond ArrayBuffer/TypedArray; no native memory mapping; all I/O through Cloudflare-provided bindings.

This is the target the [javascript-targeting/](../javascript-targeting/) subfolder covers.

### Tier 2 — WASM in Workers

Cloudflare Workers accept WebAssembly modules alongside JavaScript. The WASM runs inside the same V8 isolate; it's V8's WASM implementation, accessed through the JavaScript WebAssembly API. A Worker can be entirely WASM-driven (with a minimal JavaScript shim that imports and invokes the WASM) or a hybrid (JavaScript for the request lifecycle, WASM for compute-intensive inner loops).

Cloudflare shipped WASM support for Workers in 2018. The core runtime model is structurally the same today: Workers still hand you a `WebAssembly.Module` (not an `Instance`) for performance and security reasons, and the module is instantiated inside the isolate with a JavaScript-provided imports object. But the developer experience has shifted substantially between 2018 and 2026, and the shift changes what Composer should target.

**What's the same.** Module instantiation still follows the pattern Cloudflare described in 2018:

```javascript
const imports = { exampleImport(a, b) { return a + b; } };
const instance = new WebAssembly.Instance(MY_WASM_MODULE, imports);
instance.exports.exampleExport(123);
```

Cloudflare's original guidance on *when* to use WASM remains the right framing: "WASM is not always the right tool for the job. For lightweight tasks like redirecting a request to a different URL or checking an authorization token, sticking to pure JavaScript is probably both faster and easier than WASM... WASM really shines when you need to perform a resource-hungry, self-contained operation, like resizing an image, or processing an audio stream." The copy-in/copy-out cost at the JS/WASM boundary that made this true in 2018 still dominates the decision in 2026.

**What's changed.** The deployment experience, the ecosystem, and the continuation story have moved considerably:

- **Entire-Worker-in-WASM is now viable.** The 2018 post described WASM as a compute-intensive inner loop called from a JavaScript Worker. In 2026, Cloudflare ships [`workers-rs`](https://github.com/cloudflare/workers-rs), a Rust crate that lets you write an entire Worker in Rust compiled to WASM, with the full Runtime APIs and bindings (KV, R2, D1, Queues, Durable Object interaction) exposed directly from Rust. The JavaScript is still there structurally — `worker-build` generates a minimal JS entrypoint that instantiates the WASM module — but it is generated glue, not hand-written code.
- **`wasm-bindgen` handles the plumbing.** The manual `imports` object construction is replaced by `wasm-bindgen` glue generation. The Rust side annotates function signatures; `wasm-bindgen` produces both the Rust-side proxy types for JavaScript values and the JS-side glue module. Async is bridged through `wasm-bindgen-futures`, which converts Rust `Future`s to JavaScript `Promise`s and vice versa. For Composer, this matters because Fidelity's LLVM WASM pathway can target the same glue conventions and integrate with `workers-rs`-style deployments.
- **SIMD is supported.** WebAssembly SIMD works on Cloudflare Workers. Composer's numeric-dense code paths (BitNet inference, BAREWire codec hot paths with parallel byte manipulation, cryptographic bulk operations) can use SIMD ops emitted through LLVM's WASM backend without runtime gating.
- **Panic recovery for Rust (September 2025).** `workers-rs` 0.6.5+ registers a default panic handler and a `__wbg_reset_state` function that resets the WASM VM state after a panic, making in-flight requests fail with 500 but keeping the Worker instance viable for subsequent requests. Earlier, a Rust panic would corrupt the Wasm application state permanently. This is narrowly Rust-specific but illustrates the broader maturation: Cloudflare is investing in making WASM-driven Workers production-reliable, not just functional.
- **Component Model with WIT (WebAssembly Interface Types) is mainstream in 2026.** The 2018 post called out Emscripten as the expected build path. The 2026 picture is the Component Model: WIT types describe cross-language module interfaces, allowing a single application to compose a video-encoding module written in C++, a data-transformation module written in Rust, and a UI layer in JavaScript-or-otherwise as a single unified system. For Fidelity, this means that producing WASM components with WIT-described interfaces gives Clef-compiled code first-class composability with other-language WASM components in the same deployment.
- **JSPI (JavaScript Promise Integration) changes the continuation story.** JSPI provides a subset of the full Stack Switching proposal that is implementable in V8 today (and is shipped in Cloudflare Workers as part of the Chrome-aligned feature set). JSPI lets WASM code suspend at JavaScript-Promise boundaries and resume when the Promise settles — not arbitrary stack switching, but stack switching at async boundaries. This is why `wasm-bindgen-futures` can bridge Rust `Future`s to JS `Promise`s without a CPS transformation: the underlying runtime already does suspend/resume at Promise boundaries. The WASIp3 milestone released in early 2026 uses JSPI for concurrency explicitly, avoiding the CPS transformation Go's WASM backend currently relies on.
- **Threading is still not supported.** Each Worker runs in a single thread; the Web Worker API is not available. Multi-threaded WASM patterns (`wasi-threads`, `pthread`-emulation) do not work in Workers. This constraint remains unchanged since 2018 and is architectural (isolate-per-request), not a gap.

What this tier buys:

- **Typed linear memory.** The WASM module has its own linear memory (`WebAssembly.Memory`), which is typed in the sense that the compiler that produced the WASM knows the exact byte layout. This is where BAREWire codecs, BitNet model inference, crypto operations, and other byte-precise numeric work belong.
- **Near-native performance.** WASM in V8 runs at ~near-native speed for the right workloads (numeric computation, bit manipulation, dense linear algebra). Significantly faster than equivalent JavaScript for these patterns.
- **Type safety across the FFI boundary.** The WASM module's exported and imported function signatures are typed. The JavaScript/WASM boundary carries explicit type information in both directions; values crossing the boundary are either primitives (i32, f64, externref) or the caller explicitly serializes them.
- **Cloudflare-native deployment.** A WASM Worker is deployed through the same Workers infrastructure as a JavaScript Worker. Wrangler builds it, `wrangler deploy` (or Fidelity.CloudEdge's `cfs` CLI) uploads it.

What this tier costs:

- **Copy-in/copy-out overhead at the JS/WASM boundary.** WASM operates in its own linear memory, separate from the JavaScript heap that holds the Worker's request/response objects, environment bindings, and KV/R2/Queue values. Any data moving between JavaScript and WASM has to be copied across the boundary. For a Worker that spends most of its time shuffling small values between Cloudflare bindings and the response — a routing decision, a header check, a KV lookup — the copy cost can dominate whatever the WASM saves in compute. Cloudflare's 2018 guidance named this explicitly: "Code that mostly interacts with external objects without doing any serious 'number crunching' likely does not benefit from WASM." The implication for Fidelity is that the right Tier 2 candidates are code paths that pull a single batch of data into WASM memory, do substantial compute on it, and return a single result — not code paths that make many small round trips across the boundary.
- **Cold start grows.** WASM modules have to be instantiated on cold start. For small modules, the overhead is negligible; for large modules (multi-MB), it is significant. Cloudflare caches compiled WASM between requests for the same Worker, but the first hit after a deploy pays the full cost.
- **No stack switching currently.** Workers' isolates run WASM without the stack-switching feature enabled. This means WASM in Workers today must use the DCont-via-Coroutines strategy for continuation-bearing code. The WAMI pathway targeting Workers would fall back to state machines.
- **Same isolate constraints.** WASM in a Worker is still inside a V8 isolate. No SharedArrayBuffer across isolates; no traditional filesystem; no raw sockets; I/O still goes through Cloudflare-provided bindings.
- **Size and compile budget.** Cloudflare enforces module size and compile-time budgets. Very large WASM modules (hundreds of MB) do not fit; extremely complex modules may exceed the compile-time limit.

This tier is the natural home for code that needs the type-carrying properties WASM provides but doesn't need a full Linux process. BitNet inference, BAREWire codec hot paths, JSON/binary parsers with auditable byte handling, cryptographic operations — all of this belongs here when the constraints fit.

### Tier 3 — Containers

Cloudflare Containers run actual Linux processes. Full POSIX environment, filesystem, `mmap(2)`, shared memory via `/dev/shm`, arbitrary binaries. Cold start is measured in seconds (though Cloudflare's Container runtime has warm-pool optimizations); memory is measured in gigabytes; compute-intensive workloads run without the isolate's per-request constraints.

Strengths: full Linux, full memory mapping, any binary.
Constraints: slowest cold start, highest per-request cost, different request model (not per-invocation isolated).

Container workloads talk to Workers through the new egress-interception APIs (Container's `interceptOutboundHttp`, `interceptOutboundHttps`) and through BAREWire-framed messages over HTTP/WebSocket when typed messaging is required. The [03_four_wings.md](../javascript-targeting/03_four_wings.md) doc touches on this; it's where BAREWire's cross-substrate invariant (same byte layout from native `memref` lowering and from JavaScript `DataView` ops) earns its keep.

### The Graduation

The three tiers form a monotonic progression on several axes:

| Axis | Pure JS Worker | WASM in Worker | Container |
|:-----|:--------------|:---------------|:----------|
| Cold start | ~5 ms | ~5–50 ms | ~seconds |
| Memory limit | 128 MB | 128 MB (same isolate) | Multi-GB |
| Type carrying | ArrayBuffer/TypedArray only | Full WASM typed memory | Full native process |
| Memory mapping | None across isolates | Linear memory within module | Full `mmap(2)` |
| Compute density | Highest | High | Moderate |
| Per-request cost | Cheapest | Low | Moderate |
| Deployment model | Script upload | Module + shim upload | Image deploy |
| DCont strategy | State machines only | State machines (today); stack switching when Cloudflare enables it | State machines or stack switching, depending on runtime config |

A project's compute profile determines where it fits on this graduation. Latency-sensitive request handlers with modest compute go in Tier 1. Numeric-heavy or byte-precise code goes in Tier 2. Full applications with filesystem, arbitrary binaries, or extensive memory mapping go in Tier 3. A single project can span multiple tiers, with BAREWire framing the communication between them.

### Why This Tier Matters to Composer

The JSIR-based path emits JavaScript for Tier 1. The LLVM WASM and WAMI paths emit WebAssembly for Tier 2. The LLVM native path emits ELF for Tier 3 (a Container running a native Fidelity binary). All three paths share CCS, PSG, Baker, and most of Alex; they diverge at the backend and in which witnesses Alex registers.

From a Composer architecture standpoint, WASM in Workers is not a new backend — it is the LLVM WASM pathway (or WAMI, when Stack Switching lands in Workers) with a Cloudflare-specific deployment target profile. The WASM compilation pressure is the same as for any WASM target; the Cloudflare-specific aspects are size/compile budget enforcement, the JavaScript shim that imports the module, and the Workers-specific bindings the WASM imports (`env.KV.get` becomes a function imported from the JavaScript shim, not a direct WASM call).

## The Converging Story: Stack Switching as the Universal Path

The conservative framing in this document — "LLVM WASM + Coroutines for now, WAMI + Stack Switching as horizon 2/3 investment" — is accurate for 2026 engineering decisions but understates the trajectory. The WebAssembly Stack Switching proposal is moving toward universality, and the proposal's own explainer enumerates the use cases: *coroutines, async/await, generators, lightweight threads, and other advanced non-local control flow idioms*. That list is Fidelity's DCont unification almost verbatim. The WebAssembly community and Fidelity are converging on the same abstraction from different directions.

**Current phase status (April 2026):**

- The Stack Switching proposal advanced to Phase 2 in August 2024.
- Reference interpreter implementation exists.
- Wasmtime has a production-grade implementation (x86_64 Linux initially), with reported 6x performance improvements on micro-benchmarks compared to CPS-transformation fallbacks.
- Phase 3 advancement is in progress pending upstream integration and browser implementation commitments.
- WASIp3 (released early 2026) uses JSPI rather than full Stack Switching because JSPI is universally available today, but the WASIp3 design is explicitly compatible with the broader Stack Switching proposal's feature set.
- JSPI (the subset implementable via JavaScript Promise Integration) is already shipping in V8 and in Cloudflare Workers' WASM runtime.

**The implication for Fidelity.** Stack Switching is not a niche capability that will serve one corner of WASM deployment. It is the standards-track answer for continuation support across every WASM runtime that will matter — browsers, Node.js, Wasmtime, WasmEdge, Cloudflare Workers, embedded WASM runtimes. Once Phase 4 completes and implementations ship broadly (timeline plausible for 2027, aggressive cases possible sooner), *delimited continuations become a runtime-native capability everywhere WASM runs*.

Fidelity's DCont dialect, the WAMI lowering from DCont to stack-switching ops, and the actor/async/computation-expression unification through DCont are all positioned for this convergence. When the proposal reaches broad availability, Composer's WASM output does not need a strategy change; the same DCont ops lower to the same stack-switching instructions, and the runtime carries the continuation work that was previously reified into state machines.

**What this changes in the near-term plan.** Three things:

1. **WAMI is not "horizon 3 research."** It is the pathway that naturally receives the universal standard when it lands. The work to integrate WAMI's `SsaWasm`/`Wasm` dialects and define the DCont-to-stack-switching lowering should be scoped as horizon-2 engineering, not deferred as research.
2. **JSPI is the bridge worth using now.** Any runtime that ships JSPI (Workers today, browsers behind flags or shipped) can already do suspend/resume at Promise boundaries. The `wasm-bindgen-futures` pattern demonstrates this for Rust; Composer can apply the same pattern for Clef. Async-at-Promise-boundaries is a proper subset of DCont semantics, and targeting that subset through JSPI gives runtime-native continuation handling today for the most common async shape.
3. **The "coroutines fallback" framing is temporary.** The LLVM WASM + Coroutines pathway is the right answer for runtimes that never receive Stack Switching support (legacy embedded WASM, some constrained edge runtimes). For mainstream deployments, the coroutine strategy is transitional. Code that is written against Clef's DCont surface today will lower through coroutines where it must and through Stack Switching where it can, and neither the source nor the PSG codata needs to know which strategy is active.

**The delimited-continuation-everywhere observation.** When Stack Switching ships broadly, every WASM deployment inherits delimited continuations as a native capability. This reaches further than just Fidelity's targets: any language that compiles to WASM gets DCont-shape async without the CPS transformation Go currently performs, and any JavaScript environment that can host WASM with Stack Switching gets the same. Fidelity is not unique in wanting this; the WebAssembly community has committed to delivering it as a universal standard.

From Composer's position, this convergence is the strongest possible validation of the DCont unification thesis. The framework picked the abstraction the WebAssembly standards process is settling on independently. The architecture does not need to bet on Fidelity's choice being right in isolation; it needs to position the tooling to receive the standard when it lands.

## Concurrency Is Not Parallelism (and Why Single-Threaded Stack Switching Makes Sense)

The WebAssembly Stack Switching explainer uses language like "concurrent task execution" that can be misread as requiring multiple threads. It does not. This section exists because the confusion is common and because getting it right matters for understanding what Cloudflare Workers can and cannot do with Stack Switching — and, more broadly, for understanding how Fidelity's DCont model fits into Workers' single-threaded isolate constraint.

**Cloudflare Workers are single-threaded, and that has not changed.** Each Worker runs in one thread inside a V8 isolate. The Web Worker API is not available. WebAssembly threading (`wasi-threads`, pthread emulation) is not supported. `SharedArrayBuffer` is not exposed across Worker invocations. This is architectural, not a gap — the per-request isolation model depends on it.

**"Concurrency" in the Stack Switching explainer is cooperative concurrency, not parallelism.** The explainer's task scheduler example runs `$task_0` through `$task_n` on a single thread, interleaved via voluntary suspension:

- The scheduler picks a task.
- The task runs until it hits a suspension point (an I/O op, a yield, a `cont.suspend`).
- Control returns to the scheduler, which picks another task — possibly the same one if it's ready to resume, possibly a different one.
- Many tasks make progress over time. None of them execute simultaneously on separate cores. One thread, interleaved computation.

This is what generators, coroutines, async/await, and green threads all are. None of them require threads. All of them require suspension primitives. Stack Switching is the suspension primitive that WASM was previously missing; with it, WASM code can participate in the same cooperative-concurrency model that JavaScript already has through Promises and the event loop.

The distinction is actually three-way, not two-way, because *data parallelism* (SIMD) is a third category that lives on its own axis: same instruction, many data elements, one thread. SIMD is supported in Cloudflare Workers. Task parallelism (multiple threads) is not.

| Property | Concurrency | Task Parallelism | Data Parallelism (SIMD) |
|:---------|:-----------|:-----------------|:------------------------|
| Multiple tasks in flight | Yes | Yes | No (one task, many elements) |
| Multiple cores simultaneously | Not required | Required | Not required (single core SIMD units) |
| Needs OS threads | No | Yes | No |
| Supported in Cloudflare Workers | Yes | No | Yes |
| Mechanism in JS today | Event loop, Promises, async/await | — | `Uint8Array`-style ops; limited native SIMD surface |
| Mechanism in WASM today | JSPI (at Promise boundaries) | — | WebAssembly SIMD (`v128` type and ops) |
| Mechanism in WASM future | Stack Switching | — | Unchanged |
| Example | Coroutines, goroutines on one thread | pthread, Web Workers | Matrix multiply, pixel ops, BitNet inference, SIMD'd codec loops |

Data parallelism does not suffer from the constraints task parallelism does because it does not require threads. A single thread executing SIMD instructions processes 4, 8, or 16 elements per operation using the CPU's vector registers. The programmer is not managing threads, the isolate is not sharing memory across isolates, and the Worker's single-threaded model is not violated. SIMD is just a better instruction choice within the same thread.

This matters because SIMD is genuinely common in the workload shapes where Tier 2 pays off. Numeric kernels, codec hot paths, and bit-packed low-precision inference are all naturally SIMD-friendly. The section below on Composer's memory marshaling expands on why this is a particularly strong combination.

**Why this matches Fidelity's DCont unification.** The DCont dialect provides delimited continuations — suspend/resume primitives that express cooperative concurrency. A Clef actor inside a Durable Object can handle many in-flight operations (awaiting messages, storage reads, inter-DO fetches) on the DO's single thread. The actor does not need parallelism to have concurrency; it needs suspension. DCont provides that. Stack Switching provides it at the WASM level. The two abstractions match.

**What this means for Workers architecture decisions.** When a workload genuinely needs *task parallelism* — multiple cores executing different code simultaneously — the answer is not "WASM in a Worker." The answer is "Containers" (Tier 3), where real OS threads and full process semantics are available, or splitting the workload across multiple Workers that communicate via BAREWire frames. WASM-in-Worker does not lift the single-thread constraint; it lets WASM code participate in the same cooperative concurrency that JavaScript already has, and it adds access to data parallelism through SIMD.

A Worker with a WASM module that performs suspension-heavy logic (many outstanding BAREWire decodes, many in-flight Promise-based I/O operations, generator-style processing of a streaming input) benefits from Stack Switching and JSPI once they arrive. A Worker that needs to run eight independent CPU-bound tasks simultaneously does not; that workload needs Tier 3 or workload partitioning across multiple Workers. A Worker that processes many *elements of the same computation* — dense numeric loops, byte-level parallel codec operations, low-bit inference — benefits from SIMD regardless of the task-parallelism question, and that benefit is available today. Keeping these three categories distinct in architectural conversations prevents overpromising what WASM-in-Worker can do and underpromising what it can do.

## SIMD and Composer's Memory Marshaling

WebAssembly SIMD is the one form of parallelism Cloudflare Workers supports fully, and it is a much more significant capability than its single-thread status implies. A `v128` vector operation processes four 32-bit lanes, eight 16-bit lanes, or sixteen 8-bit lanes per instruction — a 4x, 8x, or 16x multiplier on the single-thread throughput of the underlying compute. For the workload shapes described in [01_type_carrying_and_workloads.md](./01_type_carrying_and_workloads.md) (codec hot paths, dense numerics, bit-packed inference), this multiplier is the point.

The advantage is amplified in Composer compared to what a hand-written JavaScript Worker or a typical C/C++/Rust WASM module can reach, because Composer owns the memory layout end-to-end.

**Why layout ownership matters for SIMD.** SIMD operations have strong preferences about how their input data is arranged. Aligned loads (16-byte aligned for `v128`) are faster than unaligned. Structure-of-arrays (SoA) layout, where the same field across many records is contiguous in memory, is faster than array-of-structures (AoS), where each record occupies contiguous memory but the same field across records is strided. Packing multiple low-bit values into a single lane (e.g., 16 ternary BitNet weights into a v128) depends on the layout knowing the bit pattern at compile time.

In hand-written code, the programmer has to arrange data this way deliberately, often fighting the language's default memory layout. In a compiler that owns the layout — which Composer does, via the PSG's memory-layout elaboration and the codata-driven memref lowering — the layout can be chosen at compile time based on how the data will be consumed.

**Three specific synergies:**

1. **Alignment is automatic.** Composer can emit memory allocations with the SIMD-friendly alignment the target operation wants, without the programmer annotating alignment anywhere in the source. If a particular loop will consume a buffer via `v128` loads, Composer allocates the buffer on a 16-byte boundary. The programmer wrote `let buffer = Array.create(size)` in Clef; Composer emitted the aligned allocation. This is the kind of thing Rust handles with `#[repr(align(16))]` and C handles with `alignas(16)` — both of which require the programmer to think about alignment. Composer's codata-driven layout means the programmer does not have to.

2. **AoS/SoA transposition is a compilation decision.** For a given data structure, whether the in-memory layout should be array-of-structures or structure-of-arrays depends on how the data is consumed. A structure with a SIMD-heavy consumer typically wants SoA, because the SIMD loop reads one field from many records, which SoA makes contiguous. A structure with a record-at-a-time consumer typically wants AoS, because random access to full records matches AoS naturally. Composer has enough type information and dataflow information to pick the right layout per consumer. The programmer writes a discriminated union or a record in Clef; Composer lays it out in memory according to how it will be used. This is a compile-time optimization that hand-written code rarely reaches because it requires restructuring source at both definition and use sites.

3. **Bit-packed low-precision work is a natural target.** BitNet's 1.58-bit (ternary) weights pack sixteen weights into a 32-byte vector, or more aggressively, many weights into a single `v128`. The packing and unpacking operations are bit-level shuffles that SIMD supports natively (WebAssembly SIMD has `i8x16.shuffle`, `i16x8.narrow_*`, and similar ops). Composer knows the bit-width from Clef's dimensional types, knows the packing strategy from the codata, and emits the SIMD shuffles directly. The alternative is hand-written SIMD intrinsics in Rust or C++, which is productive but very specific to each packing scheme. Composer can generate the right shuffle sequence for any packing Clef's type system describes.

**Concrete workloads that combine well:**

- **BitNet inference in a Cloudflare Worker.** 1.58-bit weight packing, SIMD-accelerated ternary arithmetic, dense matrix-vector multiplication. The compute pattern is exactly what WASM SIMD was built for. Composer's dimensional types express the weight precision; the memory marshaling emits the packed layout; the witness-level SIMD emission produces v128 ops that are more aggressive than a hand-written Rust BitNet implementation would typically reach without substantial hand-tuning.
- **BAREWire batch decode.** A stream of BAREWire frames, each one a structured message with fixed-layout fields. In a naive codec, each frame is decoded one at a time. In a SIMD-aware codec, multiple frames can be decoded in lockstep where the structure allows — header parsing across N frames simultaneously, length-prefix extraction across N frames, common-case field reads across N frames. Composer's codec-derivation infrastructure can emit the SIMD-batched form when the consumer pattern allows it.
- **Media and image processing.** Resizing, format conversion, filter application, compression preprocessing. All have dense per-pixel loops that SIMD vectorizes naturally. A Composer-emitted pixel processing kernel on Cloudflare Workers is a reasonable candidate to outperform an equivalent in hand-written WASM, specifically because Composer's layout ownership removes the alignment and AoS/SoA concerns the hand-written code has to manage.
- **Cryptographic primitive acceleration.** Block cipher inner loops, hash function rounds, byte-permutation-heavy crypto. WebAssembly SIMD has the `i8x16` and `i32x4` operations that these kernels need. Composer emitting these from Clef source (assuming the crypto primitive is expressed in Clef rather than bound from a C library) gives the SIMD acceleration without the programmer building it by hand.

**Where Composer's advantage is most distinctive.** Not in outright instruction count — a competent hand-written Rust SIMD implementation is usually close to optimal for a single kernel. Composer's advantage is *consistency across the codebase*: every hot path that touches memory gets the same layout-aware, SIMD-capable treatment without the programmer having to rewrite each one. A large Fidelity application deployed to Cloudflare Workers Tier 2 would see SIMD utilization across the board (codec paths, numerical kernels, bit-level operations) in a way that a mixed-language application built from hand-tuned pieces usually does not.

**This is a Tier 2 capability that does not exist in Tier 1 at all.** JavaScript cannot emit SIMD instructions directly — V8 does some auto-vectorization of JavaScript arithmetic, but it is opportunistic, not predictable, and does not handle structured layouts or bit-packing. A JavaScript Worker doing BitNet inference is simply slower than a WASM Worker doing BitNet inference, by a factor that depends on the kernel but is typically an order of magnitude or more. The decision to move a workload from Tier 1 to Tier 2 should weight this heavily: Tier 2 is not just "the same code, slightly faster"; for SIMD-amenable workloads, Tier 2 is a categorically different performance class.



## Axis Triangulation: The Matrix

Pulling the three axes together:

| Pathway × Strategy × Deployment | Works today? | Notes |
|:---------------------------------|:-------------|:------|
| LLVM WASM + Coroutines + Pure JS Worker (JS-side calls into WASM) | Yes | Primary path for numeric hot paths in current Workers |
| LLVM WASM + Coroutines + WASM-in-Worker | Yes | Tier 2 of Cloudflare's step-grade, today |
| LLVM WASM + Coroutines + Container | Yes | Container running a WASM runtime (Wasmtime); unusual but supported |
| LLVM WASM + Coroutines + Browser | Yes | Standard browser WASM target |
| LLVM WASM + Stack Switching + * | Partial | LLVM's stack-switching support is partial; V8 behind flags, not Cloudflare |
| WAMI + Coroutines + Pure JS Worker | Future | Pathway exists in research; Composer integration is horizon 2/3 |
| WAMI + Coroutines + WASM-in-Worker | Future | Same; no DCont advantage over LLVM WASM without stack switching |
| WAMI + Stack Switching + Wasmtime | Future | Wasmtime has experimental stack-switching support |
| WAMI + Stack Switching + WASM-in-Worker | Future (gated on Cloudflare) | The payoff scenario: WAMI preserving DCont structure into Cloudflare's WASM runtime |
| WAMI + Stack Switching + Browser | Future (gated on engine) | V8 stack-switching behind flag; Firefox and Safari not yet shipped |

The "future" rows are not abstract. The WebAssembly Stack Switching proposal is in Phase 3 as of early 2026, which means it is on track for standardization. Cloudflare tracks WebAssembly proposals closely and has historically enabled features soon after they reach Phase 4. Composer's architecture is preparing for the scenario where Stack Switching is broadly available: the WAMI pathway is being scoped, the DCont dialect already exposes the right primitives, and the target-profile mechanism can switch strategies without requiring source code changes.

## Where BAREWire Fits

BAREWire is the wire format. It does not care which tier the sender or receiver occupies; it guarantees byte-identical layout for a given schema across all compilation paths.

- Native ELF binaries (Tier 3) encode BAREWire frames through `memref` operations lowered to native code.
- WASM modules (Tier 2) encode BAREWire frames through `memref` operations lowered to WASM linear memory.
- JavaScript Workers (Tier 1) encode BAREWire frames through `DataView` operations on `ArrayBuffer`s.

All three lowerings derive from the same PSG discriminated union definition. A `CounterMsg.Add(42)` serialized in a Container, in a WASM-in-Worker, or in a pure JS Worker produces identical bytes. This is what makes the step-graded compute model tractable: each tier can hold the piece of the application that fits it, and the tiers communicate through BAREWire frames without ambiguity about what the bytes mean.

The trust argument from [Fidelity.CloudEdge/docs/08b_actor_core.md §3.4](../../../Fidelity.CloudEdge/docs/08b_actor_core.md) applies uniformly: the sender's compiler verified the schema, the codec encoded the verified structure into bytes, the wire preserves the bytes, the receiver's codec reconstructs the structure. The runtime tier at either end is irrelevant to the trust chain.

## Implications for Downstream Agent Work

**1. Treat pathway and strategy as separate target-profile axes.** A Cloudflare Worker WASM target today is `pathway=llvm, strategy=coroutines`. A future Workers target with stack switching enabled is `pathway=llvm, strategy=stack-switching` or `pathway=wami, strategy=stack-switching`. The `.fidproj` target declaration should expose both axes; Composer should make the pathway decision explicit rather than implicit.

**2. Prioritize LLVM WASM + Coroutines as the near-term production path.** WAMI is research-grade. Stack switching is gated on runtime support. The near-term deployments to Cloudflare Workers, browsers, and standalone WASM runtimes should all use the LLVM WASM pathway with coroutine lowering. This gets Fidelity WASM to production-viable status without depending on experimental features.

**3. Scope WAMI integration as horizon-2 engineering, not horizon-3 research.** The Stack Switching proposal is a standards-track convergence with Fidelity's DCont unification; the WAMI pathway is how Composer receives that standard when it ships. The engineering work — integrating `SsaWasm`/`Wasm` dialects into Composer's MLIR pipeline, defining the DCont-to-stack-switching lowering, building target-profile capability detection — should begin in 2026 with the intent to have production-ready WAMI output by the time Stack Switching reaches Phase 4 broad availability. Near-term Cloudflare deployments are not gated on WAMI (LLVM WASM + Coroutines covers the immediate need), but the WAMI investment is the strategic bet, and it is a bet on a standards process that has already signaled its intent.

**4. BitNet and similar density candidates belong in Tier 2.** The [bitnet-fidelity-integration-analysis.md](../../../bitnet-fidelity-integration-analysis.md) material examines BitNet's compute profile. Low-bit model inference with tight numeric loops is exactly what WASM-in-Worker is built for: typed memory, near-native SIMD, cacheable compiled modules. The architecture should treat this as a Tier 2 deployment, not a Container deployment, unless the model size exceeds Tier 2's budget.

**5. Containers are the escape hatch for things the isolate can't do.** Full filesystem access, arbitrary binary execution, memory mapping, shared memory through `/dev/shm`, process-level APIs — these live in Tier 3. The framework should not pretend these can be emulated in Tier 1 or Tier 2; when they're needed, Container is the right answer, and BAREWire is how the Container talks to the Worker layer.

**6. Stack Switching advancement is a capability to track at the ecosystem level.** The proposal's phase transitions, browser and Wasmtime implementation milestones, and Cloudflare's adoption timeline are material for Composer's target-profile logic. A decision infrastructure (similar to how Fidelity.CloudEdge tracks workers-types releases) that watches Stack Switching availability across runtimes should be part of the WASM targeting story. JSPI adoption in specific runtimes is also worth tracking separately, since it is a shipping subset of the full capability and is where the immediate DCont-native wins live.

**7. The JavaScript wing inherits this convergence.** JavaScript targeting through JSIR (see [javascript-targeting/](../javascript-targeting/)) today emits JavaScript that uses async/await and Promise chains. Once Stack Switching ships broadly in browsers (JSPI is already partial here), applications that compile JavaScript-targeted Clef to deploy-to-WASM targets instead gain native DCont handling without source changes. The JavaScript targeting work and the WASM targeting work converge on the same DCont abstraction via different runtime mechanisms.

## Cross-References

- [javascript-targeting/](../javascript-targeting/) — the sister subfolder for JavaScript (Tier 1) targeting
- [javascript-targeting/03_four_wings.md](../javascript-targeting/03_four_wings.md) — the four deployment wings, of which Cloudflare (Tiers 1–3) is the largest
- [Async_LLVM_Coroutines.md](../Async_LLVM_Coroutines.md) — the current coroutine lowering for DCont
- [Actor_Substrate_Independence.md](../Actor_Substrate_Independence.md) — the cross-substrate actor model that spans the three tiers
- [Composer/.serena/memories/delimited_continuations_architecture.md](../../.serena/memories/delimited_continuations_architecture.md) — the DCont unification argument
- [Composer/.serena/memories/coroutines_as_dcont_substrate.md](../../.serena/memories/coroutines_as_dcont_substrate.md) — why coroutines are the fallback substrate for DCont on constrained targets
- [Fidelity.CloudEdge/docs/10_jsir_strategic_assessment.md](../../../Fidelity.CloudEdge/docs/10_jsir_strategic_assessment.md) §3 — WAMI precedent citation
- WAMI paper: https://arxiv.org/html/2506.16048v1
- WebAssembly Stack Switching proposal: https://github.com/WebAssembly/stack-switching
