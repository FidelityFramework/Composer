# Build, Bundle, and Execution Models: WASM Beyond Cloudflare

**SpeakEZ Technologies | Fidelity Framework**
**April 2026**

[02_deployment_and_lifecycle.md](./02_deployment_and_lifecycle.md) covered how WASM deploys specifically on Cloudflare Workers. This document zooms out: how Composer builds a WASM artifact in the first place, how that single artifact adapts to different deployment substrates, and what kinds of execution models the resulting WASM instance can embody. The architectural interest is at the top end of this stack — the "unikernel thinking" that continuation-based execution enables, and the break point where a single WASM instance stops being enough and multiple instances need to coordinate.

The unifying observation: Composer's WASM emission is substrate-agnostic. The same `.wasm` byte sequence can run in Cloudflare Workers, a browser, Wasmtime, WasmEdge, a WAMR embedded runtime, or a Component-Model-capable host. What varies is the *bundle* that wraps the WASM and makes it deployable to a specific host. This separation — target-agnostic compilation, target-specific bundling — is what makes the execution-model discussion useful across substrates rather than stuck on one.

## The Target-Agnostic Build Pipeline

Both Composer pathways to WASM — LLVM WASM and WAMI — converge on the same output format: a `.wasm` module that conforms to the WebAssembly specification and uses the feature set the target runtime supports. A Composer build for WASM looks conceptually the same regardless of pathway:

```
Clef source (.clef)
  │
  ▼
CCS → PSG → Baker → Alex
  │ (target profile selects WASM witnesses)
  │
  ▼  (pathway branch)
  │
  ├── LLVM WASM:   Alex → MLIR → LLVM IR → llc --target=wasm32
  │
  └── WAMI:        Alex → WAMI dialects (SsaWasm/Wasm) → direct WASM emission
  │
  ▼  (common)
.wasm module (WASM MVP + feature set matching target runtime profile)
  │
  ▼
wasm-opt (Binaryen) size and speed optimization pass
  │
  ▼
Final .wasm ready for target-specific bundling
```

The pathway choice (LLVM vs. WAMI) is an internal Composer decision, driven by the target profile's feature requirements. LLVM WASM is the default for runtimes without stack-switching support; WAMI is preferred for runtimes with it. From outside Composer's compilation boundary, the output is the same kind of artifact either way.

### LLVM WASM Pathway Specifics

The LLVM pathway is the well-understood route: Alex emits `func`/`memref`/`arith`/`scf` ops, the standard MLIR lowering passes produce LLVM IR, and LLVM's WASM backend emits the `.wasm` output. The target triple is either `wasm32-unknown-unknown` (for runtimes where WASI imports are not expected or are provided by the host-specific shim, which is Cloudflare Workers' situation) or `wasm32-wasi` (for standalone WASI runtimes).

Key tools:
- `llc` with `--target=wasm32-unknown-unknown` or `--target=wasm32-wasi`
- `wasm-ld` for final linking of WASM modules
- `wasm-opt` (from Binaryen) for size and performance optimization
- Optional `wasm-bindgen`-style glue generation where JS-host interop is needed (the bundle layer, not the build layer)

Composer's LLVM WASM build is mechanically similar to Rust's `cargo build --target wasm32-unknown-unknown`. The tooling is battle-tested; the work Composer does is on the MLIR → LLVM IR side (Alex's witnesses and Baker's saturation choices), not on the LLVM → WASM side.

### WAMI Pathway Specifics

The WAMI pathway stays within MLIR's infrastructure all the way to WASM emission. The WAMI project (arxiv.org/html/2506.16048v1) defines MLIR dialects — `SsaWasm` and `Wasm` — that represent WebAssembly directly. Alex emits into the existing Clef-native dialects (`func`/`memref`/`arith`/`scf`); a lowering pass converts those to `SsaWasm` and `Wasm` ops; a final emission step writes the MLIR module to `.wasm` bytes.

Key tools (speculative, based on WAMI's structure):
- An MLIR binary (either `mlir-opt` with WAMI extensions, or a dedicated `wami-opt`) for running the lowering passes
- A WAMI translation tool that reads the final MLIR and emits `.wasm`
- `wasm-opt` for post-emission optimization (same as the LLVM pathway)

The WAMI pathway does not currently include a production-grade all-the-way-from-high-level-MLIR-to-WASM toolchain; integrating it into Composer is the horizon-2 engineering work. Once in place, the key capability WAMI adds over LLVM WASM is direct lowering from DCont ops to WebAssembly stack-switching ops — the continuation primitives survive the pipeline without reification into state machines.

### WASI Considerations

WASI (WebAssembly System Interface) defines a standardized set of syscall-equivalent imports that WASM modules can use: file I/O, environment variables, random number generation, process operations, networking (in newer revisions). A WASM module compiled with `wasm32-wasi` imports these functions from the host runtime.

For Cloudflare Workers specifically, WASI's role is limited. Cloudflare Workers exposes its own set of imports (the bindings described through the JS shim), not WASI. A Clef WASM module targeting Workers should use `wasm32-unknown-unknown` and call out to the JS shim for any I/O-like operation, rather than assuming WASI syscalls exist.

For other WASM runtimes, WASI matters more:
- **Wasmtime** supports WASI fully, including the newer WASIp3 (released early 2026) which uses JSPI-style concurrency instead of CPS transformation.
- **WasmEdge** supports WASI and has additional extensions for AI/ML workloads, sockets, and more.
- **WAMR** (embedded) supports a subset of WASI suitable for constrained devices.
- **Component Model hosts** increasingly use WASI Preview 2/3 as their standard interface layer.

Composer's WASI handling is a target-profile decision: if the target runtime expects WASI imports, emit `wasm32-wasi` output and ensure Clef's runtime expectations (Clef stdlib operations, actor I/O) map cleanly to WASI calls. If the target provides a different import set (Cloudflare's bindings, browser APIs, custom embedded host), emit `wasm32-unknown-unknown` and let the bundling layer specify the import mapping.

For Clef specifically, WASI is less critical than for languages with POSIX-expecting standard libraries (Rust, Go). Clef's stdlib is built around dimensional types, coeffects, and BAREWire — none of which have POSIX analogues. The I/O surface is supplied by the target host (bindings in Cloudflare, DOM APIs in browsers, Wasmtime's WASI when targeting standalone runtimes). This means Composer can target WASI where it helps and skip it where the host provides its own imports, with relatively little ceremony.

## Target-Specific Bundling

The bundle is what wraps a `.wasm` module and makes it deployable to a specific host. The bundle shape varies substantially by target, even though the WASM module inside is the same:

| Target | Bundle Contents | Host Integration |
|:-------|:----------------|:-----------------|
| Cloudflare Workers | JS shim + .wasm + wrangler.toml | JS shim instantiates WASM, exposes bindings as imports |
| Browser | HTML + JS host + .wasm + optional wasm-bindgen glue | JS host fetches and instantiates, calls exports |
| Node.js | .mjs entry + .wasm + package.json | ES module imports WASM, uses Node.js WASI when needed |
| Wasmtime (standalone) | .wasm + optional config + CLI invocation | `wasmtime run module.wasm` |
| WasmEdge | .wasm + AOT-compiled .wasm option | `wasmedge module.wasm` with extensions for AI/networking |
| Embedded (WAMR, wasm3) | .wasm linked into host firmware + runtime library | WASM runtime library statically linked; module loaded at boot or dynamically |
| Component Model | .wasm + .wit interface + adapter modules | Host instantiates component, wires up imports/exports per WIT |

In every row, the core `.wasm` is the same artifact from Composer's perspective. What differs is the shell around it — the code that loads the WASM, wires it up to the host runtime's services, and gives it a lifecycle.

For Composer's build tooling, this implies a two-layer architecture:

1. **WASM emission layer.** Produces the `.wasm` module from Clef source. Target-agnostic in output; target-aware in compilation (the target profile informs which feature set, which import expectations, which DCont strategy).
2. **Bundling layer.** Wraps the `.wasm` for a specific host. A Cloudflare Worker bundler generates the JS shim and wrangler.toml; a browser bundler generates the HTML/JS host; a standalone bundler packages the `.wasm` with an invocation script.

The Fidelity equivalent of `worker-build` (Rust's Cloudflare-specific bundler) should live in the bundling layer. Call it `fidelity-worker-build` provisionally; its input is the Composer-produced `.wasm` plus a set of import declarations; its output is a deployable Worker bundle. `cfs` (the Fidelity.CloudEdge CLI) is the natural host for this tool, since it already handles Worker deployment through the Cloudflare Management API. For other targets, analogous bundlers would exist — `fidelity-browser-build`, `fidelity-wasmtime-build`, etc. — but the WASM emission layer they all consume is shared.

## Unikernel Execution Model

This is where the architecture gets interesting, and where Composer's distinctive capabilities compound.

A unikernel, as the term is normally used, is a single-purpose OS image: the application and its kernel are fused into one bootable image with no extraneous OS services. WrenHello is Fidelity's native unikernel exemplar — a 17KB binary that contains everything needed to run the application, with no separate operating system underneath.

**A WASM module can be a unikernel in the same sense**, if the execution model inside the module is self-contained enough to not require rich host services. Composer's combination of features — static typing, DCont unification, stack-switching-ready continuation semantics, compiler-owned memory layout, and BAREWire for structured data — makes this genuinely viable rather than aspirational.

The elements:

**The "kernel" is the DCont scheduler.** In conventional programming, a kernel provides threads, scheduling, synchronization primitives, I/O orchestration, and memory management. A WASM module with stack-switching support can provide all of these except threads (which require host cooperation) within its own linear memory and continuation machinery. The DCont dialect's `shift`/`reset`/`resume` primitives lower to stack-switching ops (WAMI pathway) or state machines (LLVM pathway); either way, the WASM module holds a run queue of continuations, a scheduler that selects the next continuation to resume, and the suspension points where cooperative yields happen. This is a cooperative scheduler written in Clef, lowered through Alex, emitted as WASM — not a library the programmer bolts on.

**Memory is compiler-managed.** The memory-marshaling discussion from [README.md](./README.md)'s SIMD section — layout ownership, alignment control, AoS/SoA transposition, bit-packed representations — applies to the unikernel case too. The "heap manager" inside the WASM module doesn't need to be general-purpose because Composer knows the allocation patterns at compile time. Arena allocation, region-based memory, stack allocation where lifetimes fit — all of this is compiler-planned rather than programmer-managed. For many Fidelity workloads, there is no dynamic heap at all; memory usage is statically bounded because the type system's dimensional types and escape analysis have resolved the lifetimes.

**I/O is specific, not general.** The unikernel doesn't need a generic file system or socket API. It needs exactly the imports its actual workload uses: BAREWire frame send/receive, storage binding reads, maybe an HTTP dispatch. These are narrow, typed imports the host provides. No POSIX emulation, no WASI layer, no fallback glue — just the specific entry points the Clef code actually uses.

**The module is self-scheduling.** When a request arrives (or an event, or a timer, or whatever the host's activation event is), the module's entry point doesn't immediately dispatch to a single function. It hands the request to its internal scheduler, which runs the appropriate continuation, which suspends when it needs I/O, which gets resumed when the I/O completes. Multiple requests can be in flight (cooperative concurrency via the DCont-scheduler pattern); they interleave on the single thread the host provides.

**What this means for Fidelity on Cloudflare Workers Tier 2:**

A Fidelity Worker that embraces the unikernel execution model is a WASM module containing:
- A complete actor runtime (Olivier actors, Prospero supervisors) scheduled by DCont continuations
- BAREWire codecs for all message types used
- The application logic as a graph of actors communicating through BAREWire frames
- A thin interface to the host's imports: fetch requests in, responses out, bindings accessed as specific import calls

The Worker's JS shim becomes minimal — it just instantiates the WASM module, forwards the fetch request into the module's entry point, awaits the response, and returns it to Cloudflare. Essentially all of the logic is WASM; the JavaScript is a pure bridge.

This is not how most WASM-using Workers are structured today. Most Rust-via-`workers-rs` Workers still delegate substantial orchestration to the JavaScript shim and invoke WASM only for specific compute. A Fidelity unikernel-shape Worker puts the orchestration in WASM too, with the JS shim reduced to the absolute minimum the Cloudflare runtime requires.

**What this means beyond Cloudflare:**

The same unikernel pattern applies anywhere WASM runs. A Wasmtime-hosted Clef module behaves the same way — its entry point is a Wasmtime invocation rather than a Cloudflare fetch, but the internal structure (DCont scheduler, actor runtime, BAREWire codecs) is identical. Deploying to a browser WebAssembly context, the internal structure doesn't change; only the host's import set changes. Deploying to an embedded WAMR runtime on a microcontroller, same story.

This is the payoff from having a compiler that owns the execution model. Clef code targeting "WASM" produces a self-contained runtime per module; the module adapts to different hosts via the bundling layer without the internal structure needing to vary.

## The Parallelism Break Point

Single-instance unikernel thinking takes you a long way. But there's a break point where a single WASM instance stops being enough, and the architecture has to shift from "one instance with cooperative concurrency" to "multiple instances with coordination." Understanding where that break point is matters for architectural decisions about how to structure a Fidelity deployment.

**What a single WASM instance can do (the concurrency ceiling):**
- Arbitrarily many cooperative concurrent tasks via DCont scheduling. Limited only by the module's memory budget for continuation state.
- Data-parallel operations via SIMD, 4x–16x throughput per instruction within a single thread.
- Event-driven orchestration with suspension at every I/O point, giving high overlap of I/O and compute.
- Request pipelining where the incoming event stream drives many concurrent tasks.

**What a single WASM instance cannot do (the ceiling):**
- True simultaneous execution on multiple CPU cores. One instance lives in one thread.
- Compute density beyond what a single core sustains. If the hot path saturates one core and there is more work to do, cooperative concurrency cannot help — you are computation-bound, not concurrency-bound.
- Geographic distribution of work. One instance runs in one place.
- Fault isolation between independent work units. If one cooperative task in the instance corrupts shared state, other tasks in the same instance see the corruption.

**The break point is compute saturation of a single core combined with independence of work units.** When you have:

1. More CPU work than one core can do in the available wall time, and
2. Work units that can be independently assigned to separate executors (not requiring shared mutable state),

...then the architecture should fan out to multiple WASM instances. Coordination between them happens through BAREWire-framed messages over whatever transport the deployment substrate provides.

**The continuity is important: fan-out is architecturally continuous with single-instance cooperative concurrency.** A Clef actor that sends a BAREWire-framed message to another actor doesn't care whether the other actor is in the same WASM instance (DCont-dispatched, in-memory message delivery) or a different WASM instance (network-delivered, cross-process message). The actor's code is the same. The BAREWire schema is the same. The dispatch mechanism underneath differs, but the actor-level semantics are uniform.

This is the `ActorRef<'Msg>` abstraction from [Fidelity.CloudEdge/docs/08a_actor_model_overview.md](../../../Fidelity.CloudEdge/docs/08a_actor_model_overview.md): local dispatch when the target is in the same instance, edge dispatch (DurableObjectStub.fetch / WebSocket) when the target is in a different Worker or DO, remote dispatch (network) when the target is in a different cluster entirely. The compiler resolves the dispatch shape based on where the target actor is deployed; the programmer writes the same code in all cases.

**Target-specific fan-out mechanisms:**

| Deployment substrate | Fan-out mechanism | Coordination transport |
|:---------------------|:------------------|:----------------------|
| Cloudflare Workers | Multiple Worker invocations (concurrent fetch) + Durable Objects for stateful coordination | BAREWire over HTTP/WebSocket/MoQ; DO storage for state |
| Wasmtime / Wasmtime-based service | Multiple Wasmtime processes, orchestrated by external scheduler | BAREWire over sockets |
| Browser | Web Workers (different OS threads), each running its own WASM instance | BAREWire via `postMessage` with Transferables |
| Embedded / IoT | Multiple WASM instances on different devices | BAREWire over whatever link layer the devices use |
| Component Model | Multiple component instances composed by the host | BAREWire via component-level interfaces |

The pattern is uniform: each "executor" is a WASM instance, each instance runs a DCont-scheduled Clef runtime, coordination happens via BAREWire messages, and the coordination transport adapts to whatever the substrate provides.

**When a deployment should fan out:**

- When BitNet inference or similar dense compute exceeds one-core throughput and the workload admits batch partitioning.
- When a streaming pipeline has stages that can run in parallel (e.g., decode → process → encode), each becoming its own instance.
- When geographically distributed actors need local compute for low latency (Cloudflare's edge network makes this natural).
- When fault isolation matters — one instance's crash shouldn't take down the whole application.
- When the problem is embarrassingly parallel and the coordination overhead is much less than the compute savings.

**When a deployment should not fan out:**

- When the workload fits comfortably in one core. Overhead from fan-out (message serialization, dispatch, coordination) can exceed the benefit for small workloads.
- When work units have tight coupling (shared mutable state, fine-grained synchronization). Cooperative concurrency in one instance handles this better than distributed coordination.
- When the problem is bound by I/O rather than compute. More instances doing more I/O doesn't help if the I/O backend is the bottleneck.

## Implications for Downstream Work

**1. Keep the WASM emission layer target-agnostic.** Composer should produce a `.wasm` artifact without baking in any Cloudflare-specific or substrate-specific assumptions at the emission level. Target-specific concerns (import shapes, bundle layout, lifecycle events) belong in the bundling layer, which is per-substrate.

**2. Design the bundling layer to be pluggable.** `fidelity-worker-build` for Cloudflare Workers, `fidelity-browser-build` for browsers, `fidelity-wasmtime-build` for Wasmtime — each is a thin tool that consumes the `.wasm` and emits a target-specific bundle. The common layer underneath each is `cfs` or a similar deployment CLI that knows how to upload to whatever registry the target uses (Cloudflare API, browser bundler output, container registry, etc.).

**3. The unikernel execution model should be the recommended shape for Fidelity WASM deployments.** Not the only shape, but the recommended one. A WASM module that contains its own DCont scheduler, actor runtime, BAREWire codecs, and minimal host-import surface is easier to reason about, port across substrates, and scale via fan-out than a fragmented module that relies heavily on host-provided orchestration. Documentation, tooling, and examples should default to this shape.

**4. Fan-out should be an ActorRef-level decision, not a code-level decision.** The Clef programmer writes actors that exchange messages. The compiler and deployment configuration decide whether those actors run in the same instance (DCont-dispatched locally) or different instances (BAREWire-dispatched remotely). The source code should not change when the deployment topology changes.

**5. Beyond Cloudflare matters for the framework's generality.** Fidelity.CloudEdge is the current visible deployment, but the same WASM emission and unikernel execution model should work on Wasmtime (for private/on-prem deployments), on browsers (for serverless-in-the-client patterns), on embedded runtimes (for IoT), and on Component Model hosts (for composable microservice architectures). The strategic value of the compilation work is proportional to how many of these substrates the same artifacts actually serve.

**6. The Stack Switching convergence described in [README.md](./README.md) is what makes all of this clean.** Today, the DCont scheduler inside a WASM module has to be implemented via coroutine-style state machines. Once Stack Switching ships broadly, the scheduler becomes runtime-native — the module uses `cont.new`, `cont.suspend`, `cont.resume` primitives directly, and the runtime does the scheduling work. The unikernel thinking doesn't require Stack Switching to be viable, but Stack Switching makes the implementation much cleaner and the performance substantially better.

## Cross-References

- [README.md](./README.md) — pathway/strategy/deployment triangulation; includes the SIMD and Stack Switching convergence discussion that this document builds on
- [01_type_carrying_and_workloads.md](./01_type_carrying_and_workloads.md) — the workload-selection rubric that determines whether WASM (and thus this execution-model discussion) is relevant for a given piece of code
- [02_deployment_and_lifecycle.md](./02_deployment_and_lifecycle.md) — Cloudflare-specific deployment mechanics, which this document generalizes beyond
- [Fidelity.CloudEdge/docs/08a_actor_model_overview.md](../../../Fidelity.CloudEdge/docs/08a_actor_model_overview.md) — the actor model that the unikernel execution pattern and the fan-out coordination build on
- [Composer/.serena/memories/delimited_continuations_architecture.md](../../.serena/memories/delimited_continuations_architecture.md) — the DCont unification that makes the continuation-as-scheduler pattern viable
- WrenHello — the native unikernel precedent that this document's WASM unikernel thinking draws from
