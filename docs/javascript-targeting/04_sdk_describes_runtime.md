# SDKs Describe the Runtime; Composer Emits Into It

**SpeakEZ Technologies | Fidelity Framework**
**April 2026**

A persistent source of confusion in early Composer architectural discussions was the scope of what the framework actually builds. This document establishes the convention that governs Composer's relationship to every JavaScript runtime it emits into: **SDKs describe the runtime; Composer emits JavaScript that calls it.** Composer does not reimplement any of the runtimes it targets.

## What "SDK" Means Here

An SDK in this context is a set of bindings that describes a runtime's API surface in the source language being compiled. For F#, these are type definitions and function signatures. For Clef, they will be the same kind of thing — type definitions and function signatures, authored in Clef rather than F#.

SDKs describe what the runtime offers. They do not implement it. A Clef binding for Durable Objects describes the `DurableObject` base class, its `fetch` method, its `state.storage` namespace, and so on. It does not implement any of those things. The implementation lives inside Cloudflare's Workers runtime. Composer's emitted JavaScript, at runtime, calls into that implementation.

This is no different from how any typed language consumes a foreign runtime. TypeScript has `@types/node` describing Node.js's API; TypeScript does not reimplement Node.js. Haskell's FFI describes C libraries; Haskell does not reimplement the C standard library. F# bindings for Cloudflare describe the Workers API; F# does not reimplement the Workers API.

## Glutinum and Hawaii Are Not Clef Products

This distinction was muddled in earlier drafts and is worth naming explicitly.

**Glutinum** is an F# tool that generates F# bindings from TypeScript `.d.ts` files. It is a community-maintained .NET/F# ecosystem tool. It has nothing to do with Clef or Composer.

**Hawaii** is an F# tool that generates F# clients from OpenAPI specifications. It is also community-maintained, also part of the .NET/F# ecosystem, and also unrelated to Clef or Composer.

Fidelity.CloudEdge uses both to generate its F# bindings for Cloudflare runtime types (via Glutinum against `@cloudflare/workers-types`) and for Cloudflare management APIs (via Hawaii against the Cloudflare OpenAPI specification). The generated F# code is then consumed by F# applications that compile through Fable.

**Xantham** is the Clef equivalent of Glutinum — a Clef tool that will generate Clef bindings from TypeScript `.d.ts` files. It is a Composer-ecosystem tool.

**Farscape** is the Clef equivalent of Hawaii — a Clef tool that will generate Clef clients from OpenAPI specifications. It is also a Composer-ecosystem tool.

The current state (April 2026):

- F# applications use Glutinum and Hawaii to generate bindings. These applications compile through Fable to JavaScript.
- Clef applications will use Xantham and Farscape to generate bindings. These applications will compile through Composer to JavaScript.

Horizon 1 is about Xantham replacing Glutinum *within the existing F#/Fable pipeline*, because Clef's type system is richer and can produce higher-fidelity bindings than Glutinum can. Horizon 2 is about Farscape producing Clef bindings consumed by Composer. Horizon 3 is about Transcribe absorbing foreign JavaScript implementations into native Clef. None of these horizons involve Composer reimplementing the runtimes that the bindings describe.

## Composer Does Not Rebuild Cloudflare Tooling

The strongest form of this confusion was framing Composer as somehow rebuilding Durable Objects, KV, R2, or Workers. Composer does not do any of that. Cloudflare owns and operates those runtimes. Composer emits JavaScript that runs inside them.

Specifically:

- **Workers runtime:** Cloudflare's V8-based isolate. Composer emits JavaScript modules that the runtime loads and executes. Composer has no runtime of its own inside a Worker.
- **Durable Objects:** Cloudflare's stateful actor runtime. Composer emits Babel-AST-shaped class declarations (`jsir.class_declaration` with `jsir.class_method` children) that the DO runtime instantiates when a request arrives. The DO lifecycle is Cloudflare's; Composer's emitted classes participate in that lifecycle.
- **KV, R2, D1, Queues, Vectorize, Hyperdrive, etc.:** Cloudflare services exposed through bindings that the Workers runtime injects into the module's environment object. Composer emits JavaScript that reads `env.MY_KV` and calls `env.MY_KV.get(key)`. Cloudflare implements the `.get` method. Composer does not.
- **MoQ (Media over QUIC):** Cloudflare's media streaming runtime, surfaced via `@moq/lite`, `@moq/watch`, `@moq/publish`. Composer emits JavaScript that imports those packages. The packages are Cloudflare/IETF-standard runtime code. Composer does not reimplement MoQ.
- **Wrangler / `wrangler`:** Cloudflare's CLI for managing Workers. Fidelity.CloudEdge has its own CLI (`cfs`) that talks to the Cloudflare Management API directly, but this is a management tool (tier 2/3 of the Fidelity.CloudEdge package architecture), not a reimplementation of the Workers runtime. The `cfs` CLI is to Wrangler what the Kubernetes client library is to `kubectl` — a different client for the same API.

The same principle holds for every other wing:

- **SolidJS:** Composer emits JavaScript that imports `solid-js` and calls its exported functions. Composer does not reimplement Solid's fine-grained reactivity.
- **WebKit / GTK (WrenHello):** Composer emits native bindings that call into the platform's C libraries. The native binary links against WebKitGTK; Composer doesn't reimplement the web engine.
- **BAREWire:** Composer emits codec functions that serialize discriminated unions according to the BAREWire specification. Composer doesn't reimplement the wire format — it honors the format specification during code generation.

## Why This Convention Matters

The convention matters because it bounds Composer's scope precisely. The Fidelity Framework is not a "Cloudflare platform competitor" or a "SolidJS replacement" or a "desktop application framework." It is a compiler that targets JavaScript (among other things), and the JavaScript it produces is designed to integrate cleanly with whatever runtime it's deployed into.

Bounding the scope this way has several consequences:

1. **Composer's work stays tractable.** Reimplementing Cloudflare's infrastructure would be an open-ended project with no completion criterion. Emitting JavaScript that correctly calls Cloudflare's infrastructure is a well-defined problem.

2. **Upstream updates come for free.** When Cloudflare ships a new feature (Facets, Container egress interception, snapshot APIs — all landed in April 2026), the response in Composer is to update the binding descriptions and emit code that calls the new APIs. The runtime change is absorbed by Cloudflare; Composer absorbs only the surface change.

3. **The two targeting models coexist naturally.** Both the F#/.NET model and the fully-decomposed-AST model emit JavaScript that calls the same runtimes. A Worker emitted by Fable and a Worker emitted by Composer look different in their internals but call the same `env.MY_KV.get(key)` at runtime. The interop story is simple because the runtime is shared.

4. **The SDK layer is where innovation happens.** Clef's type system is richer than TypeScript's or F#'s in specific ways (dimensional types, coeffects, structural BAREWire integration). Those advantages show up in the bindings themselves — a Xantham-generated Clef binding for `@cloudflare/workers-types` can express things that a Glutinum-generated F# binding cannot. But the underlying runtime is the same, and the advantage is in how Composer's emitted JavaScript uses the runtime, not in a reimplemented runtime.

## What Composer Does Own

Composer owns the compilation path from Clef source to the JavaScript emission that calls into these runtimes. Specifically:

- **The PSG and middle-end elaboration.** Clef-level analysis, coeffect computation, selective saturation. None of this exists in the runtime ecosystem; it's compile-time work Composer performs.
- **The witness architecture.** Alex's Elements/Patterns/Witnesses stratification, parallel nanopass execution, codata-driven extension. This is how Composer decides what JavaScript to emit.
- **The JSIR lowering.** Composer's backend translates PSG nodes to JSIR ops through the witness set. This is new code that doesn't exist in Google's JSIR repository; Google's repository contains JSIR the dialect, and Composer contains the lowering from Clef into that dialect.
- **BAREWire codec derivation.** Given a DU definition in the PSG, Composer derives byte-identical codec implementations for the native and JavaScript targets. The BAREWire specification is external; the codec derivation is Composer's.
- **Cross-substrate invariants.** The guarantee that a BAREWire-framed message serialized by the native compilation and a BAREWire-framed message serialized by the JavaScript compilation are byte-identical. This is enforced by Composer's code generation, not by any runtime.

In short: Composer owns the compile-time work. Runtimes own the runtime work. SDKs are the interface that lets Composer's compile-time work correctly target the runtime's runtime work.

This is the wrap strategy, and it is the permanent right answer for runtime surfaces Composer does not own (Cloudflare's services, the browser DOM, WebKit in a WebView). For JavaScript libraries whose behavior Composer *should* own — supply-chain-sensitive dependencies, crypto primitives, security-critical deserializers — the complementary absorb strategy is described in [05_supply_chain_and_transcribe.md](./05_supply_chain_and_transcribe.md). The two strategies coexist; neither replaces the other.

## The Four Wings Through This Lens

| Wing | Runtime owner | SDK describes | Composer emits |
|:-----|:--------------|:--------------|:---------------|
| Cloudflare Workers/DOs | Cloudflare | `@cloudflare/workers-types`, management OpenAPI | JavaScript Worker modules, DO classes, Facet init code |
| Browser UI | Browser vendors + SolidJS team | DOM APIs, `solid-js` exports | JavaScript that calls createSignal/createElement |
| WebView Desktop | WebKit/GTK teams + platform vendors | WebKit headers, GTK headers, custom IPC protocols | JavaScript (inside WebView) + native ELF (outside); BAREWire between them |
| Shared BAREWire | BAREWire specification | DU schemas | Codec functions in whatever language is in use |

Every row has a "runtime owner" column that isn't Composer. That's the point.

## Cross-References

- [01_two_models.md](./01_two_models.md) — where Glutinum/Hawaii fit (.NET model) vs. Xantham/Farscape (Clef model)
- [02_jsir_tooling.md](./02_jsir_tooling.md) — what Composer invokes; what it doesn't reimplement
- [03_four_wings.md](./03_four_wings.md) — per-wing runtime owners and SDK surfaces
- [Fidelity.CloudEdge/docs/00_architecture_decisions.md](../../../Fidelity.CloudEdge/docs/00_architecture_decisions.md) — the three-tier Fidelity.CloudEdge package architecture (Runtime/Management/Tenancy) that descriptively wraps the Cloudflare runtime
