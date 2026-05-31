# The Four JavaScript Wings

**SpeakEZ Technologies | Fidelity Framework**
**April 2026**

JavaScript targeting at Composer is not a single deployment story. Four distinct contexts consume the JavaScript the framework emits, and each has its own runtime environment, SDK surface, and deployment model. Cloudflare Workers is the largest of the four by API surface area and current strategic weight, but the architecture does not privilege it. The same compilation mechanics (either model from [01_two_models.md](./01_two_models.md)) serve all four.

This document names the four wings, describes what's common and what differs, and positions Composer's responsibilities relative to each.

## Wing 1: Cloudflare Workers and Durable Objects

**Runtime:** Cloudflare's Workers runtime (a V8-based isolate with a custom runtime API), Durable Objects, Containers, Facets.

**SDK surface:** `@cloudflare/workers-types` (TypeScript definitions for the runtime API), plus the REST management APIs for provisioning D1 databases, KV namespaces, R2 buckets, Queues, and so on. Fidelity.CloudEdge generates F# bindings for both surfaces — runtime types in the Runtime tier, management clients in the Management and Tenancy tiers.

**Deployment:** Worker scripts deployed via the Cloudflare API (not via Wrangler when Fidelity is driving it — Fidelity.CloudEdge's `cfs` CLI uploads Workers through the Management Layer's REST clients). Durable Objects are instantiated by the runtime on demand; Facets are instantiated from within a DO.

**Composer's role:** Emit JavaScript that runs inside the Workers runtime and calls the Workers API. Emit Durable Object classes (Babel `ClassDeclaration` with `ClassMethod` members for `fetch`, `webSocketMessage`, `alarm`, etc.). Emit BAREWire codecs for actor message types. Emit `fetch` handler modules for the Worker entry point.

**Composer does not:** Rebuild the Workers runtime. Rebuild Durable Objects. Rebuild KV, R2, D1, Queues, Vectorize, or any other Cloudflare service. All of those are consumed as runtime dependencies from the JavaScript Composer emits.

**Current mechanism:** F#/.NET model via Fable, with F# bindings from Fidelity.CloudEdge. The fully-decomposed-AST model via JSIR is in development under Horizon 2.

## Wing 2: Browser UI (Reactive Frontends)

**Runtime:** The browser's JavaScript engine (V8 in Chrome/Edge, SpiderMonkey in Firefox, JavaScriptCore in Safari). Reactive UI frameworks layered on top: SolidJS, React, Svelte.

**SDK surface:** Framework runtime libraries (`solid-js`, `react`, `svelte`) from npm. DOM APIs exposed by the browser. Fetch API, WebSocket API, WebTransport API (where supported), IndexedDB, service workers.

**Deployment:** Static assets served via HTTP, typically bundled by Vite, esbuild, or equivalent. Composer produces the JavaScript modules; the bundler composes them with framework runtime libraries and app-level glue.

**Composer's role:** Emit JavaScript that calls into the framework of choice. In the F#/.NET model, Partas.Solid handles the JSX surface and Fable compiles to Solid runtime calls. In the fully-decomposed-AST model, Clef's reactive surface (first-class signals, effects, memos) lowers through Alex into JSIR ops that represent the post-JSX-transform JavaScript. Both models emit code that consumes `solid-js` (or the equivalent for another framework) from npm.

**Composer does not:** Reimplement SolidJS's fine-grained reactivity, React's fiber scheduler, Svelte's compiler output, or any other framework's runtime. The framework is a runtime dependency, not a Composer-owned artifact.

**Signal-actor relationship:** The fully-decomposed-AST model has an interesting architectural property here. Signals are a specialized actor topology — a signal is a minimal actor that holds a value and broadcasts changes to subscribers. Clef's unified actor model can surface both ordinary actors (for Cloudflare DOs and native OS actors) and signal-actors (for reactive UI) from the same source-language construct, with target-profile-driven lowering deciding which runtime representation is emitted.

**Current mechanism:** F#/.NET model via Fable + Partas.Solid. The fully-decomposed-AST model via JSIR is under investigation; whether Composer pursues a Clef-native signals surface is a strategic decision still in flight.

## Wing 3: WebView Desktop (WrenHello)

**Runtime:** A native desktop binary that embeds a WebView (WebKit on Linux via WebKitGTK, WebView2 on Windows, WKWebView on macOS). The WebView runs JavaScript as the frontend; the native binary provides the backend and IPC boundary.

**SDK surface:** Platform WebView APIs exposed through the native side (GTK, WebKit C headers on Linux). Standard browser APIs inside the WebView. Custom IPC bridge (typically BAREWire over a local channel) between the JavaScript frontend and the native backend.

**Deployment:** A single native binary (~17KB in WrenHello's current form) that embeds the JavaScript bundle as rodata and launches the WebView on startup. No separate installation of the JavaScript frontend; it ships inside the binary.

**Composer's role:** Emit both sides of the application from one Clef codebase. The native side compiles through the LLVM backend to ELF. The JavaScript side compiles through the JSIR backend (or Fable, for existing F# frontends). The shared protocol module, defined once in Clef, compiles both ways — LLVM produces the native BAREWire codec (via `memref` ops), JSIR produces the JavaScript BAREWire codec (via `DataView` ops). The byte layout is guaranteed identical because both derive from the same PSG discriminated union definition.

**Composer does not:** Reimplement WebKit, GTK, or any WebView. Those are consumed as runtime/system dependencies of the native binary. It also does not build a new desktop application framework; WrenHello's "framework" is the convention of "native binary + embedded WebView + BAREWire IPC," not a runtime library.

**Key distinction from Wing 2:** Wing 2's JavaScript runs in a browser that the user's OS installed. Wing 3's JavaScript runs in a WebView that the native binary spawns. Both ultimately use a browser engine; the deployment and lifecycle differ.

**Current mechanism:** F#/Fable for the JavaScript side in WrenHello today. The fully-decomposed-AST model via JSIR is a natural fit for desktop applications because cross-substrate compilation (native + WebView from one source) is already how WrenHello's build works.

## Wing 4: Shared BAREWire (Cross-Cutting Typed Messaging)

**Runtime:** Whatever environment the sender and receiver happen to be running in. BAREWire is not a runtime itself — it is a wire format. The BAREWire codec code runs in every wing and in the native process too.

**SDK surface:** Not applicable in the same sense as the other wings. BAREWire's "surface" is the codec functions (encode/decode pairs) emitted for each discriminated union type. These are ordinary JavaScript functions in the JavaScript wings and ordinary native functions in LLVM targets.

**Deployment:** Emitted inline into whatever module uses the codec. A Worker that exchanges BAREWire-framed messages with a native cluster has the codec compiled into its JavaScript bundle. The native cluster has the same codec compiled into its ELF binary. The wire format on the WebSocket (or MoQ stream) between them is byte-identical.

**Composer's role:** Derive codec implementations from discriminated union type definitions in the PSG. Emit them through both the LLVM pipeline (as `memref`-based byte manipulation) and the JSIR pipeline (as `DataView`/`ArrayBuffer`-based byte manipulation). Preserve the BAREWire invariant: any given DU's byte layout is determined by the PSG definition, not by the lowering target.

**Composer does not:** Define the BAREWire format. BAREWire is documented independently (see `08b_actor_core.md` §3.4 in Fidelity.CloudEdge for the trust argument across the JavaScript erasure boundary). The BAREWire patent (USPTO Provisional 63/786,247) describes the format. Composer's role is to honor the specification during code generation, not to define it.

**Trust argument:** In the F#/.NET model, BAREWire is what carries type safety across the boundary between statically typed F# and the dynamically typed JavaScript runtime. The sender's compiler verified the discriminated union structure; the codec encoded that structure into the byte layout; the wire format preserves the structure in transit; the receiver's codec reconstructs the structure on arrival. The JavaScript runtime never has to reason about the types — it just encodes and decodes bytes according to a schema both ends agreed on at compile time. The fully-decomposed-AST model has the same property by different mechanics: Composer's type-checker verified the DU, the JSIR-lowered codec encodes the same byte layout, and the receiving side reconstructs identically.

## What's Common Across All Four

**Emission substrate:** JavaScript, produced by whichever of the two models ([01_two_models.md](./01_two_models.md)) is in use. In the fully-decomposed-AST model, all four wings use the same `jsir_gen --passes=hir2ast,ast2source` pipeline; they differ in which witnesses Alex registers, which SDK types the code imports, and which deployment artifact format wraps the final `.js`.

**Runtime library consumption:** All four consume JavaScript libraries from npm as runtime dependencies. Composer does not reimplement those libraries.

**BAREWire-style typed messaging:** All four can use BAREWire when cross-boundary messaging is needed. Workers talking to native clusters use it. Browser UI talking to Workers uses it (indirectly, through Worker-mediated message forwarding). Desktop frontends talking to desktop backends use it. Native clusters talking to each other use it.

**Alex's witness architecture:** Elements, Patterns, Witnesses. Parallel nanopass execution. Read-only coeffects. The architecture doesn't care which wing is the target; it processes the PSG and emits ops. The target-specific variation is in the witness registration.

## What Differs

**Runtime environment:** V8 isolate vs. browser V8 vs. embedded WebKit. API surfaces differ accordingly.

**SDK surface that Clef bindings describe:** `@cloudflare/workers-types` for Wing 1, `solid-js` (and DOM) for Wing 2, `@webkit2gtk/*` and custom IPC protocols for Wing 3, BAREWire schema definitions for Wing 4.

**Deployment model:** Upload via Cloudflare API; static asset hosting; embedded rodata in a native binary; inline codec emission. The compiler emits `.js`; what happens to the `.js` varies.

**Baker saturation strategy:** Reactive UI preserves signal/effect calls; Cloudflare Worker code preserves async/await patterns and Response construction; BAREWire codec emission is fully saturated because it's a straightforward byte-manipulation problem. The per-target Baker recipe exclusion list varies.

**Consumer role:** Who ultimately uses the output. Wing 1 serves traffic. Wing 2 renders UI. Wing 3 is part of an installed desktop application. Wing 4 is plumbing inside the other three.

## Why Cloudflare Looks Like the Architecture

Cloudflare is the largest of the four wings by several measures:

- **API surface:** Fidelity.CloudEdge 0.2.0 ships F# bindings for 40 Management-tier services and 2 Tenancy-tier services, plus Runtime types for Workers, Durable Objects, Containers, and Facets. That's an order of magnitude more surface area than the other wings combined.
- **Opinionation:** Cloudflare's runtime has strong opinions about how code is structured (modules with default exports, DO classes with specific method names, `fetch` handler conventions). The emission has to respect those conventions precisely.
- **Current reach:** Fidelity.CloudEdge is the primary target audience for the framework's JavaScript emission story today. MoQ support landed in April 2026. Agents Week landed shortly after. Each announcement adds to the SDK surface.

All of that is real, and it deservedly gets attention. But it is one wing of four, and the architectural decisions Composer makes about how JavaScript is emitted are not Cloudflare-specific. The same witness architecture, the same JSIR pipeline, the same Baker selective saturation, serves all four wings. Cloudflare is a large consumer of Composer's JavaScript output. It is not the designer of Composer's JavaScript emission architecture.

## Cross-References

- [01_two_models.md](./01_two_models.md) — the F#/.NET vs. fully-decomposed-AST distinction across all four wings
- [02_jsir_tooling.md](./02_jsir_tooling.md) — the `jsir_gen` invocation that services the fully-decomposed-AST path
- [04_sdk_describes_runtime.md](./04_sdk_describes_runtime.md) — the SDK-as-description convention that applies to every wing
- [Fidelity.CloudEdge/docs/08b_actor_core.md](../../../Fidelity.CloudEdge/docs/08b_actor_core.md) §3.4 — BAREWire trust argument (Wing 4 context)
- [WebView_Desktop_Architecture.md](../WebView_Desktop_Architecture.md) — Wing 3 specifics
- [Actor_Substrate_Independence.md](../Actor_Substrate_Independence.md) — cross-wing actor compilation (Wings 1, 3, and 4)
