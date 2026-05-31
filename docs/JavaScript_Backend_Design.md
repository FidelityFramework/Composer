# Composer JavaScript Backend

**SpeakEZ Technologies | Fidelity Framework**
**March 2026 | Horizon 2 Design Document**

> **Revision note (April 2026).** This document predates the JSIR adoption decision. Sections describing Alex witnesses emitting JavaScript AST or JSX source directly reflect the pre-JSIR design. The current design uses JSIR as the emission dialect; see [JSIR_Backend_Design.md](./JSIR_Backend_Design.md) for the superseding specification. The witness-level architectural invariants (Elements/Patterns/Witnesses, parallel nanopass) remain correct, but the emission target changed from JavaScript AST nodes to `jsir.*` / `jshir.*` MLIR ops. The subfolder [javascript-targeting/](./javascript-targeting/) covers the two coexisting targeting models (F#/.NET via Fable; fully decomposed AST via JSIR) and the four JavaScript deployment wings.

## 1. Context

Composer compiles Clef source through CCS → PSG → Baker → Alex → MLIR, then channels MLIR through backend-specific toolchains to produce target artifacts. The LLVM backend produces native binaries. The CIRCT backend produces FPGA bitstreams via Vivado synthesis. The MLIR-AIE backend produces NPU kernels via Peano's specialized LLVM.

This document specifies a fourth backend: **JavaScript**. The JS backend compiles Clef source to deployable JavaScript and JSX, replacing Fable's role in the Fidelity Framework. The output flows through standard web toolchains (bundlers, minifiers) the same way CIRCT output flows through Vivado, producing deployment-ready artifacts for Cloudflare Workers, WREN Stack desktop applications (via WebView), and browser-based front-ends.

This document is one half of the Horizon 2 plan. The companion document, *Farscape TypeScript and OpenAPI Ingestion Roadmap*, describes how Clef bindings for JavaScript libraries are generated. Both are required for Fidelity.CloudEdge to migrate from F#/Fable to native Clef.

The companion document *Actor Substrate Independence* provides the detailed code trace demonstrating that a single Clef actor definition compiles to both native (MLIR/LLVM) and Cloudflare (JavaScript/Durable Objects) substrates through different witness configurations. The actor model is the primary use case that validates the JavaScript backend's architecture: one source, two compilation targets, identical actor semantics, BAREWire-compatible on the wire.

### Horizon Context

**Horizon 1** (current, 8-12 days): Xantham replaces Glutinum within the existing F#/Fable pipeline. Tactical.

**Horizon 2** (this document + companion): Composer gains a JavaScript backend. Farscape produces Clef bindings for TypeScript and OpenAPI. Fable exits the pipeline entirely. Strategic.

**Horizon 3** (future): Composer Transcribe absorbs foreign implementations. TypeScript algorithms become native Clef. Visionary.

## 2. Architectural Position

### 2.1 Alex as Witness Library

Alex is the PSG witness library. It observes the saturated PSG via zipper traversal, and witnesses emit target representations through XParsec pattern matching and coeffect consultation. Today, every witness emits MLIR operations because every existing backend consumes MLIR. The Elements/Patterns/Witnesses stratification is a general observation-and-emission framework; the emission of MLIR is an implementation detail of the current witness set, not an architectural constraint.

The JavaScript backend adds a parallel set of Elements, Patterns, and Witnesses to Alex that emit JavaScript AST or source instead of MLIR operations. The same zipper traversal, the same XParsec combinators, the same coeffect system. The architectural invariants hold:

- Witnesses MUST NOT call other witnesses (parallelism preserved)
- Witnesses MAY share Elements and Patterns (common vocabulary)
- Witnesses MUST return `WitnessOutput.skip` for unhandled nodes
- All state is read-only coeffects (no mutable shared state)

### 2.2 Where Alex Produces MLIR and Where It Doesn't

Alex always produces MLIR for native targets. For the JavaScript target, Alex produces JavaScript. This is not a violation of Alex's role; Alex is the witness library, and witnesses emit whatever the target requires. The MLIR-specific SSA assignment pass and memref-oriented coeffect analysis are skipped or replaced with lighter analysis appropriate for JS emission.

MLIR can still appear in the JavaScript backend's downstream toolchain. If a Clef project targets both native and JavaScript (e.g., WrenHello's backend compiles to native while its frontend compiles to JS), the same PSG is observed by different witness configurations for each target. Alex runs twice, once per target, with different witness registrations.

### 2.3 Backend Pipeline Position

```
CCS → PSG → Baker → Alex (JS witnesses)
                          ↓
                    JavaScript/JSX source
                          ↓
                    Bundler (Vite, esbuild, Wrangler)
                          ↓
                    Deployment artifact
```

Parallel to:

```
CCS → PSG → Baker → Alex (MLIR witnesses)
                          ↓
                    Portable MLIR
                          ↓
                    mlir-opt / CIRCT / Peano
                          ↓
                    Native binary / FPGA bitstream / NPU kernel
```

Both paths share CCS, PSG construction, and Baker saturation. They diverge at Alex's witness registration.

## 3. The Elaboration Depth Question

### 3.1 Baker's Role for JavaScript

Baker saturates the PSG by expanding higher-order patterns into primitive operations. For native targets, this is exhaustive: `List.map` becomes an explicit recursive loop, closures become struct pairs with code pointers and captured environments, pattern matching becomes conditional branches.

For JavaScript, exhaustive saturation is counterproductive for certain constructs:

- **Higher-order functions**: `Array.map(f)` should survive as `array.map(f)` in the JS output, not become an explicit loop. JavaScript engines optimize HOF calls aggressively; the explicit loop is often slower.
- **Reactive primitives**: Partas.Solid's `createSignal`, `createEffect`, and component construction must survive as recognizable calls. SolidJS's fine-grained reactivity depends on seeing these primitives at the framework level.
- **Closure semantics**: JavaScript closures are native. Baker's expansion of closures into struct pairs with code pointers and environments produces correct but unidiomatic JavaScript. The JS backend should emit native closures.

Other Baker saturations remain valuable:

- **Intrinsic elaboration**: Expanding `Console.writeln` into the appropriate JS `console.log` call.
- **Partial application**: Resolving curried functions into concrete parameter lists that map to JS function signatures.
- **Pipe operator reduction**: `x |> f |> g` becomes `g(f(x))`, which is correct JS.

### 3.2 Selective Saturation

Baker's saturation becomes target-aware. The `.fidproj` target declaration informs Baker which saturations to apply:

| Saturation | LLVM/CIRCT/AIE | JavaScript |
|:-----------|:--------------:|:----------:|
| Intrinsic elaboration | Yes | Yes (to JS equivalents) |
| Pipe operator reduction | Yes | Yes |
| Partial application resolution | Yes | Yes |
| HOF expansion to explicit loops | Yes | **No** (preserve HOFs) |
| Closure struct pair construction | Yes | **No** (use native closures) |
| Match expansion to conditionals | Yes | Configurable |
| Reactive primitive expansion | N/A | **No** (preserve signals/effects) |

This is not a new architectural concept. Baker already operates as a series of discrete nanopass recipes (Fan-Out/Fold-In). The JS target simply excludes certain recipe categories from the saturation pass.

### 3.3 Elaboration Depth as Project Configuration

The `.fidproj` target configuration determines elaboration depth:

```toml
[compilation]
target = "js"

[compilation.js]
# High-level: preserve framework idioms, minimal saturation
# Used for: Cloudflare Workers, Partas.Solid UI, library interop
mode = "reactive"

# OR

# Low-level: saturate HOFs, optimize for throughput
# Used for: D3 visualization pipelines, WebGPU compute, data processing
mode = "computational"
```

Alternatively, Composer infers the elaboration strategy from the project's dependency declarations. A project depending on `Partas.Solid` activates reactive-preserving witnesses. A project depending on `Fidelity.D3` or `Fidelity.WebGPU` activates computational witnesses. A project like Fidelity.CloudEdge that's primarily binding interop gets the lightest touch.

## 4. JavaScript Witness Architecture

### 4.1 Elements Layer

JavaScript Elements are atomic JS emission operations, parallel to the existing MLIR Elements:

```
Alex/
  Elements/
    ArithElements.fs          # Existing: MLIR arith dialect
    MemRefElements.fs         # Existing: MLIR memref dialect
    CFElements.fs             # Existing: MLIR control flow
    SCFElements.fs            # Existing: MLIR structured control flow
    ...
    JsExprElements.fs         # New: JS expression emission
    JsStmtElements.fs         # New: JS statement emission
    JsModuleElements.fs       # New: JS import/export emission
    JsxElements.fs            # New: JSX component emission
```

Elements are `module internal`. Witnesses cannot import them directly; they compose through Patterns.

### 4.2 Patterns Layer

JavaScript Patterns compose Elements into JS-specific idioms:

```
Alex/
  Patterns/
    ApplicationPatterns.fs    # Existing: MLIR function call patterns
    ClosurePatterns.fs        # Existing: MLIR closure pair patterns
    MemoryPatterns.fs         # Existing: MLIR memref patterns
    StringPatterns.fs         # Existing: MLIR string literal patterns
    ...
    JsApplicationPatterns.fs  # New: JS function call, method chain
    JsClosurePatterns.fs      # New: JS arrow functions, native closures
    JsStringPatterns.fs       # New: JS template literals, string ops
    JsReactivePatterns.fs     # New: SolidJS signal/effect patterns
    JsxComponentPatterns.fs   # New: JSX element construction
    JsInteropPatterns.fs      # New: import/export, FFI boundary
```

### 4.3 Witnesses Layer

JavaScript Witnesses observe PSG nodes and delegate to JS Patterns:

```
Alex/
  Witnesses/
    LiteralWitness.fs         # Existing: MLIR literals
    ArithIntrinsicWitness.fs  # Existing: MLIR arithmetic
    ApplicationWitness.fs     # Existing: MLIR function calls
    ...
    JsLiteralWitness.fs       # New: JS literal emission
    JsArithWitness.fs         # New: JS arithmetic (respecting float semantics)
    JsApplicationWitness.fs   # New: JS function calls, method invocation
    JsBindingWitness.fs       # New: const/let bindings
    JsControlFlowWitness.fs   # New: if/else, for, while, switch
    JsLambdaWitness.fs        # New: arrow functions
    JsModuleWitness.fs        # New: import/export statements
    JsInteropWitness.fs       # New: [<JsImport>], [<JsInterface>] handling
    JsReactiveWitness.fs      # New: SolidJS signal/effect primitives
    JsxWitness.fs             # New: JSX component rendering
```

### 4.4 Witness Registration

The witness registry activates JS witnesses based on the `.fidproj` target:

```fsharp
match target with
| "cpu" | "native" ->
    // Register MLIR witnesses (existing behavior)
    registerMLIRWitnesses registry
| "fpga" ->
    registerMLIRWitnesses registry
    |> registerCIRCTWitnesses
| "npu" ->
    registerMLIRWitnesses registry
    |> registerAIEWitnesses
| "js" | "javascript" ->
    // Register JavaScript witnesses
    registerJavaScriptWitnesses registry
```

## 5. Coeffect Analysis for JavaScript

The existing coeffect analysis pipeline includes SSA assignment, mutability analysis, escape classification, and string table construction. For the JavaScript target, some of these are unnecessary and others need adaptation:

| Coeffect | LLVM Target | JavaScript Target |
|:---------|:------------|:------------------|
| SSA Assignment | Required (MLIR SSA form) | Not required (JS uses `const`/`let`) |
| Mutability Analysis | Required (memref.alloca vs. SSA) | Lightweight (mutable → `let`, immutable → `const`) |
| Escape Classification | Required (stack vs. heap allocation) | Lightweight (closures are native; GC handles allocation) |
| String Table | Required (global memref dedup) | Not required (JS strings are interned by the engine) |
| Yield State Tracking | Required (state machine for seq) | Adapted (JS generators or async iterators) |
| Pattern Binding | Required (conditional branch setup) | Adapted (destructuring assignment) |

A lighter coeffect pass for JavaScript targets would skip SSA assignment and string table construction, perform simplified mutability and escape analysis, and focus on information that improves JS output quality: identifying pure functions (for potential inlining), detecting unused bindings (for tree-shaking hints), and classifying reactive dependencies (for SolidJS signal tracking).

## 6. JS Interop Attribute Handling

Farscape produces Clef source with JS interop attributes. The JavaScript witnesses in Alex interpret these attributes at the FFI boundary:

| Attribute | Witness Behavior |
|:----------|:----------------|
| `[<JsImport("name", "module")>]` | Emit `import { name } from "module"` |
| `[<JsImportDefault("module")>]` | Emit `import name from "module"` |
| `[<JsInterface>]` | Emit as JS object type; no runtime representation |
| `[<JsStringEnum>]` | Emit string literal type; values are string constants |
| `[<JsErase>]` | Type-level only; erased at runtime (union types) |
| `[<JsEmit("expression")>]` | Inline JavaScript expression verbatim |

The `JsInteropWitness` observes PSG nodes with these attributes and delegates to `JsInteropPatterns`, which compose `JsModuleElements` for import statements and `JsExprElements` for inline expressions.

## 7. Output Targets

### 7.1 JavaScript (ES Modules)

Standard ES module output for Node.js, Cloudflare Workers, and bundler consumption:

```javascript
import { fetch, Request, Response } from "@cloudflare/workers";

export default {
    async fetch(request) {
        // Compiled Clef application code
    }
};
```

### 7.2 Reactive UI and JSX

> **Revised (April 2026):** JSIR has no JSX operations. This section is superseded by [JSIR_Backend_Design.md §8.3](./JSIR_Backend_Design.md) and [javascript-targeting/01_two_models.md](./javascript-targeting/01_two_models.md).

JSX is a source-level surface. It is handled by whichever source compiler sits upstream of JSIR:

- **F#/.NET path.** F# authors use Partas.Solid bindings; Fable transforms JSX into Solid runtime calls before any MLIR is involved. JSIR is not in this pipeline.
- **Fully decomposed AST path.** A Clef reactive surface (first-class signals, effects, memos) lowers through Alex. The JSX transform happens in Clef's front end; Alex witnesses emit JSIR ops that represent the post-transform JavaScript (function calls to `createSignal`, `createElement`, etc.).

In both models, `solid-js` is consumed as a runtime dependency from npm. Composer does not reimplement Solid, and JSIR does not represent JSX.

### 7.3 Downstream Toolchain

The JavaScript/JSX output flows through standard web toolchains:

| Deployment Target | Bundler | Output |
|:-----------------|:--------|:-------|
| Cloudflare Workers | Wrangler | Single JS bundle |
| WREN Stack (WebView) | Vite | Optimized bundle embedded in native binary |
| Browser SPA | Vite / esbuild | Static assets + JS bundles |
| Node.js service | None (ES modules) | Direct execution |

This parallels the existing backend toolchains: CIRCT → Vivado, MLIR-AIE → Peano, LLVM → system linker.

## 8. WREN Stack Integration

The WREN stack (WebView + Reactive + Embedded + Native) is the concrete deployment model for desktop applications. Under Horizon 2:

```
Farscape:  SolidJS .d.ts    → Clef bindings (Partas.Solid foundation)
           GTK .h / WebKit .h → Clef bindings (native backend)

Composer (JS backend):
    Frontend: App.clef + Partas.Solid bindings → JavaScript/JSX
              → Vite → optimized bundle

Composer (LLVM backend):
    Backend: Main.clef + GTK/WebKit bindings → MLIR → LLVM
             → native binary (embeds frontend bundle)

IPC: BAREWire over WebSocket
     Shared Protocol.clef compiled by BOTH backends
     Identical binary encoding guaranteed
```

The WrenHello reference application (currently 17KB native binary + embedded SolidJS frontend) serves as the integration test for the full pipeline.

## 9. Relationship to Fable

Fable compiles F# to JavaScript by walking the F# AST (via FSharp.Compiler.Service) and emitting Babel AST nodes. Composer's JavaScript backend replaces this with a different approach:

| Aspect | Fable | Composer JS Backend |
|:-------|:------|:-------------------|
| Source language | F# | Clef |
| Frontend | FSharp.Compiler.Service | CCS (Clef Compiler Services) |
| Intermediate form | Fable AST | PSG (Program Semantic Graph) |
| Emission model | AST walk → Babel AST → JS source | Witness observation → JS source |
| Optimization | Fable-specific passes | Baker saturation + coeffect analysis |
| Interop attributes | `[<Import>]`, `[<Emit>]`, `[<Erase>]` | `[<JsImport>]`, `[<JsEmit>]`, `[<JsErase>]` |
| Framework | Fable runtime library | No runtime (or minimal, platform-specific) |
| Compilation model | Transpilation (F# AST → JS AST) | Compilation (PSG → JS, js_of_ocaml model) |

The js_of_ocaml precedent is relevant. OCaml's ecosystem has both `ocamlopt` (native compiler) and `js_of_ocaml` (JavaScript compiler). They share the same source language and type system. `js_of_ocaml` was a separate project that became a standard part of the OCaml Platform. Composer follows the same trajectory: one source language (Clef), multiple compilation targets, JavaScript being one of them.

## 10. Design Documents Required

The following documents need to be written to fully specify the JavaScript backend:

1. **JavaScript Elements Specification**: Define the atomic JS emission operations. What does a `JsExprElement` produce? How are source maps tracked? What is the accumulator model for JS (parallel to the MLIR accumulator)?

2. **JavaScript Patterns Catalog**: Enumerate the composable templates for JS idioms. Application patterns, closure patterns, reactive patterns, JSX patterns. Each pattern is ~50 lines, following the existing Pattern convention.

3. **JavaScript Witness Registry**: Define which witnesses activate for the JS target, how they interact with the existing nanopass architecture, and how witness output is collected into JS module structure.

4. **Selective Saturation Specification**: Formalize which Baker recipes apply to the JS target. Define the interface between Baker and the target declaration. This may require refactoring Baker's recipe system to support exclusion lists.

5. **JavaScript Coeffect Analysis**: Specify the lighter coeffect pass for JS targets. What analysis is needed, what can be skipped, and how the results are consumed by JS witnesses.

6. **JS Interop Attribute Semantics**: Complete specification of the `[<Js*>]` attribute vocabulary, their PSG representation, and their witness interpretation. This document is shared with the Farscape companion document.

7. **Output Format and Source Map Specification**: How the JS backend produces ES modules, how source maps trace back to Clef source, how the output integrates with standard bundlers.

8. **WrenHello Migration Guide**: Step-by-step plan for converting WrenHello's frontend from F#/Fable to Clef/Composer, validating the JS backend against a concrete application.
