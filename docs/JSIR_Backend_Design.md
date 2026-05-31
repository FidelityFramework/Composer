# Composer JSIR Backend Design

**SpeakEZ Technologies | Fidelity Framework**
**April 2026 | Horizon 2 Design Document (JSIR Revision)**

## 1. Context

This document revises the JavaScript Backend Design to incorporate Google's JSIR (JavaScript Intermediate Representation), an MLIR dialect for JavaScript with full AST round-trip fidelity. JSIR was proposed for upstream inclusion in MLIR in April 2026. Its availability changes the architecture of JavaScript emission in Composer from direct AST generation to MLIR dialect lowering, aligning the JavaScript backend with the structural pattern established by the AIE and CIRCT backends.

The original `JavaScript_Backend_Design.md` describes a pipeline where Alex's JavaScript witnesses directly emit JavaScript source. That design remains valid as a conceptual specification of *what* the JavaScript output looks like. This document describes *how* the emission happens when JSIR is the intermediate representation between Alex and JavaScript source.

### Relationship to Existing Documents

- **JavaScript_Backend_Design.md**: Specifies the witness architecture (Elements, Patterns, Witnesses), selective saturation, coeffect analysis, and output targets. These remain correct. The change is that witness emission targets JSIR ops, not a JavaScript AST data structure.
- **AIE_Backend_Design.md**: Provides the structural template for this document. The AIE pipeline (CCS → PSG → Baker → Alex → MLIR-AIE → aiecc.py → xclbin) is the closest analogy to the JSIR pipeline (CCS → PSG → Baker → Alex → JSIR ops → `jsir_gen --passes=hir2ast,ast2source` → JavaScript).
- **javascript-targeting/**: The companion subfolder for this document. Covers the two targeting models (F#/.NET via Fable vs. fully decomposed AST via JSIR), the four JavaScript deployment wings, and the SDK-describes-the-runtime convention. Start with [javascript-targeting/README.md](./javascript-targeting/README.md).
- **Actor_Substrate_Independence.md**: Demonstrates the dual-target compilation model. The JavaScript emission in §6 of that document is what the JSIR backend produces.
- **JSIR-Strategic-Assessment.md** (Fidelity.CloudEdge/docs): Strategic analysis of JSIR's implications across the Fidelity ecosystem.

### Horizon Context

**Horizon 1** (current): Xantham replaces Glutinum within the existing F#/Fable pipeline. Tactical.

**Horizon 2** (this document + companions): Composer gains a JSIR-based JavaScript backend. Farscape produces Clef bindings via Transpose. Fable exits the pipeline entirely. Strategic.

**Horizon 3** (future): Composer Transcribe absorbs foreign implementations. TypeScript algorithms become native Clef. Visionary.

## 2. Architectural Position

### 2.1 JSIR as an MLIR Dialect Target

The existing Composer backends follow a consistent pattern: Alex observes the saturated PSG and emits MLIR dialect operations. A backend toolchain consumes those operations and produces the final artifact.

| Backend | Alex emits | Backend tool | Output artifact |
|:--------|:-----------|:-------------|:----------------|
| CPU | `func`/`memref`/`arith`/`scf` | `mlir-opt` + `llc` + linker | ELF binary |
| FPGA | `hw`/`comb`/`seq` | `circt-opt` | Verilog (.sv) |
| NPU | `aie.*`/`arith`/`memref`/`scf` | `aiecc.py` (aie-opt + Peano) | xclbin + insts.bin |
| **JS** | **`jsir.*` / `jshir.*`** | **`jsir_gen --passes=hir2ast,ast2source`** | **JavaScript modules (.js)** |

The JavaScript backend was originally designed as an exception to this pattern: Alex witnesses would emit a JavaScript AST data structure directly, bypassing MLIR. JSIR removes the need for that exception. JavaScript emission becomes dialect lowering, the same as every other target.

### 2.2 The EmitC Precedent

EmitC is an MLIR dialect already upstream in MLIR core, designed for lowering MLIR to C source code. It proves that MLIR dialects can serve as source language emission targets. JSIR applies the same pattern to JavaScript. The structural analogy is exact:

| | EmitC | JSIR |
|:--|:--|:--|
| Target language | C | JavaScript |
| Dialect position | Emission target | Emission target |
| Round-trip fidelity | High | 99.9%+ |
| MLIR upstream status | Core | Proposed (April 2026 RFC) |

### 2.3 Open Design Question: Middle-End Participation vs. Backend Isolation

The precise boundary between Alex's middle-end dialects and JSIR is not settled. Two models are under investigation:

**Model A: Backend isolation (CIRCT analogy).** Alex works in Clef-native dialects with target-agnostic semantics. At the backend boundary, a lowering pass converts Alex's output to JSIR ops. JSIR is consumed only by the backend toolchain. This is the cleaner separation and the current leaning.

```
Alex middle-end (Clef dialects, target-agnostic)
    ↓ lowering pass
JSIR ops (backend-only)
    ↓ jsir_gen --passes=hir2ast,ast2source
JavaScript source
```

**Model B: Middle-end catalog extension.** Alex acquires a small set of JavaScript-target-specific dialect ops that participate in the nanopass pipeline alongside existing dialects. These ops lower to JSIR at the backend boundary. This model is justified if certain JavaScript-specific semantics (async/await patterns, Durable Object lifecycle hooks, WebSocket hibernation protocol) benefit from middle-end representation.

```
Alex middle-end (Clef dialects + JS-specific ops where needed)
    ↓ lowering pass
JSIR ops
    ↓ jsir_gen --passes=hir2ast,ast2source
JavaScript source
```

The investigation will determine which model, or what hybrid, serves best. The key principle: do not fragment the middle-end unnecessarily when the type system's target profile mechanism handles the difference cleanly, but do not force artificial similarity where target-specific representation serves compilation better.

**What is settled:** JSIR is the emission dialect for JavaScript targets. The PSG enters Alex for JavaScript compilation. Fable becomes optional.

### 2.4 Target Profiles

Clef's type system uses dimensional types with inference. When targeting native via LLVM, the compiler resolves concrete widths (i32, f64, etc.) based on the target architecture. When targeting JavaScript via JSIR, the same inference mechanism applies with JavaScript's default widths (Number, BigInt, typed array backing).

JavaScript is a CPU-based runtime. It is another target profile for Clef's dimensional type inference, alongside ARM, x86, RISC-V, and FPGA. The type inference system handles target-specific width resolution uniformly. This narrows the design space: there is less reason for a separate JavaScript-specific dialect catalog in Alex when the type system's target profile mechanism handles the differences.

## 3. Compilation Pipeline

### 3.1 Pipeline Diagram

```
Clef source (.clef)
    │
    ▼
[FrontEnd] CCS: .clef → PSG
    │
    ▼
[MiddleEnd] Baker: Selective saturation (target-aware)
    │
    ▼
[MiddleEnd] Alex: PSG → JSIR dialect ops
    │  JS witnesses observe PSG nodes
    │  emit jsir.* operations (expressions, statements, modules)
    │
    ▼
JSIR module (MLIR text with jsir.* ops)
    │
    ▼
[BackEnd] JSIR: jsir_gen --passes=hir2ast,ast2source → JavaScript source
    │  Source maps generated (JSIR locations → Clef source)
    │  Optional: JSDoc annotations from Clef type information
    │  Optional: .d.ts companion files for TypeScript consumers
    │
    ▼
JavaScript modules (.js)
    │
    ▼
[Downstream] Bundler / deployment toolchain
    │  Vite, esbuild, or direct ES module usage
    │  Fidelity.CloudEdge Management Layer deploys Workers
    │
    ▼
Deployment artifact
```

Parallel to the native pipeline:

```
CCS → PSG → Baker → Alex (MLIR witnesses) → func/memref/arith/scf
                                                    ↓
                                              mlir-opt / llc / linker
                                                    ↓
                                              Native binary (ELF)
```

Both paths share CCS, PSG construction, and Baker saturation. They diverge at Alex's witness registration.

### 3.2 Comparison with Other Backends

| Aspect | CPU (LLVM) | FPGA (CIRCT) | NPU (AIE) | JS (JSIR) |
|:-------|:-----------|:-------------|:-----------|:----------|
| DeclRoot | EntryPoint | HardwareModule | KernelModule | WorkerModule |
| Attribute | `[<EntryPoint>]` | `[<HardwareModule>]` | `[<KernelModule>]` | `[<WorkerModule>]` |
| Witness | (standard) | HardwareModuleWitness | KernelModuleWitness | WorkerModuleWitness |
| Source pattern | `main : int` | `Design<'S>` | `ElementKernel<'T>` | `Olivier<'Msg>` / module |
| Primary field | (function body) | `Step: 'S -> 'S` | `Compute: 'T -> 'T -> 'T` | `Handle: 'Msg -> Async<unit>` |
| Metadata fields | (none) | `InitialState: 'S` | `Shape: Shape` | substrate bindings |
| Alex emits | func/memref/arith/scf | hw/comb/seq | aie.*/arith/memref/scf | jsir.* / jshir.* |
| Backend tool | mlir-opt + llc + clang | circt-opt | aiecc.py (aie-opt + Peano) | `jsir_gen --passes=hir2ast,ast2source` |
| Output artifact | NativeBinary (ELF) | Verilog (.sv) | Xclbin (.xclbin + .bin) | JavaScriptModule (.js) |

### 3.3 BackEndArtifact Extension

```fsharp
type BackEndArtifact =
    | NativeBinary of path: string
    | Verilog of path: string
    | Xclbin of xclbinPath: string * instsPath: string
    | JavaScriptModule of jsPath: string * sourceMapPath: string option
    | IntermediateOnly of format: string
```

## 4. Selective Saturation

Baker's saturation is target-aware. The `.fidproj` target declaration informs Baker which saturations to apply. This is documented in detail in `JavaScript_Backend_Design.md` §3 and carries forward unchanged.

The key decisions for the JS target:

| Saturation | LLVM/CIRCT/AIE | JSIR |
|:-----------|:--------------:|:----:|
| Intrinsic elaboration | Yes | Yes (to JS equivalents) |
| Pipe operator reduction | Yes | Yes |
| Partial application resolution | Yes | Yes |
| HOF expansion to explicit loops | Yes | **No** (preserve HOFs) |
| Closure struct pair construction | Yes | **No** (use native closures) |
| Match expansion to conditionals | Yes | Configurable |
| Reactive primitive expansion | N/A | **No** (preserve signals/effects) |

Baker already operates as a series of discrete nanopass recipes (Fan-Out/Fold-In). The JS target excludes certain recipe categories. This is not a new concept; it is existing Baker infrastructure with a different exclusion list.

## 5. JSIR Witness Architecture

### 5.1 Witness Registration

The witness registry activates JSIR witnesses based on the `.fidproj` target:

```fsharp
match target with
| "cpu" | "native" ->
    registerMLIRWitnesses registry
| "fpga" ->
    registerMLIRWitnesses registry
    |> registerCIRCTWitnesses
| "npu" ->
    registerMLIRWitnesses registry
    |> registerAIEWitnesses
| "js" | "javascript" ->
    registerJSIRWitnesses registry
```

### 5.2 Elements, Patterns, Witnesses

The three-layer architecture (Alex_Architecture_Overview.md) applies to the JSIR witness set. The architectural invariants hold:

- Witnesses MUST NOT call other witnesses (parallelism preserved)
- Witnesses MAY share Elements and Patterns (common vocabulary)
- Witnesses MUST return `WitnessOutput.skip` for unhandled nodes
- All state is read-only coeffects (no mutable shared state)

The difference from the original `JavaScript_Backend_Design.md`: witness Elements emit JSIR ops, not JavaScript AST nodes. The Elements layer wraps JSIR's operation set.

```
Alex/
  Elements/
    JsirExprElements.fs         # jsir.numeric_literal, jsir.identifier,
                                # jsir.binary_expression, jsir.call_expression,
                                # jsir.assignment_expression
    JsirStmtElements.fs         # jsir.expression_statement, jsir.variable_declaration,
                                # jsir.return_statement
    JsirControlElements.fs      # jshir.if_statement, jshir.while_statement,
                                # jshir.logical_expression
    JsirModuleElements.fs       # jsir.import_declaration, jsir.export_declaration
    JsirFunctionElements.fs     # jsir.function_declaration, jsir.arrow_function_expression

  Patterns/
    JsirApplicationPatterns.fs  # Function call, method chain composition
    JsirClosurePatterns.fs      # Arrow functions, native closure emission
    JsirModulePatterns.fs       # ES module import/export structure
    JsirReactivePatterns.fs     # SolidJS signal/effect/memo patterns
    JsirActorPatterns.fs        # Durable Object class, WebSocket handler structure
    JsirBAREWirePatterns.fs     # DataView/ArrayBuffer serialization ops

  Witnesses/
    JsirLiteralWitness.fs       # Numeric, string, boolean, null literals
    JsirArithWitness.fs         # Arithmetic, comparison (Number semantics)
    JsirApplicationWitness.fs   # Function calls, method invocations
    JsirBindingWitness.fs       # const/let bindings
    JsirControlFlowWitness.fs   # if/else, for, while, switch
    JsirLambdaWitness.fs        # Arrow functions
    JsirModuleWitness.fs        # import/export statements
    JsirInteropWitness.fs       # [<JsImport>], [<JsInterface>] handling
    JsirReactiveWitness.fs      # SolidJS signal/effect primitives
    JsirActorWitness.fs         # Olivier → DO class, Prospero → supervisor DO
    JsirBAREWireWitness.fs      # DU → BAREWire codec (DataView ops)
    WorkerModuleWitness.fs      # [<WorkerModule>] → Worker entry (fetch handler)
```

### 5.3 JSIR Op Mapping

The witness Elements emit JSIR operations. The key mappings from Clef PSG nodes to JSIR ops:

| PSG Node | JSIR Op | Notes |
|:---------|:--------|:------|
| Let binding (immutable) | `jsir.variable_declaration` (const) | |
| Let binding (mutable) | `jsir.variable_declaration` (let) | |
| Integer literal | `jsir.numeric_literal` | |
| String literal | `jsir.string_literal` (template_literal for interpolation) | |
| Binary arithmetic | `jsir.binary_expression` | Operator resolved by element type |
| Function application | `jsir.call_expression` | |
| Lambda / closure | `jsir.arrow_function_expression` | Native JS closures |
| If/then/else | `jshir.if_statement` | Uses JSIR's region-based control flow |
| Match (DU dispatch) | `jshir.switch_statement` or chained `jshir.if_statement` | Tag-based dispatch |
| Async computation | `jsir.await_expression` inside `async` function | |
| Module import | `jsir.import_declaration` | From `[<JsImport>]` attributes |
| Module export | `jsir.export_declaration` | |

### 5.4 WorkerModuleWitness

The `[<WorkerModule>]` attribute is the JavaScript-target analogue of `[<EntryPoint>]` (CPU), `[<HardwareModule>]` (FPGA), and `[<KernelModule>]` (NPU). It marks the root of a Cloudflare Worker compilation unit.

```fsharp
// WorkerModuleWitness matches:
//   SemanticKind.Binding(_, _, _, Some DeclRoot.WorkerModule)
//
// Emits:
//   - ES module with default export { fetch, ... }
//   - Durable Object class exports for each Olivier<'Msg> type
//   - BAREWire codec functions for each message DU
```

When `Fidelity.CloudEdge` is in the dependency scope, the WorkerModuleWitness activates CloudEdge-specific emission: DO classes with `webSocketMessage` handlers, `fetch` ingress, `blockConcurrencyWhile` activation, and the full lifecycle mapping documented in `Actor_Substrate_Independence.md` §6.

When CloudEdge is not in scope (e.g., a WREN Stack frontend module), the witness emits standard ES modules with SolidJS component exports.

## 6. JSIR Backend Toolchain

### 6.1 `jsir_gen`: JSIR to JavaScript Source

JSIR ships a single CLI binary, `jsir_gen`, whose behavior is controlled by a `--passes` argument naming a pipeline of transforms. The pass names are conversion steps between four representations: JavaScript source, Babel AST, JSHIR (region-based high-level IR), and JSIR (flat SSA-shaped low-level IR).

Composer's backend invokes the reverse pipeline:

```
jsir_gen --input=module.mlir --passes=hir2ast,ast2source --output=module.js
```

JSIR's 99.9%+ round-trip fidelity means the emitted JavaScript is structurally faithful, readable, and idiomatic.

```
[BackEnd] JSIR Pipeline:
  1. Write JSIR MLIR text to file (.mlir)
  2. Invoke jsir_gen --passes=hir2ast,ast2source to produce JavaScript source
  3. Generate source maps (JSIR locations → Clef source positions)
  4. Optionally emit JSDoc annotations from Clef type metadata
  5. Optionally emit .d.ts companion files for TypeScript consumers
  6. Return JavaScriptModule artifact
```

The forward pipeline `--passes=source2ast,ast2hir` is used only during development to validate that Clef-emitted JavaScript round-trips through JSIR without loss. Composer's production pipeline does not need the forward direction.

### 6.2 Source Maps

JSIR operations carry MLIR location attributes. The JSIR backend maps these through two levels:

1. **Clef source → MLIR location**: CCS attaches Clef source positions to PSG nodes. Alex propagates these as MLIR `FileLineCol` locations on JSIR ops.
2. **MLIR location → JavaScript source position**: the `ast2source` pass tracks the relationship between JSIR ops and their emitted JavaScript text positions.

The composite mapping (Clef source line → JavaScript source line) is emitted as a standard V3 source map. This enables:

- Browser devtools showing Clef source during debugging
- Cloudflare dashboard traces pointing to Clef source
- Error stack traces mapping back to the developer's Clef code

### 6.3 JSDoc Annotations and .d.ts Companions

JSIR itself has no type system. Clef's type system carries full type information through CCS and the PSG. The JSIR backend can emit this information as JSDoc annotations in the JavaScript output and as separate `.d.ts` TypeScript declaration files.

This is a quality-of-life feature. The JavaScript output is deployable without annotations. The annotations serve consumers who interact with the compiled JavaScript from TypeScript projects.

### 6.4 Downstream Toolchain

The JavaScript/JSX output flows through standard web toolchains:

| Deployment Target | Bundler | Deployment Method | Output |
|:-----------------|:--------|:------------------|:-------|
| Cloudflare Workers | esbuild / none | `cfs` CLI via Management Layer API | Single JS bundle |
| WREN Stack (WebView) | Vite | Embedded in native binary | Optimized bundle |
| Browser SPA | Vite / esbuild | Static hosting | JS bundles + assets |
| Node.js service | None (ES modules) | Direct execution | ES module |

Deployment to Cloudflare Workers uses Fidelity.CloudEdge's Management Layer API clients and the `cfs` CLI. The Management Layer provisions infrastructure (D1 databases, KV namespaces, R2 buckets, Queues, DO namespaces) and uploads Worker scripts through Fidelity's own REST clients. No third-party CLI tooling is involved. The relationship is analogous to CIRCT emitting Verilog and a scripted pipeline invoking Vivado: the compiler produces the artifact, the framework's own tooling handles deployment.

## 7. Coeffect Analysis for JSIR

The coeffect analysis pipeline adapts for the JavaScript target. This is documented in `JavaScript_Backend_Design.md` §5 and carries forward:

| Coeffect | LLVM Target | JSIR Target |
|:---------|:------------|:------------|
| SSA Assignment | Required (MLIR SSA form) | Handled by JSIR's own SSA semantics |
| Mutability Analysis | Required (memref.alloca vs. SSA) | Lightweight (mutable → `let`, immutable → `const`) |
| Escape Classification | Required (stack vs. heap) | Lightweight (closures are native; GC handles allocation) |
| String Table | Required (global memref dedup) | Not required (JS strings are interned by the engine) |
| Yield State Tracking | Required (state machine) | Adapted (JS generators or async iterators) |
| Pattern Binding | Required (conditional branch setup) | Adapted (destructuring assignment via JSIR) |
| Reactive Dependencies | N/A | New: classify signal reads for SolidJS tracking |

JSIR's SSA form handles variable binding semantics internally. The JSIR backend does not need the full SSA assignment pass that the LLVM backend requires; it needs the mutability and reactive dependency analysis to inform witness emission quality.

## 8. Actor Compilation (Cloudflare Target)

The JSIR backend's primary use case is compiling Clef actors to Cloudflare Workers with Durable Objects. The compilation flow follows the pattern documented in `Actor_Substrate_Independence.md`, with JSIR as the intermediate.

### 8.1 PSG to JSIR

The `JsirActorWitness` observes `Olivier<'Msg>` inheritance in the PSG and emits a Durable Object class structure as JSIR ops:

```
PSG: ClassType "CounterActor" (inherits Olivier<CounterMsg>)
  │
  ▼ JsirActorWitness
  │
JSIR ops (Babel-AST-shaped; JSIR follows Babel's class model where methods are `ClassMethod` nodes, not `MethodDefinition`):
  jsir.class_declaration "CounterActor"
    jsir.class_body
      jsir.class_method "constructor" (state, env)
        jsir.assignment_expression this.state ← state
        jsir.assignment_expression this.env ← env
      jsir.class_method "webSocketMessage" (ws, message)
        [BAREWire frame parsing ops]
        [DU tag dispatch via jshir.switch_statement]
        [Handle body ops per case]
      jsir.class_method "fetch" (request)
        [WebSocket upgrade check]
        [HTTP ask ingress]
      jsir.class_method "webSocketClose" (ws, code, reason, wasClean)
        [OnStop body ops]
```

### 8.2 BAREWire Codec Emission

The `JsirBAREWireWitness` observes discriminated union type definitions and emits BAREWire encode/decode functions as JSIR ops targeting `DataView` and `Uint8Array` operations:

```
PSG: UnionType "CounterMsg" [Increment; Add of int; GetCount]
  │
  ▼ JsirBAREWireWitness
  │
JSIR ops:
  jsir.function_declaration "CounterMsg_encode"
    jsir.call_expression buffer.writeUint8(msg.tag)
    jshir.switch_statement msg.tag
      case 1: jsir.call_expression buffer.writeVarint(msg.amount)
  jsir.function_declaration "CounterMsg_decode"
    jsir.call_expression data[offset.value++]   // read tag
    jshir.switch_statement tag
      case 0: jsir.object_expression { tag: 0 }
      case 1: jsir.object_expression { tag: 1, amount: readVarint(...) }
      case 2: jsir.object_expression { tag: 2 }
```

The native MLIR backend emits the same BAREWire schema as `memref` operations. The byte layout is provably identical because both lowering paths derive from the same PSG discriminated union definition. This is the BAREWire invariant documented in `Actor_Substrate_Independence.md` §8.

### 8.3 Reactive UI Emission (Browser / WREN Stack)

JSIR does not contain JSX operations. Its dialect surface is the desugared JavaScript that every JSX framework ultimately compiles to: function calls, object literals, and arrow functions. This shapes how reactive UI compilation works on the fully-decomposed-AST path, and it is one of two targeting models supported by Composer for browser-bound JavaScript.

**Model 1: F#/.NET path (Partas.Solid via Fable).**

Partas.Solid authors JSX in F# source. Fable transforms the F# AST into JavaScript with JSX already lowered to Solid's `template()` / `createComponent()` / `createSignal()` calls. JSIR does not participate in this path. This is the mechanism WrenHello and Fable-based browser frontends rely on today.

**Model 2: Fully decomposed AST path (Composer via JSIR).**

A Clef reactive surface (signals, effects, memos as first-class source constructs) lowers through Alex into JSIR ops that represent the post-JSX-transform JavaScript directly:

```
PSG: FunctionDef "App" [reactive annotations]
  │
  ▼ JsirReactiveWitness
  │
JSIR ops (Babel-AST-shaped, post-JSX-transform):
  jsir.function_declaration "App"
    jsir.variable_declaration const [count, setCount] = createSignal(0)
    jsir.return_statement
      jsir.call_expression createElement("div", props,
        jsir.call_expression createElement("button",
          { onClick: jsir.arrow_function_expression c => setCount(c + 1) },
          "Increment"))
```

The reactive primitives (`createSignal`, `createEffect`, `createMemo`) survive Baker's selective saturation as recognized framework calls. `createElement` / `h` / `template` are whichever factory the target framework ships; Solid's runtime remains `solid-js` from npm in both models.

Composer does not reimplement Solid. It emits JavaScript that calls Solid's runtime. The F#/.NET and fully-decomposed-AST models converge at the Solid runtime boundary. See [javascript-targeting/01_two_models.md](./javascript-targeting/01_two_models.md) for the detailed comparison.

## 9. Cross-Substrate Compilation

The JSIR backend enables the cross-substrate architecture described in `Actor_Substrate_Independence.md` §9 and the JSIR Strategic Assessment §8.8.

### 9.1 Dual-Target Build

A Clef project produces both native and JavaScript artifacts from the same source:

```
CounterActor.clef
    │
    ├── myapp-native.fidproj (target=cpu)
    │     CCS → PSG → Baker → Alex (MLIR witnesses) → LLVM → ELF
    │
    └── myapp-edge.fidproj (target=js)
          CCS → PSG → Baker → Alex (JSIR witnesses) → JSIR → jsir_gen --passes=hir2ast,ast2source → JS
```

The PSG is identical in both paths. Baker's saturation differs per target (§4). Alex runs the appropriate witness set per target.

### 9.2 BAREWire as the Seam

The native process and the Cloudflare Worker communicate via BAREWire frames over WebSocket (or Media over QUIC). The wire format is defined once at the PSG level and lowered to both `memref` ops (native) and `DataView` ops (JSIR/JS). The cross-substrate bridge Worker is itself a JSIR-compiled Clef module that holds a WebSocket to the native cluster and translates between transport mechanisms.

The `ActorRef<'Msg>` dispatch documented in `Actor_Substrate_Independence.md` §4 resolves at runtime:

- `Local`: direct function call (native) or DO internal dispatch (CloudEdge)
- `Edge`: `DurableObjectStub.fetch()` or WebSocket frame
- `Remote`: WebSocket/MoQ to the native cluster

## 10. JSIR Integration Path

### 10.1 Dialect Extraction

JSIR's dialect definitions (TableGen files, C++ op implementations) are build-system agnostic. The Bazel build configuration in Google's repository is for their tooling (parser, frontend); the dialect itself is standard MLIR ODS (Operation Definition Specification).

Extraction steps:

1. Clone `google/jsir`
2. Extract TableGen dialect definitions from `maldoca/js/ir/`
3. Integrate into Composer's MLIR build as an external dialect
4. Build the C++ op implementations against Composer's MLIR version

### 10.2 `jsir_gen` Invocation

JSIR's round-trip tooling is built into `jsir_gen` itself. The reverse pipeline `jsir_gen --passes=hir2ast,ast2source` performs:

1. Parse JSIR MLIR text (JSHIR ops with regions)
2. `hir2ast`: convert region-based JSHIR into Babel AST JSON (the reverse of `ast2hir`)
3. `ast2source`: print Babel AST to JavaScript source via the Babel printer (the reverse of `source2ast`)

Babel is the AST substrate throughout. JSIR's C++ code uses `BabelAstString`, `BabelParseRequest`, and the Babel printer for both input parsing and output generation. Composer does not reimplement any of this; it invokes `jsir_gen` as a subprocess in the backend.

Source map generation attaches to the `ast2source` pass via MLIR location attributes carried on the JSIR ops.

### 10.3 Prototype Sequence

1. **Validate JSIR op coverage.** Feed JavaScript resembling Fable output for a Cloudflare Worker (fetch handler, async/await, Response construction) through `jsir-gen`. Inspect the MLIR ops. Confirm the op set covers Worker-shaped JavaScript.

2. **Write a single witness.** Implement `JsirLiteralWitness`: observe PSG literal nodes, emit `jsir.numeric_literal` / `jsir.string_literal` ops. Verify the round-trip: Clef literal → PSG → JSIR op → `jsir_gen --passes=hir2ast,ast2source` → JavaScript literal.

3. **Build up to a Worker.** Incrementally add witnesses (binding, function, control flow, module) until a minimal Cloudflare Worker compiles end-to-end: Clef source → JavaScript Worker script.

4. **Port Actor_Substrate_Independence CounterActor.** Compile the CounterActor from `Actor_Substrate_Independence.md` §2 through the JSIR backend. Compare the output against the hand-written JavaScript in §6 of that document.

5. **BAREWire cross-substrate test.** Compile CounterActor for both native and JS targets. Verify that a message serialized by the native actor is byte-identical to the same message serialized by the JS actor. This validates the BAREWire invariant through the JSIR pipeline.

## 11. Project Structure

### 11.1 Backend Files

```
Composer/
  src/
    MiddleEnd/
      Alex/
        Elements/
          JsirExprElements.fs
          JsirStmtElements.fs
          JsirControlElements.fs
          JsirModuleElements.fs
          JsirFunctionElements.fs
        Patterns/
          JsirApplicationPatterns.fs
          JsirClosurePatterns.fs
          JsirModulePatterns.fs
          JsirReactivePatterns.fs
          JsirActorPatterns.fs
          JsirBAREWirePatterns.fs
        Witnesses/
          JsirLiteralWitness.fs
          JsirArithWitness.fs
          JsirApplicationWitness.fs
          JsirBindingWitness.fs
          JsirControlFlowWitness.fs
          JsirLambdaWitness.fs
          JsirModuleWitness.fs
          JsirInteropWitness.fs
          JsirReactiveWitness.fs
          JsirActorWitness.fs
          JsirBAREWireWitness.fs
          WorkerModuleWitness.fs
    BackEnd/
      JSIR/
        Pipeline.fs            # Orchestrates: write MLIR, invoke jsir_gen, emit source maps
        Lowering.fs            # Tool path resolution, jsir_gen --passes=hir2ast,ast2source invocation
        SourceMapEmitter.fs    # Clef → JS source map generation
        JSDocEmitter.fs        # Optional JSDoc annotation emission
        DtsEmitter.fs          # Optional .d.ts companion file emission
    Core/
      Types/
        Pipeline.fs            # BackEndArtifact extended with JavaScriptModule
      PlatformPipeline.fs      # JS → BackEnd.JSIR.Pipeline.backend
```

### 11.2 fidproj Configuration

```toml
[compilation]
target = "js"

[compilation.js]
# Elaboration mode (documented in JavaScript_Backend_Design.md §3.3)
mode = "reactive"         # preserve framework idioms
# mode = "computational"  # saturate HOFs, optimize throughput

# Source map emission
source_maps = true

# JSDoc annotations from Clef types
jsdoc = true

# .d.ts companion files
dts = false
```

## 12. Design Documents Required

The following specifications complete the JSIR backend:

1. **JSIR Elements Specification.** Define the wrapper functions in `JsirExprElements.fs` etc. that emit JSIR ops with correct region structure, SSA semantics, and l-value/r-value distinctions.

2. **JSIR Lowering Rules.** Formalize the lowering from Alex's Clef-native dialect ops to JSIR ops for Model A (backend isolation), or specify which ops participate in the middle-end for Model B.

3. **Selective Saturation Specification.** (Shared with `JavaScript_Backend_Design.md` §3.) Formalize the Baker recipe exclusion interface. This may require refactoring Baker's recipe system to support per-target exclusion lists.

4. **JSIR Coeffect Analysis.** Specify the lighter coeffect pass for JS targets, focusing on reactive dependency classification and mutability analysis.

5. **Source Map Specification.** Define the two-level mapping (Clef → MLIR location → JS source position) and the V3 source map emission format.

6. **Actor Witness Specification.** Formalize the `JsirActorWitness` emission for each `Olivier` lifecycle hook, the supervision protocol, and the elastic scaling patterns documented in `Fidelity.CloudEdge/docs/08a-08e`.

7. **BAREWire JSIR Lowering.** Specify how BAREWire dialect ops lower to `DataView`/`ArrayBuffer` operations in JSIR, preserving byte-layout identity with the native `memref` lowering.

## 13. Navigation

- Predecessor: [JavaScript_Backend_Design.md](./JavaScript_Backend_Design.md) (pre-JSIR design, remains valid for witness specifications)
- Reference: [AIE_Backend_Design.md](./AIE_Backend_Design.md) (structural template)
- Reference: [Alex_Architecture_Overview.md](./Alex_Architecture_Overview.md) (Elements/Patterns/Witnesses architecture)
- Companion: [Actor_Substrate_Independence.md](./Actor_Substrate_Independence.md) (dual-target compilation model)
- Strategic context: [JSIR-Strategic-Assessment.md](../../Fidelity.CloudEdge/docs/JSIR-Strategic-Assessment.md)
