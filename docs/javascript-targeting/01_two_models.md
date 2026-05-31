# Two Models for JavaScript Targeting

**SpeakEZ Technologies | Fidelity Framework**
**April 2026**

Composer reaches JavaScript through two distinct compilation models. Both are first-class. Neither is a fallback for the other. They exist because the source languages involved, the consumer ecosystems, and the idioms each one produces are different enough that collapsing them into a single path would be a mistake.

## The Two Models

**Model 1: The F#/.NET model.** F# source is compiled by Fable, which walks the F# AST through FSharp.Compiler.Service and emits a Babel AST that becomes JavaScript. Partas.Solid is an F# library that surfaces SolidJS's JSX as a typed DSL in F#; Fable handles the JSX transform during compilation. The output is JavaScript (and `.jsx` when desired) that consumes `solid-js` from npm at runtime. JSIR does not participate in this pipeline.

**Model 2: The fully decomposed AST model.** Clef source is compiled by Composer. The compilation passes through CCS → PSG → Baker → Alex, and Alex's witnesses emit JSIR operations (a Babel-AST-shaped MLIR dialect). The JSIR output is then run through `jsir_gen --passes=hir2ast,ast2source`, which converts the MLIR ops to Babel AST and then prints JavaScript source. The output is JavaScript (no JSX — JSIR has no JSX operations) that consumes whichever JavaScript runtime libraries the target uses: `solid-js` for Solid, Cloudflare's `@cloudflare/workers-types`-shaped APIs for Workers, etc.

## What Each Model Is Good At

| Axis | F#/.NET model | Fully decomposed AST model |
|:-----|:--------------|:---------------------------|
| Source language | F# (and by extension, C# via shared runtime) | Clef |
| Compiler | Fable | Composer |
| Intermediate form | Fable AST → Babel AST | PSG → JSIR ops → Babel AST |
| Type system carrier | .NET CLR types erased to JS | Clef's dimensional type system, target-profile-resolved |
| JSX handling | Source-level (Partas.Solid) | Not applicable; JSX is upstream of JSIR |
| Package ecosystem | NuGet for F# bindings (Glutinum, Hawaii, Partas.Solid) | npm for JavaScript runtime libraries; Clef-side SDKs for binding descriptions |
| Current reach | Full: WrenHello, Fidelity.CloudEdge, browser frontends in production | Emerging: prototype backends, Horizon 2 target |
| Runtime dependencies | Fable runtime (minimal), F# core JS library | None added; JavaScript runtime libraries consumed as-is |
| Interop attributes | `[<Import>]`, `[<Emit>]`, `[<Erase>]` from Fable | `[<JsImport>]`, `[<JsEmit>]`, `[<JsErase>]` emitted by Clef/Composer |
| Best fit | Existing F# codebases; teams with .NET fluency; Fable's reactive UI idioms | Clef-native projects; cross-substrate actors (native + JavaScript from one source); dimensional-typed numerical computation reaching JavaScript |

Neither column dominates the other on every axis. Which model a given project uses is a function of what the source codebase already is, not an architectural ranking.

## The js_of_ocaml Precedent

The closer architectural precedent for Composer's fully-decomposed-AST model is not Fable. It is js_of_ocaml, and the comparison is worth making in detail because js_of_ocaml is essentially a proof, delivered over fifteen years of production use, that the compile-through-IR approach works for reaching JavaScript from an ML-family language. Fable was inspired by js_of_ocaml but took a different tactical route; Composer follows js_of_ocaml's architectural route and extends it with MLIR's multi-target infrastructure.

**What js_of_ocaml actually does.** Jérôme Vouillon and Vincent Balat published the first release of js_of_ocaml in 2010 at PPS/CNRS Paris Diderot. The architectural commitment, stated in the project README, is to compile from OCaml bytecode rather than source:

> We believe this compiler will prove much easier to maintain than a retargeted OCaml compiler, as the bytecode provides a very stable API.

A look at the repository under [external/js_of_ocaml/](../../../oxcaml/external/js_of_ocaml/) in the oxcaml tree confirms what "compile from bytecode" actually means in practice. `compiler/lib/parse_bytecode.ml` lifts OCaml's bytecode format into js_of_ocaml's internal IR, called `Code.program`. That IR lives in `compiler/lib/code.ml` and is a full SSA-style representation with basic blocks (`Addr`), variables (`Var`), and structured control flow. Multiple optimization passes run over `Code.program`: `deadcode.ml`, `tailcall.ml`, `flow.ml`, `global_flow.ml`, `effects.ml` (CPS transformation for effect handlers), `generate_closure.ml`. Only after those passes complete does `generate.ml` translate `Code.program` to `Javascript.program` — an in-memory JavaScript AST — which is then printed by `js_output.ml` with source maps.

The signature of `generate.ml`'s main entry point is worth quoting (condensed from the actual `.mli`):

```ocaml
val f : Code.program
     -> live_vars:Deadcode.variable_uses
     -> trampolined_calls:Effects.trampolined_calls
     -> in_cps:Effects.in_cps
     -> deadcode_sentinal:Code.Var.t
     -> ...
     -> Javascript.program
```

The function takes an optimized IR program and context carried forward from earlier analysis passes, and returns a JavaScript AST. This is a conventional compiler backend shape. The equivalent shape in Composer is Alex's witness layer consuming the PSG (plus coeffects carried forward from Baker) and emitting JSIR ops.

**The multi-backend generalization.** In 2024, the same project shipped `wasm_of_ocaml`, a WebAssembly backend. The critical detail: `wasm_of_ocaml` shares `Code.program` with `js_of_ocaml`. Both backends sit atop the same IR and consume the same upstream optimization passes. The directory structure in the repository makes this explicit:

```
compiler/
  lib/              # Shared IR + optimization passes (Code.program, etc.)
  lib-wasm/         # WebAssembly-specific lowering
  bin-js_of_ocaml/  # JavaScript backend entry point
  bin-wasm_of_ocaml/# WebAssembly backend entry point
```

That is the multi-target-from-shared-IR pattern in miniature. One frontend (OCaml bytecode parser), one middle-end (`Code.program` plus its passes), two backends (`generate.ml` for JavaScript, `lib-wasm/` for WebAssembly). The OCaml source compiles once; two artifacts come out.

This is the pattern Composer generalizes. Composer's middle-end is Alex and the PSG. Composer's backends are LLVM (native binaries), CIRCT (FPGA Verilog), MLIR-AIE (NPU xclbin), and JSIR (JavaScript). One Clef source reaches any of those, and the backend chosen is a compilation decision, not a source decision. The structural correspondence to js_of_ocaml/wasm_of_ocaml is direct; the substrate that makes it scale to four (and counting) backends instead of two is MLIR.

**Why this is the right comparison.** A few properties of js_of_ocaml map onto Composer's architecture more cleanly than Fable's:

| Property | js_of_ocaml | Composer (JSIR path) | Fable |
|:---------|:-----------|:---------------------|:------|
| Starts from | OCaml bytecode (post-compile) | Clef PSG (post-elaboration) | F# source AST (pre-compile) |
| Internal IR | `Code.program` (SSA) | MLIR ops (SSA) | Fable AST (walked) |
| Optimization passes run on IR | Yes: deadcode, flow, tailcall, effects CPS | Yes: Baker saturation, coeffect analysis, witness emission | Limited: some AST rewriting |
| Shared IR across backends | Yes: `Code.program` feeds js_of_ocaml and wasm_of_ocaml | Yes: PSG feeds LLVM, CIRCT, MLIR-AIE, JSIR | No: Fable's AST is JavaScript-specific |
| Emission step | `generate.ml` → `Javascript.program` → source | JSIR ops → `jsir_gen --passes=hir2ast,ast2source` → source | Fable AST → Babel AST → source |
| Production lifetime | 2010 to present (15+ years) | 2026 forward | 2016 to present (9 years) |

js_of_ocaml is the answer to the question "does this architecture actually work at scale, for a long time?" because the answer has been demonstrated, reproducibly, since 2010. Some of the JavaScript people run in production today — in Ocsigen-based web applications, in ReScript's earlier history, in the OCaml toplevels embedded in tutorial sites — is js_of_ocaml output. The IR-driven compilation approach has been battle-tested by the OCaml community for over a decade.

Composer's JSIR-based backend is the same architecture realized on MLIR's infrastructure. The novelty is the substrate (MLIR instead of a bespoke IR), the scope (four backends instead of two), and the type system that drives middle-end reasoning (Clef's dimensional types and codata instead of OCaml's types). The architectural pattern itself — one semantic IR, multiple backends, heavy optimization pressure applied before the source language's AST ever gets reconstructed for emission — is what js_of_ocaml has been proving all along.

## Why Fable Did Not Follow js_of_ocaml's Path

Fable, built by Alfonso García-Caro starting in 2016, was explicitly inspired by js_of_ocaml. The early Fable README cited js_of_ocaml as the reference. Fable does not, however, compile through a post-compile IR the way js_of_ocaml does. It compiles from the F# source AST as provided by FSharp.Compiler.Service. This is not a matter of tactical preference. It is a consequence of how different OCaml's and F#'s post-compile representations are.

**What OCaml bytecode gave js_of_ocaml.** OCaml bytecode is the output of `ocamlc`, and it is designed for a runtime built specifically for a single-inheritance ML-family language. OCaml's type system is erased to bytecode, but the runtime representation preserves the structural semantics of the source: closures are ML closures, tag dispatch on variants is tag dispatch on variants, currying is currying. The bytecode is a small, stable instruction set. Lifting bytecode into js_of_ocaml's `Code.program` IR is direct; the work of translating OCaml semantics into JavaScript happens in the IR-to-JavaScript generation step, but the semantic information the compiler needs is still visible in bytecode.

**What .NET CLR IL does not give Fable.** F#'s post-compile representation is .NET CLR Intermediate Language (IL). CLR IL is an IR, but it is not an F# IR. It is the IR of a multi-language runtime designed around C#'s semantics — reified generics, reference/value type distinctions, boxing rules, structural variance constraints, interface dispatch, and a specific implementation of garbage collection. By the time F# source has been lowered to IL, much of what makes F# code F# code has been translated into CLR-shaped constructs. Discriminated unions become sealed class hierarchies. F# records become classes with properties. Pipe operators become nested method calls. Curried functions become chains of `FSharpFunc` instances. Async workflows become state machines expressed through CLR task types. Pattern matching is expanded into cascading type tests.

None of that lowering is reversible from IL alone. A hypothetical IL-to-JavaScript compiler would be compiling the CLR, not F#. It would produce JavaScript that mirrored C#-shaped object hierarchies, chains of `FSharpFunc` objects, and CLR task state machines — not JavaScript that looked like what an F# developer would recognize as "their code." It would also pull the full weight of the CLR's assumptions into JavaScript: the boxing rules, the dispatch semantics, the GC contract, the exception model. js_of_ocaml does not face this problem because OCaml's runtime is simpler and its bytecode is closer to ML semantics. Fable does face it, because F#'s runtime is the CLR and the CLR's IL is not F#-shaped.

**The consequence.** For Fable, the F# source AST is the last stable representation in which F# idioms are still visible as F# idioms. Pipe operators are still pipe operators. Curried functions are still curried functions. Pattern matching is still pattern matching, not a cascade of type tests. Discriminated unions are still DUs, not sealed hierarchies. The AST is where a JavaScript-targeting F# compiler has to start if the output is to be recognizably F#-derived JavaScript rather than C#-derived JavaScript that went through F# syntax on the way in. The alternative — working from IL — was not really an alternative. It would have meant building a CLR-to-JavaScript compiler, which is a different project with a different output character.

This is why calling the F#/.NET model "AST-walking" is accurate but the AST-walking isn't a choice Fable preferred over IR-walking. It's the highest point in F#'s compilation stack at which F# semantic structure is still intact. Fable works there because that's where the semantics live.

Composer is not subject to this constraint because Clef does not target the CLR. Clef compiles to its own PSG, and the PSG is designed to preserve Clef semantics (dimensional types, coeffects, BAREWire schemas, codata annotations) through elaboration. The PSG is Clef's equivalent of OCaml bytecode in this context: a post-compile IR that keeps the source language's semantic shape. JSIR then sits further downstream, specifically as the emission substrate for JavaScript. Composer's fully-decomposed-AST model is the js_of_ocaml-shaped approach applied to a language that, like OCaml and unlike F#, has a post-compile IR it can freely design.

The name "fully decomposed AST" refers to the Babel AST that surfaces at the very end of the pipeline — the form JSIR ops are printed as — not to where the compiler does its work. Upstream of that Babel AST, Composer is operating entirely in compiler IRs: the PSG, then MLIR dialects (including JSIR), with optimization and analysis passes running throughout. The compiler IR choice is available to Clef because Clef owns its post-frontend representation. It is unavailable to F# because F# does not.

## Where They Share Ground

Both models emit JavaScript that runs on the same runtimes (V8, Cloudflare Workers runtime, Hermes, WebKit in a WebView, etc.). Both consume the same ecosystem of JavaScript libraries from npm. Both rely on Babel AST as the emission substrate — Fable writes Babel AST directly; JSIR emits MLIR ops that `jsir_gen` converts to Babel AST before printing source.

This shared surface is why Composer's JSIR backend and Fable's F#/JSX pipeline coexist without interfering. The JavaScript produced by either path is ordinary JavaScript. A Cloudflare Worker script produced by Fable and a Cloudflare Worker script produced by Composer call the same Cloudflare runtime APIs; the Worker runtime cannot distinguish them.

## Where They Diverge

The divergence is at the source language boundary and in what the compiler can reason about.

**F#/.NET model:** The F# type system and Fable's transformation rules determine what the output looks like. .NET types are erased to JavaScript equivalents; F# union types become Fable's tag-based object representation; F# reference types become JavaScript objects. JSX in Partas.Solid is a typed DSL; Fable's JSX transform produces the JavaScript that actually calls Solid's runtime. The compiler's leverage ends where Fable's output ends.

**Fully decomposed AST model:** Clef's type system carries dimensional types, coeffects, escape classifications, BAREWire schemas, and other annotations that persist into the PSG as codata. Composer's middle-end can reason about this codata during elaboration (selective saturation, reactive primitive preservation, cross-substrate invariants). The output reaches JavaScript, but the compiler has done substantially more semantic analysis than Fable can. The JavaScript that emerges is consequently more targeted: Baker's per-target saturation strategy means the JSIR path can preserve higher-order functions where the native path would have expanded them into explicit loops, and it can emit BAREWire codecs that share byte layout with the native path's `memref` lowering.

## What Neither Model Is

Neither model is "JavaScript with a type system bolted on." JavaScript's type system is what the JavaScript engine enforces at runtime, which is close to nothing at the VM level. Both models produce JavaScript that runs under whatever the JavaScript engine does. What each model provides is a compile-time type system in the source language that constrains what the compiler emits. The runtime is agnostic to the source.

Neither model reimplements JavaScript's runtime libraries. Fable does not reimplement SolidJS; Composer does not reimplement SolidJS or `@cloudflare/workers-types`. Both models emit JavaScript that consumes those libraries at runtime. Composer's relationship to Cloudflare's tooling is not "we're replacing it" — it's "we're emitting JavaScript that calls it."

Neither model does source-to-source rewriting within its own language. Fable compiles F# to JavaScript; it does not translate between two JavaScript dialects. Composer compiles Clef to JavaScript; it is not a JavaScript-to-JavaScript transformer. Composer's Transcribe service (Horizon 3) may absorb foreign JavaScript implementations by lifting them into Clef, but that is a separate feature from the targeting models described here.

## Why Both Models Stay

The question the user asked during architectural review — "are we keeping Fable?" — has a crisp answer: yes. The F#/.NET model has F# code and F# developers behind it. WrenHello's frontend is F#/Partas.Solid. Fidelity.CloudEdge's F# bindings for Cloudflare services are consumed by F# projects. Removing Fable would strand that code. There is no technical reason to do so and considerable cost in doing it.

The Clef-native model is what Composer brings to the table. It is the path through which dimensional types, codata extension, and cross-substrate BAREWire reach JavaScript. It is a genuinely new capability, not a replacement of an existing one.

Both models serve different populations of users and different shapes of projects. Both remain.

## Cross-References

- [JSIR_Backend_Design.md](../JSIR_Backend_Design.md) — how the fully-decomposed-AST model actually emits JSIR
- [JavaScript_Backend_Design.md](../JavaScript_Backend_Design.md) — original (pre-JSIR) design; witness architecture still valid
- [Actor_Substrate_Independence.md](../Actor_Substrate_Independence.md) — cross-substrate compilation made possible by the fully-decomposed-AST model
- [02_jsir_tooling.md](./02_jsir_tooling.md) — what `jsir_gen` actually does
- [03_four_wings.md](./03_four_wings.md) — the four deployment wings both models target
