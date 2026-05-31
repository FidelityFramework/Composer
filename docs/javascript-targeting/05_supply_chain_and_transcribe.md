# Supply Chain Independence and Source Absorption

**SpeakEZ Technologies | Fidelity Framework**
**April 2026**

The js_of_ocaml architectural precedent documented in [01_two_models.md](./01_two_models.md) carries a property that deserves its own treatment, because it shapes what tooling Composer needs to build over the next two horizons. That property is supply chain independence from npm.

This document names the property, explains why js_of_ocaml has it by default and why Fable does not, distinguishes the two strategies Composer can use to reach the same independence (wrap vs. absorb), and specifies the new tooling the absorb strategy requires. Subsequent agent work on Composer's JavaScript-targeting pipeline should be grounded in this framing.

## The npm Supply Chain Reality

Every JavaScript application that consumes packages from npm is exposed to the transitive dependency graph of those packages. A typical Cloudflare Worker built with Wrangler pulls in hundreds of packages through its transitive dependencies. Each package is code written by some other author, published to the npm registry, and loaded into the Worker runtime at build or deploy time. This is the standard JavaScript development posture, and it is the posture Fable occupies today: Fable emits JavaScript that imports from npm packages, and the Fable toolchain itself depends on npm packages during build.

npm has a well-documented history of supply chain incidents: typosquatting attacks, compromised maintainer accounts, protestware events, transitive dependency takeovers. The mitigation is defense in depth at several layers — lockfiles, signed commits, dependency scanning, SBOM generation, registry policy — but the fundamental exposure is structural. JavaScript runtimes expect package resolution at runtime; the ecosystem is built around it; every import is potentially a trust decision about another author's code.

For software that has to meet a high bar of supply chain integrity (defense, finance, healthcare, edge infrastructure operated by regulated entities), this exposure is a material concern. The ability to produce JavaScript artifacts whose bytes are traceable end-to-end to auditable Clef source is a property worth building toward deliberately.

## What js_of_ocaml Gets By Default

js_of_ocaml produces standalone JavaScript. The OCaml program's standard library is the OCaml stdlib, which is compiled through the same js_of_ocaml pipeline as the application code. The OCaml runtime primitives that the stdlib depends on (exception handling, GC interaction, channel I/O, numeric operations) are provided by a JavaScript runtime layer that js_of_ocaml bundles. The result is a single JavaScript artifact that the OCaml program depends on, and nothing from npm is required to run it.

This isn't a security posture js_of_ocaml was built to provide; it's a consequence of the architectural commitment. By compiling through a post-compile IR (`Code.program`) and linking in OCaml's own stdlib through the same IR, the project never needed to pull dependencies from JavaScript's package ecosystem. The runtime layer js_of_ocaml ships (`runtime/`, written in JavaScript) is the project's own code under the project's own license, audited in one place. Users do install third-party OCaml libraries, but those libraries compile through js_of_ocaml too; they don't introduce npm dependencies unless the library explicitly interops with a JavaScript package through js_of_ocaml's FFI (which, when it happens, is an explicit and visible choice, not an implicit transitive inclusion).

An OCaml program compiled with js_of_ocaml and deployed as a Worker script does not have a `package.json`. It does not have a `node_modules`. It is a JavaScript file, it calls into the JavaScript engine's built-in APIs (or the host runtime's APIs like Workers' `env.KV.get`), and whatever OCaml libraries it uses are compiled-in rather than linked at runtime. The supply chain surface is: the OCaml source written by the application author, the OCaml libraries they chose to depend on (all of them building through js_of_ocaml), and js_of_ocaml itself. No npm packages appear anywhere.

## What Fable Does Not Get By Default

Fable produces JavaScript that imports from npm. The compiled F# code references `@cloudflare/workers-types` shapes, imports from `solid-js`, uses `fable-library` for runtime support, and consumes whatever Fable bindings packages the application depends on. Those bindings (Feliz for React, Partas.Solid for SolidJS, Fidelity.CloudEdge's workers bindings) are themselves F# libraries that compile through Fable, but the JavaScript they compile to expects npm-provided runtime libraries on the other end of each import.

This isn't a Fable defect. It's a consequence of Fable's position: because Fable compiles from the F# AST and produces Babel AST for output, and because F# doesn't have its own stdlib compiled-to-JavaScript equivalent (the .NET base class library is not portable in that shape), Fable has to emit JavaScript that uses JavaScript's libraries. Fable ships its own small runtime (`fable-library`) for language-level concerns like sequence evaluation and option types, but anything beyond that — HTTP, WebSockets, JSON parsing, UI frameworks, date handling — comes from npm.

A Cloudflare Worker built from F# through Fable has a `package.json`, pulls `fable-library` and whatever bindings packages from npm, and sits inside the JavaScript supply chain the same way a TypeScript-authored Worker does. The F# source is the author's code; most of what runs in production is other people's code from npm.

## The Fork: Wrap vs. Absorb

Composer has two paths to reaching the supply chain integrity that js_of_ocaml provides, and the two paths coexist rather than competing. Each is appropriate for different kinds of code.

**Wrap (Xantham/Farscape-style).** Generate Clef bindings from a TypeScript `.d.ts` file or an OpenAPI specification. The bindings are Clef types and function signatures that describe the JavaScript library's API. The Clef application code uses those bindings idiomatically. When Composer compiles, it emits JavaScript that calls into the wrapped library at runtime — the library is still an npm dependency of the final artifact. This is the model described in [04_sdk_describes_runtime.md](./04_sdk_describes_runtime.md): the SDK describes the runtime; Composer emits JavaScript that calls it.

The wrap strategy is right when:

- The library in question is itself a Cloudflare-owned runtime surface (`@cloudflare/workers-types`, `@moq/lite`) where the npm package is just a thin accessor over a Cloudflare-provided runtime feature. The actual code is Cloudflare's, running inside their runtime; the npm package is a TypeScript types file plus a tiny shim.
- The library is large, actively evolving, and pulling its behavior into Clef source would be an ongoing maintenance burden larger than the library is worth to Clef applications.
- The supply chain exposure of the specific package is acceptable — audited, pinned, small transitive dependency graph, trusted maintainer.

**Absorb (Transcribe-style).** Lift the JavaScript implementation of the library into Clef source. The result is Clef code that, when compiled through Composer, emits JavaScript that replicates the library's behavior without depending on the original npm package. The absorbed library becomes part of the Clef source tree (or a separately-distributed Clef package). This is the Transcribe service, listed as Horizon 3 in the broader roadmap.

The absorb strategy is right when:

- The library in question is load-bearing for the application's security or correctness guarantees (crypto, JSON parsing, deserializers, auth flows).
- Supply chain integrity is a contractual or regulatory requirement.
- The library's footprint is small enough that Clef-native absorption is feasible, or large but stable enough that one-time absorption is worth the effort.
- The library's behavior is worth owning as Clef source — i.e., future maintenance and extension will happen in Clef, not through npm updates.

Most projects will use both strategies. A Worker that uses Cloudflare's KV and R2 bindings (wrap is right) while needing supply-chain-clean crypto and auth (absorb is right) is perfectly coherent. The architecture does not force a project-wide choice.

## Why the Wrap Result Is Useful Even for the Absorb Strategy

Even when a project's ultimate goal is absorption, the wrap-style pattern established by Glutinum and Hawaii — and brought into the Clef ecosystem by Xantham and Farscape — is useful as a framing of the result rather than the mechanics. What does a correctly-absorbed JavaScript library look like from the Clef side? It looks like Clef code using bindings that resemble what Xantham would have generated from the library's `.d.ts` file. The shape of the Clef API surface is the same; what differs is whether calls into that surface lower to "emit JavaScript that imports from npm" or "emit JavaScript that contains the library's absorbed implementation inline."

This is what the user-facing observation captures: the Glutinum/Hawaii one-time-transform model is useful as a description of the result (Clef code that lowers to appropriate JavaScript), but not as a description of the pipeline mechanics (generate bindings, emit imports, rely on runtime). The pipeline mechanics for absorption are different, and that's what the rest of this document specifies.

## Mechanics of Absorption: JS-to-Clef Source Transforms

Transcribe operates in the opposite direction from Composer's normal compilation flow. Composer normally takes Clef source, produces JSIR ops through Alex, and emits JavaScript through `jsir_gen --passes=hir2ast,ast2source`. Transcribe takes JavaScript source, and needs to produce Clef source that, when compiled back through the normal Composer pipeline, reproduces semantically equivalent JavaScript.

The pipeline is:

```
JavaScript source (library.js)
    │
    ▼
jsir_gen --passes=source2ast,ast2hir → JSIR MLIR module
    │
    ▼
PSG Reconstructor (new tooling) → Clef PSG
    │
    ▼
Clef Printer (new tooling) → Clef source
    │
    ▼
Human review and curation (types, names, structural improvements)
    │
    ▼
Clef library source ready for inclusion
```

Several observations about this pipeline:

**JSIR is the bridge.** The forward pipeline `source2ast,ast2hir` already exists in JSIR. Transcribe reuses it. The JavaScript library enters the MLIR world through the same pass JSIR was originally designed for (analysis-oriented lifting from source). Transcribe does not need to write its own JavaScript parser; Babel is Google's dependency inside JSIR, and `jsir_gen` shells it out for Transcribe the same way it does for Composer.

**The novel tooling is PSG reconstruction.** JSIR ops represent JavaScript semantics; the PSG represents Clef semantics. The lift from JSIR back to a PSG requires inferring Clef-shaped structure (discriminated unions where the JavaScript uses tag-and-payload objects, closures with explicit environments where the JavaScript uses native closures, dimensional-type annotations where the JavaScript uses `number`). This inference is inherently lossy — the JavaScript didn't have the Clef-specific information — so the output PSG needs to be curated by human review to add type annotations, normalize naming, and refactor structures that JSIR's round-trip preserved verbatim but which don't fit Clef idioms.

**The output is Clef source, not a binary absorption.** Transcribe's output is code a Clef developer can read, edit, audit, and own. This is important for the supply-chain-integrity argument: a binary absorption that opaquely carries the library's behavior inside Composer's emission would satisfy the "no npm import" property but would fail the "auditable Clef source" property. The whole point of absorption is that the library's behavior becomes part of the owning organization's code, reviewable and modifiable on the same terms as any other Clef source.

**Each absorbed library is a one-time transform with ongoing curation.** The initial lift produces Clef source from JavaScript source. From that point forward, the library lives as Clef and is maintained as Clef. Upstream updates to the original JavaScript library (security patches, feature additions) are not automatically picked up; they have to be re-absorbed or hand-ported. For libraries where upstream churn is rapid and security-sensitive, this is a tradeoff that needs careful thought — wrapping may be the better choice if keeping up with upstream is essential. For libraries that are stable and whose behavior the organization wants to own, absorption is a one-time cost with predictable ongoing maintenance.

## Relationship to Farscape

Farscape generates Clef bindings from C/C++ headers (`.h` files). It is the analogue of Xantham (TypeScript `.d.ts`) and of Hawaii (OpenAPI spec). All three are wrap-style tools: they produce Clef bindings that let Clef code call into an external runtime (native C/C++ library for Farscape, JavaScript library for Xantham, HTTP service for Hawaii/Farscape's OpenAPI mode).

Transcribe is structurally similar to Farscape in that it reads existing source material and produces Clef code. The similarity ends there. Farscape produces bindings; Transcribe produces implementations. Farscape's result is Clef that calls into a preserved external library; Transcribe's result is Clef that replaces the external library entirely.

There is a conceivable Farscape-mode-for-JavaScript that would be Xantham's equivalent of Farscape's binding generation — read the library's `.d.ts`, generate Clef bindings that import from the library at runtime. That's the wrap strategy, and Xantham already does it (or will, under Horizon 1 as Xantham replaces Glutinum). The novel piece that Transcribe adds is the absorb strategy: read the library's JavaScript implementation (not just its types), lift it into Clef, and produce Clef source that compiles back to standalone JavaScript.

The two tools are complementary:

| Tool | Input | Output | Strategy | When to Use |
|:-----|:------|:-------|:---------|:------------|
| Xantham | TypeScript `.d.ts` | Clef bindings | Wrap | Library's npm dependency is acceptable; library is large or rapidly evolving |
| Farscape (C/C++) | C/C++ headers | Clef bindings | Wrap | Native library dependency is acceptable |
| Farscape (OpenAPI) | OpenAPI spec | Clef HTTP client | Wrap | HTTP service is the consumer relationship |
| Transcribe | JavaScript source | Clef source | Absorb | Supply chain integrity required; library behavior should be owned as Clef |

## Implications for Downstream Agent Work

This framing is meant to guide future work on Composer's JavaScript-targeting pipeline. Specifically:

**1. Default to the js_of_ocaml shape for new applications.** When designing new Clef applications that target JavaScript, prefer patterns that minimize npm runtime dependencies. The PSG should carry enough semantic information that Clef stdlib primitives (collections, async, I/O, serialization) can be witnessed directly into JSIR ops rather than lowering into calls into npm packages. The Clef stdlib itself should compile through Composer the way OCaml's stdlib compiles through js_of_ocaml.

**2. Treat Xantham as the Horizon 1 tool for necessary npm interop.** Some libraries (Cloudflare-owned runtime surfaces, well-audited small packages with tight transitive graphs) are fine to wrap. Xantham generates the bindings. The resulting Clef code lowers to JavaScript that imports from npm, and that's appropriate for those cases. Don't treat Xantham as a supply-chain compromise; treat it as the right tool for wrap-appropriate libraries.

**3. Plan Transcribe as a Horizon 3 investment for absorb-appropriate libraries.** The tooling described above (PSG reconstruction from JSIR, Clef printer, curation workflow) is non-trivial to build. It should be scoped to specific libraries with clear absorption value, not treated as a general "absorb every npm package" project. Crypto libraries, JSON parsers, BAREWire codecs, auth flow implementations are reasonable early candidates.

**4. Do not plan to build npm-free versions of Cloudflare services.** Durable Objects, KV, R2, and the rest are Cloudflare-owned runtimes. Composer does not absorb them; they are the target runtime. Xantham-style wrap bindings are the permanent right answer for these — npm-independence in this dimension means avoiding `@cloudflare/workers-types` as a runtime dep (which a Worker doesn't actually need at runtime anyway; the types are build-time only), not replacing Cloudflare's runtime.

**5. SDK documents (the Xantham patterns) remain the long-term result shape.** Even fully-absorbed libraries, once imported into Clef, should present a Clef API surface that looks like what Xantham would have generated from their TypeScript definitions. This keeps the application-author-facing experience consistent whether a given library is wrapped or absorbed. The mechanics underneath the surface differ; the surface itself is convergent.

## Cross-References

- [01_two_models.md](./01_two_models.md) — the architectural basis for why the absorb strategy is feasible
- [02_jsir_tooling.md](./02_jsir_tooling.md) — `jsir_gen`'s forward pipeline is what Transcribe reuses
- [03_four_wings.md](./03_four_wings.md) — different wings have different wrap/absorb profiles (Cloudflare wings heavy on wrap, browser UI and crypto-sensitive desktop wings heavier on absorb)
- [04_sdk_describes_runtime.md](./04_sdk_describes_runtime.md) — the wrap-strategy convention that remains for runtime-owned surfaces
- [Fidelity.CloudEdge/docs/00_architecture_decisions.md](../../../Fidelity.CloudEdge/docs/00_architecture_decisions.md) — the three-tier Fidelity.CloudEdge package architecture (all wrap-strategy for the Cloudflare runtime)
