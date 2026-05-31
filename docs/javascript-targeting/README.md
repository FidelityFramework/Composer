# JavaScript Targeting

**SpeakEZ Technologies | Fidelity Framework**
**April 2026**

JavaScript is a compilation target for Clef. Two distinct models reach that target, both of which Composer supports as first-class citizens of the Fidelity ecosystem. This subfolder documents the distinctions between them, the tooling `jsir_gen` brings to bear, and the deployment surface (four JavaScript "wings") that absorbs the output.

## What's Here

- [01_two_models.md](./01_two_models.md) — The F#/.NET model (Partas.Solid via Fable) vs. the fully decomposed AST model (JSIR via Composer). Includes the js_of_ocaml precedent that is the closer architectural referent for Composer than Fable is, and why F#'s relationship to the CLR forecloses the js_of_ocaml-style path for Fable.
- [02_jsir_tooling.md](./02_jsir_tooling.md) — What `jsir_gen` actually is: pass names, AST substrate (Babel), dialects, and what Composer invokes.
- [03_four_wings.md](./03_four_wings.md) — The four deployment wings: Cloudflare Workers/DOs, Browser UI, WebView desktop (WrenHello), Shared BAREWire. What's common and what differs.
- [04_sdk_describes_runtime.md](./04_sdk_describes_runtime.md) — The SDK-describes-the-runtime convention. Glutinum/Hawaii are not Clef products. Composer does not rebuild Cloudflare tooling or Solid.
- [05_supply_chain_and_transcribe.md](./05_supply_chain_and_transcribe.md) — Supply chain independence from npm as a first-class design property, following js_of_ocaml's shape. The wrap strategy (Xantham/Farscape bindings) and the absorb strategy (Transcribe source lifting), with guidance for when each applies and what tooling downstream agent work will need to build.
- [06_obj_and_null_at_the_boundary.md](./06_obj_and_null_at_the_boundary.md) — How Clef handles the `obj` and `null` concepts it doesn't have, at the boundary with runtimes that treat both as first-class citizens. Covers the three strategies (opaque handles, typed `JsValue` DU, schema-directed narrowing), the risk/mitigation story for schema validation, and the direct parallel to Rust's `serde` + `serde-wasm-bindgen` approach that proves the pattern is production-viable.

## Why a Separate Subfolder

The primary Composer backend documents (`JSIR_Backend_Design.md`, `JavaScript_Backend_Design.md`) specify how Alex emits JSIR ops and how `jsir_gen` transforms those ops to JavaScript. They answer the question "how does Composer emit JavaScript?"

This subfolder answers a different set of questions:

1. **Which model are we in?** Composer supports two paths to JavaScript output. One of them does not use JSIR at all; Fable continues to serve F#/.NET code that reaches JavaScript through Partas.Solid and the existing .NET toolchain. Both paths coexist because both have use cases that benefit from them.

2. **What does `jsir_gen` actually do?** Earlier drafts of the backend design referred to a non-existent `jsir-lift` tool and to ESTree as the AST substrate. Neither is correct. JSIR ships one binary, `jsir_gen`, and it uses Babel AST. The details matter when implementing the backend.

3. **Where does the output go?** JavaScript targeting at Composer spans four deployment contexts with different runtime environments and different SDK surfaces. Cloudflare is the largest of these but not the only one, and the architecture does not privilege it.

4. **What does Composer not do?** Composer does not reimplement Solid, Durable Objects, KV, R2, or any other runtime service. SDKs describe those runtimes; Composer emits JavaScript that calls into them.

## Companion Subfolder

- [wasm-targeting/](../wasm-targeting/) — the WASM story as a distinct target family, with the two pathways (LLVM WASM and WAMI/MLIR WASM), the DCont-via-Coroutines vs. DCont-Native (Stack Switching / JSPI) continuation strategies, and Cloudflare's step-graded compute model where WASM sits between pure JS Workers and Containers. Start there when the question is "should this compute go in a Worker or a WASM-in-Worker or a Container" rather than "should this compute be authored in F# or Clef."

## Position in the Broader Docs

This subfolder is referenced from:

- [JSIR_Backend_Design.md](../JSIR_Backend_Design.md) — the superseding backend specification
- [JavaScript_Backend_Design.md](../JavaScript_Backend_Design.md) — the pre-JSIR design document (remains valid for witness architecture)
- [Alex_Architecture_Overview.md](../Alex_Architecture_Overview.md) — the Elements/Patterns/Witnesses framework
- [Fidelity.CloudEdge/docs/10_jsir_strategic_assessment.md](../../../Fidelity.CloudEdge/docs/10_jsir_strategic_assessment.md) — the ecosystem-level strategic framing

## Reading Order

New readers should start with [01_two_models.md](./01_two_models.md) to internalize the core distinction (including the js_of_ocaml precedent that shapes Composer's approach more than Fable does), then read [02_jsir_tooling.md](./02_jsir_tooling.md) for the corrected tooling reality, then [03_four_wings.md](./03_four_wings.md) for the deployment surface, then [04_sdk_describes_runtime.md](./04_sdk_describes_runtime.md) for the wrap-strategy SDK convention that covers Cloudflare-owned runtime surfaces, and finally [05_supply_chain_and_transcribe.md](./05_supply_chain_and_transcribe.md) for the npm-independence property, the absorb strategy that complements wrapping, and the downstream-agent-work implications that follow from both.
