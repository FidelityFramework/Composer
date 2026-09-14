# JavaScript targeting

**Design review: September 2026**

Composer's JavaScript target compiles Clef-native programs and libraries into JavaScript that satisfies the selected host contract. The output may differ substantially from the JavaScript produced by a vendor's TypeScript SDK. Correctness concerns the supported observable behavior, including failure and lifecycle behavior, rather than reproduction of the vendor's source or internal representations.

Clef does not acquire `obj` or `null` to reach this target. Foreign declarations and implementations supply evidence for binding generation. Analysis recovers what it can, retains unresolved constraints, and presents the remaining decisions through the design-time tooling. The selected structures enter the ordinary CCS/PSG/Baker/Alex pathway. JSHIR/JSIR realizes the settled computation below the witness boundary.

## Governing architecture

The [Thin Middle End doctrine](../Thin_Middle_End_Design.md) and [Backend Lowering Architecture](../../../clef-lang-spec/spec/backend-lowering-architecture.md) govern this folder:

- CCS owns semantic facts in the PSG. Dimensions, ranges, identities, effects, capture relationships and obligations remain available while they are useful.
- Baker fan-out composes recipes from Ingredients; generic fold-in incorporates their structure. Target realization does not bypass this separation.
- Alex's Library of Alexandria witnesses supported declaration and graph shapes through patterns and elements. Its witnessed vocabulary is `func`, `scf`, `arith`, `memref` and `index`.
- JSHIR/JSIR is a backend realization. No JavaScript-specific or Clef semantic dialect crosses the portable witness boundary.
- A lowering preserves each affected property or re-establishes it. A missing premise remains visible until it is resolved, explicitly assumed under a boundary contract, or diagnosed where commitment requires it.

The [JavaScript Boundary Semantics](../../../clef-lang-spec/spec/javascript-boundary.md) defines foreign values, narrowing, absence and failure. [Option Operations Representation](../../../clef-lang-spec/spec/option-operations-representation.md) separately defines interior Option realization.

## Current ground and intended work

| Area | Status and scope |
|---|---|
| F#/Fable | Working path for FSharp.CloudEdge bindings and existing Partas.Solid frontends. Fable has its own IR and transformations. |
| BAREWire JavaScript | Working Fable codecs and selected byte, framing and rejection tests. This does not establish Composer's JavaScript lowering. |
| FSharp.CloudEdge | September 13 selected delivery accepted; exact scope and limits are in its [acceptance record](../../../FSharp.CloudEdge/docs/SDK-DELIVERY-ACCEPTANCE-20260913.md). |
| Clef to JSHIR/JSIR | Design and implementation work. The JavaScript Substrate profile explicitly has no conforming implementation yet. |
| Clef-native foreign ingestion and dependency replacement | Design direction: analysis, deferred inference, developer curation and supported witnessing rules. No general automatic JavaScript-to-Clef recovery is claimed. |
| Atelier interaction | [Transcribe/Transpose design](../../../Atelier/docs/10_transcribe.md); the editor presents analysis and diagnostics rather than computing independent semantic facts. |

## Reading order

1. [Two source paths, one host contract](01_two_models.md): the compilation boundary and the meaning of different but valid output.
2. [JSHIR/JSIR tooling and lowering](02_jsir_tooling.md): the pinned analysis substrate, backend role and required correspondence.
3. [Deployment contexts and BAREWire](03_four_wings.md): Cloudflare, browsers, WebViews and the shared memory/IPC/wire contract.
4. [From foreign declarations to Clef-native bindings](04_sdk_describes_runtime.md): contract recovery, annotations, rule coverage and compiler ownership.
5. [Dependency replacement through deferred inference](05_supply_chain_and_transcribe.md): the interactive recovery loop and an artifact without third-party JavaScript dependencies.
6. [Opaque values and absence](06_obj_and_null_at_the_boundary.md): what can stay unknown, what must be checked, and what the backend can emit.
7. [Identity, preservation and acceptance](07_dependency_identity_and_validation.md): dependency provenance, proof scope and executable acceptance.
8. [Numeric selection and precision across strata](08_numeric_selection_and_precision.md): representation, arithmetic construction, transfer fidelity and design-time diagnostics through the JavaScript pathway.

## Design rationale

- [The Gift of Deferred Inference](../../../clef-lang-site/hugo/content/blog/deferred-inference.md): consistent partial programs retain open decisions until enough evidence exists.
- [Pondering Fearless Parallelism](../../../clef-lang-site/hugo/content/blog/pondering-fearless-parallelism.md): capacity, accuracy, permitted decomposition and placement have distinct obligations.
- [Carrying Proofs into JavaScript](../../../clef-lang-site/hugo/content/blog/carrying-proofs-into-javascript.md): proofs concern generated behavior and relationships across suspension, messaging and recovery.
- [JSIR: JavaScript as an MLIR Backend](../../../clef-lang-site/hugo/content/docs/design/javascript-targeting/jsir-javascript-as-mlir-backend.md): September status, carrier realization and the contract-to-artifact acceptance path.
- [The Foreign Pair](../../../clef-lang-site/hugo/content/docs/design/javascript-targeting/the-foreign-pair.md): explicit foreign values without a universal source type.
- [Fully Informed Bindings](../../../clef-lang-site/hugo/content/docs/design/javascript-targeting/fully-informed-bindings.md): declaration/body correspondence and conservative analysis.

WebAssembly has its own [targeting material](../wasm-targeting/README.md). A JavaScript artifact and a WASM module hosted by a Worker have distinct realization obligations.
