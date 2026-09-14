# JavaScript targeting

**Design review: September 2026**

Composer's JavaScript target compiles Clef-native programs and libraries into JavaScript that satisfies the selected host contract. The output may differ substantially from the JavaScript produced by a vendor's TypeScript SDK. Correctness concerns the supported observable behavior, including failure and lifecycle behavior, rather than reproduction of the vendor's source or internal representations.

Clef does not acquire `obj` or `null` to reach this target. A single coordinated foreign-language frontend combines JavaScript structure with TypeScript declarations and SDK uses. It can accept several contributing tools and representations. JSHIR is the worked ingestion candidate; the [substrate comparison](02_jsir_tooling.md#ingestion-substrate-choice) keeps interfaces and useful combinations open. The frontend contributes candidate Clef and available structural and semantic evidence to ordinary CCS/PSG elaboration. Consistent unresolved types, ranges, effects and representation choices remain pending as application and target context arrives. The frontend need not resolve them all before ingestion.

Build or REPL evaluation commits the required portion of the computation. The obligations needed for that commitment must then be established; unrelated open work need not be settled. A foreign value used only through an admitted opaque contract can remain opaque throughout execution. An unresolved inference variable is not automatically `JsValue`. In forward compilation, Baker and Alex retain their existing roles, and JSHIR/JSIR realizes the witnessed portable computation in the backend.

The [tooling contribution map](02_jsir_tooling.md#contributions-to-a-fused-pipeline) gives Bun high weight for JavaScript structure, binding/dependency relationships and transformations; Dafny high weight for semantic contracts, functional/numeric realization and preservation testing; JSIR high weight for structured analysis and target realization; and the working F#/Fable pathway high immediate weight as an executable oracle. These are overlapping strengths across a fused pipeline, not exclusive stage assignments. Their contributions meet through shared identities and obligations in ordinary elaboration, with borrowed ideas, candidate integrations and established results distinguished.

## Governing architecture

The [Thin Middle End doctrine](../Thin_Middle_End_Design.md) and [Backend Lowering Architecture](../../../clef-lang-spec/spec/backend-lowering-architecture.md) govern this folder:

- CCS owns semantic facts in the PSG. Dimensions, ranges, identities, effects, capture relationships and obligations remain available while they are useful.
- Baker fan-out composes recipes from Ingredients; generic fold-in incorporates their structure. Target realization does not bypass this separation.
- Alex's Library of Alexandria witnesses supported declaration and graph shapes through patterns and elements. Its witnessed vocabulary is `func`, `scf`, `arith`, `memref` and `index`.
- In the worked frontend route, JSHIR is foreign-program analysis input. In forward lowering, JSHIR/JSIR belongs to the backend; no JavaScript-specific or Clef semantic dialect enters Alex's portable witnessed vocabulary.
- A lowering preserves each affected property or re-establishes it. A missing premise remains visible until it is resolved, explicitly assumed under a boundary contract, or diagnosed where commitment requires it.

The [JavaScript Boundary Semantics](../../../clef-lang-spec/spec/javascript-boundary.md) defines foreign values, narrowing, absence and failure. [Option Operations Representation](../../../clef-lang-spec/spec/option-operations-representation.md) separately defines interior Option realization.

## The Clef library ecosystem being built

The intended destination is a set of Clef-native SDKs and supporting Clef libraries whose reachable implementations participate in the same semantic graph as application code. Xantham supplies the declaration and ownership analysis; the JavaScript frontend extends the ingestion to the dependency behavior those SDKs actually call. Original package payloads remain pinned inputs for provenance, comparison and regeneration.

| Input or facility | Place in the Clef ecosystem |
|---|---|
| TypeScript SDK declarations and executable entry points | Clef SDK declarations and implementations, with the required host boundaries. |
| Called dependency behavior and its necessary initialization/effects | Owned supporting Clef libraries, linked by resolved declaration and implementation identity. |
| A dependency shared by several SDKs | Shared ownership where identity and compatible contracts establish it; preserve per-use instantiations and runtime-linkage provenance. |
| Platform-supplied operations | Declared host capabilities and boundary bindings. |
| Application entry points | Demand for ordinary reachability, elaboration, refinement and final compilation across these libraries. |

This creates room to refine dependency implementations into functional Clef structures while preserving boundary data contracts and required behavior. As context accumulates, shared graph analysis can expose simplifications across library boundaries. A deployed artifact can then contain compiled Clef implementations and declared host calls without carrying the replaced vendor runtime code. The [worked frontend guide](09_contract_directed_dependency_recovery.md) connects this destination to partial inference, source generation and executable acceptance.

The converted libraries and their accumulated correspondence, constraints and validation evidence are durable project assets. Later work follows an [incremental lifecycle](05_supply_chain_and_transcribe.md#incremental-library-lifecycle): new applications use the existing Clef libraries, new context refines their uses, and changed inputs invalidate affected relationships. Expanded demand adds newly required behavior. Ordinary builds do not restart foreign conversion, and functional refactoring continues from the owned Clef source.

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

1. [Two source paths, one host contract](01_two_models.md): the compilation boundary, the F#/Fable oracle and the meaning of different but valid output.
2. [JavaScript tooling, analysis and lowering](02_jsir_tooling.md): Bun, Dafny, JSIR and F#/Fable contributions, their relative weighting and combined use, pinned source reviews and preservation requirements.
3. [Deployment contexts and BAREWire](03_four_wings.md): Cloudflare, browsers, WebViews and the shared memory/IPC/wire contract.
4. [From foreign declarations to Clef-native bindings](04_sdk_describes_runtime.md): contract recovery, annotations, rule coverage and compiler ownership.
5. [Dependency replacement through deferred inference](05_supply_chain_and_transcribe.md): the interactive recovery loop and an artifact without third-party JavaScript dependencies.
6. [Opaque values and absence](06_obj_and_null_at_the_boundary.md): what can stay unknown, what must be checked, and what the backend can emit.
7. [Identity, preservation and acceptance](07_dependency_identity_and_validation.md): dependency provenance, proof scope and executable acceptance.
8. [Numeric selection and precision across strata](08_numeric_selection_and_precision.md): representation, arithmetic construction, transfer fidelity and design-time diagnostics through the JavaScript pathway.
9. [JavaScript frontend and deferred dependency translation](09_contract_directed_dependency_recovery.md): a worked TypeScript/JSHIR-to-Clef example, partial elaboration, offline and application reachability, and the owned SDK dependency edge.

## Design rationale

- [The Gift of Deferred Inference](../../../clef-lang-site/hugo/content/blog/deferred-inference.md): consistent partial programs retain open decisions until enough evidence exists.
- [Pondering Fearless Parallelism](../../../clef-lang-site/hugo/content/blog/pondering-fearless-parallelism.md): capacity, accuracy, permitted decomposition and placement have distinct obligations.
- [Carrying Proofs into JavaScript](../../../clef-lang-site/hugo/content/blog/carrying-proofs-into-javascript.md): proofs concern generated behavior and relationships across suspension, messaging and recovery.
- [JSIR: JavaScript as an MLIR Backend](../../../clef-lang-site/hugo/content/docs/design/javascript-targeting/jsir-javascript-as-mlir-backend.md): September status, carrier realization and the contract-to-artifact acceptance path.
- [The Foreign Pair](../../../clef-lang-site/hugo/content/docs/design/javascript-targeting/the-foreign-pair.md): explicit foreign values without a universal source type.
- [Fully Informed Bindings](../../../clef-lang-site/hugo/content/docs/design/javascript-targeting/fully-informed-bindings.md): declaration/body correspondence and conservative analysis.

WebAssembly has its own [targeting material](../wasm-targeting/README.md). A JavaScript artifact and a WASM module hosted by a Worker have distinct realization obligations.
