# Two source paths, one host contract

**Design review: September 2026**

F#/Fable and Clef/Composer can produce JavaScript for the same host while retaining different source languages, intermediate representations and implementation choices. The Clef pathway supports the design of Clef-native bindings and library implementations elaborated from JavaScript frontend facts and TypeScript/SDK constraints. Its output can differ from Fable output or the JavaScript emitted from a vendor's TypeScript source.

## The working F# path

Fable compiles F# through its own intermediate representation and target transformations to JavaScript. FSharp.CloudEdge supplies generated F# bindings; Partas.Solid supplies the existing F# frontend surface. Their behavior tests provide useful characterization of host contracts and library interactions.

Fable is neither a CLR-IL translator nor a string-template wrapper. Its use does not inherently require every application operation to import an npm library: host APIs, compiler support code and third-party libraries are separate dependency categories. The actual emitted dependency closure determines what an application carries.

### An executable oracle for the Clef pathway

The working F#/Fable pathway has high value as an oracle throughout this development. Existing binding and application cases characterize actual JavaScript/host interactions. Corresponding F# implementations of functional fixtures can also provide executable reference behavior for Option, closures and callback composition while Composer's JavaScript realization is being built. BAREWire fixtures contribute concrete encoding, decoding and rejection observations.

Compare implementations under an explicit shared contract: related inputs, results, failures, state changes and invocation traces. F# and Clef can express that contract differently, and their emitted JavaScript can have different representations. The oracle supplies observations against which recovery, refinement and lowering can be checked. Its source and generated JavaScript can also help localize a disagreement between those stages.

Record which behavior the reference actually exercises. A Fable binding calling the original vendor dependency characterizes that boundary; an F# implementation of the dependency's algorithm exercises a separate realization. Both are useful. Their assumptions, runtime dependencies and shared components remain visible in the [oracle comparison procedure](07_dependency_identity_and_validation.md#ffable-as-an-executable-oracle).

## The Clef path

```text
Clef program + Clef-native libraries + declared foreign boundaries
    -> CCS: PSG with types, relationships, effects and obligations
    -> Baker: recipe fan-out and generic fold-in
    -> Alex: Library of Alexandria patterns and elements
    -> portable MLIR, with useful PSG/codata retained alongside it
    -> JavaScript backend: JSHIR/JSIR realization
    -> JavaScript module and its declared host imports/exports
```

The [backend specification](../../../clef-lang-spec/spec/backend-lowering-architecture.md) fixes the boundary. Alex emits `func`, `scf`, `arith`, `memref` and `index`. JSIR belongs to the target pathway. Backend lowering may read structural identity from the PSG; it does not recover semantics by guessing from emitted instruction sequences or symbol spellings.

For a Solid UI profile, the target handoff can include JSX that retains component and reactive intent for the Solid compiler. Vite then coordinates compilation and bundling for the WebView. This adds downstream transformations to the preservation chain; it does not make JSX necessary for ordinary JavaScript modules. The [WREN toolchain guide](10_jsx_and_webview_toolchain.md) separates the existing Fable producer from proposed Clef lowering and JSIR JSX extensions.

The JavaScript frontend contributes implementation structure, established relationships and remaining inference variables to ordinary CCS/PSG elaboration. Consistent partial graphs remain available for further inference; a fully resolved implementation is not required before ingestion. Build or REPL commitment requires the premises of the computation being realized. Frontend JSHIR provides foreign-program analysis input, while the forward JSIR backend consumes the witnessed portable computation.

Representation and analysis remain distinct. Source dimensions identify quantities. Range and relational evidence constrain numeric selection. Captures, sharing and lifetime constrain closure realization. Platform capabilities constrain which forms are available. These facts can remain pending during elaboration, but a concrete commitment must have its required premises.

### Portable carriers, host realizations

The portable storage carrier does not force an ordinary JavaScript record to acquire a native byte layout. The pathway can realize record access as property access and a closure as a host function, while preserving initialization, capture mode, case identity and the other required observables. Byte-backed BAREWire views still require their explicit offset, extent and encoding contracts.

| Clef structure | Required JavaScript realization property |
|---|---|
| Record | Preserve declared fields and accesses; a host object is permitted without introducing source `obj`. |
| Closure | Preserve actual callable boundaries, immutable snapshots and shared mutable captures. A host function is permitted. |
| Option | Preserve Some/None and nested distinctions; use the per-instantiation erased or reified form admitted by the Option specification. |
| Foreign boundary | Preserve declared conversion, calling convention, failure behavior and tracked foreign contact. |
| Numeric computation | Preserve dimensions and the selected arithmetic contract, including exactness or admitted error. |
| Suspension | Preserve the continuation's required state and the validity of facts used after resumption. |

These are obligations of the realization, not a list of completed JSIR features. HOFs do not bypass Baker merely because JavaScript has similarly named library functions. Any selected alternative decomposition must preserve the operation contract and be established above witnessing.

## Different output is a supported outcome

A vendor may implement a protocol with mutable property bags, sentinel values and callback adapters. A recovered Clef implementation may use records, unions, Option, Result and explicitly captured computations. Its emitted JavaScript may have different functions, object layouts, control flow and module boundaries.

The comparison is against the supported behavior at the application and host boundaries. Relevant observations can include values, errors, property presence, identity and aliasing, callback invocation, asynchronous ordering, resource release and message encoding. Internal choices may change where those observations are preserved. A changed public Clef API can be intentional; the generated foreign boundary must still satisfy the contract on its other side.

This freedom is what makes replacing a subsidiary JavaScript library useful. The accepted Clef implementation can provide its required behavior without reproducing its incidental source organization or retaining its runtime import. [Dependency replacement](05_supply_chain_and_transcribe.md) describes that work separately from binding generation.

## What survives erasure

[Carrying Proofs into JavaScript](../../../clef-lang-site/hugo/content/blog/carrying-proofs-into-javascript.md) places the argument on behavior. A payload can remain opaque while the graph establishes which request it belongs to, which continuation may consume it, or which accepted contribution it represents. Those properties do not require complete knowledge of its internal shape.

A lowering must preserve or re-establish the relationships used by the proof. Runtime checks can establish observable premises for incoming values. Host scheduling, storage and foreign implementation guarantees remain named assumptions where they are relied upon. Source annotations disappearing from JavaScript does not make their justified behavior disappear.

## Acceptance

Fable and vendor outputs are useful comparison implementations, with their versions and assumptions recorded. Matching their text or normalized JSHIR is not the acceptance criterion. The Clef pathway must satisfy its declared contract, preserve the affected properties through lowering, link under the selected host profile and execute the relevant success and failure cases.

The JavaScript Substrate profile remains design-stage. See [identity and acceptance](07_dependency_identity_and_validation.md) for the evidence needed to claim support.
