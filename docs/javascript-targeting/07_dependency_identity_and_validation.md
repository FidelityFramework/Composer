# Identity, preservation and acceptance

**Design review: September 2026**

Clef-native binding generation and dependency recovery need a traceable connection between the declared contract, accepted implementation, graph obligations, emitted JavaScript and selected host. Different JavaScript output is permitted; the required behavior and the evidence supporting it must survive.

## Preserve three related graphs

| Graph | Facts to retain |
|---|---|
| Declaration graph | Canonical owners, aliases, parameters, applied arguments, constraints and the environment that resolved them. |
| Runtime module graph | Executable exports, import specifiers, selected export conditions and installed implementation versions. |
| Generated library graph | Which library owns each public Clef declaration, how consumers reference it, and its representation constraints, pending choices and eventual selected realization. |

One declaration can be reached through several public imports with different runtime implementations. Deduplicating its type must not erase those paths. Equal member lists do not establish a common declaration owner. A valid JavaScript import graph can coexist with an invalid generated-library ownership cycle.

Record established relationships and unresolved linkage constraints during ingestion; ordinary semantic enrichment resolves them as context accumulates. Alex consumes the facts settled for the required computation. It does not infer ownership from a symbol spelling, choose a constructor from the first interface-shaped edge or repair generic constraints at emission.

## Concrete lessons from Xantham and CloudEdge

The September FSharp.CloudEdge exercise provides transferable regression cases. Its F# encodings and remaining mapping losses are not Clef implementation prescriptions.

| Observed boundary | Requirement for the Clef path |
|---|---|
| Shared declarations reached through different public imports | Keep canonical type identity separate from per-use executable linkage. |
| Producer and consumer infer different generic bounds | Keep declaration parameters, contextual bounds and applied arguments in their proper scopes. |
| A nullable alias gains a second Option layer or loses nullability | Preserve the complete absence contract through aliases and generation contexts. |
| Nearby aliases or compiler-interned IDs change ownership | Authenticate declaration provenance independently from incidental names and the broader input fingerprint. |
| A generic empty marker widens to `obj` in a consumer | Preserve meaningful owner and argument identity even when the runtime member list is empty. |
| A class's `implements` and `extends` edges collapse together | Separate declared conformance, executable inheritance and what the target representation supports. |
| A callback compiles under a curried alias but fails when returned or partially applied | Preserve actual calling convention and argument boundaries independently of type names and generic arity. |
| An erased union hides a widening behind an alias | Report the actual loss consistently; normalization does not recover the erased information. |

The accepted [CloudEdge delivery](../../../FSharp.CloudEdge/docs/SDK-DELIVERY-ACCEPTANCE-20260913.md) records generator, compilation, typed-composition and bounded runtime checks. Its documented generic RPC widening and callback-injection limitations remain evidence for analysis. Do not import `obj`, delegates or F# representation restrictions into Clef merely because they occur in that output.

## Deferred obligations and developer input

The [deferred-inference discipline](../../../clef-lang-site/hugo/content/blog/deferred-inference.md) permits consistent partial programs while facts accumulate. Keep established relationships alongside unresolved requirements. A pending range, unknown callback retention policy and unsupported witness shape are different findings and need different remedies.

An unresolved inference variable is not automatically `JsValue`. A value intentionally admitted through an opaque foreign contract may remain opaque throughout execution, while constraints required for its uses still apply. Partial foreign-program structure enters ordinary CCS/PSG elaboration without first completing all type or representation decisions.

Atelier/LSP should present the premise, provenance, affected use and available sources of evidence. A developer may establish intent or supply a documented external contract. That input must not silently turn an assumption into a proof. If analysis can settle a choice, it should do so without demanding an early manual representation decision.

At build or REPL commitment, each property required by the selected reachable computation needs sufficient evidence, a sound generated boundary check where the contract permits one, or an explicit permitted external premise. This does not require settling unrelated partial work or discovering every opaque payload's structure. Known contradictions and established unavailable capabilities are located findings during elaboration. A solver timeout or unknown result remains unresolved evidence.

## Preserve behavior through each affected edge

[Carrying Proofs into JavaScript](../../../clef-lang-site/hugo/content/blog/carrying-proofs-into-javascript.md) identifies the argument's scope: generated behavior can preserve a property after source annotations disappear. Source typing, graph relationships, numeric laws, protocol state and runtime assumptions supply different parts of that argument.

| Property | Correspondence to establish |
|---|---|
| Optional value | Some/None, nested distinctions, eager operands and conditional callback invocation survive realization. Boundary absence remains position-specific. |
| Closure | Definitions, calls and returns agree on actual argument boundaries; immutable captures are snapshots and mutable captures share the intended cells. |
| Foreign narrowing | Every admitted value satisfies the required predicate, failures are typed, and the premise remains valid until use. |
| Numeric computation | Selected Number/BigInt or other supported construction preserves the admitted range, exactness or error contract. Bitwise coercions cannot silently truncate it. |
| BAREWire access | Buffer origin, offsets, extent, endian order, numeric conversion and failure behavior implement the agreed encoding. |
| Suspension and joins | Captured facts remain valid; replies match the logical computation; required contributions are accepted with the specified multiplicity. |
| Durable recovery | Retained data, acceptance state and control position reconstruct the required computation under the declared storage and retry contract. |

An exact accumulator does not prevent duplicate contribution acceptance. A well-formed reply does not prove its producer computed correctly. A decoded control frame does not establish session agreement. Unknown foreign effects must remain in the model across composition and suspension.

The [backend specification](../../../clef-lang-spec/spec/backend-lowering-architecture.md) permits the pathway to read useful PSG/codata during realization. A transformation that can disturb a carried property needs preservation evidence or a re-check. Shared derivation of native and JavaScript codecs provides a common reference, not automatic proof that either lowering implements it.

## Dependency-free artifact claim

The deployed artifact's dependency closure includes bundled implementations as well as explicit imports. To claim no third-party JavaScript implementation dependencies, record which required behaviors are compiled from owned Clef and which are supplied by the declared host. Any retained foreign implementation remains a dependency, even if inlined or renamed.

The build toolchain and deployment tooling have separate closures. Host/runtime assumptions remain explicit. Declaration-only packages are analysis inputs, not executable SDKs. Bounded source replacement can remove a runtime dependency without proving all behavior of the original package or all Cloudflare services.

## Acceptance sequence

This sequence records acceptance of a concrete artifact; it is not a requirement that every fact be resolved before a partial program joins the PSG.

1. **Inventory and environment.** Identify selected public entry points, package contents, declaration providers, export conditions, host configuration and exact tool payloads. Record exclusions and unresolved imports.
2. **Resolution and correspondence.** Authenticate declaration owners and runtime exports. Relate lifted bodies to their declarations and retain unknown or lossy mappings.
3. **Clef contract.** Review candidate source, assumptions, supported witnessing rules and outstanding constraints. Elaborate actual producer and consumer libraries together through ordinary CCS/PSG, retaining unresolved constraints until the relevant commitment.
4. **Lowering.** Check the selected JSHIR route structurally and establish or re-check semantic correspondence for affected operations. Retain source/obligation provenance.
5. **Behavior.** Execute representative application and boundary cases: values, errors, absence, nested Options, returned and partial callbacks, aliasing and ordering. For codecs, include byte vectors, invalid extents and encoding failures.
6. **Host operation.** Load the artifact under its selected Cloudflare profile. Exercise required exports, imports, callback conventions, instance isolation and applicable suspension/recovery behavior. Service tests need controlled setup and cleanup.
7. **Artifact closure.** Bind evidence to emitted bytes, owned and retained dependencies, target policy and tool revisions. Claim only the supported behavior actually covered.

JSHIR round trips and differential execution against Fable or vendor output are useful checks. They must compare the declared observables rather than assume one implementation is universally correct. Normalization requires a justified relation; equal normalized IR is not itself a general semantic-equivalence proof. Mutation cases such as wrong offsets, endian reversal, dropped rejection handling or changed callback arity should demonstrate that relevant checks detect a broken correspondence.

### Combine analysis and validation evidence

The [Bun, Dafny and JSIR contribution map](02_jsir_tooling.md#contributions-to-a-fused-pipeline) applies across this acceptance sequence. Bun's source and dependency relationships can inform both ingestion and artifact closure. Dafny's contract and compiler-checking patterns can inform both recovered abstractions and executable preservation checks. JSHIR structure can expose relationships during ingestion, refinement and output validation. Several contributions can address one obligation.

Associate each result with the source/candidate/artifact identities it concerns, its premises, tool or adapted implementation, and transformation history. Distinguish structural validity, a discharged semantic obligation, an external assumption, a bounded execution result and a pending analysis result. Checks derived from the same model share that model's assumptions; multiple agreeing tools do not automatically establish independent corroboration.

Following Dafny's documented assertion/expectation pattern, a supported preservation predicate can guide both static reasoning and an executable check of the realized behavior. A failing execution invalidates the affected support claim; a passing execution covers its admitted case. For the reader fixture, the same callback-state and trace relation should govern source replacement, Option/closure realization and the negative mutations. Changes to the relation or its premises invalidate the dependent evidence across all contributing tools.

The [numeric implementation and acceptance guide](08_numeric_selection_and_precision.md#7-implementation-progression-and-acceptance) adds intermediate-capacity, rescaling, rounding-tree, residual, partial-transfer and recovery cases. Report numerical accuracy, reproducibility and execution cost separately. A representation-error score is not an application error bound, and a repeatable result is not evidence of accuracy.

## Current evidence boundaries

The FSharp.CloudEdge September 13 acceptance records a completed selected delivery, including a 50-project build, typed composition and bounded local Durable Object/Agents checks. Its [ByteBridge fixture](../../../FSharp.CloudEdge/tests/ByteBridge/README.md) records 13 checks through generated Workers body bindings and BAREWire in Node Fetch, including nonzero view offsets, detached buffers and typed failures. That fixture is not a general workerd or distributed-protocol proof.

BAREWire's [intersection-subset inventory](../../../BAREWire/docs/12%20Intersection%20Subset.md) distinguishes executable codec checks from solver queries about declarations. Declaration-level query results do not prove the JavaScript emitter or every emitted byte operation.

The [JavaScript Substrate profile](../../../clef-lang-spec/spec/javascript-boundary.md) remains design-stage with no conforming implementation. The requirements here describe how Clef-specific bindings and owned implementations earn supported JavaScript targeting, without borrowing a completion claim from the existing F#/Fable path.
