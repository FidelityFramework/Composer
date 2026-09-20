# Identity, preservation and acceptance

**Design review: September 2026**

Clef-native binding generation and dependency recovery need a traceable connection between the declared contract, accepted implementation, graph obligations, emitted JavaScript and selected host. Different JavaScript output is permitted; the required behavior and the evidence supporting it must survive.

## Retain a versioned package graph

Every version of the Clef library system must retain an inspectable snapshot of the complete dependency graph of its pinned source packages under the recorded resolution profiles. Package mapping is established independently of which SDK operations or application bodies are currently used. Capture the package graph before either offline SDK demand analysis or application reachability selects executable behavior. An unused package or edge remains in that version's record; its absence from an application artifact is a separate finding.

Bun supplies concrete precedents through its committed [lockfile](https://bun.com/docs/pm/lockfile) and [`bun why`](https://bun.com/docs/pm/cli/why), which explains dependency paths with versions, requested ranges and dependency roles. The [Bun source review](02_jsir_tooling.md#bun-internal-structure-and-the-ingestion-seam) complements that package mapping with module and binding relationships before tree shaking and generation. A Bun integration and the persistent Clef package mapping described here remain implementation work; the reviewed parser records alone do not supply a complete package-resolution snapshot.

| Retained information | Required identity and relationships |
|---|---|
| Library-system version | Snapshot identity, source package roots, owned Clef library versions and the predecessor snapshot when one exists. Accepted snapshots are immutable; an update creates a successor, retaining earlier snapshots for comparison. |
| Package instance | Package name, exact resolved version or source revision, source location, payload integrity/hash and resolution context. Distinct versions, payloads or peer environments must remain distinguishable even when names coincide. |
| Dependency edge | The requesting package instance, declared dependency name or alias and version request, and the exact resolved provider instance. Preserve direct and transitive paths, shared providers and cycles; a deduplicated name list loses the dependency structure. |
| Dependency role | Preserve regular, development, optional and peer declarations, including the selected peer provider. Record omitted optional dependencies, unsatisfied requests and the reasons or conditions for each unresolved or inactive edge. Keep build/deployment and runtime roles identifiable. |
| Resolution profile | Manifest and lockfile inputs, resolver version/settings, relevant platform and peer context, and module export conditions. Relate the package instance to its selected declaration and executable entry points without merging those identities. |
| Clef correspondence | Links from package instances and dependency edges to declaration owners, runtime modules and maintained Clef libraries. Record replaced behavior, retained foreign behavior and behavior not yet translated. |

The persisted form must retain the literal dependency tree or graph, with exact nodes and edges. A tree view may repeat shared nodes, but the underlying identities must expose sharing and cycles. Known omissions and unresolved relationships remain visible; they cannot be represented as a complete resolved closure. The design does not require resolving every possible platform configuration: each snapshot records the profiles it covers and their exclusions.

The source package graph underpins the three related graphs below. A behavioral demand graph selects bodies, initialization, captures and calls requiring recovery. Later application reachability selects from the maintained Clef implementation graph. Both selections refer back to the retained package graph; neither replaces it. Thus a package can remain part of the library system's provenance while contributing no executable code to a particular artifact. Keeping its mapping also does not require translating or type-checking all its bodies before a partial program can enter elaboration.

### Evaluate and track pin changes

A proposed pin change produces a candidate resolution snapshot and a before/after graph comparison. Record added or removed package instances and edges, version or payload changes, changed alias/peer providers, optional-dependency decisions and declaration/export resolution changes. Follow transitive changes even when a root package name or version is unchanged. Keep the old and proposed identities, the reason for the update and its assessment with the library-system version that accepts or defers it.

Cloudflare's Agents repository provides an existing upstream precedent. At the linked revision, its [install action](https://github.com/cloudflare/agents/blob/46760e635ce9599add0abbfe6c1a34af0d5d44f1/.github/actions/install/action.yml) uses a frozen lockfile, while its [AI SDK compatibility workflow](https://github.com/cloudflare/agents/blob/46760e635ce9599add0abbfe6c1a34af0d5d44f1/.github/workflows/ai-sdk-compat.yml) exercises majors 6 and 7, re-resolving the v6 dependencies before type-checking Think and running its Workers tests. These upstream resolutions and checks can supply a baseline for Clef's input capture and change assessment. Consumers still need the graph resolved for their own package roots and peer context; the upstream lockfile does not define every consumer installation.

Trace each change through the declaration, runtime-module and owned-library relationships to the affected source correspondence, pending constraints, assumptions and validation evidence. A package-level change identifies work to assess; it does not by itself establish that every converted function changed. Use finer source and obligation dependencies where available to preserve valid evidence and identify what requires reanalysis. If those relationships cannot justify a narrow impact assessment, widen it and record the reason. A change to currently unused behavior still appears in the package history, even when it does not require a new application artifact or immediate translation.

This assessment follows the [incremental library lifecycle](05_supply_chain_and_transcribe.md#incremental-library-lifecycle). Reconcile affected upstream declarations and bodies with maintained Clef source, retain developer refinements, and invalidate evidence whose premises no longer hold. Rerun the affected correspondence and oracle checks before carrying their support claims into the updated version. A pin change does not automatically overwrite curated source or restart conversion of the entire library system. Acceptance records connect the resulting source, graph snapshot and evidence; unresolved effects of the update remain explicit.

## Preserve three related graphs

| Graph | Facts to retain |
|---|---|
| Declaration graph | Canonical owners, aliases, parameters, applied arguments, constraints and the environment that resolved them. |
| Runtime module graph | Executable exports, import specifiers, selected export conditions and installed implementation versions. |
| Generated library graph | Which library owns each public Clef declaration, how consumers reference it, and its representation constraints, pending choices and eventual selected realization. |

One declaration can be reached through several public imports with different runtime implementations. Deduplicating its type must not erase those paths. Equal member lists do not establish a common declaration owner. A valid JavaScript import graph can coexist with an invalid generated-library ownership cycle.

Record established relationships and unresolved linkage constraints during ingestion; ordinary semantic enrichment resolves them as context accumulates. Alex consumes the facts settled for the required computation. It does not infer ownership from a symbol spelling, choose a constructor from the first interface-shaped edge or repair generic constraints at emission.

### Gauge implementation demand from bindings

The bindings' runtime import specifiers and exported-value identities provide roots for measuring the implementation they bring into a program. Type provenance and `.d.ts` references alone do not establish which implementation functions call which dependencies. Use a synthetic entry that retains the selected public exports, or an actual Fable consumer, and bundle it with pinned packages, tool version, export conditions and host externals. Record which of those two entry scopes was measured. Bun's [`metafile: true`](https://bun.com/docs/bundler#metafile) supplies input files and imports, output exports and per-input `bytesInOutput`; [esbuild's metafile](https://esbuild.github.io/api/#metafile) exposes corresponding measurements.

```text
Binding runtime imports/exports → selected-export entry or actual consumer
    → pinned bundle and module metadata
    → package-snapshot annotations and roots for implementation analysis
```

Report modules and package instances encountered separately from those contributing output bytes. This gauges the retained implementation for that entry and configuration; a metafile does not provide a precise function-call graph, runtime coverage or a minimal Clef replacement size. Refine body-level calls, captures and effects through the frontend analysis, retaining required module initialization and explicit unresolved dynamic imports or dispatch. Excluded host operations remain named boundaries. Attach the entry, settings, artifact and measurements to the same versioned package snapshot. These annotations guide recovery and comparison without deleting the full dependency tree or treating unobserved behavior as absent.

Build a directed dependency graph rooted at each SDK's public executable exports. Package edges retain exact versions, declared roles and resolution instances. Module edges identify the importing source file, import specifier, selected export condition and resolved provider. Symbol and function relationships refine those edges with the dependency implementation brought into the SDK. Preserve shared libraries, transitive paths, initialization and unresolved dynamic imports. This dependency mapping is a static source-analysis task.

Each library's percentage needs an explicit whole-library denominator. Inventory all implementation functions in the selected published distribution, including files absent from the SDK's import graph. Record distribution rules for parallel ESM/CommonJS, browser/Node and source/generated copies. Keep declarations and native/Wasm payloads separately identified. Merge repeated installations by exact package version, relative source path and verified content. Prebundled dependency code remains attributed to its containing file until source correspondence establishes its upstream owner.

The machine report contains every dependency in every SDK's pinned graph. Each SDK/library occurrence has a tuple `(SDK name and version, dependency name and version, mapped functions, total library functions, mapped percentage)`, with immediate importers and complete dependency paths. If the same library maps 2% into one SDK and 98% into another, retain both rows. Keep the denominator consistent for the same library distribution, and preserve worker, browser and tooling contexts on the supporting edges. Retain dependencies with no mapped implementation as rows with their resolution status.

An initial bundler-based implementation can map emitted source positions back to dependency function intervals and report statically retained functions. Label that measurement precisely: CommonJS, class members and dynamic dispatch can retain more functions than an exact call graph would select. Unknown imports remain explicit graph boundaries. On a pin change, compare package and import edges, regenerate affected function mappings and denominators, and retain the old and new graph snapshots with the library-system version.

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

1. **Inventory and environment.** Identify the library-system version and retained package-graph snapshot, selected public entry points, package contents, declaration providers, export conditions, host configuration and exact tool payloads. Record exclusions and unresolved imports. Keep the full package mapping distinct from the behavior selected for this artifact.
2. **Resolution and correspondence.** Authenticate declaration owners and runtime exports against their package instances. Relate lifted bodies to their declarations and retain unknown or lossy mappings. For changed pins, record the graph comparison, affected correspondence and disposition of invalidated or pending evidence.
3. **Clef contract.** Review candidate source, assumptions, supported witnessing rules and outstanding constraints. Elaborate actual producer and consumer libraries together through ordinary CCS/PSG, retaining unresolved constraints until the relevant commitment.
4. **Lowering.** Check the selected JSHIR route structurally and establish or re-check semantic correspondence for affected operations. Retain source/obligation provenance.
5. **Behavior.** Execute representative application and boundary cases: values, errors, absence, nested Options, returned and partial callbacks, aliasing and ordering. For codecs, include byte vectors, invalid extents and encoding failures.
6. **Host operation.** Load the artifact under its selected host profile. For Cloudflare, exercise required exports, imports, callback conventions, instance isolation and applicable suspension/recovery behavior. For a Solid/WebView profile, exercise the final embedded page's reactive updates, event dispatch, subscription disposal and native bridge shutdown. Service tests need controlled setup and cleanup.
7. **Artifact closure.** Bind evidence to emitted bytes, owned and retained dependencies, target policy, tool revisions and the retained library-system snapshot. Relate the artifact's executable closure to that snapshot without deleting unused package relationships from it. Claim only the supported behavior actually covered.

JSHIR round trips and differential execution against Fable or vendor output are useful checks. They must compare the declared observables rather than assume one implementation is universally correct. Normalization requires a justified relation; equal normalized IR is not itself a general semantic-equivalence proof. Mutation cases such as wrong offsets, endian reversal, dropped rejection handling or changed callback arity should demonstrate that relevant checks detect a broken correspondence.

For [JSX/Solid output](10_jsx_and_webview_toolchain.md#proofs-and-the-final-artifact), include the framework transform, bundler and embedding steps in the preservation chain. Evidence concerns the final assets and shipped dependency closure, with their tool/configuration and source/obligation identities. JSX parsing, source maps and hashes do not establish reactive behavior. Compare admitted structure before Solid compilation and execution after it; a shared Solid transform is a shared assumption in the Fable/Clef comparison. WrenHello's existing native UI gate is a bounded reference, not a completed Clef frontend proof.

### F#/Fable as an executable oracle

The [working F#/Fable pathway](01_two_models.md#an-executable-oracle-for-the-clef-pathway) provides executable reference behavior while Clef's JavaScript pathway develops. Use it for shared functional contracts, generated-binding interactions and BAREWire byte/failure cases. The existing [CloudEdge acceptance record](../../../FSharp.CloudEdge/docs/SDK-DELIVERY-ACCEPTANCE-20260913.md) and [ByteBridge fixture](../../../FSharp.CloudEdge/tests/ByteBridge/README.md) provide bounded starting evidence; the combined comparison harness described here is work to implement.

1. **Establish the comparison domain.** State related inputs, expected results/failures, callback conventions, observable identity/state relationships and numerical semantics. F#/.NET-specific behavior or a different numeric contract needs an explicit relation or a separate case; it does not silently define Clef behavior.
2. **Pin and identify each reference.** Record F# source, Fable/compiler/runtime versions, SDK and dependency payloads, execution host and boundary adapters. Distinguish a Fable binding that invokes vendor code from a corresponding algorithm implemented in F# and compiled by Fable.
3. **Run related cases.** Where available, compare the original dependency, the F#/Fable reference and the Clef-generated artifact. Give each run equivalent initial state and controlled inputs. Compare results, failures, effect order and multiplicity, relevant aliasing and contract-defined bytes. Adapters must preserve absence, nested Option and numeric distinctions rather than conceal a mismatch.
4. **Use disagreements to refine the analysis.** Locate whether the difference concerns declaration/body correspondence, foreign admission, recovered source, saturation, target realization or an incorrect reference assumption. Feed the finding and its provenance into the same obligations. The declared contract governs resolution; agreement between implementations does not override it.
5. **Bind observations to support claims.** Retain cases, traces, artifact identities and negative mutations. If Fable and another run share the same vendor implementation, that agreement checks the exercised adaptation and interaction, not an independent implementation of the algorithm. Numerical proofs and Clef-only capabilities retain their own acceptance requirements.

Reference cases can be prepared before Composer can execute the corresponding JavaScript, then reused as its feature coverage grows. A stronger Clef representation or different functional implementation remains acceptable when it preserves the agreed observations. Oracle evidence helps discover and test semantic requirements across the pipeline; it does not require Clef to reproduce Fable's output or dependencies.

Retain the fixtures and their contract relations through the [incremental library lifecycle](05_supply_chain_and_transcribe.md#incremental-library-lifecycle). Reuse valid evidence and rerun checks affected by source, contract, compiler or host changes. Equivalent initial execution state is a per-case testing requirement; it does not mean recreating or reconverting the library. A mismatch triggers analysis of the affected correspondence and its dependents.

### Combine analysis and validation evidence

The [tooling contribution map](02_jsir_tooling.md#contributions-to-a-fused-pipeline) applies across this acceptance sequence. Bun's source and dependency relationships can inform both ingestion and artifact closure. Dafny's contract and compiler-checking patterns can inform both recovered abstractions and executable preservation checks. JSHIR structure can expose relationships during ingestion, refinement and output validation. F#/Fable supplies executable oracle observations for the shared contracts. Several contributions can address one obligation.

Associate each result with the source/candidate/artifact identities it concerns, its premises, tool or adapted implementation, and transformation history. Distinguish structural validity, a discharged semantic obligation, an external assumption, a bounded execution result and a pending analysis result. Checks derived from the same model share that model's assumptions; multiple agreeing tools do not automatically establish independent corroboration.

Following Dafny's documented assertion/expectation pattern, a supported preservation predicate can guide both static reasoning and an executable check of the realized behavior. A failing execution invalidates the affected support claim; a passing execution covers its admitted case. For the reader fixture, the same callback-state and trace relation should govern source replacement, Option/closure realization and the negative mutations. Changes to the relation or its premises invalidate the dependent evidence across all contributing tools.

The [numeric implementation and acceptance guide](08_numeric_selection_and_precision.md#7-implementation-progression-and-acceptance) adds intermediate-capacity, rescaling, rounding-tree, residual, partial-transfer and recovery cases. Report numerical accuracy, reproducibility and execution cost separately. A representation-error score is not an application error bound, and a repeatable result is not evidence of accuracy.

## Current evidence boundaries

The FSharp.CloudEdge September 13 acceptance records a completed selected delivery, including a 50-project build, typed composition and bounded local Durable Object/Agents checks. Its [ByteBridge fixture](../../../FSharp.CloudEdge/tests/ByteBridge/README.md) records 13 checks through generated Workers body bindings and BAREWire in Node Fetch, including nonzero view offsets, detached buffers and typed failures. That fixture is not a general workerd or distributed-protocol proof.

BAREWire's [intersection-subset inventory](../../../BAREWire/docs/12%20Intersection%20Subset.md) distinguishes executable codec checks from solver queries about declarations. Declaration-level query results do not prove the JavaScript emitter or every emitted byte operation.

The [JavaScript Substrate profile](../../../clef-lang-spec/spec/javascript-boundary.md) remains design-stage with no conforming implementation. The requirements here describe how Clef-specific bindings and owned implementations earn supported JavaScript targeting, without borrowing a completion claim from the existing F#/Fable path.
