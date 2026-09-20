# JSX and the WREN frontend toolchain

**Design review: September 15, 2026.** WrenHello supplies a working F#/Fable frontend and Composer native host. Clef-to-JSX lowering, JSIR JSX support and unified Composer build orchestration described here are proposed work.

**UI direction updated September 18, 2026.** The [Fidelity.UI reconsideration](../../../Fidelity.UI/docs/08_ui_model_reconsideration.md) makes a new Clef-native reactive-area update model the design priority, with quiet functional composition preferred and optional computation expressions over the same portable semantics. WrenHello and its Solid toolchain are experimental reference hosts. This document develops a candidate JSX/Solid realization; it does not select Solid, Fabulous-style CEs or Elmish as requirements of the portable UI architecture.

## The output contract

JSX is a useful structured handoff to a framework compiler. For WREN applications, emitting Solid-compatible JSX lets the Solid compiler realize declarative views as DOM construction and reactive updates. It also produces inspectable output and a substantial reference corpus. General JavaScript libraries, codecs and Worker entry points still need ordinary JavaScript realization; JSX does not supply their semantics.

The preferred Clef authoring model uses quiet functional constructors and composition. Optional CEs elaborate to the same owned UI and reactive-area operations. Fabulous, Partas.Solid and Fable.Ripple contribute design evidence rather than a required API shape or runtime. Pure Elmish model/update functions remain an optional state discipline, with selective projections into the reactive model. View construction and reactive updates are separate responsibilities: mount an owned factory, preserve its reactive reads and dispose subscriptions when their owner ends. An Elmish update need not reconstruct the entire DOM or introduce a second authoritative application model.

The UI construction contract starts cold. Composing a view must not create live DOM/native controls, subscribe to producers, start validation or dispatch workers. Explicit owned activation admits the required demand, normally at first mount or earlier through a service/preparation scope; stale unobserved derivations remain unevaluated. `Cold<Incremental<'T>>` can defer graph construction as well as the demanded evaluation supplied by `Incremental<'T>`. Existing `Effect.create` semantics remain active: a cold UI factory delays that call until activation. The proposed Solid adapter must preserve these distinctions, including separate state for independent instances, rather than inherit eager setup merely because its generated JavaScript is valid.

Demand can intentionally remain active outside a visible surface. A background observer may keep a time-series projection current for fast view switching while visual layout/paint remain inactive. A preparation scope can demand further view stages within a budget; later attachment to that instance must not repeat setup or duplicate subscriptions. Shared source and independent prepared-instance lifetimes must survive lowering. Retaining a stale cache, maintaining a current projection and preparing a complete view are distinct policies and conformance cases.

The portable contract must distinguish dependency granularity, area-update granularity and presentation damage. A reactive area can own an update without requiring a separate thread, actor or framebuffer. Native and DOM realizations must preserve its identity, input versions, lifetime and commit behavior. Compare leaf bindings, area recomputation and hybrids before fixing that granularity. Solid is a candidate browser adapter and comparator; Clef-owned incremental behavior with direct DOM updates is another candidate. WrenHello's existing packaging reduces integration work for a Solid experiment, but does not establish its fit to the new area semantics.

Existing Partas.Solid is an F# library with Fable-specific compilation support. A Clef counterpart needs its own library contracts and supported lowering. TypeScript bindings alone do not implement that transformation, and Composer cannot simply execute a Fable plugin as a Clef lowering rule.

## Existing and proposed paths

```text
Existing WrenHello:
  F# / Partas.Solid → Fable → .fs.jsx
                               │
  Vite invokes vite-plugin-solid
    → Babel with babel-preset-solid → JavaScript
    → bundle JS/CSS/assets with vite-plugin-singlefile → dist/index.html
    → scripts/weld.js → EmbeddedAssets.fs → Composer native binary
    → system WebView executes the frontend

Proposed additional producer for the candidate Solid profile:
  Clef UI → CCS/PSG → Baker → Alex's portable MLIR
    → Clef-specific JavaScript/JSX realization, retaining PSG/codata
    → extended JSHIR + native Babel-AST bridge → Babel generator → .jsx
    → the same Solid/Vite/bundle/weld path
```

The existing path is visible in WrenHello's [package scripts](../../../WrenHello/package.json), [Vite configuration](../../../WrenHello/vite.config.js), [weld](../../../WrenHello/scripts/weld.js) and [build guide](../../../WrenHello/README.md#the-weld). Those scripts currently coordinate the stages; `composer build` does not yet orchestrate them. The frontend is compiled by Fable and the native host by Composer.

Vite plus the Solid compiler is the **downstream frontend toolchain** in this arrangement. Composer still owns Clef semantic lowering. The native host has its own backend, and the WebView is the execution environment. These responsibilities remain useful distinctions even if one future build command invokes all of them.

## Which tool does what?

| Tool | Role in this design |
|---|---|
| Xantham / TypeScript Compiler API | Declaration, symbol, ownership and resolution evidence for bindings; combined with implementation analysis under the [fused frontend design](02_jsir_tooling.md). |
| JSIR / JSHIR | MLIR infrastructure for JavaScript analysis and proposed target realization. JSHIR is already an MLIR dialect. Its current native representation needs extension to retain JSX. |
| Babel | Independent open-source JavaScript tooling: parser, AST definitions, traversal/transforms and generator. JSIR uses an embedded Babel payload through QuickJS; that is not a full Babel source checkout or an installed Solid compiler. |
| Solid compiler | Framework-specific JSX transformation, exposed through `babel-preset-solid` and invoked here by `vite-plugin-solid`. It produces DOM/reactive JavaScript and references the separate Solid runtime. |
| Vite | Coordinates frontend transforms and bundles the application and its assets under the selected configuration. |
| Bun | Candidate structural analysis and validation contributions, plus runtime/build capabilities where selected. Its JSX transform is not automatically a substitute for Solid compilation or WebView execution tests. |
| Bazel | Build orchestrator used by JSIR. It can build native tools and could coordinate more stages if adopted. It neither supplies JSX semantics nor becomes a required WREN application build system because JSIR uses it. |
| Fable | Existing F# compiler and bounded executable reference. A second producer can share the downstream Solid/Vite stages. |

Babel appears at two distinct points: **printing JSX without consuming it**, then **running Solid's transformation that consumes it**. Babel already has JSX nodes in [`@babel/types`](https://github.com/babel/babel/blob/main/packages/babel-types/src/definitions/jsx.ts). Its [parser](https://babeljs.io/docs/babel-parser#output) accepts JSX when enabled, and its [generator](https://babeljs.io/docs/babel-generator) can print it. The generator expects Babel AST, which is ESTree-derived with documented differences. A React JSX transform or React preset would select different framework behavior; it is not the Solid compilation step. See the [Solid Vite plugin](https://github.com/solidjs/solid-vite-plugin).

## The upstream JSIR gap

The review inspected upstream [`d5322bda6e1311357ead5e20376e28461c8cbc2a`](https://github.com/google/jsir/tree/d5322bda6e1311357ead5e20376e28461c8cbc2a), independently of the older local fork. Its native [AST classes](https://github.com/google/jsir/blob/d5322bda6e1311357ead5e20376e28461c8cbc2a/maldoca/js/ast/ast.generated.h) and [generated IR definitions](https://github.com/google/jsir/blob/d5322bda6e1311357ead5e20376e28461c8cbc2a/maldoca/js/ir/jsir_ops.generated.td) do not represent JSX. Enabling Babel's parser plugin alone therefore does not provide JSX round trips through JSHIR. There is no missing Babel JSX repository to install.

The proposed extension needs a reproducible source of definitions. JSIR's [`ast_def.proto`](https://github.com/google/jsir/blob/d5322bda6e1311357ead5e20376e28461c8cbc2a/maldoca/astgen/ast_def.proto) describes the input schema; [`ast_gen_main.cc`](https://github.com/google/jsir/blob/d5322bda6e1311357ead5e20376e28461c8cbc2a/maldoca/astgen/ast_gen_main.cc) generates native AST classes, JSON conversion, visitors, TableGen operations and AST/IR conversions. Editing generated `.td` files alone would miss that source and the bridge.

At this revision the public tree contains test schemas but does **not** contain the JavaScript `maldoca/js/ast/ast_def.textproto` named in the generator's example. Obtain or reconstruct that input, or establish an explicitly maintained extension mechanism, before claiming reproducible regeneration. Babel's TypeScript node definitions are useful mapping references; this generator does not consume them directly.

There is also a driver seam. The CLI names `source2ast,ast2jsir` and `jsir2ast,ast2source` identify conversions, but [`jsir_gen_lib.cc`](https://github.com/google/jsir/blob/d5322bda6e1311357ead5e20376e28461c8cbc2a/maldoca/js/ir/jsir_gen_lib.cc) initializes the input as JavaScript source. A reverse-only invocation with an MLIR file is not an established emission command. Composer integration needs the [representation conversion APIs](https://github.com/google/jsir/blob/d5322bda6e1311357ead5e20376e28461c8cbc2a/maldoca/js/driver/conversion.h) or a driver that accepts constructed/parsed JSHIR. Characterize supported operations and verification for that route.

## Lowering and reactive semantics

CCS owns semantic facts; Baker composes recipes through fan-out and generic fold-in; Alex witnesses `func`, `scf`, `arith`, `memref` and `index`. JavaScript and JSX operations belong below that portable witness boundary. A Clef-to-JSHIR conversion deliberately realizes the supported portable computation with retained PSG/codata. UI component and area identities, reactive reads, effects, input/commit versions and lifetimes must remain structurally available until realization; recovering them from printed names or already flattened calls would lose the contract. Functional and CE authoring share this obligation; JSX is a target handoff, not the portable UI representation.

The JSX extension must cover the admitted vocabulary explicitly: elements and fragments, intrinsic and component names (including member names), ordered attributes and spreads, text/whitespace, children and expression containers. Preserve source/obligation identities and lexical captures through native AST, IR and Babel export. Diagnose unsupported constructs.

MLIR regions can retain embedded expression structure, but JSX `{...}` does **not** create a new JavaScript lexical scope. A representation must admit outer captures. Neither a region nor its verifier alone establishes evaluation timing or Solid's tracking semantics.

For example, these forms can have different behavior under Solid compilation:

```jsx
// Reactive read remains available to the Solid transform.
const view = <span>{total()}</span>;

// Reading before view construction can capture a nonreactive snapshot.
const snapshot = total();
const viewWithSnapshot = <span>{snapshot}</span>;
```

Early hoisting, eager prop evaluation, altered spread order or duplicated callbacks can change the view's behavior while still producing valid JavaScript. Lowering must preserve the distinction between a value, a deferred accessor and an event handler. Keep transformations conservative until their effect and tracking contracts are established. JSX syntax can be a general JSIR extension; Solid-specific realization remains a separate contract.

Preserve the cold activation boundary as well: lifting mounted setup or a live control expression into module initialization can start work even when the corresponding view is never admitted. Conversely, a cold description may contain static data that the compiler can safely share. Conformance must distinguish description allocation, owner-local state, demanded evaluation and external effects; laziness alone does not prove zero allocation or safe sharing.

Extending JSHIR and its generated bridge is the concrete candidate developed here. A structured Babel-AST exporter after the same portable witness boundary is another possible engineering choice. It would still owe Clef semantics, UI-intent retention and the same acceptance evidence; choosing an AST exporter does not inherently require bypassing Alex.

## Proofs and the final artifact

This fits [Carrying Proofs into JavaScript](../../../clef-lang-site/hugo/content/blog/carrying-proofs-into-javascript.md): properties concern realized behavior, even when the final payload no longer carries compiler metadata. JSX adds a transformation edge to that argument. It does not itself carry or discharge a proof.

For a Solid profile, preserve or re-check affected obligations through JSX generation, Solid compilation, bundling and embedding. Bind evidence to the final HTML/JS/CSS assets, shipped dependency closure, source/projection identities, tool payloads and configuration, and the selected WebView host. Source maps and hashes support provenance; they are not semantic proofs. Solid's runtime remains a shipped implementation dependency even inside one native executable.

Use the existing [oracle procedure](07_dependency_identity_and_validation.md#ffable-as-an-executable-oracle) at suitable boundaries:

1. Compare admitted JSX structure before Solid compilation, including captures, accessor placement, spreads and events. Upstream JSIR cannot perform this comparison until its JSX representation exists.
2. Compare supported ordinary JavaScript after Solid compilation and execute related cases. JSIR normalization is an inspection aid; it is not an equivalence theorem.
3. Exercise the final embedded page in the actual WebView: updates, dispatch multiplicity, disposal, malformed messages and shutdown. WrenHello's [native UI gate](../../../WrenHello/tests/native-ui/README.md) is existing bounded evidence for its current path. Rebuild frontend and weld inputs before using that gate to assess a frontend change.

Two producers using the same Solid compiler or vendor runtime share possible failure modes. Their agreement checks the exercised translation and integration; independent expected behavior and negative mutations are still needed. Bun/Node execution can check portable logic but cannot establish DOM or native-WebView behavior alone.

## Bundling pages, panels and windows

Source pages, HTML documents, asset bundles and native windows are different units. Several authored pages can become one embedded HTML document with a persistent shell and section changes. Keeping that document alive permits smooth transitions and retained application state. Navigation between actual documents requires an explicit state and lifecycle protocol; bundling alone does not preserve the previous document's DOM or JavaScript heap.

WrenHello currently inlines dynamic imports and disables CSS splitting to produce one HTML file. It supplies an asset-packaging precedent, not an implemented router or multiwindow shell. A single bundle can mount only the active panel, but embedded bytes do not eliminate module initialization, DOM construction or memory costs.

[Atelier's design](../../../Atelier/docs/00_architecture.md) motivates a later choice: one document with docked editor/debugger panels, or several native WebViews loading distinct entry documents. [Vite supports multiple HTML entries](https://vite.dev/guide/build#multi-page-app); adopting that model would require an embedded asset collection, entry/chunk manifest and native resource resolution instead of the current single `EmbeddedAssets.IndexHtml` literal. Shared asset bytes do not mean shared JavaScript state.

Docking is a UI/runtime capability, separate from JSX compilation. Floating native WebView inspectors need explicit window creation, host messaging and lifetime ownership. Treat the observed workload, its subscription and its view as separate lifetimes: closing an inspector normally unsubscribes without stopping the workload; reopening obtains a snapshot and subsequent updates. Such a WREN capability could serve Atelier and standalone Clef diagnostic apps. A separate WebView does not by itself promise a separate renderer process on every platform.

## Implementation gates

These gates apply to the candidate JSX/Solid profile. Define and exercise the portable reactive-area contract with a native realization alongside this experiment; adopting the existing frontend toolchain must not settle the UI update model by default.

1. Pin tools and preserve WrenHello's Fable-produced JSX and native UI cases as reference fixtures.
2. Establish reproducible JSIR AST/IR generation and the JSHIR-input emission driver; add the minimal JSX vocabulary and supported round trips.
3. Connect one Clef view through portable witnessing to that vocabulary, with a signal read and an event callback. Preserve captures and source identities; reject mutations that hoist reads, activate an unused description, reorder effectful attributes or duplicate dispatch. Test independent mounts, unobserved stale branches and demand withdrawal/reacquisition. Then exercise an owned reactive area with versioned updates, resize and disposal against the shared semantic cases.
4. Run Solid compilation, bundle, weld and native acceptance on the generated result. Record the final artifact and dependencies with the covered obligations.
5. Compare the adapter's area behavior and costs with native evidence and a bounded direct-DOM experiment before choosing the browser substrate. Expand to optional Elmish adaptation, page/panel disposal and then optional multi-entry or multiwindow hosting, each with its own lifecycle and asset contract.

This advances the JavaScript target without making JSX mandatory for non-UI code or claiming that the existing downstream tools already implement Clef lowering.
