# JSHIR/JSIR tooling and lowering

**Design review: September 2026**

JSIR provides a JavaScript analysis and source-emission substrate within MLIR. Composer uses that substrate in two different roles: inspecting foreign implementations during recovery, and realizing compiled Clef computations in the JavaScript backend. Neither role makes JSIR the authority for Clef types or proof obligations.

## Pin the reviewed tool

The September [site review](../../../clef-lang-site/hugo/content/docs/design/javascript-targeting/jsir-javascript-as-mlir-backend.md) records upstream revision `1488d9bd408ec9163ac7051252dfe80e40a4e26a`. This document carries that review's tooling facts; it does not claim a newly built or integrated Composer toolchain.

At that revision, `jsir_gen` exposes these conversion sequences:

| Direction | Pass sequence | Role |
|---|---|---|
| JavaScript source to high-level IR | `source2ast,ast2jsir` | Parse through Babel AST and produce JSHIR for analysis. |
| High-level IR to JavaScript source | `jsir2ast,ast2source` | Convert JSHIR through Babel AST and print JavaScript. |

Commands, accepted inputs and output formats must come from the pinned tool revision. Integration must record the actual invocation and tool payload. A pass-name table is not a build receipt.

Babel is the AST substrate. JSHIR supplies region-based high-level structure; the repository also defines JSIR operations. The existence of both dialects does not establish a separately supported low-level route to source generation. Supported operation and conversion coverage must be characterized for the selected revision. JSX handling, if required by a source frontend, is upstream of this JavaScript representation.

## The forward analysis route

```text
Pinned JavaScript package and selected executable entry points
    -> Babel/JSHIR lift
    -> language-specific semantic analysis
    + Xantham declaration analysis and application context
    -> candidate Clef bindings or implementations, with pending requirements
```

The analysis must relate bodies to declarations through authenticated module/export resolution. A shared name is insufficient. Dynamic dispatch, callbacks, unavailable imports and runtime-generated behavior retain unknowns until evidence closes them.

JSHIR's JavaScript value types do not recover Clef dimensions, lifetime ownership or a lost TypeScript generic contract. Recovery combines evidence from the available sources and exposes its unresolved parts through the [deferred inference workflow](05_supply_chain_and_transcribe.md). A recovered source candidate enters ordinary CCS checking; a syntax lift does not directly certify a Clef graph.

## The backend route

```text
Portable witnessed operations + retained PSG/codata
    -> JavaScript carrier and boundary realization
    -> supported JSHIR/JSIR operations
    -> Babel AST
    -> JavaScript module
```

Alex does not emit JSHIR. It witnesses settled graph structure through the five portable dialects. The JavaScript backend realizes those operations under [Backend Lowering Architecture §4.5](../../../clef-lang-spec/spec/backend-lowering-architecture.md#45-carrier-realization-on-pathways-without-linear-memory).

The backend can read field names, case identities and capture structure from the graph where the JavaScript model needs them. Form selection and semantic judgments remain above the witness boundary. No Option-specific reconstruction or vendor-name dispatch belongs in emission.

A host function realization must retain actual argument boundaries and capture semantics. A property access must correspond to the declared record access. A byte operation must retain its view origin, extent, encoding and conversion. A boundary operation must retain its absence and failure disposition. The target's richer syntax is not permission to omit these relationships.

[Numeric selection and precision](08_numeric_selection_and_precision.md) details the arithmetic correspondence: preserve selected representations, intermediate capacity, rounding points and admitted merge laws. A valid numeric JSHIR operation alone does not establish those properties, and emission does not repeat the selector or choose a cheaper precision policy.

## Verification scope

The reviewed upstream revision invokes MLIR verification during AST-to-JSHIR conversion, while its transformation runner disables pass-manager verification pending an IR-design fix. Do not describe that as a universally verified pipeline. Integration must establish which structural checks actually ran on the selected route.

Structural validity is necessary but does not prove semantic preservation. A valid DataView operation can still use the wrong offset or endian order. An ordinary JavaScript callback can still receive the wrong arguments. For every admitted operation family, the lowering needs a stated correspondence and preservation evidence or a re-check at the affected edge.

Round trips and normalized JSHIR comparisons are regression instruments. Normalization must respect binding, capture and observable distinctions. Neither matching IR nor parseable JavaScript proves a foreign library's effects or Cloudflare behavior.

## Integration evidence

A backend acceptance record should identify:

- The exact JSIR, MLIR, parser/printer and compiler tool payloads and invocation.
- Supported operations and explicit unsupported cases for both lift and emission.
- Structural verification actually performed, alongside semantic correspondence checks.
- Source locations and obligation provenance through transformations.
- The emitted module, dependency closure and selected host configuration.
- Executable success, failure and boundary cases under that configuration.

These are implementation requirements, not claims that Composer currently supplies those records. [Identity and acceptance](07_dependency_identity_and_validation.md) connects them to the binding and application contracts.
