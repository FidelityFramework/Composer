# JavaScript tooling, analysis and lowering

**Design review: September 2026**

Composer's JavaScript pathway combines declaration analysis, foreign implementation analysis, functional refinement, proof obligations and target realization. Bun, Dafny, JSIR and the working F#/Fable pathway offer overlapping contributions to that work. JSHIR is a candidate analysis representation and part of the designed JavaScript backend; frontend interfaces and useful combinations remain open. CCS retains authority for Clef semantics throughout.

## Contributions to a fused pipeline

The relative weighting below describes the strongest contributions established by the reviewed material and where to concentrate implementation study first. It does not allocate exclusive stages to projects. A project can inform several stages, and several projects can contribute to the same analysis or transformation.

| Resource | Greatest current weight | Contribution across the pipeline |
|---|---|---|
| Bun | High weight for concrete JavaScript structure, binding identity, module/dependency relationships and transformation behavior. | Its representations can feed ingestion; its linking and rewrite decisions also supply cases for reachability, functional refinement, effect preservation, diagnostics and final dependency-closure checks. The internal adapter is a candidate to build and validate. |
| Dafny | High weight for semantic translation, specifications, representation choices and preservation testing. | Contracts and executable/proof distinctions can inform recovered Clef abstractions and their obligations; the JavaScript compiler supplies functional and numeric realization cases; external-contract and compiler checks inform acceptance. Design and code are precedents. Direct reuse or use as a scoped reference/checking tool requires a separate validated integration. |
| JSIR/JSHIR and MLIR analysis | High weight for structured operation/region analysis and the designed forward JavaScript realization path. | The source lift can contribute ingestion evidence; structured comparisons can inspect candidate refinements and expose preservation obligations; target conversion and printing realize the witnessed computation. Coverage and verification must be established for each used route. |
| F#/Fable pathway | High immediate weight as an executable oracle, grounded in the working bindings and bounded behavior tests. | Existing SDK/host interactions and BAREWire cases characterize contracts; corresponding functional F# implementations can supply reference results and traces for recovery, refinement and lowering. Source and emitted JavaScript can help localize discrepancies. Follow the [oracle procedure](07_dependency_identity_and_validation.md#ffable-as-an-executable-oracle) and record the shared semantic domain. |

Xantham's declaration/ownership analysis and SDK uses provide the contextual contracts that these contributions meet. The TypeScript Compiler API remains a possible source of additional syntax, symbol and resolution evidence. None of the resource weightings changes the authority of the Clef specifications or Composer's architecture.

**One frontend means coordinated elaboration into the same authoritative PSG.** It can accept several evidence producers. For example, Bun may supply binding and module relationships while JSHIR exposes operations for further analysis, and Dafny's contract/representation examples inform the recovered functional structure and its correspondence obligations. These contributions can cooperate before payload types, effects or target representations are fully settled.

Fusion requires common source/declaration identities, related operation identities, explicit assumptions and pending constraints, and dependency tracking for invalidation. A finding from one analysis can refine another through those relationships. Agreement between tools is useful corroboration; shared assumptions or a common parser do not make two results independent proofs. Where analyses disagree, retain the conflicting evidence and its affected uses for ordinary diagnostics and resolution.

For each contribution, distinguish a borrowed design principle, adapted implementation, invoked analysis tool and observed integration result. Tool adoption follows demonstrated usefulness against the shared fixtures; a build need not execute every reference tool. Any optional external checker receives a scoped obligation/model with stated assumptions through the established proof machinery. That does not require completing inference for the entire imported program or create a parallel authority for Clef correctness.

The resulting pipeline preserves its architectural ownership: CCS elaborates semantic constraints; Baker composes recipes through fan-out and incorporates them through generic fold-in; Alex observes settled joint constraints with its zipper, patterns and elements; the backend realizes portable witnessed operations. The [worked integration example](09_contract_directed_dependency_recovery.md#combined-contributions-to-this-fixture) makes the overlapping contributions concrete.

## Pin the reviewed JSIR tool

The September [site review](../../../clef-lang-site/hugo/content/docs/design/javascript-targeting/jsir-javascript-as-mlir-backend.md) records upstream revision `1488d9bd408ec9163ac7051252dfe80e40a4e26a`. This document carries that review's tooling facts; it does not claim a newly built or integrated Composer toolchain.

At that revision, `jsir_gen` exposes these conversion sequences:

| Direction | Pass sequence | Role |
|---|---|---|
| JavaScript source to high-level IR | `source2ast,ast2jsir` | Parse through Babel AST and produce JSHIR for analysis. |
| High-level IR to JavaScript source | `jsir2ast,ast2source` | Convert JSHIR through Babel AST and print JavaScript. |

Commands, accepted inputs and output formats must come from the pinned tool revision. Integration must record the actual invocation and tool payload. A pass-name table is not a build receipt.

Babel is the AST substrate. JSHIR supplies region-based high-level structure; the repository also defines JSIR operations. The existence of both dialects does not establish a separately supported low-level route to source generation. Supported operation and conversion coverage must be characterized for the selected revision. JSX handling, if required by a source frontend, is upstream of this JavaScript representation.

## The worked frontend analysis route

```text
Pinned JavaScript package and selected executable entry points
    -> Babel/JSHIR lift
    + Xantham declaration analysis and available SDK/application constraints
    -> one foreign-language frontend: structure, candidate Clef and partial evidence
    -> ordinary CCS/PSG elaboration, retaining unresolved requirements
    <-> further source, application and target context
```

The analysis must relate bodies to declarations through authenticated module/export resolution. A shared name is insufficient. Dynamic dispatch, callbacks, unavailable imports and runtime-generated behavior retain unknowns until evidence closes them.

JSHIR supplies structured operations and regions, with represented values and references that analysis can use to establish data flow, bindings, captures and call relationships. Those facts are useful before every type or representation is resolved. JSHIR's value types alone do not establish Clef dimensions, lifetime ownership or a lost TypeScript generic contract; the frontend combines the available evidence and carries remaining constraints into the [deferred inference workflow](05_supply_chain_and_transcribe.md).

Candidate source participates in ordinary CCS checking while inference remains partial. Structural recovery is not conditional on first completing type inference, numeric selection or proof discharge. The frontend preserves correspondence obligations between the foreign operations and the candidate; a syntax lift does not itself discharge them.

The [deferred frontend example](09_contract_directed_dependency_recovery.md) specifies demands, body observations, partial elaboration states and eventual correspondence checks. Its analysis-region labels are not JSHIR operation names, and the documented lift does not itself provide a Clef emitter.

## Ingestion substrate choice

JSHIR is a concrete candidate for the frontend's analysis representation. The ingestion design can also use other parsers and their exposed relationships, individually or together, provided the adapters preserve the information needed for ordinary Clef elaboration. Choosing those interfaces does not choose a different forward backend or restrict which projects can inform later stages.

The interfaces below were reviewed on September 13, 2026; none is an adopted new Composer dependency:

| Substrate | Exposed information and assessment |
|---|---|
| JSIR lift through Babel | The pinned route described above produces structured JSHIR. It provides an MLIR analysis surface; its operation coverage and semantic relationships still need validation for the fixture. |
| TypeScript Compiler API | The documented `Program`/`SourceFile` interfaces accept JavaScript using `allowJs`, expose AST traversal, and support symbol/type queries and module resolution. This makes the API behind `tsc` a concrete candidate for structural ingestion. The reviewed guide covers TypeScript 6.0 and earlier and warns that 7.1 has a different API; pin and validate the selected interface. [Official Compiler API guide](https://github.com/microsoft/TypeScript/wiki/Using-the-Compiler-API). |
| Bun.Transpiler | The documented interface returns JavaScript text from transforms and import/export metadata from scans. The transform does not resolve modules; scans omit type-only imports/exports. This can contribute source inventory or preprocessing, but the reviewed public interface does not supply a general AST or semantic-graph export. That additional access would need to be established before using it as the sole structural substrate. [Official transpiler documentation](https://bun.com/docs/runtime/transpiler). |
| Bun internal parser and AST | The local source exposes structured syntax, symbols, scopes, import/export relations and statement-part dependencies. This is a structural ingestion candidate beyond the public scan API. The integration must account for transformations performed during parsing and visiting, and export owned evidence from Bun's arena-backed structures. See the pinned source review below. |

A syntax tree plus declaration, binding and use relations can support graph construction. An import list alone cannot establish closure captures, execution order or the full behavior required by a call. An MLIR graph likewise needs the relevant language analysis; its existence is not a complete call, effect or proof graph.

The adapter contract is to preserve source identities and locations, operation structure, available scope/binding/module relationships, and explicit unresolved relationships. Further analysis contributes value flow, captures, call targets and effects as evidence permits. Neither a completed foreign type check nor a fully resolved call graph is an upfront condition for contributing partial information to CCS. Foreign `any`, unknown types and unresolved targets retain their provenance rather than silently becoming Clef commitments.

Compare individual interfaces and useful combinations on the same [worked frontend fixture](09_contract_directed_dependency_recovery.md): recover its closure and branch structure, relate declaration owners and calls, retain pending payload/effect constraints, and expose the initialization and dynamic-call mutations. Keep the contributions that preserve those facts with the least unnecessary reconstruction and maintenance burden. Parser speed alone does not establish that result. The composition remains open.

### Bun internal structure and the ingestion seam

The local Bun review is pinned to `86771d09fd486a7256790d6f36602b683f7a19de`. Its [repository architecture guidance](../../../bun/AGENTS.md) separates the Rust parser, AST, resolver, printer and bundler crates. The [linker design notes](../../../bun/src/bundler/linker_context/README.md) describe import/export analysis followed by tree shaking, chunking and generation. These establish where to inspect the implementation; they do not establish a Clef integration.

The source supplies concrete evidence for graph construction before Clef types are settled:

- [`Ast`](../../../bun/src/ast/ast_result.rs) carries statement parts, symbols, module scope, named imports/exports and dynamic-import alias information. Its dynamic-import record distinguishes tracked property uses from an escaped namespace that requires retaining all exports.
- [`Expr`](../../../bun/src/ast/expr.rs) retains locations and distinct expression cases. Binding references and scopes supply relationships beyond a syntax tree. For example, the property-inlining predicate explicitly preserves optional-chain boundaries instead of treating superficially similar expressions as interchangeable.
- [`Scope`](../../../bun/src/ast/scope.rs) retains parent/child relationships, member bindings and direct-eval constraints. [`Symbol`](../../../bun/src/ast/symbol.rs) retains merge links and namespace identity. Export these relationships; names or source locations alone do not identify a binding across transformations.
- [`Part`](../../../bun/src/ast/nodes.rs) relates statements to declarations, symbol uses, imports and dependencies. Its comments distinguish estimated use counts from exact facts, and defer imported-property dependencies until linking determines whether an enum property is inlined. These are useful analysis distinctions, not Clef proof discharge.

Bun's [`Call` representation](../../../bun/src/ast/e.rs) makes another relevant distinction explicit: a `PURE` annotation permits removing an unused call, but does not establish mathematical purity, and effectful arguments must remain. The frontend must retain that annotation's limited meaning and provenance. It cannot turn it into a Clef law authorizing repeated, reordered or parallel calls.

The public [`JSTranspiler.scan`](../../../bun/src/runtime/api/JSTranspiler.rs) already obtains a full parse result, then exposes only import and export information. A structural adapter would need access before that projection. The lower-level [`Parser::init` and `Parser::parse`](../../../bun/src/js_parser/parse/parse_entry.rs) are Rust entry points, but their result is not an untouched source tree: the first pass parses without binding symbols; the subsequent visit binds symbols and also transforms syntax. Disabling tree shaking groups statements into one part; it does not establish a transformation-free parse mode. Preserve TypeScript declaration analysis separately: Bun's [`STypeScript` statement](../../../bun/src/ast/s.rs) is an empty stand-in for erased type-only syntax.

The proposed seam is an adapter over a pinned parser phase, with source/declaration identity retained and each active transformation accounted for. Where a transformation loses a distinction required by correspondence checking, export the earlier structure and its relationship to the bound result, or separate that transformation from binding. Do not treat a post-bundle output as the complete original dependency. Required module initialization and unresolved dynamic relationships must survive offline selection.

Export owned operation and relationship records while their source and AST storage remain alive, mapping Bun references to persistent source identities. Bun's [parser crate](../../../bun/src/js_parser/lib.rs) documents arena-backed, lifetime-erased storage and higher-tier macro hooks; a separately linkable parser adapter has not been built or established by this review. Parser settings, source payload and transformation provenance belong with the exported evidence. Macro execution and environment substitution must be explicit inputs to the ingestion contract rather than accidental consequences of a runtime transpiler configuration.

That evidence joins Xantham's declaration/use analysis in ordinary CCS/PSG elaboration, with open payload, effect and representation constraints intact. An adapter may materialize JSHIR when MLIR analysis is useful; MLIR is not a prerequisite for obtaining the initial graph. The first comparison should exercise chapter 09's closure, nullish branch, callback-order and initialization cases, including initially unresolved payloads. This evaluates the proposed seam without adopting Bun's bundler decisions as Clef semantic authority or changing Baker, Alex or the forward JSIR backend.

## Dafny's semantic and implementation contribution

The local Dafny review is pinned to `98ac8c0a443e1ca4348d0d01bda618894dee0d1e`. Its [language introduction](../../../dafny/docs/DafnyRef/Introduction.md) describes source specifications, verification through Boogie and an SMT solver, and compilation to supported targets including JavaScript. The [JavaScript integration guide](../../../dafny/docs/DafnyRef/integration-js/IntegrationJS.md) describes its target output and runtime requirement. The contribution to this design is substantial, with concrete lessons spanning ingestion, refinement, realization and acceptance.

| Reviewed mechanism | Application to Clef |
|---|---|
| Preconditions, postconditions, frame specifications and ghost state | State which values, effects and state relationships a recovered operation must preserve. These can guide candidate functional structures and correspondence obligations during partial elaboration, with unresolved premises retained. |
| Lambda/application emission and datatype construction/destruction in the [JavaScript generator](../../../dafny/Source/DafnyCore/Backends/JavaScript/JavaScriptCodeGenerator.cs) | Examine concrete realizations of function values, argument binding, constructor distinctions and wrapper elimination. Use their semantic cases when designing both recovery patterns and forward preservation checks for Clef's Option and flat closures. |
| Selective [runtime type descriptors](../../../dafny/docs/Compilation/ReferenceTypes.md) | Determine which operations actually need executable representation information. Dafny uses descriptors for purposes such as type-parameter defaults; this is a case study in semantic need. Clef retains dimensions and other facts in the PSG/codata as long as useful and applies its own realization rules. |
| Carrier-dependent arithmetic, exact construction of large literals, and source-specific division/modulo/shift realization | Identify numeric correspondence requirements from the foreign input onward, then realize the selected Clef operations faithfully. These cases strengthen [numeric selection and precision diagnostics](08_numeric_selection_and_precision.md#lessons-from-the-combined-tooling-review). |
| Paired static assertions and executable expectations, plus external-contract checks | Derive tests from the same properties used in the semantic argument. Keep successful executions, discharged obligations, external assumptions and unresolved results distinct in the [acceptance record](07_dependency_identity_and_validation.md#combine-analysis-and-validation-evidence). |

Dafny's [statement guide](../../../dafny/docs/DafnyRef/Statements.md) explains using a proved assertion and an executable expectation of the same predicate to detect discrepancies introduced by compilation. Its [contract-testing description](../../../dafny/docs/DafnyRef/UserGuide.md) also describes wrappers that check external preconditions and postconditions. These are useful patterns for exposing a mismatch between a declared SDK contract, its dependency implementation, the recovered Clef and generated JavaScript. Runtime checks provide evidence for their executions; they do not discharge universal obligations by themselves.

The same lessons can improve upstream recovery. A representation case in the Dafny backend may expose an input-preservation premise missing from a proposed Clef translation. A state/frame specification may reveal that a property-bag implementation admits a functional replacement, or that observable sharing must remain. Those findings re-enter ordinary elaboration with their provenance; Dafny is not confined to an emission reference.

Its runtime representations and verification workflow implement Dafny's language. Adapt the relevant mechanisms to Clef's contracts and owned-library model. No complete Dafny program, eager foreign type completion or additional Dafny compilation stage is required by this design. Direct use of Dafny for a reference model or scoped verification remains a candidate integration; the reviewed source does not establish automatic verification of vendor JavaScript or Composer's emitter.

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
