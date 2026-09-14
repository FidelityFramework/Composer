# Dependency replacement through deferred inference

**Design review: September 2026**

The intended Clef JavaScript path can replace the required behavior of a subsidiary JavaScript library with an owned Clef implementation. The resulting JavaScript can differ from the vendor's output while remaining valid under the selected host and application contracts. This is a constructive path to Cloudflare artifacts without third-party JavaScript implementation dependencies.

Generating a binding and recovering an implementation are distinct operations. A binding can call the supplied host or a retained library. An accepted replacement supplies the behavior itself. Dependency removal follows from replacing the required executable behavior and its reachable dependencies, not from changing type names or bundling imports into one file.

## Three implementation relationships

| Relationship | What executes |
|---|---|
| Direct host binding | The declared host facility, reached through a generated boundary. Declaration packages are build inputs. |
| Retained foreign library | The selected vendor implementation, with explicit runtime linkage, assumptions and dependency ownership. |
| Owned Clef implementation | Clef source compiled with the application; any remaining foreign calls still have declared boundaries. |

A program can combine these relationships per operation. A dependency-free artifact needs every required executable operation supplied by owned code or the declared host. A partially replaced library is still a dependency wherever its implementation remains reachable.

Cloudflare's execution and service facilities remain part of the target contract. Replacing an SDK's protocol logic does not replace the host's storage, scheduling or network implementation. Build tools and deployment clients have their own dependency closures; removing runtime package dependencies does not imply a toolchain with no dependencies.

## Analysis and editing loop

This is the Transcribe/Transpose work discussed in [Atelier](../../../Atelier/docs/10_transcribe.md). The responsibilities below distinguish binding ingestion from implementation recovery without depending on a final product-name allocation between them.

```text
Pinned declarations + executable entry points + dependency resolution
    -> one frontend combines Xantham and JavaScript structural evidence
    -> candidate Clef and partial facts enter ordinary CCS/PSG elaboration
    <-> application/target context and further analysis refine constraints
    <-> Atelier/LSP presents findings; developer supplies residual intent if needed
    -> build or REPL commitment of the required computation
    -> Baker, Alex and JavaScript backend realize that computation
```

The lift supplies the implementation's structured operations and relationships. The frontend can translate available structure and contribute constraints before every type, range, effect or representation has an answer. It does not restore information the JavaScript never represented, and candidate type-checking does not alone prove behavioral correspondence. Those obligations accompany ordinary elaboration under the same deferred-inference discipline as authored Clef.

JSHIR supplies the worked analysis route; [parser and API composition](02_jsir_tooling.md#ingestion-substrate-choice) remains open. Each contributing substrate or adapter must preserve the required structure, source identity and pending relationships. One frontend coordinates those contributions into ordinary CCS/PSG elaboration. Bun, Dafny and JSIR can each inform recovery, refinement, realization and validation according to their [documented strengths](02_jsir_tooling.md#contributions-to-a-fused-pipeline); the synthesis retains the forward witness boundary.

[The worked frontend example](09_contract_directed_dependency_recovery.md) combines SDK demand and reachable JSHIR structure in one translation path. Its Option/closure example shows partial elaboration, correspondence obligations and the SDK call redirected to an owned supporting library. Offline reachability bounds the SDK behavior requiring translation; application reachability later selects from the resulting Clef graph. Witnessing retains its existing forward role.

### Infer as much as the evidence supports

A library may encode a finite choice as a property bag, or implement a closure protocol through callbacks and shared state. Analysis can recover candidate unions, records, capture relationships and effect summaries from declarations and uses. It may also recover only a supported projection while leaving the rest of a value opaque.

Retain evidence about the producer, consumer, identity, guards and dependencies even when the value's complete shape is unknown. An opaque payload can still belong to a particular logical request or continuation. Widening must not discard those relationships.

[Deferred inference](../../../clef-lang-site/hugo/content/blog/deferred-inference.md) keeps independent decisions open. A value's semantic role can be known before its representation; a callback's shape can be known before its retention policy. Do not ask the developer to decide what further analysis can determine. Do not use the absence of a decision as permission to erase the requirement.

### Make the remainder actionable

A design-time finding should locate the missing premise and explain the consequence of each supported resolution. Examples include whether absence means keep or clear, whether a callback is retained, and whether an object must preserve identity or only an agreed value projection.

Developers supply intent, domain facts or a documented foreign contract. Those inputs re-enter analysis with provenance. An override is not proof of the vendor implementation. Where an inbound runtime value must satisfy a predicate, the compiler can generate the corresponding total check and typed failure exit. That is different from asking a developer to assert that all future inputs satisfy it.

Pending obligations remain resident during elaboration of a consistent partial program, including imported library work. Build or REPL evaluation requires the obligations for the computation being committed, not a fully resolved description of every imported value or unused operation. A required property must then be established, checked at a runtime boundary where the contract permits it, or supported by an explicit permitted external assumption. Otherwise the compiler reports the located requirement. Known contradictions and established unsupported shapes can be reported earlier; an open choice is not such a failure.

## Incremental library lifecycle

Conversion establishes a maintained Clef library. Subsequent applications, refinements and builds operate on that owned source and its accumulated semantic relationships. They do not restart JavaScript-to-Clef conversion. The same continuity applies to consistent partial work: new context refines the existing candidate and its pending obligations.

Retain the owned SDK/supporting source, declaration/body correspondence, established and unresolved constraint relationships, obligation premises and dependencies, and validation cases/evidence across sessions and builds. Persistent identity must survive regenerated parser indices and source edits. The exact storage and incremental interfaces remain implementation work; durable reuse is a requirement of that design.

| Change or new demand | Incremental response |
|---|---|
| Another application uses the converted SDK surface | Resolve against the existing Clef libraries, refine use-specific constraints and run ordinary application reachability. Reuse applicable conversion evidence. |
| A use supplies new type, range, effect or target context | Refine the affected instantiation and its dependent obligations. Reuse the retained generic/body structure; a new instantiation does not inherently require another source conversion. |
| The supported SDK surface expands | Reuse represented behavior and translate newly required foreign behavior with its initialization and dependency closure. Extend the existing library and correspondence. |
| Owned Clef code is functionally refactored | Maintain library ownership and update the affected correspondence and preservation argument. Continue from the curated source. |
| Upstream declarations, bodies or resolution conditions change | Reconcile the changed inputs with the owned library. Reanalyze affected relationships and their dependent closure, preserving unaffected source and valid evidence. Regeneration must not overwrite curated Clef with a fresh whole-library translation. |
| Compiler rules, target policy or boundary contracts change | Reconsider affected elaboration, realizations and evidence under their recorded premises. Preserve the owned source and any correspondence that remains valid. |

Incremental validity follows dependencies. Reuse evidence only where its premises and identities still hold; a change to shared state or a common boundary can affect several consumers. If recorded dependencies cannot justify a narrow recheck, widen that recheck explicitly. Revalidation of affected behavior does not imply discarding the converted library or repeating all ingestion.

The [F#/Fable oracle](07_dependency_identity_and_validation.md#ffable-as-an-executable-oracle) is part of this retained work. Keep its fixtures, input relations, traces and contract expectations; rerun affected cases as implementations or host conditions change. A new artifact needs applicable validation evidence, while the reference cases and accepted library work remain reusable.

## Preserve behavior while changing structure

The contract determines which observations a replacement must preserve. Depending on the supported library surface, those can include:

- Results, failures and the distinctions between missing, undefined, null and present values.
- Object identity, aliasing, mutation and any observed property/prototype behavior.
- Callable arity, captured snapshots or shared cells, callback retention and release.
- Promise completion/rejection and ordering relevant to the application.
- Wire encoding, numeric conversion and recovery or protocol state.

A replacement may use different internal data structures, functions or control flow where those observations are preserved. It may expose a more idiomatic Clef API with explicit adapters at retained foreign boundaries. Matching the vendor's JavaScript text or reproducing its internal object layout is not required.

Replacing a mutable object with a record snapshot is valid only where the contract permits the snapshot. Replacing an SDK's two-argument callback with a curried function requires the corresponding call adaptation. These are semantic obligations, not formatting differences.

## Support grows through bounded replacements

Choose a useful supported fragment, establish its dependency closure and identify the behavior to preserve. Combine declaration/body correspondence, proofs for supported operations, differential execution and host acceptance. Unknown effects stay in the model; an analysis that found no known effect has not proved purity.

JSHIR comparisons help characterize transformations, but any normalization needs a justified relation. Matching normalized IR is not a general equivalence theorem. Different IR can be correct, and identical-looking calls can reach different runtime implementations if linkage was lost.

Once accepted, the replacement follows the incremental lifecycle above. Record its origin, license obligations, accepted contract and validation evidence. An upstream change triggers review of the relevant assumptions and behavior; pinning makes the comparison reproducible rather than making the old contract timeless. Grow coverage by accepting further fragments with their obligations, not by claiming whole-library absorption from one successful example.

## The deployed artifact

For a Worker with no third-party JavaScript implementation dependencies, the output contains the compiled application, owned library behavior and generated boundary/entry code. Its remaining runtime imports or bindings are those explicitly supplied by the selected host. The final dependency audit must include bundled code as well as unresolved imports.

The host loads JavaScript that meets its export, calling and lifecycle contract. It does not require the output to match a TypeScript compiler's choices. The [acceptance chapter](07_dependency_identity_and_validation.md) defines the checks connecting this freedom to a supported claim of correctness.
