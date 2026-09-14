# From foreign declarations to Clef-native bindings

**Design review: September 2026**

A Clef binding generation path consuming Xantham's analysis can produce a Clef-native API rather than transliterating every TypeScript construct. That freedom requires preserving the supported foreign contract: what operation is called, what it accepts and returns, which effects and lifetimes matter, and how failure is represented.

An `obj` widening or nullable declaration is evidence to investigate, not a command to add `obj` or `null` to Clef. The analysis may recover a supported structure, retain a foreign value opaquely, or leave a requirement pending while more context arrives.

## Separate declarations, executable libraries and host services

| Input | Evidence and resulting work |
|---|---|
| Host API declarations | Describe entry points and shapes. Generate boundaries to the supplied host facilities under explicit host assumptions. |
| Executable JavaScript/TypeScript SDK | Supplies implementation behavior as well as declarations. Analyze the body and dependency closure; retain it as a dependency or replace the supported behavior with accepted Clef. |
| OpenAPI contract | Describes the management/HTTP surface. Generate the client from that contract rather than assuming a language-specific SDK is the authority. |

A declaration-only package does not supply runtime code. Conversely, an SDK published by a platform vendor may contain substantial executable logic that is not part of the host. Ownership and package branding do not settle which category applies.

The current FSharp.CloudEdge delivery uses Xantham for TypeScript-derived F# SDK bindings and Hawaii for OpenAPI-derived F# clients. Farscape supplies the C/C++ binding-analysis and marshaling precedent. The Clef-target generation path is work to build from those analysis contracts; this document does not describe Xantham's shipped F# output as already being Clef.

## Recover a contract, not just a name

Xantham's structural analysis supplies declarations, parameters, aliases and references. JSHIR can expose supported executable control flow and operations. Application use contributes additional constraints. Their join requires resolved declaration ownership and runtime linkage; equal names or member lists are not proof of identity.

Useful evidence can include a callback's actual argument convention, fields consumed together, a discriminant's use, resource ownership, nullability behavior and effects reachable through dependencies. A dimension or domain law cannot be invented from a variable name. Dynamic or unavailable behavior stays explicit.

[The Gift of Deferred Inference](../../../clef-lang-site/hugo/content/blog/deferred-inference.md) supplies the discipline: retain established facts and their dependencies while other choices remain open. Do not prematurely commit a widened value to a concrete record or permanently discard the evidence around it. A consistent partial binding may remain under analysis in the editor; concrete compilation needs the premises required by its selected realization.

The foreign frontend contributes that partial structure to ordinary CCS/PSG elaboration. A call relationship, branch condition or capture can be known while its type arguments, effect requirements or representation remain open. The tables below describe supported semantic dispositions as they become justified; they are not an admission checklist requiring all choices to be completed before the graph can be built.

## Clef-native dispositions

The [JavaScript boundary specification](../../../clef-lang-spec/spec/javascript-boundary.md) governs the admitted surface:

| Foreign role | Clef disposition |
|---|---|
| Required value of an established shape | Declared Clef type, with the required inbound narrowing. |
| Optional or nullable position | Option by default; preserve contractually distinct states with a generated union. |
| Options bag | Generated nominal record with explicitly optional fields. |
| Position whose established contract admits an undetermined foreign value | `JsValue`, with explicit foreign contact and narrowing only for uses requiring it. |
| Unresolved inference or mapping choice | Retain inference variables, candidate constraints and provenance in elaboration; do not default to `JsValue`. |
| Known opaque host reference | `JsRef<'T>` where the declaration establishes that identity. |
| Concrete foreign union | Binding-owned introduction/elimination under the specification's erased-union rules; no universal NTU type. |
| Callback | Declared calling convention, parameter/result conversions and applicable lifetime/effect contract. |
| Host throw or awaited rejection | Generated interception and a typed Result error. |

This table describes semantic roles, not a string-replacement algorithm. A generator's `obj` may record an unsupported mapping of a richer declaration, rather than genuine source dynamism. Preserve that loss and consult the available declaration/body/use evidence. The [opaque-value chapter](06_obj_and_null_at_the_boundary.md) covers cases that remain foreign.

Removing CLR-specific top bounds does not remove meaningful generic constraints. Type parameters, their declaration scopes, applied arguments and contextual constraints remain distinct facts through specialization and target realization.

## Declarations join the graph

Atelier's [Transcribe design](../../../Atelier/docs/10_transcribe.md) makes the result a Clef binding declaration with binding-strategy annotations. These describe how a supported declaration participates in the foreign boundary. They are not an embedded JavaScript body or a license for unchecked lowering.

The Library of Alexandria supplies witnessing rules for supported declaration shapes. Ingestion reports rule coverage as the shape becomes known: an established shape without a rule is a located Composer/Alex support requirement; an unresolved shape has a pending coverage question. Partially elaborated candidates remain available to the editor and compiler analysis. A declaration cannot be claimed executable under a selected realization without the required coverage. Transcribe does not invent a witness, and rule availability does not settle unrelated type, effect or representation constraints.

CCS and Baker establish the typed conversion structure and its obligations. Fan-out composes Ingredients into recipes; generic fold-in incorporates them. Alex observes the settled structure through patterns and elements and emits the portable vocabulary. The JSIR backend realizes it in JavaScript and may read the retained graph facts it needs.

No witness decides that a value must be a particular record because the vendor named it `options`. No backend treats a widening annotation as evidence that an unchecked operation is safe.

## The Farscape lesson

A callback pointer, its context and its destroy callback describe a relationship across declarations. The joint constraint concerns callable shape, representation and lifetime ownership. The obligations identify missing information, such as which event releases the captured environment and whether invocation can outlive registration.

The JavaScript analogue is the method of analysis, not a requirement to copy a C pointer representation into a managed host. Callback retention, aliasing, mutation and release/cancellation behavior may be distributed across several SDK declarations and bodies. Recover the supported relationship, preserve unresolved premises and ask for the remaining contract in the design-time environment. A function name alone does not establish a lifecycle.

## Developer participation

Atelier displays compiler and substrate findings, proposed bindings and outstanding requirements. Developers can select the intended contract, provide documented overrides, choose what is wire-typed versus interop-typed, and review supported target realizations. The editor computes no independent semantic facts.

A supplied assumption remains identifiable as an assumption. A checked guard establishes only its predicate under its validity conditions. Regeneration must reconsider evidence affected by changed declarations, bodies or dependencies; it must not silently carry an old decision into a new contract.

See [dependency recovery](05_supply_chain_and_transcribe.md) for the interactive sequence and [identity and acceptance](07_dependency_identity_and_validation.md) for the evidence retained with the result.

The [worked frontend example](09_contract_directed_dependency_recovery.md) shows how TypeScript demand and JSHIR implementation evidence constrain a Clef-owned dependency through partial elaboration. It makes the structural facts, deferred decisions, declaration/body correspondence and owned SDK call edge concrete.
