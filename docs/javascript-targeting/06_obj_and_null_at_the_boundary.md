# Opaque values and absence at the JavaScript boundary

**Design review: September 2026**

Clef does not need `obj` or `null` to produce JavaScript that uses objects and foreign absence conventions. A source-language type, a retained semantic contract and a target representation are distinct. JavaScript can keep its host-managed dynamic representation at a declared boundary without creating a universal type in Clef.

The governing rules are [JavaScript Boundary Semantics](../../../clef-lang-spec/spec/javascript-boundary.md) and [Option Operations Representation](../../../clef-lang-spec/spec/option-operations-representation.md). This chapter describes their application to binding generation and deferred recovery. The JavaScript Substrate profile has no conforming implementation yet.

## Widening is a finding

An `obj` in generated F# bindings may mean a genuinely dynamic TypeScript position, an opaque runtime handle, an options bag, or information lost by that generator's mapping. The spelling alone does not identify the semantic role. Recover what declarations, bodies and uses establish, and retain the source of any remaining loss.

There is no requirement to recover every object's complete shape. Analysis may establish a useful projection, or settle on an opaque foreign crossing. It may also keep the decision pending until application context supplies more constraints. Do not manufacture a record or nominal handle identity to conceal an unsupported mapping.

An unresolved inference variable is also distinct from `JsValue`. The former is an open compiler decision with constraints still accumulating; the latter is a deliberate contract for a foreign value. The frontend must not turn missing inference into `JsValue`, `obj` or a guessed concrete type merely to finish generation. Structural facts and unresolved type relationships can enter ordinary CCS/PSG elaboration together.

## The foreign pair

| Type | Contract |
|---|---|
| `JsValue` | Foreign value with undetermined shape. No subtyping edges, implicit conversion, property access, invocation, arithmetic or comparison. Its elimination is generated narrowing. |
| `JsRef<'T>` | Opaque host reference whose declared type identity is recorded by `'T`. It can be held and passed to declared boundary functions; it grants no interior operations. |

Injection occurs only through declared boundary positions. Neither type is a universal top or bottom. `JsValue` is not an interior discriminated union exposing arbitrary JavaScript values to unchecked traversal.

A value can remain opaque throughout its lifetime: received, stored, forwarded and returned through a declared boundary. This does not require converting it into a Clef record. It also does not establish that the host value is serializable or survives isolate destruction. Those uses require their own contract.

The boundary grade tracks modeled foreign contact transitively. Opaque passage preserves that contact; it does not establish that an unknown operation is pure. Projects can require grade zero outside a designated interop layer under the specification's rules.

## Narrow only what a use requires

Deferred inference and runtime boundary narrowing occur at different points. Inference can leave the required predicate, type or realization pending while the program is elaborated. When build or REPL evaluation commits a use, the compiler must establish its required static facts and any permitted generated check. An actual incoming value must pass that check before the typed operation uses it. The binding can declare a supported projection rather than promise complete knowledge of the original object; other manipulation can remain within a declared foreign operation.

Runtime narrowing is total: it yields a converted value or a Result error identifying the failed path and expected shape. It does not admit a partially converted runtime record. This says nothing against a partially inferred program during elaboration. Callback parameters entering from JavaScript receive the same boundary treatment. The concrete narrowing-error representation remains a specification item; this folder does not invent one.

Property names and `typeof` tests are evidence a generated check can use. Getters, proxies, thrown values, mutation and buffer detachment can affect the check or invalidate a premise before use. An immutable binding to a mutable object does not freeze its contents. The graph must preserve the relevant identity, effects and validity interval rather than treat one successful observation as permanent.

## Absence is position-specific

For a declared inbound `Option<'T>`, the default boundary conversion maps absent, undefined and null to None, and narrows other values into Some. Some APIs assign different meanings to those states. At a declared-distinction site, the binding uses a generated union, such as keep, clear and set, rather than collapsing the distinction.

TypeScript optional and nullable syntax helps locate the question. It does not alone prove an API's update semantics. Declaration/body evidence and explicit curation supply the supported contract.

For each outbound optional position, generation selects the representation the host contract requires: omit the property or argument, emit null, or emit undefined. There is no universal JavaScript lowering of None at a foreign boundary. Host throws and awaited rejections are intercepted there and surfaced through Result.

### Interior Option realization

The JSIR pathway selects an erased or reified form per instantiation before emission:

- Erasure can realize Some as its payload and None as undefined only when undefined cannot also represent an admitted Some payload.
- Reification preserves the case and payload explicitly. Nested options must retain every distinction the erasure proof cannot discharge.

Interior None is never represented as null. The selected form is not renegotiated at runtime. At an outbound boundary, its absence is converted again according to that position's contract; an erased interior None does not dictate an undefined argument to every host API.

Option operations retain their specified evaluation and callback behavior in either form. Representation choice does not authorize dropping an eagerly evaluated callback factory merely because the input is None.

## Where the work belongs

Bindings carry declarations and binding-strategy annotations. Semantic analysis and Baker establish conversions, supported forms and obligations in the PSG. Alex witnesses the settled structure through patterns and elements into portable operations. The JSIR backend realizes those operations in the host's value model, reading useful retained codata.

An annotation can preserve a foreign convention and its provenance. It cannot waive the language's type rules or supply missing proof. The backend can emit a host object, function, boundary null or permitted undefined without making that construct an unrestricted Clef source value.

This is the same relationship-preserving discipline as the Farscape callback/context/destroy boundary: obligations expose what the graph must retain. JavaScript realization uses the host's representations and lifecycle facilities rather than importing a C memory model.

## Proofs can concern opaque values

[Carrying Proofs into JavaScript](../../../clef-lang-site/hugo/content/blog/carrying-proofs-into-javascript.md) makes the distinction concrete. A payload may remain opaque while generated code preserves its logical request identity, contribution multiplicity or eligibility to resume a continuation. A known record shape alone would not establish any of those properties.

A widening therefore must not erase the surrounding semantic evidence. Retain dependencies, provenance and protocol relationships. Where a property depends on foreign behavior, record the accepted contract or generated check. Preserving an opaque value does not establish the correctness of the computation that produced it.

## BAREWire today

BAREWire's built codecs are typed module functions. Its [encoding design](../../../BAREWire/docs/02%20Encoding%20and%20Decoding%20Engine.md) identifies runtime schema walking with boxed objects as an unbuilt approach, not the implementation to reproduce.

`writeOption` writes a zero presence byte for None and a one followed by the typed payload for Some. `readOption` distinguishes valid absence from decoding failure through its cursor result: an accepted None has a valid next offset; an invalid presence flag or failed payload read has a fault cursor. Whole-value decoding additionally checks complete consumption. A later API refactor enabled by stronger Clef functional support must preserve that distinction between absence and failure.

The final payload is untagged with respect to compiler type, schema, dimension and proof metadata. Presence bytes, union case indices and framing fields select structure inside an agreed contract; they are not runtime type descriptions. Bounds and encoding checks remain necessary. Hosted text decoding has its own replacement behavior for malformed UTF-8, so this is not a claim that every malformed byte sequence is rejected.

The current Fable path can use undefined for Option absence and tagged host objects for Result around the codecs. Those are host realizations, not BAREWire payload type tags or a prescription for Clef's JSIR ABI. BAREWire byte/view checks do not validate arbitrary SDK object shapes or decide an API's missing/null semantics.

See [Substrate Formalism](../../../BAREWire/docs/Substrate_Formalism.md), the [evidence inventory](../../../BAREWire/docs/12%20Intersection%20Subset.md), and [deployment contexts](03_four_wings.md).
