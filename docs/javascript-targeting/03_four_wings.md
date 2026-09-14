# Deployment contexts and BAREWire

**Design review: September 2026**

Cloudflare, browsers and WebViews are JavaScript deployment contexts. BAREWire is a shared contract across those contexts and native targets, encompassing memory description, IPC and network encoding. Their common compilation architecture does not make their host capabilities or physical representations interchangeable.

## Cloudflare Workers and applications

The intended artifact is JavaScript with the exports, entry adapters and host calls required by its selected Cloudflare profile. An application using Durable Objects needs per-instance state and the applicable constructor, method and lifecycle behavior. A workflow realization needs an explicit relation between source continuation state and the host's persistence, retry and resumption facilities.

Cloudflare supplies execution, dispatch, object identity and service facilities under their declared contracts. Compiler-generated code must use those facilities correctly. It does not establish a new Cloudflare scheduler or prove the platform implementation. A source orchestration policy must have a supported host realization; an unsupported policy remains a diagnosed capability requirement.

Three dependency categories matter:

| Category | Relationship to the artifact |
|---|---|
| Host facilities and declaration-only packages | Generated boundary calls use the host implementation. Type declarations do not become executable SDK dependencies. |
| Executable SDK wrappers and subsidiary libraries | Their behavior remains a dependency until the required functionality has an accepted replacement. |
| Clef-owned implementations | Compile with the application through the same semantic and lowering pathway. |

An artifact without third-party JavaScript dependencies is possible when all required executable library behavior is supplied by compiled Clef or the declared host facilities. Bundling vendor code into one file does not remove that dependency. [Source recovery](05_supply_chain_and_transcribe.md) explains the replacement process.

FSharp.CloudEdge's working F#/Fable route and its September [selected delivery](../../../FSharp.CloudEdge/docs/SDK-DELIVERY-ACCEPTANCE-20260913.md) provide concrete binding and runtime cases. They do not establish completion of the Clef/JSIR route. Provisioning and uploading the resulting artifact belong to deployment tooling using the management API, independently of the compiler that produced it.

## Browser applications

A browser profile identifies its available host APIs and execution assumptions. Direct host bindings and owned Clef code can form an application without a third-party framework. If the application uses SolidJS or another executable library, that library remains in the dependency closure unless its required behavior is replaced.

Partas.Solid through Fable is the existing frontend path. A Clef reactive surface and its supported realizations require their own language and library design; this folder does not assume such a surface is implemented or equate reactive signals with actors by analogy. Browser capabilities such as shared buffers or particular transports must be declared for the selected environment rather than inferred from the word JavaScript.

## WebView desktop applications

The WREN stack combines a Composer-compiled native host with a WebView frontend. Existing exemplars use F#/Fable on the frontend. A future Clef JavaScript frontend would use the same backend contract as other JavaScript applications, with WebView-specific host and IPC boundaries.

The native host and frontend have distinct resource lifetimes and representations. A shared declaration can support both endpoints, but does not itself prove codec agreement or make a JavaScript object a native memory block. Embedding a frontend bundle in a native artifact also does not change its dependency ownership.

## BAREWire across the contexts

[BAREWire's substrate formalism](../../../BAREWire/docs/Substrate_Formalism.md) separates source structure, carrier realization and the final payload. Types, dimensions, schemas, constraints and proof evidence remain in PSG/codata through the decisions and preservation checks that need them. They need not accompany each final payload.

Untagged means the payload carries no self-describing compiler type, schema, dimension or proof metadata. A union case index, optional-value presence flag, length or protocol identifier can still be ordinary data required by the agreed contract. These fields do not identify the contract itself.

Ordinary JavaScript records and closures can use host objects and functions without a native address layout. Byte-backed regions use an explicit buffer/view contract. Native ABI padding and pointer layout are not automatically wire encoding; a zero-copy path needs additional ownership, validity and host-capability evidence.

### Working JavaScript evidence

The current BAREWire JavaScript implementation compiles shared codec sources through Fable, selecting the appropriate text and floating-point substrate implementations. Its tests exercise byte vectors, round trips and selected rejection cases. The CloudEdge [ByteBridge fixture](../../../FSharp.CloudEdge/tests/ByteBridge/README.md) additionally exercises generated Workers body bindings, view offsets, detached buffers and typed failures in Node Fetch.

These checks give future JSIR lowering a concrete contract to preserve. They do not establish arbitrary foreign-object admission, all host profiles or completed artifact-bound proofs. See the [BAREWire evidence inventory](../../../BAREWire/docs/12%20Intersection%20Subset.md).

### Obligations shared by both endpoints

The useful parallel with typed remoting is a shared contract exposed through typed interfaces on both sides of a transport. HTTP/JSON and BAREWire binary encoding can realize such an agreement with different representation obligations. The contract alone does not establish that either endpoint implements it correctly. For Clef, the constructive path is to connect the agreed meaning to each endpoint's graph obligations, marshaling and emitted operations, then to the protocol relationships maintained across communication.

The endpoints must agree on the declared meaning and encoding. Both lowerings must preserve numeric conversion, field order, byte order, extent checks and relevant failure behavior. Decoders still check bounds and encoding constraints; an agreed schema does not make malformed input impossible. Decoding a Hello frame is not enforcement of session agreement.

[Numeric selection and precision](08_numeric_selection_and_precision.md#5-preserve-precision-across-transport-and-suspension) specifies the numerical part of that agreement. Transfer fidelity is directional; coverage alone does not establish exactness. An exact global reduction must retain its partial information until the permitted final rounding, including through checkpoints and recovery.

[Carrying Proofs into JavaScript](../../../clef-lang-site/hugo/content/blog/carrying-proofs-into-javascript.md) adds the relationships that transport alone cannot establish: job identity, contribution multiplicity, continuation eligibility and consistent recovery state. Those are application and protocol obligations, preserved using the supported host facilities.

“JavaScript you can prove correct” therefore means generated JavaScript with an argument for stated properties under explicit assumptions. It can realize those properties with different code and structures from the vendor SDK. The claim depends on the correspondence through lowering and the declared host contract, with runtime checks establishing premises that incoming values must satisfy.

## One witness boundary

All these contexts use the same architectural ownership: semantic facts in the PSG, Baker decomposition, Alex's portable witnessing, then target realization. Host differences enter through declared capabilities and contracts, not a parallel middle end that dispatches on library names or skips semantic elaboration.
