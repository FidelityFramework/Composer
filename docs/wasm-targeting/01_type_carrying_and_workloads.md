# Type Carrying: JSON, JavaScript, WASM, and What Belongs in Tier 2

**SpeakEZ Technologies | Fidelity Framework**
**April 2026**

The earlier discussion of JavaScript's type-carrying capability ([javascript-targeting/](../javascript-targeting/), and in prior architectural exchanges) established that JavaScript at the language level carries only runtime type tags — `Number`, `String`, `Boolean`, `Object`, `Function`, `Symbol`, `BigInt`, `null`, `undefined` — with no ahead-of-time type information reachable from user code. The typed-bytes surface (`ArrayBuffer` + `TypedArray` + `DataView`) is where JavaScript carries structural type information at runtime, and BAREWire is Fidelity's answer for carrying schema-verified structure through that surface.

This document extends that discussion into the WASM context. The specific question is: given that JavaScript's effective on-the-wire vocabulary in a Cloudflare Worker is JSON (plus binary where the author explicitly reaches for it), and given that WASM has a very different type-carrying story, *what workloads actually benefit from Tier 2 (WASM-in-Worker) over Tier 1 (pure JavaScript Worker)?* The answer is mechanical once the type-carrying capacities of both formats are made explicit.

## The JSON Ceiling

JSON is the default structured-data format inside the Cloudflare Workers runtime. HTTP request bodies arrive as JSON via `await request.json()`. Response bodies leave as JSON via `Response.json(...)`. KV values, Queue message bodies, R2 metadata, D1 rows — all of them flow through JSON at some point in the typical Worker pipeline. Even Durable Object storage serializes to JSON-compatible forms for most calls.

What JSON carries:

- **Strings.** UTF-16-origin, serialized as UTF-8, canonical in JSON. No length prefix limits beyond the encoding.
- **Numbers.** IEEE 754 double precision only. No integer/float distinction. No fixed-width integers. Integers larger than 2^53 lose precision. No BigInt (JSON.stringify throws on BigInt).
- **Booleans.**
- **`null`.**
- **Arrays.** Ordered, heterogeneous. No fixed length, no typed element constraint.
- **Objects.** String-keyed only. Unordered by spec (though most implementations preserve insertion order). Values are any of the above.

What JSON explicitly does not carry:

- **Binary data.** There is no binary type in JSON. Binary payloads are carried as base64-encoded strings, which costs ~33% size overhead and requires explicit encode/decode on both sides. The base64 convention is just that — a convention; JSON itself does not know it.
- **Dates.** There is no date type in JSON. Dates are carried as ISO 8601 strings by convention. Parsing is a separate step and is error-prone (timezone handling, millisecond precision, leap seconds, etc.).
- **Integer precision beyond 2^53.** IDs, timestamps in nanoseconds, and cryptographic values that need 64-bit precision do not round-trip through JSON reliably. BigInt was added to JavaScript to solve this at runtime, but JSON cannot carry BigInt.
- **Discriminated unions.** There is no sum-type representation in JSON. The convention is a discriminant field (`{"tag": "Add", "amount": 42}`), but the JSON parser does not enforce the discriminant-to-shape mapping. The receiver has to re-validate which fields are expected for each tag.
- **References and cycles.** JSON is a tree. Shared references and cycles must be encoded through explicit ID/reference conventions (which most libraries do not handle automatically).
- **Functions, Symbols, class instances.** JavaScript values that do not map to JSON's value set are silently dropped or throw on `JSON.stringify`.
- **Type identity.** A JSON object does not carry which TypeScript interface or F# record type it corresponds to. Every deserialization is "I hope this matches what I expect."

The operational consequence: every JSON-carried value arrives at the receiver as structurally opaque data that the receiver must validate before trusting. In practice, this means a Zod/ajv/io-ts/TypeBox schema check at every boundary, or a decision to trust the sender and risk shape mismatches at runtime. In a Cloudflare Worker, the cost of schema validation per request is real — often single-digit milliseconds for non-trivial schemas, which is a significant fraction of a Worker's typical compute budget.

JSON is, in short, a *convention for string-based structured data*. It is useful, widely supported, human-readable, and completely inadequate as a type-preserving transport. Everything that makes it useful also makes it lossy.

## The WASM Floor

WebAssembly's type-carrying capacity is categorically different, and the categorical difference is what makes Tier 2 worth considering in the first place.

**At the module boundary.** WASM functions have typed signatures. Each parameter and each return value is one of a fixed set of value types:

- `i32`, `i64` — signed or unsigned integers of known width. No IEEE 754 double coercion; a 64-bit integer is a 64-bit integer.
- `f32`, `f64` — single- and double-precision floats. Distinct types; the compiler and runtime know which is which.
- `v128` — SIMD vector value (four 32-bit lanes, eight 16-bit lanes, or sixteen 8-bit lanes, depending on operation). Cloudflare Workers supports SIMD; this type is real and usable.
- `funcref` — reference to a WASM function. First-class.
- `externref` — opaque reference to an external (typically JavaScript) object. Carried by handle; the WASM module cannot introspect the referent but can pass it back to JavaScript unchanged. This is how a WASM Worker holds onto a KV namespace binding or a Durable Object stub without serializing through JSON.

The boundary itself enforces type safety. Calling a WASM function with the wrong argument types produces a trap; a mismatched return expectation is impossible. This is more than JSON carries at its own boundary, by several orders of magnitude.

**Inside the module's linear memory.** WASM linear memory is a flat byte array, but the code operating on it was compiled with full knowledge of the layout. Offsets are compile-time constants. Field accesses are direct pointer arithmetic. The compiler produced code that reads exactly the bytes it expected to find. There is no runtime type tag in linear memory because the code doesn't need one — the code already knows what's there.

This is the same model native Fidelity uses. A `memref` operation in MLIR reads a known offset from a known buffer; the LLVM backend lowers it to a load instruction. The WASM backend lowers it to a WASM load instruction on linear memory. The byte layout, established by the compiler, is identical between native and WASM. BAREWire codecs exploit this: a BAREWire-encoded `CounterMsg.Add(42)` occupies the same bytes on the native heap and in WASM linear memory because both lowerings derive from the same PSG discriminated union.

**Across WASM module boundaries via WIT and the Component Model.** The WebAssembly Component Model, reaching mainstream in 2026, defines WebAssembly Interface Types (WIT) that describe module-to-module interfaces precisely:

- `record` — fixed-shape structures with named fields and declared types.
- `variant` — discriminated unions; each case has a name and optional payload type. *This is the WASM-native answer to JSON's discriminant-field convention.* The Component Model enforces tag-to-payload correspondence at the boundary.
- `list<T>` — homogeneous typed list.
- `tuple<T, U, V>` — fixed-arity tuple with per-position types.
- `option<T>` — optional value; `none` or `some(T)`.
- `result<T, E>` — success-or-error; Rust-style error handling as a first-class type.
- `resource` — handle type with ownership semantics (`own` vs. `borrow`). Lifetimes and move semantics tracked at the interface.
- `string` — UTF-8 with explicit length.

A component that exports a WIT-typed interface and another component that imports it can communicate without a JSON round-trip. Values cross the boundary with their types intact. This is qualitatively different from the JS/WASM boundary through `wasm-bindgen`, which still has to convert between JavaScript values and WASM values at each call; WIT boundaries between components carry typed data natively.

**BAREWire on top of all of this.** BAREWire is Fidelity's canonical wire format. On WASM, BAREWire codecs emit `memref` operations that read and write linear memory according to the compiler-known layout. No byte escapes the format the sender and receiver agreed on at compile time. The structural type safety argument from [Fidelity.CloudEdge/docs/08b_actor_core.md §3.4](../../../Fidelity.CloudEdge/docs/08b_actor_core.md) applies uniformly: the sender's compiler verified the DU, the codec encoded the verified structure, the wire preserves the bytes, the receiver's codec reconstructs. Nothing about this story relies on JavaScript's runtime type tags or on JSON's string-and-number vocabulary.

## The Delta

Side by side, the type-carrying capacity is:

| Capability | JSON | JS TypedArray/DataView | WASM linear memory + WIT |
|:-----------|:----:|:---------------------:|:-------------------------:|
| Fixed-width integers (i8, i16, i32, i64) | No | Yes | Yes |
| Unsigned integers | No | Yes | Yes |
| Integer precision beyond 2^53 | No | Yes (BigInt64Array) | Yes |
| Float vs. double distinction | No | Yes | Yes |
| SIMD vectors | No | No (approximation via TypedArray) | Yes (`v128`) |
| Binary data without base64 | No | Yes | Yes |
| Discriminated unions | Convention | Manual via DataView | Yes (WIT `variant`) |
| Records with typed fields | Convention | Manual via DataView | Yes (WIT `record`) |
| Resource/handle types with ownership | No | Weakly (external refs) | Yes (WIT `resource`, `own`/`borrow`) |
| Result/option types | No | No | Yes (WIT `result`, `option`) |
| Type identity preservation | No | Author convention | Compiler-enforced |
| Zero-copy access to structured data | No | Yes with discipline | Yes natively |
| Schema validation cost per access | High (runtime) | Author-written | Compile-time |

The delta between JSON and WASM is vast. The delta between JavaScript's typed-bytes surface and WASM is narrower — `TypedArray` + `DataView` gets you most of what WASM linear memory gives you in terms of byte-level type carrying — but WASM wraps that surface in a compiler-enforced type system at the module boundary, which JavaScript does not have.

## The Cloudflare-Specific Implications

A Cloudflare Worker's default position is JSON-over-HTTP. HTTP request arrives, JSON body parsed, business logic runs over JavaScript objects, JSON response serialized. KV bindings return JSON (or binary-as-base64-wrapped JSON). Queues send and receive JSON messages. D1 returns JSON-like row data. This pipeline is JSON end to end in the overwhelming majority of Workers, and for the overwhelming majority of Workers, that is the right choice.

WASM-in-Worker changes the picture only for specific workload shapes. The decision rubric:

**Workloads where WASM's type-carrying advantage translates to Worker-level benefits:**

- **Binary codec hot paths.** BAREWire encode/decode, Protobuf, MessagePack, CBOR, Cap'n Proto, FlatBuffers — any format whose bytes are structured and whose access is random-by-offset. JSON-level access to these payloads requires parsing the binary, constructing JavaScript objects, and then discarding those objects on the response path. WASM-level access reads exact bytes at exact offsets without intermediate object construction.
- **Numeric computation over typed buffers.** BitNet model inference, image processing, audio filtering, cryptographic operations, hashing, dense linear algebra. The data arrives as an ArrayBuffer (from a POST body, a KV value, a Queue message with a binary payload), gets mapped into WASM linear memory once, processed with the full WASM typed memory model, and the result is written back to an ArrayBuffer. No JSON involved on the hot path.
- **Structured data with precision requirements.** 64-bit integer IDs (Snowflake-shaped), nanosecond timestamps, cryptographic nonces, bit-packed flags. JavaScript's Number type loses precision; JSON can't carry these faithfully. WASM operates on exact 64-bit integers without coercion.
- **SIMD-accelerated workloads.** Cloudflare Workers supports WebAssembly SIMD, which is the one form of parallelism available inside the single-threaded isolate: same instruction, many data lanes, one thread. Composer's memory marshaling is a particularly strong fit here because the compiler owns the layout end-to-end and can emit SIMD-friendly alignment, structure-of-arrays transposition, and bit-packed low-precision representations automatically. See the "SIMD and Composer's Memory Marshaling" section in [README.md](./README.md) for the detailed discussion. Natural candidates include BitNet inference, BAREWire batch decode, media and image processing, and cryptographic primitive acceleration.
- **Component Model compositions.** When the application composes multiple WASM components communicating via WIT interfaces, the inter-component boundary is typed without JSON. This is a 2026 capability that changes what "modular compute" looks like in a Worker.

**Workloads where WASM offers no advantage, and the copy-in/out cost makes it worse than pure JS:**

- **HTTP routing and redirection.** Read a path, decide a target, write a location header. WASM cannot do this faster than JavaScript can, and the copy-in/out overhead dominates.
- **Authorization and token validation.** Parse a JWT, check signatures, look up a permission. JavaScript's `SubtleCrypto` is efficient; a WASM crypto library adds copy overhead without compute gain for single-token operations.
- **Simple key-value passthroughs.** Read from KV, return as response. No transformation; no compute. Copy-in/out is pure overhead.
- **JSON-centric API gateways.** Receive JSON, transform fields, send JSON. JavaScript's JSON handling is V8-optimized; routing data through WASM adds cost without benefit when the data never leaves JSON shape on either side.
- **High-fanout I/O orchestration.** Issue N KV/D1/R2 fetches in parallel, combine results. The compute per fetch is tiny; the I/O dominates wall time. WASM adds no wall-time benefit and costs setup time.

The heuristic: if the Worker's critical path is *compute over structured bytes*, Tier 2 is the right home. If the critical path is *JSON transformation and I/O orchestration*, Tier 1 is the right home. Many Workers are mixed — a routing shell that occasionally needs a crypto operation, a validation layer that occasionally needs image resizing — and the right answer is to keep the shell in pure JavaScript and invoke WASM only for the inner hot path, which is exactly the pattern Cloudflare described in the 2018 post and which remains the correct pattern for mixed workloads.

## The BAREWire-Specific Case

For Fidelity applications specifically, BAREWire is the reason Tier 2 matters. A Fidelity actor exchanging BAREWire frames with other actors — over WebSocket, MoQ, or Queue messages — has its hot path in binary codec work. The codec is compile-time derived from the PSG DU definition; the bytes on the wire are structured; the access pattern is exactly what WASM linear memory is good at.

In Tier 1 (pure JavaScript Worker), BAREWire codecs compile to JavaScript functions that call `DataView.getUint8`, `DataView.getInt32`, `DataView.getBigInt64`, etc. The emitted code is idiomatic TypedArray-style JavaScript. It works, it is correct, and it is significantly slower than the equivalent WASM.

In Tier 2 (WASM-in-Worker), the same BAREWire codec, derived from the same PSG DU definition, compiles to WASM linear memory operations. The encode path writes bytes at compile-time-known offsets; the decode path reads bytes at compile-time-known offsets. The bytes are byte-identical to what the Tier 1 codec produces, because the BAREWire invariant holds across lowering targets, but the execution is orders-of-magnitude faster for codec-heavy workloads.

A Fidelity Worker whose job is to bridge between a native actor cluster (BAREWire over WebSocket) and a Cloudflare service (KV, R2, D1) is a natural Tier 2 candidate: the BAREWire frame parsing is compute-dense, the cluster-facing path is binary, and the service-facing path is the only part that needs JSON conversion. Splitting the Worker into a JavaScript shell (service-facing I/O and JSON translation) plus a WASM core (BAREWire codec operations) is the canonical shape.

## What This Means for Downstream Agent Work

**1. The Tier 1 vs. Tier 2 decision is a data-shape decision.** The shape of data flowing through a Worker determines the tier more than the language it was authored in. A Clef-authored Worker that does nothing but JSON routing should compile to Tier 1 output. A Clef-authored Worker that processes BAREWire frames should compile to Tier 2 output. Composer's target-profile logic should include this classification — probably as codata on the PSG, derived from the types actually in use on the critical path.

**2. BAREWire is the strongest case for Tier 2.** Every Fidelity project that exchanges BAREWire frames with anything benefits from Tier 2 for those exchanges. This is the concrete early-adopter workload for WASM-in-Worker in the Fidelity ecosystem. Prototyping the Fidelity BAREWire WASM codec and measuring it against the JavaScript codec equivalent should be an early milestone in the WASM work.

**3. WIT-typed component boundaries are worth targeting deliberately.** The 2026 Component Model is the first time WASM has a standards-track typed inter-module interface. Composer emitting WIT-described components, rather than monolithic WASM modules, positions Fidelity applications to compose with other-language components (Rust, C++, Go) without JSON round-trips. This is a downstream feature, but the WAMI pathway should expose WIT emission as a first-class capability, not an afterthought.

**4. JSON remains the right answer for the majority of Workers.** This document's emphasis on WASM's type-carrying advantage should not be read as "JSON is obsolete" or "Tier 2 is the default." JSON's ubiquity and V8-optimized JavaScript handling make it the right answer for the majority of Worker workloads, which are not compute-dense. WASM is a tool for specific workload shapes; most Workers are not those shapes.

**5. The type-carrying framing gives Composer a principled compilation decision.** Given a Clef program targeting Cloudflare Workers, Composer can reason about the types flowing through the critical path. If the critical path is dominated by types that JSON represents faithfully and inexpensively (strings, small objects, shallow arrays), Tier 1 emission is optimal. If the critical path is dominated by types that JSON represents poorly or expensively (64-bit integers, binary blobs, discriminated unions with large payloads, dense numeric arrays), Tier 2 emission is optimal. This decision can be partially automated from PSG type analysis rather than left entirely to the application author.

## Cross-References

- [README.md](./README.md) — the WASM triangulation with Tier 1/2/3 structure
- [javascript-targeting/01_two_models.md](../javascript-targeting/01_two_models.md) — JavaScript's type-carrying story in detail
- [javascript-targeting/03_four_wings.md](../javascript-targeting/03_four_wings.md) — the four deployment wings; Wing 4 (Shared BAREWire) is the workload this document treats as the canonical Tier 2 case
- [Fidelity.CloudEdge/docs/08b_actor_core.md](../../../Fidelity.CloudEdge/docs/08b_actor_core.md) §3.4 — the BAREWire trust argument across the erasure boundary
- [BAREWire](../../../BAREWire/) — the wire format specification
- [WebAssembly Interface Types (WIT)](https://component-model.bytecodealliance.org/design/wit.html) — the Component Model's type system
- [WebAssembly SIMD](https://v8.dev/features/simd) — V8's SIMD implementation, which Cloudflare Workers exposes
