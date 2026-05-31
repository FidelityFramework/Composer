# `obj` and `null` at the JavaScript Boundary

**SpeakEZ Technologies | Fidelity Framework**
**April 2026**

Clef has no `obj` (open, untyped reference) and no `null` (bottom value inhabiting any reference type). JavaScript and its runtimes — including Cloudflare Workers, which treats `null` as a first-class API citizen — are built around both. This document addresses how Clef crosses that conceptual gap in both directions: handing values to a runtime that expects `obj`/`null`, and receiving values from a runtime that produces them.

The short answers up front:

- **Null.** `null` is an FFI-boundary concept only. Within Clef source code, nothing is ever null. Optional values are `Option<T>`, and the binding layer translates between `Option` and the runtime's `null` convention. You never type `null` in a Clef program.
- **`obj`.** Three strategies cover the full surface: opaque handles (passthrough without introspection), typed discriminated unions (Clef's representation of JS's value space), and schema-directed narrowing (the "full-loop knowledge" pattern that uses compile-time information to pre-determine the shape).
- **The compiler's advantage.** Because Composer sees both the binding side (what the runtime produces) and the consumer side (what the Clef code expects), it can insert the right narrowing, validation, or passthrough at each boundary without the programmer specifying it manually.

## The JavaScript Side of the Gap: What `obj` Actually Is

Before discussing how Clef crosses the gap, it's worth naming precisely what `obj` means in JavaScript, because the intuitive reading "it's a record" is almost right but misleading in specific ways.

**JavaScript `Object` is semantically an associative map.** The language specification defines `Object` as a collection of key-value pairs where keys are strings or Symbols. You can add properties (`o.x = 1`), delete them (`delete o.x`), check existence (`"x" in o`), and iterate (`Object.keys(o)`, `for (const k in o)`). All of this is map behavior, not record behavior. Records have a fixed shape determined at definition; objects are open-ended.

**V8's implementation, however, treats stable-shape objects as if they were records.** V8 maintains *hidden classes* (internally called `Map`) that track the shape of an object: which properties it has, in what order, and which storage slots hold their values. When many objects are created with the same property names added in the same order, V8 shares one hidden class across all of them and generates specialized access code — essentially treating them as records for the purpose of field access. When a program adds a new property to an existing object, V8 creates a new hidden class (a transition from the old class), and the object migrates. The V8 optimizer is tuned around the pattern of objects keeping stable shapes; code that mutates object shapes frequently gets slower.

So the "hidden record" intuition is correct at V8's implementation level but wrong at the language-specification level. An `Object` that never gets properties added or removed is, functionally, a record — and V8 treats it as one. An `Object` that does have properties added dynamically is a real associative map and incurs the cost of being one.

**Is iteration order guaranteed?** Yes, and the guarantee is stronger than most developers realize. Since ES2015, and tightened further in ES2020, the JavaScript specification requires a specific iteration order for property access (`Object.keys`, `Object.entries`, `Object.getOwnPropertyNames`, `for...in`, `JSON.stringify`). The order is:

1. **Integer-indexed keys first**, in ascending numeric order, *regardless of insertion order*. A key like `"42"` is treated as an integer index if it parses as a non-negative integer without leading zeros. These come out in `0, 1, 2, ...` order.
2. **String keys next**, in insertion order. `{a: 1, b: 2, c: 3}` iterates as `a, b, c`.
3. **Symbol keys last**, in insertion order.

The integer-first rule is the one gotcha. `{"2": "x", "1": "y", "foo": "z"}` iterates as `"1", "2", "foo"`, not in source order. For JSON-parsed objects, the same rule applies: numeric-string keys are reordered, alphabetic-string keys preserve source order.

**This is substantially different from .NET's `Dictionary<K, V>`.** Historically, `Dictionary<K, V>` has *no spec-guaranteed iteration order*. Modern .NET runtimes preserve insertion order in most cases as an implementation detail, but code that depends on the order is formally wrong per the spec. For a .NET collection with guaranteed insertion order, the correct type is `OrderedDictionary` (non-generic, awkward) or `List<KeyValuePair<K, V>>`. JavaScript's plain `Object` in ES2015+ is closer to those ordered collections than to `Dictionary<K, V>`. JavaScript also has a separate `Map` type (ES6+), which is explicitly insertion-ordered and does not have the integer-keys-first reordering quirk — it iterates exactly in insertion order regardless of key shape.

**Why this matters for Clef interop:**

| Question | Answer | Implication for Clef |
|:---------|:-------|:---------------------|
| Is `Object` a record? | No, it's an associative map — but V8 optimizes stable-shape objects as if they were records | Don't rely on record-shape assumptions; rely on field-name access |
| Is iteration order guaranteed? | Yes, since ES2015, with the integer-key-first quirk | Clef validators can traverse fields in a known order when needed |
| Can you rely on property-order stability across API versions? | Only as much as the API promises | Validators must access by name, not by position |
| Is this like .NET's `Dictionary`? | No — `Dictionary<K,V>` has no spec-guaranteed order | Treat JS `Object` as closer to an ordered dict than to an unordered one |
| What about JS `Map`? | Explicitly insertion-ordered, no integer-key quirk | When a Clef binding needs guaranteed-ordered map semantics, `Map` is the better runtime target than `Object` |

**Two practical consequences for Clef's interop strategy:**

**1. Prefer name-based access, always.** The schema-directed validator reads fields by name, not by iteration position. If an API returns `{id, email, createdAt}` today and returns `{email, id, createdAt}` tomorrow, the validator doesn't care — field order doesn't affect name-based access. This is the safe default and is what every production validating deserializer (serde, Zod, ajv, io-ts, TypeBox) does. Clef follows the same rule. The integer-key-reordering quirk never affects Clef binding code because Clef bindings never depend on iteration order.

**2. V8's hidden-class optimization comes for free.** When Clef compiles a record type to a JavaScript object, the generated code consistently produces the same property order. This lets V8 settle on a single hidden class for all instances of that Clef record type and specialize access. The Clef programmer gets this performance characteristic without having to think about it, because Clef's strict typing naturally produces stable-shape output. The contrast is with hand-written JavaScript that dynamically builds objects (adding properties conditionally, mixing shapes) and degrades to V8's slow path; Clef's output does not do this because Clef records have fixed shapes by construction.

**The "hidden record" intuition, sharpened:** a JavaScript `Object` that was produced by Clef-compiled code *is effectively a record* — fixed shape, stable property order, V8-hidden-class-friendly — even though JavaScript itself would let that object be modified into something else at runtime. Clef's emission pattern produces record-shaped objects, and V8's implementation optimizes them accordingly. The association only weakens when Clef interacts with third-party JavaScript that mutates object shapes or produces objects with variable property sets. For that case, the schema-directed validator is the answer — it imposes a stable record-shape interpretation on whatever the runtime produces, and fails cleanly when the interpretation doesn't fit.

### Historical Note: JavaScript's `Object` as LISP Tagged Structure

JavaScript's associative-map object model is not a modern invention. It is a direct descendant of LISP's associative data structure tradition, and the parallel runs deep enough that naming it makes the whole discussion clearer.

Brendan Eich designed JavaScript in 1995 with Scheme as one of his explicit references. The "object as property bag" model — dynamically extensible, string-keyed, iterable, and checkable for property existence — is the same shape as Scheme's association lists and hash tables. The `Object.prototype` chain (prototype-based inheritance) is a direct descendant of Self, which was itself a simplification of Smalltalk, which inherited its object model from LISP's late-binding dispatch traditions. The name `Object` is a choice of vocabulary; the structure underneath is *tagged associative data*, just as it is in LISP.

The architectural consequence: LISP dialects with static-typed refinements have been doing exactly what Clef's schema-directed narrowing does, for decades. Typed Racket is the clearest modern example — it takes dynamically-typed Racket values (which include tagged structures behaviorally identical to JavaScript's objects) and applies compile-time shape declarations that generate validators at the dynamic/static boundary. The validator narrows the dynamic structure to the declared shape, returning a typed result-or-error. Clef's pattern is the same, applied to a different dynamic language.

**Common Lisp's `DEFSTRUCT` is the closest analogue to V8's hidden classes.** `DEFSTRUCT` declares a named tagged structure with typed slots in specific positions; Common Lisp implementations optimize slot access by specializing on the structure tag. V8 does the same thing under a different name: the hidden class is the specialization key, and stable-shape objects hit the specialized fast path. "LISP would have called this a tagged structure with declared slots" is essentially where V8's optimizer arrives when it observes stable-shape usage. The intuition that `<obj>` is a tagged structure à la LISP is not merely analogical; it is an accurate description of what V8's runtime actually does with stable-shape objects, expressed in older vocabulary.

This grounds Clef's interop pattern in a longer tradition than the 2026 JavaScript ecosystem discussion suggests. The problem of bridging static types to tagged associative structures is not new; the mature solutions (shape declaration, validator generation, typed narrowing, failure-as-`Result`) have been refined across LISP, ML-family, and Haskell-family languages for the better part of four decades. Clef inherits this body of work through a compiler that can generate the validators automatically, rather than requiring programmer-written `DEFSTRUCT`-like declarations or Typed-Racket-style annotations.

## The Clef Side of the Gap

Clef's type system, as documented in the [FFI Boundary Semantics](../../../clef-lang-spec/spec/ffi-boundary.md) specification, states the invariant:

> **Null exists ONLY at the FFI boundary. Within Clef code, `nativeptr<'T>` and `FnPtr<'F>` are NEVER null.**

This invariant was established for the C FFI. It extends directly to JavaScript and to WebAssembly's `externref` handles. Within Clef source, no value has `null` as a possible inhabitant. The type system enforces this at compilation. Where a C API would expose `T*` (potentially null), Clef's binding surface exposes `Option<nativeptr<'T>>`. Where a JavaScript API exposes `T | null` (the `string | null` return of `env.KV.get`, the `T | undefined` return of `Map.get`, the nullable fields on JSON-shaped responses), Clef's binding surface exposes `Option<T>`.

This is not a convention the programmer has to apply. It is a structural requirement of the binding generator (Xantham for TypeScript definitions, Farscape for C/C++ headers and OpenAPI specifications). When Xantham sees `string | null` in a `.d.ts` file, it emits `Option<string>` in the Clef binding. When it sees `T | undefined`, same thing. When it sees `T | null | undefined`, same thing — the distinction between `null` and `undefined` is absorbed into the `None` case. The Cloudflare binding for `KVNamespace.get` reads:

```clef
type KVNamespace =
    member Get : key:string -> Async<Option<string>>
```

Not `Async<string>` with a secret null possibility. Not `Async<string | null>` (Clef does not have anonymous unions with nullable-reference semantics). `Async<Option<string>>`, which is exactly what the caller sees, and which forces the caller to handle the "key not found" case explicitly through pattern matching or `Option.defaultValue`.

## "Is There a Flat Non-Null Pointer Type?"

Yes. It is `Option<T>`. Taking the common Cloudflare shape where null is first-class:

**Runtime surface:**
```typescript
// TypeScript
interface KVNamespace {
  get(key: string): Promise<string | null>;
}
```

**Clef binding (generated by Xantham):**
```clef
type KVNamespace =
    member Get : key:string -> Async<Option<string>>
```

**Clef consumer code:**
```clef
async {
    let! result = env.KV.Get("user:42")
    match result with
    | Some value -> return processValue value
    | None -> return defaultResponse ()
}
```

Or, more concisely when there is a natural default:

```clef
async {
    let! name = env.KV.Get("user:42")
                |> Async.map (Option.defaultValue "anonymous")
    return processName name
}
```

The programmer never writes `null`. The runtime, on the other end of the FFI boundary, sees `null` when the key is missing. The translation happens in the generated glue code that Xantham produces — the programmer does not see it and does not have to maintain it.

This is what "flat non-null pointer type" looks like in practice. There is exactly one type constructor that captures optionality (`Option<T>`), it is statically visible at every use site (you cannot forget the `None` case because pattern matching exhaustiveness checks catch that), and it becomes invisible below the binding layer (the compiler lowers `None` to `null` when marshaling back to the runtime).

**Cloudflare's first-class treatment of null is not a barrier; it is absorbed.** Every Cloudflare API that returns `T | null` gets surfaced to Clef as `Option<T>`. Every optional field in a Cloudflare request or response becomes an `Option` field. The Clef code above reads naturally, handles all cases, and does not contain the word `null` anywhere. The lowering to the Worker runtime below the Clef/JS boundary emits the JavaScript null where the runtime expects it.

The tradeoff: every value that can be absent is wrapped explicitly in `Option`. A programmer coming from a null-pervasive language might find this verbose at the boundary. The verbosity is real but local — it only appears at the points where optionality matters, which is exactly where it should be visible. Values that are always present are not wrapped; values that might be absent are. The programmer's attention goes to the right places.

## Three Strategies for `obj`

The `obj` question is more complex than the `null` question because JavaScript's `Object` is genuinely polymorphic. An `Object` can be a plain key-value bag, an array, a class instance, a DOM node, a function, a Promise, a Map, a Set, or any combination thereof through composition. Clef needs three strategies to cover this surface, and the right choice depends on what the compiler knows at compile time.

### Strategy 1: Opaque Handles (Passthrough Without Introspection)

The simplest case is: a value comes from JavaScript, Clef holds onto it, Clef hands it back to JavaScript later. Clef never needs to know what's inside. The value is effectively a token identifying a JavaScript object, carried through Clef code without being inspected.

WebAssembly's `externref` is the canonical form of this pattern at the WASM level. Clef's equivalent on the JavaScript interop side is `JsRef<'T>` (where `'T` is a phantom type parameter that lets the type system distinguish different reference kinds without implying Clef knows the shape):

```clef
// Store a KV binding handle received from the runtime
let kvHandle : JsRef<KVNamespace> = env.KV

// Pass it to a helper that also doesn't introspect it
let result = useKvBinding kvHandle "user:42"

// Eventually, the handle crosses back to JavaScript
someJavaScriptApi.acceptBinding kvHandle
```

The value is opaque. Clef cannot read fields off it. Clef cannot pattern-match on its shape. Clef can store it, pass it, return it, and eventually send it back to JavaScript — and that's all.

**When this is the right strategy:** holding runtime-provided handles (KV, R2, Queue, DO stub, WebSocket), wrapping DOM nodes in a browser context, threading Promise handles through async logic without awaiting them, anywhere the shape inside doesn't matter to Clef.

**Why it's safe:** the type system still tracks `'T`, so `JsRef<KVNamespace>` cannot be accidentally passed where `JsRef<R2Bucket>` is expected. The type is phantom — Clef doesn't know what's inside — but the type is still checked. Opaque does not mean untyped.

### Strategy 2: Typed Discriminated Union (`JsValue` DU)

The second case is: a value comes from JavaScript, Clef needs to look inside it, and the shape is not predictable at compile time. The canonical example is `JSON.parse(userInput)` where the input is user-controlled and might be anything.

For this case, Clef provides a first-class discriminated union representing JavaScript's runtime value space:

```clef
type JsValue =
    | JsString of string
    | JsNumber of float
    | JsBigInt of int64
    | JsBool of bool
    | JsNull
    | JsUndefined
    | JsArray of List<JsValue>
    | JsObject of Map<string, JsValue>
```

Clef code pattern-matches on this to extract whatever structure is actually present:

```clef
let rec extractUserId (value: JsValue) : Option<string> =
    match value with
    | JsObject fields ->
        fields
        |> Map.tryFind "user"
        |> Option.bind (fun userObj ->
            match userObj with
            | JsObject userFields ->
                userFields
                |> Map.tryFind "id"
                |> Option.bind (fun idValue ->
                    match idValue with
                    | JsString s -> Some s
                    | _ -> None)
            | _ -> None)
    | _ -> None
```

This is verbose for nested access, which is correct — deeply nested optional access in arbitrary JSON structure is genuinely a chain of "is this the shape I expected at this level?" checks. Helper functions (`JsValue.tryGetString`, `JsValue.tryGetField`, path-based accessors) smooth out the common cases, but the underlying model is that Clef refuses to pretend structure exists until it's been verified.

**When this is the right strategy:** parsing user-uploaded JSON, handling event payloads with open-ended shape, dynamic plugin systems, exploratory interaction with JavaScript libraries whose types aren't well-described.

**Why it's correct:** Clef's type system insists on exhaustive handling. The programmer cannot accidentally access a field on something that isn't an object; the type system forces the match on `JsObject` first. The pattern is verbose but sound.

### Strategy 3: Schema-Directed Narrowing (Full-Loop Knowledge)

The third case is the architecturally interesting one, and it's what your question was pointing at. When the compiler sees both:

- What the runtime produced (from the binding), and
- What shape the Clef consumer expects (from the use site),

it can insert a validator at the boundary that enforces the expected shape. The Clef programmer writes code against the target type; the compiler ensures the runtime value matches that type before the Clef code ever runs against it. If validation fails, a structured error propagates (typically as `Result<T, DeserializationError>`), never as a shape mismatch that crashes mid-access.

```clef
type UserProfile = {
    Id: string
    Email: Option<string>
    CreatedAt: int64
    Preferences: Map<string, string>
}

async {
    let response = fetch url
    let! data : Result<UserProfile, DeserializationError> = response.Json()
    // ^ Compiler sees the target type UserProfile and emits a validator
    //   that checks the JSON shape against UserProfile's structure.
    match data with
    | Ok profile -> return processProfile profile
    | Error err -> return errorResponse err
}
```

The Clef programmer names the target type. The compiler inspects the PSG to see what `UserProfile` looks like structurally — its field names, types, nullability, nested shapes. It generates validator code that:

- Reads the JSON.
- Checks that the top-level is a `JsObject`.
- For each required field of `UserProfile`, verifies presence and type.
- For `Option` fields (like `Email`), treats `null`, `undefined`, or absence as `None`.
- Recursively validates nested records and list elements.
- Returns `Ok profile` if everything matches, `Error` with details if not.

**The mechanics of "verifies presence and type" — what the validator actually reads.** JavaScript objects carry two kinds of runtime tags that the validator uses, and naming them explicitly clarifies why this works:

- **Property-name tags.** Every property is keyed by a string (or Symbol). The keys are observable through `Object.keys`, `hasOwnProperty`, and `in`. When the validator needs the `email` field, it uses the name "email" as the tag to access the corresponding value from the obj. This is the same mechanism LISP property-list access (`getf`) and hash-table lookup (`gethash`) use, just with JavaScript's vocabulary.
- **Value type-tags.** Every JavaScript value carries a runtime type classification accessible via `typeof`, plus more specific operators like `Array.isArray`, `instanceof`, and strict null checks. The validator reads these to confirm each field's value matches its declared type — a `typeof === "string"` result for a declared `string` field, a `typeof === "number"` for an `int64` (with additional range checks), `value === null || value === undefined` treated as `None` for `Option<_>` fields.

Together, these two tag systems carry exactly the information needed to narrow an unknown obj to a typed record. Without property-name tags, there would be no way to know which value corresponds to which field. Without value type-tags, there would be no way to verify each field's type. Both are part of JavaScript's standard runtime API; Clef's compiler-generated validator is built on top of them.

**The useful contrast with BAREWire.** BAREWire's codecs don't read runtime tags at all because BAREWire's tags are *positional* — the schema says "byte 0 is the discriminant, bytes 1-4 are the payload length, bytes 5-N are the payload." The decoder knows what's at each offset because schema and byte layout agree by construction. No reflection needed.

JSON/obj parsing is "tagged data, parse by inspecting runtime tags." BAREWire is "tagged data, parse by layout position." Both are conceptually the same operation — narrow tagged data to a typed record — but the tags live in different places. The compiler generates the appropriate parser for whichever tag location the source data uses; both paths converge on the same final typed record at the consumer.

This is exactly the "full-loop knowledge" pattern. The compiler sees:

- The `fetch` binding's return type (something that wraps an opaque JSON response).
- The `Json()` method's signature (polymorphic, `'T` determined by the call site's type annotation).
- The consumer's target type (`UserProfile`).

With all three pieces visible, the compiler emits the validator that connects them. The programmer writes the declarative target type; the imperative validation work is derived.

**When this is the right strategy:** consuming typed JSON APIs (Cloudflare bindings, OpenAPI-described services, third-party APIs with known schemas), deserializing BAREWire frames (where the schema is even stronger than JSON), any boundary where "the shape should be this, and if it isn't, that's an error worth reporting."

**Why it's powerful:** the programmer never writes validation code. The compiler writes it, and the compiler writes correctly (no missed fields, no type coercion bugs, no silent truncation). The generated validator is also typically more efficient than hand-written validation code because the compiler can specialize it to the exact target type, inline the field checks, and avoid the overhead of a general-purpose schema library.

**This is the strategy that makes Cloudflare's null-heavy API ergonomic in Clef.** When the Cloudflare binding describes a response with ten optional fields, and the Clef code declares the target type, the compiler inserts a validator that handles every `null` or missing field as `None` automatically. The programmer never sees any individual null check.

## The Risk, and How Validation Mitigates It

A natural worry at this point: "If the TypeScript binding says the runtime returns `{id: string, email: string | null}`, and I build a Clef record to match, aren't I just trusting the TypeScript annotation? What if the runtime returns something different — a missing field, a wrong type, an outright malformed response?"

The concern is correct. Blind structural mapping — assuming the runtime's value matches the type and accessing fields directly — *would* be risky. Schema-directed narrowing is not blind mapping. It is compile-time type declaration *plus* runtime validation, composed by the compiler.

A Clef expression like `response.Json() : UserProfile` does not return a raw `UserProfile`. It returns `Result<UserProfile, DeserializationError>`. The compiler emits code that:

1. Parses the JSON into a structural value.
2. Checks that the top-level is a `JsObject`.
3. For each required field of `UserProfile`, verifies presence and type. Missing required field → `Error`.
4. For each `Option<_>` field, treats `null`, `undefined`, or absence as `None`. No error.
5. For each nested record or list, recursively validates.
6. On complete success, returns `Ok profile`. On any mismatch, returns `Error(ValidationError { path, expected, actual })`.

The TypeScript type is a *contract* the runtime promises but might occasionally violate. The validator makes violations visible and handleable. If Cloudflare changes their API and the Clef binding hasn't been regenerated, or if the runtime returns an edge-case shape the TypeScript types didn't describe, the validator catches it at the boundary. The Clef handler code returns a clean error response instead of crashing mid-handler.

This is the same pattern that every validating deserializer in every other language does. It is not a new idea. The Clef-specific improvement is that the validator is compiler-generated from the target type rather than library-provided via annotations or runtime reflection.

## How Rust Handles This for WASM (and Why the Parallel Matters)

Rust's WebAssembly ecosystem — specifically `serde`, `serde-wasm-bindgen`, and `workers-rs` — addresses the exact same problem with the exact same strategy, and the parallel is worth making explicit because it validates Clef's approach against a production-tested reference.

**The Rust pattern:**

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct UserProfile {
    id: String,
    email: Option<String>,       // nullable becomes Option
    created_at: i64,
    preferences: HashMap<String, String>,
}

// Inside a Worker handler:
let js_value: JsValue = /* received from the runtime */;
let profile: Result<UserProfile, _> = serde_wasm_bindgen::from_value(js_value);

match profile {
    Ok(profile) => process_profile(&profile),
    Err(e) => Response::error(&format!("invalid shape: {}", e), 400),
}
```

The mechanics:

- `#[derive(Deserialize)]` tells `serde` to generate a validator for `UserProfile` at compile time. The validator knows every field, its type, whether it's `Option`, how it's named in JSON vs. Rust.
- `serde_wasm_bindgen::from_value` is the glue that takes a JavaScript value (received through `wasm-bindgen`) and runs it through serde's deserialization pipeline.
- The result is `Result<UserProfile, serde_wasm_bindgen::Error>`. Success gives a fully-validated `UserProfile`; failure gives a structured error with path information ("field `email` expected String, found Number").
- The programmer handles the error path explicitly. No panic, no undefined behavior, no partial access to a malformed shape.

Rust Workers built with `workers-rs` use this pattern constantly. Every boundary — KV values decoded from JSON, fetch response bodies, Queue message payloads, Durable Object state — goes through serde validation. The TypeScript types from `@cloudflare/workers-types` describe the *intent* of what each API returns; serde validates the *actual shape* that arrives; when they disagree (rarely but reliably), the programmer gets a typed error instead of a crash.

**The Clef parallel:**

| Rust mechanism | Clef equivalent |
|:---------------|:---------------|
| `#[derive(Serialize, Deserialize)]` | Compiler-emitted validator from target type |
| `serde_wasm_bindgen::from_value` | `response.Json() : TargetType` (type annotation triggers validator generation) |
| `serde::Error` | `DeserializationError` with path information |
| `Option<T>` for nullable | `Option<T>` for nullable (identical) |
| `Result<T, E>` for fallibility | `Result<T, DeserializationError>` for fallibility |
| `#[serde(default)]` for defaults | Pattern-match on `Error` and fall back |
| `#[serde(rename = "...")]` for field renames | Clef attribute on record field (generated by Xantham) |
| `#[serde(flatten)]` for embedded objects | Record composition (structural, no attribute needed) |
| Error handling is mandatory (compiler checks) | Error handling is mandatory (exhaustiveness checks on Result) |

The mapping is one-to-one. Every capability Rust's ecosystem has built to make TypeScript-typed JavaScript interop production-safe maps onto something Clef either already has or will have via the binding generator.

**The practical implication.** A Rust Worker interacting with Cloudflare APIs looks like this in practice (paraphrased from `workers-rs` examples):

```rust
#[event(fetch)]
async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    let kv = env.kv("USER_DATA")?;
    let raw: Option<String> = kv.get("user:42").text().await?;
    
    match raw {
        Some(s) => {
            let profile: UserProfile = serde_json::from_str(&s)?;
            Response::from_json(&profile)
        }
        None => Response::error("user not found", 404),
    }
}
```

Every boundary is an explicit `Result`. Every null is an explicit `Option`. Every shape assumption is validated. The programmer never writes `.unwrap()` on a boundary call in production code (and if they do, the reviewer catches it). The compiler ensures the pattern holds structurally.

The Clef equivalent reads nearly identically, with the same discipline enforced by the same compile-time mechanisms:

```clef
[<WorkerEvent("fetch")>]
let main (req: Request) (env: Env) (ctx: Context) : Async<Response> =
    async {
        let kv = env.GetKv "USER_DATA"
        let! raw = kv.Get "user:42"
        
        match raw with
        | Some s ->
            match JsonDecode.fromString<UserProfile> s with
            | Ok profile -> return Response.fromJson profile
            | Error err -> return Response.error (sprintf "invalid stored shape: %A" err) 500
        | None ->
            return Response.error "user not found" 404
    }
```

Same pattern. Same discipline. Different surface syntax. The Rust-WASM story proves this works at Cloudflare-deployed scale; Clef inherits the pattern with compiler-generated validators replacing the `derive` macros.

**Why this is not an additional engineering burden.** A common concern when proposing "generate validators at every boundary" is the engineering cost of building and maintaining the validator machinery. But the machinery is already factored out: one general-purpose validator generator, keyed by Clef types, serves every boundary. The work happens once in Composer's codegen path; it then applies uniformly to every JSON boundary, every KV value, every Queue message, every fetch response body in every Clef Worker ever deployed. This is the same economics as Rust's serde — one mature library, derive-applied everywhere, cost amortized across the whole ecosystem.

## Which Strategy, When

| Situation | Strategy | Example |
|:----------|:---------|:--------|
| Clef holds a runtime value it never introspects | Opaque handle (`JsRef<'T>`) | KV namespace binding, DO stub, WebSocket |
| Clef parses a structured value whose shape is known at compile time | Schema-directed (`Result<T, _>`) | API response to a known-schema endpoint |
| Clef parses a value whose shape is genuinely dynamic | Typed DU (`JsValue`) | User-uploaded JSON, exploratory interaction |
| Clef emits a value for the runtime | Target shape determined by binding | Returning a Response, setting a KV value |
| Clef exchanges typed messages with known schema | BAREWire | Actor-to-actor messaging, wire protocols |

The strategy is a per-use-site choice, and the compiler often makes it automatically from the type annotation the programmer provides. A Clef function that takes `JsRef<KVNamespace>` accepts an opaque handle. A function that takes `UserProfile` triggers schema-directed narrowing at its entry. A function that takes `JsValue` explicitly asks for the typed DU form. The strategy is visible in the type signature.

## BAREWire as the Sidestep

When the data in question is Clef-to-Clef (one Clef-compiled process talking to another), BAREWire sidesteps the whole obj/null question. BAREWire's wire format is derived from the discriminated union definition in the PSG; both ends of the wire share that definition; the bytes carry structure that the schema has pre-verified.

BAREWire's `None` case is a tag in the serialized bytes, not a JSON `null`. The sender's `None` is encoded as a specific byte pattern; the receiver's decoder sees that pattern and produces `None` directly. There is no intermediate JSON representation, no `null` sentinel, no schema validation step, because both ends agree on the byte layout from the same compile-time definition.

The pattern extends: a Fidelity Worker talking to a Fidelity native cluster over BAREWire-framed WebSocket messages does not encounter the obj/null issue at all. The issue only arises at Cloudflare/external-JavaScript boundaries, and that's where Xantham bindings + schema-directed narrowing do the work.

## Implications for Downstream Agent Work

**1. Xantham emits `Option<T>` wherever TypeScript has nullable or undefinable types.** The mapping is mechanical: `T | null`, `T | undefined`, `T | null | undefined`, and optional fields (`field?: T`) all become `Option<T>` in Clef. There is no other mapping. Xantham does not expose `null` to Clef source under any circumstances.

**2. The compiler emits schema-directed validators from target type annotations.** When a Clef expression has an explicit target type at a JSON-parsing boundary (`response.Json() : UserProfile`, `JSON.parse source : Config`), the compiler generates a validator for that type. This is a generic facility that applies to every JSON boundary; it should not be a per-binding feature.

**3. `JsValue` is the escape hatch for genuinely dynamic data.** The typed DU representation is in the Clef standard library surface for JavaScript interop. Code that needs to handle open-ended JSON reaches for it explicitly; code that knows the shape does not see it.

**4. `JsRef<'T>` is the phantom-typed opaque handle.** Runtime-provided handles (bindings, stubs, connection handles) are carried as `JsRef<'T>` with `'T` naming the runtime type. The handle is opaque but typed.

**5. The Cloudflare API's first-class null is absorbed entirely at the binding layer.** A Clef application targeting Cloudflare Workers does not contain `null` anywhere in its source. The nulls exist, they are real, and the runtime sees them — but they are hidden inside the `None` case of `Option` at the points where they enter or leave Clef's world. Programmers write Clef code that pattern-matches on `Option`; the generated glue code handles the null/Option translation.

**6. Full-loop knowledge is the driver, not a nice-to-have.** The whole model relies on the compiler seeing both ends of each data flow. The binding declares what the runtime produces; the consumer declares what Clef expects. Between them, the compiler inserts the right marshaling, narrowing, or validation. This is codata-driven compilation applied to the interop boundary, the same pattern documented in [Fidelity.CloudEdge/docs/00_architecture_decisions.md](../../../Fidelity.CloudEdge/docs/00_architecture_decisions.md) Decision 6 (Library Extension via PSG Codata), specialized to the specific question of "what does this JavaScript value look like, structurally?"

## Clef's Position: `obj` in the LISP Sense, Not the .NET Sense

The framing that ties the preceding discussion together, stated compactly:

> **Clef has no `obj` in the .NET sense, at any target. Clef has `obj` in the LISP sense only at the JavaScript/WASM interop boundary, and only through `JsValue`, `JsRef<'T>`, and schema-directed narrowing. Everywhere else in the language — including the FFI boundary with C, the wire boundary with BAREWire, and every native target — the type system is closed.**

Breaking this into its two halves:

**The .NET sense of `obj` does not exist.** .NET's `System.Object` is a universal root type: every reference type inherits from it, every value can be boxed as one, runtime reflection can inspect any instance, and null is a valid inhabitant of every reference. F# exposes this as `obj`. Clef does not have any of it. There is no universal root. There is no implicit boxing of value types into a reference container. There is no runtime reflection over arbitrary values. There is no `null` that can stand in for any reference. These are absent by construction, for every compilation target Clef reaches — native ELF, FPGA Verilog, NPU xclbin, JavaScript, WebAssembly. The absence is not "available but discouraged"; it is "not part of the language."

**The LISP sense of `obj` exists, with strict scoping.** LISP's tagged associative structure — a dynamic value whose shape is accessible by key and inspectable at runtime — has a direct analogue in Clef: the `JsValue` discriminated union, which represents JavaScript's runtime value space as a tagged Clef type. It exists for exactly one reason: Clef needs to interact with JavaScript and WebAssembly runtimes that produce genuinely dynamic values, and needs a way to represent "the actual shape is determined by what the runtime returns, not by the Clef source." `JsValue` provides that representation with the same shape-narrowing discipline that Typed Racket and similar LISP descendants have been applying to dynamic tagged structures for decades. `JsRef<'T>` is the companion facility for opaque handles that Clef carries through without introspection. Schema-directed narrowing is the compile-time mechanism that bridges typed Clef records to runtime-supplied `JsValue`s via compiler-generated validators.

Crucially: these facilities are **scoped to the interop boundary**. You cannot use `JsValue` as a catch-all dynamic type in ordinary Clef logic. It has no meaning outside the JavaScript/WASM context; no other target has a runtime that produces tagged dynamic values in the relevant sense. Native C FFI handles this differently — through `Option<nativeptr<'T>>` for null-pointer semantics and declared struct layouts for data shape — because C's model is static, not tagged-dynamic. FPGA/NPU/embedded targets have no analogue at all because their runtime models don't include tagged values. The LISP-sense `obj` in Clef is a facility specifically for bridging to LISP-influenced dynamic runtimes (JavaScript being the contemporary one that matters), and it is absent everywhere else by design.

**Why this matters.** The framing is not just terminological cleanup. It informs how Clef users reason about their code, how binding generators shape their output, and how Composer's backend decides when to emit validators. A Clef programmer reading code that manipulates `JsValue` knows immediately: this is interop code, this is the dynamic edge, this is where runtime shapes are narrowed to static types. A Clef programmer reading code without `JsValue` or `JsRef` knows immediately: this is fully-typed Clef, no runtime shape uncertainty, no possibility of undeclared values reaching this code. The scoping makes the boundary visible, and visible boundaries are the point.

The LISP echo is the historically accurate one. The architectural lineage — tagged associative structures with shape-narrowing at the access boundary — runs from LISP through Scheme, through Typed Racket, through JavaScript (by descent from Scheme), and into Clef's interop facility. Clef does not inherit LISP's dynamism as a pervasive language feature; it inherits LISP's vocabulary for *describing* the tagged-dynamic structures it needs to interoperate with. The result is a language that is statically typed everywhere except at explicit interop edges, and at those edges, uses the mature LISP-descended toolkit for narrowing dynamism back to static types.

## Cross-References

- [FFI Boundary Semantics](../../../clef-lang-spec/spec/ffi-boundary.md) — Clef's null-safety principle, originally for C FFI, extended here to JavaScript
- [01_two_models.md](./01_two_models.md) — how the F#/.NET model and the fully-decomposed-AST model both arrive at this same boundary question from different directions
- [04_sdk_describes_runtime.md](./04_sdk_describes_runtime.md) — bindings describe the runtime; this document describes what the bindings have to express
- [05_supply_chain_and_transcribe.md](./05_supply_chain_and_transcribe.md) — Xantham's role as the binding generator that applies these rules
- [wasm-targeting/01_type_carrying_and_workloads.md](../wasm-targeting/01_type_carrying_and_workloads.md) — the type-carrying story, of which this document's obj/null handling is a specialization for the interop edge
- [Fidelity.CloudEdge/docs/08b_actor_core.md](../../../Fidelity.CloudEdge/docs/08b_actor_core.md) §3.4 — BAREWire trust argument across the type-erasure boundary
