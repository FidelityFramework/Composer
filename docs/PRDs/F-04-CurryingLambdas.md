# F-04: Currying & Lambdas

> **Layout note (2026-09).** The packed environment layouts below (`{code_ptr, …}`; captures from `[1]`, or `[3]` for lazy and seq) are historical sketches. The current contract is the two-value pair `(fn, env)` with no function address stored in the environment as data — [closure representation](../../../clef-lang-spec/spec/closure-representation.md), [lazy representation](../../../clef-lang-spec/spec/lazy-representation.md) and [sequence representation](../../../clef-lang-spec/spec/seq-representation.md). The [C-01 contract](C-01-Closures.md) and recorded implementation evidence govern current lowering; these sketches do not describe current representation authority.

> **Sample**: `04_HelloWorldFullCurried` | **Status**: Retrospective | **Category**: Foundation

> **Current acceptance, 2026-09-26: In-Progress.** The
> [full-manifest audit](../C_F_Checkpoint_2026-09-26.md#subsequent-full-manifest-audit--2026-09-26)
> records original04 failing callable declaration/result/storage witnessing.
> Earlier passing evidence is historical. Restore the existing oracle through
> the C-01/C-02 contracts; the retrospective label is not a current pass.

## 1. Executive Summary

This sample introduces curried functions, lambda expressions (`fun`), and partial application. These are the building blocks for closures (C-01), but this sample focuses on the structural representation without capture analysis.

**Key Achievement**: Established `SemanticKind.Lambda` representation in PSG, preparing for capture analysis in C-01.

---

## 2. Surface Feature

```fsharp
module HelloWorld

let greet (salutation: string) (name: string) =
    Console.writeln $"{salutation}, {name}!"

let sayHello = greet "Hello"

let arena = Arena.create<'a> 1024
Console.readlnFrom &arena |> sayHello
```

**Key Constructs**:
- Curried function `greet` with two parameters
- Partial application creating `sayHello`
- Lambda (implicit in currying)

---

## 3. Infrastructure Contributions

### 3.1 Lambda Representation

Lambda expressions are represented in PSG with `SemanticKind.Lambda`:

```fsharp
type SemanticKind =
    | Lambda of
        parameters: (string * NativeType * NodeId) list *
        body: NodeId *
        captures: CaptureInfo list  // Empty until C-01
```

**Structure**:
- Parameters: List of (name, type, binding node)
- Body: NodeId pointing to lambda body
- Captures: Empty in F-04, populated in C-01

### 3.2 Curried Functions

Clef curried functions are desugared to nested lambdas:

```fsharp
let greet salutation name = ...
```

Becomes:

```
Lambda: salutation ->
  Lambda: name ->
    body
```

This is the standard ML representation.

### 3.3 Function Types

Multi-parameter curried functions have chained function types:

```fsharp
greet: string -> string -> unit
```

Represented as:
```
TFun(string, TFun(string, unit))
```

### 3.4 Partial Application

Applying one argument to a two-argument curried function:

```fsharp
let sayHello = greet "Hello"
```

Creates a value of type `string -> unit` - a function awaiting its second argument.

**PSG Representation**:
```
LetBinding: sayHello
└── Application
    ├── Function: greet
    └── Argument: "Hello"
```

The result is a **thunk** - a partially applied function.

### 3.5 Thunk Structure (Pre-Closure)

Without captures, thunks are simple function pointers with pre-filled arguments:

```
Thunk (no captures)
┌─────────────────────────────────────┐
│ code_ptr: ptr to greet_stage2      │
│ arg0: "Hello" (pre-filled)         │
└─────────────────────────────────────┘
```

C-01 extends this to include environment captures.

---

## 4. PSG Structure

```
ModuleOrNamespace: HelloWorld
├── LetBinding: greet
│   └── Lambda: salutation ->
│       └── Lambda: name ->
│           └── Application: Console.writeln
│               └── InterpolatedString
│
├── LetBinding: sayHello
│   └── Application
│       ├── Function: Ident(greet)
│       └── Argument: "Hello"
│
└── StatementSequence
    └── Application (pipe-reduced)
        ├── Function: sayHello
        └── Argument: readlnFrom result
```

---

## 5. Type Inference

The type checker resolves:

| Binding | Type |
|---------|------|
| `greet` | `string -> string -> unit` |
| `sayHello` | `string -> unit` |
| `arena` | `Arena<'a>` |

Type inference ensures partial application produces the correct residual type.

---

## 6. Coeffects

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | SSA for all bindings |
| ClosureLayout | Prepared (empty) for C-01 |

The ClosureLayout coeffect exists but contains no captures - this sample has no variable capture.

---

## 7. MLIR Output Pattern

```mlir
// greet_stage2: receives pre-filled salutation, takes name
// Strings are memref<?xi8> — no separate length parameter
func.func @greet_stage2(%salutation: memref<?xi8>, %name: memref<?xi8>) {
  // Build interpolated string
  // Call Console.writeln
}

// sayHello thunk creation
func.func @main() -> i32 {
  // Create thunk with pre-filled "Hello"
  %hello = memref.get_global @str_hello : memref<5xi8>

  // Read name
  %name = func.call @Console.readln() : () -> memref<?xi8>

  // Apply thunk (call greet_stage2 with both args)
  func.call @greet_stage2(%hello, %name) : (memref<?xi8>, memref<?xi8>) -> ()

  %zero = arith.constant 0 : i32
  func.return %zero : i32
}
```

---

## 8. Validation

```bash
cd samples/console/FidelityHelloWorld/04_HelloWorldFullCurried
/path/to/Composer compile HelloWorld.fidproj
echo "Lambda" | ./HelloWorld
# Output: Hello, Lambda!
```

---

## 9. Architectural Lessons

1. **Nested Lambdas**: Curried functions are sequences of single-argument lambdas
2. **Thunks Without Closures**: Partial application works without capture analysis
3. **Type Chain**: `a -> b -> c` is `TFun(a, TFun(b, c))`
4. **Preparation for C-01**: Lambda structure ready for capture analysis

---

## 10. Relationship to C-01

This sample establishes lambda **structure**. C-01 adds:
- Capture analysis (what free variables are referenced)
- Capture modes (by-value vs by-ref)
- Flat closure representation

The transition from F-04 to C-01:

| Aspect | F-04 | C-01 |
|--------|------|------|
| Lambda representation | Yes | Yes |
| Parameters | Yes | Yes |
| Body reference | Yes | Yes |
| Captures list | Empty | Populated |
| Closure struct | Thunk only | Full closure |

---

## 11. Downstream Dependencies

This sample's infrastructure enables:
- **C-01**: Closure infrastructure (extends lambda with captures)
- **C-02**: Higher-order functions (uses curried patterns)
- **C-05**: Lazy evaluation (thunk pattern)

---

## 12. Related Documents

- [F-03-PipeOperators](F-03-PipeOperators.md) - Pipes with curried functions
- [C-01-Closures](C-01-Closures.md) - Full closure implementation
- [PSG_architecture.md](../PSG_architecture.md) - SemanticKind definitions
