# C-01: MLKit-Style Flat Closures

> **Layout note (2026-09).** This PRD describes the interim environment layout, in which the code pointer is a field of the environment (`{code_ptr, …}`; captures from `[1]`, or `[3]` for lazy and seq). The settled form is the two-value pair `(fn, env)` with no function address stored in the environment as data — spec `closure-representation.md` §2.1/§6.3, `lazy-representation.md` §3, `seq-representation.md` §4. The code moves under `clef/docs/fidelity/phg/Closure_Retooling_Plan.md`, and this PRD moves with it; until then the layout sections below describe what the code does, not the design.

> **Surface note (2026-09).** `nativeptr<'T>`, `NativePtr.*`, `voidptr`, and `FSharp.NativeInterop` are not denotable in Clef source (spec `ffi-boundary.md` §1, `special-attributes-and-types.md`; `TNativePtr` is compiler-internal only). Where this PRD shows them, it records the pre-strip surface the code was written against; the settled surfaces are the opaque `Ptr<'T, 'Region, 'Access>` handle in the interior and `CHandle<'T>` at the C boundary, with buffers as bounded arrays and captures as `memref` views.

> **Sample**: `11_Closures` | **Status**: In Progress | **Depends On**: Samples 01-10 (complete)

## 1. Executive Summary

This PRD specifies the implementation of MLKit-style flat closures for Clef Native compilation. Closures are foundational to functional programming — they enable functions to capture variables from enclosing scopes. This feature unlocks higher-order functions (C-02), lazy evaluation (C-05), sequences (C-06/C-07), async (A-01+), and ultimately the MailboxProcessor capstone (T-03+).

**Key Architectural Decisions**:

1. **Capture analysis belongs in CCS**, not Composer. Captures are scope analysis; scope is resolved during type checking. CCS computes captures during PSG construction; Alex handles only SSA assignment and MLIR emission.

2. **Portable memref dialect throughout** — no LLVM dialect. Closures use byte-level `TMemRefStatic(totalBytes, TInt(IntWidth 8))` for struct representation, `pInsertValue`/`pExtractValue` for field access, `pFuncCallIndirect` for code pointer invocation. This keeps closures target-agnostic: CPU → native pointers, FPGA → typed channels, NPU → typed descriptors.

3. **FFI boundary marshaling is explicit** — closures are Clef-internal values. When closure state (function pointers, struct references) must cross the C boundary, extraction to raw pointers happens as the natural residual at the boundary, not as a general-purpose conversion.

## 2. Language Feature Specification

### 2.1 Clef Closure Semantics

A closure is a function bundled with its captured environment:

```fsharp
let makeCounter (arena: byref<Arena<'a>>) (start: int) : (unit -> int) =
    let countPtr = Arena.alloc &arena (Platform.wordSize ())
    NativePtr.write (NativePtr.ofNativeInt<int> countPtr) start
    fun () ->
        let ptr = NativePtr.ofNativeInt<int> countPtr
        let current = NativePtr.read ptr
        let next = current + 1
        NativePtr.write ptr next
        next
```

The inner lambda captures `countPtr` (the arena-allocated storage). Each call to `makeCounter` produces an independent closure with its own storage.

### 2.2 Capture Modes

| Variable Kind | Capture Mode | Environment Entry | Semantics |
|---|---|---|---|
| Immutable `let x = ...` | ByValue | `T` | Copy value into closure struct |
| Mutable `let mutable x = ...` | ByRef | `ptr<T>` | Pointer to original alloca |

**Critical**: Mutable variables MUST be captured by reference to enable mutation through the closure. Multiple closures capturing the same mutable variable share the same storage.

### 2.3 Flat vs Linked Closures

Based on Shao & Appel (1994) "Space-Efficient Closure Representations":

| Property | Flat Closures (MLKit) | Linked Closures |
|---|---|---|
| Memory layout | All captures inline | Pointer to outer env |
| Access time | O(1) direct offset | O(depth) chain walk |
| Space safety | Safe for GC | Keeps outer env alive |
| Creation cost | Copy all captures | Store one pointer |
| Cache behavior | Contiguous | Scattered |

**Fidelity uses FLAT closures** because:
1. No GC — space safety is structural, not runtime
2. Arena allocation — closures live in arenas, not heap
3. Predictable performance — no chain traversal
4. Target portability — flat structs map cleanly to FPGA/NPU

### 2.4 Memory Layout

```
Closure Struct (Flat, byte-level memref)
┌─────────────────────────────────────────────────────────┐
│ code_ptr (TIndex): pointer to lambda implementation     │
├─────────────────────────────────────────────────────────┤
│ capture_0: T₀  (value or pointer depending on mode)     │
│ capture_1: T₁                                           │
│ ...                                                     │
└─────────────────────────────────────────────────────────┘

MLIR representation: memref<N x i8> where N = sum of field sizes
Field access: arith.constant (offset) + memref.load/store
```

For `makeCounter` capturing `countPtr: nativeint`, the returned closure:
```
┌──────────────────────────┬──────────────────────────┐
│ code_ptr (8 bytes)       │ countPtr: nativeint (8b)  │
└──────────────────────────┴──────────────────────────┘
= memref<16 x i8>
```

### 2.5 Arena-Based Lifetime Management

With no runtime/GC, closures that outlive their defining scope need explicit lifetime management:

```fsharp
let main _ =
    let arenaMem = NativePtr.stackalloc<byte> 4096
    let mutable arena = Arena.fromPointer (NativePtr.toNativeInt arenaMem) 4096

    let counter = makeCounter &arena 0   // closure state lives in arena
    counter ()  // 1 — arena still alive, safe
    counter ()  // 2
    0           // arena dies with main's stack frame
```

The arena's lifetime is the caller's stack frame. Factory functions receive the arena and allocate from it. All closure state lives as long as the arena.

## 3. CCS Layer Implementation

### 3.1 Type Definitions

**File**: `src/Compiler/NativeTypedTree/SemanticGraph.fs`

```fsharp
type CaptureInfo = {
    Name: string
    Type: NativeType
    IsMutable: bool
    SourceNodeId: NodeId option
}

type SemanticKind =
    | Lambda of
        parameters: (string * NativeType * NodeId) list *
        body: NodeId *
        captures: CaptureInfo list *
        closureId: ClosureId option *
        isEntryPoint: bool
```

### 3.2 Capture Analysis Algorithm

**File**: `src/Compiler/NativeTypedTree/Expressions/Applications.fs`

CCS computes captures during `checkLambda` via free variable analysis:

1. Create parameter bindings in extended environment
2. Check body expression in extended environment
3. Collect VarRefs in body (recursive traversal)
4. Identify captures: VarRefs not in parameter set, looked up in *outer* environment
5. Build `CaptureInfo` list from outer scope bindings
6. Create Lambda node with captures embedded

**Key insight**: Captures are determined by checking which VarRefs in the body refer to bindings in the *outer* environment (before parameters were added). This is scope analysis, performed once during type checking.

### 3.3 Unit-Parameterized Lambda Types

`fun () -> body` has type `unit -> bodyType`. CCS handles empty parameter lists:
```fsharp
let funcType =
    if List.isEmpty paramTypes then
        NativeType.TFun(env.Globals.UnitType, bodyNode.Type)
    else
        mkFunctionType paramTypes bodyNode.Type
```

### 3.4 Lambda Children Structure

Lambda nodes include parameter PatternBinding NodeIds in Children for proper traversal ordering:
```fsharp
children = paramNodeIds @ [bodyNode.Id]  // Params THEN body
```

The `foldWithSCFRegions` traversal walks parameter PatternBindings before the body region.

## 4. Composer/Alex Layer Implementation

### 4.1 Coeffect Types (Pre-computed Mise-en-Place)

**File**: `src/MiddleEnd/PSGElaboration/Coeffects.fs`

All closure layout information is pre-computed during SSAAssignment — witnesses observe the result, never compute it.

```fsharp
type CaptureMode =
    | ByValue  // Immutable: copy value into closure struct
    | ByRef    // Mutable: store pointer to alloca in closure struct

type CaptureSlot = {
    Name: string
    SlotIndex: int           // 0 = code_ptr, 1+ = captures
    SlotType: MLIRType
    SourceNodeId: NodeId option
    Mode: CaptureMode
}

type LambdaContext =
    | RegularClosure         // captures start at index 1
    | LazyThunk              // captures start at index 3 (after computed, value, code_ptr)
    | SeqGenerator           // captures start at index 3 (after state, current, code_ptr)

type ClosureLayout = {
    LambdaNodeId: NodeId
    Captures: CaptureSlot list

    // FLAT STRUCT CONSTRUCTION SSAs
    CodeAddrSSA: SSA
    ClosureUndefSSA: SSA
    ClosureWithCodeSSA: SSA
    CaptureInsertSSAs: SSA list

    // HEAP ALLOCATION SSAs (arena bump allocator)
    HeapPosPtrSSA: SSA; HeapPosSSA: SSA; HeapBaseSSA: SSA
    HeapResultPtrSSA: SSA; HeapNewPosSSA: SSA
    SizeGepSSA: SSA; SizeSSA: SSA; SizeOneSSA: SSA

    // UNIFORM PAIR CONSTRUCTION SSAs
    PairUndefSSA: SSA; PairWithCodeSSA: SSA; ClosureResultSSA: SSA

    // CAPTURE EXTRACTION SSA (inner function)
    StructLoadSSA: SSA

    // TYPE INFORMATION
    EnvStructType: MLIRType
    ClosureStructType: MLIRType    // memref<N x i8>
    Context: LambdaContext
    LazyStructType: MLIRType option
}
```

### 4.2 SSA Assignment

**File**: `src/MiddleEnd/PSGElaboration/SSAAssignment.fs`

For Lambda nodes with captures, `buildClosureLayout` allocates SSAs for closure CONSTRUCTION in the parent scope. The SSA layout for N captures (total N+3 SSAs):

```
ssas[0]        = addressof code_ptr
ssas[1]        = undef closure struct
ssas[2]        = insertvalue code_ptr at [0]
ssas[3..N+2]   = insertvalue each capture at [1..N]
```

Plus additional SSAs for heap allocation (6), size computation (3), and uniform pair construction (3).

**Two-scope model**:
- **Parent scope**: closure CONSTRUCTION (these SSAs)
- **Child scope (inner function)**: capture EXTRACTION (via `pExtractCaptures` with base index from `LambdaContext`)

### 4.3 Pattern Layer — ClosurePatterns.fs

**File**: `src/MiddleEnd/Alex/Patterns/ClosurePatterns.fs`

Patterns compose Elements into semantic closure operations. All use the `parser { }` CE with XParsec state threading.

#### 4.3.1 Materialized environment construction and invocation

The unused `pFlatClosure` and capture-prepending `ClosurePatterns.pClosureCall`
prototypes were removed during C-07. They computed storage from field counts,
stored a function address in that storage and inferred call arguments in Alex.
They never constituted the native callable contract.

For the bounded materialized callback path, Baker establishes `ClosureValue`,
`EnvironmentCreate`, explicit capture access and applications containing the
actual environment argument. `EnvironmentPatterns` reads the settled layout and
residence; `ApplicationPatterns` realizes the already formed call. The separate
full-pair call path remains in `ApplicationPatterns.pClosureCall`.
See [Closure values as data](../Closure_As_Data.md) for the supported forms,
exact prerequisites and remaining work.

#### 4.3.3 Capture Extraction — `pExtractCaptures`

At inner function entry, extracts captures from the env_ptr (Arg 0):

```fsharp
let pExtractCaptures (baseIndex: int) (captureTypes: MLIRType list)
                     (structType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list>
```

`baseIndex` comes from `closureExtractionBaseIndex` — 1 for regular closures, 3 for lazy/seq.

#### 4.3.4 Allocation residence

The unused global-arena prototype was removed during C-07. It had no admitted
allocation provenance and fixed its position arithmetic to 64 bits. Materialized
environment allocation now requires Baker's residence and placed extent facts;
escaping values require their own storage contract.

#### 4.3.5 Function Definition — `pFunctionDef`

Coeffect-aware function definition wrapping. Observes `TargetPlatform` to elide:
- **CPU** → `func.func` (portable MLIR)
- **FPGA** → `hw.module` (hardware description)

### 4.4 Element Layer

Closure patterns compose from these Elements:

| Element | File | Purpose |
|---|---|---|
| `pUndef` | MLIRAtomics.fs | Create uninitialized memref (closure struct) |
| `pInsertValue` | MLIRAtomics.fs | Store field at index in struct (code_ptr, captures) |
| `pExtractValue` | MLIRAtomics.fs | Load field at index from struct |
| `pLoad` | MemRefElements.fs | Load value from memref |
| `pStore` | MemRefElements.fs | Store value to memref |
| `pAlloca` | MemRefElements.fs | Stack allocate memref |
| `pSubView` | MemRefElements.fs | Compute subview offset |
| `pExtractBasePtr` | MemRefElements.fs | Extract raw pointer from memref (FFI boundary) |
| `pConstI` | MLIRAtomics.fs | Integer constant |
| `pFuncCallIndirect` | FuncElements.fs | Indirect function call via code pointer |
| `pFuncDef` | FuncElements.fs | Function definition |

### 4.5 Witness Layer — LambdaWitness.fs

**File**: `src/MiddleEnd/Alex/Witnesses/LambdaWitness.fs`

Categorized as Lambda nanopass. Uses Y-combinator thunk (`getCombinator`) for recursive self-reference — nested lambdas require the combinator to include itself.

Matches `pLambdaWithCaptures` from PSGCombinators:
- **No captures** → simple `func.func` definition
- **Has captures** → follow the admitted environment representation; materialized callbacks use `EnvironmentWitness` and a separate implementation Lambda

Three Lambda flavors handled:
1. **Entry point** (`DeclRoot.EntryPoint`) → `func.func @main` wrapper
2. **Hardware module** (`DeclRoot.HardwareModule`) → `hw.module`
3. **Non-root** (`None`) → module-level `func.func` with qualified name

## 5. MLIR Output Specification

### 5.1 Closure Struct Type

```mlir
// Closure for makeCounter capturing countPtr (8 bytes code_ptr + 8 bytes capture)
// Represented as: memref<16 x i8>
```

### 5.2 Closure Construction

```mlir
// makeCounter returning a closure
func.func private @makeCounter(%arg0: memref<16xi8>, %arg1: i64) -> memref<16xi8> {
    // ... arena allocation, store start value ...

    // Build closure struct (16 bytes: code_ptr + countPtr)
    %undef = memref.alloca() : memref<16xi8>

    // Insert code_ptr at [0]
    %off0 = arith.constant 0 : index
    memref.store @counter_impl_ptr, %undef[%off0] : memref<16xi8>

    // Insert captured countPtr at [1]
    %off1 = arith.constant 1 : index
    memref.store %countPtr, %undef[%off1] : memref<16xi8>

    func.return %undef : memref<16xi8>
}
```

### 5.3 Closure Implementation Function

```mlir
// The lambda body — receives captures as explicit parameters
func.func private @counter_impl(%countPtr: index) -> i64 {
    // NativePtr.read: load through pointer
    %ptr = ... // ofNativeInt<int> countPtr
    %current = memref.load %ptr[%zero] : memref<1xi64>

    // Increment
    %one = arith.constant 1 : i64
    %next = arith.addi %current, %one : i64

    // NativePtr.write: store through pointer
    memref.store %next, %ptr[%zero] : memref<1xi64>

    func.return %next : i64
}
```

### 5.4 Closure Invocation

```mlir
// counter() — extract code_ptr, extract captures, indirect call
%off0 = arith.constant 0 : index
%code_ptr = memref.load %closure[%off0] : index     // Extract code_ptr from [0]
%off1 = arith.constant 1 : index
%capture0 = memref.load %closure[%off1] : index     // Extract countPtr from [1]

%result = func.call_indirect %code_ptr(%capture0) : (index) -> i64
```

## 6. FFI Boundary Marshaling

### 6.1 The Boundary Problem

Closures are Clef-internal values. C functions know nothing about flat closure structs. When Clef code interacts with C at function pointer boundaries, explicit marshaling is required.

Three boundary scenarios:

| Direction | Clef Side | C Side | Marshaling |
|---|---|---|---|
| Clef → C callback | Flat closure struct | `void (*fn)(void*, ...)` + `void* userdata` | Decompose: trampoline + struct-as-userdata |
| C fn ptr → Clef | Callable value | `nativeint` (raw pointer) | Wrap: zero-capture closure with C fn ptr as code_ptr |
| Clef struct → C | memref struct value | `void*` raw pointer | Extract: `pExtractBasePtr` at boundary |

### 6.2 Clef → C Callback (Pattern A: Registration)

When passing a Clef closure to a C function expecting `void (*callback)(T1, T2, ..., void* userdata)`:

```fsharp
// Clef code: register a handler with a C library
let handler = fun (data: nativeint) ->
    Console.writeln "event received"
wl_display_add_listener display (nativeint &handler) 0n
```

The closure struct must be decomposed at the boundary:
1. **Trampoline function**: A static C-ABI-compatible function that receives `void*`, casts it back to the closure struct, extracts code_ptr and captures, and calls the Clef lambda
2. **Userdata**: Pointer to the closure struct (arena-allocated, lifetime must exceed the registration)

**Implementation**: This is Farscape's Layer 2 responsibility. Farscape generates registration wrappers that take symbol names, resolve via `dlsym`, and pass `nativeint` to L1 extern bindings. The trampoline is a Clef function with C-compatible signature.

### 6.3 C Function Pointer → Clef Callable (Pattern B: dlsym)

When receiving a C function pointer (from `dlsym`, from a listener struct field):

```fsharp
// Farscape L2 generates:
let buildWlPointerListener (enter: nativeint) (leave: nativeint) ... =
    { wl_pointer_listener.enter = enter; leave = leave; ... }
```

These are raw `nativeint` values — C function pointers. They cannot be called as Clef closures directly. They're stored in listener structs and passed back to C via `wl_proxy_add_listener`.

**When Clef code needs to call a C function pointer**: Use `NativePtr.invoke<'T>` intrinsic (future) or explicit extern binding.

### 6.4 Clef Struct → C Raw Pointer

When a Clef struct (like a listener containing function pointers) must be passed to a C function expecting a raw pointer:

```fsharp
// HelloWayland: pass listener struct to C function
wl_proxy_add_listener proxy (nativeint &xdgSurfaceListener) 0n
```

The `nativeint &struct` expression:
1. `&struct` → `AddressOf` node → `pBuildAddressOf` (alloca + store) → `memref<1 x StructType>`
2. `nativeint memref` → type conversion → `pExtractBasePtr` → `memref.extract_aligned_pointer_as_index`

The type conversion from memref to index happens in `pTypeConversion` (`ApplicationPatterns.fs`):
```fsharp
| TMemRef _, TIndex | TMemRefStatic _, TIndex ->
    pExtractBasePtr resultSSA srcSSA srcType
```

### 6.5 FFI Argument Marshaling at ExternCall Boundaries

When `pExternCallResolved` (`PlatformPatterns.fs`) calls a C function, arguments with memref types are automatically marshaled to raw pointers using pre-allocated cast SSAs:

```fsharp
// For each argument:
// - If type is TMemRef/TMemRefStatic → pExtractBasePtr, pass as TIndex
// - Otherwise → pass through unchanged
```

This uses the pre-allocated SSAs from `SSAAssignment.fs` (indices `1..argCount` for non-option, `11..11+argCount` for option return). No runtime SSA allocation.

### 6.6 Design Principle: C Does Not Leak In

The marshaling boundary is ONE-WAY and EXPLICIT:
- **Inside Clef**: Everything is memref, portable, target-agnostic
- **At the boundary**: Extraction/injection is the natural residual of observing a type mismatch
- **C semantics never propagate inward**: No raw pointers inside Clef code, no `void*` semantics, no C calling conventions

This is the same principle as `pSysWrite`: the syscall boundary extracts `memref.extract_aligned_pointer_as_index` + `index.casts i64`, but the string value inside Clef is always a fat pointer (memref + length).

### 6.7 The Boundary Contract as a Joint Constraint

The three scenarios of Section 6.1 read as separate marshaling problems, but they are instances of one constraint with three parties: the Clef type of the crossing value, the C ABI contract it must meet on the far side, and the owner of its lifetime. Each scenario strains a different party. The callback direction (6.2) strains the lifetime owner, because the environment handed to C as userdata must outlive the registration. The function pointer direction (6.3) strains the Clef type, because a raw address must be readmitted as a callable value. The struct direction (6.4) strains the ABI contract, because a memref must present itself as the raw pointer C expects. No two parties determine the third, so the claim each crossing makes is irreducibly joint.

For callback surfaces, the constraint classifies by the lifetime plumbing the C API offers:

| Tier | C Surface Shape | Marshaling Obligation |
|---|---|---|
| A | `userdata` plus destroy hook | Full closure; environment arena-hoisted; the destroy hook releases it; nothing for the application to manage |
| B | `userdata`, no destroy hook | Full closure; registration returns a linear handle whose consumption (disconnect, destroy) releases the environment |
| C | No `userdata` | Closed functions only; the boundary type requires an empty environment |

Tier A is the GLib shape: the destroy hook is C's own acknowledgment that callbacks are closures with lifetimes, and wiring it makes release automatic. Tier B is the Wayland listener shape, where registration, every invocation, and release form one lifetime claim that no single call site can witness, so the registration handle carries the claim as a linear value. Tier C is where the flat representation pays off structurally: a flat closure with an empty environment is a bare code pointer, arity analysis already certifies which lambdas qualify, and the boundary type demands the property instead of assuming it.

The jointness dictates the representation. A boundary contract is one hyperedge, a single edge whose participant set spans all of its sites: the extern declaration, the marshaled arguments, the lifetime owner, and the ABI contract, with tier and byte contract carried in the annotation. Encoding the contract as pairwise edges asserts strictly less, because each binary edge can be discharged in isolation while the joint claim fails; the Tier B lifetime is exactly such a claim, and it is never lowered to binary edges.

The representation also makes the boundary auditable. Lowering resolves the deferred casts that a closure crossing creates, and those cast-resolution statistics reconcile against the boundary obligations the graph records as discharged. A cast with no contract is a crossing the front end never saw; a contract with no cast is an obligation that never reached mechanism. The contract is therefore a decidability condition, not housekeeping: a crossing with enumerated participants, a flat environment of known extent, and a single release site is exactly what keeps the boundary judgment inside the finiteness lemma of `Closure_Nanopass_Architecture.md` Section 4, and the provable region of the computation graph stays closed only while every crossing has that form.

All of this is future work in the same sense as Section 10.8: the trampoline generation deferred there is the mechanism that discharges these contracts, and the contract representation given here is what that mechanism discharges against. Both attach after application-surface lowering completes. The promotion of boundary edges to hyperedges is specified in the PHG addendum to `PSG_Nanopass_Architecture.md`; the saturation leaf the contract anchors to is described in D-01 Section 4.2; the interior forms that arrive at this fence are enumerated in Section 14.

## 7. Lazy and Seq Extensions

### 7.1 Lazy Thunk (C-05)

Lazy values reuse the closure struct layout with additional prefix fields:

```
Lazy Struct: {computed: i1, value: T, code_ptr: ptr, capture₀, capture₁, ...}
             [0]           [1]       [2]            [3...]
```

- `LambdaContext.LazyThunk` → extraction base index = 3
- `pLazyStruct`: builds the struct with `computed=false` initial state
- `pBuildLazyForce`: extracts code_ptr, alloca struct, store, call thunk with pointer

### 7.2 Seq Generator (C-06/C-07)

Sequence iterators use:

```
Seq Struct: {state: i32, current: T, code_ptr: ptr, captures..., internal_state...}
            [0]         [1]         [2]            [3...]
```

- `LambdaContext.SeqGenerator` → extraction base index = 3
- `pSeqStruct`: builds struct with initial state
- `pSeqMoveNext`: extracts state + code_ptr + captures, calls MoveNext
- `pBuildForEachLoop`: (gap — needs SCF.While integration)

## 8. Validation

### 8.1 Sample Code

**File**: `samples/console/FidelityHelloWorld/11_Closures/Closures.fs`

The sample covers:

| Test Case | What It Validates |
|---|---|
| `makeCounter` | Mutable capture via arena, ByRef semantics, independent state |
| `makeGreeter` | Immutable capture (string), ByValue semantics |
| `makeAccumulator` | Mutable capture, arithmetic accumulation |
| `makeRangeChecker` | Multiple immutable captures |
| Independent closures | Two counters from same factory, independent state |

See Section 10 for expanded boundary marshaling test cases.

### 8.2 Expected Output

```
=== Closures Test ===
--- Counter ---
First call: 1
Second call: 2
Third call: 3

--- Greeter ---
Hello, Alice!
Goodbye, Alice!
Welcome, Bob!

--- Accumulator ---
Add 10: 110
Add 25: 135
Add 5: 140

--- Range Checker ---
5 in range 10-20: false
15 in range 10-20: true
25 in range 10-20: false

--- Independent Closures ---
counter1: 1
counter2: 101
counter1: 2
counter2: 102
```

### 8.3 Regression Tests

ALL samples 01-10 must continue to pass after closure implementation.

## 9. Implementation Status

### Phase 1: CCS Changes — COMPLETE
- [x] `CaptureInfo` type in SemanticGraph.fs
- [x] `SemanticKind.Lambda` updated with captures field
- [x] `collectVarRefs` free variable analysis
- [x] Capture analysis in `checkLambda`
- [x] Unit-parameterized lambda types
- [x] Lambda Children include parameter PatternBindings
- [x] `foldWithSCFRegions` traversal updated
- [x] All Lambda pattern matches updated

### Phase 2: Composer Infrastructure — COMPLETE
- [x] `CaptureMode`, `CaptureSlot`, `ClosureLayout` coeffect types (Coeffects.fs)
- [x] `LambdaContext` DU (RegularClosure, LazyThunk, SeqGenerator)
- [x] `buildClosureLayout` in SSAAssignment.fs
- [x] `lookupClosureLayout` / `hasClosure` coeffect lookups
- [x] Lambda pattern matches updated in PSGCombinators, CCSTransfer
- [x] Samples 01-10 verified

### Phase 3: Closure Patterns — COMPLETE
- Retired unused closure-construction prototype; see §4.3.1.
- Retired unused capture-prepending invocation prototype; see §4.3.1.
- [x] `pExtractCaptures` — environment extraction at function entry
- Retired unused global-arena prototype; see §4.3.4.
- [x] `pFunctionDef` — coeffect-aware (CPU → func.func, FPGA → hw.module)
- [x] `pLazyStruct` / `pBuildLazyForce` — lazy thunk patterns
- [x] `pSeqStruct` / `pSeqMoveNext` — seq generator patterns

### Phase 4: LambdaWitness Integration — IN PROGRESS
- [x] Entry point Lambda handling
- [x] Non-root Lambda handling (qualified names)
- [x] Parameter PatternBinding visiting
- [x] Y-combinator recursive self-reference
- Materialized callback construction uses `EnvironmentWitness`; coverage is recorded in C-07.
- Materialized callback applications carry explicit environment operands from Baker; general full-pair coverage remains separate.
- [ ] Nested lambda / returning function values

### Phase 5: FFI Boundary Marshaling — IN PROGRESS
- [x] `pTypeConversion` TMemRef → TIndex case (ApplicationPatterns.fs)
- [x] `pExternCallResolved` argument marshaling (PlatformPatterns.fs)
- [ ] AddressOf + type conversion PSG connection (CCS enrichment investigation)
- [ ] End-to-end verification with HelloWayland

### Phase 6: ForEach / MoveNext — PENDING
- [ ] SCF.While integration for `pBuildForEachLoop`
- [ ] MoveNext calling convention implementation

### Phase 7: Validation — PENDING
- [ ] Sample 11 compiles without errors
- [ ] Sample 11 produces correct output
- [ ] Expanded boundary tests pass (Section 10)
- [ ] All regression tests pass (samples 01-10)

## 10. Expanded Sample 11 — Boundary Marshaling Coverage

Sample 11 should be expanded to cover the full feature area, including the boundary edge cases that are literally at the edge between Clef's portable world and C's raw pointer world.

### 10.1 Core Closure Cases (existing)

```fsharp
// Already covered:
// - makeCounter: mutable capture via arena, ByRef
// - makeGreeter: immutable capture (string), ByValue
// - makeAccumulator: mutable capture, accumulation
// - makeRangeChecker: multiple immutable captures
// - Independent closures from same factory
```

### 10.2 Zero-Capture Closures

```fsharp
/// Lambda with no captures — pure function value
/// Should be a minimal closure struct (code_ptr only, no env)
let applyOp (f: int -> int -> int) (a: int) (b: int) : int =
    f a b

// Usage: applyOp (fun a b -> a + b) 3 4  // 7
```

### 10.3 Multi-Type Captures

```fsharp
/// Captures of different types: int, string, bool
let makeFormatter (prefix: string) (width: int) (showSign: bool) : (int -> string) =
    fun value ->
        let sign = if showSign && value > 0 then "+" else ""
        $"{prefix}{sign}{Format.int value}"
```

### 10.4 Nested Closures (Closure Returning Closure)

```fsharp
/// Outer closure captures 'base', returns inner closure that captures 'multiplier'
/// Tests nested flat closure construction — each level has its own struct
let makeScaledAdder (base': int) : (int -> (int -> int)) =
    fun multiplier ->
        fun x -> base' + multiplier * x
```

### 10.5 Closure as Function Argument (HOF Bridge)

```fsharp
/// Pass closure to a higher-order function
/// Tests that closure struct flows correctly as parameter
let twice (f: int -> int) (x: int) : int =
    f (f x)

// Usage: twice (makeAdder 10) 5  // 25
```

### 10.6 Closure Over Arena-Allocated Struct

```fsharp
/// Capture a pointer to an arena-allocated struct
/// Tests that struct references survive in closure environment
let makePointMover (arena: byref<Arena<'a>>) (x: int) (y: int) : (int -> int -> string) =
    let xPtr = Arena.alloc &arena (Platform.wordSize ())
    let yPtr = Arena.alloc &arena (Platform.wordSize ())
    NativePtr.write (NativePtr.ofNativeInt<int> xPtr) x
    NativePtr.write (NativePtr.ofNativeInt<int> yPtr) y
    fun dx dy ->
        let newX = NativePtr.read (NativePtr.ofNativeInt<int> xPtr) + dx
        let newY = NativePtr.read (NativePtr.ofNativeInt<int> yPtr) + dy
        NativePtr.write (NativePtr.ofNativeInt<int> xPtr) newX
        NativePtr.write (NativePtr.ofNativeInt<int> yPtr) newY
        $"({Format.int newX}, {Format.int newY})"
```

### 10.7 Boundary: Struct Reference to C (the HelloWayland pattern)

```fsharp
/// Build a struct of function pointers (simulating a C listener)
/// Then pass its address as nativeint to an extern function
/// This is the exact pattern that HelloWayland uses

// Simulated C struct (mirrors wl_*_listener pattern)
type EventHandler = {
    on_enter: nativeint    // C function pointer
    on_leave: nativeint
}

// Simulated extern (would be [<FidelityExtern>] in real code)
[<FidelityExtern>]
let register_handler (target: nativeint) (handler: nativeint) (data: nativeint) : int64 =
    Unchecked.defaultof<int64>

let testBoundaryMarshal () =
    let handler = { on_enter = 0n; on_leave = 0n }
    // nativeint &handler — AddressOf produces memref, nativeint extracts pointer
    let result = register_handler 0n (nativeint &handler) 0n
    Console.writeln $"register result: {Format.int64 result}"
```

### 10.8 Boundary: Closure State Pointer to C (future — trampoline)

```fsharp
/// Pass closure's capture environment to C as void* userdata
/// This requires decomposing the closure at the boundary:
///   1. Extract code_ptr → wrap in C-ABI trampoline
///   2. Extract env_ptr → pass as void* userdata
///
/// NOTE: This is future work — requires trampoline generation.
/// Documenting the pattern here so the PRD covers the full edge.
/// The interior form that reaches this boundary is specified in Section 14.
```

## 11. Files to Create/Modify

### 11.1 CCS — COMPLETE

| File | Status | Purpose |
|---|---|---|
| SemanticGraph.fs | DONE | CaptureInfo, Lambda with captures |
| Applications.fs | DONE | collectVarRefs, capture analysis |
| Bindings.fs | DONE | Lambda creation with Children |
| SemanticGraph.fs traversal | DONE | foldWithSCFRegions Lambda region |

### 11.2 Composer — IN PROGRESS

| File | Status | Purpose |
|---|---|---|
| Coeffects.fs | DONE | CaptureMode, CaptureSlot, ClosureLayout, LambdaContext |
| SSAAssignment.fs | DONE | buildClosureLayout, closure SSA allocation |
| ClosurePatterns.fs | DONE | Function definition and legacy capture/lazy patterns; materialized callback environments are in EnvironmentPatterns |
| LambdaWitness.fs | PARTIAL | Entry/non-root handling done; closure emission pending |
| ApplicationPatterns.fs | DONE | pTypeConversion TMemRef→TIndex |
| PlatformPatterns.fs | DONE | pExternCallResolved argument marshaling |
| PSGCombinators.fs | DONE | pLambdaWithCaptures |

## 12. Academic References

1. Shao & Appel (1994), "Space-Efficient Closure Representations"
2. Tofte & Talpin (1997), "Region-Based Memory Management"
3. MLKit Programming with Regions (Elsman, 2021)
4. Perconti & Ahmed (2019), "Closure Conversion is Safe for Space"

## 13. Related PRDs

- **C-02**: Higher-Order Functions — closures as values passed to/from functions
- **C-05**: Lazy Evaluation — reuses closure struct layout with prefix fields
- **C-06/C-07**: Sequences — reuses closure struct for MoveNext state machine
- **A-01 to A-03**: Async — uses closures for continuation callbacks
- **T-03 to T-05**: MailboxProcessor — synthesizes closures + async + threading

## 14. The Closure Saturation Form Family

This section specifies the saturated representation of environments as a family of forms. Each form is selected by a condition that is decidable at saturation. Each interior form is expressible in standard MLIR dialects. Each materializing form carries verification conditions stated as formulas over literals, dischargeable by SMT proof dispatch. Status is stated plainly at each point: the two-pass capture and layout machinery of Section 4 and the sample corpus of Sections 8 and 10 are the present state; the saturation recipes, the PSG proof dispatch, and the smt-dialect re-check are design. Section 14.7 itemizes which is which.

### 14.1 The Environment as the General Object

Fan-out lowers a capturing lambda to five kinds of PSG structure:

| Structure | Content |
|---|---|
| Code reference | The symbol of the implementation function |
| Capture nodes | One per captured binding; each carries a mode, ByValue or ByRef, read from the mutability coeffect (Section 2.2) |
| Environment node | Ordered; its frontier is its child list; field order is child order |
| Escape-class edge | Places the environment in the escape order: local, downward, or region-escaping |
| Release site | The node at which the environment's storage is released |

The environment node is the general object. Three instances share it:

1. **Closure environment**: the captures of a lambda.
2. **DCont continuation frame**: the captures are the variables live across a suspension point.
3. **Actor state cell**: the state record an actor carries between message receipts.

One fold-in rule serves all three. The rule reads the capture modes, the escape class, and the field sizes, then selects a form from the family in Section 14.2. Nothing in the rule is specific to lambdas.

### 14.2 The Form Family and Its Selection at Saturation

| # | Form | Saturation condition | Representation |
|---|---|---|---|
| 1 | Vacant | Capture set is empty | Bare code reference; no environment; the arity gate's case, and the form Section 6.7 Tier C demands |
| 2 | Unmaterialized | Callee known at every application site; no escape | Captures dissolve to SSA arguments; zero bytes |
| 3 | Stack flat | All captures ByValue; escape is downward only | Byte buffer of literal extent on the stack |
| 4 | Mixed ByRef | At least one capture is ByRef | Mutable captures held as region-scoped references |
| 5 | Region escaping | Environment is returned or stored | Materialization in the target region's arena; release at region end |
| 6 | Large capture | A ByValue field exceeds the target-parameterized size class | Per-field policy; see below |
| 7 | Descriptor shared | Environment crosses a memory fabric | BAREWire descriptor; fields region-scoped by construction |

Forms 1 and 2 are terminal: they materialize nothing. Forms 4, 5, and 6 refine form 3 and compose with each other; a single environment can be Mixed ByRef and Region escaping at once.

**Large capture policy.** The decision is per field. A field below the size class is copied. A field above it is demoted to a region-scoped reference. The size class is a target parameter, so the policy is inference with zero annotation in the common case. A binding-level attribute overrides the inference per binding. Every override is checked, and an override that would introduce chain-scoped sharing is rejected.

**Descriptor shared status.** This form is specified now and implemented at the memory-fabrics horizon. It is design. Its admissibility argument is structural: BAREWire descriptor fields are region-scoped by construction.

**Fence packing is not a form.** Packing a closure into words for a C crossing is a boundary event under the Section 6.7 contract. It occurs only at the fence and never defines an interior representation.

**Safe for space bounds the family.** Following Shao & Appel (1994) and Perconti & Ahmed (2019) (Section 12), a form is admissible only if the environment's reachable set equals its field list and its lifetime equals its region bounds. Chain-scoped sharing, a field that retains an enclosing environment, is inadmissible. This restates the flat-closure decision of Section 2.3 as an admissibility bound on the whole family.

### 14.3 Full Expression in Standard MLIR Primitives

Every interior form in Section 14.2 is expressible with the `func`, `memref`, and `arith` dialects only. No LLVM dialect, no pointer type, no custom dialect. This subsection demonstrates the claim primitive by primitive.

**Code value.** `func.constant @f : (T...) -> R` produces a function-typed SSA value; the verifier requires that `@f` resolve to a `func.func` of that exact type. Known-callee application sites use direct `func.call @f(...)`; the arity gate certifies these sites. Unknown-callee application uses `func.call_indirect %fn(%args)`. All three operations are standard `func` dialect.

**Materialized heterogeneous environment.** The environment is a 1-D byte buffer of literal extent plus one typed view per field. The MLIR memref dialect documentation defines the primitive: "The 'view' operation extracts an N-D contiguous memref with empty layout map with arbitrary element type from a 1-D contiguous memref with empty layout map of i8 element type," and it requires "a single dynamic byte-shift operand" that shifts the base pointer to produce "the resulting contiguous memref view with identity layout." The primitive's own precondition is a 1-D i8 source with identity layout and a byte shift. That is exactly the saturated environment's shape: byte extent literal in the type, field offsets literal, materialized as `arith.constant` byte shifts. The standard dialect already contains the elaborated form; nothing needs to be invented.

```mlir
%env = memref.alloca() : memref<8xi8>
%c0  = arith.constant 0 : index
%f0  = memref.view %env[%c0][] : memref<8xi8> to memref<1xi32>
```

The result of `memref.view` has identity layout, so an environment view carved from an arena buffer is itself a valid base for field views. The construction is closed under the primitive's own precondition.

**ByRef fields.** A reference field is a typed view into the region's arena buffer: `memref.get_global` fetches the buffer, and `memref.view` at the hoisted cell's offset yields the typed cell. A reference is a static symbol plus a byte offset. Provenance is static. No pointer type appears anywhere in the interior.

**The closure value.** A closure value is two SSA values traveling together: the function value and the environment buffer. They are passed and returned as ordinary multiple values. No packing, no casts, no new dialect. Packing into words happens only at the FFI fence, under the Section 6.7 boundary contract. The two-pass machinery of Section 4 currently extracts captures and prepends them as explicit arguments (Section 4.3.2); the pair convention here is the saturated target of the recipe migration (Section 14.7). The fence contract is identical under both conventions.

**Correspondence.**

| Saturated PSG structure | Standard MLIR construct | What the verifier checks structurally |
|---|---|---|
| Code reference | `func.constant @f : (T...) -> R` | Symbol resolves to a `func.func` of the stated type |
| Known-callee application | `func.call @f(...)` | Operand and result types match the callee signature |
| Unknown-callee application | `func.call_indirect %fn(...)` | Operand and result types match the function type of `%fn` |
| Environment node, extent E | `memref.alloca() : memref<Exi8>` or an arena view | E is a literal in the type |
| Field (off, size, T) | `memref.view %env[%off][] : memref<Exi8> to memref<1xT>` | Source is 1-D i8 with identity layout; result is typed with identity layout |
| ByRef referent | `memref.get_global @r` plus `memref.view` at the cell offset | Symbol resolves to a `memref.global` of the stated type |
| Closure value | Two SSA values `(%fn, %env)` in multi-value signatures | Function type agreement at definitions, calls, and returns |
| Release site | Region end; not an MLIR construct | Checked in the PSG, not by the MLIR verifier |

The MLIR verifier checks types and layout preconditions. It cannot bound a byte-shift operand against the source extent, because the operand is dynamic in the op's definition. The verification conditions of Section 14.4 close exactly that gap, over the same literals.

### 14.4 Verification Conditions per Form

For a materializing form with fields f_0 .. f_(n-1) in frontier order, offsets off_i, sizes size_i, padding pad, and extent E, all literals at saturation:

| VC | Formula | Fragment | Discharge |
|---|---|---|---|
| VC-EXT, extent bound | size_0 + ... + size_(n-1) + pad = E, with E literal | QF_LIA | Ground arithmetic over literals |
| VC-DIS, disjointness and coverage | for all i < j: off_i + size_i <= off_j; and off_(n-1) + size_(n-1) <= E | QF_LIA | Finite conjunction over literals |
| VC-REG, region ordering | for each reference field f_i with referent region r_i: lifetime(r_i) >= lifetime(r_env) | None; lattice | Order check in the escape order; no solver |
| VC-REL, single release | count(releaseSites(env)) = 1 | None; graph | Graph check |
| VC-APP, application type agreement | for every application site s: typeof(callee_s) = the carried function type | EUF, structural | Congruence check; no quantifiers |

The discharge point is fixed: at saturation, over the PSG, before witnessing. Each form discharges its applicable subset. Vacant and Unmaterialized discharge VC-APP only. Stack flat adds VC-EXT and VC-DIS. Mixed ByRef adds VC-REG. Region escaping adds VC-REL. Large capture reruns VC-EXT and VC-DIS after per-field demotion. Descriptor shared satisfies VC-REG by construction.

### 14.5 The Two-Sided Check: PSG Dispatch and the MLIR smt Dialect

**Status: design.** This subsection follows the standing plan for SMT annotations: each obligation generates its own PSG node with dependency edges to the structures it constrains, and dispatch runs over those nodes. The MLIR side is contingent on the pending MLIR/CIRCT smt dialect scoping study.

The check has two sides:

1. **PSG side, at saturation.** The Section 14.4 conditions are dispatched over PSG literals before witnessing. This is the discharge of record.
2. **MLIR side, after witnessing.** The same layout formulas are re-read off the witnessed MLIR. The literals are already in the artifact: E is the static memref shape, off_i is the `arith.constant` operand of each view, size_i is each view's result type. The formulas are encoded as smt dialect operations (`smt.declare_fun`, `smt.assert`, `smt.check` over `!smt.int` or bitvectors) and re-dispatched. The witnessed artifact is therefore re-checkable without trusting the emitter.

Encoding discipline: declare a symbol per derived quantity, assert its definition as read from the artifact, assert the negation of the conjoined conditions, and expect unsat. The smt module is a verification artifact beside the program, not part of it. Section 14.6 lists the concrete encoding for the worked example.

Audit consequence: the check is per obligation, per site, machine-checkable. Every obligation node names its site and its formulas; the MLIR re-check re-derives the same formulas from the artifact. The reconciliation counters of the interim cast-resolution pass (Section 6.7) are superseded by exact correspondence: obligations and assertions match one for one, and any mismatch identifies its site.

### 14.6 Worked Example: The Counter End to End

The counter of Sections 2.1 and 8.1: one mutable int capture, and the closure escapes to the caller. The manual arena plumbing in Section 2.1 is the hand-written form of what the region machinery performs here. This trace is the design target of the recipe migration; the literals below are the saturated output for one instance.

```fsharp
let makeCounter (start: int) : (unit -> int) =
    let mutable count = start
    fun () ->
        count <- count + 1
        count
```

**Fan-out.** Code reference `@counter_impl`. One capture node for `count`. One ordered environment node `env0` with frontier `[count]`. One escape-class edge: region-escaping, target region `r0`. One release site: the end of `r0`.

**Coeffects.** `count` is mutable, so the mode is ByRef. The lambda is returned, so the class is escaping. The target region is `r0`, the caller's arena.

**Fold-in selection.** The capture set is nonempty: not Vacant. The closure is applied through a returned value, so the callee is unknown at the application site: not Unmaterialized. A ByRef capture is present: Mixed ByRef. The environment escapes: Region escaping. Selected form: Mixed ByRef composed with Region escaping.

**Layout, all literals.** Region `r0` has extent 16. The hoisted cell `c0` (i64, 8 bytes) sits at region offset 0. The environment `env0` sits at region offset 8 with extent E = 8. The environment has one field `f0` at off_0 = 0, size_0 = 8, type i64. Its content is the byte offset of `c0` inside `r0`, which is 0 for this instance; the offset is data because instances differ, while the buffer symbol is static. pad = 0. Unit erases at the interior, so the implementation takes only the environment.

**Witnessed listing, standard dialects only.**

```mlir
memref.global "private" @r0 : memref<16xi8> = uninitialized

func.func private @counter_impl(%env: memref<8xi8>) -> i64 {
  %z     = arith.constant 0 : index
  %f0    = memref.view %env[%z][] : memref<8xi8> to memref<1xi64>      // field f0: off_0 = 0, size_0 = 8
  %off   = memref.load %f0[%z] : memref<1xi64>                         // byte offset of c0 in r0
  %offx  = arith.index_cast %off : i64 to index
  %arena = memref.get_global @r0 : memref<16xi8>
  %cell  = memref.view %arena[%offx][] : memref<16xi8> to memref<1xi64>
  %cur   = memref.load %cell[%z] : memref<1xi64>
  %one   = arith.constant 1 : i64
  %next  = arith.addi %cur, %one : i64
  memref.store %next, %cell[%z] : memref<1xi64>
  func.return %next : i64
}

func.func private @makeCounter(%start: i64) -> ((memref<8xi8>) -> i64, memref<8xi8>) {
  %arena = memref.get_global @r0 : memref<16xi8>
  %z     = arith.constant 0 : index
  %cell  = memref.view %arena[%z][] : memref<16xi8> to memref<1xi64>   // c0 at region offset 0
  memref.store %start, %cell[%z] : memref<1xi64>
  %c8    = arith.constant 8 : index
  %env   = memref.view %arena[%c8][] : memref<16xi8> to memref<8xi8>   // env0 at region offset 8, E = 8
  %f0    = memref.view %env[%z][] : memref<8xi8> to memref<1xi64>      // f0 at off_0 = 0
  %ref   = arith.constant 0 : i64                                      // c0's byte offset in r0
  memref.store %ref, %f0[%z] : memref<1xi64>
  %fn    = func.constant @counter_impl : (memref<8xi8>) -> i64
  func.return %fn, %env : (memref<8xi8>) -> i64, memref<8xi8>
}
```

Application site, both callee kinds:

```mlir
%fn, %env = func.call @makeCounter(%start) : (i64) -> ((memref<8xi8>) -> i64, memref<8xi8>)
%r1 = func.call_indirect %fn(%env) : (memref<8xi8>) -> i64
```

**The VC set instantiated.**

- VC-EXT: size_0 + pad = E instantiates to 8 + 0 = 8. Holds.
- VC-DIS: n = 1, so the pairwise clause is vacuous; coverage instantiates to off_0 + size_0 <= E, that is 0 + 8 <= 8. Holds.
- Region fit, the same formula family one level up: 0 + 8 <= 16 for `c0` and 8 + 8 <= 16 for `env0`. Holds.
- VC-REG: lifetime(r0) >= lifetime(r0). Holds reflexively; the cell and the environment share one region.
- VC-REL: releaseSites(env0) = { end of r0 }; count = 1. Holds.
- VC-APP: the carried type is (memref<8xi8>) -> i64; the callee type at the one application site is (memref<8xi8>) -> i64. Equal.

Every literal in these formulas appears in the listing: 8 from `memref<8xi8>`, 16 from `memref<16xi8>`, 0 from `%z`, 8 from `%c8`, and size_0 = 8 from `memref<1xi64>`.

**smt-dialect re-check of the two layout formulas (design).** Read off the witnessed listing: E = 8 from the type of `%env`, off_0 = 0 from the byte shift of the view producing `%f0`, size_0 = 8 from that view's result type `memref<1xi64>`. Validity is checked by asserting the negation and expecting unsat.

```mlir
smt.solver() : () -> () {
  %E     = smt.declare_fun "E" : !smt.int
  %c8    = smt.int.constant 8
  %defE  = smt.eq %E, %c8 : !smt.int          // E = 8, read from memref<8xi8>
  smt.assert %defE
  %off0  = smt.int.constant 0                 // byte shift of the f0 view
  %sz0   = smt.int.constant 8                 // extent of memref<1xi64>
  %pad   = smt.int.constant 0
  %sizes = smt.int.add %sz0, %pad
  %vc1   = smt.eq %sizes, %E : !smt.int       // VC-EXT: size_0 + pad = E
  %end0  = smt.int.add %off0, %sz0
  %vc2   = smt.int.cmp le %end0, %E           // VC-DIS: off_0 + size_0 <= E
  %both  = smt.and %vc1, %vc2
  %neg   = smt.not %both
  smt.assert %neg                             // assert the negation; expect unsat
  smt.check sat { smt.yield } unknown { smt.yield } unsat { smt.yield }
  smt.yield
}
```

`%vc1` encodes VC-EXT and `%vc2` encodes VC-DIS coverage, instantiated with the artifact's own literals. The fragment is QF_LIA over `!smt.int`. An equivalent encoding over `!smt.bv<64>` uses the `smt.bv` operations; every term here is a small nonnegative literal, so the two encodings agree.

### 14.7 Status and Sequencing

Plain accounting. The present state is the two-pass machinery and the sample corpus. Everything else in this section is design.

| Item | Status | Basis |
|---|---|---|
| Two-pass capture analysis and closure layout in Alex (Sections 4.1, 4.2, 4.3) | DONE | Present state; Phases 2 and 3 of Section 9 |
| Witness integration and validation samples (Sections 8 and 10) | IN PROGRESS | Present state; tracked in Section 9 Phases 4 and 7 |
| Recipe migration: closure elaboration and saturation as Baker recipes | PENDING | Design; this section is its specification |
| Form selection (Section 14.2) and VC dispatch (Section 14.4) at saturation | PENDING | Design; part of the recipe migration |
| smt-dialect re-check of witnessed MLIR (Sections 14.5, 14.6) | PENDING | Design; contingent on the MLIR/CIRCT smt dialect scoping study |
| Descriptor shared form (Section 14.2, form 7) | PENDING | Design; memory-fabrics horizon |

The recipe migration is the work item:

- [ ] Move closure elaboration and saturation into Baker's recipes, ending the closure exception
- [ ] Select forms per Section 14.2 at saturation
- [ ] Dispatch the Section 14.4 conditions over the PSG before witnessing
- [ ] Witness the pair convention of Section 14.3 from saturated forms

Sequencing: the smt re-check follows the scoping study, and the descriptor form follows the memory-fabrics work. Packing at the fence stays governed by the boundary contract of Section 6.7, and the trampoline generation deferred in Section 10.8 is the mechanism that packs these forms at that fence.
