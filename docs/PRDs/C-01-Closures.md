# C-01: MLKit-Style Flat Closures

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

#### 4.3.1 Closure Construction — `pFlatClosure`

Builds closure struct by inserting code_ptr at index [0], then captures at indices [1..N]:

```fsharp
let pFlatClosure (codePtr: SSA) (codePtrTy: MLIRType) (captures: Val list)
                 (ssas: SSA list) : PSGParser<MLIROp list>
```

SSA layout: `[0]` = undef, `[1-2]` = insert code_ptr (offset, result), then for each capture: `[3+2*i]` = offsetSSA, `[4+2*i]` = resultSSA.

The closure struct is `TMemRefStatic(totalBytes, TInt(IntWidth 8))` — a byte-level memref whose total size is the sum of all field sizes.

#### 4.3.2 Closure Invocation — `pClosureCall`

Extracts code_ptr from [0], captures from [1..N], then emits indirect function call with captures prepended to explicit args:

```fsharp
let pClosureCall (closureSSA: SSA) (closureTy: MLIRType) (captureTypes: MLIRType list)
                 (args: Val list) (extractSSAs: SSA list) (resultSSA: SSA)
                 : PSGParser<MLIROp list>
```

Calling convention: `call code_ptr(capture_0, capture_1, ..., explicit_arg_0, explicit_arg_1, ...)`

Captures are explicit parameters in the generated code — not passed via context/userdata. This is the Clef calling convention, not the C calling convention.

#### 4.3.3 Capture Extraction — `pExtractCaptures`

At inner function entry, extracts captures from the env_ptr (Arg 0):

```fsharp
let pExtractCaptures (baseIndex: int) (captureTypes: MLIRType list)
                     (structType: MLIRType) (ssas: SSA list) : PSGParser<MLIROp list>
```

`baseIndex` comes from `closureExtractionBaseIndex` — 1 for regular closures, 3 for lazy/seq.

#### 4.3.4 Arena Allocation — `pAllocateInArena`

Bump allocator for closures that escape their defining scope:

```fsharp
let pAllocateInArena (sizeSSA: SSA) (ssas: SSA list) : PSGParser<MLIROp list * SSA>
```

SSA layout: `[0]` = heap_pos_ptr, `[1]` = heap_pos, `[2]` = heap_base, `[3]` = result_ptr, `[4]` = new_pos, `[5]` = index.

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
- **Has captures** → emit flat closure via `pFlatClosure`

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
- [x] `pFlatClosure` — struct construction via insertvalue chain
- [x] `pClosureCall` — extract + indirect call with captures prepended
- [x] `pExtractCaptures` — environment extraction at function entry
- [x] `pAllocateInArena` — heap arena bump allocation
- [x] `pFunctionDef` — coeffect-aware (CPU → func.func, FPGA → hw.module)
- [x] `pLazyStruct` / `pBuildLazyForce` — lazy thunk patterns
- [x] `pSeqStruct` / `pSeqMoveNext` — seq generator patterns

### Phase 4: LambdaWitness Integration — IN PROGRESS
- [x] Entry point Lambda handling
- [x] Non-root Lambda handling (qualified names)
- [x] Parameter PatternBinding visiting
- [x] Y-combinator recursive self-reference
- [ ] Closure construction emission (connecting `pFlatClosure` to witness)
- [ ] Closure invocation emission (connecting `pClosureCall` to witness)
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
| ClosurePatterns.fs | DONE | pFlatClosure, pClosureCall, pExtractCaptures, arena, lazy, seq |
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
