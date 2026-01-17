# PRD-16: Sequence Operations

> **Sample**: `16_SeqOperations` | **Status**: Planned | **Depends On**: PRD-15 (SimpleSeq), PRD-12 (HOFs)

## 1. Executive Summary

This PRD covers the core sequence operations: `Seq.map`, `Seq.filter`, `Seq.take`, `Seq.fold`, etc. These are higher-order functions over sequences - they transform or consume lazy enumerations.

**Key Insight**: Seq operations create **composed flat closures**. `Seq.map f xs` produces a new seq struct that contains:
1. The original sequence (inlined, not by pointer)
2. The mapper closure (flat closure, no `env_ptr`)
3. Its own state machine fields

**Builds on PRD-15**: Seq operations wrap inner sequences. The wrapper is itself a flat closure with state machine fields. The inner sequence and transformation closure are inlined captures.

## 2. Language Feature Specification

### 2.1 Seq.map

```fsharp
let doubled = Seq.map (fun x -> x * 2) numbers
```

Transforms each element lazily.

### 2.2 Seq.filter

```fsharp
let evens = Seq.filter (fun x -> x % 2 = 0) numbers
```

Yields only elements matching predicate.

### 2.3 Seq.take

```fsharp
let first5 = Seq.take 5 infiniteSeq
```

Yields at most N elements.

### 2.4 Seq.fold

```fsharp
let sum = Seq.fold (fun acc x -> acc + x) 0 numbers
```

Eager reduction to single value (consumes sequence).

### 2.5 Seq.collect (flatMap)

```fsharp
let flattened = Seq.collect (fun x -> seq { yield x; yield x * 10 }) numbers
```

Maps then flattens.

## 3. Architectural Principles

### 3.1 Composed Flat Closures (No `env_ptr`)

Seq operations create wrapper sequences. Following the flat closure model:

```
MapSeq<A,B> = {state: i32, current: B, moveNext_ptr: ptr, inner_seq: Seq<A>, mapper: (A -> B)}
```

Both `inner_seq` and `mapper` are **inlined** (flat), not stored by pointer.

| Field | Type | Purpose |
|-------|------|---------|
| `state` | `i32` | Wrapper's own state |
| `current` | `B` | Current transformed value |
| `moveNext_ptr` | `ptr` | Wrapper's MoveNext function |
| `inner_seq` | `Seq<A>` | Inlined inner sequence struct |
| `mapper` | `(A -> B)` | Inlined mapper closure |

**No `env_ptr` anywhere.** The mapper closure itself is flat: `{code_ptr, cap₀, cap₁, ...}`.

### 3.2 Struct Layout Examples

**Seq.map** (mapper with no captures):
```
{state: i32, current: i32, moveNext_ptr: ptr,
 inner: {inner_state: i32, inner_current: i32, inner_moveNext_ptr: ptr},
 mapper: {mapper_code_ptr: ptr}}
```

**Seq.map** (mapper captures `factor`):
```
{state: i32, current: i32, moveNext_ptr: ptr,
 inner: {inner_state: i32, inner_current: i32, inner_moveNext_ptr: ptr},
 mapper: {mapper_code_ptr: ptr, factor: i32}}
```

**Seq.filter**:
```
{state: i32, current: i32, moveNext_ptr: ptr,
 inner: {inner_state: i32, inner_current: i32, inner_moveNext_ptr: ptr},
 predicate: {pred_code_ptr: ptr, ...captures...}}
```

**Seq.take**:
```
{state: i32, current: i32, moveNext_ptr: ptr,
 inner: {inner_state: i32, inner_current: i32, inner_moveNext_ptr: ptr},
 remaining: i32}
```

### 3.3 The Composition Challenge

When sequences are composed (e.g., `Seq.take 5 (Seq.map f (Seq.filter p xs))`), the struct grows:

```
TakeSeq {
    state, current, moveNext_ptr,
    inner: MapSeq {
        state, current, moveNext_ptr,
        inner: FilterSeq {
            state, current, moveNext_ptr,
            inner: OriginalSeq { ... },
            predicate: {...}
        },
        mapper: {...}
    },
    remaining: i32
}
```

This is a **compile-time known** nested struct. No heap allocation. Each composition adds to the struct size, but the exact size is known at compile time.

### 3.4 Why Inline Instead of Pointer?

**Flat closure philosophy**: Self-contained, no indirection.

If we stored inner sequences by pointer:
- Need arena/heap allocation for the inner sequence
- Add indirection (cache misses)
- Lifetime management complexity

With inlining:
- Single contiguous struct
- Stack allocation works
- Predictable memory layout
- No lifetime issues (everything has same lifetime)

**Trade-off**: Struct size grows with composition depth. For typical pipeline depths (3-5), this is acceptable. Very deep pipelines would benefit from alternative strategies (future optimization).

## 4. FNCS Layer Implementation

### 4.1 Seq Module Intrinsics

**File**: `~/repos/fsnative/src/Compiler/Checking.Native/CheckExpressions.fs`

```fsharp
// Seq intrinsic module
| "Seq.map" ->
    // ('a -> 'b) -> seq<'a> -> seq<'b>
    let aVar = freshTypeVar ()
    let bVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TFun(aVar, bVar),
        NativeType.TFun(NativeType.TSeq(aVar), NativeType.TSeq(bVar)))

| "Seq.filter" ->
    // ('a -> bool) -> seq<'a> -> seq<'a>
    let aVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TFun(aVar, env.Globals.BoolType),
        NativeType.TFun(NativeType.TSeq(aVar), NativeType.TSeq(aVar)))

| "Seq.take" ->
    // int -> seq<'a> -> seq<'a>
    let aVar = freshTypeVar ()
    NativeType.TFun(
        env.Globals.IntType,
        NativeType.TFun(NativeType.TSeq(aVar), NativeType.TSeq(aVar)))

| "Seq.fold" ->
    // ('state -> 'a -> 'state) -> 'state -> seq<'a> -> 'state
    let stateVar = freshTypeVar ()
    let aVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TFun(stateVar, NativeType.TFun(aVar, stateVar)),
        NativeType.TFun(stateVar,
            NativeType.TFun(NativeType.TSeq(aVar), stateVar)))
```

### 4.2 SemanticKind for Seq Operations

Use existing `Application` with `IntrinsicInfo` marking:

```fsharp
// In SemanticGraph.fs
type IntrinsicModule =
    | Console
    | Format
    | Sys
    | NativePtr
    | Math
    | Lazy
    | Seq  // NEW

type IntrinsicInfo = {
    Module: IntrinsicModule
    Operation: string
    // ...
}
```

The PSG represents `Seq.map f xs` as:
```
Application(
    Application(VarRef "Seq.map", [mapperNode]),
    [sourceSeqNode])
```

With intrinsic info attached marking it as `{Module = Seq; Operation = "map"}`.

### 4.3 Files to Modify (FNCS)

| File | Action | Purpose |
|------|--------|---------|
| `CheckExpressions.fs` | MODIFY | Add Seq.map, filter, take, fold intrinsics |
| `SemanticGraph.fs` | MODIFY | Add `Seq` to IntrinsicModule |
| `NativeGlobals.fs` | MODIFY | Seq module registration |

## 5. Alex Layer Implementation

### 5.1 SeqOpLayout Coeffect

Compute layouts for sequence operation wrappers:

**File**: `src/Alex/Preprocessing/SeqOpLayout.fs`

```fsharp
/// Layout for a Seq.map wrapper
type MapSeqLayout = {
    ElementType: MLIRType           // Output element type (B)
    InnerElementType: MLIRType      // Input element type (A)
    InnerSeqLayout: SeqLayout       // Layout of inner sequence (inlined)
    MapperLayout: ClosureLayout     // Layout of mapper closure (flat)
    StructType: MLIRType            // Complete wrapper struct type
    MoveNextFuncName: string
}

/// Layout for a Seq.filter wrapper
type FilterSeqLayout = {
    ElementType: MLIRType
    InnerSeqLayout: SeqLayout
    PredicateLayout: ClosureLayout
    StructType: MLIRType
    MoveNextFuncName: string
}

/// Layout for a Seq.take wrapper
type TakeSeqLayout = {
    ElementType: MLIRType
    InnerSeqLayout: SeqLayout
    StructType: MLIRType
    MoveNextFuncName: string
}

/// Generate struct type for MapSeq
let mapSeqStructType (outElemType: MLIRType) (innerSeqType: MLIRType) (mapperType: MLIRType) : MLIRType =
    // {state: i32, current: B, moveNext_ptr: ptr, inner: InnerSeq, mapper: Mapper}
    TStruct [TInt I32; outElemType; TPtr; innerSeqType; mapperType]
```

### 5.2 Seq.map MoveNext Implementation

```fsharp
/// Generate MoveNext for Seq.map wrapper
/// Algorithm:
/// 1. Call inner.MoveNext()
/// 2. If false, return false
/// 3. Get inner.current
/// 4. Apply mapper to get transformed value
/// 5. Store in self.current
/// 6. Return true
let emitMapMoveNext (layout: MapSeqLayout) : MLIROp list =
    // MoveNext signature: (ptr) -> i1

    mlir {
        // Get pointer to inner sequence (inlined at fixed offset)
        yield "%inner_ptr = llvm.getelementptr %self[0, 3] : !llvm.ptr"

        // Call inner's MoveNext
        yield "%inner_moveNext_ptr = llvm.getelementptr %inner_ptr[0, 2] : !llvm.ptr"
        yield "%inner_moveNext = llvm.load %inner_moveNext_ptr : !llvm.ptr"
        yield "%has_next = llvm.call %inner_moveNext(%inner_ptr) : (!llvm.ptr) -> i1"
        yield "llvm.cond_br %has_next, ^transform, ^done"

        // Transform block
        yield "^transform:"
        // Get inner's current value
        yield "%inner_curr_ptr = llvm.getelementptr %inner_ptr[0, 1] : !llvm.ptr"
        yield "%inner_val = llvm.load %inner_curr_ptr"

        // Get mapper closure (inlined at offset 4)
        yield "%mapper_ptr = llvm.getelementptr %self[0, 4] : !llvm.ptr"
        yield "%mapper = llvm.load %mapper_ptr : !mapper_closure_type"

        // Extract code_ptr and captures from mapper (flat closure)
        yield "%mapper_code = llvm.extractvalue %mapper[0] : !mapper_closure_type"
        // Extract captures if any...

        // Call mapper: code_ptr(captures..., inner_val) -> B
        yield "%result = llvm.call %mapper_code(..., %inner_val)"

        // Store transformed value
        yield "%curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr"
        yield "llvm.store %result, %curr_ptr"
        yield "%true = arith.constant true"
        yield "llvm.return %true : i1"

        // Done block
        yield "^done:"
        yield "%false = arith.constant false"
        yield "llvm.return %false : i1"
    }
```

### 5.3 Seq.filter MoveNext Implementation

```fsharp
/// Generate MoveNext for Seq.filter wrapper
/// Algorithm:
/// 1. Loop: call inner.MoveNext()
/// 2. If false, return false
/// 3. Get inner.current
/// 4. Apply predicate
/// 5. If true: store in self.current, return true
/// 6. If false: continue loop
let emitFilterMoveNext (layout: FilterSeqLayout) : MLIROp list =
    mlir {
        yield "llvm.br ^loop"

        yield "^loop:"
        // Call inner MoveNext
        yield "%inner_ptr = llvm.getelementptr %self[0, 3] : !llvm.ptr"
        yield "%inner_moveNext_ptr = llvm.getelementptr %inner_ptr[0, 2] : !llvm.ptr"
        yield "%inner_moveNext = llvm.load %inner_moveNext_ptr : !llvm.ptr"
        yield "%has_next = llvm.call %inner_moveNext(%inner_ptr) : (!llvm.ptr) -> i1"
        yield "llvm.cond_br %has_next, ^check, ^done"

        yield "^check:"
        // Get inner current
        yield "%inner_curr_ptr = llvm.getelementptr %inner_ptr[0, 1] : !llvm.ptr"
        yield "%val = llvm.load %inner_curr_ptr"

        // Apply predicate (flat closure)
        yield "%pred_ptr = llvm.getelementptr %self[0, 4] : !llvm.ptr"
        yield "%pred = llvm.load %pred_ptr"
        yield "%pred_code = llvm.extractvalue %pred[0]"
        yield "%matches = llvm.call %pred_code(..., %val) : (...) -> i1"
        yield "llvm.cond_br %matches, ^yield, ^loop"

        yield "^yield:"
        yield "%curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr"
        yield "llvm.store %val, %curr_ptr"
        yield "%true = arith.constant true"
        yield "llvm.return %true : i1"

        yield "^done:"
        yield "%false = arith.constant false"
        yield "llvm.return %false : i1"
    }
```

### 5.4 Seq.take MoveNext Implementation

```fsharp
/// Generate MoveNext for Seq.take wrapper
/// Algorithm:
/// 1. Check remaining > 0
/// 2. If false, return false
/// 3. Call inner.MoveNext()
/// 4. If false, return false
/// 5. Copy inner.current to self.current
/// 6. Decrement remaining
/// 7. Return true
let emitTakeMoveNext (layout: TakeSeqLayout) : MLIROp list =
    mlir {
        // Check remaining
        yield "%remaining_ptr = llvm.getelementptr %self[0, 4] : !llvm.ptr"
        yield "%remaining = llvm.load %remaining_ptr : i32"
        yield "%zero = arith.constant 0 : i32"
        yield "%has_remaining = arith.cmpi sgt, %remaining, %zero : i32"
        yield "llvm.cond_br %has_remaining, ^try_inner, ^done"

        yield "^try_inner:"
        // Call inner MoveNext
        yield "%inner_ptr = llvm.getelementptr %self[0, 3] : !llvm.ptr"
        yield "%inner_moveNext_ptr = llvm.getelementptr %inner_ptr[0, 2] : !llvm.ptr"
        yield "%inner_moveNext = llvm.load %inner_moveNext_ptr : !llvm.ptr"
        yield "%has_next = llvm.call %inner_moveNext(%inner_ptr) : (!llvm.ptr) -> i1"
        yield "llvm.cond_br %has_next, ^yield, ^done"

        yield "^yield:"
        // Copy current
        yield "%inner_curr_ptr = llvm.getelementptr %inner_ptr[0, 1] : !llvm.ptr"
        yield "%val = llvm.load %inner_curr_ptr"
        yield "%curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr"
        yield "llvm.store %val, %curr_ptr"

        // Decrement remaining
        yield "%one = arith.constant 1 : i32"
        yield "%new_remaining = arith.subi %remaining, %one : i32"
        yield "llvm.store %new_remaining, %remaining_ptr"

        yield "%true = arith.constant true"
        yield "llvm.return %true : i1"

        yield "^done:"
        yield "%false = arith.constant false"
        yield "llvm.return %false : i1"
    }
```

### 5.5 Seq.fold (Eager Consumer)

Fold is NOT a sequence transformer - it consumes the sequence eagerly:

```fsharp
/// Emit Seq.fold - consumes sequence to produce single value
let witnessSeqFold
    (z: PSGZipper)
    (folderClosure: Val)       // Flat closure: (state, elem) -> state
    (initialVal: Val)          // Initial accumulator
    (seqVal: Val)              // Source sequence
    (layout: SeqLayout)
    : (MLIROp list * TransferResult) =

    mlir {
        // Allocate seq on stack for mutation
        yield "%seq_alloca = llvm.alloca 1 x !seq_type : !llvm.ptr"
        yield "llvm.store %seq_val, %seq_alloca"

        // Allocate accumulator
        yield "%acc_alloca = llvm.alloca 1 x !state_type : !llvm.ptr"
        yield "llvm.store %initial_val, %acc_alloca"

        yield "llvm.br ^loop"

        yield "^loop:"
        // Call MoveNext
        yield "%moveNext_ptr = llvm.getelementptr %seq_alloca[0, 2] : !llvm.ptr"
        yield "%moveNext = llvm.load %moveNext_ptr : !llvm.ptr"
        yield "%has_next = llvm.call %moveNext(%seq_alloca) : (!llvm.ptr) -> i1"
        yield "llvm.cond_br %has_next, ^body, ^done"

        yield "^body:"
        // Get current element
        yield "%curr_ptr = llvm.getelementptr %seq_alloca[0, 1] : !llvm.ptr"
        yield "%elem = llvm.load %curr_ptr"

        // Get current accumulator
        yield "%acc = llvm.load %acc_alloca"

        // Apply folder (flat closure)
        yield "%folder_code = llvm.extractvalue %folder[0]"
        // ... extract captures ...
        yield "%new_acc = llvm.call %folder_code(..., %acc, %elem)"

        // Store new accumulator
        yield "llvm.store %new_acc, %acc_alloca"
        yield "llvm.br ^loop"

        yield "^done:"
        yield "%result = llvm.load %acc_alloca"
        yield "llvm.return %result"
    }
```

### 5.6 Files to Create/Modify (Alex)

| File | Action | Purpose |
|------|--------|---------|
| `Alex/Preprocessing/SeqOpLayout.fs` | CREATE | Wrapper sequence layouts |
| `Alex/Witnesses/SeqWitness.fs` | MODIFY | Add map, filter, take, fold witnesses |

## 6. MLIR Output Specification

### 6.1 Seq.map Example

```mlir
// Source: Seq.map (fun x -> x * 2) numbers
// where numbers: seq { yield 1; yield 2; yield 3 }

// Inner seq type (from PRD-15)
!inner_seq = !llvm.struct<(i32, i32, ptr)>

// Mapper closure type (no captures - just code_ptr)
!mapper = !llvm.struct<(ptr)>

// MapSeq wrapper type
!map_seq = !llvm.struct<(i32, i32, ptr, !inner_seq, !mapper)>

// Map MoveNext function
llvm.func @map_double_moveNext(%self: !llvm.ptr) -> i1 {
    // Get inner seq pointer
    %inner_ptr = llvm.getelementptr %self[0, 3] : !llvm.ptr

    // Call inner MoveNext
    %inner_moveNext_ptr = llvm.getelementptr %inner_ptr[0, 2] : !llvm.ptr
    %inner_moveNext = llvm.load %inner_moveNext_ptr : !llvm.ptr
    %has_next = llvm.call %inner_moveNext(%inner_ptr) : (!llvm.ptr) -> i1
    llvm.cond_br %has_next, ^transform, ^done

^transform:
    // Get inner current
    %inner_curr_ptr = llvm.getelementptr %inner_ptr[0, 1] : !llvm.ptr
    %inner_val = llvm.load %inner_curr_ptr : i32

    // Apply mapper (x * 2)
    %two = arith.constant 2 : i32
    %result = arith.muli %inner_val, %two : i32

    // Store result
    %curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr
    llvm.store %result, %curr_ptr : i32

    %true = arith.constant true
    llvm.return %true : i1

^done:
    %false = arith.constant false
    llvm.return %false : i1
}

// Creation: Seq.map (fun x -> x * 2) numbers
// 1. Create inner seq (numbers)
%inner_seq = ... // from PRD-15

// 2. Create mapper closure (just code_ptr for no-capture lambda)
%mapper_code = llvm.mlir.addressof @double_func : !llvm.ptr
%mapper_undef = llvm.mlir.undef : !mapper
%mapper = llvm.insertvalue %mapper_code, %mapper_undef[0] : !mapper

// 3. Create map wrapper
%zero = arith.constant 0 : i32
%undef = llvm.mlir.undef : !map_seq
%s0 = llvm.insertvalue %zero, %undef[0] : !map_seq
%moveNext_ptr = llvm.mlir.addressof @map_double_moveNext : !llvm.ptr
%s1 = llvm.insertvalue %moveNext_ptr, %s0[2] : !map_seq
%s2 = llvm.insertvalue %inner_seq, %s1[3] : !map_seq
%map_seq_val = llvm.insertvalue %mapper, %s2[4] : !map_seq
```

### 6.2 Seq.filter Example

```mlir
// Source: Seq.filter (fun x -> x % 2 = 0) numbers

!filter_seq = !llvm.struct<(i32, i32, ptr, !inner_seq, !predicate)>

llvm.func @filter_even_moveNext(%self: !llvm.ptr) -> i1 {
    llvm.br ^loop

^loop:
    %inner_ptr = llvm.getelementptr %self[0, 3] : !llvm.ptr
    %inner_moveNext_ptr = llvm.getelementptr %inner_ptr[0, 2] : !llvm.ptr
    %inner_moveNext = llvm.load %inner_moveNext_ptr : !llvm.ptr
    %has_next = llvm.call %inner_moveNext(%inner_ptr) : (!llvm.ptr) -> i1
    llvm.cond_br %has_next, ^check, ^done

^check:
    %inner_curr_ptr = llvm.getelementptr %inner_ptr[0, 1] : !llvm.ptr
    %val = llvm.load %inner_curr_ptr : i32

    // Check x % 2 == 0
    %two = arith.constant 2 : i32
    %rem = arith.remsi %val, %two : i32
    %zero = arith.constant 0 : i32
    %is_even = arith.cmpi eq, %rem, %zero : i32
    llvm.cond_br %is_even, ^yield, ^loop

^yield:
    %curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr
    llvm.store %val, %curr_ptr : i32
    %true = arith.constant true
    llvm.return %true : i1

^done:
    %false = arith.constant false
    llvm.return %false : i1
}
```

### 6.3 Seq.fold Example

```mlir
// Source: Seq.fold (fun acc x -> acc + x) 0 numbers

llvm.func @fold_sum(%folder: !folder_closure, %initial: i32, %seq_val: !seq_type) -> i32 {
    // Allocate seq for mutation
    %one = arith.constant 1 : i64
    %seq_alloca = llvm.alloca %one x !seq_type : !llvm.ptr
    llvm.store %seq_val, %seq_alloca : !seq_type

    // Accumulator
    %acc_alloca = llvm.alloca %one x i32 : !llvm.ptr
    llvm.store %initial, %acc_alloca : i32

    llvm.br ^loop

^loop:
    %moveNext_ptr = llvm.getelementptr %seq_alloca[0, 2] : !llvm.ptr
    %moveNext = llvm.load %moveNext_ptr : !llvm.ptr
    %has_next = llvm.call %moveNext(%seq_alloca) : (!llvm.ptr) -> i1
    llvm.cond_br %has_next, ^body, ^done

^body:
    %curr_ptr = llvm.getelementptr %seq_alloca[0, 1] : !llvm.ptr
    %elem = llvm.load %curr_ptr : i32
    %acc = llvm.load %acc_alloca : i32

    // acc + x
    %new_acc = arith.addi %acc, %elem : i32
    llvm.store %new_acc, %acc_alloca : i32
    llvm.br ^loop

^done:
    %result = llvm.load %acc_alloca : i32
    llvm.return %result : i32
}
```

## 7. Validation

### 7.1 Sample Code

```fsharp
module SeqOperationsSample

let numbers = seq {
    let mutable i = 1
    while i <= 10 do
        yield i
        i <- i + 1
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Seq Operations Test ==="

    Console.writeln "--- Seq.map ---"
    let doubled = Seq.map (fun x -> x * 2) numbers
    for x in Seq.take 5 doubled do
        Console.writeln (Format.int x)

    Console.writeln "--- Seq.filter ---"
    let evens = Seq.filter (fun x -> x % 2 = 0) numbers
    for x in evens do
        Console.writeln (Format.int x)

    Console.writeln "--- Seq.fold ---"
    let sum = Seq.fold (fun acc x -> acc + x) 0 numbers
    Console.write "Sum: "
    Console.writeln (Format.int sum)

    0
```

### 7.2 Expected Output

```
=== Seq Operations Test ===
--- Seq.map ---
2
4
6
8
10
--- Seq.filter ---
2
4
6
8
10
--- Seq.fold ---
Sum: 55
```

## 8. Implementation Checklist

### Phase 1: Seq.map
- [ ] Add Seq.map intrinsic to FNCS
- [ ] Create MapSeqLayout coeffect
- [ ] Implement MapSeq MoveNext generation
- [ ] Test: map doubles values

### Phase 2: Seq.filter
- [ ] Add Seq.filter intrinsic
- [ ] Create FilterSeqLayout coeffect
- [ ] Implement FilterSeq MoveNext generation
- [ ] Test: filter for evens

### Phase 3: Seq.take
- [ ] Add Seq.take intrinsic
- [ ] Create TakeSeqLayout coeffect
- [ ] Implement TakeSeq MoveNext generation
- [ ] Test: take limits sequence

### Phase 4: Seq.fold
- [ ] Add Seq.fold intrinsic
- [ ] Implement fold as eager consumer (not a wrapper)
- [ ] Test: fold sums sequence

### Validation
- [ ] Sample 16 compiles without errors
- [ ] Sample 16 produces correct output
- [ ] Composed pipelines work (e.g., `take (map (filter xs))`)
- [ ] Samples 01-15 still pass

## 9. Related PRDs

- **PRD-11**: Closures - Mapper/predicate are flat closures
- **PRD-12**: HOFs - Seq operations are HOFs
- **PRD-14**: Lazy - Foundation for deferred computation
- **PRD-15**: SimpleSeq - Foundation for sequences

## 10. Architectural Alignment

This PRD aligns with the flat closure architecture:

**Key principles maintained:**
1. **No nulls** - Every field initialized
2. **No env_ptr** - Closures and inner sequences are flat
3. **Self-contained structs** - Wrappers inline their dependencies
4. **Coeffect-based layout** - Wrapper layouts computed before witnessing
5. **Composition = nesting** - Composed sequences are nested structs

**Trade-off acknowledged:**
- Struct size grows with composition depth
- For typical depths (3-5 operations), this is acceptable
- Very deep pipelines may need alternative strategies (future optimization)
