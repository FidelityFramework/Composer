# PRD-15: Simple Sequence Expressions

> **Sample**: `15_SimpleSeq` | **Status**: Planned | **Depends On**: PRD-14 (Lazy), PRD-11 (Closures)

## 1. Executive Summary

Sequence expressions (`seq { }`) provide lazy, on-demand iteration in F#. Unlike `Lazy<'T>` (single deferred value), `Seq<'T>` produces multiple values through resumable computation.

**Key Insight**: A sequence is an **extended flat closure with state machine fields**. Like Lazy (PRD-14), captures are inlined directly into the struct. The sequence adds state tracking for resumable computation at yield points.

**Builds on PRD-14**: Both `Lazy<'T>` and `Seq<'T>` are flat closures with extra state. Lazy has `{computed, value, code_ptr, captures...}`. Seq has `{state, current, code_ptr, captures...}`.

## 2. Language Feature Specification

### 2.1 Basic Sequence Expression

```fsharp
let numbers = seq {
    yield 1
    yield 2
    yield 3
}
```

Produces values 1, 2, 3 on demand.

### 2.2 Sequence with Computation

```fsharp
let squares n = seq {
    let mutable i = 1
    while i <= n do
        yield i * i
        i <- i + 1
}
```

### 2.3 Sequence with Conditional Yields

```fsharp
let evens max = seq {
    let mutable n = 0
    while n <= max do
        if n % 2 = 0 then
            yield n
        n <- n + 1
}
```

### 2.4 Sequence Consumption

```fsharp
for x in numbers do
    Console.writeln (Format.int x)
```

The `for...in` construct drives the state machine.

### 2.5 Semantic Laws

1. **Laziness**: Elements computed on demand, not eagerly
2. **Pull-based**: Consumer controls iteration pace
3. **Restartability**: Iterating a seq twice re-executes from the beginning
4. **Capture semantics**: Variables captured at creation time (flat closure model)

## 3. Architectural Principles

### 3.1 Flat Closure Extension (No `env_ptr`)

Following the flat closure model from PRD-11 and PRD-14:

```
Seq<T> = {state: i32, current: T, code_ptr: ptr, capture₀, capture₁, ...}
```

| Field | Type | Purpose |
|-------|------|---------|
| `state` | `i32` | Current state: 0=initial, N=after yield N, -1=done |
| `current` | `T` | Current value (valid after MoveNext returns true) |
| `code_ptr` | `ptr` | MoveNext function pointer |
| `capture₀...captureₙ` | varies | Inlined captured variables |

**There is no `env_ptr` field.** Captures are stored directly in the struct.

### 3.2 Capture Semantics (from PRD-11)

| Variable Kind | Capture Mode | Storage in Seq |
|---------------|--------------|----------------|
| Immutable | ByValue | Copy of value |
| Mutable | ByRef | Pointer to storage location |

### 3.3 Struct Layout Examples

**No captures** (`seq { yield 1; yield 2; yield 3 }`):
```
{state: i32, current: i32, code_ptr: ptr}
```

**With captures** (`seq { for i in 1..n do yield i * factor }` where n, factor are captured):
```
{state: i32, current: i32, code_ptr: ptr, n: i32, factor: i32}
```

### 3.4 State Machine Model

Each `yield` becomes a state transition point. The MoveNext function:
1. Switches on current state
2. Executes code until next yield (or end)
3. Stores yielded value in `current` field
4. Updates state to next yield point
5. Returns `true` (has value) or `false` (done)

## 4. FNCS Layer Implementation

### 4.1 NTUKind and NativeType Extensions

```fsharp
// In NativeTypes.fs
type NTUKind =
    // ... existing kinds ...
    | NTUseq   // Sequence/generator

type NativeType =
    // ... existing types ...
    | TSeq of elementType: NativeType
```

### 4.2 SemanticKind Extensions

```fsharp
type SemanticKind =
    // ... existing kinds ...

    /// Sequence expression - a resumable flat closure producing values on demand
    /// body: The sequence body containing yields
    /// captures: Variables captured from enclosing scope (flat closure model)
    | SeqExpr of body: NodeId * captures: CaptureInfo list

    /// Yield point within a sequence expression
    /// value: The expression to yield
    | Yield of value: NodeId

    /// For-each loop consuming an enumerable
    /// loopVar: Name of the loop variable
    /// varType: Type of the loop variable
    /// source: The sequence/enumerable to iterate
    /// body: Loop body executed for each element
    | ForEach of loopVar: string * varType: NativeType * source: NodeId * body: NodeId
```

**Note**: No `stateIndex` in Yield - that's computed by Alex as a coeffect (YieldStateIndices).

### 4.3 TypeEnv Extension

Following PRD-13's environment enrichment pattern:

```fsharp
type TypeEnv = {
    // ... existing fields ...

    /// Enclosing sequence expression (None at top level)
    /// Used to validate that yields appear inside seq { }
    EnclosingSeqExpr: NodeId option
}
```

### 4.4 Checking `seq { }` Expressions

```fsharp
/// Check a sequence expression
let checkSeqExpr
    (checkExpr: CheckExprFn)
    (env: TypeEnv)
    (builder: NodeBuilder)
    (seqBody: SynExpr)
    (range: range)
    : SemanticNode =

    // STEP 1: Pre-create SeqExpr node to get NodeId (PRD-13 pattern)
    let elementType = freshTypeVar range
    let seqNode = builder.Create(
        SemanticKind.SeqExpr(NodeId.Empty, []),  // Placeholder
        NativeType.TSeq(elementType),
        range,
        children = [])

    // STEP 2: Extend environment with enclosing seq context
    let seqEnv = { env with EnclosingSeqExpr = Some seqNode.Id }

    // STEP 3: Check body - yields will validate against EnclosingSeqExpr
    let bodyNode = checkExpr seqEnv builder seqBody

    // STEP 4: Collect captures (reuse closure logic from PRD-11)
    let captures = collectCaptures env builder bodyNode.Id

    // STEP 5: Update SeqExpr node with actual body and captures
    builder.SetChildren(seqNode.Id, [bodyNode.Id])
    builder.UpdateKind(seqNode.Id, SemanticKind.SeqExpr(bodyNode.Id, captures))

    seqNode
```

### 4.5 Checking `yield` Expressions

```fsharp
/// Check a yield expression within a sequence
let checkYield
    (checkExpr: CheckExprFn)
    (env: TypeEnv)
    (builder: NodeBuilder)
    (valueExpr: SynExpr)
    (range: range)
    : SemanticNode =

    // Validate we're inside a seq { }
    match env.EnclosingSeqExpr with
    | None ->
        failwith "yield can only appear inside a sequence expression"
    | Some seqId ->
        let valueNode = checkExpr env builder valueExpr

        builder.Create(
            SemanticKind.Yield(valueNode.Id),
            NativeType.TUnit,  // yield itself returns unit
            range,
            children = [valueNode.Id])
```

### 4.6 Checking `for...in` Expressions

```fsharp
/// Check a for-each loop
let checkForEach
    (checkExpr: CheckExprFn)
    (env: TypeEnv)
    (builder: NodeBuilder)
    (loopVar: string)
    (sourceExpr: SynExpr)
    (bodyExpr: SynExpr)
    (range: range)
    : SemanticNode =

    // Check the source sequence
    let sourceNode = checkExpr env builder sourceExpr

    // Extract element type from TSeq
    let elemType =
        match sourceNode.Type with
        | NativeType.TSeq elemTy -> elemTy
        | _ -> failwith "for...in requires a sequence source"

    // Extend environment with loop variable
    let loopEnv = env.AddLocal(loopVar, elemType)

    // Check body
    let bodyNode = checkExpr loopEnv builder bodyExpr

    builder.Create(
        SemanticKind.ForEach(loopVar, elemType, sourceNode.Id, bodyNode.Id),
        NativeType.TUnit,
        range,
        children = [sourceNode.Id; bodyNode.Id])
```

### 4.7 Files to Modify (FNCS)

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add `NTUseq`, `TSeq` |
| `SemanticGraph.fs` | MODIFY | Add `SeqExpr`, `Yield`, `ForEach` |
| `Types.fs` | MODIFY | Add `EnclosingSeqExpr` to TypeEnv |
| `Expressions/Computations.fs` | MODIFY | Seq/yield/for-in checking |
| `Expressions/Coordinator.fs` | MODIFY | Route expressions |

## 5. Alex Layer Implementation

### 5.1 YieldStateIndices Coeffect

Following the SSAAssignment pattern, assign state indices as coeffects:

**File**: `src/Alex/Preprocessing/YieldStateIndices.fs`

```fsharp
/// Coeffect: Maps each Yield NodeId to its state index
type YieldStateCoeffect = {
    /// SeqExpr NodeId -> (Yield NodeId -> state index)
    StateIndices: Map<NodeId, Map<NodeId, int>>
}

/// Assign state indices to all yield points in order of appearance
let run (graph: SemanticGraph) : YieldStateCoeffect =
    let mutable indices = Map.empty

    for node in graph.Nodes.Values do
        match node.Kind with
        | SemanticKind.SeqExpr(bodyId, _) ->
            let yields = collectYieldsInBody graph bodyId
            let yieldIndices =
                yields
                |> List.mapi (fun i yieldId -> (yieldId, i + 1))  // States 1, 2, 3...
                |> Map.ofList
            indices <- Map.add node.Id yieldIndices indices
        | _ -> ()

    { StateIndices = indices }
```

### 5.2 SeqLayout Coeffect

Following PRD-14's LazyLayout pattern:

**File**: `src/Alex/Preprocessing/SeqLayout.fs`

```fsharp
/// Layout information for a sequence expression
type SeqLayout = {
    /// NodeId of the SeqExpr
    SeqId: NodeId
    /// Element type (T in Seq<T>)
    ElementType: MLIRType
    /// Capture layouts (reuse from closure)
    Captures: CaptureLayout list
    /// Total struct type
    StructType: MLIRType
    /// Number of yield points
    NumYields: int
    /// MoveNext function name
    MoveNextFuncName: string
}

/// Generate MLIR struct type for a sequence expression
let seqStructType (elementType: MLIRType) (captureTypes: MLIRType list) : MLIRType =
    // {state: i32, current: T, code_ptr: ptr, cap₀, cap₁, ...}
    TStruct ([TInt I32; elementType; TPtr] @ captureTypes)
```

### 5.3 SeqWitness - Creation

```fsharp
/// Emit MLIR for sequence expression initialization
let witnessSeqCreate
    (z: PSGZipper)
    (layout: SeqLayout)
    (captureVals: Val list)
    : (MLIROp list * TransferResult) =

    let structType = layout.StructType
    let ssas = requireNodeSSAs layout.SeqId z

    let ops = [
        // Create initial state = 0
        MLIROp.ArithOp (ArithOp.ConstI (ssas.[0], 0L, MLIRTypes.i32))

        // Create undef seq struct
        MLIROp.LLVMOp (LLVMOp.Undef (ssas.[1], structType))

        // Insert state=0 at index 0
        MLIROp.LLVMOp (LLVMOp.InsertValue (ssas.[2], ssas.[1], ssas.[0], [0], structType))

        // Get MoveNext function address and insert at index 2
        MLIROp.LLVMOp (LLVMOp.AddressOf (ssas.[3], GFunc layout.MoveNextFuncName))
        MLIROp.LLVMOp (LLVMOp.InsertValue (ssas.[4], ssas.[2], ssas.[3], [2], structType))
    ]

    // Insert each capture at indices 3, 4, 5, ...
    let captureOps = ... // Same pattern as LazyWitness

    (ops @ captureOps, TRValue { SSA = resultSSA; Type = structType })
```

### 5.4 MoveNext Function Generation

The MoveNext function is generated as a state machine:

```fsharp
/// Generate MoveNext function for a sequence
/// Signature: (ptr-to-seq-struct) -> i1
/// Mutates the struct (state, current) through the pointer
let emitMoveNextFunction
    (seqId: NodeId)
    (layout: SeqLayout)
    (yieldIndices: Map<NodeId, int>)
    (bodyEmitter: int -> MLIRBuilder)  // state -> code for that state
    : MLIROp list =

    // MoveNext takes a POINTER to the seq struct (for mutation)
    // Extracts current state, switches, executes code, stores new state/current

    let numYields = layout.NumYields

    // Generate:
    // 1. Load state from struct
    // 2. Switch on state: 0 -> ^state0, 1 -> ^state1, ...
    // 3. Each state block: execute code until yield, store current, update state, return true
    // 4. Done block: set state = -1, return false

    ...
```

**Key insight**: MoveNext takes a **pointer** to the seq struct because it needs to mutate `state` and `current`. This is different from force (which can work with by-value for pure computation).

### 5.5 ForEachWitness

```fsharp
/// Emit MLIR for for-each loop
let witnessForEach
    (z: PSGZipper)
    (loopVar: string)
    (seqLayout: SeqLayout)
    (seqVal: Val)
    (bodyEmitter: Val -> MLIROp list)  // current value -> body ops
    : MLIROp list =

    // 1. Allocate seq struct on stack (if not already a pointer)
    // 2. Loop: call MoveNext, check result, extract current, emit body, repeat

    [
        // Alloca for seq struct
        MLIROp.LLVMOp (LLVMOp.Alloca (seqAllocaSSA, one, seqLayout.StructType, None))
        MLIROp.LLVMOp (LLVMOp.Store (seqVal.SSA, seqAllocaSSA, seqLayout.StructType, NotAtomic))

        // Loop header
        // %has_next = call @moveNext(%seq_alloca)
        // cond_br %has_next, ^body, ^done

        // Loop body
        // %current_ptr = gep %seq_alloca[0, 1]
        // %current = load %current_ptr
        // ... body using %current ...
        // br ^header

        // Done
        // ...
    ]
```

### 5.6 SSA Cost Computation

```fsharp
/// SSA cost for SeqExpr with N captures
let seqExprSSACost (numCaptures: int) : int =
    // 1: state constant (0)
    // 1: undef struct
    // 1: insert state
    // 1: addressof code_ptr
    // 1: insert code_ptr
    // N: insert each capture
    5 + numCaptures

/// SSA cost for ForEach
let forEachSSACost : int =
    // 1: alloca for seq
    // 1: store seq value
    // 1: moveNext result
    // 1: current_ptr (gep)
    // 1: current value (load)
    5
```

### 5.7 Files to Create/Modify (Alex)

| File | Action | Purpose |
|------|--------|---------|
| `Alex/Preprocessing/YieldStateIndices.fs` | CREATE | State index coeffects |
| `Alex/Preprocessing/SeqLayout.fs` | CREATE | Seq struct layout coeffects |
| `Alex/Witnesses/SeqWitness.fs` | CREATE | Seq creation, MoveNext emission |
| `Alex/Witnesses/ForEachWitness.fs` | CREATE | For-each loop emission |
| `Alex/Traversal/FNCSTransfer.fs` | MODIFY | Handle SeqExpr, Yield, ForEach |

## 6. MLIR Output Specification

### 6.1 Simple Sequence: `seq { yield 1; yield 2; yield 3 }`

```mlir
// Sequence struct type (no captures)
!seq_int_0 = !llvm.struct<(i32, i32, ptr)>

// MoveNext function
llvm.func @seq_123_moveNext(%self: !llvm.ptr) -> i1 {
    // Load current state
    %state_ptr = llvm.getelementptr %self[0, 0] : !llvm.ptr
    %state = llvm.load %state_ptr : i32

    // Switch on state
    llvm.switch %state : i32 [
        0: ^state0,
        1: ^state1,
        2: ^state2
    ], ^done

^state0:  // Initial -> yield 1
    %curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr
    %c1 = arith.constant 1 : i32
    llvm.store %c1, %curr_ptr : i32
    %s1 = arith.constant 1 : i32
    llvm.store %s1, %state_ptr : i32
    %true = arith.constant true
    llvm.return %true : i1

^state1:  // After yield 1 -> yield 2
    %curr_ptr_1 = llvm.getelementptr %self[0, 1] : !llvm.ptr
    %c2 = arith.constant 2 : i32
    llvm.store %c2, %curr_ptr_1 : i32
    %s2 = arith.constant 2 : i32
    llvm.store %s2, %state_ptr : i32
    %true1 = arith.constant true
    llvm.return %true1 : i1

^state2:  // After yield 2 -> yield 3
    %curr_ptr_2 = llvm.getelementptr %self[0, 1] : !llvm.ptr
    %c3 = arith.constant 3 : i32
    llvm.store %c3, %curr_ptr_2 : i32
    %s3 = arith.constant 3 : i32
    llvm.store %s3, %state_ptr : i32
    %true2 = arith.constant true
    llvm.return %true2 : i1

^done:
    %neg1 = arith.constant -1 : i32
    llvm.store %neg1, %state_ptr : i32
    %false = arith.constant false
    llvm.return %false : i1
}

// Creation
%zero = arith.constant 0 : i32
%undef = llvm.mlir.undef : !seq_int_0
%s0 = llvm.insertvalue %zero, %undef[0] : !seq_int_0
%code_ptr = llvm.mlir.addressof @seq_123_moveNext : !llvm.ptr
%seq_val = llvm.insertvalue %code_ptr, %s0[2] : !seq_int_0
```

### 6.2 Sequence with Captures: `seq { for i in 1..n do yield i * factor }`

```mlir
// Sequence struct type (captures n: i32, factor: i32, plus mutable i)
!seq_int_3 = !llvm.struct<(i32, i32, ptr, i32, i32, i32)>
// Fields: state, current, code_ptr, n, factor, i

// MoveNext function
llvm.func @seq_factors_moveNext(%self: !llvm.ptr) -> i1 {
    // Load state
    %state_ptr = llvm.getelementptr %self[0, 0] : !llvm.ptr
    %state = llvm.load %state_ptr : i32

    // Load captures (n, factor) and mutable (i)
    %n_ptr = llvm.getelementptr %self[0, 3] : !llvm.ptr
    %n = llvm.load %n_ptr : i32
    %factor_ptr = llvm.getelementptr %self[0, 4] : !llvm.ptr
    %factor = llvm.load %factor_ptr : i32
    %i_ptr = llvm.getelementptr %self[0, 5] : !llvm.ptr
    %i = llvm.load %i_ptr : i32

    llvm.switch %state : i32 [
        0: ^state0,
        1: ^state1
    ], ^done

^state0:  // Initial: i = 1
    %one = arith.constant 1 : i32
    llvm.store %one, %i_ptr : i32
    llvm.br ^check

^state1:  // After yield: i <- i + 1
    %i_val = llvm.load %i_ptr : i32
    %i_plus_1 = arith.addi %i_val, %one : i32
    llvm.store %i_plus_1, %i_ptr : i32
    llvm.br ^check

^check:
    %i_current = llvm.load %i_ptr : i32
    %done_cond = arith.cmpi sgt, %i_current, %n : i32
    llvm.cond_br %done_cond, ^done, ^yield

^yield:
    %i_for_yield = llvm.load %i_ptr : i32
    %product = arith.muli %i_for_yield, %factor : i32
    %curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr
    llvm.store %product, %curr_ptr : i32
    %s1 = arith.constant 1 : i32
    llvm.store %s1, %state_ptr : i32
    %true = arith.constant true
    llvm.return %true : i1

^done:
    %neg1 = arith.constant -1 : i32
    llvm.store %neg1, %state_ptr : i32
    %false = arith.constant false
    llvm.return %false : i1
}
```

### 6.3 For-Each Loop: `for x in numbers do ...`

```mlir
// Allocate seq struct on stack
%seq_alloca = llvm.alloca 1 x !seq_int_0 : !llvm.ptr
llvm.store %seq_val, %seq_alloca : !seq_int_0

llvm.br ^loop_check

^loop_check:
    %has_next = llvm.call @seq_123_moveNext(%seq_alloca) : (!llvm.ptr) -> i1
    llvm.cond_br %has_next, ^loop_body, ^loop_done

^loop_body:
    %curr_ptr = llvm.getelementptr %seq_alloca[0, 1] : !llvm.ptr
    %x = llvm.load %curr_ptr : i32
    // ... loop body using %x ...
    llvm.br ^loop_check

^loop_done:
```

## 7. Validation

### 7.1 Sample Code

```fsharp
/// Sample 15: Simple Sequence Expressions
module SimpleSeqSample

// Basic sequence - literal yields
let threeNumbers = seq {
    yield 1
    yield 2
    yield 3
}

// Sequence with while loop
let countUp (start: int) (stop: int) = seq {
    let mutable i = start
    while i <= stop do
        yield i
        i <- i + 1
}

// Sequence with conditional yields
let evenNumbersUpTo (max: int) = seq {
    let mutable n = 0
    while n <= max do
        if n % 2 = 0 then
            yield n
        n <- n + 1
}

// Sequence with captures
let multiplesOf (factor: int) (count: int) = seq {
    let mutable i = 1
    while i <= count do
        yield factor * i
        i <- i + 1
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Sample 15: Simple Sequences ==="
    Console.writeln ""

    Console.writeln "--- Basic Sequence ---"
    for x in threeNumbers do
        Console.writeln (Format.int x)

    Console.writeln ""
    Console.writeln "--- Count 1 to 5 ---"
    for x in countUp 1 5 do
        Console.writeln (Format.int x)

    Console.writeln ""
    Console.writeln "--- Evens up to 10 ---"
    for x in evenNumbersUpTo 10 do
        Console.writeln (Format.int x)

    Console.writeln ""
    Console.writeln "--- Multiples of 3 (first 5) ---"
    for x in multiplesOf 3 5 do
        Console.writeln (Format.int x)

    0
```

### 7.2 Expected Output

```
=== Sample 15: Simple Sequences ===

--- Basic Sequence ---
1
2
3

--- Count 1 to 5 ---
1
2
3
4
5

--- Evens up to 10 ---
0
2
4
6
8
10

--- Multiples of 3 (first 5) ---
3
6
9
12
15
```

## 8. Implementation Checklist

### Phase 1: FNCS Foundation
- [ ] Add `NTUseq` to NTUKind enum
- [ ] Add `TSeq` to NativeType
- [ ] Add `SeqExpr`, `Yield`, `ForEach` to SemanticKind
- [ ] Add `EnclosingSeqExpr` to TypeEnv
- [ ] Implement seq { } expression checking with capture analysis
- [ ] Implement yield checking
- [ ] Implement for...in checking
- [ ] FNCS builds successfully

### Phase 2: Alex Implementation
- [ ] Create `YieldStateIndices.fs` coeffect pass
- [ ] Create `SeqLayout.fs` coeffect pass
- [ ] Create `SeqWitness.fs` with flat closure model
- [ ] Create `ForEachWitness.fs`
- [ ] Handle SeqExpr, Yield, ForEach in FNCSTransfer
- [ ] Generate MoveNext functions with state machines
- [ ] Firefly builds successfully

### Phase 3: Validation
- [ ] Sample 15 compiles without errors
- [ ] Binary executes correctly
- [ ] State machine transitions verified
- [ ] Captures work correctly
- [ ] Samples 01-14 still pass (regression)

## 9. Lessons Applied

| Lesson | Application |
|--------|-------------|
| Flat closure model (PRD-11) | Captures inlined in seq struct |
| Pre-creation pattern (PRD-13) | SeqExpr node created before checking body |
| Environment enrichment (PRD-13) | `EnclosingSeqExpr` added to TypeEnv |
| Coeffect pattern (PRD-14) | YieldStateIndices, SeqLayout computed before witnessing |
| Extended closure (PRD-14) | Seq struct = closure + state machine fields |

## 10. Related PRDs

- **PRD-11**: Closures - Sequences reuse flat closure model
- **PRD-13**: Recursion - Pre-creation and environment enrichment patterns
- **PRD-14**: Lazy - Foundation for extended flat closures
- **PRD-16**: SeqOperations - `Seq.map`, `Seq.filter`, etc.
- **PRD-17**: Async - Builds on same deferred computation model

## 11. Architectural Alignment

This PRD aligns with the flat closure architecture:

**Key principles maintained:**
1. **No nulls** - Every field initialized
2. **No env_ptr** - Captures inlined directly
3. **Self-contained structs** - No pointer chains
4. **Coeffect-based layout** - State indices and layout computed before witnessing
5. **Capture reuse** - Same analysis as PRD-11 closures
6. **MoveNext by pointer** - Mutation requires pointer to struct (stack-allocated for for-each)

> **Critical:** See Serena memory `compose_from_standing_art_principle` for why composing from PRD-11/14 patterns is essential. New features MUST extend standing art, not reinvent.
