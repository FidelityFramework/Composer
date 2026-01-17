# PRD-15: Simple Sequence Expressions

> **Sample**: `15_SimpleSeq` | **Status**: Planned | **Depends On**: PRD-14 (Lazy), PRD-11 (Closures)

## 1. Executive Summary

Sequence expressions (`seq { }`) provide lazy, on-demand iteration in F#. Unlike `Lazy<'T>` (single deferred value), `Seq<'T>` produces multiple values through resumable computation.

**Key Insight**: A sequence is a **resumable closure** - it captures variables from scope and suspends/resumes at yield points. The state machine tracks which yield to resume from.

**Builds on PRD-14**: Both `Lazy<'T>` and `Seq<'T>` are deferred computations with captures. Lazy evaluates once; Seq evaluates repeatedly until exhausted.

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
4. **Capture semantics**: Variables captured at creation time

## 3. FNCS Layer Implementation

### 3.1 NTUKind Extension

**File**: `~/repos/fsnative/src/Compiler/Checking.Native/NativeTypes.fs`

```fsharp
type NTUKind =
    // ... existing kinds ...
    | NTUlazy  // From PRD-14
    | NTUseq   // NEW: Sequence/generator
```

### 3.2 NativeType Extension

```fsharp
type NativeType =
    // ... existing types ...
    | TLazy of valueType: NativeType  // From PRD-14
    | TSeq of elementType: NativeType  // NEW
```

### 3.3 SemanticKind Extensions

**File**: `~/repos/fsnative/src/Compiler/Checking.Native/SemanticGraph.fs`

```fsharp
type SemanticKind =
    // ... existing kinds ...

    /// Sequence expression - a resumable closure producing values on demand
    /// body: The sequence body containing yields
    /// captures: Variables captured from enclosing scope
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

**Note**: No `stateIndex` in Yield - that's computed by Alex as a coeffect.

### 3.4 TypeEnv Extension

Following PRD-13's environment enrichment pattern:

```fsharp
type TypeEnv = {
    // ... existing fields ...

    /// Enclosing sequence expression (None at top level)
    /// Used to validate that yields appear inside seq { }
    EnclosingSeqExpr: NodeId option
}
```

### 3.5 Checking seq { } Expressions

**File**: `~/repos/fsnative/src/Compiler/Checking.Native/Expressions/Computations.fs`

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

### 3.6 Checking yield Expressions

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

### 3.7 Checking for...in Expressions

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

### 3.8 Files to Modify (FNCS)

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add `NTUseq`, `TSeq` type constructor |
| `SemanticGraph.fs` | MODIFY | Add `SeqExpr`, `Yield`, `ForEach` SemanticKinds |
| `Types.fs` | MODIFY | Add `EnclosingSeqExpr` to TypeEnv |
| `Expressions/Computations.fs` | MODIFY | Add seq/yield/for-in checking |
| `Expressions/Coordinator.fs` | MODIFY | Route `seq { }` and `for...in` expressions |

## 4. Alex Layer Implementation

### 4.1 YieldStateIndices Coeffect

**File**: `src/Alex/Preprocessing/YieldStateIndices.fs`

Following the SSAAssignment pattern, assign state indices as coeffects:

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

### 4.2 Sequence State Machine Structure

```fsharp
type SeqStateMachine<'T> = {
    State: int              // 0 = initial, N = after yield N, -1 = done
    Current: 'T             // Current value (valid after MoveNext returns true)
    // ...captured variables...
}
```

### 4.3 SeqWitness - State Machine Emission

**File**: `src/Alex/Witnesses/SeqWitness.fs`

```fsharp
/// Emit MLIR for sequence expression initialization
let witnessSeqExpr
    (z: PSGZipper)
    (seqExprId: NodeId)
    (captures: CaptureInfo list)
    : MLIRBuilder =

    mlir {
        let structName = generateSeqStructName seqExprId

        // Allocate state machine struct
        yield $"%seq = llvm.alloca 1 x !{structName} : !llvm.ptr"

        // Initialize state = 0 (initial)
        yield "%state_ptr = llvm.getelementptr %seq[0, 0] : !llvm.ptr"
        yield "%zero = llvm.mlir.constant(0 : i32) : i32"
        yield "llvm.store %zero, %state_ptr : i32"

        // Store captured variables
        for (i, capture) in captures |> List.indexed do
            yield $"%cap_{i}_ptr = llvm.getelementptr %seq[0, {2 + i}] : !llvm.ptr"
            yield $"llvm.store %{capture.Name}, %cap_{i}_ptr"
    }

/// Emit the MoveNext function for a sequence
let emitMoveNextFunction
    (seqExprId: NodeId)
    (yieldIndices: Map<NodeId, int>)
    (body: SemanticNode)
    : MLIRBuilder =

    let numYields = yieldIndices.Count

    mlir {
        yield $"llvm.func @{seqExprId}_moveNext(%self: !llvm.ptr) -> i1 {{"
        yield "    %state_ptr = llvm.getelementptr %self[0, 0] : !llvm.ptr"
        yield "    %state = llvm.load %state_ptr : i32"
        yield ""
        yield $"    llvm.switch %state : i32 ["

        // State 0 = initial, States 1..N = after yield N
        for i in 0 .. numYields do
            yield $"        {i}: ^state{i},"
        yield "    ], ^done"

        // Generate state blocks based on body structure
        // Each yield becomes a state transition
        // ... emit body with state transitions ...

        yield "^done:"
        yield "    %neg1 = llvm.mlir.constant(-1 : i32) : i32"
        yield "    llvm.store %neg1, %state_ptr : i32"
        yield "    %false = llvm.mlir.constant(false) : i1"
        yield "    llvm.return %false : i1"
        yield "}"
    }
```

### 4.4 ForEachWitness

**File**: `src/Alex/Witnesses/ForEachWitness.fs`

```fsharp
/// Emit MLIR for for-each loop
let witnessForEach
    (z: PSGZipper)
    (loopVar: string)
    (sourceSSA: string)
    (bodyEmitter: unit -> MLIRBuilder)
    : MLIRBuilder =

    mlir {
        yield "llvm.br ^loop_check"

        yield "^loop_check:"
        yield $"    %has_next = llvm.call @moveNext(%{sourceSSA}) : (!llvm.ptr) -> i1"
        yield "    llvm.cond_br %has_next, ^loop_body, ^loop_done"

        yield "^loop_body:"
        yield $"    %curr_ptr = llvm.getelementptr %{sourceSSA}[0, 1] : !llvm.ptr"
        yield $"    %{loopVar} = llvm.load %curr_ptr"

        // Emit loop body
        yield! bodyEmitter()

        yield "    llvm.br ^loop_check"

        yield "^loop_done:"
    }
```

## 5. MLIR Output Specification

### 5.1 Simple Sequence: `seq { yield 1; yield 2; yield 3 }`

```mlir
// Sequence struct type
!seq_numbers = !llvm.struct<(
    i32,     // state: 0=initial, 1=after yield 1, etc., -1=done
    i32      // current value
)>

// MoveNext function
llvm.func @seq_numbers_moveNext(%self: !llvm.ptr) -> i1 {
    %state_ptr = llvm.getelementptr %self[0, 0] : !llvm.ptr
    %state = llvm.load %state_ptr : i32

    llvm.switch %state : i32 [
        0: ^state0,
        1: ^state1,
        2: ^state2
    ], ^done

^state0:  // Initial -> yield 1
    %curr_ptr = llvm.getelementptr %self[0, 1] : !llvm.ptr
    %c1 = llvm.mlir.constant(1 : i32) : i32
    llvm.store %c1, %curr_ptr : i32
    %s1 = llvm.mlir.constant(1 : i32) : i32
    llvm.store %s1, %state_ptr : i32
    %true1 = llvm.mlir.constant(true) : i1
    llvm.return %true1 : i1

^state1:  // After yield 1 -> yield 2
    %curr_ptr_1 = llvm.getelementptr %self[0, 1] : !llvm.ptr
    %c2 = llvm.mlir.constant(2 : i32) : i32
    llvm.store %c2, %curr_ptr_1 : i32
    %s2 = llvm.mlir.constant(2 : i32) : i32
    llvm.store %s2, %state_ptr : i32
    %true2 = llvm.mlir.constant(true) : i1
    llvm.return %true2 : i1

^state2:  // After yield 2 -> yield 3
    %curr_ptr_2 = llvm.getelementptr %self[0, 1] : !llvm.ptr
    %c3 = llvm.mlir.constant(3 : i32) : i32
    llvm.store %c3, %curr_ptr_2 : i32
    %s3 = llvm.mlir.constant(3 : i32) : i32
    llvm.store %s3, %state_ptr : i32
    %true3 = llvm.mlir.constant(true) : i1
    llvm.return %true3 : i1

^done:
    %neg1 = llvm.mlir.constant(-1 : i32) : i32
    llvm.store %neg1, %state_ptr : i32
    %false = llvm.mlir.constant(false) : i1
    llvm.return %false : i1
}
```

### 5.2 For-Each Loop: `for x in numbers do ...`

```mlir
// Initialize sequence
%seq = llvm.alloca 1 x !seq_numbers : !llvm.ptr
%init_state = llvm.mlir.constant(0 : i32) : i32
%state_ptr = llvm.getelementptr %seq[0, 0] : !llvm.ptr
llvm.store %init_state, %state_ptr : i32

// Loop
llvm.br ^loop_check

^loop_check:
    %has_next = llvm.call @seq_numbers_moveNext(%seq) : (!llvm.ptr) -> i1
    llvm.cond_br %has_next, ^loop_body, ^loop_done

^loop_body:
    %curr_ptr = llvm.getelementptr %seq[0, 1] : !llvm.ptr
    %x = llvm.load %curr_ptr : i32
    // ... loop body using %x ...
    llvm.br ^loop_check

^loop_done:
```

## 6. Validation

### 6.1 Sample Code

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

### 6.2 Expected Output

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

## 7. Files to Create/Modify

### 7.1 FNCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add `NTUseq`, `TSeq` |
| `SemanticGraph.fs` | MODIFY | Add `SeqExpr`, `Yield`, `ForEach` |
| `Types.fs` | MODIFY | Add `EnclosingSeqExpr` to TypeEnv |
| `Expressions/Computations.fs` | MODIFY | Seq/yield/for-in checking |
| `Expressions/Coordinator.fs` | MODIFY | Route expressions |

### 7.2 Firefly

| File | Action | Purpose |
|------|--------|---------|
| `Alex/Preprocessing/YieldStateIndices.fs` | CREATE | State index coeffects |
| `Alex/Witnesses/SeqWitness.fs` | CREATE | Seq state machine emission |
| `Alex/Witnesses/ForEachWitness.fs` | CREATE | For-each loop emission |
| `Alex/Traversal/FNCSTransfer.fs` | MODIFY | Handle SeqExpr, Yield, ForEach |

## 8. Implementation Checklist

### Phase 1: FNCS Foundation
- [ ] Add `NTUseq` to NTUKind enum
- [ ] Add `TSeq` to NativeType
- [ ] Add `SeqExpr`, `Yield`, `ForEach` to SemanticKind
- [ ] Add `EnclosingSeqExpr` to TypeEnv
- [ ] Implement seq { } expression checking
- [ ] Implement yield checking
- [ ] Implement for...in checking
- [ ] FNCS builds successfully

### Phase 2: Alex Implementation
- [ ] Create `YieldStateIndices.fs` coeffect pass
- [ ] Create `SeqWitness.fs`
- [ ] Create `ForEachWitness.fs`
- [ ] Handle SeqExpr, Yield, ForEach in transfer
- [ ] Firefly builds successfully

### Phase 3: Validation
- [ ] Sample 15 compiles without errors
- [ ] Binary executes correctly
- [ ] State machine transitions verified
- [ ] Samples 01-14 still pass (regression)

## 9. Lessons Applied

| Lesson | Application |
|--------|-------------|
| Pre-creation pattern (PRD-13) | SeqExpr node created before checking body |
| Environment enrichment (PRD-13) | `EnclosingSeqExpr` added to TypeEnv |
| Capture analysis (PRD-11) | Seq reuses closure capture logic |
| FNCS captures semantics | No stateIndex in Yield - semantic only |
| Alex computes coeffects | `YieldStateIndices` assigns state numbers |
| Build foundationally (PRD-14) | Seq builds on Lazy thunk concepts |

## 10. Related PRDs

- **PRD-11**: Closures - Sequences reuse capture analysis
- **PRD-13**: Recursion - Pre-creation and environment enrichment patterns
- **PRD-14**: Lazy - Foundation for deferred computation
- **PRD-16**: SeqOperations - `Seq.map`, `Seq.filter`, etc.
- **PRD-17**: Async - Builds on same deferred computation model
