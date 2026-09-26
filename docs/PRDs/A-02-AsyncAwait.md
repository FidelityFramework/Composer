# A-02: Async Await (let! and Suspension)

> **Sample**: `18_AsyncAwait` | **Status**: Planned | **Depends On**: A-01 (BasicAsync)

## 1. Executive Summary

This PRD adds `let!` - the ability to await other async computations. This is where LLVM coroutines become essential: each `let!` is a **suspension point** where the coroutine may pause and later resume.

**Key Insight**: `let!` compiles to `llvm.coro.suspend`. The CoroSplit pass transforms the function into a state machine that can pause at each suspension point and resume with the awaited value.

## 2. Language Feature Specification

### 2.1 let! (Async Bind)

```fsharp
let composed = async {
    let! x = async { return 10 }
    let! y = async { return 20 }
    return x + y
}
```

Each `let!` awaits the inner async before continuing.

### 2.2 do! (Async Ignore)

```fsharp
let withSideEffect = async {
    do! async { Console.writeln "Side effect" }
    return 42
}
```

Like `let!` but discards the result.

### 2.3 Suspension Semantics

At each `let!`:
1. Evaluate the inner async
2. If inner async is incomplete, suspend
3. When inner completes, resume with its result
4. Bind result to name and continue

For now (single-threaded), inner asyncs complete immediately, so suspension is technically immediate resumption. True suspension matters when combined with I/O (I-01/I-02) or threading (T-01/T-02).

## 3. CCS Layer Implementation

### 3.1 SemanticKind.AsyncBind

```fsharp
type SemanticKind =
    | AsyncBind of
        name: string *
        inner: NodeId *        // The async being awaited
        continuation: NodeId * // Code after the let!
        suspensionIndex: int   // Unique index for this suspension point
```

### 3.2 SemanticKind.AsyncDo

```fsharp
| AsyncDo of
    inner: NodeId *
    continuation: NodeId *
    suspensionIndex: int
```

Same as AsyncBind but without binding a name.

### 3.3 Type Checking let!

```fsharp
let checkAsyncBind env builder name innerExpr contExpr =
    // 1. Check inner expression - must be Async<'a>
    let innerNode = checkExpr env builder innerExpr
    match innerNode.Type with
    | TAsync elemTy ->
        // 2. Bind name to 'a in continuation
        let envWithBinding = addBinding name elemTy env

        // 3. Check continuation
        let contNode = checkExpr envWithBinding builder contExpr

        // 4. Assign suspension index
        let suspIdx = nextSuspensionIndex ()

        builder.Create(
            SemanticKind.AsyncBind(name, innerNode.Id, contNode.Id, suspIdx),
            contNode.Type,  // Type is continuation's type
            range)
    | _ ->
        error "let! requires Async<_> on right-hand side"
```

## 4. Source Settlement and Passive Witnessing

CCS/Baker owns suspension discovery, delimiter and continuation identity,
liveness across suspension, frame storage and definite initialization. It
constructs and settles the bind, resume, completion and cleanup protocol in the
PSG before publication.

Suspension ordering and frame membership are source facts with complete joint
premises. A frame change invalidates its dependent layouts, calls and witnesses.
A sequential execution sketch cannot substitute for the required async semantics.

Alex witnesses settled control and storage from immutable node-local codata.
It neither numbers source suspension points nor reconstructs continuation
control from async bodies. Composer's backend realizes the admitted coroutine
operations for the selected target.

## 5. MLIR Output Specification

### 5.1 Frame with Suspension Points

```mlir
// async { let! x = ...; let! y = ...; return x + y }
!composed_frame = !llvm.struct<(
    i32,    // state
    i32,    // result
    i32,    // x_slot (bound by first let!)
    i32     // y_slot (bound by second let!)
)>
```

### 5.2 State Machine Switch

```mlir
// Entry point dispatches based on state
%state = llvm.load %state_ptr : i32
llvm.switch %state : i32 [
    0: ^state0,
    1: ^state1,
    2: ^state2
], ^done
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module AsyncAwaitSample

let asyncAdd a b = async {
    return a + b
}

let composed = async {
    let! x = asyncAdd 10 20
    let! y = asyncAdd x 5
    return y
}

let withDo = async {
    do! async { Console.writeln "Step 1" }
    do! async { Console.writeln "Step 2" }
    return 42
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Async Await Test ==="

    Console.writeln "--- Composed Async ---"
    let v1 = Async.RunSynchronously composed
    Console.write "Result: "
    Console.writeln (Format.int v1)

    Console.writeln "--- Async with do! ---"
    let v2 = Async.RunSynchronously withDo
    Console.write "Result: "
    Console.writeln (Format.int v2)

    0
```

### 6.2 Expected Output

```
=== Async Await Test ===
--- Composed Async ---
Result: 35
--- Async with do! ---
Step 1
Step 2
Result: 42
```

## 7. Files to Create/Modify

### 7.1 CCS

| File | Action | Purpose |
|------|--------|---------|
| `SemanticGraph.fs` | MODIFY | Add AsyncBind, AsyncDo SemanticKinds |
| `Expressions/Computations.fs` | MODIFY | Handle let! and do! in async |

### 7.2 Composer

| File | Action | Purpose |
|------|--------|---------|
| CCS/Baker continuation settlement | MODIFY | Settle suspension identity, liveness, frame and control |
| Alex async witnesses | MODIFY | Witness settled continuation control and frame operations |
| `src/Alex/Traversal/CCSTransfer.fs` | MODIFY | Handle AsyncBind, AsyncDo |

## 8. Implementation Checklist

### Phase 1: CCS Suspension Points
- [ ] Add AsyncBind, AsyncDo to SemanticKind
- [ ] Implement let!/do! type checking
- [ ] Assign suspension indices during checking

### Phase 2: Settlement and Witnessing
- [ ] Settle frame membership, storage and definite initialization in CCS/Baker
- [ ] Construct resume, completion and cleanup control through Baker recipes
- [ ] Publish immutable facts and passively witness settled operations
- [ ] Realize admitted coroutine operations in the selected backend

### Phase 3: Validation
- [ ] Sample 18 compiles without errors
- [ ] Sample 18 produces correct output
- [ ] Samples 01-17 still pass

## 9. Future: True Suspension

Currently, inner asyncs complete immediately. True suspension requires:

1. **I/O Operations** (I-01/I-02): Socket read that may block
2. **Thread Integration** (T-01/T-02): Async running on different thread

When these exist, `llvm.coro.suspend` actually suspends, and a scheduler resumes the coroutine when the awaited operation completes.

## 10. Related PRDs

- **A-01**: BasicAsync - Foundation
- **A-03**: AsyncParallel - Running multiple asyncs
- **I-01/I-02**: Networking - True async I/O
- **T-03-31**: MailboxProcessor - Async message loop
