# A-01: Basic Async (LLVM Coroutines Foundation)

> **Sample**: `17_BasicAsync` | **Status**: Planned | **Depends On**: C-01 to C-07 (Closures through Lazy)

## 1. Executive Summary

This PRD establishes the foundation for async computation using **LLVM coroutine intrinsics**. Unlike .NET's Task-based async (which requires a runtime), Fidelity's async compiles directly to LLVM coroutine state machines that the CoroSplit pass transforms at compile time.

**Key Insight**: LLVM coroutines are a compile-time transformation, not a runtime feature. The compiler marks suspension points, and LLVM's passes handle frame allocation, state saving, and resumption code generation.

**Reference**: See `async_llvm_coroutines` memory for the StateMachine strategy.

## 2. Language Feature Specification

### 2.1 Basic Async Expression

```fsharp
let simpleAsync = async {
    return 42
}
```

A trivial async that immediately returns - no suspension points.

### 2.2 Async.RunSynchronously

```fsharp
let result = Async.RunSynchronously simpleAsync
```

Runs the async to completion, blocking the current thread.

### 2.3 Async Return Types

```fsharp
// Async<int> - async returning int
let intAsync = async { return 42 }

// Async<string> - async returning string
let strAsync = async { return "hello" }

// Async<unit> - async with no meaningful return
let unitAsync = async { Console.writeln "done" }
```

## 3. CCS Layer Implementation

### 3.1 Async Type

```fsharp
// In NativeTypes.fs
| TAsync of resultType: NativeType

// Type constructor
| "Async" -> fun resTy -> NativeType.TAsync(resTy)
```

### 3.2 SemanticKind.AsyncExpr

```fsharp
type SemanticKind =
    | AsyncExpr of
        body: NodeId *
        suspensionPoints: int list *  // Indices for let! points
        captures: CaptureInfo list
```

### 3.3 SemanticKind.AsyncReturn

```fsharp
| AsyncReturn of value: NodeId
```

The `return` keyword in async context.

### 3.4 Async Intrinsics

```fsharp
// In CheckExpressions.fs
| "Async.RunSynchronously" ->
    // Async<'a> -> 'a
    let aVar = freshTypeVar ()
    NativeType.TFun(NativeType.TAsync(aVar), aVar)

| "async.Return" ->  // Internal, for return keyword
    // 'a -> Async<'a>
    let aVar = freshTypeVar ()
    NativeType.TFun(aVar, NativeType.TAsync(aVar))
```

## 4. Source Settlement and Passive Witnessing

CCS/Baker elaborates the async body, captures and activation boundary into the
PSG. Formation preserves the deferred computation; `Async.RunSynchronously`
activates it. Absence of suspension points does not authorize Alex to execute
the body during formation.

Owning nanopasses settle the callable identity, captured values, result storage,
frame layout, lifetime, completion protocol and physical signature. Their
ingredients, recipes and joint premises remain available for invalidation.

Alex passively witnesses the published control, calls and storage operations
through Huet Element/Pattern/Witness composition. Coroutine intrinsics and their
target-specific realization belong to Composer's backend. Missing source
settlement fails at its owning contract.

## 5. Backend Realization Contract

The selected backend realizes the source-settled async frame, activation call
and completion operations. Physical widths and layouts come from the published
representation and boundary facts.

Formation and activation must remain observably distinct, including for bodies
without suspension. Any source-owned simplification must prove preservation of
demand, effects and lifetime before publication. Backend output alone does not
authorize eager body execution.

## 6. Validation

### 6.1 Sample Code

```fsharp
module BasicAsyncSample

let simpleAsync = async {
    return 42
}

let asyncWithCapture x = async {
    return x * 2
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Basic Async Test ==="

    Console.writeln "--- Simple Async ---"
    let v1 = Async.RunSynchronously simpleAsync
    Console.write "Result: "
    Console.writeln (Format.int v1)

    Console.writeln "--- Async with Capture ---"
    let v2 = Async.RunSynchronously (asyncWithCapture 21)
    Console.write "Result: "
    Console.writeln (Format.int v2)

    0
```

### 6.2 Expected Output

```
=== Basic Async Test ===
--- Simple Async ---
Result: 42
--- Async with Capture ---
Result: 42
```

## 7. Files to Create/Modify

### 7.1 CCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TAsync type constructor |
| `SemanticGraph.fs` | MODIFY | Add AsyncExpr, AsyncReturn SemanticKinds |
| `CheckExpressions.fs` | MODIFY | Add Async.RunSynchronously intrinsic |
| `Expressions/Computations.fs` | MODIFY | Handle async { } expressions |

### 7.2 Composer

| File | Action | Purpose |
|------|--------|---------|
| Alex async witnesses | CREATE | Passively compose settled async control, storage and calls |
| CCS/Baker continuation settlement | CREATE | Settle activation, suspension identity, frame and lifetime |
| `src/Alex/Traversal/CCSTransfer.fs` | MODIFY | Handle AsyncExpr, AsyncReturn |

## 8. Implementation Checklist

### Phase 1: CCS Foundation
- [ ] Add TAsync to NativeTypes
- [ ] Add AsyncExpr, AsyncReturn to SemanticKind
- [ ] Implement async { } checking
- [ ] Add Async.RunSynchronously intrinsic

### Phase 2: Settlement and Witnessing
- [ ] Settle deferred body, captures, frame, lifetime and activation in CCS/Baker
- [ ] Publish immutable facts for passive async witnesses
- [ ] Verify RunSynchronously activates the body at the admitted demand boundary

### Phase 3: Validation
- [ ] Sample 17 compiles without errors
- [ ] Sample 17 produces correct output
- [ ] Samples 01-16 still pass

## 9. Why LLVM Coroutines?

| Alternative | Drawback |
|-------------|----------|
| .NET Task | Requires runtime, GC, thread pool |
| libco/boost.context | External dependency, non-portable |
| Custom state machine | Complex, error-prone |
| **LLVM Coroutines** | **Built-in, cross-platform, no runtime** |

LLVM's coroutine passes are production-quality and handle all the difficult state machine generation automatically.

## 10. Related PRDs

- **A-02**: Async Await - Adds `let!` (suspension points)
- **A-03**: Async Parallel - Adds `Async.Parallel`
- **T-03 to T-05**: MailboxProcessor - Uses async for message loop
