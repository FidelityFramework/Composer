# A-03: Async.Parallel

> **Sample**: `19_AsyncParallel` | **Status**: Planned | **Depends On**: A-01-18 (BasicAsync, AsyncAwait)

## 1. Executive Summary

`Async.Parallel` runs multiple async computations and collects their results. This is a composition operation - it doesn't introduce new suspension semantics, but combines multiple asyncs into one.

**Key Insight**: `Async.Parallel` in single-threaded mode runs asyncs sequentially. True parallelism requires threading (T-01/T-02). But the API and semantics are established here.

## 2. Language Feature Specification

### 2.1 Async.Parallel

```fsharp
let tasks = [|
    async { return 1 }
    async { return 2 }
    async { return 3 }
|]

let results = Async.Parallel tasks |> Async.RunSynchronously
// results = [| 1; 2; 3 |]
```

### 2.2 Async.Sequential (For Comparison)

```fsharp
let results = Async.Sequential tasks |> Async.RunSynchronously
```

Explicitly sequential execution (same as Parallel in single-threaded mode).

### 2.3 Type Signatures

```fsharp
Async.Parallel : Async<'T>[] -> Async<'T[]>
Async.Sequential : Async<'T>[] -> Async<'T[]>
```

## 3. CCS Layer Implementation

### 3.1 Async.Parallel Intrinsic

```fsharp
// In CheckExpressions.fs
| "Async.Parallel" ->
    // Async<'a>[] -> Async<'a[]>
    let aVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TArray(NativeType.TAsync(aVar)),
        NativeType.TAsync(NativeType.TArray(aVar)))

| "Async.Sequential" ->
    // Same signature
    let aVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TArray(NativeType.TAsync(aVar)),
        NativeType.TAsync(NativeType.TArray(aVar)))
```

### 3.2 No New SemanticKind

CCS recognizes Async.Parallel as an intrinsic call. Baker constructs and settles
its protocol in the PSG before publication; an intrinsic tag alone is insufficient
for a witness to invent parallel execution behavior.

## 4. Source Settlement and Passive Witnessing

CCS/Baker constructs the parallel or sequential protocol required by the
selected source operation. It settles activation, work dependencies, result
ordering, join conditions, storage capacity, lifetime and synchronization
obligations in the PSG.

A witness cannot silently replace `Async.Parallel` with a sequential loop or
choose a thread-per-computation algorithm. Any admitted realization must preserve
the source contract and carry the premises required for its scheduling and
resource commitments.

Alex passively composes the published control, storage and calls. Composer's
backend realizes the selected target's threading or scheduling operations.
Native concurrency, ordering and completion observations remain distinct
acceptance requirements.

## 5. Backend Realization Contract

The selected backend realizes the source-settled scheduling, result-storage and
join operations. It preserves the admitted concurrency, ordering and lifetime
contract. A sequential-loop sketch does not establish Async.Parallel acceptance;
native concurrency and completion evidence must cover the admitted realization.

## 6. Validation

### 6.1 Sample Code

```fsharp
module AsyncParallelSample

let makeAsync (n: int) = async {
    Console.write "Computing "
    Console.writeln (Format.int n)
    return n * n
}

[<EntryPoint>]
let main _ =
    Console.writeln "=== Async Parallel Test ==="

    let tasks = [|
        makeAsync 1
        makeAsync 2
        makeAsync 3
        makeAsync 4
        makeAsync 5
    |]

    Console.writeln "--- Running Parallel ---"
    let results = Async.Parallel tasks |> Async.RunSynchronously

    Console.writeln "--- Results ---"
    for r in results do
        Console.writeln (Format.int r)

    0
```

### 6.2 Expected Output

```
=== Async Parallel Test ===
--- Running Parallel ---
Computing 1
Computing 2
Computing 3
Computing 4
Computing 5
--- Results ---
1
4
9
16
25
```

Note: In single-threaded mode, "Computing N" appears in order. With true threading, the order may vary.

## 7. Files to Create/Modify

### 7.1 CCS

| File | Action | Purpose |
|------|--------|---------|
| `CheckExpressions.fs` | MODIFY | Add Async.Parallel, Async.Sequential intrinsics |

### 7.2 Composer

| File | Action | Purpose |
|------|--------|---------|
| Alex async witnesses | MODIFY | Witness source-settled parallel/sequential protocols |

## 8. Implementation Checklist

### Phase 1: Source-Owned Protocol
- [ ] Add Async.Parallel intrinsic to CCS
- [ ] Construct and settle admitted activation, join and result-storage operations in Baker
- [ ] Witness published operations passively
- [ ] Test ordering and completion with an array of asyncs

### Phase 2: Validation
- [ ] Sample 19 compiles without errors
- [ ] Sample 19 produces correct output
- [ ] Samples 01-18 still pass

### Phase 3 (Future): True Parallelism
- [ ] After T-01/T-02: Implement threaded version
- [ ] Add thread pool or worker spawn
- [ ] Implement completion synchronization

## 9. Design Decision: Arrays vs Lists

Using arrays (`Async<'T>[]`) rather than lists because:
1. Known length enables result array pre-allocation
2. Index-based access for parallel assignment
3. Cache-friendly iteration

Lists could be supported via `Async.ParallelSeq : seq<Async<'T>> -> Async<'T[]>` that first collects to array.

## 10. Related PRDs

- **A-01-18**: BasicAsync, AsyncAwait - Foundation
- **T-01/T-02**: Threading - True parallelism
- **T-03-31**: MailboxProcessor - Parallel actors
