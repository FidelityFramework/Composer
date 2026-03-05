# T-02: Mutex Synchronization

> **Sample**: `28_MutexSync` | **Status**: Planned | **Depends On**: T-01 (BasicThread)

## 1. Executive Summary

This PRD adds mutual exclusion (mutex) primitives for thread synchronization. Mutexes prevent data races when multiple threads access shared state.

**Key Insight**: Mutexes are OS primitives (pthread_mutex on Linux, CRITICAL_SECTION on Windows). The raw pthread_mutex functions are accessed via Farscape-generated `Fidelity.Pthread` bindings. The language-level `Mutex.create`/`lock`/`unlock` API is a **CCS intrinsic** that wraps the raw bindings with opaque handle management.

**Architecture**: Same two-layer model as T-01:
- **Raw binding** (L1): `Fidelity.Pthread` — `[<FidelityExtern("pthread", "pthread_mutex_init")>]`
- **Language API** (CCS intrinsic): `Mutex.create` — allocates opaque storage, initializes via ExternCall to pthread

The CCS intrinsic is justified because mutex handles require compiler-managed opaque storage sizing (40 bytes for pthread_mutex_t on Linux) and destruction guarantees.

## 2. Language Feature Specification

### 2.1 Mutex Creation

```clef
let mutex = Mutex.create ()
```

### 2.2 Lock/Unlock

```clef
Mutex.lock mutex
// Critical section - only one thread at a time
sharedCounter <- sharedCounter + 1
Mutex.unlock mutex
```

### 2.3 Lock Guard Pattern

```clef
let withLock (mutex: Mutex) (f: unit -> 'T) : 'T =
    Mutex.lock mutex
    let result = f ()
    Mutex.unlock mutex
    result

// Usage
let value = withLock mutex (fun () ->
    sharedData.[index])
```

### 2.4 Condition Variables

```clef
let cond = CondVar.create ()
CondVar.wait cond mutex    // Atomically unlock and wait
CondVar.signal cond        // Wake one waiter
CondVar.broadcast cond     // Wake all waiters
```

## 3. CCS Layer Implementation

### 3.1 Mutex Type

```fsharp
// In NativeTypes.fs
| TMutex      // Opaque mutex handle
| TCondVar    // Opaque condition variable handle
```

### 3.2 Mutex Intrinsics

```fsharp
// In CheckExpressions.fs
| "Mutex.create" ->
    NativeType.TFun(env.Globals.UnitType, NativeType.TMutex)

| "Mutex.lock" ->
    NativeType.TFun(NativeType.TMutex, env.Globals.UnitType)

| "Mutex.unlock" ->
    NativeType.TFun(NativeType.TMutex, env.Globals.UnitType)

| "Mutex.tryLock" ->
    NativeType.TFun(NativeType.TMutex, env.Globals.BoolType)

| "Mutex.destroy" ->
    NativeType.TFun(NativeType.TMutex, env.Globals.UnitType)
```

### 3.3 CondVar Intrinsics

```fsharp
| "CondVar.create" ->
    NativeType.TFun(env.Globals.UnitType, NativeType.TCondVar)

| "CondVar.wait" ->
    NativeType.TFun(NativeType.TCondVar,
        NativeType.TFun(NativeType.TMutex, env.Globals.UnitType))

| "CondVar.signal" ->
    NativeType.TFun(NativeType.TCondVar, env.Globals.UnitType)

| "CondVar.broadcast" ->
    NativeType.TFun(NativeType.TCondVar, env.Globals.UnitType)
```

## 4. Binding Generation (Farscape)

### 4.1 pthread_mutex in Pilot TOML

```toml
# pthread.pilot.toml
[[namespace]]
name = "Fidelity.Pthread.Mutex"
description = "POSIX mutex and condition variable operations"
library = "pthread"
functions = ["pthread_mutex_init", "pthread_mutex_lock", "pthread_mutex_unlock",
             "pthread_mutex_trylock", "pthread_mutex_destroy",
             "pthread_cond_init", "pthread_cond_wait", "pthread_cond_signal",
             "pthread_cond_broadcast", "pthread_cond_destroy"]
```

### 4.2 Generated L1 Declarations

```clef
module Fidelity.Pthread.Mutex

[<FidelityExtern("pthread", "pthread_mutex_init")>]
let pthread_mutex_init (mutex: nativeint) (attr: nativeint) : int = Unchecked.defaultof<int>

[<FidelityExtern("pthread", "pthread_mutex_lock")>]
let pthread_mutex_lock (mutex: nativeint) : int = Unchecked.defaultof<int>

[<FidelityExtern("pthread", "pthread_mutex_unlock")>]
let pthread_mutex_unlock (mutex: nativeint) : int = Unchecked.defaultof<int>

[<FidelityExtern("pthread", "pthread_mutex_destroy")>]
let pthread_mutex_destroy (mutex: nativeint) : int = Unchecked.defaultof<int>

[<FidelityExtern("pthread", "pthread_cond_init")>]
let pthread_cond_init (cond: nativeint) (attr: nativeint) : int = Unchecked.defaultof<int>

[<FidelityExtern("pthread", "pthread_cond_wait")>]
let pthread_cond_wait (cond: nativeint) (mutex: nativeint) : int = Unchecked.defaultof<int>

[<FidelityExtern("pthread", "pthread_cond_signal")>]
let pthread_cond_signal (cond: nativeint) : int = Unchecked.defaultof<int>
```

## 5. Composer/Alex Layer Implementation

### 5.1 Mutex.create Witness

The witness allocates opaque storage and calls `pthread_mutex_init` via ExternCall:

```mlir
// Mutex.create() generates:
%mutex = memref.alloca() : memref<40xi8>  // sizeof(pthread_mutex_t)
%mutex_ptr = memref.extract_aligned_pointer_as_index %mutex : memref<40xi8> -> index
%null = arith.constant 0 : index
func.call @pthread_mutex_init(%mutex_ptr, %null) : (index, index) -> i32
```

### 5.2 Mutex.lock/unlock Witness

```mlir
// Mutex.lock mutex
func.call @pthread_mutex_lock(%mutex_ptr) : (index) -> i32

// ... critical section ...

// Mutex.unlock mutex
func.call @pthread_mutex_unlock(%mutex_ptr) : (index) -> i32
```

### 5.3 pthread declarations via ExternCall

```mlir
// Generated from [<FidelityExtern>] metadata — same pathway as D-01
func.func private @pthread_mutex_init(index, index) -> i32 attributes { "link" = "pthread" }
func.func private @pthread_mutex_lock(index) -> i32 attributes { "link" = "pthread" }
func.func private @pthread_mutex_unlock(index) -> i32 attributes { "link" = "pthread" }
func.func private @pthread_mutex_destroy(index) -> i32 attributes { "link" = "pthread" }
func.func private @pthread_cond_init(index, index) -> i32 attributes { "link" = "pthread" }
func.func private @pthread_cond_wait(index, index) -> i32 attributes { "link" = "pthread" }
func.func private @pthread_cond_signal(index) -> i32 attributes { "link" = "pthread" }
```

## 6. Validation

### 6.1 Sample Code

```clef
module MutexSyncSample

let mutable sharedCounter = 0

let incrementer (mutex: Mutex) (count: int) () =
    for _ in 1..count do
        Mutex.lock mutex
        sharedCounter <- sharedCounter + 1
        Mutex.unlock mutex

[<EntryPoint>]
let main _ =
    Console.writeln "=== Mutex Sync Test ==="

    let mutex = Mutex.create ()

    // Without mutex (race condition)
    sharedCounter <- 0
    let t1_unsafe = Thread.create (fun () ->
        for _ in 1..100000 do
            sharedCounter <- sharedCounter + 1)
    let t2_unsafe = Thread.create (fun () ->
        for _ in 1..100000 do
            sharedCounter <- sharedCounter + 1)
    Thread.join t1_unsafe
    Thread.join t2_unsafe
    Console.write "Without mutex (expect < 200000): "
    Console.writeln (Format.int sharedCounter)

    // With mutex (correct)
    sharedCounter <- 0
    let t1_safe = Thread.create (incrementer mutex 100000)
    let t2_safe = Thread.create (incrementer mutex 100000)
    Thread.join t1_safe
    Thread.join t2_safe
    Console.write "With mutex (expect 200000): "
    Console.writeln (Format.int sharedCounter)

    Mutex.destroy mutex
    0
```

### 6.2 Expected Output

```
=== Mutex Sync Test ===
Without mutex (expect < 200000): 156789  (varies, usually < 200000)
With mutex (expect 200000): 200000
```

## 7. Implementation Checklist

### Phase 1: CCS Foundation
- [ ] Add TMutex, TCondVar types to NativeTypes.fs
- [ ] Add Mutex/CondVar intrinsics to CheckExpressions.fs

### Phase 2: Binding Generation
- [ ] Confirm `pthread.pilot.toml` includes mutex/condvar functions
- [ ] Run `farscape generate` — verify Fidelity.Pthread output

### Phase 3: Alex Implementation
- [ ] Implement Mutex.create witness (alloc + ExternCall pthread_mutex_init)
- [ ] Implement Mutex.lock/unlock witnesses (ExternCall pthread_mutex_lock/unlock)
- [ ] Implement CondVar witnesses

### Phase 4: Validation
- [ ] Sample 28 compiles
- [ ] Race condition visible without mutex
- [ ] Correct result with mutex

## 8. Design Decision: Why Mutex.create is an Intrinsic

Same reasoning as T-01's `Thread.create`:
1. **Opaque storage sizing**: `pthread_mutex_t` is 40 bytes on Linux, different on other platforms. The compiler must know the size.
2. **Destruction guarantees**: Future integration with linear types (A-04) may require compiler tracking of mutex lifecycle.
3. **Platform abstraction**: The `Mutex` type abstracts over platform differences. The raw pthread calls go through ExternCall; the abstraction is compiler-managed.

The raw `pthread_mutex_*` functions are library functions (ExternCall via Farscape). The `Mutex.*` wrapper API is a compiler intrinsic.

## 9. Related PRDs

- **T-01**: BasicThread — threading foundation
- **D-01**: GTKWindow — establishes ExternCall pathway used by pthread bindings
- **T-03-31**: MailboxProcessor — uses mutex for message queue
- **A-04**: BasicRegion — linear types may govern mutex lifecycle
