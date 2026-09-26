# T-03: MailboxProcessor Basic Actor

> **Surface note (2026-09).** `nativeptr<'T>`, `NativePtr.*`, `voidptr`, and `FSharp.NativeInterop` are not denotable in Clef source (spec `ffi-boundary.md` §1, `special-attributes-and-types.md`; `TNativePtr` is compiler-internal only). Where this PRD shows them, it records the pre-strip surface the code was written against; the settled surfaces are the opaque `Ptr<'T, 'Region, 'Access>` handle in the interior and `CHandle<'T>` at the C boundary, with buffers as bounded arrays and captures as `memref` views.

> **Sample**: `29_BasicActor` | **Status**: Planned | **Depends On**: C-01-28 (All Prior Features)

## 1. Executive Summary

MailboxProcessor is the **capstone feature** of the WREN Stack - it synthesizes closures, async, threading, mutex synchronization, and regions into a single coherent abstraction. An actor is a concurrent unit with a private mailbox that processes messages sequentially.

**Key Insight**: MailboxProcessor is a composition, not a primitive. It emerges from combining existing capabilities:
- Thread (T-01) for concurrent execution
- Mutex + CondVar (T-02) for message queue synchronization
- Async (A-01-19) for message loop coroutine
- Closures (C-01) for behavior function capture
- Regions (A-04 to A-06) for per-batch memory management

**Reference**: See `mailboxprocessor_first_stage` memory for implementation strategy.

## 2. Language Feature Specification

### 2.1 Actor Creation

```fsharp
let counter = MailboxProcessor.Start(fun inbox ->
    let rec loop count = async {
        let! msg = inbox.Receive()
        match msg with
        | Increment -> return! loop (count + 1)
        | Get reply -> reply.Reply count; return! loop count
    }
    loop 0)
```

### 2.2 Posting Messages

```fsharp
counter.Post(Increment)
counter.Post(Increment)
counter.Post(Increment)
```

`Post` is asynchronous - it enqueues and returns immediately.

### 2.3 Message Type

```fsharp
type CounterMessage =
    | Increment
    | Get of AsyncReplyChannel<int>
```

Messages are discriminated unions (already supported via F-05+).

## 3. CCS Layer Implementation

### 3.1 MailboxProcessor Type

```fsharp
// In NativeTypes.fs
| TMailboxProcessor of messageType: NativeType
```

### 3.2 Inbox Type

```fsharp
// The inbox passed to behavior function
| TInbox of messageType: NativeType
```

### 3.3 MailboxProcessor Intrinsics

```fsharp
// In CheckExpressions.fs
| "MailboxProcessor.Start" ->
    // (Inbox<'Msg> -> Async<unit>) -> MailboxProcessor<'Msg>
    let msgVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TFun(
            NativeType.TInbox(msgVar),
            NativeType.TAsync(env.Globals.UnitType)),
        NativeType.TMailboxProcessor(msgVar))

| "MailboxProcessor.Post" ->
    // MailboxProcessor<'Msg> -> 'Msg -> unit
    // Note: 'this' parameter in member syntax
    let msgVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TMailboxProcessor(msgVar),
        NativeType.TFun(msgVar, env.Globals.UnitType))

| "Inbox.Receive" ->
    // Inbox<'Msg> -> Async<'Msg>
    let msgVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TInbox(msgVar),
        NativeType.TAsync(msgVar))
```

## 4. Source Settlement and Passive Witnessing

CCS/Baker constructs and settles actor activation, behavior closure calls,
queue operations, message ordering and publication, waiting, completion and
cleanup. Queue/message storage and the worker's captures retain their actual
identity, extent, capacity, ownership and lifetime relationships.

These protocols are represented in the PSG through owning ingredients and
recipes. Their joint premises cover queue state and every participating producer,
consumer and message occurrence; emitter-local pointer manipulations cannot
supply that proof.

Alex passively witnesses settled declarations, control, storage and calls.
It does not invent queue algorithms, worker wrappers, frame layouts or message
allocation. Composer's backend realizes the admitted threading and synchronization
operations for the selected target.

## 5. MLIR Output Specification

### 5.1 Actor Types

```mlir
!message_node = !llvm.struct<(ptr, i32)>  // next, data (example: int message)

!message_queue = !llvm.struct<(
    ptr,     // head
    ptr,     // tail
    !mutex,  // mutex
    !condvar // condvar
)>

!mailbox_processor = !llvm.struct<(
    !message_queue,   // queue
    i64,              // thread handle
    !closure_type     // behavior
)>
```

### 5.2 Start Implementation

```mlir
// MailboxProcessor.Start(behavior)
%actor = llvm.alloca 1 x !mailbox_processor

// Initialize queue
%queue = llvm.getelementptr %actor[0, 0]
llvm.call @queue_init(%queue)

// Store behavior
%behavior_slot = llvm.getelementptr %actor[0, 2]
llvm.store %behavior_closure, %behavior_slot

// Create thread (attr=0 means default attributes)
%thread_slot = llvm.getelementptr %actor[0, 1]
%attr_zero = llvm.mlir.zero : !llvm.ptr
llvm.call @pthread_create(%thread_slot, %attr_zero, @actor_loop, %actor)
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module BasicActorSample

type Message =
    | Increment
    | Decrement
    | Print

let counter = MailboxProcessor.Start(fun inbox ->
    let rec loop count = async {
        let! msg = inbox.Receive()
        match msg with
        | Increment ->
            Console.writeln "Incrementing"
            return! loop (count + 1)
        | Decrement ->
            Console.writeln "Decrementing"
            return! loop (count - 1)
        | Print ->
            Console.write "Count: "
            Console.writeln (Format.int count)
            return! loop count
    }
    loop 0)

[<EntryPoint>]
let main _ =
    Console.writeln "=== Basic Actor Test ==="

    counter.Post(Increment)
    counter.Post(Increment)
    counter.Post(Print)
    counter.Post(Decrement)
    counter.Post(Print)
    counter.Post(Increment)
    counter.Post(Increment)
    counter.Post(Print)

    // Give actor time to process
    Thread.sleep 100

    Console.writeln "Done"
    0
```

### 6.2 Expected Output

```
=== Basic Actor Test ===
Incrementing
Incrementing
Count: 2
Decrementing
Count: 1
Incrementing
Incrementing
Count: 3
Done
```

## 7. Files to Create/Modify

### 7.1 CCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TMailboxProcessor, TInbox |
| `CheckExpressions.fs` | MODIFY | Add MailboxProcessor intrinsics |

### 7.2 Composer

| File | Action | Purpose |
|------|--------|---------|
| Alex actor witnesses | CREATE | Passively compose settled actor operations |
| CCS/Baker actor settlement | CREATE | Settle actor/message storage, layouts, lifetimes and declarations |

## 8. Implementation Checklist

### Phase 1: Core Types
- [ ] Add TMailboxProcessor, TInbox types
- [ ] Add Start, Post, Receive intrinsics

### Phase 2: CCS/Baker Queue Construction
- [ ] Implement thread-safe message queue
- [ ] Implement enqueue (Post)
- [ ] Implement dequeue (Receive)

### Phase 3: CCS/Baker Actor Loop and Passive Witnessing
- [ ] Implement worker thread entry
- [ ] Integrate with the settled async continuation protocol
- [ ] Publish immutable facts for passive actor witnesses

### Phase 4: Validation
- [ ] Sample 29 compiles
- [ ] Messages process in order
- [ ] Actor runs on separate thread
- [ ] Samples 01-28 still pass

## 9. Why This Is the Capstone

MailboxProcessor demonstrates mastery of:

| Capability | PRD | Usage in Actor |
|------------|-----|----------------|
| Closures | 11 | Behavior function |
| HOFs | 12 | Recursive loop |
| Recursion | 13 | `let rec loop` |
| Async | 17-19 | `async { }` body |
| Regions | 20-22 | Message batch memory |
| Threading | 27 | Worker thread |
| Mutex | 28 | Queue synchronization |
| DUs | 05 | Message types |

## 10. Related PRDs

- **T-04**: PostAndReply - Two-way communication
- **T-05**: ParallelActors - Multiple interacting actors
