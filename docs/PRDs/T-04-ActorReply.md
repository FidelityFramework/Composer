# T-04: Actor PostAndReply

> **Surface note (2026-09).** `nativeptr<'T>`, `NativePtr.*`, `voidptr`, and `FSharp.NativeInterop` are not denotable in Clef source (spec `ffi-boundary.md` §1, `special-attributes-and-types.md`; `TNativePtr` is compiler-internal only). Where this PRD shows them, it records the pre-strip surface the code was written against; the settled surfaces are the opaque `Ptr<'T, 'Region, 'Access>` handle in the interior and `CHandle<'T>` at the C boundary, with buffers as bounded arrays and captures as `memref` views.

> **Sample**: `30_ActorReply` | **Status**: Planned | **Depends On**: T-03 (BasicActor)

## 1. Executive Summary

This PRD adds two-way communication to MailboxProcessor via `PostAndReply`. The caller sends a message and blocks until the actor replies. This enables request-response patterns.

**Key Insight**: `PostAndReply` bundles a reply channel with the message. The reply channel contains a condvar that the caller waits on and the actor signals.

## 2. Language Feature Specification

### 2.1 PostAndReply

```fsharp
let result = actor.PostAndReply(fun reply ->
    GetValue reply)
```

The lambda receives an `AsyncReplyChannel<'Reply>` and constructs the message.

### 2.2 AsyncReplyChannel

```fsharp
type AsyncReplyChannel<'Reply> =
    member Reply : 'Reply -> unit
```

The actor calls `reply.Reply(value)` to send the response.

### 2.3 Complete Example

```fsharp
type BankMessage =
    | Deposit of int
    | Withdraw of int * AsyncReplyChannel<Result<int, string>>
    | GetBalance of AsyncReplyChannel<int>

let bank = MailboxProcessor.Start(fun inbox ->
    let rec loop balance = async {
        let! msg = inbox.Receive()
        match msg with
        | Deposit amount ->
            return! loop (balance + amount)
        | Withdraw (amount, reply) ->
            if amount > balance then
                reply.Reply(Error "Insufficient funds")
                return! loop balance
            else
                reply.Reply(Ok (balance - amount))
                return! loop (balance - amount)
        | GetBalance reply ->
            reply.Reply(balance)
            return! loop balance
    }
    loop 0)

// Usage
bank.Post(Deposit 100)
let balance = bank.PostAndReply(fun reply -> GetBalance reply)
```

## 3. CCS Layer Implementation

### 3.1 AsyncReplyChannel Type

```fsharp
// In NativeTypes.fs
| TAsyncReplyChannel of replyType: NativeType
```

### 3.2 PostAndReply Intrinsic

```fsharp
// In CheckExpressions.fs
| "MailboxProcessor.PostAndReply" ->
    // MailboxProcessor<'Msg> -> (AsyncReplyChannel<'Reply> -> 'Msg) -> 'Reply
    let msgVar = freshTypeVar ()
    let replyVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TMailboxProcessor(msgVar),
        NativeType.TFun(
            NativeType.TFun(NativeType.TAsyncReplyChannel(replyVar), msgVar),
            replyVar))

| "AsyncReplyChannel.Reply" ->
    // AsyncReplyChannel<'Reply> -> 'Reply -> unit
    let replyVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TAsyncReplyChannel(replyVar),
        NativeType.TFun(replyVar, env.Globals.UnitType))
```

## 4. Source Settlement and Passive Witnessing

CCS/Baker owns reply-channel identity, message-builder activation, publication,
waiting, result delivery, completion and cleanup. It settles the actual actor,
channel, result storage and participating call occurrences, including their
lifetime and synchronization premises.

The source-owned synchronous RPC and liveness contracts govern blocking waits.
Owning recipes express the protocol and callable declarations in the PSG before
publication; a witness cannot infer a wait loop or choose result-slot residence.

Alex witnesses immutable settled control, storage and calls. Composer's backend
realizes the selected target's synchronization operations while preserving the
source-to-artifact correspondence. Missing channel or lifetime settlement fails
at its source owner.

## 5. MLIR Output Specification

### 5.1 Reply Channel Type

```mlir
!reply_channel = !llvm.struct<(
    ptr,      // result slot
    !mutex,   // mutex
    !condvar, // condvar
    i1        // completed
)>
```

### 5.2 PostAndReply Implementation

```mlir
// result = actor.PostAndReply(fun reply -> GetBalance reply)

// Allocate channel
%channel = llvm.alloca 1 x !reply_channel
%result_slot = llvm.alloca 1 x i32
%slot_ptr = llvm.getelementptr %channel[0, 0]
llvm.store %result_slot, %slot_ptr

// Init sync primitives
%mutex = llvm.getelementptr %channel[0, 1]
llvm.call @pthread_mutex_init(%mutex, %null)
%cond = llvm.getelementptr %channel[0, 2]
llvm.call @pthread_cond_init(%cond, %null)
%completed = llvm.getelementptr %channel[0, 3]
llvm.store %false, %completed

// Build and post message
%msg = llvm.call %msgBuilder_code(%msgBuilder_env, %channel)
llvm.call @actor_post(%actor, %msg)

// Wait for reply
llvm.call @pthread_mutex_lock(%mutex)
llvm.br ^check

^check:
    %done = llvm.load %completed : i1
    llvm.cond_br %done, ^got_reply, ^wait

^wait:
    llvm.call @pthread_cond_wait(%cond, %mutex)
    llvm.br ^check

^got_reply:
    llvm.call @pthread_mutex_unlock(%mutex)
    %result = llvm.load %result_slot : i32
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module ActorReplySample

type CalcMessage =
    | Add of int * int * AsyncReplyChannel<int>
    | Multiply of int * int * AsyncReplyChannel<int>

let calculator = MailboxProcessor.Start(fun inbox ->
    async {
        while true do
            let! msg = inbox.Receive()
            match msg with
            | Add (a, b, reply) ->
                reply.Reply(a + b)
            | Multiply (a, b, reply) ->
                reply.Reply(a * b)
    })

[<EntryPoint>]
let main _ =
    Console.writeln "=== Actor Reply Test ==="

    let sum = calculator.PostAndReply(fun reply -> Add(10, 20, reply))
    Console.write "10 + 20 = "
    Console.writeln (Format.int sum)

    let product = calculator.PostAndReply(fun reply -> Multiply(6, 7, reply))
    Console.write "6 * 7 = "
    Console.writeln (Format.int product)

    let sum2 = calculator.PostAndReply(fun reply -> Add(sum, product, reply))
    Console.write "(10+20) + (6*7) = "
    Console.writeln (Format.int sum2)

    0
```

### 6.2 Expected Output

```
=== Actor Reply Test ===
10 + 20 = 30
6 * 7 = 42
(10+20) + (6*7) = 72
```

## 7. Files to Create/Modify

### 7.1 CCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TAsyncReplyChannel |
| `CheckExpressions.fs` | MODIFY | Add PostAndReply, Reply intrinsics |

### 7.2 Composer

| File | Action | Purpose |
|------|--------|---------|
| Alex actor witnesses | MODIFY | Passively compose settled reply-channel operations |

## 8. Implementation Checklist

### Phase 1: CCS/Baker Reply-Channel Settlement
- [ ] Add TAsyncReplyChannel type
- [ ] Implement Reply intrinsic
- [ ] Generate reply channel struct

### Phase 2: CCS/Baker PostAndReply Construction
- [ ] Implement PostAndReply intrinsic
- [ ] Implement blocking wait
- [ ] Implement channel cleanup
- [ ] Publish immutable facts for passive reply-channel witnesses

### Phase 3: Validation
- [ ] Sample 30 compiles
- [ ] Replies return correct values
- [ ] No deadlocks or races
- [ ] Samples 01-29 still pass

## 9. Timeout Support (Future)

```fsharp
// With timeout
let result = actor.TryPostAndReply(
    (fun reply -> GetValue reply),
    timeout = 1000)  // ms
// Returns: Some value | None (timeout)
```

This requires timer integration - deferred to future work.

## 10. Related PRDs

- **T-03**: BasicActor - Foundation
- **T-05**: ParallelActors - Multiple actors with replies
