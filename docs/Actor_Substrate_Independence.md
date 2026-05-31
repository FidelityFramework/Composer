# Actor Substrate Independence

**SpeakEZ Technologies | Fidelity Framework**
**March 2026 | Companion to JavaScript Backend Design**

## 1. The Principle

A developer writes an actor once, in Clef. The compilation target determines how that actor is expressed at runtime. The actor's message types, `Handle` dispatch, supervision relationships, persistence model, and observability instrumentation are identical across substrates. The `.fidproj` target declaration and the substrate library opened in the project are the only variables.

This document traces a single actor definition through two compilation paths: native (MLIR/LLVM) and Cloudflare (JavaScript). The purpose is to demonstrate that the source code is the same, the PSG representation is the same, and only the witness emission diverges.

## 2. The Actor Source

One file. One type definition. No substrate-specific imports.

```clef
open Fidelity.Actor

type CounterMsg =
    | Increment
    | Add of amount: int
    | GetCount

type CounterActor() =
    inherit Olivier<CounterMsg>()

    let mutable count = 0

    override this.OnActivate() = async {
        let! stored = this.Storage.get<int>("count")
        count <- stored |> Option.defaultValue 0
    }

    override this.Handle(msg) = async {
        match msg with
        | Increment ->
            count <- count + 1
        | Add n ->
            count <- count + n
        | GetCount ->
            this.Reply(count)
    }

    override this.OnStop() = async {
        do! this.Storage.put("count", count)
    }
```

The caller:

```clef
open Fidelity.Actor

let counter : ActorRef<CounterMsg> = actors.Get("user-123")
counter.Tell(Increment)
counter.Tell(Add 5)
let! count = counter.Ask(GetCount)
```

There is no `open Fidelity.CloudEdge` here. There is no `open Fidelity.Native` here. The actor code uses `Fidelity.Actor` exclusively. The substrate library is opened in the project configuration, not in the actor source.

## 3. The Project Configurations

### 3.1 Cloudflare Target

```toml
# myapp-edge.fidproj
[package]
name = "MyApp.Edge"

[compilation]
target = "js"

[dependencies]
actor = { path = "../Fidelity.Actor" }
cloudedge = { path = "../Fidelity.CloudEdge" }

[build]
sources = ["CounterActor.clef", "Main.clef"]
output = "counter-worker"
```

Opening `Fidelity.CloudEdge` as a dependency tells Composer that the actor substrate is Cloudflare Durable Objects. The JS witnesses in Alex know how to emit DO-backed actor infrastructure when they see `Olivier<'Msg>` patterns in the PSG with CloudEdge bindings available in scope.

### 3.2 Native Target

```toml
# myapp-native.fidproj
[package]
name = "MyApp.Native"

[compilation]
target = "cpu"

[dependencies]
actor = { path = "../Fidelity.Actor" }
platform = { path = "../Fidelity.Platform" }

[build]
sources = ["CounterActor.clef", "Main.clef"]
output = "counter-service"
```

Opening `Fidelity.Platform` as a dependency tells Composer that the actor substrate is native OS processes with IPC. The MLIR witnesses in Alex emit delimited continuations, IPC channel operations, and native memory access.

### 3.3 Cross-Substrate

```toml
# myapp-bridge.fidproj
[package]
name = "MyApp.Bridge"

[compilation]
target = "cpu"

[dependencies]
actor = { path = "../Fidelity.Actor" }
platform = { path = "../Fidelity.Platform" }
cloudedge = { path = "../Fidelity.CloudEdge" }

[build]
sources = ["Supervisor.clef", "Bridge.clef"]
output = "bridge-service"
```

When both are present, the native binary can hold `ActorRef<'Msg>` references to both local native actors and remote CloudEdge actors. BAREWire frames cross the substrate boundary over WebSocket. The bridge Worker on the Cloudflare side translates between the WebSocket transport and the DO-internal WebSocket mesh.

## 4. Fidelity.Actor: The Substrate-Independent Layer

`Fidelity.Actor` defines the types that the developer writes against. These types carry no substrate commitment. They are pure semantic declarations.

```clef
/// The worker actor base type.
/// Substrate-independent. Compiled differently per target.
[<AbstractClass>]
type Olivier<'Msg>() =

    /// Called on first activation or post-eviction recovery.
    abstract member OnActivate : unit -> Async<unit>
    default _.OnActivate() = async { return () }

    /// Called for each inbound message, sequentially.
    abstract member Handle : 'Msg -> Async<unit>

    /// Called when the supervisor issues a stop directive.
    abstract member OnStop : unit -> Async<unit>
    default _.OnStop() = async { return () }

    /// Called when Handle throws an unhandled exception.
    abstract member OnError : exn -> Async<unit>
    default _.OnError(e) = async { return () }

    /// Reply to the current ask message (correlation-based).
    member this.Reply<'T>(value: 'T) : unit = ...

    /// Access to substrate-provided storage.
    /// On Cloudflare: DO transactional storage.
    /// On native: configurable backend (file, SQLite, external store).
    member this.Storage : IActorStorage = ...

    /// Pending message count (diagnostic, not for flow control).
    member this.PendingCount : int = ...

    /// Schedule an alarm (DO alarm on Cloudflare, timer on native).
    member this.ScheduleAlarm(delay: TimeSpan) : unit = ...

    /// Default timeout for ask operations originating from this actor.
    member val DefaultTimeout : TimeSpan = TimeSpan.FromSeconds(30.0)
```

```clef
/// Typed reference to an actor. Substrate-resolved at runtime.
type ActorRef<'Msg> =
    | Local of localRef: ILocalActorRef<'Msg>
    | Edge of actorId: string * transport: IActorTransport
    | Remote of endpoint: Uri * actorId: string

    /// Fire-and-forget message send.
    member this.Tell(msg: 'Msg) : unit = ...

    /// Request-response with correlation ID.
    member this.Ask<'Reply>(msg: 'Msg) : Async<'Reply> = ...

    /// Request-response with explicit timeout.
    member this.Ask<'Reply>(msg: 'Msg, timeout: TimeSpan) : Async<'Reply> = ...
```

```clef
/// Supervisor actor base type.
type Prospero<'Msg>() =
    inherit Olivier<'Msg>()

    /// Child registry with supervision metadata.
    member this.Children : ChildRegistry = ...

    /// Called when a supervised child fails.
    abstract member OnChildFailed : childId: string * error: exn -> Async<SupervisionDirective>

    /// Called when a supervised child stops.
    abstract member OnChildStopped : childId: string -> Async<unit>

type SupervisionStrategy = OneForOne | OneForAll | RestForOne
type SupervisionDirective = Restart | Stop | Escalate

type ChildSpec<'Msg> = {
    ActorType: System.Type
    Strategy: SupervisionStrategy
    Scaling: ScalingPolicy
}

type ScalingPolicy =
    | SingleInstance
    | Elastic of ElasticConfig

type ElasticConfig = {
    MinReplicas: int
    MaxReplicas: int
    QueueThreshold: int
    ReplicaStrategy: ReplicaStrategy
}

type ReplicaStrategy =
    | Isolate
    | DurableObject
```

```clef
/// BAREWire serialization interface.
/// Generated at compile time for message DUs.
type IBARECodec<'T> =
    abstract member Encode : IBuffer -> 'T -> unit
    abstract member Decode : Span<byte> -> byref<int> -> 'T
```

None of these types reference Cloudflare, Durable Objects, WebSocket, MLIR, IPC, or any substrate-specific concept. They are the actor vocabulary. The substrate fills in the implementation.

## 5. PSG Representation

CCS compiles the `CounterActor` source into a PSG. The PSG for the actor looks the same regardless of target. The key nodes:

```
Module "CounterActor"
  ├── UnionType "CounterMsg"
  │     ├── Case "Increment" (no payload)
  │     ├── Case "Add" (payload: int)
  │     └── Case "GetCount" (no payload)
  │
  ├── ClassType "CounterActor"
  │     ├── Inherits: Olivier<CounterMsg>
  │     ├── MutableLocal "count" : int = 0
  │     │
  │     ├── Override "OnActivate" : Async<unit>
  │     │     └── LetBang (Storage.get<int>("count"))
  │     │           └── Assignment: count <- defaultValue 0
  │     │
  │     ├── Override "Handle" : CounterMsg -> Async<unit>
  │     │     └── Match msg
  │     │           ├── Case Increment → Assignment: count <- count + 1
  │     │           ├── Case Add n → Assignment: count <- count + n
  │     │           └── Case GetCount → Call: this.Reply(count)
  │     │
  │     └── Override "OnStop" : Async<unit>
  │           └── DoBang (Storage.put("count", count))
  │
  └── Module "Caller"
        ├── Let "counter" : ActorRef<CounterMsg> = actors.Get("user-123")
        ├── Call: counter.Tell(Increment)
        ├── Call: counter.Tell(Add 5)
        └── LetBang: count = counter.Ask(GetCount)
```

Baker saturates this PSG. For both targets, Baker elaborates:

- The `match msg` into conditional dispatch on the DU tag
- The `async { }` computation expressions into continuation structures
- The `this.Reply(count)` into the appropriate reply mechanism (correlation-based)

Baker does **not** expand the `Olivier<'Msg>` inheritance, the `Handle` override dispatch, or the lifecycle hooks. These are structural patterns that the witnesses need to recognize.

## 6. JavaScript Witness Emission

When `target = "js"` and `Fidelity.CloudEdge` is in scope, Alex's JavaScript witnesses observe the saturated PSG and emit a Cloudflare Worker with a Durable Object.

### 6.1 CounterMsg → BAREWire Codec

The `UnionType "CounterMsg"` node triggers the BAREWire codec witness:

```javascript
// Generated: CounterMsg BAREWire serialization
const CounterMsg = {
    Increment: 0,
    Add: 1,
    GetCount: 2,

    encode(buffer, msg) {
        buffer.writeUint8(msg.tag);
        switch (msg.tag) {
            case 1: // Add
                buffer.writeVarint(msg.amount);
                break;
        }
    },

    decode(data, offset) {
        const tag = data[offset.value++];
        switch (tag) {
            case 0: return { tag: 0 }; // Increment
            case 1: return { tag: 1, amount: readVarint(data, offset) }; // Add
            case 2: return { tag: 2 }; // GetCount
        }
    }
};
```

### 6.2 CounterActor → Durable Object Class

The `ClassType "CounterActor"` with `Inherits: Olivier<CounterMsg>` triggers the Olivier witness, which emits a Durable Object class:

```javascript
// Generated: CounterActor as Durable Object
export class CounterActor {
    constructor(state, env) {
        this.state = state;
        this.env = env;
        this.count = 0;
    }

    // OnActivate → constructor + blockConcurrencyWhile
    async onActivate() {
        await this.state.blockConcurrencyWhile(async () => {
            const stored = await this.state.storage.get("count");
            this.count = stored ?? 0;
        });
    }

    // fetch → HTTP ingress (ask semantics)
    async fetch(request) {
        // External boundary: translate HTTP to internal message
        const upgradeHeader = request.headers.get("Upgrade");
        if (upgradeHeader === "websocket") {
            const pair = new WebSocketPair();
            this.state.acceptWebSocket(pair[1]);
            return new Response(null, { status: 101, webSocket: pair[0] });
        }
        // HTTP ask: deserialize, handle, respond
        const body = await request.arrayBuffer();
        const msg = CounterMsg.decode(new Uint8Array(body), { value: 0 });
        return await this.handleWithResponse(msg);
    }

    // Handle → webSocketMessage dispatch
    async webSocketMessage(ws, message) {
        const data = new Uint8Array(
            typeof message === "string"
                ? new TextEncoder().encode(message)
                : message
        );
        // Parse BAREWire frame header
        const frameLen = readUint32LE(data, 0);
        const offset = { value: 4 };
        const msgTag = readVarint(data, offset);
        const correlationId = readOptionalUint32(data, offset);
        const msg = CounterMsg.decode(data, offset);

        // Sequential dispatch (DO single-concurrency guarantee)
        await this.handle(msg, ws, correlationId);
    }

    // Handle(msg) → the developer's match expression
    async handle(msg, ws, correlationId) {
        switch (msg.tag) {
            case 0: // Increment
                this.count += 1;
                break;
            case 1: // Add
                this.count += msg.amount;
                break;
            case 2: // GetCount
                // Reply via correlation ID
                if (correlationId !== null && ws) {
                    const reply = encodeReply(correlationId, this.count);
                    ws.send(reply);
                }
                break;
        }
    }

    // OnStop → webSocketClose
    async webSocketClose(ws, code, reason, wasClean) {
        await this.state.storage.put("count", this.count);
    }

    // OnError → error propagation to supervisor
    async webSocketError(ws, error) {
        // Default: propagate to supervising Prospero
        console.error("Actor error:", error);
    }
}
```

### 6.3 ActorRef.Tell → WebSocket Frame Send

The `Call: counter.Tell(Increment)` node triggers the Tell witness:

```javascript
// Generated: Tell dispatch
function tell(actorRef, msg) {
    const frame = new Uint8Array(64);
    const buf = new BAREBuffer(frame);
    buf.writeUint32LE(0); // placeholder for frame length
    CounterMsg.encode(buf, msg);
    // No correlation ID (tell semantics)
    buf.setUint32LE(0, buf.position); // write actual frame length
    actorRef.ws.send(frame.subarray(0, buf.position));
}

tell(counter, { tag: 0 }); // Increment
tell(counter, { tag: 1, amount: 5 }); // Add 5
```

### 6.4 ActorRef.Ask → Correlation-Based Promise

The `LetBang: count = counter.Ask(GetCount)` node triggers the Ask witness:

```javascript
// Generated: Ask dispatch
function ask(actorRef, msg, timeout = 30000) {
    return new Promise((resolve, reject) => {
        const correlationId = actorRef.nextCorrelationId++;
        actorRef.pendingReplies.set(correlationId, { resolve, reject });

        const frame = new Uint8Array(64);
        const buf = new BAREBuffer(frame);
        buf.writeUint32LE(0); // placeholder
        CounterMsg.encode(buf, msg);
        buf.writeUint32(correlationId); // correlation ID present
        buf.setUint32LE(0, buf.position);
        actorRef.ws.send(frame.subarray(0, buf.position));

        setTimeout(() => {
            if (actorRef.pendingReplies.has(correlationId)) {
                actorRef.pendingReplies.delete(correlationId);
                reject(new Error("Ask timeout"));
            }
        }, timeout);
    });
}

const count = await ask(counter, { tag: 2 }); // GetCount
```

## 7. MLIR Witness Emission

When `target = "cpu"` and `Fidelity.Platform` is in scope, Alex's MLIR witnesses observe the same saturated PSG and emit native actor infrastructure.

### 7.1 CounterMsg → BAREWire Codec (Native)

The same DU produces a native BAREWire encoder/decoder. The encoding is byte-identical to the JavaScript version:

```mlir
// CounterMsg.encode: tag byte + optional varint payload
func.func @CounterMsg_encode(%buf: memref<?xi8>, %msg_tag: i8, %msg_payload: i64)
    -> index {
    // Write tag byte
    memref.store %msg_tag, %buf[%offset] : memref<?xi8>
    // Conditional payload write for Add case (tag 1)
    // ... varint encoding ...
}
```

### 7.2 CounterActor → Delimited Continuation

The actor's `Handle` becomes a delimited continuation that suspends when waiting for messages and resumes when one arrives via the IPC channel:

```mlir
// Actor message loop as structured control flow
func.func @CounterActor_loop(%state: memref<1xi64>, %channel: !ipc.channel)
    -> () {
    scf.while : () -> () {
        // Suspend: wait for message on IPC channel
        %msg = ipc.receive %channel : !barewire.frame
        %tag = barewire.read_tag %msg : i8
        scf.yield %true : i1
    } do {
        // Dispatch on tag
        scf.if %tag_is_increment {
            %count = memref.load %state[%zero] : memref<1xi64>
            %new = arith.addi %count, %one : i64
            memref.store %new, %state[%zero] : memref<1xi64>
        } else {
            scf.if %tag_is_add {
                // ... decode payload, add to count ...
            } else {
                // GetCount: send reply on correlation channel
                // ... encode count, send on reply channel ...
            }
        }
        scf.yield
    }
}
```

### 7.3 ActorRef.Tell → IPC Channel Write

```mlir
// Tell: encode BAREWire frame, write to IPC channel
func.func @tell(%channel: !ipc.channel, %msg_tag: i8, %payload: i64) {
    %frame = memref.alloca() : memref<64xi8>
    func.call @CounterMsg_encode(%frame, %msg_tag, %payload) -> index
    ipc.send %channel, %frame : !ipc.channel, memref<64xi8>
    // Returns immediately. No reply expected.
}
```

### 7.4 ActorRef.Ask → Continuation Suspend

```mlir
// Ask: encode frame with correlation ID, suspend until reply
func.func @ask(%channel: !ipc.channel, %reply_channel: !ipc.channel,
               %msg_tag: i8, %correlation_id: i32) -> i64 {
    %frame = memref.alloca() : memref<64xi8>
    func.call @CounterMsg_encode_with_correlation(
        %frame, %msg_tag, %correlation_id) -> index
    ipc.send %channel, %frame : !ipc.channel, memref<64xi8>
    // Suspend: delimited continuation yields here
    %reply_frame = ipc.receive %reply_channel : !barewire.frame
    %result = barewire.read_i64 %reply_frame : i64
    return %result : i64
}
```

## 8. The BAREWire Invariant

The binary encoding on the wire is identical across substrates. A `CounterMsg.Increment` serialized by the JavaScript actor:

```
04 00 00 00    frame length (4 bytes, little-endian)
00             tag: Increment (varint 0)
               no correlation ID (tell)
               no payload
```

A `CounterMsg.Increment` serialized by the native actor:

```
04 00 00 00    frame length (4 bytes, little-endian)
00             tag: Increment (varint 0)
               no correlation ID (tell)
               no payload
```

Byte-identical. This is what enables the `Remote` case of `ActorRef<'Msg>`: a native Prospero supervising a Cloudflare Olivier, with BAREWire frames on WebSocket as the bridge. The native process sends a frame; the Cloudflare Worker receives it; the bytes are the same.

## 9. Supervision Across Substrates

A Prospero supervisor on native can manage Olivier workers on Cloudflare:

```clef
open Fidelity.Actor

type SessionSupervisor() =
    inherit Prospero<SupervisorMsg>()

    override this.OnActivate() = async {
        // Spawn local workers for compute-heavy tasks
        let localWorker = this.SpawnLocal<ComputeWorker>("compute-1")

        // Reference remote CloudEdge workers for edge tasks
        let edgeWorker = this.SpawnRemote<EdgeWorker>(
            "edge-1",
            Uri("wss://edge-bridge.example.com"))

        // Both are ActorRef<'Msg>. Tell/Ask works identically.
        localWorker.Tell(StartCompute data)
        edgeWorker.Tell(CacheAtEdge key)
    }

    override this.OnChildFailed(childId, error) = async {
        // Same supervision logic regardless of substrate
        return Restart
    }
```

The `SpawnLocal` call creates an in-process actor via the native substrate. The `SpawnRemote` call creates an `ActorRef` pointing at a Cloudflare Worker via WebSocket. The Prospero's supervision logic (restart, stop, escalate) is identical for both. The WebSocket connection to the bridge Worker is the supervision channel for the remote child; `webSocketClose` signals child failure, just as process termination signals failure for the local child.

## 10. What the Substrate Library Provides

`Fidelity.CloudEdge` is not an actor framework. It is the set of substrate bindings that teach Composer's JavaScript witnesses how to map actor semantics onto Cloudflare's runtime:

| Actor Concept | `Fidelity.Actor` (abstract) | `Fidelity.CloudEdge` (substrate) |
|:--|:--|:--|
| Actor identity | `ActorRef<'Msg>` | `DurableObjectId`, `DurableObjectNamespace` |
| Message transport | `Tell` / `Ask` | WebSocket binary frames |
| Single-concurrency | Sequential `Handle` dispatch | DO runtime scheduler enforcement |
| State storage | `IActorStorage` | `DurableObjectStorage` (transactional KV) |
| Activation | `OnActivate` | DO first-access activation |
| Hibernation | Implicit (no messages → idle) | DO hibernation API (WebSocket survives eviction) |
| Supervision channel | Typed connection to parent | WebSocket connection to Prospero DO |
| Lifecycle signals | `OnStop`, `OnError` | `webSocketClose`, exception propagation |
| Elastic scaling | `ScalingPolicy.Elastic` | Cloudflare Queues + Worker Loader isolates |
| Alarms / timers | `ScheduleAlarm` | DO Alarm API |
| Observability | `IActorMetrics`, `IActorDiagnostics` | Analytics Engine, Diagnostics Channel |

`Fidelity.Platform` provides the equivalent mapping for native:

| Actor Concept | `Fidelity.Actor` (abstract) | `Fidelity.Platform` (substrate) |
|:--|:--|:--|
| Actor identity | `ActorRef<'Msg>` | Process ID, Unix socket path |
| Message transport | `Tell` / `Ask` | IPC channel (Unix socket, shared memory) |
| Single-concurrency | Sequential `Handle` dispatch | Event loop / delimited continuation |
| State storage | `IActorStorage` | Configurable backend (file, SQLite, external) |
| Activation | `OnActivate` | Process start / first message |
| Hibernation | Implicit | Process sleep / signal wait |
| Supervision channel | Typed connection to parent | IPC channel to supervisor process |
| Lifecycle signals | `OnStop`, `OnError` | SIGTERM, process exit code |
| Elastic scaling | `ScalingPolicy.Elastic` | Process fork / thread pool |
| Alarms / timers | `ScheduleAlarm` | OS timer (timerfd, kqueue) |
| Observability | `IActorMetrics`, `IActorDiagnostics` | OpenTelemetry, structured logging |

The developer never sees these tables. They write `Olivier<'Msg>` with `Handle(msg)`. The `.fidproj` opens the substrate library. Composer's witnesses do the rest.
