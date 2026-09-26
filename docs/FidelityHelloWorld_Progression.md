# FidelityHelloWorld Sample Progression

## Overview

This document retains the historical sample progression and its recorded findings. Current sample identities, delivery scope and acceptance belong to the [PRD index](PRDs/README.md), [C-series acceptance contract](PRDs/C-Series-Acceptance.md) and [coverage waypoints](Language_Coverage_Waypoints.md). A source example or historical status does not establish a fresh compiler result.

**Ownership**: CCS/Baker elaborates and saturates source semantics through nanopass ingredients and recipes, preserving scope, ordered joint premises and the intermediate rewrite record. Alex reads immutable settled codata at its actual Huet occurrence and passively composes Elements/Patterns/Witnesses. Target-specific realization belongs to Composer's backend. No analysis, inference, emitter hyperedge queries or semantic MLIR repair belongs in Alex; all custom plugins and compatibility paths are retired.

## Historical sample inventory (01-13)

### Working Samples

| Sample | Name | Features Proven | Status |
|--------|------|-----------------|--------|
| 01 | HelloWorldDirect | Static strings, direct Console.writeln calls | Working |
| 02 | HelloWorldSaturated | Let bindings, string literals, function calls | Working |
| 03 | HelloWorldHalfCurried | Pipe operators (`\|>`), partial application | Working |
| 04 | HelloWorldFullCurried | Full currying, Result.map, lambdas | Working |
| 05 | AddNumbers | Integer arithmetic, pattern matching on DUs | Working |
| 06 | AddNumbersInteractive | Console.readln, string parsing, interactive I/O | Working |
| 07 | BitsTest | Bit manipulation operators (`&&&`, `\|\|\|`, `^^^`, `<<<`, `>>>`) | Working |
| 08 | Option | Option type (Some/None), 2-element DU | Working |
| 09 | Result | Result type (Ok/Error), multi-payload DU | Working |

### Samples Needing Fixes

| Sample | Name | Issue | Root Cause |
|--------|------|-------|------------|
| 10 | Records | Record pattern matching not implemented | Compiler limitation |
| 11 | HigherOrderFunctions | Source code type errors | Source bugs (not compiler) |
| 12 | Closures | Source code type error | Source bugs (not compiler) |
| 13 | Recursion | Recursive function bindings not resolved | Compiler limitation |

## Planned Development Phases

### Phase A: Foundation Fixes

Fix existing samples that have compiler limitations.

#### A.1: Fix Sample 10 (Records)

**Problem**: Record pattern matching in match expressions not implemented.

**What needs to work**:
```fsharp
type Person = { Name: string; Age: int }

match person with
| { Name = n; Age = a } -> printfn "%s is %d" n a
```

**Owning contract**: CCS/Baker resolves record patterns, field declarations, demand, projection and representation. Alex witnesses the published field-access form without field-name inference or pattern analysis.

#### A.2: Fix Sample 12 (Closures)

**Problem**: Source code has type errors.

**Owning contract**: Preserve the reported failure and its source form. CCS/Baker settles capture identity, the actual function/environment components, typed application, layout and covering residence. Alex witnesses the published allocation and access forms; it does not construct a closure from a capture list.

#### A.3: Fix Sample 13 (Recursion)

**Problem**: Recursive function bindings not found in VarBindings.

**What needs to work**:
```fsharp
let rec factorial n =
    if n <= 1 then 1
    else n * factorial (n - 1)
```

**Owning contract**: CCS/Baker establishes recursive identities, group constraints, captures, effect/range fixed points, declaration scope and call conventions. Alex emits declarations at their settled scope and consumes the published calls; no declaration discovery or hoisting occurs during witnessing.

---

### Phase B: Sequences

Implement `seq` computation expression using MoveNext struct pattern.

CCS/Baker constructs sequence control, cut/resume relations, live-across storage, fresh enumeration and successful-current premises. Alex witnesses the settled machine through passive Huet composition; traversal does not synthesize transitions from yield syntax.

#### B.1: Sample 14 - SimpleSeq

**Features**:
- Basic `seq { }` builder
- `for ... in ... do yield` pattern
- Iterator state machine generation

**Source**:
```fsharp
let numbers = seq {
    for i in 1..10 do
        yield i
}

for n in numbers do
    Console.writeln (Format.int n)
```

**Owning contract**: CCS/Baker recipes elaborate the sequence and consumer, then settle control, demand, actual iterator identity, current-read guards, frame layout and residence. Alex consumes that immutable publication through ordinary admitted control, call and memory forms.

#### B.2: Sample 15 - SeqOperations

**Features**:
- `Seq.map`
- `Seq.filter`
- `Seq.take`

**Source**:
```fsharp
let doubled = Seq.map (fun x -> x * 2) numbers
let evens = Seq.filter (fun x -> x % 2 = 0) numbers
let first5 = Seq.take 5 numbers
```

**Owning contract**: CCS/Baker operation recipes compose the shared producer/consumer protocol, preserving deferred operands, callback identity, pull order and stopping. Frame and backing-storage premises settle before publication. Alex does not recognize combinators to synthesize wrappers or delegate algorithms.

---

### Phase C: Lazy Evaluation

Implement `lazy` thunks - frozen computation that executes at most once.

A Lazy value retains a separate thunk and typed memo environment. CCS/Baker establishes first successful force, cached reads, shared instance identity and covering storage. Its normal memoization contract is distinct from ordinary demand, concurrent force and reentry.

#### C.1: Sample 16 - LazyValues

**Features**:
- `lazy { }` expression
- `Lazy.force` / `.Value`
- Memoization

**Source**:
```fsharp
let expensive = lazy {
    Console.writeln "Computing..."
    42
}

let value1 = Lazy.force expensive  // Prints "Computing..."
let value2 = Lazy.force expensive  // No print, cached
```

**Owning contract**: CCS/Baker constructs the guard, cold computation, typed cache store, completion publication and hot read as PSG structure, with exact capture and memo-instance identities. Alex passively witnesses those operations. The witness contains no force algorithm or invented cache representation.

---

### Phase D: Async

Async admission follows the source suspension, scheduling, effect and lifetime contracts.

CCS/Baker settles suspension and resumption structure, frame storage, ownership and progress premises. A backend may realize an admitted target protocol only while preserving those facts; this document establishes no runtime-cost claim.

See: [Async_LLVM_Coroutines.md](./Async_LLVM_Coroutines.md)

#### D.1: Sample 17 - BasicAsync

**Features**:
- `async { return value }`
- `Async.RunSynchronously`

**Source**:
```fsharp
let simple = async {
    return 42
}

let result = Async.RunSynchronously simple
Console.writeln (Format.int result)
```

**Owning contract**: CCS/Baker settles the computation's activation, effects and return form. Frame elision requires source-owned premises. Alex consumes the published result without choosing an async strategy.

#### D.2: Sample 18 - AsyncAwait

**Features**:
- `let!` binding (await)
- Suspension points
- State machine with multiple states

**Source**:
```fsharp
let fetchData = async {
    return "data"
}

let process = async {
    let! data = fetchData
    return String.length data
}
```

**Owning contract**: CCS/Baker elaborates cuts, resumes, liveness, cleanup, frame layout and declaration/call relationships. Alex witnesses that settled structure; target-specific coroutine instructions belong to the backend.

#### D.3: Sample 19 - AsyncParallel

**Features**:
- `Async.Parallel`
- Multiple concurrent asyncs

**Source**:
```fsharp
let task1 = async { return 1 }
let task2 = async { return 2 }
let task3 = async { return 3 }

let results = Async.Parallel [task1; task2; task3]
              |> Async.RunSynchronously
```

**Owning contract**: CCS/Baker establishes the admitted execution, result, effect, resource and progress relationships of parallel composition. Alex witnesses published structure and cannot replace concurrency with an inferred sequential loop. The selected backend realizes the declared execution capabilities.

---

### Phase E: Scoped Regions

Implement compiler-inferred deterministic memory regions - dynamic allocation without runtime overhead.

CCS/Baker settles region ownership, scope, escape, placement, capacity and release, retaining the actual declarations and uses. Alex reads those facts. Platform-specific allocation and release APIs are backend realization under the declared storage contract.

**Stack-first proof**: Phases B-D prove Seq/Lazy/Async with stack-only allocation. Regions unlock realistic I/O workloads.

#### E.1: Sample 20 - BasicRegion

**Features**:
- `Region.create` / `Region.alloc`
- Compiler-inferred disposal at scope exit
- Bump-pointer allocation

**Source**:
```fsharp
open Fidelity.Memory

let main () =
    let region = Region.create 4  // 4 pages initial

    // Allocate in region (fast bump-pointer)
    let buffer = Region.alloc<int> region 1000

    // Fill buffer
    for i in 0..999 do
        buffer.[i] <- i * 2

    // Sum values
    let mutable sum = 0
    for i in 0..999 do
        sum <- sum + buffer.[i]

    Console.writeln ("Sum: " + Format.int sum)

    // Compiler inserts: Region.release region
```

**Expected Output**:
```
Sum: 999000
```

**Owning contract**: CCS/Baker establishes allocation and release sites, source lifetimes, initialization, capacity and cleanup on every admitted exit. Alex witnesses those explicit operations; the backend realizes the selected platform allocation protocol.

#### E.2: Sample 21 - RegionPassing

**Features**:
- Region passed to functions
- Caller scope determines lifetime
- Multiple allocations in same region

**Source**:
```fsharp
open Fidelity.Memory

let processData (r: Region) (size: int) =
    let temp = Region.alloc<float> r size
    for i in 0..(size-1) do
        temp.[i] <- float i * 1.5
    let sum = Array.fold (+) 0.0 temp
    sum

let main () =
    let region = Region.create 8

    let result1 = processData region 1000
    let result2 = processData region 500   // Same region, more allocation

    Console.writeln ("Result 1: " + Format.float result1)
    Console.writeln ("Result 2: " + Format.float result2)
    Console.writeln ("Used: " + Format.int (Region.usedBytes region) + " bytes")

    // Compiler inserts: Region.release region
```

**Owning contract**: CCS/Baker establishes borrowed authority, exact allocation extent/alignment, capacity and lifetime. Any allocation algorithm is elaborated above Alex's boundary. Alex composes the published physical operations; it supplies no allocator or storage policy.

#### E.3: Sample 22 - RegionEscape

**Features**:
- `Region.copyOut` for escaping data
- Compiler prevents implicit escape

**Source**:
```fsharp
open Fidelity.Memory

let createResult (size: int) =
    let region = Region.create 2
    let data = Region.alloc<int> region size

    for i in 0..(size-1) do
        data.[i] <- i * i

    // Must explicitly copy to escape region
    let result = Region.copyOut data

    // Compiler inserts: Region.release region
    result  // Returns copy, not region-allocated data

let main () =
    let squares = createResult 10

    for i in 0..9 do
        Console.writeln (Format.int squares.[i])
```

**Owning contract**: CCS/Baker analyzes escape and proves the admitted destination, extent, sharing and covering lifetime for any transfer. Alex reads the settled transfer and actual storage identities; it does not select a caller allocation or invent a copy to repair a lifetime.

---

### Phase F: Networking

Implement socket operations for WebSocket support. With Regions available, we can allocate proper I/O buffers.

#### F.1: Sample 23 - SocketBasics

**Features**:
- `Sys.socket` / `Sys.connect` / `Sys.bind` / `Sys.listen` / `Sys.accept`
- `Sys.read` / `Sys.write` on socket FDs
- `Sys.close`

**Source**:
```fsharp
let server () =
    let region = Region.create 2
    let buffer = Region.alloc<byte> region 1024

    let sock = Sys.socket AF_INET SOCK_STREAM 0
    Sys.bind sock addr port
    Sys.listen sock 5
    let client = Sys.accept sock
    let bytesRead = Sys.read client buffer 1024
    Sys.write client response (String.length response)
    Sys.close client
    Sys.close sock
```

**Owning contract**: CCS/Baker settles typed declarations, demand/effect order, buffer access, extent and residence. Alex witnesses the published call; the backend realizes the selected syscall ABI.

#### F.2: Sample 24 - WebSocketEcho

**Features**:
- HTTP upgrade handshake
- WebSocket frame encoding/decoding
- Echo server loop

**Source**:
```fsharp
let wsServer () =
    let sock = acceptConnection ()
    WebSocket.handshake sock
    while true do
        let frame = WebSocket.readFrame sock
        WebSocket.writeFrame sock frame
```

**Owning contract**: Protocol logic remains library source. CCS/Baker settles its calls, buffers and proof premises; Alex passively composes the published operations, with platform calls realized by the backend.

---

### Phase G: Desktop Scaffold

Implement GTK/WebView integration via FFI bindings.

#### G.1: Sample 25 - GTKWindow

**Features**:
- GTK initialization
- Window creation
- Event loop

**Source**:
```fsharp
open Fidelity.Desktop.GTK

let main () =
    GTK.init ()
    let window = GTK.windowNew "Hello GTK"
    GTK.windowShow window
    GTK.main ()
```

**Owning contract**: Farscape declarations provide typed foreign boundaries. CCS/Baker settles declaration identity, ABI and joint callback/storage premises. Alex reads the published convention; the backend realizes it without toolkit-specific semantic repair.

#### G.2: Sample 26 - WebViewBasic

**Features**:
- WebKitGTK WebView widget
- HTML content loading
- JavaScript evaluation (optional)

**Source**:
```fsharp
open Fidelity.Desktop.GTK
open Fidelity.WebView

let main () =
    GTK.init ()
    let window = GTK.windowNew "WebView Demo"
    let webview = WebView.create ()
    WebView.loadHtml webview "<h1>Hello from Fidelity!</h1>"
    GTK.containerAdd window webview
    GTK.windowShow window
    GTK.main ()
```

**Implementation**: WebKitGTK FFI bindings from Fidelity.Platform. HTML string passed as native string (UTF-8 `memref<?xi8>` view). Widget hierarchy managed through GTK container API.

---

### Phase H: Threading Primitives

Implement OS threading for true parallelism.

Threading primitives are CCS intrinsics that map to platform syscalls. The Thread coeffect marks functions that spawn threads - affecting what can be captured and how resources are managed.

#### H.1: Sample 27 - BasicThread

**Features**:
- `Thread.create` / `Thread.join`
- Parallel execution of computation

**Source**:
```fsharp
open Fidelity.Threading

let main () =
    let compute () =
        Console.writeln "Worker thread running"
        Thread.sleep 100
        Console.writeln "Worker thread done"

    Console.writeln "Main thread starting worker"
    let worker = Thread.create compute
    Console.writeln "Main thread waiting"
    Thread.join worker
    Console.writeln "All done"
```

**Expected Output**:
```
Main thread starting worker
Main thread waiting
Worker thread running
Worker thread done
All done
```

**Owning contract**: CCS/Baker settles the admitted thread entry, actual environment, shared-storage permissions, lifetime and completion/release premises. Alex witnesses the published calls and storage. Target thread APIs and adapters belong to backend realization.

#### H.2: Sample 28 - MutexSync

**Features**:
- `Mutex.create` / `Mutex.lock` / `Mutex.unlock`
- Shared mutable state with synchronization

**Source**:
```fsharp
open Fidelity.Threading

let main () =
    let mutable counter = 0
    let mutex = Mutex.create ()

    let increment () =
        for _ in 1..1000 do
            Mutex.lock mutex
            counter <- counter + 1
            Mutex.unlock mutex

    let t1 = Thread.create increment
    let t2 = Thread.create increment
    Thread.join t1
    Thread.join t2

    Console.writeln (Format.int counter)  // Should be 2000
```

**Owning contract**: CCS/Baker establishes synchronization operations and their actual shared-storage, ordering and lifetime premises. Alex passively composes those operations. The backend realizes the selected platform synchronization mechanism.

---

### Phase I: MailboxProcessor (CAPSTONE)

**MailboxProcessor is the capstone feature** - it synthesizes all prior capabilities:
- Async (for message loop) via LLVM coroutines
- Closures (for behavior function capture)
- Threading (for true parallelism via OS threads)
- **Scoped Regions (for dynamic memory in worker threads)**
- Records/DUs (for message type definitions)

This proves the compiler can handle Clef's actor model primitive with full native compilation.

**Foundational Implementation**: OS thread per actor + mutex-protected queue + LLVM coroutine for async loop. No DCont, no Olivier/Prospero supervision - just the core actor semantics. This foundation works for desktop AND embedded/MCU/unikernel targets.

#### I.1: Sample 29 - BasicActor

**Features**:
- `MailboxProcessor.Start`
- `Post` (fire-and-forget)
- `Receive` in async loop

**Source**:
```fsharp
open Fidelity.Actors

type Message =
    | Greet of string
    | Shutdown

let main () =
    let actor = MailboxProcessor.Start(fun inbox ->
        let rec loop () = async {
            let! msg = inbox.Receive()
            match msg with
            | Greet name ->
                Console.writeln ("Hello, " + name + "!")
                return! loop ()
            | Shutdown ->
                Console.writeln "Shutting down"
                return ()
        }
        loop ()
    )

    actor.Post(Greet "Alice")
    actor.Post(Greet "Bob")
    actor.Post(Shutdown)
    Thread.sleep 100  // Give actor time to process
```

**Expected Output**:
```
Hello, Alice!
Hello, Bob!
Shutting down
```

**Owning contract**: CCS/Baker elaborates admitted actor behavior, mailbox, activation and resource protocols through reusable ingredients and recipes. It settles all scheduling, layout, publication and lifetime premises. Alex consumes the resulting structure; it does not synthesize an actor from templates.

#### I.2: Sample 30 - ActorReply

**Features**:
- `PostAndReply` (request-response)
- `AsyncReplyChannel`
- Blocking wait for response

**Source**:
```fsharp
open Fidelity.Actors

type CounterMsg =
    | Increment
    | Decrement
    | GetValue of AsyncReplyChannel<int>

let main () =
    let counter = MailboxProcessor.Start(fun inbox ->
        let rec loop count = async {
            let! msg = inbox.Receive()
            match msg with
            | Increment ->
                return! loop (count + 1)
            | Decrement ->
                return! loop (count - 1)
            | GetValue reply ->
                reply.Reply(count)
                return! loop count
        }
        loop 0
    )

    counter.Post(Increment)
    counter.Post(Increment)
    counter.Post(Increment)
    counter.Post(Decrement)

    let value = counter.PostAndReply(fun reply -> GetValue reply)
    Console.writeln ("Counter value: " + Format.int value)
```

**Expected Output**:
```
Counter value: 2
```

**Owning contract**: CCS/Baker settles request/reply identity, result publication, waiting, completion and storage lifetime under the admitted protocol. Alex witnesses the explicit structure, and the backend realizes selected synchronization primitives.

#### I.3: Sample 31 - ParallelActors

**Features**:
- Multiple actors running in parallel
- Inter-actor communication via Post
- True parallelism (multiple OS threads)
- **Region-based worker memory**

**Source**:
```fsharp
open Fidelity.Actors
open Fidelity.Memory

type WorkerMsg =
    | Compute of int * MailboxProcessor<CoordinatorMsg>

type CoordinatorMsg =
    | Result of int

let main () =
    let coordinator = MailboxProcessor.Start(fun inbox ->
        let rec loop results = async {
            if List.length results >= 3 then
                let sum = List.fold (+) 0 results
                Console.writeln ("Total: " + Format.int sum)
                return ()
            else
                let! msg = inbox.Receive()
                match msg with
                | Result n -> return! loop (n :: results)
        }
        loop []
    )

    let createWorker id =
        MailboxProcessor.Start(fun inbox ->
            async {
                // Each worker has its own region for computation scratch space
                let region = Region.create 4
                let! msg = inbox.Receive()
                match msg with
                | Compute (n, reply) ->
                    let buffer = Region.alloc<int> region 100
                    // ... computation using buffer ...
                    let result = n * n
                    reply.Post(Result result)
                // Region released when actor terminates
            }
        )

    let w1 = createWorker 1
    let w2 = createWorker 2
    let w3 = createWorker 3

    w1.Post(Compute(10, coordinator))  // 100
    w2.Post(Compute(20, coordinator))  // 400
    w3.Post(Compute(30, coordinator))  // 900

    Thread.sleep 100
```

**Expected Output**:
```
Total: 1400
```

**Owning contract**: Actor scheduling and storage follow their declared source/platform contracts. CCS/Baker retains exact activation, allocation, use, completion and release participants. Alex witnesses their settled composition; neither one thread per actor nor an implicit region is selected during emission.

---

## Sample Directory Structure

```
samples/console/FidelityHelloWorld/
├── 01_HelloWorldDirect/
├── 02_HelloWorldSaturated/
├── 03_HelloWorldHalfCurried/
├── 04_HelloWorldFullCurried/
├── 05_AddNumbers/
├── 06_AddNumbersInteractive/
├── 07_BitsTest/
├── 08_Option/
├── 09_Result/
├── 10_Records/              # Needs fix
├── 11_HigherOrderFunctions/
├── 12_Closures/             # Source fix needed
├── 13_Recursion/            # Needs fix
├── 14_SimpleSeq/            # Planned (Phase B)
├── 15_SeqOperations/        # Planned (Phase B)
├── 16_LazyValues/           # Planned (Phase C)
├── 17_BasicAsync/           # Planned (Phase D)
├── 18_AsyncAwait/           # Planned (Phase D)
├── 19_AsyncParallel/        # Planned (Phase D)
├── 20_BasicRegion/          # Planned (Phase E)
├── 21_RegionPassing/        # Planned (Phase E)
├── 22_RegionEscape/         # Planned (Phase E)
├── 23_SocketBasics/         # Planned (Phase F)
├── 24_WebSocketEcho/        # Planned (Phase F)
├── 25_GTKWindow/            # Planned (Phase G)
├── 26_WebViewBasic/         # Planned (Phase G)
├── 27_BasicThread/          # Planned (Phase H)
├── 28_MutexSync/            # Planned (Phase H)
├── 29_BasicActor/           # Planned (Phase I - CAPSTONE)
├── 30_ActorReply/           # Planned (Phase I - CAPSTONE)
└── 31_ParallelActors/       # Planned (Phase I - CAPSTONE)
```

## Validation Protocol

For each sample:

1. **Build compiler**: `cd /home/hhh/repos/Composer/src && dotnet build`
2. **Compile sample**: `Composer compile <Sample>.fidproj`
3. **Execute binary**: `./<output_binary>`
4. **Verify output**: Compare with expected output
5. **Keep intermediates**: Use `-k` flag for debugging if needed

```bash
# Example validation
cd /home/hhh/repos/Composer/samples/console/FidelityHelloWorld/20_BasicRegion
/home/hhh/repos/Composer/src/bin/Debug/net10.0/Composer compile BasicRegion.fidproj
./targets/basicregion
# Expected: Sum: 999000
```

## Dependencies by Phase

| Feature | CCS/Baker settlement | Passive witnessing and backend realization |
|---|---|---|
| Records and callables | Pattern/capture identity, typed access/application, declaration scope and residence | Alex consumes published fields, calls and placements |
| Seq and Lazy | Demand, memoization or enumeration, control, typed storage and complete-use premises | Alex witnesses explicit operations; backend preserves the realized protocol |
| Async, regions and threading | Suspension, ownership, scheduling, allocation, synchronization and release premises | Alex reads immutable publication; backend realizes selected platform operations |
| Foreign and platform calls | Declaration identity, typed ABI, materialization/effect order and joint lifetime premises | Passive calls carry settled facts into backend ABI realization |
| Actors and reply protocols | Exact activation, mailbox, request/reply, resource and publication relationships | No actor synthesis or scheduling analysis in Alex |

### Capstone Feature Dependencies

```
Phase I (MailboxProcessor) - CAPSTONE
    │
    ├── requires Phase D (Async)
    │   └── LLVM coroutine for message loop
    │
    ├── requires Phase E (Scoped Regions)
    │   └── dynamic memory for worker threads
    │
    ├── requires Phase H (Threading)
    │   └── OS thread per actor
    │
    ├── requires Phase A (Closures)
    │   └── behavior function capture
    │
    └── requires Samples 05, 08, 09 (DUs)
        └── message type definitions
```

## Related Documentation

- [WRENStack_Roadmap.md](./WRENStack_Roadmap.md) - Overall architecture
- [Async_LLVM_Coroutines.md](./Async_LLVM_Coroutines.md) - Async implementation
- [CCS_Architecture.md](./CCS_Architecture.md) - Adding new intrinsics
- [Architecture_Canonical.md](./Architecture_Canonical.md) - Compiler pipeline

**Serena Memories** (architectural guidance):
- `four_pillars_of_transfer` - Coeffects, Active Patterns, Zipper, Templates
- `codata_photographer_principle` - Witness, don't construct
- `architecture_principles` - Layer separation, non-dispatch model
