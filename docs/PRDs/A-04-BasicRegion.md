# A-04: Basic Region Allocation

> **Surface note (2026-09).** `nativeptr<'T>`, `NativePtr.*`, `voidptr`, and `FSharp.NativeInterop` are not denotable in Clef source (spec `ffi-boundary.md` §1, `special-attributes-and-types.md`; `TNativePtr` is compiler-internal only). Where this PRD shows them, it records the pre-strip surface the code was written against; the settled surfaces are the opaque `Ptr<'T, 'Region, 'Access>` handle in the interior and `CHandle<'T>` at the C boundary, with buffers as bounded arrays and captures as `memref` views.

> **Sample**: `20_BasicRegion` | **Status**: Planned | **Depends On**: C-01 to A-03

## 1. Executive Summary

Scoped Regions provide dynamic memory allocation with **compiler-inferred deterministic disposal**. Unlike GC-based allocation, regions have lexically-scoped lifetimes - the compiler inserts deallocation at scope exit.

**Key Insight**: Regions are MLKit-style memory management. All allocations in a region are freed together when the region is released. This is bulk deallocation - no per-object tracking, no GC pauses.

**Reference**: See `scoped_regions_architecture` memory for design details.

## 2. Language Feature Specification

### 2.1 Region Creation

```fsharp
let region = Region.create 4  // 4 pages (16KB)
```

Creates a region backed by OS virtual memory (mmap/VirtualAlloc).

### 2.2 Region Allocation

```fsharp
let buffer = Region.alloc<byte> region 1024
```

Bump-pointer allocation within the region.

### 2.3 Automatic Disposal

```fsharp
let processData () =
    let region = Region.create 4
    let temp = Region.alloc<int> region 100

    // ... use temp ...

    // Region.release region  ← COMPILER-INSERTED at scope exit
```

The compiler tracks Region bindings and inserts `Region.release` at all scope exit points.

### 2.4 No IDisposable

Regions are NOT IDisposable. There's no `use` keyword. The compiler manages lifetime automatically based on lexical scope.

## 3. CCS Layer Implementation

### 3.1 Region Type

```fsharp
// In NativeTypes.fs
| TRegion  // Opaque region handle

// Region is a built-in type, not parameterized
```

### 3.2 Region Intrinsics

> **Membrane note.** The `nativeptr` and `nativeint` appearances in this PRD are membrane plumbing recorded point-in-time, internal `TNativePtr` surface governed by the exit in `Closure_Nanopass_Architecture.md` Section 4 ("Why Flat: the Finiteness Lemma") and the boundary contract of C-01 Section 6.7.

```fsharp
// In CheckExpressions.fs
| "Region.create" ->
    // int -> Region (pages argument)
    NativeType.TFun(env.Globals.IntType, NativeType.TRegion)

| "Region.alloc" ->
    // Region -> int -> nativeptr<'T>
    // Type parameter 'T determines element size
    let tVar = freshTypeVar ()
    NativeType.TFun(
        NativeType.TRegion,
        NativeType.TFun(env.Globals.IntType, NativeType.TNativePtr(tVar)))

| "Region.release" ->
    // Region -> unit
    NativeType.TFun(NativeType.TRegion, env.Globals.UnitType)
```

### 3.3 Linear Resource Tracking

CCS marks Region bindings as linear resources:

```fsharp
type BindingKind =
    | Normal
    | LinearResource of resourceType: LinearResourceType

type LinearResourceType =
    | Region
    // Future: FileHandle, Socket, etc.
```

During scope exit analysis (a nanopass), the compiler ensures each linear resource is consumed exactly once.

### 3.4 Scope Exit Coeffect

```fsharp
type ScopeExitCoeffect = {
    ScopeId: NodeId
    LinearResources: (string * NodeId) list  // Name and release point
}
```

## 4. Source Settlement and Passive Witnessing

CCS/Baker owns region identity and authority, allocation extent, capacity,
alignment, placement, borrow/lifetime relationships and every required cleanup
edge. Owning recipes construct allocation and release operations in the PSG,
including release at all applicable scope exits.

Platform memory declarations provide page size, allocation capabilities and
physical call signatures. Neither a guessed page size nor emitter-generated
bump-pointer arithmetic establishes a safe allocation. Missing extent, capacity
or lifetime premises prevent the commitment that requires them.

Alex witnesses the settled storage, control and external calls from immutable
codata. It does not insert cleanup, choose placement or construct a region
allocator. Composer's backend realizes admitted OS memory operations and their
target ABI.

## 5. MLIR Output Specification

### 5.1 Region Type

```mlir
!region_type = !llvm.struct<(
    ptr,     // base pointer
    i64,     // capacity
    i64,     // used
    i1       // growable
)>
```

### 5.2 Region Create

```mlir
// let region = Region.create 4
%size = arith.muli %pages, %c4096 : i64
%addr_zero = llvm.mlir.zero : !llvm.ptr  // OS chooses address
%base = llvm.call @mmap(
    %addr_zero,   // addr (0 = OS chooses)
    %size,        // length
    i32 3,        // PROT_READ | PROT_WRITE
    i32 34,       // MAP_PRIVATE | MAP_ANONYMOUS
    i32 -1,       // fd (unused)
    i64 0         // offset
) : (!llvm.ptr, i64, i32, i32, i32, i64) -> !llvm.ptr

%region = llvm.alloca 1 x !region_type
%base_slot = llvm.getelementptr %region[0, 0]
llvm.store %base, %base_slot
%cap_slot = llvm.getelementptr %region[0, 1]
llvm.store %size, %cap_slot
%used_slot = llvm.getelementptr %region[0, 2]
llvm.store %c0, %used_slot
```

### 5.3 Region Alloc (Bump Pointer)

```mlir
// let buffer = Region.alloc<int> region 100
%elem_size = arith.constant 4 : i64  // sizeof(int)
%bytes = arith.muli %count, %elem_size : i64

%used_ptr = llvm.getelementptr %region[0, 2]
%used = llvm.load %used_ptr : i64
%new_used = arith.addi %used, %bytes : i64
llvm.store %new_used, %used_ptr

%base_ptr = llvm.getelementptr %region[0, 0]
%base = llvm.load %base_ptr : !llvm.ptr
%result = llvm.getelementptr %base[%used] : (!llvm.ptr, i64) -> !llvm.ptr
```

### 5.4 Region Release

```mlir
// Region.release region (compiler-inserted)
%base = llvm.load %base_ptr : !llvm.ptr
%cap = llvm.load %cap_ptr : i64
llvm.call @munmap(%base, %cap) : (!llvm.ptr, i64) -> i32
```

## 6. Validation

### 6.1 Sample Code

```fsharp
module BasicRegionSample

let processInRegion () =
    let region = Region.create 4  // 16KB

    // Allocate temporary buffer
    let buffer = Region.alloc<int> region 100

    // Use buffer
    for i in 0..99 do
        NativePtr.set buffer i (i * i)

    // Compute sum
    let mutable sum = 0
    for i in 0..99 do
        sum <- sum + NativePtr.get buffer i

    // Region.release region  ← compiler-inserted
    sum

[<EntryPoint>]
let main _ =
    Console.writeln "=== Basic Region Test ==="

    let result = processInRegion ()
    Console.write "Sum of squares 0-99: "
    Console.writeln (Format.int result)

    0
```

### 6.2 Expected Output

```
=== Basic Region Test ===
Sum of squares 0-99: 328350
```

## 7. Files to Create/Modify

### 7.1 CCS

| File | Action | Purpose |
|------|--------|---------|
| `NativeTypes.fs` | MODIFY | Add TRegion type |
| `CheckExpressions.fs` | MODIFY | Add Region intrinsics |
| `SemanticGraph.fs` | MODIFY | Add LinearResource binding kind |
| `ScopeAnalysis.fs` | CREATE | Track linear resources and exit points |

### 7.2 Composer

| File | Action | Purpose |
|------|--------|---------|
| CCS/Baker scope-exit recipe | CREATE | Settle lifetime and construct required Region.release operations |
| Alex region witnesses | CREATE | Passively compose settled region operations |

## 8. Implementation Checklist

### Phase 1: CCS Foundation
- [ ] Add TRegion to NativeTypes
- [ ] Add Region.create/alloc/release intrinsics
- [ ] Implement linear resource tracking
- [ ] Create ScopeAnalysis pass

### Phase 2: Settlement and Witnessing
- [ ] Construct required scope-exit releases in CCS/Baker
- [ ] Settle allocation extent, capacity, layout and lifetime
- [ ] Publish immutable facts for passive region witnesses
- [ ] Realize declared memory boundaries through Composer's backend

### Phase 3: Validation
- [ ] Sample 20 compiles without errors
- [ ] Sample 20 produces correct output
- [ ] Memory is properly released (no leaks)
- [ ] Samples 01-19 still pass

## 9. Platform Bindings

All platform-specific memory functions are accessed via Farscape-generated bindings (ExternCall pathway):

| Operation | Linux (Fidelity.Libc.Memory) | Windows (Fidelity.Win32.Memory) | Embedded |
|-----------|------------------------------|----------------------------------|----------|
| Create | mmap | VirtualAlloc | Static buffer |
| Grow | mremap | VirtualAlloc | ERROR |
| Release | munmap | VirtualFree | No-op |

`Region.create`/`alloc`/`release` remain CCS intrinsics because they involve compiler-managed state: bump pointer arithmetic, struct layout, and scope exit insertion. The raw OS memory calls go through ExternCall.

## 10. Related PRDs

- **D-01**: GTKWindow — establishes ExternCall pathway used by mmap/munmap bindings
- **A-05**: Region Passing — regions as parameters
- **A-06**: Region Escape — copyOut for escaping data
- **T-03 to T-05**: MailboxProcessor — region per message batch
