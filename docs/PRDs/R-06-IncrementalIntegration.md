# R-06: Incremental Integration

> **Sample**: `37_IncrementalIntegration` (Future) | **Status**: Planned | **Category**: Reactive

## 1. Executive Summary

This PRD defines how `Incremental<'T>` integrates with the rest of the concurrent model and how it lowers across hardware targets. Three integrations are specified: the structural correspondence between an incremental node and an Olivier actor, demand registration through Prospero, and fusion with `Observable<'T>` (R-01 through R-03). The PRD then specifies target lowering, where the same incremental abstraction reaches CPU, NPU, and GPU through the shared middle-end.

This is the first PRD in which the heterogeneous-target story is a core concern. The applicative core (R-04) and dynamism (R-05) are specified against CPU. Here the measure-typed hardware variant enters, dependency edges cross device boundaries through BAREWire, and stabilization maps onto tile activation waves and wavefront dispatch. CPU lowering is demonstrated; NPU and GPU lowering are architectural and described as design intent.

**Key Insight**: An incremental node and an actor are the same structure viewed two ways. The cached value is actor state, the recompute function is the receive handler, dependency edges are message channels, and demand is supervision.

Normative specification: clef-lang-spec `spec/incremental-computation.md`, sections 1.2, 6.3, 8, and 11.

---

## 2. Integration Surface

### 2.1 Incremental ↔ Actor

| Incremental Concept | Actor Concept |
|---|---|
| Cached value | Actor state in arena memory |
| Recompute function | Actor receive handler |
| Dependency edges | Message channels (BAREWire) |
| Staleness flag | Incoming message differs from last processed |
| Cutoff predicate | Structural comparison on output |
| Demand registration | Prospero supervision |
| Stabilization order | Actor scheduling waves |

An incremental node's lifetime is the enclosing actor's lifetime. When Prospero retires an actor, the node, its cached value, and its dependency edges are freed with the arena. No garbage collection participates.

### 2.2 Incremental ↔ Observable

| Function | Type | Description |
|----------|------|-------------|
| `Incremental.ofObservable` | `Observable<'T, _> -> Incremental<'T>` | Event source drives invalidation |
| `Incremental.toObservable` | `Incremental<'T> -> Observable<'T, Hot>` | Emit on each post-cutoff change |

An `Observable<'T>` feeding an `Incremental<'T>` is the common case: an event source drives a cached derived computation. Because both are intrinsic, the compiler fuses the observable subscription directly into the incremental node's invalidation trigger, removing the intermediate allocation and callback indirection a library bridge would add.

### 2.3 Incremental ↔ BAREWire

| Function | Type | Description |
|----------|------|-------------|
| dependency edge (same device) | pointer handoff | Zero-copy on unified memory |
| dependency edge (cross device) | DMA descriptor | BAREWire-generated from the edge type |

Dependency edges between nodes on different hardware targets use BAREWire descriptors for zero-copy data movement. On HSA-unified memory such as Strix Halo, CPU-to-GPU edges are pointer handoffs; CPU or GPU to NPU edges use DMA descriptors that BAREWire generates from the edge's type information.

---

## 3. Architecture

### 3.1 Demand and Prospero, By Contrast

A node that no consumer observes does not stabilize, even when its inputs are stale. Demand is the transitive closure of observers in the dependency direction, the dual of the variables' reachability. Prospero manages demand for actor-based nodes; for non-actor contexts the compiler tracks demand statically through the PSG.

This is where a structural design replaces the machinery a runtime library needs. The OCaml Incremental reached the same place over several versions: explicit observers to track the demanded part, quiescing the unobserved part to stop nested-bind garbage, and finally an invariant that no pointer crosses from the observed world to the unobserved world so that finalizers could be retired. Here that invariant holds by construction. Demand gating stops undemanded nodes from stabilizing, and arena lifetime frees detached subgraphs deterministically, so there are no sentinels, finalizers, or reference counts.

### 3.2 Observable Fusion

```
Observable.OnNext  ──►  Incremental invalidation trigger
        (fused at compile time; no intermediate observer allocation)
```

The compiler recognizes an observable feeding an incremental input and emits the subscription as the node's invalidation path. The push side marks the node stale; the next stabilization pulls the new value through with cutoff.

### 3.3 Hardware-Targeted Type

```fsharp
type Incremental<'T, [<Measure>] 'Target when 'T : equality> = intrinsic

[<Measure>] type cpu
[<Measure>] type gpu_cu
[<Measure>] type npu_tile

let routingDecision : Incremental<RoutingVector>            = ...   // CPU (default)
let visionFeatures  : Incremental<FeatureMap, npu_tile>     = ...   // NPU
let languageEmbed   : Incremental<Embedding, gpu_cu>        = ...   // GPU
```

The target measure survives through the PSG to code generation without erasure. Absent a target, the compiler infers it from context or defaults to CPU.

### 3.4 Target Lowering Overview

| Target | Status | Node maps to | Cutoff becomes |
|--------|--------|--------------|----------------|
| CPU | Demonstrated | Arena struct, inline stabilization | `structural_eq` branch |
| NPU (MLIR-AIE) | Architectural | AIE tile with local SRAM | ObjectFIFO not written |
| GPU (RDNA) | Architectural | CU wavefront group | Ballot/vote across lanes |

The CPU path is the baseline this PRD validates. The NPU and GPU paths describe how the same abstraction is designed to lower; they are not claimed as demonstrated.

---

## 4. Implementation Strategy

### 4.1 NPU (MLIR-AIE), Architectural

The design maps each node to an AIE tile, the cached value to tile-local SRAM, the recompute function to the tile ELF, and each dependency edge to a stream-switch route with a DMA descriptor. For applicative subgraphs, the compiler is designed to generate a static overlay: tile placement, routing, and descriptors are fixed once, and incremental recomputation activates only the DMA descriptors whose inputs changed. For monadic subgraphs, the compiler is designed to emit ctrlcode that the embedded runtime executes to select which tiles to activate per cycle.

```
Height 0 tiles: input DMA activated (leaf nodes)
Height 1 tiles: activated when height 0 outputs land in ObjectFIFO
Height h tiles: activated when height h-1 outputs land in ObjectFIFO
```

Cutoff at a tile means its output ObjectFIFO is not written, so downstream tiles see no new input and stay idle. This is the height-ordered stabilization of R-04 expressed as a tile activation wave.

### 4.2 GPU (RDNA), Architectural

The design maps each node to a CU wavefront group, the cached value to LDS or VGPR contents, and each dependency edge to a global-memory handoff via a BAREWire descriptor. Staleness is a dispatch predicate, so a CU whose input buffers are unchanged is not dispatched. Cutoff is a ballot or vote across wavefront lanes. Height-ordered stabilization becomes sequential dispatch waves, with independent nodes at the same height dispatched in parallel within a wave.

### 4.3 Demand Through Prospero

```fsharp
// An incremental node hosted by an actor; Prospero registers demand
let router =
    IncrementalActor.create (arena: byref<Arena<'lifetime>>) {
        dependencies = [inputStream]
        cutoff = RoutingDecision.structuralEquality
        compute = fun input -> bitnetRoute input
    }
```

When a downstream consumer subscribes, Prospero records demand and the node participates in stabilization. When demand drops, the node quiesces and is freed with the actor's arena.

---

## 5. Coeffects

| Coeffect | Purpose |
|----------|---------|
| IncrementalLayout | Extended with `TargetMeasure` |
| TargetMeasure | Hardware target carried to code generation |
| DemandRegistration | Observer/Prospero demand state |
| BridgeFusion | Observable subscription fused into invalidation |
| BAREWireEdge | Descriptor generation for cross-target dependency edges |

---

## 6. Sample Code

```fsharp
module IncrementalIntegration

// Observable event source drives a cached derived computation
let temperatureEvents : Observable<float<celsius>, Hot> = Sensor.stream ()

let display : Incremental<string> =
    incremental {
        let! c = Incremental.ofObservable temperatureEvents
        return Format.celsius c
    }

// Heterogeneous graph: routing on CPU, vision on NPU, fused
let visionFeatures : Incremental<FeatureMap, npu_tile> =
    incremental {
        let! routing = routingDecision           // CPU node
        let! frame   = cameraFrame               // CPU node
        return extractFeatures routing frame     // NPU tile
    }
// CPU-to-NPU dependency edges lower to BAREWire DMA descriptors.
```

---

## 7. Target Applicability

| Target | Applicable | Notes |
|--------|------------|-------|
| WREN Stack | Yes | CPU integration with Observable and actors |
| QuantumCredential | Partial | Actor-hosted derived computation |
| LVGL MCU | No | Cross-target lowering out of scope |
| Unikernel | Optional | Actor-hosted nodes, no GPU/NPU |
| Strix Halo (heterogeneous) | Architectural | CPU/NPU/GPU nodes over unified memory and BAREWire |

---

## 8. Dependencies

| Dependency | Purpose |
|------------|---------|
| R-04 IncrementalFoundations | Static core, stabilization, cutoff |
| R-05 IncrementalDynamism | Monadic dispatch for dynamic graphs |
| R-01 ObservableFoundations | Observable source for fusion |
| R-03 ObservableIntegration | Observable/actor bridge patterns |
| T-03 BasicActor | Actor hosting and lifetime |
| BAREWire | Cross-target dependency edges |
| AIE_Backend_Design | NPU tile overlay and ObjectFIFO model |

---

## 9. Validation Criteria

- [ ] An incremental node hosted by an actor is freed with the actor's arena
- [ ] An undemanded node does not stabilize even when inputs are stale
- [ ] An Observable feeding an Incremental is fused into the invalidation trigger
- [ ] `Incremental.toObservable` emits only on post-cutoff changes
- [ ] The target measure survives to code generation without erasure
- [ ] CPU integration with Observable and actors is demonstrated end to end
- [ ] (Architectural) NPU lowering maps cutoff to an unwritten ObjectFIFO
- [ ] (Architectural) GPU lowering skips dispatch for unchanged input buffers
- [ ] Cross-target dependency edges generate BAREWire DMA descriptors

---

## 10. Related Documents

- [R-04-IncrementalFoundations](R-04-IncrementalFoundations.md) - Static core
- [R-05-IncrementalDynamism](R-05-IncrementalDynamism.md) - Dynamic graphs
- [R-01-ObservableFoundations](R-01-ObservableFoundations.md) - Push-based source
- [R-03-ObservableIntegration](R-03-ObservableIntegration.md) - Bridge patterns
- [T-03-BasicActor](T-03-BasicActor.md) - Actor model
- [AIE_Backend_Design](../AIE_Backend_Design.md) - NPU lowering
- clef-lang-spec `spec/incremental-computation.md` - Normative specification
