# R-04: Incremental Foundations

> **Sample**: `35_IncrementalFoundations` (Future) | **Status**: Planned | **Category**: Reactive

## 1. Executive Summary

This PRD introduces `Incremental<'T>` as a compiler-known intrinsic for dependency-tracked, demand-driven, change-minimizing computation. It covers the static core: the type, `return`/`map`/`map2`, the applicative dependency DAG, height assignment, height-ordered stabilization, structural cutoff, and CPU lowering with arena-allocated cached values. Dynamism via `bind` is specified in R-05. Actor mapping, demand through Prospero, Observable fusion, and hardware-target lowering are specified in R-06.

`Incremental<'T>` is the demand-driven dual of `Observable<'T>` (R-01). Observable pushes values forward as they arrive; Incremental pulls values on demand, caches them, tracks which inputs produced them, and recomputes only the affected part of the graph when an input changes. The two are complementary points on the evaluation-strategy spectrum the compiler understands natively.

Unlike Jane Street's library-level `Incremental` for OCaml, `Incremental<'T>` is not a runtime abstraction built on the garbage collector. It is a compile-time annotation the PSG preserves through lowering, so Firefly reasons about the dependency graph directly and emits selective recomputation per target. The seven-version runtime history that the OCaml library worked through (sentinels and finalizers for collection, timestamp topological sort for ordering, observer tracking for demand) reduces to compile-time structure plus arena lifetime here.

**Key Insight**: The dependency graph is the Program Semantic Graph. Incremental does not build a second graph at runtime; it annotates the one the compiler already has.

Normative specification: clef-lang-spec `spec/incremental-computation.md`.

---

## 2. Language Feature Specification

### 2.1 Position on the Evaluation-Strategy Spectrum

| Property | Observable&lt;'T&gt; | Lazy&lt;'T&gt; | Incremental&lt;'T&gt; |
|---|---|---|---|
| Deferred evaluation | No (push) | Yes | Yes (demand-driven) |
| Cached result | No | Yes | Yes |
| Dependency tracking | No | No | Yes |
| Invalidation | No | No | Yes |
| Cutoff (change detection) | No | No | Yes |
| Propagation bound | Unbounded | N/A | Bounded by cutoff |

Each property to the right hands the compiler more information at lowering. `Incremental<'T>` is transparent: the compiler knows the dependency graph, knows which downstream nodes a change affects, and can prove that unaffected subgraphs produce identical results.

`Incremental<'T>` extends `Lazy<'T>` (C-05) with two properties: dependency tracking and invalidation. A `Lazy<'T>` computes once and caches indefinitely. An `Incremental<'T>` computes on demand, caches, tracks the inputs that contributed, and invalidates when those inputs change. Recomputation occurs only when a node is both stale and demanded.

### 2.2 Core Type

```fsharp
type Incremental<'T when 'T : equality> = intrinsic
```

The `'T : equality` constraint provides the default cutoff predicate. Structural equality on records and discriminated unions generates the cutoff automatically. Types without equality are rejected at compile time. The hardware-targeted variant `Incremental<'T, [<Measure>] 'Target>` is specified in R-06; absent a target measure, the foundations layer lowers to CPU.

### 2.3 Applicative Operations

This PRD covers the operators that build static dependency graphs:

```fsharp
Incremental.return : 'T -> Incremental<'T>                              // const node
Incremental.map    : ('a -> 'b) -> Incremental<'a> -> Incremental<'b>   // 1-input node
Incremental.map2   : ('a -> 'b -> 'c)
                     -> Incremental<'a> -> Incremental<'b>
                     -> Incremental<'c>                                  // 2-input join
```

`map3` through `mapN` compose from `map2`. `return` produces a constant node that is never stale. `map` and `map2` add nodes that re-fire when their inputs change and survive cutoff. `bind`, which introduces dynamic structure, is deferred to R-05.

### 2.4 Node Logical Fields

Each node carries PSG annotations that lower differently per target:

| Field | Type | Semantics |
|-------|------|-----------|
| `value` | `'T` | Cached result of the most recent computation |
| `stale` | `bool` | Whether any dependency changed since last computation |
| `height` | `int` | Topological depth in the DAG; sets evaluation order |
| `dependencies` | `NodeId list` | PSG nodes this node reads from |
| `dependents` | `NodeId list` | PSG nodes that read from this node |
| `cutoff` | `'T -> 'T -> bool` | Equality predicate; defaults to structural equality |
| `recompute` | `unit -> 'T` | Deferred computation (thunk with captured environment) |

### 2.5 Variables and Demand

Input enters the graph through variables, and output leaves through demand:

```fsharp
Var.create : 'T -> Var<'T>
Var.set    : Var<'T> -> 'T -> unit
Var.watch  : Var<'T> -> Incremental<'T>     // read the incremental view
```

Demand registration in the foundations layer is the compiler tracking, statically through the PSG, which incremental values a downstream consumer reads. The actor and Prospero demand model is specified in R-06.

---

## 3. Architecture

### 3.1 Composition from Standing Art

| Component | Reused From | Purpose |
|-----------|-------------|---------|
| Thunk + memoization | C-05 Lazy | Deferred recompute, cached value |
| Structural equality | F-10 Records, F-05 DUs | Default cutoff predicate |
| Arena allocation | A-04 BasicRegion | Cached value storage with scoped lifetime |
| Closures | C-01 | Captured environment of the recompute thunk |
| Push dual | R-01 Observable | Conceptual counterpart; fusion specified in R-06 |

### 3.2 Static Dependency DAG and Height

The applicative subgraph has structure known at compile time: all dependencies are declared unconditionally, so Firefly compiles it to a fixed configuration. Each node is assigned a height equal to the longest path from any leaf input:

```
height(node) = 0                                        if node has no dependencies
height(node) = 1 + max(height(d) for d in dependencies) otherwise
```

Height sets evaluation order. Nodes at height 0 evaluate first; nodes at height `h` evaluate after all nodes at `h-1`. For applicative subgraphs, height is computed once at compile time. This is a partial order, not a total order: nodes at the same height are independent and may execute concurrently.

### 3.3 CPU Memory Layout

An applicative node with element type `T` and `N` dependencies materializes as:

```
IncrementalNode<T>
┌──────────────────────────────────────────────┐
│ stale: i1            (padded to alignment)     │
│ height: i32          (4 bytes)                 │
│ value: T             (sizeof(T), aligned)      │
│ recompute_ptr: ptr   (8 bytes on 64-bit)       │
│ dep_count: i32       (4 bytes)                 │
│ dep_ptrs: ptr[N]     (N x 8 bytes)             │
└──────────────────────────────────────────────┘
```

The cached value is allocated in the enclosing arena, so its lifetime equals the arena's. No GC-managed heap is involved.

### 3.4 NativeType Representation

```fsharp
type NativeType =
    // ...
    | TIncremental of
        elementType: NativeType *
        targetMeasure: MeasureType option   // None at foundations layer (CPU)
```

---

## 4. Implementation Strategy

### 4.1 Height-Ordered Stabilization

Stabilization brings demanded nodes up to date in a single pass that supports cutoff. This is the endpoint of the algorithm the OCaml library reached in its later versions, taken here as the normative procedure:

1. Collect all stale nodes whose outputs are demanded.
2. Sort by height, ascending (a topological order over the DAG).
3. For each node in height order:
   a. Recompute the value from current dependencies.
   b. Compare the new value against the cached value with the cutoff predicate.
   c. If the cutoff returns `true` (unchanged): clear the stale flag, keep the cached value, remove dependents from the stale set.
   d. If the cutoff returns `false` (changed): store the new value, clear the stale flag, leave dependents stale for processing at their height.

Step 3c is the bound: when a node's output is unchanged despite a changed input, the downstream subgraph is skipped. A height array of node lists replaces a priority heap, so selecting the next node is constant-time.

### 4.2 Staleness and Cutoff

Staleness propagates forward conservatively: a set or invalidation marks the source node stale, then its dependents transitively. The marked set is the candidate recomputation set; the cutoff decides which candidates actually propagate.

The default cutoff is structural equality derived from `'T : equality`, generated field-by-field for records and discriminated unions and emitted as native equality for primitives. A custom cutoff attaches to a type:

```fsharp
[<IncrementalCutoff>]
let epsilonEqual (a: float<celsius>) (b: float<celsius>) =
    abs (a - b) < 0.01<celsius>
```

The DTS constrains cutoff operands: two values with incompatible units cannot be compared without explicit conversion, which prevents spurious "unchanged" results from comparing across unit systems.

### 4.3 Computation Expression

```fsharp
incremental {
    let! a = sensorA      // edge: sensorA -> a
    let! b = sensorB      // edge: sensorB -> b   (independent of a)
    return combine a b    // edges: a -> result, b -> result
}
```

Each `let!` desugars to a dependency edge in the PSG. When sequential `let!` bindings carry no data dependency between them, the compiler classifies the subgraph as applicative: `a` and `b` share a height and may execute concurrently, and `combine` sits one height above. The builder methods relevant to the applicative subset are `Return`, `ReturnFrom`, `Combine`, and `Zero`; `Bind` with dynamic structure is R-05.

### 4.4 CPU MLIR

```mlir
// Stabilization check for one node
%is_stale = llvm.load %node_stale_ptr : !llvm.ptr -> i1
llvm.cond_br %is_stale, ^recompute, ^use_cached

^recompute:
    %new_value = llvm.call %recompute_ptr(%node_ptr) : (!llvm.ptr) -> !result_type
    %cached = llvm.load %node_value_ptr : !llvm.ptr -> !result_type
    %unchanged = llvm.call @structural_eq(%new_value, %cached) : (...) -> i1
    llvm.cond_br %unchanged, ^cutoff_hit, ^propagate

^cutoff_hit:
    %false = arith.constant 0 : i1
    llvm.store %false, %node_stale_ptr : i1, !llvm.ptr
    llvm.br ^use_cached

^propagate:
    llvm.store %new_value, %node_value_ptr : !result_type, !llvm.ptr
    %false2 = arith.constant 0 : i1
    llvm.store %false2, %node_stale_ptr : i1, !llvm.ptr
    llvm.br ^done(%new_value : !result_type)

^use_cached:
    %val = llvm.load %node_value_ptr : !llvm.ptr -> !result_type
    llvm.br ^done(%val : !result_type)

^done(%result: !result_type):
```

### 4.5 CCS / Composer Pipeline

| Phase | Pass | Responsibility |
|-------|------|----------------|
| CCS | `checkIncremental` (Coordinator) | Check the body, compute dependencies from `let!`, classify as Applicative, assign heights, build `SemanticKind.IncrementalExpr` |
| CCS | `inferCutoff` (TypeChecker) | Resolve cutoff per node; structural equality by default; honor `[<IncrementalCutoff>]`; verify dimensional compatibility |
| Alex | `SSAAssignment` | Compute `IncrementalLayout`; assign SSAs for node construction |
| Alex | `HeightAnalysis` | Validate height assignments for applicative subgraphs |
| Alex | `IncrementalWitness` | Emit arena-allocated node struct and inline stabilization loop |
| Alex | `CutoffWitness` | Generate the element-type equality comparison; inline into the stabilization check |

---

## 5. Coeffects

| Coeffect | Purpose |
|----------|---------|
| IncrementalLayout | Node element type, height, dependency slots, cutoff SSA, construction SSAs |
| ClosureLayout | Capture set of the recompute thunk (from C-01) |
| SSA | Values threaded through stabilization |

---

## 6. Sample Code

```fsharp
module IncrementalFoundations

// Inputs
let celsius = Var.create 20.0<celsius>

// Derived, applicative graph
let view =
    incremental {
        let! c = Var.watch celsius
        let f = c * 1.8<fahrenheit/celsius> + 32.0<fahrenheit>
        return f
    }

// Drive it
Var.set celsius 25.0<celsius>     // marks the graph stale
// stabilization recomputes `view` once, in height order

// A cutoff stops propagation when the rounded display value is unchanged
[<IncrementalCutoff>]
let sameTenth (a: float<fahrenheit>) (b: float<fahrenheit>) =
    floor (a * 10.0) = floor (b * 10.0)
```

---

## 7. Target Applicability

| Target | Applicable | Notes |
|--------|------------|-------|
| WREN Stack | Yes | CPU stabilization, reactive derived values for UI and compute |
| QuantumCredential | Partial | Derived computation without UI |
| LVGL MCU | Partial | Static applicative graphs only, fixed arena |
| Unikernel | Optional | Derived network/state computation |

CPU lowering is the demonstrated baseline for this PRD. NPU and GPU lowering are architectural and specified in R-06.

---

## 8. Dependencies

| Dependency | Purpose |
|------------|---------|
| C-05 Lazy | Thunk and memoization basis |
| F-10 Records | Structural equality for cutoff |
| F-05 DiscriminatedUnions | Structural equality for cutoff |
| A-04 BasicRegion | Arena-allocated cached value |
| C-01 Closures | Recompute thunk captures |
| R-01 ObservableFoundations | Conceptual dual; fusion in R-06 |

---

## 9. Validation Criteria

- [ ] `Incremental.return` produces a constant node that never goes stale
- [ ] `map` re-fires only when its input changes
- [ ] `map2` joins two inputs and re-fires when either changes
- [ ] Height is computed at compile time for an applicative graph
- [ ] Stabilization processes nodes in ascending height order
- [ ] Structural cutoff stops propagation when an output is unchanged
- [ ] A recombinant (fan-out then fan-in) graph recomputes each node once
- [ ] Cached values reside in arena memory, freed with the arena
- [ ] A cutoff comparing incompatible units is a compile-time error

---

## 10. Related Documents

- [R-01-ObservableFoundations](R-01-ObservableFoundations.md) - The push-based dual
- [R-05-IncrementalDynamism](R-05-IncrementalDynamism.md) - bind and monadic graphs
- [R-06-IncrementalIntegration](R-06-IncrementalIntegration.md) - actors, demand, hardware targets
- [C-05-Lazy](C-05-Lazy.md) - Thunk and memoization
- clef-lang-spec `spec/incremental-computation.md` - Normative specification
