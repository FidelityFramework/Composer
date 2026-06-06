# R-05: Incremental Dynamism

> **Sample**: `36_IncrementalDynamism` (Future) | **Status**: Planned | **Category**: Reactive

## 1. Executive Summary

This PRD adds `bind` to `Incremental<'T>`, the operator that lets the dependency graph change shape as input values change. Where the applicative core (R-04) builds a static DAG whose structure is fixed at compile time, `bind` chooses or constructs its downstream subgraph from a runtime value. That single addition is the source of nearly all the difficulty in incremental computation, and it is where a structural, compiler-resident design earns its keep against a runtime library.

The contract is to keep the static majority of a graph free of runtime overhead, and to confine the dynamic machinery to the subgraphs that genuinely need it. The compiler classifies each subgraph as Applicative, Monadic, or Mixed, lowers applicative regions to static dispatch, and generates dynamic height tracking and dispatch only for monadic regions.

**Key Insight**: `map` and `bind` differ by one type constructor and a large amount of cost. The compiler must see the difference, because the difference decides whether a subgraph is a fixed configuration or a runtime-reconfigured one.

Normative specification: clef-lang-spec `spec/incremental-computation.md`, sections 4 and 9.

---

## 2. Motivation: Structural Incremental, By Contrast

Jane Street's "Seven Implementations of Incremental" is the clearest account of what a runtime library pays to support dynamic incremental graphs in a garbage-collected host. It is worth reading as the contrast case for why `Incremental<'T>` is a compiler intrinsic here. Each problem the talk describes is downstream of two facts: their Incremental is a library on top of OCaml's GC, and the dependency graph is a separate runtime structure from the program. This PRD removes both premises, so most of those problems move to compile time or do not arise.

| Problem in the runtime library | Where it came from | Resolution in a structural design |
|---|---|---|
| Nodes kept alive from inputs, collected late via sentinels and finalizers and reference counts | Library on a GC heap, upward pointers from leaves | Node lifetime is the enclosing arena and actor; deterministic free, no finalizers (R-06) |
| Exponential garbage from nested `bind`, old subgraphs still alive and recomputing | Dynamic reallocation with delayed collection | Demand gating: unobserved subgraphs do not stabilize; arena frees them deterministically (R-06) |
| Cutoff impossible under the first two-pass marking scheme | Marking pass had no values in hand | Height-ordered single pass with cutoff (R-04) |
| Topological order maintained at runtime with logical timestamps and periodic de-compaction | Total order needed for dynamic graphs | Height is a compile-time partial order for applicative regions; dynamic only for monadic regions |
| `map` implemented via `bind` was crushingly expensive | No distinction between static and dynamic nodes | Applicative and Monadic are distinct graph categories; `map` never allocates a dynamic node |
| Back edges from memoization tables introduced cycles | Hand-rolled common-subexpression caching | Interaction-net sharing reuses equal subgraphs without a side table |
| Record-of-closures node representation, slow and opaque | Type erasure across heterogeneous nodes | The PSG is the typed, inspectable graph; nodes are `SemanticKind` variants |

The talk also reports that once an application is incrementalized, the framework overhead dominates its runtime, so the returns to optimizing the framework are large. The structural answer is that for applicative regions there is no runtime framework to dominate, because stabilization is inlined and the graph is compiled. For monadic regions, the runtime orchestration described in this PRD is the remaining cost, and it is bounded to the subgraphs that require it. That confinement is the point of the Applicative/Monadic classification.

---

## 3. Language Feature Specification

### 3.1 The bind Operator

```fsharp
Incremental.bind : Incremental<'a> -> ('a -> Incremental<'b>) -> Incremental<'b>
```

`bind` differs from `map` only in the result of the function: `map` takes `'a -> 'b`, `bind` takes `'a -> Incremental<'b>`. The extra `Incremental` is the dynamism. A `bind` node has a left input and a right-hand side; depending on the left value, the right-hand side selects an existing incremental or constructs a new subgraph. The set of active dependencies can change per stabilization cycle.

### 3.2 map vs bind

```fsharp
// map: static. All three inputs are always wired in; recomputes on any change.
let ifMap cond t e =
    Incremental.map3 (fun c x y -> if c then x else y) cond t e

// bind: dynamic. Only the taken branch is wired in.
let ifBind cond t e =
    Incremental.bind cond (fun c -> if c then t else e)
```

Both produce the same value. The `map` form re-fires whenever either branch changes, even the branch not selected. The `bind` form wires in only the selected branch, so a change to the unselected branch does no work. The cost is that `bind` allocates and rewires, where `map` does not.

### 3.3 Graph Category

```fsharp
type IncrementalGraphCategory =
    | Applicative    // Static structure, known at compile time
    | Monadic        // Dynamic structure, determined at runtime
    | Mixed          // Applicative skeleton with monadic subgraphs
```

The compiler infers the category from the computation-expression desugaring. A `let!` whose result does not influence which subsequent `let!` executes is applicative. A `let!` whose result determines which subsequent `let!` runs is monadic. Most real graphs are Mixed: a static skeleton with dynamic pockets.

---

## 4. Architecture

### 4.1 Monadic Subgraphs

```fsharp
incremental {
    let! routing = bitnetRouter input          // monadic: result selects the next stage
    let! result =
        match routing.selectedExpert with
        | Vision   -> visionExpert input
        | Language -> languageExpert input
    return result
}
```

The active dependency set after the `bind` depends on `routing`. When `routing` changes the selected expert, the previously active subgraph is detached and the newly selected subgraph is attached. Within each branch, the subgraph is itself applicative and lowers statically; the `bind` node is the only dynamic element.

### 4.2 Dynamic Height

Applicative height is computed once at compile time. A `bind` can change the height of its right-hand side between cycles, so height for monadic regions is tracked dynamically. The structural design uses height directly as a partial order. A memory-sensitive variant that never decreases its recorded height avoids renumbering churn when a `bind` flips its right-hand side back and forth, which keeps height maintenance bounded instead of linear in graph size.

### 4.3 Sharing Without Back Edges

Common-subexpression reuse is provided by the interaction-net substrate, which shares two subgraphs that compute the same value from the same dependencies. Because sharing is a property of the substrate, there is no separate memoization table to introduce back edges, which removes the cycle hazard the runtime library hit when it added memoization.

### 4.4 What Stays Hard

`bind` does not become free under a structural design; its cost is confined, not removed. A deeply monadic graph that is genuinely observed still pays for reallocation and dynamic dispatch, and a pathological nesting can still blow up the work per cycle. The honest position is that the static majority pays nothing and the dynamic minority is isolated and bounded by demand. The performance of monadic lowering is the measured risk to carry into implementation, with nested `bind` as the named failure mode.

---

## 5. Implementation Strategy

### 5.1 Category Classification

`checkIncremental` (CCS Coordinator) walks the `Bind` chain and classifies each subgraph. Applicative regions get static heights and a fixed dispatch configuration. Monadic regions get dynamic height tracking and a dynamic dispatch site. Mixed graphs get both, region by region.

### 5.2 Dynamic Dispatch

On CPU, a `bind` node lowers to a dispatch that detaches the prior right-hand side, evaluates the function, attaches the selected subgraph, and recomputes from there in height order. Detached subgraphs that are no longer demanded are quiesced and freed with the arena (the demand mechanics are specified in R-06).

### 5.3 MLIR Sketch

```mlir
// bind node stabilization (CPU)
%lhs = llvm.call @stabilize_node(%lhs_node) : (!llvm.ptr) -> !lhs_type
%changed = llvm.call @cutoff_changed(%bind_node, %lhs) : (...) -> i1
llvm.cond_br %changed, ^rewire, ^reuse_rhs

^rewire:
    // detach prior rhs, run the bind function to choose/construct the new rhs
    llvm.call @detach_rhs(%bind_node)
    %new_rhs = llvm.call %bind_fn_ptr(%lhs) : (!lhs_type) -> !llvm.ptr
    llvm.call @attach_rhs(%bind_node, %new_rhs)
    llvm.br ^stabilize_rhs(%new_rhs : !llvm.ptr)

^reuse_rhs:
    %rhs = llvm.call @current_rhs(%bind_node) : (!llvm.ptr) -> !llvm.ptr
    llvm.br ^stabilize_rhs(%rhs : !llvm.ptr)

^stabilize_rhs(%rhs: !llvm.ptr):
    %val = llvm.call @stabilize_node(%rhs) : (!llvm.ptr) -> !result_type
```

---

## 6. Coeffects

| Coeffect | Purpose |
|----------|---------|
| IncrementalLayout | Extended with `GraphCategory` per region |
| DynamicHeight | Runtime height slot for monadic nodes |
| ClosureLayout | Capture set of the bind function |
| RhsLifetime | Arena binding for dynamically constructed right-hand sides |

---

## 7. Sample Code

```fsharp
module IncrementalDynamism

type Expert = Vision | Language
type Routing = { selectedExpert: Expert }

let pipeline (input: Incremental<Tensor>) (router: Tensor -> Incremental<Routing>) =
    incremental {
        let! routing = router input            // monadic boundary
        let! result =
            match routing.selectedExpert with
            | Vision   -> visionExpert input    // applicative subgraph
            | Language -> languageExpert input  // applicative subgraph
        return result
    }

// Changing `input` re-fires only the selected expert.
// Changing the routing decision detaches one expert subgraph and attaches the other.
```

---

## 8. Target Applicability

| Target | Applicable | Notes |
|--------|------------|-------|
| WREN Stack | Yes | Dynamic UI and compute graphs |
| QuantumCredential | Partial | Static-skeleton graphs preferred |
| LVGL MCU | No | Dynamic reallocation conflicts with fixed memory |
| Unikernel | Optional | Routing-style dynamism where demanded |

---

## 9. Dependencies

| Dependency | Purpose |
|------------|---------|
| R-04 IncrementalFoundations | Static core, height, stabilization, cutoff |
| C-01 Closures | Bind function captures |
| A-04 BasicRegion | Arena for dynamically constructed subgraphs |
| Interaction-net substrate | Sharing without back edges |

---

## 10. Validation Criteria

- [ ] `bind` wires in only the selected branch
- [ ] A change to an unselected branch does no work under `bind`
- [ ] The compiler classifies a subgraph as Applicative, Monadic, or Mixed from the CE
- [ ] `map` never allocates a dynamic node
- [ ] A changed routing value detaches the prior subgraph and attaches the new one
- [ ] A detached, undemanded subgraph is freed with the arena
- [ ] Dynamic height does not renumber the whole graph when a bind flips repeatedly
- [ ] Equal subgraphs from equal dependencies are shared, with no back edge introduced

---

## 11. Related Documents

- [R-04-IncrementalFoundations](R-04-IncrementalFoundations.md) - Static core
- [R-06-IncrementalIntegration](R-06-IncrementalIntegration.md) - Demand, actors, hardware targets
- clef-lang-spec `spec/incremental-computation.md` - Normative specification
- Minsky, Y. "Seven Implementations of Incremental," Jane Street - Runtime-library contrast
