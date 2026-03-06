# Partial Application Closure Reification — Research Spike

> **Status**: Research spike. Documents the gap between curry flattening detection and escaping partial application emission.
>
> **Context**: Sample 11 closures pass for all cases except nested closures (`makeScaledAdder`), which requires partial application results to be reified as closure values.

## 1. Problem Statement

Baker's curry flattening correctly merges nested lambdas:

```
fun multiplier -> fun x -> base' + multiplier * x
    ==>  lambda_flat(env, multiplier, x) -> base' + multiplier * x
```

And the saturation analysis correctly detects:
- **Partial application**: `addScaled 3` supplies 1 of 2 args
- **Saturated call**: `scale3 4` completes the application

When both sites are **local** (same function scope), the current system works: ApplicationWitness emits a direct call with combined args at the saturation site, and the partial application emits nothing (`TRVoid`).

**The gap**: When the partial application result **escapes** — returned from a function, stored in a data structure, passed as an argument — there is no saturation site to absorb it. The partial result must become a **first-class closure value** that can be called later with the remaining arguments.

```clef
let makeScaledAdder (base': int) =
    fun multiplier ->           // This lambda gets flattened with inner lambda
        fun x -> base' + multiplier * x

let addScaled = makeScaledAdder 100   // Returns closure (int -> (int -> int))
let scale3 = addScaled 3              // Partial application — result escapes!
let result = scale3 4                 // Saturated call on escaped PAP
```

At `addScaled 3`: the caller has a closure for the flattened `lambda_flat(env, multiplier, x)` and supplies `multiplier=3`. It expects a new closure `(int -> int)` back. But `lambda_flat` takes both args — there's nothing to return a closure from.

## 2. Taxonomy of Partial Application Scenarios

| Scenario | PAP Escapes? | Current Status |
|---|---|---|
| `let f = add 2; f 3` | No (local saturation) | **Working** — direct call with combined args |
| `let f = add 2; return f` | Yes (via return) | **Gap** — needs PAP closure |
| `let f = add 2; g f` | Yes (via argument) | **Gap** — needs PAP closure |
| `let f = add 2; [f]` | Yes (via data structure) | **Gap** — needs PAP closure |
| Nested lambda from outside | Yes (structural) | **Gap** — the `makeScaledAdder` case |

## 3. What Exists Today

### 3.1 Detection (Complete)

`CurryFlattening.analyze` produces:

```
CurryFlatteningResult = {
    PartialApplications: Map<NodeId, PartialApplicationInfo>
    SaturatedCalls: Map<NodeId, SaturatedCallInfo>
    PartialAppBindings: Set<NodeId>
    AbsorbedLambdas: Set<NodeId>
    DeferredArgNodes: Set<NodeId>
}
```

All detection is correct. The `PartialApplicationInfo` knows:
- Which function is being partially applied (`TargetBindingId`)
- Which arguments were supplied (`SuppliedArgNodes`)
- How many total parameters the flattened function takes (`TotalParams`)

### 3.2 Emission (Local Saturation Only)

- **ApplicationWitness**: Saturated calls emit direct call with all args combined
- **BindingWitness**: Partial app bindings emit `TRVoid`
- **VarRefWitness**: References to partial app bindings emit `TRVoid`

### 3.3 Closure Infrastructure (Complete for Regular Closures)

ClosurePatterns provides:
- `pFlatClosure`: Build env buffer + uniform pair from captures
- `pClosureCall` / `pClosureCallIndirect`: Extract code_ptr + env, call with prepended env
- `pExtractCaptures`: Load captures from env buffer at byte offsets

This infrastructure is the right shape for PAP closures — the only difference is what goes into the env buffer.

## 4. Design Space

### 4.1 PAP as Closure (Thunk Approach)

A partial application `f x₁ x₂` where `f` takes `x₁ x₂ x₃ x₄` produces a **PAP closure**:

```
PAP Closure for `f x₁ x₂`:
┌──────────────────────────────────────────┐
│ code_ptr: pointer to PAP_thunk_f_2       │
├──────────────────────────────────────────┤
│ capture_0: x₁  (first supplied arg)      │
│ capture_1: x₂  (second supplied arg)     │
│ capture_2: original_env (if f is closure)│
└──────────────────────────────────────────┘

PAP_thunk_f_2(env, x₃, x₄):
    // Extract captured partial args from env
    x₁ = extract env[0]
    x₂ = extract env[1]
    original_env = extract env[2]
    // Forward to flattened function with all args
    return f_flat(original_env, x₁, x₂, x₃, x₄)
```

The PAP thunk is a **generated forwarding function** — it reconstructs the full argument list and delegates to the original flattened function.

**Uniform pair representation** (same as regular closures):
```
memref<2xindex> = [code_ptr_as_index, env_ptr_as_index]
```

This means PAP closures and regular closures have the **same calling convention** — callers don't need to distinguish them.

### 4.2 Key Architectural Question: Where to Generate PAP Thunks?

**Option A: Baker (PSG-level, before Alex)**
- Baker detects escaping partial applications and inserts synthetic Lambda nodes into the PSG
- These Lambdas have the PAP thunk body structure
- Alex sees them as regular closures — no changes needed in witnesses
- Pro: Keeps Alex pure (observe, don't construct)
- Con: Baker would need to construct PSG nodes for the thunk body (forwarding call)

**Option B: Alex (MLIR-level, during transfer)**
- ApplicationWitness detects escaping partial applications
- Instead of `TRVoid`, emits PAP closure construction + generates a thunk function
- Pro: Close to emission, direct MLIR generation
- Con: Witness does more than observe — violates photographer principle

**Option C: Coeffect + Pattern (Hybrid)**
- A new coeffect `PAPClosures` identifies which partial applications need reification
- SSAAssignment pre-computes the PAP layout (thunk struct size, capture types)
- A new pattern `pPAPClosure` (in ClosurePatterns) handles construction
- ApplicationWitness PULLs from the coeffect and delegates to the pattern
- Pro: Follows four pillars — coeffect pre-computes, pattern PULLs, witness observes
- Con: More infrastructure, but architecturally clean

### 4.3 Relationship to Escape Analysis

The escape analysis coeffect (`EscapeAnalysis.fs`) now recognizes `ClosureConstruction` as an allocating site. PAP closures are the same — they allocate an env buffer that escapes.

The question is whether PAP closure construction should be detected as a new `AllocSiteKind` (e.g., `PAPConstruction`) or whether it composes with existing infrastructure:

- **If Baker generates synthetic Lambdas**: Existing `ClosureConstruction` detection handles it
- **If Alex handles it**: A new `PAPConstruction` site kind is needed

### 4.4 Relationship to Curry Flattening

Curry flattening currently makes a **global decision** to flatten nested lambdas. An alternative: **only flatten when all call sites are saturated**. If any call site is a partial application that escapes, don't flatten that lambda chain — keep it as nested closures.

This would make `makeScaledAdder` work without PAP thunks: the inner `fun x -> ...` stays as a separate Lambda, and calling `addScaled 3` invokes the outer Lambda which constructs and returns the inner closure.

**Trade-off**: Less flattening means less optimization opportunity. But correctness first.

## 5. Proposed Investigation Path

### Phase 1: Characterize the Design Space (This Document)

Understand the three options, their trade-offs, and their fit with the four pillars architecture.

### Phase 2: Prototype — Selective Flattening (Simplest Path)

Modify `CurryFlattening.flatten` to **not flatten** lambda chains where any call site is a non-local partial application. This is the least invasive change:

- CurryFlattening already detects partial applications
- If a partial application's binding escapes (returned, passed to another function), skip flattening for that target function
- The inner Lambda stays intact → regular closure construction handles it

**Escape detection for PAP bindings**: Check if `PartialAppBindings` entries are:
- Returned from their enclosing function (similar to `isTransitivelyReturned`)
- Passed as arguments to other functions
- Stored in data structures

If any escape, mark the target function as "do not flatten."

### Phase 3: PAP Thunks (Full Generality)

If selective flattening is insufficient (e.g., we want to flatten for performance but still support escaping PAPs), implement Option C:

1. New coeffect: `PAPClosureAnalysis` — identifies escaping PAPs, computes thunk layouts
2. New pattern: `pPAPClosure` — constructs PAP thunk env buffer + uniform pair
3. Thunk function generation: New nanopass or Alex top-level op emission
4. ApplicationWitness: PULL from coeffect, delegate to pattern

### Phase 4: Validation

- `makeScaledAdder` produces correct output
- Regression: samples 01-10, 17 unaffected
- Performance: saturated calls still emit direct calls (no regression)

## 6. Open Questions

1. **Baker vs Alex for thunk generation**: Does Baker have the right primitives to construct synthetic PSG nodes for forwarding calls? Or is this better handled at MLIR level?

2. **Interaction with DMM**: PAP thunk env buffers have the same lifetime questions as regular closure env buffers. Does escape analysis compose correctly?

3. **Higher-order PAP**: Can a PAP itself be partially applied? (`let f = add 1; let g = f 2; g 3` where `add` takes 3 args). The current detection handles multi-level PAP chains, but emission needs to compose.

4. **Performance**: Is selective flattening sufficient, or do we need both flattening + PAP thunks for the common case to be fast?

## 7. Files

| File | Role |
|---|---|
| `PSGElaboration/CurryFlattening.fs` | Detection — already complete |
| `Alex/Witnesses/ApplicationWitness.fs` | Emission — needs PAP closure path |
| `Alex/Patterns/ClosurePatterns.fs` | Closure construction patterns — reusable |
| `PSGElaboration/EscapeAnalysis.fs` | Escape detection — may need `PAPConstruction` |
| `PSGElaboration/SSAAssignment.fs` | SSA pre-allocation — needs PAP layout |
| `Alex/Witnesses/BindingWitness.fs` | PAP binding handling — currently `TRVoid` |
| `Alex/Witnesses/VarRefWitness.fs` | PAP reference handling — currently `TRVoid` |
