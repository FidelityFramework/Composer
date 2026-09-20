# Closure Architecture

> **MLKit-style flat closures with CCS-computed captures.**
> See [C-01 PRD](PRDs/C-01-Closures.md) for the original roadmap.
> **C-07 update:** [Closure values as data](Closure_As_Data.md) records the implemented
> materialized callback path. The older layout/coeffect sketches below are not
> an alternative contract for that path; unused code-pointer-in-environment and
> global-arena prototypes have been removed.

## 1. Executive Summary

Clef Native uses **MLKit-style flat closures** where all captured variables are stored inline in the closure struct, not via pointer chains to enclosing environments.

**Key Architectural Decision**: Capture analysis is **scope analysis**, and scope is resolved during type checking. Therefore, **capture analysis belongs in CCS**, not Composer. CCS computes captures during PSG construction and includes them directly in `SemanticKind.Lambda`. SSA assignment and closure struct layout are derived by the post-saturation coeffect nanopass (`PSGElaboration.SSAAssignment` today, scheduled into CCS; see CCS_Architecture.md); Alex reads both and computes neither.

**Implementation**: All closures use the **portable memref dialect** — byte-level `TMemRefStatic(N, TInt(IntWidth 8))` for struct representation, `pInsertValue`/`pExtractValue` for field access, `pFuncCallIndirect` for code pointer invocation. No LLVM dialect.

## 2. Layer Responsibilities

| Layer | Responsibility |
|-------|---------------|
| **CCS** | Compute captures during scope analysis, embed in `SemanticKind.Lambda` |
| **PSGElaboration/SSAAssignment** | Build `ClosureLayout` coeffect, assign SSAs |
| **Alex/Patterns/ClosurePatterns** | Compose Elements into closure construction/invocation |
| **Alex/Witnesses/LambdaWitness** | Observe Lambda nodes, delegate to ClosurePatterns |

**Capture analysis is NOT a Composer nanopass.** The PSG arrives from CCS with complete capture information.

## 3. Memory Layout

### 3.1 Flat Closure Structure

```
Closure = (fn, env)
fn:  func.constant @lambda_impl — the function-value half, never stored in env
env: environment (byte-level memref)
┌─────────────────────────────────────────────────────────┐
│ capture_0: T₀  (value for ByValue, pointer for ByRef)   │
│ capture_1: T₁                                           │
│ ...                                                     │
└─────────────────────────────────────────────────────────┘

MLIR type of env: memref<N x i8> where N = sum of capture byte sizes
```

(Interim: the code today writes a `code_ptr` word at offset 0 of the buffer and reads it back through a cast; that is the retired encoding, and it is removed by `clef/docs/fidelity/phg/Closure_Retooling_Plan.md` steps 4–5.)

### 3.2 Capture Modes

| Variable Kind | Capture Mode | In Struct | Semantics |
|---|---|---|---|
| Immutable `let x = ...` | ByValue | `T` | Copy value |
| Mutable `let mutable x = ...` | ByRef | `memref<1xT>` | A view of the binding's storage cell, never a raw pointer |

### 3.3 Extended Struct Layouts

The same byte-level struct pattern supports three contexts:

| Context | Environment layout | Extraction Base Index |
|---|---|---|
| RegularClosure | `{cap₀, cap₁, ...}` | 0 |
| LazyThunk | `{computed, value, cap₀, ...}` | 2 |
| SeqGenerator | `{state, current, cap₀, ...}` | 2 |

(Interim: the code's base indices are 1/3/3 because it still stores the code pointer; they become 0/2/2 with the retooling.)

## 4. Why Flat: the Finiteness Lemma

Flat closure is foundational to the entire structure of the Fidelity Framework: it is the **finiteness lemma** the proof stack rests on. Because the capture set is enumerated (CCS, Section 2), the layout is fixed, and every field is assigned an offset (Alex, Sections 3 and 5), a closure's reachability frontier is exactly its field list. The memory-safety judgments the PSG emits therefore quantify over finite, enumerated structure: liveness is a field list, extent is a literal, release is a single site. That keeps verification conditions in the quantifier-free fragment that discharges at Tier 2 and Tier 3.

Two representation choices lose the lemma:

| Choice | Failure Mode | Scope of Loss |
|---|---|---|
| Linked environments | Unbounded reachability through environment chains; recursive heap predicates; interactive proof territory | Transitive |
| `nativeptr` | Authority forged from an integer opens the frame: anything may alias anything; the judgment degrades to assumption | Global |

The FFI boundary (Section 7) is memory-safe and bounded exactly when every crossing has enumerated participants (the hyperedge's source set), carries a flat closure of known extent, and releases exactly once. The provable region of the computation graph is closed precisely when every crossing has that form. An unwitnessed cast is an open edge in the boundary of the provable region.

**`nativeptr`'s exit**: per the spec (`clef-lang-spec/spec/ffi-boundary.md`, `ntu-types.md`, `special-attributes-and-types.md`), `nativeptr` is not user-denotable and survives as internal `TNativePtr` plumbing. It is confined to the generated Layer 1/2 membrane and counted as the TCB metric; replaced by use-class (closure environments to the flat closure primitive of Section 3, handles to `CHandle` and branded types, buffers and strings to length-carried memref and bounded arrays, registers to `Mmio`, shared regions to `Ptr<'T, Region, Access>` with BAREWire descriptors); removed from generated code at the corpus-wide regeneration. The audit equation (cast-resolution statistics reconciled against discharged boundary obligations, C-01 PRD Section 6.7) verifies no unwitnessed cast survives.

**Proof-theoretic lineage.** The finiteness lemma is the proof shape MLKit's region discipline formalized: a type-and-effect system whose soundness theorem bounds what a computation can reach by static structure (Tofte & Talpin, "Region-Based Memory Management", Information and Computation 132(2), 1997), made compiler-inferred by the region inference algorithm (Tofte & Birkedal, "A Region Inference Algorithm", ACM TOPLAS 20(4), 1998) and kept safe under collection in Elsman, "Garbage Collection Safety for Region-based Memory Management" (TLDI 2003). Safe-for-space closure conversion is the closure-specific instance of the same bound (Shao & Appel, "Space-Efficient Closure Representations", LFP 1994; "Efficient and Safe-for-Space Closure Conversion", ACM TOPLAS 22(1), 2000). The lemma inherits that lineage and narrows it: where the region calculus bounds reachability by effect annotations over region variables, the flat closure bounds it by the enumerated field list itself.

## 5. Coeffect — ClosureLayout

All closure layout information is pre-computed during SSAAssignment (Four Pillars: Codata/Coeffects). Witnesses observe the result.

**File**: `src/MiddleEnd/PSGElaboration/Coeffects.fs`

```fsharp
type ClosureLayout = {
    LambdaNodeId: NodeId
    Captures: CaptureSlot list
    // Construction SSAs (parent scope)
    CodeAddrSSA, ClosureUndefSSA, ClosureWithCodeSSA: SSA
    CaptureInsertSSAs: SSA list
    // Arena allocation SSAs
    HeapPosPtrSSA, HeapPosSSA, HeapBaseSSA, HeapResultPtrSSA, HeapNewPosSSA: SSA
    SizeGepSSA, SizeSSA, SizeOneSSA: SSA
    // Uniform pair SSAs
    PairUndefSSA, PairWithCodeSSA, ClosureResultSSA: SSA
    // Extraction SSA (child scope)
    StructLoadSSA: SSA
    // Types
    ClosureStructType: MLIRType  // memref<N x i8>
    Context: LambdaContext
}
```

## 6. Pipeline Flow

```
Clef Source
    │
    ▼
CCS (checkLambda with free variable analysis)
    │
    ├─ Computes captures via scope analysis
    ├─ Creates SemanticKind.Lambda(params, body, captures, ...)
    │
    ▼
PSG with complete closure information
    │
    ▼
PSGElaboration/SSAAssignment
    │
    ├─ Reads captures from SemanticKind.Lambda
    ├─ Builds ClosureLayout coeffect (complete pre-computation)
    ├─ Assigns SSAs for construction (parent) and extraction (child)
    │
    ▼
Alex/Witnesses/LambdaWitness
    │
    ├─ Observes Lambda node
    ├─ Looks up ClosureLayout from coeffects
    ├─ Delegates to ClosurePatterns
    │
    ▼
Alex/Patterns/ClosurePatterns
    │
    ├─ pExtractCaptures: legacy env extraction at function entry
    ├─ EnvironmentPatterns: materialized environment allocation/access
    ├─ ApplicationPatterns: graph-established call operands
    │
    ▼
Alex/Elements (MLIRAtomics, MemRefElements, FuncElements)
    │
    ├─ pInsertValue/pExtractValue: field access
    ├─ pUndef/pAlloca/pLoad/pStore: memory ops
    ├─ pFuncCallIndirect: indirect call
    │
    ▼
MLIR (memref, arith, func dialects — portable)
    │
    ▼
mlir-opt → mlir-translate → llc → linker → Native Binary
```

## 7. FFI Boundary

Closures are Clef-internal. A native callback uses an explicit `FnPtr` entry and
an opaque `CHandle` context where the foreign signature supplies one. Generated
descriptors govern argument and result representations. Scalar-array reference
parameters are checked for sufficient storage before Composer extracts their
address; the source does not cast a numeric value into a pointer.

See the [foreign boundary specification](../../clef-lang-spec/spec/ffi-boundary.md).

## 8. Function values and the remaining declaration-promotion gap

Anonymous function expressions, including those without captures, have explicit closure-pair
planning. An alias of an existing function value preserves the value read at the binding:
`let saved = selected` snapshots `selected`, including when `selected` is mutable. It does not
create a forwarding function that reads `selected` later. A named capture-free declaration
used as a value receives a compiler-generated pair whose forwarding body saturates the
original declaration's direct-call arity.

Captured arrays retain their element representation and actual extent. Mutable
cells, function pairs and records carry an address with a compiler-settled view;
record extraction retains the field layout used by construction. Closure code
symbols are unique to anonymous lambda nodes, so equally named local bindings
in different scopes cannot collide. Parent SSA associations are restored after
each lambda body is emitted. These paths are exercised by fresh native programs
in `tests/NativeCallbacks`.

The closure environment allocator does not yet reclaim arbitrary escaping
closures. A region releasing its borrowed callback values establishes the end
of their use by carriers; it does not itself reclaim the compiler's allocations.

Promotion of a **capturing named local declaration** into a first-class value is still missing.
For example, a local `let render lo hi = ...` that captures its enclosing frame does not yet
receive the pair needed when passed to a higher-order function. The current supported source
form is `let render = fun lo hi -> ...`, which follows anonymous-expression planning. Supporting
the named form requires a compiler promotion plan that preserves its direct-call ABI and
capture lifetime; a downstream missing-SSA or missing-return error is a symptom of this gap.

## 9. References

- Shao & Appel (1994), "Space-Efficient Closure Representations"
- MLKit Programming with Regions (Tofte, Elsman)
- C-01 PRD: `docs/PRDs/C-01-Closures.md`
