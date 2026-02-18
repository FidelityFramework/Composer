# CCS: Clef Compiler Services

> **Reference**: See `docs/DTS_DMM_Architecture.md` for the full formal specification of the
> Dimensional Type System and Deterministic Memory Management that CCS implements.
> See `Architecture_Canonical.md` for the Composer pipeline that consumes the PSG CCS produces.

---

## What CCS Is

CCS (Clef Compiler Services) is the type checker, semantic analysis engine, and PSG constructor for
the Clef programming language. It is the authoritative source of type information, dimensional
constraints, escape classification, and memory placement annotations for the entire Fidelity stack.

CCS implements two joint systems that define Clef's type-safety and memory-safety guarantees:

| System | Formal Basis | Purpose |
|--------|-------------|---------|
| **DTS** (Dimensional Type System) | Finitely generated abelian groups + extended Hindley-Milner | Dimensional inference, numeric representation selection |
| **DMM** (Deterministic Memory Management) | Coeffect discipline in the PSG | Escape analysis, allocation strategy, lifetime tracking |

Both are resolved during type checking and recorded in the PSG. Composer consumes the PSG as
"correct by construction" and does not re-derive these properties.

---

## The DTS Type System

### Formal Characterization

DTS assigns every numeric value a dimension drawn from a finitely generated free abelian group
$\mathcal{D} = \mathbb{Z}^n$. Dimensional constraints are linear equations over $\mathbb{Z}$:

- Addition: operands must carry equal dimensions
- Multiplication: dimensions add (as exponent vectors)
- Division: dimensions subtract

Inference is an extension of Hindley-Milner unification. Dimension variables are associated with
type variables and are solved by Gaussian elimination over $\mathbb{Z}$. The inference is:
- **Decidable** in polynomial time (linear algebra, not SMT)
- **Complete** — no annotations required in the common case
- **Principal** — the inferred type is the most general

### Why DTS Is Not Dependent Typing

| Property | DTS | Dependent Types |
|---|---|---|
| Type checking | Decidable (linear algebra over $\mathbb{Z}$) | Undecidable in general |
| Inference | Complete and principal | Incomplete; requires annotations |
| Runtime representation | No runtime cost; metadata only | May require runtime witnesses |
| Expressiveness | Abelian group constraints on numerics | Arbitrary predicates over values |

DTS occupies a specific algebraic niche. Decidability guarantees bounded-time inference for every
query — critical for language server response-time guarantees.

### Preservation Through Compilation

Unlike F# Units of Measure (which erases at IL generation), DTS annotations survive the full
compilation pipeline:

```
Source → Typed AST → PSG (as node attributes) → MLIR (as clef.dim attributes)
       → Target dialects (guide representation selection)
       → Machine code (lowered to debug metadata only)
```

Dimensional annotations never influence control flow or data layout. They are metadata that guides
**numeric representation selection** per target.

### Representation Selection

Because dimensions survive to code generation, the compiler can select numeric formats based on
dimensional domain:

| Target | Selection |
|--------|-----------|
| x86_64 | IEEE 754 float64 (default) |
| Xilinx FPGA | posit\<32, es=2\> (tapered precision matches domain range) |
| RISC-V + Xposit | Hardware quire (exact accumulation) |
| Neuromorphic (Loihi 2) | Capability failure if quire required |

The selection is a deterministic function from dimensional range + target capabilities — not a
heuristic. The language server surfaces this as a per-target resolution panel.

---

## The Native Type Universe (NTU)

CCS maps standard Clef types to platform-generic NTUKind representations. Width is resolved by
Alex/platform bindings at code generation time; CCS enforces type *identity*, not type *width*.

### Platform-Dependent Types

| Clef Type | NTUKind | x86_64 | ARM32 |
|-----------|---------|--------|-------|
| `int` | `NTUint` | i64 | i32 |
| `uint` | `NTUuint` | u64 | u32 |
| `nativeint` | `NTUnint` | i64 | i32 |
| `nativeptr<'T>` | `NTUptr<'T>` | 8 bytes | 4 bytes |

### Fixed-Width Types

| Clef Type | NTUKind | Always |
|-----------|---------|--------|
| `int32` | `NTUint32` | i32 |
| `int64` | `NTUint64` | i64 |
| `float` | `NTUfloat64` | f64 |
| `float32` | `NTUfloat32` | f32 |
| `bool` | `NTUbool` | i1 |
| `char` | `NTUchar` | u32 (UTF-32 code point) |
| `string` | `NTUstring` | fat pointer `{ptr<u8>, length: i64}` |
| `unit` | `NTUunit` | zero-sized |

**Important**: `int` in Clef follows ML/Rust semantics (platform word = 64-bit on x86_64), not
.NET's 32-bit `System.Int32`. Use `int32` for explicit 32-bit values.

**Important**: `string` is a UTF-8 fat pointer, not `System.String`. `char` is a UTF-32 code
point (4 bytes), not .NET's UTF-16 `System.Char`.

DTS extends NTU numerics with dimensional annotations. A `float<newtons>` has NTUKind `NTUfloat64`
plus dimensional metadata `(1, -2, 1, 0, ...)` (length¹ · time⁻² · mass¹).

---

## DMM: Deterministic Memory Management

### Escape Analysis as Coeffect Propagation

Memory allocation strategy is a **coeffect**: a contextual requirement, not an effect. CCS tracks
allocation, lifetime, and escape behavior in the PSG as annotations on computation nodes.

Escape analysis propagates lifetime constraints through the graph. When a value's required
lifetime exceeds its tentative lifetime, the value is promoted and the promotion is recorded in
the PSG. The promotion is visible and navigable — it is not a silent optimization.

### Escape Classification

| Classification | Allocation Strategy | Lifetime Bound |
|---|---|---|
| `StackScoped` | Stack (`memref.alloca`) | Lexical scope of binding |
| `ClosureCapture(t)` | Arena (closure environment) | Lifetime of closure `t` |
| `ReturnEscape` | Arena (caller's scope) | Caller's scope |
| `ByRefEscape` | Arena (parameter's origin scope) | Origin scope of aliased reference |

The language server surfaces the escape path and proposes concrete restructuring alternatives
(caller-provided buffer, continuation-passing, explicit annotation).

### The `inline` Keyword and Escape

When a function allocates on the stack and returns a pointer, marking it `inline` causes CCS to
expand the body at call sites, lifting the allocation to the caller's frame. This is a mandatory
inline constraint (not performance-motivated inlining). CCS verifies that the inlined allocation
does not escape the caller.

### Coeffect Specification Models

Three models form a spectrum of inference scope:

| Model | Developer Provides | Compiler Infers | PSG Saturation |
|---|---|---|---|
| **Push** (explicit) | Full coeffect constraints via `[<Target: ...>]` `[<Memory: ...>]` | Internal details | Immediate |
| **Bounded** (scoped) | Scope boundary via computation expression (`arena { ... }`) | Allocation within scope | Fast |
| **Poll** (implicit) | Nothing | All coeffects from usage context | Context-dependent |

No model is incorrect. The push model produces PSGs that saturate faster and remain stable under
dependency changes. The poll model imposes no annotation burden. The language server shows the
consequences of each choice.

---

## PSG as Design-Time Resource

CCS produces the Program Semantic Graph, which serves two roles simultaneously:
1. Compilation artifact consumed by Composer
2. Persistent design-time resource consumed by Lattice (language server)

### Three-State Node Model

| State | Elaborated | Saturated | Active | Visible to Optimizer | Visible to Language Server |
|---|---|---|---|---|---|
| **Live** | Yes | Yes | Yes | Yes | Full resolution |
| **Latent** | Yes | Yes | No | No | Dimmed, resolution preserved |
| **Fresh** | No | No | No | No | Syntax only |

**Latent preservation**: when a node becomes unreachable (feature flag disabled, dependency
removed), CCS marks it latent rather than deleting it. Reactivation is $O(\text{boundary})$;
re-elaboration from source is $O(\text{subgraph})$.

### Coeffects Computed During Elaboration

All of these are computed *before* the graph traversal that generates MLIR:

| Coeffect | What It Resolves | Consumed By |
|---|---|---|
| Emission strategy | Inline / separate function / module init | MLIR generation |
| Capture analysis | Lambda capture set | Closure layout, escape |
| Lifetime requirements | Minimum value lifetime | Allocation strategy |
| SSA pre-assignment | SSA identifier for each node's result | MLIR emission |
| Dimensional resolution | Physical dimension, representation | Representation selection |
| Target reachability | Per-target reachability bitvector | Code generation filtering |

The Zipper traversal in Alex is **purely navigational** — it observes pre-computed coeffects and
emits the corresponding MLIR. It does not compute, infer, or decide.

### Design-Time Feedback (via Lattice)

Because the PSG persists, Lattice surfaces compilation-internal analysis as interactive guidance:

- **Dimensional resolution** — hover shows dimension, per-target representation
- **Escape diagnostics** — shows escape path, allocation promotion, restructuring alternatives
- **Cache locality estimates** — for hot loops, estimates residency based on size + allocation
- **Cross-target transfer** — at hardware boundaries (FPGA→CPU), shows protocol, latency, fidelity

---

## Layer Separation

| Layer | Responsibility | Does NOT |
|---|---|---|
| **CCS** | DTS inference, DMM coeffects, NTU type universe, PSG construction, editor services | Generate code or know targets |
| **Composer** | Receives PSG, applies lowering nanopasses, MLIR generation | Re-derive types or coeffects |
| **Alex/Zipper** | Traverses PSG, emits MLIR via platform bindings | Pattern-match on names, re-infer |
| **Fidelity.Platform** | Resolves NTU widths, posit configs, syscall numbers per target | Typecheck or infer |
| **Lattice** | Language server consuming live PSG for design-time services | Drive compilation |

CCS builds the PSG with:
- Native types attached during type checking (not overlaid later)
- SRTP resolved during type checking (not in a separate pass)
- Dimensional constraints propagated during unification
- Escape classification complete before MLIR traversal begins

---

## What CCS Does NOT Contain

- **Runtime implementations** — in Fidelity.Platform (Console, Span, Arena, etc.)
- **MLIR code generation** — in Composer/Alex
- **Platform-specific bindings** — in Fidelity.Platform substrate packages
- **Arbitrary predicate verification** — DTS restricts to abelian group constraints; for full
  dependent-type-style proofs, see the F* interoperability path described in the DTS paper

---

## Related Documentation

| Document | Purpose |
|---|---|
| `docs/DTS_DMM_Architecture.md` | Full formal specification of DTS + DMM |
| `Architecture_Canonical.md` | Composer pipeline, Alex traversal model |
| `Alex_Architecture_Overview.md` | Three-layer Elements/Patterns/Witnesses model |
| `NTU_Architecture.md` | NTUKind type system detail |
| `PSG_Nanopass_Architecture.md` | Nanopass principles, soft-delete reachability |
| `Platform_Binding_Model.md` | Fidelity.Platform binding architecture |
| `~/repos/Clef_migration/DTS-DMM-CLEF_MIME/dts-dmm-paper.md` | Authoritative formal specification |
