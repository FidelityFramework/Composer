# Clef Semantics: Where We Stand

> Three classical approaches to formal semantics (operational, denotational, axiomatic)
> and how they map onto Composer's architecture.

---

## The Three Lenses

Operational semantics models execution.
Denotational semantics models mathematical meaning.
Axiomatic semantics models provability.

These are not competing views. They are complementary perspectives on what programs
are and what it means for them to be correct. The question for Clef and Composer is:
which of these three has the architecture already committed to, which does it partially
realize, and where are the gaps?

---

## 1. Operational Semantics: The Pipeline IS the Abstract Machine

Operational semantics defines meaning by describing how programs execute on an abstract
machine. A machine state is a term; evaluation is transitions between states.

**Composer's nanopass pipeline is a big-step operational semantics in disguise.**

Each nanopass takes a PSG in language L_n and produces a PSG in language L_{n+1}. The
language transformations are explicit:

```
L_SynExpr → L_PSG₁ (structural)
L_PSG₁   → L_PSG₂ (symbol-correlated)
L_PSG₂   → L_PSG₃ (reachability-marked)
L_PSG₃   → L_PSG₄ (typed overlay)
L_PSG₄   → L_PSG₅ (flattened)
...
L_PSG_n  → MLIR
```

Each intermediate form is inspectable via the `-k` flag. Each transition does one
thing. The correspondence with nanopass framework theory (Sarkar, Keep; Indiana) is
exact: each pass is a **single-step reduction** in a meta-level operational semantics
where the "term" being reduced is the entire program representation.

The small-step character shows up in the zipper traversal. Alex's PSGZipper walks the
enriched PSG node by node. At each node, an XParsec pattern fires or doesn't. The
zipper carries state (SSA counters, builder accumulator). This is a **small-step
abstract machine** where:

- The **state** is `(ZipperFocus, AccumulatorState)`
- The **transition** is: match pattern, emit MLIR, advance focus
- The **terminal state** is: zipper exhausted, MLIR module complete

**What we have**: A concrete operational semantics embodied in code. The pipeline
defines how a Clef program runs, not on hardware, but through the compiler's abstract
machine.

**What's formalized**: The nanopass intermediate languages are implicitly defined by
the PSG node types at each phase. The `-k` artifacts make each intermediate state
observable.

**What's missing**: The transition rules exist as F# code but not as inference rules
on paper. A formal small-step relation `⟨PSG_n, σ⟩ → ⟨PSG_{n+1}, σ'⟩` has not been
written down. This would be the natural next step for anyone wanting to prove compiler
correctness.

---

## 2. Denotational Semantics: CCS as Interpretation Function

Denotational semantics maps programs to mathematical objects: numbers, functions,
elements of semantic domains. The interpretation function `⟦·⟧` assigns meaning
compositionally.

**CCS is a denotational semantics for Clef.**

Consider what CCS actually does:

| Clef Construct | CCS Denotation |
|----------------|----------------|
| `42` | NTUint64 with value 42 |
| `"hello"` | NTUstring: fat pointer `{ptr<u8>, length: i64}` |
| `fun x -> x + 1` | NTUfun: function type with native calling convention |
| `3.0<m/s²>` | NTUfloat64 + dimensional vector `(1, -2, 0, ...)` |
| `let mutable x = 0` | PSG node with ArenaAffinity, escape classification |

CCS maps every syntactic construct to a semantic object. Not to a runtime value, but
to a **native type in the NTU (Native Type Universe)** enriched with dimensional
metadata and memory coefficients. This is a denotational semantics where:

- The **semantic domain** is `NativeType × DimensionalVector × EscapeClassification`
- The **interpretation function** is type checking + dimensional inference + escape analysis
- **Compositionality** holds: the meaning of `f x` is determined by the meanings of `f` and `x`

The DTS (Dimensional Type System) makes this particularly clean. Dimensions form a
finitely generated free abelian group D = Z^n. Dimensional inference is Gaussian
elimination over Z: decidable, complete, principal. This is not an approximation or
heuristic; it is a **mathematical denotation** of what numeric values mean in terms of
physical quantity.

The PSG itself is the denotation. When CCS produces a `SemanticGraph` with nodes
carrying `NativeType`, `SRTPResolution`, `ArenaAffinity`, and `EmissionStrategy`, it
has assigned a complete mathematical meaning to every subexpression. Composer then
consumes this as "correct by construction."

**What we have**: A compositional interpretation from Clef syntax to native semantic
objects, implemented as CCS type checking and PSG construction.

**What's formalized**: The NTU type universe is specified. DTS inference is
mathematically characterized (abelian groups, Gaussian elimination). Escape
classification has a defined lattice.

**What's missing**: The interpretation function itself, the mapping from syntax to
NTU/DTS/DMM denotations, exists as CCS code but not as a formal `⟦·⟧` on paper.
Writing it down would yield a denotational semantics for Clef in the classical sense.

---

## 3. Axiomatic Semantics: DMM and Escape Analysis Through Coeffects

Axiomatic semantics shifts focus from behavior to what can be proven about a program.
Meaning is given by logical rules taken as primary. The tradition gave us invariants,
Hoare logic, and the idea that reasoning is central to programming.

**Clef's DMM (Deterministic Memory Management) is an axiomatic semantics for resource
safety, realized through coeffect propagation in the PSG.**

This is the most architecturally novel of the three perspectives. Rust enforces memory
safety through a borrow checker on the surface syntax. Traditional Hoare logic reasons
about heap assertions in pre/post conditions. Clef takes a different path: it embeds
resource axioms directly into the intermediate representation as **coeffects**,
contextual requirements that propagate through the Program Semantic Graph and are
verified before any code generation begins.

### What Coeffects Are (and Are Not)

In PL theory, **effects** describe what a computation does to its environment (mutation,
I/O, exceptions). **Coeffects** describe what a computation requires from its
environment (context, resources, capabilities). The term was introduced by Petricek,
Orchard, and Mycroft (ICALP 2013) for context-dependent computations.

Clef extends this idea to memory. Every computation node in the PSG carries coeffect
annotations that express what that computation requires from the memory system:

```
┌──────────────────────────────────────────────────────┐
│  SemanticNode                                        │
│  ├── Type: NativeType          (what it IS)          │
│  ├── ArenaAffinity: region     (where it LIVES)      │
│  ├── EmissionStrategy: how     (how it's LOWERED)    │
│  └── IsReachable: bool         (whether it MATTERS)  │
│                                                      │
│  These are not runtime values. They are compile-time │
│  assertions about the program's resource behavior.   │
└──────────────────────────────────────────────────────┘
```

The critical distinction: coeffects are not effects tracked through a monad. They are
**properties of the computation context** that flow backward from usage sites to
definition sites. When CCS discovers that a value escapes its defining scope, it does
not insert a runtime check. Instead, it reclassifies the value's allocation strategy
and records the promotion in the PSG. The promotion is visible, navigable, and
auditable.

### Escape Analysis as Axiom Verification

DMM performs escape analysis as coeffect propagation. The escape classifications form
an ordered lattice of increasingly permissive allocation strategies:

```
StackScoped  ⊑  ClosureCapture(t)  ⊑  ReturnEscape  ⊑  ByRefEscape
    │                  │                    │                 │
    │                  │                    │                 │
  "lives on       "lives as long       "must be in       "origin scope
   the stack"      as closure t"        caller arena"     must persist"
```

Each classification is an **axiom about lifetime**, a provable property that the
compiler verifies statically and that downstream stages trust without re-verification:

| Classification | Axiom | Proof Obligation |
|----------------|-------|------------------|
| `StackScoped` | Value does not outlive its lexical scope | No reference escapes scope boundary |
| `ClosureCapture(t)` | Value lives as long as closure `t` | Closure lifetime ≥ value lifetime |
| `ReturnEscape` | Value must be allocated in caller's arena | Return type analysis confirms escape |
| `ByRefEscape` | Reference's origin scope must be preserved | Alias analysis tracks origin |

When a value's **required lifetime** (determined by how it is used) exceeds its
**tentative lifetime** (determined by where it is defined), the value is **promoted**
to a longer-lived region. This promotion is not a silent optimization. It is recorded
in the PSG as a change in `ArenaAffinity`, and the language server (Lattice) surfaces
it to the developer with the escape path and restructuring alternatives.

This is the Hoare logic parallel: each escape classification is a precondition on the
code that uses the value, and the promotion rule is an inference rule that weakens the
precondition when the original one cannot be satisfied.

### Region-Typed Pointers: Axioms in the Type System

Clef makes memory regions first-class in the type system. Every pointer carries its
region and access permissions:

```
Ptr<'T, 'Region, 'Access>

Ptr<uint32, Peripheral, ReadWrite>   -- GPIO register (volatile, uncacheable)
Ptr<byte, Flash, ReadOnly>           -- Constant data (read-only, link-time placed)
Ptr<int, Stack, ReadWrite>           -- Stack buffer (scope-bounded)
Ptr<float, Arena, ReadWrite>         -- Arena allocation (bulk lifetime)
```

Region mismatches are type errors, axiom violations caught at compile time. You cannot
assign a `Peripheral` pointer to a `Stack` variable. You cannot write through a
`ReadOnly` access kind. These are not runtime checks; they are proof obligations
discharged by the type checker.

The five memory regions (Stack, Arena, Peripheral, Sram, Flash) each carry distinct
semantic axioms about volatility, cacheability, and lifetime:

| Region | Volatile | Cacheable | Lifetime |
|--------|----------|-----------|----------|
| `Stack` | No | Yes | Lexical scope |
| `Arena` | No | Yes | Arena lifetime (batch dealloc) |
| `Peripheral` | Yes | No | Hardware-mapped (infinite) |
| `Sram` | No | Yes | Explicit management |
| `Flash` | No | Yes | Program lifetime (read-only) |

These are axiomatic properties. They hold by construction because the type system
enforces them. Code generation consumes these classifications to select the correct
load/store instructions, barrier insertions, and allocation strategies per target.

### Three Specification Models: How Axioms Enter the System

Clef provides three ways for developers to introduce memory axioms, forming a spectrum
from explicit annotation to full inference:

| Model | Developer Provides | Compiler Infers | Character |
|-------|-------------------|-----------------|-----------|
| **Push** (explicit) | Full coeffect constraints via attributes | Internal details only | Axioms stated |
| **Bounded** (scoped) | Scope boundary via `arena { ... }` CE | Allocation within scope | Axioms bounded |
| **Poll** (implicit) | Nothing | All coeffects from usage context | Axioms derived |

All three models converge on the same verified properties. The push model produces PSGs
that saturate faster because axioms are given, not discovered. The poll model imposes no
annotation burden because axioms are inferred from usage. The bounded model sits between:
the developer draws the scope boundary, and the compiler fills in the details.

This is analogous to the relationship between Hoare logic (where invariants are stated)
and abstract interpretation (where invariants are computed). Both produce the same class
of verified properties; they differ in how much the human states versus how much the
machine derives.

### The `inline` Keyword: Axiom-Driven Code Transformation

When a function allocates on the stack and returns a pointer, the value escapes its
defining scope. That is an axiom violation of `StackScoped`. Marking the function
`inline` triggers mandatory inlining: CCS expands the body at call sites, lifting the
allocation to the caller's frame. This is not performance-motivated inlining. It is an
**axiom-restoring transformation**: inlining converts a `ReturnEscape` back into a
`StackScoped` by eliminating the scope boundary that caused the escape.

CCS verifies that after inlining, the allocation does not escape the caller. If it
still does, that is a compile-time error. The axiom cannot be satisfied.

### How This Differs from Rust, Linear Types, and Traditional GC

The natural question: how is this different from Rust's borrow checker, linear type
systems, or garbage collection?

| Approach | Where Safety Lives | Developer Experience | Runtime Cost |
|----------|-------------------|---------------------|-------------|
| **GC** (Java, Go, C#) | Runtime collector | Invisible (no annotations) | Unpredictable pauses |
| **Borrow checker** (Rust) | Surface syntax constraints | Lifetime annotations in signatures | Zero (compile-time) |
| **Linear types** (Clean, Idris) | Type system with usage tracking | Linear/affine annotations | Zero (compile-time) |
| **DMM coeffects** (Clef) | PSG node attributes propagated by CCS | Three-level spectrum (explicit to inferred) | Zero (compile-time) |

Key differences from Rust specifically:

1. **Inference depth**: Rust's borrow checker operates on surface syntax. Clef's escape
   analysis operates on the PSG after type checking and SRTP resolution, so it sees the
   program after all type-level computation has been resolved. This means escape
   classification can account for information that is not syntactically visible.

2. **Promotion not rejection**: When Rust detects a lifetime violation, it rejects the
   program. When Clef detects a lifetime violation, it promotes the allocation to a
   longer-lived region and records the promotion. The developer sees what happened and
   can choose to restructure, but the program compiles.

3. **Region is in the type, not in the annotation**: Rust lifetimes are annotation
   parameters (`'a`). Clef regions are type parameters (`Ptr<T, Region, Access>`).
   Region constraints participate in type inference and are checked by the same
   unification engine that handles all other type constraints.

4. **Coeffect not effect**: Rust's borrow checker is fundamentally about tracking
   *effects* (mutation, moves). Clef's DMM is about tracking *coeffects* (what the
   context must provide). This inversion means the analysis flows backward from usage
   to definition, rather than forward from definition to usage.

### The Codata Connection

The codata principle says: coeffects are computed before the walk, observed during the
walk. This is the architectural realization of the axiomatic perspective. By the time
the zipper traverses the PSG for MLIR generation, every node already carries its proven
resource properties. The walk does not decide whether code is safe. That was already
proven by CCS. The walk merely observes the proofs and emits code accordingly.

This is precisely the axiomatic semantics paradigm: meaning is given by what has been
proven about the program, and execution (code generation) follows from those proofs.

### DTS as Algebraic Axiom System

DTS (Dimensional Type System) adds a second axiomatic dimension orthogonal to memory.
Dimensions form a finitely generated free abelian group D = Z^n. The group axioms are
enforced by the type checker:

- `mass<kg> + mass<lb>` produces a type error (dimension mismatch; axiom violation)
- `distance<m> / time<s>` yields `velocity<m/s>` (axiom-derived conclusion)
- `force<N> = mass<kg> * acceleration<m/s²>` is verified (dimensional identity)

These are not heuristics or warnings. They are algebraic laws, axioms of the group
structure, that the type system enforces as provable properties of every numeric
expression.

### What We Have

A coeffect discipline that proves resource-safety properties at compile time. Escape
analysis through DMM that promotes rather than rejects. Region-typed pointers that make
memory axioms first-class in the type system. A dimensional type system that enforces
algebraic laws. Three specification models that let developers choose their annotation
level. All producing results that downstream stages trust without re-verification.

### What's Formalized

Escape classifications form a lattice with a well-defined promotion order. DTS
constraints are abelian group equations over Z. Region types carry semantic axioms
(volatility, cacheability, lifetime) enforced by construction. The PSG records all
verified properties as node attributes, serializable for inspection.

### What's Missing

A formal proof calculus, whether a Hoare logic or separation logic variant, that
axiomatically characterizes the full set of provable properties. The compiler *does*
prove things (escape safety, dimensional consistency, region correctness), but the
logic it uses has not been extracted as an independent formal system. Writing it down
would yield a resource logic for Clef that could be studied, extended, and verified
independently of the compiler implementation.

The closest existing formal framework is Petricek et al.'s coeffect calculus, but
Clef's coeffects are richer. They include escape classification, region tracking, and
dimensional constraints, not just flat context requirements. A Clef-specific coeffect
calculus would extend the Petricek framework with the lattice structure of escape
classifications and the abelian group structure of DTS.

---

## Synthesis: The Three Lenses in Composer

```
              Clef Source
                  │
                  │ ⟦·⟧  (denotational: CCS type checking)
                  ▼
              PSG with NTU types, DTS dimensions, DMM coeffects
                  │
                  │ ├── Coeffects: axiomatic properties (escape, dimension, lifetime)
                  │ └── Codata: lazy observations computed before walk
                  │
                  │ →  (operational: nanopass reductions)
                  ▼
              Enriched PSG
                  │
                  │ →  (operational: zipper small-step machine)
                  ▼
              MLIR → LLVM → Native Binary
```

The three semantic perspectives are not layered on top of each other. They are
**interleaved through the pipeline**:

1. **CCS** is primarily denotational: it assigns meaning (types, dimensions,
   coeffects) compositionally
2. **Coeffects** are primarily axiomatic: they are provable properties that
   downstream stages trust
3. **The nanopass pipeline + zipper** is primarily operational: it defines how the
   program transforms step by step toward executable form

This is not accidental. The four pillars (Codata, Zipper, Combinators, Elision) map
onto these semantic traditions:

| Pillar | Semantic Tradition | Connection |
|--------|-------------------|------------|
| **Codata/Coeffects** | Axiomatic | Pre-proven properties, observed not re-derived |
| **Zipper** | Operational (small-step) | State machine traversal, attention over sub-tree |
| **Combinators** | Denotational | XParsec patterns as compositional meaning extraction |
| **Elision** | Operational | PSG to MLIR as the residual of observation (final reduction) |

---

## What Formalization Would Buy Us

The architecture already *implements* all three semantic perspectives. Formalizing them
would provide:

1. **Compiler correctness proofs**: show that the operational semantics (nanopass
   pipeline) preserves the denotational semantics (CCS-assigned meanings). This is
   the classical adequacy theorem.

2. **Soundness of coeffect discipline**: show that if CCS says `StackScoped`, the
   generated native code actually respects stack discipline. This is the axiomatic
   counterpart.

3. **Cross-target equivalence**: show that the same PSG compiled for CPU (LLVM) and
   FPGA (CIRCT) produces programs with the same denotational meaning, modulo
   representation differences like IEEE float vs. posit.

4. **DTS metatheory**: decidability, completeness, and principality of dimensional
   inference are claimed. A formal proof would make this a theorem, not a claim.

These are not prerequisites for shipping. The compiler works. But for a language that
carries proofs through compilation (the Fidelity mission), having the meta-level proofs
about the compiler itself is the natural completion of the story.

---

## Practical Next Steps

| Step | Scope | Artifact |
|------|-------|----------|
| Write down nanopass transition rules | Operational | Inference rules for each PSG phase transition |
| Extract CCS interpretation function | Denotational | Formal `⟦·⟧` from Clef syntax to NTU × DTS × DMM |
| Characterize coeffect proof rules | Axiomatic | Separation logic or Hoare logic for DMM discipline |
| Prove adequacy | All three | Pipeline preserves denotational meaning |
| DTS metatheory | Denotational + Axiomatic | Decidability/completeness/principality proofs |

The right order is: operational first (it's closest to what exists), then denotational
(CCS already computes it), then axiomatic (coeffects are already enforced, just not
formally axiomatized).
