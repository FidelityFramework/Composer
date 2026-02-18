# Alex Architecture Overview

> **See `Architecture_Canonical.md` for the authoritative two-layer model.**

## The "Library of Alexandria" Model

Alex is Composer's **multi-dimensional hardware targeting layer**. It consumes the PSG (Program Semantic Graph) and generates platform-optimized MLIR.

## Core Responsibility: The Non-Dispatch Model

> **Key Insight: Centralization belongs at the OUTPUT (MLIR Builder), not at DISPATCH (traversal logic).**

Alex generates MLIR through Zipper traversal and platform Bindings. There is **NO central dispatch hub**.

```
PSG Entry Point
    ↓
Zipper.create(psg, entryNode)     -- provides "attention"
    ↓
Fold over structure (pre-order/post-order)
    ↓
At each node: XParsec matches locally → MLIR emission
    ↓
Extern primitive? → ExternDispatch.dispatch(primitive)
    ↓
MLIR Builder accumulates           -- correct centralization
    ↓
Output: Complete MLIR module
```

Platform differences are **data** (syscall numbers, register conventions), not routing logic.

## Key Components

| Component | Purpose |
|-----------|---------|
| `Traversal/PSGZipper.fs` | Bidirectional PSG traversal ("attention") |
| `Traversal/PSGXParsec.fs` | Local pattern matching combinators |
| `Bindings/BindingTypes.fs` | ExternDispatch registry, platform types |
| `Bindings/*/` | Platform bindings (data, not routing) |
| `CodeGeneration/MLIRBuilder.fs` | MLIR accumulation (correct centralization) |
| `Pipeline/CompilationOrchestrator.fs` | Entry point |

**Note:** PSGEmitter.fs and PSGScribe.fs were removed - they were central dispatch antipatterns.

## The Three-Layer Architecture: Elements, Patterns, Witnesses

> **CRITICAL: Shared Infrastructure ≠ Witness Dependencies**

Alex uses a three-layer architecture inspired by the "Baker" metaphor (Elements = Primitives, Patterns = Ingredients, Witnesses = Recipes):

```
Elements/    (module internal)  →  Atomic MLIR operations with XParsec state threading
Patterns/    (public)           →  Composable elision templates (~50 lines each)
Witnesses/   (public)           →  Thin observers (~20 lines each)
```

### Layer Responsibilities

| Layer | Visibility | Purpose | Line Limit |
|-------|-----------|---------|------------|
| **Elements/** | `module internal` | Atomic MLIR ops by tier | N/A |
| **Patterns/** | `public` | Composable elision templates | ~50 lines |
| **Witnesses/** | `public` | Thin observers | ~20 lines |

**Type-Level Firewall**: Elements are `module internal` - witnesses physically CANNOT import them. Witnesses must delegate to Patterns, which use Elements.

### The Key Distinction: Shared Vocabulary vs Execution Coupling

**Parallelism is preserved when witnesses share VOCABULARY (Elements/Patterns), not when they share EXECUTION (calling each other).**

#### ✅ PARALLEL-SAFE: Using Shared Elements/Patterns

Multiple witnesses can use the same Element or Pattern without creating dependencies:

```fsharp
// LazyWitness.fs
let witnessLazyForce ctx node =
    // Uses pCondBranch from LLVMElements
    let! branchOp = pCondBranch condSSA "then_block" "else_block"
    ...

// ControlFlowWitness.fs
let witnessIfThenElse ctx node =
    // Also uses pCondBranch from LLVMElements
    let! branchOp = pCondBranch condSSA "then_block" "else_block"
    ...
```

**Why this is parallel-safe**:
- Both witnesses import `Alex.Elements.LLVMElements` (shared vocabulary)
- NO witness-to-witness calls
- Execution is fully independent
- Can run in parallel via IcedTasks

#### ❌ CREATES DEPENDENCY: Calling Another Witness

```fsharp
// LazyWitness.fs - WRONG!
let witnessLazyForce ctx node =
    // Calls ControlFlowWitness - creates dependency!
    ControlFlowWitness.witnessIfThenElse ctx node
```

**Why this breaks parallelism**:
- LazyWitness now depends on ControlFlowWitness completing first
- Creates execution ordering constraint
- Forces sequential execution
- Violates the parallel nanopass architecture

### Architectural Invariants for Parallelism

For witnesses to remain truly parallel (via IcedTasks):

1. **Witnesses MUST NOT call other witnesses** - This creates dependencies
2. **Witnesses MAY share Elements** - Common vocabulary, no execution coupling
3. **Witnesses MAY share Patterns** - Patterns compose Elements, still no coupling
4. **Witnesses MUST return `WitnessOutput.skip`** for unhandled nodes - Other nanopasses handle them
5. **All state is read-only coeffects** - No mutable shared state

### Case Study: LazyForce and Control Flow

**Initial Finding**: LazyForce needs control flow (conditional branching) to check if a lazy value has been computed.

**Question**: Does this create a dependency on ControlFlowWitness?

**Answer**: No, if implemented correctly. Three parallel-safe approaches:

#### Option 1: Shared Element (Cleanest)

Both LazyWitness and ControlFlowWitness use `pCondBranch` from LLVMElements:

```fsharp
// LLVMElements.fs (shared vocabulary)
let pCondBranch (cond: SSA) (thenLabel: string) (elseLabel: string) : PSGParser<MLIROp> =
    parser {
        return MLIROp.LLVMOp (LLVMOp.CondBranch (cond, thenLabel, elseLabel))
    }

// LazyWitness.fs (independent)
let witnessLazyForce ctx node =
    let! branchOp = pCondBranch computedSSA "return_cached" "compute_value"
    ...

// ControlFlowWitness.fs (independent)
let witnessIfThenElse ctx node =
    let! branchOp = pCondBranch condSSA "then_block" "else_block"
    ...
```

**Why parallel-safe**: Shared Element, no witness-to-witness calls.

#### Option 2: Shared Pattern (Compositional)

Create a Pattern in ElisionPatterns.fs that both witnesses use:

```fsharp
// ElisionPatterns.fs (shared vocabulary)
let pBuildConditionalBranch (cond: SSA) (thenOps: MLIROp list) (elseOps: MLIROp list) =
    parser {
        let! branchOp = pCondBranch cond "then_block" "else_block"
        let! thenBlock = pBlock "then_block" thenOps
        let! elseBlock = pBlock "else_block" elseOps
        return [branchOp; thenBlock; elseBlock]
    }

// LazyWitness.fs (uses pattern)
let witnessLazyForce ctx node =
    let! ops = pBuildConditionalBranch computedSSA returnOps computeOps
    ...

// ControlFlowWitness.fs (uses pattern)
let witnessIfThenElse ctx node =
    let! ops = pBuildConditionalBranch condSSA thenOps elseOps
    ...
```

**Why parallel-safe**: Shared Pattern, no witness-to-witness calls.

#### Option 3: CCS Elaboration (Moves Logic Upstream)

Have CCS elaborate `Lazy.force` into explicit control flow nodes in the PSG:

```fsharp
// CCS elaborates:
Lazy.force x

// Into PSG nodes:
IfThenElse(
    cond = x.computed,
    thenBranch = x.value,
    elseBranch = Call(x.code_ptr, x.captures)
)

// ControlFlowWitness handles the IfThenElse node
// LazyWitness only handles LazyExpr construction
```

**Why parallel-safe**: No LazyForce witness needed. ControlFlowWitness handles the PSG structure that CCS created.

### Enforcement Mechanisms

#### 1. Type-Level Firewall

Elements are `module internal` - witnesses cannot import them:

```fsharp
// Elements/LLVMElements.fs
module internal Alex.Elements.LLVMElements  // INTERNAL!

// Witnesses/LazyWitness.fs
open Alex.Elements.LLVMElements  // COMPILATION ERROR!
```

This forces witnesses to go through Patterns, preventing direct Element usage.

#### 2. CI Validation

Detect witness-to-witness imports:

```bash
#!/bin/bash
# In .github/workflows/validate-architecture.yml

if grep -r "open Alex.Witnesses" src/Alex/Witnesses/ --exclude-dir=obj; then
    echo "FAIL: Witnesses cannot import other witnesses (creates dependencies)"
    exit 1
fi

if grep -r "open Alex.Elements" src/Alex/Witnesses/ --exclude-dir=obj; then
    echo "FAIL: Witnesses cannot import Elements (use Patterns instead)"
    exit 1
fi
```

#### 3. Nanopass Execution Test

Verify parallel execution produces identical results to sequential:

```fsharp
[<Test>]
let ``Parallel nanopass execution equals sequential execution`` () =
    let graph = loadGraph "samples/FidelityHelloWorld/04_HelloWorldFullCurried/psg.json"
    let coeffects = computeCoeffects graph

    let seqResult = executeNanopasses { EnableParallel = false } registry graph coeffects
    let parResult = executeNanopasses { EnableParallel = true } registry graph coeffects

    Assert.AreEqual(seqResult.TopLevelOps, parResult.TopLevelOps)
```

If witnesses have hidden dependencies, parallel execution will produce different results.

### The Dependency Tree is Not a Failure

**Key Insight**: A dependency tree between LAYERS (Elements → Patterns → Witnesses) is correct and intentional. This is composition, not coupling.

**What would be a failure**: Dependencies WITHIN a layer (Witness A → Witness B) breaks parallelism.

**The architecture ensures**:
- Elements compose into Patterns (bottom-up composition)
- Patterns compose into Witnesses (bottom-up composition)
- Witnesses do NOT compose into each other (flat parallelism)

### Summary

| Relationship | Effect on Parallelism | Example |
|--------------|----------------------|---------|
| Witness → Element | ✅ Parallel-safe | LazyWitness uses `pCondBranch` |
| Witness → Pattern | ✅ Parallel-safe | LazyWitness uses `pBuildLazyStruct` |
| Witness → Witness | ❌ Creates dependency | LazyWitness calls ControlFlowWitness |
| Element → Element | ✅ Composition layer | `pCondBranch` uses MLIR types |
| Pattern → Element | ✅ Composition layer | `pBuildLazyStruct` uses `pInsertValue` |

**The standing art composes up. Shared vocabulary enables parallel execution.**

## External Tool Integration

Alex delegates to battle-tested infrastructure:
- `mlir-opt` for dialect conversion
- `mlir-translate` for LLVM IR generation
- `llc` for machine code generation
- System linker for final executable

## OutputKind

```fsharp
type OutputKind =
    | Console       // Uses libc, main entry point
    | Freestanding  // No libc, _start wrapper, direct syscalls
    | Embedded      // No OS, custom startup
    | Library       // No entry point, exported symbols
```

---

*For detailed architecture decisions, see `Architecture_Canonical.md`.*
