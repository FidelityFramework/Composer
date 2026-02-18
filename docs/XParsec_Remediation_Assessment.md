# Alex XParsec Remediation Assessment

**Date:** February 2026
**Status:** Largely complete — vestiges remain

---

## Executive Summary

The XParsec remediation of Alex is substantially done. Witness code has been reduced from **5,773 lines to ~2,225 lines** (61% reduction). All witnesses now follow the Element/Pattern/Witness architecture and use XParsec combinators for PSG traversal.

**Remaining work is vestige cleanup, not a full refactoring.** Two vestige patterns persist in some witnesses:
1. **Hidden helpers** — private functions that should live in Patterns
2. **TMemRef push-passing** — witnesses load values and pass them downstream as parameters instead of using the PULL/catamorphism model where Patterns detect and handle `TMemRef` at point of use

The next major engineering milestone is **DMM escape analysis integration**, not further XParsec refactoring.

---

## Current State Metrics

### Total Lines by Layer

| Layer | Current LOC | Notes |
|-------|-------------|-------|
| **Witnesses** | ~2,225 | 21 files — 61% reduction from 5,773 |
| **Elements** | ~882 | 8 files — fully extracted from witnesses |
| **Patterns** | ~3,067 | 9 files — MemoryPatterns 768, ClosurePatterns 509, StringPatterns 477 |
| **XParsec Combinators** | ~849 | PSGCombinators 810 + Extensions 39 |

### Witness File Inventory (Current)

| File | Lines | Status |
|------|-------|--------|
| LambdaWitness.fs | 336 | XParsec — closure complexity |
| ControlFlowWitness.fs | 266 | XParsec |
| ApplicationWitness.fs | 185 | XParsec |
| MapWitness.fs | 140 | XParsec |
| SeqWitness.fs | 127 | XParsec |
| ListWitness.fs | ~110 | XParsec |
| OptionWitness.fs | ~100 | XParsec |
| SetWitness.fs | ~95 | XParsec |
| MemoryWitness.fs | ~90 | XParsec |
| VarRefWitness.fs | ~85 | XParsec — TMemRef vestige possible |
| LazyWitness.fs | 38 | XParsec — canonical pilot ✅ |
| PlatformWitness.fs | 22 | XParsec — intrinsic thin wrapper |
| ArithIntrinsicWitness.fs | 23 | XParsec |
| MemoryIntrinsicWitness.fs | 25 | XParsec |
| StringIntrinsicWitness.fs | 25 | XParsec |
| *(additional witnesses)* | ~754 | XParsec |

---

## Vestige Patterns

Two anti-patterns from the pre-remediation era occasionally surface during new witness development. They are not systemic failures — they are localized and fixable.

### Vestige 1: Hidden Helper Functions

**What it is:** A private function defined inside a witness module that computes something which properly belongs in a Pattern.

**Why it's wrong:** Witnesses are observers (the codata/photographer principle). Computation and composition belong in Patterns. Hidden helpers accumulate witness-specific logic that can't be reused.

**Before (WRONG):**
```fsharp
// Inside VarRefWitness.fs — private helper doing Pattern work
let private buildLoadForMutableRef (node: SemanticNode) (state: PSGState) : MLIROp list =
    let memrefType = ...
    let ptrSSA = ...
    // builds GEP + Load ops manually
    [gepOp; loadOp]

let witnessVarRef (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match node.Type with
    | TMemRef _ -> { InlineOps = buildLoadForMutableRef node ctx.State; ... }
    | _ -> ...
```

**After (RIGHT):**
```fsharp
// The load logic lives in MemoryPatterns.fs as a composable Pattern
// Witness delegates entirely:
let witnessVarRef (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch (pVarRef >>= MemoryPatterns.pLoadIfMemRef) ... with
    | Some result -> result
    | None -> WitnessOutput.error "VarRef pattern failed"
```

### Vestige 2: TMemRef Push-Passing

**What it is:** A witness detects that a value has type `TMemRef`, loads it eagerly, and passes the loaded value as a parameter to downstream Patterns.

**Why it's wrong:** This is the PUSH model — it makes the witness stateful and breaks monadic composition. The correct model is PULL: Patterns detect `TMemRef` at the point of use and compose load operations there. This is the catamorphism model described in the Managed Mutability blog.

**Before (WRONG — push model):**
```fsharp
// Witness eagerly loads, pushes value downstream
let witnessApplication (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    let argVal = resolveArg ctx node.Arg
    let loadedVal =
        match argVal.Type with
        | TMemRef innerTy ->
            // Load eagerly before calling pattern
            let loadSSA = freshSSA()
            let loadOp = MLIROp.LLVMOp (LLVMOp.Load (loadSSA, argVal.SSA, innerTy, ...))
            { SSA = loadSSA; Type = innerTy; ExtraOps = [loadOp] }
        | _ -> argVal
    ApplicationPatterns.pFuncCall ctx loadedVal  // receives pre-loaded value
```

**After (RIGHT — pull model):**
```fsharp
// Pattern detects TMemRef at point of use and composes load
let witnessApplication (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch (pApplication >>= ApplicationPatterns.pEmitCall) ctx.Graph node ... with
    | Some result -> result
    | None -> WitnessOutput.error "Application pattern failed"

// In ApplicationPatterns.fs — PULL: detect and load at point of use
let pEmitCall : PSGParser<MLIROp list> = parser {
    let! (func, args) = pApplication
    let! state = getUserState
    // Pattern internally handles TMemRef args by composing load ops
    let! resolvedArgs = args |> List.map pResolveValue |> sequence
    ...
}
```

The PULL model ensures witnesses remain stateless observers; all composition lives in Patterns where it can be reused and tested independently.

---

## Architecture Invariants (Enforced by Compiler)

The three-layer invariant is structurally enforced:

| Layer | Visibility | Can Import |
|-------|-----------|------------|
| **Elements** | `module internal` | XParsec, MLIR types |
| **Patterns** | public | Elements, XParsec, PSG |
| **Witnesses** | public | Patterns, XParsec, PSG |

Witnesses **cannot** import Elements directly — the F# `module internal` declaration makes this a compile error. This firewall is maintained.

The **parallelism invariant** is architectural: witnesses are pure functions of (PSG node, state) → WitnessOutput. No witness calls another witness. This enables parallel witness evaluation and makes the system amenable to the concurrent zipper traversal described in `Parallel_Zipper_Architecture.md`.

---

## Next: DMM Escape Analysis

The remediation provides the right foundation for DMM integration. The PULL model is essential: once escape classifications are attached to PSG nodes as coeffects, Patterns can detect them at point of use and select the appropriate allocation strategy — exactly as they detect `TMemRef` and compose load operations.

The escape analysis roadmap (from the Managed Mutability blog):

**Phase 1 — Closure capture integration:**
- Detect when a mutable allocation escapes via closure capture
- Select arena allocation strategy in `ClosurePatterns.pBuildClosure`
- `EscapeKind.EscapesViaClosure` already defined in CCS

**Phase 2 — Arena hoisting:**
- Allocations that escape their lexical scope promoted to arena
- `arena { ... }` computation expression scope (Bounded model)
- Alex `MemoryPatterns` generates `memref.alloc` vs `memref.alloca` based on escape kind

**Phase 3 — Full lifetime inference:**
- L1 model: compiler infers all escape classifications
- No annotations needed for the common case
- Language server (Lattice) surfaces escape path and promotion decisions

---

## Success Criteria (Updated)

### Achieved ✅

- [x] XParsec-based architecture throughout all 21 witnesses
- [x] Elements layer extracted and `module internal`
- [x] Zero direct MLIR op constructions in witnesses (no ad-hoc LLVMOp construction)
- [x] Patterns layer composable (~3,067 lines)
- [x] Total witness LOC: ~2,225 (vs 5,773 at start — 61% reduction)

### Vestige Cleanup

- [ ] No hidden helper functions in any witness
- [ ] No TMemRef push-passing — all TMemRef handling via PULL model in Patterns
- [ ] All witnesses under ~100 lines (simple: 20-40, complex: 50-100)

### DMM Integration (Next Milestone)

- [ ] `EscapeKind` coeffect attached to PSG allocation nodes
- [ ] `MemoryPatterns.pAllocate` selects `alloca` vs `alloc` based on escape kind
- [ ] `ClosurePatterns.pBuildClosure` detects captured mutable refs and upcasts to arena
- [ ] `arena { ... }` computation expression compiles correctly (Bounded model)

---

## Related Documentation

- `Architecture_Canonical.md` - Overall Composer pipeline architecture
- `CCS_Architecture.md` - DTS/DMM coeffect model
- `Alex_Architecture_Overview.md` - Alex component overview
- `XParsec_PSG_Architecture.md` - XParsec integration details
- SpeakEZ blog: "Managed Mutability" — PULL model, TMemRef, escape analysis roadmap
- SpeakEZ blog: "Inferring Memory Lifetimes" — L1/L2/L3 lifetime model
