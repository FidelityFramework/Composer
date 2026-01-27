# Alex XParsec Remediation Assessment

**Date:** January 2026
**Status:** Architecture established, systematic refactoring required

---

## Executive Summary

Alex currently contains **5,773 lines of witness code** that manually constructs MLIR operations instead of using the XParsec-based Element/Pattern/Witness architecture. The target is **~600 lines** of thin witnesses using XParsec combinators, representing a **90% reduction** in witness code.

**Key Finding:** Only 1 out of 14 witnesses (LazyWitness.fs) correctly uses the XParsec architecture. The other 13 witnesses contain **564 direct MLIR op constructions** and **49 private helper functions** that should be factored into Elements and Patterns layers.

---

## Current State Metrics

### Total Lines by Layer

| Layer | Current LOC | Target LOC | Gap |
|-------|-------------|------------|-----|
| **Witnesses** | 5,773 | ~600 | **90% reduction needed** |
| **Elements** | 47 | ~800 | Need extraction from witnesses |
| **Patterns** | 466 + 1,273 | ~2,000 | Good foundation, needs XParsec |
| **XParsec Combinators** | 268 | ~300-400 | Good foundation |
| **Total Alex** | ~8,000 | ~3,500 | **56% reduction** |

### Code Quality Metrics

| Metric | Current | Target | Reduction |
|--------|---------|--------|-----------|
| **Direct MLIR Ops in Witnesses** | 564 | 0 | **100%** |
| **Private Helper Functions** | 49 | 0 | **100%** |
| **XParsec-Based Witnesses** | 1 (LazyWitness) | 14 | 1300% increase |

---

## Component-by-Component Analysis

### 1. Witnesses/ - CRITICAL GAP

| File | Lines | MLIR Ops | Helpers | Assessment |
|------|-------|----------|---------|------------|
| SeqWitness.fs | 1,021 | 69 | 1 | **SEVERE** - Manual struct layout, state machine |
| MemoryWitness.fs | 878 | 72 | 4 | **SEVERE** - Direct GEP/Load/Store emission |
| FormatOps.fs | 862 | 181 | 1 | **CRITICAL** - 181 MLIR ops! Transform logic? |
| LambdaWitness.fs | 556 | 27 | 9 | **SEVERE** - Manual closure construction |
| SeqOpWitness.fs | 549 | 106 | 8 | **SEVERE** - Manual moveNext emission |
| ControlFlowWitness.fs | 487 | 46 | 6 | **HIGH** - Manual block/branch emission |
| ArithOps.fs | 286 | 13 | 5 | **MEDIUM** - String matching on operator names |
| OptionWitness.fs | 257 | 19 | 2 | **MEDIUM** - DU construction |
| SetWitness.fs | 191 | 3 | 2 | **LOW** - Mostly delegates |
| ListWitness.fs | 190 | 13 | 2 | **LOW** - DU construction |
| MapWitness.fs | 183 | 3 | 2 | **LOW** - Mostly delegates |
| SyscallOps.fs | 152 | 0 | 2 | **GOOD** - Thin, uses Bindings (now deleted) |
| LiteralWitness.fs | 123 | 12 | 5 | **LOW** - Constant emission |
| LazyWitness.fs | 38 | 0 | 0 | **EXCELLENT** - XParsec pilot ✅ |

**Totals:**
- **5,773 lines** (Target: ~600 lines)
- **564 direct MLIR op constructions** (Target: 0)
- **49 private helper functions** (Target: 0)
- **1 XParsec-based witness** (Target: 14)

---

### 2. Elements/ - NEEDS EXTRACTION

**Current:** 47 lines (4 XParsec parser functions)

**Gap:** ~90% of MLIR construction code sitting in witnesses needs extraction.

#### What Needs Extraction

| Source Witness | Lines | MLIR Ops | Target Elements File | Est. Lines |
|----------------|-------|----------|---------------------|------------|
| MemoryWitness.fs | 878 | 72 | MemoryElements.fs | ~150 |
| SeqWitness.fs | 1,021 | 69 | SeqElements.fs | ~100 |
| SeqOpWitness.fs | 549 | 106 | SeqOpElements.fs | ~80 |
| LambdaWitness.fs | 556 | 27 | LambdaElements.fs | ~80 |
| ControlFlowWitness.fs | 487 | 46 | ControlFlowElements.fs | ~70 |
| ArithOps.fs | 286 | 13 | ArithElements.fs | ~50 |
| OptionWitness.fs | 257 | 19 | UnionElements.fs | ~40 |
| Others | ~1,700 | ~212 | Various | ~230 |

**Expected Elements Layer Files:**
- `ArithElements.fs` - AddI, SubI, MulI, DivI, CmpI, etc.
- `MemoryElements.fs` - Load, Store, GEP, StructGEP, Alloca
- `ControlFlowElements.fs` - Branch, CondBranch, Block, Loop
- `FunctionElements.fs` - FuncOp, Call, IndirectCall, Return
- `StructElements.fs` - ExtractValue, InsertValue, Undef (✅ already done)
- `SeqElements.fs` - Seq-specific struct operations
- `UnionElements.fs` - DU tag operations, case construction

---

### 3. Patterns/ - NEEDS XPARSEC INTEGRATION

**Current:**
- SemanticPatterns.fs: 438 lines, 43 active patterns ✅
- ElisionPatterns.fs: 28 lines (TODOs) ❌
- Dialects/*/Templates.fs: 1,273 lines ✅

**Status:**
- ✅ SemanticPatterns.fs - Good foundation (43 active patterns)
- ❌ ElisionPatterns.fs - Needs implementation (currently stubs)
- ✅ Dialects/*/Templates.fs - Good composable patterns (1,273 lines)

#### Gap Analysis

ElisionPatterns.fs needs to be populated with composable elision functions that:
1. Use XParsec for pattern matching
2. Call Elements for atomic ops
3. Return composed MLIR structures

**Current Code (WRONG):**
```fsharp
/// Pattern: Match and elide lazy struct construction
let pBuildLazyStruct : PSGParser<unit> =
    parser {
        let! (bodyId, captures) = pLazyExpr
        // TODO: Emit MLIR via Elements
        return ()
    }
```

**Target Code (RIGHT):**
```fsharp
/// Build flat closure struct via InsertValue chain
let pBuildFlatClosure (codePtr: SSA) (captures: Val list) (ssas: SSA list) : PSGParser<MLIROp list> =
    parser {
        let! state = getUserState
        let structTy = buildClosureType state.Current.Type captures
        let! undefOp = pEmitUndef ssas.[0]
        let! insertOps = captures
            |> List.mapi (fun i cap -> pEmitInsertValue ssas.[i+1] (if i = 0 then ssas.[0] else ssas.[i]) cap.SSA [i])
            |> sequence
        return undefOp :: insertOps
    }
```

**What's Missing:**
- `pBuildFlatClosure` - Compose InsertValue chain for closures
- `pBuildSeqStruct` - Compose seq state machine struct
- `pElideMoveNext` - Compose moveNext function with blocks
- `pBuildDUCase` - Compose discriminated union case
- `pBuildRecordStruct` - Compose record struct layout
- And ~20 more composable patterns

---

### 4. XParsec/PSGCombinators.fs - GOOD FOUNDATION

**Current:** 268 lines

**What Exists:** ✅
- Basic node patterns: `pLiteral`, `pVarRef`, `pApplication`
- Navigation: `onChild`, `onChildren`, `focusChild`
- Lazy patterns: `pLazyExpr`, `pLazyForce`
- Intrinsic patterns: `pIntrinsic`, `pIntrinsicModule`, `pIntrinsicNamed`
- Control flow: `pIfThenElse`, `pWhileLoop`, `pForLoop`
- DU patterns: `pDUGetTag`, `pDUEliminate`, `pDUConstruct`

**What's Missing (Low Priority):**
- Multi-arg application patterns (e.g., `pBinaryApp`, `pTernaryApp`)
- Record/tuple projection patterns
- More semantic sugar for common idioms

**Assessment:** Good foundation. Incrementally add patterns as witnesses are refactored.

---

## The Reimplement-vs-Use-XParsec Problem

### Anti-Pattern: Manual String Matching

**Current Code (WRONG):**
```fsharp
// ArithOps.fs - Manual string matching and MLIR construction
let witnessBinaryArith (opName: string) (lhs: Val) (rhs: Val) =
    match opName with
    | "op_Addition" -> Some (ArithOp.AddI (resultSSA, lhs.SSA, rhs.SSA, ty))
    | "op_Subtraction" -> Some (ArithOp.SubI (resultSSA, lhs.SSA, rhs.SSA, ty))
    | "op_Multiply" -> Some (ArithOp.MulI (resultSSA, lhs.SSA, rhs.SSA, ty))
    // ... 10 more cases
```

**Correct Pattern (RIGHT):**
```fsharp
// Should use XParsec
let pBinaryArith : PSGParser<MLIROp list> = parser {
    let! (opName, lhs, rhs) = pBinaryApp
    let! state = getUserState
    let resultSSA = requireSSA node.Id state.Coeffects.SSA
    match opName with
    | "op_Addition" -> return! pEmitAddI resultSSA lhs.SSA rhs.SSA
    | "op_Subtraction" -> return! pEmitSubI resultSSA lhs.SSA rhs.SSA
    | "op_Multiply" -> return! pEmitMulI resultSSA lhs.SSA rhs.SSA
}
```

### Anti-Pattern: Direct MLIR Construction in Witnesses

**Current Code (WRONG):**
```fsharp
// MemoryWitness.fs - 878 lines, 72 direct MLIR ops
let witnessFieldGet structVal fieldIndex fieldType =
    let resultSSA = requireSSA nodeId ssa
    let ptrSSA = V (freshSSA())
    let gepOp = MLIROp.LLVMOp (LLVMOp.GEP (ptrSSA, structVal.SSA, [...], ptrTy, None))
    let loadOp = MLIROp.LLVMOp (LLVMOp.Load (resultSSA, ptrSSA, fieldType, AtomicOrdering.NotAtomic))
    [gepOp; loadOp], TRValue { SSA = resultSSA; Type = fieldType }
```

**Correct Pattern (RIGHT):**
```fsharp
// Witness delegates to Patterns
let witnessFieldGet (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch (pFieldGet >>= ElisionPatterns.pBuildFieldAccess)
                   ctx.Graph node ... with
    | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.error "FieldGet pattern match failed"
```

---

## Systematic Refactoring Strategy

### Phase 1: Extract Elements (HIGH PRIORITY)

For each bloated witness:
1. Identify atomic MLIR ops being constructed
2. Extract to Elements layer with XParsec state threading
3. Make `module internal`

**Example: MemoryWitness.fs (878 lines) → MemoryElements.fs (~150 lines)**

```fsharp
// Elements/MemoryElements.fs
module internal Alex.Elements.MemoryElements =
    let pEmitLoad (ssa: SSA) (ptr: SSA) : PSGParser<MLIROp> = parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Load (ssa, ptr, ty, AtomicOrdering.NotAtomic))
    }

    let pEmitStore (value: SSA) (ptr: SSA) : PSGParser<MLIROp> = parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Store (value, ptr, ty, AtomicOrdering.NotAtomic))
    }

    let pEmitGEP (ssa: SSA) (ptr: SSA) (indices: (SSA * MLIRType) list) : PSGParser<MLIROp> = parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.GEP (ssa, ptr, indices, ty, None))
    }
```

### Phase 2: Populate Patterns (MEDIUM PRIORITY)

Compose Elements into semantic patterns:

```fsharp
// Patterns/ElisionPatterns.fs
module Alex.Patterns.ElisionPatterns

open Alex.Elements.MLIRElements
open Alex.Elements.MemoryElements

/// Build field access: GEP + Load
let pBuildFieldAccess (structPtr: SSA) (fieldIndex: int) (ssas: SSA list) : PSGParser<MLIROp list> = parser {
    let gepSSA = ssas.[0]
    let loadSSA = ssas.[1]
    let! gepOp = pEmitStructGEP gepSSA structPtr fieldIndex
    let! loadOp = pEmitLoad loadSSA gepSSA
    return [gepOp; loadOp]
}

/// Build flat closure struct
let pBuildFlatClosure (codePtr: SSA) (captures: Val list) (ssas: SSA list) : PSGParser<MLIROp list> = parser {
    let! state = getUserState
    let structTy = buildClosureType state.Current.Type captures
    let! undefOp = pEmitUndef ssas.[0]
    let! insertOps = captures
        |> List.mapi (fun i cap ->
            let targetSSA = ssas.[i+1]
            let sourceSSA = if i = 0 then ssas.[0] else ssas.[i]
            pEmitInsertValue targetSSA sourceSSA cap.SSA [i])
        |> sequence
    return undefOp :: insertOps
}
```

### Phase 3: Refactor Witnesses (SYSTEMATIC)

Rewrite each witness to use XParsec + Patterns:

```fsharp
// Witnesses/MemoryWitness.fs (878 → ~50 lines)
module Alex.Witnesses.MemoryWitness

open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns
open Alex.Traversal.TransferTypes

/// Witness field get operation
let witnessFieldGet (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch (pFieldGet >>= fun (structVal, fieldIdx) ->
                    ElisionPatterns.pBuildFieldAccess structVal fieldIdx)
                   ctx.Graph node ... with
    | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.error "FieldGet pattern match failed"

/// Witness record construction
let witnessRecordExpr (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch (pRecordExpr >>= ElisionPatterns.pBuildRecordStruct)
                   ctx.Graph node ... with
    | Some (ops, result) -> { InlineOps = ops; TopLevelOps = []; Result = result }
    | None -> WitnessOutput.error "RecordExpr pattern match failed"
```

---

## Prioritized Work Queue

### CRITICAL (Immediate)

1. ✅ **LazyWitness** - Already done (pilot)
2. **FormatOps** (862 lines, 181 MLIR ops) - Check for transform logic, may need FNCS fix
3. **SeqWitness** (1,021 lines → ~60 lines) - Extract state machine patterns
4. **MemoryWitness** (878 lines → ~50 lines) - Extract GEP/Load/Store patterns

### HIGH Priority

5. **LambdaWitness** (556 lines → ~40 lines) - Extract closure construction
6. **SeqOpWitness** (549 lines → ~50 lines) - Extract map/filter patterns
7. **ControlFlowWitness** (487 lines → ~60 lines) - Extract block/branch patterns

### MEDIUM Priority

8. **ArithOps** (286 → ~40) - Replace string matching with XParsec
9. **OptionWitness** (257 → ~30) - Extract DU patterns
10. **ListWitness** (190 → ~20) - Extract cons/nil patterns
11. **MapWitness** (183 → ~20) - Mostly delegates, minimal work
12. **SetWitness** (191 → ~20) - Mostly delegates, minimal work

### LOW Priority (Already Small)

13. **SyscallOps** (152 → ~30) - Needs Bindings replacement (deleted)
14. **LiteralWitness** (123 → ~20) - Extract constant emission

---

## Estimated Effort

**Per-Witness Effort:**
- Extract Elements: 2-3 hours
- Populate Patterns: 1-2 hours
- Refactor Witness: 1 hour
- **Total per witness: 4-6 hours**

**Total Effort:**
- 13 witnesses to refactor (LazyWitness already done)
- At 4-6 hours per witness: **52-78 hours**
- At 8 hours/day: **7-10 days**
- With testing/validation: **1.5-2 weeks**

**Parallelization Opportunity:**
Witnesses can be refactored independently once Elements/Patterns foundation is established.

---

## Success Criteria

### Quantitative

- [ ] All witnesses under 50 lines (target ~20-30)
- [ ] Zero direct MLIR op construction in witnesses
- [ ] Zero private helper functions in witnesses
- [ ] All 14 witnesses use XParsec architecture
- [ ] Elements layer ~800 lines
- [ ] Patterns layer ~2,000 lines (including Templates)
- [ ] Total Alex LOC reduced from ~8,000 to ~3,500 (**56% reduction**)

### Qualitative

- [ ] Consistent XParsec-based architecture throughout
- [ ] Readable: witnesses ~20 lines of XParsec patterns
- [ ] Maintainable: atomic ops in Elements, composition in Patterns
- [ ] Type-safe: compiler enforces firewall (witnesses can't import Elements)

### Architectural

- [ ] Witnesses use XParsec for pattern matching
- [ ] Patterns compose Elements for elision
- [ ] Elements are `module internal`
- [ ] No transform logic in witnesses (return `TRError` for gaps)
- [ ] CI enforces witness line limits
- [ ] All samples continue to pass

---

## Benefits of Remediation

### Code Size

**56% reduction in total Alex code:**
- From ~8,000 lines to ~3,500 lines
- 90% reduction in witness code specifically
- Eliminates 564 direct MLIR constructions
- Eliminates 49 helper functions

### Consistency

**Uniform architecture:**
- All witnesses follow same XParsec pattern
- No ad-hoc string matching
- No manual MLIR construction
- Predictable structure makes onboarding easier

### Readability

**Witnesses become declarative:**
```fsharp
// Before: 878 lines of imperative MLIR construction
// After: ~50 lines of XParsec patterns
let witnessFieldGet (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch (pFieldGet >>= ElisionPatterns.pBuildFieldAccess) ... with
    | Some (ops, result) -> { InlineOps = ops; ... }
    | None -> WitnessOutput.error "..."
```

### Maintainability

**Atomic operations centralized:**
- Need to change Load emission? Edit MemoryElements.fs
- Need to change closure layout? Edit ElisionPatterns.fs
- Witnesses don't need to change

**Type-safe firewall:**
- Compiler prevents witnesses from importing Elements
- Architecture violations caught at compile time

---

## Related Documentation

- `Architecture_Canonical.md` - Overall Firefly pipeline architecture
- `Alex_Architecture_Overview.md` - Alex component overview
- `XParsec_PSG_Architecture.md` - XParsec integration details

## Related Memories

- `alex_element_pattern_witness_architecture` - The three-layer model
- `mlir_transfer_canonical_architecture` - MLIRTransfer.fs role and limits
- `alex_xparsec_throughout_architecture` - XParsec usage pattern
- `four_pillars_of_transfer` - XParsec, Patterns, Zipper, Templates
- `codata_photographer_principle` - Witnesses observe and return
