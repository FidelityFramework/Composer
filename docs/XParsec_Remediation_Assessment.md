# Alex XParsec Remediation Assessment

**Date:** February 2026
**Status:** February 2026 assessment; measurements below are historical evidence.

---

## Executive Summary

The February assessment reported the XParsec remediation of Alex as substantially done. Witness code has been reduced from **5,773 lines to ~2,225 lines** (61% reduction). All witnesses now follow the Element/Pattern/Witness architecture and use XParsec combinators for PSG traversal.

**Remaining work is vestige cleanup, not a full refactoring.** Two vestige patterns persist in some witnesses:
1. **Hidden helpers** — private functions that should live in Patterns
2. **TMemRef push-passing** — eager loads and downstream passing were recorded as a vestige. The source-owned occurrence contract now governs load versus address use; Patterns consume that settlement.

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

## Passive Pattern Composition

The February review identified hidden witness helpers and eager load-passing as
vestiges. Moving either behavior into a Pattern is insufficient if it still
infers source semantics.

CCS/Baker settles the load or address use for each occurrence, including native
type, demand, actual storage identity, mutability and lifetime. It constructs
required operations in the PSG before publication. A physical `TMemRef` shape,
SSA count or source spelling is not authority to insert a load.

Patterns compose physical Elements from those immutable settled facts.
Witnesses observe through the Huet zipper. Neither layer reconstructs source
demand, chooses storage or repairs missing premises.

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

## Source-Owned DMM Settlement

CCS/Baker owns escape analysis, capture relationships, allocation residence and
complete lifetime proof. Owning recipes construct any required promotion,
allocation and release operations in the PSG before publication.

Alex's Patterns compose the physical operations already admitted by that
settlement. They do not detect captures, choose stack versus arena residence or
promote escaping allocations. The publication retains actual storage identity,
extent, capacity, alignment, lifetime and the complete joint premises.

The editor consumes the same source-owned conclusions and located failures.
Missing facts fail at their owner; emitter fallbacks cannot substitute for proof.

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
- [ ] Load/address operations follow source-settled per-occurrence facts; no physical-type-driven inference in Patterns
- [ ] All witnesses under ~100 lines (simple: 20-40, complex: 50-100)

### DMM Integration (Next Milestone)

- [ ] CCS/Baker settles complete capture, escape, residence and lifetime premises
- [ ] Baker recipes construct admitted allocation and promotion operations
- [ ] Alex passively witnesses immutable source-settled storage operations
- [ ] `arena { ... }` computation expression compiles correctly (Bounded model)

---

## Related Documentation

- `Architecture_Canonical.md` - Overall Composer pipeline architecture
- `CCS_Architecture.md` - DTS/DMM coeffect model
- `Alex_Architecture_Overview.md` - Alex component overview
- `XParsec_PSG_Architecture.md` - XParsec integration details
- SpeakEZ blog: "Managed Mutability" — PULL model, TMemRef, escape analysis roadmap
- SpeakEZ blog: "Inferring Memory Lifetimes" — L1/L2/L3 lifetime model
