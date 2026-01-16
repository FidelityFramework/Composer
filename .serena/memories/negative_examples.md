# NEGATIVE EXAMPLES: Condensed Anti-Patterns

> Reference these patterns when making architectural decisions. Detailed code examples exist in the codebase itself.

## Layer Violations

| # | Anti-Pattern | Why Wrong | Fix |
|---|--------------|-----------|-----|
| 1 | Library-specific logic in MLIR generation | Couples codegen to namespace structure | Alex recognizes FNCS intrinsic markers, not symbol names |
| 2 | Stub implementations in libraries | PSG shows empty body, no structure to witness | Real implementations decompose to primitives |
| 3 | Nanopass logic in code generation | Nanopasses enrich PSG BEFORE emission | Consume enriched PSG, don't compute during gen |
| 4 | Mutable state tracking in codegen | Transformation logic belongs in nanopasses | Follow edges to find values |
| 5 | Central dispatch/emitter/scribe | Removed TWICE - attracts special cases | Zipper folds, XParsec matches locally, no router |
| 6 | String/name-based pattern matching | Couples to surface syntax | Match on PSG node structure |
| 7 | Premature centralization | Pooling decisions too early | Centralize at OUTPUT (MLIR), not DISPATCH |

## Error Handling

| # | Anti-Pattern | Why Wrong | Fix |
|---|--------------|-----------|-----|
| 8 | Silent failures (printfn + Void) | Hides bugs behind bugs, causes downstream symptoms | Return EmitError, halt compilation |

## Type System Violations

| # | Anti-Pattern | Why Wrong | Fix |
|---|--------------|-----------|-----|
| 9 | Hardcoding types instead of using FNCS flow | Type info exists upstream, being discarded | Use node.Type + mapType + Serialize.mlirType |
| 10 | Imperative "push" with "wasn't traversed yet" fallbacks | Zipper provides attention to any node | Pull/codata model - graph is complete |
| 11 | BCL stubs for platform operations (`Unchecked.defaultof`) | Violates BCL-free principle | Use FNCS Sys intrinsics |
| 12 | TVar → Pointer default for polymorphic ops | Erases type instantiation from SRTP | Typed tree overlay should capture concrete types |

## Architectural Pollution

| # | Anti-Pattern | Why Wrong | Fix |
|---|--------------|-----------|-----|
| 13 | Marker strings ($pipe:, $partial:, $platform:) | Indicates incomplete upstream transforms | Fix FNCS transforms so markers never needed |
| 14 | Config defaults as bug fixes | Masks architectural deficiency | Fix the architecture (e.g., reachability following type refs) |
| 15 | Fresh type variables for pattern bindings | Ignores known constructor types | Look up constructor, extract payload types |
| 16 | TRecord/TApp dual representation | Causes unification failures | Single TApp representation + SemanticGraph lookup |
| 17 | Incomplete conversion witness coverage | Valid code fails at emission | Add witnesses for ALL operations in intrinsic modules |

## FOUNDATIONAL VIOLATIONS

| # | Anti-Pattern | Why Wrong | Fix |
|---|--------------|-----------|-----|
| 18 | sprintf-based MLIR generation | No type safety, no composition, stringly typed | Structured types + XParsec composition + boundary-only serialization |
| 19 | Quotations in Alex templates | Quotations define source semantics, not target generation | Templates return structured MLIROp values |
| 20 | Imperative MLIR construction for intrinsics | FNCS punts implementation, Alex builds structure | FNCS intrinsics must decompose to functional constructs |

## Implementation Bugs (Reference)

| # | Issue | Root Cause |
|---|-------|------------|
| 21 | AddressOf returns loaded value, not alloca | Need alloca pointer from SSA coeffect |
| 22 | Nested lambda scope corruption | Missing parameter stack save/restore |
| 23 | LLVM store operand order | `store value, ptr` - value first |

## The Acid Test

Before committing any change, ask:

> "If someone deleted all the comments and looked only at what this code DOES, would they see library-specific logic in MLIR generation?"

If yes, you have violated the layer separation principle. Revert and fix upstream.

> "Am I creating a central dispatch mechanism?"

If yes, STOP. This is the antipattern that was removed twice.
