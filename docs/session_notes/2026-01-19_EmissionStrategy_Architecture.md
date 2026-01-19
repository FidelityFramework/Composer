# Session Notes: EmissionStrategy Architecture Fix (January 19, 2026)

## Summary

This session addressed a fundamental architectural issue: **traversal/emission decisions were being made in Alex (Firefly) that should be in FNCS (PSG construction)**. The fix involved adding `EmissionStrategy` metadata to FNCS and cleaning up Alex to consume it uniformly.

## The Architectural Principle

**FNCS/PSG**: Pure semantic graph. Knows NOTHING about MLIR or SSA. Provides:
- Node structure (Lambda has body, captures list, etc.)
- Enriched metadata like `EmissionStrategy.SeparateFunction(captureCount)`
- Type information, memory scaffolding markers

**Alex**: Receives PSG as input. From there:
- SSA Assignment reads PSG structure and assigns SSAs to nodes
- Emission (zipper + witnesses) traverses PSG, generates MLIR
- Witnesses should derive SSAs from PSG structure, not pre-stored values

## Changes Made

### 1. FNCS (fsnative) - EmissionStrategy Type

**File**: `/home/hhh/repos/fsnative/src/Compiler/Checking.Native/SemanticGraph.fs`

Added:
```fsharp
type EmissionStrategy =
    | Inline
    | SeparateFunction of captureCount: int
```

- Added `EmissionStrategy` field to `SemanticNode`
- Added `SetEmissionStrategy` method to `NodeBuilder`

### 2. FNCS - Lambda/SeqExpr Body Marking

Updated all Lambda creation sites to mark bodies with capture count:
- `Collections.fs:327` - Function keyword (0 captures)
- `Applications.fs:630` - Anonymous Lambda (List.length captures)
- `Bindings.fs:247,364` - Named functions and eta-expansion
- `Coordinator.fs:462` - DotLambda (0 captures)
- `Coordinator.fs:995` - checkLazy thunk (List.length captures)
- `Coordinator.fs:1051` - checkSeq MoveNext (List.length captures)

### 3. Alex - Removed Architectural Pollution (GepSSA)

**Key insight**: `GepSSA` in `CaptureSlot` was SSA Assignment trying to pre-compute emission details. This violated separation of concerns.

**File**: `/home/hhh/repos/Firefly/src/Alex/Preprocessing/SSAAssignment.fs`

- Removed `GepSSA` from `CaptureSlot` type
- Changed `computeLambdaSSACost` from `2*n+3` to `n+3` (no longer pre-allocating extraction SSAs)
- Updated `buildClosureLayout` SSA indices

**File**: `/home/hhh/repos/Firefly/src/Alex/Witnesses/LambdaWitness.fs`

- Changed `buildCaptureExtractionOps` to derive extraction SSAs from `SlotIndex`:
  ```fsharp
  let extractSSA = V slot.SlotIndex  // Derived from PSG structure
  ```
- Updated `preBindParams` similarly

### 4. Alex - Children Filtering by EmissionStrategy

**File**: `/home/hhh/repos/Firefly/src/Alex/Preprocessing/SSAAssignment.fs` (~line 694)
**File**: `/home/hhh/repos/Firefly/src/Alex/Traversal/FNCSTransfer.fs` (~line 411)

Both now filter children by EmissionStrategy:
```fsharp
match child.EmissionStrategy with
| EmissionStrategy.SeparateFunction _ -> false  // Skip, parent handles it
| EmissionStrategy.Inline -> true
```

### 5. Alex - Lambda Body SSA Offset

**File**: `/home/hhh/repos/Firefly/src/Alex/Preprocessing/SSAAssignment.fs` (~line 711)

Lambda body SSA assignment now reads captureCount from EmissionStrategy:
```fsharp
let startCounter =
    match bodyNode.EmissionStrategy with
    | EmissionStrategy.SeparateFunction captureCount -> captureCount
    | _ -> 0
let innerStartScope = { FunctionScope.empty with Counter = startCounter }
```

### 6. Alex - Cleaned Up Witness SSA Costs

**File**: `/home/hhh/repos/Firefly/src/Alex/Preprocessing/SSAAssignment.fs` (~line 570)

Fixed LazyExpr/LazyForce to match witness documentation:
```fsharp
| SemanticKind.LazyExpr (_, captures) -> 5 + List.length captures  // Was hardcoded 12
| SemanticKind.LazyForce _ -> 4  // Was hardcoded 20
```

## Current Issue (Unresolved)

**Sample 14 (Lazy) still fails** with dominance error:
```
%v4 used at line 126 (LazyForce extractvalue)
%v4 defined at line 725 (LazyExpr insertvalue)
```

**Root cause identified**: Module-level lazy values (`expensive`) are processed BEFORE main, but their ops go to `PendingModuleLevelOps` and are injected INTO main. The SSAs are assigned at module level but emitted inside main's function body.

**The architectural question**: Are module-level bindings being SSA-assigned in the wrong scope? The LazyExpr for `expensive` gets SSAs at module level, but those ops are then emitted inside main where LazyForce expects to use them.

**Key files for investigation**:
- `FNCSTransfer.fs:584-594` - MODULE-LEVEL BINDING INJECTION pattern
- `PSGZipper.fs:524,537` - PendingModuleLevelOps collection
- SSAAssignment - how module-level vs function-level scopes are handled

## Sample Status

- Samples 01-03, 05-13: PASS
- Sample 04: FAIL (unrelated)
- Sample 11 (Closures): PASS - basic closure model works correctly
- Sample 14 (Lazy): FAIL - module-level lazy SSA scoping issue
- Sample 15-16: FAIL - likely same issue

## Pending Tasks

1. Fix module-level lazy SSA scoping issue
2. Document EmissionStrategy in fsnative-spec
3. Full witness SSA cost audit (all witnesses should derive from PSG structure)
4. Resume PRD-16 SeqOperations
5. Write SpeakEZ blog "Learning to Walk" - nanopass architecture case study

## Key Architectural Lessons

1. **PSG is markup only** - no MLIR/SSA knowledge
2. **SSA Assignment derives from structure** - no pre-computed emission details
3. **Witnesses derive SSAs from PSG** - `V slot.SlotIndex`, not stored values
4. **Separation is directional** - FNCS -> Alex, never backwards
5. **Hardcoded costs are cruft** - derive from actual structure
