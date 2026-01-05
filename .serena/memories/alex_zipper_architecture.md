# Alex Zipper Architecture (Updated January 2026)

## Current Architecture: Observation-Driven Emission

The Alex emission layer uses an **observation-driven model** based on codata and coeffects:

```
PSG from FNCS (already transformed)
         ↓
Alex Preprocessing (analysis → coeffects)
         ↓
MLIRZipper fold (observation → template selection → MLIR)
         ↓
MLIR Output
```

**Key Principle**: "The photographer doesn't build the scene - they arrive AFTER the scene is complete and witness it."

## Core Components

### 1. MLIRZipper (`Alex/Traversal/MLIRZipper.fs`)

The MLIRZipper provides:
- **Navigation**: Move through PSG structure
- **State threading**: SSA counters, NodeSSA map, string literals
- **Fold operations**: `foldPostOrder` for emission traversal
- **NO transform logic** - pure navigation

```fsharp
type MLIRZipper = {
    Focus: MLIRFocus
    Path: MLIRPath
    CurrentOps: MLIROp list
    State: MLIRState
    Globals: MLIRGlobal list
}
```

### 2. Witnesses (`Alex/Witnesses/`)

Each witness:
1. **OBSERVES** the focused node
2. **OBSERVES** coeffects (pre-computed in Preprocessing)
3. **SELECTS** appropriate template
4. **RETURNS** updated state with emitted MLIR

Witness files:
- `LiteralWitness.fs` - Integer, float, string constants
- `BindingWitness.fs` - Let bindings (mutable/immutable)
- `ApplicationWitness.fs` - Function applications, intrinsics
- `ControlFlowWitness.fs` - If/while/for, SCF operations
- `MemoryWitness.fs` - Load/store, alloca, GEP
- `LambdaWitness.fs` - Lambda expressions

### 3. Templates (`Alex/Templates/`)

Quotation-based MLIR structure definitions:
- `TemplateTypes.fs` - Core template infrastructure
- `ArithTemplates.fs` - Arithmetic operations
- `MemoryTemplates.fs` - Memory operations
- `ControlFlowTemplates.fs` - Control flow (SCF dialect)
- `LLVMTemplates.fs` - LLVM dialect operations

```fsharp
type MLIRTemplate<'Params> = Expr<'Params -> string>

let addI = <@ fun p -> sprintf "%s = arith.addi %s, %s : %s" p.Res p.Lhs p.Rhs p.Ty @>
```

### 4. XParsec Combinators (`Alex/XParsec/`)

**NEW (January 2026)**: Combinator-based pattern recognition and emission.

- `PSGCombinators.fs` - XParsec-style parser combinators for PSG nodes
- `MLIREmitters.fs` - Combinator-based MLIR emission

```fsharp
type PSGParser<'T> = PSGParserState -> PSGMatchResult<'T> * PSGParserState

// Composite patterns
let pBinaryArithApp : PSGParser<IntrinsicInfo * NodeId * NodeId>
let pComparisonApp : PSGParser<IntrinsicInfo * NodeId * NodeId>

// Full emitters using computation expression
let emitBinaryArithOp : EmitterParser
let emitComparisonOp : EmitterParser
```

### 5. Preprocessing (`Alex/Preprocessing/`)

**Analysis only, no transforms**:
- `MutabilityAnalysis.fs` - Which bindings need alloca vs SSA
- `SSAAssignment.fs` - Pre-assign SSA names
- `PatternBindingAnalysis.fs` - Pattern binding extraction

Output: Coeffects (metadata attached to nodes)

### 6. Transfer (`Alex/Traversal/FNCSTransfer.fs`)

The main transfer pipeline that:
- Receives SemanticGraph from FNCS
- Creates MLIRZipper from entry point
- Folds with witness functions
- Produces MLIR output

## The Four Pillars

| Pillar | Role | Location |
|--------|------|----------|
| **Coeffects** | Pre-computed analysis results | `Alex/Preprocessing/` |
| **Templates** | MLIR structure as data | `Alex/Templates/` |
| **Zipper** | Navigation + fold | `Alex/Traversal/MLIRZipper.fs` |
| **XParsec** | Combinator-based recognition | `Alex/XParsec/` |

## What Changed (January 2026)

| Before | After |
|--------|-------|
| Multiple transfer files | Single `FNCSTransfer.fs` |
| Transform logic in witnesses | Observation only |
| sprintf throughout | Template-based emission |
| String matching on names | SemanticKind/IntrinsicInfo dispatch |
| No XParsec infrastructure | Full combinator library |

## Current Metrics

- **sprintf in Witnesses**: 57 (down from 138)
- **sprintf in MLIRZipper**: 47 (down from 53)
- **MLIRZipper lines**: 1493 (SCF complexity)

## Marker Strings (Technical Debt)

Current workarounds for incomplete FNCS transforms:
- `$pipe:` - Pipe operator chains
- `$partial:` - Partial application
- `$platform:` - Platform bindings

These markers are necessary until FNCS provides complete semantic transforms.

## Removed Files

- `PSGEmitter.fs` - Central dispatch (antipattern)
- `PSGScribe.fs` - Recreated same antipattern
- `MLIRTransfer.fs` - Replaced by FNCSTransfer
- `SaturateApplications.fs` - Moved to FNCS

## Validation

Samples 01-04 compile and run correctly with this architecture.
