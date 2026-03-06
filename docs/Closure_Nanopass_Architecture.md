# Closure Architecture

> **MLKit-style flat closures with CCS-computed captures.**
> See [C-01 PRD](PRDs/C-01-Closures.md) for the full specification.

## 1. Executive Summary

Clef Native uses **MLKit-style flat closures** where all captured variables are stored inline in the closure struct, not via pointer chains to enclosing environments.

**Key Architectural Decision**: Capture analysis is **scope analysis**, and scope is resolved during type checking. Therefore, **capture analysis belongs in CCS**, not Composer. CCS computes captures during PSG construction and includes them directly in `SemanticKind.Lambda`. Alex only handles SSA assignment and struct layout for code generation.

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
Closure Structure (byte-level memref)
┌─────────────────────────────────────────────────────────┐
│ code_ptr (TIndex): pointer to lambda implementation     │
├─────────────────────────────────────────────────────────┤
│ capture_0: T₀  (value for ByValue, pointer for ByRef)   │
│ capture_1: T₁                                           │
│ ...                                                     │
└─────────────────────────────────────────────────────────┘

MLIR type: memref<N x i8> where N = sum of field byte sizes
```

### 3.2 Capture Modes

| Variable Kind | Capture Mode | In Struct | Semantics |
|---|---|---|---|
| Immutable `let x = ...` | ByValue | `T` | Copy value |
| Mutable `let mutable x = ...` | ByRef | `ptr<T>` | Pointer to alloca |

### 3.3 Extended Struct Layouts

The same byte-level struct pattern supports three contexts:

| Context | Layout | Extraction Base Index |
|---|---|---|
| RegularClosure | `{code_ptr, cap₀, cap₁, ...}` | 1 |
| LazyThunk | `{computed, value, code_ptr, cap₀, ...}` | 3 |
| SeqGenerator | `{state, current, code_ptr, cap₀, ...}` | 3 |

## 4. Coeffect — ClosureLayout

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

## 5. Pipeline Flow

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
    ├─ pFlatClosure: struct construction
    ├─ pClosureCall: extract + indirect call
    ├─ pExtractCaptures: env extraction at function entry
    ├─ pAllocateInArena: arena bump allocation
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

## 6. FFI Boundary

Closures are Clef-internal. At the C boundary:
- **memref → raw pointer**: `pExtractBasePtr` (memref.extract_aligned_pointer_as_index)
- **ExternCall marshaling**: Automatic in `pExternCallResolved` for memref-typed args
- **`nativeint &struct`**: AddressOf → pBuildAddressOf → pExtractBasePtr at type conversion

See C-01 PRD Section 6 for full boundary marshaling specification.

## 7. References

- Shao & Appel (1994), "Space-Efficient Closure Representations"
- MLKit Programming with Regions (Tofte, Elsman)
- C-01 PRD: `docs/PRDs/C-01-Closures.md`
