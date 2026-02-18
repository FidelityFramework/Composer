# NTU (Native Type Universe) Architecture

## Executive Summary

The NTU (Native Type Universe) architecture provides platform-generic types for Fidelity that resolve via quotation-based platform bindings. Type WIDTH is an **erased assumption**, not part of type identity — CCS enforces type identity; Alex resolves widths from platform quotations.

**Core Principle**: Platform awareness flows FROM THE TOP via quotation-based binding libraries, not from CCS type inference.

## Nomenclature

NTU = **N**ative **T**ype **U**niverse

The "NTU" prefix marks platform-generic types internally in CCS. Types come in three families: platform-dependent (resolved via quotations), fixed-width (platform-independent), and numeric formats (including posit for FPGA targets).

## NTU Type System

### Platform-Dependent Types (Resolved via Quotations)

| NTU Type | Meaning | x86_64 | ARM32 |
|----------|---------|--------|-------|
| `NTUint` | Platform word (signed) | i64 | i32 |
| `NTUuint` | Platform word (unsigned) | i64 | i32 |
| `NTUnint` | Native int (pointer-sized signed) | i64 | i32 |
| `NTUunint` | Native uint (pointer-sized unsigned) | u64 | u32 |
| `NTUptr<'T>` | Native pointer | 8 bytes | 4 bytes |
| `NTUsize` | Size type (`size_t`) | u64 | u32 |
| `NTUdiff` | Pointer difference (`ptrdiff_t`) | i64 | i32 |

### Fixed-Width Integer and Float Types (Platform-Independent)

| NTU Type | Meaning | Always |
|----------|---------|--------|
| `NTUint8` | 8-bit signed | i8 |
| `NTUint16` | 16-bit signed | i16 |
| `NTUint32` | 32-bit signed | i32 |
| `NTUint64` | 64-bit signed | i64 |
| `NTUuint8` | 8-bit unsigned | u8 |
| `NTUuint16` | 16-bit unsigned | u16 |
| `NTUuint32` | 32-bit unsigned | u32 |
| `NTUuint64` | 64-bit unsigned | u64 |
| `NTUfloat32` | 32-bit IEEE 754 float | f32 |
| `NTUfloat64` | 64-bit IEEE 754 float | f64 |

### Posit Types (Gustafson Type III Unum)

| NTU Type | Meaning | Target |
|----------|---------|--------|
| `NTUposit(w, es)` | Posit arithmetic, width `w`, exponent bits `es` | FPGA, specialized hardware |

Convenience aliases: `posit8` = `NTUposit(8, 0)`, `posit16` = `NTUposit(16, 1)`, `posit32` = `NTUposit(32, 2)`, `posit64` = `NTUposit(64, 3)`.

Posit arithmetic is selected automatically by the DTS representation selection logic when the dimensional domain and target capabilities warrant it (e.g., Xilinx FPGA with tapered-precision needs). See `CCS_Architecture.md` for representation selection rules.

### Memory Space Qualifiers

| NTU Qualifier | Meaning | MLIR Mapping |
|---------------|---------|--------------|
| `Default` | Implicit (stack or heap per escape) | default address space |
| `Stack` | Stack-allocated, lexically scoped | `alloca` |
| `Global` | Global/static lifetime | `global` |
| `Shared` | GPU shared memory | `gpu.shared` |

`NTUMemorySpace` qualifiers are attached to allocations during DMM coeffect analysis and consumed by Alex during MLIR emission.

## Layered Type Abstraction

```
┌─────────────────────────────────────────────────────────┐
│  Clef Application Code                                   │
│  int, uint, nativeint, nativeptr<'T>                    │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  CCS                                                    │
│  NTU types: NTUint, NTUuint, NTUsize, NTUposit, etc.   │
│  Dimensional metadata: Measure (abelian group)          │
└─────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────┐
│  Alex                                                    │
│  Witnesses quotations → concrete MLIR (i32/i64/posit)  │
└─────────────────────────────────────────────────────────┘
```

### Mapping Table

| Clef Surface Type | CCS Internal |
|-------------------|--------------|
| `int` | `NTUint` |
| `uint` | `NTUuint` |
| `nativeint` | `NTUnint` |
| `nativeptr<'T>` | `NTUptr<'T>` |
| `int32` | `NTUint32` |
| `int64` | `NTUint64` |
| `float` | `NTUfloat64` |
| `float32` | `NTUfloat32` |

## Type Identity vs Type Width

**Key Insight**: Type WIDTH is an erased assumption, not part of type identity.

### What CCS Enforces (Type Identity)
- `NTUint ≠ NTUint64` — These are **different types**
- `NTUint + NTUint` ✓ — Same type, valid
- `NTUint + NTUint64` ✗ — Different types, compile error

### What CCS Does NOT Assume (Type Width)
- Whether `NTUint` is 32 or 64 bits
- Memory layout of platform-dependent types
- Exact byte sizes

### What Alex Resolves (Type Width)
- Platform quotations provide width information
- `NTUint` on x86_64 → `i64`
- `NTUint` on ARM32 → `i32`
- `NTUposit(32, 2)` on Xilinx → posit32 softcore or hardened IP

## DTS Integration: Dimensions as Metadata on NTU Types

DTS (Dimensional Type System) extends NTU numerics with **dimensional annotations** — metadata drawn from a finitely generated free abelian group. The `Measure` type captures this:

```fsharp
type Measure =
    | MOne           // dimensionless
    | MVar of string // dimension variable (e.g., 'Length)
    | MProd of Measure * Measure  // dimension product
    | MInv of Measure             // dimension inverse
    | MCon of string              // named dimension (e.g., "kg")
```

A `float<newtons>` carries `NTUfloat64` as its NTU kind plus dimensional metadata `MProd(MCon "kg", MProd(MCon "m", MInv(MProd(MCon "s", MCon "s"))))` (kg·m·s⁻²). This metadata is **orthogonal to width** — CCS enforces dimensional correctness during unification; Alex ignores dimensional metadata during code generation.

The DTS extends NTU without complicating it. Width identity (CCS) and dimensional correctness (DTS) are independent axes of type safety.

See `CCS_Architecture.md` for the full DTS treatment.

## Implementation in CCS

### NTUKind Discriminated Union

```fsharp
/// NTU (Native Type Universe) type kinds
type NTUKind =
    // Platform-dependent (resolved via quotations)
    | NTUint      // Platform word, signed
    | NTUuint     // Platform word, unsigned
    | NTUnint     // Native int (pointer-sized signed)
    | NTUunint    // Native uint (pointer-sized unsigned)
    | NTUptr of NativeType  // Pointer to type
    | NTUsize     // size_t equivalent
    | NTUdiff     // ptrdiff_t equivalent

    // Fixed width (platform-independent)
    | NTUint8 | NTUint16 | NTUint32 | NTUint64
    | NTUuint8 | NTUuint16 | NTUuint32 | NTUuint64
    | NTUfloat32 | NTUfloat64

    // Posit (Gustafson Type III Unum)
    | NTUposit of NTUWidth * es: int  // width, exponent bits
```

### NTULayout (Erased Assumptions)

```fsharp
/// Platform-resolved type layout (erased at runtime)
type NTULayout = {
    Kind: NTUKind
    /// Erased - only for type checking, resolved by Alex
    AssumedSize: int option
    AssumedAlignment: int option
}
```

## Pipeline Flow

```
Clef Source (int, nativeint, float<kg·m·s⁻²>)
    │
    ▼
┌──────────────────────────────────────────────┐
│ CCS: Maps to NTU types                      │
│ - Type identity checking                     │
│ - SRTP resolution                            │
│ - Dimensional constraint propagation (DTS)   │
│ - DMM escape classification                  │
│ - Width is NOT resolved here                 │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ PSG: Carries NTU + dimensional annotations  │
│ - Platform quotations attached               │
│ - Escape classifications annotated          │
│ - Type identity preserved                    │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ Alex: Witnesses quotations                  │
│ - Reads Fidelity.Platform bindings           │
│ - Resolves NTUint → i64 on x86_64           │
│ - Selects posit for FPGA targets where DTS  │
│   dimensional domain warrants it            │
│ - Generates platform-specific MLIR          │
└──────────────────────────────────────────────┘
    │
    ▼
Native Binary (IEEE 754, posit, or fixed-point per target)
```

## Platform Predicates (F*-Inspired)

Following the `fits_u32`/`fits_u64` pattern:

```fsharp
module Platform.Predicates =
    /// Platform supports 64-bit word operations
    val fits_u64 : Expr<bool>

    /// Platform has AVX-512 vector support
    val has_avx512 : Expr<bool>

    /// Platform has posit hardware support
    val has_posit_hw : Expr<bool>
```

These predicates enable conditional compilation without runtime checks. Alex witnesses them to eliminate dead branches at compile time.

## Related Documentation

- `CCS_Architecture.md` - DTS/DMM architecture and NTU type checking
- `Platform_Binding_Model.md` - Fidelity.Platform binding architecture
- `Architecture_Canonical.md` - Composer pipeline overview
