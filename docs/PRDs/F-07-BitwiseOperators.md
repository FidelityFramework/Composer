# F-07: Bitwise Operators

> **Sample**: `07_BitsTest` | **Status**: Active | **Category**: Foundation

## 1. Executive Summary

This sample establishes the lowering path for F#'s native bitwise operators
(`&&&`, `|||`, `^^^`, `<<<`, `>>>`).  These operators are **type-preserving**
— result type equals operand type — making them dimensionally safe for use in
Fidelity without any hidden coercion machinery.

**What was removed**: The original F-07 defined `Bits.htons`/`ntohs`/`htonl`/`ntohl`
(C-style network byte order) and float↔int bitcast reinterpretation.  Both are
architecturally wrong for Fidelity:

- Byte-order conversion is an *algorithmic* operation the user builds explicitly
  using shifts and masks — not a magic intrinsic.
- Float↔int reinterpretation crosses type boundaries in ways that violate the
  Dimensional Type System.  If a user needs this, they build the machinery via
  NTUTypes; there is no "assumed behavior."

**What replaces them**: F#'s idiomatic bitwise operators, which operate *within*
a type and map one-to-one to `arith` dialect operations in MLIR.

---

## 2. Surface Feature

```fsharp
module BitwiseOps

[<EntryPoint>]
let main _ =
    // Isolate a bit field
    let field = 0b11010110 &&& 0b00111100  // 20

    // Combine flag sets
    let combined = 0b10100000 ||| 0b00000101  // 165

    // Toggle bits
    let toggled = 0b11001100 ^^^ 0b01010101  // 153

    // Bitwise complement (all bits flipped)
    let complement = ~~~0   // -1 (all ones in two's complement)

    // Scale by power of 2
    let scaled = 1 <<< 4  // 16

    // Halve (arithmetic, sign-preserving)
    let halved = 256 >>> 3  // 32
    0
```

No module prefix, no magic function calls — these are F# operators that belong
to every ML programmer's mental model.

---

## 3. Infrastructure Contributions

### 3.1 Operator → MLIR Mapping

Added to `classifyAtomicOp` in `PSGCombinators.fs`:

| F# Operator | FNCS Operation    | MLIR (arith dialect)       | Notes                              |
|-------------|-------------------|----------------------------|------------------------------------|
| `&&&`       | `op_BitwiseAnd`   | `arith.andi`               | Always integer; no float analog    |
| `\|\|\|`    | `op_BitwiseOr`    | `arith.ori`                | Always integer; no float analog    |
| `^^^`       | `op_ExclusiveOr`  | `arith.xori`               | Always integer; no float analog    |
| `~~~`       | `op_LogicalNot`   | `arith.constant -1; xori`  | Bitwise complement via all-ones mask |
| `<<<`       | `op_LeftShift`    | `arith.shli`               | Same encoding signed/unsigned      |
| `>>>`       | `op_RightShift`   | `arith.shrsi`              | Arithmetic (sign-preserving); signed integers |

**Future**: `op_RightShift` on unsigned types should lower to `arith.shrui`
(logical shift right).  This requires a type-aware dispatch in `pBinaryArithOp`,
analogous to how `div` dispatches to `divsi`/`divui`.

**`~~~` (bitwise complement)**: `op_LogicalNot` → `UnaryArith "complement"` → `pBitwiseNot`.
Emits `arith.constant -1 : operandType` then `arith.xori %x, %allones`.
In two's complement, `-1` is all-ones — `~~~0 = -1`, `~~~0xFF = -256`, etc.

### 3.2 PSGCombinators.fs

```fsharp
// Bitwise operators — always integer, type-preserving (no float analog)
| IntrinsicModule.Operators, "op_BitwiseAnd"  -> BinaryArith "andi"
| IntrinsicModule.Operators, "op_BitwiseOr"   -> BinaryArith "ori"
| IntrinsicModule.Operators, "op_ExclusiveOr" -> BinaryArith "xori"
| IntrinsicModule.Operators, "op_LogicalNot"  -> UnaryArith "complement"  // ~~~
// Shift operators — integer only; shrsi = arithmetic (sign-preserving) for signed int
| IntrinsicModule.Operators, "op_LeftShift"   -> BinaryArith "shli"
| IntrinsicModule.Operators, "op_RightShift"  -> BinaryArith "shrsi"
```

These feed into the existing `pBinaryArithIntrinsic` pipeline — no new pattern
infrastructure required.

---

## 4. PSG Representation

```
ModuleOrNamespace: BitwiseOps
├── LetBinding: field
│   └── Application
│       ├── Intrinsic: Operators.op_BitwiseAnd
│       ├── Argument: 0b11010110 (Const 214)
│       └── Argument: 0b00111100 (Const 60)
├── LetBinding: combined
│   └── Application
│       ├── Intrinsic: Operators.op_BitwiseOr
│       └── ...
├── LetBinding: toggled (op_ExclusiveOr)
├── LetBinding: scaled  (op_LeftShift)
├── LetBinding: halved  (op_RightShift)
└── StatementSequence (Console.write / writeln calls)
```

---

## 5. MLIR Output

```mlir
// field = 214 &&& 60 = 20
%c214 = arith.constant 214 : i64
%c60  = arith.constant 60  : i64
%field = arith.andi %c214, %c60 : i64

// combined = 160 ||| 5 = 165
%c160 = arith.constant 160 : i64
%c5   = arith.constant 5   : i64
%combined = arith.ori %c160, %c5 : i64

// toggled = 204 ^^^ 85 = 153
%c204 = arith.constant 204 : i64
%c85  = arith.constant 85  : i64
%toggled = arith.xori %c204, %c85 : i64

// scaled = 1 <<< 4 = 16
%c1  = arith.constant 1 : i64
%c4  = arith.constant 4 : i64
%scaled = arith.shli %c1, %c4 : i64

// halved = 256 >>> 3 = 32
%c256 = arith.constant 256 : i64
%c3   = arith.constant 3   : i64
%halved = arith.shrsi %c256, %c3 : i64
```

---

## 6. Expected Output

```
AND: 20
OR:  165
XOR: 153
NOT: -1
SHL: 16
SHR: 32
```

---

## 7. Coeffects

| Coeffect | Purpose |
|----------|---------|
| NodeSSAAllocation | SSA for all bindings |

No new coeffects — bitwise operators are pure computations.

---

## 8. Design Principles Embodied

1. **Type-preserving**: `int &&& int = int`.  No implicit widening, narrowing, or
   reinterpretation.  Dimensionally safe by construction.

2. **No magic**: The user does not need any "Bits.*" namespace.  These are
   core F# language operators (`Microsoft.FSharp.Core.Operators`).

3. **Direct MLIR mapping**: Each operator is a single `arith` instruction.
   The compiler does not synthesize multi-op sequences for basic bit operations.

4. **Byte order is user code**: If a protocol needs big-endian encoding of a
   little-endian value, the user writes `(x >>> 8) ||| (x <<< 8)` — exactly
   what C's `htons` does under the hood, but visibly and explicitly.

---

## 9. Downstream Dependencies

- **F-08 / F-09**: DU tag extraction (`DUGetTag`) and payload access use
  `arith.andi` / `arith.shrsi` patterns internally in the compiler.  The
  operator path validated here is the same path the DU machinery uses.

- **I-01 Socket Basics**: Network byte order for protocol fields is composed
  from `<<<`/`>>>` and `|||` — user-visible, no hidden endianness magic.

---

## 10. Related Documents

- [F-05-DiscriminatedUnions](F-05-DiscriminatedUnions.md) — DU payload storage uses the same arith ops internally
- [Architecture_Canonical.md](../Architecture_Canonical.md) — Layer separation principles
- [XParsec_PSG_Architecture.md](../XParsec_PSG_Architecture.md) — classifyAtomicOp dispatch model
