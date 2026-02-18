# CCS: Clef Compiler Services

## Executive Summary

CCS (Clef Compiler Services) is a minimal, surgical fork of the legacy compiler-services lineage that provides native-first type resolution for the Fidelity framework. Rather than fighting legacy BCL-centric assumptions downstream in Composer, CCS addresses the fundamental problem at its source: the type system itself.

CCS is **not** a full rewrite of the upstream stack. It is a targeted modification that:

1. Provides native semantics for standard Clef types (`string`, `option`, `array`, etc.)
2. Types string literals with UTF-8 fat pointer semantics instead of `System.String`
3. Resolves SRTP against a native witness hierarchy
4. Enforces null-free semantics where BCL would allow nulls
5. Maintains a compatible API surface for seamless Composer integration

**Key Principle**: Users write standard Clef type names. CCS provides native semantics transparently. No "NativeStr" or other internal naming - `string` is `string` everywhere.

## The Problem: BCL-Centric Type Universe

The legacy compiler-services model was designed for .NET compilation. Its type system assumes BCL primitives:

```fsharp
// In legacy CheckExpressions.fs, line ~7342
| false, LiteralArgumentType.Inline ->
    TcPropagatingExprLeafThenConvert cenv overallTy g.string_ty env m (fun () ->
        mkString g m s, tpenv)
```

When you write `"Hello"` in Clef code, the legacy service model **always** types it with BCL string semantics (`System.String`). This is hardcoded. No amount of type shadows, namespace tricks, or downstream transformations can change this fundamental behavior.

### Why Type Shadows Don't Work

Alloy attempted to shadow BCL types:

```fsharp
// In Alloy - this was the attempted solution (now removed)
type string = NativeStr
```

But type shadows only affect **type annotations**, not **literal inference**:

```fsharp
let x: string = value  // Shadow works here
let y = "Hello"        // Shadow IGNORED: y is still inferred with BCL string semantics
```

This asymmetry creates an impossible situation. User code looks correct (`Console.Write "Hello"`), but the checker has already fixed `"Hello"` to BCL string semantics before Alloy-level abstractions can influence typing.

### The Downstream Consequence

Because the inherited checker model outputs BCL-typed results, every downstream component must either:

1. **Reject BCL types** - Composer's `ValidateNativeTypes` does this, but then valid-looking code fails
2. **Transform BCL types** - Violates "no lowering" principle, creates semantic mismatches
3. **Accept BCL types** - Requires BCL runtime, defeats purpose of native compilation

None of these options are acceptable. The fix must happen at the source: the type system.

## The Solution: Native Semantics for Standard Clef Types

CCS replaces the inherited BCL-centric semantics with **native semantics** for standard Clef types:

```
┌─────────────────────────────────────────────────────────────────────┐
│  Legacy Service Type Universe (BCL-centric)                         │
│                                                                     │
│  string         → System.String (UTF-16, heap-allocated, nullable)  │
│  option<'T>     → Reference-oriented option semantics (nullable)    │
│  array<'T>      → System.Array (runtime-managed)                    │
│  String literal → System.String (hardcoded)                         │
│  SRTP           → Searches BCL method tables                        │
└─────────────────────────────────────────────────────────────────────┘

                           ↓ CCS Migration ↓

┌─────────────────────────────────────────────────────────────────────┐
│  CCS Type Universe (Native semantics)                              │
│                                                                     │
│  string         → UTF-8 fat pointer {Pointer, Length}               │
│  option<'T>     → Value-type, stack-allocated, never null           │
│  array<'T>      → Fat pointer {Pointer, Length}                     │
│  String literal → UTF-8 fat pointer (native semantics)              │
│  SRTP           → Searches native witness hierarchy                 │
└─────────────────────────────────────────────────────────────────────┘
```

**Note**: The type NAME stays `string` - only the SEMANTICS change. No cognitive overhead for maintainers.

## Architectural Layering with CCS

CCS changes the fundamental layering of the Fidelity stack:

### Before (Legacy Service Model)

```
User Code: Console.Write "Hello"
    ↓
Legacy checker: Types "Hello" as System.String (BCL)
Legacy checker: SRTP searches BCL method tables
    ↓
Composer/Baker: Receives BCL-typed tree
Composer: ValidateNativeTypes FAILS (BCL detected)
    ↓
ERROR: BCL types in native compilation
```

### After (CCS)

```
User Code: Console.Write "Hello"
    ↓
CCS: Types "Hello" as string with native semantics (UTF-8 fat pointer)
CCS: SRTP searches native witness hierarchy
    ↓
Composer/Baker: Receives native-typed tree
Composer: ValidateNativeTypes PASSES (native semantics)
    ↓
Alex: Generates MLIR directly
    ↓
Native binary
```

## What CCS Contains

CCS is a focused modification. It contains:

### 1. Native Semantics for Primitive Types

The core type semantics redefinitions:

```fsharp
// Conceptual - actual implementation in TcGlobals.fs
// Type NAMES remain standard Clef; SEMANTICS are native
type CCSSemantics = {
    // string has UTF-8 fat pointer semantics
    string_semantics: {| Pointer: nativeptr<byte>; Length: int |}

    // option has value semantics (stack-allocated, never null)
    option_semantics: voption<'T>

    // array has fat pointer semantics
    array_semantics: {| Pointer: nativeptr<'T>; Length: int |}

    // Span for memory views
    span_semantics: {| Pointer: nativeptr<'T>; Length: int |}
}
```

### 2. String Literal Type Resolution

The key modification in `CheckExpressions.fs`:

```fsharp
// CCS modification - string literals have native semantics
| false, LiteralArgumentType.Inline ->
    TcPropagatingExprLeafThenConvert cenv overallTy g.string_ty env m (fun () ->
        mkString g m s, tpenv)  // Same API shape, but creates string with native semantics
```

### 3. Native SRTP Witness Resolution

SRTP resolution searches native witnesses instead of BCL method tables:

```fsharp
// CCS SRTP resolution
let resolveTraitCall (traitInfo: TraitConstraintInfo) =
    // Search native witness hierarchy, not BCL
    match traitInfo.MemberName with
    | "op_Addition" -> searchNativeWitness NumericOps.Add
    | "op_Dollar" -> searchNativeWitness WritableString.op_Dollar
    | "get_Length" -> searchNativeWitness Measurable.Length
    // ... native witnesses for all SRTP-resolvable operations
```

### 4. Null-Free Semantics

CCS enforces null-free semantics where BCL would allow nulls:

```fsharp
// CCS null handling
let checkNullAssignment targetType sourceExpr =
    match targetType with
    | t when hasNativeSemantics t ->
        // Native types are NEVER null - error if null assigned
        if isNullLiteral sourceExpr then
            error "Native types cannot be null"
    | _ -> ()
```

### 5. Inline Expansion for Escape Analysis

CCS captures function bodies for functions marked `inline` and expands them at call sites during type checking. This is critical for **escape analysis** of stack-allocated memory.

**The Problem**: When a function allocates via `NativePtr.stackalloc` and returns a pointer, that pointer is invalid after the function returns (stack frame deallocated).

**The Solution**: CCS expands `inline` function bodies at call sites, lifting allocations to the caller's frame:

```fsharp
// In Platform/Console.fs
let inline readln () : string =
    let buffer = NativePtr.stackalloc<byte> 256  // Allocation
    let len = readLineInto buffer 256
    NativeStr.fromPointer buffer len             // Returns fat pointer

// In user code
let hello() =
    Console.readln() |> greet
    // CCS expands readln inline:
    // - stackalloc now in hello's frame
    // - fat pointer valid through hello's scope
```

**Implementation**:
- `Bindings.fs`: Captures `InlineBody` for functions with `isInline = true`
- `Applications.fs`: Checks `InlineBody` at call sites and expands when present
- Environment substitution binds parameters to argument values

**When to Mark Functions `inline`**:
- Allocates via `stackalloc` or `Arena.alloc`
- Returns pointer/reference/fat pointer to that allocation
- Caller needs returned value to remain valid

This enables **Level 1 (Implicit) memory management** - developers write standard Clef while the compiler ensures safety via inline expansion.

### 6. BCL-Sympathetic API Surface

Despite native semantics, the API surface remains familiar:

```fsharp
// User code looks exactly like BCL Clef
module Console =
    let Write (s: string) = ...      // string has native semantics in CCS
    let WriteLine (s: string) = ...
    let ReadLine () : string = ...

module String =
    let length (s: string) : int = ...
    let concat (sep: string) (strings: string seq) : string = ...
```

## What CCS Does NOT Contain

CCS is intentionally minimal. It does NOT include:

### Platform Bindings

Platform-specific operations remain in Composer/Alex:

```fsharp
// NOT in CCS - stays in Alloy/Composer
module Platform.Bindings =
    let writeBytes fd buffer count : int = ...
    let readBytes fd buffer maxCount : int = ...
```

CCS defines type semantics; Composer implements operations on those types.

### Runtime Implementations

CCS provides type resolution, not runtime code:

```fsharp
// NOT in CCS - runtime implementation in Alloy
let inline concat2 (dest: nativeptr<byte>) (s1: string) (s2: string) : string =
    // Actual byte-copying implementation
    ...
```

### Code Generation

CCS produces typed trees. Code generation is Composer/Alex's domain:

```fsharp
// NOT in CCS - stays in Alex
let emitString (ctx: EmissionContext) (str: string) : MLIR =
    // Place bytes in data section, emit struct construction
    ...
```

## Integration with Composer

CCS produces **Clef.Compiler.Service.dll** - a distinct library with native-first type resolution.

### Repository Structure

```
dotnet/fsharp (upstream reference)
    │
    └──→ SpeakEZ/clef (pure divergence)
              │
              ├── main          ← Initial fork point (reference only)
              │
              └── clef      ← Active development branch
                                   All CCS modifications here
```

**Key decisions:**
- **Pure divergence** - No ongoing merge from upstream. `dotnet/fsharp` is reference only.
- **Distinct identity** - Output is `Clef.Compiler.Service.dll`, not a drop-in replacement
- **Companion spec** - `SpeakEZ/clef-spec` documents the native Clef dialect

### Bridge Baseline (Package Mode)

```xml
<!-- Composer.fsproj -->
<PackageReference Include="Clef.Compiler.Service" Version="43.8.x" />
```

### With CCS (Local Build)

```xml
<!-- Composer.fsproj -->
<Reference Include="Clef.Compiler.Service">
  <HintPath>../clef/artifacts/bin/Clef.Compiler.Service/Release/netstandard2.0/Clef.Compiler.Service.dll</HintPath>
</Reference>
```

### API Compatibility

CCS keeps compatibility with the legacy service API shape. Existing Composer code requires minimal changes:

```fsharp
// Namespace migration target: legacy compiler namespace -> Clef-native compiler namespace
open Clef.Native.Compiler.CodeAnalysis
open Clef.Native.Compiler.Symbols

let checker = createChecker()
let results = checker.ParseAndCheckFileInProject(...)
let typedTree = results.TypedTree  // Contains string with native semantics
```

The API shape is the same; the namespace and type semantics differ.

## The Resulting Layer Separation

With CCS, the Fidelity stack has clean layer separation:

| Layer | Responsibility |
|-------|---------------|
| **CCS** | Type universe, literal typing, type inference, SRTP resolution, **PSG construction**, editor services |
| **Alloy** | Library implementations using standard Clef types (Console, String, etc.) |
| **Composer/Alex** | **Consumes PSG from CCS**, platform-aware MLIR generation |
| **Platform Bindings** | Syscall implementations per platform |

**Key Architecture Change**: CCS now builds the PSG (Program Semantic Graph). Composer consumes the PSG as "correct by construction" and focuses purely on code generation.

Each layer has a single responsibility. No layer needs to "work around" another layer's assumptions.

## Impact on Existing Components

### ValidateNativeTypes

With CCS, `ValidateNativeTypes` becomes simpler:

```fsharp
// Before: Complex classification, many edge cases
// After: Simple - CCS guarantees native semantics
let validateNode (node: PSGNode) =
    // CCS already ensured all types have native semantics
    // This pass becomes a sanity check, not a gatekeeper
    ()
```

### Baker/TypedTreeZipper

Baker's job becomes easier:

```fsharp
// Before: Extract types, handle BCL/native mismatches
// After: Types already have native semantics from CCS
let overlayTypes (node: PSGNode) (typedExpr: TypedExpr) =
    // Types from typedExpr already have native semantics
    // No translation needed
    node.Type <- typedExpr.Type
```

### Alloy Type Shadows

Type shadows are no longer needed:

```fsharp
// Before: Attempted shadowing (didn't work for literals)
type string = NativeStr  // This was wrong and is now removed

// After: Not needed - CCS provides native semantics for string
// Alloy is a pure library with no type system workarounds
```

### Console.Write and SRTP

SRTP resolution becomes straightforward:

```fsharp
// Before: SRTP resolved against BCL, then we tried to redirect
// After: SRTP resolves against native witnesses directly

type WritableString =
    | WritableString
    static member inline ($) (WritableString, s: string) = writeString s
    // CCS ensures string has native semantics
```

## Implementation Roadmap

### Phase 0: Repository Setup (DONE)

- **`SpeakEZ/clef`** - Fork of `dotnet/fsharp` on GitHub
- **`SpeakEZ/clef-spec`** - Fork of `fsharp/fslang-spec` on GitHub
- **Branch**: `clef` for all development (main frozen at fork point)
- **Pure divergence** - `dotnet/fsharp` is reference only, kept at `~/repos/fsharp`

### Phase 1: Assembly Identity and Build Infrastructure

1. Rename assembly to `Clef.Compiler.Service`
2. Update namespaces: `Clef.Compiler.*` → `Clef.Native.Compiler.*`
3. Verify build produces correctly named DLL
4. Create test harness for validating type resolution

### Phase 2: Native Semantics for Primitive Types

1. Define native semantics for `string` (UTF-8 fat pointer)
2. Define native semantics for `option<'T>` (value type, never null)
3. Define native semantics for `array<'T>`, `Span<'T>`
4. Wire these into `TcGlobals.fs`

### Phase 3: Literal Type Resolution

1. Modify `TcConstStringExpr` to produce string with native semantics
2. Update string literal handling throughout checker
3. Ensure string operations type-check correctly

### Phase 4: SRTP Native Witnesses

1. Define native witness hierarchy in CCS
2. Modify constraint solver to search native witnesses
3. Ensure common operations (`+`, `$`, `Length`, etc.) resolve correctly

### Phase 5: Null-Free Enforcement

1. Add null checks where BCL would allow null
2. Emit errors for null assignments to native types
3. Ensure `option` has value semantics (never null)

### Phase 6: Integration and Testing

1. Update Composer to reference CCS DLL
2. Verify HelloWorld samples compile
3. Run full test suite
4. Document any API differences

## CCS Files Requiring Modification

Based on the "From Bridged to Self Hosted" analysis and CCS structure:

| File | Modification |
|------|-------------|
| `src/Compiler/Checking/TcGlobals.fs` | Native semantics definitions |
| `src/Compiler/Checking/CheckExpressions.fs` | String literal typing (~line 7342) |
| `src/Compiler/Checking/ConstraintSolver.fs` | SRTP native witness resolution |
| `src/Compiler/TypedTree/TypedTree.fs` | Native type representations |
| `src/Compiler/TypedTree/TypedTreeOps.fs` | `mkString` produces native semantics |

The modification surface is intentionally small - surgical changes to type resolution, not a rewrite.

## Relationship to Self-Hosting Roadmap

CCS is an **intermediate step** on the path to full self-hosting:

```
Current:     Legacy Service (BCL) → Composer → Native Binary
                ↓
Near-Term:   CCS (Native) → Composer → Native Binary  ← WE ARE HERE
                ↓
Extracted:   Composer.Syntax (Native) → Composer → Native Binary
                ↓
Self-Hosted: Native Composer → Native Composer
```

CCS provides immediate relief from BCL contamination while maintaining .NET tooling for Composer itself. The full extraction (as described in "From Bridged to Self Hosted") remains the long-term goal, but CCS unblocks current development.

## Success Criteria

CCS is successful when:

1. `Console.Write "Hello"` compiles without BCL type errors
2. String literals in the typed tree have native semantics, not `System.String`
3. SRTP resolves against native witnesses
4. `ValidateNativeTypes` passes without special cases
5. HelloWorld samples execute correctly
6. No changes required to Alloy namespace structure
7. No "lowering" or "transformation" passes needed in Composer

## Conclusion

CCS represents a pragmatic middle ground between fighting inherited BCL assumptions downstream and undertaking a full compiler extraction. By making targeted modifications to compiler-service type semantics, we get:

- **Correctness**: Types have native semantics from the start
- **Simplicity**: No downstream workarounds, no internal naming differences
- **Compatibility**: Compatible API surface, same Composer integration model
- **Progress**: Immediate unblocking of current development
- **Foundation**: Clear path to full self-hosting

The key insight is that the problem was never in Composer, Baker, or Alloy - it was in the inherited assumption that all Clef code targets .NET/BCL semantics. CCS corrects that assumption at the source while maintaining standard Clef type names for zero cognitive overhead.
