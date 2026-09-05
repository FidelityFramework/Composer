/// TypeMapping - CCS NativeType to MLIR type conversion
///
/// Maps CCS native types to their MLIR representations.
/// Uses structured MLIRType from Alex.Dialects.Core.Types.
///
/// CCS-native: Uses NativeType from Clef.Compiler.NativeTypedTree
module Alex.CodeGeneration.TypeMapping

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.NativeTypedTree.UnionFind
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Core.Types.Dialects

// ═══════════════════════════════════════════════════════════════════════════
// TYPE MAPPING DIAGNOSTIC COLLECTION
// ═══════════════════════════════════════════════════════════════════════════

/// Collects AX1001 diagnostics during type mapping so compilation can continue
/// and report ALL unbound type variables, not just the first one.
/// Drained by mapType after each call.
let private typeMappingErrors = System.Collections.Generic.List<string>()

/// Drain collected type mapping errors. Returns the list and clears the collector.
let drainTypeMappingErrors () : string list =
    let errors = typeMappingErrors |> Seq.toList
    typeMappingErrors.Clear()
    errors

/// The target platform the current compilation lowers to. Set once by MLIRGeneration
/// before any type is mapped. It decides representations that differ per target but are
/// reached through platform-agnostic entry points (mapNativeTypeForArch): the enum DU tag.
let mutable private currentTargetPlatform : TargetPlatform option = None

/// Record the target platform for representation decisions made during type mapping.
let setTargetPlatform (platform: TargetPlatform) : unit =
    currentTargetPlatform <- Some platform

/// Representation of a nullary-cases-only DU (an enumeration) for the current target.
/// FPGA: abstract TTag (platform elision picks the width). CPU/MCU: a one-byte memref,
/// the same shape every other DU has, so construction, tag extraction and field storage
/// share one path.
let private enumTagRepresentation (caseCount: int) : MLIRType =
    match currentTargetPlatform with
    | Some FPGA -> TTag caseCount
    | _ -> TMemRefStatic (1, TInt (IntWidth 8))

// ═══════════════════════════════════════════════════════════════════════════
// TYPE SIZE COMPUTATION (for DU slot sizing)
// ═══════════════════════════════════════════════════════════════════════════

/// Compute max payload size in bytes for heterogeneous DUs
let maxPayloadBytes (arch: Architecture) (ty1: MLIRType) (ty2: MLIRType) : int =
    max (mlirTypeSize arch ty1) (mlirTypeSize arch ty2)

/// Physical storage type of a value: records and tuples are semantic TStruct values whose
/// storage is a byte memref of the struct's size (the size RecordWitness allocates and the
/// size FieldGet views assume). Every container element type (array element, option payload,
/// slot element) must use this physical type so that all uses of a record agree on its size.
let physicalStorageType (arch: Architecture) (ty: MLIRType) : MLIRType =
    match ty with
    | TStruct fields -> TMemRefStatic (fields |> List.sumBy (fun (_, t) -> mlirTypeSize arch t), TInt (IntWidth 8))
    | t -> t

// ═══════════════════════════════════════════════════════════════════════════
// NTUKind DIRECT MAPPING (for literals)
// ═══════════════════════════════════════════════════════════════════════════

/// Map NTUKind directly to MLIRType with platform and architecture awareness.
/// Used for NativeLiteral where we have the kind without a full NativeType.
///
/// PRINCIPLED DESIGN (February 2026 — DTS/Width Inference):
/// The NTUKind in a literal IS the type. Fixed-width kinds (int8, int16, etc.)
/// resolve to their declared width on all platforms. Platform-word kinds
/// (int, uint — NTUWidth.Resolved WidthDimension.Register) resolve differently:
///   - CPU: architecture register width (32 or 64 bits)
///   - FPGA: IntWidth 0 (abstract — width comes from interval analysis coeffect)
/// On FPGA there is no "register width." Every integer is exactly as wide as the
/// design requires. Width is a design property, not a platform property.
let mapNTUKindToMLIRType (platform: TargetPlatform) (arch: Architecture) (kind: NTUKind) : MLIRType =
    match kind with
    // Fixed-width signed integers — same on all platforms
    | NTUKind.NTUint (NTUWidth.Fixed 8) -> TInt (IntWidth 8)
    | NTUKind.NTUint (NTUWidth.Fixed 16) -> TInt (IntWidth 16)
    | NTUKind.NTUint (NTUWidth.Fixed 32) -> TInt (IntWidth 32)
    | NTUKind.NTUint (NTUWidth.Fixed 64) -> TInt (IntWidth 64)
    // Fixed-width unsigned integers (same MLIR type, signedness is in ops)
    | NTUKind.NTUuint (NTUWidth.Fixed 8) -> TInt (IntWidth 8)
    | NTUKind.NTUuint (NTUWidth.Fixed 16) -> TInt (IntWidth 16)
    | NTUKind.NTUuint (NTUWidth.Fixed 32) -> TInt (IntWidth 32)
    | NTUKind.NTUuint (NTUWidth.Fixed 64) -> TInt (IntWidth 64)
    // Platform-word integers — width depends on target platform
    | NTUKind.NTUint (NTUWidth.Resolved WidthDimension.Register)
    | NTUKind.NTUuint (NTUWidth.Resolved WidthDimension.Register) ->
        match platform with
        | FPGA -> TInt (IntWidth 0)  // Abstract: width from interval analysis, not architecture
        | _ -> TInt (declaredWordWidth arch)   // CPU/MCU: the declared Register width
    // Native pointer-sized types - map to MLIR index for memref operations
    | NTUKind.NTUint (NTUWidth.Resolved WidthDimension.Pointer)  // nativeint
    | NTUKind.NTUuint (NTUWidth.Resolved WidthDimension.Pointer) // unativeint
    | NTUKind.NTUsize     // size_t
    | NTUKind.NTUdiff     // ptrdiff_t
        -> TIndex
    // Floating point
    | NTUKind.NTUfloat (NTUWidth.Fixed 32) -> TFloat F32
    | NTUKind.NTUfloat (NTUWidth.Fixed 64) -> TFloat F64
    // Boolean
    | NTUKind.NTUbool -> TInt (IntWidth 1)
    // Character (Unicode codepoint = i32)
    | NTUKind.NTUchar -> TInt (IntWidth 32)
    // Unit
    | NTUKind.NTUunit -> TInt (IntWidth 32)  // Unit represented as i32 0
    // Pointers
    | NTUKind.NTUptr | NTUKind.NTUfnptr -> TIndex
    // String as memref (portable MLIR type, not LLVM struct)
    // memref<?xi8> represents a dynamic-sized buffer with length tracked in descriptor
    | NTUKind.NTUstring -> TMemRef (TInt (IntWidth 8))
    // Composite/complex types - representation comes from platform tier, not here
    | kind -> failwithf "NTUKind %A requires platform-tier resolution, not scalar mapping" kind

// ═══════════════════════════════════════════════════════════════════════════
// MAIN TYPE MAPPING
// ═══════════════════════════════════════════════════════════════════════════

/// Map CCS NativeType to structured MLIRType with architecture awareness.
/// This is the canonical conversion used throughout Alex.
/// Uses NTU layout information for platform-aware type mapping.
///
/// PRINCIPLED DESIGN (January 2026):
/// PlatformWord types (int, uint, nativeint, size_t, ptrdiff_t) resolve to
/// the actual word size of the target architecture. This is NOT hardcoded to i64!
/// - 64-bit targets (x86_64, ARM64, RISCV64): i64
/// - 32-bit targets (ARM32, RISCV32, WASM32): i32
///
/// The architecture is passed explicitly to ensure correct codegen for all targets.
let rec mapNativeTypeForArch (arch: Architecture) (ty: NativeType) : MLIRType =
    let rec stripQualifiedLayout (layout: TypeLayout) : TypeLayout =
        match layout with
        | TypeLayout.Qualified (inner, _) -> stripQualifiedLayout inner
        | other -> other

    /// One type-constructor table for both the `TApp` and the `TNum` forms: a numeric type is
    /// read off its carrier (the tycon, with its NTUKind and layout) exactly as the arity-0
    /// `TApp` was. Composer reads; it decides no width here.
    let mapTyCon (tycon: TypeConRef) (args: NativeType list) : MLIRType =
        let tyconLayout = stripQualifiedLayout tycon.Layout
        // FIRST: Check NTU layout for types that have it - this is the authoritative source
        // for platform-dependent types like int (PlatformWord)
        match tyconLayout, tycon.NTUKind with
        // Zero-size unit type
        | TypeLayout.Inline (0, 1), Some NTUKind.NTUunit -> TInt (IntWidth 32)
        // Boolean: 1-bit
        | TypeLayout.Inline (1, 1), Some NTUKind.NTUbool -> TInt (IntWidth 1)
        // Fixed-width integers by NTUKind
        | _, Some (NTUKind.NTUint (NTUWidth.Fixed 8)) -> TInt (IntWidth 8)
        | _, Some (NTUKind.NTUuint (NTUWidth.Fixed 8)) -> TInt (IntWidth 8)
        | _, Some (NTUKind.NTUint (NTUWidth.Fixed 16)) -> TInt (IntWidth 16)
        | _, Some (NTUKind.NTUuint (NTUWidth.Fixed 16)) -> TInt (IntWidth 16)
        | _, Some (NTUKind.NTUint (NTUWidth.Fixed 32)) -> TInt (IntWidth 32)
        | _, Some (NTUKind.NTUuint (NTUWidth.Fixed 32)) -> TInt (IntWidth 32)
        | _, Some (NTUKind.NTUint (NTUWidth.Fixed 64)) -> TInt (IntWidth 64)
        | _, Some (NTUKind.NTUuint (NTUWidth.Fixed 64)) -> TInt (IntWidth 64)
        // Platform-word integers (int, uint) - size depends on architecture
        | TypeLayout.PlatformWord, Some (NTUKind.NTUint (NTUWidth.Resolved WidthDimension.Register))
        | TypeLayout.PlatformWord, Some (NTUKind.NTUuint (NTUWidth.Resolved WidthDimension.Register))
        | TypeLayout.PlatformWord, None -> TInt (declaredWordWidth arch)  // the declared Register width
        // Native pointer-sized types (nativeint, size_t, etc.) - map to index for memref
        | TypeLayout.PlatformWord, Some (NTUKind.NTUint (NTUWidth.Resolved WidthDimension.Pointer))
        | TypeLayout.PlatformWord, Some (NTUKind.NTUuint (NTUWidth.Resolved WidthDimension.Pointer))
        | TypeLayout.PlatformWord, Some NTUKind.NTUsize
        | TypeLayout.PlatformWord, Some NTUKind.NTUdiff
            -> TIndex
        // Pointers
        | TypeLayout.PlatformWord, Some NTUKind.NTUptr
        | TypeLayout.PlatformWord, Some NTUKind.NTUfnptr -> TIndex
        | _, Some NTUKind.NTUptr -> TIndex
        // Floats
        | _, Some (NTUKind.NTUfloat (NTUWidth.Fixed 32)) -> TFloat F32
        | _, Some (NTUKind.NTUfloat (NTUWidth.Fixed 64)) -> TFloat F64
        // Char (Unicode codepoint)
        | _, Some NTUKind.NTUchar -> TInt (IntWidth 32)
        // String as memref (portable MLIR type, not LLVM struct)
        // At F# level: string has .Pointer/.Length accessors (CCS synthetic members)
        // At MLIR level: memref<?xi8> (dynamic buffer)
        // Descriptor (ptr+size) is MLIR's concern, not explicitly modeled here
        | TypeLayout.FatPointer, Some NTUKind.NTUstring -> TMemRef (TInt (IntWidth 8))
        // String with Opaque layout (memref transition - January 2026)
        // After CCS memref transition, strings use TypeLayout.Opaque instead of FatPointer
        // Both layouts map to the same MLIR type: memref<?xi8>
        | TypeLayout.Opaque, Some NTUKind.NTUstring -> TMemRef (TInt (IntWidth 8))
        // SECOND: Name-based resolution for types without NTU metadata
        // Arrays have FatPointer layout but no specific NTUKind, handled here
        | _ ->
            match tycon.Name with
            // Byref types: all variants map to pointers
            | "byref" | "inref" | "outref" -> TIndex
            | "option" ->
                // Option is a DU with 2 cases (None, Some) - tag must be i8, not i1
                // DU tags are ALWAYS i8 (or i16 for >256 cases), never boolean
                match args with
                | [innerTy] ->
                    let innerMlir = mapNativeTypeForArch arch innerTy
                    let totalBytes = 1 + mlirTypeSize arch innerMlir
                    TMemRefStatic(totalBytes, TInt (IntWidth 8))
                | _ -> failwithf "option type requires exactly one type argument: %A" ty
            | "voption" ->
                // ValueOption is a DU with 2 cases (ValueNone, ValueSome) - tag must be i8
                match args with
                | [innerTy] ->
                    let innerMlir = mapNativeTypeForArch arch innerTy
                    let totalBytes = 1 + mlirTypeSize arch innerMlir
                    TMemRefStatic(totalBytes, TInt (IntWidth 8))
                | _ -> failwithf "voption type requires exactly one type argument: %A" ty
            | "Result" ->
                // Result<'T, 'E> is stored inline as a byte-level memref.
                // Allocate for the larger of the two case payloads so either case fits.
                // Layout: i8 tag + max(sizeof('T), sizeof('E)) bytes
                match args with
                | [okTy; errTy] ->
                    let okMlir  = mapNativeTypeForArch arch okTy
                    let errMlir = mapNativeTypeForArch arch errTy
                    let payloadBytes = max (mlirTypeSize arch okMlir) (mlirTypeSize arch errMlir)
                    TMemRefStatic(1 + payloadBytes, TInt (IntWidth 8))
                | _ -> failwithf "result type requires exactly two type arguments: %A" ty
            | "list" ->
                // PRD-13a: list<'T> is a pointer to cons cell (linked list)
                TIndex
            | "array" | "Array" ->
                // Array<T>: Following string migration pattern - use memref descriptor (ptr + len implicit)
                // Phase 2: memref<?xT> represents array with runtime length
                match args with
                | [elemTy] -> TMemRef (mapNativeTypeForArch arch elemTy)
                | _ -> failwithf "array<'T> requires exactly one type argument, got %d" args.Length
            | _ ->
                // Check FieldCount for record types
                if tycon.FieldCount > 0 then
                    match tyconLayout with
                    | TypeLayout.Inline (size, _align) when size > 0 ->
                        // Record with known layout — use computed size
                        TMemRefStatic (size, TInt (IntWidth 8))
                    | _ ->
                        // Record with Opaque/unknown layout (e.g. contains strings or other memref views)
                        // Estimate: field count × word size as upper bound
                        let estimatedSize = tycon.FieldCount * declaredPointerBytes arch
                        TMemRefStatic (estimatedSize, TInt (IntWidth 8))
                else
                    match tyconLayout with
                    | TypeLayout.Inline (size, align) when size > 8 ->
                        // DU layout: CCS provides size & align - type uses size, allocation uses align
                        // Heterogeneous struct → TMemRefStatic (size, TInt (IntWidth 8))
                        // This is the CORRECT portable representation for WASM and other backends
                        TMemRefStatic (size, TInt (IntWidth 8))
                    | TypeLayout.Inline (_size, _align) when tycon.CaseCount > 0 ->
                        // Small enum DU: abstract tag on FPGA, one-byte memref on CPU/MCU
                        enumTagRepresentation tycon.CaseCount
                    | TypeLayout.Inline (size, _align) when size > 0 ->
                        // C-style integer enum (CaseCount = 0, known size): map to integer of matching width
                        TInt (IntWidth (size * 8))
                    | TypeLayout.FatPointer ->
                        // FatPointer types should have been handled earlier by NTUKind or name
                        // Strings: TypeLayout.FatPointer + NTUKind.NTUstring → TMemRef (line 151)
                        // Arrays: Name match "array"|"Array" → TMemRef (line 207)
                        // If we reach here, check if it's a string by name (defensive)
                        if tycon.Name.ToLowerInvariant().Contains("string") then
                            // String without proper NTUKind - use memref but warn
                            printfn "WARNING: String type '%s' lacks NTUKind.NTUstring - fix CCS intrinsic definition" tycon.Name
                            TMemRef <| TInt (IntWidth 8)
                        else
                            // Unknown FatPointer type - fail loudly
                            failwithf "FatPointer type '%s' lacks proper NTUKind or name match - fix CCS metadata" tycon.Name
                    | TypeLayout.PlatformWord ->
                        // PlatformWord without NTUKind — the declared Register width
                        TInt (declaredWordWidth arch)
                    | TypeLayout.Opaque ->
                        failwithf "TApp with Opaque layout - CCS must resolve type '%s'" tycon.Name
                    | TypeLayout.Reference _ ->
                        failwithf "Reference type not yet implemented: %s" tycon.Name
                    | TypeLayout.NTUCompound n ->
                        // Arena<'lifetime> and similar compound types: N platform words
                        // Phase 2: Use memref array for multiple pointer fields (homogeneous)
                        if n = 1 then TIndex
                        else TMemRefStatic (n, TIndex)  // Array of N indices (portable)
                    | TypeLayout.Qualified _ ->
                        failwithf "Qualified layout should have been normalized before mapping: %s" tycon.Name
                    | TypeLayout.Inline _ ->
                        failwithf "Unknown inline type '%s' with no fields" tycon.Name

    match ty with
    | NativeType.TApp(tycon, args) -> mapTyCon tycon args
    // The carrier is read through the one carrier read; a carrier variable the checker left
    // unresolved is a checker failure surfaced here, never a width chosen by Composer.
    | NativeType.TNum(carrier, _) ->
        match CarrierRef.tryConstructor carrier with
        | Some tc -> mapTyCon tc []
        | None -> failwithf "mapNativeTypeForArch: unresolved carrier variable in numeric type '%s'; CCS must resolve it" (formatType ty)

    | NativeType.TFun _ ->
        // Closures: {codePtr: ptr, envPtr: ptr} - homogeneous, use memref array
        // Phase 2: Memref-backed pattern - array of 2 indices (portable, platform-sized)
        // Use TIndex (not TPtr) because index can be memref element type
        TMemRefStatic (2, TIndex)

    | NativeType.TTuple(elements, _) ->
        // Tuples are materialized as TStruct with positional field names on all platforms.
        let fields = elements |> List.mapi (fun i e -> sprintf "Item%d" (i + 1), mapNativeTypeForArch arch e)
        TStruct fields

    | NativeType.TVar tvar ->
        // Use Union-Find to resolve type variable chains
        match find tvar with
        | (_, Some boundTy) -> mapNativeTypeForArch arch boundTy
        | (root, None) ->
            // AX1001: Unbound type variable at MLIR generation time.
            // All type variables must be resolved by CCS/Baker before Alex runs.
            // Collect diagnostic and continue with TIndex so all errors are reported.
            typeMappingErrors.Add(sprintf "AX1001: Unbound type variable '%s' — CCS/Baker must resolve all type variables before MLIR generation" root.Name)
            TIndex

    | NativeType.TByref _ -> TIndex
    | NativeType.TNativePtr _ -> TIndex
    | NativeType.TForall(_, body) -> mapNativeTypeForArch arch body

    // PRD-14: Lazy<T> - FLAT CLOSURE: { computed: i1, value: T, code_ptr: ptr }
    // Captures are added dynamically at witness time, not in type mapping
    | NativeType.TLazy elemTy ->
        let elemMlir = mapNativeTypeForArch arch elemTy
        // Base layout: i1 + T + ptr - convert to byte-level memref
        let totalSize = mlirTypeSize arch (TInt (IntWidth 1)) + mlirTypeSize arch elemMlir + mlirTypeSize arch TIndex
        TMemRefStatic (totalSize, TInt (IntWidth 8))

    // PRD-15: Seq<T> - FLAT CLOSURE: { state: i32, current: T, moveNext_ptr: ptr }
    // Captures are added dynamically at witness time, not in type mapping
    | NativeType.TSeq elemTy ->
        let elemMlir = mapNativeTypeForArch arch elemTy
        // Base layout: i32 + T + ptr - convert to byte-level memref
        let totalSize = mlirTypeSize arch (TInt (IntWidth 32)) + mlirTypeSize arch elemMlir + mlirTypeSize arch TIndex
        TMemRefStatic (totalSize, TInt (IntWidth 8))

    // PRD-15/16: SeqEnumerator<T> - mutable iteration state over a seq
    // { seq_ptr: ptr, state: i32, current: T, hasValue: i1 }
    | NativeType.TSeqEnumerator elemTy ->
        let elemMlir = mapNativeTypeForArch arch elemTy
        // Layout: ptr + i32 + T + i1 - convert to byte-level memref
        let totalSize = mlirTypeSize arch TIndex + mlirTypeSize arch (TInt (IntWidth 32)) + mlirTypeSize arch elemMlir + mlirTypeSize arch (TInt (IntWidth 1))
        TMemRefStatic (totalSize, TInt (IntWidth 8))

    // PRD-13a: Immutable collection types - all are reference types (pointer to nodes)
    | NativeType.TList _ -> TIndex  // Pointer to cons cell
    | NativeType.TMap _ -> TIndex   // Pointer to tree root
    | NativeType.TSet _ -> TIndex   // Pointer to tree root

    // Named records are TApp with FieldCount > 0 - handled in TApp case above

    | NativeType.TUnion (tycon, cases) ->
        // DU layout: (tag, payload) where payload accommodates all cases
        // Tag type: i8 for ≤256 cases, i16 for more
        let tagType = if List.length cases <= 256 then TInt (IntWidth 8) else TInt (IntWidth 16)

        // Compute max payload size from case field types
        // Each case can have multiple fields (tuple payload) or single field
        let casePayloadTypes =
            cases
            |> List.map (fun case ->
                match case.Fields with
                | [] -> None  // No payload (e.g., None case)
                | [(_, ty)] -> Some (mapNativeTypeForArch arch ty)  // Single field
                | fields ->  // Multiple fields = tuple payload
                    let fieldTypes = fields |> List.map (fun (_, ty) -> mapNativeTypeForArch arch ty)
                    let totalBytes = fieldTypes |> List.sumBy (mlirTypeSize arch)
                    Some (TMemRefStatic(totalBytes, TInt (IntWidth 8))))

        // Find the "largest" payload type for union storage
        // For now, use the first non-None case's type (proper size comparison would need layout info)
        let payloadType =
            casePayloadTypes
            |> List.choose id
            |> List.tryHead
            |> Option.defaultValue (TInt (IntWidth 8))  // Empty union: tag-only storage

        // Convert to byte-level memref: tag + payload
        let totalSize = mlirTypeSize arch tagType + mlirTypeSize arch payloadType
        TMemRefStatic (totalSize, TInt (IntWidth 8))

    | NativeType.TAnon(fields, _) ->
        // Anonymous records - convert to byte-level memref
        let fieldTypes = fields |> List.map (fun (_, ty) -> mapNativeTypeForArch arch ty)
        let totalSize = fieldTypes |> List.sumBy (mlirTypeSize arch)
        TMemRefStatic (totalSize, TInt (IntWidth 8))

    | NativeType.TMeasure _ ->
        failwith "Measure type should have been stripped - this is an CCS issue"

    | NativeType.TError msg ->
        failwithf "NativeType.TError: %s" msg

// ═══════════════════════════════════════════════════════════════════════════
// FIELD OFFSET CALCULATION (for byte-level memref field access)
// ═══════════════════════════════════════════════════════════════════════════

/// Calculate byte offset for a field within a struct
/// Uses CCS-provided type structure and arch-aware size computation
let calculateFieldOffsetForArch (arch: Architecture) (nativeType: NativeType) (fieldIndex: int) : int =
    match nativeType with
    | NativeType.TTuple(elements, _) ->
        // Offset = sum of sizes of all fields before fieldIndex
        elements
        |> List.take fieldIndex
        |> List.map (mapNativeTypeForArch arch >> mlirTypeSize arch)
        |> List.sum

    | NativeType.TAnon(fields, _) ->
        // Offset = sum of sizes of all fields before fieldIndex
        fields
        |> List.take fieldIndex
        |> List.map (snd >> mapNativeTypeForArch arch >> mlirTypeSize arch)
        |> List.sum

    | NativeType.TLazy elemTy ->
        // Layout: evaluated (I1) | value (elemTy) | thunk (TIndex)
        match fieldIndex with
        | 0 -> 0  // evaluated flag
        | 1 -> mlirTypeSize arch (TInt (IntWidth 1))  // value after flag
        | 2 -> mlirTypeSize arch (TInt (IntWidth 1)) + mlirTypeSize arch (mapNativeTypeForArch arch elemTy)  // thunk after value
        | _ -> failwith $"Invalid field index {fieldIndex} for TLazy"

    | NativeType.TSeq elemTy ->
        // Layout: state (I32) | current (elemTy) | moveNext (TIndex)
        match fieldIndex with
        | 0 -> 0  // state
        | 1 -> mlirTypeSize arch (TInt (IntWidth 32))  // current after state
        | 2 -> mlirTypeSize arch (TInt (IntWidth 32)) + mlirTypeSize arch (mapNativeTypeForArch arch elemTy)  // moveNext after current
        | _ -> failwith $"Invalid field index {fieldIndex} for TSeq"

    | NativeType.TSeqEnumerator elemTy ->
        // Layout: source (TIndex) | index (I32) | current (elemTy) | hasValue (I1)
        match fieldIndex with
        | 0 -> 0  // source
        | 1 -> mlirTypeSize arch TIndex  // index after source
        | 2 -> mlirTypeSize arch TIndex + mlirTypeSize arch (TInt (IntWidth 32))  // current after index
        | 3 -> mlirTypeSize arch TIndex + mlirTypeSize arch (TInt (IntWidth 32)) + mlirTypeSize arch (mapNativeTypeForArch arch elemTy)  // hasValue after current
        | _ -> failwith $"Invalid field index {fieldIndex} for TSeqEnumerator"

    | NativeType.TUnion (_, cases) ->
        // Layout: tag | payload (max size of all cases)
        match fieldIndex with
        | 0 -> 0  // tag at offset 0
        | 1 ->
            // Payload offset = tag size
            let tagType = if List.length cases <= 256 then TInt (IntWidth 8) else TInt (IntWidth 16)
            mlirTypeSize arch tagType
        | _ -> failwith $"Invalid field index {fieldIndex} for TUnion"

    | NativeType.TApp ({ Name = name }, _) when name = "Closure" || name = "FunctionPointer" ->
        // Layout: codePtr (TIndex) | closure (TIndex)
        match fieldIndex with
        | 0 -> 0  // codePtr
        | 1 -> mlirTypeSize arch TIndex  // closure after codePtr
        | _ -> failwith $"Invalid field index {fieldIndex} for {name}"

    | NativeType.TFun _ ->
        // TFun is closures: {codePtr, envPtr} - same as Closure
        match fieldIndex with
        | 0 -> 0  // codePtr
        | 1 -> mlirTypeSize arch TIndex  // envPtr after codePtr
        | _ -> failwith $"Invalid field index {fieldIndex} for TFun"

    | _ -> failwith $"Cannot calculate field offset for type {nativeType} - not a struct type"

// ═══════════════════════════════════════════════════════════════════════════
// GRAPH-AWARE TYPE MAPPING (for record types)
// ═══════════════════════════════════════════════════════════════════════════

/// Case payloads of a user union type, from its TypeDef node (None for records, options,
/// abbreviations and primitives).
let private tryGetUnionCases (typeName: string) (graph: SemanticGraph) : (string * (string option * NativeType) list) list option =
    match SemanticGraph.recallType typeName graph with
    | Some nodeId ->
        match SemanticGraph.tryGetNode nodeId graph with
        | Some node ->
            match node.Kind with
            | Clef.Compiler.PSGSaturation.SemanticGraph.Types.SemanticKind.TypeDef (_, Clef.Compiler.PSGSaturation.SemanticGraph.Types.TypeDefKind.UnionDef cases, _) -> Some cases
            | _ -> None
        | None -> None
    | None -> None

/// Bytes one union case payload occupies at offset 1 of the union's memory (pDUCase stores it
/// there, pExtractDUPayload reads it back). A scalar is stored by value; every memory-backed
/// value (string, array, record, tuple, option, union) is stored as its memref descriptor, whose
/// size does not depend on the pointee, so a union that mentions itself through a payload needs
/// no recursion here.
let private unionPayloadSlotBytes (arch: Architecture) (graph: SemanticGraph) (ty: NativeType) : int =
    let wordBytes = declaredPointerBytes arch
    let descriptorBytes = 5 * wordBytes
    match ty with
    | NativeType.TApp (tycon, _) when tycon.FieldCount > 0 -> descriptorBytes
    | NativeType.TApp (tycon, _) when (SemanticGraph.tryGetRecordFields tycon.Name graph).IsSome -> descriptorBytes
    | NativeType.TApp (tycon, _) when (tryGetUnionCases tycon.Name graph).IsSome -> descriptorBytes
    | NativeType.TApp _ | NativeType.TNum _ ->
        let mapped = try Some (mapNativeTypeForArch arch ty) with _ -> None
        match mapped with
        | Some (TStruct _ | TMemRef _ | TMemRefStatic _ | TMemRefScalar _) | None -> descriptorBytes
        | Some other -> mlirTypeSize arch other
    | NativeType.TTuple _ -> descriptorBytes
    | NativeType.TFun _ -> 2 * wordBytes
    | _ -> descriptorBytes

/// CPU/MCU representation of a user union: a byte tag at offset 0 and the widest case payload
/// slot at offset 1. The front end's Inline layout counts a string or record payload as one
/// pointer, but the emitted store at offset 1 is the payload's memref descriptor (five words),
/// so the slot is sized from the emitted representation, not from the front end's estimate.
/// A union of nullary cases only is an enumeration tag.
let private unionRepresentation (arch: Architecture) (graph: SemanticGraph) (cases: (string * (string option * NativeType) list) list) : MLIRType =
    let maxPayload =
        cases
        |> List.map (fun (_, fields) -> fields |> List.sumBy (fun (_, fty) -> unionPayloadSlotBytes arch graph fty))
        |> List.max
    if maxPayload = 0 then enumTagRepresentation (List.length cases)
    else TMemRefStatic (1 + maxPayload, TInt (IntWidth 8))

/// Map a NativeType to MLIRType with architecture awareness, using graph lookup for record field types.
/// This is the principled approach per spec type-representation-architecture.md:
/// record fields are looked up via tryGetRecordFields, not guessed from layout.
/// RECURSIVE: nested record types also use graph lookup.
///
/// PRINCIPLED DESIGN (January 2026):
/// Takes Architecture explicitly to ensure PlatformWord types resolve correctly.
let rec mapNativeTypeWithGraphForArch (arch: Architecture) (graph: SemanticGraph) (ty: NativeType) : MLIRType =
    match ty with
    | NativeType.TApp(tycon, args) when tycon.FieldCount > 0 ->
        // Record type: look up field types from TypeDef → TStruct with named fields
        match SemanticGraph.tryGetRecordFields tycon.Name graph with
        | Some fields ->
            // Map each field type to MLIR RECURSIVELY (nested records also use graph lookup)
            let mlirFields = fields |> List.map (fun (name, fieldTy) -> (name, mapNativeTypeWithGraphForArch arch graph fieldTy))
            TStruct mlirFields
        | None ->
            // AX1002: Record type not found — CCS must create TypeDef nodes for all record types
            failwithf "AX1002: Record type '%s' not found in TypeDef nodes — CCS must create TypeDef for records" tycon.Name
    | NativeType.TApp(tycon, args) ->
        // Non-record TApp (FieldCount = 0) - but check if it might be a record by name lookup
        // This handles cases where FieldCount wasn't preserved in type extraction
        match SemanticGraph.tryGetRecordFields tycon.Name graph with
        | Some fields ->
            // Found record definition - use graph lookup → TStruct
            let mlirFields = fields |> List.map (fun (name, fieldTy) -> (name, mapNativeTypeWithGraphForArch arch graph fieldTy))
            TStruct mlirFields
        | None ->
            match tryGetUnionCases tycon.Name graph with
            | Some cases when not (List.isEmpty cases) && currentTargetPlatform <> Some FPGA ->
                // User union: sized from the emitted payload representation
                unionRepresentation arch graph cases
            | _ ->
            // Containers: the element/payload type must be the graph-aware PHYSICAL type, so an
            // array of records (or an option of a record) agrees with the record's own storage.
            match tycon.Name, args with
            | ("array" | "Array"), [elemTy] ->
                TMemRef (physicalStorageType arch (mapNativeTypeWithGraphForArch arch graph elemTy))
            | ("option" | "voption"), [innerTy] ->
                let innerMlir = physicalStorageType arch (mapNativeTypeWithGraphForArch arch graph innerTy)
                TMemRefStatic (1 + mlirTypeSize arch innerMlir, TInt (IntWidth 8))
            | "Result", [okTy; errTy] ->
                let okMlir = physicalStorageType arch (mapNativeTypeWithGraphForArch arch graph okTy)
                let errMlir = physicalStorageType arch (mapNativeTypeWithGraphForArch arch graph errTy)
                TMemRefStatic (1 + max (mlirTypeSize arch okMlir) (mlirTypeSize arch errMlir), TInt (IntWidth 8))
            | _ ->
                // Not a record - use standard mapping with architecture
                mapNativeTypeForArch arch ty
    | NativeType.TTuple(elements, _) ->
        // Tuples are materialized as TStruct with positional field names on all platforms.
        let fields = elements |> List.mapi (fun i e -> sprintf "Item%d" (i + 1), mapNativeTypeWithGraphForArch arch graph e)
        TStruct fields
    | NativeType.TAnon(fields, _) ->
        // Anonymous records → TStruct with named fields
        let mlirFields = fields |> List.map (fun (name, fieldTy) -> (name, mapNativeTypeWithGraphForArch arch graph fieldTy))
        TStruct mlirFields
    // PRD-14: Lazy<T> - FLAT CLOSURE, need recursive mapping in case T is a record
    | NativeType.TLazy elemTy ->
        let elemMlir = mapNativeTypeWithGraphForArch arch graph elemTy
        let totalBytes = 1 + mlirTypeSize arch elemMlir + mlirTypeSize arch TIndex
        TMemRefStatic(totalBytes, TInt (IntWidth 8))  // Flat: just code_ptr, captures added at witness
    | _ ->
        // Non-record types: use standard mapping with architecture
        mapNativeTypeForArch arch ty

/// Collect unique unbound type variables from a NativeType, in order of first appearance.
/// Follows Union-Find chains to find root TVars that are Unbound.
let rec private collectUnboundTVars (seen: Set<int>) (ty: NativeType) : (TypeParam * Set<int>) list =
    match ty with
    | NativeType.TVar tvar ->
        match find tvar with
        | (root, None) when not (Set.contains root.Id seen) ->
            [(root, Set.add root.Id seen)]
        | _ -> []
    | NativeType.TApp(_, args) ->
        args |> List.fold (fun acc arg ->
            let currentSeen = match acc with [] -> seen | _ -> snd (List.last acc)
            let results = collectUnboundTVars currentSeen arg
            acc @ results) []
    | NativeType.TFun(a, b) ->
        let aVars = collectUnboundTVars seen a
        let bSeen = match aVars with [] -> seen | _ -> snd (List.last aVars)
        aVars @ collectUnboundTVars bSeen b
    | NativeType.TTuple(elements, _) ->
        elements |> List.fold (fun acc elem ->
            let currentSeen = match acc with [] -> seen | _ -> snd (List.last acc)
            acc @ collectUnboundTVars currentSeen elem) []
    | _ -> []

/// When a TApp carries type arguments but the TypeDef's fields contain unbound TVars,
/// bind them in the Union-Find so downstream type mapping resolves correctly.
/// This compensates for clef not propagating type arguments to expression-level nodes.
let private bindTypeArgsToFieldTVars (fields: (string * NativeType) list) (args: NativeType list) =
    if args.IsEmpty then ()
    else
        let unboundTVars =
            fields
            |> List.collect (fun (_, fieldTy) -> collectUnboundTVars Set.empty fieldTy)
            |> List.map fst
        // Bind positionally: first unique unbound TVar → first type arg, etc.
        for i in 0 .. min (unboundTVars.Length - 1) (args.Length - 1) do
            unboundTVars.[i].Parent <- TypeParamState.Bound args.[i]

/// Eagerly walk a NativeType to bind type parameters from TApp type arguments.
/// Call this on function signatures BEFORE traversing body nodes, so inner expression
/// types resolve correctly through the Union-Find.
let rec resolveTypeParams (graph: SemanticGraph) (ty: NativeType) =
    match ty with
    | NativeType.TApp(tycon, args) ->
        if not args.IsEmpty then
            match SemanticGraph.tryGetRecordFields tycon.Name graph with
            | Some fields -> bindTypeArgsToFieldTVars fields args
            | None -> ()
        args |> List.iter (resolveTypeParams graph)
    | NativeType.TFun(a, b) ->
        resolveTypeParams graph a
        resolveTypeParams graph b
    | NativeType.TTuple(elements, _) ->
        elements |> List.iter (resolveTypeParams graph)
    | _ -> ()

/// Map leaf NativeType to MLIRType with platform-aware width resolution.
/// On FPGA, platform-word integers (NTUWidth.Resolved WidthDimension.Register) produce
/// IntWidth 0 — an abstract sentinel meaning "width from interval analysis, not architecture."
/// All other types (fixed-width integers, booleans, pointers, etc.) pass through unchanged.
let private mapLeafTypeForPlatform (platform: TargetPlatform) (arch: Architecture) (ty: NativeType) : MLIRType =
    match platform with
    | FPGA ->
        match ty with
        | NativeType.TApp _ | NativeType.TNum _ ->
            // The numeric carrier's kind is read through the one carrier read (Types.tryGetNTUKind).
            match Types.tryGetNTUKind ty with
            | Some (NTUKind.NTUint (NTUWidth.Resolved WidthDimension.Register))
            | Some (NTUKind.NTUuint (NTUWidth.Resolved WidthDimension.Register)) ->
                TInt (IntWidth 0)  // Abstract: width from interval analysis
            | _ -> mapNativeTypeForArch arch ty
        | _ -> mapNativeTypeForArch arch ty
    | _ -> mapNativeTypeForArch arch ty

/// Platform-aware type mapping — the canonical entry point for target-dependent code.
/// On FPGA, TTuple maps to TStruct with positional field names (first-class value).
/// On CPU, TTuple maps to TMemRefStatic (byte blob for memory layout).
/// On FPGA, platform-word integers produce IntWidth 0 (abstract — resolved by interval analysis).
/// Recursive: nested tuples within tuple elements also get the platform treatment.
let rec mapNativeTypeForTarget (platform: TargetPlatform) (arch: Architecture) (graph: SemanticGraph) (ty: NativeType) : MLIRType =
    let recurse = mapNativeTypeForTarget platform arch graph
    match ty with
    | NativeType.TApp(tycon, args) when tycon.FieldCount > 0 ->
        // Record type: look up field types from TypeDef → TStruct with named fields
        match SemanticGraph.tryGetRecordFields tycon.Name graph with
        | Some fields ->
            // Bind type arguments to unbound TVars in fields (compensates for clef gap)
            bindTypeArgsToFieldTVars fields args
            let mlirFields = fields |> List.map (fun (name, fieldTy) -> (name, recurse fieldTy))
            TStruct mlirFields
        | None ->
            failwithf "Record type '%s' not found in TypeDef nodes - CCS must create TypeDef for records" tycon.Name
    | NativeType.TApp(tycon, args) ->
        // Non-record TApp (FieldCount = 0) - check if it might be a record by name lookup
        match SemanticGraph.tryGetRecordFields tycon.Name graph with
        | Some fields ->
            bindTypeArgsToFieldTVars fields args
            let mlirFields = fields |> List.map (fun (name, fieldTy) -> (name, recurse fieldTy))
            TStruct mlirFields
        | None ->
            // FPGA: option/voption → hw.struct with tag + value
            match platform with
            | FPGA ->
                match tycon.Name with
                | "option" | "voption" ->
                    match args with
                    | [innerTy] ->
                        let innerMlir = recurse innerTy
                        TStruct [("tag", TInt (IntWidth 1)); ("value", innerMlir)]
                    | _ -> mapLeafTypeForPlatform platform arch ty
                | _ -> mapLeafTypeForPlatform platform arch ty
            | _ ->
                match tryGetUnionCases tycon.Name graph with
                | Some cases when not (List.isEmpty cases) ->
                    // User union: sized from the emitted payload representation
                    unionRepresentation arch graph cases
                | _ ->
                // CPU/MCU containers: element/payload types are the graph-aware PHYSICAL types,
                // so an array of records or an option of a record agrees with the record's storage.
                match tycon.Name, args with
                | ("array" | "Array"), [elemTy] ->
                    TMemRef (physicalStorageType arch (recurse elemTy))
                | ("option" | "voption"), [innerTy] ->
                    let innerMlir = physicalStorageType arch (recurse innerTy)
                    TMemRefStatic (1 + mlirTypeSize arch innerMlir, TInt (IntWidth 8))
                | "Result", [okTy; errTy] ->
                    let okMlir = physicalStorageType arch (recurse okTy)
                    let errMlir = physicalStorageType arch (recurse errTy)
                    TMemRefStatic (1 + max (mlirTypeSize arch okMlir) (mlirTypeSize arch errMlir), TInt (IntWidth 8))
                | _ -> mapLeafTypeForPlatform platform arch ty
    | NativeType.TTuple(elements, _) ->
        // Tuples are materialized as TStruct with positional field names on all platforms.
        // CPU uses memref alloca + byte-offset stores; FPGA uses hw.struct_create.
        // Both need TStruct for field-level access (pRecordFieldGet, TupleGet extraction).
        let fields = elements |> List.mapi (fun i e -> sprintf "Item%d" (i + 1), recurse e)
        TStruct fields
    | NativeType.TAnon(fields, _) ->
        // Anonymous records → TStruct with named fields
        let mlirFields = fields |> List.map (fun (name, fieldTy) -> (name, recurse fieldTy))
        TStruct mlirFields
    | NativeType.TLazy elemTy ->
        // Lazy<T> - flat closure
        let elemMlir = recurse elemTy
        let totalBytes = 1 + mlirTypeSize arch elemMlir + mlirTypeSize arch TIndex
        TMemRefStatic(totalBytes, TInt (IntWidth 8))
    | NativeType.TVar tvar ->
        // Resolve type variable through Union-Find and recurse through target-aware mapper
        match find tvar with
        | (_, Some boundTy) -> recurse boundTy
        | (root, None) ->
            // AX1001: Unbound type variable at MLIR generation time.
            // All type variables must be resolved by CCS/Baker before Alex runs.
            // Collect diagnostic and continue with TIndex so all errors are reported.
            typeMappingErrors.Add(sprintf "AX1001: Unbound type variable '%s' — CCS/Baker must resolve all type variables before MLIR generation" root.Name)
            TIndex
    | _ ->
        // Leaf types: platform-aware mapping (FPGA: IntWidth 0 for platform-word integers)
        mapLeafTypeForPlatform platform arch ty
