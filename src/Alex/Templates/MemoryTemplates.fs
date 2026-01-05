/// Memory Templates - LLVM dialect memory operations
///
/// Templates for LLVM dialect memory operations:
/// - Load/Store
/// - Alloca
/// - GEP (pointer arithmetic)
/// - Pointer conversions
/// - Memcpy/Memset intrinsics
module Alex.Templates.MemoryTemplates

open Alex.Templates.TemplateTypes

// ═══════════════════════════════════════════════════════════════════════════
// LOAD / STORE
// ═══════════════════════════════════════════════════════════════════════════

/// Load from pointer: llvm.load
let load = simple "llvm" "memory" (fun (p: LoadParams) ->
    sprintf "%s = llvm.load %s : !llvm.ptr -> %s" p.Result p.Pointer p.Type)

/// Store to pointer: llvm.store
let store = simple "llvm" "memory" (fun (p: StoreParams) ->
    sprintf "llvm.store %s, %s : %s, !llvm.ptr" p.Value p.Pointer p.Type)

// ═══════════════════════════════════════════════════════════════════════════
// ALLOCA
// ═══════════════════════════════════════════════════════════════════════════

/// Stack allocation: llvm.alloca
let alloca = simple "llvm" "memory" (fun (p: AllocaParams) ->
    sprintf "%s = llvm.alloca %s x %s : (i64) -> !llvm.ptr" p.Result p.Count p.ElementType)

/// Single element alloca (count = 1)
let allocaSingle = simple "llvm" "memory" (fun (result: string, elemType: string) ->
    sprintf "%s = llvm.alloca 1 x %s : (i64) -> !llvm.ptr" result elemType)

// ═══════════════════════════════════════════════════════════════════════════
// GEP (POINTER ARITHMETIC)
// ═══════════════════════════════════════════════════════════════════════════

/// GEP with i32 offset: llvm.getelementptr
let gepI32 = simple "llvm" "memory" (fun (p: GepParams) ->
    sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i32) -> !llvm.ptr, %s" 
        p.Result p.Base p.Offset p.ElementType)

/// GEP with i64 offset: llvm.getelementptr
let gepI64 = simple "llvm" "memory" (fun (p: GepParams) ->
    sprintf "%s = llvm.getelementptr %s[%s] : (!llvm.ptr, i64) -> !llvm.ptr, %s"
        p.Result p.Base p.Offset p.ElementType)

// ═══════════════════════════════════════════════════════════════════════════
// POINTER CONVERSIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Pointer to integer: llvm.ptrtoint
let ptrToInt = simple "llvm" "conversion" (fun (result: string, ptr: string, intType: string) ->
    sprintf "%s = llvm.ptrtoint %s : !llvm.ptr to %s" result ptr intType)

/// Integer to pointer: llvm.inttoptr
let intToPtr = simple "llvm" "conversion" (fun (result: string, int: string, intType: string) ->
    sprintf "%s = llvm.inttoptr %s : %s to !llvm.ptr" result int intType)

/// Bitcast (type punning) - typically for pointer types
let bitcast = simple "llvm" "conversion" (fun (result: string, operand: string, fromType: string, toType: string) ->
    sprintf "%s = llvm.bitcast %s : %s to %s" result operand fromType toType)

// ═══════════════════════════════════════════════════════════════════════════
// MEMORY INTRINSICS
// ═══════════════════════════════════════════════════════════════════════════

/// Memcpy intrinsic: llvm.intr.memcpy
/// dest, src, len - copies len bytes from src to dest
let memcpy = simple "llvm" "intrinsic" (fun (dest: string, src: string, len: string) ->
    sprintf "\"llvm.intr.memcpy\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()"
        dest src len)

/// Memset intrinsic: llvm.intr.memset
/// dest, value, len - fills len bytes with value
let memset = simple "llvm" "intrinsic" (fun (dest: string, value: string, len: string) ->
    sprintf "\"llvm.intr.memset\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, i8, i64) -> ()"
        dest value len)

/// Memmove intrinsic: llvm.intr.memmove
/// dest, src, len - copies len bytes, handles overlapping regions
let memmove = simple "llvm" "intrinsic" (fun (dest: string, src: string, len: string) ->
    sprintf "\"llvm.intr.memmove\"(%s, %s, %s) <{isVolatile = false}> : (!llvm.ptr, !llvm.ptr, i64) -> ()"
        dest src len)

// ═══════════════════════════════════════════════════════════════════════════
// STRUCT OPERATIONS (for fat pointers, etc.)
// ═══════════════════════════════════════════════════════════════════════════

/// Undef value (for struct construction)
let undef = simple "llvm" "aggregate" (fun (result: string, ty: string) ->
    sprintf "%s = llvm.mlir.undef : %s" result ty)

/// Insert value into struct
let insertValue = simple "llvm" "aggregate" (fun (result: string, aggregate: string, value: string, aggType: string, index: int) ->
    sprintf "%s = llvm.insertvalue %s, %s[%d] : %s" result value aggregate index aggType)

/// Extract value from struct
let extractValue = simple "llvm" "aggregate" (fun (result: string, aggregate: string, aggType: string, index: int) ->
    sprintf "%s = llvm.extractvalue %s[%d] : %s" result aggregate index aggType)

// ═══════════════════════════════════════════════════════════════════════════
// NATIVE STRING (FAT POINTER) HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// The MLIR type for native strings (fat pointer)
let nativeStrType = "!llvm.struct<(ptr, i64)>"

/// Construct a native string from pointer and length
/// Returns the sequence of operations needed
let constructNativeStr (resultSSA: string) (ptrSSA: string) (lenSSA: string) : string list =
    let undefSSA = resultSSA + "_undef"
    let withPtrSSA = resultSSA + "_ptr"
    [
        sprintf "%s = llvm.mlir.undef : %s" undefSSA nativeStrType
        sprintf "%s = llvm.insertvalue %s, %s[0] : %s" withPtrSSA ptrSSA undefSSA nativeStrType
        sprintf "%s = llvm.insertvalue %s, %s[1] : %s" resultSSA lenSSA withPtrSSA nativeStrType
    ]

/// Extract pointer from native string
let extractStrPointer (resultSSA: string) (strSSA: string) : string =
    sprintf "%s = llvm.extractvalue %s[0] : %s" resultSSA strSSA nativeStrType

/// Extract length from native string
let extractStrLength (resultSSA: string) (strSSA: string) : string =
    sprintf "%s = llvm.extractvalue %s[1] : %s" resultSSA strSSA nativeStrType

// ═══════════════════════════════════════════════════════════════════════════
// ADDRESS SPACE OPERATIONS (for future GPU/accelerator support)
// ═══════════════════════════════════════════════════════════════════════════

/// Address space cast
let addrSpaceCast = simple "llvm" "memory" (fun (result: string, ptr: string, fromAS: int, toAS: int) ->
    sprintf "%s = llvm.addrspacecast %s : !llvm.ptr<%d> to !llvm.ptr<%d>" result ptr fromAS toAS)
