/// Alex MLIR Type System
///
/// Structured MLIR type representation for Alex code generation.
/// This is the canonical type system used throughout Alex.
module Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER AND FLOAT WIDTHS
// ═══════════════════════════════════════════════════════════════════════════

/// Integer bit width — arbitrary precision.
/// Width is a number, not an enum of CPU-blessed sizes.
/// MLIR natively supports iN for any N. CIRCT hw dialect supports arbitrary widths.
/// On FPGA, the width is exactly what the design needs — no more, no less.
[<Struct>]
type IntWidth = IntWidth of bits: int

/// Extract the bit count from an IntWidth
let inline intWidthBits (IntWidth bits) = bits

/// Size in bytes for a given bit width (ceiling division by 8)
let intWidthBytes (IntWidth bits) = (bits + 7) / 8

/// Floating-point widths supported by MLIR
type FloatWidth =
    | F32   // Single-precision
    | F64   // Double-precision

// ═══════════════════════════════════════════════════════════════════════════
// MLIR TYPE SYSTEM
// ═══════════════════════════════════════════════════════════════════════════

/// The settled byte layout of a struct on a core, read from the graph's settled layouts
/// (`SemanticGraph.Layouts`, Dimensional_Range_Design.md ruling 2): one offset per field in field
/// order, the size and the alignment. `None` on fabric, where a struct is an `hw.struct` value
/// with no byte layout. Composer reads these and computes none of them.
type StructBytes = { Offsets: int list; Size: int; Align: int }

/// Structured MLIR type representation
type MLIRType =
    | TInt of IntWidth
    | TFloat of FloatWidth
    | TFunc of MLIRType list * MLIRType list // Code value's function type (args, results); environment is separate
    | TMemRef of MLIRType                   // MemRef type (dynamic 1D)
    | TMemRefStatic of int * MLIRType       // Static-sized MemRef type (1D with known size)
    | TMemRefScalar of MLIRType             // Scalar MemRef type (0D)
    | TVector of int * MLIRType             // Vector type (SIMD)
    | TIndex                                // Index type
    | TUnit                                 // Unit type (represented as i32 0)
    | TVoid                                 // Foreign ABI: no returned value
    | TStruct of (string * MLIRType) list * StructBytes option   // Named struct type (record fields) and, on a core, its settled bytes
    | TSeqClock                             // CIRCT !seq.clock type (clock signal for registers)
    | TTag of int                           // DU tag discriminant (case count). Platform elision decides concrete width.
    | TError of string                      // Error type

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Target operating system family
type OSFamily =
    | Linux
    | Windows
    | MacOS
    | FreeBSD

/// The instruction set, for the OS and syscall selection.
type Isa =
    | X86_64
    | ARM64
    | ARM32_Thumb
    | RISCV64
    | RISCV32
    | WASM32

/// The target architecture as Composer reads it (plan D8, L-10; Dimensional_Range_Design.md
/// §8.3): the instruction set, and the width dimensions the platform description declares,
/// `Register` and `Pointer`, read once from the CCS context (MLIRGeneration.architectureOf). There
/// is no architecture table: a width the description does not declare carries CCS8203's text,
/// and the site that needs it fails with that text, never with a number of its own. An FPGA
/// description declares neither, and no site on the fabric leg reads them.
type Architecture = {
    Isa: Isa
    Register: Result<int, string>
    Pointer: Result<int, string>
}

/// The platform's word: the declared Register width as an MLIR integer width.
let declaredWordWidth (arch: Architecture) : IntWidth =
    match arch.Register with
    | Ok bits -> IntWidth bits
    | Error message -> failwith message

/// The declared Pointer width in bytes: the size of an index, and the unit of a memref
/// descriptor (five words). Function code values have no implicit data-storage layout.
let declaredPointerBytes (arch: Architecture) : int =
    match arch.Pointer with
    | Ok bits -> (bits + 7) / 8
    | Error message -> failwith message

/// The size in bytes of a value as it is stored: a read, never a computation (§8.3, one size
/// model). A scalar is the bytes of its selected width; every pointer-sized type (an index; a
/// rank-1 memref descriptor, {allocPtr, alignPtr, offset, size, stride}, five words)
/// is the declared Pointer width times its words, and `pointer` is `Error`
/// where no declaration is in hand (serialization), so that a pointer-sized value reaching such a
/// path is a loud defect and never a silent eight bytes; a struct is its settled size, read from
/// the layout CCS settled at saturation, and a struct with none (fabric, an opaque field) is a
/// stop. A width the range was to give (`IntWidth 0`) has no size and is a stop naming it.
let rec mlirTypeSizeWith (pointer: Result<int, string>) (ty: MLIRType) : int =
    let pointerBytes () =
        match pointer with
        | Ok bits -> (bits + 7) / 8
        | Error message -> failwith message
    match ty with
    | TInt (IntWidth 0) -> failwith "mlirTypeSize: an integer whose width the node's range was to give reached a size read unnarrowed"
    | TInt w -> intWidthBytes w
    | TFloat F32 -> 4 | TFloat F64 -> 8
    | TFunc _ -> failwith "mlirTypeSize: a function code SSA has no admitted data-storage layout; settle callable storage before sizing it"
    | TMemRef _ | TMemRefStatic _ | TMemRefScalar _ -> 5 * pointerBytes ()
    | TVector (_, elemTy) -> mlirTypeSizeWith pointer elemTy
    | TIndex -> pointerBytes ()
    | TStruct (_, Some bytes) -> bytes.Size
    | TStruct (fields, None) ->
        failwithf "mlirTypeSize: the struct {%s} has no settled layout to read its size from (a fabric struct, or a field the placement could not settle)"
            (fields |> List.map fst |> String.concat ", ")
    | TSeqClock -> 1
    | TTag _ -> 1  // Tag is at least 1 byte; platform elision determines actual width
    | TUnit | TVoid -> 0 | TError _ -> 0

/// The settled byte offset of a struct's field, read from the layout CCS settled; a struct with
/// no settled layout, or a field outside it, is a stop.
let structFieldOffset (ty: MLIRType) (fieldIndex: int) : int =
    match ty with
    | TStruct (_, Some bytes) ->
        match List.tryItem fieldIndex bytes.Offsets with
        | Some offset -> offset
        | None -> failwithf "structFieldOffset: field %d is outside the settled layout of %d fields" fieldIndex bytes.Offsets.Length
    | TStruct (fields, None) ->
        failwithf "structFieldOffset: the struct {%s} has no settled layout to read an offset from" (fields |> List.map fst |> String.concat ", ")
    | other -> failwithf "structFieldOffset: %A is not a struct" other

/// The size model read through the architecture's declared Pointer width.
let mlirTypeSize (arch: Architecture) (ty: MLIRType) : int =
    mlirTypeSizeWith arch.Pointer ty

// ═══════════════════════════════════════════════════════════════════════════
// SSA VALUES AND BLOCK REFERENCES
// ═══════════════════════════════════════════════════════════════════════════

/// SSA value reference - the currency of MLIR operations
/// V = value from computation, Arg = function/block argument
/// A value name. `V (node, k)` is the k-th value emission names for a graph node, a pure
/// derivation from the node's identity (Alex.Traversal.Values); `Arg n` a block argument. No
/// pass assigns names and no witness holds a counter: the graph identifies the node, the witness
/// numbers what it emits for it.
[<Struct>]
type SSA =
    | V of node: int * ordinal: int   // %v<node>_<k>
    | CallableAlternative of node: int * alternative: int // Source-settled mutable read dispatch arm
    | Arg of int                      // %arg0, %arg1, ...

/// Block label reference
[<Struct>]
type BlockRef = BlockRef of string

/// A typed SSA value (value + its type)
type Val = {
    SSA: SSA
    Type: MLIRType
}

/// Create a typed value
let val' ssa ty = { SSA = ssa; Type = ty }

// ═══════════════════════════════════════════════════════════════════════════
// COMPARISON PREDICATES
// ═══════════════════════════════════════════════════════════════════════════

/// Integer comparison predicates (for arith.cmpi)
type ICmpPred =
    | Eq | Ne                       // Equal, Not equal
    | Slt | Sle | Sgt | Sge         // Signed comparisons
    | Ult | Ule | Ugt | Uge         // Unsigned comparisons

/// Float comparison predicates (for arith.cmpf)
type FCmpPred =
    // Ordered comparisons (return false if either operand is NaN)
    | OEq | ONe | OLt | OLe | OGt | OGe
    // Unordered comparisons (return true if either operand is NaN)
    | UEq | UNe | ULt | ULe | UGt | UGe
    // Special predicates
    | Ord   // Ordered (neither operand is NaN)
    | Uno   // Unordered (either operand is NaN)
    // False/True (always)
    | AlwaysFalse | AlwaysTrue

/// Index comparison predicates (for index.cmp)
type IndexCmpPred =
    | Eq | Ne                       // Equal, Not equal
    | Slt | Sle | Sgt | Sge         // Signed comparisons
    | Ult | Ule | Ugt | Uge         // Unsigned comparisons

// ═══════════════════════════════════════════════════════════════════════════
// MEMORY ORDERING AND VISIBILITY
// ═══════════════════════════════════════════════════════════════════════════

/// Function visibility for external linking
type FuncVisibility =
    | Public                        // Visible to all modules
    | Private                       // Visible only within module

// ═══════════════════════════════════════════════════════════════════════════
// MLIR OPERATIONS (for type-safe MLIR construction)
// ═══════════════════════════════════════════════════════════════════════════

/// Dimension parameter for memref.subview: compile-time constant or runtime SSA value
[<RequireQualifiedAccess>]
type SubViewParam =
    | Static of int64
    | Dynamic of SSA

/// MemRef dialect operations (standard MLIR memory operations)
type MemRefOp =
    | Load of SSA * SSA * SSA list * MLIRType * MLIRType               // result, memref, indices, elemType, memrefType
    | Store of SSA * SSA * SSA list * MLIRType * MLIRType              // value, memref, indices, elemType, memrefType
    | LoadAligned of SSA * SSA * SSA list * MLIRType * MLIRType * int // explicit byte alignment, including packed fields
    | StoreAligned of SSA * SSA * SSA list * MLIRType * MLIRType * int
    /// Standard contiguous representation copy. This confers no source-value
    /// initialization or current-read evidence on the destination.
    | Copy of source: SSA * destination: SSA * sourceType: MLIRType * destinationType: MLIRType
    | Alloca of SSA * MLIRType * int option                            // result, memrefType, alignment (stack, compile-time size)
    | Alloc of SSA * SSA * MLIRType                                    // result, sizeSSA, elementType (heap, runtime size)
    | Dealloc of SSA * MLIRType                                        // owned heap memref, released after its last use
    | AllocStatic of SSA * MLIRType * int option                        // result, memrefType, alignment (heap, compile-time size)
    | SubView of SSA * SSA * SSA list * MLIRType                       // result, source, offsets, resultType (legacy element access)
    | SubViewSlice of SSA * SSA * SSA list * SubViewParam list * SubViewParam list * MLIRType  // result, source, offsets, sizes, strides, sourceType (proper MLIR 3-group, strided result)
    | SubViewCopy of SSA * SSA * SSA list * SubViewParam list * SubViewParam list * SSA * MLIRType  // result, source, offsets, sizes, strides, sizeIndexSSA, sourceType (subview → alloc → copy: fresh contiguous buffer)
    | ExtractBasePtr of SSA * SSA * MLIRType                           // result, memref, memrefType → !llvm.ptr (for FFI)
    | GetGlobal of SSA * string * MLIRType                             // result, globalName, memrefType
    | Dim of SSA * SSA * SSA * MLIRType                                // result, memref, dimIndex, memrefType (returns index)
    | Cast of SSA * SSA * MLIRType * MLIRType                          // result, source, srcType, destType (memref type cast)
    | ReinterpretCast of SSA * SSA * int * int * MLIRType * MLIRType   // result, source, byteOffset, size, srcType, destType
    | ReinterpretCastDynamic of SSA * SSA * int * SSA * MLIRType * MLIRType  // result, source, offset, sizeSSA, srcType, destType (dynamic size for memref reconstruction)
    | View of SSA * SSA * SSA * MLIRType * MLIRType                   // result, source, offsetSSA, srcType, destType (different element type: byte buffer → typed view)
    | IndexToMemRef of SSA * SSA * MLIRType                            // result, sourceIndex, destMemRefType (internal index→memref seam: TNativePtr-as-index at closure/seq/FFI boundaries)
    | MemRefToIndex of SSA * SSA * MLIRType                            // result, sourceMemRef, srcMemRefType (internal memref→index seam at FFI boundaries)

/// Arithmetic dialect operations
type ArithOp =
    // Constants
    | ConstI of SSA * int64 * MLIRType                      // result, value, type
    | ConstF of SSA * float * MLIRType                      // result, value, type
    // Integer arithmetic
    | AddI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | SubI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | MulI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | DivSI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (signed)
    | DivUI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (unsigned)
    | RemSI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (signed)
    | RemUI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (unsigned)
    | CmpI of SSA * ICmpPred * SSA * SSA * MLIRType         // result, predicate, lhs, rhs, type
    // Float arithmetic
    | AddF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | SubF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | NegF of SSA * SSA * MLIRType                          // result, operand, type
    | MulF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | DivF of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | CmpF of SSA * FCmpPred * SSA * SSA * MLIRType         // result, predicate, lhs, rhs, type
    // Type conversions
    | ExtSI of SSA * SSA * MLIRType * MLIRType              // result, value, srcType, destType
    | ExtUI of SSA * SSA * MLIRType * MLIRType              // result, value, srcType, destType
    | ExtF of SSA * SSA * MLIRType * MLIRType
    | TruncF of SSA * SSA * MLIRType * MLIRType
    | TruncI of SSA * SSA * MLIRType * MLIRType             // result, value, srcType, destType
    | SIToFP of SSA * SSA * MLIRType * MLIRType             // result, value, srcType, destType
    | FPToSI of SSA * SSA * MLIRType * MLIRType             // result, value, srcType, destType
    // Bitwise operations (migrated from LLVM dialect)
    | AndI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | OrI of SSA * SSA * SSA * MLIRType                     // result, lhs, rhs, type
    | XorI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | ShLI of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type (shift left)
    | ShRUI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (logical shift right)
    | ShRSI of SSA * SSA * SSA * MLIRType                   // result, lhs, rhs, type (arithmetic shift right)
    // Other
    | Select of SSA * SSA * SSA * SSA * MLIRType            // result, cond, trueVal, falseVal, type

/// Index dialect operations
type IndexOp =
    | IndexConst of SSA * int64                       // result, value
    | IndexBoolConst of SSA * bool                    // result, value
    | IndexAdd of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexSub of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexMul of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexDivS of SSA * SSA * SSA                    // result, lhs, rhs (signed)
    | IndexDivU of SSA * SSA * SSA                    // result, lhs, rhs (unsigned)
    | IndexCeilDivS of SSA * SSA * SSA                // result, lhs, rhs (signed ceiling)
    | IndexCeilDivU of SSA * SSA * SSA                // result, lhs, rhs (unsigned ceiling)
    | IndexFloorDivS of SSA * SSA * SSA               // result, lhs, rhs (signed floor)
    | IndexRemS of SSA * SSA * SSA                    // result, lhs, rhs (signed)
    | IndexRemU of SSA * SSA * SSA                    // result, lhs, rhs (unsigned)
    | IndexMaxS of SSA * SSA * SSA                    // result, lhs, rhs (signed)
    | IndexMaxU of SSA * SSA * SSA                    // result, lhs, rhs (unsigned)
    | IndexMinS of SSA * SSA * SSA                    // result, lhs, rhs (signed)
    | IndexMinU of SSA * SSA * SSA                    // result, lhs, rhs (unsigned)
    | IndexShl of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexShrS of SSA * SSA * SSA                    // result, lhs, rhs (signed)
    | IndexShrU of SSA * SSA * SSA                    // result, lhs, rhs (unsigned)
    | IndexAnd of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexOr of SSA * SSA * SSA                      // result, lhs, rhs
    | IndexXor of SSA * SSA * SSA                     // result, lhs, rhs
    | IndexCmp of SSA * IndexCmpPred * SSA * SSA      // result, predicate, lhs, rhs
    | IndexCastS of SSA * SSA * MLIRType * MLIRType    // result, operand, srcType, destType (signed)
    | IndexCastU of SSA * SSA * MLIRType * MLIRType    // result, operand, srcType, destType (unsigned)
    | IndexSizeOf of SSA * MLIRType                   // result, type

/// CIRCT Combinational Logic Dialect (pure, stateless operations)
/// Maps to hw/comb dialect — synthesizable combinational logic for FPGA targets
type CombOp =
    // Integer arithmetic (combinational — no clock, no state)
    | CombAdd of SSA * SSA * SSA * MLIRType                    // result, lhs, rhs, type
    | CombSub of SSA * SSA * SSA * MLIRType
    | CombMul of SSA * SSA * SSA * MLIRType
    | CombDivS of SSA * SSA * SSA * MLIRType                   // signed
    | CombDivU of SSA * SSA * SSA * MLIRType                   // unsigned
    | CombMod of SSA * SSA * SSA * MLIRType                    // signed (comb.mods)
    | CombModU of SSA * SSA * SSA * MLIRType                   // unsigned (comb.modu)
    // Bitwise
    | CombAnd of SSA * SSA * SSA * MLIRType
    | CombOr of SSA * SSA * SSA * MLIRType
    | CombXor of SSA * SSA * SSA * MLIRType
    | CombShl of SSA * SSA * SSA * MLIRType
    | CombShrU of SSA * SSA * SSA * MLIRType
    | CombShrS of SSA * SSA * SSA * MLIRType
    // Comparison
    | CombICmp of SSA * ICmpPred * SSA * SSA * MLIRType        // result, pred, lhs, rhs, type
    // Multiplexer (if/else → combinational select)
    | CombMux of SSA * SSA * SSA * SSA * MLIRType              // result, cond, trueVal, falseVal, type

/// CIRCT Sequential Logic Dialect (clocked state, registers)
/// Maps to seq dialect — flip-flops and clocked elements for FPGA targets
type SeqOp =
    | SeqCompreg of SSA * SSA * SSA * (SSA * SSA) option * MLIRType    // result, input, clk, (resetSignal, resetValue) option, type

/// SMT dialect types (upstream `smt` dialect, adopted from CIRCT).
/// Used ONLY inside verification modules — never mixes with program value types.
type SMTType =
    | SMTBool
    | SMTInt
    | SMTBV of int          // bit-vector width

/// Integer comparison predicates for smt.int.cmp
type SMTCmpPred =
    | SmtLt | SmtLe | SmtGt | SmtGe

/// Top-level MLIR operation (all dialects)
/// Single-phase execution with nested accumulators - no scope markers needed
type MLIROp =
    /// Typed MMIO only becomes LLVM pointers at serialization. The address is
    /// an opaque handle (index); no generic source dereference is introduced.
    | MmioLoad of result: SSA * address: SSA * integerAddress: SSA * pointer: SSA * bits: int
    | MmioStore of value: SSA * address: SSA * integerAddress: SSA * pointer: SSA * bits: int
    | MemRefOp of MemRefOp
    | ArithOp of ArithOp
    | SCFOp of SCFOp
    /// A required runtime boundary check; failure terminates before the foreign access.
    | Assert of condition: SSA * message: string
    /// MCU functions terminate on failure; unwinding across the exception
    /// adapter is unsupported. Applied only to definitions, never foreign declarations.
    | NoUnwindFunction of FuncOp
    | FuncOp of FuncOp
    | IndexOp of IndexOp
    | Block of string * MLIROp list                                 // label, ops
    | Region of MLIROp list                                         // blocks
    // Module-level declarations (backend-agnostic)
    | GlobalString of name: string * content: string * byteLength: int * obligations: string list  // obligations: anchor names of the obligations constraining this storage (PHG 2.4b), reified as {clef.obligations = [...]}
    /// One immutable allocation whose bytes and alignment were settled in the PSG.
    | GlobalBytePool of name: string * bytes: byte list * alignment: int * obligations: string list
    /// Exact source inventory accompanies writable storage. None identifies a
    /// legacy unowned declaration; it cannot pass storage correspondence.
    | GlobalMemref of string * MLIRType * Clef.Compiler.PSGSaturation.SemanticGraph.Types.ProgramStorageEntry option
    // CIRCT hardware dialects (FPGA targets)
    | CombOp of CombOp
    | HWOp of HWOp
    | SeqOp of SeqOp
    // SMT dialect (verification modules — proof obligations as IR)
    | SMTOp of SMTOp
    // Raw MLIR text (dialect-opaque: AIE, GPU, etc.)
    | RawMLIR of string

/// Structured Control Flow (SCF) dialect operations
and SCFOp =
    | If of SSA * MLIROp list * MLIROp list option * (SSA * MLIRType) option  // cond, thenOps, elseOps, result (None = void)
    | IndexSwitch of selector: SSA * cases: (int64 * MLIROp list) list * defaultBody: MLIROp list * results: (SSA * MLIRType) list
    | While of MLIROp list * MLIROp list                      // condOps, bodyOps
    | For of SSA * SSA * SSA * MLIROp list                    // lower, upper, step, bodyOps
    | Yield of (SSA * MLIRType) list                          // values with types
    | Condition of SSA * SSA list                             // cond, args

/// Control Flow (CF) dialect operations - unstructured control flow
/// passed via `byval` — the struct data goes on the stack, not in a register.
and ByvalParam = { ParamIndex: int; SizeBytes: int; AlignBytes: int }

/// Function dialect operations
and FuncOp =
    // Function definition/declaration
    | FuncDef of string * (SSA * MLIRType) list * MLIRType list * MLIROp list * FuncVisibility  // name, args, resultTypes, body, visibility
    | FuncDecl of string * MLIRType list * MLIRType list * FuncVisibility * ByvalParam list     // name, paramTypes, resultTypes, visibility, byvalParams (external decl)
    // Function calls
    | FuncCall of Val list * string * Val list                                             // results, func, args
    | FuncCallIndirect of Val list * SSA * Val list                                        // results, callee, args
    | FuncConstant of SSA * string * MLIRType                                              // result, funcName, funcType
    // Cast index → function type for call_indirect (unrealized_conversion_cast)
    | IndexToFunc of SSA * SSA * MLIRType list * MLIRType                                 // result, sourceIndex, argTypes, retType
    // Explicit foreign function-address boundary, not canonical closure storage
    | FuncToIndex of SSA * SSA * MLIRType list * MLIRType                                 // result, sourceFunc, argTypes, retType
    // Return
    | Return of Val list                                                                  // ordered typed results

/// CIRCT Hardware Construction Dialect (structural, module-level)
/// Maps to hw dialect — hardware modules, instances, and ports for FPGA targets
and HWOp =
    | HWModule of string * (string * MLIRType) list * (string * MLIRType) list * MLIROp list
      // name, inputs (name*type), outputs (name*type), body
    | HWOutput of (SSA * MLIRType) list
    // Struct operations (hw.struct_create, hw.struct_extract, hw.struct_inject)
    | HWStructCreate of SSA * (SSA * MLIRType) list * MLIRType
      // result, field values (ssa * fieldType), structType
    | HWStructExtract of SSA * SSA * string * MLIRType
      // result, input, fieldName, structType
    | HWStructInject of SSA * SSA * string * SSA * MLIRType
      // result, input, fieldName, newValue, structType
    // Module instantiation (hw.instance)
    | HWInstance of SSA * string * string * (string * SSA * MLIRType) list * (string * MLIRType) list
      // result, instanceName, moduleName, inputs (portName*ssa*type), outputs (portName*type)
    // Aggregate constant (hw.aggregate_constant) — zero-initialize structs
    | HWAggregateConstant of SSA * MLIRType
      // result, structType (all fields zero-initialized)

/// SMT dialect operations (upstream `smt` dialect — proof obligations as IR).
/// Emitted into verification modules by SMTTransfer; exported to SMT-LIB via
/// mlir-translate --export-smtlib. Solver-neutral: any SMT-LIB solver dispatches.
and SMTOp =
    | SMTSolver of MLIROp list                      // smt.solver () : () -> () { body }
    | SMTSetLogic of string                         // smt.set_logic "QF_LIA"
    | SMTDeclareFun of SSA * string * SMTType       // result, name (exported verbatim), type
    | SMTIntConstant of SSA * int64                 // smt.int.constant
    | SMTBigIntConstant of SSA * bigint             // exact unbounded smt.int.constant
    | SMTBVConstant of SSA * int64 * int            // result, value, width
    | SMTIntAdd of SSA * SSA * SSA                  // result, lhs, rhs
    | SMTIntSub of SSA * SSA * SSA                  // result, lhs, rhs
    | SMTIntMod of SSA * SSA * SSA                  // result, lhs, rhs
    | SMTIntDiv of SSA * SSA * SSA                  // result, lhs, rhs (integer quotient)
    | SMTIntMul of SSA * SSA * SSA                  // result, lhs, rhs
    | SMTIntCmp of SSA * SMTCmpPred * SSA * SSA     // result, predicate, lhs, rhs
    | SMTEq of SSA * SSA * SSA * SMTType            // result, lhs, rhs, operand type
    | SMTAnd of SSA * SSA list                      // result, operands (variadic)
    | SMTOr of SSA * SSA list                       // result, operands (variadic)
    | SMTNot of SSA * SSA                           // result, operand
    | SMTAssert of SSA                              // assert a !smt.bool value
    | SMTCheck                                      // smt.check sat {} unknown {} unsat {}
