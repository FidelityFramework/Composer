/// MLIR Serialization
///
/// Converts structured MLIR types and operations to MLIR text format.
/// This is the ONLY place where sprintf is used for MLIR text generation.
/// All upstream code (witnesses, patterns, elements) works with structured MLIROp.
module Alex.Dialects.Core.Serialize

open Alex.Dialects.Core.Types

/// MLIR symbol names are bare identifiers ([A-Za-z_][A-Za-z0-9_$.]*). A Clef function name may
/// carry an apostrophe (`process'`), which is legal in F# and in the linked ELF symbol but not
/// in a bare MLIR id; it is spelled `$` in the symbol (a character no Clef name contains).
let symbolName (name: string) : string =
    name |> String.map (fun c -> if c = '\'' then '$' elif System.Char.IsLetterOrDigit c || c = '_' || c = '.' || c = '$' then c else '_')


// ═══════════════════════════════════════════════════════════════════════════
// TYPE SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert IntWidth to MLIR type string.
/// IntWidth 0 is a sentinel for "abstract width — must be resolved by interval analysis."
/// If it reaches serialization, width inference has failed — this is a hard error.
let intWidthToString (IntWidth bits) : string =
    if bits = 0 then
        failwith
            "Width inference failure: IntWidth 0 reached MLIR serialization. \
             A hardware integer's width is the width of the range CCS wrote on its node \
             (RangeAnalysis); a sentinel here is a value the witness did not narrow through \
             narrowType, a defect of the pipeline rather than of the program."
    sprintf "i%d" bits

/// Convert FloatWidth to MLIR type string
let floatWidthToString (width: FloatWidth) : string =
    match width with
    | F32 -> "f32"
    | F64 -> "f64"

/// Convert MLIRType to MLIR text format string
let rec typeToString (pointer: Result<int, string>) (ty: MLIRType) : string =
    match ty with
    | TInt width -> intWidthToString width
    | TFloat width -> floatWidthToString width
    | TFunc (paramTypes, retType) ->
        let paramStrs = paramTypes |> List.map (typeToString pointer) |> String.concat ", "
        sprintf "(%s) -> %s" paramStrs (typeToString pointer retType)
    | TMemRef elemTy ->
        sprintf "memref<?x%s>" (typeToString pointer elemTy)
    | TMemRefStatic (size, elemTy) ->
        sprintf "memref<%dx%s>" size (typeToString pointer elemTy)
    | TMemRefScalar elemTy ->
        // Scalar memref (0D) is represented as 1-element static memref in MLIR
        sprintf "memref<1x%s>" (typeToString pointer elemTy)
    | TVector (count, elemTy) ->
        sprintf "vector<%dx%s>" count (typeToString pointer elemTy)
    | TIndex -> "index"
    | TUnit -> "i32"  // Unit represented as i32 (value 0)
    | TStruct fields ->
        // TStruct serializes as !hw.struct for CIRCT (FPGA) or memref for CPU.
        // This default path is CPU; hwTypeToString pointer handles the FPGA case.
        // The byte size reads the declared Pointer width the caller hands in (the one size model).
        sprintf "memref<%dxi8>" (mlirTypeSizeWith pointer (TStruct fields))
    | TSeqClock -> "!seq.clock"
    | TTag caseCount ->
        // Default serialization: smallest power-of-2 integer that fits the case count
        // Platform-specific serializers (hw dialect) may override with exact bit widths
        if caseCount <= 2 then "i1"
        elif caseCount <= 256 then "i8"
        elif caseCount <= 65536 then "i16"
        else "i32"
    | TError msg -> sprintf "<<ERROR: %s>>" msg

/// FPGA-aware type serialization: TStruct → !hw.struct<...>, all others → typeToString pointer
/// Used by comb.* and other CIRCT ops that carry struct types on FPGA.
let rec hwTypeToString (pointer: Result<int, string>) (ty: MLIRType) : string =
    match ty with
    | TStruct fields ->
        let fs = fields |> List.map (fun (n, t) -> sprintf "%s: %s" n (hwTypeToString pointer t))
        sprintf "!hw.struct<%s>" (String.concat ", " fs)
    | _ -> typeToString pointer ty

// ═══════════════════════════════════════════════════════════════════════════
// SSA SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert SSA to MLIR SSA value string
let ssaToString (ssa: SSA) : string =
    match ssa with
    | V n -> sprintf "%%v%d" n
    | Arg n -> sprintf "%%arg%d" n

/// Convert Val (SSA + type) to typed SSA value string
let valToString (pointer: Result<int, string>) (v: Val) : string =
    sprintf "%s : %s" (ssaToString v.SSA) (typeToString pointer v.Type)

// ═══════════════════════════════════════════════════════════════════════════
// OPERATION SERIALIZATION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert ICmpPred to MLIR predicate string
let icmpPredToString (pred: ICmpPred) : string =
    match pred with
    | ICmpPred.Eq -> "eq"
    | ICmpPred.Ne -> "ne"
    | ICmpPred.Slt -> "slt"
    | ICmpPred.Sle -> "sle"
    | ICmpPred.Sgt -> "sgt"
    | ICmpPred.Sge -> "sge"
    | ICmpPred.Ult -> "ult"
    | ICmpPred.Ule -> "ule"
    | ICmpPred.Ugt -> "ugt"
    | ICmpPred.Uge -> "uge"

/// Convert FCmpPred to MLIR predicate string
let fcmpPredToString (pred: FCmpPred) : string =
    match pred with
    | OEq -> "oeq"
    | OGt -> "ogt"
    | OGe -> "oge"
    | OLt -> "olt"
    | OLe -> "ole"
    | ONe -> "one"
    | Ord -> "ord"
    | UEq -> "ueq"
    | UGt -> "ugt"
    | UGe -> "uge"
    | ULt -> "ult"
    | ULe -> "ule"
    | UNe -> "une"
    | Uno -> "uno"
    | AlwaysFalse -> "false"
    | AlwaysTrue -> "true"

/// Serialize ArithOp to MLIR text
let arithOpToString (pointer: Result<int, string>) (op: ArithOp) : string =
    match op with
    | ConstI (result, value, ty) ->
        sprintf "%s = arith.constant %d : %s" (ssaToString result) value (typeToString pointer ty)
    | ConstF (result, value, ty) ->
        sprintf "%s = arith.constant %f : %s" (ssaToString result) value (typeToString pointer ty)
    | AddI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.addi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | SubI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.subi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | MulI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.muli %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | DivSI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.divsi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | DivUI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.divui %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | RemSI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.remsi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | RemUI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.remui %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | AddF (result, lhs, rhs, ty) ->
        sprintf "%s = arith.addf %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | SubF (result, lhs, rhs, ty) ->
        sprintf "%s = arith.subf %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | NegF (result, operand, ty) ->
        sprintf "%s = arith.negf %s : %s" (ssaToString result) (ssaToString operand) (typeToString pointer ty)
    | MulF (result, lhs, rhs, ty) ->
        sprintf "%s = arith.mulf %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | DivF (result, lhs, rhs, ty) ->
        sprintf "%s = arith.divf %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | CmpI (result, pred, lhs, rhs, ty) ->
        sprintf "%s = arith.cmpi %s, %s, %s : %s" (ssaToString result) (icmpPredToString pred) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | CmpF (result, pred, lhs, rhs, ty) ->
        sprintf "%s = arith.cmpf %s, %s, %s : %s" (ssaToString result) (fcmpPredToString pred) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | ExtSI (result, value, srcTy, destTy) ->
        sprintf "%s = arith.extsi %s : %s to %s" (ssaToString result) (ssaToString value) (typeToString pointer srcTy) (typeToString pointer destTy)
    | ExtUI (result, value, srcTy, destTy) ->
        sprintf "%s = arith.extui %s : %s to %s" (ssaToString result) (ssaToString value) (typeToString pointer srcTy) (typeToString pointer destTy)
    | TruncI (result, value, srcTy, destTy) ->
        sprintf "%s = arith.trunci %s : %s to %s" (ssaToString result) (ssaToString value) (typeToString pointer srcTy) (typeToString pointer destTy)
    | SIToFP (result, value, srcTy, destTy) ->
        sprintf "%s = arith.sitofp %s : %s to %s" (ssaToString result) (ssaToString value) (typeToString pointer srcTy) (typeToString pointer destTy)
    | FPToSI (result, value, srcTy, destTy) ->
        sprintf "%s = arith.fptosi %s : %s to %s" (ssaToString result) (ssaToString value) (typeToString pointer srcTy) (typeToString pointer destTy)
    | Select (result, cond, trueVal, falseVal, ty) ->
        sprintf "%s = arith.select %s, %s, %s : %s"
            (ssaToString result) (ssaToString cond) (ssaToString trueVal) (ssaToString falseVal) (typeToString pointer ty)
    // Bitwise operations (migrated from LLVM dialect)
    | AndI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.andi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | OrI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.ori %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | XorI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.xori %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | ShLI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.shli %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | ShRUI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.shrui %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)
    | ShRSI (result, lhs, rhs, ty) ->
        sprintf "%s = arith.shrsi %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (typeToString pointer ty)

/// Serialize CombOp to CIRCT MLIR text (comb dialect)
/// Uses hwTypeToString: comb.* ops only appear on FPGA where TStruct → !hw.struct
let combOpToString (pointer: Result<int, string>) (op: CombOp) : string =
    match op with
    | CombAdd (result, lhs, rhs, ty) ->
        sprintf "%s = comb.add %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombSub (result, lhs, rhs, ty) ->
        sprintf "%s = comb.sub %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombMul (result, lhs, rhs, ty) ->
        sprintf "%s = comb.mul %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombDivS (result, lhs, rhs, ty) ->
        sprintf "%s = comb.divs %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombDivU (result, lhs, rhs, ty) ->
        sprintf "%s = comb.divu %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombMod (result, lhs, rhs, ty) ->
        sprintf "%s = comb.mods %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombModU (result, lhs, rhs, ty) ->
        sprintf "%s = comb.modu %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombAnd (result, lhs, rhs, ty) ->
        sprintf "%s = comb.and %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombOr (result, lhs, rhs, ty) ->
        sprintf "%s = comb.or %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombXor (result, lhs, rhs, ty) ->
        sprintf "%s = comb.xor %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombShl (result, lhs, rhs, ty) ->
        sprintf "%s = comb.shl %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombShrU (result, lhs, rhs, ty) ->
        sprintf "%s = comb.shru %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombShrS (result, lhs, rhs, ty) ->
        sprintf "%s = comb.shrs %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombICmp (result, pred, lhs, rhs, ty) ->
        sprintf "%s = comb.icmp %s %s, %s : %s" (ssaToString result) (icmpPredToString pred) (ssaToString lhs) (ssaToString rhs) (hwTypeToString pointer ty)
    | CombMux (result, cond, trueVal, falseVal, ty) ->
        sprintf "%s = comb.mux %s, %s, %s : %s" (ssaToString result) (ssaToString cond) (ssaToString trueVal) (ssaToString falseVal) (hwTypeToString pointer ty)

/// Serialize HWOp to CIRCT MLIR text (hw dialect)
/// Note: HWModule body serialization delegates to opToString pointer (defined later),
/// which handles all MLIROp cases. A non-CIRCT op in an hw.module body is a
/// pipeline error — it means a CPU-dialect op leaked into FPGA output.
let hwOpToString (pointer: Result<int, string>) (inner: MLIROp -> string) (op: HWOp) : string =
    match op with
    | HWModule (name, inputs, outputs, body) ->
        let inputsStr = inputs |> List.map (fun (n, ty) -> sprintf "in %%%s: %s" n (hwTypeToString pointer ty)) |> String.concat ", "
        let outputsStr = outputs |> List.map (fun (n, ty) -> sprintf "out %s: %s" n (hwTypeToString pointer ty)) |> String.concat ", "
        let portsStr =
            match inputs, outputs with
            | [], [] -> ""
            | _, [] -> inputsStr
            | [], _ -> outputsStr
            | _, _ -> sprintf "%s, %s" inputsStr outputsStr
        let bodyStr = body |> List.map inner |> String.concat "\n    "
        // hw.module uses named ports — replace %argN with port names (highest index first to avoid partial matches)
        let bodyWithPorts =
            inputs
            |> List.indexed
            |> List.rev
            |> List.fold (fun (s: string) (i, (portName, _)) ->
                s.Replace(sprintf "%%arg%d" i, sprintf "%%%s" portName)) bodyStr
        sprintf "hw.module @%s(%s) {\n    %s\n}" name portsStr bodyWithPorts
    | HWOutput vals ->
        match vals with
        | [] -> "hw.output"
        | _ ->
            let valsStr = vals |> List.map (fun (ssa, _) -> ssaToString ssa) |> String.concat ", "
            let typesStr = vals |> List.map (fun (_, ty) -> hwTypeToString pointer ty) |> String.concat ", "
            sprintf "hw.output %s : %s" valsStr typesStr
    | HWStructCreate (result, fieldVals, structTy) ->
        let valsStr = fieldVals |> List.map (fun (ssa, _) -> ssaToString ssa) |> String.concat ", "
        let tyStr = hwTypeToString pointer structTy
        sprintf "%s = hw.struct_create (%s) : %s" (ssaToString result) valsStr tyStr
    | HWStructExtract (result, input, fieldName, structTy) ->
        let tyStr = hwTypeToString pointer structTy
        sprintf "%s = hw.struct_extract %s[\"%s\"] : %s" (ssaToString result) (ssaToString input) fieldName tyStr
    | HWStructInject (result, input, fieldName, newValue, structTy) ->
        let tyStr = hwTypeToString pointer structTy
        sprintf "%s = hw.struct_inject %s[\"%s\"], %s : %s" (ssaToString result) (ssaToString input) fieldName (ssaToString newValue) tyStr
    | HWInstance (result, instName, moduleName, inputs, outputs) ->
        // hw.instance "instName" @moduleName(portName: %ssa : type, ...) -> (outName: type, ...)
        let inputsStr = inputs |> List.map (fun (pn, ssa, ty) -> sprintf "%s: %s: %s" pn (ssaToString ssa) (hwTypeToString pointer ty)) |> String.concat ", "
        let outputsStr = outputs |> List.map (fun (pn, ty) -> sprintf "%s: %s" pn (hwTypeToString pointer ty)) |> String.concat ", "
        sprintf "%s = hw.instance \"%s\" @%s(%s) -> (%s)" (ssaToString result) instName moduleName inputsStr outputsStr
    | HWAggregateConstant (result, structTy) ->
        let rec zeroLiteral (ty: MLIRType) : string =
            match ty with
            | TStruct fields ->
                let inner = fields |> List.map (fun (_, ft) -> zeroLiteral ft) |> String.concat ", "
                sprintf "[%s]" inner
            | _ -> sprintf "0 : %s" (hwTypeToString pointer ty)
        sprintf "%s = hw.aggregate_constant [%s] : %s"
            (ssaToString result)
            (match structTy with
             | TStruct fields -> fields |> List.map (fun (_, ft) -> zeroLiteral ft) |> String.concat ", "
             | _ -> zeroLiteral structTy)
            (hwTypeToString pointer structTy)

/// Serialize SeqOp to CIRCT MLIR text (seq dialect)
let seqOpToString (pointer: Result<int, string>) (op: SeqOp) : string =
    match op with
    | SeqCompreg (result, input, clk, resetOpt, ty) ->
        match resetOpt with
        | Some (resetSignal, resetValue) ->
            sprintf "%s = seq.compreg %s, %s reset %s, %s : %s"
                (ssaToString result) (ssaToString input) (ssaToString clk)
                (ssaToString resetSignal) (ssaToString resetValue) (typeToString pointer ty)
        | None ->
            sprintf "%s = seq.compreg %s, %s : %s"
                (ssaToString result) (ssaToString input) (ssaToString clk) (typeToString pointer ty)

/// Serialize an SMT dialect type to MLIR text
let smtTypeToString (ty: SMTType) : string =
    match ty with
    | SMTBool -> "!smt.bool"
    | SMTInt -> "!smt.int"
    | SMTBV w -> sprintf "!smt.bv<%d>" w

/// Serialize SMTOp to MLIR text (verification modules)
let smtOpToString (pointer: Result<int, string>) (inner: MLIROp -> string) (op: SMTOp) : string =
    match op with
    | SMTSolver body ->
        let bodyStr = body |> List.map inner |> String.concat "\n    "
        sprintf "smt.solver () : () -> () {\n    %s\n  }" bodyStr
    | SMTSetLogic logic ->
        sprintf "smt.set_logic \"%s\"" logic
    | SMTDeclareFun (result, name, ty) ->
        sprintf "%s = smt.declare_fun \"%s\" : %s" (ssaToString result) name (smtTypeToString ty)
    | SMTIntConstant (result, value) ->
        sprintf "%s = smt.int.constant %d" (ssaToString result) value
    | SMTBVConstant (result, value, width) ->
        sprintf "%s = smt.bv.constant #smt.bv<%d> : !smt.bv<%d>" (ssaToString result) value width
    | SMTIntAdd (result, lhs, rhs) ->
        sprintf "%s = smt.int.add %s, %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs)
    | SMTIntSub (result, lhs, rhs) ->
        sprintf "%s = smt.int.sub %s, %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs)
    | SMTIntCmp (result, pred, lhs, rhs) ->
        let predStr =
            match pred with
            | SmtLt -> "lt" | SmtLe -> "le" | SmtGt -> "gt" | SmtGe -> "ge"
        sprintf "%s = smt.int.cmp %s %s, %s" (ssaToString result) predStr (ssaToString lhs) (ssaToString rhs)
    | SMTEq (result, lhs, rhs, ty) ->
        sprintf "%s = smt.eq %s, %s : %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs) (smtTypeToString ty)
    | SMTAnd (result, operands) ->
        sprintf "%s = smt.and %s" (ssaToString result) (operands |> List.map ssaToString |> String.concat ", ")
    | SMTOr (result, operands) ->
        sprintf "%s = smt.or %s" (ssaToString result) (operands |> List.map ssaToString |> String.concat ", ")
    | SMTNot (result, operand) ->
        sprintf "%s = smt.not %s" (ssaToString result) (ssaToString operand)
    | SMTAssert operand ->
        sprintf "smt.assert %s" (ssaToString operand)
    | SMTCheck ->
        "smt.check sat {} unknown {} unsat {}"

/// Serialize MemRefOp to MLIR text
let memrefOpToString (pointer: Result<int, string>) (op: MemRefOp) : string =
    match op with
    | MemRefOp.Load (result, memref, indices, _elemType, memrefType) ->
        // Build indices string
        let indicesStr = if List.isEmpty indices then "" else sprintf "[%s]" (indices |> List.map ssaToString |> String.concat ", ")
        // Use the passed memrefType directly (no heuristic reconstruction)
        sprintf "%s = memref.load %s%s : %s"
            (ssaToString result) (ssaToString memref) indicesStr (typeToString pointer memrefType)
    | MemRefOp.Store (value, memref, indices, _elemType, memrefType) ->
        // Build indices string
        let indicesStr = if List.isEmpty indices then "" else sprintf "[%s]" (indices |> List.map ssaToString |> String.concat ", ")
        // Use the passed memrefType directly (no heuristic reconstruction)
        sprintf "memref.store %s, %s%s : %s"
            (ssaToString value) (ssaToString memref) indicesStr (typeToString pointer memrefType)
    | MemRefOp.Alloca (result, memrefType, alignmentOpt) ->
        match alignmentOpt with
        | Some alignment ->
            sprintf "%s = memref.alloca() {alignment = %d : i64} : %s"
                (ssaToString result) alignment (typeToString pointer memrefType)
        | None ->
            sprintf "%s = memref.alloca() : %s" (ssaToString result) (typeToString pointer memrefType)
    | MemRefOp.Alloc (result, sizeSSA, elemType) ->
        // Heap allocation with runtime size: memref.alloc(%size) : memref<?xelemType>
        let memrefType = TMemRef elemType
        sprintf "%s = memref.alloc(%s) : %s"
            (ssaToString result) (ssaToString sizeSSA) (typeToString pointer memrefType)
    | MemRefOp.AllocStatic (result, memrefType, alignmentOpt) ->
        // Heap allocation with compile-time size: memref.alloc() : memref<NxT>
        // Like Alloca but heap-allocated — survives function return
        match alignmentOpt with
        | Some alignment ->
            sprintf "%s = memref.alloc() {alignment = %d : i64} : %s"
                (ssaToString result) alignment (typeToString pointer memrefType)
        | None ->
            sprintf "%s = memref.alloc() : %s" (ssaToString result) (typeToString pointer memrefType)
    | MemRefOp.SubView (result, source, offsets, resultType) ->
        let offsetsStr = offsets |> List.map ssaToString |> String.concat ", "
        sprintf "%s = memref.subview %s[%s] : %s"
            (ssaToString result) (ssaToString source) offsetsStr (typeToString pointer resultType)
    | MemRefOp.SubViewSlice (result, source, offsets, sizes, strides, sourceType) ->
        // Proper MLIR memref.subview with 3 bracket groups: [offsets] [sizes] [strides]
        // Result has strided layout — callers must copy to contiguous buffer for FFI use.
        let fmtParam = function
            | SubViewParam.Static n -> string n
            | SubViewParam.Dynamic s -> ssaToString s
        let offsetsStr = offsets |> List.map ssaToString |> String.concat ", "
        let sizesStr = sizes |> List.map fmtParam |> String.concat ", "
        let stridesStr = strides |> List.map fmtParam |> String.concat ", "
        let elemType =
            match sourceType with
            | TMemRef t | TMemRefStatic (_, t) | TMemRefScalar t -> t
            | t -> t
        let elemStr = typeToString pointer elemType
        let sizeStr =
            match sizes with
            | [SubViewParam.Static n] -> string n
            | _ -> "?"
        let strideStr =
            match strides with
            | [SubViewParam.Static n] -> string n
            | _ -> "?"
        let stridedTypeStr = sprintf "memref<%sx%s, strided<[%s], offset: ?>>" sizeStr elemStr strideStr
        sprintf "%s = memref.subview %s[%s] [%s] [%s] : %s to %s"
            (ssaToString result) (ssaToString source) offsetsStr sizesStr stridesStr
            (typeToString pointer sourceType) stridedTypeStr
    | MemRefOp.SubViewCopy (result, source, offsets, sizes, strides, sizeIndexSSA, sourceType) ->
        // SubView + Alloc + Copy: create a fresh contiguous buffer from a slice.
        // This is needed because memref.extract_aligned_pointer_as_index on a subview
        // gives the BASE pointer, not the data pointer. Copying to a fresh alloc fixes this.
        let fmtParam = function
            | SubViewParam.Static n -> string n
            | SubViewParam.Dynamic s -> ssaToString s
        let offsetsStr = offsets |> List.map ssaToString |> String.concat ", "
        let sizesStr = sizes |> List.map fmtParam |> String.concat ", "
        let stridesStr = strides |> List.map fmtParam |> String.concat ", "
        let elemType =
            match sourceType with
            | TMemRef t | TMemRefStatic (_, t) | TMemRefScalar t -> t
            | t -> t
        let elemStr = typeToString pointer elemType
        let sizeStr =
            match sizes with
            | [SubViewParam.Static n] -> string n
            | _ -> "?"
        let strideStr =
            match strides with
            | [SubViewParam.Static n] -> string n
            | _ -> "?"
        let stridedTypeStr = sprintf "memref<%sx%s, strided<[%s], offset: ?>>" sizeStr elemStr strideStr
        let plainTypeStr = sprintf "memref<?x%s>" elemStr
        let intermSSA = ssaToString result + "_sv"
        // 1. SubView: create strided view into source
        let subviewLine = sprintf "%s = memref.subview %s[%s] [%s] [%s] : %s to %s"
                            intermSSA (ssaToString source) offsetsStr sizesStr stridesStr
                            (typeToString pointer sourceType) stridedTypeStr
        // 2. Alloc: fresh contiguous buffer
        let allocLine = sprintf "%s = memref.alloc(%s) : %s"
                            (ssaToString result) (ssaToString sizeIndexSSA) plainTypeStr
        // 3. Copy via scf.for loop (avoids memref.copy → memrefCopy runtime dependency)
        let resultName = ssaToString result
        let c0Name = resultName + "_c0"
        let c1Name = resultName + "_c1"
        let ivName = resultName + "_iv"
        let ldName = resultName + "_ld"
        let c0Line = sprintf "%s = arith.constant 0 : index" c0Name
        let c1Line = sprintf "%s = arith.constant 1 : index" c1Name
        let loadLine = sprintf "%s = memref.load %s[%s] : %s" ldName intermSSA ivName stridedTypeStr
        let storeLine = sprintf "memref.store %s, %s[%s] : %s" ldName resultName ivName plainTypeStr
        let forLoop = sprintf "scf.for %s = %s to %s step %s {\n      %s\n      %s\n    }" ivName c0Name (ssaToString sizeIndexSSA) c1Name loadLine storeLine
        sprintf "%s\n    %s\n    %s\n    %s\n    %s" subviewLine allocLine c0Line c1Line forLoop
    | MemRefOp.ExtractBasePtr (result, memref, ty) ->
        // Extract pointer as platform word (index type) - PORTABLE!
        // This replaces the old LLVM-specific unrealized_conversion_cast
        // Returns index (platform word size), caller must cast to target type if needed
        sprintf "%s = memref.extract_aligned_pointer_as_index %s : %s -> index"
            (ssaToString result) (ssaToString memref) (typeToString pointer ty)
    | MemRefOp.GetGlobal (result, globalName, memrefType) ->
        // memref.get_global @symbol_name : memref<...>
        sprintf "%s = memref.get_global @%s : %s"
            (ssaToString result) globalName (typeToString pointer memrefType)
    | MemRefOp.Dim (result, memref, index, memrefType) ->
        // memref.dim %memref, %index : memref<...>
        sprintf "%s = memref.dim %s, %s : %s"
            (ssaToString result) (ssaToString memref) (ssaToString index) (typeToString pointer memrefType)
    | MemRefOp.Cast (result, source, srcType, destType) ->
        // memref.cast %source : srcType to destType
        sprintf "%s = memref.cast %s : %s to %s"
            (ssaToString result) (ssaToString source) (typeToString pointer srcType) (typeToString pointer destType)
    | MemRefOp.ReinterpretCast (result, source, byteOffset, size, srcType, destType) ->
        // Check if source and dest have different element types
        let getElemType = function
            | TMemRef e | TMemRefStatic (_, e) | TMemRefScalar e -> Some e
            | _ -> None
        match getElemType srcType, getElemType destType with
        | Some srcElem, Some destElem when srcElem <> destElem ->
            // Different element types: emit memref.view (allows cross-element-type access)
            // Generate inline offset constant + view as two ops on separate lines
            let resultStr = ssaToString result
            let offsetName = sprintf "%s_off" resultStr
            sprintf "%s = arith.constant %d : index\n    %s = memref.view %s[%s][] : %s to %s"
                offsetName byteOffset resultStr (ssaToString source) offsetName (typeToString pointer srcType) (typeToString pointer destType)
        | _ ->
            // Same element type: standard memref.reinterpret_cast
            sprintf "%s = memref.reinterpret_cast %s to offset: [%d], sizes: [%d], strides: [1] : %s to %s"
                (ssaToString result) (ssaToString source) byteOffset size (typeToString pointer srcType) (typeToString pointer destType)
    | MemRefOp.ReinterpretCastDynamic (result, source, offset, sizeSSA, srcType, destType) ->
        // memref.reinterpret_cast with dynamic size: reconstruct memref from pointer + known length
        // Used for string/array capture extraction where size is loaded from closure struct
        sprintf "%s = memref.reinterpret_cast %s to offset: [%d], sizes: [%s], strides: [1] : %s to %s"
            (ssaToString result) (ssaToString source) offset (ssaToString sizeSSA) (typeToString pointer srcType) (typeToString pointer destType)
    | MemRefOp.View (result, source, offsetSSA, srcType, destType) ->
        // memref.view: typed view of byte buffer (different element type allowed)
        // Portable across all targets: CPU (→ GEP), FPGA (→ typed memory port), NPU (→ typed channel)
        sprintf "%s = memref.view %s[%s][] : %s to %s"
            (ssaToString result) (ssaToString source) (ssaToString offsetSSA) (typeToString pointer srcType) (typeToString pointer destType)
    | MemRefOp.IndexToMemRef (result, source, destType) ->
        // builtin.unrealized_conversion_cast: FFI boundary crossing (raw pointer → memref)
        // Internal index→memref seam (platform pointer as index → memref for typed access)
        sprintf "%s = builtin.unrealized_conversion_cast %s : index to %s"
            (ssaToString result) (ssaToString source) (typeToString pointer destType)
    | MemRefOp.MemRefToIndex (result, source, srcType) ->
        // builtin.unrealized_conversion_cast: memref → raw pointer (index)
        // Internal memref→index seam (stack alloc → index for FFI boundary crossing)
        sprintf "%s = builtin.unrealized_conversion_cast %s : %s to index"
            (ssaToString result) (ssaToString source) (typeToString pointer srcType)

/// Serialize top-level MLIROp to MLIR text
let rec opToString (pointer: Result<int, string>) (op: MLIROp) : string =
    match op with
    | MLIROp.ArithOp aop -> arithOpToString pointer aop
    | MLIROp.MemRefOp mop -> memrefOpToString pointer mop
    | MLIROp.FuncOp fop ->
        match fop with
        | FuncDef (name, args, retTy, body, _visibility) ->
            let argsStr = args |> List.map (fun (ssa, ty) -> sprintf "%s: %s" (ssaToString ssa) (typeToString pointer ty)) |> String.concat ", "
            let bodyStr = body |> List.map (opToString pointer) |> String.concat "\n    "
            sprintf "func.func @%s(%s) -> %s {\n    %s\n}" (symbolName name) argsStr (typeToString pointer retTy) bodyStr
        | FuncDecl (name, paramTypes, retTy, _visibility, byvalParams) ->
            let paramsStr = paramTypes |> List.map (typeToString pointer) |> String.concat ", "
            let attrsStr =
                match byvalParams with
                | [] -> ""
                | bvs ->
                    // Encode byval metadata as function attribute for reconcile-ffi-externs plugin.
                    // Format: "idx:size:align,idx:size:align,..."
                    let bvStr = bvs |> List.map (fun bv -> sprintf "%d:%d:%d" bv.ParamIndex bv.SizeBytes bv.AlignBytes) |> String.concat ","
                    sprintf " attributes {ffi.byval = \"%s\"}" bvStr
            sprintf "func.func private @%s(%s) -> %s%s" (symbolName name) paramsStr (typeToString pointer retTy) attrsStr
        | FuncCall (resultOpt, funcName, args, retTy) ->
            let argSSAs = args |> List.map (fun v -> ssaToString v.SSA) |> String.concat ", "
            let argTypes = args |> List.map (fun v -> typeToString pointer v.Type) |> String.concat ", "
            match resultOpt with
            | Some result -> sprintf "%s = func.call @%s(%s) : (%s) -> %s" (ssaToString result) (symbolName funcName) argSSAs argTypes (typeToString pointer retTy)
            | None -> sprintf "func.call @%s(%s) : (%s) -> %s" (symbolName funcName) argSSAs argTypes (typeToString pointer retTy)
        | FuncCallIndirect (resultOpt, callee, args, retTy) ->
            let argSSAs = args |> List.map (fun v -> ssaToString v.SSA) |> String.concat ", "
            let argTypes = args |> List.map (fun v -> typeToString pointer v.Type) |> String.concat ", "
            match resultOpt with
            | Some result -> sprintf "%s = func.call_indirect %s(%s) : (%s) -> %s" (ssaToString result) (ssaToString callee) argSSAs argTypes (typeToString pointer retTy)
            | None -> sprintf "func.call_indirect %s(%s) : (%s) -> %s" (ssaToString callee) argSSAs argTypes (typeToString pointer retTy)
        | FuncConstant (result, funcName, funcTy) ->
            sprintf "%s = func.constant @%s : %s" (ssaToString result) (symbolName funcName) (typeToString pointer funcTy)
        | IndexToFunc (result, source, argTypes, retTy) ->
            let funcTyStr =
                let argsStr = argTypes |> List.map (typeToString pointer) |> String.concat ", "
                sprintf "(%s) -> %s" argsStr (typeToString pointer retTy)
            sprintf "%s = builtin.unrealized_conversion_cast %s : index to %s" (ssaToString result) (ssaToString source) funcTyStr
        | FuncToIndex (result, source, argTypes, retTy) ->
            let funcTyStr =
                let argsStr = argTypes |> List.map (typeToString pointer) |> String.concat ", "
                sprintf "(%s) -> %s" argsStr (typeToString pointer retTy)
            sprintf "%s = builtin.unrealized_conversion_cast %s : %s to index" (ssaToString result) (ssaToString source) funcTyStr
        | Return (valueOpt, tyOpt) ->
            match valueOpt, tyOpt with
            | Some value, Some ty -> sprintf "func.return %s : %s" (ssaToString value) (typeToString pointer ty)
            | Some value, None -> sprintf "func.return %s" (ssaToString value)
            | None, _ -> "func.return"
    | MLIROp.GlobalString (name, content, storageLength, obligations) ->
        // Emit memref.global (portable MLIR) with null sentinel byte for C interop.
        // Clef strings are (ptr, length) — the sentinel is a storage detail invisible
        // to the type system, ensuring .Pointer yields C-compatible null-terminated data.
        let bytes = System.Text.Encoding.UTF8.GetBytes(content)
        let bytesWithSentinel = Array.append bytes [| 0uy |]
        let denseStr = bytesWithSentinel |> Array.map (sprintf "%d") |> String.concat ", "
        // The obligations constraining this storage, reified on the op as a
        // discardable attribute (PHG paper 2.4b). The artifact carries the
        // correspondence explicitly; the artifact-side check reads it here
        // rather than reconstructing it by matching content.
        let attrs =
            match obligations with
            | [] -> ""
            | names -> sprintf " {clef.obligations = [%s]}" (names |> List.map (sprintf "\"%s\"") |> String.concat ", ")
        sprintf "memref.global \"private\" constant @%s : memref<%dxi8> = dense<[%s]>%s" name storageLength denseStr attrs
    | MLIROp.GlobalMemref (name, memrefType) ->
        // Zero-initialized static storage for a program-lifetime value (the program-lifetime
        // point of the lifetime lattice). Not `constant`: the closure struct is written into
        // this storage at construction. `uninitialized` is correct because every read is
        // preceded by the construction store; a heap-free target places this in .bss/Sram.
        sprintf "memref.global \"private\" @%s : %s = uninitialized" name (typeToString pointer memrefType)
    | MLIROp.IndexOp iop ->
        match iop with
        | IndexOp.IndexConst (result, value) ->
            sprintf "%s = arith.constant %d : index" (ssaToString result) value
        | IndexOp.IndexBoolConst (result, value) ->
            let boolVal = if value then 1 else 0
            sprintf "%s = arith.constant %d : i1" (ssaToString result) boolVal
        | IndexOp.IndexCastS (result, operand, srcTy, destTy) ->
            sprintf "%s = index.casts %s : %s to %s" (ssaToString result) (ssaToString operand) (typeToString pointer srcTy) (typeToString pointer destTy)
        | IndexOp.IndexCastU (result, operand, srcTy, destTy) ->
            sprintf "%s = index.castu %s : %s to %s" (ssaToString result) (ssaToString operand) (typeToString pointer srcTy) (typeToString pointer destTy)
        | IndexOp.IndexCmp (result, pred, lhs, rhs) ->
            let predStr =
                match pred with
                | Eq -> "eq" | Ne -> "ne"
                | Slt -> "slt" | Sle -> "sle"
                | Sgt -> "sgt" | Sge -> "sge"
                | Ult -> "ult" | Ule -> "ule"
                | Ugt -> "ugt" | Uge -> "uge"
            sprintf "%s = index.cmp %s(%s, %s)" (ssaToString result) predStr (ssaToString lhs) (ssaToString rhs)
        | IndexOp.IndexAdd (result, lhs, rhs) ->
            sprintf "%s = index.add %s, %s" (ssaToString result) (ssaToString lhs) (ssaToString rhs)
        | _ ->
            sprintf "// TODO: Serialize IndexOp %A" iop
    | MLIROp.SCFOp scfOp ->
        match scfOp with
        | SCFOp.While (condOps, bodyOps) ->
            // scf.while with condition and body regions
            let condStr = condOps |> List.map (opToString pointer) |> String.concat "\n      "
            let bodyStr = bodyOps |> List.map (opToString pointer) |> String.concat "\n      "
            sprintf "scf.while : () -> () {\n      %s\n    } do {\n      %s\n    }" condStr bodyStr
        | SCFOp.If (cond, thenOps, elseOpsOpt, resultOpt) ->
            let thenStr = thenOps |> List.map (opToString pointer) |> String.concat "\n      "
            // Result annotation: %result = scf.if %cond -> (type) { ... }
            let prefix, suffix =
                match resultOpt with
                | Some (resultSSA, resultType) ->
                    sprintf "%s = " (ssaToString resultSSA), sprintf " -> (%s)" (typeToString pointer resultType)
                | None -> "", ""
            match elseOpsOpt with
            | Some elseOps ->
                let elseStr = elseOps |> List.map (opToString pointer) |> String.concat "\n      "
                sprintf "%sscf.if %s%s {\n      %s\n    } else {\n      %s\n    }" prefix (ssaToString cond) suffix thenStr elseStr
            | None ->
                sprintf "%sscf.if %s%s {\n      %s\n    }" prefix (ssaToString cond) suffix thenStr
        | SCFOp.For (lower, upper, step, bodyOps) ->
            let bodyStr = bodyOps |> List.map (opToString pointer) |> String.concat "\n      "
            sprintf "scf.for %s = %s to %s step %s {\n      %s\n    }" 
                (ssaToString lower) (ssaToString upper) (ssaToString step) (ssaToString step) bodyStr
        | SCFOp.Yield vals ->
            match vals with
            | [] -> "scf.yield"
            | _ ->
                let valStr = vals |> List.map (fun (ssa, ty) -> sprintf "%s : %s" (ssaToString ssa) (typeToString pointer ty)) |> String.concat ", "
                sprintf "scf.yield %s" valStr
        | SCFOp.Condition (cond, args) ->
            let argsStr = args |> List.map ssaToString |> String.concat ", "
            sprintf "scf.condition(%s) %s" (ssaToString cond) argsStr
    | MLIROp.CombOp cop -> combOpToString pointer cop
    | MLIROp.HWOp hop -> hwOpToString pointer (opToString pointer) hop
    | MLIROp.SeqOp sop -> seqOpToString pointer sop
    | MLIROp.SMTOp sop -> smtOpToString pointer (opToString pointer) sop
    | MLIROp.RawMLIR text -> text
    | _ ->
        // Placeholder for operations with no serializer yet (Block, Region)
        sprintf "// TODO: Serialize %A" op

/// Serialize a list of operations with proper indentation
/// Serialize one op; a width failure inside it is re-raised naming the op, so that the
/// stop says which value had no width rather than only that one did.
let private opToStringNamed (pointer: Result<int, string>) (op: MLIROp) : string =
    try opToString pointer op
    with ex when ex.Message.StartsWith "Width inference failure" ->
        let rendered = sprintf "%A" op
        let shown = if rendered.Length > 400 then rendered.Substring(0, 400) + " ..." else rendered
        failwith (ex.Message + "\nWhile serialising: " + shown)

let opsToString (pointer: Result<int, string>) (ops: MLIROp list) (indent: string) : string =
    ops
    |> List.map (opToStringNamed pointer)
    |> List.map (fun line -> indent + line)
    |> String.concat "\n"

/// Serialize a complete MLIR module
let moduleToString (pointer: Result<int, string>) (moduleName: string) (ops: MLIROp list) : string =
    let opsText = opsToString pointer ops "  "
    sprintf "module @%s {\n%s\n}" moduleName opsText
