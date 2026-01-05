/// Arithmetic Templates - arith dialect operations
///
/// Templates for MLIR arith dialect operations:
/// - Binary integer/float arithmetic
/// - Comparison operations
/// - Unary operations
/// - Constants
module Alex.Templates.ArithTemplates

open Alex.Templates.TemplateTypes
module Patterns = Alex.Patterns.SemanticPatterns

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER BINARY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Integer addition: arith.addi
let addI = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.addi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Integer subtraction: arith.subi
let subI = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.subi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Integer multiplication: arith.muli
let mulI = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.muli %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Signed integer division: arith.divsi
let divSI = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.divsi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Signed integer remainder: arith.remsi
let remSI = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.remsi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER BITWISE OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Bitwise AND: arith.andi
let andI = simple "arith" "bitwise" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.andi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Bitwise OR: arith.ori
let orI = simple "arith" "bitwise" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.ori %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Bitwise XOR: arith.xori
let xorI = simple "arith" "bitwise" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.xori %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Shift left: arith.shli
let shlI = simple "arith" "bitwise" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.shli %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Arithmetic shift right (signed): arith.shrsi
let shrSI = simple "arith" "bitwise" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.shrsi %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

// ═══════════════════════════════════════════════════════════════════════════
// INTEGER COMPARISON OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Comparison predicate type
type CmpIPredicate = 
    | Eq | Ne | Slt | Sle | Sgt | Sge
    | Ult | Ule | Ugt | Uge

/// Convert predicate to MLIR string
let cmpIPredicateStr = function
    | Eq -> "eq" | Ne -> "ne"
    | Slt -> "slt" | Sle -> "sle" | Sgt -> "sgt" | Sge -> "sge"
    | Ult -> "ult" | Ule -> "ule" | Ugt -> "ugt" | Uge -> "uge"

/// Integer comparison: arith.cmpi
let cmpI predicate = simple "arith" "compare" (fun (p: CompareParams) ->
    sprintf "%s = arith.cmpi %s, %s, %s : %s" p.Result (cmpIPredicateStr predicate) p.Lhs p.Rhs p.Type)

/// Map Patterns.CompareOp to signed predicate
let signedPredicate (op: Patterns.CompareOp) : CmpIPredicate =
    match op with
    | Patterns.Lt -> Slt 
    | Patterns.Le -> Sle 
    | Patterns.Gt -> Sgt 
    | Patterns.Ge -> Sge 
    | Patterns.Eq -> Eq 
    | Patterns.Ne -> Ne

// ═══════════════════════════════════════════════════════════════════════════
// FLOAT BINARY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Float addition: arith.addf
let addF = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.addf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Float subtraction: arith.subf
let subF = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.subf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Float multiplication: arith.mulf
let mulF = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.mulf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

/// Float division: arith.divf
let divF = simple "arith" "binary" (fun (p: BinaryOpParams) ->
    sprintf "%s = arith.divf %s, %s : %s" p.Result p.Lhs p.Rhs p.Type)

// ═══════════════════════════════════════════════════════════════════════════
// FLOAT COMPARISON OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Float comparison predicate (ordered)
type CmpFPredicate =
    | OEq | ONE | OLt | OLe | OGt | OGe
    | UEq | UNE | ULt | ULe | UGt | UGe
    | Ord | Uno  // ordered / unordered

/// Convert predicate to MLIR string
let cmpFPredicateStr = function
    | OEq -> "oeq" | ONE -> "one"
    | OLt -> "olt" | OLe -> "ole" | OGt -> "ogt" | OGe -> "oge"
    | UEq -> "ueq" | UNE -> "une"
    | ULt -> "ult" | ULe -> "ule" | UGt -> "ugt" | UGe -> "uge"
    | Ord -> "ord" | Uno -> "uno"

/// Float comparison: arith.cmpf
let cmpF predicate = simple "arith" "compare" (fun (p: CompareParams) ->
    sprintf "%s = arith.cmpf %s, %s, %s : %s" p.Result (cmpFPredicateStr predicate) p.Lhs p.Rhs p.Type)

/// Map Patterns.CompareOp to ordered float predicate
let orderedPredicate (op: Patterns.CompareOp) : CmpFPredicate =
    match op with
    | Patterns.Lt -> OLt 
    | Patterns.Le -> OLe 
    | Patterns.Gt -> OGt 
    | Patterns.Ge -> OGe 
    | Patterns.Eq -> OEq 
    | Patterns.Ne -> ONE

// ═══════════════════════════════════════════════════════════════════════════
// UNARY OPERATIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Integer negation (0 - x)
/// Note: Requires constant 0 first, then subi
let negI = simple "arith" "unary" (fun (p: UnaryOpParams) ->
    // This returns just the subi; caller must emit the constant 0 separately
    sprintf "%s = arith.subi %%zero, %s : %s" p.Result p.Operand p.Type)

/// Float negation: arith.negf
let negF = simple "arith" "unary" (fun (p: UnaryOpParams) ->
    sprintf "%s = arith.negf %s : %s" p.Result p.Operand p.Type)

// ═══════════════════════════════════════════════════════════════════════════
// CONSTANTS
// ═══════════════════════════════════════════════════════════════════════════

/// Integer constant: arith.constant
let constantI = simple "arith" "constant" (fun (p: ConstantParams) ->
    sprintf "%s = arith.constant %s : %s" p.Result p.Value p.Type)

/// Float constant: arith.constant
let constantF = simple "arith" "constant" (fun (p: ConstantParams) ->
    sprintf "%s = arith.constant %s : %s" p.Result p.Value p.Type)

/// Boolean true constant
let constantTrue = simple "arith" "constant" (fun result ->
    sprintf "%s = arith.constant true" result)

/// Boolean false constant
let constantFalse = simple "arith" "constant" (fun result ->
    sprintf "%s = arith.constant false" result)

// ═══════════════════════════════════════════════════════════════════════════
// TYPE CONVERSIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Sign extend: arith.extsi
let extSI = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.extsi %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Zero extend: arith.extui
let extUI = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.extui %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Truncate: arith.trunci
let truncI = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.trunci %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Float to signed int: arith.fptosi
let fpToSI = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.fptosi %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Signed int to float: arith.sitofp
let siToFP = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.sitofp %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Float extend: arith.extf
let extF = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.extf %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

/// Float truncate: arith.truncf
let truncF = simple "arith" "conversion" (fun (p: ConversionParams) ->
    sprintf "%s = arith.truncf %s : %s to %s" p.Result p.Operand p.FromType p.ToType)

// ═══════════════════════════════════════════════════════════════════════════
// DISPATCH HELPERS - Select template based on witness
// ═══════════════════════════════════════════════════════════════════════════

/// Select integer binary template from BinaryArithOp witness
let intBinaryTemplate (op: Patterns.BinaryArithOp) =
    match op with
    | Patterns.Add -> addI
    | Patterns.Sub -> subI
    | Patterns.Mul -> mulI
    | Patterns.Div -> divSI
    | Patterns.Mod -> remSI
    | Patterns.BitAnd -> andI
    | Patterns.BitOr -> orI
    | Patterns.BitXor -> xorI
    | Patterns.ShiftLeft -> shlI
    | Patterns.ShiftRight -> shrSI

/// Select float binary template from BinaryArithOp witness
let floatBinaryTemplate (op: Patterns.BinaryArithOp) =
    match op with
    | Patterns.Add -> addF
    | Patterns.Sub -> subF
    | Patterns.Mul -> mulF
    | Patterns.Div -> divF
    | _ -> failwith "Unsupported float operation"

/// Select integer comparison template from CompareOp witness
let intCompareTemplate op = cmpI (signedPredicate op)

/// Select float comparison template from CompareOp witness
let floatCompareTemplate op = cmpF (orderedPredicate op)
