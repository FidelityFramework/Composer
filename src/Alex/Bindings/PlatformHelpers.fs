/// Platform Helper Functions - MLIR function definitions for common operations
///
/// ARCHITECTURAL FOUNDATION:
/// Instead of expanding complex operations inline at every call site,
/// we define them once as func.func definitions that get emitted at module level.
/// This follows the "platform as library" pattern used by libc but without runtime overhead.
///
/// These helpers use the func and scf dialects and will be lowered to LLVM by mlir-opt.
module Alex.Bindings.PlatformHelpers

open Alex.Traversal.MLIRZipper

// ═══════════════════════════════════════════════════════════════════════════
// Helper Names (used for registration and call emission)
// ═══════════════════════════════════════════════════════════════════════════

[<Literal>]
let ParseIntHelper = "fidelity_parse_int"

[<Literal>]
let ParseFloatHelper = "fidelity_parse_float"

[<Literal>]
let StringContainsCharHelper = "fidelity_string_contains_char"

// ═══════════════════════════════════════════════════════════════════════════
// Helper Bodies (MLIR func.func definitions)
// ═══════════════════════════════════════════════════════════════════════════

/// Parse a string to int64
/// Input: fat string (!llvm.struct<(ptr, i64)>)
/// Output: i64
let parseIntBody = """
func.func @fidelity_parse_int(%str: !llvm.struct<(ptr, i64)>) -> i64 {
  // Extract pointer and length
  %ptr = llvm.extractvalue %str[0] : !llvm.struct<(ptr, i64)>
  %len = llvm.extractvalue %str[1] : !llvm.struct<(ptr, i64)>

  // Constants
  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %c10 = arith.constant 10 : i64
  %c48 = arith.constant 48 : i64
  %c45_i8 = arith.constant 45 : i8

  // Check if first char is '-'
  %first_char = llvm.load %ptr : !llvm.ptr -> i8
  %is_neg = arith.cmpi eq, %first_char, %c45_i8 : i8

  // Starting position: 1 if negative, 0 if positive
  %start_pos = arith.select %is_neg, %c1, %c0 : i64

  // Parse digits with scf.while loop
  %result:2 = scf.while (%val = %c0, %pos = %start_pos) : (i64, i64) -> (i64, i64) {
    %in_bounds = arith.cmpi slt, %pos, %len : i64
    scf.condition(%in_bounds) %val, %pos : i64, i64
  } do {
  ^bb0(%val_arg: i64, %pos_arg: i64):
    %char_ptr = llvm.getelementptr %ptr[%pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char = llvm.load %char_ptr : !llvm.ptr -> i8
    %char_i64 = arith.extui %char : i8 to i64
    %digit = arith.subi %char_i64, %c48 : i64
    %val_times_10 = arith.muli %val_arg, %c10 : i64
    %new_val = arith.addi %val_times_10, %digit : i64
    %new_pos = arith.addi %pos_arg, %c1 : i64
    scf.yield %new_val, %new_pos : i64, i64
  }

  // Apply sign: if negative, negate result
  %negated = arith.subi %c0, %result#0 : i64
  %final = arith.select %is_neg, %negated, %result#0 : i64

  return %final : i64
}
"""

/// Parse a string to float64
/// Input: fat string (!llvm.struct<(ptr, i64)>)
/// Output: f64
let parseFloatBody = """
func.func @fidelity_parse_float(%str: !llvm.struct<(ptr, i64)>) -> f64 {
  // Extract pointer and length
  %ptr = llvm.extractvalue %str[0] : !llvm.struct<(ptr, i64)>
  %len = llvm.extractvalue %str[1] : !llvm.struct<(ptr, i64)>

  // Constants
  %c0_i64 = arith.constant 0 : i64
  %c1_i64 = arith.constant 1 : i64
  %c10_i64 = arith.constant 10 : i64
  %c48 = arith.constant 48 : i64
  %c45_i8 = arith.constant 45 : i8
  %c46_i8 = arith.constant 46 : i8
  %c0_f64 = arith.constant 0.0 : f64
  %c1_f64 = arith.constant 1.0 : f64
  %c10_f64 = arith.constant 10.0 : f64

  // Check if first char is '-'
  %first_char = llvm.load %ptr : !llvm.ptr -> i8
  %is_neg = arith.cmpi eq, %first_char, %c45_i8 : i8
  %start_pos = arith.select %is_neg, %c1_i64, %c0_i64 : i64

  // Parse integer part (before decimal point)
  %int_result:2 = scf.while (%val = %c0_i64, %pos = %start_pos) : (i64, i64) -> (i64, i64) {
    %in_bounds = arith.cmpi slt, %pos, %len : i64
    %char_ptr_check = llvm.getelementptr %ptr[%pos] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char_check = llvm.load %char_ptr_check : !llvm.ptr -> i8
    %not_dot = arith.cmpi ne, %char_check, %c46_i8 : i8
    %continue = arith.andi %in_bounds, %not_dot : i1
    scf.condition(%continue) %val, %pos : i64, i64
  } do {
  ^bb0(%val_arg: i64, %pos_arg: i64):
    %char_ptr = llvm.getelementptr %ptr[%pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char = llvm.load %char_ptr : !llvm.ptr -> i8
    %char_i64 = arith.extui %char : i8 to i64
    %digit = arith.subi %char_i64, %c48 : i64
    %val_times_10 = arith.muli %val_arg, %c10_i64 : i64
    %new_val = arith.addi %val_times_10, %digit : i64
    %new_pos = arith.addi %pos_arg, %c1_i64 : i64
    scf.yield %new_val, %new_pos : i64, i64
  }

  // Convert integer part to float
  %int_f64 = arith.sitofp %int_result#0 : i64 to f64

  // Check if we have decimal point
  %has_decimal = arith.cmpi slt, %int_result#1, %len : i64

  // Parse fractional part if present
  %frac_start = arith.addi %int_result#1, %c1_i64 : i64

  %frac_result:3 = scf.while (%frac = %c0_f64, %div = %c1_f64, %pos = %frac_start) : (f64, f64, i64) -> (f64, f64, i64) {
    %in_bounds = arith.cmpi slt, %pos, %len : i64
    %continue = arith.andi %has_decimal, %in_bounds : i1
    scf.condition(%continue) %frac, %div, %pos : f64, f64, i64
  } do {
  ^bb0(%frac_arg: f64, %div_arg: f64, %pos_arg: i64):
    %char_ptr = llvm.getelementptr %ptr[%pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char = llvm.load %char_ptr : !llvm.ptr -> i8
    %char_i64 = arith.extui %char : i8 to i64
    %digit_i64 = arith.subi %char_i64, %c48 : i64
    %digit_f64 = arith.sitofp %digit_i64 : i64 to f64
    %new_div = arith.mulf %div_arg, %c10_f64 : f64
    %scaled_digit = arith.divf %digit_f64, %new_div : f64
    %new_frac = arith.addf %frac_arg, %scaled_digit : f64
    %new_pos = arith.addi %pos_arg, %c1_i64 : i64
    scf.yield %new_frac, %new_div, %new_pos : f64, f64, i64
  }

  // Combine integer and fractional parts
  %combined = arith.addf %int_f64, %frac_result#0 : f64

  // Apply sign
  %negated = arith.negf %combined : f64
  %final = arith.select %is_neg, %negated, %combined : f64

  return %final : f64
}
"""

/// Check if string contains a character
/// Input: fat string, i8 char
/// Output: i1 (bool)
let stringContainsCharBody = """
func.func @fidelity_string_contains_char(%str: !llvm.struct<(ptr, i64)>, %target: i8) -> i1 {
  %ptr = llvm.extractvalue %str[0] : !llvm.struct<(ptr, i64)>
  %len = llvm.extractvalue %str[1] : !llvm.struct<(ptr, i64)>

  %c0 = arith.constant 0 : i64
  %c1 = arith.constant 1 : i64
  %false = arith.constant false

  %result:2 = scf.while (%found = %false, %pos = %c0) : (i1, i64) -> (i1, i64) {
    %in_bounds = arith.cmpi slt, %pos, %len : i64
    %not_found = arith.cmpi eq, %found, %false : i1
    %continue = arith.andi %in_bounds, %not_found : i1
    scf.condition(%continue) %found, %pos : i1, i64
  } do {
  ^bb0(%found_arg: i1, %pos_arg: i64):
    %char_ptr = llvm.getelementptr %ptr[%pos_arg] : (!llvm.ptr, i64) -> !llvm.ptr, i8
    %char = llvm.load %char_ptr : !llvm.ptr -> i8
    %is_match = arith.cmpi eq, %char, %target : i8
    %new_pos = arith.addi %pos_arg, %c1 : i64
    scf.yield %is_match, %new_pos : i1, i64
  }

  return %result#0 : i1
}
"""

// ═══════════════════════════════════════════════════════════════════════════
// Helper Registration Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Register the Parse.int helper and return updated zipper
let ensureParseIntHelper (zipper: MLIRZipper) : MLIRZipper =
    MLIRZipper.registerPlatformHelperWithBody ParseIntHelper parseIntBody zipper

/// Register the Parse.float helper and return updated zipper
let ensureParseFloatHelper (zipper: MLIRZipper) : MLIRZipper =
    MLIRZipper.registerPlatformHelperWithBody ParseFloatHelper parseFloatBody zipper

/// Register the String.containsChar helper and return updated zipper
let ensureStringContainsCharHelper (zipper: MLIRZipper) : MLIRZipper =
    MLIRZipper.registerPlatformHelperWithBody StringContainsCharHelper stringContainsCharBody zipper

// ═══════════════════════════════════════════════════════════════════════════
// Call Emission Functions
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a call to fidelity_parse_int and return the result SSA
let emitParseIntCall (strSSA: string) (zipper: MLIRZipper) : string * MLIRZipper =
    // Register the helper if needed
    let zipper1 = ensureParseIntHelper zipper

    // Emit the call
    let resultSSA, zipper2 = MLIRZipper.yieldSSA zipper1
    let callText = sprintf "%s = func.call @%s(%s) : (!llvm.struct<(ptr, i64)>) -> i64" resultSSA ParseIntHelper strSSA
    let zipper3 = MLIRZipper.witnessOpWithResult callText resultSSA (Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I64) zipper2
    resultSSA, zipper3

/// Emit a call to fidelity_parse_float and return the result SSA
let emitParseFloatCall (strSSA: string) (zipper: MLIRZipper) : string * MLIRZipper =
    // Register the helper if needed
    let zipper1 = ensureParseFloatHelper zipper

    // Emit the call
    let resultSSA, zipper2 = MLIRZipper.yieldSSA zipper1
    let callText = sprintf "%s = func.call @%s(%s) : (!llvm.struct<(ptr, i64)>) -> f64" resultSSA ParseFloatHelper strSSA
    let zipper3 = MLIRZipper.witnessOpWithResult callText resultSSA (Alex.CodeGeneration.MLIRTypes.Float Alex.CodeGeneration.MLIRTypes.F64) zipper2
    resultSSA, zipper3

/// Emit a call to fidelity_string_contains_char and return the result SSA
let emitStringContainsCharCall (strSSA: string) (charSSA: string) (zipper: MLIRZipper) : string * MLIRZipper =
    // Register the helper if needed
    let zipper1 = ensureStringContainsCharHelper zipper

    // Emit the call
    let resultSSA, zipper2 = MLIRZipper.yieldSSA zipper1
    let callText = sprintf "%s = func.call @%s(%s, %s) : (!llvm.struct<(ptr, i64)>, i8) -> i1" resultSSA StringContainsCharHelper strSSA charSSA
    let zipper3 = MLIRZipper.witnessOpWithResult callText resultSSA (Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I1) zipper2
    resultSSA, zipper3
