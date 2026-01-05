/// Literal Witness - Witness literal values to MLIR
///
/// Observes literal PSG nodes and generates corresponding MLIR constants.
/// Follows the codata/photographer principle: observe, don't compute.
module Alex.Witnesses.LiteralWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper

// ═══════════════════════════════════════════════════════════════════════════
// MAIN WITNESS FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a literal value and generate corresponding MLIR
let witness (lit: LiteralValue) (zipper: MLIRZipper) : MLIRZipper * TransferResult =
    match lit with
    | LiteralValue.Unit ->
        let ssaName, zipper' = MLIRZipper.witnessConstant 0L I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Bool b ->
        let value = if b then 1L else 0L
        let ssaName, zipper' = MLIRZipper.witnessConstant value I1 zipper
        zipper', TRValue (ssaName, "i1")

    | LiteralValue.Int8 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I8 zipper
        zipper', TRValue (ssaName, "i8")

    | LiteralValue.Int16 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I16 zipper
        zipper', TRValue (ssaName, "i16")

    | LiteralValue.Int32 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Int64 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant n I64 zipper
        zipper', TRValue (ssaName, "i64")

    | LiteralValue.UInt8 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I8 zipper
        zipper', TRValue (ssaName, "i8")

    | LiteralValue.UInt16 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I16 zipper
        zipper', TRValue (ssaName, "i16")

    | LiteralValue.UInt32 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.UInt64 n -> 
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 n) I64 zipper
        zipper', TRValue (ssaName, "i64")

    | LiteralValue.Char c ->
        let ssaName, zipper' = MLIRZipper.witnessConstant (int64 c) I32 zipper
        zipper', TRValue (ssaName, "i32")

    | LiteralValue.Float32 f ->
        let ssaName, zipper' = MLIRZipper.yieldSSA zipper
        let text = sprintf "%s = arith.constant %e : f32" ssaName (float f)
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Float F32) zipper'
        zipper'', TRValue (ssaName, "f32")

    | LiteralValue.Float64 f ->
        let ssaName, zipper' = MLIRZipper.yieldSSA zipper
        let text = sprintf "%s = arith.constant %e : f64" ssaName f
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName (Float F64) zipper'
        zipper'', TRValue (ssaName, "f64")

    | LiteralValue.String s ->
        // Native string: fat pointer struct {ptr: *u8, len: i64}
        let globalName, zipper1 = MLIRZipper.observeStringLiteral s zipper
        let ptrSSA, zipper2 = MLIRZipper.witnessAddressOf globalName zipper1
        let lenSSA, zipper3 = MLIRZipper.witnessConstant (int64 s.Length) I64 zipper2
        
        // Build fat pointer struct
        let undefSSA, zipper4 = MLIRZipper.yieldSSA zipper3
        let undefText = sprintf "%s = llvm.mlir.undef : %s" undefSSA NativeStrTypeStr
        let zipper5 = MLIRZipper.witnessOpWithResult undefText undefSSA NativeStrType zipper4
        
        let withPtrSSA, zipper6 = MLIRZipper.yieldSSA zipper5
        let insertPtrText = sprintf "%s = llvm.insertvalue %s, %s[0] : %s" withPtrSSA ptrSSA undefSSA NativeStrTypeStr
        let zipper7 = MLIRZipper.witnessOpWithResult insertPtrText withPtrSSA NativeStrType zipper6
        
        let fatPtrSSA, zipper8 = MLIRZipper.yieldSSA zipper7
        let insertLenText = sprintf "%s = llvm.insertvalue %s, %s[1] : %s" fatPtrSSA lenSSA withPtrSSA NativeStrTypeStr
        let zipper9 = MLIRZipper.witnessOpWithResult insertLenText fatPtrSSA NativeStrType zipper8
        
        zipper9, TRValue (fatPtrSSA, NativeStrTypeStr)

    | _ ->
        zipper, TRError (sprintf "Unsupported literal: %A" lit)
