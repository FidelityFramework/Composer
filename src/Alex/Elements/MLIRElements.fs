/// MLIRElements - Atomic MLIR operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides atomic MLIR op construction via XParsec-compatible interface.
module internal Alex.Elements.MLIRElements

open XParsec
open XParsec.Parsers     // getUserState
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.CodeGeneration.TypeMapping
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types

// Elements use XParsec state for platform/type context

/// Emit ExtractValue with XParsec state threading
let pEmitExtractValue (ssa: SSA) (value: SSA) (indices: int list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.ExtractValue (ssa, value, indices, ty))
    }

/// Emit InsertValue with XParsec state threading
let pEmitInsertValue (ssa: SSA) (struct_: SSA) (value: SSA) (indices: int list) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.InsertValue (ssa, struct_, value, indices, ty))
    }

/// Emit Undef with XParsec state threading
let pEmitUndef (ssa: SSA) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.LLVMOp (LLVMOp.Undef (ssa, ty))
    }

/// Emit ConstI with XParsec state threading
let pEmitConstI (ssa: SSA) (value: int64) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        let ty = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type
        return MLIROp.ArithOp (ArithOp.ConstI (ssa, value, ty))
    }
