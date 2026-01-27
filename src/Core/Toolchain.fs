/// Toolchain - COMPATIBILITY WRAPPER for MLIR/LLVM compilation
///
/// DEPRECATED: This module exists for backward compatibility only.
/// New code should call MiddleEnd.MLIROpt.Lowering and BackEnd.LLVM.Codegen directly.
///
/// Will be removed once all call sites are updated to use the proper modules.
module Core.Toolchain

open Core.Types.Dialects

/// Lower MLIR to LLVM IR using mlir-opt and mlir-translate
/// DEPRECATED: Use MiddleEnd.MLIROpt.Lowering.lowerToLLVM instead
let lowerMLIRToLLVM (mlirPath: string) (llvmPath: string) : Result<unit, string> =
    MiddleEnd.MLIROpt.Lowering.lowerToLLVM mlirPath llvmPath

/// Compile LLVM IR to native binary using llc and clang
/// DEPRECATED: Use BackEnd.LLVM.Codegen.compileToNative instead
let compileLLVMToNative
    (llvmPath: string)
    (outputPath: string)
    (targetTriple: string)
    (outputKind: OutputKind) : Result<unit, string> =
    BackEnd.LLVM.Codegen.compileToNative llvmPath outputPath targetTriple outputKind

/// Get default target triple based on host platform
/// DEPRECATED: Use BackEnd.LLVM.Codegen.getDefaultTarget instead
let getDefaultTarget() =
    BackEnd.LLVM.Codegen.getDefaultTarget()
