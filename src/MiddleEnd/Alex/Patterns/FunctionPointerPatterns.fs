module Alex.Patterns.FunctionPointerPatterns

open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
open XParsec
open XParsec.Parsers

let pFunctionAddress (_site: NodeId) (_symbol: string) (_parameters: MLIRType list) (_result: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    fail (Message "A native function address requires a source-published target ABI realization; function/index repair casts are retired")

let pFunctionPointerCall (_site: NodeId) (_pointer: SSA) (_arguments: Val list) (_result: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    fail (Message "A native function-pointer call requires a source-published target ABI realization; function/index repair casts are retired")
