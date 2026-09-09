module Alex.Patterns.FunctionPointerPatterns

open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.Values
open Alex.XParsec.PSGCombinators
open Alex.Elements.MLIRAtomics
open XParsec

let pFunctionAddress (site: NodeId) symbol parameters result : PSGParser<MLIROp list * TransferResult> =
    parser {
        let address = value site 0
        let pointer = value site 1
        return [ MLIROp.FuncOp (FuncOp.FuncConstant(address, symbol, TFunc(parameters, result)))
                 MLIROp.FuncOp (FuncOp.FuncToIndex(pointer, address, parameters, result)) ],
               TRValue { SSA = pointer; Type = TIndex }
    }

let pFunctionPointerCall (site: NodeId) pointer (arguments: Val list) result : PSGParser<MLIROp list * TransferResult> =
    parser {
        let address = value site 0
        let output = value site 1
        let operations =
            [ MLIROp.FuncOp (FuncOp.IndexToFunc(address, pointer, List.map (fun v -> v.Type) arguments, result))
              MLIROp.FuncOp (FuncOp.FuncCallIndirect((if result = TVoid then None else Some output), address, arguments, result)) ]
        if result = TVoid then
            let! unitTy = pMapType Types.unitType
            let! unitValue = pConstI output 0L unitTy
            return operations @ [unitValue], TRValue { SSA = output; Type = unitTy }
        else
            return operations, TRValue { SSA = output; Type = result }
    }
