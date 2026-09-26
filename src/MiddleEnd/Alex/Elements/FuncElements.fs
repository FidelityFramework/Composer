/// FuncElements - Atomic Func dialect operation emission
///
/// INTERNAL: Witnesses CANNOT import this. Only Patterns can.
/// Provides ALL Func dialect operations from Types.fs.
module internal Alex.Elements.FuncElements

open XParsec
open XParsec.Parsers     // getUserState
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

// All Elements use XParsec state for platform/type context

// Note: FuncVisibility is assumed to be defined in Types.fs

// ═══════════════════════════════════════════════════════════
// FUNCTION DEFINITION
// ═══════════════════════════════════════════════════════════

let private scalarResultTypes = function TVoid -> [] | result -> [result]

let private pResultTypes (results: MLIRType list) =
    ensure (not (List.contains TVoid results)) "Function result lists use [] for no result; TVoid is not a value type"

let private pScalarResult (result: SSA option) (retTy: MLIRType) : PSGParser<Val list> =
    parser {
        match result, retTy with
        | None, TVoid -> return []
        | Some _, TVoid -> return! fail (Message "A void function call cannot define an SSA result")
        | Some ssa, ty -> return [{ SSA = ssa; Type = ty }]
        | None, _ -> return! fail (Message "A nonvoid function call requires its typed SSA result, even when unused")
    }

let pFuncDefResults (name: string) (args: (SSA * MLIRType) list) (resultTypes: MLIRType list)
                 (body: MLIROp list) (visibility: FuncVisibility) : PSGParser<MLIROp> =
    parser {
        do! pResultTypes resultTypes
        return MLIROp.FuncOp (FuncOp.FuncDef (name, args, resultTypes, body, visibility))
    }

let pFuncDef name args retTy body visibility =
    pFuncDefResults name args (scalarResultTypes retTy) body visibility

// ═══════════════════════════════════════════════════════════
// FUNCTION DECLARATION (external)
// ═══════════════════════════════════════════════════════════

let pFuncDeclResults (name: string) (argTypes: MLIRType list) (resultTypes: MLIRType list)
                  (visibility: FuncVisibility) : PSGParser<MLIROp> =
    parser {
        do! pResultTypes resultTypes
        return MLIROp.FuncOp (FuncOp.FuncDecl (name, argTypes, resultTypes, visibility, []))
    }

let pFuncDecl name argTypes retTy visibility =
    pFuncDeclResults name argTypes (scalarResultTypes retTy) visibility

let pFuncDeclByvalResults (name: string) (argTypes: MLIRType list) (resultTypes: MLIRType list)
                       (visibility: FuncVisibility) (byvalParams: ByvalParam list) : PSGParser<MLIROp> =
    parser {
        do! pResultTypes resultTypes
        return MLIROp.FuncOp (FuncOp.FuncDecl (name, argTypes, resultTypes, visibility, byvalParams))
    }

let pFuncDeclByval name argTypes retTy visibility byvalParams =
    pFuncDeclByvalResults name argTypes (scalarResultTypes retTy) visibility byvalParams

// ═══════════════════════════════════════════════════════════
// DIRECT CALL
// ═══════════════════════════════════════════════════════════

let pFuncCallResults (results: Val list) (func: string) (args: Val list) : PSGParser<MLIROp> =
    parser {
        do! pResultTypes (List.map (fun (value: Val) -> value.Type) results)
        return MLIROp.FuncOp (FuncOp.FuncCall (results, func, args))
    }

let pFuncCall result func args retTy =
    parser {
        let! results = pScalarResult result retTy
        return! pFuncCallResults results func args
    }

// ═══════════════════════════════════════════════════════════
// INDIRECT CALL
// ═══════════════════════════════════════════════════════════

let pFuncCallIndirectResults (results: Val list) (callee: SSA) (args: Val list) : PSGParser<MLIROp> =
    parser {
        do! pResultTypes (List.map (fun (value: Val) -> value.Type) results)
        return MLIROp.FuncOp (FuncOp.FuncCallIndirect (results, callee, args))
    }

let pFuncCallIndirect result callee args retTy =
    parser {
        let! results = pScalarResult result retTy
        return! pFuncCallIndirectResults results callee args
    }

// ═══════════════════════════════════════════════════════════
// FUNCTION CONSTANT (pointer to function)
// ═══════════════════════════════════════════════════════════

let pFuncConstant (result: SSA) (funcName: string) (funcTy: MLIRType) : PSGParser<MLIROp> =
    parser {
        return MLIROp.FuncOp (FuncOp.FuncConstant (result, funcName, funcTy))
    }

// ═══════════════════════════════════════════════════════════
// RETURN
// ═══════════════════════════════════════════════════════════

let pFuncReturnResults (results: Val list) : PSGParser<MLIROp> =
    parser {
        do! pResultTypes (List.map (fun (value: Val) -> value.Type) results)
        return MLIROp.FuncOp (FuncOp.Return results)
    }

let pFuncReturn (valueOpt: SSA option) (tyOpt: MLIRType option) : PSGParser<MLIROp> =
    parser {
        match valueOpt, tyOpt with
        | None, None | None, Some TVoid -> return! pFuncReturnResults []
        | Some value, Some ty when ty <> TVoid -> return! pFuncReturnResults [{ SSA = value; Type = ty }]
        | _ -> return! fail (Message "Function return requires one type per value; a void return has neither")
    }
