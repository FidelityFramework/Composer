/// ClosurePatterns - Closure and lambda operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns for lambda and closure operations.
/// Patterns compose Elements into semantic closure/lambda operations.
module Alex.Patterns.ClosurePatterns

open XParsec
open XParsec.Parsers     // preturn, fail
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.XParsec.Extensions // sequence combinator
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRAtomics
open Alex.Elements.MemRefElements
open Alex.Elements.ArithElements
open Alex.Elements.FuncElements
open Alex.Elements.HWElements   // pHWModule, pHWOutput (FPGA function wrapping)
open Alex.CodeGeneration.TypeMapping
open Core.Types.Dialects        // TargetPlatform (codata-dependent elision)
open Clef.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════
// FUNCTION DEFINITION
// ═══════════════════════════════════════════════════════════

/// Wrap already witnessed, ordered result components. A callable result stays
/// two independent values; this pattern performs no graph or closure analysis.
let pFunctionDefResults (visibility: FuncVisibility) (name: string) (parameters: (SSA * MLIRType) list)
                        (parameterNames: string list option) (resultTypes: MLIRType list)
                        (bodyOps: MLIROp list) (results: Val list) : PSGParser<MLIROp> =
    parser {
        do! ensure ((results |> List.map _.Type) = resultTypes) $"pFunctionDefResults: function '{name}' has mismatched result components"
        let! targetPlatform = getTargetPlatform
        match targetPlatform with
        | FPGA ->
            let names = parameterNames |> Option.defaultValue (parameters |> List.mapi (fun i _ -> sprintf "in%d" i))
            do! ensure (names.Length = parameters.Length) $"pFunctionDefResults: function '{name}' has mismatched input names"
            let inputs = List.map2 (fun name (_, ty) -> name, ty) names parameters
            let outputs = resultTypes |> List.mapi (fun index ty -> sprintf "result%d" index, ty)
            let! terminal = pHWOutput (results |> List.map (fun value -> value.SSA, value.Type))
            return! pHWModule name inputs outputs (bodyOps @ [terminal])
        | _ ->
            let! terminal = pFuncReturnResults results
            return! pFuncDefResults name parameters resultTypes (bodyOps @ [terminal]) visibility
    }

/// Create function definition (func.func for named calls, llvm.func for closures)
/// Coeffect-aware function definition wrapping.
/// Observes TargetPlatform to elide to func.func (CPU) or hw.module (FPGA).
/// Handles the function terminator internally: func.return (CPU) or hw.output (FPGA).
///
/// `paramNames`: optional port names for hw.module (defaults to "in0", "in1", ...)
/// `returnSSA`: the SSA of the return value (None for a unit function)
/// `unitReturnSSA`: for a unit function, the zero constant it returns, derived by SSAAssignment
let pFunctionDef (visibility: FuncVisibility) (name: string) (params': (SSA * MLIRType) list) (paramNames: string list option)
                 (retTy: MLIRType) (bodyOps: MLIROp list) (returnSSA: SSA option) (unitReturnSSA: SSA option)
                 : PSGParser<MLIROp> =
    parser {
        let! targetPlatform = getTargetPlatform
        match targetPlatform with
        | FPGA ->
            // hw.module with named input/output ports
            let names = paramNames |> Option.defaultValue (params' |> List.mapi (fun i _ -> sprintf "in%d" i))
            let inputs = List.map2 (fun pname (_, ty) -> (pname, ty)) names params'
            let outputs =
                match returnSSA with
                | Some _ -> [("result", retTy)]
                | None -> []
            let outputOp = MLIROp.HWOp (HWOp.HWOutput (match returnSSA with
                                                        | Some ssa -> [(ssa, retTy)]
                                                        | None -> []))
            let body = bodyOps @ [outputOp]
            return! pHWModule name inputs outputs body
        | _ ->
            // func.func with positional parameters
            // Preserve earlier witness diagnostics when a non-unit body failed
            // to produce a value, instead of throwing over the accumulated errors.
            do! ensure (returnSSA.IsSome || unitReturnSSA.IsSome) $"pFunctionDef: function '{name}' has no witnessed return value"
            // A unit function: returnSSA = None but retTy is concrete (e.g. i32), and
            // func.return needs an operand of that type. The zero constant it returns is the
            // value SSAAssignment derived for the Lambda (the first of its body's scope).
            let actualReturnSSA, extraOps =
                match returnSSA, unitReturnSSA with
                | Some _, _ -> returnSSA, []
                | None, Some zeroSSA ->
                    let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, retTy))
                    (Some zeroSSA, [zeroOp])
                | None, None ->
                    failwithf "pFunctionDef: function '%s' returns no value and SSAAssignment derived no unit-return value for its Lambda; the derivation covers every unit-typed body" name
            let! returnOp = pFuncReturn actualReturnSSA (Some retTy)
            let body = bodyOps @ extraOps @ [returnOp]
            return! pFuncDef name params' retTy body visibility
    }
