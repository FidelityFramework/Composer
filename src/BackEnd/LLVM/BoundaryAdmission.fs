/// Admit only the scalar C ABI for which the backend has a physical realization.
/// Baker already owns declarations, signedness, ranges and adaptation proofs.
module BackEnd.LLVM.BoundaryAdmission

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Core.Types.Pipeline

/// This initial SysV AMD64 profile uses unextended i32/i64 C carriers. Narrow
/// integers and Boolean need explicit argument AND result extension attributes;
/// admitting them from their signless MLIR types would lose source ABI facts.
/// LLVM ABI attributes: https://llvm.org/docs/LangRef.html#parameter-attributes
let validate (input: BackEndInput) (context: BackEndContext) =
    let imports = input.Operations |> List.choose (function
        | MLIROp.FuncOp(BoundaryFuncDecl declaration) -> Some declaration
        | _ -> None)
    let runtime = context.TargetTripleOverride |> Option.bind (TargetProfiles.libcAmd64 context)
    let unsupportedScalar = function
        | BoundaryScalar.Integer((32 | 64), _) -> false
        | _ -> true
    imports |> List.tryPick (fun declaration ->
        if runtime.IsNone then
            Some $"Boundary '{declaration.Symbol}' has no admitted C ABI realization for the selected target/runtime"
        elif input.PointerBits <> Ok 64 then
            Some $"Boundary '{declaration.Symbol}' witnessed pointer dimension disagrees with the selected 64-bit ABI"
        elif declaration.CallingConvention <> "CDecl" then
            Some $"Boundary '{declaration.Symbol}' requires an unimplemented calling convention: {declaration.CallingConvention}"
        else
            let scalars = List.map snd declaration.Parameters @ Option.toList declaration.Result
            scalars |> List.tryFind unsupportedScalar |> Option.map (fun scalar ->
                $"Boundary '{declaration.Symbol}' requires ABI realization for {scalar}; the LLVM SysV AMD64 profile currently realizes only signed/unsigned 32- and 64-bit scalar carriers"))
    |> function Some reason -> Error reason | None -> Ok ()
