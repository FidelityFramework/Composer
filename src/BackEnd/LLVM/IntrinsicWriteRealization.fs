/// Linux AMD64 realization of the source-published bounded Sys.write boundary.
/// The backend preserves the portable call and realizes its exact declaration;
/// it neither inspects source expressions nor supplies missing source premises.
module BackEnd.LLVM.IntrinsicWriteRealization

open System
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Core.Types.Pipeline

let private declarations (input: BackEndInput) =
    input.Operations |> List.choose (function
        | MLIROp.FuncOp(IntrinsicWriteDecl declaration) -> Some declaration
        | _ -> None)

/// The source endpoint grants the operation; this profile checks its Linux
/// AMD64 realization. Neither libc nor a startup choice grants that capability.
let validate (input: BackEndInput) (context: BackEndContext) =
    let target = context.TargetTripleOverride |> Option.bind (TargetProfiles.linuxAmd64Syscalls context)
    declarations input |> List.tryPick (fun declaration ->
        if target.IsNone then
            Some $"Intrinsic '{declaration.Symbol}' has no admitted Linux AMD64 Sys.write realization for the selected target"
        elif input.PointerBits <> Ok 64 then
            Some $"Intrinsic '{declaration.Symbol}' witnessed pointer dimension disagrees with the selected 64-bit Sys.write ABI"
        elif declaration.SyscallNumber <> 1I then
            Some $"Intrinsic '{declaration.Symbol}' source endpoint does not declare Linux AMD64 write syscall 1"
        elif declaration.Fd <> BoundaryScalar.Integer(64, true)
             || declaration.Count <> BoundaryScalar.Integer(64, false)
             || declaration.Result <> BoundaryScalar.Integer(64, true) then
            Some $"Intrinsic '{declaration.Symbol}' requires unsupported scalar carriers; Linux AMD64 Sys.write realizes declared signed/unsigned/signed 64-bit fd/count/result"
        elif declaration.ByteRepresentation.Bits <> 8 || declaration.ByteRepresentation.Family <> "uint" then
            Some $"Intrinsic '{declaration.Symbol}' requires an explicitly declared unsigned octet buffer representation"
        else None)
    |> function Some message -> Error message | None -> Ok ()

/// One kernel write, with the caller's count and returned signed byte count or
/// negative errno. No retries, added newline, string scan or libc import belongs
/// to this operation. The descriptor's logical offset survives the boundary.
let private helper (declaration: IntrinsicWriteImport) =
    let body = """func.func private @$SYMBOL(%fd: i64, %buffer: memref<?xi8>, %count: i64) -> i64 {
  %base, %offset, %size, %stride = memref.extract_strided_metadata %buffer : memref<?xi8> -> memref<i8>, index, index, index
  %base_address = memref.extract_aligned_pointer_as_index %base : memref<i8> -> index
  %address = arith.addi %base_address, %offset : index
  %address_bits = arith.index_castui %address : index to i64
  %pointer = llvm.inttoptr %address_bits : i64 to !llvm.ptr
  %number = llvm.mlir.constant($NUMBER : i64) : i64
  %written = llvm.inline_asm has_side_effects "syscall", "={rax},0,{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory},~{flags}" %number, %fd, %pointer, %count : (i64, i64, !llvm.ptr, i64) -> i64
  func.return %written : i64
}"""
    body.Replace("$SYMBOL", symbolName declaration.Symbol).Replace("$NUMBER", string declaration.SyscallNumber)
    |> MLIROp.RawMLIR

let realize (context: BackEndContext) (input: BackEndInput) : Result<BackEndInput, string> =
    validate input context |> Result.bind (fun () ->
        if List.isEmpty (declarations input) then Ok input
        else
            try
                let operations = input.Operations |> List.map (function
                    | MLIROp.FuncOp(IntrinsicWriteDecl declaration) -> helper declaration
                    | operation -> operation)
                let text =
                    match input.ModuleName with
                    | Some name -> moduleToString input.PointerBits name operations
                    | None -> sprintf "module {\n%s\n}" (opsToString input.PointerBits operations "  ")
                Ok { input with Operations = operations; Text = text }
            with error -> Error ("LLVM Sys.write realization failed: " + error.Message))
