/// Realize the diagnostic effect of an already witnessed portable assertion.
/// This target pass consumes typed operations, never source nodes or MLIR text.
module BackEnd.LLVM.RequirementRealization

open System
open System.Text
open System.Text.RegularExpressions
open System.Collections.Generic
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Core.Types.Pipeline

/// Linux AMD64 process ABI, selected from checked platform/runtime facts.
/// Fidelity.Platform/Environments/Linux/x86_64/Environment.clef declares
/// sysWrite=1, sysExitGroup=231 and negative-return errno; Console.clef declares
/// STDERR=2. The syscall register convention belongs to this backend profile.
type LinuxX64Process = private | LinuxX64Process

let selectRuntime (context: BackEndContext) (triple: string) =
    let processDeployment =
        match context.DeploymentMode with
        | Core.Types.Dialects.DeploymentMode.Console | Core.Types.Dialects.DeploymentMode.Library -> true
        | _ -> false
    let parts = triple.Split '-'
    let linuxAbi =
        match parts with
        | [| "x86_64"; "linux"; environment |]
        | [| "x86_64"; _; "linux"; environment |] -> environment = "gnu" || environment = "musl"
        | _ -> false
    if context.PlatformOS = Some "linux"
       && context.RuntimeModel = Some Clef.Compiler.NativeTypedTree.NativeTypes.RuntimeModel.Libc
       && context.TargetPointerBits = Some 64 && processDeployment && linuxAbi then
        Some LinuxX64Process
    else None

/// UTF8 is encoded byte-for-byte, including NUL, quotes and newlines. A final
/// newline is the process diagnostic convention; no C string or format string
/// interpretation can truncate or reinterpret the specified diagnostic.
let private diagnosticBytes message =
    let encoding = UTF8Encoding(false, true)
    encoding.GetBytes(message + "\n")

/// Backend-owned target helper serialization. The condition is the original
/// witnessed i1; its success edge performs no diagnostic IO. RawMLIR here is
/// solely the output of this fixed target serializer, never parsed input text.
let private helper name dataName (bytes: byte array) =
    let data = bytes |> Array.map (fun value -> sprintf "\\%02X" value) |> String.concat ""
    let globalText = sprintf "llvm.mlir.global private constant @%s(\"%s\") : !llvm.array<%d x i8>" dataName data bytes.Length
    let body = """func.func private @$FUNCTION(%condition: i1) {
  cf.cond_br %condition, ^satisfied, ^failed
^satisfied:
  func.return
^failed:
  %data = llvm.mlir.addressof @$DATA : !llvm.ptr
  %length = llvm.mlir.constant($LENGTH : i64) : i64
  %write_number = llvm.mlir.constant(1 : i64) : i64
  %stderr = llvm.mlir.constant(2 : i64) : i64
  %zero = llvm.mlir.constant(0 : i64) : i64
  %eintr = llvm.mlir.constant(-4 : i64) : i64
  cf.br ^write(%data, %length : !llvm.ptr, i64)
^write(%cursor: !llvm.ptr, %remaining: i64):
  %written = llvm.inline_asm has_side_effects "syscall", "={rax},0,{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory},~{flags}" %write_number, %stderr, %cursor, %remaining : (i64, i64, !llvm.ptr, i64) -> i64
  %interrupted = llvm.icmp "eq" %written, %eintr : i64
  cf.cond_br %interrupted, ^write(%cursor, %remaining : !llvm.ptr, i64), ^progress
^progress:
  %positive = llvm.icmp "sgt" %written, %zero : i64
  cf.cond_br %positive, ^advance, ^terminate
^advance:
  %next = llvm.getelementptr %cursor[%written] : (!llvm.ptr, i64) -> !llvm.ptr, i8
  %left = llvm.sub %remaining, %written : i64
  %complete = llvm.icmp "eq" %left, %zero : i64
  cf.cond_br %complete, ^terminate, ^write(%next, %left : !llvm.ptr, i64)
^terminate:
  %exit_number = llvm.mlir.constant(231 : i64) : i64
  %status = llvm.mlir.constant(1 : i64) : i64
  %exited = llvm.inline_asm has_side_effects "syscall", "={rax},0,{rdi},~{rcx},~{r11},~{memory},~{flags}" %exit_number, %status : (i64, i64) -> i64
  llvm.unreachable
}"""
    let body = body.Replace("$FUNCTION", name).Replace("$DATA", dataName).Replace("$LENGTH", string bytes.Length)
    [MLIROp.RawMLIR globalText; MLIROp.RawMLIR body]

let realize (runtime: LinuxX64Process option) (input: BackEndInput) : Result<BackEndInput, string> =
    try
        // Reserve typed symbols, and the literal spelling in opaque backend ops
        // conservatively. This is collision avoidance only, not IR parsing or
        // semantic reconstruction; an opaque occurrence can only reserve a name.
        let names = HashSet<string>(StringComparer.Ordinal)
        let opaque = ResizeArray<string>()
        let reserveOpaque text =
            // Quoted MLIR symbols can encode ASCII bytes as \HH. Decoding
            // these solely for conservative name reservation also covers
            // escaped spellings, without interpreting any opaque operation.
            let decoded = Regex.Replace(text, @"\\([0-9a-fA-F]{2})",
                              MatchEvaluator(fun occurrence -> string (char (Convert.ToInt32(occurrence.Groups[1].Value, 16)))))
            opaque.Add decoded
        let rec inspect operation =
            match operation with
            | MLIROp.FuncOp(FuncDef(name, _, _, body, _))
            | MLIROp.NoUnwindFunction(FuncDef(name, _, _, body, _)) ->
                names.Add name |> ignore
                List.iter inspect body
            | MLIROp.FuncOp(FuncDecl(name, _, _, _, _))
            | MLIROp.FuncOp(FuncCall(_, name, _))
            | MLIROp.FuncOp(FuncConstant(_, name, _))
            | MLIROp.MemRefOp(GetGlobal(_, name, _))
            | MLIROp.GlobalString(name, _, _, _)
            | MLIROp.GlobalBytePool(name, _, _, _)
            | MLIROp.GlobalMemref(name, _, _) -> names.Add name |> ignore
            | MLIROp.HWOp(HWModule(name, _, _, body)) ->
                names.Add name |> ignore
                List.iter inspect body
            | MLIROp.SCFOp(If(_, yes, no, _)) -> List.iter inspect yes; no |> Option.iter (List.iter inspect)
            | MLIROp.SCFOp(While(condition, body)) -> List.iter inspect condition; List.iter inspect body
            | MLIROp.SCFOp(For(_, _, _, body))
            | MLIROp.Block(_, body)
            | MLIROp.Region body -> List.iter inspect body
            | MLIROp.SCFOp(IndexSwitch(_, cases, fallback, _)) ->
                cases |> List.iter (snd >> List.iter inspect)
                List.iter inspect fallback
            | MLIROp.RawMLIR text -> reserveOpaque text
            | _ -> ()
        List.iter inspect input.Operations
        let mutable next = 0
        let fresh () =
            let rec choose () =
                let name = sprintf "__clef_requirement_%d" next
                next <- next + 1
                let dataName = name + "_bytes"
                if names.Contains name || names.Contains dataName
                   || (opaque |> Seq.exists (fun text -> text.Contains(name, StringComparison.Ordinal))) then choose ()
                else
                    names.Add name |> ignore
                    names.Add dataName |> ignore
                    name, dataName
            choose ()
        let definitions = ResizeArray<MLIROp>()
        let diagnostics = Dictionary<string, string>(StringComparer.Ordinal)
        let rec transform operation =
            let map = List.map transform
            match operation with
            | MLIROp.Assert(condition, message) ->
                if runtime.IsNone then
                    failwith "The selected platform has no admitted diagnostic-and-termination realization for a required assertion."
                if input.PointerBits <> Ok 64 then
                    failwith "Linux AMD64 diagnostic realization requires the witnessed 64-bit Pointer dimension."
                let name =
                    match diagnostics.TryGetValue message with
                    | true, name -> name
                    | _ ->
                        let name, dataName = fresh ()
                        definitions.AddRange(helper name dataName (diagnosticBytes message))
                        diagnostics.Add(message, name)
                        name
                MLIROp.FuncOp(FuncCall([], name, [{ SSA = condition; Type = TInt(IntWidth 1) }]))
            | MLIROp.FuncOp(FuncDef(name, arguments, results, body, visibility)) ->
                MLIROp.FuncOp(FuncDef(name, arguments, results, map body, visibility))
            | MLIROp.NoUnwindFunction(FuncDef(name, arguments, results, body, visibility)) ->
                MLIROp.NoUnwindFunction(FuncDef(name, arguments, results, map body, visibility))
            | MLIROp.SCFOp(If(condition, yes, no, result)) ->
                MLIROp.SCFOp(If(condition, map yes, Option.map map no, result))
            | MLIROp.SCFOp(While(condition, body)) -> MLIROp.SCFOp(While(map condition, map body))
            | MLIROp.SCFOp(For(lower, upper, step, body)) -> MLIROp.SCFOp(For(lower, upper, step, map body))
            | MLIROp.SCFOp(IndexSwitch(selector, cases, fallback, results)) ->
                MLIROp.SCFOp(IndexSwitch(selector, cases |> List.map (fun (tag, body) -> tag, map body), map fallback, results))
            | MLIROp.Block(label, body) -> MLIROp.Block(label, map body)
            | MLIROp.Region body -> MLIROp.Region(map body)
            | MLIROp.HWOp(HWModule(name, inputs, outputs, body)) ->
                MLIROp.HWOp(HWModule(name, inputs, outputs, map body))
            | _ -> operation
        let transformed = List.map transform input.Operations
        if definitions.Count = 0 then Ok input
        else
            let operations = List.ofSeq definitions @ transformed
            let text =
                match input.ModuleName with
                | Some name -> moduleToString input.PointerBits name operations
                | None -> sprintf "module {\n%s\n}" (opsToString input.PointerBits operations "  ")
            Ok { input with Operations = operations; Text = text }
    with error -> Error ("LLVM requirement realization failed: " + error.Message)
