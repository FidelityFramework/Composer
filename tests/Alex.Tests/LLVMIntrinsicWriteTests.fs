module Alex.Tests.LLVMIntrinsicWriteTests

open System
open System.Diagnostics
open System.IO
open System.Text
open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Core.Types.Pipeline
open BackEnd.LLVM.IntrinsicWriteRealization

let private context =
    { Timing = Core.Timing.silent(); OutputPath = "unused"; IntermediatesDir = None
      TargetTripleOverride = Some "x86_64-unknown-linux-gnu"; TargetPointerBits = Some 64; TargetCpu = None
      PlatformOS = Some "linux"; RuntimeModel = Some RuntimeModel.Libc
      DeploymentMode = Core.Types.Dialects.Console; EmitIntermediateOnly = false
      ExternLibraries = Set.empty; NativeLink = NativeLinkOptions.Empty
      EmbeddedTarget = None; XtensaTarget = None; Deploy = false }

// Isolated target realization fixture. IntrinsicWriteWitnessTests exercises
// the actual source publication, occurrence and artifact admission boundary.
let private declaration : IntrinsicWriteImport =
    { Identity = NodeId 1; Scope = NodeId 2; Symbol = "published_write"
      Fd = BoundaryScalar.Integer(64, true); Count = BoundaryScalar.Integer(64, false)
      Result = BoundaryScalar.Integer(64, true)
      ByteRepresentation = { Name="uint8"; Capability="native"; Family="uint"; Bits=8
                             MinMagnitude="0"; MaxMagnitude="255"; Boundary="wrap" }
      Core = NodeId 3; ReturnContract = NodeId 4; Endpoint = NodeId 5; Surface = NodeId 6
      SyscallNumber = 1I; Participants = Set.ofList [NodeId 1; NodeId 2; NodeId 3; NodeId 4; NodeId 5; NodeId 6] }

let private input operations =
    { Operations = operations; PointerBits = Ok 64; ModuleName = Some "intrinsic_write"
      Text = moduleToString (Ok 64) "intrinsic_write" operations; WritableStorage = []; Catalog = None }

let private declared declaration = input [MLIROp.FuncOp(IntrinsicWriteDecl declaration)]
let private good result = Result.defaultWith failwith result

[<Fact>]
let ``modules without Sys write preserve their exact target handoff`` () =
    let original = input []
    let actual = realize { context with PlatformOS = None; TargetTripleOverride = None } original |> good
    Assert.Same(original, actual)

[<Theory>]
[<InlineData("os")>]
[<InlineData("triple")>]
[<InlineData("pointer")>]
[<InlineData("architecture")>]
[<InlineData("foreign-os")>]
[<InlineData("x32")>]
let ``Sys write realization requires explicit selected Linux AMD64 target facts`` defect =
    let changed =
        match defect with
        | "os" -> { context with PlatformOS = None }
        | "triple" -> { context with TargetTripleOverride = None }
        | "pointer" -> { context with TargetPointerBits = Some 32 }
        | "architecture" -> { context with TargetTripleOverride = Some "aarch64-unknown-linux-gnu" }
        | "foreign-os" -> { context with TargetTripleOverride = Some "x86_64-pc-windows-gnu" }
        | _ -> { context with TargetTripleOverride = Some "x86_64-unknown-linux-gnux32" }
    match realize changed (declared declaration) with
    | Error message -> Assert.Contains("no admitted Linux AMD64 Sys.write", message)
    | Ok _ -> failwith "A different or undeclared target acquired the Sys.write realization"

[<Fact>]
let ``source syscall capability does not depend on libc or startup selection`` () =
    for runtime in [None; Some RuntimeModel.Bare; Some RuntimeModel.Libc] do
        for deployment in [Core.Types.Dialects.DeploymentMode.Console; Core.Types.Dialects.DeploymentMode.Freestanding; Core.Types.Dialects.DeploymentMode.Library] do
            validate (declared declaration) { context with RuntimeModel = runtime; DeploymentMode = deployment } |> good

[<Theory>]
[<InlineData("endpoint")>]
[<InlineData("fd")>]
[<InlineData("count")>]
[<InlineData("result")>]
[<InlineData("octet")>]
[<InlineData("byte-family")>]
let ``target realization refuses incompatible published syscall facts`` defect =
    let changed =
        match defect with
        | "endpoint" -> { declaration with SyscallNumber = 64I }
        | "fd" -> { declaration with Fd = BoundaryScalar.Integer(32, true) }
        | "count" -> { declaration with Count = BoundaryScalar.Integer(64, true) }
        | "result" -> { declaration with Result = BoundaryScalar.Integer(64, false) }
        | "octet" -> { declaration with ByteRepresentation = { declaration.ByteRepresentation with Bits = 16 } }
        | _ -> { declaration with ByteRepresentation = { declaration.ByteRepresentation with Family = "int" } }
    Assert.True(Result.isError (validate (declared changed) context))

[<Fact>]
let ``Sys write cannot replace the witnessed pointer dimension`` () =
    for pointer in [Ok 32; Error "undeclared"] do
        match validate { declared declaration with PointerBits = pointer } context with
        | Error message -> Assert.Contains("witnessed pointer dimension disagrees", message)
        | Ok () -> failwith "Sys.write replaced the witnessed pointer dimension"

[<Fact>]
let ``typed target realization preserves the exact portable call and descriptor offset`` () =
    let parameters = IntrinsicWriteAbi.parameters declaration |> List.mapi (fun ordinal ty -> Arg ordinal, ty)
    let result = { SSA = Alex.Traversal.Values.value (NodeId 10) 0; Type = BoundaryAbi.scalarType declaration.Result }
    let call = MLIROp.FuncOp(FuncCall([result], declaration.Symbol, parameters |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })))
    let wrapper = MLIROp.FuncOp(FuncDef("invoke", parameters, [result.Type], [call; MLIROp.FuncOp(Return [result])], FuncVisibility.Public))
    let original = input [MLIROp.FuncOp(IntrinsicWriteDecl declaration); wrapper]
    let actual = realize context original |> good
    Assert.Equal<MLIROp>(wrapper, actual.Operations[1])
    Assert.DoesNotContain("llvm.inline_asm", original.Text)
    Assert.Contains("memref.extract_strided_metadata", actual.Text)
    Assert.Contains("arith.addi %base_address, %offset", actual.Text)
    Assert.Contains("has_side_effects", actual.Text)
    Assert.Contains("~{memory}", actual.Text)
    Assert.DoesNotContain("@write(", actual.Text)
    Assert.Single(actual.Operations |> List.choose (function MLIROp.RawMLIR helper when helper.Contains("llvm.inline_asm") -> Some helper | _ -> None)) |> ignore
    MlirComponentTests.mlirOpt ["--verify-each"] actual.Text |> ignore

let private execute (program: string) =
    let start = ProcessStartInfo(program, UseShellExecute = false,
                                RedirectStandardOutput = true, RedirectStandardError = true,
                                StandardOutputEncoding = Encoding.UTF8, StandardErrorEncoding = Encoding.UTF8)
    use child = new Process(StartInfo = start)
    Assert.True(child.Start())
    let stdout, stderr = child.StandardOutput.ReadToEndAsync(), child.StandardError.ReadToEndAsync()
    if not (child.WaitForExit 20000) then
        child.Kill(true)
        child.WaitForExit()
        failwith "Sys.write executable did not terminate"
    child.ExitCode, stdout.GetAwaiter().GetResult(), stderr.GetAwaiter().GetResult()

[<Theory>]
[<InlineData(1L, 3L, 3L, "A\000B")>]
[<InlineData(1L, 2L, 2L, "A\000")>]
[<InlineData(1L, 0L, 0L, "")>]
[<InlineData(-1L, 3L, -9L, "")>]
let ``native Sys write preserves byte view count NUL and signed return`` (fd: int64) (count: int64) (expectedReturn: int64) (expectedOutput: string) =
    let native = """memref.global "private" constant @bytes : memref<8xi8> = dense<[88, 88, 65, 0, 66, 90, 90, 90]>
func.func @main() -> i32 {
  %storage = memref.get_global @bytes : memref<8xi8>
  %offset = arith.constant 2 : index
  %extent = arith.constant 3 : index
  %view = memref.view %storage[%offset][%extent] : memref<8xi8> to memref<?xi8>
  %fd = arith.constant $FD : i64
  %count = arith.constant $COUNT : i64
  %written = func.call @published_write(%fd, %view, %count) : (i64, memref<?xi8>, i64) -> i64
  %expected = arith.constant $EXPECTED : i64
  %equal = arith.cmpi eq, %written, %expected : i64
  %zero = arith.constant 0 : i32
  %one = arith.constant 1 : i32
  %status = arith.select %equal, %zero, %one : i32
  func.return %status : i32
}"""
    let native = native.Replace("$FD", string fd).Replace("$COUNT", string count).Replace("$EXPECTED", string expectedReturn)
    let realized = input [MLIROp.FuncOp(IntrinsicWriteDecl declaration); MLIROp.RawMLIR native] |> realize context |> good
    let directory = Path.Combine(Path.GetTempPath(), "clef-intrinsic-write-" + Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory directory |> ignore
    try
        let source, llvm, binary = Path.Combine(directory, "input.mlir"), Path.Combine(directory, "output.ll"), Path.Combine(directory, "program")
        File.WriteAllText(source, realized.Text)
        BackEnd.LLVM.Lowering.lowerToLLVM source llvm context.TargetTripleOverride.Value (Some 64) |> good
        BackEnd.LLVM.Codegen.compileToNative llvm binary context.TargetTripleOverride.Value Core.Types.Dialects.Console Set.empty NativeLinkOptions.Empty None |> good
        let exitCode, stdout, stderr = execute binary
        Assert.Equal(0, exitCode)
        Assert.Equal<string>(expectedOutput, stdout)
        Assert.Equal("", stderr)
    finally
        Directory.Delete(directory, true)
