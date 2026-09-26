module Alex.Tests.LLVMRequirementTests

open System
open System.Diagnostics
open System.IO
open System.Text
open Xunit
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Core.Types.Pipeline
open BackEnd.LLVM.RequirementRealization

let private triple = "x86_64-unknown-linux-gnu"

let private context =
    { OutputPath = "unused"; IntermediatesDir = None; TargetTripleOverride = Some triple
      TargetPointerBits = Some 64; TargetCpu = None; PlatformOS = Some "linux"
      RuntimeModel = Some Clef.Compiler.NativeTypedTree.NativeTypes.RuntimeModel.Libc
      DeploymentMode = Core.Types.Dialects.Console; EmitIntermediateOnly = false
      ExternLibraries = Set.empty; NativeLink = NativeLinkOptions.Empty
      EmbeddedTarget = None; XtensaTarget = None; Deploy = false }

let private input operations =
    { Operations = operations; PointerBits = Ok 64; ModuleName = Some "requirement_backend"
      Text = moduleToString (Ok 64) "requirement_backend" operations; WritableStorage = [] }

let private required message =
    input [MLIROp.FuncOp(FuncDef("check", [(Arg 0, TInt(IntWidth 1))], [],
        [MLIROp.Assert(Arg 0, message); MLIROp.FuncOp(Return [])], FuncVisibility.Public))]

let private succeed = function Ok value -> value | Error message -> failwith message

[<Theory>]
[<InlineData("os")>]
[<InlineData("runtime")>]
[<InlineData("pointer")>]
[<InlineData("deployment")>]
[<InlineData("architecture")>]
[<InlineData("foreign-os")>]
[<InlineData("foreign-os-linux-suffix")>]
[<InlineData("x32")>]
let ``diagnostic realization requires the selected process runtime and exact target ABI`` defect =
    let changed, target =
        match defect with
        | "os" -> { context with PlatformOS = None }, triple
        | "runtime" -> { context with RuntimeModel = None }, triple
        | "pointer" -> { context with TargetPointerBits = Some 32 }, triple
        | "deployment" -> { context with DeploymentMode = Core.Types.Dialects.Embedded }, triple
        | "architecture" -> context, "aarch64-unknown-linux-gnu"
        | "foreign-os" -> context, "x86_64-pc-windows-gnu"
        | "foreign-os-linux-suffix" -> context, "x86_64-pc-windows-linux"
        | _ -> context, "x86_64-unknown-linux-gnux32"
    Assert.True((selectRuntime context triple).IsSome)
    let selected = selectRuntime changed target
    Assert.True(selected.IsNone)
    match realize selected (required "must survive") with
    | Error message -> Assert.Contains("no admitted diagnostic-and-termination", message)
    | Ok _ -> failwith "An undeclared runtime acquired a diagnostic sink."

[<Fact>]
let ``modules without requirements retain their exact portable handoff`` () =
    let original = input [MLIROp.FuncOp(FuncDef("plain", [], [], [MLIROp.FuncOp(Return [])], FuncVisibility.Public))]
    let actual = realize None original |> succeed
    Assert.True(Object.ReferenceEquals(original, actual))

[<Fact>]
let ``target realization preserves portable assertions and original ordered condition operands`` () =
    let collision = MLIROp.FuncOp(FuncDef("__clef_requirement_0", [], [], [MLIROp.FuncOp(Return [])], FuncVisibility.Private))
    let escapedCollision = MLIROp.RawMLIR "func.func private @\"__clef_requirement_\\31\"() { func.return }"
    let functionBody =
        [ MLIROp.Assert(Arg 0, "first")
          MLIROp.SCFOp(If(Arg 1, [MLIROp.Assert(Arg 2, "second")], None, None))
          MLIROp.FuncOp(Return []) ]
    let original = input [collision; escapedCollision; MLIROp.FuncOp(FuncDef("ordered", [Arg 0, TInt(IntWidth 1); Arg 1, TInt(IntWidth 1); Arg 2, TInt(IntWidth 1)], [], functionBody, FuncVisibility.Public))]
    let actual = realize (selectRuntime context triple) original |> succeed
    Assert.Contains("cf.assert %arg0", original.Text)
    Assert.DoesNotContain("cf.assert", actual.Text)
    let body = actual.Operations |> List.pick (function MLIROp.FuncOp(FuncDef("ordered", _, _, body, _)) -> Some body | _ -> None)
    match body with
    | [MLIROp.FuncOp(FuncCall([], first, [{ SSA = Arg 0; Type = TInt(IntWidth 1) }]));
       MLIROp.SCFOp(If(Arg 1, [MLIROp.FuncOp(FuncCall([], second, [{ SSA = Arg 2; Type = TInt(IntWidth 1) }]))], None, None));
       MLIROp.FuncOp(Return [])] ->
        Assert.NotEqual<string>("__clef_requirement_0", first)
        Assert.NotEqual<string>("__clef_requirement_1", first)
        Assert.NotEqual<string>(first, second)
    | other -> failwithf "Target realization changed assertion order/conditions: %A" other
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] actual.Text
    Assert.Contains("has_side_effects", verified)
    let lowered = MlirComponentTests.mlirOpt
                      ["--verify-each"; "--pass-pipeline=builtin.module(convert-scf-to-cf,convert-cf-to-llvm,convert-arith-to-llvm,convert-func-to-llvm,reconcile-unrealized-casts)"] verified
    Assert.Contains("llvm.unreachable", lowered)
    Assert.DoesNotContain("@abort", lowered)
    Assert.Contains("llvm.icmp \"eq\"", lowered)

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
        failwith "Requirement executable did not terminate."
    child.ExitCode, stdout.GetAwaiter().GetResult(), stderr.GetAwaiter().GetResult()

[<Theory>]
[<InlineData(true, false)>]
[<InlineData(false, false)>]
[<InlineData(false, true)>]
let ``native runtime preserves exact diagnostic bytes and never reports a satisfied requirement`` satisfied large =
    let message =
        let special = "quoted \"requirement\" \\ newline\nUnicode λ 雪 and NUL\000end"
        if large then String.replicate 8192 special else special
    let boolean, result = V(1, 0), V(2, 0)
    let operations =
        [MLIROp.FuncOp(FuncDef("main", [], [TInt(IntWidth 32)],
            [MLIROp.ArithOp(ConstI(boolean, (if satisfied then 1L else 0L), TInt(IntWidth 1)))
             MLIROp.Assert(boolean, message)
             MLIROp.ArithOp(ConstI(result, 43L, TInt(IntWidth 32)))
             MLIROp.FuncOp(Return [{ SSA = result; Type = TInt(IntWidth 32) }])], FuncVisibility.Public))]
    let realized = realize (selectRuntime context triple) (input operations) |> succeed
    let directory = Path.Combine(Path.GetTempPath(), "composer requirement " + Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory directory |> ignore
    try
        let source, llvm, binary = Path.Combine(directory, "runtime.mlir"), Path.Combine(directory, "runtime.ll"), Path.Combine(directory, "program")
        File.WriteAllText(source, realized.Text)
        BackEnd.LLVM.Lowering.lowerToLLVM source llvm triple (Some 64) |> succeed
        BackEnd.LLVM.Codegen.compileToNative llvm binary triple Core.Types.Dialects.Console Set.empty NativeLinkOptions.Empty None |> succeed
        let exitCode, stdout, stderr = execute binary
        Assert.Equal((if satisfied then 43 else 1), exitCode)
        Assert.Equal("", stdout)
        Assert.Equal((if satisfied then "" else message + "\n"), stderr)
    finally
        Directory.Delete(directory, true)
