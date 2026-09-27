module Alex.Tests.BoundaryAbiTests

open System
open System.Diagnostics
open System.IO
open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Core.Types.Pipeline

let private context =
    { Timing = Core.Timing.silent(); OutputPath = "unused"; IntermediatesDir = None
      TargetTripleOverride = Some "x86_64-unknown-linux-gnu"; TargetPointerBits = Some 64; TargetCpu = None
      PlatformOS = Some "linux"; RuntimeModel = Some Clef.Compiler.NativeTypedTree.NativeTypes.RuntimeModel.Libc
      DeploymentMode = Core.Types.Dialects.Console; EmitIntermediateOnly = false
      ExternLibraries = Set.singleton "c"; NativeLink = NativeLinkOptions.Empty
      EmbeddedTarget = None; XtensaTarget = None; Deploy = false }

// Isolated backend contract fixture. Production correspondence is exercised by
// ForeignDeclarationTests using actual source publication and the catalog.
let private declaration scalar =
    { Identity = NodeId 1; Binding = NodeId 2; Scope = NodeId 3; Library = "c"; Symbol = "external"
      CallingConvention = "CDecl"; DeclarationPath = [NodeId 2; NodeId 1]
      Parameters = [NodeId 4, scalar]; Result = Some scalar
      Participants = Set.empty; SourceTypes = Map.empty; DeclarationFacts = Map.empty }

let private input operations =
    { Operations = operations; PointerBits = Ok 64; ModuleName = Some "boundary_abi"
      Text = moduleToString (Ok 64) "boundary_abi" operations; WritableStorage = []; Catalog = None }

let private checkedInput declaration = input [MLIROp.FuncOp(BoundaryFuncDecl declaration)]
let private good = Result.defaultWith failwith

[<Theory>]
[<InlineData(1, false)>]
[<InlineData(8, true)>]
[<InlineData(8, false)>]
[<InlineData(16, true)>]
[<InlineData(16, false)>]
[<InlineData(24, true)>]
[<InlineData(128, false)>]
let ``ABI carriers without target realization are refused without width substitution`` bits signed =
    let scalar = if bits = 1 then BoundaryScalar.Boolean else BoundaryScalar.Integer(bits, signed)
    let original = declaration scalar
    for candidate in [original; { original with Parameters = []; Result = Some scalar }; { original with Result = None }] do
        match BackEnd.LLVM.BoundaryAdmission.validate (checkedInput candidate) context with
        | Error message -> Assert.Contains("requires ABI realization", message)
        | Ok () -> failwith "An unsupported scalar crossed the target ABI"

[<Theory>]
[<InlineData("x86_64-pc-windows-gnu")>]
[<InlineData("aarch64-unknown-linux-gnu")>]
[<InlineData("x86_64-unknown-linux-gnux32")>]
let ``scalar boundary admission belongs to the selected target ABI`` triple =
    let original = checkedInput (declaration (BoundaryScalar.Integer(32, true)))
    match BackEnd.LLVM.BoundaryAdmission.validate original { context with TargetTripleOverride = Some triple } with
    | Error message -> Assert.Contains("no admitted C ABI realization", message)
    | Ok () -> failwith "A different target acquired the AMD64 C ABI"

[<Fact>]
let ``C ABI availability is independent of owned or hosted startup`` () =
    let original = checkedInput (declaration (BoundaryScalar.Integer(32, true)))
    for deployment in [Core.Types.Dialects.DeploymentMode.Console; Core.Types.Dialects.DeploymentMode.Freestanding; Core.Types.Dialects.DeploymentMode.Library] do
        BackEnd.LLVM.BoundaryAdmission.validate original { context with DeploymentMode = deployment } |> good
    for runtime in [None; Some Clef.Compiler.NativeTypedTree.NativeTypes.RuntimeModel.Bare] do
        match BackEnd.LLVM.BoundaryAdmission.validate original { context with RuntimeModel = runtime } with
        | Error message -> Assert.Contains("no admitted C ABI realization", message)
        | Ok () -> failwith "Startup supplied an undeclared C runtime"

[<Fact>]
let ``target lowering cannot reinterpret the witnessed pointer dimension`` () =
    let original = checkedInput (declaration (BoundaryScalar.Integer(32, true)))
    for pointer in [Ok 32; Error "undeclared"] do
        match BackEnd.LLVM.BoundaryAdmission.validate { original with PointerBits = pointer } context with
        | Error message -> Assert.Contains("witnessed pointer dimension disagrees", message)
        | Ok () -> failwith "Target selection replaced the witnessed pointer dimension"

let private run program arguments =
    let start = ProcessStartInfo(program, UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true)
    for argument in arguments do start.ArgumentList.Add argument
    use child = new Process(StartInfo = start)
    Assert.True(child.Start())
    let stdout, stderr = child.StandardOutput.ReadToEndAsync(), child.StandardError.ReadToEndAsync()
    if not (child.WaitForExit 30000) then
        child.Kill(true)
        failwith $"{program} timed out"
    let output, errors = stdout.GetAwaiter().GetResult(), stderr.GetAwaiter().GetResult()
    Assert.True(child.ExitCode = 0, $"{program} exited {child.ExitCode}: {output}\n{errors}")

[<Fact>]
let ``admitted signed and unsigned 32 and 64 bit carriers interoperate with C in both directions`` () =
    let contracts = ["s32", 32, true; "u32", 32, false; "s64", 64, true; "u64", 64, false]
    let operations = contracts |> List.collect (fun (suffix, bits, signed) ->
        let scalar = BoundaryScalar.Integer(bits, signed)
        let declaration = { declaration scalar with Symbol = "external_" + suffix; Parameters = [NodeId 4, scalar; NodeId 5, scalar] }
        let ty = BoundaryAbi.scalarType scalar
        let result = { SSA = V(20, 0); Type = ty }
        [ MLIROp.FuncOp(BoundaryFuncDecl declaration)
          MLIROp.FuncOp(FuncDef("invoke_" + suffix, [Arg 0, ty; Arg 1, ty], [ty],
              [MLIROp.FuncOp(FuncCall([result], declaration.Symbol, [{ SSA = Arg 0; Type = ty }; { SSA = Arg 1; Type = ty }]))
               MLIROp.FuncOp(Return [result])], FuncVisibility.Public)) ])
    let witnessed = input operations
    BackEnd.LLVM.BoundaryAdmission.validate witnessed context |> good
    let directory = Path.Combine(Path.GetTempPath(), "clef-boundary-abi-" + Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory directory |> ignore
    try
        let mlir, llvm, c, executable = Path.Combine(directory, "input.mlir"), Path.Combine(directory, "output.ll"), Path.Combine(directory, "foreign.c"), Path.Combine(directory, "check")
        File.WriteAllText(mlir, witnessed.Text)
        BackEnd.LLVM.Lowering.lowerToLLVM mlir llvm context.TargetTripleOverride.Value (Some 64) |> good
        File.WriteAllText(c, """
#include <stdint.h>
int32_t external_s32(int32_t x, int32_t y) { return x - y; }
uint32_t external_u32(uint32_t x, uint32_t y) { return x - y; }
int64_t external_s64(int64_t x, int64_t y) { return x - y; }
uint64_t external_u64(uint64_t x, uint64_t y) { return x - y; }
extern int32_t invoke_s32(int32_t, int32_t);
extern uint32_t invoke_u32(uint32_t, uint32_t);
extern int64_t invoke_s64(int64_t, int64_t);
extern uint64_t invoke_u64(uint64_t, uint64_t);
int main(void) {
  if (invoke_s32(-200, 3) != -203) return 1;
  if (invoke_u32(UINT32_C(4000000000), 3) != UINT32_C(3999999997)) return 2;
  if (invoke_s64(-INT64_C(1099511627776), 7) != -INT64_C(1099511627783)) return 3;
  if (invoke_u64(UINT64_C(18000000000000000000), 7) != UINT64_C(17999999999999999993)) return 4;
  return 0;
}
        """)
        run "clang" ["--target=" + context.TargetTripleOverride.Value; llvm; c; "-o"; executable]
        run executable []
    finally
        Directory.Delete(directory, true)
