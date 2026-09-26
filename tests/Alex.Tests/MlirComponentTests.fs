module Alex.Tests.MlirComponentTests

open System
open System.Diagnostics
open Xunit
open Alex.Tests.Fixtures
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize

/// Runs the real tool over text serialized from Alex operations. Missing tools
/// fail the test; these are component gates, not language or native-run oracles.
let mlirOpt arguments input =
    let start = ProcessStartInfo("mlir-opt", UseShellExecute = false,
                                RedirectStandardInput = true,
                                RedirectStandardOutput = true,
                                RedirectStandardError = true)
    for argument in arguments do start.ArgumentList.Add argument
    use child = new Process(StartInfo = start)
    if not (child.Start()) then failwith "Cannot start mlir-opt"
    let stdout, stderr = child.StandardOutput.ReadToEndAsync(), child.StandardError.ReadToEndAsync()
    child.StandardInput.Write(input: string)
    child.StandardInput.Close()
    if not (child.WaitForExit 20000) then
        child.Kill(true)
        child.WaitForExit()
        failwith "mlir-opt component verification timed out"
    let output, errors = stdout.GetAwaiter().GetResult(), stderr.GetAwaiter().GetResult()
    Assert.True(child.ExitCode = 0, $"mlir-opt exited {child.ExitCode}:\n{errors}\nInput:\n{input}")
    output

[<Theory>]
[<InlineData(true, 32, "llvm.zext")>]
[<InlineData(false, 32, "llvm.sext")>]
[<InlineData(true, 64, "llvm.zext")>]
[<InlineData(false, 64, "llvm.sext")>]
let ``pattern output verifies and retains index signedness through standard lowering`` (unsigned: bool) (pointerBits: int) (extension: string) =
    let text = readModule (arrayRead unsigned) pointerBits
    let verified = mlirOpt ["--verify-each"] text
    Assert.Contains("func.func @read_index", verified)
    Assert.DoesNotContain("TODO", verified)
    let atWidth pass = $"{pass}{{index-bitwidth={pointerBits}}}"
    let passes =
        ["expand-strided-metadata"; "memref-expand"; atWidth "finalize-memref-to-llvm"
         atWidth "convert-index-to-llvm"; atWidth "convert-func-to-llvm"
         atWidth "convert-arith-to-llvm"; "reconcile-unrealized-casts"]
    let pipeline = "builtin.module(" + String.concat "," passes + ")"
    let lowered = mlirOpt ["--verify-each"; "--pass-pipeline=" + pipeline] verified
    Assert.Contains("llvm.func @read_index", lowered)
    Assert.Contains(extension, lowered)
    Assert.Contains($"i8 to i{pointerBits}", lowered)
    Assert.DoesNotContain("index.cast", lowered)
    Assert.DoesNotContain("unrealized_conversion_cast", lowered)

[<Theory>]
[<InlineData(32)>]
[<InlineData(64)>]
let ``opaque static frame copy lowers to memcpy without field loads`` (pointerBits: int) =
    let storage = TMemRefStatic(136, TInt(IntWidth 8))
    let copy = MLIROp.MemRefOp(MemRefOp.Copy(Arg 0, Arg 1, storage, storage))
    let definition =
        MLIROp.FuncOp(FuncOp.FuncDef("copy_frame", [Arg 0, storage; Arg 1, storage], [],
            [copy; MLIROp.FuncOp(FuncOp.Return [])], FuncVisibility.Public))
    let text = moduleToString (Ok pointerBits) "opaque_copy_component" [definition]
    let verified = mlirOpt ["--verify-each"] text
    Assert.Contains("memref.copy", verified)
    Assert.DoesNotContain("memref.load", verified)
    Assert.DoesNotContain("memref.store", verified)
    let atWidth pass = $"{pass}{{index-bitwidth={pointerBits}}}"
    let passes =
        [atWidth "finalize-memref-to-llvm"; atWidth "convert-func-to-llvm"
         atWidth "convert-arith-to-llvm"; "reconcile-unrealized-casts"]
    let lowered = mlirOpt ["--verify-each"; "--pass-pipeline=builtin.module(" + String.concat "," passes + ")"] verified
    Assert.Contains("llvm.intr.memcpy", lowered)
    Assert.DoesNotContain("llvm.load", lowered)
    Assert.DoesNotContain("llvm.store", lowered)
    Assert.DoesNotContain("memrefCopy", lowered)
