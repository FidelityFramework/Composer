module Alex.Tests.FunctionResultTests

open Xunit
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures

let private integer = TInt(IntWidth 32)
let private environment = TMemRef integer
let private callback = TFunc([environment; integer], [integer])
let private results = [callback; environment]
let private value ssa ty : Val = { SSA = ssa; Type = ty }
let private f operation = MLIROp.FuncOp operation

/// The graph supplies only the parser occurrence. These inputs are already
/// physical components; this gate does not invent a source callable carrier.
let private definition pointerBits name parameters body (returned: Val list) =
    let position = (arrayRead true).Position
    let pattern = Alex.Patterns.ClosurePatterns.pFunctionDefResults FuncVisibility.Public name parameters None
                      (List.map _.Type returned) body returned
    match matchAt pattern position pointerBits (MLIRAccumulator.empty ()) with
    | Result.Ok (operation, _) -> operation
    | Result.Error reason -> failwith reason

[<Theory>]
[<InlineData(32)>]
[<InlineData(64)>]
let ``separate code and environment returns survive direct and indirect calls and real lowering`` pointerBits =
    let env, argument = value (Arg 0) environment, value (Arg 1) integer
    let zero, loaded, sum = V(800, 0), V(800, 1), V(800, 2)
    let implementation = definition pointerBits "apply_environment" [env.SSA, env.Type; argument.SSA, argument.Type]
                             [MLIROp.IndexOp(IndexOp.IndexConst(zero, 0L))
                              MLIROp.MemRefOp(MemRefOp.Load(loaded, env.SSA, [zero], integer, environment))
                              MLIROp.ArithOp(ArithOp.AddI(sum, loaded, argument.SSA, integer))]
                             [value sum integer]
    let code = value (V(801, 0)) callback
    let factory = definition pointerBits "make_callable" [env.SSA, env.Type]
                      [f (FuncOp.FuncConstant(code.SSA, "apply_environment", callback))] [code; env]
    // A single function-typed result exercises the nested arrow's parentheses.
    let codeIdentity = definition pointerBits "code_identity" [Arg 0, callback] [] [value (Arg 0) callback]
    let directCode, directEnv = value (V(802, 0)) callback, value (V(802, 1)) environment
    let indirectCode, indirectEnv = value (V(802, 2)) callback, value (V(802, 3)) environment
    let factoryType = TFunc([environment], results)
    let factoryCode = value (V(802, 4)) factoryType
    let first, second, total = value (V(802, 5)) integer, value (V(802, 6)) integer, value (V(802, 7)) integer
    let body =
        [f (FuncOp.FuncCall([directCode; directEnv], "make_callable", [env]))
         f (FuncOp.FuncCallIndirect([first], directCode.SSA, [directEnv; argument]))
         f (FuncOp.FuncConstant(factoryCode.SSA, "make_callable", factoryType))
         f (FuncOp.FuncCallIndirect([indirectCode; indirectEnv], factoryCode.SSA, [env]))
         f (FuncOp.FuncCallIndirect([second], indirectCode.SSA, [indirectEnv; argument]))
         MLIROp.ArithOp(ArithOp.AddI(total.SSA, first.SSA, second.SSA, integer))]
    let consumer = definition pointerBits "consume_callable_results" [env.SSA, env.Type; argument.SSA, argument.Type] body [total]
    let operations = Alex.Pipeline.MLIRNanopass.declarationCollectionPass [implementation; factory; codeIdentity; consumer]
    let source = Alex.Dialects.Core.Serialize.moduleToString (Ok pointerBits) "function_results" operations
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] source
    // MLIR's printer elides the enclosing func dialect inside function bodies.
    Assert.Contains("func.call @make_callable", source)
    Assert.Contains("call @make_callable", verified)
    Assert.Contains("call_indirect", verified)
    Assert.Contains("return", verified)
    let atWidth name = $"{name}{{index-bitwidth={pointerBits}}}"
    let passes = ["expand-strided-metadata"; "memref-expand"; atWidth "finalize-memref-to-llvm"
                  atWidth "convert-index-to-llvm"; atWidth "convert-func-to-llvm"
                  atWidth "convert-arith-to-llvm"; "reconcile-unrealized-casts"]
    let lowered = MlirComponentTests.mlirOpt ["--verify-each"; "--pass-pipeline=builtin.module(" + String.concat "," passes + ")"] verified
    Assert.Contains("llvm.func @make_callable", lowered)
    Assert.Contains("llvm.call", lowered)
    Assert.Contains("llvm.load", lowered)
    Assert.DoesNotContain("func.call", lowered)
    Assert.DoesNotContain("call_indirect", lowered)
    Assert.DoesNotContain("unrealized_conversion_cast", lowered)

[<Fact>]
let ``declaration collection preserves result lists inside nounwind bodies`` () =
    let declaration = f (FuncOp.FuncDecl("foreign_callable", [environment], results, FuncVisibility.Private, []))
    let returned = [value (V(803, 0)) callback; value (V(803, 1)) environment]
    let wrapper = MLIROp.NoUnwindFunction(FuncOp.FuncDef("forward_callable", [Arg 0, environment], results,
                      [declaration; f (FuncOp.FuncCall(returned, "foreign_callable", [value (Arg 0) environment])); f (FuncOp.Return returned)], FuncVisibility.Public))
    let collected = Alex.Pipeline.MLIRNanopass.declarationCollectionPass [wrapper]
    Assert.Equal(declaration, collected.Head)
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "nounwind_results" collected
    MlirComponentTests.mlirOpt ["--verify-each"] text |> ignore

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``declaration collection rejects missing or reordered mandatory results`` reorder =
    let declaration = f (FuncOp.FuncDecl("required_results", [], results, FuncVisibility.Private, []))
    let supplied = if reorder then [value (V(804, 0)) environment; value (V(804, 1)) callback] else []
    let call = f (FuncOp.FuncCall(supplied, "required_results", []))
    let error = Assert.Throws<System.Exception>(fun () -> Alex.Pipeline.MLIRNanopass.declarationCollectionPass [declaration; call] |> ignore)
    Assert.Contains("ordered results", error.Message)

[<Fact>]
let ``declaration relocation cannot conceal conflicting result signatures`` () =
    let declare returns = f (FuncOp.FuncDecl("same_name", [], returns, FuncVisibility.Private, []))
    let error = Assert.Throws<System.Exception>(fun () ->
        Alex.Pipeline.MLIRNanopass.declarationCollectionPass [declare results; declare (List.rev results)] |> ignore)
    Assert.Contains("Conflicting function signatures", error.Message)

[<Fact>]
let ``function code SSA has no implicit packed storage size`` () =
    let error = Assert.Throws<System.Exception>(fun () -> mlirTypeSizeWith (Ok 64) callback |> ignore)
    Assert.Contains("no admitted data-storage layout", error.Message)

[<Fact>]
let ``void scalar call cannot silently create a result`` () =
    let position = (arrayRead true).Position
    let pattern = Alex.Patterns.ApplicationPatterns.pDirectCall position.Focus.Id "void_effect" [] TVoid None
    match matchAt pattern position 64 (MLIRAccumulator.empty ()) with
    | Result.Error message -> Assert.Contains("void function call cannot define an SSA result", message)
    | Result.Ok _ -> failwith "A scalar call created a void SSA"

[<Fact>]
let ``nounwind wrapping preserves storage correspondence validation`` () =
    let graph = (arrayRead true).Graph
    let operation = MLIROp.NoUnwindFunction(FuncOp.FuncDef("bad_pool", [], [],
                        [MLIROp.GlobalBytePool("unexpected_pool", [0uy], 1, []); f (FuncOp.Return [])], FuncVisibility.Public))
    match Alex.Traversal.StaticStorageValidation.validate graph [operation] with
    | Result.Error message -> Assert.Contains("without a settled BAREWire plan", message)
    | Result.Ok _ -> failwith "A nounwind wrapper hid an unplanned byte pool"
