module Alex.Tests.ForeignDeclarationTests

open System
open Xunit
open Alex.Dialects.Core.Types

let private serialize operations =
    Alex.Dialects.Core.Serialize.moduleToString (Result.Ok 64) "foreign_declarations" operations

[<Fact>]
let ``foreign declarations retain their published symbols without repair attributes`` () =
    let scalar = TInt(IntWidth 32)
    let declaration = MLIROp.FuncOp(FuncDecl("published_scalar", [scalar], [scalar], FuncVisibility.Private, []))
    let source = serialize [declaration]
    Assert.Contains("func.func private @published_scalar(i32) -> i32", source)
    Assert.DoesNotContain("ffi.", source)
    Assert.DoesNotContain("attributes", source)

[<Theory>]
[<InlineData(0, 8, 4)>]
[<InlineData(1, 24, 8)>]
let ``unrealized aggregate byval metadata cannot serialize as a foreign declaration`` ordinal size alignment =
    let descriptor = TMemRefStatic(size, TInt(IntWidth 8))
    let declaration = MLIROp.FuncOp(FuncDecl("foreign_record", [descriptor; descriptor], [], FuncVisibility.Private,
                                          [{ ParamIndex = ordinal; SizeBytes = size; AlignBytes = alignment }]))
    let error = Assert.ThrowsAny<Exception>(fun () -> serialize [declaration] |> ignore)
    Assert.Contains("source-settled aggregate ABI realization", error.Message)
    Assert.Contains("foreign_record", error.Message)
