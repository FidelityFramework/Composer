module Alex.Tests.UnionLayoutPatternTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Patterns.DUPatterns
open Alex.Patterns.MemoryPatterns
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper

let private platform bits : PlatformContext =
    { PlatformId = "union-component-test"; Dimensions = Map.ofList ["Pointer", bits; "Register", bits]
      Representations = Map.empty; EndpointReturns = Map.empty; PlatformLibraryPath = None
      PlatformDescription = None; PlatformArchitecture = None; PlatformOS = None
      PlatformSourcePaths = Set.empty; Predicates = Map.empty; FreestandingStartup = None
      SubstrateKind = None; RuntimeModel = None; AvailableMemorySpaces = []; DefaultMemorySpace = None
      ClockFrequencyMhz = None; NsPerWeightUnit = None }

[<Theory>]
[<InlineData(32, false)>]
[<InlineData(32, true)>]
[<InlineData(64, false)>]
[<InlineData(64, true)>]
let ``union allocation and selected descriptor stores consume the source aligned layout`` bits present =
    let builder = NodeBuilder()
    let payloadType = Types.mkArrayType Types.uint8Type
    let unionType = NativeType.TApp(Types.optionTyCon, [payloadType])
    let logical = builder.Create(SemanticKind.PatternBinding "logical", unionType, dummyRange)
    let storage = builder.Create(SemanticKind.AggregateStorage logical.Id, unionType, dummyRange)
    let payload = builder.Create(SemanticKind.PatternBinding "payload", payloadType, dummyRange)
    let selected = builder.Create(SemanticKind.DUInitialize(storage.Id, (if present then "Some" else "None"),
                                                          (if present then 1 else 0), (if present then Some payload.Id else None)), Types.unitType, dummyRange)
    let raw = builder.Build []
    // Placement is read from the source pass. This supplied component
    // residence exercises witnessing; it does not establish a program lifetime.
    let context = platform bits
    let placed = Clef.Compiler.PSGSaturation.SemanticGraph.Placement.settle (Some context) { raw with Platform = Some context }
    let graph = { placed with Codata = lazy { placed.Codata.Value with Escapes = Map.ofList [storage.Id, EscapeKind.StackScoped] } }
    let operands = MLIRAccumulator.empty ()
    let focus id = Zipper.create graph id |> require "Missing union occurrence"
    let allocations, destination =
        match matchAt (pBuildAggregateStorage storage.Id) (focus storage.Id) bits operands with
        | Result.Ok ((operations, TRValue value), _) -> operations, value
        | other -> failwithf "Union allocation lost source layout: %A" other
    let word, bytes = bits / 8, 6 * (bits / 8)
    match Assert.Single allocations with
    | MLIROp.MemRefOp(MemRefOp.Alloca(_, TMemRefStatic(actualBytes, TInt(IntWidth 8)), Some alignment)) ->
        Assert.Equal(bytes, actualBytes)
        Assert.Equal(word, alignment)
    | other -> failwithf "Allocation lost source alignment: %A" other
    let descriptor = { SSA = Arg 0; Type = TMemRef(TInt(IntWidth 8)) }
    let fields = if present then [descriptor] else []
    let writes =
        match matchAt (pDUCaseAt selected.Id destination unionType (if present then 1L else 0L) fields) (focus selected.Id) bits operands with
        | Result.Ok ((operations, TRVoid), _) -> operations
        | other -> failwithf "Selected union initialization failed: %A" other
    let views = writes |> List.choose (function
        | MLIROp.MemRefOp(MemRefOp.View(_, _, offset, _, _)) -> Some offset
        | _ -> None)
    if present then
        let offset = Assert.Single views
        Assert.Contains(writes, function MLIROp.ArithOp(ArithOp.ConstI(actual, value, TIndex)) -> actual = offset && value = int64 word | _ -> false)
    else Assert.Empty views
    let stores = writes |> List.filter (function MLIROp.MemRefOp(MemRefOp.StoreAligned _) -> true | _ -> false)
    Assert.Equal((if present then 2 else 1), stores.Length)
    let definition = MLIROp.FuncOp(FuncOp.FuncDef("initialize_aligned_union", [descriptor.SSA, descriptor.Type], [],
        allocations @ writes @ [MLIROp.FuncOp(FuncOp.Return [])], FuncVisibility.Public))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok bits) "union_layout_component" [definition]
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
    Assert.DoesNotContain("memref.load", verified)
    Assert.Equal((if present then 2 else 1), verified.Split("memref.store").Length - 1)
