module Alex.Tests.IntrinsicWriteWitnessTests

open System.IO
open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes

module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission
module Catalog = Core.WitnessArtifacts

// Exercise the actual platform declarations, Console implementation and sample.
// The fixture cannot manufacture borrowing or intrinsic ABI authority.
let private sample = lazy (
    let path = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../../samples/console/FidelityHelloWorld/01_HelloWorldDirect/HelloWorld.fidproj"))
    let checkedProject = Clef.Compiler.Project.ProjectChecker.checkProject path |> Result.defaultWith failwith
    let errors = checkedProject.CheckResult.Diagnostics |> List.filter (fun diagnostic ->
        Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.Diagnostic.effectiveSeverity diagnostic =
            Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
    Assert.Empty errors
    let graph = checkedProject.CheckResult.Graph
    let boundary = Publication.tryBoundary graph |> Result.defaultWith failwith
    let sourceProof = Core.ProofDispatch.dischargeSource graph None |> Result.defaultWith failwith
    let witnessed, _ = MiddleEnd.MLIRGeneration.generateWithLinkedLibrariesAndProof graph graph.Platform.Value
                           Core.Types.Dialects.Console Core.Types.Dialects.CPU None Set.empty (Some sourceProof) |> Result.defaultWith failwith
    graph, boundary, witnessed)

[<Fact>]
let ``Hello World witnesses source byte views and explicit ordered intrinsic operands`` () =
    let _, boundary, witnessed = sample.Value
    Assert.NotEmpty boundary.ByteViews
    Assert.NotEmpty boundary.IntrinsicWrites
    Core.Types.Pipeline.WitnessedInput.validate witnessed |> Result.defaultWith failwith
    let imports = witnessed.Operations |> List.choose (function MLIROp.FuncOp(IntrinsicWriteDecl declaration) -> Some declaration | _ -> None)
    Assert.Equal<IntrinsicWriteImport list>(boundary.IntrinsicWriteImports.Values |> Seq.toList, imports |> List.sortBy _.Identity)
    let calls = witnessed.Operations |> List.collect (Catalog.flatten >> Seq.toList) |> List.choose (function
        | MLIROp.FuncOp(FuncCall(results, symbol, args)) when imports |> List.exists (fun declaration -> declaration.Symbol = symbol) -> Some(results, symbol, args)
        | _ -> None)
    Assert.Equal(boundary.IntrinsicWrites.Count, calls.Length)
    for results, symbol, args in calls do
        let declaration = imports |> List.find (fun declaration -> declaration.Symbol = symbol)
        Assert.Equal<MLIRType list>(IntrinsicWriteAbi.parameters declaration, args |> List.map _.Type)
        Assert.Equal<MLIRType list>(IntrinsicWriteAbi.results declaration, results |> List.map _.Type)
    MlirComponentTests.mlirOpt ["--verify-each"] witnessed.Text |> ignore

[<Fact>]
let ``intrinsic Pattern preserves distinct actuals and their canonical source meets in operand order`` () =
    let graph, boundary, _ = sample.Value
    Assert.NotEmpty boundary.IntrinsicWrites
    for KeyValue(site, call) in boundary.IntrinsicWrites do
        Assert.Equal(3, Set.ofList [call.Fd; call.Buffer; call.Count] |> Set.count)
        let declaration = boundary.IntrinsicWriteImports[call.Import]
        let position = Alex.Traversal.PSGZipper.create graph site |> Option.defaultWith (fun () -> failwith "Missing intrinsic write")
        let accumulator = MLIRAccumulator.empty ()
        let seed actual supplied abi =
            let ty =
                match meetFor graph site actual with
                | Some(meet, _) -> TInt(IntWidth meet.From)
                | None -> BoundaryAbi.scalarType abi
            MLIRAccumulator.bindNode actual supplied ty accumulator
            { SSA = supplied; Type = ty }
        let fd = seed call.Fd (Arg 11) declaration.Fd
        let buffer = { SSA = Arg 17; Type = IntrinsicWriteAbi.bufferType declaration }
        MLIRAccumulator.bindNode call.Buffer buffer.SSA buffer.Type accumulator
        let count = seed call.Count (Arg 23) declaration.Count
        let expected actual original =
            match meetFor graph site actual with
            | Some(meet, canonical) -> { SSA = canonical; Type = TInt(IntWidth meet.To) }
            | None -> original
        let operations, result =
            match Fixtures.matchAt Alex.Patterns.PlatformPatterns.pIntrinsicWrite position 64 accumulator with
            | Ok ((operations, result), _) -> operations, result
            | Result.Error reason -> failwith reason
        let results, symbol, arguments =
            operations |> List.choose (function
                | MLIROp.FuncOp(FuncCall(results, symbol, arguments)) -> Some(results, symbol, arguments)
                | _ -> None) |> Assert.Single
        Assert.Equal(declaration.Symbol, symbol)
        Assert.Equal<Val list>([expected call.Fd fd; buffer; expected call.Count count], arguments)
        let rawResult = Assert.Single results
        let conversions =
            operations |> List.choose (function
                | MLIROp.ArithOp(ArithOp.ExtSI(destination, source, fromType, toType))
                | MLIROp.ArithOp(ArithOp.ExtUI(destination, source, fromType, toType))
                | MLIROp.ArithOp(ArithOp.TruncI(destination, source, fromType, toType)) ->
                    Some(destination, source, fromType, toType)
                | _ -> None)
        let expectedConversions =
            [call.Fd, fd; call.Count, count; site, rawResult]
            |> List.choose (fun (actual, original) ->
                meetFor graph site actual |> Option.map (fun (meet, canonical) ->
                    canonical, original.SSA, TInt(IntWidth meet.From), TInt(IntWidth meet.To)))
        Assert.Equal<(SSA * SSA * MLIRType * MLIRType) list>(expectedConversions, conversions)
        Assert.Equal(1 + expectedConversions.Length, operations.Length)
        match result with
        | TRValue value -> Assert.Equal<Val>(expected site rawResult, value)
        | other -> failwithf "Intrinsic write lost its source-settled result: %A" other

[<Fact>]
let ``byte borrowing Pattern aliases the actual source descriptor without operations`` () =
    let graph, boundary, _ = sample.Value
    Assert.NotEmpty boundary.ByteViews
    for view in boundary.ByteViews.Values do
        let position = Alex.Traversal.PSGZipper.create graph view.Site |> Option.defaultWith (fun () -> failwith "Missing byte view")
        let accumulator = MLIRAccumulator.empty ()
        let supplied = { SSA = Arg 37; Type = TMemRef(TInt(IntWidth view.Representation.Bits)) }
        MLIRAccumulator.bindNode view.Source supplied.SSA supplied.Type accumulator
        match Fixtures.matchAt Alex.Patterns.PlatformPatterns.pPublishedByteView position 64 accumulator with
        | Ok ((operations, TRValue value), _) ->
            Assert.Empty operations
            Assert.Equal<Val>(supplied, value)
        | Ok ((_, result), _) -> failwithf "Byte borrowing lost its descriptor value: %A" result
        | Result.Error reason -> failwith reason

[<Fact>]
let ``catalog rejects erased or changed intrinsic authority despite identical portable spelling`` () =
    let _, _, witnessed = sample.Value
    let catalog = witnessed.Catalog.Value
    for erase in [false; true] do
        let operations = witnessed.Operations |> List.map (function
            | MLIROp.FuncOp(IntrinsicWriteDecl declaration) ->
                if erase then MLIROp.FuncOp(FuncDecl(declaration.Symbol, IntrinsicWriteAbi.parameters declaration,
                                                     IntrinsicWriteAbi.results declaration, FuncVisibility.Private, []))
                else MLIROp.FuncOp(IntrinsicWriteDecl { declaration with SyscallNumber = declaration.SyscallNumber + 1I })
            | other -> other)
        let text =
            match witnessed.ModuleName with
            | Some name -> Alex.Dialects.Core.Serialize.moduleToString witnessed.PointerBits name operations
            | None -> sprintf "module {\n%s\n}" (Alex.Dialects.Core.Serialize.opsToString witnessed.PointerBits operations "  ")
        Assert.Equal(witnessed.Text, text)
        match Catalog.create catalog.Scope catalog.Activation catalog.Units.Head.Definitions operations text witnessed.WritableStorage with
        | Result.Error reason -> Assert.Contains("source-published ABI", reason)
        | Ok _ -> failwith "Identical portable spelling erased intrinsic authority"

[<Fact>]
let ``byte borrowing witness refuses a different physical carrier`` () =
    let graph, boundary, _ = sample.Value
    let view = boundary.ByteViews.Values |> Seq.head
    let position = Alex.Traversal.PSGZipper.create graph view.Site |> Option.defaultWith (fun () -> failwith "Missing byte view")
    let accumulator = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode view.Source (Arg 0) (TMemRef(TInt(IntWidth (view.Representation.Bits + 1)))) accumulator
    match Fixtures.matchAt Alex.Patterns.PlatformPatterns.pPublishedByteView position 64 accumulator with
    | Result.Error reason -> Assert.Contains("carrier disagrees", reason)
    | Ok _ -> failwith "Byte borrowing witness accepted an unsettled carrier"

[<Fact>]
let ``backend without intrinsic realization refuses before invoking its implementation`` () =
    let _, _, witnessed = sample.Value
    let mutable called = false
    let implementation _ _ = called <- true; Ok (Core.Types.Pipeline.IntermediateOnly "sentinel")
    match Core.Types.Pipeline.WitnessedInput.compile implementation witnessed Fixtures.unrealizedBackendContext with
    | Result.Error reason -> Assert.Contains("no boundary ABI realization", reason)
    | Ok _ -> failwith "Backend admitted an intrinsic without a realization"
    Assert.False called
