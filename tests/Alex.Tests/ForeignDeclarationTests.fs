module Alex.Tests.ForeignDeclarationTests

open System
open Xunit
open Alex.Dialects.Core.Types
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.ScopeContext
open Alex.Traversal.NanopassArchitecture

module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission
module Zipper = Alex.Traversal.PSGZipper
module Catalog = Core.WitnessArtifacts

let private boundaryFixtureWith tail =
    let source = """module PublishedForeign
type Signedness = Signed | Unsigned
type TypeRef = Integer of Signedness * int | Void
type PassBy = Value | Reference
type CallConv = | CDecl
type Transfer = | Borrowed
type ParameterInfo = { Name: string; Type: TypeRef; PassBy: PassBy }
type FunctionDescriptor = { CName: string; Parameters: ParameterInfo array; ReturnType: TypeRef; CallingConvention: CallConv; OwnershipTransfer: Transfer }
[<FidelityExtern("c", "combine")>]
let combine (left: int) (right: int) : int = NativeDefault.zeroed ()
let combineDescriptor: Expr<FunctionDescriptor> = <@ {
    CName = "combine"
    Parameters = [| { Name="left"; Type=Integer(Signed,32); PassBy=Value }; { Name="right"; Type=Integer(Signed,32); PassBy=Value } |]
    ReturnType=Integer(Signed,32); CallingConvention=CDecl; OwnershipTransfer=Borrowed } @>
"""
    let graph = Fixtures.checkScalarProgram (source + tail) "published-boundary.clef"
    let boundary = Publication.tryBoundary graph |> Result.defaultWith failwith
    let declaration = Assert.Single boundary.Imports.Values
    let call = Assert.Single boundary.Calls.Values
    graph, declaration, call

let private boundaryFixture () =
    boundaryFixtureWith "[<EntryPoint>]\nlet main _ = combine (-11) 7\n"

let private position graph id = Zipper.create graph id |> Option.defaultWith (fun () -> failwith "Missing source occurrence")

let private recalledBoundary (call: BoundaryCall) wrongType =
    let accumulator = MLIRAccumulator.empty ()
    let parameters = call.Arguments |> List.mapi (fun ordinal operand ->
        let bits =
            match operand.Adaptation, operand.Abi with
            | Some meet, _ -> meet.From
            | None, BoundaryScalar.Integer(bits, _) -> bits
            | None, BoundaryScalar.Boolean -> 1
        let ty = TInt(IntWidth(if wrongType && ordinal = 0 then bits + 1 else bits))
        MLIRAccumulator.bindNode operand.Actual (Arg ordinal) ty accumulator
        Arg ordinal, ty)
    accumulator, parameters

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

[<Fact>]
let ``catalog retains published boundary signedness when portable types are identical`` () =
    let graph, declaration, _ = boundaryFixture ()
    let operation = MLIROp.FuncOp(FuncOp.BoundaryFuncDecl declaration)
    let text = serialize [operation]
    let scope = Catalog.beginWholeGraphWitness graph
    let catalog =
        Catalog.create scope Core.Types.WitnessArtifacts.TargetModuleActivation [] [operation] text []
        |> Result.defaultWith failwith
    Assert.Equal(Some declaration, (Assert.Single catalog.Units.Head.Imports).Boundary)
    let changed =
        { declaration with
            Parameters = declaration.Parameters |> List.map (fun (formal, scalar) ->
                formal, match scalar with BoundaryScalar.Integer(bits, sign) -> BoundaryScalar.Integer(bits, not sign) | other -> other) }
    let changedOperation = MLIROp.FuncOp(FuncOp.BoundaryFuncDecl changed)
    let erased = MLIROp.FuncOp(FuncDecl(declaration.Symbol, BoundaryAbi.parameters declaration,
                                       BoundaryAbi.results declaration, FuncVisibility.Private, []))
    for changedOperation in [changedOperation; erased] do
        Assert.Equal(text, serialize [changedOperation])
        match Catalog.create scope Core.Types.WitnessArtifacts.TargetModuleActivation [] [changedOperation] text [] with
        | Result.Error reason -> Assert.Contains("source-published ABI", reason)
        | Result.Ok _ -> failwith "Catalog erased boundary signedness because its portable spelling was unchanged"

[<Fact>]
let ``backend without a boundary realization refuses before its implementation runs`` () =
    let graph, declaration, _ = boundaryFixture ()
    let operations = [MLIROp.FuncOp(BoundaryFuncDecl declaration)]
    let text = serialize operations
    let catalog = Catalog.create (Catalog.beginWholeGraphWitness graph)
                      Core.Types.WitnessArtifacts.TargetModuleActivation [] operations text [] |> Result.defaultWith failwith
    let input: Core.Types.Pipeline.BackEndInput =
        { Operations = operations; Text = text; PointerBits = Ok 64; ModuleName = Some "foreign_declarations"
          WritableStorage = []; Catalog = Some catalog }
    let mutable called = false
    let implementation _ _ = called <- true; Ok (Core.Types.Pipeline.IntermediateOnly "sentinel")
    match Core.Types.Pipeline.WitnessedInput.compile implementation input Unchecked.defaultof<_> with
    | Result.Error message -> Assert.Contains("no boundary ABI realization", message)
    | Result.Ok _ -> failwith "A backend accepted a source boundary without a realization"
    Assert.False called

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

[<Fact>]
let ``published import and ordered scalar call witness separately and verify as portable MLIR`` () =
    let graph, declaration, call = boundaryFixture ()
    let imports =
        match Fixtures.matchAt Alex.Patterns.PlatformPatterns.pBoundaryImports (position graph declaration.Scope) 64 (MLIRAccumulator.empty ()) with
        | Result.Ok (operations, _) -> operations
        | Result.Error reason -> failwith reason
    match Assert.Single imports with
    | MLIROp.FuncOp(FuncOp.BoundaryFuncDecl published) ->
        Assert.Equal<BoundaryImport>(declaration, published)
        Assert.Equal("combine", published.Symbol)
        Assert.Equal<BoundaryScalar list>([BoundaryScalar.Integer(32, true); BoundaryScalar.Integer(32, true)], published.Parameters |> List.map snd)
        Assert.Equal(Some (BoundaryScalar.Integer(32, true)), published.Result)
    | other -> failwithf "Expected published module import, got %A" other
    let operands, parameters = recalledBoundary call false
    let operations, result =
        match Fixtures.matchAt Alex.Patterns.PlatformPatterns.pBoundaryCall (position graph call.Site) 64 operands with
        | Result.Ok (result, _) -> result
        | Result.Error reason -> failwith reason
    Assert.DoesNotContain(operations, fun operation -> match operation with MLIROp.FuncOp(FuncOp.FuncDecl _ | FuncOp.BoundaryFuncDecl _) -> true | _ -> false)
    let calls = operations |> List.choose (function MLIROp.FuncOp(FuncOp.FuncCall(results, symbol, values)) -> Some(results, symbol, values) | _ -> None)
    let _, symbol, actuals = Assert.Single calls
    Assert.Equal("combine", symbol)
    let expectedActuals = call.Arguments |> List.mapi (fun ordinal operand ->
        { SSA = match operand.Adaptation with
                | Some published ->
                    let settled, value = meetFor graph call.Site operand.Actual |> Option.defaultWith (fun () -> failwith "Missing canonical boundary meet")
                    Assert.Equal<Meet>(published, settled)
                    value
                | None -> Arg ordinal
          Type = TInt(IntWidth 32) })
    Assert.Equal<Val list>(expectedActuals, actuals)
    Assert.Contains(call.Arguments, fun operand -> operand.Adaptation |> Option.exists (fun meet -> meet.Adapt = MeetKind.ExtendSigned && meet.From = 8 && meet.To = 32))
    match result with
    | TRValue value ->
        let body = operations @ [MLIROp.FuncOp(FuncOp.Return [value])]
        let functionOp = MLIROp.FuncOp(FuncOp.FuncDef("boundary_roundtrip", parameters, [value.Type], body, FuncVisibility.Public))
        let text = serialize (imports @ [functionOp])
        MlirComponentTests.mlirOpt ["--verify-each"] text |> ignore
    | other -> failwithf "Expected scalar boundary result, got %A" other

[<Fact>]
let ``boundary witness refuses mismatched physical operands instead of adapting by inspection`` () =
    let graph, _, call = boundaryFixture ()
    let operands, _ = recalledBoundary call true
    match Fixtures.matchAt Alex.Patterns.PlatformPatterns.pBoundaryCall (position graph call.Site) 64 operands with
    | Result.Error reason -> Assert.Contains("disagrees", reason)
    | Result.Ok _ -> failwith "Witness accepted an operand outside its published contract"

[<Fact>]
let ``boundary witnessing never republishes an invalidated source graph`` () =
    let graph, declaration, _ = boundaryFixture ()
    let draft = Fixtures.unpublished graph
    match Fixtures.matchAt Alex.Patterns.PlatformPatterns.pBoundaryImports (position draft declaration.Scope) 64 (MLIRAccumulator.empty ()) with
    | Result.Error reason -> Assert.Contains("source witness projection", reason)
    | Result.Ok _ -> failwith "Witness repaired missing source publication"

[<Fact>]
let ``declaration scope is witnessed once independently of runtime reachability`` () =
    let original, declaration, _ = boundaryFixture ()
    let owner = original.Nodes[declaration.Scope]
    let draft = { Fixtures.unpublished original with Nodes = original.Nodes.Add(owner.Id, { owner with IsReachable = false }) }
    let graph =
        match Publication.prepare draft with
        | Result.Ok graph -> graph
        | Result.Error reasons -> failwithf "Source did not admit declaration scope: %A" reasons
    let witnessed = ResizeArray<Zipper.PSGZipper>()
    // Isolate scheduling and scope routing; unrelated source occurrences do not
    // claim that their expressions have been physically composed by this test.
    let scopeWitness: Nanopass =
        { Name = "BoundaryScopeComponent"
          Witness = fun context node ->
              if node.Id = declaration.Scope then
                  witnessed.Add context.Zipper
                  Alex.Witnesses.StructuralWitness.nanopass.Witness context node
              else { InlineOps = []; TopLevelOps = []; Result = TRVoid } }
    let accumulator = MLIRAccumulator.empty ()
    let root = ref (ScopeContext.root ())
    let visited = ref Set.empty
    runAllNanopasses [scopeWitness] graph (Fixtures.coeffects graph 64) accumulator root visited
    Assert.Empty accumulator.Errors
    let scopeCoverage =
        Alex.Traversal.CoverageValidation.validateCoverage graph visited.Value
        |> List.filter (fun diagnostic -> diagnostic.NodeId = Some declaration.Scope)
    Assert.Empty scopeCoverage
    let missingImportScope = Alex.Traversal.CoverageValidation.validateCoverage graph (visited.Value.Remove declaration.Scope)
    Assert.Contains(missingImportScope, fun diagnostic -> diagnostic.NodeId = Some declaration.Scope)
    let occurrence = Assert.Single witnessed
    Assert.Same(graph.Nodes[declaration.Scope], occurrence.Focus)
    Assert.Same(graph, occurrence.Graph)
    match Assert.Single (ScopeContext.getOps root.Value) with
    | MLIROp.FuncOp(FuncOp.BoundaryFuncDecl published) -> Assert.Equal<BoundaryImport>(declaration, published)
    | other -> failwithf "Expected one module import at its declared occurrence, got %A" other

[<Fact>]
let ``source scalar boundary traverses the complete witness registry`` () =
    let graph, declaration, call = boundaryFixture ()
    let coeffects = Fixtures.coeffects graph 64
    let registry = Alex.Traversal.WitnessRegistry.createRegistry coeffects.TargetPlatform
    let accumulator = executeNanopasses registry graph coeffects None
    Assert.Empty accumulator.Errors
    let placeholderDefinitions = accumulator.EmittedDefinitions |> List.choose (fun definition ->
        match definition.Operation with
        | MLIROp.FuncOp(FuncOp.FuncDef(symbol, _, _, _, _))
            when List.contains definition.Occurrence.Focus.Id declaration.DeclarationPath ->
            Some (definition.Occurrence.Focus.Id, symbol)
        | _ -> None)
    Assert.Empty placeholderDefinitions
    let operations = List.rev accumulator.AllOps
    let definitions = operations |> List.choose (function MLIROp.FuncOp(FuncOp.FuncDef(_, _, _, body, _)) -> Some body | _ -> None)
    let body, resultValues, actualValues =
        definitions |> List.collect (fun body ->
            body |> List.choose (function
                | MLIROp.FuncOp(FuncOp.FuncCall(results, "combine", values)) -> Some(body, results, values)
                | _ -> None))
        |> Assert.Single
    Assert.Equal(2, actualValues.Length)
    let constant value =
        body |> List.choose (function MLIROp.ArithOp(ArithOp.ConstI(ssa, actual, ty)) when actual = value -> Some(ssa, ty) | _ -> None)
        |> Assert.Single
    let negative, negativeType = constant -11L
    Assert.Equal(TInt(IntWidth 8), negativeType)
    let extension, source, fromType, toType =
        body |> List.choose (function MLIROp.ArithOp(ArithOp.ExtSI(result, source, fromType, toType)) when source = negative -> Some(result, source, fromType, toType) | _ -> None)
        |> Assert.Single
    Assert.Equal(negative, source)
    Assert.Equal(TInt(IntWidth 8), fromType)
    Assert.Equal(TInt(IntWidth 32), toType)
    let publishedMeet, canonicalExtension = meetFor graph call.Site call.Arguments.Head.Actual |> Option.defaultWith (fun () -> failwith "Missing source boundary meet")
    Assert.Equal(call.Arguments.Head.Adaptation, Some publishedMeet)
    Assert.Equal(canonicalExtension, extension)
    Assert.Equal<Val>({ SSA = extension; Type = TInt(IntWidth 32) }, actualValues[0])
    let positive, positiveType = constant 7L
    let positiveAtCall =
        if positiveType = TInt(IntWidth 32) then positive else
        let result, fromType, toType =
            body |> List.choose (function MLIROp.ArithOp(ArithOp.ExtUI(result, source, fromType, toType)) when source = positive -> Some(result, fromType, toType) | _ -> None)
            |> Assert.Single
        Assert.Equal(positiveType, fromType)
        Assert.Equal(TInt(IntWidth 32), toType)
        result
    Assert.Equal<Val>({ SSA = positiveAtCall; Type = TInt(IntWidth 32) }, actualValues[1])
    let callResult = Assert.Single resultValues
    let canonicalResult = Alex.Traversal.Values.resultOf coeffects.TargetPlatform graph (Zipper.enclosingLambdaIds (position graph call.Site)) call.Site
    Assert.Equal(canonicalResult, callResult.SSA)
    let text = serialize operations
    MlirComponentTests.mlirOpt ["--verify-each"] text |> ignore

[<Fact>]
let ``external declaration supplied as a callback crosses the production witness boundary`` () =
    let graph, declaration, _ = boundaryFixtureWith """
let invoke callback = callback (-11) 7
[<EntryPoint>]
let main _ = invoke combine
"""
    let witnessed, libraries =
        MiddleEnd.MLIRGeneration.generate graph graph.Platform.Value Core.Types.Dialects.Console
            Core.Types.Dialects.CPU None |> Result.defaultWith failwith
    Core.Types.Pipeline.WitnessedInput.validate witnessed |> Result.defaultWith failwith
    Assert.Contains(declaration.Library, libraries)
    let imports = witnessed.Operations |> List.choose (function MLIROp.FuncOp(BoundaryFuncDecl imported) -> Some imported | _ -> None)
    Assert.Equal<BoundaryImport>(declaration, Assert.Single imports)
    let calls = witnessed.Operations |> List.collect (Catalog.flatten >> Seq.toList) |> List.choose (function
        | MLIROp.FuncOp(FuncCall(_, symbol, arguments)) when symbol = declaration.Symbol -> Some arguments
        | _ -> None)
    Assert.Equal(2, (Assert.Single calls).Length)
    MlirComponentTests.mlirOpt ["--verify-each"] witnessed.Text |> ignore
