module Alex.Tests.MemoryWitnessTests

open System.IO
open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.ScopeContext
module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission
module Zipper = Alex.Traversal.PSGZipper
module Catalog = Core.WitnessArtifacts

// The source owner checks real declarations and publishes the memory contract.
// Physical component inputs below do not manufacture semantic authority.
let private source = lazy (
    let path = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../../samples/console/FidelityHelloWorld/01_HelloWorldDirect/HelloWorld.fidproj"))
    let project = Clef.Compiler.Project.ProjectChecker.checkProject path |> Result.defaultWith failwith
    let errors = project.CheckResult.Diagnostics |> List.filter (fun diagnostic ->
        Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.Diagnostic.effectiveSeverity diagnostic =
            Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
    Assert.Empty errors
    let graph = project.CheckResult.Graph
    let memory = Publication.tryMemory graph |> Result.defaultWith failwith
    let extents = memory.Operations.Values |> Seq.choose (function
        | MemoryWitnessOperation.BufferExtent extent -> Some extent
        | _ -> None) |> Seq.toList
    Assert.NotEmpty extents
    graph, extents)

let private context graph site operands =
    let position = Zipper.create graph site |> Fixtures.require "Missing memory occurrence"
    let scope = ref (ScopeContext.root ())
    { Coeffects = Fixtures.coeffects graph 64; Accumulator = operands; RootAccumulator = operands
      ScopeContext = scope; RootScopeContext = scope; Graph = graph; Zipper = position
      GlobalVisited = ref Set.empty; TraversalVisited = ref Set.empty }

[<Fact>]
let ``extent witness retains the published scalar carrier and exact buffer operand`` () =
    let graph, extents = source.Value
    for extent in extents do
        let operands = MLIRAccumulator.empty ()
        let bufferType = TMemRef(TInt(IntWidth extent.Element.Bits))
        MLIRAccumulator.bindNode extent.Source (Arg 0) bufferType operands
        let ctx = context graph extent.Site operands
        let associations, types = operands.NodeAssoc, operands.SSATypes
        let output = Alex.Witnesses.MemoryWitness.nanopass.Witness ctx ctx.Zipper.Focus
        let result = match output.Result with TRValue value -> value | other -> failwithf "Extent witness failed: %A" other
        let bits = match extent.Result.Slot with SettledSlot.Integer(bits, _) -> bits | other -> failwithf "Noninteger extent: %A" other
        Assert.True(bits > 0)
        Assert.Equal(TInt(IntWidth bits), result.Type)
        let dimensions = output.InlineOps |> List.choose (function
            | MLIROp.MemRefOp(MemRefOp.Dim(_, buffer, _, ty)) -> Some(buffer, ty)
            | _ -> None)
        Assert.Equal<SSA * MLIRType>((Arg 0, bufferType), Assert.Single dimensions)
        let casts = output.InlineOps |> List.choose (function
            | MLIROp.IndexOp(IndexOp.IndexCastU(result, _, TIndex, ty)) -> Some(true, result, ty)
            | MLIROp.IndexOp(IndexOp.IndexCastS(result, _, TIndex, ty)) -> Some(false, result, ty)
            | _ -> None)
        Assert.Equal<bool * SSA * MLIRType>((extent.IndexUnsigned, result.SSA, result.Type), Assert.Single casts)
        Assert.Same(associations, operands.NodeAssoc)
        Assert.Same(types, operands.SSATypes)
        Assert.Empty output.TopLevelOps
        let definition = MLIROp.FuncOp(FuncDef("extent", [Arg 0, bufferType], [result.Type],
                            output.InlineOps @ [MLIROp.FuncOp(Return [result])], FuncVisibility.Public))
        Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "memory_component" [definition]
        |> MlirComponentTests.mlirOpt ["--verify-each"] |> ignore

[<Fact>]
let ``extent witness refuses a recalled buffer with a different element carrier`` () =
    let graph, extents = source.Value
    for extent in extents do
        let operands = MLIRAccumulator.empty ()
        MLIRAccumulator.bindNode extent.Source (Arg 0) (TMemRef(TInt(IntWidth(extent.Element.Bits + 1)))) operands
        let ctx = context graph extent.Site operands
        let output = Alex.Witnesses.MemoryWitness.nanopass.Witness ctx ctx.Zipper.Focus
        match output.Result with
        | TRError diagnostic -> Assert.Contains("published buffer carrier", diagnostic.Message)
        | other -> failwithf "Mismatched buffer acquired extent authority: %A" other
        Assert.Empty output.InlineOps
        Assert.Empty operands.AllOps

[<Fact>]
let ``memory witness rejects changed source premises before recalling operands`` () =
    let graph, extents = source.Value
    let extent = List.head extents
    let node = graph.Nodes[extent.Site]
    let changed = { graph with Nodes = graph.Nodes.Add(node.Id, { node with ValueRange = Some(ValueRange.point 0I) }) }
    let ctx = context changed extent.Site (MLIRAccumulator.empty ())
    let output = Alex.Witnesses.MemoryWitness.nanopass.Witness ctx ctx.Zipper.Focus
    match output.Result with
    | TRError diagnostic -> Assert.Contains("source republication", diagnostic.Message)
    | other -> failwithf "Changed source premises retained memory authority: %A" other
    Assert.Empty output.InlineOps

[<Fact>]
let ``memory witness requires source publication even when operands were witnessed`` () =
    let original, extents = source.Value
    let extent = List.head extents
    let graph = Fixtures.unpublished original
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode extent.Source (Arg 0) (TMemRef(TInt(IntWidth extent.Element.Bits))) operands
    let ctx = context graph extent.Site operands
    let output = Alex.Witnesses.MemoryWitness.nanopass.Witness ctx ctx.Zipper.Focus
    match output.Result with
    | TRError diagnostic -> Assert.Contains("no complete source witness projection", diagnostic.Message)
    | other -> failwithf "Operands substituted for source memory authority: %A" other
    Assert.Empty output.InlineOps

// These are the same complete, checked source programs used by the source
// memory owner. The test never inserts publication facts into a graph.
let private memoryPlatform : PlatformContext =
    let representation name family bits minimum maximum : NumericRepresentation =
        { Name=name; Capability="native"; Family=family; Bits=bits
          MinMagnitude=minimum; MaxMagnitude=maximum; Boundary="wrap" }
    let representations =
        [ representation "octet" "uint" 8 "0" "255"
          representation "signed32" "int" 32 "-2147483648" "2147483647"
          representation "unsigned32" "uint" 32 "0" "4294967295"
          representation "signed64" "int" 64 "-9223372036854775808" "9223372036854775807"
          representation "unsigned64" "uint" 64 "0" "18446744073709551615" ]
    { PlatformId="memory-witness-test"; Dimensions=Map.ofList ["Pointer",64;"Register",64]
      Representations=representations |> List.map (fun value -> value.Name,value) |> Map.ofList
      EndpointReturns=Map.empty; PlatformLibraryPath=None; PlatformDescription=Some "MemoryAuthority.description"
      PlatformArchitecture=None; PlatformOS=None; PlatformSourcePaths=Set.singleton(Path.GetFullPath "memory-authority.clef")
      Predicates=Map.empty; FreestandingStartup=None; SubstrateKind=None; RuntimeModel=None
      AvailableMemorySpaces=[]; DefaultMemorySpace=None; ClockFrequencyMhz=None; NsPerWeightUnit=None }

let private memoryAuthority =
    let representations =
        memoryPlatform.Representations.Values |> Seq.map (fun rep ->
            sprintf "{ Name=%A; Capability=%A; Family=%A; Bits=%d; MinMagnitude=%A; MaxMagnitude=%A; Boundary=%A }"
                rep.Name rep.Capability rep.Family rep.Bits rep.MinMagnitude rep.MaxMagnitude rep.Boundary) |> String.concat "; "
    let template = """module MemoryAuthority
type WidthDeclaration = { Name: string; Bits: int }
type Representation = { Name: string; Capability: string; Family: string; Bits: int; MinMagnitude: string; MaxMagnitude: string; Boundary: string }
type TargetCore = { Arch: string; Os: string; Runtime: string; Triple: string; Widths: WidthDeclaration array; Representations: Representation array }
type MemorySpace = { Name: string; Kind: string; Capacity: int; Alignment: int; Granularity: int; Growth: string; Access: string; Base: int option }
type ProgramLifetimeSpaces = { Immutable: string; Mutable: string option }
type PlatformDescription = { Id: string; Core: TargetCore option; Spaces: MemorySpace array; ProgramLifetime: ProgramLifetimeSpaces option }
let rodata = { Name="rodata"; Kind="rodata"; Capacity=4096; Alignment=16; Granularity=16; Growth="fixed"; Access="r"; Base=None }
let stack = { Name="stack"; Kind="stack"; Capacity=4096; Alignment=16; Granularity=16; Growth="down"; Access="rw"; Base=None }
let description = { Id="memory-witness-test"
                    Core=Some { Arch="x86_64"; Os="linux"; Runtime="freestanding"; Triple="x86_64-unknown-linux-gnu"; Widths=[| { Name="Pointer"; Bits=64 }; { Name="Register"; Bits=64 } |]; Representations=[| REPRESENTATIONS |] }
                    Spaces=[| rodata; stack |]; ProgramLifetime=Some { Immutable="rodata"; Mutable=None } }
"""
    template.Replace("REPRESENTATIONS",representations)

let private program body = "module MemoryFixture\n[<EntryPoint>]\nlet main _ =\n" + body
let private programs =
    [ "readonly", program "    let values = [| 7; 11 |]\n    values.[1]\n"
      "mutable", program "    let values = [| 7; 11 |]\n    values.[0] <- 19\n    values.[1]\n"
      "cell", program "    let mutable value = 7\n    let _ = eager (&value)\n    value <- 11\n    value\n"
      "element", program "    let values = [| 7; 11 |]\n    let _ = eager (&values.[1])\n    values.[0]\n"
      "field", "module MemoryFixture\ntype Cell = { Other: int; mutable Value: int }\n[<EntryPoint>]\nlet main _ =\n    let cell = { Other = 11; Value = 7 }\n    let _ = eager (&cell.Value)\n    cell.Value\n" ]

let checkMemoryProgram source path =
    let parse source path =
        match Clef.Compiler.NativeService.parseStringWithDefaults source path with
        | Clef.Compiler.NativeService.ParseSuccess input -> input
        | Clef.Compiler.NativeService.ParseError errors -> failwithf "Memory source did not parse: %A" errors
    Clef.Compiler.NativeService.checkParsedInputsWithPlatform
        [parse memoryAuthority "memory-authority.clef"; parse source path] (Some memoryPlatform)

let private checkedPrograms =
    programs |> List.map (fun (name, source) -> name, lazy (
        let checkedSource = checkMemoryProgram source ("memory-" + name + ".clef")
        let errors = checkedSource.Diagnostics |> List.filter (fun diagnostic ->
            Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.Diagnostic.effectiveSeverity diagnostic =
                Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
        Assert.Empty errors
        let graph = checkedSource.Graph
        let memory = Publication.tryMemory graph |> Result.defaultWith failwith
        graph,memory)) |> Map.ofList

let private witnessedPrograms =
    checkedPrograms |> Map.map (fun _ checkedSource -> lazy (
        let graph,_ = checkedSource.Value
        let proof = Core.ProofDispatch.dischargeSource graph None |> Result.defaultWith failwith
        let witnessed,_ = MiddleEnd.MLIRGeneration.generateWithLinkedLibrariesAndProof graph graph.Platform.Value
                              Core.Types.Dialects.Console Core.Types.Dialects.CPU None Set.empty (Some proof) |> Result.defaultWith failwith
        witnessed))

let private allOperations (input: Core.Types.Pipeline.BackEndInput) = input.Operations |> List.collect (Catalog.flatten >> Seq.toList)

[<Theory>]
[<InlineData("readonly")>]
[<InlineData("mutable")>]
[<InlineData("cell")>]
[<InlineData("element")>]
[<InlineData("field")>]
let ``actual registry witnesses admitted array storage guards and actual mutable places`` name =
    let _,memory = checkedPrograms[name].Value
    Assert.NotEmpty memory.Operations
    let witnessed = witnessedPrograms[name].Value
    Core.Types.Pipeline.WitnessedInput.validate witnessed |> Result.defaultWith failwith
    MlirComponentTests.mlirOpt ["--verify-each"] witnessed.Text |> ignore
    let operations = allOperations witnessed
    let expectedGuards =
        memory.Operations.Values |> Seq.choose (function
            | MemoryWitnessOperation.ArrayAccess access -> Some access.Bounds.Requirement
            | MemoryWitnessOperation.Address { Place=MemoryPlace.ArrayElement(_,_,bounds) } -> Some bounds.Requirement
            | _ -> None) |> Seq.toList
    for guard in expectedGuards do
        let guardPosition = operations |> List.findIndex (function MLIROp.Assert(_,diagnostic) -> diagnostic = guard.Diagnostic | _ -> false)
        let value = Alex.Traversal.Values.value guard.Continuation
        let accessPosition = operations |> List.findIndex (function
            | MLIROp.MemRefOp(MemRefOp.Load(result,_,_,_,_)) -> result = value 1
            | MLIROp.MemRefOp(MemRefOp.Store(_,_,[index],_,_)) -> index = value 0
            | MLIROp.MemRefOp(MemRefOp.ExtractStridedMetadata(baseBuffer,_,_,_,_,_,_)) -> baseBuffer = value 0
            | _ -> false)
        Assert.True(guardPosition < accessPosition, "The actual source requirement must precede its exact memory continuation.")
    let arrays = operations |> List.choose (function MLIROp.GlobalArray(_,_,authority) -> Some authority | _ -> None)
    let expected =
        memory.Operations.Values |> Seq.choose (function
            | MemoryWitnessOperation.ArrayLiteral({ Residence=MemoryResidence.ImmutableProgram _ } as literal) -> Some literal | _ -> None) |> Seq.toList
    Assert.Equal<MemoryArrayLiteralWitness list>(expected,arrays)
    if name = "mutable" || name = "element" then
        Assert.Contains(operations, fun operation -> match operation with MLIROp.MemRefOp(MemRefOp.Alloca(_,TMemRefStatic(2,_),_)) -> true | _ -> false)
    if name = "cell" || name = "element" || name = "field" then
        Assert.Contains(operations, fun operation -> match operation with MLIROp.MemRefOp(MemRefOp.ExtractStridedMetadata _) -> true | _ -> false)

[<Fact>]
let ``element address preserves actual descriptor offset stride index and source element bytes`` () =
    let graph,memory = checkedPrograms["element"].Value
    let address = memory.Operations.Values |> Seq.choose (function MemoryWitnessOperation.Address fact -> Some fact | _ -> None) |> Assert.Single
    let buffer,index,bounds = match address.Place with MemoryPlace.ArrayElement(buffer,index,bounds) -> buffer,index,bounds | _ -> failwith "Expected actual indexed place"
    let element = SettledScalar.tryType address.Element.Value |> Fixtures.require "Missing published element"
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode buffer (Arg 13) (TMemRef element) operands
    MLIRAccumulator.bindNode index (Arg 19) (Alex.CodeGeneration.TypeMapping.scalarCarrierType bounds.IndexCarrier) operands
    let ctx = context graph address.Site operands
    let output = Alex.Witnesses.MemoryWitness.nanopass.Witness ctx ctx.Zipper.Focus
    let result = match output.Result with TRValue value -> value | other -> failwithf "%A" other
    let metadata =
        output.InlineOps |> List.choose (function
            | MLIROp.MemRefOp(MemRefOp.ExtractStridedMetadata(baseBuffer,offset,_,stride,source,_,_)) -> Some(baseBuffer,offset,stride,source)
            | _ -> None) |> Assert.Single
    let baseBuffer,offset,stride,source = metadata
    Assert.Equal(Arg 13,source)
    let castIndex =
        output.InlineOps |> List.choose (function
            | MLIROp.IndexOp(IndexOp.IndexCastS(result,Arg 19,_,TIndex))
            | MLIROp.IndexOp(IndexOp.IndexCastU(result,Arg 19,_,TIndex)) -> Some result
            | _ -> None) |> Assert.Single
    let product =
        output.InlineOps |> List.choose (function
            | MLIROp.IndexOp(IndexOp.IndexMul(result,left,right)) when left=castIndex && right=stride -> Some result
            | _ -> None) |> Assert.Single
    let elementOffset =
        output.InlineOps |> List.choose (function
            | MLIROp.IndexOp(IndexOp.IndexAdd(result,left,right)) when left=offset && right=product -> Some result
            | _ -> None) |> Assert.Single
    let bytes =
        output.InlineOps |> List.choose (function
            | MLIROp.IndexOp(IndexOp.IndexConst(result,value)) when value=int64 address.ElementBytes.Value -> Some result
            | _ -> None) |> Assert.Single
    let byteOffset =
        output.InlineOps |> List.choose (function
            | MLIROp.IndexOp(IndexOp.IndexMul(result,left,right)) when left=elementOffset && right=bytes -> Some result
            | _ -> None) |> Assert.Single
    let pointer =
        output.InlineOps |> List.choose (function
            | MLIROp.MemRefOp(MemRefOp.ExtractBasePtr(result,source,_)) when source=baseBuffer -> Some result
            | _ -> None) |> Assert.Single
    Assert.Contains(MLIROp.IndexOp(IndexOp.IndexAdd(result.SSA,pointer,byteOffset)), output.InlineOps)
    Assert.DoesNotContain(output.InlineOps, fun operation -> match operation with MLIROp.MemRefOp(MemRefOp.Load _ | MemRefOp.Alloca _) -> true | _ -> false)
    let definition = MLIROp.FuncOp(FuncDef("element_address",
                        [Arg 13,TMemRef element;Arg 19,Alex.CodeGeneration.TypeMapping.scalarCarrierType bounds.IndexCarrier], [TIndex],
                        output.InlineOps @ [MLIROp.FuncOp(Return [result])], FuncVisibility.Public))
    Alex.Dialects.Core.Serialize.moduleToString (Ok address.PointerBits) "address_component" [definition]
    |> MlirComponentTests.mlirOpt ["--verify-each"] |> ignore

[<Fact>]
let ``record field address refuses physical storage inconsistent with the published receiver`` () =
    let graph,memory = checkedPrograms["field"].Value
    let address = memory.Operations.Values |> Seq.choose (function MemoryWitnessOperation.Address fact -> Some fact | _ -> None) |> Assert.Single
    let receiver,bytes = match address.Place with MemoryPlace.RecordField(receiver,bytes,_) -> receiver,bytes | _ -> failwith "Expected actual record field"
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode receiver (Arg 31) (TMemRefStatic(bytes + 1,TInt(IntWidth 8))) operands
    let ctx = context graph address.Site operands
    let output = Alex.Witnesses.MemoryWitness.nanopass.Witness ctx ctx.Zipper.Focus
    match output.Result with
    | TRError diagnostic -> Assert.Contains("source-published receiver storage",diagnostic.Message)
    | other -> failwithf "Different physical receiver retained actual-place authority: %A" other
    Assert.Empty output.InlineOps
    Assert.Empty output.TopLevelOps

[<Theory>]
[<InlineData("erased")>]
[<InlineData("initializer")>]
[<InlineData("residence")>]
let ``immutable array catalog refuses erased or altered source storage inventory`` mutation =
    let witnessed = witnessedPrograms["readonly"].Value
    let catalog = witnessed.Catalog.Value
    let original = witnessed.Operations |> List.choose (function MLIROp.GlobalArray _ as operation -> Some operation | _ -> None) |> Assert.Single
    let changed = match original with
                  | MLIROp.GlobalArray(symbol,ty,authority) when mutation="initializer" ->
                      let initializers =
                          match authority.Initializers.Value with
                          | NativeLiteral.Int(_,kind) :: rest -> NativeLiteral.Int(13L,kind) :: rest
                          | _ -> failwith "Expected the checked integer literal fixture"
                      Some(MLIROp.GlobalArray(symbol,ty,{ authority with Initializers=Some initializers }))
                  | MLIROp.GlobalArray(symbol,ty,authority) when mutation="residence" ->
                      Some(MLIROp.GlobalArray(symbol,ty,{ authority with Residence=MemoryResidence.Stack(authority.Site,authority.Site) }))
                  | _ -> None
    let replace operation = if operation=original then changed else Some operation
    let operations = witnessed.Operations |> List.choose replace
    let definitions = catalog.Units.Head.Definitions |> List.choose (fun row -> replace row.Operation |> Option.map (fun operation -> { row with Operation=operation }))
    let text =
        match witnessed.ModuleName with
        | Some name -> Alex.Dialects.Core.Serialize.moduleToString witnessed.PointerBits name operations
        | None -> "module {\n" + Alex.Dialects.Core.Serialize.opsToString witnessed.PointerBits operations "  " + "\n}"
    match Catalog.create catalog.Scope catalog.Activation definitions operations text witnessed.WritableStorage with
    | Result.Error reason -> Assert.Contains("immutable array inventory",reason)
    | Result.Ok _ -> failwith "Altered physical declarations retained source array authority."
