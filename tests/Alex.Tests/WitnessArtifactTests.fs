module Alex.Tests.WitnessArtifactTests

open System
open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.ScopeContext
open Core.Types.WitnessArtifacts
open Core.Types.Pipeline
open Alex.Tests.Fixtures
module Catalog = Core.WitnessArtifacts
module Zipper = Alex.Traversal.PSGZipper

let private definition name body = MLIROp.FuncOp(FuncDef(name, [], [], body @ [MLIROp.FuncOp(Return [])], FuncVisibility.Private))
let private text ops = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "catalog" ops
let private good result = Result.defaultWith failwith result
let private refused (part: string) (result: Result<'a, string>) =
    match result with
    | Result.Error message -> Assert.Contains(part, message)
    | Result.Ok _ -> failwith "Expected exact correspondence to be refused"

let private fixtureWith publish =
    let builder = NodeBuilder()
    let first = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let sibling = builder.Create(SemanticKind.Literal(NativeLiteral.Bool false), Types.boolType, dummyRange)
    let root = builder.Create(SemanticKind.Sequential [first.Id; sibling.Id], Types.boolType, dummyRange)
    let graph = builder.Build [] |> publish
    let position = Zipper.create graph root.Id |> require "Missing fixture root" |> Zipper.down 0 |> require "Missing child"
    let accumulator = MLIRAccumulator.empty ()
    let rootScope = ref (ScopeContext.root ())
    let visited = ref Set.empty
    let context: WitnessContext =
        { Graph = graph; Zipper = position; Accumulator = accumulator; RootAccumulator = accumulator
          Coeffects = coeffects graph 64; ScopeContext = rootScope; RootScopeContext = rootScope
          GlobalVisited = visited; TraversalVisited = visited }
    context

let private fixture () = fixtureWith id

let private recorded ctx operations =
    EmissionCorrespondence.record ctx operations
    Assert.Empty ctx.Accumulator.Errors
    let scope = ctx.Accumulator.WitnessScope |> require "Missing witness scope"
    let rows = ctx.Accumulator.EmittedDefinitions
    let catalog = Catalog.create scope CheckedProgramStartup rows operations (text operations) [] |> good
    scope, rows, catalog

[<Fact>]
let ``actual witness occurrence accompanies explicit module import inventory`` () =
    let ctx = fixture ()
    let external = MLIROp.FuncOp(FuncDecl("foreign", [], [], FuncVisibility.Private, []))
    let op = definition "body" [MLIROp.FuncOp(FuncCall([], "foreign", []))]
    EmissionCorrespondence.record ctx [op]
    let scope = ctx.Accumulator.WitnessScope.Value
    let operations = [external; op]
    let rows = ctx.Accumulator.EmittedDefinitions
    let catalog = Catalog.create scope CheckedProgramStartup rows operations (text operations) [] |> good
    Assert.Equal("foreign", (Assert.Single catalog.Units.Head.Imports).Symbol)
    let row = Assert.Single catalog.Units.Head.Definitions
    Assert.Same(ctx.Zipper.Focus, row.Occurrence.Focus)
    Assert.Same(ctx.Zipper.Path.Head.Parent, row.Occurrence.Anchor)
    Assert.Equal<NodeId list>(ctx.Zipper.Path.Head.RightSiblings, (List.head row.Occurrence.Path |> fun (_, _, right) -> right))

[<Theory>]
[<InlineData(0)>]
[<InlineData(1)>]
[<InlineData(2)>]
let ``missing duplicate or changed definition ownership is refused`` mutation =
    let ctx = fixture ()
    let operations = [definition "body" []]
    let scope, rows, catalog = recorded ctx operations
    let changed =
        match mutation with
        | 0 -> []
        | 1 -> rows @ rows
        | _ -> [{ rows.Head with Operation = definition "different" [] }]
    let catalog = { catalog with Units = [{ catalog.Units.Head with Definitions = changed }] }
    Catalog.validate scope operations (text operations) [] catalog |> refused "definition correspondence"

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``same numeric ids do not authorize stale graph or mixed generation occurrences`` changedGraph =
    let ctx = fixture ()
    let operations = [definition "body" []]
    let scope, _, catalog = recorded ctx operations
    let graph = if changedGraph then { ctx.Graph with DeclarationRoots = [] } else ctx.Graph
    let foreignScope = Catalog.beginWholeGraphWitness graph
    let row = catalog.Units.Head.Definitions.Head
    let foreign = { row with Occurrence = { row.Occurrence with Scope = foreignScope } }
    let changed = { catalog with Units = [{ catalog.Units.Head with Definitions = [foreign] }] }
    Catalog.validate scope operations (text operations) [] changed |> refused "different checked graph snapshot or witness run"

[<Theory>]
[<InlineData(0)>]
[<InlineData(1)>]
[<InlineData(2)>]
let ``shared occurrence rejects truncated path wrong siblings or copied focus`` mutation =
    let ctx = fixture ()
    let operations = [definition "body" []]
    let scope, _, catalog = recorded ctx operations
    let row = catalog.Units.Head.Definitions.Head
    let occurrence =
        match mutation with
        | 0 -> { row.Occurrence with Path = [] }
        | 1 ->
            let parent, _, right = row.Occurrence.Path.Head
            { row.Occurrence with Path = [parent, [ctx.Zipper.Focus.Id], right] }
        | _ -> { row.Occurrence with Focus = { row.Occurrence.Focus with IsReachable = not row.Occurrence.Focus.IsReachable } }
    let changed = { catalog with Units = [{ catalog.Units.Head with Definitions = [{ row with Occurrence = occurrence }] }] }
    Catalog.validate scope operations (text operations) [] changed |> refused "actual current PSG focus and Huet path"

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``global reference needs actual data owner and exact view type`` useFunction =
    let ctx = fixture ()
    let owner = if useFunction then definition "data" [] else MLIROp.GlobalBytePool("data", [1uy; 0uy], 1, [])
    let read = MLIROp.MemRefOp(MemRefOp.GetGlobal(V(100, 0), "data", TMemRefStatic(1, TInt(IntWidth 8))))
    let operations = [owner; definition "read" [read]]
    EmissionCorrespondence.record ctx operations
    Catalog.create ctx.Accumulator.WitnessScope.Value CheckedProgramStartup ctx.Accumulator.EmittedDefinitions operations (text operations) []
    |> refused "mistyped global reference"

[<Fact>]
let ``identical duplicate typed imports are refused as duplicate inventory`` () =
    let ctx = fixture ()
    let declared = MLIROp.FuncOp(FuncDecl("foreign", [], [], FuncVisibility.Private, []))
    let operations = [declared; declared; definition "body" []]
    EmissionCorrespondence.record ctx operations
    Catalog.create ctx.Accumulator.WitnessScope.Value CheckedProgramStartup ctx.Accumulator.EmittedDefinitions operations (text operations) []
    |> refused "duplicate typed import inventory"

[<Theory>]
[<InlineData(0)>]
[<InlineData(1)>]
[<InlineData(2)>]
[<InlineData(3)>]
let ``catalog rejects duplicate imports with conflicting byval or visibility`` mutation =
    let descriptor = TMemRefStatic(2, TInt(IntWidth 32))
    let actual = { ParamIndex = 0; SizeBytes = 8; AlignBytes = 4 }
    let altered =
        match mutation with
        | 0 -> { actual with SizeBytes = 16 }
        | 1 -> { actual with AlignBytes = 8 }
        | 2 -> { actual with ParamIndex = 1 }
        | _ -> actual
    let first = MLIROp.FuncOp(FuncDecl("foreign_record", [descriptor; descriptor], [], FuncVisibility.Private, [actual]))
    let second =
        MLIROp.FuncOp(FuncDecl("foreign_record", [descriptor; descriptor], [],
            (if mutation = 3 then FuncVisibility.Public else FuncVisibility.Private), [altered]))
    let ctx = fixture ()
    let operations = [first; second; definition "body" []]
    EmissionCorrespondence.record ctx operations
    Catalog.create ctx.Accumulator.WitnessScope.Value CheckedProgramStartup ctx.Accumulator.EmittedDefinitions operations "physical inventory" []
    |> refused "duplicate typed import inventory"

[<Fact>]
let ``one explicit module declaration serves repeated calls without a repair pass`` () =
    let ctx = fixture ()
    let declaration = MLIROp.FuncOp(FuncDecl("foreign", [], [], FuncVisibility.Private, []))
    let call = MLIROp.FuncOp(FuncCall([], "foreign", []))
    let operations = [declaration; definition "body" [call; call]]
    let _, _, catalog = recorded ctx operations
    Assert.Equal("foreign", (Assert.Single catalog.Units.Head.Imports).Symbol)

[<Fact>]
let ``a nested import is rejected rather than hoisted`` () =
    let ctx = fixture ()
    let declaration = MLIROp.FuncOp(FuncDecl("foreign", [], [], FuncVisibility.Private, []))
    let operations = [definition "body" [declaration]]
    EmissionCorrespondence.record ctx operations
    Catalog.create ctx.Accumulator.WitnessScope.Value CheckedProgramStartup ctx.Accumulator.EmittedDefinitions operations (text operations) []
    |> refused "nested external declaration"

[<Fact>]
let ``planned startup cannot disappear from the emitted unit`` () =
    let builder = NodeBuilder()
    let parameter = builder.Create(SemanticKind.PatternBinding "argument", Types.unitType, dummyRange)
    let body = builder.Create(SemanticKind.Literal(NativeLiteral.Int(0L, NTUKind.NTUint(NTUWidth.Fixed 64))), Types.intType, dummyRange)
    let lambda = builder.Create(SemanticKind.Lambda(["argument", Types.unitType, parameter.Id], body.Id, [], None, LambdaContext.RegularClosure),
                                NativeType.TFun(Types.unitType, Types.intType), dummyRange)
    let entry = builder.Create(SemanticKind.Binding("main", false, false, Some DeclRoot.EntryPoint), lambda.Type, dummyRange, children = [lambda.Id])
    let graph, errors = Clef.Compiler.Nanopass.ProgramInitialization.normalize [entry.Id] (builder.Build [entry.Id, DeclRoot.EntryPoint])
    Assert.Empty errors
    let plan = Clef.Compiler.PSGSaturation.SemanticGraph.ProgramInitialization.read graph |> require "Missing fixture startup"
    let scope = Catalog.beginWholeGraphWitness graph
    let operation = definition plan.Symbol []
    let occurrence = { Scope = scope; Focus = graph.Nodes[plan.EntryLambda]; Anchor = graph.Nodes[plan.EntryLambda]; Path = [] }
    let row = { Operation = operation; Occurrence = occurrence }
    Catalog.create scope CheckedProgramStartup [row] [operation] (text [operation]) [] |> good |> ignore
    Catalog.create scope CheckedProgramStartup [] [] (text []) [] |> refused "planned startup"

[<Fact>]
let ``backend entry rejects missing catalog changed text and changed content`` () =
    let ctx = fixtureWith prepareSource
    let operations = [definition "body" []]
    let _, _, catalog = recorded ctx operations
    let input: BackEndInput =
        { Operations = operations; PointerBits = Ok 64; ModuleName = Some "catalog"
          Text = text operations; WritableStorage = []; Catalog = Some catalog }
    WitnessedInput.validate input |> good
    WitnessedInput.validate { input with Catalog = None } |> refused "requires a current source"
    WitnessedInput.validate { input with Text = input.Text + "\n" } |> refused "portable text differs"
    let changed = [definition "replacement" []]
    WitnessedInput.validate { input with Operations = changed; Text = text changed } |> refused "definition correspondence"

[<Fact>]
let ``unpublished physical catalog cannot enter production or trigger serialization`` () =
    let ctx = fixture ()
    let operations = [definition "body" []]
    let _, _, catalog = recorded ctx operations
    // Serializing this unimplemented aggregate ABI would throw. Source
    // admission must refuse the unpublished input before serialization starts.
    let unsupported = MLIROp.FuncOp(FuncDecl("foreign", [], [], FuncVisibility.Private,
                                           [{ ParamIndex = 0; SizeBytes = 8; AlignBytes = 8 }]))
    let input: BackEndInput =
        { Operations = [unsupported]; PointerBits = Ok 64; ModuleName = Some "catalog"
          Text = text operations; WritableStorage = []; Catalog = Some catalog }
    let mutable called = false
    let implementation _ _ = called <- true; Ok (IntermediateOnly "sentinel")
    WitnessedInput.compile implementation input Unchecked.defaultof<_> |> refused "Source emission admission"
    Assert.False called

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``erasing boundary tags cannot authorize a stale or unpublished source catalog`` invalidate =
    let source = """module BackendAdmission
type Signedness = Signed | Unsigned
type TypeRef = Integer of Signedness * int | Void
type PassBy = Value | Reference
type CallConv = | CDecl
type Transfer = | Borrowed
type ParameterInfo = { Name: string; Type: TypeRef; PassBy: PassBy }
type FunctionDescriptor = { CName: string; Parameters: ParameterInfo array; ReturnType: TypeRef; CallingConvention: CallConv; OwnershipTransfer: Transfer }
[<FidelityExtern("c", "observe")>]
let observe (value: int) : int = NativeDefault.zeroed ()
let observeDescriptor: Expr<FunctionDescriptor> = <@ {
    CName = "observe"
    Parameters = [| { Name="value"; Type=Integer(Signed,32); PassBy=Value } |]
    ReturnType=Integer(Signed,32); CallingConvention=CDecl; OwnershipTransfer=Borrowed } @>
[<EntryPoint>]
let main _ = observe (-11)
"""
    let original = checkScalarProgram source "backend-admission.clef"
    let publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryBoundary original |> good
    let declaration = Assert.Single publication.Imports.Values
    let operation = MLIROp.FuncOp(BoundaryFuncDecl declaration)
    let originalCatalog = Catalog.create (Catalog.beginWholeGraphWitness original)
                              TargetModuleActivation [] [operation] (text [operation]) [] |> good
    let input: BackEndInput =
        { Operations = [operation]; PointerBits = Ok 64; ModuleName = Some "catalog"
          Text = text [operation]; WritableStorage = []; Catalog = Some originalCatalog }
    WitnessedInput.validate input |> good
    let changed =
        if invalidate then Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.invalidate original
        else { original with DeclarationRoots = [] }
    let erased = MLIROp.FuncOp(FuncDecl(declaration.Symbol, BoundaryAbi.parameters declaration,
                                       BoundaryAbi.results declaration, FuncVisibility.Private, []))
    Assert.Equal(input.Text, text [erased])
    // The isolated physical catalog is deliberately possible. It carries no
    // permission to enter a production backend after source authority is lost.
    let catalog = Catalog.create (Catalog.beginWholeGraphWitness changed)
                      TargetModuleActivation [] [erased] input.Text [] |> good
    let mutable called = false
    let implementation _ _ = called <- true; Ok (IntermediateOnly "sentinel")
    WitnessedInput.compile implementation { input with Operations = [erased]; Catalog = Some catalog } Unchecked.defaultof<_>
    |> refused "Source emission admission"
    Assert.False called
    WitnessedInput.validate input |> good

[<Fact>]
let ``queued globals retain the draining witness occurrence`` () =
    let ctx = fixtureWith prepareSource
    let visited = ctx.GlobalVisited
    let globalOp = MLIROp.GlobalMemref("queued", TMemRefStatic(1, TInt(IntWidth 8)), None)
    let witness (current: WitnessContext) (_: SemanticNode) =
        MLIRAccumulator.tryEmitGlobalMemref "queued" (TMemRefStatic(1, TInt(IntWidth 8))) None current.Accumulator
        WitnessOutput.empty
    Alex.Traversal.NanopassArchitecture.visitAllNodes witness ctx ctx.Zipper.Focus visited
    Assert.Empty ctx.Accumulator.Errors
    let row = Assert.Single ctx.Accumulator.EmittedDefinitions
    Assert.Equal(globalOp, row.Operation)
    Assert.Same(ctx.Zipper.Focus, row.Occurrence.Focus)
    Assert.Equal(1, row.Occurrence.Path.Length)
    Assert.Empty ctx.Accumulator.PendingStaticGlobals
