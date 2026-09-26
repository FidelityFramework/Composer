module Alex.Tests.LambdaOccurrenceTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.ScopeContext
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper

/// A settled component graph shares one environment read and formal across two
/// function occurrences. The second occurrence places that formal at argument 1;
/// the nodes' single Parent fields deliberately describe only the first use.
[<Fact>]
let ``shared body reads each lambda occurrence's block argument and restores outer operands`` () =
    let builder = NodeBuilder()
    let environmentType = Types.mkArrayType Types.uint8Type
    let formal = builder.Create(SemanticKind.PatternBinding "environment", environmentType, dummyRange)
    let padding = builder.Create(SemanticKind.PatternBinding "padding", Types.boolType, dummyRange)
    let capture = builder.Create(SemanticKind.PatternBinding "captured", Types.boolType, dummyRange)
    let externalValue = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let externalBinding = builder.Create(SemanticKind.Binding("alreadyFormed", false, false, None),
                                         Types.boolType, dummyRange, children = [externalValue.Id])
    let externalRead = builder.Create(SemanticKind.VarRef("alreadyFormed", Some externalBinding.Id), Types.boolType, dummyRange)
    let environmentRead = builder.Create(SemanticKind.VarRef("environment", Some formal.Id), environmentType, dummyRange)
    let read = builder.Create(SemanticKind.EnvironmentRead(environmentRead.Id, capture.Id), Types.boolType, dummyRange)
    let body = builder.Create(SemanticKind.Sequential [externalRead.Id; read.Id], Types.boolType, dummyRange)
    let makeLambda parameters ty =
        builder.Create(SemanticKind.Lambda(parameters, body.Id, [], None, LambdaContext.RegularClosure), ty, dummyRange)
    let first = makeLambda ["environment", environmentType, formal.Id] (NativeType.TFun(environmentType, Types.boolType))
    let second = makeLambda ["padding", Types.boolType, padding.Id; "environment", environmentType, formal.Id]
                            (NativeType.TFun(Types.boolType, first.Type))
    let root = builder.Create(SemanticKind.Sequential [first.Id; second.Id], Types.unitType, dummyRange)
    for child, parent in [formal.Id, first.Id; padding.Id, second.Id; body.Id, first.Id
                          environmentRead.Id, read.Id; read.Id, body.Id; externalRead.Id, body.Id
                          first.Id, root.Id; second.Id, root.Id] do
        builder.SetParent(child, parent)
    let slot: ContinuationSlot =
        { Source = capture.Id; ValueType = Types.boolType; IsCapture = true
          Holds = CaptureSlotKind.Scalar SettledSlot.Bool
          Field = { Name = "captured"; Slot = SettledSlot.Bool; Offset = Some 0; Size = Some 1; Align = Some 1 } }
    let proof = builder.Create(SemanticKind.Obligation
        { Id = "shared_environment_layout"; Kind = "continuation-layout"; Logic = "QF_LIA"
          Statement = "The supplied bool field occupies its one-byte environment"
          Source = "Alex component fixture"; Refs = []
          Body = ObligationBody.ContinuationLayout([0, 1, 1], 1, 1) }, Types.unitType, dummyRange)
    let layout: EnvironmentLayout =
        { Owner = first.Id; Implementation = first.Id; Formal = formal.Id
          Slots = [slot]; Bytes = 1; Alignment = 1; Obligations = [proof.Id] }
    let raw = builder.Build []
    let graph =
        { raw with Codata = lazy { raw.Codata.Value with
                                    EnvironmentLayouts = Map.ofList [first.Id, layout]
                                    EnvironmentOrigins = Map.ofList [formal.Id, first.Id; environmentRead.Id, first.Id] } }
    let operands = MLIRAccumulator.empty ()
    let carrier = TMemRefStatic(1, TInt(IntWidth 8))
    MLIRAccumulator.bindNode formal.Id (V(-20, 0)) carrier operands
    MLIRAccumulator.bindNode externalBinding.Id (V(-21, 0)) (TInt(IntWidth 1)) operands
    let parentAssociations, parentTypes = operands.NodeAssoc, operands.SSATypes
    let rootScope = ref (ScopeContext.root ())
    let visited = ref (Set.ofList [externalBinding.Id; externalValue.Id])
    let position = Zipper.create graph root.Id |> require "Missing shared-body root"
    let context =
        { Coeffects = coeffects graph 64; Accumulator = operands; RootAccumulator = operands
          ScopeContext = rootScope; RootScopeContext = rootScope; Graph = graph; Zipper = position
          GlobalVisited = visited; TraversalVisited = visited }
    let readOccurrences = ResizeArray<NodeId>()
    let rec witness ctx node =
        if node.Id = read.Id then
            readOccurrences.Add((Zipper.findEnclosingLambda ctx.Zipper |> require "Read lost its actual lambda occurrence").Id)
        Assert.NotEqual(externalBinding.Id, node.Id)
        Assert.NotEqual(externalValue.Id, node.Id)
        match node.Kind with
        | SemanticKind.Lambda _ ->
            (Alex.Witnesses.LambdaWitness.createNanopass (fun () -> witness)).Witness ctx node
        | SemanticKind.VarRef _ -> Alex.Witnesses.VarRefWitness.nanopass.Witness ctx node
        | SemanticKind.EnvironmentRead _ -> Alex.Witnesses.EnvironmentWitness.nanopass.Witness ctx node
        | _ -> Alex.Witnesses.StructuralWitness.nanopass.Witness ctx node
    visitAllNodes witness context position.Focus visited
    Assert.Empty operands.Errors
    Assert.Equal<NodeId list>([first.Id; second.Id], List.ofSeq readOccurrences)
    Assert.Same(parentAssociations, operands.NodeAssoc)
    Assert.Same(parentTypes, operands.SSATypes)
    let operations = ScopeContext.getOps rootScope.Value
    let definitions =
        operations |> List.choose (function
            | MLIROp.FuncOp(FuncOp.FuncDef(name, parameters, returnType, body, _)) -> Some(name, parameters, returnType, body)
            | _ -> None)
    Assert.Equal(2, definitions.Length)
    for fn, argument in [first, Arg 0; second, Arg 1] do
        let name = Alex.CodeGeneration.CallableSymbols.lambda graph graph.Nodes[fn.Id] false
        let _, parameters, returnTypes, emitted = definitions |> List.find (fun (actual, _, _, _) -> actual = name)
        let returnType = Assert.Single returnTypes
        Assert.Contains((argument, carrier), parameters)
        Assert.Equal(TInt(IntWidth 1), returnType)
        let views = emitted |> List.choose (function
            | MLIROp.MemRefOp(MemRefOp.View(_, source, _, sourceType, _)) -> Some(source, sourceType)
            | _ -> None)
        Assert.Equal((argument, carrier), Assert.Single views)
        let loaded = emitted |> List.choose (function
            | MLIROp.MemRefOp(MemRefOp.LoadAligned(result, _, _, _, _, _)) -> Some result
            | _ -> None) |> Assert.Single
        Assert.Contains(MLIROp.FuncOp(FuncOp.Return([{ SSA = loaded; Type = returnType }])), emitted)
    // Verify the actual witness output, including per-function definitions and
    // block argument uses. This is a component gate, not a source/native oracle.
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "lambda_occurrences" operations
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
    Assert.DoesNotContain("unrealized_conversion_cast", verified)

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``definition discovered inside a dependency is reused by later reference and structural occurrences`` (structuralOccurrence: bool) =
    let builder = NodeBuilder()
    let functionType = NativeType.TFun(Types.boolType, Types.boolType)
    let makeFunction name (dependencies: SemanticNode list) =
        let formal = builder.Create(SemanticKind.PatternBinding "value", Types.boolType, dummyRange)
        let read = builder.Create(SemanticKind.VarRef("value", Some formal.Id), Types.boolType, dummyRange)
        let calls = dependencies |> List.map (fun dependency ->
            let reference = builder.Create(SemanticKind.VarRef("dependency", Some dependency.Id), functionType, dummyRange)
            let call = builder.Create(SemanticKind.Application(reference.Id, [read.Id]), Types.boolType, dummyRange)
            builder.SetParent(reference.Id, call.Id)
            call.Id)
        let body = builder.Create(SemanticKind.Sequential (calls @ [read.Id]), Types.boolType, dummyRange)
        let fn = builder.Create(SemanticKind.Lambda(["value", Types.boolType, formal.Id], body.Id,
                                                   [], None, LambdaContext.RegularClosure), functionType, dummyRange)
        let binding = builder.Create(SemanticKind.Binding(name, false, false, None), functionType,
                                     dummyRange, children = [fn.Id])
        for child, parent in [formal.Id, fn.Id; body.Id, fn.Id; fn.Id, binding.Id; read.Id, body.Id] do
            builder.SetParent(child, parent)
        for call in calls do builder.SetParent(call, body.Id)
        binding, fn
    let dependency, dependencyLambda = makeFunction "dependency" []
    let middle, _ = makeFunction "middle" [dependency]
    let caller, callerLambda = makeFunction "caller" [middle; dependency]
    let raw = builder.Build []
    let graph =
        if structuralOccurrence then
            let bodyId =
                match callerLambda.Kind with
                | SemanticKind.Lambda (_, body, _, _, _) -> body
                | _ -> failwith "Fixture caller is not a lambda"
            let body = raw.Nodes[bodyId]
            let children = body.Children.Head :: dependencyLambda.Id :: body.Children.Tail
            // Materialized ClosureValue nodes and the implementation's named
            // binding both structurally reference the same code Lambda. Its
            // canonical parent still identifies the one code declaration.
            let updated = { body with Kind = SemanticKind.Sequential children; Children = children }
            { raw with Nodes = Map.add bodyId updated raw.Nodes }
        else raw
    let operands = MLIRAccumulator.empty ()
    let rootScope = ref (ScopeContext.root ())
    let visited = ref Set.empty
    let position = Zipper.create graph caller.Id |> require "Missing dependency caller"
    let context =
        { Coeffects = coeffects graph 64; Accumulator = operands; RootAccumulator = operands
          ScopeContext = rootScope; RootScopeContext = rootScope; Graph = graph; Zipper = position
          GlobalVisited = visited; TraversalVisited = visited }
    let dependencyVisits = ResizeArray<NodeId>()
    let rec witness ctx node =
        if node.Id = dependencyLambda.Id then dependencyVisits.Add node.Id
        match node.Kind with
        | SemanticKind.Lambda _ -> (Alex.Witnesses.LambdaWitness.createNanopass (fun () -> witness)).Witness ctx node
        | SemanticKind.VarRef _ -> Alex.Witnesses.VarRefWitness.nanopass.Witness ctx node
        | SemanticKind.Binding _ -> Alex.Witnesses.BindingWitness.nanopass.Witness ctx node
        | SemanticKind.Application _ -> Alex.Witnesses.ApplicationWitness.nanopass.Witness ctx node
        | _ -> Alex.Witnesses.StructuralWitness.nanopass.Witness ctx node
    visitAllNodes witness context position.Focus visited
    Assert.Empty operands.Errors
    Assert.Equal(dependencyLambda.Id, Assert.Single dependencyVisits)
    let operations = ScopeContext.getOps rootScope.Value
    let definitions = operations |> List.choose (function
        | MLIROp.FuncOp(FuncOp.FuncDef(name, _, _, _, _)) -> Some name
        | _ -> None)
    Assert.Equal(3, definitions.Length)
    Assert.Equal(3, definitions |> Set.ofList |> Set.count)
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "dependency_reuse" operations
    MlirComponentTests.mlirOpt ["--verify-each"] text |> ignore

[<Fact>]
let ``occurrence-bound formal read emits its settled refinement before consumption`` () =
    let builder = NodeBuilder()
    let formal = builder.Create(SemanticKind.PatternBinding "value", Types.intType, dummyRange)
    let read = builder.Create(SemanticKind.VarRef("value", Some formal.Id), Types.intType, dummyRange)
    let raw = builder.Build []
    // The component receives the already settled read meet. It must retain the
    // actual occurrence's argument SSA while emitting that physical conversion.
    let meet = { Consumer = read.Id; Operand = read.Id; From = 64; To = 8; Adapt = MeetKind.Truncate }
    let graph = { raw with Codata = lazy { raw.Codata.Value with Meets = Map.ofList [read.Id, [meet]] } }
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode formal.Id (Arg 1) (TInt(IntWidth 64)) operands
    let position = Zipper.create graph read.Id |> require "Missing refined formal read"
    let scope = ref (ScopeContext.root ())
    let visited = ref (Set.singleton formal.Id)
    let context =
        { Coeffects = coeffects graph 64; Accumulator = operands; RootAccumulator = operands
          ScopeContext = scope; RootScopeContext = scope; Graph = graph; Zipper = position
          GlobalVisited = visited; TraversalVisited = visited }
    let output = Alex.Witnesses.VarRefWitness.nanopass.Witness context position.Focus
    match output.Result with
    | TRValue value ->
        Assert.Equal(TInt(IntWidth 8), value.Type)
        Assert.Equal(MLIROp.ArithOp(ArithOp.TruncI(value.SSA, Arg 1, TInt(IntWidth 64), TInt(IntWidth 8))),
                     Assert.Single output.InlineOps)
        let body = output.InlineOps @ [MLIROp.FuncOp(FuncOp.Return([{ SSA = value.SSA; Type = value.Type }]))]
        let definition = MLIROp.FuncOp(FuncOp.FuncDef("refined_read",
            [Arg 0, TInt(IntWidth 1); Arg 1, TInt(IntWidth 64)], [value.Type], body, FuncVisibility.Public))
        let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "refined_formal" [definition]
        MlirComponentTests.mlirOpt ["--verify-each"] text |> ignore
    | other -> failwithf "No value from refined formal read: %A" other
