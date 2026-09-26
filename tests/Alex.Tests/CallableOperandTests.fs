module Alex.Tests.CallableOperandTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures
module Carriers = Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers
module Operands = Alex.Traversal.CallableOperands
module Zipper = Alex.Traversal.PSGZipper

let private ok = function Result.Ok value -> value | Result.Error reason -> failwith reason
let private failure = function Result.Error reason -> reason | Result.Ok _ -> failwith "Expected a rejected callable contract"
let private boolean = TInt(IntWidth 1)

type private Fixture = {
    Graph: SemanticGraph
    Inputs: Carriers.Inputs
    Owner: NodeId
    Alias: NodeId
    Other: NodeId
    Implementation: NodeId
    Formal: NodeId
}

/// These are already settled component participants. They establish no source
/// residence proof; source admission and native execution remain separate gates.
let private fixture captured =
    let builder = NodeBuilder()
    let sourceType = NativeType.TFun(Types.boolType, Types.boolType)
    let environmentType = Types.mkArrayType Types.uint8Type
    let formal = builder.Create(SemanticKind.PatternBinding "environment", environmentType, dummyRange)
    let argument = builder.Create(SemanticKind.PatternBinding "argument", Types.boolType, dummyRange)
    let capture = builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let body = builder.Create(SemanticKind.VarRef("argument", Some argument.Id), Types.boolType, dummyRange)
    let parameters =
        (if captured then ["environment", environmentType, formal.Id] else []) @
        ["argument", Types.boolType, argument.Id]
    let implementation = builder.Create(
        SemanticKind.Lambda(parameters, body.Id, [], None, LambdaContext.RegularClosure),
        (if captured then NativeType.TFun(environmentType, sourceType) else sourceType), dummyRange)
    let owner = builder.Create(SemanticKind.PatternBinding "owner", sourceType, dummyRange)
    let environment = builder.Create(SemanticKind.EnvironmentCreate(owner.Id, [capture.Id, capture.Id]), environmentType, dummyRange)
    let owner =
        if captured then builder.CompleteNode(owner.Id, SemanticKind.ClosureValue(implementation.Id, environment.Id), [])
        else builder.CompleteNode(owner.Id, SemanticKind.Binding("plainCode", false, false, None), [implementation.Id])
    let alias = builder.Create(SemanticKind.VarRef("first", Some owner.Id), sourceType, dummyRange)
    let other = builder.Create(SemanticKind.VarRef("second", Some owner.Id), sourceType, dummyRange)
    let slot: ContinuationSlot =
        { Source = capture.Id; ValueType = Types.boolType; IsCapture = true; Holds = CaptureSlotKind.Scalar SettledSlot.Bool
          Field = { Name = "capture"; Slot = SettledSlot.Bool; Offset = Some 0; Size = Some 1; Align = Some 1 } }
    let layout: EnvironmentLayout =
        { Owner = owner.Id; Implementation = implementation.Id; Formal = formal.Id
          Slots = [slot]; Bytes = 1; Alignment = 1; Obligations = [] }
    let ids = [owner.Id; alias.Id; other.Id]
    let inputs: Carriers.Inputs =
        { Layouts = if captured then Map.ofList [owner.Id, layout] else Map.empty
          Origins = if captured then (formal.Id :: ids) |> List.map (fun id -> id, owner.Id) |> Map.ofList else Map.empty
          Known = if captured then ids |> List.map (fun id -> id, { Implementation = implementation.Id; EnvironmentOwner = owner.Id }) |> Map.ofList else Map.empty }
    let raw = builder.Build []
    let raw =
        { raw with Edges = raw.Edges @
                            (if captured then
                                [{ Class = EdgeClass.Provenance; Role = EdgeRole.EnvironmentFormal
                                   Sources = [owner.Id; implementation.Id]; Target = formal.Id; Ordinal = 0 }
                                 { Class = EdgeClass.Provenance; Role = EdgeRole.EnvironmentCapture false
                                   Sources = [owner.Id; capture.Id; capture.Id]; Target = environment.Id; Ordinal = 0 }]
                             else []) }
    // The source projection must be usable while final Codata is being built.
    let unpublished = { raw with Codata = lazy (failwith "Premature Codata force") }
    let carriers, residuals = Carriers.settle inputs unpublished
    Assert.Empty residuals
    let graph =
        { raw with Codata = lazy { raw.Codata.Value with
                                      CallableCarriers = carriers; KnownCallables = inputs.Known
                                      EnvironmentLayouts = inputs.Layouts; EnvironmentOrigins = inputs.Origins } }
    { Graph = graph; Inputs = inputs; Owner = owner.Id; Alias = alias.Id; Other = other.Id
      Implementation = implementation.Id; Formal = formal.Id }

let private context graph occurrence =
    let accumulator = MLIRAccumulator.empty ()
    let scope = ref (Alex.Traversal.ScopeContext.ScopeContext.root ())
    let visited = ref Set.empty
    { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
      ScopeContext = scope; RootScopeContext = scope; Graph = graph
      Zipper = Zipper.create graph occurrence |> require "Missing callable occurrence"
      GlobalVisited = visited; TraversalVisited = visited }

let private code shape ssa : Val = { SSA = ssa; Type = Operands.functionType shape }
let private environment shape ssa : Val =
    { SSA = ssa; Type = Operands.environmentType shape |> require "No environment in captured fixture" }
let private heldEnvironment value = Operands.environment value |> require "Captured operand lost its environment"

[<Fact>]
let ``source carrier keeps declared physical formals and distinct actual occurrences`` () =
    let fixture = fixture true
    let first, second = fixture.Graph.Codata.Value.CallableCarriers[fixture.Alias], fixture.Graph.Codata.Value.CallableCarriers[fixture.Other]
    Assert.Equal(fixture.Implementation, first.Implementation)
    Assert.Equal(first.Implementation, second.Implementation)
    Assert.Equal(fixture.Alias, first.Occurrence)
    Assert.Equal(fixture.Other, second.Occurrence)
    Assert.Equal(2, first.Parameters.Length)
    Assert.Equal(fixture.Formal, let _, _, id = first.Parameters.Head in id)

[<Theory>]
[<InlineData("environment-last")>]
[<InlineData("foreign-origin")>]
[<InlineData("body-type")>]
[<InlineData("source-type")>]
[<InlineData("capture-authority")>]
let ``source carrier rejects changed signature and environment participants`` defect =
    let fixture = fixture true
    let implementation = fixture.Graph.Nodes[fixture.Implementation]
    let graph, inputs =
        match defect, implementation.Kind with
        | "environment-last", SemanticKind.Lambda(parameters, body, captures, enclosing, context) ->
            let parameters = List.rev parameters
            let ty = NativeType.TFun(Types.boolType, NativeType.TFun(Types.mkArrayType Types.uint8Type, Types.boolType))
            let changed = { implementation with Kind = SemanticKind.Lambda(parameters, body, captures, enclosing, context); Type = ty }
            { fixture.Graph with Nodes = fixture.Graph.Nodes.Add(implementation.Id, changed) }, fixture.Inputs
        | "foreign-origin", _ -> fixture.Graph, { fixture.Inputs with Origins = fixture.Inputs.Origins.Add(fixture.Alias, fixture.Implementation) }
        | "body-type", SemanticKind.Lambda(_, body, _, _, _) ->
            let result = fixture.Graph.Nodes[body]
            { fixture.Graph with Nodes = fixture.Graph.Nodes.Add(body, { result with Type = Types.unitType }) }, fixture.Inputs
        | "source-type", _ ->
            let source = fixture.Graph.Nodes[fixture.Alias]
            let changed = { source with Type = NativeType.TFun(Types.boolType, Types.unitType) }
            { fixture.Graph with Nodes = fixture.Graph.Nodes.Add(source.Id, changed) }, fixture.Inputs
        | "capture-authority", _ ->
            let edges = fixture.Graph.Edges |> List.filter (fun edge ->
                match edge.Role with EdgeRole.EnvironmentCapture _ -> false | _ -> true)
            { fixture.Graph with Edges = edges }, fixture.Inputs
        | _ -> failwith "Unknown carrier defect"
    let carriers, residuals = Carriers.settle inputs graph
    Assert.False(carriers.ContainsKey fixture.Alias)
    Assert.Contains(residuals, fun residual -> residual.Occurrence = fixture.Alias)

[<Fact>]
let ``same code with different environments survives recall and alias copy without a scalar fallback`` () =
    let fixture = fixture true
    let ctx = context fixture.Graph fixture.Owner
    let shape = Operands.project ctx fixture.Owner |> ok
    let fn = code shape (Arg 0)
    Operands.bind ctx fixture.Owner fn (Some(environment shape (Arg 1))) |> ok
    Operands.bind ctx fixture.Other fn (Some(environment shape (Arg 2))) |> ok
    Operands.copy ctx fixture.Other fixture.Alias |> ok
    let recalled id = MLIRAccumulator.recallCallable id ctx.Accumulator |> require "Callable pair was lost"
    Assert.Equal(Arg 1, (heldEnvironment (recalled fixture.Owner)).SSA)
    Assert.Equal(Arg 2, (heldEnvironment (recalled fixture.Alias)).SSA)
    Assert.Equal(Arg 0, (Operands.code (recalled fixture.Alias)).SSA)
    Assert.True((MLIRAccumulator.recallNode fixture.Alias ctx.Accumulator).IsNone)
    Assert.Equal(Some fn.Type, MLIRAccumulator.recallSSAType fn.SSA ctx.Accumulator)

[<Fact>]
let ``operation scope restore keeps callable pairs and physical SSA types together`` () =
    let fixture = fixture true
    let ctx = context fixture.Graph fixture.Owner
    let shape = Operands.project ctx fixture.Owner |> ok
    Operands.bind ctx fixture.Owner (code shape (Arg 0)) (Some(environment shape (Arg 1))) |> ok
    let outer = MLIRAccumulator.snapshotOperands ctx.Accumulator
    Operands.bind ctx fixture.Owner (code shape (V(900, 0))) (Some(environment shape (V(900, 1)))) |> ok
    Operands.copy ctx fixture.Owner fixture.Alias |> ok
    MLIRAccumulator.restoreOperands outer ctx.Accumulator
    Assert.Same(outer.Callables, ctx.Accumulator.CallableAssoc)
    Assert.Same(outer.Scalars, ctx.Accumulator.NodeAssoc)
    Assert.Same(outer.Types, ctx.Accumulator.SSATypes)
    Assert.True((MLIRAccumulator.recallCallable fixture.Alias ctx.Accumulator).IsNone)
    Assert.True((MLIRAccumulator.recallSSAType (V(900, 1)) ctx.Accumulator).IsNone)
    Assert.Equal(Arg 1, (heldEnvironment (MLIRAccumulator.recallCallable fixture.Owner ctx.Accumulator).Value).SSA)

[<Fact>]
let ``code-only callable rejects an invented environment and captured callable rejects scalar packing`` () =
    let plain = fixture false
    let ctx = context plain.Graph plain.Owner
    let shape = Operands.project ctx plain.Owner |> ok
    let fn = code shape (Arg 0)
    Operands.bind ctx plain.Owner fn None |> ok
    Assert.Single(Operands.values (MLIRAccumulator.recallCallable plain.Owner ctx.Accumulator).Value) |> ignore
    Operands.bind ctx plain.Owner fn (Some { SSA = Arg 1; Type = TMemRefStatic(0, TInt(IntWidth 8)) })
    |> failure |> ignore
    let captured = fixture true
    let ctx = context captured.Graph captured.Owner
    let shape = Operands.project ctx captured.Owner |> ok
    Operands.bind ctx captured.Owner { SSA = Arg 0; Type = TMemRefStatic(2, TIndex) } (Some(environment shape (Arg 1)))
    |> failure |> ignore
    Assert.Empty ctx.Accumulator.CallableAssoc
    Assert.Empty ctx.Accumulator.SSATypes

[<Fact>]
let ``callable component yields code and matching environment together through standard structured control`` () =
    let fixture = fixture true
    let ctx = context fixture.Graph fixture.Owner
    let shape = Operands.project ctx fixture.Owner |> ok
    let left = Operands.create shape (code shape (Arg 1)) (Some(environment shape (Arg 2))) |> ok
    let right = Operands.create shape (code shape (Arg 3)) (Some(environment shape (Arg 4))) |> ok
    let selected = Operands.create shape (code shape (V(901, 0))) (Some(environment shape (V(901, 1)))) |> ok
    let parser = Alex.Patterns.ControlFlowPatterns.pBuildIndexSwitch
                    { SSA = Arg 0; Type = TIndex } [0L, ([], Operands.values left)]
                    ([], Operands.values right) (Operands.values selected)
    let operations = match matchAt parser ctx.Zipper 64 ctx.Accumulator with Result.Ok(operations, _) -> operations | Result.Error reason -> failwith reason
    let result = { SSA = V(901, 2); Type = boolean }
    let call = MLIROp.FuncOp(FuncOp.FuncCallIndirect([result], (Operands.code selected).SSA,
                    [heldEnvironment selected; { SSA = Arg 5; Type = boolean }]))
    let parameters = [Arg 0, TIndex; Arg 1, (Operands.code left).Type; Arg 2, (heldEnvironment left).Type
                      Arg 3, (Operands.code right).Type; Arg 4, (heldEnvironment right).Type; Arg 5, boolean]
    let definition = MLIROp.FuncOp(FuncOp.FuncDef("callable_selection_component", parameters, [boolean],
                        operations @ [call; MLIROp.FuncOp(FuncOp.Return [result])], FuncVisibility.Public))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "callable_selection" [definition]
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] text
    Assert.Contains("call_indirect", verified)
    Assert.DoesNotContain("unrealized_conversion_cast", verified)
    Assert.DoesNotContain("memref<2xindex>", verified)

let private higherOrder captured returnsCallable =
    let fixture = fixture captured
    let builder = NodeBuilder()
    let callableType = fixture.Graph.Nodes[fixture.Owner].Type
    let parameterType = if returnsCallable then Types.boolType else callableType
    let parameter = builder.Create(SemanticKind.PatternBinding "input", parameterType, dummyRange)
    let body =
        if returnsCallable then builder.Create(SemanticKind.VarRef("returned", Some fixture.Owner), callableType, dummyRange)
        else builder.Create(SemanticKind.Literal(NativeLiteral.Bool true), Types.boolType, dummyRange)
    let ty = NativeType.TFun(parameterType, body.Type)
    let implementation = builder.Create(SemanticKind.Lambda(["input", parameterType, parameter.Id], body.Id,
                                                            [], None, LambdaContext.RegularClosure), ty, dummyRange)
    let binding = builder.Create(SemanticKind.Binding("higher", false, false, None), ty, dummyRange, children = [implementation.Id])
    let main =
        if returnsCallable then None
        else
            // A formal may inherit an actual's carrier only through complete
            // internal ingress rooted in a real startup activation.
            let callee = builder.Create(SemanticKind.VarRef("higher", Some binding.Id), ty, dummyRange)
            let call = builder.Create(SemanticKind.Application(callee.Id, [fixture.Owner]), body.Type, dummyRange)
            let unit = builder.Create(SemanticKind.PatternBinding "unit", Types.unitType, dummyRange)
            let entryType = NativeType.TFun(Types.unitType, body.Type)
            let entry = builder.Create(SemanticKind.Lambda(["unit", Types.unitType, unit.Id], call.Id,
                                                         [], None, LambdaContext.RegularClosure), entryType, dummyRange)
            let main = builder.Create(SemanticKind.Binding("main", false, false, Some DeclRoot.EntryPoint), entryType,
                                      dummyRange, children = [entry.Id])
            builder.SetParent(entry.Id, main.Id)
            Some main.Id
    let roots = main |> Option.map (fun id -> [id, DeclRoot.EntryPoint]) |> Option.defaultValue []
    let extension = builder.Build roots
    let raw =
        { fixture.Graph with Nodes = extension.Nodes |> Map.fold (fun nodes id node -> Map.add id node nodes) fixture.Graph.Nodes
                             Edges = fixture.Graph.Edges @ extension.Edges
                             DeclarationRoots = roots }
    let raw, startupErrors = Clef.Compiler.Nanopass.ProgramInitialization.normalize (Option.toList main) raw
    Assert.Empty startupErrors
    let callable = if returnsCallable then body.Id else parameter.Id
    let inputs =
        if captured then
            { fixture.Inputs with
                Known = fixture.Inputs.Known.Add(callable, fixture.Inputs.Known[fixture.Owner])
                Origins = fixture.Inputs.Origins.Add(callable, fixture.Owner) }
        else fixture.Inputs
    let carriers, residuals = Carriers.settle inputs raw
    Assert.Empty residuals
    let graph =
        { raw with Codata = lazy { raw.Codata.Value with
                                      CallableCarriers = carriers; KnownCallables = inputs.Known
                                      EnvironmentLayouts = inputs.Layouts; EnvironmentOrigins = inputs.Origins } }
    graph, binding.Id, callable, fixture.Owner

[<Theory>]
[<InlineData(false, false)>]
[<InlineData(true, false)>]
[<InlineData(false, true)>]
[<InlineData(true, true)>]
let ``higher order and returned values expand only their settled callable components`` captured returnsCallable =
    let graph, binding, callable, original = higherOrder captured returnsCallable
    let ctx = context graph binding
    let shape = Operands.project ctx binding |> ok
    let held = Operands.project ctx original |> ok
    let components = Operands.functionType held :: (Operands.environmentType held |> Option.toList)
    if returnsCallable then
        Assert.Equal<MLIRType list>(components, Operands.resultTypes shape)
        Assert.Equal<MLIRType list list>([[boolean]], Operands.parameterTypes shape)
        Assert.Equal(CallableValueShape.Callable callable, graph.Codata.Value.CallableCarriers[binding].ResultShape)
    else
        Assert.Equal<MLIRType list list>([components], Operands.parameterTypes shape)
        Assert.Equal<MLIRType list>([boolean], Operands.resultTypes shape)
        Assert.Equal<CallableValueShape list>([CallableValueShape.Callable callable], graph.Codata.Value.CallableCarriers[binding].ParameterShapes)
    let missing = { graph with Codata = lazy { graph.Codata.Value with CallableCarriers = graph.Codata.Value.CallableCarriers.Remove callable } }
    let reason = Operands.project (context missing binding) binding |> failure
    Assert.Contains("no settled carrier", reason)

[<Fact>]
let ``equal physical shapes cannot change the settled callable identity during copy`` () =
    let fixture = fixture false
    let builder = NodeBuilder()
    let original = fixture.Graph.Nodes[fixture.Implementation]
    let replacement = builder.Create(original.Kind, original.Type, dummyRange)
    let raw = builder.Build []
    let alien = { fixture.Graph.Codata.Value.CallableCarriers[fixture.Other] with Implementation = replacement.Id }
    let graph =
        { fixture.Graph with
            Nodes = fixture.Graph.Nodes.Add(replacement.Id, raw.Nodes[replacement.Id])
            Codata = lazy { fixture.Graph.Codata.Value with CallableCarriers = fixture.Graph.Codata.Value.CallableCarriers.Add(fixture.Other, alien) } }
    let ctx = context graph fixture.Owner
    let shape = Operands.project ctx fixture.Owner |> ok
    Operands.bind ctx fixture.Owner (code shape (Arg 0)) None |> ok
    Operands.copy ctx fixture.Owner fixture.Other |> failure |> ignore
    Assert.True((MLIRAccumulator.recallCallable fixture.Other ctx.Accumulator).IsNone)

[<Fact>]
let ``cyclic callable signature references fail without recursive emission or a scalar substitute`` () =
    let graph, binding, input, _ = higherOrder false false
    let source = graph.Codata.Value.CallableCarriers[binding]
    let cyclic = { source with Occurrence = input; SourceType = graph.Nodes[input].Type }
    let graph = { graph with Codata = lazy { graph.Codata.Value with CallableCarriers = graph.Codata.Value.CallableCarriers.Add(input, cyclic) } }
    let ctx = context graph binding
    let reason = Operands.project ctx binding |> failure
    Assert.Contains("recursive component reference", reason)
    Assert.Empty ctx.Accumulator.AllOps
    Assert.Empty ctx.Accumulator.NodeAssoc
    Assert.Empty ctx.Accumulator.CallableAssoc

/// Component fixture for a checker-quantified, representation-neutral measure.
/// The code and capture layout stay shared; each alias retains its exact unit.
let private measuredFixture () =
    let existing = fixture true
    let graph = existing.Graph
    let variable = Clef.Compiler.NativeTypedTree.UnionFind.freshMeasureVar (Some "u")
    let parameter = Clef.Compiler.NativeTypedTree.UnionFind.measureCellOf variable
    let measure = Clef.Compiler.NativeTypedTree.DimensionAlgebra.Dimension.ofVar variable
    let measured dimension = NativeType.TNum(CarrierRef.Carrier Types.floatTyCon, dimension)
    let functionOf dimension = NativeType.TFun(measured dimension, measured dimension)
    let genericType = functionOf measure
    let scheme = NativeType.TForall([parameter], genericType)
    let declarationId = NodeId.fresh()
    let code = graph.Nodes[existing.Implementation]
    let parameters, body =
        match code.Kind with
        | SemanticKind.Lambda(parameters, body, [], _, _) -> parameters, body
        | _ -> failwith "Expected fixture code"
    let name, _, argument = List.last parameters
    let parameters = [List.head parameters; name, measured measure, argument]
    let code =
        { code with Kind = SemanticKind.Lambda(parameters, body, [], None, LambdaContext.RegularClosure)
                    Type = NativeType.TFun(graph.Nodes[existing.Formal].Type, genericType)
                    Metadata = code.Metadata.Add(ClosureMetadata.SourceSignature, MetadataValue.Type genericType)
                                             .Add(SchemeMetadata.Declaration, MetadataValue.Type scheme)
                                             .Add(SchemeMetadata.ImplementationDeclaration, MetadataValue.NodeId declarationId) }
    let owner = { graph.Nodes[existing.Owner] with Type = genericType; Parent = Some declarationId }
    let declaration =
        { owner with Id = declarationId; Kind = SemanticKind.Binding("measured", false, false, None)
                     Parent = None; Children = [owner.Id]
                     Metadata = Map.ofList [SchemeMetadata.Declaration, MetadataValue.Type scheme] }
    let alias id unitName =
        let dimension = Clef.Compiler.NativeTypedTree.DimensionAlgebra.Dimension.ofBase { Name = unitName; Module = ["Fixture"] }
        { graph.Nodes[id] with Kind = SemanticKind.VarRef("measured", Some declarationId); Type = functionOf dimension
                               Metadata = Map.ofList [SchemeMetadata.Definition, MetadataValue.NodeId declarationId
                                                      SchemeMetadata.Declaration, MetadataValue.Type scheme
                                                      SchemeMetadata.argument 0, MetadataValue.Type(NativeType.TMeasure dimension)] }
    let nodes =
        graph.Nodes.Add(code.Id, code).Add(owner.Id, owner).Add(declarationId, declaration)
            .Add(argument, { graph.Nodes[argument] with Type = measured measure })
            .Add(body, { graph.Nodes[body] with Type = measured measure })
            .Add(existing.Alias, alias existing.Alias "m")
            .Add(existing.Other, alias existing.Other "s")
    let inputs =
        { existing.Inputs with Known = existing.Inputs.Known.Add(declarationId, existing.Inputs.Known[owner.Id])
                               Origins = existing.Inputs.Origins.Add(declarationId, owner.Id) }
    let raw = { graph with Nodes = nodes }
    let carriers, residuals = Carriers.settle inputs raw
    Assert.Empty residuals
    let graph = { raw with Codata = lazy { graph.Codata.Value with CallableCarriers = carriers; KnownCallables = inputs.Known; EnvironmentOrigins = inputs.Origins } }
    graph, declarationId, existing.Alias, existing.Other, code.Id

[<Fact>]
let ``quantified measure aliases forward the exact recalled code and environment without erasing units`` () =
    let graph, declaration, metre, second, _ = measuredFixture ()
    let ctx = context graph declaration
    let sourceShape = Operands.project ctx declaration |> ok
    Operands.bind ctx declaration (code sourceShape (Arg 0)) (Some(environment sourceShape (Arg 1))) |> ok
    for destination in [metre; second] do
        let value = Operands.reproject ctx declaration destination |> ok
        Assert.Equal(Arg 0, (Operands.code value).SSA)
        Assert.Equal(Arg 1, (heldEnvironment value).SSA)
        Assert.Equal(graph.Nodes[destination].Type, (Operands.carrier value).SourceType)
    Assert.NotEqual(graph.Nodes[metre].Type, graph.Nodes[second].Type)
    Operands.copy ctx declaration metre |> ok
    Operands.reproject ctx metre second |> failure |> ignore

[<Fact>]
let ``unquantified dimensions cannot borrow the shared signature permission`` () =
    let graph, declaration, _, _, implementation = measuredFixture ()
    let changedCode = { graph.Nodes[implementation] with Metadata = graph.Nodes[implementation].Metadata.Remove SchemeMetadata.Declaration }
    let changed = { graph with Nodes = graph.Nodes.Add(implementation, changedCode) }
    Operands.project (context changed declaration) declaration |> failure |> ignore
    let carrier = graph.Codata.Value.CallableCarriers[declaration]
    Operands.components (context graph declaration) (List.last carrier.ParameterShapes) |> failure |> ignore

let private measuredCallFixture () =
    let graph, _, occurrence, _, implementation = measuredFixture ()
    let code = graph.Nodes[implementation]
    let parameters, _ = match code.Kind with SemanticKind.Lambda(parameters, body, _, _, _) -> parameters, body | _ -> failwith "Missing code"
    let valueType = match graph.Nodes[occurrence].Type with NativeType.TFun(input, _) -> input | _ -> failwith "Missing public signature"
    let model = graph.Nodes[occurrence]
    let fresh kind ty children = { model with Id = NodeId.fresh(); Kind = kind; Type = ty; Children = children; Metadata = Map.empty; Parent = None }
    let binding = fresh (SemanticKind.Binding("physical", false, false, None)) code.Type [implementation]
    let callee = fresh (SemanticKind.VarRef("physical", Some binding.Id)) code.Type []
    let environment = fresh (SemanticKind.EnvironmentReference occurrence) (List.head parameters |> fun (_, ty, _) -> ty) [occurrence]
    let argument = fresh (SemanticKind.Literal(NativeLiteral.Float(3.0, NTUKind.NTUfloat(NTUWidth.Fixed 64)))) valueType []
    let call = fresh (SemanticKind.Application(callee.Id, [environment.Id; argument.Id])) valueType [callee.Id; environment.Id; argument.Id]
    let nodes = [binding; callee; environment; argument; call] |> List.fold (fun nodes node -> Map.add node.Id node nodes) graph.Nodes
    let row = { Sources = [occurrence; implementation; environment.Id; argument.Id]; Target = call.Id
                Class = EdgeClass.Provenance; Role = EdgeRole.EnvironmentInvocation; Ordinal = 0 }
    { graph with Nodes = nodes; Edges = row :: graph.Edges }, call.Id, implementation, parameters, row

[<Fact>]
let ``direct physical parameters require the current instantiated call and retain symbolic shared code`` () =
    let graph, site, implementation, parameters, _ = measuredCallFixture ()
    let projected = Operands.parametersAtCall (context graph site) site implementation parameters |> ok
    Assert.Equal(2, projected.Length)
    Assert.Equal<MLIRType list>([TFloat F64], List.last projected)
    Assert.NotEmpty(Clef.Compiler.NativeTypedTree.UnionFind.freeMeasureVars (List.last parameters |> fun (_, ty, _) -> ty))
    Operands.parametersAtCall (context graph implementation) site implementation parameters |> failure |> ignore

[<Theory>]
[<InlineData("missing source call")>]
[<InlineData("wrong source callable")>]
[<InlineData("changed argument dimension")>]
let ``direct parameter projection refuses broken instance correspondence`` change =
    let graph, site, implementation, parameters, row = measuredCallFixture ()
    let graph =
        match change with
        | "missing source call" -> { graph with Edges = List.tail graph.Edges }
        | "wrong source callable" -> { graph with Edges = { row with Sources = implementation :: List.tail row.Sources } :: List.tail graph.Edges }
        | "changed argument dimension" ->
            let argument = graph.Nodes[List.last row.Sources]
            { graph with Nodes = graph.Nodes.Add(argument.Id, { argument with Type = Types.floatType }) }
        | _ -> failwith "Unknown mutation"
    Operands.parametersAtCall (context graph site) site implementation parameters |> failure |> ignore
