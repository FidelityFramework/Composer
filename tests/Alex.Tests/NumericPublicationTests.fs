module Alex.Tests.NumericPublicationTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.CodeGeneration.TypeMapping
open Alex.XParsec.PSGCombinators
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.Traversal.ScopeContext
open Alex.Tests.Fixtures
module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission
module Zipper = Alex.Traversal.PSGZipper

// Numeric facts come from the real source owners and declared offered carriers.
// Negative seven fits four bits; its offered int8 carrier discriminates
// publication reads from a witness recomputing ValueRange.width.
let private sample = lazy (
    let source = """module NumericWitness
type Pair = { Value: int; Ready: bool }
[<EntryPoint>]
let main _ =
    let pair = { Value = -7; Ready = true }
    if pair.Ready then pair.Value else 0
"""
    let graph = checkScalarProgram source "numeric-witness.clef"
    let facts = Publication.tryNumeric graph |> Result.defaultWith failwith
    let literal =
        graph.Nodes.Values |> Seq.find (fun node ->
            node.IsReachable &&
            (match node.Kind with
             | SemanticKind.Literal(NativeLiteral.Int(value, _)) -> value = -7L
             | _ -> false))
    let record =
        graph.Nodes.Values |> Seq.find (fun node ->
            match node.Kind with
            | SemanticKind.RecordExpr(fields, _) -> List.map fst fields = ["Value"; "Ready"]
            | _ -> false)
    Assert.Equal(TInt(IntWidth 8), scalarCarrierType facts.Values[literal.Id])
    Assert.Contains(literal.Id, facts.ResultSites)
    graph, facts, literal, record)

let private platform graph fabric =
    { coeffects graph 64 with TargetPlatform = if fabric then Core.Types.Dialects.FPGA else Core.Types.Dialects.CPU }

let private context graph nodeId fabric : WitnessContext =
    let position = Zipper.create graph nodeId |> require "Missing numeric occurrence"
    let accumulator = MLIRAccumulator.empty ()
    let scope = ref (ScopeContext.root ())
    let visited = ref Set.empty
    { Graph = graph; Zipper = position; Coeffects = platform graph fabric
      Accumulator = accumulator; RootAccumulator = accumulator
      ScopeContext = scope; RootScopeContext = scope
      GlobalVisited = visited; TraversalVisited = visited }

let private refuses (part: string) (action: unit -> 'a) =
    let error = Assert.ThrowsAny<System.Exception>(fun () -> action() |> ignore)
    Assert.Contains(part, error.Message)

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``all target readers retain the source offered carrier instead of range width`` fabric =
    let graph, facts, literal, _ = sample.Value
    Assert.Equal(Some(IntWidth 8), nodeWidth graph literal.Id)
    Assert.Equal(IntWidth 8, requireNodeWidth graph literal.Id)
    let selected = scalarCarrierType facts.Values[literal.Id]
    Assert.Equal(selected, mapTypeAt literal.Id (context graph literal.Id fabric))
    Assert.Equal(selected, requireValueType graph literal.Id selected)

[<Theory>]
[<InlineData(0)>]
[<InlineData(1)>]
[<InlineData(16)>]
[<InlineData(32)>]
let ``a missing or different carrier cannot bypass exact source selection`` bits =
    let graph, _, literal, _ = sample.Value
    refuses "instead of its source-published carrier" (fun () ->
        requireValueType graph literal.Id (TInt(IntWidth bits)))

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``missing or stale numeric publication cannot rerun width selection`` stale =
    let graph, _, literal, _ = sample.Value
    let changed =
        if stale then
            { graph with Nodes = graph.Nodes.Add(literal.Id, { literal with ValueRange = Some(ValueRange.Bounded(0I, 65535I)) }) }
        else Publication.invalidate graph
    refuses "source Numeric publication is unavailable" (fun () -> nodeWidth changed literal.Id)
    refuses "source Numeric publication is unavailable" (fun () -> requireValueType changed literal.Id (TInt(IntWidth 8)))

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``aggregate readers on every target use the same published field slots`` fabric =
    let graph, facts, _, record = sample.Value
    let mapped = valueTypeAt graph record.Id
    let actual = mapTypeAt record.Id (context graph record.Id fabric)
    Assert.Equal(mapped, requireValueType graph record.Id actual)
    match facts.Layouts[facts.SourceTypes[record.Id]], actual with
    | SettledLayout.Record(slots, _, _), TStruct(fields, _) ->
        Assert.Equal<string list>(slots |> List.map _.Name, fields |> List.map fst)
        Assert.Equal(TInt(IntWidth 8), fields |> List.find (fst >> (=) "Value") |> snd)
        Assert.Equal(TInt(IntWidth 1), fields |> List.find (fst >> (=) "Ready") |> snd)
    | other -> failwithf "Expected an exact published record layout: %A" other

[<Theory>]
[<InlineData("width")>]
[<InlineData("order")>]
[<InlineData("missing")>]
let ``aggregate correspondence refuses wrong fields and concrete widths`` corruption =
    let graph, _, _, record = sample.Value
    let mapped = valueTypeAt graph record.Id
    let changed =
        match mapped with
        | TStruct(fields, bytes) ->
            let fields =
                match corruption with
                | "width" -> fields |> List.map (fun (name, ty) -> name, if name = "Value" then TInt(IntWidth 32) else ty)
                | "order" -> List.rev fields
                | _ -> List.tail fields
            TStruct(fields, bytes)
        | other -> failwithf "Expected a record carrier, got %A" other
    refuses "source-published" (fun () -> requireValueType graph record.Id changed)

[<Theory>]
[<InlineData("wrong-width")>]
[<InlineData("unresolved-width")>]
[<InlineData("void")>]
let ``driver rejects arbitrary witness result bypass before committing operations or value`` corruption =
    let graph, _, literal, _ = sample.Value
    let position = Zipper.create graph literal.Id |> require "Missing numeric occurrence"
    let accumulator = MLIRAccumulator.empty ()
    let scope = ref (ScopeContext.root ())
    let visited = ref Set.empty
    let ctx: WitnessContext =
        { Graph = graph; Zipper = position; Coeffects = coeffects graph 64
          Accumulator = accumulator; RootAccumulator = accumulator
          ScopeContext = scope; RootScopeContext = scope
          GlobalVisited = visited; TraversalVisited = visited }
    let supplied = Alex.Traversal.Values.value literal.Id 0
    let result =
        match corruption with
        | "void" -> TRVoid
        | "unresolved-width" -> TRValue { SSA = supplied; Type = TInt(IntWidth 0) }
        | _ -> TRValue { SSA = supplied; Type = TInt(IntWidth 32) }
    let bypass _ _ =
        { InlineOps = [MLIROp.RawMLIR "// must not be committed"]
          TopLevelOps = []; Result = result }
    visitAllNodes bypass ctx position.Focus visited
    let error = Assert.Single accumulator.Errors
    Assert.Equal(Some "published numeric result", error.Phase)
    Assert.Contains("source", error.Message)
    Assert.Empty (ScopeContext.getOps scope.Value)
    Assert.Equal(None, MLIRAccumulator.recallNode literal.Id accumulator)
    Assert.Empty accumulator.SSATypes

[<Fact>]
let ``driver commits an arbitrary witness only at its exact published scalar carrier`` () =
    let graph, facts, literal, _ = sample.Value
    let position = Zipper.create graph literal.Id |> require "Missing numeric occurrence"
    let accumulator = MLIRAccumulator.empty ()
    let scope = ref (ScopeContext.root ())
    let visited = ref Set.empty
    let ctx: WitnessContext =
        { Graph = graph; Zipper = position; Coeffects = coeffects graph 64
          Accumulator = accumulator; RootAccumulator = accumulator
          ScopeContext = scope; RootScopeContext = scope
          GlobalVisited = visited; TraversalVisited = visited }
    let supplied = Alex.Traversal.Values.value literal.Id 0
    let ty = scalarCarrierType facts.Values[literal.Id]
    Assert.Equal(ty, mapTypeAt literal.Id ctx)
    let exact _ _ =
        { InlineOps = [MLIROp.ArithOp(ArithOp.ConstI(supplied, -7L, ty))]
          TopLevelOps = []; Result = TRValue { SSA = supplied; Type = ty } }
    visitAllNodes exact ctx position.Focus visited
    Assert.Empty accumulator.Errors
    Assert.Single (ScopeContext.getOps scope.Value) |> ignore
    Assert.Equal(Some(supplied, ty), MLIRAccumulator.recallNode literal.Id accumulator)
