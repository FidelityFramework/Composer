module Alex.Tests.MutableClosureTests

open System.Diagnostics
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

let private carrier = TMemRefStatic(2, TIndex)

/// Already elaborated function values arrive as operands. This fixture exercises
/// their binding/read/write boundary, not lambda construction or closure proofs.
type private Fixture =
    { Graph: SemanticGraph
      Root: Zipper.PSGZipper
      Initial: NodeId
      Replacement: NodeId
      Cell: NodeId
      FirstRead: NodeId
      Snapshot: NodeId
      Assignment: NodeId
      LatestRead: NodeId
      SnapshotRead: NodeId
      ImmutableFunction: NodeId }

let private fixture () =
    let builder = NodeBuilder()
    let functionType = NativeType.TFun(Types.unitType, Types.boolType)
    let lambda value =
        let parameter = builder.Create(SemanticKind.PatternBinding "_", Types.unitType, dummyRange)
        let body = builder.Create(SemanticKind.Literal(NativeLiteral.Bool value), Types.boolType, dummyRange)
        builder.Create(SemanticKind.Lambda(["_", Types.unitType, parameter.Id], body.Id, [], None,
                                          LambdaContext.RegularClosure), functionType, dummyRange)
    let initial, replacement = lambda false, lambda true
    let cell = builder.Create(SemanticKind.Binding("selectedThunk", true, false, None), functionType,
                              dummyRange, children = [initial.Id])
    let firstRead = builder.Create(SemanticKind.VarRef("selectedThunk", Some cell.Id), functionType, dummyRange)
    let snapshot = builder.Create(SemanticKind.Binding("snapshot", false, false, None), functionType,
                                  dummyRange, children = [firstRead.Id])
    let target = builder.Create(SemanticKind.VarRef("selectedThunk", Some cell.Id), functionType, dummyRange)
    let assignment = builder.Create(SemanticKind.Set(target.Id, replacement.Id), Types.unitType, dummyRange)
    let latestRead = builder.Create(SemanticKind.VarRef("selectedThunk", Some cell.Id), functionType, dummyRange)
    let snapshotRead = builder.Create(SemanticKind.VarRef("snapshot", Some snapshot.Id), functionType, dummyRange)
    let immutableFunction = builder.Create(SemanticKind.Binding("unchanged", false, false, None), functionType,
                                           dummyRange, children = [initial.Id])
    let root = builder.Create(SemanticKind.Sequential
                                [cell.Id; snapshot.Id; assignment.Id; latestRead.Id; snapshotRead.Id; immutableFunction.Id],
                              functionType, dummyRange)
    let raw = builder.Build []
    let nodes =
        [initial.Id; replacement.Id] |> List.fold (fun nodes id ->
            let node = Map.find id nodes
            Map.add id { node with Metadata =
                                    node.Metadata
                                    |> Map.add ClosureMetadata.LambdaExpression (MetadataValue.Bool true)
                                    |> Map.add ClosureMetadata.RequiresClosurePair (MetadataValue.Bool true) } nodes) raw.Nodes
    let graph = { raw with Nodes = nodes }
    { Graph = graph; Root = Zipper.create graph root.Id |> require "Missing component root"
      Initial = initial.Id; Replacement = replacement.Id; Cell = cell.Id; FirstRead = firstRead.Id
      Snapshot = snapshot.Id; Assignment = assignment.Id; LatestRead = latestRead.Id
      SnapshotRead = snapshotRead.Id; ImmutableFunction = immutableFunction.Id }

let private inputs fixture =
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode fixture.Initial (Arg 0) carrier operands
    MLIRAccumulator.bindNode fixture.Replacement (Arg 1) carrier operands
    operands

/// Invoke one public witness at an explicit Huet position. No traversal, semantic
/// elaboration or closure emission is reproduced by the component harness.
let private observe (nanopass: Nanopass) (position: Zipper.PSGZipper) pointerBits operands =
    let rootScope = ref (ScopeContext.root ())
    let scope = ref (ScopeContext.createChild rootScope.Value FunctionLevel)
    let visited = ref Set.empty
    let context =
        { Coeffects = coeffects position.Graph pointerBits
          Accumulator = operands; RootAccumulator = operands
          ScopeContext = scope; RootScopeContext = rootScope
          Graph = position.Graph; Zipper = position
          GlobalVisited = visited; TraversalVisited = visited }
    let nodes, edges, codata, associations = position.Graph.Nodes, position.Graph.Edges, position.Graph.Codata, operands.NodeAssoc
    let output = nanopass.Witness context position.Focus
    Assert.Same(position.Graph, context.Graph)
    Assert.Same(nodes, context.Graph.Nodes)
    Assert.Same(edges, context.Graph.Edges)
    Assert.Same(codata, context.Graph.Codata)
    Assert.Same(associations, operands.NodeAssoc)
    Assert.Empty(visited.Value)
    Assert.Empty(scope.Value.Operations)
    Assert.Empty(rootScope.Value.Operations)
    Assert.Empty(operands.AllOps)
    Assert.Empty(output.TopLevelOps)
    output

let private value output =
    match output.Result with
    | TRValue result -> result
    | other -> failwithf "Expected a witnessed value, got %A" other

let private remember id output operands =
    let result = value output
    MLIRAccumulator.bindNode id result.SSA result.Type operands
    result

let private stages pointerBits =
    let f = fixture ()
    let operands = inputs f
    let binding = observe Alex.Witnesses.BindingWitness.nanopass (f.Root |> atChild f.Cell) pointerBits operands
    let cell = remember f.Cell binding operands
    Assert.Equal(TMemRef carrier, cell.Type)
    Assert.NotEqual(Arg 0, cell.SSA)
    Assert.Equal(Some(TMemRefStatic(1, carrier)), MLIRAccumulator.recallSSAType cell.SSA operands)

    let firstRead = observe Alex.Witnesses.VarRefWitness.nanopass
                        (f.Root |> atChild f.Snapshot |> atChild f.FirstRead) pointerBits operands
    let before = remember f.FirstRead firstRead operands
    Assert.Equal(carrier, before.Type)
    let snapshot = observe Alex.Witnesses.BindingWitness.nanopass (f.Root |> atChild f.Snapshot) pointerBits operands
    let saved = remember f.Snapshot snapshot operands
    Assert.Equal(before.SSA, saved.SSA)
    Assert.Empty(snapshot.InlineOps)

    let assignment = observe Alex.Witnesses.MutableAssignmentWitness.nanopass
                         (f.Root |> atChild f.Assignment) pointerBits operands
    match assignment.Result with
    | TRVoid -> ()
    | other -> failwithf "Assignment did not complete: %A" other
    let latestRead = observe Alex.Witnesses.VarRefWitness.nanopass (f.Root |> atChild f.LatestRead) pointerBits operands
    let after = value latestRead
    let snapshotRead = observe Alex.Witnesses.VarRefWitness.nanopass (f.Root |> atChild f.SnapshotRead) pointerBits operands
    Assert.Equal(before.SSA, (value snapshotRead).SSA)
    Assert.Equal(carrier, (value snapshotRead).Type)
    Assert.Empty(snapshotRead.InlineOps)
    Assert.Equal(carrier, after.Type)
    Assert.NotEqual(before.SSA, after.SSA)
    Assert.Empty(operands.Errors)
    f, operands, cell, before, after, [binding; firstRead; snapshot; assignment; latestRead; snapshotRead]

[<Fact>]
let ``mutable function reassignment targets its cell and leaves a value snapshot intact`` () =
    let _, _, cell, before, after, outputs = stages 64
    let operations = outputs |> List.collect _.InlineOps
    let allocations = operations |> List.choose (function MLIROp.MemRefOp(MemRefOp.Alloca(ssa, ty, _)) -> Some(ssa, ty) | _ -> None)
    Assert.Equal((cell.SSA, TMemRefStatic(1, carrier)), Assert.Single allocations)
    let stores = operations |> List.choose (function MLIROp.MemRefOp(MemRefOp.Store(v, destination, _, ty, memory)) -> Some(v, destination, ty, memory) | _ -> None)
    Assert.Equal<(SSA * SSA * MLIRType * MLIRType) list>(
        [Arg 0, cell.SSA, carrier, TMemRefStatic(1, carrier)
         Arg 1, cell.SSA, carrier, TMemRefStatic(1, carrier)], stores)
    let loads = operations |> List.choose (function MLIROp.MemRefOp(MemRefOp.Load(v, source, _, ty, memory)) -> Some(v, source, ty, memory) | _ -> None)
    Assert.Equal<(SSA * SSA * MLIRType * MLIRType) list>(
        [before.SSA, cell.SSA, carrier, TMemRefStatic(1, carrier)
         after.SSA, cell.SSA, carrier, TMemRefStatic(1, carrier)], loads)

[<Fact>]
let ``immutable lambda binding still forwards the already witnessed function value`` () =
    let f = fixture ()
    let output = observe Alex.Witnesses.BindingWitness.nanopass (f.Root |> atChild f.ImmutableFunction) 64 (inputs f)
    Assert.Equal(Arg 0, (value output).SSA)
    Assert.Equal(carrier, (value output).Type)
    Assert.Empty(output.InlineOps)

[<Fact>]
let ``mutable lambda binding rejects a missing initializer value`` () =
    let f = fixture ()
    let output = observe Alex.Witnesses.BindingWitness.nanopass (f.Root |> atChild f.Cell) 64 (MLIRAccumulator.empty ())
    match output.Result with
    | TRError diagnostic -> Assert.Equal("Mutable binding 'selectedThunk': Initial value not yet witnessed", diagnostic.Message)
    | other -> failwithf "Missing initializer was accepted: %A" other
    Assert.Empty(output.InlineOps)

[<Fact>]
let ``mutable function read rejects a missing binding value`` () =
    let f = fixture ()
    let output = observe Alex.Witnesses.VarRefWitness.nanopass (f.Root |> atChild f.LatestRead) 64 (inputs f)
    match output.Result with
    | TRError diagnostic -> Assert.Equal("VarRef 'selectedThunk': Binding not yet witnessed", diagnostic.Message)
    | other -> failwithf "Missing mutable cell was accepted: %A" other
    Assert.Empty(output.InlineOps)

let private mlirOpt arguments (input: string) =
    let start = ProcessStartInfo("mlir-opt", UseShellExecute = false, RedirectStandardInput = true,
                                RedirectStandardOutput = true, RedirectStandardError = true)
    for argument in arguments do start.ArgumentList.Add argument
    use child = new Process(StartInfo = start)
    if not (child.Start()) then failwith "Cannot start mlir-opt"
    let stdout, stderr = child.StandardOutput.ReadToEndAsync(), child.StandardError.ReadToEndAsync()
    child.StandardInput.Write input
    child.StandardInput.Close()
    if not (child.WaitForExit 20000) then
        child.Kill(true)
        child.WaitForExit()
        failwith "Mutable closure component verification timed out"
    let output, errors = stdout.GetAwaiter().GetResult(), stderr.GetAwaiter().GetResult()
    Assert.True(child.ExitCode = 0, $"mlir-opt exited {child.ExitCode}:\n{errors}\nInput:\n{input}")
    output

[<Theory>]
[<InlineData(32)>]
[<InlineData(64)>]
let ``witnessed closure cell and snapshot verify and lower through standard MLIR`` (pointerBits: int) =
    let _, _, _, snapshot, _, outputs = stages pointerBits
    let operations = outputs |> List.collect _.InlineOps
    let body = operations @ [MLIROp.FuncOp(FuncOp.Return([{ SSA = snapshot.SSA; Type = snapshot.Type }]))]
    let definition = MLIROp.FuncOp(FuncOp.FuncDef("closure_snapshot", [Arg 0, carrier; Arg 1, carrier], [carrier], body, FuncVisibility.Public))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok pointerBits) "mutable_closure_component" [definition]
    let verified = mlirOpt ["--verify-each"] text
    Assert.Contains("memref<1xmemref<2xindex>>", verified)
    let atWidth pass = $"{pass}{{index-bitwidth={pointerBits}}}"
    let passes =
        ["expand-strided-metadata"; "memref-expand"; atWidth "finalize-memref-to-llvm"
         atWidth "convert-index-to-llvm"; atWidth "convert-func-to-llvm"
         atWidth "convert-arith-to-llvm"; "reconcile-unrealized-casts"]
    let lowered = mlirOpt ["--verify-each"; "--pass-pipeline=builtin.module(" + String.concat "," passes + ")"] verified
    Assert.Contains("llvm.func @closure_snapshot", lowered)
    Assert.Contains("llvm.alloca", lowered)
    Assert.Contains("llvm.store", lowered)
    Assert.Contains("llvm.load", lowered)
    Assert.DoesNotContain("memref.", lowered)
    Assert.DoesNotContain("unrealized_conversion_cast", lowered)
