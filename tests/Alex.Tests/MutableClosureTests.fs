module Alex.Tests.MutableClosureTests

open System.Diagnostics
open System.IO
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
module Operands = Alex.Traversal.CallableOperands
module Carriers = Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers
module Storage = Clef.Compiler.PSGSaturation.SemanticGraph.MutableCallableStorage

let private boolean = TInt(IntWidth 1)
let private unitType = TInt(IntWidth 32)
let private codeType = TFunc([unitType], [boolean])
let private ok = function Result.Ok value -> value | Result.Error reason -> failwith reason

/// These component events have already been demanded. The fixture establishes
/// neither eager source evaluation policy nor a source allocation lifetime proof.
type private Fixture = {
    Graph: SemanticGraph
    Root: Zipper.PSGZipper
    Initial: NodeId
    Replacement: NodeId
    Cell: NodeId
    FirstRead: NodeId
    Snapshot: NodeId
    Assignment: NodeId
    Target: NodeId
    LatestRead: NodeId
    SnapshotRead: NodeId
    ImmutableFunction: NodeId
}

let private fixture () =
    let builder = NodeBuilder()
    let functionType = NativeType.TFun(Types.unitType, Types.boolType)
    let named name value =
        let parameter = builder.Create(SemanticKind.PatternBinding "_", Types.unitType, dummyRange)
        let body = builder.Create(SemanticKind.Literal(NativeLiteral.Bool value), Types.boolType, dummyRange)
        let code = builder.Create(SemanticKind.Lambda(["_", Types.unitType, parameter.Id], body.Id, [], None,
                                                    LambdaContext.RegularClosure), functionType, dummyRange)
        let declaration = builder.Create(SemanticKind.Binding(name, false, false, None), functionType,
                                         dummyRange, children = [code.Id])
        let reference = builder.Create(SemanticKind.VarRef(name, Some declaration.Id), functionType, dummyRange)
        declaration, reference
    let firstCode, initial = named "initial_code" false
    let secondCode, replacement = named "replacement_code" true
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
                                [firstCode.Id; secondCode.Id; cell.Id; snapshot.Id; assignment.Id
                                 latestRead.Id; snapshotRead.Id; immutableFunction.Id], functionType, dummyRange)
    let raw = builder.Build []
    let carriers, residuals = Carriers.settle { Layouts = Map.empty; Origins = Map.empty; Known = Map.empty } raw
    Assert.Empty residuals
    let storage = Storage.settle carriers Map.empty raw
    Assert.Empty storage.Residuals
    let codata =
        { raw.Codata.Value with
            CallableCarriers = carriers
            CallableJoins = storage.Joins
            MutableCallableStorage = storage.Storage
            Escapes = Map.ofList [cell.Id, EscapeKind.StackScoped] }
    let graph = { raw with Codata = lazy codata }
    { Graph = graph; Root = Zipper.create graph root.Id |> require "Missing component root"
      Initial = initial.Id; Replacement = replacement.Id; Cell = cell.Id; FirstRead = firstRead.Id
      Snapshot = snapshot.Id; Assignment = assignment.Id; Target = target.Id; LatestRead = latestRead.Id
      SnapshotRead = snapshotRead.Id; ImmutableFunction = immutableFunction.Id }

let private context (position: Zipper.PSGZipper) pointerBits operands =
    let rootScope = ref (ScopeContext.root ())
    let scope = ref (ScopeContext.createChild rootScope.Value FunctionLevel)
    let visited = ref Set.empty
    { Coeffects = coeffects position.Graph pointerBits; Accumulator = operands; RootAccumulator = operands
      ScopeContext = scope; RootScopeContext = rootScope; Graph = position.Graph; Zipper = position
      GlobalVisited = visited; TraversalVisited = visited }

let private inputValue id = { SSA = Alex.Traversal.Values.callableCode id; Type = codeType }

let private inputs fixture =
    let operands = MLIRAccumulator.empty ()
    for id in [fixture.Initial; fixture.Replacement] do
        let position = Zipper.create fixture.Graph id |> require "Missing input occurrence"
        Operands.bind (context position 64 operands) id (inputValue id) None |> ok
    operands

/// One public witness at its actual Huet position. No traversal, elaboration,
/// cell selection, or source demand rule is reproduced by the harness.
let private observe (nanopass: Nanopass) position pointerBits operands =
    let ctx = context position pointerBits operands
    let nodes, edges, codata = position.Graph.Nodes, position.Graph.Edges, position.Graph.Codata
    let scalars, callables, cells = operands.NodeAssoc, operands.CallableAssoc, operands.CallableCellAssoc
    let output = nanopass.Witness ctx position.Focus
    Assert.Same(position.Graph, ctx.Graph)
    Assert.Same(nodes, ctx.Graph.Nodes)
    Assert.Same(edges, ctx.Graph.Edges)
    Assert.Same(codata, ctx.Graph.Codata)
    Assert.Same(scalars, operands.NodeAssoc)
    Assert.Same(callables, operands.CallableAssoc)
    Assert.Same(cells, operands.CallableCellAssoc)
    Assert.Empty(ctx.GlobalVisited.Value)
    Assert.Empty(ctx.ScopeContext.Value.Operations)
    Assert.Empty(ctx.RootScopeContext.Value.Operations)
    Assert.Empty(operands.AllOps)
    Assert.Empty(output.TopLevelOps)
    output

let private callable output =
    match output.Result with TRCallable value -> value | other -> failwithf "Expected callable operands, got %A" other

let private remember id output operands =
    let value = callable output
    MLIRAccumulator.bindCallable id value operands |> ok
    value

let private stages pointerBits =
    let f = fixture ()
    let operands = inputs f
    let binding = observe Alex.Witnesses.BindingWitness.nanopass (f.Root |> atChild f.Cell) pointerBits operands
    let cell =
        match binding.Result with
        | TRCallableCell cell -> MLIRAccumulator.bindCallableCell f.Cell cell operands |> ok; cell
        | other -> failwithf "Expected admitted callable storage, got %A" other
    Assert.Equal(TMemRefStatic(1, TIndex), (Operands.cellDiscriminator cell).Type)
    Assert.Equal(None, Operands.cellEnvironment cell)
    let firstRead = observe Alex.Witnesses.VarRefWitness.nanopass
                        (f.Root |> atChild f.Snapshot |> atChild f.FirstRead) pointerBits operands
    let before = remember f.FirstRead firstRead operands
    let snapshot = observe Alex.Witnesses.BindingWitness.nanopass (f.Root |> atChild f.Snapshot) pointerBits operands
    let saved = remember f.Snapshot snapshot operands
    Assert.Equal(Operands.code before, Operands.code saved)
    Assert.Empty(snapshot.InlineOps)
    let destination = observe Alex.Witnesses.VarRefWitness.nanopass
                          (f.Root |> atChild f.Assignment |> atChild f.Target) pointerBits operands
    Assert.True(match destination.Result with TRVoid -> true | _ -> false)
    Assert.Empty(destination.InlineOps)
    let assignment = observe Alex.Witnesses.MutableAssignmentWitness.nanopass (f.Root |> atChild f.Assignment) pointerBits operands
    Assert.True(match assignment.Result with TRVoid -> true | _ -> false)
    let latestRead = observe Alex.Witnesses.VarRefWitness.nanopass (f.Root |> atChild f.LatestRead) pointerBits operands
    let after = callable latestRead
    let snapshotRead = observe Alex.Witnesses.VarRefWitness.nanopass (f.Root |> atChild f.SnapshotRead) pointerBits operands
    Assert.Equal(Operands.code before, Operands.code (callable snapshotRead))
    Assert.Empty(snapshotRead.InlineOps)
    Assert.NotEqual((Operands.code before).SSA, (Operands.code after).SSA)
    Assert.Equal(codeType, (Operands.code before).Type)
    Assert.Equal(codeType, (Operands.code after).Type)
    Assert.Empty(operands.Errors)
    f, operands, cell, before, after, [binding; firstRead; snapshot; assignment; latestRead; snapshotRead]

[<Fact>]
let ``mutable function reassignment targets its cell and leaves a value snapshot intact`` () =
    let f, _, cell, before, after, outputs = stages 64
    let discriminator = Operands.cellDiscriminator cell
    let operations = outputs |> List.collect _.InlineOps
    let allocations = operations |> List.choose (function MLIROp.MemRefOp(MemRefOp.Alloca(ssa, ty, _)) -> Some(ssa, ty) | _ -> None)
    Assert.Equal((discriminator.SSA, TMemRefStatic(1, TIndex)), Assert.Single allocations)
    let stores = operations |> List.choose (function MLIROp.MemRefOp(MemRefOp.Store(value, destination, _, ty, _)) -> Some(value, destination, ty) | _ -> None)
    Assert.Equal<(SSA * SSA * MLIRType) list>(
        [Alex.Traversal.Values.value f.Cell 3, discriminator.SSA, TIndex
         Alex.Traversal.Values.value f.Assignment 3, discriminator.SSA, TIndex], stores)
    let loads = operations |> List.choose (function MLIROp.MemRefOp(MemRefOp.Load(_, source, _, ty, _)) -> Some(source, ty) | _ -> None)
    Assert.Equal<(SSA * MLIRType) list>([discriminator.SSA, TIndex; discriminator.SSA, TIndex], loads)
    let switches = operations |> List.choose (function MLIROp.SCFOp(SCFOp.IndexSwitch(_, _, _, results)) -> Some results | _ -> None)
    Assert.Equal<(SSA * MLIRType) list list>([[(Operands.code before).SSA, codeType]; [(Operands.code after).SSA, codeType]], switches)
    match Operands.carrier before, Operands.carrier after with
    | Joined first, Joined second -> Assert.Equal(f.FirstRead, first.Read); Assert.Equal(f.LatestRead, second.Read)
    | other -> failwithf "Mutable snapshots acquired a fictitious exact identity: %A" other

[<Fact>]
let ``immutable lambda binding still forwards the already witnessed function value`` () =
    let f = fixture ()
    let output = observe Alex.Witnesses.BindingWitness.nanopass (f.Root |> atChild f.ImmutableFunction) 64 (inputs f)
    Assert.Equal(inputValue f.Initial, Operands.code (callable output))
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

[<Fact>]
let ``mutable read retracts after a write changes even when prior cell operands remain`` () =
    let f, operands, _, _, _, _ = stages 64
    let assignment = f.Graph.Nodes[f.Assignment]
    let changed = { assignment with Kind = SemanticKind.Set(f.Target, f.Initial) }
    let graph = { f.Graph with Nodes = f.Graph.Nodes.Add(assignment.Id, changed) }
    let position = Zipper.create graph f.LatestRead |> require "Missing changed read"
    let output = observe Alex.Witnesses.VarRefWitness.nanopass position 64 operands
    match output.Result with
    | TRError diagnostic -> Assert.Contains("complete source protocol", diagnostic.Message)
    | other -> failwithf "Stale source evidence admitted a read: %A" other
    Assert.Empty(output.InlineOps)

[<Fact>]
let ``scoped operand snapshots retain the shared cell beside callable value snapshots`` () =
    let f, operands, cell, before, _, _ = stages 64
    let scope = MLIRAccumulator.snapshotOperands operands
    MLIRAccumulator.bindNode f.Cell (V(9999, 0)) TIndex operands
    Assert.Equal(None, MLIRAccumulator.recallCallableCell f.Cell operands)
    MLIRAccumulator.restoreOperands scope operands
    let restored = MLIRAccumulator.recallCallableCell f.Cell operands |> require "Lost scoped callable cell"
    Assert.Equal(Operands.cellDiscriminator cell, Operands.cellDiscriminator restored)
    let saved = MLIRAccumulator.recallCallable f.Snapshot operands |> require "Lost scoped value snapshot"
    Assert.Equal(Operands.code before, Operands.code saved)
    let output = observe Alex.Witnesses.VarRefWitness.nanopass (f.Root |> atChild f.LatestRead) 64 operands
    Assert.Equal(codeType, (Operands.code (callable output)).Type)

let private run command arguments (input: string) =
    let start = ProcessStartInfo(command, UseShellExecute = false, RedirectStandardInput = true,
                                RedirectStandardOutput = true, RedirectStandardError = true)
    for argument in arguments do start.ArgumentList.Add argument
    use child = new Process(StartInfo = start)
    if not (child.Start()) then failwithf "Cannot start %s" command
    let stdout, stderr = child.StandardOutput.ReadToEndAsync(), child.StandardError.ReadToEndAsync()
    child.StandardInput.Write input
    child.StandardInput.Close()
    if not (child.WaitForExit 20000) then
        child.Kill(true)
        child.WaitForExit()
        failwith "Mutable closure component verification timed out"
    let output, errors = stdout.GetAwaiter().GetResult(), stderr.GetAwaiter().GetResult()
    Assert.True(child.ExitCode = 0, $"{command} exited {child.ExitCode}:\n{errors}\nInput:\n{input}")
    output

let private mlirOpt arguments input = run "mlir-opt" arguments input

[<Theory>]
[<InlineData(32)>]
[<InlineData(64)>]
let ``witnessed closure cell and snapshot verify and lower through standard MLIR`` pointerBits =
    let f, _, _, snapshot, current, outputs = stages pointerBits
    let implementations =
        [f.Initial, false; f.Replacement, true] |> List.map (fun (source, value) ->
            let carrier = f.Graph.Codata.Value.CallableCarriers[source]
            let symbol = Alex.CodeGeneration.CallableSymbols.lambda f.Graph f.Graph.Nodes[carrier.Implementation] false
            let result = { SSA = V(NodeId.value carrier.Implementation, 0); Type = boolean }
            MLIROp.FuncOp(FuncOp.FuncDef(symbol, [Arg 0, unitType], [boolean],
                [MLIROp.ArithOp(ArithOp.ConstI(result.SSA, (if value then 1L else 0L), boolean))
                 MLIROp.FuncOp(FuncOp.Return [result])], FuncVisibility.Private)))
    let inputOps =
        [f.Initial; f.Replacement] |> List.map (fun source ->
            let carrier = f.Graph.Codata.Value.CallableCarriers[source]
            let symbol = Alex.CodeGeneration.CallableSymbols.lambda f.Graph f.Graph.Nodes[carrier.Implementation] false
            MLIROp.FuncOp(FuncOp.FuncConstant((inputValue source).SSA, symbol, codeType)))
    // Invoke both actual function operands after the write. This artifact is
    // also suitable for the native oracle: saved=false and current=true.
    let savedResult = { SSA = V(900001, 0); Type = boolean }
    let currentResult = { SSA = V(900001, 1); Type = boolean }
    let unitArgument = { SSA = V(900001, 2); Type = unitType }
    let body = inputOps @ (outputs |> List.collect _.InlineOps) @
               [MLIROp.ArithOp(ArithOp.ConstI(unitArgument.SSA, 0L, unitType))
                MLIROp.FuncOp(FuncOp.FuncCallIndirect([savedResult], (Operands.code snapshot).SSA, [unitArgument]))
                MLIROp.FuncOp(FuncOp.FuncCallIndirect([currentResult], (Operands.code current).SSA, [unitArgument]))
                MLIROp.FuncOp(FuncOp.Return [savedResult; currentResult])]
    let definition = MLIROp.FuncOp(FuncOp.FuncDef("closure_snapshot", [], [boolean; boolean], body, FuncVisibility.Public))
    let read ordinal ty : Val = { SSA = V(900002, ordinal); Type = ty }
    let saved, latest, truth, unchanged, passed = read 0 boolean, read 1 boolean, read 2 boolean, read 3 boolean, read 4 boolean
    let zero, one, status = read 5 unitType, read 6 unitType, read 7 unitType
    let main = MLIROp.FuncOp(FuncOp.FuncDef("main", [], [unitType],
        [MLIROp.FuncOp(FuncOp.FuncCall([saved; latest], "closure_snapshot", []))
         MLIROp.ArithOp(ArithOp.ConstI(truth.SSA, 1L, boolean))
         MLIROp.ArithOp(ArithOp.XorI(unchanged.SSA, saved.SSA, truth.SSA, boolean))
         MLIROp.ArithOp(ArithOp.AndI(passed.SSA, unchanged.SSA, latest.SSA, boolean))
         MLIROp.ArithOp(ArithOp.ConstI(zero.SSA, 0L, unitType))
         MLIROp.ArithOp(ArithOp.ConstI(one.SSA, 1L, unitType))
         MLIROp.ArithOp(ArithOp.Select(status.SSA, passed.SSA, zero.SSA, one.SSA, unitType))
         MLIROp.FuncOp(FuncOp.Return [status])], FuncVisibility.Public))
    let text = Alex.Dialects.Core.Serialize.moduleToString (Ok pointerBits) "mutable_closure_component" (implementations @ [definition; main])
    let verified = mlirOpt ["--verify-each"] text
    Assert.Contains("memref<1xindex>", verified)
    Assert.Contains("scf.index_switch", verified)
    // The stock Func printer elides its dialect prefix inside func.func.
    Assert.Contains("call_indirect", verified)
    Assert.DoesNotContain("memref<2xindex>", verified)
    Assert.DoesNotContain("unrealized_conversion_cast", verified)
    let atWidth pass = $"{pass}{{index-bitwidth={pointerBits}}}"
    let passes =
        ["convert-scf-to-cf"; "expand-strided-metadata"; "memref-expand"; atWidth "finalize-memref-to-llvm"
         atWidth "convert-index-to-llvm"; atWidth "convert-func-to-llvm"; atWidth "convert-arith-to-llvm"
         "convert-cf-to-llvm"; "reconcile-unrealized-casts"]
    let lowered = mlirOpt ["--verify-each"; "--pass-pipeline=builtin.module(" + String.concat "," passes + ")"] verified
    Assert.Contains("llvm.func @closure_snapshot", lowered)
    Assert.Contains("llvm.alloca", lowered)
    Assert.Contains("llvm.store", lowered)
    Assert.Contains("llvm.load", lowered)
    Assert.DoesNotContain("memref.", lowered)
    Assert.DoesNotContain("unrealized_conversion_cast", lowered)
    if pointerBits = 64 then
        let directory = Path.Combine(Path.GetTempPath(), "composer-mutable-callable-" + System.Guid.NewGuid().ToString("N"))
        Directory.CreateDirectory directory |> ignore
        let llvm = run "mlir-translate" ["--mlir-to-llvmir"] lowered
        let input, executable = Path.Combine(directory, "snapshot.ll"), Path.Combine(directory, "snapshot")
        File.WriteAllText(input, llvm)
        run "clang" [input; "-O0"; "-o"; executable] "" |> ignore
        Assert.Equal("", run executable [] "")
        Directory.Delete(directory, true)
