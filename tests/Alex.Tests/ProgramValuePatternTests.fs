module Alex.Tests.ProgramValuePatternTests

open Xunit
open Clef.Compiler.NativeService
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Patterns.MemRefPatterns
open Alex.Tests.Fixtures
module Startup = Clef.Compiler.PSGSaturation.SemanticGraph.ProgramInitialization
module Zipper = Alex.Traversal.PSGZipper

let private fixture () =
    let file = System.IO.Path.GetFullPath "program-value-pattern.clef"
    let integer: NumericRepresentation =
        { Name = "signed64"; Capability = "native"; Family = "int"; Bits = 64
          MinMagnitude = "-9223372036854775808"; MaxMagnitude = "9223372036854775807"; Boundary = "wrap" }
    let context: PlatformContext = {
        PlatformId = "program-value-pattern"; Dimensions = Map.ofList ["Pointer", 64; "Register", 64]
        Representations = Map.ofList [integer.Name, integer]; EndpointReturns = Map.empty; PlatformLibraryPath = None
        PlatformDescription = Some "ProgramValuePattern.description"; PlatformArchitecture = None; PlatformOS = None
        PlatformSourcePaths = Set.singleton file; Predicates = Map.empty; FreestandingStartup = None
        SubstrateKind = None; RuntimeModel = None; AvailableMemorySpaces = []
        DefaultMemorySpace = None; ClockFrequencyMhz = None; NsPerWeightUnit = None }
    let source = """module ProgramValuePattern
 type WidthDeclaration = { Name: string; Bits: int }
 type Representation = { Name: string; Capability: string; Family: string; Bits: int; MinMagnitude: string; MaxMagnitude: string; Boundary: string }
 type TargetCore = { Widths: WidthDeclaration array; Representations: Representation array }
 type MemorySpace = { Name: string; Kind: string; Capacity: int; Alignment: int; Granularity: int; Growth: string; Access: string; Base: int option }
 type ProgramLifetimeSpaces = { Immutable: string; Mutable: string option }
 type PlatformDescription = { Id: string; Core: TargetCore option; Spaces: MemorySpace array; ProgramLifetime: ProgramLifetimeSpaces option }
 let image = { Name = "image"; Kind = "rodata"; Capacity = 1024; Alignment = 16; Granularity = 16; Growth = "fixed"; Access = "r"; Base = None }
 let state = { Name = "state"; Kind = "data"; Capacity = 1024; Alignment = 16; Granularity = 16; Growth = "fixed"; Access = "rw"; Base = None }
 let core = { Widths = [| { Name = "Pointer"; Bits = 64 }; { Name = "Register"; Bits = 64 } |]; Representations = [| { Name = "signed64"; Capability = "native"; Family = "int"; Bits = 64; MinMagnitude = "-9223372036854775808"; MaxMagnitude = "9223372036854775807"; Boundary = "wrap" } |] }
 let description = { Id = "program-value-pattern"; Core = Some core; Spaces = [| image; state |]; ProgramLifetime = Some { Immutable = "image"; Mutable = Some "state" } }
 let mutable flag = true
 [<EntryPoint>]
 let main _ = if flag then 0 else 1
"""
    let input = match parseStringWithDefaults source file with ParseSuccess input -> input | ParseError errors -> failwithf "Fixture parse failed: %A" errors
    let result = checkParsedInputsWithPlatform [input] (Some context)
    let errors = result.Diagnostics |> List.filter (fun diagnostic -> Diagnostic.effectiveSeverity diagnostic = NativeDiagnosticSeverity.Error)
    Assert.Empty errors
    let plan = Startup.read result.Graph |> require "Fixture startup was not settled"
    let binding = plan.ValueBindings |> Seq.filter (fun id ->
        match result.Graph.Nodes[id].Kind with SemanticKind.Binding("flag", true, _, _) -> true | _ -> false) |> Assert.Single
    Assert.True((Startup.tryValueAuthority result.Graph binding).IsSome)
    result.Graph, binding

[<Fact>]
let ``program slot initialization consumes the declared writable authority`` () =
    let graph, binding = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    let focus = Zipper.create graph binding |> require "Missing slot fixture"
    let ty = TInt(IntWidth 1)
    let ops =
        match matchAt (pGlobalSlotInit binding "program_slot" (Arg 0) ty) focus 64 accumulator with
        | Result.Ok ((ops, TRValue _), _) -> ops
        | result -> failwithf "Program slot initialization failed: %A" result
    let globals = MLIRAccumulator.drainPendingStaticGlobals accumulator
    Assert.Single globals |> ignore
    let zero = V(-2, 0)
    let resultType = TInt(IntWidth 32)
    let body = ops @ [MLIROp.ArithOp(ArithOp.ConstI(zero, 0L, resultType)); MLIROp.FuncOp(FuncOp.Return([{ SSA = zero; Type = resultType }]))]
    let functionOp = MLIROp.FuncOp(FuncOp.FuncDef("initialize_program_slot", [Arg 0, ty], [resultType], body, FuncVisibility.Public))
    let source = Alex.Dialects.Core.Serialize.moduleToString (Ok 64) "program_slot_component" (globals @ [functionOp])
    let verified = MlirComponentTests.mlirOpt ["--verify-each"] source
    Assert.Contains("memref.global", verified)
    Assert.Contains("memref.store", verified)

[<Theory>]
[<InlineData(false)>]
[<InlineData(true)>]
let ``missing or mismatched authority preserves slot intent and emits no storage`` malformed =
    let graph, binding = fixture ()
    let edges = graph.Edges |> List.choose (fun edge ->
        if edge.Role <> EdgeRole.ProgramValue then Some edge
        elif malformed then Some { edge with Sources = edge.Sources |> List.rev }
        else None)
    let graph = { graph with Edges = edges }
    Assert.True(Startup.isSlotBinding graph binding)
    let accumulator = MLIRAccumulator.empty ()
    let focus = Zipper.create graph binding |> require "Missing slot fixture"
    match matchAt (pGlobalSlotInit binding "program_slot" (Arg 0) (TInt(IntWidth 1))) focus 64 accumulator with
    | Result.Error message -> Assert.Contains("writable-space authority", message)
    | result -> failwithf "Expected a missing authority boundary: %A" result
    Assert.Empty(MLIRAccumulator.drainPendingStaticGlobals accumulator)

let private programCallables () =
    let file = System.IO.Path.GetFullPath "program-callable-pattern.clef"
    let integer: NumericRepresentation =
        { Name = "signed64"; Capability = "native"; Family = "int"; Bits = 64
          MinMagnitude = "-9223372036854775808"; MaxMagnitude = "9223372036854775807"; Boundary = "wrap" }
    let octet: NumericRepresentation =
        { Name = "uint8"; Capability = "native"; Family = "uint"; Bits = 8
          MinMagnitude = "0"; MaxMagnitude = "255"; Boundary = "wrap" }
    let platform: PlatformContext =
        { PlatformId = "program-callable-pattern"; Dimensions = Map.ofList ["Pointer", 64; "Register", 64]
          Representations = Map.ofList [integer.Name, integer; octet.Name, octet]; EndpointReturns = Map.empty
          PlatformLibraryPath = None; PlatformDescription = Some "ProgramCallablePattern.description"
          PlatformArchitecture = None; PlatformOS = None; PlatformSourcePaths = Set.singleton file
          Predicates = Map.empty; FreestandingStartup = None; SubstrateKind = None; RuntimeModel = None
          AvailableMemorySpaces = []; DefaultMemorySpace = None; ClockFrequencyMhz = None; NsPerWeightUnit = None }
    let source = """module ProgramCallablePattern
type WidthDeclaration = { Name: string; Bits: int }
type Representation = { Name: string; Capability: string; Family: string; Bits: int; MinMagnitude: string; MaxMagnitude: string; Boundary: string }
type TargetCore = { Widths: WidthDeclaration array; Representations: Representation array }
type MemorySpace = { Name: string; Kind: string; Capacity: int; Alignment: int; Granularity: int; Growth: string; Access: string; Base: int option }
type ProgramLifetimeSpaces = { Immutable: string; Mutable: string option }
type PlatformDescription = { Id: string; Core: TargetCore option; Spaces: MemorySpace array; ProgramLifetime: ProgramLifetimeSpaces option }
let image = { Name = "image"; Kind = "rodata"; Capacity = 4096; Alignment = 16; Granularity = 16; Growth = "fixed"; Access = "r"; Base = None }
let state = { Name = "state"; Kind = "data"; Capacity = 4096; Alignment = 16; Granularity = 16; Growth = "fixed"; Access = "rw"; Base = None }
let core = { Widths = [| { Name = "Pointer"; Bits = 64 }; { Name = "Register"; Bits = 64 } |]; Representations = [| { Name = "signed64"; Capability = "native"; Family = "int"; Bits = 64; MinMagnitude = "-9223372036854775808"; MaxMagnitude = "9223372036854775807"; Boundary = "wrap" }; { Name = "uint8"; Capability = "native"; Family = "uint"; Bits = 8; MinMagnitude = "0"; MaxMagnitude = "255"; Boundary = "wrap" } |] }
let description = { Id = "program-callable-pattern"; Core = Some core; Spaces = [| image; state |]; ProgramLifetime = Some { Immutable = "image"; Mutable = Some "state" } }
[<Measure>] type m
let make (offset: int<m>) = fun (value: int<m>) -> value + offset
let first = make 3<m>
let second = make 4<m>
let alias = first
[<EntryPoint>]
let main _ =
    ignore (first 1<m>)
    ignore (second 2<m>)
    ignore (alias 3<m>)
    0
"""
    let input = match parseStringWithDefaults source file with ParseSuccess input -> input | ParseError errors -> failwithf "%A" errors
    let result = checkParsedInputsWithPlatform [input] (Some platform)
    Assert.Empty(result.Diagnostics |> List.filter (fun diagnostic -> Diagnostic.effectiveSeverity diagnostic = NativeDiagnosticSeverity.Error))
    result.Graph

let private callableContext graph occurrence accumulator =
    let scope = ref (Alex.Traversal.ScopeContext.ScopeContext.root ())
    let visited = ref Set.empty
    { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
      ScopeContext = scope; RootScopeContext = scope; Graph = graph
      Zipper = Zipper.create graph occurrence |> require "Missing program callable occurrence"
      GlobalVisited = visited; TraversalVisited = visited }

[<Fact>]
let ``program callable references retain distinct measured instances through passive witnesses`` () =
    let graph = programCallables ()
    let source name = graph.Nodes.Values |> Seq.filter (fun node ->
        node.IsReachable && match node.Kind with SemanticKind.Binding(actual, false, _, _) -> actual = name | _ -> false) |> Assert.Single
    let instances = ["first"; "second"; "alias"] |> List.map (fun name ->
        let binding = source name
        let instance = Clef.Compiler.Nanopass.ClosureEnvironmentSettlement.programInstance graph binding.Id |> require "Missing settled program instance"
        binding, instance)
    Assert.Equal((snd instances[0]).Carrier.Implementation, (snd instances[1]).Carrier.Implementation)
    Assert.NotEqual((snd instances[0]).Allocation, (snd instances[1]).Allocation)
    Assert.Equal((snd instances[0]).Allocation, (snd instances[2]).Allocation)
    for binding, instance in instances do
        let references = graph.Nodes.Values |> Seq.filter (fun node ->
            node.IsReachable && match node.Kind with SemanticKind.VarRef(_, Some id) -> id = binding.Id | _ -> false) |> Seq.toList
        Assert.NotEmpty references
        for reference in references do
            let accumulator = MLIRAccumulator.empty ()
            let ctx = callableContext graph reference.Id accumulator
            let output = Alex.Witnesses.VarRefWitness.nanopass.Witness ctx reference
            let callable =
                match output.Result with TRCallable value -> value | other -> failwithf "Expected program callable: %A; %A" other output.InlineOps
            Assert.Equal(reference.Id, (Alex.Traversal.CallableOperands.carrier callable).Occurrence)
            match output.InlineOps with
            | [MLIROp.MemRefOp(MemRefOp.GetGlobal(_, name, _)); MLIROp.FuncOp _] ->
                Assert.Equal(Alex.Patterns.MemoryPatterns.staticValueName instance.Allocation.Value, name)
            | operations -> failwithf "Expected descriptor access and function value: %A" operations
            Assert.Empty output.TopLevelOps
            Assert.Empty(MLIRAccumulator.drainPendingStaticGlobals accumulator)
            Assert.Empty ctx.TraversalVisited.Value

[<Theory>]
[<InlineData("startup")>]
[<InlineData("destination")>]
[<InlineData("foreign-occurrence")>]
[<InlineData("foreign-graph")>]
let ``program callable witness retracts with its exact source authority`` defect =
    let original = programCallables ()
    let binding = original.Nodes.Values |> Seq.filter (fun node ->
        node.IsReachable && match node.Kind with SemanticKind.Binding("first", false, _, _) -> true | _ -> false) |> Assert.Single
    let reference = original.Nodes.Values |> Seq.find (fun node ->
        node.IsReachable && match node.Kind with SemanticKind.VarRef(_, Some id) -> id = binding.Id | _ -> false)
    let graph =
        match defect with
        | "startup" -> { original with Edges = original.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.ProgramValue) }
        | "destination" -> { original with Edges = original.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.EnvironmentResultCall) }
        | _ -> original
    let accumulator = MLIRAccumulator.empty ()
    let current = callableContext graph reference.Id accumulator
    let ctx = if defect = "foreign-graph" then { current with Graph = { graph with Edges = graph.Edges |> List.rev } } else current
    let position = if defect = "foreign-occurrence" then Zipper.create graph binding.Id |> Option.get else ctx.Zipper
    match matchAt (Alex.Patterns.CallablePatterns.pProgramCallableReference ctx binding.Id) position 64 accumulator with
    | Result.Error _ -> ()
    | result -> failwithf "Expected stale or foreign source authority rejection: %A" result
    Assert.Empty accumulator.AllOps
    Assert.Empty(MLIRAccumulator.drainPendingStaticGlobals accumulator)

[<Fact>]
let ``writable declaration correspondence retains exact source identity and type`` () =
    let graph, binding = fixture ()
    let accumulator = MLIRAccumulator.empty ()
    let focus = Zipper.create graph binding |> require "Missing slot fixture"
    let ty = TInt(IntWidth 1)
    let ops =
        match matchAt (pGlobalSlotInit binding "owned_slot" (Arg 0) ty) focus 64 accumulator with
        | Ok((ops, _), _) -> ops
        | result -> failwithf "Expected source-owned slot: %A" result
    let declaration = MLIRAccumulator.drainPendingStaticGlobals accumulator |> Assert.Single
    let arch = (coeffects graph 64).Platform.TargetArch
    let validate operations = Alex.Traversal.StaticStorageValidation.validateWritable arch graph operations
    let entry =
        match declaration with
        | MLIROp.GlobalMemref(_, _, Some entry) -> entry
        | other -> failwithf "Expected exact storage authority: %A" other
    Assert.Equal(ProgramStorageIdentity.BindingSlot binding, entry.Identity)
    match validate (declaration :: ops) with
    | Ok [name, held] -> Assert.Equal("owned_slot", name); Assert.Equal(entry, held)
    | other -> failwithf "Expected complete inventory correspondence: %A" other
    let variants = [
        ops
        declaration :: declaration :: ops
        MLIROp.GlobalMemref("owned_slot", TMemRefStatic(1, ty), None) :: ops
        MLIROp.GlobalMemref("owned_slot", TMemRefStatic(2, ty), Some entry) :: ops
        MLIROp.GlobalMemref("owned_slot", TMemRefStatic(1, ty), Some { entry with Bytes = entry.Bytes + 1 }) :: ops
        [declaration]
    ]
    for changed in variants do
        match validate changed with
        | Result.Error _ -> ()
        | Ok _ -> failwith "A missing, duplicate, stale or differently typed writable declaration was admitted"
    let staleEdges = graph.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.ProgramValue)
    let stale = { graph with Edges = staleEdges }
    match Alex.Traversal.StaticStorageValidation.validateWritable arch stale (declaration :: ops) with
    | Result.Error _ -> ()
    | Ok _ -> failwith "Removed startup/storage authority retained native permission"

[<Fact>]
let ``writable declaration deduplication rejects conflicting identity and type`` () =
    let graph, binding = fixture ()
    let entry = graph.Codata.Value.ProgramStorage.Entries[ProgramStorageIdentity.BindingSlot binding]
    let accumulator = MLIRAccumulator.empty ()
    let ty = TMemRefStatic(1, TInt(IntWidth 1))
    MLIRAccumulator.tryEmitGlobalMemref "one" ty (Some entry) accumulator
    MLIRAccumulator.tryEmitGlobalMemref "one" ty (Some entry) accumulator
    Assert.Single(MLIRAccumulator.drainPendingStaticGlobals accumulator) |> ignore
    Assert.ThrowsAny<System.Exception>(fun () -> MLIRAccumulator.tryEmitGlobalMemref "one" (TMemRefStatic(2, TInt(IntWidth 1))) (Some entry) accumulator) |> ignore
    Assert.ThrowsAny<System.Exception>(fun () -> MLIRAccumulator.tryEmitGlobalMemref "one" ty None accumulator) |> ignore
