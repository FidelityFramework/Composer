module Alex.Tests.ProgramSequencePatternTests

open Xunit
open Clef.Compiler.NativeService
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics
module SequencePrograms = Clef.Compiler.Nanopass.SequenceProgramInstances
module SequenceProgramAuthority = Clef.Compiler.PSGSaturation.SemanticGraph.ProgramStorageAuthority
module SequenceProgramCopies = Clef.Compiler.Nanopass.SequenceFamilies

module SequenceProgramFixture =
    let file = System.IO.Path.GetFullPath "sequence-program.clef"
    let context: PlatformContext =
        let integer name family bits minimum maximum: NumericRepresentation =
            { Name = name; Family = family; Bits = bits; MinMagnitude = minimum; MaxMagnitude = maximum
              Capability = "native"; Boundary = "wrap" }
        let representations = [integer "signed64" "int" 64 "-9223372036854775808" "9223372036854775807"; integer "uint8" "uint" 8 "0" "255"]
        { PlatformId = "sequence-program"; Dimensions = Map.ofList ["Pointer", 64; "Register", 64]
          Representations = representations |> List.map (fun value -> value.Name, value) |> Map.ofList
          EndpointReturns = Map.empty; PlatformLibraryPath = None; PlatformDescription = Some "SequenceProgram.description"
          PlatformArchitecture = None; PlatformOS = None; PlatformSourcePaths = Set.singleton file
          Predicates = Map.empty; FreestandingStartup = None; SubstrateKind = None; RuntimeModel = None
          AvailableMemorySpaces = []; DefaultMemorySpace = None; ClockFrequencyMhz = None; NsPerWeightUnit = None }

    let header = """module SequenceProgram
type WidthDeclaration = { Name: string; Bits: int }
type Representation = { Name: string; Capability: string; Family: string; Bits: int; MinMagnitude: string; MaxMagnitude: string; Boundary: string }
type TargetCore = { Widths: WidthDeclaration array; Representations: Representation array }
type MemorySpace = { Name: string; Kind: string; Capacity: int; Alignment: int; Granularity: int; Growth: string; Access: string; Base: int option }
type ProgramLifetimeSpaces = { Immutable: string; Mutable: string option }
type PlatformDescription = { Id: string; Core: TargetCore option; Spaces: MemorySpace array; ProgramLifetime: ProgramLifetimeSpaces option }
let image = { Name = "image"; Kind = "rodata"; Capacity = 4096; Alignment = 16; Granularity = 16; Growth = "fixed"; Access = "r"; Base = None }
let state = { Name = "state"; Kind = "data"; Capacity = 4096; Alignment = 16; Granularity = 16; Growth = "fixed"; Access = "rw"; Base = None }
let core = { Widths = [| { Name = "Pointer"; Bits = 64 }; { Name = "Register"; Bits = 64 } |]; Representations = [| { Name = "signed64"; Capability = "native"; Family = "int"; Bits = 64; MinMagnitude = "-9223372036854775808"; MaxMagnitude = "9223372036854775807"; Boundary = "wrap" }; { Name = "uint8"; Capability = "native"; Family = "uint"; Bits = 8; MinMagnitude = "0"; MaxMagnitude = "255"; Boundary = "wrap" } |] }
let description = { Id = "sequence-program"; Core = Some core; Spaces = [| image; state |]; ProgramLifetime = Some { Immutable = "image"; Mutable = Some "state" } }
[<Measure>] type m
"""
    let check body =
        let source = header + body
        let input = match parseStringWithDefaults source file with ParseSuccess input -> input | ParseError errors -> failwithf "%A" errors
        let result = checkParsedInputsWithPlatform [input] (Some context)
        Assert.Empty(result.Diagnostics |> List.filter (fun diagnostic -> Diagnostic.effectiveSeverity diagnostic = NativeDiagnosticSeverity.Error))
        result.Graph

    let direct () = check """
let first = seq { yield 1<m>; yield 2<m> }
let second = seq { yield 3<m> }
let alias = first
[<EntryPoint>]
let main _ =
    for value in first do ignore value
    for value in alias do ignore value
    for value in second do ignore value
    0
"""

    let binding name (graph: SemanticGraph) =
        graph.Nodes.Values |> Seq.filter (fun node ->
            node.IsReachable && match node.Kind with SemanticKind.Binding(actual, false, _, _) -> actual = name | _ -> false)
        |> Assert.Single

    let instance graph name =
        SequencePrograms.programInstance graph (binding name graph).Id
        |> Option.defaultWith (fun () -> failwithf "Missing actual program sequence %s" name)

open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Tests.Fixtures
module Zipper = Alex.Traversal.PSGZipper
module SequenceValues = Alex.Traversal.SequenceOperands

let private context graph occurrence accumulator =
    let scope = ref (Alex.Traversal.ScopeContext.ScopeContext.root ())
    let visited = ref Set.empty
    { Coeffects = coeffects graph 64; Accumulator = accumulator; RootAccumulator = accumulator
      ScopeContext = scope; RootScopeContext = scope; Graph = graph
      Zipper = Zipper.create graph occurrence |> require "Missing program sequence occurrence"
      GlobalVisited = visited; TraversalVisited = visited }

[<Fact>]
let ``program references read exact template code and allocation without replaying formation`` () =
    let graph = SequenceProgramFixture.direct ()
    for name in ["first"; "second"; "alias"] do
        let binding = SequenceProgramFixture.binding name graph
        let instance = SequenceProgramFixture.instance graph name
        let references = graph.Nodes.Values |> Seq.filter (fun node ->
            node.IsReachable && match node.Kind with SemanticKind.VarRef(_, Some source) -> source = binding.Id | _ -> false) |> Seq.toList
        Assert.NotEmpty references
        for reference in references do
            let accumulator = MLIRAccumulator.empty ()
            let ctx = context graph reference.Id accumulator
            let output = Alex.Witnesses.VarRefWitness.nanopass.Witness ctx reference
            let sequence = match output.Result with TRSequence sequence -> sequence | other -> failwithf "Missing sequence pair: %A %A" other output.InlineOps
            MLIRAccumulator.bindSequence reference.Id sequence accumulator |> Result.defaultWith failwith
            match output.InlineOps with
            | [MLIROp.MemRefOp(MemRefOp.GetGlobal(_, name, _)); MLIROp.FuncOp _] ->
                Assert.Equal(Alex.Patterns.MemoryPatterns.staticValueName instance.Allocation, name)
            | operations -> failwithf "Expected only descriptor access and code value: %A" operations
            Assert.Empty output.TopLevelOps
            Assert.Empty(MLIRAccumulator.drainPendingStaticGlobals accumulator)
            Assert.Empty ctx.TraversalVisited.Value

[<Fact>]
let ``program binding forwards the actual witnessed pair without allocation or replay`` () =
    let graph = SequenceProgramFixture.direct ()
    let binding = SequenceProgramFixture.binding "first" graph
    let source = binding.Children |> List.exactlyOne
    let accumulator = MLIRAccumulator.empty ()
    let sourceCtx = context graph source accumulator
    let shape = SequenceValues.project sourceCtx source |> Result.defaultWith failwith
    let code = { SSA = Arg 0; Type = SequenceValues.functionType shape }
    let environment = { SSA = Arg 1; Type = SequenceValues.environmentType shape }
    let value = SequenceValues.create shape code environment |> Result.defaultWith failwith
    MLIRAccumulator.bindSequence source value accumulator |> Result.defaultWith failwith
    let ctx = context graph binding.Id accumulator
    let output = Alex.Witnesses.BindingWitness.nanopass.Witness ctx binding
    match output.Result with
    | TRSequence actual ->
        Assert.Equal(code, SequenceValues.code actual)
        Assert.Equal(environment, SequenceValues.environment actual)
        MLIRAccumulator.bindSequence binding.Id actual accumulator |> Result.defaultWith failwith
    | other -> failwithf "Missing actual initialized pair: %A %A" other output.InlineOps
    Assert.Empty output.InlineOps
    Assert.Empty output.TopLevelOps
    Assert.Empty(MLIRAccumulator.drainPendingStaticGlobals accumulator)

[<Theory>]
[<InlineData("storage")>]
[<InlineData("startup")>]
[<InlineData("foreign-focus")>]
[<InlineData("foreign-graph")>]
let ``program sequence references refuse changed authority or occurrence without emitting operations`` defect =
    let original = SequenceProgramFixture.direct ()
    let binding = SequenceProgramFixture.binding "first" original
    let reference = original.Nodes.Values |> Seq.find (fun node ->
        node.IsReachable && match node.Kind with SemanticKind.VarRef(_, Some source) -> source = binding.Id | _ -> false)
    let graph =
        match defect with
        | "storage" -> { original with Edges = original.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.SequenceProgramStorage) }
        | "startup" -> { original with Edges = original.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.ProgramValue) }
        | _ -> original
    let accumulator = MLIRAccumulator.empty ()
    let originalCtx = context graph reference.Id accumulator
    let ctx = if defect = "foreign-graph" then { originalCtx with Graph = { graph with Edges = List.rev graph.Edges } } else originalCtx
    let position = if defect = "foreign-focus" then Zipper.create graph binding.Id |> Option.get else ctx.Zipper
    match matchAt (Alex.Patterns.SequencePatterns.pProgramSequenceReference ctx binding.Id) position 64 accumulator with
    | Result.Error _ -> ()
    | result -> failwithf "Expected source authority refusal: %A" result
    Assert.Empty accumulator.AllOps
    Assert.Empty(MLIRAccumulator.drainPendingStaticGlobals accumulator)
