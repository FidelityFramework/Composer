module Alex.Tests.Fixtures

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.NodeBuilder
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
module Zipper = Alex.Traversal.PSGZipper

let require label = Option.defaultWith (fun () -> failwith label)

/// A backend component with no target realization authority.
let unrealizedBackendContext : Core.Types.Pipeline.BackEndContext =
    { Timing = Core.Timing.silent(); OutputPath = "unused"; IntermediatesDir = None
      TargetTripleOverride = None; TargetPointerBits = None; TargetCpu = None
      PlatformOS = None; RuntimeModel = None
      DeploymentMode = Core.Types.Dialects.Console; EmitIntermediateOnly = false
      ExternLibraries = Set.empty; NativeLink = Core.Types.Pipeline.NativeLinkOptions.Empty
      EmbeddedTarget = None; XtensaTarget = None; Deploy = false }

/// Source edits invalidate the publication before any new zipper is created.
/// All semantic premises remain present for source revalidation or refusal.
let unpublished (graph: SemanticGraph) =
    Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.invalidate graph

/// A component fixture publishes through the same source owner as compilation.
/// Callers must use the returned graph when constructing their zipper/context.
let prepareSource (graph: SemanticGraph) =
    let settled =
        graph
        |> unpublished
        |> Clef.Compiler.Nanopass.NumericSettlement.normalize
        |> Clef.Compiler.Nanopass.MemorySettlement.normalize
        |> Clef.Compiler.Nanopass.SpatialSettlement.normalize
        |> Clef.Compiler.Nanopass.BoundarySettlement.normalize
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.prepare settled with
    | Result.Ok prepared -> prepared
    | Result.Error failures ->
        failures |> List.map (fun failure -> sprintf "%A: %s" failure.Occurrence failure.Reason)
        |> String.concat "\n" |> failwith

/// Check a complete program with explicit source-declared scalar platform
/// authority, then ask CCS to publish every witness domain. No host defaults
/// or fixture-created empty projection stand in for source settlement.
let checkScalarProgram source path =
    let representation: NumericRepresentation =
        { Name="int32"; Capability="native"; Family="int"; Bits=32
          MinMagnitude="-2147483648"; MaxMagnitude="2147483647"; Boundary="wrap" }
    let unsignedRepresentation: NumericRepresentation =
        { Name="uint32"; Capability="native"; Family="uint"; Bits=32
          MinMagnitude="0"; MaxMagnitude="4294967295"; Boundary="wrap" }
    let narrowRepresentation: NumericRepresentation =
        { Name="int8"; Capability="native"; Family="int"; Bits=8
          MinMagnitude="-128"; MaxMagnitude="127"; Boundary="wrap" }
    let authority = """module PublishedAuthority
type WidthDeclaration = { Name: string; Bits: int }
type Representation = { Name: string; Capability: string; Family: string; Bits: int; MinMagnitude: string; MaxMagnitude: string; Boundary: string }
type TargetCore = { Runtime: string; Widths: WidthDeclaration array; Representations: Representation array }
type PlatformDescription = { Id: string; Core: TargetCore option }
let description = {
    Id = "published-boundary-witness"
    Core = Some {
        Runtime = "libc"
        Widths = [| { Name="Pointer"; Bits=64 }; { Name="Register"; Bits=32 } |]
        Representations = [|
            { Name="int8"; Capability="native"; Family="int"; Bits=8; MinMagnitude="-128"; MaxMagnitude="127"; Boundary="wrap" }
            { Name="int32"; Capability="native"; Family="int"; Bits=32; MinMagnitude="-2147483648"; MaxMagnitude="2147483647"; Boundary="wrap" }
            { Name="uint32"; Capability="native"; Family="uint"; Bits=32; MinMagnitude="0"; MaxMagnitude="4294967295"; Boundary="wrap" } |] } }
"""
    let platform: PlatformContext =
        { PlatformId="published-boundary-witness"; Dimensions=Map.ofList ["Pointer",64; "Register",32]
          Representations=Map.ofList [representation.Name, representation; unsignedRepresentation.Name, unsignedRepresentation; narrowRepresentation.Name, narrowRepresentation]; EndpointReturns=Map.empty
          PlatformLibraryPath=None; PlatformDescription=Some "PublishedAuthority.description"; PlatformArchitecture=None; PlatformOS=None
          PlatformSourcePaths=Set.singleton (System.IO.Path.GetFullPath "published-authority.clef"); Predicates=Map.empty; FreestandingStartup=None; SubstrateKind=None
          RuntimeModel=Some RuntimeModel.Libc; AvailableMemorySpaces=[]; DefaultMemorySpace=None
          ClockFrequencyMhz=None; NsPerWeightUnit=None }
    let parse source path =
        match Clef.Compiler.NativeService.parseStringWithDefaults source path with
        | Clef.Compiler.NativeService.ParseSuccess input -> input
        | Clef.Compiler.NativeService.ParseError errors -> failwithf "Scalar program did not parse: %A" errors
    let result = Clef.Compiler.NativeService.checkParsedInputsWithPlatform [parse authority "published-authority.clef"; parse source path] (Some platform)
    let errors = result.Diagnostics |> List.filter (fun diagnostic ->
        diagnostic.Severity = Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
    if not errors.IsEmpty then failwithf "Source did not admit scalar program: %A" errors
    prepareSource result.Graph

/// Complete the demand owner's source step after a deliberate fixture edit,
/// then publish every domain. This never repairs other owners' proof relations.
let settleDemand (graph: SemanticGraph) =
    graph |> unpublished |> Clef.Compiler.Nanopass.OrdinaryDemand.normalize |> prepareSource

/// Retraction tests must establish the actual source refusal before exercising
/// passive consumers with the unpublished graph. No empty authority is supplied.
let refusePublication (reasonFragment: string) (graph: SemanticGraph) =
    let graph = graph |> unpublished |> Clef.Compiler.Nanopass.OrdinaryDemand.normalize
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.prepare graph with
    | Result.Ok _ -> failwith "An invalid component fixture was accepted by source publication"
    | Result.Error failures ->
        if not (failures |> List.exists (fun failure -> failure.Reason.Contains(reasonFragment, System.StringComparison.OrdinalIgnoreCase))) then
            failwithf "Expected source refusal containing '%s', got %A" reasonFragment failures
        graph

let coeffects (graph: SemanticGraph) pointerBits : TransferCoeffects =
    { Platform =
        { TargetArch = { Register = Ok pointerBits; Pointer = Ok pointerBits }
          LinkedLibraries = Set.empty }
      TargetPlatform = Core.Types.Dialects.CPU }

let atChild childId (position: Zipper.PSGZipper) =
    let index = position.Focus.Children |> List.findIndex ((=) childId)
    Zipper.down index position |> require "Fixture child is missing"

/// Component-boundary fixture, not a source program or a substitute for Baker.
/// The index already has its analysed range and an i8 operand representation.
/// Neither this fixture nor the tested pattern establishes array bounds.
type ArrayRead =
    { Graph: SemanticGraph
      Binding: NodeId
      Lambda: NodeId
      Array: NodeId
      Index: NodeId
      Call: NodeId
      Proof: NodeId
      Position: Zipper.PSGZipper }

let arrayRead unsigned =
    let builder = NodeBuilder()
    let arrayType = Types.mkArrayType Types.boolType
    let array = builder.Create(SemanticKind.PatternBinding "values", arrayType, dummyRange)
    let index = builder.Create(SemanticKind.PatternBinding "index", Types.intType, dummyRange)
    let intrinsic = builder.Create(
        SemanticKind.Intrinsic
            { Module = IntrinsicModule.Array; Operation = "get"
              Category = IntrinsicCategory.Memory; FullName = "Array.get" },
        NativeType.TFun(arrayType, NativeType.TFun(Types.intType, Types.boolType)), dummyRange)
    let call = builder.Create(SemanticKind.Application(intrinsic.Id, [array.Id; index.Id]), Types.boolType, dummyRange)
    let lambda = builder.Create(
        SemanticKind.Lambda(["values", arrayType, array.Id; "index", Types.intType, index.Id],
                            call.Id, [], None, LambdaContext.RegularClosure),
        intrinsic.Type, dummyRange, children = [array.Id; index.Id; call.Id])
    let binding = builder.Create(SemanticKind.Binding("read", false, false, None), lambda.Type,
                                 dummyRange, children = [lambda.Id])
    for child in [array; index; call] do builder.SetParent(child.Id, lambda.Id)
    builder.SetParent(lambda.Id, binding.Id)
    let lower, upper = if unsigned then 128I, 255I else -128I, 127I
    let minimum, maximum = if unsigned then 0I, 255I else -128I, 127I
    let proof = builder.Create(
        SemanticKind.Obligation
            { Id = "index_carrier"; Kind = "integer-representation-coverage"; Logic = "QF_LIA"
              Statement = "The supplied index range fits its supplied operand carrier"
              Source = "Alex component fixture"; Refs = []
              Body = ObligationBody.IntegerRepresentationCoverage(lower, upper, minimum, maximum) },
        Types.unitType, dummyRange)
    let raw = builder.Build []
    let graph =
        { raw with
            Nodes = raw.Nodes.Add(index.Id, { raw.Nodes[index.Id] with ValueRange = Some (ValueRange.Bounded(lower, upper)) })
            Edges =
                [{ Sources = [index.Id]; Target = proof.Id; Class = EdgeClass.Obligation
                   Role = EdgeRole.Constrains; Ordinal = 0 }] }
        |> prepareSource
    let position =
        Zipper.create graph binding.Id |> require "Missing fixture binding"
        |> atChild lambda.Id |> atChild call.Id
    { Graph = graph; Binding = binding.Id; Lambda = lambda.Id; Array = array.Id
      Index = index.Id; Call = call.Id; Proof = proof.Id; Position = position }

/// The current public parser API requires a table of already witnessed operands.
/// Seed those two inputs only; tests do not run or reproduce the mutable traversal.
let recalledOperands fixture =
    let operands = MLIRAccumulator.empty ()
    MLIRAccumulator.bindNode fixture.Array (Arg 0) (TMemRef(TInt(IntWidth 1))) operands
    MLIRAccumulator.bindNode fixture.Index (Arg 1) (TInt(IntWidth 8)) operands
    operands

let matchAt parser (position: Zipper.PSGZipper) pointerBits operands =
    tryMatchWithDiagnostics parser position.Graph position.Focus position
        (coeffects position.Graph pointerBits) operands

let readArray fixture pointerBits operands =
    match matchAt Alex.Patterns.MemoryPatterns.pIndexGetArray fixture.Position pointerBits operands with
    | Result.Ok ((operations, result), position) -> operations, result, position
    | Result.Error message -> failwith message

let readModule fixture pointerBits =
    let operations, result, _ = readArray fixture pointerBits (recalledOperands fixture)
    match result with
    | TRValue value ->
        let body = operations @ [MLIROp.FuncOp(FuncOp.Return([{ SSA = value.SSA; Type = value.Type }]))]
        let definition = MLIROp.FuncOp(FuncOp.FuncDef(
            "read_index", [Arg 0, TMemRef(TInt(IntWidth 1)); Arg 1, TInt(IntWidth 8)], [value.Type], body, FuncVisibility.Public))
        Alex.Dialects.Core.Serialize.moduleToString (Ok pointerBits) "index_component" [definition]
    | other -> failwithf "Array.get did not produce a value: %A" other
