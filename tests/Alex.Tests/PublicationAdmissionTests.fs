module Alex.Tests.PublicationAdmissionTests

open System
open System.IO
open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Tests.Fixtures

module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission
module Transfer = Alex.Traversal.MLIRTransfer

let private program () =
    let source = """module PublicationInputs
let choose left right = left
[<EntryPoint>]
let main _ = choose 11 7
    """
    checkScalarProgram source "publication-inputs.clef"

let private entry (graph: SemanticGraph) = graph.DeclarationRoots |> List.head |> fst

let private accepted graph =
    match Transfer.transferWithCorrespondence graph (entry graph) (coeffects graph 64) None with
    | Result.Ok (operations, _, _) -> operations
    | Result.Error reason -> failwith reason

let private refused original changed =
    // Supply the original coeffects: rejection must inspect the changed graph's
    // provenance before witnesses or output, not rely on fixture invalidation.
    let output = Path.Combine(Path.GetTempPath(), "clef-publication-admission-" + Guid.NewGuid().ToString("N"))
    match Transfer.transferWithCorrespondence changed (entry original) (coeffects original 64) (Some output) with
    | Result.Error reason -> Assert.Contains("Source emission admission", reason)
    | Result.Ok _ -> failwith "A changed raw graph retained production witness authority"
    Assert.False(Directory.Exists output)
    Assert.True(Publication.tryRead original |> Result.isOk)

[<Fact>]
let ``production witnessing permits an exact root copy and preserves its operations`` () =
    let original = program ()
    let copied = { original with Nodes = original.Nodes }
    Assert.False(obj.ReferenceEquals(original, copied))
    let expected = accepted original
    Assert.NotEmpty expected
    Assert.Equal<Alex.Dialects.Core.Types.MLIROp list>(expected, accepted copied)

[<Theory>]
[<InlineData("actual-order")>]
[<InlineData("module-member")>]
[<InlineData("new-source-use")>]
[<InlineData("omission-proof")>]
[<InlineData("declaration-roots")>]
[<InlineData("runtime")>]
let ``production transfer rejects changed raw source roots retaining published codata`` change =
    let original = program ()
    let changed =
        match change with
        | "actual-order" ->
            let site = original.Nodes.Values |> Seq.find (fun node ->
                match node.Kind with SemanticKind.Application(_, [_; _]) -> true | _ -> false)
            let changedKind =
                match site.Kind with
                | SemanticKind.Application(callee, actuals) -> SemanticKind.Application(callee, List.rev actuals)
                | _ -> failwith "Expected the two-argument application"
            { original with Nodes = original.Nodes.Add(site.Id, { site with Kind = changedKind }) }
        | "module-member" ->
            let owner = original.Nodes.Values |> Seq.find (fun node ->
                match node.Kind with SemanticKind.ModuleDef("PublicationInputs", _ :: _) -> true | _ -> false)
            let changedKind =
                match owner.Kind with
                | SemanticKind.ModuleDef(name, _ :: members) -> SemanticKind.ModuleDef(name, members)
                | _ -> failwith "Expected the owning module"
            { original with Nodes = original.Nodes.Add(owner.Id, { owner with Kind = changedKind }) }
        | "new-source-use" ->
            let actual = original.Nodes.Values |> Seq.find (fun node ->
                match node.Kind with SemanticKind.VarRef _ -> true | _ -> false)
            let added = { actual with Id = NodeId.fresh() }
            { original with Nodes = original.Nodes.Add(added.Id, added) }
        | "omission-proof" ->
            Assert.Contains(original.Edges, fun edge -> edge.Role = EdgeRole.OrdinaryUnusedActual)
            { original with Edges = original.Edges |> List.filter (fun edge -> edge.Role <> EdgeRole.OrdinaryUnusedActual) }
        | "declaration-roots" ->
            Assert.NotEmpty original.DeclarationRoots
            { original with DeclarationRoots = [] }
        | "runtime" ->
            let platform = original.Platform |> Option.get
            { original with Platform = Some { platform with RuntimeModel = Some RuntimeModel.Bare } }
        | _ -> failwithf "Unknown source change: %s" change
    Assert.Same(original.Codata, changed.Codata)
    refused original changed

[<Theory>]
[<InlineData("types")>]
[<InlineData("classifications")>]
[<InlineData("field-ranges")>]
[<InlineData("element-ranges")>]
[<InlineData("layouts")>]
[<InlineData("escaping")>]
[<InlineData("codata")>]
let ``production transfer refuses replacement deferred facts without evaluating them`` domain =
    let original = program ()
    let mutable forced = false
    let unavailable () = forced <- true; failwith "Witness admission evaluated a replacement source computation"
    let changed =
        match domain with
        | "types" -> { original with Types = lazy (unavailable ()) }
        | "classifications" -> { original with ModuleClassifications = lazy (unavailable ()) }
        | "field-ranges" -> { original with FieldRanges = lazy (unavailable ()) }
        | "element-ranges" -> { original with ElementRanges = lazy (unavailable ()) }
        | "layouts" -> { original with Layouts = lazy (unavailable ()) }
        | "escaping" -> { original with Escaping = lazy (unavailable ()) }
        | "codata" -> { original with Codata = lazy (unavailable ()) }
        | _ -> failwithf "Unknown source domain: %s" domain
    refused original changed
    Assert.False forced

[<Fact>]
let ``materializing replacement codata does not authorize a copied publication`` () =
    let original = program ()
    let copiedFacts = lazy original.Codata.Value
    copiedFacts.Force() |> ignore
    let changed = { original with Codata = copiedFacts }
    Assert.True changed.Codata.IsValueCreated
    Assert.True changed.Codata.Value.WitnessEmission.IsSome
    refused original changed

[<Fact>]
let ``public generation refuses stale input before platform readers codata or output`` () =
    let original = program ()
    let mutable forced = false
    let changed = { original with Codata = lazy (forced <- true; failwith "Premature codata read") }
    let output = Path.Combine(Path.GetTempPath(), "clef-publication-generation-" + Guid.NewGuid().ToString("N"))
    // The invalid platform argument is a sentinel: admission must precede the
    // platform reader as well as the deliberately deferred codata computation.
    match MiddleEnd.MLIRGeneration.generateWithLinkedLibraries changed Unchecked.defaultof<PlatformContext>
              Core.Types.Dialects.DeploymentMode.Console Core.Types.Dialects.TargetPlatform.CPU
              (Some output) Set.empty with
    | Result.Error reason -> Assert.Contains("Source emission admission", reason)
    | Result.Ok _ -> failwith "Public generation accepted stale publication"
    Assert.False forced
    Assert.False(Directory.Exists output)
    Assert.True(Publication.tryRead original |> Result.isOk)
