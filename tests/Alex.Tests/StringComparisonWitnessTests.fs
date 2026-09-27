module Alex.Tests.StringComparisonWitnessTests

open Xunit
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission
module Catalog = Core.WitnessArtifacts

let private source (category:string) =
    if category.StartsWith("dynamic-empty-") then
        let operator = if category.EndsWith("-ne") then "<>" else "="
        let comparison = if category.Contains("-left-") then $"\"\" {operator} expected" else $"expected {operator} \"\""
        $"module EmptySelectedEquality\nlet compareSelected select =\n    let expected = if select then \"\" else \"a\\000b\"\n    {comparison}\n[<EntryPoint>]\nlet main _ = if compareSelected true then 0 else 1\n"
    elif category="selected" then
        "module SelectedEquality\nlet compareSelected select =\n    let expected = if select then \"a\\000b\" else \"a\\000c\"\n    expected = \"a\\000b\"\n[<EntryPoint>]\nlet main _ = if compareSelected true then 0 else 1\n"
    else
        let operator,left,right =
            match category with
            | "empty" -> "=","",""
            | "empty-ne" -> "<>","",""
            | "empty-left-eq" -> "=","","a"
            | "empty-left-ne" -> "<>","","a"
            | "empty-right-eq" -> "=","a",""
            | "empty-right-ne" -> "<>","a",""
            | "unequal-length" -> "<>","a","longer"
            | "embedded-zero-equal" -> "=","a\\000b","a\\000b"
            | "embedded-zero-different" -> "=","a\\000b","a\\000c"
            | _ -> "=","same","same"
        $"module TextEquality\nlet compareText first second =\n    let alias = first\n    alias {operator} second\n[<EntryPoint>]\nlet main _ = if compareText \"{left}\" \"{right}\" then 0 else 1\n"

[<Theory>]
[<InlineData("formal-alias")>]
[<InlineData("empty")>]
[<InlineData("empty-ne")>]
[<InlineData("empty-left-eq")>]
[<InlineData("empty-left-ne")>]
[<InlineData("empty-right-eq")>]
[<InlineData("empty-right-ne")>]
[<InlineData("dynamic-empty-left-eq")>]
[<InlineData("dynamic-empty-left-ne")>]
[<InlineData("dynamic-empty-right-eq")>]
[<InlineData("dynamic-empty-right-ne")>]
[<InlineData("unequal-length")>]
[<InlineData("embedded-zero-equal")>]
[<InlineData("embedded-zero-different")>]
[<InlineData("selected")>]
let ``actual registry witnesses source string comparison lengths guards and byte traversal`` category =
    let checkedSource = MemoryWitnessTests.checkMemoryProgram (source category) ("string-comparison-"+category+".clef")
    let errors = checkedSource.Diagnostics |> List.filter (fun diagnostic ->
        Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.Diagnostic.effectiveSeverity diagnostic =
            Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
    Assert.Empty errors
    let graph = checkedSource.Graph
    let boundary = Publication.tryBoundary graph |> Result.defaultWith failwith
    let memory = Publication.tryMemory graph |> Result.defaultWith failwith
    let lengthOnly = category.StartsWith("empty") || category.StartsWith("dynamic-empty-")
    Assert.Equal((if lengthOnly then 0 else 2),boundary.ByteViews.Count)
    Assert.Empty boundary.Imports
    Assert.Empty boundary.IntrinsicWrites
    if category="selected" then Assert.Contains(boundary.ByteViews.Values,fun view -> view.StaticOrigins.IsEmpty)
    let reads = memory.Operations.Values |> Seq.choose (function MemoryWitnessOperation.ArrayAccess read -> Some read | _ -> None) |> Seq.toList
    Assert.Equal((if lengthOnly then 0 else 2),reads.Length)
    let proof = Core.ProofDispatch.dischargeSource graph None |> Result.defaultWith failwith
    let witnessed,_ = MiddleEnd.MLIRGeneration.generateWithLinkedLibrariesAndProof graph graph.Platform.Value
                          Core.Types.Dialects.Console Core.Types.Dialects.CPU None Set.empty (Some proof) |> Result.defaultWith failwith
    Core.Types.Pipeline.WitnessedInput.validate witnessed |> Result.defaultWith failwith
    MlirComponentTests.mlirOpt ["--verify-each"] witnessed.Text |> ignore
    let operations = witnessed.Operations |> List.collect (Catalog.flatten >> Seq.toList)
    if lengthOnly then
        Assert.DoesNotContain(operations,fun operation -> match operation with MLIROp.SCFOp(SCFOp.While _) -> true | _ -> false)
    else Assert.Contains(operations,fun operation -> match operation with MLIROp.SCFOp(SCFOp.While _) -> true | _ -> false)
    Assert.DoesNotContain("memcmp",witnessed.Text)
    for read in reads do
        Assert.Contains(operations,fun operation -> match operation with MLIROp.Assert(_,diagnostic) -> diagnostic=read.Bounds.Requirement.Diagnostic | _ -> false)
        let value = Alex.Traversal.Values.value read.Site
        Assert.Contains(operations,fun operation -> match operation with MLIROp.MemRefOp(MemRefOp.Load(result,_,_,_,_)) -> result=value 1 | _ -> false)

[<Theory>]
[<InlineData("property")>]
[<InlineData("intrinsic")>]
[<InlineData("alias")>]
[<InlineData("curried")>]
[<InlineData("first-class")>]
let ``actual registry witnesses source string extents through callable forms`` form =
    let declarations, invocation =
        match form with
        | "property" -> "let measure (text:string) = text.Length\n", "measure text"
        | "alias" -> "let measure (text:string) =\n    let extent = String.length\n    extent text\n", "measure text"
        | "curried" -> "let measure ignored (text:string) = String.length text\n", "let length = measure 7\n    length text"
        | "first-class" -> "let apply measure text = measure text\n", "apply String.length text"
        | _ -> "let measure (text:string) = String.length text\n", "measure text"
    let program = "module OrdinaryExtent\nlet select choose = if choose then \"a\\000b\" else \"longer\"\n"+declarations+"[<EntryPoint>]\nlet main _ =\n    let text = select true\n    "+invocation+"\n"
    let checkedSource = MemoryWitnessTests.checkMemoryProgram program ("string-extent-"+form+".clef")
    let errors = checkedSource.Diagnostics |> List.filter (fun diagnostic ->
        Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.Diagnostic.effectiveSeverity diagnostic =
            Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
    Assert.Empty errors
    let graph = checkedSource.Graph
    let memory = Publication.tryMemory graph |> Result.defaultWith failwith
    let extents = memory.Operations.Values |> Seq.choose (function MemoryWitnessOperation.BufferExtent extent -> Some extent | _ -> None) |> Seq.toList
    Assert.NotEmpty extents
    Assert.Contains(extents,fun extent -> extent.Extent.StaticOrigins.IsEmpty)
    let proof = Core.ProofDispatch.dischargeSource graph None |> Result.defaultWith failwith
    let witnessed,_ = MiddleEnd.MLIRGeneration.generateWithLinkedLibrariesAndProof graph graph.Platform.Value
                          Core.Types.Dialects.Console Core.Types.Dialects.CPU None Set.empty (Some proof) |> Result.defaultWith failwith
    Core.Types.Pipeline.WitnessedInput.validate witnessed |> Result.defaultWith failwith
    MlirComponentTests.mlirOpt ["--verify-each"] witnessed.Text |> ignore
    Assert.Contains("memref.dim", witnessed.Text)
