module Alex.Tests.StringBoundaryTests

open Xunit
open System.Diagnostics
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission

// Byte conversion preserves independent snapshots. Carrier equality alone is
// insufficient: these fixtures enter through the actual source compiler.
let private program direction =
    let body =
        if direction="fromBytes" then
            "    let bytes = [| 65; 0; 66 |]\n    let text = eager (String.fromBytes bytes)\n    bytes.[0] <- 67\n    if text = \"A\\000B\" then 0 else 1\n"
        else
            "    let text = \"A\\000B\"\n    let first = eager (String.toBytes text)\n    let second = eager (String.toBytes text)\n    first.[0] <- 512\n    if second.[0] = 65 && first.[0] = 512 then 0 else 1\n"
    "module StringSnapshots\n[<EntryPoint>]\nlet main _ =\n" + body

let private checkedGraph direction =
    let result = MemoryWitnessTests.checkMemoryProgram (program direction) ("string-snapshot-"+direction+".clef")
    let errors = result.Diagnostics |> List.filter (fun diagnostic ->
        Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.Diagnostic.effectiveSeverity diagnostic =
            Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
    Assert.True(errors.IsEmpty,sprintf "Source snapshot contract failed: %A" errors)
    result.Graph

let private isSnapshot direction role =
    if direction="fromBytes" then role=EdgeRole.StringByteSnapshot else role=EdgeRole.StringToBytesSnapshot

// This executes witnessed behavior using stock MLIR. The typed no-argument
// harness is test-only; it grants no native artifact or Rocq admission.
let private executeSnapshot (witnessed:Core.Types.Pipeline.BackEndInput) =
    let entry=witnessed.Operations |> List.choose (function
        | MLIROp.FuncOp(FuncOp.FuncDef("StringSnapshots.main",[],[result],_,_)) -> Some result
        | _ -> None) |> Assert.Single
    let raw,result=V(0,0),V(0,1)
    let i64=TInt(IntWidth 64)
    let body=
        match entry with
        | TInt(IntWidth width) when width>0 && width<64 ->
            [MLIROp.FuncOp(FuncOp.FuncCall([{SSA=raw;Type=entry}],"StringSnapshots.main",[]))
             MLIROp.ArithOp(ArithOp.ExtUI(result,raw,entry,i64))
             MLIROp.FuncOp(FuncOp.Return [{SSA=result;Type=i64}])]
        | TInt(IntWidth 64) ->
            [MLIROp.FuncOp(FuncOp.FuncCall([{SSA=raw;Type=i64}],"StringSnapshots.main",[]))
             MLIROp.FuncOp(FuncOp.Return [{SSA=raw;Type=i64}])]
        | _ -> failwithf "Snapshot behavior fixture has no admitted integer result: %A" entry
    let harness=MLIROp.FuncOp(FuncOp.FuncDef("__snapshot_test_entry",[],[i64],body,FuncVisibility.Public))
    let text=Alex.Dialects.Core.Serialize.moduleToString witnessed.PointerBits "snapshot_behavior" (witnessed.Operations@[harness])
    let width=witnessed.PointerBits |> Result.defaultWith failwith
    let atWidth pass=sprintf "%s{index-bitwidth=%d}" pass width
    let passes=["expand-strided-metadata";atWidth "finalize-memref-to-llvm";"convert-scf-to-cf";"convert-cf-to-llvm"
                atWidth "convert-index-to-llvm";atWidth "convert-func-to-llvm";atWidth "convert-arith-to-llvm";"reconcile-unrealized-casts"]
    let lowered=MlirComponentTests.mlirOpt ["--verify-each";"--pass-pipeline=builtin.module("+String.concat "," passes+")"] text
    let start=ProcessStartInfo("mlir-runner",UseShellExecute=false,RedirectStandardInput=true,RedirectStandardOutput=true,RedirectStandardError=true)
    for argument in ["-e";"__snapshot_test_entry";"--entry-point-result=i64";"-"] do start.ArgumentList.Add argument
    use child=new Process(StartInfo=start)
    Assert.True(child.Start(),"Cannot start the required stock MLIR behavior runner")
    let output,errors=child.StandardOutput.ReadToEndAsync(),child.StandardError.ReadToEndAsync()
    child.StandardInput.Write lowered
    child.StandardInput.Close()
    let exited=child.WaitForExit 20000
    if not exited then
        child.Kill true
        child.WaitForExit 5000 |> ignore
    Assert.True(exited,"Snapshot behavior did not terminate")
    Assert.True(System.Threading.Tasks.Task.WhenAll([|output :> System.Threading.Tasks.Task;errors :> System.Threading.Tasks.Task|]).Wait 5000,
                "Snapshot behavior runner did not close its output")
    Assert.True(child.ExitCode=0,sprintf "Snapshot runner failed: %s" errors.Result)
    Assert.Equal("0",output.Result.Trim())

[<Theory>]
[<InlineData("fromBytes")>]
[<InlineData("toBytes")>]
let ``actual string conversions retain independent snapshot storage before subsequent mutation`` direction =
    let graph = checkedGraph direction
    let edges = graph.Edges |> List.filter (fun edge -> isSnapshot direction edge.Role)
    Assert.Equal((if direction="fromBytes" then 1 else 2),edges.Length)
    for edge in edges do
        match edge.Sources with
        | [input;snapshot] when direction="fromBytes" -> Assert.NotEqual(input,snapshot)
        | [input;view;snapshot] ->
            Assert.NotEqual(input,snapshot)
            Assert.NotEqual(view,snapshot)
        | _ -> failwith "Snapshot lost its exact input/view/copy source identities."
    let memory = Publication.tryMemory graph |> Result.defaultWith failwith
    let writes = memory.Operations.Values |> Seq.choose (function
        | MemoryWitnessOperation.ArrayAccess { Value=Some value } as operation -> Some(value,operation)
        | _ -> None) |> Seq.toList
    Assert.NotEmpty writes
    let proof = Core.ProofDispatch.dischargeSource graph None |> Result.defaultWith failwith
    let witnessed,_ = MiddleEnd.MLIRGeneration.generateWithLinkedLibrariesAndProof graph graph.Platform.Value
                          Core.Types.Dialects.DeploymentMode.Console Core.Types.Dialects.CPU None Set.empty (Some proof) |> Result.defaultWith failwith
    Core.Types.Pipeline.WitnessedInput.validate witnessed |> Result.defaultWith failwith
    MlirComponentTests.mlirOpt ["--verify-each"] witnessed.Text |> ignore
    Assert.DoesNotContain("@memcpy",witnessed.Text)
    let operations = witnessed.Operations |> List.collect (Core.WitnessArtifacts.flatten >> Seq.toList)
    Assert.Contains(operations,fun operation -> match operation with MLIROp.MemRefOp(MemRefOp.Store _) -> true | _ -> false)
    executeSnapshot witnessed

[<Theory>]
[<InlineData("fromBytes")>]
[<InlineData("toBytes")>]
let ``removing snapshot evidence retracts the source contract`` direction =
    let graph = checkedGraph direction
    let changed = {graph with Edges=graph.Edges |> List.filter (fun edge -> not(isSnapshot direction edge.Role))}
    match Publication.prepare changed with
    | Error failures -> Assert.NotEmpty failures
    | Ok _ -> failwith "A byte conversion retained authority after losing its snapshot relation."

[<Theory>]
[<InlineData("view")>]
[<InlineData("copy")>]
let ``constructed string comparison retracts when admitted snapshot authority disappears`` missing =
    let graph=checkedGraph "fromBytes"
    let domain=graph.Edges |> List.choose(fun edge ->
        match edge.Role with EdgeRole.BoundaryDomain domain -> Some domain | _ -> None) |> Assert.Single
    let snapshot=Assert.Single domain.StringComparisonSnapshots
    let copy=Assert.Single domain.StringComparisonCopies
    Assert.Equal(snapshot.Snapshot,copy.Site)
    let retained edge =
        match missing,edge.Role with
        | "view",EdgeRole.MemoryOperation(MemoryWitnessOperation.StringView fact) -> fact.Site<>snapshot.Site
        | "copy",EdgeRole.MemoryArrayCopy fact -> fact.Site<>copy.Site
        | _ -> true
    let changed={graph with Edges=List.filter retained graph.Edges}
    Assert.Equal(graph.Edges.Length-1,changed.Edges.Length)
    match Clef.Compiler.PSGSaturation.SemanticGraph.BoundaryEmission.project changed with
    | Error failures ->
        Assert.Contains(failures,fun failure -> failure.Reason.Contains("local string comparison"))
    | Ok _ -> failwith "A constructed comparison survived removal of its admitted snapshot authority."
    Publication.tryRead graph |> Result.defaultWith failwith |> ignore

[<Theory>]
[<InlineData("255")>]
[<InlineData("169")>]
[<InlineData("512")>]
let ``byte carrier alone never admits invalid text`` value =
    let source = "module InvalidText\n[<EntryPoint>]\nlet main _ =\n    let text = eager (String.fromBytes [| "+value+" |])\n    String.length text\n"
    let result = MemoryWitnessTests.checkMemoryProgram source "invalid-string-snapshot.clef"
    let reason = if value="512" then "0..255" else "UTF-8"
    Assert.Contains(result.Diagnostics,fun diagnostic ->
        diagnostic.Code="CCS8404" && diagnostic.Message.Contains("String.fromBytes") &&
        diagnostic.Message.Contains(reason) && not diagnostic.RelatedNodes.IsEmpty &&
        Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.Diagnostic.effectiveSeverity diagnostic =
            Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
