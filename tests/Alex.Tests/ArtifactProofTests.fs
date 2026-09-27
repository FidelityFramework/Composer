module Alex.Tests.ArtifactProofTests

open System
open System.IO
open System.Text
open System.Security.Cryptography
open System.Text.Json
open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Core.Types.Pipeline
open BackEnd.LLVM.ArtifactProof

let private good = function Ok value -> value | Error reason -> failwith reason
let private temporary name =
    let directory=Path.Combine(Path.GetTempPath(),"composer-artifact-"+name+"-"+Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory directory |> ignore
    directory

let private context output =
    {Timing=Core.Timing.silent();OutputPath=output;IntermediatesDir=None
     TargetTripleOverride=Some "x86_64-unknown-linux-gnu";TargetPointerBits=Some 64;TargetCpu=None
     PlatformOS=Some "linux";RuntimeModel=Some RuntimeModel.Libc;DeploymentMode=Core.Types.Dialects.Console
     EmitIntermediateOnly=false;ExternLibraries=Set.empty;NativeLink=NativeLinkOptions.Empty
     EmbeddedTarget=None;XtensaTarget=None;Deploy=false}

let private witness graph =
    let source=Core.ProofDispatch.dischargeSource graph None |> good
    let input,_=MiddleEnd.MLIRGeneration.generateWithLinkedLibrariesAndProof graph graph.Platform.Value
                    Core.Types.Dialects.Console Core.Types.Dialects.CPU None Set.empty (Some source) |> good
    let catalog=input.Catalog.Value
    let proof=catalog.Proof.Value
    let mlir=Core.ProofDispatch.dischargeMlir proof None |> good
    {input with Catalog=Some {catalog with Proof=Some {proof with Mlir=Some mlir}}}

let private sample=lazy (
    let path=Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__,"../../samples/console/FidelityHelloWorld/01_HelloWorldDirect/HelloWorld.fidproj"))
    let checkedProject=Clef.Compiler.Project.ProjectChecker.checkProject path |> good
    let graph=checkedProject.CheckResult.Graph
    graph,witness graph)

let private checkProgram body =
    let source="module ArtifactFixture\n[<EntryPoint>]\nlet main _ =\n"+body
    let checkedSource=MemoryWitnessTests.checkMemoryProgram source "artifact-source.clef"
    let errors=checkedSource.Diagnostics |> List.filter (fun diagnostic ->
        Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.Diagnostic.effectiveSeverity diagnostic =
            Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
    Assert.Empty errors
    checkedSource.Graph,witness checkedSource.Graph

let private scalar=lazy(checkProgram "    7\n")

// Every artifact used below is produced by this current compiler invocation.
let private linked=lazy (
    let graph,input=sample.Value
    let directory=temporary "fresh"
    let output=Path.Combine(directory,"hello")
    let ctx=context output
    let result=BackEnd.LLVM.Pipeline.backend.Compile input ctx
    let artifact=
        match result with
        | Ok artifact -> artifact
        | Error reason ->
            let publication=Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryRead graph |> good
            let operands=publication.Boundary.IntrinsicWrites.Values |> Seq.map (fun call ->
                let view=publication.Boundary.ByteViews[call.Buffer]
                let target=publication.Callable.AliasTargets[view.Source]
                let owners=publication.Callable.Arguments |> Map.toList |> List.choose (fun (owner,arguments) -> if arguments.ContainsKey target then Some(owner,arguments[target]) else None)
                call.Site,view.Source,target,owners) |> Seq.toList
            let definitions=input.Catalog.Value.Units |> List.collect _.Definitions |> List.choose (fun row ->
                match row.Operation with MLIROp.FuncOp(FuncDef(name,_,_,_,_)) -> Some(name,row.Occurrence.Focus.Id) | _ -> None)
            failwithf "%s\nPublished write operands: %A\nDefinition occurrences: %A" reason operands definitions
    match artifact with
    | NativeBinary path -> Assert.Equal(output,path)
    | other -> failwithf "Expected checked native artifact: %A" other
    let runtime=BackEnd.LLVM.IntrinsicWriteRealization.realize ctx input |> good
    let runtime=BackEnd.LLVM.RequirementRealization.realize (BackEnd.LLVM.RequirementRealization.selectRuntime ctx ctx.TargetTripleOverride.Value) runtime |> good
    let evidence=Directory.GetDirectories(directory,"hello.proof-*") |> Assert.Single
    graph,input,output,runtime.Text,evidence)

[<Fact>]
let ``native publication requires fresh source MLIR and actual ELF Rocq evidence`` () =
    let graph,input,binary,_,directory=linked.Value
    use program=new System.Diagnostics.Process()
    program.StartInfo<-System.Diagnostics.ProcessStartInfo(binary,UseShellExecute=false,RedirectStandardOutput=true,RedirectStandardError=true)
    Assert.True(program.Start(),"The freshly verified F01 artifact did not start")
    let output,errors=program.StandardOutput.ReadToEndAsync(),program.StandardError.ReadToEndAsync()
    let exited=program.WaitForExit 5000
    if not exited then
        program.Kill true
        program.WaitForExit 5000 |> ignore
    Assert.True(exited,"The freshly verified F01 artifact did not terminate")
    Assert.True(System.Threading.Tasks.Task.WhenAll([|output :> System.Threading.Tasks.Task;errors :> System.Threading.Tasks.Task|]).Wait 5000,
                "The freshly verified F01 artifact did not close its output")
    Assert.Equal(0,program.ExitCode)
    Assert.Equal("Hello, World!\n",output.Result)
    Assert.Equal("",errors.Result)
    Assert.True(File.Exists(Path.Combine(directory,"MemoryMap.vo")))
    use evidence=JsonDocument.Parse(File.ReadAllText(Path.Combine(directory,"artifact-receipt.json")))
    let root=evidence.RootElement
    Assert.Equal("elf-rocq",root.GetProperty("Stage").GetString())
    Assert.Equal(input.Catalog.Value.Proof.Value.Source.Value.Evidence.Invocation,root.GetProperty("Invocation").GetGuid())
    let checkEvidence (expected:Core.Types.WitnessArtifacts.ProofStageEvidence) (actual:JsonElement) =
        Assert.Equal(expected.Invocation,actual.GetProperty("Invocation").GetGuid())
        Assert.Equal(expected.Stage,actual.GetProperty("Stage").GetString())
        Assert.Equal(expected.InputSha256,actual.GetProperty("InputSha256").GetString())
        Assert.Equal(expected.QuerySha256,actual.GetProperty("QuerySha256").GetString())
        Assert.Equal(expected.StandardOutput,actual.GetProperty("StandardOutput").GetString())
        Assert.Equal(expected.StandardError,actual.GetProperty("StandardError").GetString())
        let tools=actual.GetProperty("Tools").EnumerateArray() |> Seq.toList
        Assert.NotEmpty tools
        Assert.Equal(expected.Tools.Length,tools.Length)
        for tool,row in List.zip expected.Tools tools do
            Assert.Equal(tool.Executable,row.GetProperty("Executable").GetString())
            Assert.Equal(tool.Sha256,row.GetProperty("Sha256").GetString())
            Assert.Equal(tool.Version,row.GetProperty("Version").GetString())
            Assert.True(tool.Arguments=(row.GetProperty("Arguments").EnumerateArray() |> Seq.map _.GetString() |> Seq.toList))
        let outcomes=actual.GetProperty("Outcomes").EnumerateArray() |> Seq.toList
        Assert.NotEmpty outcomes
        Assert.Equal(expected.Outcomes.Length,outcomes.Length)
        for outcome,row in List.zip expected.Outcomes outcomes do
            Assert.Equal(outcome.Anchor,row.GetProperty("Anchor").GetString())
            Assert.Equal(outcome.Source,row.GetProperty("Source").GetString())
            Assert.Equal("unsat",row.GetProperty("Verdict").GetString())
    checkEvidence input.Catalog.Value.Proof.Value.Source.Value.Evidence (root.GetProperty("SourceEvidence"))
    checkEvidence input.Catalog.Value.Proof.Value.Mlir.Value.Evidence (root.GetProperty("MlirEvidence"))
    Assert.Equal(SHA256.HashData(File.ReadAllBytes binary) |> Convert.ToHexString,root.GetProperty("ArtifactSha256").GetString())
    Assert.True(root.GetProperty("Claims").GetArrayLength()>0)
    let boundary=Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryBoundary graph |> good
    let expected=boundary.IntrinsicWrites.Values |> Seq.sumBy (fun call -> boundary.ByteViews[call.Buffer].StaticOrigins.Count)
    Assert.Equal(expected,root.GetProperty("Origins").GetArrayLength())
    Assert.True(root.GetProperty("ForeignExtents").GetArrayLength()>0)
    let proof=File.ReadAllText(Path.Combine(directory,"MemoryMap.v"))
    Assert.Contains("_bytes_accounted",proof)
    Assert.Contains("_tiled",proof)
    Assert.Contains("trusted_foreign",proof)
    Assert.Empty(Directory.GetFiles(Path.GetDirectoryName binary,"*.provisional"))

[<Theory>]
[<InlineData("source")>]
[<InlineData("mlir")>]
[<InlineData("catalog")>]
let ``artifact admission refuses missing prior stage receipts`` missing =
    let _,original=sample.Value
    let catalog=original.Catalog.Value
    let proof=catalog.Proof.Value
    let input=
        match missing with
        | "source" -> {original with Catalog=Some {catalog with Proof=Some {proof with Source=None}}}
        | "mlir" -> {original with Catalog=Some {catalog with Proof=Some {proof with Mlir=None}}}
        | _ -> {original with Catalog=None}
    match prepare input with
    | Error reason -> Assert.True(reason.Contains("receipt") || reason.Contains("PSG") || reason.Contains("catalog") || reason.Contains("discharge"),reason)
    | Ok _ -> failwith "An absent checked stage acquired artifact authority"

[<Fact>]
let ``equal source query from a different graph object cannot authorize the artifact`` () =
    let graph,input=sample.Value
    let catalog=input.Catalog.Value
    let proof=catalog.Proof.Value
    let other=Core.ProofDispatch.dischargeSource {graph with Nodes=graph.Nodes} None |> good
    Assert.Equal(proof.Source.Value.Query,other.Query)
    let changed={input with Catalog=Some {catalog with Proof=Some {proof with Source=Some other}}}
    match prepare changed with
    | Error reason -> Assert.Contains("different graph snapshot",reason)
    | Ok _ -> failwith "Equal query text authorized a different graph object"

[<Theory>]
[<InlineData("argument")>]
[<InlineData("dimension")>]
[<InlineData("conversion")>]
let ``write operand dimension and conversion tampering fails production artifact admission`` defect =
    let graph,original=sample.Value
    let catalog=original.Catalog.Value
    let boundary=Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryBoundary graph |> good
    let counts=boundary.IntrinsicWrites.Values |> Seq.map _.Count |> Set.ofSeq
    let imports=catalog.Units.Head.Imports |> List.choose _.IntrinsicWrite |> List.map _.Symbol |> Set.ofList
    let mutable changed=false
    let rec change operation =
        match operation with
        | MLIROp.FuncOp(FuncDef(symbol,args,results,body,visibility)) -> MLIROp.FuncOp(FuncDef(symbol,args,results,List.map change body,visibility))
        | MLIROp.FuncOp(FuncCall(results,symbol,[fd;buffer;count])) when defect="argument" && imports.Contains symbol ->
            Assert.Equal(fd.Type,count.Type)
            changed<-true
            MLIROp.FuncOp(FuncCall(results,symbol,[fd;buffer;fd]))
        | MLIROp.IndexOp(IndexOp.IndexConst((V(id,0) as result),0L)) when defect="dimension" && counts.Contains(NodeId id) ->
            changed<-true
            MLIROp.IndexOp(IndexOp.IndexConst(result,1L))
        | MLIROp.IndexOp(IndexOp.IndexCastU((V(id,_) as result),_,TIndex,ty))
        | MLIROp.IndexOp(IndexOp.IndexCastS((V(id,_) as result),_,TIndex,ty)) when defect="conversion" && counts.Contains(NodeId id) ->
            changed<-true
            MLIROp.ArithOp(ArithOp.ConstI(result,1L,ty))
        | _ -> operation
    let operations=List.map change original.Operations
    Assert.True changed
    let definitions=catalog.Units.Head.Definitions |> List.map (fun row -> {row with Operation=change row.Operation})
    let text=Alex.Dialects.Core.Serialize.moduleToString original.PointerBits original.ModuleName.Value operations
    let altered=Core.WitnessArtifacts.createWithProof catalog.Proof.Value catalog.Scope catalog.Activation definitions operations text original.WritableStorage |> good
    let input={original with Operations=operations;Text=text;Catalog=Some altered}
    match prepare input with
    | Error reason -> Assert.Contains((if defect="argument" then "ordered source operands" else "dimension"),reason)
    | Ok _ -> failwith "A same-width count replacement retained artifact authority"

[<Theory>]
[<InlineData("byte")>]
[<InlineData("sentinel")>]
[<InlineData("symbol")>]
[<InlineData("load-write")>]
let ``fresh linked artifact tampering cannot retain checked source storage claims`` defect =
    let graph,input,binary,runtime,_=linked.Value
    let directory=temporary defect
    try
        let output=Path.Combine(directory,"altered")
        File.Copy(binary,output)
        let pool=graph.StaticStringPool.Value
        let observed=observe output (Some pool.Symbol) |> good
        let object=observed.Pool.Value
        let bytes=File.ReadAllBytes output
        match defect with
        | "byte" -> bytes[object.Offset]<-bytes[object.Offset] ^^^ 1uy
        | "sentinel" -> bytes[object.Offset+pool.Entries.Head.Offset+pool.Entries.Head.Length]<-1uy
        | "symbol" ->
            let expected=Encoding.UTF8.GetBytes(pool.Symbol+"\000")
            let position=[0..bytes.Length-expected.Length] |> List.find (fun at -> bytes.AsSpan(at,expected.Length).SequenceEqual(expected.AsSpan()))
            bytes[position]<-byte 'X'
        | _ ->
            let programAt=int(BitConverter.ToUInt64(bytes,32))
            let width,count=int(BitConverter.ToUInt16(bytes,54)),int(BitConverter.ToUInt16(bytes,56))
            let at=[0..count-1] |> List.map (fun i -> programAt+width*i) |> List.find (fun at ->
                BitConverter.ToUInt32(bytes,at)=1u && bigint(BitConverter.ToUInt64(bytes,at+16))<=object.Address &&
                object.Address<bigint(BitConverter.ToUInt64(bytes,at+16))+bigint(BitConverter.ToUInt64(bytes,at+40)))
            bytes[at+4]<-bytes[at+4] ||| 2uy
        File.WriteAllBytes(output,bytes)
        match verifyWith "rocq" (Path.Combine(directory,"proof")) runtime output (prepare input |> good) with
        | Error reason -> Assert.Contains((if defect="symbol" then "pool symbol" else "Rocq"),reason)
        | Ok _ -> failwith "Altered ELF retained the original source claim"
    finally Directory.Delete(directory,true)

[<Fact>]
let ``missing Rocq refuses publication and preserves an existing output`` () =
    let _,input,binary,runtime,_=linked.Value
    let directory=temporary "missing-checker"
    try
        let output=Path.Combine(directory,"program")
        File.WriteAllText(output,"previous owner output")
        let mutable linkedPath=""
        let produce provisional=linkedPath<-provisional;File.Copy(binary,provisional);Ok ()
        match publishWith (Path.Combine(directory,"absent-rocq")) input runtime (context output) produce with
        | Error reason -> Assert.Contains("proof",reason)
        | Ok _ -> failwith "A missing checker published a native binary"
        Assert.NotEqual<string>("",linkedPath)
        Assert.Equal(directory,Path.GetDirectoryName linkedPath)
        Assert.False(File.Exists linkedPath)
        Assert.Equal("previous owner output",File.ReadAllText output)
    finally Directory.Delete(directory,true)

[<Theory>]
[<InlineData("allocation")>]
[<InlineData("load")>]
[<InlineData("store")>]
let ``empty source memory inventory does not authorize residual typed memory instructions`` defect =
    let graph,original=scalar.Value
    Assert.True graph.StaticStringPool.IsNone
    prepare original |> good |> ignore
    let catalog=original.Catalog.Value
    let row=catalog.Units.Head.Definitions |> List.find (fun row ->
        match row.Operation with MLIROp.FuncOp(FuncDef _) -> true | _ -> false)
    let buffer=Alex.Traversal.Values.value row.Occurrence.Focus.Id 41
    let result=Alex.Traversal.Values.value row.Occurrence.Focus.Id 42
    let index=Alex.Traversal.Values.value row.Occurrence.Focus.Id 43
    let element=TInt(IntWidth 8)
    let memory=TMemRefStatic(1,element)
    let operation=
        match defect with
        | "allocation" -> MemRefOp.Alloca(buffer,memory,None)
        | "load" -> MemRefOp.Load(result,buffer,[index],element,memory)
        | _ -> MemRefOp.Store(result,buffer,[index],element,memory)
    let changed=
        match row.Operation with
        | MLIROp.FuncOp(FuncDef(name,args,results,body,visibility)) ->
            MLIROp.FuncOp(FuncDef(name,args,results,MLIROp.MemRefOp operation::body,visibility))
        | _ -> failwith "Fixture has no witnessed function definition"
    let replace operation=if operation=row.Operation then changed else operation
    let operations=List.map replace original.Operations
    let definitions=catalog.Units.Head.Definitions |> List.map (fun row -> {row with Operation=replace row.Operation})
    let text=Alex.Dialects.Core.Serialize.moduleToString original.PointerBits original.ModuleName.Value operations
    let altered=Core.WitnessArtifacts.createWithProof catalog.Proof.Value catalog.Scope catalog.Activation definitions operations text original.WritableStorage |> good
    let input={original with Operations=operations;Text=text;Catalog=Some altered}
    WitnessedInput.validate input |> good
    match prepare input with
    | Error reason -> Assert.Contains("Unsupported required artifact typed memory operation",reason)
    | Ok _ -> failwith "Residual memory acquired authority from an empty source claim inventory"

[<Fact>]
let ``no pool still checks actual ELF sections while required array storage refuses explicitly`` () =
    let graph,input=scalar.Value
    Assert.True graph.StaticStringPool.IsNone
    let directory=temporary "no-pool"
    try
        BackEnd.LLVM.Pipeline.backend.Compile input (context(Path.Combine(directory,"scalar"))) |> good |> ignore
        let evidence=Directory.GetDirectories(directory,"scalar.proof-*") |> Assert.Single
        use receipt=JsonDocument.Parse(File.ReadAllText(Path.Combine(evidence,"artifact-receipt.json")))
        Assert.Equal(0,receipt.RootElement.GetProperty("Claims").GetArrayLength())
        Assert.True(receipt.RootElement.GetProperty("ForeignExtents").GetArrayLength()>0)
        Assert.True(File.Exists(Path.Combine(evidence,"MemoryMap.vo")))
    finally Directory.Delete(directory,true)
    let _,array=checkProgram "    let values = [| 7; 11 |]\n    values.[1]\n"
    match prepare array with
    | Error reason -> Assert.Contains("Unsupported required artifact memory operation",reason)
    | Ok _ -> failwith "Array storage disappeared into an empty string-pool proof"
