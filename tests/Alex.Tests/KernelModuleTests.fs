module Alex.Tests.KernelModuleTests

open Xunit
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Tests.Fixtures
module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission

let private aieOpt (text:string) =
    let selected = System.Environment.GetEnvironmentVariable "AIE_OPT_PATH"
    let executable =
        if System.String.IsNullOrWhiteSpace selected then
            System.IO.Path.Combine(System.Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile),"aie-toolchain","bin","aie-opt")
        else selected
    let start=System.Diagnostics.ProcessStartInfo(executable,UseShellExecute=false,RedirectStandardInput=true,RedirectStandardOutput=true,RedirectStandardError=true)
    start.ArgumentList.Add "--verify-each"
    use child=new System.Diagnostics.Process(StartInfo=start)
    if not(child.Start()) then failwith "Cannot start native aie-opt."
    let stdout,stderr=child.StandardOutput.ReadToEndAsync(),child.StandardError.ReadToEndAsync()
    child.StandardInput.Write text
    child.StandardInput.Close()
    if not(child.WaitForExit 20000) then child.Kill(true); failwith "Native AIE verification timed out."
    Assert.True(child.ExitCode=0,sprintf "aie-opt rejected the complete target realization:\n%s\n%s" (stderr.GetAwaiter().GetResult()) text)
    stdout.GetAwaiter().GetResult()

let private check expression alterAuthority =
    let authority = """namespace Fidelity.Platform.Contracts
type KernelTarget = { Device: string; Columns: int; ShimRow: int; ComputeRow: int; FifoDepth: int; Iterations: int; Inputs: string array; Result: string }
type WidthDeclaration = { Name: string; Bits: int }
type Representation = { Name: string; Capability: string; Family: string; Bits: int; MinMagnitude: string; MaxMagnitude: string; Boundary: string }
type TargetCore = { Arch: string; Runtime: string; Widths: WidthDeclaration array; Representations: Representation array }
type PlatformDescription = { Id: string; Core: TargetCore option }
module KernelAuthority =
    let target: KernelTarget = { Device="npu2"; Columns=8; ShimRow=0; ComputeRow=2; FifoDepth=2; Iterations=1; Inputs=[|"int16";"int16"|]; Result="int32" }
    let description: PlatformDescription = {
        Id="kernel-witness"
        Core=Some {
            Arch="xdna2"; Runtime="xdna"
            Widths=[| {Name="Pointer";Bits=32}; {Name="Register";Bits=16} |]
            Representations=[|
                {Name="int16";Capability="native";Family="int";Bits=16;MinMagnitude="-32768";MaxMagnitude="32767";Boundary="wrap"}
                {Name="uint16";Capability="native";Family="uint";Bits=16;MinMagnitude="0";MaxMagnitude="65535";Boundary="wrap"}
                {Name="int32";Capability="native";Family="int";Bits=32;MinMagnitude="-2147483648";MaxMagnitude="2147483647";Boundary="wrap"}
                {Name="uint32";Capability="native";Family="uint";Bits=32;MinMagnitude="0";MaxMagnitude="4294967295";Boundary="wrap"} |] } }
"""
    let representation name family bits minimum maximum : NumericRepresentation =
        {Name=name;Family=family;Bits=bits;Capability="native";MinMagnitude=minimum;MaxMagnitude=maximum;Boundary="wrap"}
    let offers = [representation "int16" "int" 16 "-32768" "32767"; representation "uint16" "uint" 16 "0" "65535"
                  representation "int32" "int" 32 "-2147483648" "2147483647"; representation "uint32" "uint" 32 "0" "4294967295"]
    let context : PlatformContext =
        { PlatformId="kernel-witness"; Dimensions=Map.ofList["Pointer",32;"Register",16]
          Representations=offers |> List.map (fun offer -> offer.Name,offer) |> Map.ofList; EndpointReturns=Map.empty
          PlatformLibraryPath=None; PlatformDescription=Some "Fidelity.Platform.Contracts.KernelAuthority.description"
          PlatformArchitecture=Some "xdna2"; PlatformOS=None
          PlatformSourcePaths=Set.singleton(System.IO.Path.GetFullPath "kernel-authority.clef")
          Predicates=Map.empty;FreestandingStartup=None;SubstrateKind=Some SubstrateKind.NPU;RuntimeModel=Some RuntimeModel.XDNA
          AvailableMemorySpaces=[];DefaultMemorySpace=None;ClockFrequencyMhz=None;NsPerWeightUnit=None }
    let sourceTemplate = """module KernelComputation
type Shape = { Elements: int; Grain: int }
type ElementKernel = { Compute: int -> int -> int; Shape: Shape }
let calculate (left: int) (right: int) = EXPRESSION
[<KernelModule>]
let kernel: ElementKernel = { Compute=calculate; Shape={Elements=64;Grain=16} }
"""
    let source = sourceTemplate.Replace("EXPRESSION",expression)
    let parse (text:string) path =
        match Clef.Compiler.NativeService.parseStringWithDefaults text path with
        | Clef.Compiler.NativeService.ParseSuccess input -> input
        | Clef.Compiler.NativeService.ParseError errors -> failwithf "Kernel source fixture parse failed: %A" errors
    Clef.Compiler.NativeService.checkParsedInputsWithPlatform [parse (alterAuthority authority) "kernel-authority.clef";parse source "kernel-computation.clef"] (Some context)

let private fixture expression =
    let checkedProgram=check expression id
    let errors=checkedProgram.Diagnostics |> List.filter (fun diagnostic -> diagnostic.Severity=Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
    Assert.True(errors.IsEmpty,sprintf "Kernel source was not admitted: %A" errors)
    let graph=checkedProgram.Graph
    let spatial=Publication.trySpatial graph |> Result.defaultWith failwith
    graph,Assert.Single spatial.Kernels.Values

[<Theory>]
[<InlineData("left * right", "muli", 1)>]
[<InlineData("right - left", "subi", 1)>]
[<InlineData("let product = left * right in product + left", "addi", 2)>]
let ``complete source kernel expression survives portable token and backend scalar realization`` expression opcode operationCount =
    let graph,plan=fixture expression
    let operations=plan.Steps |> List.choose (function KernelScalarStep.Operation operation -> Some operation | _ -> None)
    Assert.Equal(operationCount,operations.Length)
    Assert.Equal<int list>([16;16],plan.Ingress.Inputs |> List.map _.Representation.Bits)
    Assert.Equal(32,plan.Ingress.Output.Representation.Bits)
    Assert.Equal<int list>([0;16;32;48],plan.Tiles |> List.map _.Offset)
    let scalar=BackEnd.AIE.KernelRealization.scalarFunction plan |> Result.defaultWith failwith
    let verified=MlirComponentTests.mlirOpt ["--verify-each"] ("module {\n"+scalar+"\n}")
    Assert.Contains("arith."+opcode,verified)
    if expression="right - left" then
        let operation=Assert.Single operations
        let actualNames=operation.Operands |> List.map (fun operand ->
            match graph.Nodes[operand.Actual].Kind with SemanticKind.VarRef(name,_) -> name | _ -> failwith "Expected exact source formal reference.")
        Assert.Equal<string list>(["right";"left"],actualNames)
    let proof=Core.ProofDispatch.dischargeSource graph None |> Result.defaultWith failwith
    let witnessed,_=MiddleEnd.MLIRGeneration.generateWithLinkedLibrariesAndProof graph graph.Platform.Value
                        Core.Types.Dialects.DeploymentMode.Library Core.Types.Dialects.NPU None Set.empty (Some proof) |> Result.defaultWith failwith
    let tokens=witnessed.Operations |> List.choose (function MLIROp.SpatialModule(SpatialModuleWitness.Kernel declaration) -> Some declaration | _ -> None)
    Assert.Equal<KernelModuleWitness list>([plan],tokens)
    Assert.DoesNotContain("aie.device",witnessed.Text)
    Assert.DoesNotContain(witnessed.Operations,fun operation -> match operation with MLIROp.RawMLIR _ -> true | _ -> false)
    MlirComponentTests.mlirOpt ["--verify-each"] witnessed.Text |> ignore
    let realized,_=BackEnd.AIE.KernelRealization.realize unrealizedBackendContext witnessed |> Result.defaultWith failwith
    Assert.Contains("aie.device(npu2)",realized.Text)
    Assert.Contains("arith."+opcode,realized.Text)
    Assert.Contains("memref<16xi16>",realized.Text)
    Assert.Contains("memref<16xi32>",realized.Text)
    Assert.Contains("(%host_0: memref<64xi16>, %host_1: memref<64xi16>, %host_2: memref<64xi32>)",realized.Text)
    let target=aieOpt realized.Text
    Assert.Contains("arith."+opcode,target)

[<Theory>]
[<InlineData("shape")>]
[<InlineData("ordered-operand")>]
[<InlineData("missing-proof")>]
[<InlineData("missing-ingress")>]
[<InlineData("transport-declaration")>]
[<InlineData("source-authority")>]
[<InlineData("held-transport")>]
let ``spatial source publication rejects changed shape compute order or proof`` defect =
    let graph,plan=fixture "right - left"
    let changed=
        match defect with
        | "shape" ->
            let node=graph.Nodes[plan.GrainSite]
            {graph with Nodes=graph.Nodes.Add(node.Id,{node with Kind=SemanticKind.Literal(NativeLiteral.Int(8L,NTUKind.NTUint(NTUWidth.Fixed 16)))})}
        | "ordered-operand" ->
            let operation=plan.Steps |> List.choose (function KernelScalarStep.Operation operation -> Some operation | _ -> None) |> Assert.Single
            let node=graph.Nodes[operation.Site]
            let kind=match node.Kind with SemanticKind.Application(callee,args) -> SemanticKind.Application(callee,List.rev args) | _ -> failwith "Expected source operation."
            {graph with Nodes=graph.Nodes.Add(node.Id,{node with Kind=kind})}
        | "missing-ingress" ->
            {graph with Edges=graph.Edges |> List.filter (fun edge -> match edge.Role with EdgeRole.KernelIngress _ -> false | _ -> true)}
        | "transport-declaration" ->
            let declaration=graph.Nodes[plan.Ingress.Inputs.Head.Declaration]
            let fields=match declaration.Kind with SemanticKind.RecordExpr(fields,_) -> fields | _ -> failwith "Expected exact transport representation."
            let bits=fields |> List.find (fst >> (=) "Bits") |> snd
            let node=graph.Nodes[bits]
            let kind=match node.Kind with SemanticKind.Literal(NativeLiteral.Int(_,kind)) -> SemanticKind.Literal(NativeLiteral.Int(8L,kind)) | _ -> failwith "Expected exact declared transport bits."
            {graph with Nodes=graph.Nodes.Add(bits,{node with Kind=kind})}
        | "source-authority" ->
            let declaration=graph.Nodes[plan.Ingress.Target]
            {graph with Nodes=graph.Nodes.Add(declaration.Id,{declaration with Range={declaration.Range with File="unselected-authority.clef"}})}
        | "held-transport" ->
            let first=plan.Ingress.Inputs.Head
            let ingress={plan.Ingress with Inputs={first with Range=ValueRange.point 0I}::plan.Ingress.Inputs.Tail}
            let changedPlan={plan with Ingress=ingress}
            let edges=
                graph.Edges |> List.map (fun edge ->
                    match edge.Role with
                    | EdgeRole.KernelIngress held when held.Site=plan.Site -> {edge with Role=EdgeRole.KernelIngress ingress}
                    | EdgeRole.KernelModule held when held.Site=plan.Site -> {edge with Role=EdgeRole.KernelModule changedPlan}
                    | EdgeRole.SpatialModuleDomain domain -> {edge with Role=EdgeRole.SpatialModuleDomain {domain with Kernels=domain.Kernels |> List.map (fun held -> if held.Site=plan.Site then changedPlan else held)}}
                    | _ -> edge)
            {graph with Edges=edges}
        | _ -> {graph with Nodes=graph.Nodes.Remove plan.Obligations.Head}
    match Clef.Compiler.PSGSaturation.SemanticGraph.SpatialPublication.project changed with
    | Result.Error errors -> Assert.NotEmpty errors
    | Result.Ok _ -> failwith "Changed spatial premises retained publication."

[<Fact>]
let ``kernel output transport cannot replace the complete mathematical result range`` () =
    let checkedProgram=check "left * right" (fun authority -> authority.Replace("Result=\"int32\"", "Result=\"int16\""))
    let graph=checkedProgram.Graph
    let mathematicalResults=graph.Nodes.Values |> Seq.filter (fun node ->
        match node.Kind with SemanticKind.Application _ when node.Range.File.EndsWith("kernel-computation.clef") -> true | _ -> false) |> Seq.toList
    Assert.Contains(mathematicalResults, fun node -> node.ValueRange |> Option.exists (fun range -> not(ValueRange.contains (ValueRange.twosComplement 16) range)))
    match Clef.Compiler.PSGSaturation.SemanticGraph.SpatialPublication.project graph with
    | Result.Error errors -> Assert.Contains(errors,fun error -> error.Reason.Contains("output representation and range coverage"))
    | Result.Ok _ -> failwith "Narrow output transport silently replaced or truncated the mathematical result."

[<Fact>]
let ``new runtime compute use retracts external-only ingress without replacing actual ranges`` () =
    let graph,plan=fixture "left * right"
    let reference=plan.Ingress.ComputePath |> Seq.find (fun id ->
        match graph.Nodes[id].Kind with SemanticKind.VarRef(_,Some binding) -> binding=plan.ComputeBinding | _ -> false)
    let original=graph.Nodes[plan.Result]
    let site=NodeId.fresh()
    let arguments=[plan.ElementsSite;plan.GrainSite]
    let call={original with Id=site;Kind=SemanticKind.Application(reference,arguments);Children=reference::arguments;Parent=Some plan.Scope;IsReachable=true}
    let changed={graph with Nodes=graph.Nodes.Add(site,call)}
    match Clef.Compiler.Baker.Recipes.KernelDeclarations.read changed plan.Site with
    | Result.Error _ -> ()
    | Result.Ok _ -> failwith "A new ordinary invocation retained a declaration-only external input contract."
