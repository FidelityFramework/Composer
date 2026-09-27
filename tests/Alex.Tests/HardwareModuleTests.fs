module Alex.Tests.HardwareModuleTests

open System.Diagnostics
open Xunit
open Clef.Compiler.NativeService
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission

let private authority = """module HardwareAuthority
type PlatformDescription = { Id: string }
type ClockEndpoint = { Name: string; FrequencyHz: int; PackagePin: string; Standard: string }
type ResetEndpoint = { Name: string; Kind: string; PackagePin: string; Standard: string; ActiveLevel: string }
type PinEndpoint = { LogicalName: string; PackagePin: string; Standard: string; Direction: string }
type PlatformDescriptor = { Device: string; Package: string; SpeedGrade: string }
let description = { Id = "hardware-test" }
let device = { Device = "xc7a100t"; Package = "csg324"; SpeedGrade = "-1" }
let clock = { Name = "clk"; FrequencyHz = 100000000; PackagePin = "E3"; Standard = "LVCMOS33" }
let reset = { Name = "por"; Kind = "Internal"; PackagePin = ""; Standard = ""; ActiveLevel = "High" }
let switch = { LogicalName = "switch"; PackagePin = "A8"; Standard = "LVCMOS33"; Direction = "Input" }
let led = { LogicalName = "led"; PackagePin = "H5"; Standard = "LVCMOS33"; Direction = "Output" }
"""

let private program = """module HardwareFixture
open HardwareAuthority
type State = { On: bool }
type Inputs = { [<Pin("switch")>] Switch: bool }
type Outputs = { [<Pin("led")>] Led: bool }
type Design = { InitialState: State; Step: State -> Inputs -> State * Outputs; Clock: ClockEndpoint }
let step (state: State) (inputs: Inputs) = ({ On = inputs.Switch }, { Led = state.On })
[<HardwareModule>]
let design = { InitialState = { On = false }; Step = step; Clock = clock }
"""

let private platform : PlatformContext =
    { PlatformId="hardware-test"; Dimensions=Map.empty; Representations=Map.empty; EndpointReturns=Map.empty
      PlatformLibraryPath=None; PlatformDescription=Some "HardwareAuthority.description"; PlatformArchitecture=None; PlatformOS=None
      PlatformSourcePaths=Set.singleton(System.IO.Path.GetFullPath "hardware-authority.clef")
      Predicates=Map.empty; FreestandingStartup=None; SubstrateKind=Some SubstrateKind.FPGA; RuntimeModel=None
      AvailableMemorySpaces=[]; DefaultMemorySpace=None; ClockFrequencyMhz=Some 100; NsPerWeightUnit=Some 1.6 }

let private check declaration =
    let parse text path =
        match parseStringWithDefaults text path with
        | ParseSuccess value -> value
        | ParseError errors -> failwithf "Hardware fixture parse failed: %A" errors
    checkParsedInputsWithPlatform [parse declaration "hardware-authority.clef"; parse program "hardware-program.clef"] (Some platform)

let private admitted = lazy (
    let checkedSource = check authority
    let errors = checkedSource.Diagnostics |> List.filter (fun diagnostic ->
        Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.Diagnostic.effectiveSeverity diagnostic =
            Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics.NativeDiagnosticSeverity.Error)
    Assert.True(errors.IsEmpty, sprintf "Hardware source was not admitted: %A" errors)
    let graph = checkedSource.Graph
    let spatial = Publication.trySpatial graph |> Result.defaultWith failwith
    graph, Assert.Single spatial.Hardware.Values)

let private circtOpt (text:string) =
    let start = ProcessStartInfo("circt-opt", UseShellExecute=false, RedirectStandardInput=true, RedirectStandardOutput=true, RedirectStandardError=true)
    start.ArgumentList.Add "--verify-each"
    use child = new Process(StartInfo=start)
    if not(child.Start()) then failwith "Cannot start circt-opt"
    let stdout,stderr = child.StandardOutput.ReadToEndAsync(),child.StandardError.ReadToEndAsync()
    child.StandardInput.Write text
    child.StandardInput.Close()
    if not(child.WaitForExit 20000) then child.Kill(true); failwith "CIRCT hardware verification timed out"
    Assert.True(child.ExitCode=0, sprintf "CIRCT rejected source hardware realization:\n%s\n%s" (stderr.GetAwaiter().GetResult()) text)
    stdout.GetAwaiter().GetResult()

[<Fact>]
let ``source hardware contract retains reset pins and exact Step declaration`` () =
    let graph,plan = admitted.Value
    Assert.Equal<string list>(["state";"inputs"],plan.Parameters |> List.map fst)
    Assert.Equal<string list>(["Switch"],(Assert.Single plan.InputPorts).Path)
    Assert.Equal<string list>(["Led"],(Assert.Single plan.OutputPorts).Path)
    Assert.Equal("switch",(Assert.Single plan.InputPorts).Name)
    Assert.Equal("led",(Assert.Single plan.OutputPorts).Name)
    Assert.Equal(2,plan.Obligations.Length)
    Assert.Contains(plan.ClockDeclaration,plan.Participants)
    Assert.Contains(plan.ResetDeclaration,plan.Participants)
    Assert.DoesNotContain(plan.StepBinding,plan.MetadataOnly)
    let proof = Core.ProofDispatch.dischargeSource graph None |> Result.defaultWith failwith
    let witnessed,_ = MiddleEnd.MLIRGeneration.generateWithLinkedLibrariesAndProof graph graph.Platform.Value
                          Core.Types.Dialects.DeploymentMode.Library Core.Types.Dialects.FPGA None Set.empty (Some proof) |> Result.defaultWith failwith
    Core.Types.Pipeline.WitnessedInput.validate witnessed |> Result.defaultWith failwith
    let plans = witnessed.Operations |> List.choose (function MLIROp.SpatialModule(SpatialModuleWitness.Hardware value) -> Some value | _ -> None)
    Assert.Equal<HardwareModuleWitness list>([plan],plans)
    Assert.DoesNotContain("seq.compreg",witnessed.Text)
    let realized = BackEnd.CIRCT.HardwareRealization.realize witnessed |> Result.defaultWith failwith
    let checkedText = circtOpt realized.Text
    Assert.Contains("seq.initial",checkedText)
    Assert.Contains("seq.compreg",checkedText)
    Assert.Contains("hw.instance",checkedText)
    Assert.DoesNotContain(realized.Operations,fun operation -> match operation with MLIROp.SpatialModule _ -> true | _ -> false)

[<Theory>]
[<InlineData("reset")>]
[<InlineData("clock")>]
[<InlineData("proof")>]
let ``hardware publication retracts changed reset clock or proof`` defect =
    let graph,plan = admitted.Value
    let changed =
        match defect with
        | "proof" -> {graph with Nodes=graph.Nodes.Remove plan.Obligations.Head}
        | _ ->
            let id = if defect="clock" then plan.ClockDeclaration else plan.ResetDeclaration
            let node = graph.Nodes[id]
            {graph with Nodes=graph.Nodes.Add(id,{node with Children=[]})}
    match Clef.Compiler.PSGSaturation.SemanticGraph.SpatialPublication.project changed with
    | Error failures -> Assert.NotEmpty failures
    | Ok _ -> failwith "Changed source hardware retained its old publication."

[<Fact>]
let ``a changed held reset plan cannot borrow the unchanged source proof`` () =
    let graph,plan = admitted.Value
    let changedPlan = {plan with ResetFields=plan.ResetFields |> List.map (fun field -> {field with Reset=1I})}
    // Keep both held copies consistent to exercise the proof/plan boundary,
    // rather than merely detecting disagreement between two duplicate rows.
    let edges = graph.Edges |> List.map (fun edge ->
        let role =
            match edge.Role with
            | EdgeRole.HardwareModule held when held.Site=plan.Site -> EdgeRole.HardwareModule changedPlan
            | EdgeRole.SpatialModuleDomain domain ->
                EdgeRole.SpatialModuleDomain {domain with Hardware=domain.Hardware |> List.map (fun held -> if held.Site=plan.Site then changedPlan else held)}
            | role -> role
        {edge with Role=role})
    match Clef.Compiler.PSGSaturation.SemanticGraph.SpatialPublication.project {graph with Edges=edges} with
    | Error failures -> Assert.NotEmpty failures
    | Ok _ -> failwith "A different reset value acquired the original reset coverage proof."

[<Fact>]
let ``absent reset cannot acquire power on behavior in a witness`` () =
    let source = authority.Split('\n') |> Array.filter (fun line -> not(line.StartsWith("let reset ="))) |> String.concat "\n"
    let result = check source
    match Clef.Compiler.PSGSaturation.SemanticGraph.SpatialPublication.project result.Graph with
    | Error failures -> Assert.Contains(failures,fun failure -> failure.Reason.Contains("reset",System.StringComparison.OrdinalIgnoreCase))
    | Ok _ -> failwith "Missing reset declaration acquired invented hardware semantics."
