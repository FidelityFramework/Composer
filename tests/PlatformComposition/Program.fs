module PlatformCompositionTests

open System
open System.IO
open Clef.Compiler.Project
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution

let repos = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../../.."))
let metadata = Path.Combine(repos, "BAREWire/src/BAREWire.PlatformMetadata.fidproj")
let fullBarewire = Path.Combine(repos, "BAREWire/src/BAREWire.fidproj")
let contracts = Path.Combine(repos, "Fidelity.Platform/Contracts/Fidelity.Platform.Contracts.fidproj")
let work = Path.Combine(Path.GetTempPath(), "composer-platform-composition-" + Guid.NewGuid().ToString("N"))
let require condition message = if not condition then failwith message
let write (relative: string) (text: string) =
    let path = Path.Combine(work, relative)
    Directory.CreateDirectory(Path.GetDirectoryName path) |> ignore
    File.WriteAllText(path, text)
    path
let package name sources dependencies extra =
    let quoted = sources |> List.map (sprintf "\"%s\"") |> String.concat ", "
    let deps = dependencies |> List.map (fun (name, path) -> sprintf "%s = { path = \"%s\" }" name path) |> String.concat "\n"
    sprintf "[package]\nname = \"%s\"\n[compilation]\ntarget = \"library\"\n[build]\noutput_kind = \"library\"\nsources = [%s]\n[dependencies]\n%s\n%s" name quoted deps extra
let load path = FidprojLoader.load path |> function Ok options -> options | Error message -> failwith message
let resolve path = SourceResolver.getSourcesAndLibraries (load path) |> function Ok value -> value | Error error -> failwith (SourceResolutionError.format error)
let check path = ProjectChecker.checkProject path |> function Ok value -> value | Error message -> failwith message
let clean project = require (not (ProjectChecker.hasErrors project)) (String.concat "\n" (ProjectChecker.getErrorMessages project))
let diagnostic (code: string) (message: string) project =
    require (project.CheckResult.Diagnostics |> List.exists (fun d -> d.Code = code && d.Message.Contains(message, StringComparison.Ordinal)))
        (sprintf "Expected %s containing '%s':\n%s" code message (String.concat "\n" (ProjectChecker.getErrorMessages project)))

let declaration moduleName id bits =
    sprintf """namespace Fixture
open BAREWire.Platform
module %s =
    let core: TargetCore = {
        Os = "none"; Arch = "x86_64"; WordSizeBits = %d; Endianness = "little"; Runtime = "bare"
        Triple = "x86_64-unknown-none"; CpuModel = ""
        Widths = [| { Name = "Pointer"; Bits = %d }; { Name = "Register"; Bits = %d } |]
        Representations = [|
            { Name = "int32"; Capability = "native"; Family = "int"; Bits = 32; MinMagnitude = "-2147483648"; MaxMagnitude = "2147483647"; Boundary = "wrap" }
            { Name = "int64"; Capability = "native"; Family = "int"; Bits = 64; MinMagnitude = "-9223372036854775808"; MaxMagnitude = "9223372036854775807"; Boundary = "wrap" }
        |]
    }
    let port: MemorySpace = {
        Name = "port"; Kind = "peripheral"; Capacity = 4096; Alignment = 4096; Granularity = 4
        Growth = "fixed"; Base = Some 4096; Access = "rw"; Notes = ""; MapKind = ""; Since = ""; Until = ""
    }
    let descriptor: PlatformDescription = {
        Id = "%s"; DisplayName = "%s"; Substrate = "cpu"; Core = Some core
        Spaces = [| port |]; ProgramLifetime = None; Surfaces = [||]; Buffers = [||]; Transports = [||]
        Lifecycle = { Clocks = [||]; Resets = [||]; Entry = "main"; Teardown = ""; Persistence = "volatile" }
        Notes = [||]; Limits = [||]
    }
        """ moduleName bits bits bits id id

let createSelected name selector ownerSource aliasSource metadataExtra =
    write (name + "/owner/Description.clef") ownerSource |> ignore
    // A second file in the same namespace must not erase the first module's
    // canonical namespace identity in the semantic graph.
    write (name + "/owner/Catalogue.clef") (declaration "Catalogue" "unselected-catalogue" 32) |> ignore
    let owner = write (name + "/owner/Owner.fidproj") (package "Owner" ["Description.clef"; "Catalogue.clef"] ["metadata", metadata] "")
    let sources =
        match aliasSource with
        | Some text -> write (name + "/profile/Selection.clef") text |> ignore; ["Selection.clef"]
        | None -> []
    let selection = selector |> Option.map (sprintf "description = \"%s\"\n") |> Option.defaultValue ""
    let section = "[platform]\nruntime_model = \"freestanding\"\nos = \"none\"\narch = \"x86_64\"\n" + selection + metadataExtra
    let profile = write (name + "/profile/Profile.fidproj") (package "Selected" sources ["owner", owner] section)
    write (name + "/app/Main.clef") "module App\n[<EntryPoint>]\nlet main _ = 0\n" |> ignore
    write (name + "/app/App.fidproj") ((package "App" ["Main.clef"] ["platform", profile] "").Replace("target = \"library\"", "target = \"cpu\""))

let tests = ResizeArray<string * (unit -> unit)>()
let test name body = tests.Add(name, body)

do
    test "diamond-source-and-link-identity" (fun () ->
        let shared = write "diamond/Shared.clef" "module Shared\nlet value = 1\n"
        let sharedProject = write "diamond/Shared.fidproj" (package "Shared" ["Shared.clef"] [] "[link]\nlibraries = [\"c\"]\n")
        let branch name =
            write ("diamond/" + name + ".clef") (sprintf "module %s\nlet value = Shared.value\n" name) |> ignore
            write ("diamond/" + name + ".fidproj") (package name [name + ".clef"] ["shared", sharedProject] "")
        let a, b = branch "A", branch "B"
        write "diamond/App.clef" "module App\nlet value = A.value + B.value\n" |> ignore
        let app = write "diamond/App.fidproj" (package "App" ["App.clef"] ["a", a; "b", b] "[link]\nlibraries = [\"c\", \"m\"]\n")
        let resolved = resolve app
        require (resolved.SourcePaths |> List.filter ((=) shared) |> List.length = 1) "Diamond compiled shared source twice"
        require (resolved.SourcePaths |> List.map Path.GetFileName = ["Shared.clef"; "A.clef"; "B.clef"; "App.clef"]) "Dependency order changed"
        require (resolved.LinkedLibraries = ["c"; "m"]) "Link identities changed"
        check app |> clean
        ProjectChecker.checkProjectWithVolatile app Map.empty |> function Ok p -> clean p | Error e -> failwith e)

    test "normalized-overlapping-source-identity" (fun () ->
        write "overlap/Shared.clef" "module Shared\nlet value = 1\n" |> ignore
        let a = write "overlap/A.fidproj" (package "A" ["Shared.clef"] [] "")
        let b = write "overlap/B.fidproj" (package "B" ["./Shared.clef"] [] "")
        write "overlap/App.clef" "module App\nlet value = Shared.value\n" |> ignore
        let app = write "overlap/App.fidproj" (package "App" ["Shared.clef"; "App.clef"] ["a", a; "b", b] "")
        require ((resolve app).SourcePaths.Length = 2) "Normalized source paths were compiled more than once"
        check app |> clean)

    for name, child in ["self-cycle", false; "mutual-cycle", true] do
        test name (fun () ->
            let aPath = Path.Combine(work, name, "A.fidproj")
            let bPath = if child then Path.Combine(work, name, "B.fidproj") else aPath
            let a = write (name + "/A.fidproj") (package "A" [] ["b", bPath] "")
            if child then write (name + "/B.fidproj") (package "B" [] ["a", aPath] "") |> ignore
            match SourceResolver.getSourcesAndLibraries (load a) with
            | Error(CircularDependency chain) -> require (chain.Length >= 2) "Cycle lost its dependency chain"
            | other -> failwithf "Expected circular dependency diagnostic, got %A" other
            match ProjectChecker.checkProject a with
            | Error text -> require (text.Contains "Circular dependency") "Project loader hid the dependency cycle"
            | Ok _ -> failwith "Project accepted a cycle")

    for name, deps in ["barewire-metadata-first", ["a", metadata; "b", fullBarewire]; "barewire-full-first", ["a", fullBarewire; "b", metadata]] do
        test name (fun () ->
            write (name + "/Main.clef") "module App\nlet value = 1\n" |> ignore
            let app = write (name + "/App.fidproj") (package "App" ["Main.clef"] deps "")
            let sources = (resolve app).SourcePaths
            require (List.distinct sources = sources) "BAREWire source ownership duplicates declarations"
            require (sources |> List.filter (fun p -> p.EndsWith "/Platform/Description.fs") |> List.length = 1) "BAREWire description source lost its identity"
            check app |> clean)

    test "explicit-sibling-export-ignores-catalogue" (fun () ->
        let app = createSelected "sibling" (Some "Fixture.Owner.descriptor") (declaration "Owner" "selected-owner" 64) None ""
        let project = check app
        clean project
        let platform = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.resolve project.CheckResult.Graph |> Option.get
        require (platform.Id = "selected-owner") "Catalogue became the selected platform"
        require (project.CheckResult.PlatformContext.Value.Dimensions.["Pointer"] = 64) "Selected width did not reach the platform context")

    test "explicit-quoted-alias-preserves-record-identity" (fun () ->
        let alias = "namespace Fixture\nmodule Selection =\n    let descriptor = <@ Owner.descriptor @>\n"
        let app = createSelected "alias" (Some "Fixture.Selection.descriptor") (declaration "Owner" "selected-owner" 64) (Some alias) ""
        let project = check app
        clean project
        let graph = project.CheckResult.Graph
        let platform = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.resolve graph |> Option.get
        let original = graph.Nodes.Values |> Seq.find (fun n -> qualifiedExportName graph n = Some "Fixture.Owner.descriptor")
        let record = recordOf graph (List.exactlyOne original.Children) |> Option.get |> fst
        require (platform.Node = record.Id) "Alias copied the description record"
        let port = graph.Nodes.Values |> Seq.find (fun n -> qualifiedExportName graph n = Some "Fixture.Owner.port")
        let record = recordOf graph (List.exactlyOne port.Children) |> Option.get |> fst
        require (platform.Spaces.Head.Node = record.Id) "Alias copied the memory-space declaration")

    test "explicit-file-level-module-export" (fun () ->
        let source = (declaration "Owner" "full-module-owner" 64).Replace("namespace Fixture\nopen BAREWire.Platform\nmodule Owner =", "module Fixture.Owner\nopen BAREWire.Platform")
        let app = createSelected "full-module" (Some "Fixture.Owner.descriptor") source None ""
        let project = check app
        clean project
        let platform = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.resolve project.CheckResult.Graph |> Option.get
        require (platform.Id = "full-module-owner") "Fully qualified file-level module export was not selected")

    test "selected-manifest-is-authoritative" (fun () ->
        let app = createSelected "manifest-authority" (Some "Fixture.Owner.descriptor") (declaration "Owner" "selected-owner" 64) None ""
        File.AppendAllText(app, "[platform]\nruntime_model = \"libc\"\nos = \"linux\"\narch = \"aarch64\"\ndescription = \"Fixture.Catalogue.descriptor\"\n")
        let project = check app
        clean project
        require (project.CheckResult.PlatformContext.Value.PlatformDescription = Some "Fixture.Owner.descriptor") "Application metadata overrode the selected manifest"
        let platform = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.resolve project.CheckResult.Graph |> Option.get
        require (platform.Id = "selected-owner") "Application catalogue replaced the selected root")

    test "standalone-explicit-platform" (fun () ->
        let app = createSelected "standalone" (Some "Fixture.Owner.descriptor") (declaration "Owner" "standalone-owner" 64) None ""
        let selected = (load app).PlatformPath.Value
        File.WriteAllText(selected, (File.ReadAllText selected).Replace("target = \"library\"", "target = \"cpu\""))
        let project = check selected
        clean project
        let platform = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.resolve project.CheckResult.Graph |> Option.get
        require (platform.Id = "standalone-owner") "Standalone manifest did not select its exported dependency root")

    let unrelatedPackage name =
        write (name + "/unrelated/Rogue.clef") (declaration "Rogue" "unrelated" 32) |> ignore
        write (name + "/unrelated/Rogue.fidproj") (package "Unrelated" ["Rogue.clef"] ["metadata", metadata] "")
    let addUnrelated app rogue =
        File.WriteAllText(app, (File.ReadAllText app).Replace("[dependencies]\n", sprintf "[dependencies]\na_unrelated = { path = \"%s\" }\n" rogue))

    test "application-cannot-supply-missing-platform-export" (fun () ->
        let app = createSelected "app-export-injection" (Some "Fixture.Rogue.descriptor") (declaration "Owner" "selected-owner" 64) None ""
        write "app-export-injection/app/Rogue.clef" (declaration "Rogue" "application-injection" 32) |> ignore
        File.WriteAllText(app, (File.ReadAllText app).Replace("sources = [\"Main.clef\"]", "sources = [\"Rogue.clef\", \"Main.clef\"]"))
        check app |> diagnostic "CCS8206" "was not found in the selected platform's source dependency closure")

    test "unrelated-dependency-cannot-supply-platform-export" (fun () ->
        let app = createSelected "dependency-export-injection" (Some "Fixture.Rogue.descriptor") (declaration "Owner" "selected-owner" 64) None ""
        addUnrelated app (unrelatedPackage "dependency-export-injection")
        check app |> diagnostic "CCS8206" "was not found in the selected platform's source dependency closure")

    test "unrelated-qualified-shadow-does-not-replace-export" (fun () ->
        let app = createSelected "app-shadow" (Some "Fixture.Owner.descriptor") (declaration "Owner" "selected-owner" 64) None ""
        write "app-shadow/app/Shadow.clef" (declaration "Owner" "application-shadow" 32) |> ignore
        File.WriteAllText(app, (File.ReadAllText app).Replace("sources = [\"Main.clef\"]", "sources = [\"Shadow.clef\", \"Main.clef\"]"))
        let project = check app
        clean project
        let platform = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.resolve project.CheckResult.Graph |> Option.get
        require (platform.Id = "selected-owner") "Application qualified shadow replaced the selected export")

    for name, source, alias, selector in [
        "unrelated-core-reference", (declaration "Owner" "selected-owner" 64).Replace("Core = Some core", "Core = Some Rogue.core"), None, "Fixture.Owner.descriptor"
        "unrelated-root-alias", declaration "Owner" "selected-owner" 64, Some "namespace Fixture\nmodule Selection =\n    let descriptor = Rogue.descriptor\n", "Fixture.Selection.descriptor"
    ] do
        test name (fun () ->
            let app = createSelected name (Some selector) source alias ""
            addUnrelated app (unrelatedPackage name)
            check app |> diagnostic "CCS8206" "outside the selected platform's source dependency closure")

    let abi moduleName name bits registers =
        sprintf "namespace Fixture\nopen BAREWire.Platform\nmodule %s =\n    let descriptor: CAbiDescriptor = { Name = \"%s\"; PointerBits = %d; ScalarAggregateRegisterBytes = %d }\n" moduleName name bits registers
    test "explicit-cabi-ignores-application-inventory" (fun () ->
        let app = createSelected "cabi-scope" (Some "Fixture.Owner.descriptor") ((declaration "Owner" "selected-owner" 64) + "\n" + abi "Abi" "selected-abi" 64 16) None ""
        write "cabi-scope/app/Rogue.clef" (abi "RogueAbi" "rogue-abi" 32 8) |> ignore
        File.WriteAllText(app, (File.ReadAllText app).Replace("sources = [\"Main.clef\"]", "sources = [\"Rogue.clef\", \"Main.clef\"]"))
        let project = check app
        clean project
        require (cAbiOfGraph project.CheckResult.Graph = ["selected-abi", 64, 16]) "Application ABI entered the explicit platform ABI selection")

    test "explicit-cabi-cannot-be-supplied-by-application" (fun () ->
        let app = createSelected "cabi-injection" (Some "Fixture.Owner.descriptor") (declaration "Owner" "selected-owner" 64) None ""
        write "cabi-injection/app/Rogue.clef" (abi "RogueAbi" "rogue-abi" 32 8) |> ignore
        File.WriteAllText(app, (File.ReadAllText app).Replace("sources = [\"Main.clef\"]", "sources = [\"Rogue.clef\", \"Main.clef\"]"))
        let project = check app
        clean project
        require (List.isEmpty (cAbiOfGraph project.CheckResult.Graph)) "Application supplied the selected platform's missing ABI")

    test "legacy-cabi-discovery" (fun () ->
        write "legacy-cabi/Abi.clef" (abi "Abi" "legacy-abi" 64 16) |> ignore
        let path = write "legacy-cabi/P.fidproj" (package "Legacy" ["Abi.clef"] ["metadata", metadata] "")
        let project = check path
        clean project
        require (cAbiOfGraph project.CheckResult.Graph = ["legacy-abi", 64, 16]) "Legacy structural ABI discovery changed")

    let inventory moduleName pin device =
        sprintf """namespace Fixture
open Fidelity.Platform.Contracts
module %s =
    let led: PinEndpoint = { LogicalName = "led"; PackagePin = "%s"; Direction = "Output"; Standard = "LVCMOS33"; Description = None }
    let clock: ClockEndpoint = { Name = "clk"; FrequencyHz = 100000000; PackagePin = "E3"; Standard = "LVCMOS33"; Description = None }
    let part: PlatformDescriptor = {
        Id = "part"; DisplayName = "part"; Substrate = "fpga"; Vendor = "test"; Family = "test"
        Device = "%s"; Package = "csg324"; SpeedGrade = "-1"; Core = None
        Clocks = [clock]; Resets = []; Groups = []; Uarts = []; DedicatedPins = [led]; Notes = []
    }
        """ moduleName pin device
    let pinApplication = "module App\ntype Outputs = { [<Pin(\"led\")>] Led: bool }\nlet state: Outputs = { Led = true }\n[<EntryPoint>]\nlet main _ = if state.Led then 0 else 1\n"
    test "explicit-pin-inventory-ignores-application-shadow" (fun () ->
        let app = createSelected "pin-scope" (Some "Fixture.Owner.descriptor") (declaration "Owner" "selected-owner" 64) None ""
        let owner = Path.Combine(work, "pin-scope/owner/Owner.fidproj")
        write "pin-scope/owner/Inventory.clef" (inventory "Inventory" "H5" "xc7a100t") |> ignore
        File.WriteAllText(owner, (File.ReadAllText owner).Replace("sources = [", "sources = [\"Inventory.clef\", ").Replace("[dependencies]\n", sprintf "[dependencies]\ncontracts = { path = \"%s\" }\n" contracts))
        write "pin-scope/app/Rogue.clef" (inventory "RogueInventory" "WRONG" "rogue") |> ignore
        write "pin-scope/app/Main.clef" pinApplication |> ignore
        File.WriteAllText(app, (File.ReadAllText app).Replace("sources = [\"Main.clef\"]", "sources = [\"Rogue.clef\", \"Main.clef\"]"))
        let project = check app
        clean project
        let pins = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformBindings.pins project.CheckResult.Graph |> Option.get
        require (pins.Pins |> List.exists (fun p -> p.PortName = "led" && p.PackagePin = "H5")) "Application pin shadow replaced the selected product's wiring"
        require (pins.DevicePart = "xc7a100tcsg324-1") "Application inventory supplied the device part")

    test "legacy-pin-inventory-discovery" (fun () ->
        write "legacy-pins/Inventory.clef" (inventory "Inventory" "H5" "xc7a100t") |> ignore
        write "legacy-pins/Main.clef" pinApplication |> ignore
        let path = write "legacy-pins/P.fidproj" (package "LegacyPins" ["Inventory.clef"; "Main.clef"] ["contracts", contracts] "")
        let project = check path
        clean project
        let pins = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformBindings.pins project.CheckResult.Graph |> Option.get
        require (pins.Pins |> List.exists (fun p -> p.PortName = "led" && p.PackagePin = "H5")) "Legacy pin inventory discovery changed")

    test "hardware-clock-metadata-requires-no-runtime-rodata" (fun () ->
        let app = createSelected "hardware-clock" (Some "Fixture.Owner.descriptor") ((declaration "Owner" "fpga-owner" 64).Replace("Core = Some core", "Core = None")) None ""
        let owner = Path.Combine(work, "hardware-clock/owner/Owner.fidproj")
        write "hardware-clock/owner/Inventory.clef" (inventory "Inventory" "H5" "xc7a100t") |> ignore
        File.WriteAllText(owner, (File.ReadAllText owner).Replace("sources = [", "sources = [\"Inventory.clef\", ").Replace("[dependencies]\n", sprintf "[dependencies]\ncontracts = { path = \"%s\" }\n" contracts))
        File.WriteAllText(app, (File.ReadAllText app).Replace("target = \"cpu\"", "target = \"fpga\""))
        write "hardware-clock/app/Main.clef" """module App
open Fidelity.Platform.Contracts
type State = { Value: int }
type Design = { InitialState: State; Step: State -> State; Clock: ClockEndpoint }
let step (state: State) = { Value = if state.Value = 0 then 1 else 0 }
[<HardwareModule>]
let design: Design = { InitialState = { Value = 0 }; Step = step; Clock = Fixture.Inventory.clock }
        """ |> ignore
        let project = check app
        clean project
        let graph = project.CheckResult.Graph
        let clock = graph.Nodes.Values |> Seq.find (fun n -> qualifiedExportName graph n = Some "Fixture.Inventory.clock")
        let step = graph.Nodes.Values |> Seq.find (fun n -> qualifiedExportName graph n = Some "App.step")
        require (not clock.IsReachable && step.IsReachable) "Hardware metadata and executable logic have the wrong reachability"
        require graph.StaticStringPool.IsNone "Hardware clock metadata allocated native string storage")

    test "cpu-runtime-strings-still-require-rodata" (fun () ->
        let app = createSelected "cpu-rodata" (Some "Fixture.Owner.descriptor") (declaration "Owner" "cpu-owner" 64) None ""
        write "cpu-rodata/app/Main.clef" "module App\nlet consume (value: string) = if value = \"\" then 0 else 1\n[<EntryPoint>]\nlet main _ = consume \"runtime string\"\n" |> ignore
        check app |> diagnostic "CCS8206" "the platform has no rodata memory-space declaration")

    for name, triple, message in [
        "triple-architecture-mismatch", "aarch64-unknown-none", "target triple architecture 'aarch64' disagrees"
        "triple-os-mismatch", "x86_64-unknown-linux-gnu", "target triple OS 'linux' disagrees"
        "triple-missing-os-component", "x86_64", "target triple OS '' disagrees"
    ] do
        test name (fun () ->
            let source = (declaration "Owner" "selected-owner" 64).Replace("x86_64-unknown-none", triple)
            createSelected name (Some "Fixture.Owner.descriptor") source None "" |> check |> diagnostic "CCS8207" message)

    test "supported-cortex-m33-triple" (fun () ->
        let source = (declaration "Owner" "m33-owner" 32).Replace("Arch = \"x86_64\"", "Arch = \"arm_cortex_m33\"").Replace("x86_64-unknown-none", "thumbv8m.main-none-eabi")
        let app = createSelected "m33-triple" (Some "Fixture.Owner.descriptor") source None ""
        let profile = (load app).PlatformPath.Value
        File.WriteAllText(profile, (File.ReadAllText profile).Replace("arch = \"x86_64\"", "arch = \"arm_cortex_m33\""))
        check app |> clean)

    test "supported-linux-x86-64-triple" (fun () ->
        let source = (declaration "Owner" "linux-owner" 64).Replace("Os = \"none\"", "Os = \"linux\"").Replace("Runtime = \"bare\"", "Runtime = \"libc\"").Replace("x86_64-unknown-none", "x86_64-unknown-linux-gnu")
        let app = createSelected "linux-triple" (Some "Fixture.Owner.descriptor") source None ""
        let profile = (load app).PlatformPath.Value
        File.WriteAllText(profile, (File.ReadAllText profile).Replace("os = \"none\"", "os = \"linux\"").Replace("runtime_model = \"freestanding\"", "runtime_model = \"libc\""))
        check app |> clean)

    for name, selector, source, alias, code, message in [
        "missing-export", "Fixture.Owner.missing", declaration "Owner" "selected" 64, None, "CCS8206", "was not found"
        "wrong-export-shape", "Fixture.Owner.core", declaration "Owner" "selected" 64, None, "CCS8206", "immutable PlatformDescription"
        "mutable-export", "Fixture.Selection.descriptor", declaration "Owner" "selected" 64, Some "namespace Fixture\nmodule Selection =\n    let mutable descriptor = Owner.descriptor\n", "CCS8206", "immutable PlatformDescription"
        "runtime-export", "Fixture.Selection.descriptor", declaration "Owner" "selected" 64, Some "namespace Fixture\nmodule Selection =\n    let choose () = Owner.descriptor\n    let descriptor = choose ()\n", "CCS8206", "immutable PlatformDescription"
        "ambiguous-export", "Fixture.Owner.descriptor", declaration "Owner" "selected" 64, Some (declaration "Owner" "duplicate" 64), "CCS8208", "is ambiguous"
        "core-architecture-mismatch", "Fixture.Owner.descriptor", (declaration "Owner" "selected" 64).Replace("Arch = \"x86_64\"", "Arch = \"aarch64\""), None, "CCS8207", "arch 'x86_64' disagrees"
        "core-os-mismatch", "Fixture.Owner.descriptor", (declaration "Owner" "selected" 64).Replace("Os = \"none\"", "Os = \"linux\""), None, "CCS8207", "os 'none' disagrees"
        "core-runtime-mismatch", "Fixture.Owner.descriptor", (declaration "Owner" "selected" 64).Replace("Runtime = \"bare\"", "Runtime = \"libc\""), None, "CCS8207", "runtime_model 'freestanding' disagrees"
    ] do
        test name (fun () -> createSelected name (Some selector) source alias "" |> check |> diagnostic code message)

    test "legacy-directory-scope" (fun () ->
        let app = createSelected "legacy" None (declaration "Owner" "outside-catalogue" 64) (Some (declaration "Local" "legacy-local" 32)) ""
        let project = check app
        clean project
        let platform = Clef.Compiler.PSGSaturation.SemanticGraph.PlatformResolution.resolve project.CheckResult.Graph |> Option.get
        require (platform.Id = "legacy-local") "Legacy directory selection changed")

    for invalid in ["\"\""; "\"descriptor\""; "false"; "\"Fixture..descriptor\""] do
        test ("invalid-selector-" + invalid) (fun () ->
            let path = write "invalid-selector/P.fidproj" (package "Invalid" [] [] ("[platform]\nruntime_model = \"bare\"\ndescription = " + invalid + "\n"))
            match FidprojLoader.load path with
            | Error text -> require (text.Contains "fully qualified binding") "Selector validation lost its diagnostic"
            | Ok _ -> failwith "Malformed explicit selector was silently ignored")

[<EntryPoint>]
let main args =
    Directory.CreateDirectory work |> ignore
    let mutable failed = 0
    let mutable executed = 0
    for name, body in tests do
        if args.Length = 0 || Array.contains name args then
            executed <- executed + 1
            try body(); printfn "PASS %s" name
            with error -> failed <- failed + 1; eprintfn "FAIL %s: %s" name error.Message
    printfn "%d/%d platform composition checks passed; fixtures: %s" (executed - failed) executed work
    if failed = 0 then 0 else 1
