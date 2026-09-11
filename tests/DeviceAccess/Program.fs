module DeviceAccessTests

open System
open System.IO
open System.Text.Json
open System.Text.RegularExpressions
open Clef.Compiler.Project
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

let mcuSource = """module DeviceAccessCheck
open Fidelity.Platform.Contracts
open Fidelity.Platform.Hardware.Silicon.MCU.Renesas.RA6M5.R7FA6M5BH3CFC
let region: DeviceRegion = { Space = Description.port0; AddressSpace = "cpu-physical"; Source = "test region" }
let register: DeviceRegister = { Name = "led"; Region = region; Offset = 0; TransactionBits = 16; Access = "rw"; ByteOrder = "little"; Ordering = "volatile" }
let mapping: DeviceMapping = { Name = "gpio"; Region = region; AddressSpace = "cpu-physical"; Base = Some 0x40080000; Lifetime = "image"; Establishment = "reset-identity"; Source = "test mapping premise" }
let predicate: ClefPredicate = { Name = "extent"; Condition = <@ register.Offset >= 0 && register.Offset + register.TransactionBits / 8 <= Description.port0.Capacity @>; Source = "test relation" }
let grant: DeviceGrant = { Name = "gpio"; Mapping = mapping; Registers = [| register |]; Access = "rw"; Predicates = [| predicate |] }
let plan: Expr<DeviceAccessPlan> = <@ { Name = "test"; Grants = [| grant |] } @>
[<EntryPoint>]
let main _ =
    let handle = Mmio.bind16 "gpio" "led"
    Mmio.write16 handle 255
    let _ = Mmio.read16 handle
    0
"""

let guestSource = """module DeviceAccessCheck
open Fidelity.Platform.Contracts
open Fidelity.Platform.Profiles.RestrictedGuest64
let mapping: DeviceMapping = { Name = "transport-map"; Region = Description.transportRegion; AddressSpace = "cpu-virtual"; Base = Some 8589934592; Lifetime = "image"; Establishment = "boot-contract"; Source = "synthetic boot mapping premise" }
let grant: DeviceGrant = { Name = "transport"; Mapping = mapping; Registers = [| Description.status; Description.control |]; Access = "rw"; Predicates = [||] }
let plan: Expr<DeviceAccessPlan> = <@ { Name = "restricted-guest"; Grants = [| grant |] } @>
[<EntryPoint>]
let main _ =
    let status = Mmio.bind32 "transport" "status"
    let control = Mmio.bind32 "transport" "control"
    let _ = Mmio.read32 status
    Mmio.write32 control 1
    0
"""

let replace (oldText: string) (newText: string) (source: string) =
    if not (source.Contains oldText) then failwith ("Missing test mutation: " + oldText)
    source.Replace(oldText, newText)

let cases = [|
    "mcu_bound", false, mcuSource, ""
    "region_extent", false, mcuSource |> replace "Offset = 0;" "Offset = 32;", "outside its region"
    "negative_offset", false, mcuSource |> replace "Offset = 0;" "Offset = -1;", "outside its region"
    "unaligned_register", false, mcuSource |> replace "Offset = 0;" "Offset = 1;", "misaligned"
    "transaction_width", false, mcuSource |> replace "TransactionBits = 16;" "TransactionBits = 32;", "transaction requirement"
    "fractional_transaction", false, mcuSource |> replace "TransactionBits = 16;" "TransactionBits = 7;", "multiple of eight"
    "non_device_region", false, mcuSource |> replace "Space = Description.port0;" "Space = Description.sram;", "fixed peripheral/register space"
    "different_region_identity", false, mcuSource |> replace "let mapping:" "let otherRegion: DeviceRegion = { Space = Description.port0; AddressSpace = \"cpu-physical\"; Source = \"different declaration\" }\nlet mapping:" |> replace "Name = \"gpio\"; Region = region;" "Name = \"gpio\"; Region = otherRegion;", "same DeviceRegion declaration"
    "register_read_only", false, mcuSource |> replace "TransactionBits = 16; Access = \"rw\"" "TransactionBits = 16; Access = \"r\"", "write is not permitted"
    "grant_read_only", false, mcuSource |> replace "|]; Access = \"rw\"; Predicates" "|]; Access = \"r\"; Predicates", "write is not permitted"
    "register_write_only", false, mcuSource |> replace "TransactionBits = 16; Access = \"rw\"" "TransactionBits = 16; Access = \"w\"", "read is not permitted"
    "missing_register", false, mcuSource |> replace "\"gpio\" \"led\"" "\"gpio\" \"ungranted\"", "register is not granted"
    "missing_grant", false, mcuSource |> replace "\"gpio\" \"led\"" "\"missing\" \"led\"", "grant is not in the selected plan"
    "raw_bypass", false, mcuSource |> replace "Mmio.bind16 \"gpio\" \"led\"" "Mmio.reg16 0x40080000", "Raw MMIO address construction is forbidden"
    "empty_plan", false, mcuSource |> replace "Grants = [| grant |]" "Grants = [||]", "grant is not in the selected plan"
    "duplicate_plan", false, mcuSource |> replace "[<EntryPoint>]" "let second: Expr<DeviceAccessPlan> = <@ { Name = \"second\"; Grants = [| grant |] } @>\n[<EntryPoint>]", "Exactly one DeviceAccessPlan"
    "duplicate_grant", false, mcuSource |> replace "Grants = [| grant |]" "Grants = [| grant; grant |]", "Duplicate grant name"
    "identity_base", false, mcuSource |> replace "Base = Some 0x40080000" "Base = Some 0x40081000", "identity mapping disagrees"
    "identity_space", false, mcuSource |> replace "Region = region; AddressSpace = \"cpu-physical\"; Base" "Region = region; AddressSpace = \"cpu-virtual\"; Base", "identity mapping disagrees"
    "mcu_pointer_overflow", false, mcuSource |> replace "Base = Some 0x40080000" "Base = Some 4294967296" |> replace "Establishment = \"reset-identity\"" "Establishment = \"boot-contract\"", "mapping extent lies outside"
    "runtime_base", false, mcuSource |> replace "Base = Some 0x40080000" "Base = None", "mapping base is pending"
    "runtime_establishment", false, mcuSource |> replace "Establishment = \"reset-identity\"" "Establishment = \"runtime\"", "mapping establishment is pending"
    "scoped_lifetime", false, mcuSource |> replace "Lifetime = \"image\"" "Lifetime = \"scope\"", "mapping lifetime is pending"
    "wrong_endian", false, mcuSource |> replace "ByteOrder = \"little\"" "ByteOrder = \"big\"", "byte order"
    "unsupported_ordering", false, mcuSource |> replace "Ordering = \"volatile\"" "Ordering = \"sequential\"", "ordering requirement is unsupported"
    "predicate_false", false, mcuSource |> replace "<= Description.port0.Capacity" "> Description.port0.Capacity", "is Contradicted"
    "predicate_unknown_call", false, mcuSource |> replace "let predicate:" "let requirement n = n >= 0\nlet predicate:" |> replace "register.Offset >= 0" "requirement register.Offset", "is Pending"
    "predicate_mutable", false, mcuSource |> replace "let predicate:" "let mutable requirement = 1\nlet predicate:" |> replace "register.Offset >= 0" "requirement = 1", "is Pending"
    "predicate_bad_annotation", false, mcuSource |> replace "register.Offset >= 0" "(1 : bool) = (1 : bool)", "source declaration contains a type error"
    "predicate_bad_equality", false, mcuSource |> replace "register.Offset >= 0" "1 <> false", "source declaration contains a type error"
    "predicate_fixed_arithmetic", false, mcuSource |> replace "register.Offset >= 0" "255uy + 1uy = 0uy", "Numeric literal suffixes"
    "predicate_invalid_conversion", false, mcuSource |> replace "register.Offset >= 0" "int (Mmio.read16 (Mmio.reg16 0x40080000)) = 0", "source declaration contains a type error"
    "predicate_runtime_read", false, mcuSource |> replace "register.Offset >= 0" "Mmio.read16 (Mmio.reg16 0x40080000) = 0", "is Pending"
    "handle_mutable", false, mcuSource |> replace "let handle =" "let mutable handle =", "handle provenance is pending"
    "grant_name_mutable", false, mcuSource |> replace "let handle =" "let mutable grantName = \"gpio\"\n    let handle =" |> replace "Mmio.bind16 \"gpio\"" "Mmio.bind16 grantName", "grant name must be static"
    "predicate_runtime_record", false, mcuSource |> replace "let predicate:" "type Settings = { mutable Required: int }\nlet settings = { Required = 1 }\nlet predicate:" |> replace "register.Offset >= 0" "settings.Required = 1" |> replace "let handle =" "settings.Required <- 2\n    let handle =", "is Pending"
    "deferred_unused", false, mcuSource |> replace "TransactionBits = 16;" "TransactionBits = 128;" |> replace "Base = Some 0x40080000" "Base = None" |> replace "    let handle = Mmio.bind16 \"gpio\" \"led\"\n    Mmio.write16 handle 255\n    let _ = Mmio.read16 handle\n" "", ""
    "oversized_write", false, mcuSource |> replace "Mmio.write16 handle 255" "Mmio.write16 handle 65536", "not proven within"
    "guest64_bound", true, guestSource, ""
    "guest64_bad_region_alignment", true, guestSource, "alignment is not a power of two"
    "guest64_bad_region_granularity", true, guestSource, "granularity is not a power of two"
    "guest64_high_address", true, guestSource |> replace "Base = Some 8589934592" "Base = Some (4294967296 * 4294967296 - 4096)", ""
    "guest64_pointer_overflow", true, guestSource |> replace "Base = Some 8589934592" "Base = Some (4294967296 * 4294967296)", "mapping extent lies outside"
    "guest64_granularity", true, guestSource |> replace "Registers = [| Description.status; Description.control |]" "Registers = [| Description.status; byteRegister |]" |> replace "let grant:" "let byteRegister: DeviceRegister = { Name = \"control\"; Region = Description.transportRegion; Offset = 4; TransactionBits = 8; Access = \"rw\"; ByteOrder = \"little\"; Ordering = \"volatile\" }\nlet grant:" |> replace "Mmio.bind32 \"transport\" \"control\"" "Mmio.bind8 \"transport\" \"control\"" |> replace "Mmio.write32 control" "Mmio.write8 control", "region granularity"
    "guest64_ungranted", true, guestSource |> replace "\"transport\" \"control\"" "\"transport\" \"management-control\"", "register is not granted"
    "guest64_runtime", true, guestSource |> replace "Base = Some 8589934592" "Base = None", "mapping base is pending"
    "guest64_read_only", true, guestSource |> replace "Mmio.write32 control 1" "Mmio.write32 status 1", "write is not permitted"
    "guest64_width", true, guestSource |> replace "Mmio.bind32 \"transport\" \"control\"" "Mmio.bind16 \"transport\" \"control\"" |> replace "Mmio.write32 control" "Mmio.write16 control", "transaction requirement"
|]

[<EntryPoint>]
let main args =
    let root = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../.."))
    let compiler = Path.Combine(root, "src/bin/Debug/net10.0/Composer")
    let work = Path.Combine(Path.GetTempPath(), "composer-device-access-" + Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory work |> ignore
    let results = ResizeArray<_>()
    for name, guest, source, expected in cases do
        if args.Length = 0 || Array.contains name args then
            let directory = Path.Combine(work, name)
            Directory.CreateDirectory directory |> ignore
            let originalPlatform = Path.GetFullPath(Path.Combine(root, "../Fidelity.Platform/" + (if guest then "Profiles/RestrictedGuest64" else "Profiles/EK_RA6M5_HelloBlinky") + "/Fidelity.Platform.fidproj"))
            let platform =
                if name.StartsWith "guest64_bad_region_" then
                    let local = Path.Combine(directory, "profile")
                    Directory.CreateDirectory local |> ignore
                    let field = if name.EndsWith "alignment" then "Alignment = 4096" else "Granularity = 4"
                    let replacement = if name.EndsWith "alignment" then "Alignment = 3" else "Granularity = 3"
                    let contracts = Path.GetFullPath(Path.Combine(root, "../Fidelity.Platform/Contracts/Fidelity.Platform.Contracts.fidproj"))
                    let machine = Path.GetFullPath(Path.Combine(root, "../Fidelity.Platform/Hardware/VirtualMachines/Synthetic/RestrictedGuest64/Description.clef"))
                    let machineSource = File.ReadAllText machine |> replace field replacement
                    File.WriteAllText(Path.Combine(local, "Machine.clef"), machineSource)
                    let localMachine = Path.Combine(local, "Machine.fidproj")
                    File.WriteAllText(localMachine, String.concat "\n" [
                        "[package]"; "name=\"InvalidGuestMachine\""; "version=\"0.1.0\""
                        "[compilation]"; "target=\"library\""
                        "[dependencies]"; "contracts={path=" + JsonSerializer.Serialize(contracts) + "}"
                        "[build]"; "sources=[\"Machine.clef\"]"; "output_kind=\"library\"" ])
                    File.Copy(Path.Combine(Path.GetDirectoryName originalPlatform, "Description.clef"), Path.Combine(local, "Description.clef"))
                    let manifest = Regex.Replace(File.ReadAllText(originalPlatform), "path\\s*=\\s*\"([^\"]+)\"", MatchEvaluator(fun m ->
                        let original = Path.GetFullPath(Path.Combine(Path.GetDirectoryName originalPlatform, m.Groups[1].Value))
                        let selected = if original.Contains("Hardware/VirtualMachines/Synthetic/RestrictedGuest64/") then localMachine else original
                        "path = " + JsonSerializer.Serialize selected))
                    let project = Path.Combine(local, "Fidelity.Platform.fidproj")
                    File.WriteAllText(project, manifest)
                    project
                else originalPlatform
            // Match the application structure: platform/workload declarations
            // are a separate module from the executable entry point.
            let entry = source.IndexOf("[<EntryPoint>]", StringComparison.Ordinal)
            File.WriteAllText(Path.Combine(directory, "Declarations.clef"), source.Substring(0, entry).Replace("module DeviceAccessCheck\n", "module DeviceAccessCheck.Declarations\n"))
            File.WriteAllText(Path.Combine(directory, "Main.clef"), "module DeviceAccessCheck\nopen DeviceAccessCheck.Declarations\n" + source.Substring(entry))
            let project = Path.Combine(directory, "Test.fidproj")
            File.WriteAllText(project, String.concat "\n" [
                "[package]"; "name=\"DeviceAccessCheck\""; "version=\"0.1.0\""
                "[compilation]"; "target=\"" + (if guest then "cpu" else "mcu") + "\""
                "[platform]"; "runtime_model=\"bare\""; "os=\"none\""; "arch=\"" + (if guest then "x86_64" else "arm_cortex_m33") + "\""
                "[dependencies]"; "platform={path=" + JsonSerializer.Serialize(platform) + "}"
                "[build]"; "sources=[\"Declarations.clef\",\"Main.clef\"]"; "output=\"test.elf\""; "output_kind=\"embedded\"" ])
            let checkedProject = FrontEnd.ProjectLoader.load project |> function Ok p -> p | Error e -> failwith e
            let messages = ProjectChecker.getErrorMessages checkedProject |> String.concat "\n"
            File.WriteAllText(Path.Combine(directory, "diagnostics.txt"), messages)
            if expected <> "" then
                if not (ProjectChecker.hasErrors checkedProject) || not (messages.Contains expected) then
                    failwithf "%s: expected '%s'; got %s. Source: %s" name expected messages directory
            elif name = "deferred_unused" then
                if ProjectChecker.hasErrors checkedProject then failwithf "Unused deferred declaration was rejected: %s" messages
                if not checkedProject.CheckResult.Graph.Codata.Value.Mmio.IsEmpty then failwith "Unused deferred declaration manufactured an access"
            else
                if ProjectChecker.hasErrors checkedProject then failwithf "%s failed: %s" name messages
                let sites = checkedProject.CheckResult.Graph.Codata.Value.Mmio |> Map.toList |> List.map snd
                if sites.IsEmpty || sites |> List.exists (fun s -> s.Binding.IsNone) then failwith "Bound program lost its declaration evidence"
                if guest then
                    if sites |> List.exists (fun s -> s.Address < 8589934592I || s.Bits <> 32) then failwith "Guest pointer width narrowed the wrong boundary"
                else
                    let predicates = sites |> List.collect (fun s -> s.Binding.Value.Predicates)
                    if predicates.IsEmpty || predicates |> List.exists (fun p -> p.Status <> Established || p.Dependencies.Length < 4) then failwith "Predicate relation/dependencies did not survive CCS"
                // Exercise the real lowering for both substrates; never execute MMIO on the host.
                Tests.Process.requireSuccess compiler ["compile"; project; "--emit-llvm"; "-k"; "--no-color"] 600000 (Some(Path.Combine(directory, "compile.log"))) |> ignore
                let ir = Path.Combine(directory, "targets/intermediates/08_output.ll")
                let optimized = Path.Combine(directory, "optimized.ll")
                Tests.Process.requireSuccess "opt" ["-S"; "-passes=default<O2>"; ir; "-o"; optimized] 60000 None |> ignore
                let text = File.ReadAllText optimized
                let width = if guest then "32" else "16"
                if not (text.Contains("load volatile i" + width) && text.Contains("store volatile i" + width)) then failwith "Bound volatile transaction lost in lowering"
                let baseText, nextText = if name = "guest64_high_address" then "-4096", "-4092" else "8589934592", "8589934596"
                if guest && not (text.Contains baseText && text.Contains nextText) then failwith "64-bit guest virtual addresses lost in lowering"
                let ledger = Path.Combine(directory, "targets/intermediates/device-access.json")
                use report = JsonDocument.Parse(File.ReadAllText ledger)
                let recorded = report.RootElement.GetProperty("sites")
                if recorded.GetArrayLength() <> sites.Length then failwith "Access ledger differs from CCS codata"
                for site in recorded.EnumerateArray() do
                    if site.GetProperty("binding").ValueKind = JsonValueKind.Null then failwith "Ledger lost a selected grant"
                if name = "mcu_bound" then
                    File.Copy(ledger, Path.Combine(directory, "accepted-device-access.json"))
                    let wrongTarget = Tests.Process.run compiler ["compile"; project; "--target"; "x86_64-unknown-linux-gnu"; "--emit-mlir"; "--no-color"] 600000 (Some(Path.Combine(directory, "wrong-target.log")))
                    if wrongTarget.ExitCode = 0 || not (wrongTarget.Output.Contains "CLI target triple") then failwith "Explicit MCU profile accepted an incompatible CLI target"
                    if wrongTarget.Output.Contains "MLIR Generation" then failwith "Conflicting target reached lowering"
                    if File.Exists ledger then failwith "Conflicting target retained stale access evidence"
                    let wrongBackendProject = Path.Combine(directory, "WrongBackend.fidproj")
                    File.WriteAllText(wrongBackendProject, File.ReadAllText(project).Replace("target=\"mcu\"", "target=\"cpu\""))
                    let wrongBackend = Tests.Process.run compiler ["compile"; wrongBackendProject; "--emit-mlir"; "--no-color"] 600000 (Some(Path.Combine(directory, "wrong-backend.log")))
                    if wrongBackend.ExitCode = 0 || not (wrongBackend.Output.Contains "Workload backend") then failwith "Explicit MCU profile accepted the CPU workload backend"
                    if wrongBackend.Output.Contains "MLIR Generation" then failwith "Conflicting backend reached lowering"
                    // A selected profile must never fall back to the build host
                    // when its reusable environment omits the target triple.
                    let incomplete = Path.Combine(directory, "missing-triple")
                    let environment = Path.GetFullPath(Path.Combine(root, "../Fidelity.Platform/Environments/Freestanding/arm_cortex_m33/Fidelity.Platform.fidproj"))
                    let clonePackage (original: string) (destination: string) (remap: string -> string) (transform: string -> string) =
                        Directory.CreateDirectory destination |> ignore
                        let originalDir = Path.GetDirectoryName original
                        File.WriteAllText(Path.Combine(destination, "Description.clef"), File.ReadAllText(Path.Combine(originalDir, "Description.clef")) |> transform)
                        let manifest = Regex.Replace(File.ReadAllText original, "path\\s*=\\s*\"([^\"]+)\"", MatchEvaluator(fun m ->
                            "path = " + (Path.GetFullPath(Path.Combine(originalDir, m.Groups[1].Value)) |> remap |> JsonSerializer.Serialize)))
                        let path = Path.Combine(destination, "Fidelity.Platform.fidproj")
                        File.WriteAllText(path, manifest)
                        path
                    let incompleteEnvironment = clonePackage environment (Path.Combine(incomplete, "environment")) id (replace "Triple = \"thumbv8m.main-none-eabi\"" "Triple = \"\"")
                    let incompleteProfile = clonePackage originalPlatform (Path.Combine(incomplete, "profile")) (fun path -> if path = environment then incompleteEnvironment else path) id
                    let incompleteProject = Path.Combine(directory, "MissingTriple.fidproj")
                    File.WriteAllText(incompleteProject, File.ReadAllText(project).Replace(JsonSerializer.Serialize platform, JsonSerializer.Serialize incompleteProfile))
                    File.Copy(Path.Combine(directory, "accepted-device-access.json"), ledger)
                    let missingTriple = Tests.Process.run compiler ["compile"; incompleteProject; "--emit-mlir"; "--no-color"] 600000 (Some(Path.Combine(directory, "missing-triple.log")))
                    if missingTriple.ExitCode = 0 || not (missingTriple.Output.Contains "requires a target triple") then failwith "Incomplete explicit profile accepted build-host fallback"
                    if missingTriple.Output.Contains "MLIR Generation" then failwith "Incomplete explicit profile reached lowering"
                    if File.Exists ledger then failwith "Incomplete explicit profile retained stale access evidence"
                    File.Copy(Path.Combine(directory, "accepted-device-access.json"), ledger)
                    File.WriteAllText(Path.Combine(directory, "Invalid.clef"), File.ReadAllText(Path.Combine(directory, "Main.clef")).Replace("Mmio.write16 handle 255", "Mmio.write16 handle 65536"))
                    let invalidProject = Path.Combine(directory, "Invalid.fidproj")
                    File.WriteAllText(invalidProject, File.ReadAllText(project).Replace("\"Main.clef\"", "\"Invalid.clef\""))
                    let failed = Tests.Process.run compiler ["compile"; invalidProject; "--no-color"] 600000 (Some(Path.Combine(directory, "invalidated.log")))
                    if failed.ExitCode = 0 || not (failed.Output.Contains "not proven within") then failwith "Invalid source recheck was accepted"
                    if File.Exists ledger then failwith "Failed recheck retained stale access evidence"
            printfn "PASS %s" name
            results.Add {| name = name; outcome = if expected = "" then "CCS evidence and optimized lowering" else "CCS rejection" |}
    File.WriteAllText(Path.Combine(work, "evidence.json"), JsonSerializer.Serialize(results, JsonSerializerOptions(WriteIndented = true)))
    printfn "%d device-access cases passed. Evidence: %s" results.Count work
    0
