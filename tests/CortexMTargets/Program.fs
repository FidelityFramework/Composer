module CortexMTargetTests

open System
open System.IO
open System.Text.Json
open System.Buffers.Binary
open Clef.Compiler.Project
open Core.Types.Pipeline
open BackEnd.MCU

let root = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__, "../.."))
let work = Path.Combine(Path.GetTempPath(), "composer-cortexm-targets-" + Guid.NewGuid().ToString("N"))
let compiler = Path.Combine(root, "src/bin/Debug/net10.0/Composer")
let quote (text: string) = JsonSerializer.Serialize text
let require holds message = if not holds then failwith message
let mutable passed = 0
let succeeds name action = action (); passed <- passed + 1; printfn "PASS %s" name
let rejects name fragment action =
    let error = try action (); None with ex -> Some ex.Message
    require (error |> Option.exists (fun message -> message.Contains(fragment: string))) (sprintf "%s: expected '%s', got %A" name fragment error)
    passed <- passed + 1
    printfn "PASS %s" name

// These small synthetic declarations exercise the image contract independently
// of the board's much larger inventory. No fixture is installed as a platform.
let declarations m7 =
    let arch, triple, cpu, flash, ram, flashBase, flashBytes, ramBytes, vectors, vectorCount, vectorAlignment, part, idAddress, optionAddress =
        if m7 then "arm_cortex_m7", "thumbv7em-none-eabihf", "cortex-m7", "flash-bank1", "dtcm", 0x08000000L, 0x100000L, 0x20000L, "ArmV7MVectors", 166, 1024, "STM32H747XIH6", 0x5C001000L, 0x5200201CL
        else "arm_cortex_m33", "thumbv8m.main-none-eabi", "cortex-m33", "flash", "sram", 0L, 0x200000L, 0x80000L, "ArmV8MVectors", 32, 128, "R7FA6M5BH", 0x0100A150L, 0x0100A100L
    sprintf """namespace Fixture
open BAREWire.Platform
open BAREWire.Hardware
module Description =
    let core: TargetCore = {
        Os = "none"; Arch = "%s"; WordSizeBits = 32; Endianness = "little"; Runtime = "bare"
        Triple = "%s"; CpuModel = "%s"
        Widths = [| { Name = "Pointer"; Bits = 32 }; { Name = "Register"; Bits = 32 } |]
        Representations = [|
            { Name = "int32"; Capability = "native"; Family = "int"; Bits = 32; MinMagnitude = "-2147483648"; MaxMagnitude = "2147483647"; Boundary = "wrap" }
            { Name = "int64"; Capability = "native"; Family = "int"; Bits = 64; MinMagnitude = "-9223372036854775808"; MaxMagnitude = "9223372036854775807"; Boundary = "wrap" }
            { Name = "uint32"; Capability = "native"; Family = "uint"; Bits = 32; MinMagnitude = "0"; MaxMagnitude = "4294967295"; Boundary = "wrap" }
            { Name = "float64"; Capability = "native"; Family = "ieee"; Bits = 64; MinMagnitude = "2.2250738585072014e-308"; MaxMagnitude = "1.7976931348623157e308"; Boundary = "saturate" }
        |]
    }
    let flash: MemorySpace = {
        Name = "%s"; Kind = "flash"; Capacity = %d; Alignment = 4; Granularity = 1
        Growth = "fixed"; Access = "rx"; Base = Some %d; Notes = "fixture"; MapKind = ""; Since = ""; Until = ""
    }
    let ram: MemorySpace = {
        Name = "%s"; Kind = "sram"; Capacity = %d; Alignment = 8; Granularity = 1
        Growth = "fixed"; Access = "rw"; Base = Some 536870912; Notes = "fixture"; MapKind = ""; Since = ""; Until = ""
    }
    let vectors: StructDescriptor = {
        Name = "%s"; Documentation = None
        Layout = { Size = %d; Alignment = 4; Fields = [|
            { Name = "vectors"; Offset = 0; Repr = "u32"; Count = %d; Access = "ro"; BitFields = [||]; Documentation = None }
        |] }
    }
    let image: CortexMImageDescriptor = {
        FlashSpace = "%s"; RamSpace = "%s"; VectorLayout = "%s"; VectorAlignment = %d; StackBytes = 8192
        EntrySymbol = "Reset_Handler"; DebugDevice = "%s"; PartNumber = "%s"; PartNumberAddress = %d
        PreservedOptionAddress = %d; PreservedOptionBytes = 48
    }
    let descriptor: PlatformDescription = {
        Id = "fixture-%s"; DisplayName = "Build-only fixture"; Substrate = "mcu"; Core = Some core
        Spaces = [| flash; ram |]; ProgramLifetime = Some { Immutable = "%s"; Mutable = Some "%s" }
        Surfaces = [||]; Buffers = [||]; Transports = [||]; Notes = [||]; Limits = [||]
        Lifecycle = { Clocks = [||]; Resets = [||]; Entry = "Reset_Handler"; Teardown = ""; Persistence = "volatile" }
    }
        """ arch triple cpu flash flashBytes flashBase ram ramBytes vectors (vectorCount * 4) vectorCount flash ram vectors vectorAlignment part part idAddress optionAddress cpu flash ram

let startup m7 =
    let profile = if m7 then CortexMProfile.Stm32H747HardFloat else CortexMProfile.CortexM33SoftFloat
    let slots = [0 .. (if m7 then 165 else 31)] |> List.map (fun slot ->
        if slot = 0 then ".word __stack_top"
        elif slot = 1 then ".word Reset_Handler"
        elif List.contains slot (Target.reservedVectorSlots profile) then ".word 0"
        else ".word Fault_Handler") |> String.concat "\n"
    String.concat "\n" [
        ".syntax unified"; ".cpu " + Target.cpu profile; ".thumb"
        if m7 then ".fpu fpv5-d16\n.eabi_attribute 28, 1" else ""
        ".include \"layout.inc\""; ".section .vectors,\"a\",%progbits"; ".balign VECTOR_ALIGNMENT"; slots
        ".section .text.Reset_Handler,\"ax\",%progbits"; ".global Reset_Handler"; ".type Reset_Handler,%function"; ".thumb_func"; "Reset_Handler:"
        "cpsid i"
        if m7 then "ldr r0, =0xe000ed88\nldr r1, [r0]\norr r1, r1, #0x00f00000\nstr r1, [r0]\ndsb\nisb" else ""
        "sub sp, sp, #8\nmovs r0, #1\nstr r0, [sp]\nmovs r0, #0\nmovs r1, #0\nmovs r2, #0\nmovs r3, #0\nbl main\nadd sp, sp, #8"
        "b Fault_Handler"; ".size Reset_Handler, .-Reset_Handler"
        ".section .text.Fault_Handler,\"ax\",%progbits"; ".global Fault_Handler"; ".type Fault_Handler,%function"; ".thumb_func"; "Fault_Handler: b Fault_Handler"
    ]

let create m7 name (transform: string -> string) extra =
    let directory = Path.Combine(work, name)
    Directory.CreateDirectory directory |> ignore
    let arch = if m7 then "arm_cortex_m7" else "arm_cortex_m33"
    File.WriteAllText(Path.Combine(directory, "Description.clef"), declarations m7 |> transform)
    let platform = Path.Combine(directory, "Platform.fidproj")
    File.WriteAllText(platform, String.concat "\n" [
        "[package]\nname=\"Fixture.Platform\"\n[compilation]\ntarget=\"mcu\""
        "[platform]\nruntime_model=\"bare\"\nos=\"none\"\narch=" + quote arch + "\ndescription=\"Fixture.Description.descriptor\""
        "[dependencies]\nmetadata={path=" + quote (Path.GetFullPath(Path.Combine(root, "../BAREWire/src/BAREWire.PlatformMetadata.fidproj"))) + "}"
        "[build]\nsources=[\"Description.clef\"]\noutput_kind=\"library\""
    ])
    File.WriteAllText(Path.Combine(directory, "startup.S"), startup m7)
    File.WriteAllText(Path.Combine(directory, "Main.clef"),
        "module Fixture.Main\n[<EntryPoint>]\nlet main _ =\n    let value = Mmio.read32 (Mmio.reg32 0x40000000)\n    " +
        (if m7 then "if float value * 1.5 > 10.0 then 1 else 0\n" else "if value > 10 then 1 else 0\n"))
    let app = Path.Combine(directory, "Test.fidproj")
    File.WriteAllText(app, String.concat "\n" [
        "[package]\nname=\"CortexMFixture\"\n[compilation]\ntarget=\"mcu\""
        "[platform]\nruntime_model=\"bare\"\nos=\"none\"\narch=" + quote arch
        "[dependencies]\nplatform={path=" + quote platform + "}"
        "[build]\nsources=[\"Main.clef\"]\noutput=\"test.elf\"\noutput_kind=\"embedded\""
        "[embedded]\nstartup=\"startup.S\"\nentry_abi=\"clef-empty-string-array32\"\nrecovery=\"recovery\""
        "[embedded.vector_handlers]\n\"1\"=\"Reset_Handler\"\n" + extra
    ])
    app

let checkedTarget project =
    let checkedProject = ProjectChecker.checkProject project |> function Ok p -> p | Error e -> failwith e
    if ProjectChecker.hasErrors checkedProject then failwith (ProjectChecker.getErrorMessages checkedProject |> String.concat "\n")
    Target.resolve project checkedProject.CheckResult.Graph

let axiRam (text: string) =
    text.Replace("\"dtcm\"", "\"axi-sram\"")
        .Replace("Capacity = 131072", "Capacity = 524288")
        .Replace("Base = Some 536870912", "Base = Some 603979776")

// The relocation from reset retains this BSS reservation through --gc-sections.
// Its size matches a packed 416x416 RGB565 logo; no allocator is provided.
let reservePixels (project: string) (bytes: int) =
    let path = Path.Combine(Path.GetDirectoryName project, "startup.S")
    let source = File.ReadAllText path
    File.WriteAllText(path,
        source.Replace("cpsid i", "cpsid i\nldr r4, =RAM_ORIGIN\nldr r5, =fixture_pixels") +
        sprintf "\n.section .bss.fixture_pixels,\"aw\",%%nobits\n.balign 8\n.global fixture_pixels\nfixture_pixels:\n.space %d\n" bytes)

[<EntryPoint>]
let main _ =
    Directory.CreateDirectory work |> ignore
    printfn "Evidence: %s" work
    let m33, m7 = create false "m33" id "", create true "m7" id ""
    let axi = create true "m7_axi" axiRam ""
    reservePixels axi (416 * 416 * 2)
    for name, project in ["M33", m33; "M7 DTCM", m7; "M7 AXI SRAM", axi] do
        succeeds (name + " complete native image") (fun () ->
            Tests.Process.requireSuccess compiler ["compile"; project; "-k"; "--no-color"] 600000 (Some(Path.Combine(Path.GetDirectoryName project, "compile.log"))) |> ignore)
    succeeds "layout.inc preserves the selected RAM origin for every profile" (fun () ->
        for project, expected in [m33, 0x20000000L; m7, 0x20000000L; axi, 0x24000000L] do
            let path = Path.Combine(Path.GetDirectoryName project, "targets/boot/layout.inc")
            require (File.ReadAllText(path).Contains(sprintf ".equ RAM_ORIGIN, 0x%X\n" expected)) ("Wrong assembly RAM origin: " + path))
    succeeds "AXI SRAM contains the packed logo and reserved stack" (fun () ->
        let directory = Path.Combine(Path.GetDirectoryName axi, "targets")
        let symbols = Image.symbols(File.ReadAllText(Path.Combine(directory, "test.symbols.txt")))
        require (symbols.["RAM_ORIGIN"] = 0x24000000u) "Startup did not receive the AXI SRAM origin"
        require (symbols.["__stack_top"] = 0x24080000u && symbols.["__stack_bottom"] = 0x2407E000u) "AXI SRAM stack placement changed"
        require (symbols.["fixture_pixels"] >= 0x24000000u && symbols.["fixture_pixels"] + 346112u <= symbols.["__stack_bottom"]) "Packed logo overlaps the AXI SRAM stack")
    let target = checkedTarget m7
    let directory = Path.Combine(Path.GetDirectoryName m7, "targets")
    let elf = File.ReadAllBytes(Path.Combine(directory, "test.elf"))
    let binary = File.ReadAllBytes(Path.Combine(directory, "test.bin"))
    let symbols = Image.symbols(File.ReadAllText(Path.Combine(directory, "test.symbols.txt")))
    succeeds "M7 FP code is hardware double precision" (fun () ->
        let text = File.ReadAllText(Path.Combine(directory, "test.disassembly.txt"))
        require (text.Contains "vmul.f64") "FP64 multiply disappeared or became a software call")
    rejects "M7 reserved slot 7" "Reserved vector 7" (fun () ->
        let changed = Array.copy binary
        BinaryPrimitives.WriteUInt32LittleEndian(changed.AsSpan(28, 4), 0x08000001u)
        Image.verify target elf changed symbols)
    rejects "M7 hard-float ELF identity" "hard-float" (fun () ->
        let changed = Array.copy elf
        BinaryPrimitives.WriteUInt32LittleEndian(changed.AsSpan(36, 4), 0x05000200u)
        Image.verify target changed binary symbols)
    rejects "M7 probe forbidden before library/recovery access" "image builds only" (fun () -> Probe.deploy target "/missing/firmware.elf")
    let negative name m7 fragment (before: string) (after: string) =
        rejects name fragment (fun () -> create m7 name (fun text -> text.Replace(before, after)) "" |> checkedTarget |> ignore)
    negative "m7_wrong_triple" true "triple architecture" "thumbv7em-none-eabihf" "thumbv8m.main-none-eabi"
    negative "m7_soft_float" true "target triple OS" "thumbv7em-none-eabihf" "thumbv7em-none-eabi"
    negative "m7_wrong_cpu" true "requires the declared" "cortex-m7" "cortex-m4"
    negative "m7_wrong_pointer" true "32-bit Pointer" "Name = \"Pointer\"; Bits = 32" "Name = \"Pointer\"; Bits = 64"
    negative "m7_flash_zero" true "flash-bank1" "Base = Some 134217728" "Base = Some 0"
    negative "m7_wrong_ram" true "128 KiB DTCM" "Base = Some 536870912" "Base = Some 603979776"
    for name, before, after in [
        "m7_axi_wrong_origin", "Base = Some 603979776", "Base = Some 536870912"
        "m7_axi_wrong_capacity", "Capacity = 524288", "Capacity = 131072"
        "m7_axi_wrong_name", "\"axi-sram\"", "\"other-sram\""
    ] do
        rejects name "512 KiB AXI SRAM" (fun () ->
            create true name (fun text -> (axiRam text).Replace(before, after)) "" |> checkedTarget |> ignore)
    negative "m7_wrong_identity" true "identity declaration" "PartNumberAddress = 1543507968" "PartNumberAddress = 1543507972"
    rejects "m7_wrong_vector_count" "166 words" (fun () ->
        create true "m7_wrong_vector_count" (fun text -> text.Replace("Size = 664", "Size = 660").Replace("Count = 166", "Count = 165")) "" |> checkedTarget |> ignore)
    negative "m33_flash_nonzero" false "flash at zero" "Base = Some 0" "Base = Some 134217728"
    rejects "m7 reserved handler declaration" "Invalid/reserved vector slot" (fun () ->
        create true "m7_reserved_handler" id "\"7\"=\"Fault_Handler\"" |> checkedTarget |> ignore)
    succeeds "AXI SRAM data/stack collision is rejected by the native linker" (fun () ->
        let oversized = create true "m7_axi_collision" axiRam ""
        reservePixels oversized 524288
        let result = Tests.Process.run compiler ["compile"; oversized; "-k"; "--no-color"] 600000 (Some(Path.Combine(Path.GetDirectoryName oversized, "compile.log")))
        require (result.ExitCode <> 0 && result.Output.Contains "Data/stack collision") "Oversized AXI SRAM BSS was not rejected by the linker")
    // A failed recheck cannot retain an earlier accepted image's evidence.
    let evidence = Path.Combine(directory, "test.build-evidence.json")
    let context: BackEndContext = {
        OutputPath = Path.Combine(directory, "test.elf"); IntermediatesDir = None
        TargetTripleOverride = Some "thumbv8m.main-none-eabi"; TargetPointerBits = Some 32; TargetCpu = Some "cortex-m7"
        DeploymentMode = Core.Types.Dialects.DeploymentMode.Embedded; EmitIntermediateOnly = false
        ExternLibraries = Set.empty; NativeLink = NativeLinkOptions.Empty; EmbeddedTarget = Some target; XtensaTarget = None; Deploy = false
    }
    rejects "M7 image context disagreement" "declared triple" (fun () -> Image.build "/missing/input.ll" context target |> ignore)
    succeeds "rejected image invalidates evidence" (fun () -> require (not (File.Exists evidence)) "Stale accepted build evidence survived")
    let badAbiDirectory = Path.Combine(work, "m7_startup_abi")
    Directory.CreateDirectory badAbiDirectory |> ignore
    let badStartup = Path.Combine(badAbiDirectory, "startup.S")
    File.WriteAllText(badStartup, (startup true).Replace(".eabi_attribute 28, 1", ".eabi_attribute 28, 0"))
    let badAbiContext = { context with OutputPath = Path.Combine(badAbiDirectory, "test.elf"); TargetTripleOverride = Some "thumbv7em-none-eabihf" }
    rejects "M7 startup soft-float ABI rejected before link" "VFP argument registers" (fun () ->
        Image.build (Path.Combine(directory, "intermediates/08_output.ll")) badAbiContext { target with StartupSource = badStartup } |> ignore)
    printfn "%d Cortex-M target checks passed; no probe opened and no firmware executed" passed
    0
