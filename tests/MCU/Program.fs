module McuTests

open System
open System.IO
open System.Buffers.Binary
open BackEnd.MCU

[<EntryPoint>]
let main args =
    let project = Path.GetFullPath args.[0]
    let checkedProject = FrontEnd.ProjectLoader.load project |> function Ok p -> p | Error e -> failwith e
    let target = Target.resolve project checkedProject.CheckResult.Graph
    let elfPath = Path.Combine(checkedProject.Options.ProjectDirectory, "targets", checkedProject.Options.OutputName.Value)
    let elf, image = File.ReadAllBytes elfPath, File.ReadAllBytes(Path.ChangeExtension(elfPath, "bin"))
    let table = File.ReadAllText(Path.ChangeExtension(elfPath, "symbols.txt")) |> Image.symbols
    let mutable passed = 0
    let succeeds name f = f (); passed <- passed + 1; printfn "PASS %s" name
    let rejects name fragment f =
        let failure = try f (); None with ex -> Some ex.Message
        match failure with
        | Some message when message.Contains(fragment: string) -> passed <- passed + 1; printfn "PASS %s" name
        | _ -> failwithf "%s expected '%s', got %A" name fragment failure
    let changed offset value =
        let bytes = Array.copy image
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset,4),value)
        bytes
    succeeds "valid real image" (fun () -> Image.verify target elf image table)
    rejects "truncated vectors" "extent" (fun () -> Image.verify target elf image.[0..31] table)
    rejects "flash overflow" "extent" (fun () -> Image.verify {target with Flash = { target.Flash with Capacity = 16L }} elf image table)
    rejects "wrong MSP" "MSP" (fun () -> Image.verify target elf (changed 0 0x2007fffcu) table)
    rejects "reserved vector populated" "Reserved vector" (fun () -> Image.verify target elf (changed 32 1u) table)
    rejects "handler mismatch" "Wrong vector" (fun () -> Image.verify target elf (changed 60 1u) table)
    rejects "non-Thumb vector" "Invalid Thumb" (fun () -> Image.verify target elf (changed 8 512u) table)
    rejects "vector outside image" "Invalid Thumb" (fun () -> Image.verify target elf (changed 8 0x10000001u) table)
    rejects "missing vector symbol" "Missing vector handler" (fun () -> Image.verify target elf image Map.empty)
    let badElf = Array.copy elf
    badElf.[18] <- 62uy
    rejects "wrong ELF machine" "ELF32 ARM" (fun () -> Image.verify target badElf image table)
    let wrongEntry = Array.copy elf
    BinaryPrimitives.WriteUInt32LittleEndian(wrongEntry.AsSpan(24,4),1u)
    rejects "ELF entry mismatch" "entry and reset" (fun () -> Image.verify target wrongEntry image table)
    // The linker must independently reject collisions even if the RAM profile
    // grows without a corresponding review of startup storage requirements.
    let temporary = Path.Combine(Path.GetTempPath(), "Composer MCU checks " + Guid.NewGuid().ToString("N"))
    Directory.CreateDirectory temporary |> ignore
    try
        let tiny = { target with Image = { target.Image with StackBytes = int target.Ram.Capacity - 8 } }
        Layout.generate tiny temporary
        rejects "linker data/stack collision" "Data/stack collision" (fun () ->
            Tools.run "ld.lld" ["--static";"--entry="+target.Image.EntrySymbol;"-T";Path.Combine(temporary,"memory.ld");
                               Path.Combine(Path.GetDirectoryName(elfPath),"boot/startup.o");Path.ChangeExtension(elfPath,"o");
                               "-o";Path.Combine(temporary,"must-not-deploy.elf")] None |> ignore)
        let unsupported = Path.Combine(temporary, "heap.ll")
        File.WriteAllText(unsupported, String.concat "\n" [
            "target triple = \"thumbv8m.main-none-eabi\""
            "declare ptr @malloc(i32)"
            "define i32 @main(ptr %allocated, ptr %aligned, i32 %offset, i32 %length, i32 %stride) nounwind {"
            "  %heap = call ptr @malloc(i32 16)"
            "  %address = ptrtoint ptr %heap to i32"
            "  ret i32 %address"
            "}" ])
        let output = Path.Combine(temporary,"unsupported.elf")
        let ctx: Core.Types.Pipeline.BackEndContext = {
            OutputPath = output; IntermediatesDir = None
            TargetTripleOverride = Some "thumbv8m.main-none-eabi"; TargetPointerBits = Some 32; TargetCpu = Some "cortex-m33"
            DeploymentMode = Core.Types.Dialects.DeploymentMode.Embedded; EmitIntermediateOnly = false
            ExternLibraries = Set.empty; NativeLink = Core.Types.Pipeline.NativeLinkOptions.Empty
            EmbeddedTarget = Some target; XtensaTarget = None; Deploy = false
        }
        let evidence = Path.ChangeExtension(output,"build-evidence.json")
        File.WriteAllText(evidence,"old build must be invalidated")
        rejects "allocator import cannot enter firmware" "undefined symbol: malloc" (fun () -> Image.build unsupported ctx target |> ignore)
        succeeds "failed image has no deployment evidence" (fun () -> if File.Exists evidence then failwith "Stale evidence survived failure")
    finally Directory.Delete(temporary,true)
    printfn "%d MCU checks passed; no probe opened or firmware downloaded" passed
    0
