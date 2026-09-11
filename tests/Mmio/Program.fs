module MmioTests

open System
open System.IO
open System.Text.Json
open System.Text.RegularExpressions
open System.Threading.Tasks

let cases = [|
    "width_mismatch", "Mmio.write16 (Mmio.reg8 0x40080D03) 1", "expected 'Mmio16', got 'Mmio8'"
    "misaligned", "let _ = Mmio.read32 (Mmio.reg32 0x40080819)\n    ()", "misaligned"
    "null", "let _ = Mmio.read8 (Mmio.reg8 0)\n    ()", "null"
    "address_overflow", "let _ = Mmio.read8 (Mmio.reg8 4294967296)\n    ()", "outside the platform address space"
    "negative_write", "Mmio.write8 (Mmio.reg8 0x40080D03) -1", "not proven within"
    "oversized_write", "Mmio.write8 (Mmio.reg8 0x40080D03) 256", "not proven within"
    "dynamic_address", "let address = Mmio.read32 (Mmio.reg32 0x40080818)\n    let _ = Mmio.read8 (Mmio.reg8 address)\n    ()", "statically declared"
    "positive", String.concat "\n    " [
        "let r8 = Mmio.reg8 0x40080D03"
        "let r16 = Mmio.reg16 0x4008081A"
        "let r32 = Mmio.reg32 0x40080818"
        "Mmio.write8 r8 255"
        "Mmio.write16 r16 65535"
        "Mmio.write32 r32 4294967295"
        "let _ = Mmio.read8 r8"
        "let _ = Mmio.read16 r16"
        "let _ = Mmio.read32 r32"
        "let mutable attempts = 0"
        "while attempts < 4 && Mmio.read8 r8 = 0 do"
        "    attempts <- attempts + 1" ], ""
|]

let check compiler platform work (name, body, expected) =
    let directory = Path.Combine(work, name)
    Directory.CreateDirectory directory |> ignore
    File.WriteAllText(Path.Combine(directory,"Main.clef"), "module MmioCheck\n[<EntryPoint>]\nlet main _ =\n    " + body + "\n    0\n")
    let project = Path.Combine(directory,"Test.fidproj")
    File.WriteAllText(project, String.concat "\n" [
        "[package]"; "name=\"MmioCheck\""; "version=\"0.1.0\""
        "[compilation]"; "target=\"mcu\""
        "[platform]"; "runtime_model=\"bare\""; "os=\"none\""; "arch=\"arm_cortex_m33\""
        "[dependencies]"; "platform={path=" + JsonSerializer.Serialize(platform: string) + "}"
        "[build]"; "sources=[\"Main.clef\"]"; "output=\"test.elf\""; "output_kind=\"embedded\"" ])
    let result = Tests.Process.run compiler ["compile";project;"--emit-llvm";"-k";"--no-color"] 600000 (Some (Path.Combine(directory,"compile.log")))
    if expected <> "" then
        if result.ExitCode = 0 then failwith (name + ": invalid program was accepted")
        if not (result.Output.Contains expected) then failwith (name + ": wrong rejection; see " + directory)
    else
        if result.ExitCode <> 0 then failwith ("Valid MMIO program failed; see " + directory)
        let ir, optimized = Path.Combine(directory,"targets/intermediates/08_output.ll"), Path.Combine(directory,"optimized.ll")
        Tests.Process.requireSuccess "opt" ["-S";"-passes=default<O2>";ir;"-o";optimized] 60000 None |> ignore
        let text = File.ReadAllText optimized
        if not (text.Contains "target triple = \"thumbv8m.main-none-eabi\"") then failwith "Missing MCU triple"
        if not (Regex.IsMatch(text, @"@main\([^\n]+i32[^\n]+i32[^\n]+i32")) then failwith "Entry descriptor is not 32-bit"
        for bits in [8;16;32] do
            if not (Regex.IsMatch(text, sprintf @"load volatile i%d\b" bits)) then failwithf "Discarded read%d was removed" bits
            if not (Regex.IsMatch(text, sprintf @"store volatile i%d\b" bits)) then failwithf "write%d was removed" bits
        if Regex.Matches(text, @"load volatile i8\b").Count < 2 then failwith "Repeated volatile polling vanished"
        let assembly = Path.Combine(directory,"optimized.s")
        Tests.Process.requireSuccess "llc" ["-mtriple=thumbv8m.main-none-eabi";"-mcpu=cortex-m33";"-float-abi=soft";"-O=2";optimized;"-o";assembly] 60000 None |> ignore
        let asm = File.ReadAllText assembly
        for op in ["ldrb";"ldrh";"ldr";"strb";"strh";"str"] do
            if not (Regex.IsMatch(asm, @"\b" + op + @"(?:\.w)?\s")) then failwith ("No " + op + " in ARM output")
    printfn "PASS %s" name
    {| name = name; result = if expected = "" then "optimized widths and polling preserved" else "rejected with expected diagnostic" |}

[<EntryPoint>]
let main args =
    let root = Path.GetFullPath(Path.Combine(__SOURCE_DIRECTORY__,"../.."))
    let compiler = if args.Length > 0 then Path.GetFullPath args.[0] else Path.Combine(root,"src/bin/Debug/net10.0/Composer")
    let platform = Path.GetFullPath(Path.Combine(root,"../Fidelity.Platform/Profiles/EK_RA6M5_HelloBlinky/Fidelity.Platform.fidproj"))
    let work = Path.Combine(Path.GetTempPath(),"composer-mmio-fsharp-" + Guid.NewGuid().ToString("N"))
    let results = Array.zeroCreate cases.Length
    Parallel.For(0, cases.Length, ParallelOptions(MaxDegreeOfParallelism = 2), fun i -> results.[i] <- check compiler platform work cases.[i]) |> ignore
    File.WriteAllText(Path.Combine(work,"evidence.json"), JsonSerializer.Serialize({| runner = "F# / .NET"; work = work; cases = results |},JsonSerializerOptions(WriteIndented = true)))
    printfn "%d MMIO cases passed; generated ARM code was not executed on the host. Evidence: %s" results.Length work
    0
