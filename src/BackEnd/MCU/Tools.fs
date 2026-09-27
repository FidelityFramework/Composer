module BackEnd.MCU.Tools

open System
open System.IO
open System.Diagnostics
open System.Security.Cryptography
open System.Text.Json

let sha256 bytes = SHA256.HashData(bytes: byte array) |> Convert.ToHexStringLower
let writeJson path value = File.WriteAllText(path, JsonSerializer.Serialize(value, JsonSerializerOptions(WriteIndented = true)) + "\n")

/// ArgumentList goes directly to the executable. No shell, script interpreter,
/// command substitutions, or per-application build hooks.
let run tool (args: string list) log =
    printfn "MCU: %s %s" tool (String.concat " " args)
    let start = ProcessStartInfo(tool, UseShellExecute = false, RedirectStandardOutput = true, RedirectStandardError = true)
    for arg in args do start.ArgumentList.Add arg
    use child = new Process(StartInfo = start)
    if not (child.Start()) then failwith ("Could not start " + tool)
    let stdout, stderr = child.StandardOutput.ReadToEndAsync(), child.StandardError.ReadToEndAsync()
    child.WaitForExit()
    let output = stdout.GetAwaiter().GetResult() + stderr.GetAwaiter().GetResult()
    log |> Option.iter (fun path -> File.WriteAllText(path, output))
    if child.ExitCode <> 0 then failwithf "%s failed (%d): %s" tool child.ExitCode output
    output

let private environment name = Environment.GetEnvironmentVariable name |> Option.ofObj |> Option.filter (String.IsNullOrWhiteSpace >> not)

/// The binutils directory is an explicit selection (embedded.tool_directory or
/// COMPOSER_ARM_GNU_BIN) or the one on PATH. No vendor install tree is scanned
/// for whichever version happens to sort last.
let armToolDirectory configured =
    let selected = configured |> Option.orElseWith (fun () -> environment "COMPOSER_ARM_GNU_BIN")
    match selected with
    | Some path -> Path.GetFullPath path
    | None ->
        (environment "PATH" |> Option.defaultValue "").Split(Path.PathSeparator)
        |> Array.tryFind (fun p -> File.Exists(Path.Combine(p, "arm-none-eabi-as")))
        |> Option.defaultWith (fun () ->
            failwith "backend (MCU Cortex-M) did not receive an ARM GNU binutils selection: declare embedded.tool_directory, set COMPOSER_ARM_GNU_BIN, or put arm-none-eabi-as on PATH")

/// The J-Link library is an explicit selection (embedded.probe_library or
/// COMPOSER_JLINK_LIBRARY). No IDE install tree is scanned for one.
let probeLibrary configured =
    configured |> Option.orElseWith (fun () -> environment "COMPOSER_JLINK_LIBRARY")
    |> Option.defaultWith (fun () ->
        failwith "backend (MCU Cortex-M probe) did not receive a SEGGER J-Link library selection: declare embedded.probe_library or set COMPOSER_JLINK_LIBRARY")
