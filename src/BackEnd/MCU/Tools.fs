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
let private home = Environment.GetFolderPath Environment.SpecialFolder.UserProfile
let private installed root file =
    if Directory.Exists root then Directory.GetFiles(root, file, SearchOption.AllDirectories) |> Array.sort |> Array.tryLast
    else None

let armToolDirectory configured =
    let selected = configured |> Option.orElseWith (fun () -> environment "COMPOSER_ARM_GNU_BIN")
    match selected with
    | Some path -> Path.GetFullPath path
    | None ->
        let onPath =
            (environment "PATH" |> Option.defaultValue "").Split(Path.PathSeparator)
            |> Array.tryFind (fun p -> File.Exists(Path.Combine(p, "arm-none-eabi-as")))
        onPath |> Option.orElseWith (fun () ->
            installed (Path.Combine(home, ".local/share/renesas/e2_studio/toolchains/gcc_arm")) "arm-none-eabi-as"
            |> Option.map Path.GetDirectoryName)
        |> Option.defaultWith (fun () -> failwith "Install ARM GNU binutils or set COMPOSER_ARM_GNU_BIN")

let probeLibrary configured =
    configured |> Option.orElseWith (fun () -> environment "COMPOSER_JLINK_LIBRARY")
    |> Option.orElseWith (fun () -> installed (Path.Combine(home, ".eclipse")) "libjlinkarm.so")
    |> Option.defaultWith (fun () -> failwith "Set COMPOSER_JLINK_LIBRARY to the installed SEGGER J-Link shared library")
