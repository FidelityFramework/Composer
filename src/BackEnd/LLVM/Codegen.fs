/// LLVM IR -> target bitcode -> LLD's LLVM code generation and ELF linking.
/// Runtime objects are explicit link inputs, independent of any C compiler driver.
module BackEnd.LLVM.Codegen

open System
open System.IO
open System.Diagnostics
open System.Runtime.InteropServices
open Core.Types.Dialects
open Core.Types.Pipeline

let getDefaultTarget() =
    let arch =
        match RuntimeInformation.ProcessArchitecture with
        | Architecture.X64 -> "x86_64"
        | Architecture.Arm64 -> "aarch64"
        | Architecture.X86 -> "i386"
        | Architecture.Arm -> "arm"
        | other -> other.ToString().ToLowerInvariant()
    if RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then arch + "-unknown-linux-gnu"
    elif RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then arch + "-pc-windows-gnu"
    else arch + "-apple-darwin"

let private run tool (arguments: string list) =
    let start = ProcessStartInfo(tool)
    start.UseShellExecute <- false
    start.RedirectStandardOutput <- true
    start.RedirectStandardError <- true
    for argument in arguments do start.ArgumentList.Add argument
    use toolProcess = new Process(StartInfo = start)
    if not (toolProcess.Start()) then Error (sprintf "Could not start %s" tool)
    else
        let output = toolProcess.StandardOutput.ReadToEndAsync()
        let errors = toolProcess.StandardError.ReadToEndAsync()
        toolProcess.WaitForExit()
        let stdout, stderr = output.GetAwaiter().GetResult(), errors.GetAwaiter().GetResult()
        if toolProcess.ExitCode = 0 then Ok ()
        else Error (sprintf "%s failed (%d): %s%s" tool toolProcess.ExitCode stderr stdout)

let private elfTarget (triple: string) =
    not (["windows"; "mingw"; "darwin"; "apple"; "wasm"] |> List.exists triple.Contains)

/// Resolve only the selected target's runtime. Native Linux has a convenience
/// profile for its installed libc; a cross runtime needs a sysroot or explicit inputs.
let private linkArguments target mode libraries (options: NativeLinkOptions) bitcode output =
    let root = options.Sysroot |> Option.map Path.GetFullPath
    let onTarget (path: string) =
        match root with
        | Some basePath -> Path.Combine(basePath, path.TrimStart('/'))
        | None -> path
    let nativeLinux = target = getDefaultTarget() && RuntimeInformation.IsOSPlatform(OSPlatform.Linux)
    let linux = target.Contains("linux")
    let multiarch =
        let i = target.IndexOf("linux", StringComparison.Ordinal)
        if i < 0 then target else target.Split('-')[0] + "-" + target.Substring(i)
    let runtimeDirectories =
        if linux && (nativeLinux || root.IsSome) then
            ["/usr/lib/" + multiarch; "/lib/" + multiarch; "/usr/lib64"; "/usr/lib"; "/lib64"; "/lib"]
            |> List.map onTarget |> List.filter Directory.Exists
        else []
    let directories = (options.LibraryPaths |> List.map Path.GetFullPath) @ runtimeDirectories |> List.distinct
    let findFile name = directories |> List.tryPick (fun directory ->
        let path = Path.Combine(directory, name)
        if File.Exists path then Some path else None)
    let needFile name =
        findFile name |> Option.defaultWith (fun () ->
            failwithf "Target runtime file %s is missing. Supply --sysroot or explicit --link-start-file/--link-end-file and --link-library-path inputs." name)
    let startFiles, endFiles, loader =
        match mode with
        | Console ->
            if not linux then failwith "Console startup discovery currently supports Linux ELF; provide another deployment mode for a bare-metal target."
            if not nativeLinux && root.IsNone && options.StartFiles.IsEmpty then
                failwith "A cross-target console link requires --sysroot or explicit startup objects; host runtime files are not used."
            let starts =
                if options.StartFiles.IsEmpty then [needFile "crt1.o"; needFile "crti.o"]
                else options.StartFiles |> List.map Path.GetFullPath
            let ends =
                if options.StartFiles.IsEmpty && options.EndFiles.IsEmpty then [needFile "crtn.o"]
                else options.EndFiles |> List.map Path.GetFullPath
            let loader =
                match options.DynamicLinker with
                | Some path -> path
                | None ->
                    let arch = target.Split('-')[0]
                    let name =
                        if target.Contains("musl") then "ld-musl-" + arch + ".so.1"
                        else
                            match arch with
                            | "x86_64" -> "ld-linux-x86-64.so.2"
                            | "aarch64" -> "ld-linux-aarch64.so.1"
                            | "i386" | "i686" -> "ld-linux.so.2"
                            | "riscv64" -> "ld-linux-riscv64-lp64d.so.1"
                            | "arm" | "armv7" when target.EndsWith("hf") -> "ld-linux-armhf.so.3"
                            | "arm" | "armv7" -> "ld-linux.so.3"
                            | _ -> failwith "Specify --dynamic-linker for this target runtime."
                    let path = needFile name
                    match root with
                    | Some basePath -> "/" + Path.GetRelativePath(basePath, path).Replace('\\', '/')
                    | None -> path
            starts, ends, Some loader
        | _ -> options.StartFiles |> List.map Path.GetFullPath, options.EndFiles |> List.map Path.GetFullPath, None
    for path in startFiles @ endFiles @ (options.LinkerScript |> Option.toList) do
        if not (File.Exists path) then failwithf "Target link input does not exist: %s" path
    let modeArguments =
        match mode with
        | Console -> ["--no-pie"; "--export-dynamic"; "--entry=_start"; "--dynamic-linker=" + loader.Value]
        | Freestanding | Embedded -> ["--static"; "--entry=_start"]
        | Library -> ["--shared"]
    let libraries = if mode = Console then Set.add "c" libraries else libraries
    ["--lto-O0"; "--lto-CGO0"; "--fatal-warnings"; "-o"; Path.GetFullPath output]
    @ (root |> Option.map (fun path -> "--sysroot=" + path) |> Option.toList)
    @ (if nativeLinux && root.IsNone then ["--plugin-opt=mcpu=native"] else [])
    @ modeArguments
    @ (options.LinkerScript |> Option.map (fun path -> "--script=" + Path.GetFullPath path) |> Option.toList)
    @ (directories |> List.map (fun path -> "-L" + path))
    @ startFiles @ [bitcode]
    @ (libraries |> Set.toList |> List.map (fun name -> "-l" + name))
    @ endFiles

let compileToNative
    (llvmPath: string)
    (outputPath: string)
    (targetTriple: string)
    (deploymentMode: DeploymentMode)
    (externLibraries: Set<string>)
    (linkOptions: NativeLinkOptions) : Result<unit, string> =
    try
        if not (elfTarget targetTriple) then
            Error (sprintf "The direct LLVM backend currently emits ELF. Target %s requires a separate LLD PE/COFF, Mach-O or Wasm link profile." targetTriple)
        else
            let llvmPath = Path.GetFullPath llvmPath
            let declaredTarget =
                File.ReadLines llvmPath
                |> Seq.tryPick (fun line ->
                    if line.TrimStart().StartsWith("target triple", StringComparison.Ordinal) then
                        line.Split('"') |> Array.tryItem 1
                    else None)
            match declaredTarget with
            | Some target when target <> targetTriple ->
                failwithf "LLVM IR declares target %s, but the backend selected %s. Regenerate the IR for the selected target." target targetTriple
            | _ -> ()
            let bitcodePath = Path.ChangeExtension(llvmPath, ".bc")
            let arguments = linkArguments targetTriple deploymentMode externLibraries linkOptions bitcodePath outputPath
            // TargetMachine supplies missing DataLayout from the selected triple.
            // This verifies and serializes IR without running an optimization pipeline.
            // Keep the bitcode beside retained LLVM IR for inspecting the exact LLD input.
            run "opt" ["-mtriple=" + targetTriple; "-passes=no-op-module"; llvmPath; "-o"; bitcodePath]
            |> Result.bind (fun () -> run "ld.lld" arguments)
    with ex -> Error (sprintf "Native compilation failed: %s" ex.Message)
