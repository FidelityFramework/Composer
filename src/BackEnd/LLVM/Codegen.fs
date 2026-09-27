/// LLVM IR -> target bitcode -> LLD's LLVM code generation and ELF linking.
/// Runtime objects are explicit link inputs, independent of any C compiler driver.
module BackEnd.LLVM.Codegen

open System
open System.IO
open System.Diagnostics
open System.Runtime.InteropServices
open Core.Types.Dialects
open Core.Types.Pipeline

/// The build host's triple, when its architecture and OS have an established
/// spelling. Used only to recognise that the selected target IS the host (so
/// host runtime files may be admitted); never a compilation default.
let private hostTriple () =
    let arch =
        match RuntimeInformation.ProcessArchitecture with
        | Architecture.X64 -> Some "x86_64"
        | Architecture.Arm64 -> Some "aarch64"
        | Architecture.X86 -> Some "i386"
        | Architecture.Arm -> Some "arm"
        | _ -> None
    arch |> Option.bind (fun arch ->
        if RuntimeInformation.IsOSPlatform(OSPlatform.Linux) then Some (arch + "-unknown-linux-gnu")
        elif RuntimeInformation.IsOSPlatform(OSPlatform.Windows) then Some (arch + "-pc-windows-gnu")
        elif RuntimeInformation.IsOSPlatform(OSPlatform.OSX) then Some (arch + "-apple-darwin")
        else None)

/// The build host's triple for host-native test harnesses. An unrecognised
/// host is an error, not a guessed spelling.
let getDefaultTarget() =
    hostTriple () |> Option.defaultWith (fun () ->
        failwithf "backend (LLVM) has no established triple for build host %O on %s"
            RuntimeInformation.ProcessArchitecture RuntimeInformation.OSDescription)

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
    let nativeLinux = RuntimeInformation.IsOSPlatform(OSPlatform.Linux) && hostTriple () = Some target
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
    // No host-CPU default: the CPU is the declared core's CpuModel (passed by
    // the caller), and a core that declares none gets the triple's generic CPU.
    @ modeArguments
    @ (options.LinkerScript |> Option.map (fun path -> "--script=" + Path.GetFullPath path) |> Option.toList)
    @ (directories |> List.map (fun path -> "-L" + path))
    @ startFiles @ [bitcode]
    @ (libraries |> Set.toList |> List.map (fun name -> "-l" + name))
    @ endFiles

let compileToNativeWithArtifacts
    (llvmPath: string)
    (outputPath: string)
    (targetTriple: string)
    (deploymentMode: DeploymentMode)
    (externLibraries: Set<string>)
    (linkOptions: NativeLinkOptions)
    (cpu: string option)
    (storage: (string * Clef.Compiler.PSGSaturation.SemanticGraph.Types.ProgramStorageEntry) list)
    (pool: Clef.Compiler.PSGSaturation.SemanticGraph.Types.StaticStringPool option) : Result<unit, string> =
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
            let plans =
                match StorageCommitment.plan targetTriple storage with
                | Ok plans -> plans
                | Error reason -> failwith reason
            let linkOptions =
                if linkOptions.LinkerScript.IsSome then linkOptions else
                { linkOptions with LinkerScript = StorageCommitment.linkerScript (Path.ChangeExtension(llvmPath, ".storage.ld")) plans }
            let bitcodePath = Path.ChangeExtension(llvmPath, ".bc")
            let arguments = linkArguments targetTriple deploymentMode externLibraries linkOptions bitcodePath outputPath @ (cpu |> Option.map (fun c -> ["--plugin-opt=mcpu=" + c]) |> Option.defaultValue [])
            // TargetMachine supplies missing DataLayout from the selected triple.
            // This verifies and serializes IR without running an optimization pipeline.
            // Keep the bitcode beside retained LLVM IR for inspecting the exact LLD input.
            run "opt" ["-mtriple=" + targetTriple; "-passes=no-op-module"; llvmPath; "-o"; bitcodePath]
            |> Result.bind (fun () -> StorageCommitment.realizeBitcodeWithPool bitcodePath plans pool)
            |> Result.bind (fun () -> run "ld.lld" arguments)
            |> Result.bind (fun () -> StorageCommitment.verifyNative outputPath plans)
            |> Result.map (fun commitments ->
                if not commitments.IsEmpty then
                    File.WriteAllText(Path.ChangeExtension(llvmPath, ".storage.json"), System.Text.Json.JsonSerializer.Serialize commitments))
    with ex -> Error (sprintf "Native compilation failed: %s" ex.Message)

let compileToNativeWithStorage llvmPath outputPath targetTriple deploymentMode externLibraries linkOptions cpu storage =
    compileToNativeWithArtifacts llvmPath outputPath targetTriple deploymentMode externLibraries linkOptions cpu storage None

/// Components with no source-owned writable objects retain the same backend
/// entry point. Real compilation always supplies its checked inventory.
let compileToNative llvmPath outputPath targetTriple deploymentMode externLibraries linkOptions cpu =
    compileToNativeWithStorage llvmPath outputPath targetTriple deploymentMode externLibraries linkOptions cpu []
