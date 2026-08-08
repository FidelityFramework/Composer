/// GPU Lowering - portable MLIR to an AMD GPU code object via native tools
///
/// Invokes the MLIR toolchain natively (no Python). Pipeline:
///   1. reachability — device call-graph closure from the kernel entry
///   2. wrap         — closure + generated gpu.func → gpu.module
///   3. mlir-opt     — --convert-scf-to-cf --convert-gpu-to-rocdl
///   4. mlir-opt     — --gpu-module-to-binary (embeds the AMDGPU object)
///   5. extract      — lift the object out of #gpu.object<...> into .hsaco
///
/// The MiddleEnd commits to nothing: Alex emits portable dialects (func,
/// scf, arith, memref, index) exactly as it does for every other leg, and
/// the commitment to AMDGPU happens here. Hardware targeting is backend
/// work; the Alex-witnessed MLIR is what sets the transform up.
///
/// Chip resolved from FIDELITY_GPU_CHIP env, default gfx1151.
/// Kernel entry resolved from FIDELITY_GPU_KERNEL env, default "kernel"
/// (matched as a trailing dotted segment of a Clef function name).
/// mlir-opt is expected on PATH (installed with LLVM/MLIR).
module BackEnd.GPU.Lowering

open System.IO

// ═══════════════════════════════════════════════════════════
// TOOL AND TARGET RESOLUTION
// ═══════════════════════════════════════════════════════════

/// AMDGPU target chip. Device selection is a backend concern; the
/// fidproj carries no per-target section, so this is env-driven.
let private resolveChip () : string =
    let fromEnv = System.Environment.GetEnvironmentVariable("FIDELITY_GPU_CHIP")
    if System.String.IsNullOrEmpty(fromEnv) then "gfx1151" else fromEnv

/// Trailing dotted segment identifying the kernel entry function.
let private resolveKernelName () : string =
    let fromEnv = System.Environment.GetEnvironmentVariable("FIDELITY_GPU_KERNEL")
    if System.String.IsNullOrEmpty(fromEnv) then "kernel" else fromEnv

/// Resolve a tool: env override if it exists on disk, else bare name so
/// the OS resolves it via PATH.
let private resolveTool (envVar: string) (name: string) : string =
    let fromEnv = System.Environment.GetEnvironmentVariable(envVar)
    if not (System.String.IsNullOrEmpty(fromEnv)) && File.Exists(fromEnv) then fromEnv
    else name

/// Run an external tool with environment augmentation.
/// Returns Ok(stdout) or Error(message).
let private runTool
    (tool: string)
    (args: string)
    (workDir: string)
    (extraEnv: (string * string) list)
    : Result<string, string> =
    try
        let proc = new System.Diagnostics.Process()
        proc.StartInfo.FileName <- tool
        proc.StartInfo.Arguments <- args
        proc.StartInfo.UseShellExecute <- false
        proc.StartInfo.RedirectStandardOutput <- true
        proc.StartInfo.RedirectStandardError <- true
        proc.StartInfo.WorkingDirectory <- workDir

        for (key, value) in extraEnv do
            proc.StartInfo.EnvironmentVariables.[key] <- value

        proc.Start() |> ignore
        let stdout = proc.StandardOutput.ReadToEnd()
        let stderr = proc.StandardError.ReadToEnd()
        proc.WaitForExit()

        if proc.ExitCode <> 0 then
            Error (sprintf "%s failed (exit %d):\n%s" tool proc.ExitCode stderr)
        else
            Ok stdout
    with
    | :? System.ComponentModel.Win32Exception ->
        Error (sprintf "%s not found. Ensure it is on PATH (installed with LLVM/MLIR)." tool)
    | ex ->
        Error (sprintf "%s failed: %s" tool ex.Message)

/// Run a tool, discarding stdout, returning Ok () or Error.
let private runToolUnit tool args workDir extraEnv =
    runTool tool args workDir extraEnv |> Result.map ignore

// ═══════════════════════════════════════════════════════════
// DEVICE CALL-GRAPH CLOSURE
// ═══════════════════════════════════════════════════════════

/// One top-level function lifted out of the middle end's module.
type private DeviceFunc = {
    Name: string
    Text: string
    Calls: string list
}

/// Split the emitted module into top-level func.func definitions.
/// Declarations without a body (FFI externs) are skipped: they have no
/// device implementation and must not reach the code object.
let private parseFunctions (mlirText: string) : DeviceFunc list =
    let lines = mlirText.Replace("\r\n", "\n").Split('\n')
    let header = System.Text.RegularExpressions.Regex(@"func\.func\s+(?:private\s+)?@([A-Za-z0-9_.$]+)\s*\(")
    let callRef = System.Text.RegularExpressions.Regex(@"func\.call\s+@([A-Za-z0-9_.$]+)")

    let mutable acc = []
    let mutable i = 0
    while i < lines.Length do
        let m = header.Match(lines.[i])
        if m.Success && lines.[i].Contains("{") then
            let name = m.Groups.[1].Value
            let start = i
            let mutable depth = 0
            let mutable finished = false
            while i < lines.Length && not finished do
                let line = lines.[i]
                depth <- depth + (line |> Seq.filter (fun c -> c = '{') |> Seq.length)
                depth <- depth - (line |> Seq.filter (fun c -> c = '}') |> Seq.length)
                i <- i + 1
                if depth <= 0 then finished <- true
            let text = System.String.Join("\n", lines.[start .. i - 1])
            let calls =
                callRef.Matches(text)
                |> Seq.map (fun c -> c.Groups.[1].Value)
                |> Seq.distinct
                |> List.ofSeq
            acc <- { Name = name; Text = text; Calls = calls } :: acc
        else
            i <- i + 1
    List.rev acc

/// Transitive closure of functions reachable from the kernel entry.
/// This is device-side tree shaking: the host module carries the entry
/// point, console I/O and FFI shims, none of which belong on the device.
let private reachableFrom (entry: string) (funcs: DeviceFunc list) : DeviceFunc list =
    let byName = funcs |> List.map (fun f -> f.Name, f) |> Map.ofList
    let rec walk (seen: Set<string>) (pending: string list) =
        match pending with
        | [] -> seen
        | name :: rest ->
            if Set.contains name seen then walk seen rest
            else
                match Map.tryFind name byName with
                | Some f -> walk (Set.add name seen) (f.Calls @ rest)
                | None -> walk seen rest
    let keep = walk Set.empty [entry]
    funcs |> List.filter (fun f -> Set.contains f.Name keep)

/// Find the kernel entry by trailing dotted segment.
let private findKernelEntry (funcs: DeviceFunc list) (kernelName: string) : Result<string, string> =
    let suffix = "." + kernelName
    let matches =
        funcs
        |> List.filter (fun f -> f.Name.EndsWith(suffix) || f.Name = kernelName)
    match matches with
    | [ single ] -> Ok single.Name
    | [] ->
        Error (sprintf "no kernel entry found: expected a function named '%s' (or ending in '%s').\nCandidates in module: %s"
                   kernelName suffix
                   (funcs |> List.map (fun f -> f.Name) |> String.concat ", "))
    | many ->
        Error (sprintf "ambiguous kernel entry '%s': %s"
                   kernelName (many |> List.map (fun f -> f.Name) |> String.concat ", "))

// ═══════════════════════════════════════════════════════════
// GPU MODULE EMISSION
// ═══════════════════════════════════════════════════════════

/// The generated dispatch wrapper.
///
/// The application supplies the compute function; the toolchain supplies
/// the data movement — the same division of labour the NPU leg uses. The
/// Clef side never names a buffer, which is why it lowers at all: a
/// nativeptr parameter arrives as a bare `index` carrying neither extent
/// nor address space, and the ROCDL pipeline cannot legalise it. Here the
/// buffers are genuine memrefs that the toolchain can place and size.
///
/// Kernel contract (map over a 1-D index domain with a shared table):
///   let kernel (i: int) (data: int array) (n: int) : int
/// where i is the global index, data is a read-only table every thread may
/// index freely, and n is the size of the output domain. The table makes
/// gather kernels expressible without the leg knowing anything about the
/// application's data: a Clef `int array` parameter already lowers to
/// `memref<?xi64>`, so the wrapper passes its own memref straight through.
let private emitDispatchWrapper (entryName: string) : string =
    sprintf """
    gpu.func @clef_kernel(%%data: memref<?xi64>, %%out: memref<?xi32>, %%n: i64) kernel {
      %%bid = gpu.block_id x
      %%bdim = gpu.block_dim x
      %%tid = gpu.thread_id x
      %%base = arith.muli %%bid, %%bdim : index
      %%gidx = arith.addi %%base, %%tid : index
      %%gi = arith.index_cast %%gidx : index to i64
      %%inRange = arith.cmpi slt, %%gi, %%n : i64
      scf.if %%inRange {
        %%v = func.call @%s(%%gi, %%data, %%n) : (i64, memref<?xi64>, i64) -> i64
        %%v32 = arith.trunci %%v : i64 to i32
        memref.store %%v32, %%out[%%gidx] : memref<?xi32>
      }
      gpu.return
    }""" entryName

/// Assemble the device module: reachable Clef functions plus the
/// generated dispatch wrapper, inside a chip-targeted gpu.module.
let private buildGpuModule (mlirText: string) (kernelName: string) (chip: string) : Result<string, string> =
    let funcs = parseFunctions mlirText
    if List.isEmpty funcs then
        Error "no func.func definitions found in the middle end's output"
    else
        match findKernelEntry funcs kernelName with
        | Error e -> Error e
        | Ok entry ->
            let device = reachableFrom entry funcs
            let bodies = device |> List.map (fun f -> f.Text) |> String.concat "\n"
            let sb = System.Text.StringBuilder()
            sb.AppendLine("module attributes {gpu.container_module} {") |> ignore
            sb.AppendLine(sprintf "  gpu.module @clef_device [#rocdl.target<chip = \"%s\">] {" chip) |> ignore
            sb.AppendLine(bodies) |> ignore
            sb.AppendLine(emitDispatchWrapper entry) |> ignore
            sb.AppendLine("  }") |> ignore
            sb.AppendLine("}") |> ignore
            Ok (sb.ToString())

// ═══════════════════════════════════════════════════════════
// CODE OBJECT EXTRACTION
// ═══════════════════════════════════════════════════════════

/// Decode an MLIR string-attribute literal into the bytes it denotes.
/// MLIR escapes non-printable bytes as \XX (two hex digits).
let private decodeMlirBytes (literal: string) : byte[] =
    let out = System.Collections.Generic.List<byte>()
    let hex (c: char) =
        if c >= '0' && c <= '9' then int c - int '0'
        elif c >= 'a' && c <= 'f' then int c - int 'a' + 10
        elif c >= 'A' && c <= 'F' then int c - int 'A' + 10
        else -1
    let mutable i = 0
    while i < literal.Length do
        let c = literal.[i]
        if c = '\\' && i + 2 < literal.Length then
            let h1 = hex literal.[i + 1]
            let h2 = hex literal.[i + 2]
            if h1 >= 0 && h2 >= 0 then
                out.Add(byte (h1 * 16 + h2))
                i <- i + 3
            else
                // \\ , \" and friends denote the literal second character
                out.Add(byte literal.[i + 1])
                i <- i + 2
        else
            out.Add(byte c)
            i <- i + 1
    out.ToArray()

/// Lift the embedded AMDGPU object out of the gpu.binary attribute.
let private extractCodeObject (binMlirPath: string) (hsacoPath: string) : Result<unit, string> =
    let text = File.ReadAllText(binMlirPath)
    let marker = "bin = \""
    let start = text.IndexOf(marker)
    if start < 0 then
        Error "gpu-module-to-binary produced no embedded object (no 'bin = \"...\"' attribute found)"
    else
        let contentStart = start + marker.Length
        // Scan to the closing quote, honouring backslash escapes.
        let mutable i = contentStart
        let mutable finished = false
        while i < text.Length && not finished do
            if text.[i] = '\\' then i <- i + 2
            elif text.[i] = '"' then finished <- true
            else i <- i + 1
        if not finished then
            Error "malformed gpu.binary attribute: unterminated object literal"
        else
            let bytes = decodeMlirBytes (text.Substring(contentStart, i - contentStart))
            if bytes.Length < 4 || bytes.[0] <> 0x7Fuy || bytes.[1] <> byte 'E' then
                Error (sprintf "extracted object is not an ELF code object (%d bytes)" bytes.Length)
            else
                File.WriteAllBytes(hsacoPath, bytes)
                Ok ()

// ═══════════════════════════════════════════════════════════
// LOWERING ENTRY POINT
// ═══════════════════════════════════════════════════════════

/// Lower portable MLIR to an AMD GPU code object.
let lowerToCodeObject (mlirPath: string) (hsacoPath: string) : Result<unit, string> =
    let workDir = Path.GetDirectoryName(mlirPath)
    let mlirOpt = resolveTool "FIDELITY_MLIR_OPT" "mlir-opt"
    let chip = resolveChip ()
    let kernelName = resolveKernelName ()

    let devicePath = Path.Combine(workDir, "device.mlir")
    let rocdlPath = Path.Combine(workDir, "device_rocdl.mlir")
    let binPath = Path.Combine(workDir, "device_binary.mlir")

    let mlirText = File.ReadAllText(mlirPath)

    match buildGpuModule mlirText kernelName chip with
    | Error e -> Error (sprintf "GPU module assembly failed:\n%s" e)
    | Ok deviceModule ->

    File.WriteAllText(devicePath, deviceModule)

    let rocdlArgs =
        sprintf "%s --convert-scf-to-cf --convert-gpu-to-rocdl -o %s" devicePath rocdlPath
    match runToolUnit mlirOpt rocdlArgs workDir [] with
    | Error e -> Error (sprintf "mlir-opt ROCDL conversion failed:\n%s" e)
    | Ok () ->

    let binArgs = sprintf "%s --gpu-module-to-binary -o %s" rocdlPath binPath
    match runToolUnit mlirOpt binArgs workDir [] with
    | Error e -> Error (sprintf "mlir-opt code object generation failed:\n%s" e)
    | Ok () ->

    match extractCodeObject binPath hsacoPath with
    | Error e -> Error (sprintf "code object extraction failed:\n%s" e)
    | Ok () ->

    printfn "  GPU: %s kernel entry '%s' -> %s" chip kernelName (Path.GetFileName(hsacoPath))
    Ok ()
