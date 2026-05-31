/// AIE Lowering - MLIR-AIE dialect to xclbin via native tools
///
/// Invokes the AIE toolchain natively (no Python). Pipeline:
///   1. aie-opt      — objectfifo/lock/BD/buffer/route passes
///   2. aie-translate — per-core MLIR → LLVM IR
///   3. Peano opt+llc — per-core LLVM IR → AIE2 ELF
///   4. aie-translate — CDO generation (--aie-generate-cdo)
///   5. aie-translate — NPU instruction binary (--aie-npu-to-binary)
///   6. bootgen       — CDO → PDI
///   7. xclbinutil    — package PDI + metadata → xclbin
///
/// Tool paths resolved from AIE_TOOLCHAIN env or ~/aie-toolchain.
/// Peano (llvm-aie) provides clang++/opt/llc for AIE2 targets.
/// xclbinutil is expected on PATH (installed with XRT).
module BackEnd.AIE.Lowering

open System.IO

// ═══════════════════════════════════════════════════════════
// TOOL RESOLUTION
// ═══════════════════════════════════════════════════════════

/// Resolve the AIE toolchain root directory.
/// Checks AIE_TOOLCHAIN env first, falls back to ~/aie-toolchain.
let private resolveToolchainRoot () =
    let envPath = System.Environment.GetEnvironmentVariable("AIE_TOOLCHAIN")
    if not (System.String.IsNullOrEmpty(envPath)) then envPath
    else
        let home = System.Environment.GetFolderPath(System.Environment.SpecialFolder.UserProfile)
        Path.Combine(home, "aie-toolchain")

/// Resolve a binary path: check env override, then toolchain bin/, then PATH.
let private resolveTool (toolchainRoot: string) (envVar: string) (name: string) =
    let envPath = System.Environment.GetEnvironmentVariable(envVar)
    if not (System.String.IsNullOrEmpty(envPath)) && File.Exists(envPath) then
        envPath
    else
        let inBin = Path.Combine(toolchainRoot, "bin", name)
        if File.Exists(inBin) then inBin
        else name // fall back to PATH

/// Resolve Peano (llvm-aie) bin directory for opt/llc.
/// pip installs llvm-aie under lib/python3.12/site-packages/llvm-aie/bin/
let private resolvePeanoBin (toolchainRoot: string) =
    let envPeano = System.Environment.GetEnvironmentVariable("PEANO_INSTALL_DIR")
    if not (System.String.IsNullOrEmpty(envPeano)) then
        Path.Combine(envPeano, "bin")
    else
        let sitePackages = Path.Combine(toolchainRoot, "lib", "python3.12", "site-packages")
        Path.Combine(sitePackages, "llvm-aie", "bin")

/// Resolve mlir-aie runtime lib directory for per-core link.
let private resolveRuntimeLibDir (toolchainRoot: string) (aieTarget: string) =
    let envDir = System.Environment.GetEnvironmentVariable("MLIR_AIE_INSTALL_DIR")
    let installDir =
        if not (System.String.IsNullOrEmpty(envDir)) then envDir
        else
            let sitePackages = Path.Combine(toolchainRoot, "lib", "python3.12", "site-packages")
            Path.Combine(sitePackages, "mlir_aie")
    Path.Combine(installDir, "aie_runtime_lib", aieTarget.ToUpperInvariant())

// ═══════════════════════════════════════════════════════════
// TOOL RUNNER (shared with CIRCT backend pattern)
// ═══════════════════════════════════════════════════════════

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
        Error (sprintf "%s not found. Ensure AIE_TOOLCHAIN is set or the tool is on PATH." tool)
    | ex ->
        Error (sprintf "%s failed: %s" tool ex.Message)

/// Run a tool, discarding stdout, returning Ok () or Error.
let private runToolUnit tool args workDir extraEnv =
    runTool tool args workDir extraEnv |> Result.map ignore

/// Run a tool, capturing stdout to a file.
let private runToolToFile tool args workDir extraEnv (outputPath: string) =
    match runTool tool args workDir extraEnv with
    | Ok stdout ->
        File.WriteAllText(outputPath, stdout)
        Ok ()
    | Error e -> Error e

// ═══════════════════════════════════════════════════════════
// JSON METADATA GENERATION (for xclbinutil)
// ═══════════════════════════════════════════════════════════

/// mem_topology.json - describes memory regions visible to the NPU.
/// Standard layout for NPU2: HOST (DRAM) + SRAM.
let private emitMemTopologyJson () =
    """{
  "mem_topology": {
    "m_count": "2",
    "m_mem_data": [
      { "m_type": "MEM_DRAM", "m_used": "1", "m_sizeKB": "0x10000", "m_tag": "HOST", "m_base_address": "0x4000000" },
      { "m_type": "MEM_DRAM", "m_used": "1", "m_sizeKB": "0xc000", "m_tag": "SRAM", "m_base_address": "0x4000000" }
    ]
  }
}"""

/// kernels.json - DPU kernel descriptor for xclbinutil.
/// kernel_name = "MLIR_AIE", kernel_id = 0x901 (standard for mlir-aie).
/// Arguments: opcode (scalar), instr (SRAM), ninstr (scalar), bo0..bo4 (HOST).
let private emitKernelsJson (kernelName: string) (kernelId: string) =
    sprintf """{
  "ps-kernels": {
    "kernels": [{
      "name": "%s",
      "type": "dpu",
      "extended-data": { "subtype": "DPU", "functional": "0", "dpu_kernel_id": "%s" },
      "arguments": [
        { "name": "opcode", "address-qualifier": "SCALAR", "type": "uint64_t", "offset": "0x00" },
        { "name": "instr", "memory-connection": "SRAM", "address-qualifier": "GLOBAL", "type": "char *", "offset": "0x08" },
        { "name": "ninstr", "address-qualifier": "SCALAR", "type": "uint32_t", "offset": "0x10" },
        { "name": "bo0", "memory-connection": "HOST", "address-qualifier": "GLOBAL", "type": "void*", "offset": "0x14" },
        { "name": "bo1", "memory-connection": "HOST", "address-qualifier": "GLOBAL", "type": "void*", "offset": "0x1c" },
        { "name": "bo2", "memory-connection": "HOST", "address-qualifier": "GLOBAL", "type": "void*", "offset": "0x24" },
        { "name": "bo3", "memory-connection": "HOST", "address-qualifier": "GLOBAL", "type": "void*", "offset": "0x2c" },
        { "name": "bo4", "memory-connection": "HOST", "address-qualifier": "GLOBAL", "type": "void*", "offset": "0x34" }
      ],
      "instances": [{ "name": "MLIRAIE" }]
    }]
  }
}""" kernelName kernelId

/// aie_partition.json - partition descriptor for xclbinutil.
/// NPU2: start_columns = [0], column_width depends on tile count.
let private emitPartitionJson (pdiPath: string) (kernelId: string) (numCols: int) =
    let pdiUuid = System.Guid.NewGuid().ToString()
    sprintf """{
  "aie_partition": {
    "name": "QoS",
    "operations_per_cycle": "2048",
    "inference_fingerprint": "23423",
    "pre_post_fingerprint": "12345",
    "partition": { "column_width": %d, "start_columns": [0] },
    "PDIs": [{
      "uuid": "%s",
      "file_name": "%s",
      "cdo_groups": [{
        "name": "DPU",
        "type": "PRIMARY",
        "pdi_id": "0x01",
        "dpu_kernel_ids": ["%s"],
        "pre_cdo_groups": ["0xC1"]
      }]
    }]
  }
}""" numCols pdiUuid pdiPath kernelId

/// BIF file content for bootgen (CDO → PDI).
let private emitDesignBif (tmpDir: string) (deviceName: string) =
    let cdoElfs = Path.Combine(tmpDir, sprintf "%s_aie_cdo_elfs.bin" deviceName)
    let cdoInit = Path.Combine(tmpDir, sprintf "%s_aie_cdo_init.bin" deviceName)
    let cdoEnable = Path.Combine(tmpDir, sprintf "%s_aie_cdo_enable.bin" deviceName)
    sprintf """all:
{
  id_code = 0x14ca8093
  extended_id_code = 0x01
  image
  {
    name=aie_image, id=0x1c000000
    { type=cdo file=%s file=%s file=%s }
  }
}""" cdoElfs cdoInit cdoEnable

// ═══════════════════════════════════════════════════════════
// AIE COMPILATION PIPELINE
// ═══════════════════════════════════════════════════════════

/// The aie-opt pass pipeline for NPU2 targets.
/// Replicates the standard mlir-aie flow:
///   objectfifo transform → lock/BD assignment → buffer addressing → routing → NPU lowering
let private aieOptPassPipeline (deviceName: string) =
    // Phase 1: Resource allocation and objectfifo lowering
    let phase1 =
        "builtin.module(" +
        "lower-affine,aie.device(" +
        "aie-assign-lock-ids," +
        "aie-register-objectFifos," +
        "aie-objectFifo-stateful-transform," +
        "aie-assign-bd-ids," +
        "aie-lower-cascade-flows," +
        "aie-lower-broadcast-packet," +
        "aie-lower-multicast," +
        "aie-assign-tile-controller-ids," +
        "aie-generate-column-control-overlay{route-shim-to-tile-ctrl=false}," +
        "aie-assign-buffer-addresses{alloc-scheme=basic-sequential}," +
        "aie-vector-transfer-lowering{max-transfer-rank=1}))"

    // Phase 2: Pathfinder routing
    let phase2 =
        "builtin.module(aie.device(aie-create-pathfinder-flows))"

    // Phase 3: NPU instruction lowering
    let phase3 =
        "builtin.module(aie.device(" +
        "aie-materialize-bd-chains," +
        "aie-substitute-shim-dma-allocations," +
        "aie-assign-runtime-sequence-bd-ids," +
        "aie-dma-tasks-to-npu," +
        "aie-dma-to-npu," +
        "aie-lower-set-lock))"

    (phase1, phase2, phase3)

/// Per-core LLVM lowering pass pipeline for aie-opt.
/// Device is auto-detected from the module; do NOT pass {device=...} to
/// aie-standard-lowering (it rejects string names like "npu2").
let private coreLoweringPipeline (aieTarget: string) =
    sprintf "builtin.module(aie.device(aie-localize-locks,aie-normalize-address-spaces,aie-transform-bfp-types),aie-standard-lowering,aiex-standard-lowering,convert-aievec-to-llvm{aie-target=%s},canonicalize,cse,expand-strided-metadata,lower-affine,arith-expand,finalize-memref-to-llvm,convert-func-to-llvm{use-bare-ptr-memref-call-conv=true},convert-to-llvm{dynamic=true},canonicalize,cse)"
        (aieTarget.ToLowerInvariant())

/// Extract core tile coordinates from physical MLIR text.
/// Parses aie.core ops in pretty-print format: %core_C_R = aie.core(%tile_C_R)
/// Returns list of (col, row) pairs.
let private extractCoreTiles (mlirText: string) : (int * int) list =
    let regex = System.Text.RegularExpressions.Regex(@"%core_(\d+)_(\d+)\s*=\s*aie\.core")
    [ for m in regex.Matches(mlirText) do
        yield (int m.Groups.[1].Value, int m.Groups.[2].Value) ]

/// Patch physical MLIR (generic format) to replace aie.core bodies with
/// elf_file references and empty bodies (just aie.end).
/// CDO generation requires elf_file attribute and empty core bodies.
let private patchPhysicalMlirWithElfs
    (genericMlirText: string)
    (coreElfs: Map<int * int, string>)
    : string =
    // Build tile SSA -> (col, row) mapping from generic format:
    //   %N = "aie.tile"() <{col = C : i32, row = R : i32}>
    let tileRegex =
        System.Text.RegularExpressions.Regex(
            @"(%\d+)\s*=\s*""aie\.tile""\(\)\s*<\{col\s*=\s*(\d+)\s*:\s*i32,\s*row\s*=\s*(\d+)\s*:\s*i32\}>")
    let tileMap =
        [ for m in tileRegex.Matches(genericMlirText) do
            yield (m.Groups.[1].Value, (int m.Groups.[2].Value, int m.Groups.[3].Value)) ]
        |> Map.ofList

    // Find each "aie.core"(%N) <{attrs}> ({ ... }) and replace body + add elf_file
    let coreRegex =
        System.Text.RegularExpressions.Regex(
            @"""aie\.core""\((%\d+)\)\s*<\{([^}]*)\}>\s*\(\{")
    let mutable result = System.Text.StringBuilder()
    let mutable pos = 0

    for m in coreRegex.Matches(genericMlirText) do
        let tileSsa = m.Groups.[1].Value
        let attrs = m.Groups.[2].Value
        match Map.tryFind tileSsa tileMap with
        | Some (col, row) when Map.containsKey (col, row) coreElfs ->
            let elfFile = coreElfs.[(col, row)]
            // Find the matching closing }) by counting depth
            let bodyStart = m.Index + m.Length
            let mutable depth = 1
            let mutable i = bodyStart
            while i < genericMlirText.Length && depth > 0 do
                if i + 1 < genericMlirText.Length
                   && genericMlirText.[i] = '(' && genericMlirText.[i + 1] = '{' then
                    depth <- depth + 1
                    i <- i + 2
                elif i + 1 < genericMlirText.Length
                     && genericMlirText.[i] = '}' && genericMlirText.[i + 1] = ')' then
                    depth <- depth - 1
                    if depth = 0 then () // don't advance past; bodyEnd = i
                    else i <- i + 2
                else
                    i <- i + 1

            let bodyEnd = i // position of closing }

            result.Append(genericMlirText, pos, m.Index - pos) |> ignore
            result.Append(sprintf "\"aie.core\"(%s) <{elf_file = \"%s\", %s}> ({\n      \"aie.end\"() : () -> ()\n    " tileSsa elfFile attrs) |> ignore
            pos <- bodyEnd
        | _ ->
            () // no elf for this core; leave unchanged

    result.Append(genericMlirText, pos, genericMlirText.Length - pos) |> ignore
    result.ToString()

/// Full compilation: MLIR-AIE → xclbin + insts.bin
///
/// All steps use native binaries. No Python, no scripting.
/// The pipeline mirrors aiecc.py's --no-xchesscc --no-xbridge flow
/// but implemented as direct tool invocations.
let lowerToXclbin (mlirPath: string) (xclbinPath: string) (instsPath: string) : Result<unit, string> =
    let toolchainRoot = resolveToolchainRoot ()
    let aieOpt = resolveTool toolchainRoot "AIE_OPT_PATH" "aie-opt"
    let aieTranslate = resolveTool toolchainRoot "AIE_TRANSLATE_PATH" "aie-translate"
    let bootgen = resolveTool toolchainRoot "BOOTGEN_PATH" "bootgen"
    let peanoBin = resolvePeanoBin toolchainRoot
    let peanoOpt = Path.Combine(peanoBin, "opt")
    let peanoLlc = Path.Combine(peanoBin, "llc")

    let workDir = Path.GetDirectoryName(mlirPath)
    let tmpDir = Path.Combine(workDir, "aie_prj")
    if not (Directory.Exists tmpDir) then
        Directory.CreateDirectory(tmpDir) |> ignore

    // Device sym_name from the aie.device op (must match CDO output filenames).
    // The generated MLIR-AIE uses sym_name = "main" for the aie.device block.
    let deviceName = "main"
    let aieTarget = "aie2"
    let kernelName = "MLIR_AIE"
    let kernelId = "0x901"

    // Augment PATH with toolchain and Peano bin directories
    let existingPath =
        match System.Environment.GetEnvironmentVariable("PATH") with
        | null -> ""
        | p -> p
    let toolchainBinDir = Path.Combine(toolchainRoot, "bin")
    let augmentedPath = sprintf "%s:%s:%s" toolchainBinDir peanoBin existingPath
    let env = [
        "PATH", augmentedPath
        "PEANO_INSTALL_DIR", Path.GetDirectoryName(peanoBin)
    ]

    let (phase1, phase2, phase3) = aieOptPassPipeline deviceName

    // ── Step 1: aie-opt phase 1 — resource allocation ──
    let withAddresses = Path.Combine(tmpDir, "input_with_addresses.mlir")
    match runToolUnit aieOpt (sprintf "--pass-pipeline=\"%s\" %s -o %s" phase1 mlirPath withAddresses) workDir env with
    | Error e -> Error (sprintf "aie-opt phase 1 (resource allocation) failed:\n%s" e)
    | Ok () ->

    // ── Step 2: aie-opt phase 2 — pathfinder routing ──
    let physical = Path.Combine(tmpDir, "input_physical.mlir")
    match runToolUnit aieOpt (sprintf "--pass-pipeline=\"%s\" %s -o %s" phase2 withAddresses physical) workDir env with
    | Error e -> Error (sprintf "aie-opt phase 2 (routing) failed:\n%s" e)
    | Ok () ->

    // ── Step 3: Per-core compilation via Peano ──
    // Each aie.core op must be compiled to an AIE2 ELF (.o) file, then the
    // physical MLIR is patched to reference these ELFs (with empty core bodies).
    // CDO generation requires elf_file attribute and empty bodies.
    //
    // Sub-steps:
    //   3a. Lower core ops to LLVM dialect (aie-opt core pipeline)
    //   3b. Per core: extract LLVM IR via aie-translate --mlir-to-llvmir --tilecol/row
    //   3c. Per core: Peano opt + llc → .o (AIE2 ELF)
    //   3d. Patch physical MLIR with elf_file attributes and empty bodies

    let corePipeline = coreLoweringPipeline aieTarget
    let loweredMlir = Path.Combine(tmpDir, "input_lowered.mlir")

    // 3a: Core lowering to LLVM dialect
    match runToolUnit aieOpt (sprintf "--pass-pipeline=\"%s\" %s -o %s" corePipeline withAddresses loweredMlir) workDir env with
    | Error e -> Error (sprintf "aie-opt core lowering failed:\n%s" e)
    | Ok () ->

    // Discover core tiles from the physical MLIR
    let physicalText = File.ReadAllText(physical)
    let coreTiles = extractCoreTiles physicalText

    // 3b + 3c: Per-core LLVM IR extraction and Peano compilation
    let compileCore (col: int) (row: int) : Result<string, string> =
        let coreName = sprintf "core_%d_%d" col row
        let coreLl = Path.Combine(tmpDir, sprintf "%s.ll" coreName)
        let coreOptLl = Path.Combine(tmpDir, sprintf "%s.opt.ll" coreName)
        let coreObj = Path.Combine(tmpDir, sprintf "%s.o" coreName)

        match runToolToFile aieTranslate
                (sprintf "--mlir-to-llvmir --tilecol=%d --tilerow=%d %s" col row loweredMlir)
                workDir env coreLl with
        | Error e -> Error (sprintf "aie-translate LLVM IR for core (%d,%d) failed:\n%s" col row e)
        | Ok () ->
        match runToolUnit peanoOpt
                (sprintf "--passes=default<O2> -inline-threshold=10 -S %s -o %s" coreLl coreOptLl)
                workDir env with
        | Error e -> Error (sprintf "Peano opt for core (%d,%d) failed:\n%s" col row e)
        | Ok () ->
        match runToolUnit peanoLlc
                (sprintf "%s -O2 --march=%s --function-sections --filetype=obj -o %s" coreOptLl aieTarget coreObj)
                workDir env with
        | Error e -> Error (sprintf "Peano llc for core (%d,%d) failed:\n%s" col row e)
        | Ok () -> Ok (sprintf "%s.o" coreName)

    let coreElfsResult =
        coreTiles
        |> List.fold (fun (acc: Result<Map<int * int, string>, string>) (col, row) ->
            match acc with
            | Error _ -> acc
            | Ok m ->
                match compileCore col row with
                | Error e -> Error e
                | Ok elfName -> Ok (Map.add (col, row) elfName m)
        ) (Ok Map.empty)

    match coreElfsResult with
    | Error e -> Error e
    | Ok coreElfs ->

    // 3d: Patch physical MLIR with elf_file attributes and empty core bodies.
    // Convert to generic format first (required for attribute patching), then
    // patch, and write the result as the new physical MLIR for CDO generation.
    let physicalGeneric = Path.Combine(tmpDir, "input_physical_generic.mlir")
    match runToolToFile aieOpt
            (sprintf "--mlir-print-op-generic %s" physical)
            workDir env physicalGeneric with
    | Error e -> Error (sprintf "aie-opt generic print failed:\n%s" e)
    | Ok () ->

    let genericText = File.ReadAllText(physicalGeneric)
    let patchedText = patchPhysicalMlirWithElfs genericText coreElfs
    let physicalPatched = Path.Combine(tmpDir, "input_physical_patched.mlir")
    File.WriteAllText(physicalPatched, patchedText)

    // ── Step 4: NPU instruction lowering + binary ──
    let npuLowered = Path.Combine(tmpDir, "npu_insts.mlir")
    match runToolUnit aieOpt (sprintf "--pass-pipeline=\"%s\" %s -o %s" phase3 physical npuLowered) workDir env with
    | Error e -> Error (sprintf "aie-opt phase 3 (NPU lowering) failed:\n%s" e)
    | Ok () ->

    // Generate insts.bin from lowered NPU instructions
    match runToolUnit aieTranslate (sprintf "--aie-npu-to-binary --aie-output-binary %s -o %s" npuLowered instsPath) workDir env with
    | Error e -> Error (sprintf "aie-translate (NPU binary) failed:\n%s" e)
    | Ok () ->

    // ── Step 5: CDO generation (uses patched physical MLIR with elf_file attrs) ──
    match runToolUnit aieTranslate (sprintf "--aie-generate-cdo %s --work-dir-path=%s" physicalPatched tmpDir) workDir env with
    | Error e -> Error (sprintf "aie-translate (CDO) failed:\n%s" e)
    | Ok () ->

    // ── Step 6: bootgen — CDO → PDI ──
    let bifFile = Path.Combine(tmpDir, "design.bif")
    let pdiFile = Path.Combine(tmpDir, "design.pdi")
    File.WriteAllText(bifFile, emitDesignBif tmpDir deviceName)

    match runToolUnit bootgen (sprintf "-arch versal -image %s -o %s -w" bifFile pdiFile) workDir env with
    | Error e -> Error (sprintf "bootgen failed:\n%s" e)
    | Ok () ->

    // ── Step 7: xclbinutil — package xclbin ──
    let memTopoFile = Path.Combine(tmpDir, "mem_topology.json")
    let kernelsFile = Path.Combine(tmpDir, "kernels.json")
    let partitionFile = Path.Combine(tmpDir, "aie_partition.json")

    // Estimate column count from tile count (hello world: 4 tiles = 4 columns)
    // TODO: Extract from MLIR module metadata for general case
    let numCols = 4

    File.WriteAllText(memTopoFile, emitMemTopologyJson ())
    File.WriteAllText(kernelsFile, emitKernelsJson kernelName kernelId)
    File.WriteAllText(partitionFile, emitPartitionJson pdiFile kernelId numCols)

    let xclbinutilArgs =
        sprintf "--add-replace-section MEM_TOPOLOGY:JSON:%s --add-kernel %s --add-replace-section AIE_PARTITION:JSON:%s --force --quiet --output %s"
            memTopoFile kernelsFile partitionFile xclbinPath

    match runToolUnit "xclbinutil" xclbinutilArgs workDir env with
    | Error e -> Error (sprintf "xclbinutil failed:\n%s" e)
    | Ok () ->

    // Verify outputs exist
    if not (File.Exists xclbinPath) then
        Error (sprintf "Pipeline completed but xclbin not found at %s" xclbinPath)
    elif not (File.Exists instsPath) then
        Error (sprintf "Pipeline completed but insts.bin not found at %s" instsPath)
    else
        Ok ()
