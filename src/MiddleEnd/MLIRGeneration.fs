/// MLIRGeneration - MiddleEnd orchestration layer
///
/// Composer Pipeline Context:
///   FrontEnd (CCS) → PSG → MiddleEnd → MLIR text → BackEnd (mliropt/LLVM)
///
/// This module is the PUBLIC API for the MiddleEnd. It orchestrates:
///   1. Alex transfer: witnesses traverse the saturated PSG, reading its codata → structured MLIROp
///   2. Serialization: MLIROp → MLIR text (exit point)
///
/// Nothing about the program is computed here: every fact the witnesses read is on the graph
/// (its nodes, layouts, ranges and `Codata`, settled by CCS at saturation). Composer reads.
///
/// Clean signature: PSG + PlatformContext → MLIR text
module MiddleEnd.MLIRGeneration

open System.IO
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Alex.Traversal.TransferTypes
open Alex.Traversal.MLIRTransfer

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// The instruction set and the declared Register and Pointer widths, read from the CCS context.
/// The widths are the description's (plan D8, L-10) and carry as `Result`s: a site that needs one
/// on a description declaring none fails with CCS8203's text, never with a number of its own.
let private architectureOf (ctx: PlatformContext) : Architecture =
    let isa =
        match ctx.PlatformId with
        | id when id.Contains("x86_64") || id.Contains("x86-64") -> X86_64
        | id when id.Contains("arm_cortex_m7") || id.Contains("arm_cortex_m33") || id.Contains("arm32") -> ARM32_Thumb
        | id when id.Contains("ARM64") || id.Contains("aarch64") -> ARM64
        | id when id.Contains("riscv64") -> RISCV64
        | id when id.Contains("riscv32") -> RISCV32
        | _ -> X86_64
    let width (dimension: WidthDimension) = PlatformContext.tryWidth ctx (WidthDimension.name dimension)
    { Isa = isa; Register = width WidthDimension.Register; Pointer = width WidthDimension.Pointer }

/// Generate MLIR from PSG
/// This is the single entry point for the MiddleEnd
/// Returns (mlirText, externLibraries) on success.
/// ExternLibraries is the set of shared libraries needed by resolved bindings.
let private generateCore
    (graph: SemanticGraph)
    (platformCtx: PlatformContext)
    (targetPlatform: Core.Types.Dialects.TargetPlatform)
    (intermediatesDir: string option)
    (linkedLibraries: Set<string>)
    : Result<string * Set<string>, string> =

    let arch = architectureOf platformCtx
    let codata = graph.Codata.Value

    // Representation decisions inside type mapping that depend on the target (enum DU tags)
    Alex.CodeGeneration.TypeMapping.setTargetPlatform targetPlatform

    // Proof obligations are graph citizens: minted into the PSG by the Baker obligation recipes
    // at saturation (CCS Pass 5), discharged from F at design time by CCS (06a/06b). This is a
    // READ of the same records for the build-time dispatch below (09).
    let proofObligations = Clef.Compiler.Nanopass.ObligationDischarge.ofGraph graph

    let coeffects : TransferCoeffects = {
        Platform = { TargetArch = arch; Bindings = codata.Bindings; LinkedLibraries = linkedLibraries }
        TargetPlatform = targetPlatform
    }

    // Execute Alex transfer (parallel nanopasses)
    match graph.DeclarationRoots with
    | [] -> Result.Error "No declaration roots found in PSG"
    | (entryId, _) :: _ ->
        match transfer graph entryId coeffects intermediatesDir with
        | Result.Ok (topLevelOps, _) ->
            // Filter ops by target platform — FPGA/NPU exclude CPU-only func.func ops
            let platformOps =
                match targetPlatform with
                | Core.Types.Dialects.TargetPlatform.FPGA ->
                    topLevelOps |> List.filter (fun op ->
                        match op with MLIROp.FuncOp _ -> false | _ -> true)
                | Core.Types.Dialects.TargetPlatform.NPU ->
                    // NPU: keep only RawMLIR (the aie.device block from KernelModuleWitness)
                    topLevelOps |> List.filter (fun op ->
                        match op with MLIROp.RawMLIR _ -> true | _ -> false)
                | _ -> topLevelOps

            // Apply MLIR nanopasses (MLIR→MLIR transformations)
            let transformedOps = Alex.Pipeline.MLIRNanopass.applyPasses platformOps coeffects.Platform intermediatesDir

            match Alex.Traversal.StaticStorageValidation.validate graph transformedOps with
            | Result.Error message -> Result.Error message
            | Result.Ok () ->
                // Serialize MLIROp → MLIR text (exit point of MiddleEnd)
                // NPU uses unnamed module (MLIR-AIE expects `module { aie.device(...) { } }`)
                let mlirText =
                    match targetPlatform with
                    | Core.Types.Dialects.TargetPlatform.NPU ->
                        let opsText = opsToString arch.Pointer transformedOps "  "
                        sprintf "module {\n%s\n}" opsText
                    | _ -> moduleToString arch.Pointer "main" transformedOps

                // Write final MLIR output (renamed to 10_output.mlir for nanopass visibility)
                match intermediatesDir with
                | Some dir ->
                    let finalPath = Path.Combine(dir, "10_output.mlir")
                    File.WriteAllText(finalPath, mlirText)
                    if Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig.isVerbose() then
                        printfn "[Alex] Wrote final MLIR: 10_output.mlir"
                    // SMT verification module — parallel residual from the proof
                    // obligations the graph carries (same shape as XDC from the pins)
                    if not (List.isEmpty proofObligations) then
                        let smtPath = Path.Combine(dir, "09_obligations.mlir")
                        File.WriteAllText(smtPath, Alex.Traversal.SMTTransfer.transfer proofObligations + "\n")
                        if Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig.isVerbose() then
                            printfn "[Alex] Wrote SMT verification module: 09_obligations.mlir (%d obligations)" proofObligations.Length
                | None -> ()

                // XDC transfer — parallel residual from the pin facts the graph carries (FPGA only)
                match targetPlatform, codata.Pins with
                | Core.Types.Dialects.TargetPlatform.FPGA, Some mapping ->
                    let xdcText = Alex.Traversal.XDCTransfer.transfer mapping
                    match intermediatesDir with
                    | Some dir ->
                        let xdcPath = Path.Combine(dir, "constraints.xdc")
                        File.WriteAllText(xdcPath, xdcText)
                        if Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig.isVerbose() then
                            printfn "[Alex] Wrote XDC constraints: constraints.xdc (%d pins)" mapping.Pins.Length
                    | None -> ()
                | _ -> ()

                Result.Ok (mlirText, Set.union codata.Bindings.ExternLibraries linkedLibraries)
        | Result.Error msg -> Result.Error msg

/// Generate MLIR for the graph. A core's leg reads the declared Register and Pointer widths at
/// every boundary and layout site (Types.declaredWordWidth, declaredPointerBytes); a description
/// that declares neither cannot start it, and is refused here, before any witness runs, with
/// the code PlatformDeclaration reports for a missing declaration (CCS8203). The fabric leg reads
/// neither, so a description declaring none compiles for it. The deployment mode is the
/// project's; the runtime it selects is read from the graph's platform bindings (CCS).
let generateWithLinkedLibraries
    (graph: SemanticGraph)
    (platformCtx: PlatformContext)
    (_deploymentMode: Core.Types.Dialects.DeploymentMode)
    (targetPlatform: Core.Types.Dialects.TargetPlatform)
    (intermediatesDir: string option)
    (linkedLibraries: Set<string>)
    : Result<string * Set<string>, string> =
    let arch = architectureOf platformCtx
    let undeclared =
        match targetPlatform with
        | Core.Types.Dialects.TargetPlatform.FPGA -> None
        | _ ->
            match arch.Register, arch.Pointer with
            | Result.Error message, _ | _, Result.Error message -> Some message
            | Result.Ok _, Result.Ok _ -> None
    match undeclared with
    | Some message ->
        Result.Error (sprintf "CCS8203: %s; a core's leg reads the Register and Pointer width dimensions at its boundaries and layouts and cannot start without them" message)
    | None -> generateCore graph platformCtx targetPlatform intermediatesDir linkedLibraries

/// Callers without project link declarations retain the existing binding policy.
let generate graph platformCtx deploymentMode targetPlatform intermediatesDir =
    generateWithLinkedLibraries graph platformCtx deploymentMode targetPlatform intermediatesDir Set.empty
