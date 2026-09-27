/// MLIRGeneration - MiddleEnd orchestration layer
///
/// Composer Pipeline Context:
///   FrontEnd (CCS) → PSG → MiddleEnd → typed MLIR + text → BackEnd (mliropt/LLVM)
///
/// This module is the PUBLIC API for the MiddleEnd. It orchestrates:
///   1. Alex transfer: witnesses traverse the saturated PSG, reading its codata → structured MLIROp
///   2. Serialization: MLIROp → portable MLIR text, retaining the exact operations
///      for backend realization of target runtime primitives.
///
/// Nothing about the program is computed here: every fact the witnesses read is on the graph
/// (its nodes, layouts, ranges and `Codata`, settled by CCS at saturation). Composer reads.
///
/// Clean signature: PSG + PlatformContext → witnessed MLIR and portable text
module MiddleEnd.MLIRGeneration

open System.IO
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Alex.Traversal.TransferTypes
open Alex.Traversal.MLIRTransfer
open Core.Types.Pipeline

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// The declared Register and Pointer widths, read from the CCS context.
/// The widths are the description's (plan D8, L-10) and carry as `Result`s: a site that needs one
/// on a description declaring none fails with CCS8203's text, never with a number of its own.
let private architectureOf (ctx: PlatformContext) : Architecture =
    let width (dimension: WidthDimension) = PlatformContext.tryWidth ctx (WidthDimension.name dimension)
    { Register = width WidthDimension.Register; Pointer = width WidthDimension.Pointer }

/// Generate MLIR from PSG
/// This is the single entry point for the MiddleEnd
/// Returns (witnessed module, externLibraries) on success.
/// ExternLibraries is the set of shared libraries needed by resolved bindings.
let private generateCore
    (graph: SemanticGraph)
    (platformCtx: PlatformContext)
    (targetPlatform: Core.Types.Dialects.TargetPlatform)
    (intermediatesDir: string option)
    (linkedLibraries: Set<string>)
    (sourceProof: Core.Types.WitnessArtifacts.SourceProofReceipt option)
    : Result<BackEndInput * Set<string>, string> =

    let arch = architectureOf platformCtx
    let codata = graph.Codata.Value

    // Representation decisions inside type mapping that depend on the target (enum DU tags)

    // Always witness the complete source inventory. Keeping intermediates only
    // controls files; source and backend orchestration own actual proof dispatch.
    let proofObligations = Clef.Compiler.Nanopass.ObligationDischarge.ofGraph graph
    let proofOperations = Alex.Traversal.SMTTransfer.operations proofObligations
    let proofText = Alex.Traversal.SMTTransfer.transfer proofObligations

    let coeffects : TransferCoeffects = {
        Platform = { TargetArch = arch; LinkedLibraries = linkedLibraries }
        TargetPlatform = targetPlatform
    }

    // Execute the current whole-graph Alex traversal, retaining actual source
    // occurrences. This is not a source invalidation/partition decision.
    match graph.DeclarationRoots with
    | [] -> Result.Error "No declaration roots found in PSG"
    | (entryId, _) :: _ ->
        match transferWithCorrespondence graph entryId coeffects intermediatesDir with
        | Result.Ok (topLevelOps, scope, definitions) ->
            // Preserve exactly what the PSG witnesses produced. Declaration
            // placement and target admission must be correct at their owners;
            // this boundary never repairs, drops or rewrites witnessed MLIR.
            let storageValidation =
                Alex.Traversal.StaticStorageValidation.validate graph topLevelOps
                |> Result.bind (fun () -> Alex.Traversal.StaticStorageValidation.validateWritable arch graph topLevelOps)
            match storageValidation with
            | Result.Error message -> Result.Error message
            | Result.Ok writableStorage ->
                // Serialize the portable module; retain these exact operations
                // at the backend boundary alongside this diagnostic artifact.
                // NPU uses unnamed module (MLIR-AIE expects `module { aie.device(...) { } }`)
                let mlirText =
                    match targetPlatform with
                    | Core.Types.Dialects.TargetPlatform.NPU ->
                        let opsText = opsToString arch.Pointer topLevelOps "  "
                        sprintf "module {\n%s\n}" opsText
                    | _ -> moduleToString arch.Pointer "main" topLevelOps

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
                        File.WriteAllText(smtPath, proofText)
                        if Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig.isVerbose() then
                            printfn "[Alex] Wrote SMT verification module: 09_obligations.mlir (%d obligations)" proofObligations.Length
                | None -> ()

                // XDC transfer — parallel residual from the pin facts the graph carries (FPGA only).
                // A design that declares no pins has no constraints; malformed pin facts fail here.
                let constraints =
                    match targetPlatform, codata.Pins with
                    | Core.Types.Dialects.TargetPlatform.FPGA, Some mapping ->
                        Alex.Traversal.XDCTransfer.transfer mapping
                        |> Result.map (fun xdcText ->
                            match intermediatesDir with
                            | Some dir ->
                                let xdcPath = Path.Combine(dir, "constraints.xdc")
                                File.WriteAllText(xdcPath, xdcText)
                                if Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig.isVerbose() then
                                    printfn "[Alex] Wrote XDC constraints: constraints.xdc (%d pins)" mapping.Pins.Length
                            | None -> ())
                    | _ -> Result.Ok ()

                match constraints with
                | Result.Error message -> Result.Error message
                | Result.Ok () ->
                    let activation =
                        match targetPlatform with
                        | Core.Types.Dialects.TargetPlatform.FPGA | Core.Types.Dialects.TargetPlatform.NPU -> Core.Types.WitnessArtifacts.TargetModuleActivation
                        | _ -> Core.Types.WitnessArtifacts.CheckedProgramStartup
                    let proof: Core.Types.WitnessArtifacts.ProofEnvelope =
                        { Scope = scope; Source = sourceProof; Obligations = proofObligations
                          Operations = proofOperations; Text = proofText; Mlir = None }
                    Core.WitnessArtifacts.createWithProof proof scope activation definitions topLevelOps mlirText writableStorage
                    |> Result.map (fun catalog ->
                        let witnessed =
                            { Operations = topLevelOps; PointerBits = arch.Pointer; Text = mlirText; WritableStorage = writableStorage
                              Catalog = Some catalog
                              ModuleName = if targetPlatform = Core.Types.Dialects.TargetPlatform.NPU then None else Some "main" }
                        witnessed, Set.union codata.WitnessEmission.Value.Boundary.Links linkedLibraries)
        | Result.Error msg -> Result.Error msg

/// Generate MLIR for the graph. A core's leg reads the declared Register and Pointer widths at
/// every boundary and layout site (Types.declaredWordWidth, declaredPointerBytes); a description
/// that declares neither cannot start it, and is refused here, before any witness runs, with
/// the code PlatformDeclaration reports for a missing declaration (CCS8203). The fabric leg reads
/// neither, so a description declaring none compiles for it. The deployment mode is the
/// project's; runtime and library requirements are settled by CCS/Baker.
let generateWithLinkedLibrariesAndProof
    (graph: SemanticGraph)
    (platformCtx: PlatformContext)
    (_deploymentMode: Core.Types.Dialects.DeploymentMode)
    (targetPlatform: Core.Types.Dialects.TargetPlatform)
    (intermediatesDir: string option)
    (linkedLibraries: Set<string>)
    (sourceProof: Core.Types.WitnessArtifacts.SourceProofReceipt option)
    : Result<BackEndInput * Set<string>, string> =
    // Refuse changed source input before any platform/codata reader or output.
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryRead graph with
    | Result.Error reason -> Result.Error ("Source emission admission: " + reason)
    | Result.Ok _ ->
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
        | None -> generateCore graph platformCtx targetPlatform intermediatesDir linkedLibraries sourceProof

/// Isolated witness callers may inspect the exact proof transcription without
/// executing tools. Production backend admission still requires source receipts.
let generateWithLinkedLibraries graph platformCtx deploymentMode targetPlatform intermediatesDir linkedLibraries =
    generateWithLinkedLibrariesAndProof graph platformCtx deploymentMode targetPlatform intermediatesDir linkedLibraries None

/// Callers without project link declarations retain the source-settled library requirements.
let generate graph platformCtx deploymentMode targetPlatform intermediatesDir =
    generateWithLinkedLibraries graph platformCtx deploymentMode targetPlatform intermediatesDir Set.empty
