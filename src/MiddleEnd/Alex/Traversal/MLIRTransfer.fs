/// Alex witnesses share one post-order Huet traversal. This boundary returns
/// its operations and actual declaration occurrences before flattening loses
/// the source path; it does not schedule independent witness traversals.

module Alex.Traversal.MLIRTransfer

open System.IO
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Alex.Traversal.TransferTypes
open Alex.Traversal.WitnessRegistry
open Alex.Traversal.NanopassArchitecture

// ═══════════════════════════════════════════════════════════════════════════
// PUBLIC API
// ═══════════════════════════════════════════════════════════════════════════

/// Transfer PSG to MLIR via nanopass execution
///
/// ARCHITECTURE:
/// 1. Build this transfer's target-selected witness registry
/// 2. Run every nanopass over one post-order traversal of the PSG
/// 3. Envelope collects the witness outputs
/// 4. Return cohesive MLIR graph
///
/// Registered witnesses handle their categories at the current occurrence.
let transferWithCorrespondence
    (graph: SemanticGraph)
    (entryNodeId: NodeId)
    (coeffects: TransferCoeffects)
    (intermediatesDir: string option)
    : Result<MLIROp list * Core.Types.WitnessArtifacts.SemanticScope * Core.Types.WitnessArtifacts.EmittedDefinition list, string> =

    let registry = createRegistry coeffects.TargetPlatform

    // Production admission requires the exact graph's source-settled projection.
    // Absent provenance or changed prepared roots cannot carry old demand authority.
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryRead graph,
          SemanticGraph.tryGetNode entryNodeId graph with
    | Result.Error reason, _ -> Result.Error ("Source emission admission: " + reason)
    | Result.Ok _, None ->
        Result.Error (sprintf "Entry node %d not found" (NodeId.value entryNodeId))
    | Result.Ok _, Some _ ->
        // Execute all nanopasses in single-phase traversal
        let accumulator = executeNanopasses registry graph coeffects intermediatesDir

        // Strip scope markers and prepare operations for output
        let cleanedOps =
            accumulator.AllOps
            |> List.rev

        // Write partial MLIR to intermediate file for debugging (even with errors)
        match intermediatesDir with
        | Some dir ->
            let mlirText = moduleToString coeffects.Platform.TargetArch.Pointer "main" cleanedOps
            let mlirPath = Path.Combine(dir, "07_output.mlir")
            File.WriteAllText(mlirPath, mlirText)
        | None -> ()

        // Check for errors accumulated during nanopass execution
        match accumulator.Errors with
        | [] ->
            // Success - return accumulated MLIR operations
            let scope = accumulator.WitnessScope |> Option.defaultWith (fun () -> Core.WitnessArtifacts.beginWholeGraphWitness graph)
            Result.Ok (cleanedOps, scope, List.rev accumulator.EmittedDefinitions)
        | errors ->
            // Errors occurred - format and report them (MLIR already written above)
            let formattedErrors = errors |> List.map Diagnostic.format |> String.concat "\n"
            Result.Error formattedErrors

/// Compatibility entry for isolated transfer consumers. The production
/// pipeline retains correspondence through transferWithCorrespondence.
let transfer graph entryNodeId coeffects intermediatesDir =
    transferWithCorrespondence graph entryNodeId coeffects intermediatesDir
    |> Result.map (fun (operations, _, _) -> operations, [])
