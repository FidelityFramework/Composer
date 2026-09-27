/// Shared passive spatial declaration Pattern for hardware and kernel Witnesses.
module Alex.Patterns.SpatialPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.SpatialElements
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

let pPublishedSpatialModule : PSGParser<MLIROp list * TransferResult> = parser {
    let! state = getUserState
    do! ensure (state.Current.Id=state.Zipper.Focus.Id && obj.ReferenceEquals(state.Graph,state.Zipper.Graph))
               "Spatial declaration witnessing requires the exact current source occurrence and Huet zipper."
    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.trySpatial state.Graph with
    | Result.Error reason -> return! fail (Message reason)
    | Result.Ok projection ->
        let plan =
            match projection.Hardware.TryFind state.Current.Id,projection.Kernels.TryFind state.Current.Id with
            | Some plan,None -> Some(SpatialModuleWitness.Hardware plan)
            | None,Some plan -> Some(SpatialModuleWitness.Kernel plan)
            | _ -> None
        match plan with
        | None -> return! fail (Message "This occurrence lacks its unique source-published spatial module plan.")
        | Some plan ->
            let! operation = pSpatialModule plan
            return [operation],TRVoid
}
