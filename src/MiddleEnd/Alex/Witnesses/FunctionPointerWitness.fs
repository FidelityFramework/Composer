/// Native callbacks consume the entry and signature settled by CCS.
module Alex.Witnesses.FunctionPointerWitness

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.CodeGeneration.TypeMapping
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.FunctionPointerPatterns

let private witness (ctx: WitnessContext) (node: SemanticNode) =
    let observe pattern prefix =
        match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Ok ((ops, result), _) -> { InlineOps = prefix @ ops; TopLevelOps = []; Result = result }
        | Result.Error message -> WitnessOutput.error message
    let unsettled phase message =
        WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "FunctionPointer") (Some phase) message
    match Map.tryFind node.Id ctx.Graph.Codata.Value.FunctionPointers with
    | Some (FunctionPointerPlan.Address (symbol, lambdaId)) ->
        match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryCallable ctx.Graph with
        | Result.Error reason -> unsettled "callback declaration" reason
        | Result.Ok projection ->
            match projection.Declarations.TryFind lambdaId with
            | Some declaration ->
                let types = declaration.Parameters |> List.map (fun (_, _, id) -> mapTypeAt id ctx)
                let resultType =
                    if projection.VoidCallbacks.Contains lambdaId then TVoid
                    else mapTypeAt declaration.Result ctx
                observe (pFunctionAddress node.Id symbol types resultType) []
            | None -> unsettled "callback declaration" "The callback entry has no source-published declaration."
    | Some (FunctionPointerPlan.Invoke (pointer, arguments, _, _)) ->
        match MLIRAccumulator.recallNode pointer ctx.Accumulator with
        | Some (pointerSSA, TIndex) ->
            let recalled = arguments |> List.map (fun id -> MLIRAccumulator.recallNode id ctx.Accumulator)
            if recalled |> List.exists Option.isNone then WitnessOutput.error "Native callback arguments were not witnessed."
            else
                match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryCallable ctx.Graph with
                | Result.Error reason ->
                    unsettled "pointer result"
                        $"Baker callable projection did not settle void-pointer membership for native callback pointer {NodeId.value pointer}: {reason}"
                | Result.Ok projection ->
                    let adapted = List.zip arguments (List.choose id recalled) |> List.map (fun (id, (ssa, ty)) -> adaptOperand ctx.Coeffects ctx.Graph node.Id id ssa ty)
                    let prefix = adapted |> List.collect (fun (ops, _, _) -> ops)
                    let values = adapted |> List.map (fun (_, ssa, ty) -> { SSA = ssa; Type = ty })
                    let physicalResult =
                        if projection.VoidPointers.Contains pointer then TVoid
                        else mapTypeAt node.Id ctx
                    observe (pFunctionPointerCall node.Id pointerSSA values physicalResult) prefix
        | _ -> WitnessOutput.error "Native callback pointer was not witnessed as a pointer-sized value."
    | None -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "FunctionPointer"; Witness = witness }
