/// Passive witnessing of source-published foreign declarations and calls.
/// Declaration identity, scope, ABI and adaptations belong to CCS/Baker.
module Alex.Patterns.PlatformPatterns

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.FuncElements

module Publication = Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission

let private pBoundary : PSGParser<BoundaryEmissionProjection> =
    parser {
        let! state = getUserState
        match Publication.tryBoundary state.Graph with
        | Result.Ok boundary -> return boundary
        | Result.Error reason -> return! fail (Message reason)
    }

/// The source declaration is a leaf. Its import is witnessed at the published
/// module scope; its retained typechecking placeholder is not executable code.
let pBoundaryDeclaration : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! node = getCurrentNode
        let! boundary = pBoundary
        do! ensure (boundary.DeclarationLeaves.Contains node.Id) "This occurrence has no published boundary declaration-leaf contract."
        return [], TRVoid
    }

/// Witness exactly the imports assigned to this source scope. No call inventory,
/// signature discovery, symbol reconciliation or declaration hoisting occurs here.
let pBoundaryImports : PSGParser<MLIROp list> =
    parser {
        let! node = getCurrentNode
        let! boundary = pBoundary
        let imports = boundary.ByScope.TryFind node.Id |> Option.defaultValue []
        let rec witness (ids: NodeId list) : PSGParser<MLIROp list> = parser {
            match ids with
            | [] -> return []
            | id :: rest ->
                match boundary.Imports.TryFind id with
                | None -> return! fail (Message "Published import scope names an absent declaration.")
                | Some declaration ->
                    do! ensure (declaration.Scope = node.Id) "Published import has a different source owner scope."
                    let! operation = pPublishedFuncDecl declaration
                    let! following = witness rest
                    return operation :: following
        }
        return! witness imports
    }

/// The source owns whether an adaptation exists and its exact operation.
/// SSA naming and checking physical correspondence are witness bookkeeping.
let private pBoundaryAdapt (consumer: NodeId) (operand: NodeId) (adaptation: Meet option) (value: Val) : PSGParser<MLIROp list * Val> =
    parser {
        let! state = getUserState
        let settled = meetFor state.Graph consumer operand |> Option.map fst
        do! ensure (settled = adaptation) "Published boundary adaptation disagrees with its exact source numeric meet."
        match adaptation with
        | None -> return [], value
        | Some meet ->
            do! ensure (meet.Consumer = consumer && meet.Operand = operand) "Published boundary adaptation names different source participants."
            do! ensure (value.Type = TInt(IntWidth meet.From)) "Published boundary adaptation disagrees with the witnessed operand type."
            let! operations, result, resultType = pAdapt consumer operand value.SSA value.Type
            return operations, { SSA = result; Type = resultType }
    }

/// Witness a settled call. Missing publication is a refusal, never permission to
/// inspect a descriptor, infer marshalling from an MLIR type or create an import.
let pBoundaryCall : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! node = getCurrentNode
        let! boundary = pBoundary
        match boundary.Calls.TryFind node.Id with
        | None -> return! fail (Message $"Boundary call {NodeId.value node.Id} has no source-published contract.")
        | Some call ->
            match boundary.Imports.TryFind call.Import with
            | None -> return! fail (Message "Published boundary call names an absent import.")
            | Some declaration ->
                let rec arguments (witnessed: Map<NodeId, Meet option * Val * Val>) (operands: BoundaryOperand list) : PSGParser<MLIROp list * Val list> = parser {
                    match operands with
                    | [] -> return [], []
                    | operand :: rest ->
                        let! ssa, ty = pRecallNode operand.Actual
                        let raw = { SSA = ssa; Type = ty }
                        let! operations, value =
                            match witnessed.TryFind operand.Actual with
                            | Some (adaptation, previous, value) -> parser {
                                do! ensure (adaptation = operand.Adaptation && previous = raw) "Repeated boundary actual disagrees with its witnessed source meet or value."
                                return [], value
                              }
                            | None -> pBoundaryAdapt node.Id operand.Actual operand.Adaptation raw
                        do! ensure (value.Type = boundaryScalarType operand.Abi) "Witnessed boundary argument disagrees with its published ABI."
                        // Repeated positions retain their order and reuse the
                        // one canonical meet definition for this actual.
                        let! remainingOps, values = arguments (witnessed.Add(operand.Actual, (operand.Adaptation, raw, value))) rest
                        return operations @ remainingOps, value :: values
                }
                let! argumentOps, values = arguments Map.empty call.Arguments
                let! results =
                    match call.Result with
                    | None -> preturn []
                    | Some ty -> parser {
                        let! resultSSA = getNodeSSA node.Id
                        return [{ SSA = resultSSA; Type = boundaryScalarType ty }]
                      }
                let! operation = pFuncCallResults results declaration.Symbol values
                match results with
                | [] -> return argumentOps @ [operation], TRVoid
                | [value] ->
                    let! resultOps, result = pBoundaryAdapt node.Id node.Id call.ResultAdaptation value
                    return argumentOps @ [operation] @ resultOps, TRValue result
                | _ -> return! fail (Message "Scalar boundary publication has an invalid result list.")
    }
