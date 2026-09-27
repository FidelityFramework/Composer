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
open Alex.Elements.ArithElements

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
let private pBoundaryAdapt (result: SSA) (adaptation: Meet option) (value: Val) : PSGParser<MLIROp list * Val> =
    parser {
        match adaptation with
        | None -> return [], value
        | Some meet ->
            let fromType, toType = TInt(IntWidth meet.From), TInt(IntWidth meet.To)
            do! ensure (value.Type = fromType) "Published boundary adaptation disagrees with the witnessed operand type."
            let! operation =
                match meet.Adapt with
                | MeetKind.ExtendSigned -> pExtSI result value.SSA fromType toType
                | MeetKind.ExtendUnsigned -> pExtUI result value.SSA fromType toType
                | MeetKind.Truncate -> pTruncI result value.SSA fromType toType
                | _ -> fail (Message "Published scalar boundary requires an admitted integer adaptation.")
            return [operation], { SSA = result; Type = toType }
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
                let rec arguments ordinal (operands: BoundaryOperand list) : PSGParser<MLIROp list * Val list> = parser {
                    match operands with
                    | [] -> return [], []
                    | operand :: rest ->
                        let! ssa, ty = pRecallNode operand.Actual
                        let! operations, value =
                            pBoundaryAdapt (V(NodeId.value node.Id, 2 + ordinal)) operand.Adaptation { SSA = ssa; Type = ty }
                        do! ensure (value.Type = boundaryScalarType operand.Abi) "Witnessed boundary argument disagrees with its published ABI."
                        let! remainingOps, values = arguments (ordinal + 1) rest
                        return operations @ remainingOps, value :: values
                }
                let! argumentOps, values = arguments 0 call.Arguments
                let resultSSA = V(NodeId.value node.Id, 0)
                let results = call.Result |> Option.map (fun ty -> { SSA = resultSSA; Type = boundaryScalarType ty }) |> Option.toList
                let! operation = pFuncCallResults results declaration.Symbol values
                match results with
                | [] -> return argumentOps @ [operation], TRVoid
                | [value] ->
                    let! resultOps, result = pBoundaryAdapt (V(NodeId.value node.Id, 1)) call.ResultAdaptation value
                    return argumentOps @ [operation] @ resultOps, TRValue result
                | _ -> return! fail (Message "Scalar boundary publication has an invalid result list.")
    }
