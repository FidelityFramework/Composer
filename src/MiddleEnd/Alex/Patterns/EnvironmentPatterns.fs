/// Ordinary closure environments use the same placed unboxed fields as
/// continuations. All identities, extents and residence are supplied by Baker.
module Alex.Patterns.EnvironmentPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ContinuationPatterns
open Alex.Patterns.MemoryPatterns
module Values = Alex.Traversal.Values

let private pEnvironmentType (layout: EnvironmentLayout) = parser {
    let! state = getUserState
    do! ensure (layout.Bytes >= 0 && layout.Alignment > 0 && (layout.Bytes > 0 || layout.Alignment = 1))
            $"Environment {NodeId.value layout.Owner} has no settled extent and alignment"
    do! ensure (not layout.Obligations.IsEmpty && (layout.Obligations |> List.forall (fun id ->
        match state.Graph.Nodes |> Map.tryFind id with
        | Some { Kind = SemanticKind.Obligation _ } -> true
        | _ -> false))) $"Environment {NodeId.value layout.Owner} has no resident layout obligations"
    return TMemRefStatic(layout.Bytes, TInt(IntWidth 8))
}

/// Recall this occurrence's environment. The known code identity does not
/// select or recreate an environment from another formation of that code.
let pRecallEnvironment source (layout: EnvironmentLayout) = parser {
    let! state = getUserState
    do! ensure ((state.Graph.Codata.Value.EnvironmentOrigins |> Map.tryFind source) = Some layout.Owner)
            $"Environment occurrence {NodeId.value source} lacks its exact layout identity"
    let! expected = pEnvironmentType layout
    let! value, actual = pRecallNode source
    do! ensure (actual = expected) $"Environment occurrence {NodeId.value source} lacks its settled carrier"
    return [], TRValue { SSA = value; Type = expected }
}

let pAllocateEnvironment nodeId (layout: EnvironmentLayout) = parser {
    let! state = getUserState
    let! ty = pEnvironmentType layout
    let ssa = Values.value nodeId 0
    let! allocations =
        match state.Graph.Codata.Value.Escapes |> Map.tryFind nodeId with
        | Some EscapeKind.StackScoped -> parser {
            let! operations, result = pAllocateContinuationStorage nodeId layout.Bytes layout.Alignment
            do! ensure (match result with TRValue value -> value.SSA = ssa && value.Type = ty | _ -> false)
                    $"Environment {NodeId.value nodeId} allocation disagrees with its settled carrier"
            return operations
          }
        | Some EscapeKind.StaticLifetime -> parser {
            let! allocation = pAllocValue nodeId ssa ty
            return [allocation]
          }
        | _ -> fail (Message $"Environment {NodeId.value nodeId} requires an admitted allocation residence")
    return allocations, TRValue { SSA = ssa; Type = ty }
}

let pCreateEnvironment nodeId (layout: EnvironmentLayout) initializers = parser {
    let! state = getUserState
    let! ty = pEnvironmentType layout
    let! allocations, result =
        match state.Graph.Codata.Value.EnvironmentDestinations |> Map.tryFind nodeId with
        | Some destination -> pRecallEnvironment destination layout
        | None -> pAllocateEnvironment nodeId layout
    let! ssa =
        match result with
        | TRValue value when value.Type = ty -> preturn value.SSA
        | _ -> fail (Message $"Environment {NodeId.value nodeId} destination has no settled carrier")
    let! stores = pInitializeEnvironmentSlots nodeId ssa ty layout.Bytes layout.Slots initializers
    return allocations @ stores, result
}
