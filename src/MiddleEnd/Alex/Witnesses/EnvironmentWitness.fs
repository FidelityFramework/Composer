/// Passive observation of Baker's materialized callable environments.
module Alex.Witnesses.EnvironmentWitness

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ContinuationPatterns
open Alex.Patterns.EnvironmentPatterns
open Alex.Patterns.LiteralPatterns

let private failure (node: SemanticNode) phase message =
    WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "Environment") (Some phase) message

let private observe (ctx: WitnessContext) (node: SemanticNode) pattern =
    match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Result.Ok ((operations, result), _) ->
        { InlineOps = operations; TopLevelOps = MLIRAccumulator.drainPendingStaticGlobals ctx.Accumulator; Result = result }
    | Result.Error message -> failure node "settled operands" message

let private layoutAt (ctx: WitnessContext) source =
    ctx.Graph.Codata.Value.EnvironmentOrigins |> Map.tryFind source
    |> Option.bind (fun owner -> ctx.Graph.Codata.Value.EnvironmentLayouts |> Map.tryFind owner)

let private access (ctx: WitnessContext) (node: SemanticNode) environment slotId borrow write =
    match layoutAt ctx environment with
    | None -> failure node "storage identity" $"Environment operand {NodeId.value environment} has no settled layout"
    | Some layout ->
        match layout.Slots |> List.tryFind (fun slot -> slot.Source = slotId) with
        | None -> failure node "slot identity" $"Environment {NodeId.value layout.Owner} has no slot {NodeId.value slotId}"
        | Some slot ->
            let pattern =
                match write with
                | Some value -> pWithUnitResult node.Id (pWriteContinuationSlot node.Id environment value layout.Bytes slot)
                | None when borrow -> pBorrowContinuationSlot node.Id environment layout.Bytes slot
                | None -> pReadContinuationSlot node.Id environment layout.Bytes slot
            observe ctx node pattern

let private witness (ctx: WitnessContext) (node: SemanticNode) =
    match node.Kind with
    | SemanticKind.EnvironmentRead(environment, slot) -> access ctx node environment slot false None
    | SemanticKind.EnvironmentBorrow(environment, slot) -> access ctx node environment slot true None
    | SemanticKind.EnvironmentWrite(environment, slot, value) -> access ctx node environment slot false (Some value)
    | SemanticKind.EnvironmentCreate(owner, initializers) ->
        match ctx.Graph.Codata.Value.EnvironmentLayouts |> Map.tryFind owner with
        | Some layout -> observe ctx node (pCreateEnvironment node.Id layout initializers)
        | None -> failure node "allocation layout" $"Environment {NodeId.value owner} has no settled allocation layout"
    | SemanticKind.EnvironmentAllocate owner ->
        match ctx.Graph.Codata.Value.EnvironmentLayouts |> Map.tryFind owner with
        | Some layout -> observe ctx node (pAllocateEnvironment node.Id layout)
        | None -> failure node "allocation layout" $"Environment {NodeId.value owner} has no settled allocation layout"
    | SemanticKind.ClosureValue(implementation, environment) ->
        match ctx.Graph.Codata.Value.KnownCallables |> Map.tryFind node.Id, layoutAt ctx environment with
        | Some callable, Some layout when callable.Implementation = implementation
                                         && callable.EnvironmentOwner = layout.Owner
                                         && layout.Implementation = implementation ->
            observe ctx node (pRecallEnvironment environment layout)
        | _ -> failure node "callable identity" $"Callable {NodeId.value node.Id} has no settled code/environment relationship"
    | SemanticKind.EnvironmentReference callable ->
        match ctx.Graph.Codata.Value.KnownCallables |> Map.tryFind callable, layoutAt ctx callable with
        | Some known, Some layout when known.EnvironmentOwner = layout.Owner && known.Implementation = layout.Implementation ->
            observe ctx node (pRecallEnvironment callable layout)
        | _ -> failure node "callable occurrence" $"Callable occurrence {NodeId.value callable} has no settled environment"
    | _ -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "Environment"; Witness = witness }
