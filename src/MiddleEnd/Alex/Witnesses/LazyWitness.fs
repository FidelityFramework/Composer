/// Passive observation of Baker's explicit lazy storage and value primitives.
/// The existing Huet traversal observes the source guard/computation/store/
/// publication graph through ordinary control-flow and application witnesses.
module Alex.Witnesses.LazyWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ContinuationPatterns
open Alex.Patterns.LazyPatterns
open Alex.Patterns.LiteralPatterns
open XParsec
open XParsec.Parsers
open XParsec.Combinators
module Operands = Alex.Traversal.LazyOperands

let private failure (node: SemanticNode) phase reason =
    WitnessOutput.errorCoded AX4001 (Some node.Id) (Some "Lazy") (Some phase) reason

let private observe (ctx: WitnessContext) (node: SemanticNode) pattern =
    match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Result.Ok ((operations, result), _) ->
        { InlineOps = operations; TopLevelOps = MLIRAccumulator.drainPendingStaticGlobals ctx.Accumulator; Result = result }
    | Result.Error reason -> failure node "settled operands" reason

/// Formation consumes the source declarations as typed storage fields. They
/// have no executable initializer to traverse (the cache is uninitialized),
/// but a successfully observed layout accounts for their declaration identity.
let private formation (ctx: WitnessContext) (node: SemanticNode) (layout: LazyLayout) initializers =
    let output = observe ctx node (pCreateLazyEnvironment node.Id layout initializers)
    match output.Result with
    | TRValue _ ->
        ctx.GlobalVisited.Value <-
            ctx.GlobalVisited.Value |> Set.add layout.Computed |> Set.add layout.Cached
        output
    | _ -> output

let private access (ctx: WitnessContext) (node: SemanticNode) environment slotId borrow write =
    match Operands.layoutAt ctx environment with
    | None -> failure node "storage identity" "Lazy access has no current source layout and complete-use proof."
    | Some layout ->
        match layout.Slots |> List.tryFind (fun slot -> slot.Source = slotId) with
        | None -> failure node "slot identity" "Lazy access does not name a field of its actual instance."
        | Some slot ->
            let pattern =
                match write with
                | Some value -> pWithUnitResult node.Id (pWriteContinuationSlot node.Id environment value layout.Bytes slot)
                | None when borrow -> pBorrowContinuationSlot node.Id environment layout.Bytes slot
                | None -> pReadContinuationSlot node.Id environment layout.Bytes slot
            observe ctx node pattern

let private witness (ctx: WitnessContext) (node: SemanticNode) =
    match node.Kind with
    | SemanticKind.LazyRead(environment, slot) -> access ctx node environment slot false None
    | SemanticKind.LazyBorrow(environment, slot) -> access ctx node environment slot true None
    | SemanticKind.LazyWrite(environment, slot, value) -> access ctx node environment slot false (Some value)
    | SemanticKind.LazyEnvironment(owner, initializers) ->
        match Operands.layout ctx owner with
        | Some layout -> formation ctx node layout initializers
        | None -> failure node "formation" "Lazy formation lacks its complete typed storage contract."
    | SemanticKind.LazyAllocate owner ->
        match Operands.layout ctx owner with
        | Some layout -> observe ctx node (pAllocateLazyEnvironment node.Id layout)
        | None -> failure node "allocation" "Lazy allocation lacks its complete typed storage contract."
    | SemanticKind.LazyEnvironmentReference value ->
        match Operands.layoutAt ctx value with
        | Some layout -> observe ctx node (pRecallLazyEnvironment value layout)
        | None -> failure node "actual instance" "Lazy force lacks the settled actual environment operand."
    | SemanticKind.LazyValue(thunk, environment) ->
        match Operands.project ctx node.Id with
        | Result.Error reason -> failure node "value boundary" reason
        | Result.Ok shape ->
            let layout = Operands.contract shape
            if layout.Thunk <> thunk then failure node "thunk identity" "Lazy value and settled thunk disagree."
            else
                let symbol = Alex.CodeGeneration.CallableSymbols.lambda ctx.Graph ctx.Graph.Nodes[thunk] false
                observe ctx node (parser {
                    let! operations, result = pRecallLazyEnvironment environment layout
                    match result with
                    | TRValue environment ->
                        let! code, value = pLazyValue node.Id shape symbol environment
                        return operations @ code, value
                    | _ -> return! fail (Message "Lazy formation requires its actual witnessed environment.")
                })
    | SemanticKind.LazyExpr _ | SemanticKind.LazyForce _ ->
        failure node "source settlement" "Explicit lazy source operations require their Baker memoization and storage contracts."
    | _ -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "Lazy"; Witness = witness }
