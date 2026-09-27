/// Values: the names emission gives the values it emits for a node.
///
/// A value name is a pure derivation from the graph node it belongs to: `V (node, k)` is the
/// k-th value a witness emits on that node's behalf, `Arg i` a function's i-th block argument.
/// No pass assigns names, no witness holds a counter, and nothing here reads the shape of an
/// emission (how many values a pattern needs is the pattern's, satisfied by a family large enough
/// for any). The aliasing a name follows is structural: a lambda parameter is its argument, a
/// pattern binding over a field read is that read, an immutable binding that is not a module
/// value slot is its value. Every distinct family below is disjoint from every other.
module Alex.Traversal.Values

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types

/// The values a node may name for its own emission: ordinals 0 .. Family-1.
let [<Literal>] Family = 512

let value (nodeId: NodeId) (k: int) : SSA = V (NodeId.value nodeId, k)
let values (nodeId: NodeId) : SSA list = List.init Family (value nodeId)

/// The i-th meet of a consumer (Codata.Meets order).
let meetValue (consumer: NodeId) (i: int) : SSA = V (NodeId.value consumer, 1000 + i)
/// A lambda's return meet, the last value of its body's scope.
let returnMeetValue (lambdaId: NodeId) : SSA = V (NodeId.value lambdaId, 1100)
/// The zero a unit-typed function returns.
let unitReturnValue (lambdaId: NodeId) : SSA = V (NodeId.value lambdaId, 1101)
/// Code materialized alongside an occurrence's ordinary data/environment read.
let callableCode (nodeId: NodeId) : SSA = V (NodeId.value nodeId, 1200)
/// Each finite mutable-read arm has its own code value, with no bounded ordinal window.
let callableAlternative (nodeId: NodeId) alternative : SSA = CallableAlternative (NodeId.value nodeId, alternative)
/// Each array initializer position has a distinct physical index name.
let arrayElementIndex (nodeId: NodeId) element : SSA = ArrayElementIndex (NodeId.value nodeId, element)
/// The k-th value of a closure's callee prologue (capture extraction and env reconstruction).
let prologueValue (lambdaId: NodeId) (k: int) : SSA = V (NodeId.value lambdaId, 2000 + k)
/// The fixed work lanes of a settled continuation initializer/copy slot.
let continuationValue (nodeId: NodeId) (slot: int) (lane: int) : SSA = V (NodeId.value nodeId, 4000 + 16 * slot + lane)
/// The k-th value of a hardware module's body, per role (HardwareModuleWitness).
let hardwareValue (bindingId: NodeId) (k: int) : SSA = V (NodeId.value bindingId, 3000 + k)
/// The k-th value of an isolated solver scope (the SMT module's, not the program's).
let solverValue (k: int) : SSA = V (0, k)
/// A value no operation defines: the placeholder of a guard that cannot occur.
let undefined : SSA = V (-1, -1)

/// Runtime slot intent is an explicit Baker fact, independent of whether
/// its physical authority has settled. A missing premise cannot turn a slot
/// into an inline initializer at a later reference.
let isModuleValueSlot (_platform: Core.Types.Dialects.TargetPlatform) (graph: SemanticGraph) (node: SemanticNode) : bool =
    let projection =
        Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryStorage graph
        |> Result.defaultWith (fun reason ->
            invalidOp (sprintf "PSG settlement (WitnessEmission.Storage) did not publish the storage projection for binding %d: %s" (NodeId.value node.Id) reason))
    projection.Startup
    |> Option.exists (fun plan -> plan.ValueBindings.Contains node.Id)

let private callableFacts graph =
    Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryCallable graph
    |> Result.defaultWith (fun reason ->
        invalidOp ("PSG settlement (WitnessEmission.Callable) did not publish the callable value projection: " + reason))

/// Name the source-admitted alias endpoint or formal. Resolving aliases and
/// classifying parameter/environment conventions belong to source settlement.
/// Only the current emitted function supplies block arguments. A lexical outer
/// formal requires its witnessed capture; it is never another local Arg value.
let valuesOf (_platform: Core.Types.Dialects.TargetPlatform) (graph: SemanticGraph) (owners: NodeId list) (nodeId: NodeId) : SSA list =
    let facts = callableFacts graph
    match facts.AliasTargets.TryFind nodeId with
    | None -> invalidOp (sprintf "PSG settlement (WitnessEmission.Callable AliasTargets) did not settle an alias endpoint for source value %d" (NodeId.value nodeId))
    | Some target ->
        let components =
            owners |> List.tryHead |> Option.bind (fun owner -> facts.Arguments.TryFind owner |> Option.bind (Map.tryFind target))
        match components with
        | Some ordinals -> List.map Arg ordinals
        | None when facts.Arguments.Values |> Seq.exists (Map.containsKey target) ->
            invalidOp (sprintf "PSG settlement (WitnessEmission.Callable Arguments) did not settle an owner for source formal %d at this Huet occurrence (value %d)" (NodeId.value target) (NodeId.value nodeId))
        | None -> values target

/// The result value of a node: the last of its values.
let resultOf (platform: Core.Types.Dialects.TargetPlatform) (graph: SemanticGraph) (owners: NodeId list) (nodeId: NodeId) : SSA =
    match valuesOf platform graph owners nodeId with
    | [] -> invalidOp (sprintf "PSG settlement (OrdinaryDemand) omitted formal %d, which has no physical operand to recall" (NodeId.value nodeId))
    | [Arg ordinal] -> Arg ordinal
    | Arg _ :: _ -> invalidOp (sprintf "Alex recalled multi-component formal %d as one value; all of its physical operands must be recalled" (NodeId.value nodeId))
    | values -> List.last values

/// A unit-typed body: the function returns no value and its return needs a zero constant.
let isUnitTyped (graph: SemanticGraph) (nodeId: NodeId) : bool =
    (callableFacts graph).UnitNodes.Contains nodeId
