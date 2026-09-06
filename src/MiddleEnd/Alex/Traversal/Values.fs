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
open Clef.Compiler.NativeTypedTree.UnionFind
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
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
/// The k-th value of a closure's callee prologue (capture extraction and env reconstruction).
let prologueValue (lambdaId: NodeId) (k: int) : SSA = V (NodeId.value lambdaId, 2000 + k)
/// The k-th value of a hardware module's body, per role (HardwareModuleWitness).
let hardwareValue (bindingId: NodeId) (k: int) : SSA = V (NodeId.value bindingId, 3000 + k)
/// The k-th value of an isolated solver scope (the SMT module's, not the program's).
let solverValue (k: int) : SSA = V (0, k)
/// A value no operation defines: the placeholder of a guard that cannot occur.
let undefined : SSA = V (-1, -1)

/// A module-level value binding realised as a program-lifetime slot: a module member with
/// EmissionStrategy.MainPrologue whose value is not a lambda (a core's realisation; fabric has no slot).
let isModuleValueSlot (platform: Core.Types.Dialects.TargetPlatform) (graph: SemanticGraph) (node: SemanticNode) : bool =
    match node.Kind with
    | SemanticKind.Binding _ when platform <> Core.Types.Dialects.TargetPlatform.FPGA && node.EmissionStrategy = EmissionStrategy.MainPrologue ->
        let isModuleMember = graph.ModuleClassifications.Value |> Map.exists (fun _ c -> List.contains node.Id c.ModuleInit)
        isModuleMember &&
        (match node.Children with
         | childId :: _ ->
             match SemanticGraph.tryGetNode childId graph with
             | Some { Kind = SemanticKind.Lambda _ } -> false
             | Some _ -> true
             | None -> false
         | [] -> false)
    | _ -> false

/// Whether a lambda takes an environment argument ahead of its parameters.
let takesEnvironment (lambda: SemanticNode) : bool =
    match lambda.Kind with
    | SemanticKind.Lambda (_, _, captures, _, _) ->
        not (List.isEmpty captures) ||
        (lambda.Metadata |> Map.tryFind ClosureMetadata.RequiresClosurePair |> Option.map (function MetadataValue.Bool b -> b | _ -> false) |> Option.defaultValue false)
    | _ -> false

/// The block argument a lambda parameter is, if the node is one.
let private argumentOf (graph: SemanticGraph) (node: SemanticNode) : SSA option =
    match node.Kind, node.Children with
    | SemanticKind.PatternBinding _, [] ->
        match node.Parent |> Option.bind (fun p -> SemanticGraph.tryGetNode p graph) with
        | Some ({ Kind = SemanticKind.Lambda (parameters, _, _, _, _) } as lambda) ->
            parameters
            |> List.tryFindIndex (fun (_, _, id) -> id = node.Id)
            |> Option.map (fun i -> Arg (i + (if takesEnvironment lambda then 1 else 0)))
        | _ -> None
    | _ -> None

/// The node whose values a node's name follows (itself where it aliases nothing).
let rec private aliasTarget (platform: Core.Types.Dialects.TargetPlatform) (graph: SemanticGraph) (node: SemanticNode) : SemanticNode =
    match node.Kind, node.Children with
    | SemanticKind.PatternBinding _, childId :: _ ->
        match SemanticGraph.tryGetNode childId graph with
        | Some child -> aliasTarget platform graph child
        | None -> node
    | SemanticKind.Binding (_, false, _, _), childId :: _ when not (isModuleValueSlot platform graph node) ->
        match SemanticGraph.tryGetNode childId graph with
        | Some child -> aliasTarget platform graph child
        | None -> node
    | _ -> node

/// The values a node names, following its aliases: a lambda parameter its argument alone.
let valuesOf (platform: Core.Types.Dialects.TargetPlatform) (graph: SemanticGraph) (nodeId: NodeId) : SSA list =
    match SemanticGraph.tryGetNode nodeId graph with
    | None -> values nodeId
    | Some node ->
        let target = aliasTarget platform graph node
        match argumentOf graph target with
        | Some arg -> [ arg ]
        | None -> values target.Id

/// The result value of a node: the last of its values.
let resultOf (platform: Core.Types.Dialects.TargetPlatform) (graph: SemanticGraph) (nodeId: NodeId) : SSA =
    valuesOf platform graph nodeId |> List.last

/// A unit-typed body: the function returns no value and its return needs a zero constant.
let isUnitTyped (ty: NativeType) : bool =
    let rec go t =
        match t with
        | NativeType.TApp ({ NTUKind = Some NTUKind.NTUunit }, []) -> true
        | NativeType.TVar tv ->
            match find tv with
            | (_, Some bound) -> go bound
            | (_, None) -> false
        | _ -> false
    go ty
