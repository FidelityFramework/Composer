/// Passive physical spelling of immutable representations published by Baker.
/// This module has no native type resolver, range selector or target state.
module Alex.CodeGeneration.TypeMapping

open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types

let numericProjection (graph: SemanticGraph) =
    Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryNumeric graph
    |> Result.defaultWith (fun reason -> failwith ("TypeMapping: source Numeric publication is unavailable: " + reason))

let scalarCarrierAt (graph: SemanticGraph) (nodeId: NodeId) : ScalarCarrier option =
    let publication = numericProjection graph
    match publication.Values.TryFind nodeId with
    | Some carrier -> Some carrier
    | None ->
        match publication.Unresolved.TryFind nodeId with
        | Some reason -> failwithf "TypeMapping: Numeric carrier at node %d is unresolved: %s" (NodeId.value nodeId) reason
        | None when publication.Required.Contains nodeId ->
            failwithf "TypeMapping: source Numeric publication omitted the required carrier at node %d" (NodeId.value nodeId)
        | None -> None

let settledScalarType (slot: SettledSlot) : MLIRType option = SettledScalar.tryType slot

let scalarCarrierType (carrier: ScalarCarrier) : MLIRType =
    settledScalarType carrier.Slot
    |> Option.defaultWith (fun () -> failwithf "TypeMapping: source carrier at node %d has no admitted scalar physical form: %A" (NodeId.value carrier.Site) carrier.Slot)

let rec representationType = function
    | ValueRepresentation.Scalar slot ->
        settledScalarType slot |> Option.defaultWith (fun () -> failwithf "TypeMapping: source scalar form has no physical spelling: %A" slot)
    | ValueRepresentation.Buffer(None, element) -> TMemRef(representationType element)
    | ValueRepresentation.Buffer(Some count, element) when count >= 0 -> TMemRefStatic(count, representationType element)
    | ValueRepresentation.Buffer(Some count, _) -> failwithf "TypeMapping: source buffer has negative extent %d" count
    | ValueRepresentation.Record(fields, placement) ->
        let bytes = placement |> Option.map (fun (offsets, size, alignment) -> { Offsets = offsets; Size = size; Align = alignment })
        TStruct(fields |> List.map (fun (name, form) -> name, representationType form), bytes)
    | ValueRepresentation.Tag cases when cases > 0 -> TTag cases
    | ValueRepresentation.Tag _ -> failwith "TypeMapping: source tag has no cases"

let private read what = function
    | Some(Ok form) -> representationType form
    | Some(Error reason) -> failwithf "TypeMapping: %s has no source-published representation: %s" what reason
    | None -> failwithf "TypeMapping: source publication omitted %s" what

let valueTypeAt (graph: SemanticGraph) (nodeId: NodeId) =
    (numericProjection graph).OccurrenceRepresentations.TryFind nodeId
    |> read (sprintf "value occurrence %d" (NodeId.value nodeId))

let typeOfIdentity (graph: SemanticGraph) (identity: TypeIdentity) =
    (numericProjection graph).TypeRepresentations.TryFind identity |> read "type identity"

let sourceTypeAt (graph: SemanticGraph) (nodeId: NodeId) =
    (numericProjection graph).SourceTypes.TryFind nodeId
    |> Option.defaultWith (fun () -> failwithf "TypeMapping: source publication omitted type identity at %d" (NodeId.value nodeId))

let settledLayoutFor (graph: SemanticGraph) (identity: TypeIdentity) =
    (numericProjection graph).Layouts.TryFind identity

let settledLayoutAt graph nodeId = settledLayoutFor graph (sourceTypeAt graph nodeId)

let nodeWidth graph nodeId =
    match scalarCarrierAt graph nodeId with
    | Some { Slot = SettledSlot.Integer(bits, _) } when bits > 0 -> Some(IntWidth bits)
    | Some { Slot = SettledSlot.Integer _ } -> failwith "TypeMapping: source integer width is not positive"
    | _ -> None

let requireNodeWidth graph nodeId =
    nodeWidth graph nodeId |> Option.defaultWith (fun () -> failwithf "TypeMapping: no published integer carrier at %d" (NodeId.value nodeId))

/// A container's physical byte view follows its published aggregate placement.
let physicalStorageType (_arch: Architecture) = function
    | TStruct(_, Some bytes) -> TMemRefStatic(bytes.Size, TInt(IntWidth 8))
    | TStruct _ -> failwith "TypeMapping: an aggregate storage view requires complete source byte placement"
    | ty -> ty
