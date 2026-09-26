/// Typed storage access for Baker-settled continuation frames. Slot identity,
/// placement and residence belong to the graph; these patterns only observe them.
module Alex.Patterns.ContinuationPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.XParsec.PSGCombinators
open Alex.CodeGeneration.TypeMapping
open Alex.Elements.MLIRAtomics
open Alex.Elements.MemRefElements
open Alex.Elements.FuncElements
module Values = Alex.Traversal.Values
module Sequences = Alex.Traversal.SequenceOperands

let private pSlotType (slot: ContinuationSlot) : PSGParser<MLIRType> = parser {
    let! state = getUserState
    let mapped native =
        mapNativeTypeForTarget state.Coeffects.TargetPlatform state.Platform.TargetArch state.Graph native
        |> narrowType state.Coeffects state.Graph slot.Source
    match slot.Holds with
    | CaptureSlotKind.Scalar settled ->
        do! ensure (slot.Field.Slot = settled) $"Continuation slot {NodeId.value slot.Source} disagrees with its settled scalar field"
        match settled with
        | SettledSlot.Integer(bits, _) when bits > 0 -> return TInt(IntWidth bits)
        | SettledSlot.Bool -> return TInt(IntWidth 1)
        | SettledSlot.Char | SettledSlot.Unit -> return TInt(IntWidth 32)
        | SettledSlot.Real 32 -> return TFloat F32
        | SettledSlot.Real 64 -> return TFloat F64
        | SettledSlot.Pointer 1 -> return TIndex
        | _ -> return! fail (Message $"Continuation slot {NodeId.value slot.Source} has unsupported scalar representation {settled}")
    | CaptureSlotKind.CellView payload ->
        do! ensure (slot.Field.Slot = SettledSlot.Pointer 5) $"Continuation cell {NodeId.value slot.Source} requires a complete descriptor field"
        return TMemRefStatic(1, mapped payload)
    | CaptureSlotKind.InlineValue native ->
        match settledLayout state.Graph native with
        | Some(SettledLayout.Union(_, _, Some bytes, Some alignment)) when bytes > 0 && alignment > 0 ->
            let consistent = not slot.IsCapture && slot.Field.Slot = SettledSlot.InlineBytes(bytes, alignment)
                             && slot.Field.Size = Some bytes && slot.Field.Align = Some alignment
            do! ensure consistent $"Continuation aggregate {NodeId.value slot.Source} disagrees with its owned union region"
            return TMemRefStatic(bytes, TInt(IntWidth 8))
        | _ -> return! fail (Message $"Continuation aggregate {NodeId.value slot.Source} has no settled union layout")
    | CaptureSlotKind.ValueView native ->
        do! ensure (slot.Field.Slot = SettledSlot.Pointer 5) $"Continuation value {NodeId.value slot.Source} requires a complete descriptor field"
        let carrier =
            match state.Graph.Codata.Value.SequenceOrigins |> Map.tryFind slot.Source with
            | Some owner ->
                state.Graph.Codata.Value.ContinuationFrames |> Map.tryFind owner
                |> Option.map (fun frame -> TMemRefStatic(frame.Bytes, TInt(IntWidth 8)))
            | None -> None
        let! ty =
            match carrier, native with
            | Some ty, _ -> preturn ty
            | None, (NativeType.TSeq _ | NativeType.TSeqEnumerator _) -> fail (Message $"Continuation value {NodeId.value slot.Source} has no settled sequence origin")
            | _ -> preturn (mapped native)
        match ty with
        | TMemRef _ | TMemRefStatic _ | TStruct(_, Some _) -> return ty
        | _ -> return! fail (Message $"Continuation value {NodeId.value slot.Source} has no settled descriptor carrier")
    | CaptureSlotKind.EnvironmentView owner ->
        do! ensure (slot.Field.Slot = SettledSlot.Pointer 5) $"Environment capture {NodeId.value slot.Source} requires a complete descriptor field"
        match state.Graph.Codata.Value.EnvironmentLayouts |> Map.tryFind owner with
        | Some layout when layout.Bytes >= 0 && layout.Alignment > 0 ->
            return TMemRefStatic(layout.Bytes, TInt(IntWidth 8))
        | _ -> return! fail (Message $"Environment capture {NodeId.value slot.Source} has no settled layout {NodeId.value owner}")
    | _ -> return! fail (Message $"Continuation slot {NodeId.value slot.Source} requires a scalar or complete descriptor representation")
}

let private pSlotPlacement bytes (slot: ContinuationSlot) = parser {
    let! offset =
        match slot.Field.Offset, slot.Field.Size, slot.Field.Align with
        | Some offset, Some size, Some alignment when offset >= 0 && offset <= bytes && size > 0 && alignment > 0 && offset % alignment = 0 && size <= bytes - offset -> preturn offset
        | _ -> fail (Message $"Continuation slot {NodeId.value slot.Source} has no complete placement within its {bytes}-byte frame")
    let! fieldType = pSlotType slot
    return offset, fieldType
}

let private pPlacedSlot frameId bytes (slot: ContinuationSlot) = parser {
    let! frameSSA, frameType = pRecallNode frameId
    do! ensure (frameType = TMemRefStatic(bytes, TInt(IntWidth 8))) $"Continuation frame {NodeId.value frameId} does not have its settled {bytes}-byte carrier"
    let! offset, fieldType = pSlotPlacement bytes slot
    return frameSSA, frameType, offset, fieldType
}

/// A mutable capture reads its original cell through the stored descriptor.
/// Scalar and buffer-backed values are loaded directly at their placed field.
let pReadContinuationSlot nodeId frameId bytes (slot: ContinuationSlot) : PSGParser<MLIROp list * TransferResult> = parser {
    let! frameSSA, frameType, offset, fieldType = pPlacedSlot frameId bytes slot
    let s = Values.value nodeId
    let descriptor = match slot.Holds with CaptureSlotKind.CellView _ -> s 1 | _ -> s 0
    let! load =
        match slot.Holds with
        | CaptureSlotKind.InlineValue _ -> parser {
            let! index = pConstI (s 2) (int64 offset) TIndex
            let! view = pMemRefView descriptor frameSSA (s 2) frameType fieldType
            return [index; view]
          }
        | _ -> pTypedExtractView descriptor frameSSA offset (s 2) (s 3) (s 4) fieldType frameType
    let! operations, valueType =
        match slot.Holds, fieldType with
        | CaptureSlotKind.CellView _, TMemRefStatic(1, payloadType) -> parser {
            let! zero = pConstI (s 5) 0L TIndex
            let! state = getUserState
            MLIRAccumulator.registerSSAType descriptor fieldType state.Accumulator
            let! scalar = pLoadFrom (s 0) descriptor [s 5] payloadType
            return load @ [zero; scalar], payloadType
          }
        | _ -> preturn (load, fieldType)
    let! adaptations, result, resultType = pAdapt nodeId nodeId (s 0) valueType
    return operations @ adaptations, TRValue { SSA = result; Type = resultType }
}

/// Borrow the cell explicitly named by Baker. Scalar slots expose their placed
/// view; a captured cell returns its stored descriptor. No value becomes a cell.
let pBorrowContinuationSlot nodeId frameId bytes (slot: ContinuationSlot) : PSGParser<MLIROp list * TransferResult> = parser {
    let! frameSSA, frameType, offset, fieldType = pPlacedSlot frameId bytes slot
    let s = Values.value nodeId
    match slot.Holds with
    | CaptureSlotKind.CellView _ ->
        let! read = pTypedExtractView (s 0) frameSSA offset (s 2) (s 3) (s 4) fieldType frameType
        return read, TRValue { SSA = s 0; Type = fieldType }
    | CaptureSlotKind.Scalar _ ->
        let viewType = TMemRefStatic(1, fieldType)
        let! index = pConstI (s 1) (int64 offset) TIndex
        let! view = pMemRefView (s 0) frameSSA (s 1) frameType viewType
        return [index; view], TRValue { SSA = s 0; Type = viewType }
    | _ -> return! fail (Message $"Continuation slot {NodeId.value slot.Source} has no admitted mutable-cell borrow representation")
}

/// Writes through a captured mutable cell, preserving its allocation identity.
/// Other slots store the provided value at the graph's exact physical carrier.
let pWriteContinuationSlot nodeId frameId valueId bytes (slot: ContinuationSlot) : PSGParser<MLIROp list * TransferResult> = parser {
    let! frameSSA, frameType, offset, fieldType = pPlacedSlot frameId bytes slot
    let! state = getUserState
    let! rawValue, rawType =
        match slot.Holds with
        | CaptureSlotKind.EnvironmentView owner ->
            match MLIRAccumulator.recallCallable valueId state.Accumulator with
            | Some callable ->
                match Alex.Traversal.CallableOperands.exactCarrier callable |> Option.bind _.Environment, callable.Environment with
                | Some contract, Some environment when contract.Owner = owner -> preturn (environment.SSA, environment.Type)
                | _ -> fail (Message $"Environment store {NodeId.value valueId} lacks its actual instance")
            | None ->
                match state.Graph.Nodes.TryFind valueId, state.Graph.Codata.Value.EnvironmentOrigins.TryFind valueId with
                | Some node, Some actualOwner when actualOwner = owner &&
                    Clef.Compiler.NativeTypedTree.UnionFind.applySubst node.Type =
                        Clef.Compiler.PSGSaturation.SemanticGraph.ClosureEnvironments.environmentType -> parser {
                    let! environment, actual = pRecallNode valueId
                    do! ensure (actual = fieldType) $"Environment descriptor {NodeId.value valueId} disagrees with its exact placed environment view"
                    return environment, actual
                  }
                | _ -> fail (Message $"Environment store {NodeId.value valueId} has neither witnessed callable operands nor its exact environment descriptor")
        | CaptureSlotKind.ValueView (NativeType.TSeq _ | NativeType.TSeqEnumerator _) ->
            match MLIRAccumulator.recallSequence valueId state.Accumulator with
            | Some sequence when state.Graph.Codata.Value.SequenceOrigins.TryFind valueId |> Option.exists (fun owner -> sequence.Flow.Owners = Set.singleton owner) ->
                preturn (sequence.Environment.SSA, sequence.Environment.Type)
            | None when
                Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.valueShape state.Graph state.Graph.Nodes[valueId] = CallableValueShape.Data valueId &&
                Clef.Compiler.PSGSaturation.SemanticGraph.CallableCarriers.valueShape state.Graph state.Graph.Nodes[slot.Source] = CallableValueShape.Data slot.Source &&
                state.Graph.Codata.Value.SequenceOrigins.TryFind valueId = state.Graph.Codata.Value.SequenceOrigins.TryFind slot.Source &&
                state.Graph.Codata.Value.SequenceOrigins.ContainsKey valueId -> pRecallNode valueId
            | _ -> fail (Message "Descriptor-only sequence store requires the exact source-proved function half and actual environment")
        | _ -> pRecallNode valueId
    let! adaptations, value, valueType = pAdapt nodeId valueId rawValue rawType
    let s = Values.value nodeId
    match slot.Holds, fieldType with
    | CaptureSlotKind.InlineValue _, _ ->
        return! fail (Message $"Continuation aggregate {NodeId.value slot.Source} requires Baker's explicit selected-case copy")
    | CaptureSlotKind.CellView _, TMemRefStatic(1, payloadType) ->
        do! ensure (valueType = payloadType) $"Continuation cell {NodeId.value slot.Source} write lacks its settled operand representation: actual {valueType}, expected {payloadType}"
        let! descriptor = pTypedExtractView (s 1) frameSSA offset (s 2) (s 3) (s 4) fieldType frameType
        let! zero = pConstI (s 5) 0L TIndex
        let! write = pStore value (s 1) [s 5] payloadType fieldType
        return adaptations @ descriptor @ [zero; write], TRVoid
    | _ ->
        let! viewOps, stored =
            match slot.Holds, valueType, fieldType with
            | CaptureSlotKind.ValueView _, TMemRefStatic(_, element), TMemRef expected when element = expected -> parser {
                let! cast = pMemRefCast (s 6) value valueType fieldType
                return [cast], s 6
              }
            | _ when valueType = fieldType -> preturn ([], value)
            | _ -> fail (Message $"Continuation slot {NodeId.value slot.Source} write lacks its settled operand representation: actual {valueType}, expected {fieldType}")
        let! write = pTypedInsertView frameSSA stored offset (s 2) (s 3) (s 4) fieldType frameType
        return adaptations @ viewOps @ write, TRVoid
}

/// Transient storage is explicitly graph-created at the MoveNext activation.
let pAllocateContinuationStorage nodeId bytes alignment : PSGParser<MLIROp list * TransferResult> = parser {
    do! ensure (bytes >= 0 && alignment > 0 && (bytes > 0 || alignment = 1)) $"Continuation storage {NodeId.value nodeId} has no settled extent and alignment"
    let result = Values.value nodeId 0
    let! allocation = pAlloca result bytes (TInt(IntWidth 8)) (Some alignment)
    return [allocation], TRValue { SSA = result; Type = TMemRefStatic(bytes, TInt(IntWidth 8)) }
}

/// Call the independently identified generator with its typed environment.
/// The function half was elided by Baker's per-use origin evidence.
let pMoveNext nodeId frameId (frame: ContinuationFrame) (symbol: string) : PSGParser<MLIROp list * TransferResult> = parser {
    let! environment, actual = pRecallNode frameId
    let expected = TMemRefStatic(frame.Bytes, TInt(IntWidth 8))
    do! ensure (actual = expected) $"Continuation generator {NodeId.value frame.Generator} received a frame without its settled carrier"
    let result = Values.value nodeId 0
    let! call = pFuncCall (Some result) symbol [{ SSA = environment; Type = expected }] (TInt(IntWidth 1))
    return [call], TRValue { SSA = result; Type = TInt(IntWidth 1) }
}

let private pFrameType (frame: ContinuationFrame) = parser {
    let! state = getUserState
    do! ensure (frame.Bytes > 0 && frame.Alignment > 0) $"Continuation {NodeId.value frame.Owner} has no settled frame extent and alignment"
    do! ensure (not frame.Obligations.IsEmpty) $"Continuation {NodeId.value frame.Owner} has no resident allocation obligations"
    do! ensure (frame.Obligations |> List.forall (fun id ->
        match state.Graph.Nodes |> Map.tryFind id with
        | Some { Kind = SemanticKind.Obligation _ } -> true
        | _ -> false)) $"Continuation {NodeId.value frame.Owner} references a missing allocation obligation"
    return TMemRefStatic(frame.Bytes, TInt(IntWidth 8))
}

let private pRecallContinuationValue sourceId = parser {
    let! state = getUserState
    match MLIRAccumulator.recallSequence sourceId state.Accumulator, MLIRAccumulator.recallNode sourceId state.Accumulator with
    | Some sequence, _ ->
        do! ensure (state.Graph.Codata.Value.SequenceOrigins.TryFind sourceId |> Option.exists (fun owner ->
                        sequence.Flow.Owners = Set.singleton owner))
                "A descriptor-only continuation capture requires the source-proved unique function half"
        return sequence.Environment.SSA, sequence.Environment.Type
    | None, Some value -> return value
    | None, None ->
        // A formal can be used before its first read. Its assigned argument
        // SSA/type already belongs to this function; no source walk is needed.
        do! ensure (match state.Graph.Nodes |> Map.tryFind sourceId with Some { Kind = SemanticKind.PatternBinding _ } -> true | _ -> false) $"Continuation operand {NodeId.value sourceId} is not yet witnessed"
        let! assigned = getNodeSSA sourceId
        match MLIRAccumulator.recallSSAType assigned state.Accumulator with
        | Some ty -> return assigned, ty
        | None -> return! fail (Message $"Continuation formal {NodeId.value sourceId} has no registered carrier")
}

let private pAllocateFrame nodeId (frame: ContinuationFrame) = parser {
    let! state = getUserState
    let! ty = pFrameType frame
    let ssa = Values.value nodeId 0
    match state.Graph.Codata.Value.ContinuationRegions |> Map.tryFind nodeId with
    | Some region ->
        do! ensure (region.ChildOwner = frame.Owner && region.Bytes = frame.Bytes && region.Alignment = frame.Alignment) $"Continuation region {NodeId.value nodeId} disagrees with its settled child frame"
        let! parent =
            match state.Graph.Codata.Value.ContinuationFrames |> Map.tryFind region.ParentOwner with
            | Some parent -> preturn parent
            | None -> fail (Message $"Continuation region {NodeId.value nodeId} references missing parent frame {NodeId.value region.ParentOwner}")
        do! ensure (parent.Owner = region.ParentOwner && parent.Formal = region.ParentFormal) $"Continuation region {NodeId.value nodeId} disagrees with its settled parent identity"
        let! parentType = pFrameType parent
        do! ensure (region.Offset >= 0 && region.Offset % region.Alignment = 0
                    && parent.Alignment % region.Alignment = 0
                    && int64 region.Offset + int64 region.Bytes <= int64 parent.Bytes) $"Continuation region {NodeId.value nodeId} lacks aligned containment within its parent frame"
        let! parentSSA, actual = pRecallContinuationValue region.ParentFormal
        do! ensure (actual = parentType) $"Continuation region {NodeId.value nodeId} parent formal lacks its settled frame carrier"
        let offsetSSA = Values.value nodeId 1
        let! offset = pConstI offsetSSA (int64 region.Offset) TIndex
        let! view = pMemRefView ssa parentSSA offsetSSA parentType ty
        return [offset; view], ssa, ty
    | None ->
        let! allocation =
            match state.Graph.Codata.Value.Escapes |> Map.tryFind nodeId with
            | Some EscapeKind.StackScoped -> pAlloca ssa frame.Bytes (TInt(IntWidth 8)) (Some frame.Alignment)
            | Some EscapeKind.StaticLifetime -> Alex.Patterns.MemoryPatterns.pAllocValue nodeId ssa ty
            | Some (EscapeKind.EscapesViaReturn | EscapeKind.EscapesViaClosure _ | EscapeKind.EscapesViaByRef) ->
                pAllocStatic ssa frame.Bytes (TInt(IntWidth 8)) (Some frame.Alignment)
            | None -> fail (Message $"Continuation allocation {NodeId.value nodeId} has no settled residence")
        return [allocation], ssa, ty
}

/// Caller-owned frame allocation carries no initialized state or captures.
/// The constructor writes them through the supplied typed destination.
let pAllocateContinuationFrame nodeId (frame: ContinuationFrame) : PSGParser<MLIROp list * TransferResult> = parser {
    let! allocation, value, ty = pAllocateFrame nodeId frame
    return allocation, TRValue { SSA = value; Type = ty }
}

let private pInitializeState nodeId frameSSA frameType (frame: ContinuationFrame) = parser {
    let! slot =
        match frame.Slots |> List.tryFind (fun slot -> slot.Source = frame.State) with
        | Some slot -> preturn slot
        | None -> fail (Message $"Continuation {NodeId.value frame.Owner} has no settled state slot")
    let! offset, ty = pSlotPlacement frame.Bytes slot
    do! ensure (ty = TIndex || (match ty with TInt(IntWidth width) -> width > 0 | _ -> false)) "Continuation state requires an integer carrier"
    let s = Values.continuationValue nodeId 0
    let! zero = pConstI (s 0) 0L ty
    let! store = pTypedInsertView frameSSA (s 0) offset (s 1) (s 2) (s 3) ty frameType
    return zero :: store
}

/// Initialize an exact, ordered set of settled fields. Sequence and ordinary
/// closure environments share the same scalar/descriptor store contract.
let pInitializeEnvironmentSlots nodeId frameSSA frameType bytes (slots: ContinuationSlot list) initializers = parser {
    let! state = getUserState
    let supplied = initializers |> List.map fst
    let expected = slots |> List.map _.Source
    do! ensure (Set.ofList supplied = Set.ofList expected && supplied.Length = expected.Length
                && (Set.ofList expected).Count = expected.Length) $"Environment {NodeId.value nodeId} requires its exact unique initializer set"
    let! operations =
        initializers |> List.mapi (fun ordinal (slotId, sourceId) -> parser {
            let! slot =
                match slots |> List.tryFind (fun slot -> slot.Source = slotId) with
                | Some slot -> preturn slot
                | None -> fail (Message $"Continuation initializer {NodeId.value sourceId} names an absent capture slot {NodeId.value slotId}")
            let! offset, expected = pSlotPlacement bytes slot
            let! value, recalled =
                match slot.Holds with
                | CaptureSlotKind.EnvironmentView owner ->
                    match MLIRAccumulator.recallCallable sourceId state.Accumulator with
                    | Some callable ->
                        match Alex.Traversal.CallableOperands.exactCarrier callable |> Option.bind _.Environment, callable.Environment with
                        | Some contract, Some environment when contract.Owner = owner -> preturn (environment.SSA, environment.Type)
                        | _ -> fail (Message $"Environment capture {NodeId.value sourceId} lacks its actual instance")
                    | None -> fail (Message $"Environment capture {NodeId.value sourceId} has no witnessed callable operands")
                | _ -> pRecallContinuationValue sourceId
            // Mutable bindings expose a semantic dynamic view, while their
            // allocating operation records the exact bounded cell descriptor.
            let actual = MLIRAccumulator.recallSSAType value state.Accumulator |> Option.defaultValue recalled
            let s = Values.continuationValue nodeId (ordinal + 1)
            let! adaptations, stored, storedType = pAdapt nodeId sourceId value actual
            let! viewOps, stored =
                match storedType, expected with
                | TMemRefStatic(_, element), TMemRef expectedElement when element = expectedElement -> parser {
                    let! cast = pMemRefCast (s 3) stored storedType expected
                    return [cast], s 3
                  }
                | _ when storedType = expected -> preturn ([], stored)
                | _ -> fail (Message $"Continuation capture {NodeId.value sourceId} lacks its settled descriptor or scalar carrier: actual {storedType}, expected {expected}")
            let! store = pTypedInsertView frameSSA stored offset (s 0) (s 1) (s 2) expected frameType
            return adaptations @ viewOps @ store
        }) |> Alex.XParsec.Extensions.sequence
    return List.concat operations
}

/// Initial capture values are supplied in the graph's original evaluation
/// order. CellView copies the existing cell descriptor, never its payload.
let pConstructSequence nodeId (frame: ContinuationFrame) (initializers: (NodeId * NodeId) list) : PSGParser<MLIROp list * TransferResult> = parser {
    let! state = getUserState
    let captures = frame.Slots |> List.filter _.IsCapture |> List.map _.Source |> Set.ofList
    let initialized = initializers |> List.map fst
    do! ensure (Set.ofList initialized = captures && initialized.Length = captures.Count) $"Sequence constructor {NodeId.value nodeId} does not initialize its exact settled capture set"
    let! allocations, frameSSA, frameType =
        match state.Graph.Codata.Value.SequenceDestinations |> Map.tryFind nodeId with
        | Some destination -> parser {
            let! expected = pFrameType frame
            let! value, actual = pRecallContinuationValue destination
            do! ensure (actual = expected) $"Sequence destination {NodeId.value destination} lacks its settled frame carrier"
            return [], value, expected
          }
        | None -> pAllocateFrame nodeId frame
    let! initialState = pInitializeState nodeId frameSSA frameType frame
    let! captureOps = pInitializeEnvironmentSlots nodeId frameSSA frameType frame.Bytes (frame.Slots |> List.filter _.IsCapture) initializers
    return allocations @ initialState @ captureOps, TRValue { SSA = frameSSA; Type = frameType }
}

/// Each enumeration owns fresh iteration state. Only capture fields are copied
/// from the sequence template; current and internal state remain uninitialized
/// until the graph's explicit stores dominate their reads.
let pGetEnumerator nodeId templateId (frame: ContinuationFrame) : PSGParser<MLIROp list * TransferResult> = parser {
    let! template, templateType = pRecallNode templateId
    do! ensure (templateType = TMemRefStatic(frame.Bytes, TInt(IntWidth 8))) $"Sequence template {NodeId.value templateId} lacks its settled frame carrier"
    let! allocation, frameSSA, frameType = pAllocateFrame nodeId frame
    let! initialState = pInitializeState nodeId frameSSA frameType frame
    let! captures =
        frame.Slots |> List.filter _.IsCapture |> List.mapi (fun ordinal slot -> parser {
            let! offset, ty = pSlotPlacement frame.Bytes slot
            let s = Values.continuationValue nodeId (ordinal + 1)
            let! read = pTypedExtractView (s 0) template offset (s 1) (s 2) (s 3) ty templateType
            let! write = pTypedInsertView frameSSA (s 0) offset (s 4) (s 5) (s 6) ty frameType
            return read @ write
        }) |> Alex.XParsec.Extensions.sequence
    return allocation @ initialState @ List.concat captures, TRValue { SSA = frameSSA; Type = frameType }
}

/// Formation names the actual generator independently of the environment just
/// constructed. It never discovers or emits that generator's body.
let pSequenceValue nodeId shape symbol (environment: Val) = parser {
    let code = { SSA = Values.callableCode nodeId; Type = Sequences.functionType shape }
    let! value =
        match Sequences.create shape code environment with
        | Result.Ok value -> preturn value
        | Result.Error reason -> fail (Message reason)
    let! operation = pFuncConstant code.SSA symbol code.Type
    return [operation], TRSequence value
}

let pPullSequence nodeId (sequence: SequenceOperand) = parser {
    do! ensure sequence.Flow.IsEnumerator "A pull requires an actual enumerator occurrence"
    let result = { SSA = Values.value nodeId 0; Type = TInt(IntWidth 1) }
    let! call = pFuncCallIndirectResults [result] sequence.Code.SSA [sequence.Environment]
    return [call], TRValue result
}

/// Current placement belongs to the common family. No representative owner's
/// source current identity is substituted at a multi-origin boundary.
let pSequenceCurrent nodeId (sequence: SequenceOperand) = parser {
    let! state = getUserState
    do! ensure (sequence.Flow.IsEnumerator && state.Graph.Codata.Value.SequenceCurrentReads.Contains nodeId)
            "Current access requires its exact successful-pull premise"
    let! field =
        match sequence.Family.CurrentField with
        | Some field -> preturn field
        | None -> fail (Message "A sequence family without a yielded payload has no current value")
    let! ty, inlineValue =
        match sequence.Family.CurrentRepresentation, field.Slot with
        | Some(_, CaptureSlotKind.Scalar scalar), settled when scalar = settled ->
            match settled with
            | SettledSlot.Integer(bits, _) when bits > 0 -> preturn (TInt(IntWidth bits), false)
            | SettledSlot.Bool -> preturn (TInt(IntWidth 1), false)
            | SettledSlot.Char | SettledSlot.Unit -> preturn (TInt(IntWidth 32), false)
            | SettledSlot.Real 32 -> preturn (TFloat F32, false)
            | SettledSlot.Real 64 -> preturn (TFloat F64, false)
            | _ -> fail (Message "Sequence scalar current has an unsupported settled representation")
        | Some(valueType, CaptureSlotKind.InlineValue native), SettledSlot.InlineBytes(bytes, alignment)
            when valueType = native && bytes > 0 && alignment > 0 && field.Size = Some bytes && field.Align = Some alignment ->
            match settledLayout state.Graph native with
            | Some(SettledLayout.Union(_, _, Some actualBytes, Some actualAlignment)) when actualBytes = bytes && actualAlignment = alignment ->
                preturn (TMemRefStatic(bytes, TInt(IntWidth 8)), true)
            | _ -> fail (Message "Sequence aggregate current no longer matches its exact source union layout")
        | _ -> fail (Message "Sequence common current requires its settled physical value protocol")
    let! offset =
        match field.Offset, field.Size, field.Align with
        | Some offset, Some bytes, Some alignment when offset >= 0 && bytes > 0 && alignment > 0 &&
                                                       offset % alignment = 0 && int64 offset + int64 bytes <= int64 sequence.Family.Bytes -> preturn offset
        | _ -> fail (Message "Sequence common current has no bounded aligned field placement")
    let s = Values.value nodeId
    let! read =
        if inlineValue then parser {
            let! index = pConstI (s 1) (int64 offset) TIndex
            let! view = pMemRefView (s 0) sequence.Environment.SSA (s 1) sequence.Environment.Type ty
            return [index; view]
        }
        else pTypedExtractView (s 0) sequence.Environment.SSA offset (s 1) (s 2) (s 3) ty sequence.Environment.Type
    let! adaptations, result, resultType = pAdapt nodeId nodeId (s 0) ty
    return read @ adaptations, TRValue { SSA = result; Type = resultType }
}

/// An opaque representation copy preserves descriptor identity and partial
/// definedness. It establishes only fresh-entry state, never a current value.
let pAcquireSequence nodeId (template: SequenceOperand) shape (copy: SequenceTemplateCopy) = parser {
    let! state = getUserState
    let family = Sequences.family shape
    do! ensure (not template.Flow.IsEnumerator && (Sequences.flow shape).IsEnumerator &&
                copy.Template = template.Flow.Occurrence && copy.Family = family.Identity &&
                family.Identity = template.Family.Identity && copy.Bytes = family.Bytes && copy.Alignment = family.Alignment &&
                copy.AddressSpace = NTUMemorySpace.Default)
            "Sequence copy requires its exact source/destination protocol and admitted address space"
    let environment = { SSA = Values.value nodeId 0; Type = Sequences.environmentType shape }
    let! allocation =
        match copy.Region with
        | Some region -> parser {
            let! parent =
                match state.Graph.Codata.Value.ContinuationFrames.TryFind region.ParentOwner with
                | Some parent when parent.Formal = region.ParentFormal && region.Bytes = family.Bytes &&
                                   region.Alignment = family.Alignment && region.Offset >= 0 &&
                                   region.Offset % family.Alignment = 0 && int64 region.Offset + int64 family.Bytes <= int64 parent.Bytes -> preturn parent
                | _ -> fail (Message "Fresh sequence destination lost its exact enclosing region")
            let! parentSSA, parentType = pRecallContinuationValue region.ParentFormal
            do! ensure (parentType = TMemRefStatic(parent.Bytes, TInt(IntWidth 8))) "Sequence region parent has a different physical carrier"
            let offsetSSA = Values.value nodeId 1
            let! offset = pConstI offsetSSA (int64 region.Offset) TIndex
            let! view = pMemRefView environment.SSA parentSSA offsetSSA parentType environment.Type
            return [offset; view]
          }
        | None ->
            match copy.Residence with
            | EscapeKind.StackScoped -> parser {
                let! allocation = pAlloca environment.SSA family.Bytes (TInt(IntWidth 8)) (Some family.Alignment)
                return [allocation]
              }
            | EscapeKind.StaticLifetime -> parser {
                let! allocation = Alex.Patterns.MemoryPatterns.pAllocValue nodeId environment.SSA environment.Type
                return [allocation]
              }
            | _ -> fail (Message "Fresh sequence storage has no supported admitted residence")
    let! representation = pOpaqueStorageCopy environment.SSA template.Environment.SSA copy.Bytes copy.Alignment
    let! stateType, offset =
        match family.StateField.Slot, family.StateField.Offset with
        | SettledSlot.Integer(bits, _), Some offset when bits > 0 -> preturn (TInt(IntWidth bits), offset)
        | _ -> fail (Message "Fresh sequence entry state lacks its common integer placement")
    let s = Values.continuationValue nodeId 0
    let! zero = pConstI (s 0) 0L stateType
    let! reset = pTypedInsertView environment.SSA (s 0) offset (s 1) (s 2) (s 3) stateType environment.Type
    let! result =
        match Sequences.create shape template.Code environment with
        | Result.Ok value -> preturn value
        | Result.Error reason -> fail (Message reason)
    return allocation @ [representation; zero] @ reset, TRSequence result
}
