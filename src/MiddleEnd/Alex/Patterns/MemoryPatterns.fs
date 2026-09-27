/// MemoryPatterns - Memory operation patterns composed from Elements
///
/// PUBLIC: Witnesses call these patterns to elide memory operations to MLIR.
/// Patterns compose Elements (internal) into semantic memory operations.
module Alex.Patterns.MemoryPatterns

open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId - MUST be before TransferTypes
open XParsec
open XParsec.Parsers     // fail, preturn
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.XParsec.Extensions // sequence combinator
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRAtomics
open Alex.Elements.MemRefElements
open Alex.Elements.ArithElements
open Alex.Elements.IndexElements
open Alex.Elements.FuncElements
open Alex.Elements.SCFElements
open Alex.CodeGeneration.TypeMapping
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core

// ═══════════════════════════════════════════════════════════
// FIELD EXTRACTION PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract single field from struct
/// SSA layout (2 total):
///   [0] = offsetConstSSA - index constant for memref.load
///   [1] = resultSSA - result of the load
let pExtractField (ssas: SSA list) (structSSA: SSA) (fieldIndex: int) (structTy: MLIRType) : PSGParser<MLIROp list> =
    parser {
        do! ensure (ssas.Length >= 2) $"pExtractField: Expected 2 SSAs, got {ssas.Length}"
        let offsetSSA = ssas.[0]
        let resultSSA = ssas.[1]
        return! pExtractValue resultSSA structSSA fieldIndex offsetSSA structTy
    }

// ═══════════════════════════════════════════════════════════
// DU PATTERNS
// ═══════════════════════════════════════════════════════════

/// Extract DU tag (handles both inline and pointer-based DUs)
/// Pointer-based: Load tag byte from offset 0
/// Inline: ExtractValue at index 0
/// SSAs extracted from coeffects via nodeId: [0] = indexZeroSSA, [1] = tagSSA (result)
let pExtractDUTag (nodeId: NodeId) (duSSA: SSA) (duType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        let tagTy = TInt (IntWidth 8)  // DU tags are always i8

        match duType with
        | TIndex ->
            // Pointer-based DU: load tag byte from offset 0
            do! ensure (ssas.Length >= 2) $"pExtractDUTag (pointer): Expected 2 SSAs, got {ssas.Length}"
            let indexZeroSSA = ssas.[0]
            let tagSSA = ssas.[1]
            let! indexZeroOp = pConstI indexZeroSSA 0L TIndex
            let! loadOp = pLoad tagSSA duSSA [indexZeroSSA]
            return ([indexZeroOp; loadOp], TRValue { SSA = tagSSA; Type = tagTy })
        | _ ->
            // Inline struct DU: typed extract via reinterpret_cast at byte offset 0
            do! ensure (ssas.Length >= 3) $"pExtractDUTag (inline): Expected 3 SSAs, got {ssas.Length}"
            let castSSA = ssas.[0]
            let zeroSSA = ssas.[1]
            let tagSSA = ssas.[2]
            let! ops = pTypedExtract tagSSA duSSA 0 castSSA zeroSSA tagTy duType
            return (ops, TRValue { SSA = tagSSA; Type = tagTy })
    }

/// The payload offset of a union, read from the settled layout of the union's type
/// (`SemanticGraph.Layouts`, ruling 2: the tag, then the payload slot of the widest case).
/// An absent or non-union layout is a settlement gap, reported at the reading occurrence.
let pUnionPayloadOffset (unionTy: TypeIdentity) : PSGParser<int> =
    parser {
        let! state = getUserState
        match settledLayoutFor state.Graph unionTy with
        | Some (SettledLayout.Union (_, Some offset, _, _)) -> return offset
        | Some other ->
            return! fail (Message (sprintf "PSG settlement did not publish a union payload offset at node %d: %A"
                                       (NodeId.value state.Current.Id) other))
        | None ->
            return! fail (Message (sprintf "PSG settlement did not publish a union layout at node %d" (NodeId.value state.Current.Id)))
    }

/// Extract DU payload via memref.view (different element type: byte buffer → typed payload)
/// SSAs extracted from coeffects via nodeId: [0] = offsetSSA, [1] = viewSSA, [2] = zeroSSA, [3] = extractSSA
let pExtractDUPayload (nodeId: NodeId) (duSSA: SSA) (duType: MLIRType) (unionIdentity: TypeIdentity) (payloadType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 4) $"pExtractDUPayload: Expected 4 SSAs, got {ssas.Length}"

        let offsetSSA = ssas.[0]
        let viewSSA = ssas.[1]
        let zeroSSA = ssas.[2]
        let extractSSA = ssas.[3]

        let! payloadByteOffset = pUnionPayloadOffset unionIdentity

        // Typed extract via memref.view — payload has different element type than byte buffer
        let! extractOps = pTypedExtractView extractSSA duSSA payloadByteOffset offsetSSA viewSSA zeroSSA payloadType duType
        return (extractOps, TRValue { SSA = extractSSA; Type = payloadType })
    }

// ═══════════════════════════════════════════════════════════
// ARRAY PATTERNS
// ═══════════════════════════════════════════════════════════

/// Build Arena.create pattern
/// Allocates an arena buffer on the stack
///
/// Arena.create<'lifetime>(sizeBytes: int) : Arena<'lifetime>
/// Returns: memref<sizeBytes x i8> (stack-allocated byte buffer)
/// SSA extracted from coeffects via nodeId: [0] = result
let pArenaCreate (nodeId: NodeId) (sizeBytes: int) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pArenaCreate: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Allocate arena memory block on stack as byte array
        // Arena IS the memref - no separate control struct in memref semantics
        let elemType = TInt (IntWidth 8)
        let! allocaOp = pAlloca resultSSA sizeBytes elemType None
        let memrefTy = TMemRefStatic (sizeBytes, elemType)

        // Return the arena memref (byte buffer)
        return ([allocaOp], TRValue { SSA = resultSSA; Type = memrefTy })
    }

/// Arena.alloc(arena: Arena<'lifetime> byref, sizeBytes: int) : nativeint
/// A bump allocation needs its settled offset within the arena. No stage settles one, so
/// the allocation is refused at its occurrence; the whole arena is never handed out as the
/// allocation.
let pArenaAlloc (nodeId: NodeId) (_arenaSSA: SSA) (_sizeSSA: SSA) (_arenaType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    fail (Message (sprintf "Baker (arena residence) did not settle a bump-allocation offset for Arena.alloc at node %d: the allocation has no realization inside its arena"
                       (NodeId.value nodeId)))

// ═══════════════════════════════════════════════════════════
// STRUCT FIELD ACCESS PATTERNS
// ═══════════════════════════════════════════════════════════


// ═══════════════════════════════════════════════════════════
// ESCAPE-AWARE ALLOCATION
// ═══════════════════════════════════════════════════════════

/// The static memref shape of a value's storage: a struct is a byte memref of its settled size.
/// Any other carrier has no settled static storage and is reported at the allocating node.
let pMemRefShape (nodeId: NodeId) (arch: Architecture) (ty: MLIRType) : PSGParser<int * MLIRType> =
    match ty with
    | TMemRefStatic (count, elemType) -> preturn (count, elemType)
    | TStruct _ -> preturn (mlirTypeSize arch ty, TInt (IntWidth 8))
    | _ -> fail (Message $"PSG settlement (Layouts) did not settle static storage for the allocated value at node {NodeId.value nodeId}: its carrier {ty} is neither a static memref nor a settled struct")

/// One symbol convention for a source-admitted static allocation and its reads.
let staticValueName (nodeId: NodeId) = sprintf "__clef_static_value_%d" (NodeId.value nodeId)

/// A writable declaration consumes its exact source inventory entry. It cannot
/// infer residence from a symbol name or turn individual fit into pool capacity.
let programStorageType arch graph (entry: ProgramStorageEntry) =
    match entry.Shape with
    | ProgramStorageShape.Bytes -> Some(TMemRefStatic(entry.Bytes, TInt(IntWidth 8)))
    | ProgramStorageShape.Scalar slot ->
        let scalar =
            match slot with
            | SettledSlot.Pointer 1 -> Some TIndex
            | _ -> settledScalarType slot
        scalar |> Option.map (fun scalar -> TMemRefStatic(1, scalar))
    | ProgramStorageShape.ValueView _ ->
        let site = match entry.Identity with ProgramStorageIdentity.Allocation id | ProgramStorageIdentity.BindingSlot id -> id
        valueTypeAt graph site
        |> physicalStorageType arch
        |> fun value -> Some(TMemRefStatic(1, value))

let pProgramStorageDeclaration identity storageTy : PSGParser<ProgramStorageEntry> =
    parser {
        let! state = getUserState
        let! inventory =
            match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryStorage state.Graph with
            | Result.Ok projection -> preturn projection.ProgramStorage
            | Result.Error reason ->
                fail (Message (sprintf "PSG settlement (WitnessEmission storage) did not publish the writable program inventory for %A: %s" identity reason))
        do! ensure inventory.Unresolved.IsEmpty
                ("Writable program inventory is unresolved: " + (inventory.Unresolved.Values |> String.concat "; "))
        let entry = inventory.Entries.TryFind identity
        do! ensure entry.IsSome (sprintf "Writable declaration has no source inventory entry for %A" identity)
        let entry = entry.Value
        let expected = programStorageType state.Platform.TargetArch state.Graph entry
        do! ensure (expected = Some storageTy) (sprintf "Writable declaration %A disagrees with its source-held physical shape" identity)
        return entry
    }

/// Allocate memory for a constructed value — queries escape analysis coeffect
/// PULL model: pattern pulls allocation decision from pre-computed coeffects.
/// Four-point lifetime lattice (closure-representation.md §3.3):
///   StackScoped    → memref.alloca (stack)
///   StaticLifetime → program-lifetime, belongs in static storage (memref.global)
///   EscapesVia*    → memref.alloc  (heap)
///
/// StaticLifetime is a program-lifetime DU/record: constructed once at global scope, held to
/// program end, never freed. It is placed in a module-level memref.global and referenced inline
/// via memref.get_global. Because
/// a memref.global is only valid at module scope and this runs in the PSGParser layer, the decl
/// is queued (deduped) on the shared accumulator via tryEmitGlobalMemref; the owning DU/record
/// witness drains it into WitnessOutput.TopLevelOps for module-scope placement. On a heap-free
/// target this is the only non-stack placement, so a program-lifetime DU/record no longer routes
/// through the heap allocator.
let pAllocValue (nodeId: NodeId) (ssa: SSA) (ty: MLIRType) : PSGParser<MLIROp> =
    parser {
        let! state = getUserState
        // Every allocating site carries its settled class; absence is never read as stack scope.
        let! escapeKind =
            match Map.tryFind nodeId state.Graph.Codata.Value.Escapes with
            | Some kind -> preturn kind
            | None -> fail (Message $"PSG settlement (Escape) did not settle an escape/lifetime class for the allocation at node {NodeId.value nodeId}")
        match escapeKind with
        | EscapeKind.StackScoped ->
            return! pUndef ssa ty
        | EscapeKind.StaticLifetime ->
            let! count, elemType = pMemRefShape nodeId state.Platform.TargetArch ty
            let storageTy = TMemRefStatic (count, elemType)
            let globalName = staticValueName nodeId
            let! authority = pProgramStorageDeclaration (ProgramStorageIdentity.Allocation nodeId) storageTy
            MLIRAccumulator.tryEmitGlobalMemref globalName storageTy (Some authority) state.Accumulator
            return! pMemRefGetGlobal ssa globalName storageTy
        | EscapeKind.EscapesViaReturn | EscapeKind.EscapesViaClosure _ | EscapeKind.EscapesViaByRef ->
            let! count, elemType = pMemRefShape nodeId state.Platform.TargetArch ty
            return! pAllocStatic ssa count elemType None
    }

// ═══════════════════════════════════════════════════════════
// DU CONSTRUCTION
// ═══════════════════════════════════════════════════════════

/// DU case construction: tag field (index 0) + payload fields
/// CRITICAL: This is the foundation for all collection patterns (Option, List, Map, Set, Result)
/// SSA layout: [0] = undefSSA, [1] = tagSSA, [2] = tagOffsetSSA, [3] = tagResultSSA,
///             then for each payload: [4+3*i] = offsetSSA, [5+3*i] = viewSSA, [6+3*i] = zeroSSA
let pDUCaseAt (nodeId: NodeId) (destination: Val) (identity: TypeIdentity) (tag: int64) (payload: Val list) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let s = Alex.Traversal.Values.value nodeId
        let ty = destination.Type

        // Insert tag at byte offset 0 via reinterpret_cast (same element type: i8→i8)
        let tagTy = TInt (IntWidth 8)  // DU tags are always i8
        let! tagConstOp = pConstI (s 1) tag tagTy
        let! insertTagOps = pTypedInsert destination.SSA (s 1) 0 (s 2) (s 3) tagTy ty

        // Insert payload fields at the settled payload offset (after the tag) via memref.view
        // (different element type: byte buffer → typed payload)
        let! payloadByteOffset = pUnionPayloadOffset identity
        let! payloadOpLists =
            payload
            |> List.mapi (fun i field ->
                parser {
                    let offsetSSA = s (4 + 3*i)
                    let viewSSA = s (5 + 3*i)
                    let zeroSSA = s (6 + 3*i)
                    return! pTypedInsertView destination.SSA field.SSA payloadByteOffset offsetSSA viewSSA zeroSSA field.Type ty
                })
            |> sequence

        let payloadOps = List.concat payloadOpLists
        return (tagConstOp :: (insertTagOps @ payloadOps), TRVoid)
    }

/// Construct into newly allocated storage using the same settled tag/payload
/// insertion as an explicit Baker destination. Inactive payloads are untouched.
let pDUCase (nodeId: NodeId) (tag: int64) (payload: Val list) (ty: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState
        let! result = getNodeSSA nodeId
        let! allocation = pAllocValue nodeId result ty
        let! writes, _ = pDUCaseAt nodeId { SSA = result; Type = ty } (sourceTypeAt state.Graph nodeId) tag payload
        return allocation :: writes, TRValue { SSA = result; Type = ty }
    }

// ═══════════════════════════════════════════════════════════
// SIMPLE MEMORY STORE
// ═══════════════════════════════════════════════════════════

/// MemRef copy - bulk memory copy via memcpy library function
let pMemCopy (resultSSA: SSA) (destSSA: SSA) (srcSSA: SSA) (countSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType
        let args = [
            { SSA = destSSA; Type = platformWordTy }
            { SSA = srcSSA; Type = platformWordTy }
            { SSA = countSSA; Type = platformWordTy }
        ]
        let! memcpyCall = pFuncCall (Some resultSSA) "memcpy" args platformWordTy
        let! memcpyDecl = pFuncDecl "memcpy" [platformWordTy; platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private
        return ([memcpyDecl; memcpyCall], TRVoid)
    }

// ═══════════════════════════════════════════════════════════
// MONADIC ARGUMENT RECALL
// ═══════════════════════════════════════════════════════════

/// Recall argument from accumulator.
/// VarRefWitness already auto-loads mutable variables (TMemRef) in post-order.
/// By the time Application recalls its arguments, loading is done.
/// This combinator provides a uniform (ops, ssa, type) triple interface.
let pRecallArgWithLoad (argId: NodeId) : PSGParser<MLIROp list * SSA * MLIRType> =
    parser {
        let! (ssa, ty) = pRecallNode argId
        return ([], ssa, ty)
    }

// ═══════════════════════════════════════════════════════════
// COMPOSED INTRINSIC PARSERS (per-operation, self-contained)
// ═══════════════════════════════════════════════════════════

/// Arena.create intrinsic — stack-allocated byte buffer
let pArenaCreateIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Arena
        do! ensure (info.Operation = "create") "Not Arena.create"
        let! state = getUserState
        let! node = getCurrentNode
        let sizeNodeId = argIds.[0]
        match SemanticGraph.tryGetNode sizeNodeId state.Graph with
        | Some sizeNode ->
            match sizeNode.Kind with
            | SemanticKind.Literal (NativeLiteral.Int (value, _)) ->
                return! pArenaCreate node.Id (int value)
            | _ -> return! fail (Message $"Arena.create: size must be a literal int")
        | None -> return! fail (Message "Arena.create: size node not found")
    }

/// Arena.alloc intrinsic — allocate from arena
let pArenaAllocIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Arena
        do! ensure (info.Operation = "alloc") "Not Arena.alloc"
        do! ensure (argIds.Length >= 2) "Arena.alloc: Expected 2 args"
        let! node = getCurrentNode
        let! (_, arenaSSA, arenaType) = pRecallArgWithLoad argIds.[0]
        let! (_, sizeSSA, _) = pRecallArgWithLoad argIds.[1]
        return! pArenaAlloc node.Id arenaSSA sizeSSA arenaType
    }

// ═══════════════════════════════════════════════════════════
// ARRAY INTRINSIC PARSERS
// ═══════════════════════════════════════════════════════════

/// Index extension follows the established operand range. In particular, an
/// unsigned narrow carrier's high bit is data, while a possibly negative index
/// must keep its sign. Empty does not establish a non-negative runtime value.
let indexCastForRange (range: ValueRange) (result: SSA) (operand: SSA) (operandType: MLIRType) : MLIROp =
    if range <> ValueRange.Empty && ValueRange.isNonNegative range then
        MLIROp.IndexOp (IndexOp.IndexCastU (result, operand, operandType, TIndex))
    else
        MLIROp.IndexOp (IndexOp.IndexCastS (result, operand, operandType, TIndex))

// ═══════════════════════════════════════════════════════════
// PUBLISHED MEMORY OPERATIONS
// ═══════════════════════════════════════════════════════════

let private pMemorySlot slot : PSGParser<MLIRType> =
    match SettledScalar.tryType slot with
    | Some ty -> preturn ty
    | None -> fail (Message (sprintf "Published memory slot has no admitted scalar form: %A" slot))

let private pMemoryBuffer source element : PSGParser<Val> = parser {
    let! state = getUserState
    let! ssa, ty = pRecallNode source
    let matchesElement = function
        | TMemRef actual | TMemRefStatic(_, actual) -> actual = element
        | _ -> false
    do! ensure (matchesElement ty) "Memory operand disagrees with its published buffer carrier."
    match MLIRAccumulator.recallSSAType ssa state.Accumulator with
    | Some physical when matchesElement physical -> return { SSA = ssa; Type = physical }
    | _ -> return! fail (Message "Memory operand disagrees with its published buffer carrier at the physical SSA.")
}

let private pMemoryExtent site source element resultCarrier unsigned : PSGParser<MLIROp list * TransferResult> = parser {
    let! buffer = pMemoryBuffer source element
    let! ssas = getNodeSSAs site
    do! ensure (ssas.Length >= 3) "Memory extent lacks its physical operand names."
    let result = { SSA = List.last ssas; Type = scalarCarrierType resultCarrier }
    let! dimension = pIndexConst ssas.[0] 0L
    let! length = pMemRefDim ssas.[1] buffer.SSA ssas.[0] buffer.Type
    let! adaptation =
        if unsigned then pIndexCastU result.SSA ssas.[1] TIndex result.Type
        else pIndexCastS result.SSA ssas.[1] TIndex result.Type
    return [dimension; length; adaptation], TRValue result
}

let private pMemoryIndex result (bounds: MemoryBoundsWitness) : PSGParser<MLIROp * SSA> = parser {
    let! index, ty = pRecallNode bounds.Index
    do! ensure (ty = scalarCarrierType bounds.IndexCarrier)
            "Array index disagrees with its source-published carrier."
    let! adaptation =
        if bounds.IndexUnsigned then pIndexCastU result index ty TIndex
        else pIndexCastS result index ty TIndex
    return adaptation, result
}

/// Descriptor offset and stride remain part of the actual location. The source
/// contract supplies element bytes; this Pattern does not compute a layout.
let private pMemoryAddressOfView site result (buffer: Val) element bytes index : PSGParser<MLIROp list> = parser {
    do! ensure (bytes > 0) "Memory address lacks its source-settled element extent."
    let v = Alex.Traversal.Values.value site
    let! metadata = pExtractStridedMetadata (v 0) (v 1) (v 2) (v 3) buffer.SSA buffer.Type element
    let! baseAddress = pExtractBasePtr (v 4) (v 0) (TMemRefScalar element)
    let! offsetOps, elementOffset =
        match index with
        | None -> preturn ([], v 1)
        | Some index -> parser {
            let! stride = pIndexMul (v 5) index (v 3)
            let! offset = pIndexAdd (v 6) (v 1) (v 5)
            return [stride; offset], v 6 }
    let! elementBytes = pIndexConst (v 7) (int64 bytes)
    let! byteOffset = pIndexMul (v 8) elementOffset (v 7)
    let! address = pIndexAdd result (v 4) (v 8)
    return [metadata; baseAddress] @ offsetOps @ [elementBytes; byteOffset; address]
}

let private pMemoryPointee (address: MemoryAddressWitness) : PSGParser<MLIRType * int> = parser {
    match address.Element, address.ElementBytes with
    | Some slot, Some bytes when bytes > 0 ->
        let! element = pMemorySlot slot
        return element, bytes
    | _ -> return! fail (Message "Memory address lacks its source-published pointee slot and extent.")
}

/// Compose one exact source operation. Top-level definitions retain this same
/// occurrence; the witness passes them directly to ordinary scope bookkeeping.
let pPublishedMemoryOperation : PSGParser<MLIROp list * MLIROp list * TransferResult> = parser {
    let! state = getUserState
    let! memory =
        match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryMemory state.Graph with
        | Result.Ok memory -> preturn memory
        | Result.Error reason -> fail (Message reason)
    match memory.Operations.TryFind state.Current.Id with
    | None -> return! fail (Message "Memory operation lacks its source-published contract.")
    | Some (MemoryWitnessOperation.BufferExtent extent) ->
        do! ensure (extent.Site = state.Current.Id && extent.Result.Site = state.Current.Id)
                "Memory extent and result carrier name different source occurrences."
        let! operations, result =
            pMemoryExtent extent.Site extent.Source (TInt(IntWidth extent.Element.Bits)) extent.Result extent.IndexUnsigned
        return operations, [], result
    | Some (MemoryWitnessOperation.ArrayExtent extent) ->
        let! element = pMemorySlot extent.Element
        let! operations, result = pMemoryExtent extent.Site extent.Source element extent.Result true
        return operations, [], result
    | Some (MemoryWitnessOperation.StringView view) ->
        do! ensure (view.Site=state.Current.Id) "String view names a different source occurrence."
        let! element=pMemorySlot view.Element
        let! source=pMemoryBuffer view.Source element
        let sourceType,resultType=representationType view.SourceCarrier,representationType view.ResultCarrier
        do! ensure (source.Type=sourceType) "String view source differs from its exact published physical carrier."
        if sourceType=resultType then return [],[],TRValue source
        else
            match view.SourceCarrier,view.ResultCarrier with
            | ValueRepresentation.Buffer(Some _,sourceElement),ValueRepresentation.Buffer(None,resultElement)
                when representationType sourceElement=representationType resultElement ->
                let! result=getNodeSSA view.Site
                let! cast=pMemRefCast result source.SSA sourceType resultType
                return [cast],[],TRValue {SSA=result;Type=resultType}
            | _ -> return! fail (Message "String view has no published matching buffer-carrier transport.")
    | Some (MemoryWitnessOperation.ArrayAllocation allocation) ->
        do! ensure (allocation.Site = state.Current.Id)
                "Array allocation names a different source occurrence."
        let! element = pMemorySlot allocation.Element
        let! count, countType = pRecallNode allocation.Count
        do! ensure (countType = scalarCarrierType allocation.CountCarrier)
                "Array count differs from its source-published carrier."
        let! ssas = getNodeSSAs allocation.Site
        let result = { SSA = List.last ssas; Type = TMemRef element }
        do! ensure (result.Type = valueTypeAt state.Graph allocation.Site)
                "Array allocation differs from its source-published result carrier."
        let! countCast =
            if allocation.IndexUnsigned then pIndexCastU ssas.[0] count countType TIndex
            else pIndexCastS ssas.[0] count countType TIndex
        match allocation.Residence with
        | MemoryResidence.Stack _ ->
            let! storage = pAllocaDynamic result.SSA ssas.[0] element allocation.Alignment
            return [countCast; storage], [], TRValue result
        | _ -> return! fail (Message "Dynamic array allocation lacks an admitted source storage residence.")
    | Some (MemoryWitnessOperation.ArrayAccess access) ->
        let! element = pMemorySlot access.Element
        let! buffer = pMemoryBuffer access.Buffer element
        let! ssas = getNodeSSAs access.Site
        let! cast, index = pMemoryIndex ssas.[0] access.Bounds
        match access.Value with
        | Some value ->
            let! valueSSA, valueType = pRecallNode value
            let! adaptations, adapted = pPublishedAdapt access.Site value access.Adaptation { SSA = valueSSA; Type = valueType }
            do! ensure (adapted.Type = element) "Array store value disagrees with its published element carrier."
            let! write = pStore adapted.SSA buffer.SSA [index] element buffer.Type
            return cast :: adaptations @ [write], [], TRVoid
        | None ->
            let! read = pLoadTyped ssas.[1] buffer.SSA [index] element buffer.Type
            let! adaptations, result = pPublishedAdapt access.Site access.Site access.Adaptation { SSA = ssas.[1]; Type = element }
            return [cast; read] @ adaptations, [], TRValue result
    | Some (MemoryWitnessOperation.ArrayLiteral literal) ->
        let! element = pMemorySlot literal.Element
        let! ssas = getNodeSSAs literal.Site
        let staticType = TMemRefStatic(literal.Length, element)
        let resultType = TMemRef element
        let result = { SSA = List.last ssas; Type = resultType }
        do! ensure (literal.Length = literal.Elements.Length && literal.Alignment > 0)
                "Array literal lacks its source-settled extent and alignment."
        let! allocation, declarations, buffer =
            match literal.Residence with
            | MemoryResidence.ImmutableProgram _ -> parser {
                do! ensure (literal.Initializers |> Option.exists (fun values -> values.Length = literal.Length))
                        "Immutable array lacks its complete source-published initializer."
                let symbol = staticValueName literal.Site
                let! reference = pMemRefGetGlobal ssas.[0] symbol staticType
                return [reference], [MLIROp.GlobalArray(symbol, staticType, literal)], ssas.[0] }
            | MemoryResidence.Stack _ -> parser {
                let! allocation = pAlloca ssas.[0] literal.Length element (Some literal.Alignment)
                return [allocation], [], ssas.[0] }
            | MemoryResidence.Program identity -> parser {
                let! storage =
                    match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryStorage state.Graph with
                    | Result.Error reason -> fail (Message reason)
                    | Result.Ok storage ->
                        match storage.ProgramStorage.Entries.TryFind identity with
                        | Some entry -> preturn entry
                        | None -> fail (Message "Array allocation lacks its published program-storage identity.")
                do! ensure (storage.Shape = ProgramStorageShape.Bytes)
                        "Program array requires its source-settled byte allocation."
                let storageType = TMemRefStatic(storage.Bytes, TInt(IntWidth 8))
                let! authority = pProgramStorageDeclaration identity storageType
                let symbol = staticValueName literal.Site
                let! reference = pMemRefGetGlobal ssas.[1] symbol storageType
                let! offset = pIndexConst ssas.[2] 0L
                let! view = pMemRefView ssas.[0] ssas.[1] ssas.[2] storageType staticType
                return [reference; offset; view], [MLIROp.GlobalMemref(symbol, storageType, Some authority)], ssas.[0] }
        let! stores =
            match literal.Residence with
            | MemoryResidence.ImmutableProgram _ -> preturn []
            | _ ->
                literal.Elements |> List.mapi (fun ordinal (source, adaptation) -> parser {
                    let! value, ty = pRecallNode source
                    let! adaptations, adapted = pPublishedAdapt literal.Site source adaptation { SSA = value; Type = ty }
                    do! ensure (adapted.Type = element) "Array initializer disagrees with its published element carrier."
                    let indexSSA = Alex.Traversal.Values.arrayElementIndex literal.Site ordinal
                    let! index = pIndexConst indexSSA (int64 ordinal)
                    let! store = pStore adapted.SSA buffer [indexSSA] element staticType
                    return adaptations @ [index; store] }) |> sequence |>> List.concat
        let! descriptor = pMemRefCast result.SSA buffer staticType resultType
        return allocation @ stores @ [descriptor], declarations, TRValue result
    | Some (MemoryWitnessOperation.Address address) ->
        do! ensure (state.Platform.TargetArch.Pointer = Ok address.PointerBits)
                "Address disagrees with the source-published pointer carrier."
        let! ssas = getNodeSSAs address.Site
        let result = { SSA = List.last ssas; Type = TIndex }
        match address.Place with
        | MemoryPlace.ExistingReference source ->
            let! reference, ty = pRecallNode source
            do! ensure (ty = TIndex) "Existing reference disagrees with its source-published address carrier."
            return [], [], TRValue { SSA = reference; Type = TIndex }
        | MemoryPlace.MutableCell binding ->
            let! element, bytes = pMemoryPointee address
            let! buffer = pMemoryBuffer binding element
            let! operations = pMemoryAddressOfView address.Site result.SSA buffer element bytes None
            return operations, [], TRValue result
        | MemoryPlace.ArrayElement(buffer, _, bounds) ->
            let! element, bytes = pMemoryPointee address
            let! buffer = pMemoryBuffer buffer element
            let! cast, index = pMemoryIndex ssas.[10] bounds
            let! operations = pMemoryAddressOfView address.Site result.SSA buffer element bytes (Some index)
            return cast :: operations, [], TRValue result
        | MemoryPlace.RecordField(receiver, receiverBytes, field) ->
            let! receiverSSA, _ = pRecallNode receiver
            let bufferType = TMemRefStatic(receiverBytes, TInt(IntWidth 8))
            do! ensure (MLIRAccumulator.recallSSAType receiverSSA state.Accumulator = Some bufferType)
                    "Record-field address disagrees with its source-published receiver storage."
            let! fieldOffset =
                match field.Offset with
                | Some offset when offset >= 0 -> preturn offset
                | _ -> fail (Message "Record-field address lacks its source-settled byte offset.")
            let! baseOps = pMemoryAddressOfView address.Site ssas.[9] { SSA = receiverSSA; Type = bufferType } (TInt(IntWidth 8)) 1 None
            let! offset = pIndexConst ssas.[10] (int64 fieldOffset)
            let! address = pIndexAdd result.SSA ssas.[9] ssas.[10]
            return baseOps @ [offset; address], [], TRValue result
}

/// Component entry points share the same published access contract as the
/// Memory witness. They cannot recreate a missing array operation.
let private pPublishedArrayAccess expected : PSGParser<MLIROp list * TransferResult> = parser {
    let! state = getUserState
    let! memory =
        match Clef.Compiler.PSGSaturation.SemanticGraph.WitnessEmission.tryMemory state.Graph with
        | Result.Ok memory -> preturn memory
        | Result.Error reason -> fail (Message reason)
    do! ensure (memory.Operations.TryFind state.Current.Id |> Option.exists expected)
            "Array operation lacks its matching source-published access contract."
    let! operations, declarations, result = pPublishedMemoryOperation
    do! ensure declarations.IsEmpty "Array access unexpectedly required a storage declaration."
    return operations, result
}

let pIndexGetArray = pPublishedArrayAccess (function MemoryWitnessOperation.ArrayAccess access -> access.Value.IsNone | _ -> false)
let pIndexSetArray = pPublishedArrayAccess (function MemoryWitnessOperation.ArrayAccess access -> access.Value.IsSome | _ -> false)
