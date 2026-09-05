/// RecordPatterns - Codata-dependent record elision patterns
///
/// PUBLIC: Witnesses call these patterns to elide record operations to MLIR.
/// CPU:  memref.alloca + byte-offset store/load via pTypedInsertView/pTypedExtractView
/// FPGA: hw.struct_create / hw.struct_extract / hw.struct_inject
module Alex.Patterns.RecordPatterns

open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open XParsec
open XParsec.Parsers     // fail, preturn
open XParsec.Combinators // parser { }
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRAtomics
open Alex.Elements.MemRefElements
open Alex.Elements.IndexElements
open Alex.Elements.FuncElements
open Alex.Elements.HWElements
open Alex.Patterns.MemoryPatterns
open Alex.CodeGeneration.TypeMapping

// ═══════════════════════════════════════════════════════════════════════════
// HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Compute byte offset of a field within a TStruct from preceding fields
let structFieldByteOffset (fields: (string * MLIRType) list) (idx: int) : int =
    fields |> List.take idx |> List.sumBy (fun (_, t) -> mlirTypeSize t)

/// Find field index and type by name within a TStruct
let structFieldLookup (fields: (string * MLIRType) list) (fieldName: string) : (int * MLIRType) option =
    fields |> List.tryFindIndex (fun (n, _) -> n = fieldName)
    |> Option.map (fun idx -> (idx, snd fields.[idx]))

// ═══════════════════════════════════════════════════════════════════════════
// RECORD CONSTRUCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Build a record from field values — codata-dependent on TargetPlatform.
/// CPU:  alloca + pTypedInsertView per field at computed byte offset
/// FPGA: hw.struct_create from field values
///
/// SSA layout (CPU): ssas.[0] = alloca, then per field i: ssas.[1 + 3*i] = offset,
///   ssas.[2 + 3*i] = view, ssas.[3 + 3*i] = zero
/// SSA layout (FPGA): ssas.[0] = result (struct_create)
let pBuildRecord
    (nodeId: NodeId)
    (structTy: MLIRType)
    (fieldValues: (string * SSA * MLIRType) list)
    (fieldNodes: NodeId list)
    : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        let! platform = getTargetPlatform

        match platform with
        | Core.Types.Dialects.FPGA ->
            // FPGA: hw.struct_create at the record type's settled field widths (structTy, narrowed
            // by the witness through FieldRanges). A field value narrower than its field is
            // extended by the sign of its own range (extui / extsi, read from the value's node);
            // a value wider than its field cannot occur, since the field's range is the join of
            // every construction's, and is a stop. SSA layout: [0] = result, [1 + 3*i] = the
            // extension of field i where one is needed (the CPU layout's spare per-field SSAs).
            let resultSSA = ssas.[0]
            let! state = getUserState
            let declared =
                match structTy with
                | TStruct fields -> fields
                | other -> failwithf "pBuildRecord: the record type on fabric is %A, not a struct" other
            let bits (ty: MLIRType) = match ty with TInt (IntWidth b) -> b | _ -> 0
            let placed =
                List.zip fieldValues fieldNodes
                |> List.mapi (fun i ((name, valueSSA, valueTy), valueNode) ->
                    let fieldTy = declared |> List.tryFind (fun (n, _) -> n = name) |> Option.map snd |> Option.defaultValue valueTy
                    match bits valueTy, bits fieldTy with
                    | v, f when v > 0 && f > 0 && v < f ->
                        let extSSA = ssas.[1 + 3 * i]
                        ([ extensionOp state.Graph valueNode extSSA valueSSA valueTy fieldTy ], (extSSA, fieldTy))
                    | v, f when v > 0 && f > 0 && v > f ->
                        failwithf "pBuildRecord: field '%s' is %d bits but its value (node %d) is %d bits; the field's range is the join of every construction's" name f (NodeId.value valueNode) v
                    | _ -> ([], (valueSSA, valueTy)))
            let extOps = placed |> List.collect fst
            let fieldVals = placed |> List.map snd
            let actualStructTy = TStruct (List.zip fieldValues fieldVals |> List.map (fun ((name, _, _), (_, ty)) -> (name, ty)))
            let! createOp = pHWStructCreate resultSSA fieldVals actualStructTy
            return (extOps @ [createOp], TRValue { SSA = resultSSA; Type = actualStructTy })

        | _ ->
            // CPU: alloca + pTypedInsertView per field
            let (count, elemType) = extractMemRefShape structTy
            let memrefTy = TMemRefStatic (count, elemType)
            let allocSSA = ssas.[0]
            let! allocOp = pAllocValue nodeId allocSSA memrefTy

            let! fieldOps =
                fieldValues
                |> List.mapi (fun i (fieldName, valueSSA, fieldType) ->
                    parser {
                        let offsetSSA = ssas.[1 + 3*i]
                        let viewSSA   = ssas.[2 + 3*i]
                        let zeroSSA   = ssas.[3 + 3*i]
                        // Look up field by NAME in TStruct — field must exist
                        let! byteOffset =
                            match structTy with
                            | TStruct allFields ->
                                match structFieldLookup allFields fieldName with
                                | Some (fieldIdx, _) -> parser { return structFieldByteOffset allFields fieldIdx }
                                | None -> fail (Message $"Record field '{fieldName}' not found in struct type — CCS must provide complete field metadata")
                            | _ -> fail (Message $"Record construction requires TStruct type, got {structTy} — type mapping must produce TStruct for records")
                        return! pTypedInsertView allocSSA valueSSA byteOffset offsetSSA viewSSA zeroSSA fieldType memrefTy
                    })
                |> Alex.XParsec.Extensions.sequence

            let allOps = allocOp :: (List.concat fieldOps)
            // Store TStruct (logical type) in accumulator — downstream FieldGet needs field info.
            // typeToString(TStruct) serializes as memref<Nxi8> for CPU MLIR text.
            return (allOps, TRValue { SSA = allocSSA; Type = structTy })
    }

// ═══════════════════════════════════════════════════════════════════════════
// RECORD COPY-AND-UPDATE
// ═══════════════════════════════════════════════════════════════════════════

/// Copy-and-update record: copies original, then overwrites updated fields.
/// CPU:  alloca + memcpy from original + pTypedInsertView per updated field
/// FPGA: chain of hw.struct_inject from original
///
/// SSA layout (CPU):
///   ssas.[0] = alloca
///   ssas.[1] = extract_aligned_pointer_as_index of original
///   ssas.[2] = extract_aligned_pointer_as_index of new struct
///   ssas.[3] = index.casts src ptr to platform word
///   ssas.[4] = index.casts dst ptr to platform word
///   ssas.[5] = arith.constant totalSize (platform word)
///   ssas.[6] = memcpy result
///   Per updated field i: ssas.[7 + 3*i] = offset, ssas.[8 + 3*i] = view, ssas.[9 + 3*i] = zero
///
/// SSA layout (FPGA): per updated field i: ssas.[i] = inject result
let pBuildRecordCopyWith
    (nodeId: NodeId)
    (structTy: MLIRType)
    (origSSA: SSA)
    (updatedFields: (string * SSA * MLIRType) list)
    : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        let! platform = getTargetPlatform

        match platform with
        | Core.Types.Dialects.FPGA ->
            // FPGA: chain of hw.struct_inject from original
            let mutable currentSSA = origSSA
            let mutable ops = []
            for i in 0 .. updatedFields.Length - 1 do
                let (fieldName, valueSSA, _fieldType) = updatedFields.[i]
                let resultSSA = ssas.[i]
                let! injectOp = pHWStructInject resultSSA currentSSA fieldName valueSSA structTy
                ops <- ops @ [injectOp]
                currentSSA <- resultSSA
            return (ops, TRValue { SSA = currentSSA; Type = structTy })

        | _ ->
            // CPU: alloca + memcpy from original + overwrite updated fields
            let (count, elemType) = extractMemRefShape structTy
            let memrefTy = TMemRefStatic (count, elemType)
            let totalBytes = count
            let allocSSA = ssas.[0]
            let srcPtrIdxSSA = ssas.[1]
            let dstPtrIdxSSA = ssas.[2]
            let srcPtrWordSSA = ssas.[3]
            let dstPtrWordSSA = ssas.[4]
            let sizeSSA = ssas.[5]
            let memcpyResultSSA = ssas.[6]

            // 1. Alloca new struct
            let! allocOp = pAllocValue nodeId allocSSA memrefTy

            // 2. Extract pointers for memcpy
            let! extractSrcPtr = pExtractBasePtr srcPtrIdxSSA origSSA memrefTy
            let! extractDstPtr = pExtractBasePtr dstPtrIdxSSA allocSSA memrefTy

            // 3. Cast to platform word
            let! state = getUserState
            let platformWordTy = state.Platform.PlatformWordType
            let! castSrc = pIndexCastS srcPtrWordSSA srcPtrIdxSSA TIndex platformWordTy
            let! castDst = pIndexCastS dstPtrWordSSA dstPtrIdxSSA TIndex platformWordTy

            // 4. Size constant + memcpy call
            let! sizeOp = pConstI sizeSSA (int64 totalBytes) platformWordTy
            let memcpyArgs = [
                { SSA = dstPtrWordSSA; Type = platformWordTy }
                { SSA = srcPtrWordSSA; Type = platformWordTy }
                { SSA = sizeSSA; Type = platformWordTy }
            ]
            let! memcpyCall = pFuncCall (Some memcpyResultSSA) "memcpy" memcpyArgs platformWordTy
            let! memcpyDecl = pFuncDecl "memcpy" [platformWordTy; platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private
            let copyOps = [memcpyDecl; memcpyCall]

            // 5. Overwrite updated fields at correct byte offsets
            let! fieldOps =
                updatedFields
                |> List.mapi (fun i (fieldName, valueSSA, fieldType) ->
                    parser {
                        let offsetSSA = ssas.[7 + 3*i]
                        let viewSSA   = ssas.[8 + 3*i]
                        let zeroSSA   = ssas.[9 + 3*i]
                        let byteOffset =
                            match structTy with
                            | TStruct allFields ->
                                match structFieldLookup allFields fieldName with
                                | Some (fieldIdx, _) -> structFieldByteOffset allFields fieldIdx
                                | None -> 0
                            | _ -> 0
                        return! pTypedInsertView allocSSA valueSSA byteOffset offsetSSA viewSSA zeroSSA fieldType memrefTy
                    })
                |> Alex.XParsec.Extensions.sequence

            let allOps =
                [allocOp; extractSrcPtr; extractDstPtr; castSrc; castDst; sizeOp]
                @ copyOps
                @ (List.concat fieldOps)
            return (allOps, TRValue { SSA = allocSSA; Type = structTy })
    }

// ═══════════════════════════════════════════════════════════════════════════
// RECORD FIELD ASSIGNMENT
// ═══════════════════════════════════════════════════════════════════════════

/// Store a value into a named field of a TStruct record: `r.Field <- v`.
/// Records are memref-backed, so the store mutates the record in place and is
/// visible through every reference to it (including a parameter).
///
/// SSA layout (3 SSAs): [offset, view, zero]
let pRecordFieldSet
    (nodeId: NodeId)
    (structSSA: SSA)
    (fieldName: string)
    (structTy: MLIRType)
    (valueSSA: SSA)
    : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        match structTy with
        | TStruct fields ->
            match structFieldLookup fields fieldName with
            | Some (fieldIdx, fieldType) ->
                do! ensure (ssas.Length >= 3) $"pRecordFieldSet: Expected 3 SSAs, got {ssas.Length}"
                let offsetSSA = ssas.[0]
                let viewSSA   = ssas.[1]
                let zeroSSA   = ssas.[2]
                let byteOffset = structFieldByteOffset fields fieldIdx
                let memrefTy = TMemRefStatic (fields |> List.sumBy (fun (_, t) -> mlirTypeSize t), TInt (IntWidth 8))
                let! storeOps = pTypedInsertView structSSA valueSSA byteOffset offsetSSA viewSSA zeroSSA fieldType memrefTy
                return (storeOps, TRVoid)
            | None ->
                return! fail (Message $"pRecordFieldSet: Unknown field '{fieldName}' in TStruct")
        | _ ->
            return! fail (Message $"pRecordFieldSet: Expected TStruct, got {structTy}")
    }

// ═══════════════════════════════════════════════════════════════════════════
// RECORD FIELD ACCESS
// ═══════════════════════════════════════════════════════════════════════════

/// Extract a named field from a TStruct record — codata-dependent.
/// CPU:  pTypedExtractView at computed byte offset
/// FPGA: hw.struct_extract by field name
///
/// SSA layout: ssas has 4 entries [offset, view, zero, result] on CPU; [result] on FPGA
let pRecordFieldGet
    (nodeId: NodeId)
    (structSSA: SSA)
    (fieldName: string)
    (structTy: MLIRType)
    : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        let! platform = getTargetPlatform

        match structTy with
        | TStruct fields ->
            match structFieldLookup fields fieldName with
            | Some (fieldIdx, fieldType) ->
                match platform with
                | Core.Types.Dialects.FPGA ->
                    let resultSSA = List.last ssas
                    let! extractOp = pHWStructExtract resultSSA structSSA fieldName structTy
                    return ([extractOp], TRValue { SSA = resultSSA; Type = fieldType })

                | _ ->
                    // CPU: pTypedExtractView at byte offset
                    // SSAs: [offset, view, zero, result]
                    do! ensure (ssas.Length >= 4) $"pRecordFieldGet: Expected 4 SSAs, got {ssas.Length}"
                    let offsetSSA = ssas.[0]
                    let viewSSA   = ssas.[1]
                    let zeroSSA   = ssas.[2]
                    let resultSSA = ssas.[3]
                    let byteOffset = structFieldByteOffset fields fieldIdx
                    let memrefTy = TMemRefStatic (fields |> List.sumBy (fun (_, t) -> mlirTypeSize t), TInt (IntWidth 8))
                    let! extractOps = pTypedExtractView resultSSA structSSA byteOffset offsetSSA viewSSA zeroSSA fieldType memrefTy
                    return (extractOps, TRValue { SSA = resultSSA; Type = fieldType })

            | None ->
                return! fail (Message $"pRecordFieldGet: Unknown field '{fieldName}' in TStruct")

        | _ ->
            // Not a TStruct — skip, let MemoryPatterns handle it
            return! fail (Message $"pRecordFieldGet: Expected TStruct, got {structTy}")
    }
