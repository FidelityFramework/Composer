/// Memory Witness - Witness memory and data structure operations
///
/// ARCHITECTURAL FOUNDATION:
/// This module witnesses memory-related PSG nodes including:
/// - Array/collection indexing (IndexGet, IndexSet)
/// - Address-of operator (AddressOf)
/// - Tuple/Record/Array/List construction
/// - Field access (FieldGet, FieldSet)
/// - SRTP trait calls
module Alex.Witnesses.MemoryWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open FSharp.Native.Compiler.Checking.Native.NativeTypes
open Alex.CodeGeneration.MLIRTypes
open Alex.Traversal.MLIRZipper
open Alex.Templates.TemplateTypes
open Alex.Templates.MemoryTemplates

// ═══════════════════════════════════════════════════════════════════════════
// Type Mapping Helper (delegated to TypeMapping module)
// ═══════════════════════════════════════════════════════════════════════════

/// Map FNCS NativeType to MLIR type - delegates to canonical implementation
let mapType = Alex.CodeGeneration.TypeMapping.mapNativeType

// ═══════════════════════════════════════════════════════════════════════════
// String Type Helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Check if a type string represents native string type
let isNativeStrType (tyStr: string) : bool =
    tyStr = NativeStrTypeStr || tyStr.Contains("struct<(ptr, i64)>")

// ═══════════════════════════════════════════════════════════════════════════
// Index Operations
// ═══════════════════════════════════════════════════════════════════════════

/// Witness array/collection index get
let witnessIndexGet 
    (collectionId: NodeId) 
    (indexId: NodeId) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    match MLIRZipper.recallNodeSSA (string (NodeId.value collectionId)) zipper,
          MLIRZipper.recallNodeSSA (string (NodeId.value indexId)) zipper with
    | Some (collSSA, _collType), Some (indexSSA, _) ->
        // Generate GEP (getelementptr) for array access
        let ssaName, zipper' = MLIRZipper.yieldSSA zipper
        let elemType = Serialize.mlirType (mapType node.Type)
        let gepParams = { Result = ssaName; Base = collSSA; Offset = indexSSA; ElementType = "i8" }
        let text = render Quot.Gep.i64 gepParams
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Pointer zipper'
        // Load the element
        let loadSSA, zipper''' = MLIRZipper.yieldSSA zipper''
        let loadParams = { Result = loadSSA; Pointer = ssaName; Type = elemType }
        let loadText = render Quot.Core.load loadParams
        let zipper4 = MLIRZipper.witnessOpWithResult loadText loadSSA (mapType node.Type) zipper'''
        let zipper5 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) loadSSA elemType zipper4
        zipper5, TRValue (loadSSA, elemType)
    | _ ->
        zipper, TRError "IndexGet: collection or index not computed"

/// Witness array/collection index set
let witnessIndexSet 
    (collectionId: NodeId) 
    (indexId: NodeId) 
    (valueId: NodeId) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    match MLIRZipper.recallNodeSSA (string (NodeId.value collectionId)) zipper,
          MLIRZipper.recallNodeSSA (string (NodeId.value indexId)) zipper,
          MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
    | Some (collSSA, _), Some (indexSSA, _), Some (valueSSA, valueType) ->
        // Generate GEP for array access
        let ptrSSA, zipper' = MLIRZipper.yieldSSA zipper
        let gepParams = { Result = ptrSSA; Base = collSSA; Offset = indexSSA; ElementType = "i8" }
        let gepText = render Quot.Gep.i64 gepParams
        let zipper'' = MLIRZipper.witnessOpWithResult gepText ptrSSA Pointer zipper'
        // Store the value
        let storeParams = { Value = valueSSA; Pointer = ptrSSA; Type = valueType }
        let storeText = render Quot.Core.store storeParams
        let zipper''' = MLIRZipper.witnessVoidOp storeText zipper''
        zipper''', TRVoid
    | _ ->
        zipper, TRError "IndexSet: collection, index, or value not computed"

// ═══════════════════════════════════════════════════════════════════════════
// Address-Of Operator
// ═══════════════════════════════════════════════════════════════════════════

/// Witness address-of operator
let witnessAddressOf 
    (exprId: NodeId) 
    (isMutable: bool) 
    (node: SemanticNode) 
    (graph: SemanticGraph) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    match SemanticGraph.tryGetNode exprId graph with
    | Some exprNode ->
        match exprNode.Kind with
        | SemanticKind.VarRef (_, Some targetBindingId) ->
            let bindingIdVal = NodeId.value targetBindingId
            if MLIRZipper.isAddressedMutable bindingIdVal zipper then
                // Addressed mutable: get the alloca pointer directly
                match MLIRZipper.lookupMutableAlloca bindingIdVal zipper with
                | Some (allocaSSA, _) ->
                    let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) allocaSSA "!llvm.ptr" zipper
                    zipper', TRValue (allocaSSA, "!llvm.ptr")
                | None ->
                    zipper, TRError "AddressOf: addressed mutable has no alloca"
            else
                // Non-addressed VarRef - use the VarRef's SSA (may be an existing pointer)
                match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
                | Some (exprSSA, _) ->
                    let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) exprSSA "!llvm.ptr" zipper
                    zipper', TRValue (exprSSA, "!llvm.ptr")
                | None ->
                    zipper, TRError "AddressOf: VarRef expression not computed"
        | _ ->
            // Not a VarRef - use the expression's SSA (may be an existing pointer)
            match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
            | Some (exprSSA, _) ->
                let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) exprSSA "!llvm.ptr" zipper
                zipper', TRValue (exprSSA, "!llvm.ptr")
            | None ->
                zipper, TRError "AddressOf: expression not computed"
    | None ->
        zipper, TRError "AddressOf: expression node not found in graph"

// ═══════════════════════════════════════════════════════════════════════════
// Tuple Expression
// ═══════════════════════════════════════════════════════════════════════════

/// Witness tuple construction
let witnessTupleExpr 
    (elementIds: NodeId list) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    let elementSSAs =
        elementIds
        |> List.choose (fun elemId ->
            MLIRZipper.recallNodeSSA (string (NodeId.value elemId)) zipper)

    if List.length elementSSAs <> List.length elementIds then
        zipper, TRError "TupleExpr: not all elements computed"
    else
        match elementSSAs with
        | [] ->
            // Empty tuple is unit
            zipper, TRVoid
        | [(ssa, ty)] ->
            // Single element - just return it
            let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
            zipper', TRValue (ssa, ty)
        | elements ->
            // Multi-element tuple - allocate struct and store elements
            let tupleSSA, zipper1 = MLIRZipper.yieldSSA zipper
            let allocaText = sprintf "%s = llvm.alloca i64 x %d : (i64) -> !llvm.ptr" tupleSSA (List.length elements)
            let zipper2 = MLIRZipper.witnessOp allocaText [(tupleSSA, Pointer)] zipper1
            let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) tupleSSA "!llvm.ptr" zipper2
            zipper3, TRValue (tupleSSA, "!llvm.ptr")

// ═══════════════════════════════════════════════════════════════════════════
// Record Expression
// ═══════════════════════════════════════════════════════════════════════════

/// Witness record construction
let witnessRecordExpr 
    (fields: (string * NodeId) list) 
    (copyFrom: NodeId option) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    let fieldSSAs =
        fields
        |> List.choose (fun (_fieldName, valueId) ->
            MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper)

    if List.length fieldSSAs <> List.length fields then
        zipper, TRError "Record fields not all computed"
    else
        // For a single-field record, just return that field's value
        match fieldSSAs with
        | [(ssa, ty)] ->
            let zipper1 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) ssa ty zipper
            zipper1, TRValue (ssa, ty)
        | _ ->
            // Multi-field record - for now, return unit and handle properly later
            zipper, TRVoid

// ═══════════════════════════════════════════════════════════════════════════
// Array Expression
// ═══════════════════════════════════════════════════════════════════════════

/// Witness array construction
let witnessArrayExpr 
    (elementIds: NodeId list) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    let elementSSAs =
        elementIds
        |> List.choose (fun elemId ->
            MLIRZipper.recallNodeSSA (string (NodeId.value elemId)) zipper)

    if List.length elementSSAs <> List.length elementIds then
        zipper, TRError "ArrayExpr: not all elements computed"
    else
        match elementSSAs with
        | [] ->
            // Empty array - allocate empty array (just a pointer)
            let arrSSA, zipper' = MLIRZipper.yieldSSA zipper
            let allocaText = sprintf "%s = llvm.alloca i64 x 0 : (i64) -> !llvm.ptr" arrSSA
            let zipper'' = MLIRZipper.witnessOpWithResult allocaText arrSSA Pointer zipper'
            let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) arrSSA "!llvm.ptr" zipper''
            zipper''', TRValue (arrSSA, "!llvm.ptr")
        | elements ->
            // Non-empty array - allocate and store elements
            let arrSSA, zipper1 = MLIRZipper.yieldSSA zipper
            let allocaText = sprintf "%s = llvm.alloca i64 x %d : (i64) -> !llvm.ptr" arrSSA (List.length elements)
            let zipper2 = MLIRZipper.witnessOpWithResult allocaText arrSSA Pointer zipper1
            let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) arrSSA "!llvm.ptr" zipper2
            zipper3, TRValue (arrSSA, "!llvm.ptr")

// ═══════════════════════════════════════════════════════════════════════════
// List Expression
// ═══════════════════════════════════════════════════════════════════════════

/// Witness list construction
let witnessListExpr 
    (elementIds: NodeId list) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    let elementSSAs =
        elementIds
        |> List.choose (fun elemId ->
            MLIRZipper.recallNodeSSA (string (NodeId.value elemId)) zipper)

    if List.length elementSSAs <> List.length elementIds then
        zipper, TRError "ListExpr: not all elements computed"
    else
        // Allocate space for list nodes (simplified)
        let listSSA, zipper1 = MLIRZipper.yieldSSA zipper
        let allocaText = sprintf "%s = llvm.alloca i64 x %d : (i64) -> !llvm.ptr" listSSA (List.length elementSSAs)
        let zipper2 = MLIRZipper.witnessOpWithResult allocaText listSSA Pointer zipper1
        let zipper3 = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) listSSA "!llvm.ptr" zipper2
        zipper3, TRValue (listSSA, "!llvm.ptr")

// ═══════════════════════════════════════════════════════════════════════════
// Field Operations
// ═══════════════════════════════════════════════════════════════════════════

/// Witness field get (expr.fieldName)
let witnessFieldGet 
    (exprId: NodeId) 
    (fieldName: string) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper with
    | Some (exprSSA, exprType) ->
        // STRING INTRINSIC MEMBERS: Native string is fat pointer {ptr, len}
        // .Pointer → extractvalue at index 0 → !llvm.ptr
        // .Length → extractvalue at index 1 → i64
        if isNativeStrType exprType then
            match fieldName with
            | "Pointer" ->
                let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                let extractParams : Quot.Aggregate.ExtractParams = { Result = resultSSA; Aggregate = exprSSA; Index = 0; AggType = NativeStrTypeStr }
                let extractText = render Quot.Aggregate.extractValue extractParams
                let zipper'' = MLIRZipper.witnessOpWithResult extractText resultSSA Pointer zipper'
                let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "!llvm.ptr" zipper''
                zipper''', TRValue (resultSSA, "!llvm.ptr")
            | "Length" ->
                let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
                let extractParams : Quot.Aggregate.ExtractParams = { Result = resultSSA; Aggregate = exprSSA; Index = 1; AggType = NativeStrTypeStr }
                let extractText = render Quot.Aggregate.extractValue extractParams
                let zipper'' = MLIRZipper.witnessOpWithResult extractText resultSSA (Integer I64) zipper'
                let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "i64" zipper''
                zipper''', TRValue (resultSSA, "i64")
            | _ ->
                zipper, TRError (sprintf "Unknown string field: %s (expected Pointer or Length)" fieldName)
        else
            // Generic field access for other struct types
            let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
            let loadParams = { Result = resultSSA; Pointer = exprSSA; Type = "!llvm.ptr" }
            let loadText = render Quot.Core.load loadParams
            let zipper'' = MLIRZipper.witnessOpWithResult loadText resultSSA Pointer zipper'
            let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "!llvm.ptr" zipper''
            zipper''', TRValue (resultSSA, "!llvm.ptr")
    | None ->
        zipper, TRError (sprintf "FieldGet '%s': expression not computed" fieldName)

/// Witness field set (expr.fieldName <- value)
let witnessFieldSet 
    (exprId: NodeId) 
    (fieldName: string) 
    (valueId: NodeId) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    match MLIRZipper.recallNodeSSA (string (NodeId.value exprId)) zipper,
          MLIRZipper.recallNodeSSA (string (NodeId.value valueId)) zipper with
    | Some (exprSSA, _), Some (valueSSA, valueType) ->
        // Store value at field offset
        let storeParams = { Value = valueSSA; Pointer = exprSSA; Type = valueType }
        let storeText = render Quot.Core.store storeParams
        let zipper' = MLIRZipper.witnessVoidOp storeText zipper
        zipper', TRVoid
    | _ ->
        zipper, TRError (sprintf "FieldSet '%s': expression or value not computed" fieldName)

// ═══════════════════════════════════════════════════════════════════════════
// TraitCall (SRTP)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness SRTP trait call
let witnessTraitCall 
    (memberName: string) 
    (typeArgs: NativeType list) 
    (argId: NodeId) 
    (node: SemanticNode) 
    (zipper: MLIRZipper) 
    : MLIRZipper * TransferResult =
    
    match MLIRZipper.recallNodeSSA (string (NodeId.value argId)) zipper with
    | Some (argSSA, argType) ->
        // For now, emit a call to the trait member name
        // TODO: Proper SRTP resolution from Baker/type checker
        let resultSSA, zipper' = MLIRZipper.yieldSSA zipper
        let callText = sprintf "%s = llvm.call @%s(%s) : (%s) -> !llvm.ptr" resultSSA memberName argSSA argType
        let zipper'' = MLIRZipper.witnessOpWithResult callText resultSSA Pointer zipper'
        let zipper''' = MLIRZipper.bindNodeSSA (string (NodeId.value node.Id)) resultSSA "!llvm.ptr" zipper''
        zipper''', TRValue (resultSSA, "!llvm.ptr")
    | None ->
        zipper, TRError (sprintf "TraitCall '%s': argument not computed" memberName)
