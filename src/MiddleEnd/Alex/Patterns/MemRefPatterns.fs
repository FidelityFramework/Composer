/// MemRefPatterns - Composable patterns for memref operations
///
/// PUBLIC: Witnesses import these patterns to emit memref operations.
/// Patterns compose Element primitives into domain-specific operations.
///
/// ARCHITECTURE: Patterns are PUBLIC, Elements are module internal.
/// This enforces composition - witnesses cannot directly emit Element operations.
module Alex.Patterns.MemRefPatterns

open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Traversal.TransferTypes
open Alex.Elements.MemRefElements
open Alex.Elements.IndexElements  // pIndexConst
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════
// MUTABLE VARIABLE PATTERNS
// ═══════════════════════════════════════════════════════════

/// Build complete mutable binding: alloca + initialize with store
/// For F# `let mutable x = initialValue`
///
/// Emits:
///   %xRef = memref.alloca() : memref<1x{elemType}>
///   memref.store %initialValue, %xRef[%c0] : {elemType}, memref<1x{elemType}>
///
/// Returns: memref SSA name (for VarRef to recall)
let pBuildMutableBinding (nodeId: int) (elemType: MLIRType) (initSSA: SSA) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Step 1: Get pre-allocated SSAs from Coeffects
        let! ssas = getNodeSSAs (NodeId nodeId)
        do! ensure (ssas.Length >= 2) $"pBuildMutableBinding: Expected at least 2 SSAs, got {ssas.Length}"
        let memrefSSA = ssas.[0]  // For memref allocation result
        let zeroSSA = ssas.[1]    // For zero constant index

        // Step 2: Allocate memref (rank-1 with single element for scalar)
        let! allocOp = pAlloca memrefSSA 1 elemType None

        // Step 3: Create constant index for scalar memref (always %c0)
        let! zeroOp = pIndexConst zeroSSA 0L

        // Step 4: Store initial value to memref
        let memrefType = TMemRefStatic (1, elemType)
        let! storeOp = pStore initSSA memrefSSA [zeroSSA] elemType memrefType

        let ops = [allocOp; zeroOp; storeOp]
        // Semantic type: TMemRef (abstract mutable cell) — distinguishes from TMemRefStatic (buffers)
        // Physical type (TMemRefStatic) preserved in SSATypes by pAlloca for pLoad derivation
        let result = TRValue { SSA = memrefSSA; Type = TMemRef elemType }
        return (ops, result)
    }

/// Load value from mutable variable
/// For F# VarRef to mutable binding
///
/// Emits:
///   %c0 = arith.constant 0 : index
///   %value = memref.load %memrefSSA[%c0] : memref<1x{elemType}>
///
/// Returns: loaded value SSA (type = elemType, not TMemRef)
let pLoadMutableVariable (nodeId: int) (memrefSSA: SSA) (elemType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Step 1: Get pre-allocated SSAs from Coeffects
        let! ssas = getNodeSSAs (NodeId nodeId)
        do! ensure (ssas.Length >= 2) $"pLoadMutableVariable: Expected at least 2 SSAs, got {ssas.Length}"
        let zeroSSA = ssas.[0]    // For zero constant index
        let valueSSA = ssas.[1]   // For loaded value result

        // Step 2: Create constant index for scalar memref (always %c0)
        let! zeroOp = pIndexConst zeroSSA 0L

        // Step 3: Load value from memref
        let! loadOp = pLoad valueSSA memrefSSA [zeroSSA]

        let ops = [zeroOp; loadOp]
        let result = TRValue { SSA = valueSSA; Type = elemType }
        return (ops, result)
    }

/// Store value to mutable variable
/// For F# `x <- newValue` (SemanticKind.Set)
///
/// Emits:
///   %c0 = arith.constant 0 : index
///   memref.store %newValue, %memrefSSA[%c0] : {elemType}, memref<1x{elemType}>
///
/// Returns: TRVoid (stores don't produce values)
let pStoreMutableVariable (nodeId: int) (memrefSSA: SSA) (valueSSA: SSA) (elemType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        // Step 1: Get pre-allocated SSA from Coeffects for zero constant
        let! ssas = getNodeSSAs (NodeId nodeId)
        do! ensure (ssas.Length >= 1) $"pStoreMutableVariable: Expected at least 1 SSA, got {ssas.Length}"
        let zeroSSA = ssas.[0]  // For zero constant index

        // Step 2: Create constant index for scalar memref (always %c0)
        let! zeroOp = pIndexConst zeroSSA 0L

        // Step 3: Store value to memref
        let memrefType = TMemRefStatic (1, elemType)
        let! storeOp = pStore valueSSA memrefSSA [zeroSSA] elemType memrefType

        let ops = [zeroOp; storeOp]
        return (ops, TRVoid)
    }

// ═══════════════════════════════════════════════════════════
// ADDRESS-OF PATTERN
// ═══════════════════════════════════════════════════════════

/// Build address-of for Clef `&expr`.
///
/// Two cases based on the element type:
///
/// 1. Memref-backed values (records, buffers): the data already lives behind
///    the memref descriptor's aligned pointer. Extract it directly.
///    Emits:
///      %ptr = memref.extract_aligned_pointer_as_index %value : memref<Nxi8> -> index
///
/// 2. Scalar values: materialize on the stack via alloca + store, then extract.
///    Emits:
///      %ref = memref.alloca() : memref<1x{elemType}>
///      %c0 = arith.constant 0 : index
///      memref.store %value, %ref[%c0] : {elemType}, memref<1x{elemType}>
///      %ptr = memref.extract_aligned_pointer_as_index %ref : memref<1x{elemType}> -> index
///
/// Returns: index SSA (the raw pointer)
/// TByref maps to TIndex in TypeMapping — AddressOf produces a raw pointer.
let pBuildAddressOf (nodeId: NodeId) (valueSSA: SSA) (elemType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        // Determine if the value is memref-backed (data behind the descriptor's pointer).
        // TStruct is the semantic type for records; physical representation is memref<Nxi8>.
        let physicalMemrefType =
            match elemType with
            | TMemRefStatic _ | TMemRef _ | TMemRefScalar _ -> Some elemType
            | TStruct _ -> Some (TMemRefStatic (mlirTypeSize elemType, TInt (IntWidth 8)))
            | _ -> None
        match physicalMemrefType with
        // Memref-backed values: extract the data pointer directly — no alloca.
        | Some memrefTy ->
            do! ensure (ssas.Length >= 1) $"pBuildAddressOf: Expected at least 1 SSA, got {ssas.Length}"
            let ptrSSA = ssas.[0]
            let! extractOp = pExtractBasePtr ptrSSA valueSSA memrefTy
            let result = TRValue { SSA = ptrSSA; Type = TIndex }
            return ([extractOp], result)
        // Scalar values: alloca + store + extract to materialize on stack
        | _ ->
            do! ensure (ssas.Length >= 3) $"pBuildAddressOf: Expected at least 3 SSAs, got {ssas.Length}"
            let memrefSSA = ssas.[0]
            let zeroSSA = ssas.[1]
            let ptrSSA = ssas.[2]
            let! allocOp = pAlloca memrefSSA 1 elemType None
            let! zeroOp = pIndexConst zeroSSA 0L
            let memrefType = TMemRefStatic (1, elemType)
            let! storeOp = pStore valueSSA memrefSSA [zeroSSA] elemType memrefType
            let! extractOp = pExtractBasePtr ptrSSA memrefSSA memrefType
            let ops = [allocOp; zeroOp; storeOp; extractOp]
            let result = TRValue { SSA = ptrSSA; Type = TIndex }
            return (ops, result)
    }
