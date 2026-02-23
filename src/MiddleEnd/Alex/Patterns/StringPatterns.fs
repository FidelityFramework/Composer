/// StringPatterns - String operation elision patterns
///
/// Provides composable patterns for string operations using XParsec.
/// Strings are represented as fat pointers: {ptr: index, length: int}
///
/// ARCHITECTURAL RESTORATION (Feb 2026): All patterns use NodeId-based API.
/// Patterns extract SSAs monadically via getNodeSSAs - witnesses pass NodeIds, not SSAs.
module Alex.Patterns.StringPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.FuncElements
open Alex.Elements.MemRefElements
open Alex.Elements.ArithElements
open Alex.Elements.MLIRAtomics
open Alex.Elements.IndexElements
open Alex.Elements.SCFElements
open Alex.CodeGeneration.TypeMapping
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Patterns.MemoryPatterns // pRecallArgWithLoad, pDetectMemRefAddFusion

// ═══════════════════════════════════════════════════════════
// MEMORY COPY PATTERN (composed from FuncElements)
// ═══════════════════════════════════════════════════════════

/// Build memcpy external call operation (composed from pFuncCall Element)
/// External memcpy declaration will be added at module level if needed
/// resultSSA: The SSA assigned to the memcpy result (from coeffects analysis)
let pStringMemCopy (resultSSA: SSA) (destSSA: SSA) (srcSSA: SSA) (lenSSA: SSA) : PSGParser<MLIROp list> =
    parser {
        // Get platform word type (pointers are platform words)
        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        // Build memcpy call: void* memcpy(void* dest, const void* src, size_t n)
        let args = [
            { SSA = destSSA; Type = platformWordTy }   // dest
            { SSA = srcSSA; Type = platformWordTy }    // src
            { SSA = lenSSA; Type = platformWordTy }    // len
        ]

        // Call external memcpy - uses result SSA from coeffects analysis
        let! call = pFuncCall (Some resultSSA) "memcpy" args platformWordTy
        let! memcpyDecl = pFuncDecl "memcpy" [platformWordTy; platformWordTy; platformWordTy] platformWordTy FuncVisibility.Private

        return [memcpyDecl; call]
    }

// ═══════════════════════════════════════════════════════════
// STRING CONSTRUCTION
// ═══════════════════════════════════════════════════════════

/// Convert static buffer to dynamic string (memref<?xi8>)
/// Takes buffer (memref<Nxi8>), casts to memref<?xi8>
/// Length is intrinsic to memref descriptor, no separate parameter needed
/// SSA extracted from coeffects via nodeId: [0] = result
let pStringFromBuffer (nodeId: NodeId) (bufferSSA: SSA) (bufferType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pStringFromBuffer: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Cast static memref to dynamic for function boundary
        // memref<Nxi8> -> memref<?xi8>
        let stringType = TMemRef (TInt (IntWidth 8))
        let! castOp = pMemRefCast resultSSA bufferSSA bufferType stringType
        return ([castOp], TRValue { SSA = resultSSA; Type = stringType })
    }

/// Create substring from buffer pointer and length (FNCS NativeStr.fromPointer contract)
/// FNCS contract (Intrinsics.fs:372): "creates a new memref<?xi8> with specified length"
/// This is NOT a cast - it's a substring extraction via allocate + memcpy.
/// When srcOffset is Some, the source pointer is adjusted by adding the offset (for NativePtr.add fusion).
/// SSA extracted from coeffects via nodeId (7 minimum, 8 when srcOffset provided):
///   [0] = resultBufferSSA (allocated memref<?xi8> with actual length)
///   [1] = srcPtrSSA (extract pointer from source buffer)
///   [2] = destPtrSSA (extract pointer from result buffer)
///   [3] = srcPtrWord (cast source ptr to platform word)
///   [4] = destPtrWord (cast dest ptr to platform word)
///   [5] = lenWord (cast length to platform word)
///   [6] = memcpyResultSSA (result of memcpy call)
///   [7] = adjustedSrcPtrSSA (base ptr + offset, only when srcOffset is Some)
let pStringFromPointerWithLength (nodeId: NodeId) (bufferSSA: SSA) (lengthSSA: SSA) (bufferType: MLIRType) (srcOffset: SSA option) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        let minSSAs = match srcOffset with Some _ -> 8 | None -> 7
        do! ensure (ssas.Length >= minSSAs) $"pStringFromPointerWithLength: Expected {minSSAs} SSAs, got {ssas.Length}"

        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        let resultBufferSSA = ssas.[0]
        let srcPtrSSA = ssas.[1]
        let destPtrSSA = ssas.[2]
        let srcPtrWord = ssas.[3]
        let destPtrWord = ssas.[4]
        let lenWord = ssas.[5]
        let memcpyResultSSA = ssas.[6]

        // 1. Allocate result buffer with runtime size (lengthSSA is index type from nativeint)
        let resultTy = TMemRef (TInt (IntWidth 8))
        let! allocOp = pAlloc resultBufferSSA lengthSSA (TInt (IntWidth 8))

        // 2. Extract pointers from source buffer and result buffer
        let! extractSrcPtr = pExtractBasePtr srcPtrSSA bufferSSA bufferType
        let! extractDestPtr = pExtractBasePtr destPtrSSA resultBufferSSA resultTy

        // 2b. If srcOffset provided, adjust source pointer: adjusted = base_ptr + offset
        // This handles NativePtr.add(buf, startPos) where we need to copy from buf[startPos]
        let (effectiveSrcPtrSSA, adjustOps) =
            match srcOffset with
            | Some offsetSSA ->
                let adjustedSSA = ssas.[7]
                let adjustOp = MLIROp.ArithOp (ArithOp.AddI (adjustedSSA, srcPtrSSA, offsetSSA, TIndex))
                (adjustedSSA, [adjustOp])
            | None ->
                (srcPtrSSA, [])

        // 3. Cast pointers to platform words for memcpy FFI
        let! castSrc = pIndexCastS srcPtrWord effectiveSrcPtrSSA TIndex platformWordTy
        let! castDest = pIndexCastS destPtrWord destPtrSSA TIndex platformWordTy

        // 4. Cast length to platform word for memcpy FFI (index → platform word)
        let! castLen = pIndexCastS lenWord lengthSSA TIndex platformWordTy

        // 5. Call memcpy(dest, src, len) - copies 'length' bytes from buffer to new memref
        let! copyOps = pStringMemCopy memcpyResultSSA destPtrWord srcPtrWord lenWord

        // 6. Return new memref<?xi8> with actual length
        // FNCS contract satisfied: "creates a NEW memref<?xi8> with SPECIFIED LENGTH"
        let ops =
            [allocOp] @
            [extractSrcPtr; extractDestPtr] @
            adjustOps @
            [castSrc; castDest; castLen] @
            copyOps

        return (ops, TRValue { SSA = resultBufferSSA; Type = resultTy })
    }

/// Get string length via memref.dim
/// Strings ARE memrefs (memref<?xi8>), length is intrinsic to descriptor
/// SSA extracted from coeffects via nodeId:
///   [0] = dimConstSSA: dimension constant (0 for 1D arrays)
///   [1] = lenIndexSSA: memref.dim result (index type)
///   [2] = resultSSA: cast to int result
let pStringLength (nodeId: NodeId) (stringSSA: SSA) (stringType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 3) $"pStringLength: Expected 3 SSAs, got {ssas.Length}"
        let dimConstSSA = ssas.[0]
        let lenIndexSSA = ssas.[1]
        let resultSSA = ssas.[2]

        let! state = getUserState
        let intTy = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph Types.intType

        // Get string dimension (dimension 0 for 1D memref)
        let! dimConstOp = pConstI dimConstSSA 0L TIndex
        let! dimOp = pMemRefDim lenIndexSSA stringSSA dimConstSSA stringType

        // Cast index to int
        let! castOp = pIndexCastS resultSSA lenIndexSSA TIndex intTy

        return ([dimConstOp; dimOp; castOp], TRValue { SSA = resultSSA; Type = intTy })
    }

/// Get character at index via memref.load
/// Strings ARE memrefs (memref<?xi8>), charAt loads byte at offset
/// SSA extracted from coeffects via nodeId:
///   [0] = idxIndexSSA: index_cast int→index (for memref indexing)
///   [1] = loadSSA: memref.load result (i8)
///   [2] = resultSSA: extui i8→i32 (char)
let pStringCharAt (nodeId: NodeId) (stringSSA: SSA) (indexSSA: SSA) (indexType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 3) $"pStringCharAt: Expected 3 SSAs, got {ssas.Length}"
        let idxIndexSSA = ssas.[0]
        let loadSSA = ssas.[1]
        let resultSSA = ssas.[2]

        let! state = getUserState
        let charTy = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph Types.charType

        // Cast int index to index type for memref indexing (e.g., i32 → index)
        let! castIdxOp = pIndexCastS idxIndexSSA indexSSA indexType TIndex

        // Load byte from string at index: memref.load %str[%idx] : memref<?xi8>
        let! loadOp = pLoadFrom loadSSA stringSSA [idxIndexSSA] (TInt (IntWidth 8))

        // Extend i8 to char type: arith.extui
        let! extOp = pExtUI resultSSA loadSSA (TInt (IntWidth 8)) charTy

        return ([castIdxOp; loadOp; extOp], TRValue { SSA = resultSSA; Type = charTy })
    }

// ═══════════════════════════════════════════════════════════
// STRING SCANNING PATTERN
// ═══════════════════════════════════════════════════════════

/// String.contains: scan string bytes for a target character
/// Uses scf.while with memref.alloca for mutable loop state (found flag + index counter)
/// Iteration counter uses TIndex (platform word width) — no concrete integer casts needed
/// Exact SSA expansion (20 slots, contiguous):
///   Setup [0..5]:   zeroIdx, strLen, charTrunc, foundBuf, idxBuf, zeroI8
///   Cond  [6..10]:  cLoadIdx, cCmpLt, cLoadFound, cNotFound, cBoth
///   Body  [11..17]: bLoadIdx, bLoadByte, bCmpEq, bOneI8, bSelFound, bOneIdx, bNextIdx
///   Post  [18..19]: pLoadFound, pResultBool
///   Non-SSA ops: 2 stores (init), scf.condition, 2 stores (body), scf.yield, scf.while
let pStringContains (nodeId: NodeId) (stringSSA: SSA) (charSSA: SSA) (charType: MLIRType) (stringType: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 20) $"pStringContains: Expected 20 SSAs, got {ssas.Length}"

        // Setup region (6 SSAs) [0..5]
        let zeroIdx    = ssas.[0]   // arith.constant 0 : index (dual: dim subscript + alloca index + initial counter)
        let strLen     = ssas.[1]   // memref.dim → index
        let charTrunc  = ssas.[2]   // arith.trunci char → i8
        let foundBuf   = ssas.[3]   // memref.alloca : memref<1xi8>
        let idxBuf     = ssas.[4]   // memref.alloca : memref<1xindex>
        let zeroI8     = ssas.[5]   // arith.constant 0 : i8

        // Condition region (5 SSAs) [6..10]
        let cLoadIdx   = ssas.[6]   // memref.load idxBuf → index
        let cCmpLt     = ssas.[7]   // index.cmp slt → i1
        let cLoadFound = ssas.[8]   // memref.load foundBuf → i8
        let cNotFound  = ssas.[9]   // arith.cmpi eq → i1
        let cBoth      = ssas.[10]  // arith.andi → i1

        // Body region (7 SSAs) [11..17]
        let bLoadIdx   = ssas.[11]  // memref.load idxBuf → index
        let bLoadByte  = ssas.[12]  // memref.load string[idx] → i8
        let bCmpEq     = ssas.[13]  // arith.cmpi eq → i1
        let bOneI8     = ssas.[14]  // arith.constant 1 : i8
        let bSelFound  = ssas.[15]  // arith.select → i8
        let bOneIdx    = ssas.[16]  // arith.constant 1 : index
        let bNextIdx   = ssas.[17]  // index.add → index

        // Post-loop (2 SSAs) [18..19]
        let pLoadFound = ssas.[18]  // memref.load foundBuf → i8
        let pResultBool = ssas.[19] // arith.cmpi ne → i1

        let i8Ty  = TInt (IntWidth 8)
        let i1Ty  = TInt (IntWidth 1)
        let foundBufTy = TMemRefStatic (1, i8Ty)
        let idxBufTy   = TMemRefStatic (1, TIndex)

        // ── Setup: get length, truncate char, alloca mutable state ──
        let! zeroIdxOp      = pIndexConst zeroIdx 0L
        let! strLenOp       = pMemRefDim strLen stringSSA zeroIdx stringType
        let! charTruncOp    = pTruncI charTrunc charSSA charType i8Ty
        let! foundBufOp     = pAlloca foundBuf 1 i8Ty None
        let! idxBufOp       = pAlloca idxBuf 1 TIndex None
        let! zeroI8Op       = pConstI zeroI8 0L i8Ty
        let! storeFoundInit = pStore zeroI8 foundBuf [zeroIdx] i8Ty foundBufTy
        let! storeIdxInit   = pStore zeroIdx idxBuf [zeroIdx] TIndex idxBufTy

        // ── Condition region: idx < len AND NOT found ──
        let! cLoadIdxOp    = pLoadFrom cLoadIdx idxBuf [zeroIdx] TIndex
        let! cCmpLtOp      = pIndexCmp cCmpLt IndexCmpPred.Slt cLoadIdx strLen
        let! cLoadFoundOp  = pLoadFrom cLoadFound foundBuf [zeroIdx] i8Ty
        let! cNotFoundOp   = pCmpI cNotFound ICmpPred.Eq cLoadFound zeroI8 i8Ty
        let! cBothOp       = pAndI cBoth cCmpLt cNotFound i1Ty
        let! cCondOp       = pSCFCondition cBoth []

        let condOps = [cLoadIdxOp; cCmpLtOp; cLoadFoundOp; cNotFoundOp; cBothOp; cCondOp]

        // ── Body region: load byte, compare, update found+index ──
        let! bLoadIdxOp    = pLoadFrom bLoadIdx idxBuf [zeroIdx] TIndex
        let! bLoadByteOp   = pLoadFrom bLoadByte stringSSA [bLoadIdx] i8Ty
        let! bCmpEqOp      = pCmpI bCmpEq ICmpPred.Eq bLoadByte charTrunc i8Ty
        let! bOneI8Op      = pConstI bOneI8 1L i8Ty
        let! bSelFoundOp   = pSelect bSelFound bCmpEq bOneI8 zeroI8 i8Ty
        let! bStoreFound   = pStore bSelFound foundBuf [zeroIdx] i8Ty foundBufTy
        let! bOneIdxOp     = pIndexConst bOneIdx 1L
        let! bNextIdxOp    = pIndexAdd bNextIdx bLoadIdx bOneIdx
        let! bStoreIdx     = pStore bNextIdx idxBuf [zeroIdx] TIndex idxBufTy
        let! bYieldOp      = pSCFYield []

        let bodyOps = [bLoadIdxOp; bLoadByteOp; bCmpEqOp; bOneI8Op; bSelFoundOp; bStoreFound; bOneIdxOp; bNextIdxOp; bStoreIdx; bYieldOp]

        // ── While loop ──
        let! whileOp = pSCFWhile condOps bodyOps

        // ── Post-loop: load found flag, convert to bool ──
        let! pLoadFoundOp   = pLoadFrom pLoadFound foundBuf [zeroIdx] i8Ty
        let! pResultBoolOp  = pCmpI pResultBool ICmpPred.Ne pLoadFound zeroI8 i8Ty

        let setupOps = [zeroIdxOp; strLenOp; charTruncOp; foundBufOp; idxBufOp; zeroI8Op; storeFoundInit; storeIdxInit]
        let ops = setupOps @ [whileOp] @ [pLoadFoundOp; pResultBoolOp]

        return (ops, TRValue { SSA = pResultBool; Type = i1Ty })
    }

// ═══════════════════════════════════════════════════════════
// STRING CONCATENATION PATTERN
// ═══════════════════════════════════════════════════════════

/// String.concat2: concatenate two strings using pure memref operations
/// SSA extracted from coeffects via nodeId (18 total - pure memref with index arithmetic, NO i64 round-trip):
///   [0] = dim const for str1 (dimension 0)
///   [1] = str1 len (memref.dim result, index type)
///   [2] = dim const for str2 (dimension 0)
///   [3] = str2 len (memref.dim result, index type)
///   [4] = combined length (index type - NO cast to i64!)
///   [5] = result buffer (memref)
///   [6] = str1 ptr (memref.extract_aligned_pointer_as_index)
///   [7] = str2 ptr (memref.extract_aligned_pointer_as_index)
///   [8] = result ptr (memref.extract_aligned_pointer_as_index)
///   [9] = str1 ptr word cast
///   [10] = str2 ptr word cast
///   [11] = result ptr word cast
///   [12] = str1 len word cast
///   [13] = str2 len word cast
///   [14] = memcpy1 result
///   [15] = offset ptr (result + len1, index type)
///   [16] = offset ptr word cast
///   [17] = memcpy2 result
let pStringConcat2 (nodeId: NodeId) (str1SSA: SSA) (str2SSA: SSA) (str1Type: MLIRType) (str2Type: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 18) $"pStringConcat2: Expected 18 SSAs, got {ssas.Length}"

        let! state = getUserState
        let platformWordTy = state.Platform.PlatformWordType

        let dimConst1SSA = ssas.[0]
        let str1LenSSA = ssas.[1]
        let dimConst2SSA = ssas.[2]
        let str2LenSSA = ssas.[3]
        let combinedLenSSA = ssas.[4]
        let resultBufferSSA = ssas.[5]
        let str1PtrSSA = ssas.[6]
        let str2PtrSSA = ssas.[7]
        let resultPtrSSA = ssas.[8]
        let str1PtrWord = ssas.[9]
        let str2PtrWord = ssas.[10]
        let resultPtrWord = ssas.[11]
        let len1Word = ssas.[12]
        let len2Word = ssas.[13]
        let memcpy1ResultSSA = ssas.[14]
        let offsetPtrSSA = ssas.[15]
        let offsetPtrWord = ssas.[16]
        let memcpy2ResultSSA = ssas.[17]

        // 1. Get str1 length via memref.dim (strings ARE memrefs) → index
        let! dimConst1Op = pConstI dimConst1SSA 0L TIndex
        let! dim1Op = pMemRefDim str1LenSSA str1SSA dimConst1SSA str1Type

        // 2. Get str2 length via memref.dim → index
        let! dimConst2Op = pConstI dimConst2SSA 0L TIndex
        let! dim2Op = pMemRefDim str2LenSSA str2SSA dimConst2SSA str2Type

        // 3. Compute combined length: len1 + len2 (index arithmetic via arith.addi, NO i64!)
        let addLenOp = ArithOp (AddI (combinedLenSSA, str1LenSSA, str2LenSSA, TIndex))

        // 4. Allocate result buffer with runtime size (index type, NO conversion!)
        let resultTy = TMemRef (TInt (IntWidth 8))
        let! allocOp = pAlloc resultBufferSSA combinedLenSSA (TInt (IntWidth 8))

        // 5. Extract pointers from memrefs for memcpy (FFI boundary)
        let! extractStr1Ptr = pExtractBasePtr str1PtrSSA str1SSA str1Type
        let! extractStr2Ptr = pExtractBasePtr str2PtrSSA str2SSA str2Type
        let! extractResultPtr = pExtractBasePtr resultPtrSSA resultBufferSSA resultTy

        // 6. Cast pointers to platform words for memcpy
        let! cast1 = pIndexCastS str1PtrWord str1PtrSSA TIndex platformWordTy
        let! cast2 = pIndexCastS str2PtrWord str2PtrSSA TIndex platformWordTy
        let! cast3 = pIndexCastS resultPtrWord resultPtrSSA TIndex platformWordTy

        // 7. Cast lengths to platform words for memcpy (index → platform word)
        let! castLen1 = pIndexCastS len1Word str1LenSSA TIndex platformWordTy
        let! castLen2 = pIndexCastS len2Word str2LenSSA TIndex platformWordTy

        // 8. memcpy(result, str1.ptr, len1)
        let! copy1Ops = pStringMemCopy memcpy1ResultSSA resultPtrWord str1PtrWord len1Word

        // 9. Compute offset pointer: result + len1 (index arithmetic via arith.addi)
        let addOffset = ArithOp (AddI (offsetPtrSSA, resultPtrSSA, str1LenSSA, TIndex))
        let! castOffset = pIndexCastS offsetPtrWord offsetPtrSSA TIndex platformWordTy

        // 10. memcpy(result + len1, str2.ptr, len2)
        let! copy2Ops = pStringMemCopy memcpy2ResultSSA offsetPtrWord str2PtrWord len2Word

        // 11. Return result buffer as memref (NO fat pointer struct construction)
        // In MLIR: string IS memref<?xi8> directly

        // Collect all operations (pure index arithmetic - NO i64 conversions!)
        let ops =
            [dimConst1Op; dim1Op] @
            [dimConst2Op; dim2Op] @
            [addLenOp; allocOp] @
            [extractStr1Ptr; extractStr2Ptr; extractResultPtr] @
            [cast1; cast2; cast3; castLen1; castLen2] @
            copy1Ops @
            [addOffset; castOffset] @
            copy2Ops

        return (ops, TRValue { SSA = resultBufferSSA; Type = resultTy })
    }

// ═══════════════════════════════════════════════════════════
// COMPOSED INTRINSIC PARSERS (per-operation, self-contained)
// ═══════════════════════════════════════════════════════════

/// String.length intrinsic — memref.dim to get string dimension
let pStringLengthIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.String
        do! ensure (info.Operation = "length") "Not String.length"
        do! ensure (argIds.Length >= 1) "String.length: Expected 1 arg"
        let! node = getCurrentNode
        let! (_, stringSSA, stringType) = pRecallArgWithLoad argIds.[0]
        return! pStringLength node.Id stringSSA stringType
    }

/// String.charAt intrinsic — memref.load at index offset
let pStringCharAtIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.String
        do! ensure (info.Operation = "charAt") "Not String.charAt"
        do! ensure (argIds.Length >= 2) "String.charAt: Expected 2 args"
        let! node = getCurrentNode
        let! (_, stringSSA, _) = pRecallArgWithLoad argIds.[0]
        let! (_, indexSSA, indexType) = pRecallArgWithLoad argIds.[1]
        return! pStringCharAt node.Id stringSSA indexSSA indexType
    }

/// String.concat2 intrinsic — pure memref operations with index arithmetic
let pStringConcat2Intrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.String
        do! ensure (info.Operation = "concat2") "Not String.concat2"
        do! ensure (argIds.Length >= 2) "String.concat2: Expected 2 args"
        let! node = getCurrentNode
        let! (_, str1SSA, str1Type) = pRecallArgWithLoad argIds.[0]
        let! (_, str2SSA, str2Type) = pRecallArgWithLoad argIds.[1]
        return! pStringConcat2 node.Id str1SSA str2SSA str1Type str2Type
    }

/// String.contains intrinsic — scf.while byte scan loop
let pStringContainsIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.String
        do! ensure (info.Operation = "contains") "Not String.contains"
        do! ensure (argIds.Length >= 2) "String.contains: Expected 2 args"
        let! node = getCurrentNode
        let! (_, stringSSA, stringType) = pRecallArgWithLoad argIds.[0]
        let! (_, charSSA, charType) = pRecallArgWithLoad argIds.[1]
        return! pStringContains node.Id stringSSA charSSA charType stringType
    }

/// NativeStr.fromPointer intrinsic — substring via allocate + memcpy
/// Uses pDetectMemRefAddFusion on buffer arg for NativePtr.add fusion
/// Uses pRecallArgWithLoad on length arg for TMemRef auto-load
let pNativeStrFromPointerIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.NativeStr
        do! ensure (info.Operation = "fromPointer") "Not NativeStr.fromPointer"
        do! ensure (argIds.Length >= 2) "NativeStr.fromPointer: Expected 2 args"
        let! node = getCurrentNode

        // Detect MemRef.add fusion on buffer argument
        let! (bufferSSA, fusedOffsetOpt, bufferType, fusionOps) = pDetectMemRefAddFusion argIds.[0]

        // Recall length argument with auto-load
        let! (lengthLoadOps, lengthSSA, _) = pRecallArgWithLoad argIds.[1]

        // Call pattern with resolved buffer, length, and optional source offset
        let! (ops, result) = pStringFromPointerWithLength node.Id bufferSSA lengthSSA bufferType fusedOffsetOpt
        return (fusionOps @ lengthLoadOps @ ops, result)
    }

/// String.fromBytes intrinsic — convert byte[] to string
/// Both byte[] and string are memref<?xi8>, so this is a zero-cost identity.
/// The witness simply returns the input memref as-is.
let pStringFromBytesIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.String
        do! ensure (info.Operation = "fromBytes") "Not String.fromBytes"
        do! ensure (argIds.Length >= 1) "String.fromBytes: Expected 1 arg"
        let! (_, bytesSSA, bytesType) = pRecallArgWithLoad argIds.[0]
        // Identity: byte[] and string are both memref<?xi8>
        return ([], TRValue { SSA = bytesSSA; Type = bytesType })
    }
