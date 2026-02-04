/// StringPatterns - String operation elision patterns
///
/// Provides composable patterns for string operations using XParsec.
/// Strings are represented as fat pointers: {ptr: index, length: int}
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
open Alex.CodeGeneration.TypeMapping
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════
// MEMORY COPY PATTERN
// ═══════════════════════════════════════════════════════════

/// Call external memcpy(dest, src, len) -> void*
/// External memcpy declaration will be added at module level if needed
/// resultSSA: The SSA assigned to the memcpy result (from coeffects analysis)
let pMemCopy (resultSSA: SSA) (destSSA: SSA) (srcSSA: SSA) (lenSSA: SSA) : PSGParser<MLIROp list> =
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

        return [call]
    }

// ═══════════════════════════════════════════════════════════
// STRING CONCATENATION PATTERN
// ═══════════════════════════════════════════════════════════

/// String.concat2: concatenate two strings
/// SSA layout (26 total):
///   [0-1] = str1 ptr extract (offset, result)
///   [2-3] = str1 len extract (offset, result)
///   [4-5] = str2 ptr extract (offset, result)
///   [6-7] = str2 len extract (offset, result)
///   [8] = combined length
///   [9] = result buffer
///   [10] = result ptr
///   [11] = str1 ptr word cast
///   [12] = str2 ptr word cast
///   [13] = result ptr word cast
///   [14] = len1 word cast
///   [15] = len2 word cast
///   [16] = memcpy1 result
///   [17] = offset ptr
///   [18] = offset ptr word cast
///   [19] = memcpy2 result
///   [20] = undef
///   [21-22] = insert ptr (offset, result)
///   [23-24] = insert len (offset, result)
///   [25] = resultSSA (final)
let pStringConcat2 (ssas: SSA list) (str1SSA: SSA) (str2SSA: SSA) (str1Type: MLIRType) (str2Type: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        do! ensure (ssas.Length >= 26) $"pStringConcat2: Expected 26 SSAs, got {ssas.Length}"

        let! state = getUserState
        let arch = state.Platform.TargetArch
        let platformWordTy = state.Platform.PlatformWordType
        let intTy = mapNativeTypeForArch arch Types.intType

        let str1PtrOffsetSSA = ssas.[0]
        let str1PtrSSA = ssas.[1]
        let str1LenOffsetSSA = ssas.[2]
        let str1LenSSA = ssas.[3]
        let str2PtrOffsetSSA = ssas.[4]
        let str2PtrSSA = ssas.[5]
        let str2LenOffsetSSA = ssas.[6]
        let str2LenSSA = ssas.[7]
        let combinedLenSSA = ssas.[8]
        let resultBufferSSA = ssas.[9]
        let resultPtrSSA = ssas.[10]
        let str1PtrWord = ssas.[11]
        let str2PtrWord = ssas.[12]
        let resultPtrWord = ssas.[13]
        let len1Word = ssas.[14]
        let len2Word = ssas.[15]
        let memcpy1ResultSSA = ssas.[16]
        let offsetPtrSSA = ssas.[17]
        let offsetPtrWord = ssas.[18]
        let memcpy2ResultSSA = ssas.[19]
        let undefSSA = ssas.[20]
        let insertPtrOffsetSSA = ssas.[21]
        let insertPtrResultSSA = ssas.[22]
        let insertLenOffsetSSA = ssas.[23]
        let insertLenResultSSA = ssas.[24]
        let resultSSA = ssas.[25]

        // 1. Extract components from str1: {ptr[0], length[1]}
        let! extract1Ptr = pExtractValue str1PtrSSA str1SSA 0 str1PtrOffsetSSA TIndex
        let! extract1Len = pExtractValue str1LenSSA str1SSA 1 str1LenOffsetSSA intTy

        // 2. Extract components from str2: {ptr[0], length[1]}
        let! extract2Ptr = pExtractValue str2PtrSSA str2SSA 0 str2PtrOffsetSSA TIndex
        let! extract2Len = pExtractValue str2LenSSA str2SSA 1 str2LenOffsetSSA intTy

        // 3. Compute combined length: len1 + len2
        let! addLen = pAddI combinedLenSSA str1LenSSA str2LenSSA

        // 4. Allocate result buffer (combined length bytes)
        let resultTy = TMemRef (TInt I8)
        let! allocOp = pAlloca resultBufferSSA (TInt I8) None

        // 5. Extract result buffer pointer for memcpy
        let! extractResult = pExtractBasePtr resultPtrSSA resultBufferSSA resultTy

        // 6. Cast pointers to platform words for memcpy
        let! cast1 = pIndexCastS str1PtrWord str1PtrSSA platformWordTy
        let! cast2 = pIndexCastS str2PtrWord str2PtrSSA platformWordTy
        let! cast3 = pIndexCastS resultPtrWord resultPtrSSA platformWordTy

        // 7. Cast lengths to platform words for memcpy
        let! castLen1 = pIndexCastS len1Word str1LenSSA platformWordTy
        let! castLen2 = pIndexCastS len2Word str2LenSSA platformWordTy

        // 8. memcpy(result, str1.ptr, len1)
        let! copy1Ops = pMemCopy memcpy1ResultSSA resultPtrWord str1PtrWord len1Word

        // 9. Compute offset pointer: result + len1
        let! addOffset = pIndexAdd offsetPtrSSA resultPtrSSA str1LenSSA
        let! castOffset = pIndexCastS offsetPtrWord offsetPtrSSA platformWordTy

        // 10. memcpy(result + len1, str2.ptr, len2)
        let! copy2Ops = pMemCopy memcpy2ResultSSA offsetPtrWord str2PtrWord len2Word

        // 11. Build result string fat pointer {result_ptr, combined_length}
        let totalBytes = mlirTypeSize TIndex + mlirTypeSize intTy
        let stringTy = TMemRefStatic(totalBytes, TInt I8)
        let! undefOp = pUndef undefSSA stringTy
        let! insertPtrOps = pInsertValue insertPtrResultSSA undefSSA resultPtrSSA 0 insertPtrOffsetSSA stringTy
        let! insertLenOps = pInsertValue resultSSA insertPtrResultSSA combinedLenSSA 1 insertLenOffsetSSA stringTy

        // Collect all operations
        let ops =
            extract1Ptr @ extract1Len @
            extract2Ptr @ extract2Len @
            [addLen; allocOp; extractResult] @
            [cast1; cast2; cast3; castLen1; castLen2] @
            copy1Ops @
            [addOffset; castOffset] @
            copy2Ops @
            [undefOp] @ insertPtrOps @ insertLenOps

        return (ops, TRValue { SSA = resultSSA; Type = stringTy })
    }
