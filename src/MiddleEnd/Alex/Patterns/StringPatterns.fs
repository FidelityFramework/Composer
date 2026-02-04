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

/// String.concat2: concatenate two strings using pure memref operations
/// SSA layout (20 total - pure memref, NO fat pointer extraction):
///   [0] = dim const for str1 (dimension 0)
///   [1] = str1 len (memref.dim result, index type)
///   [2] = str1 len cast to int
///   [3] = dim const for str2 (dimension 0)
///   [4] = str2 len (memref.dim result, index type)
///   [5] = str2 len cast to int
///   [6] = combined length (int)
///   [7] = result buffer (memref)
///   [8] = str1 ptr (memref.extract_aligned_pointer_as_index)
///   [9] = str2 ptr (memref.extract_aligned_pointer_as_index)
///   [10] = result ptr (memref.extract_aligned_pointer_as_index)
///   [11] = str1 ptr word cast
///   [12] = str2 ptr word cast
///   [13] = result ptr word cast
///   [14] = str1 len word cast
///   [15] = str2 len word cast
///   [16] = memcpy1 result
///   [17] = offset ptr (result + len1)
///   [18] = offset ptr word cast
///   [19] = memcpy2 result
let pStringConcat2 (ssas: SSA list) (str1SSA: SSA) (str2SSA: SSA) (str1Type: MLIRType) (str2Type: MLIRType) : PSGParser<MLIROp list * TransferResult> =
    parser {
        do! ensure (ssas.Length >= 20) $"pStringConcat2: Expected 20 SSAs, got {ssas.Length}"

        let! state = getUserState
        let arch = state.Platform.TargetArch
        let platformWordTy = state.Platform.PlatformWordType
        let intTy = mapNativeTypeForArch arch Types.intType

        let dimConst1SSA = ssas.[0]
        let str1LenIndexSSA = ssas.[1]
        let str1LenIntSSA = ssas.[2]
        let dimConst2SSA = ssas.[3]
        let str2LenIndexSSA = ssas.[4]
        let str2LenIntSSA = ssas.[5]
        let combinedLenSSA = ssas.[6]
        let resultBufferSSA = ssas.[7]
        let str1PtrSSA = ssas.[8]
        let str2PtrSSA = ssas.[9]
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

        // 1. Get str1 length via memref.dim (strings ARE memrefs)
        let! dimConst1Op = pConstI dimConst1SSA 0L TIndex  // Dimension 0 (strings are 1D)
        let! dim1Op = pMemRefDim str1LenIndexSSA str1SSA dimConst1SSA str1Type
        let! cast1LenOp = pIndexCastS str1LenIntSSA str1LenIndexSSA intTy

        // 2. Get str2 length via memref.dim
        let! dimConst2Op = pConstI dimConst2SSA 0L TIndex
        let! dim2Op = pMemRefDim str2LenIndexSSA str2SSA dimConst2SSA str2Type
        let! cast2LenOp = pIndexCastS str2LenIntSSA str2LenIndexSSA intTy

        // 3. Compute combined length: len1 + len2
        let! addLen = pAddI combinedLenSSA str1LenIntSSA str2LenIntSSA

        // 4. Allocate result buffer (combined length bytes)
        let resultTy = TMemRef (TInt I8)
        let! allocOp = pAlloca resultBufferSSA (TInt I8) None

        // 5. Extract pointers from memrefs for memcpy (FFI boundary)
        let! extractStr1Ptr = pExtractBasePtr str1PtrSSA str1SSA str1Type
        let! extractStr2Ptr = pExtractBasePtr str2PtrSSA str2SSA str2Type
        let! extractResultPtr = pExtractBasePtr resultPtrSSA resultBufferSSA resultTy

        // 6. Cast pointers to platform words for memcpy
        let! cast1 = pIndexCastS str1PtrWord str1PtrSSA platformWordTy
        let! cast2 = pIndexCastS str2PtrWord str2PtrSSA platformWordTy
        let! cast3 = pIndexCastS resultPtrWord resultPtrSSA platformWordTy

        // 7. Cast lengths to platform words for memcpy
        let! castLen1 = pIndexCastS len1Word str1LenIntSSA platformWordTy
        let! castLen2 = pIndexCastS len2Word str2LenIntSSA platformWordTy

        // 8. memcpy(result, str1.ptr, len1)
        let! copy1Ops = pMemCopy memcpy1ResultSSA resultPtrWord str1PtrWord len1Word

        // 9. Compute offset pointer: result + len1
        let! addOffset = pIndexAdd offsetPtrSSA resultPtrSSA str1LenIndexSSA
        let! castOffset = pIndexCastS offsetPtrWord offsetPtrSSA platformWordTy

        // 10. memcpy(result + len1, str2.ptr, len2)
        let! copy2Ops = pMemCopy memcpy2ResultSSA offsetPtrWord str2PtrWord len2Word

        // 11. Return result buffer as memref (NO fat pointer struct construction)
        // In MLIR: string IS memref<?xi8> directly

        // Collect all operations
        let ops =
            [dimConst1Op; dim1Op; cast1LenOp] @
            [dimConst2Op; dim2Op; cast2LenOp] @
            [addLen; allocOp] @
            [extractStr1Ptr; extractStr2Ptr; extractResultPtr] @
            [cast1; cast2; cast3; castLen1; castLen2] @
            copy1Ops @
            [addOffset; castOffset] @
            copy2Ops

        return (ops, TRValue { SSA = resultBufferSSA; Type = resultTy })
    }
