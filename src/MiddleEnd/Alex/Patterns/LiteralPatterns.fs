/// LiteralPatterns - Literal value emission and string handling
///
/// PUBLIC: Witnesses use these to emit literal constants and string operations.
/// Literal patterns map NativeLiteral values to MLIR constants.
/// String patterns handle string literal globals and pointer/length extraction for FFI.
module Alex.Patterns.LiteralPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.MLIRAtomics  // pConstI, pConstF, pUndef, pInsertValue
open Alex.Elements.MemRefElements  // pMemRefGetGlobal, pExtractBasePtr, pMemRefDim
open Alex.Elements.IndexElements  // pIndexCastS
open Alex.CodeGeneration.TypeMapping
open Clef.Compiler.NativeTypedTree.NativeTypes

// ═══════════════════════════════════════════════════════════
// XPARSEC HELPERS
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// LITERAL PATTERNS
// ═══════════════════════════════════════════════════════════

/// Build literal: Match literal from PSG and emit constant MLIR.
/// On FPGA, platform-word integer literals produce abstract types (IntWidth 0)
/// which are immediately narrowed to concrete widths via interval analysis coeffect.
let pBuildLiteral (lit: NativeLiteral) (ssa: SSA) (arch: Architecture) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState
        let platform = state.Coeffects.TargetPlatform

        match lit with
        | NativeLiteral.Unit ->
            let ty = mapNTUKindToMLIRType platform arch NTUKind.NTUunit
            let! op = pConstI ssa 0L ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Bool b ->
            let value = if b then 1L else 0L
            let ty = mapNTUKindToMLIRType platform arch NTUKind.NTUbool
            let! op = pConstI ssa value ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Int (n, kind) ->
            let ty = mapNTUKindToMLIRType platform arch kind |> narrowForCurrent state
            let! op = pConstI ssa n ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.UInt (n, kind) ->
            let ty = mapNTUKindToMLIRType platform arch kind |> narrowForCurrent state
            let! op = pConstI ssa (int64 n) ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Char c ->
            let ty = mapNTUKindToMLIRType platform arch NTUKind.NTUchar
            let! op = pConstI ssa (int64 c) ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Float (f, kind) ->
            let ty = mapNTUKindToMLIRType platform arch kind
            let! op = pConstF ssa f ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.String _ ->
            // String literals require witness-level handling with multiple SSAs
            // Use pBuildStringLiteral pattern instead
            return! fail (Message "String literals require pBuildStringLiteral pattern with SSA list")

        | _ ->
            return! fail (Message $"Unsupported literal: {lit}")
    }

// ═══════════════════════════════════════════════════════════
// STRING PATTERNS
// ═══════════════════════════════════════════════════════════

/// Derive global reference name from string content (pure, deterministic)
/// Uses FNV-1a hash — .NET String.GetHashCode() is randomized per-process.
let deriveGlobalRef (content: string) : string =
    let bytes = System.Text.Encoding.UTF8.GetBytes(content)
    let mutable hash = 2166136261u  // FNV offset basis
    for b in bytes do
        hash <- hash ^^^ (uint32 b)
        hash <- hash * 16777619u    // FNV prime
    sprintf "str_%u" hash

/// Derive byte length from string content (pure)
let deriveByteLength (content: string) : int =
    System.Text.Encoding.UTF8.GetByteCount(content)

/// Build string literal: get reference to global memref (portable MLIR)
/// TypeLayout.Opaque: Single SSA for memref.get_global
/// SSA layout: [0] = memref.get_global result
/// Returns: ((ops, globalName, content, byteLength), result)
/// NOTE: Witness must emit memref.global to TopLevelOps separately
let pBuildStringLiteral (content: string) (ssas: SSA list) (arch: Architecture)
                         : PSGParser<(MLIROp list * string * string * int) * TransferResult> =
    parser {
        do! emitTrace "pBuildStringLiteral.entry" (sprintf "content='%s', ssas=%A, arch=%A" content ssas arch)

        // Need 3 SSAs: memref.get_global (storage) + memref.reinterpret_cast (content view) + memref.cast (dynamic)
        do! ensure (ssas.Length >= 3) $"pBuildStringLiteral: Expected 3 SSAs, got {ssas.Length}"

        do! emitTrace "pBuildStringLiteral.ssa_validated" (sprintf "SSA count OK: %d" ssas.Length)

        // Use StringCollection pure derivation (coeffect model)
        let globalName = deriveGlobalRef content
        let byteLength = deriveByteLength content
        // Storage includes a null sentinel byte for C interop safety.
        // Clef strings are (ptr, length) — the sentinel is invisible to the type system
        // but ensures .Pointer yields a C-compatible null-terminated pointer.
        let storageLength = byteLength + 1

        do! emitTrace "pBuildStringLiteral.derived" (sprintf "globalName=%s, byteLength=%d, storageLength=%d" globalName byteLength storageLength)

        // Static type from global: memref<Nxi8> where N is storage length (includes null sentinel)
        let storageTy = TMemRefStatic (storageLength, TInt (IntWidth 8))
        // Content view type: memref<Mxi8> where M is byte length (semantic content, no sentinel)
        let contentTy = TMemRefStatic (byteLength, TInt (IntWidth 8))
        // Dynamic type (string): memref<?xi8>
        let dynamicTy = TMemRef (TInt (IntWidth 8))

        do! emitTrace "pBuildStringLiteral.types" (sprintf "storageTy=%A, contentTy=%A, dynamicTy=%A" storageTy contentTy dynamicTy)

        let getGlobalSSA = ssas.[0]    // SSA for memref.get_global (storage dim)
        let reinterpretSSA = ssas.[1]  // SSA for memref.reinterpret_cast (content dim)
        let castSSA = ssas.[2]         // SSA for memref.cast (content → dynamic)

        do! emitTrace "pBuildStringLiteral.ssas_extracted" (sprintf "getGlobal=%A, reinterpret=%A, cast=%A" getGlobalSSA reinterpretSSA castSSA)

        // memref.get_global @globalName : memref<Nxi8> (storage dimension, includes null sentinel)
        let! getGlobalOp = pMemRefGetGlobal getGlobalSSA globalName storageTy

        // memref.reinterpret_cast: memref<Nxi8> → memref<Mxi8> (narrow to content dimension)
        // This separates storage length (for C interop) from semantic length (for I/O operations).
        // Downstream memref.dim will return byteLength, not storageLength.
        let reinterpretOp = MLIROp.MemRefOp (MemRefOp.ReinterpretCast (reinterpretSSA, getGlobalSSA, 0, byteLength, storageTy, contentTy))

        // memref.cast: memref<Mxi8> → memref<?xi8>
        // String literals ARE strings (dynamic memref) — cast at point of creation
        let! castOp = pMemRefCast castSSA reinterpretSSA contentTy dynamicTy

        do! emitTrace "pBuildStringLiteral.elements_complete" "memref.get_global + memref.reinterpret_cast + memref.cast succeeded"

        let inlineOps = [getGlobalOp; reinterpretOp; castOp]
        let result = TRValue { SSA = castSSA; Type = dynamicTy }

        do! emitTrace "pBuildStringLiteral.returning" (sprintf "Returning %d ops" (List.length inlineOps))

        // Return ops + (globalName, content, storageLength) for witness to emit memref.global
        return ((inlineOps, globalName, content, storageLength), result)
    }

// DEAD CODE DELETED: pStringGetPtr and pStringGetLength were unused
// Pointer extraction happens inline in PlatformPatterns.pSysWrite
// Length extraction happens inline in PlatformPatterns.pSysWrite via memref.dim
