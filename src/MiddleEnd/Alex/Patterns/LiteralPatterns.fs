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
// LITERAL PATTERNS
// ═══════════════════════════════════════════════════════════

/// Build literal: Match literal from PSG and emit constant MLIR.
/// An integer literal of the bare kind is held at the width its point range selects (the
/// sentinel narrowed at the literal's node, on every substrate); a width-named literal at its
/// carrier's width (interim, CS-12). A literal needs no meet.
let pBuildLiteral (lit: NativeLiteral) (ssa: SSA) (_arch: Architecture) : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! state = getUserState

        match lit with
        | NativeLiteral.Unit ->
            let ty = mapNTUKindToMLIRType NTUKind.NTUunit
            let! op = pConstI ssa 0L ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Bool b ->
            let value = if b then 1L else 0L
            let ty = mapNTUKindToMLIRType NTUKind.NTUbool
            let! op = pConstI ssa value ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Int (n, kind) ->
            let ty = mapNTUKindToMLIRType kind |> narrowForCurrent state
            let! op = pConstI ssa n ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.UInt (n, kind) ->
            let ty = mapNTUKindToMLIRType kind |> narrowForCurrent state
            let! op = pConstI ssa (int64 n) ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Char c ->
            let ty = mapNTUKindToMLIRType NTUKind.NTUchar
            let! op = pConstI ssa (int64 c) ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.Float (f, kind) ->
            let ty = mapNTUKindToMLIRType kind
            let! op = pConstF ssa f ty
            return ([op], TRValue { SSA = ssa; Type = ty })

        | NativeLiteral.String _ ->
            // String literals require witness-level handling with multiple SSAs
            // Use pBuildStringLiteral pattern instead
            return! fail (Message "String literals require pBuildStringLiteral pattern with SSA list")

        | _ ->
            return! fail (Message $"Unsupported literal: {lit}")
    }

/// A unit-typed expression still has a value when its control-flow operations
/// return no SSA. Preserve those operations, then witness the sole unit value.
/// The caller must have observed the settled unit type; this is not a fallback
/// for an unwitnessed operand or an unresolved expression type.
let pWithUnitResult (nodeId: NodeId) (body: PSGParser<MLIROp list * TransferResult>)
                    : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! operations, result = body
        match result with
        | TRVoid ->
            let! ssa = getNodeSSA nodeId
            let! state = getUserState
            let! unitOps, unitResult = pBuildLiteral NativeLiteral.Unit ssa state.Coeffects.Platform.TargetArch
            return operations @ unitOps, unitResult
        | _ -> return! fail (Message "Unit result requires a witnessed void operation")
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

/// Read the settled pool and entry directly; no string encoding or placement occurs here.
let stringPoolView (pool: Clef.Compiler.PSGSaturation.SemanticGraph.Types.StaticStringPool)
                   (entry: Clef.Compiler.PSGSaturation.SemanticGraph.Types.StaticStringEntry)
                   (ssas: SSA list) : MLIROp list * TransferResult =
    if ssas.Length < 4 then invalidArg "ssas" "String pool view needs four SSAs."
    let storageTy = TMemRefStatic (pool.Size, TInt (IntWidth 8))
    let contentTy = TMemRefStatic (entry.Length, TInt (IntWidth 8))
    let dynamicTy = TMemRef (TInt (IntWidth 8))
    [ MLIROp.MemRefOp (MemRefOp.GetGlobal (ssas[0], pool.Symbol, storageTy))
      MLIROp.IndexOp (IndexOp.IndexConst (ssas[3], int64 entry.Offset))
      // View moves the data pointer, preserving an identity layout for memref.cast.
      MLIROp.MemRefOp (MemRefOp.View (ssas[1], ssas[0], ssas[3], storageTy, contentTy))
      MLIROp.MemRefOp (MemRefOp.Cast (ssas[2], ssas[1], contentTy, dynamicTy)) ],
    TRValue { SSA = ssas[2]; Type = dynamicTy }

// DEAD CODE DELETED: pStringGetPtr and pStringGetLength were unused
// Pointer extraction happens inline in PlatformPatterns.pSysWrite
// Length extraction happens inline in PlatformPatterns.pSysWrite via memref.dim
