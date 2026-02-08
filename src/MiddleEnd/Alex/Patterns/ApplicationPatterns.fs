/// ApplicationPatterns - Function application and invocation patterns
///
/// PUBLIC: Witnesses use these to emit function calls (direct and indirect).
/// Application patterns handle calling conventions and argument passing.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): All patterns use NodeId-based API.
/// Patterns extract SSAs monadically via getNodeSSAs - witnesses pass NodeIds, not SSAs.
module Alex.Patterns.ApplicationPatterns

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes  // NodeId
open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.FuncElements  // pFuncCall, pFuncCallIndirect
open Alex.Elements.ArithElements // Arithmetic elements for wrapper patterns
open Alex.Elements.IndexElements // pIndexConst
open Alex.CodeGeneration.TypeMapping // mapNativeTypeForArch

// ═══════════════════════════════════════════════════════════
// ARGUMENT LOADING PATTERN (Private Combinator)
// ═══════════════════════════════════════════════════════════

/// Recall argument from accumulator, automatically loading from TMemRef if needed
/// COMPOSING PATTERN: Composes Elements (pRecallNode + pIndexConst + pLoad)
/// Uses VarRef node's allocated SSAs for load operations
let private pRecallArgWithLoad (argId: NodeId) : PSGParser<MLIROp list * SSA * MLIRType> =
    parser {
        // Recall argument from accumulator (VarRef already witnessed in post-order)
        let! (ssa, ty) = pRecallNode argId

        match ty with
        | TMemRef elemType ->
            // Argument is TMemRef (mutable variable) - emit load using VarRef's SSAs
            // VarRef node has 2 SSAs allocated: [0] = index constant, [1] = load result
            let! ssas = getNodeSSAs argId
            do! ensure (ssas.Length >= 2) $"pRecallArgWithLoad: VarRef node {NodeId.value argId} expected 2 SSAs, got {ssas.Length}"
            let zeroSSA = ssas.[0]
            let valueSSA = ssas.[1]

            // Compose Elements: index constant + load
            // Mutable scalars stored as memref<1xT> (rank-1 single element)
            let! zeroOp = pIndexConst zeroSSA 0L

            // CRITICAL: Pass elemType (unwrapped), NOT TMemRefStatic (1, elemType)
            // Serialization code (Serialize.fs:173) reconstructs the memref wrapper
            // based on indices.Length. Passing already-wrapped type causes double-wrapping.
            let loadOp = MLIROp.MemRefOp (MemRefOp.Load (valueSSA, ssa, [zeroSSA], elemType))

            return ([zeroOp; loadOp], valueSSA, elemType)
        | _ ->
            // Not TMemRef - return as-is (no load needed)
            return ([], ssa, ty)
    }

// ═══════════════════════════════════════════════════════════
// APPLICATION PATTERNS (Function Calls)
// ═══════════════════════════════════════════════════════════

/// Build function application (indirect call via function pointer)
/// For known function names, use pDirectCall instead (future optimization)
/// SSA extracted from coeffects via nodeId: [0] = result
let pApplicationCall (nodeId: NodeId) (funcSSA: SSA) (args: (SSA * MLIRType) list) (retType: MLIRType)
                     : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pApplicationCall: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Emit indirect call via function pointer
        let argVals = args |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
        let! callOp = pFuncCallIndirect (Some resultSSA) funcSSA argVals retType
        return ([callOp], TRValue { SSA = resultSSA; Type = retType })
    }

/// Build direct function call (for known function names - portable)
/// Uses func.call (portable) instead of llvm.call (backend-specific)
/// SSA extracted from coeffects via nodeId: [0] = result, [1..N] = potential type compatibility casts
let pDirectCall (nodeId: NodeId) (funcName: string) (args: (SSA * MLIRType) list) (retType: MLIRType)
                : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        let expectedSSAs = 1 + args.Length  // 1 result + N potential casts
        do! ensure (ssas.Length >= expectedSSAs) $"pDirectCall: Expected {expectedSSAs} SSAs, got {ssas.Length}"
        let resultSSA = ssas.[0]
        let castSSAs = ssas |> List.skip 1  // SSAs for potential casts

        // Argument preparation: pass arguments directly to function call
        // NOTE: Prior versions had unconditional static→dynamic memref casts here.
        // After Bug 1 fix (Feb 2026), string literals already return TMemRef (dynamic)
        // in pBuildStringLiteral, so the cast is no longer needed at call boundaries.
        // DU/record/tuple args are TMemRefStatic and should stay static — their
        // function parameters expect static types.
        let processCasts (remainingArgs: (SSA * MLIRType) list) (_castSSAs: SSA list) =
            let vals = remainingArgs |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
            ([], vals)

        let (castOps, finalVals) = processCasts args castSSAs

        let! callOp = pFuncCall (Some resultSSA) funcName finalVals retType
        return (castOps @ [callOp], TRValue { SSA = resultSSA; Type = retType })
    }

// ═══════════════════════════════════════════════════════════
// ARITHMETIC WRAPPER PATTERNS
// ═══════════════════════════════════════════════════════════
// These patterns wrap arithmetic Elements to maintain Element/Pattern/Witness firewall.
// Witnesses call patterns (not Elements directly), patterns extract SSAs monadically.

/// Generic binary arithmetic pattern wrapper (PULL model)
/// Takes operation name and nodeId, pulls arguments from accumulator monadically.
/// Pattern extracts what it needs via state - witnesses don't push parameters.
/// SSA extracted from coeffects via nodeId: [0] = result
/// AUTOMATICALLY loads from TMemRef arguments using pRecallArgWithLoad
let pBinaryArithOp (nodeId: NodeId) (operation: string)
                   : PSGParser<MLIROp list * TransferResult> =
    parser {
        // PULL model: Extract argument IDs from parent Application node
        let! argIds = pGetApplicationArgs
        do! ensure (argIds.Length >= 2) $"pBinaryArithOp: Expected 2 args, got {argIds.Length}"

        // PULL model: Recall operand SSAs and types, automatically loading from TMemRef
        let! (lhsLoadOps, lhsSSA, lhsType) = pRecallArgWithLoad argIds.[0]
        let! (rhsLoadOps, rhsSSA, rhsType) = pRecallArgWithLoad argIds.[1]

        // Extract result SSA from coeffects
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pBinaryArithOp: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Dispatch to correct Element based on operation name
        let! op =
            match operation with
            // Integer arithmetic
            | "addi" -> pAddI resultSSA lhsSSA rhsSSA
            | "subi" -> pSubI resultSSA lhsSSA rhsSSA
            | "muli" -> pMulI resultSSA lhsSSA rhsSSA
            | "divsi" -> pDivSI resultSSA lhsSSA rhsSSA
            | "divui" -> pDivUI resultSSA lhsSSA rhsSSA
            | "remsi" -> pRemSI resultSSA lhsSSA rhsSSA
            | "remui" -> pRemUI resultSSA lhsSSA rhsSSA
            // Float arithmetic
            | "addf" -> pAddF resultSSA lhsSSA rhsSSA
            | "subf" -> pSubF resultSSA lhsSSA rhsSSA
            | "mulf" -> pMulF resultSSA lhsSSA rhsSSA
            | "divf" -> pDivF resultSSA lhsSSA rhsSSA
            // Bitwise
            | "andi" -> pAndI resultSSA lhsSSA rhsSSA
            | "ori" -> pOrI resultSSA lhsSSA rhsSSA
            | "xori" -> pXorI resultSSA lhsSSA rhsSSA
            | "shli" -> pShLI resultSSA lhsSSA rhsSSA
            | "shrui" -> pShRUI resultSSA lhsSSA rhsSSA
            | "shrsi" -> pShRSI resultSSA lhsSSA rhsSSA
            | _ -> fail (Message $"Unknown binary arithmetic operation: {operation}")

        // Infer result type from operand types
        let resultType = lhsType  // Binary ops preserve operand type

        return (lhsLoadOps @ rhsLoadOps @ [op], TRValue { SSA = resultSSA; Type = resultType })
    }

/// Generic comparison pattern wrapper (PULL model)
/// Takes predicate and nodeId, pulls arguments from accumulator monadically.
/// SSA extracted from coeffects via nodeId: [0] = result
/// AUTOMATICALLY loads from TMemRef arguments using pRecallArgWithLoad
let pComparisonOp (nodeId: NodeId) (pred: ICmpPred)
                  : PSGParser<MLIROp list * TransferResult> =
    parser {
        // PULL model: Extract argument IDs from parent Application node
        let! argIds = pGetApplicationArgs
        do! ensure (argIds.Length >= 2) $"pComparisonOp: Expected 2 args, got {argIds.Length}"

        // PULL model: Recall operand SSAs and types, automatically loading from TMemRef
        let! (lhsLoadOps, lhsSSA, lhsType) = pRecallArgWithLoad argIds.[0]
        let! (rhsLoadOps, rhsSSA, rhsType) = pRecallArgWithLoad argIds.[1]

        // Extract result SSA from coeffects
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pComparisonOp: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        let! op = pCmpI resultSSA pred lhsSSA rhsSSA lhsType

        return (lhsLoadOps @ rhsLoadOps @ [op], TRValue { SSA = resultSSA; Type = TInt I1 })  // Comparisons always return i1
    }

/// Unary NOT pattern (xori with constant 1) - PULL model
/// Pulls operand from accumulator monadically.
/// SSA extracted from coeffects via nodeId: [0] = constant, [1] = result
/// AUTOMATICALLY loads from TMemRef arguments using pRecallArgWithLoad
let pUnaryNot (nodeId: NodeId)
              : PSGParser<MLIROp list * TransferResult> =
    parser {
        // PULL model: Extract argument IDs from parent Application node
        let! argIds = pGetApplicationArgs
        do! ensure (argIds.Length >= 1) $"pUnaryNot: Expected 1 arg, got {argIds.Length}"

        // PULL model: Recall operand SSA and type, automatically loading from TMemRef
        let! (loadOps, operandSSA, operandType) = pRecallArgWithLoad argIds.[0]

        // Extract SSAs from coeffects
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pUnaryNot: Expected 2 SSAs, got {ssas.Length}"
        let constSSA = ssas.[0]
        let resultSSA = ssas.[1]

        // Emit constant 1 for boolean NOT
        let constOp = MLIROp.ArithOp (ArithOp.ConstI (constSSA, 1L, TInt I1))
        // Emit XOR operation
        let! xorOp = pXorI resultSSA operandSSA constSSA

        return (loadOps @ [constOp; xorOp], TRValue { SSA = resultSSA; Type = operandType })
    }

// ═══════════════════════════════════════════════════════════
// TYPE CONVERSION PATTERN (IntrinsicModule.Convert)
// ═══════════════════════════════════════════════════════════

/// Type conversion for byte(), int(), float(), etc.
/// PULL model: extracts argument and result type from XParsec state.
/// Dispatches to appropriate MLIR conversion element (TruncI, ExtSI, FPToSI, SIToFP).
let pTypeConversion (nodeId: NodeId)
                    : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! argIds = pGetApplicationArgs
        do! ensure (argIds.Length >= 1) $"pTypeConversion: Expected 1 arg, got {argIds.Length}"

        let! (loadOps, srcSSA, srcType) = pRecallArgWithLoad argIds.[0]

        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pTypeConversion: Expected at least 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        let! state = getUserState
        let dstType = mapNativeTypeForArch state.Platform.TargetArch state.Current.Type

        if srcType = dstType then
            return (loadOps, TRValue { SSA = srcSSA; Type = srcType })
        else
            let! convOp =
                match srcType, dstType with
                | TInt srcW, TInt dstW when srcW < dstW ->
                    pExtSI resultSSA srcSSA srcType dstType
                | TInt _, TInt _ ->
                    pTruncI resultSSA srcSSA srcType dstType
                | TFloat _, TInt _ ->
                    pFPToSI resultSSA srcSSA srcType dstType
                | TInt _, TFloat _ ->
                    pSIToFP resultSSA srcSSA srcType dstType
                | TIndex, TInt _ ->
                    pIndexCastS resultSSA srcSSA dstType
                | _ ->
                    fail (Message $"Unsupported type conversion: {srcType} -> {dstType}")
            return (loadOps @ [convOp], TRValue { SSA = resultSSA; Type = dstType })
    }
