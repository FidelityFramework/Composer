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
open Alex.Patterns.MemRefPatterns // pLoadMutableVariable

// ═══════════════════════════════════════════════════════════
// MUTABLE VARIABLE HANDLING (Monadic Type Coercion)
// ═══════════════════════════════════════════════════════════

/// Monadic helper: Load from memref if TMemRef type, otherwise return as-is
/// This enables patterns to accept both value SSAs and memref SSAs transparently.
/// Returns (actualSSA, actualType, loadOps) where loadOps are [] if no load needed.
let private pEnsureValue (nodeId: NodeId) (ssa: SSA) (ty: MLIRType)
                         : PSGParser<SSA * MLIRType * MLIROp list> =
    parser {
        match ty with
        | TMemRef elemType ->
            // Compose with load pattern to get the value
            let! (ops, result) = pLoadMutableVariable (NodeId.value nodeId) ssa elemType
            match result with
            | TRValue v -> return (v.SSA, v.Type, ops)
            | TRVoid -> return! fail (Message "pLoadMutableVariable returned TRVoid")
            | TRError diag -> return! fail (Message diag.Message)
            | TRSkip -> return! fail (Message "pLoadMutableVariable returned TRSkip")
        | _ ->
            // Not a memref - return as-is
            return (ssa, ty, [])
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

        // Type compatibility casting: static→dynamic memref at function boundaries
        // This is principled: maintaining flexibility where generality is required
        let rec processCasts (remainingArgs: (SSA * MLIRType) list) (castSSAs: SSA list) (accOps: MLIROp list) (accVals: Val list) =
            match remainingArgs, castSSAs with
            | [], _ -> (List.rev accOps, List.rev accVals)
            | (argSSA, argTy) :: restArgs, castSSA :: restCastSSAs ->
                // Check if cast needed: static memref → dynamic memref
                match argTy with
                | TMemRefStatic (_, elemTy) ->
                    // Emit memref.cast: static → dynamic (type-safe, maintains flexibility)
                    let targetTy = TMemRef elemTy
                    let castOp = MLIROp.MemRefOp (MemRefOp.Cast (castSSA, argSSA, argTy, targetTy))
                    let castVal = { SSA = castSSA; Type = targetTy }
                    processCasts restArgs restCastSSAs (castOp :: accOps) (castVal :: accVals)
                | _ ->
                    // No cast needed - pass argument as-is
                    let argVal = { SSA = argSSA; Type = argTy }
                    processCasts restArgs restCastSSAs accOps (argVal :: accVals)
            | _ :: _, [] ->
                // Should not happen - SSAAssignment allocated enough SSAs
                (List.rev accOps, List.rev (args |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })))

        let (castOps, finalVals) = processCasts args castSSAs [] []

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
/// ACCEPTS BOTH VALUE SSAs AND MEMREF SSAs - loads from memref monadically if needed
let pBinaryArithOp (nodeId: NodeId) (operation: string)
                   : PSGParser<MLIROp list * TransferResult> =
    parser {
        // PULL model: Extract argument IDs from parent Application node
        let! argIds = pGetApplicationArgs
        do! ensure (argIds.Length >= 2) $"pBinaryArithOp: Expected 2 args, got {argIds.Length}"
        
        // PULL model: Recall operand SSAs and types from accumulator (post-order ensures they're witnessed)
        let! (lhsSSA, lhsType) = pRecallNode argIds.[0]
        let! (rhsSSA, rhsType) = pRecallNode argIds.[1]
        
        // Extract result SSA from coeffects
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pBinaryArithOp: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Monadic load-if-needed: ensures we have values, not memrefs
        let! (actualLhsSSA, actualLhsType, lhsLoadOps) = pEnsureValue nodeId lhsSSA lhsType
        let! (actualRhsSSA, actualRhsType, rhsLoadOps) = pEnsureValue nodeId rhsSSA rhsType

        // Dispatch to correct Element based on operation name
        let! op =
            match operation with
            // Integer arithmetic
            | "addi" -> pAddI resultSSA actualLhsSSA actualRhsSSA
            | "subi" -> pSubI resultSSA actualLhsSSA actualRhsSSA
            | "muli" -> pMulI resultSSA actualLhsSSA actualRhsSSA
            | "divsi" -> pDivSI resultSSA actualLhsSSA actualRhsSSA
            | "divui" -> pDivUI resultSSA actualLhsSSA actualRhsSSA
            | "remsi" -> pRemSI resultSSA actualLhsSSA actualRhsSSA
            | "remui" -> pRemUI resultSSA actualLhsSSA actualRhsSSA
            // Float arithmetic
            | "addf" -> pAddF resultSSA actualLhsSSA actualRhsSSA
            | "subf" -> pSubF resultSSA actualLhsSSA actualRhsSSA
            | "mulf" -> pMulF resultSSA actualLhsSSA actualRhsSSA
            | "divf" -> pDivF resultSSA actualLhsSSA actualRhsSSA
            // Bitwise
            | "andi" -> pAndI resultSSA actualLhsSSA actualRhsSSA
            | "ori" -> pOrI resultSSA actualLhsSSA actualRhsSSA
            | "xori" -> pXorI resultSSA actualLhsSSA actualRhsSSA
            | "shli" -> pShLI resultSSA actualLhsSSA actualRhsSSA
            | "shrui" -> pShRUI resultSSA actualLhsSSA actualRhsSSA
            | "shrsi" -> pShRSI resultSSA actualLhsSSA actualRhsSSA
            | _ -> fail (Message $"Unknown binary arithmetic operation: {operation}")

        // Infer result type from actual operand types (after load)
        let resultType = actualLhsType  // Binary ops preserve operand type
        
        // Collect all ops: load ops + arithmetic op
        let allOps = lhsLoadOps @ rhsLoadOps @ [op]
        return (allOps, TRValue { SSA = resultSSA; Type = resultType })
    }

/// Generic comparison pattern wrapper (PULL model)
/// Takes predicate and nodeId, pulls arguments from accumulator monadically.
/// SSA extracted from coeffects via nodeId: [0] = result
/// ACCEPTS BOTH VALUE SSAs AND MEMREF SSAs - loads from memref monadically if needed
let pComparisonOp (nodeId: NodeId) (pred: ICmpPred)
                  : PSGParser<MLIROp list * TransferResult> =
    parser {
        // PULL model: Extract argument IDs from parent Application node
        let! argIds = pGetApplicationArgs
        do! ensure (argIds.Length >= 2) $"pComparisonOp: Expected 2 args, got {argIds.Length}"
        
        // PULL model: Recall operand SSAs and types from accumulator
        let! (lhsSSA, lhsType) = pRecallNode argIds.[0]
        let! (rhsSSA, rhsType) = pRecallNode argIds.[1]
        
        // Extract result SSA from coeffects
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pComparisonOp: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        // Monadic load-if-needed: ensures we have values, not memrefs
        let! (actualLhsSSA, actualLhsType, lhsLoadOps) = pEnsureValue nodeId lhsSSA lhsType
        let! (actualRhsSSA, actualRhsType, rhsLoadOps) = pEnsureValue nodeId rhsSSA rhsType

        let! op = pCmpI resultSSA pred actualLhsSSA actualRhsSSA actualLhsType

        // Collect all ops: load ops + comparison op
        let allOps = lhsLoadOps @ rhsLoadOps @ [op]
        return (allOps, TRValue { SSA = resultSSA; Type = TInt I1 })  // Comparisons always return i1
    }

/// Unary NOT pattern (xori with constant 1) - PULL model
/// Pulls operand from accumulator monadically.
/// SSA extracted from coeffects via nodeId: [0] = constant, [1] = result
/// ACCEPTS BOTH VALUE SSAs AND MEMREF SSAs - loads from memref monadically if needed
let pUnaryNot (nodeId: NodeId)
              : PSGParser<MLIROp list * TransferResult> =
    parser {
        // PULL model: Extract argument IDs from parent Application node
        let! argIds = pGetApplicationArgs
        do! ensure (argIds.Length >= 1) $"pUnaryNot: Expected 1 arg, got {argIds.Length}"
        
        // PULL model: Recall operand SSA and type from accumulator
        let! (operandSSA, operandType) = pRecallNode argIds.[0]
        
        // Extract SSAs from coeffects
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pUnaryNot: Expected 2 SSAs, got {ssas.Length}"
        let constSSA = ssas.[0]
        let resultSSA = ssas.[1]

        // Monadic load-if-needed: ensures we have value, not memref
        let! (actualOperandSSA, actualOperandType, loadOps) = pEnsureValue nodeId operandSSA operandType

        // Emit constant 1 for boolean NOT
        let constOp = MLIROp.ArithOp (ArithOp.ConstI (constSSA, 1L, TInt I1))
        // Emit XOR operation
        let! xorOp = pXorI resultSSA actualOperandSSA constSSA

        // Collect all ops: load ops + const + xor
        let allOps = loadOps @ [constOp; xorOp]
        return (allOps, TRValue { SSA = resultSSA; Type = actualOperandType })
    }
