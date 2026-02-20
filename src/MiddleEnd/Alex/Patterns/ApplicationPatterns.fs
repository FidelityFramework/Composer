/// ApplicationPatterns - Function application and invocation patterns
///
/// PUBLIC: Witnesses use these to emit function calls (direct and indirect).
/// Application patterns handle calling conventions and argument passing.
///
/// ARCHITECTURAL RESTORATION (Feb 2026): All patterns use NodeId-based API.
/// Patterns extract SSAs monadically via getNodeSSAs - witnesses pass NodeIds, not SSAs.
module Alex.Patterns.ApplicationPatterns

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.FuncElements  // pFuncCall, pFuncCallIndirect
open Alex.Elements.ArithElements // Arithmetic elements for wrapper patterns
open Alex.Elements.CombElements  // FPGA combinational elements (codata-dependent)
open Alex.Elements.IndexElements // pIndexCastS
open Core.Types.Dialects         // TargetPlatform
open Alex.CodeGeneration.TypeMapping
open Alex.Patterns.MemoryPatterns // pRecallArgWithLoad (monadic TMemRef auto-load)

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
                (paramNames: string list option)
                : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        let expectedSSAs = 1 + args.Length  // 1 result + N potential casts
        do! ensure (ssas.Length >= expectedSSAs) $"pDirectCall: Expected {expectedSSAs} SSAs, got {ssas.Length}"
        let resultSSA = ssas.[0]
        let castSSAs = ssas |> List.skip 1  // SSAs for potential casts

        // Argument preparation: pass arguments directly to function call
        let processCasts (remainingArgs: (SSA * MLIRType) list) (_castSSAs: SSA list) =
            let vals = remainingArgs |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })
            ([], vals)

        let (castOps, finalVals) = processCasts args castSSAs

        let! targetPlatform = getTargetPlatform
        match targetPlatform with
        | FPGA ->
            // FPGA: hw.instance (spatial instantiation) instead of func.call (temporal)
            let names = paramNames |> Option.defaultValue (args |> List.mapi (fun i _ -> sprintf "in%d" i))
            let inputs = List.map2 (fun pname (v: Val) -> (pname, v.SSA, v.Type)) names finalVals
            let outputs = [("result", retType)]
            let instName = funcName.Replace(".", "_") + "_inst"
            let! instanceOp = Alex.Elements.HWElements.pHWInstance resultSSA instName funcName inputs outputs
            return (castOps @ [instanceOp], TRValue { SSA = resultSSA; Type = retType })
        | _ ->
            // CPU: func.call (temporal function call)
            let! callOp = pFuncCall (Some resultSSA) funcName finalVals retType
            return (castOps @ [callOp], TRValue { SSA = resultSSA; Type = retType })
    }

// ═══════════════════════════════════════════════════════════
// ARITHMETIC WRAPPER PATTERNS
// ═══════════════════════════════════════════════════════════
// These patterns wrap arithmetic Elements to maintain Element/Pattern/Witness firewall.
// Witnesses call patterns (not Elements directly), patterns extract SSAs monadically.

/// Helper: dispatch FPGA combinational arithmetic operation
let private pFpgaCombOp (operation: string) (resultSSA: SSA) (lhs: SSA) (rhs: SSA) (opTy: MLIRType) =
    match operation with
    | "add" -> pCombAdd resultSSA lhs rhs opTy
    | "sub" -> pCombSub resultSSA lhs rhs opTy
    | "mul" -> pCombMul resultSSA lhs rhs opTy
    | "div" -> pCombDivS resultSSA lhs rhs opTy
    | "rem" -> pCombMod resultSSA lhs rhs opTy
    | "andi" -> pCombAnd resultSSA lhs rhs opTy
    | "ori" -> pCombOr resultSSA lhs rhs opTy
    | "xori" -> pCombXor resultSSA lhs rhs opTy
    | "shli" -> pCombShl resultSSA lhs rhs opTy
    | "shrui" -> pCombShrU resultSSA lhs rhs opTy
    | "shrsi" -> pCombShrS resultSSA lhs rhs opTy
    | "divu" -> pCombDivU resultSSA lhs rhs opTy
    | _ -> fail (Message $"Unsupported FPGA arithmetic operation: {operation}")

/// Generic binary arithmetic pattern wrapper (PULL model)
/// Takes operation name and nodeId, pulls arguments from accumulator monadically.
/// Pattern extracts what it needs via state - witnesses don't push parameters.
/// SSA extracted from coeffects via nodeId: [0+] = extensions (if needed), then result
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

        // Extract SSA pool from coeffects (5 SSAs for Operators intrinsics)
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pBinaryArithOp: Expected 1 SSA, got {ssas.Length}"

        // Select Element based on target platform, semantic operation, and operand type
        // This is the codata-dependent elision point: same Pattern, different Elements
        let! targetPlatform = getTargetPlatform
        let! state = getUserState
        match targetPlatform with
        | FPGA ->
            // FPGA: combinational logic (comb dialect)
            // Determine operation width: max of operand widths and inferred result width
            let lhsBits = match lhsType with TInt (IntWidth b) -> b | _ -> 0
            let rhsBits = match rhsType with TInt (IntWidth b) -> b | _ -> 0
            // Query the Application node's inferred result width via coeffect
            let inferredResultTy = narrowForCurrent state (TInt (IntWidth 0))
            let resBits = match inferredResultTy with TInt (IntWidth b) -> b | _ -> 0
            // comb.* ops require all operands AND result at same width
            let opBits = max (max lhsBits rhsBits) resBits
            let opTy = if opBits > 0 then TInt (IntWidth opBits) else lhsType

            // Emit extension ops for narrower operands (using spare SSAs from pool)
            let needExtLhs = lhsBits > 0 && lhsBits < opBits
            let needExtRhs = rhsBits > 0 && rhsBits < opBits
            // Truncation needed when operation width exceeds inferred result width
            // e.g. counter % maxCounterTicks: operands at i30, result interval → i29
            let needTrunc = resBits > 0 && resBits < opBits

            // Step 1: Emit extension ops, compute effective operand SSAs and comb result SSA
            // SSA layout: [ext0?, ext1?, combResult, trunc?] — all within 5-SSA pool
            let! (extOps, effLhs, effRhs, combResultSSA, nextSSAIdx) =
                match needExtLhs, needExtRhs with
                | true, true ->
                    parser {
                        let! extL = pExtSI ssas.[0] lhsSSA lhsType opTy
                        let! extR = pExtSI ssas.[1] rhsSSA rhsType opTy
                        return ([extL; extR], ssas.[0], ssas.[1], ssas.[2], 3)
                    }
                | true, false ->
                    parser {
                        let! extL = pExtSI ssas.[0] lhsSSA lhsType opTy
                        return ([extL], ssas.[0], rhsSSA, ssas.[1], 2)
                    }
                | false, true ->
                    parser {
                        let! extR = pExtSI ssas.[0] rhsSSA rhsType opTy
                        return ([extR], lhsSSA, ssas.[0], ssas.[1], 2)
                    }
                | false, false ->
                    parser { return ([], lhsSSA, rhsSSA, ssas.[0], 1) }

            // Step 2: Emit the comb operation
            let! op = pFpgaCombOp operation combResultSSA effLhs effRhs opTy

            // Step 3: Truncate if operation width exceeds inferred result width
            if needTrunc then
                let resTy = TInt (IntWidth resBits)
                let truncSSA = ssas.[nextSSAIdx]
                let! truncOp = pTruncI truncSSA combResultSSA opTy resTy
                return (lhsLoadOps @ rhsLoadOps @ extOps @ [op; truncOp], TRValue { SSA = truncSSA; Type = resTy })
            else
                return (lhsLoadOps @ rhsLoadOps @ extOps @ [op], TRValue { SSA = combResultSSA; Type = opTy })

        | _ ->
            // CPU/MCU: standard MLIR arithmetic (arith dialect) — no extension needed
            let resultSSA = ssas.[0]
            let! op =
                match operation, lhsType with
                | "add", TFloat _ -> pAddF resultSSA lhsSSA rhsSSA lhsType
                | "add", _ -> pAddI resultSSA lhsSSA rhsSSA lhsType
                | "sub", TFloat _ -> pSubF resultSSA lhsSSA rhsSSA lhsType
                | "sub", _ -> pSubI resultSSA lhsSSA rhsSSA lhsType
                | "mul", TFloat _ -> pMulF resultSSA lhsSSA rhsSSA lhsType
                | "mul", _ -> pMulI resultSSA lhsSSA rhsSSA lhsType
                | "div", TFloat _ -> pDivF resultSSA lhsSSA rhsSSA lhsType
                | "div", _ -> pDivSI resultSSA lhsSSA rhsSSA lhsType
                | "rem", _ -> pRemSI resultSSA lhsSSA rhsSSA lhsType
                | "andi", _ -> pAndI resultSSA lhsSSA rhsSSA lhsType
                | "ori", _ -> pOrI resultSSA lhsSSA rhsSSA lhsType
                | "xori", _ -> pXorI resultSSA lhsSSA rhsSSA lhsType
                | "shli", _ -> pShLI resultSSA lhsSSA rhsSSA lhsType
                | "shrui", _ -> pShRUI resultSSA lhsSSA rhsSSA lhsType
                | "shrsi", _ -> pShRSI resultSSA lhsSSA rhsSSA lhsType
                | "divu", _ -> pDivUI resultSSA lhsSSA rhsSSA lhsType
                | "remu", _ -> pRemUI resultSSA lhsSSA rhsSSA lhsType
                | _ -> fail (Message $"Unknown binary arithmetic operation: {operation} on {lhsType}")
            return (lhsLoadOps @ rhsLoadOps @ [op], TRValue { SSA = resultSSA; Type = lhsType })
    }

/// Generic comparison pattern wrapper (PULL model)
/// Takes predicate and nodeId, pulls arguments from accumulator monadically.
/// SSA extracted from coeffects via nodeId: [0] = result
/// AUTOMATICALLY loads from TMemRef arguments using pRecallArgWithLoad
let pComparisonOp (nodeId: NodeId) (predName: string)
                  : PSGParser<MLIROp list * TransferResult> =
    parser {
        // PULL model: Extract argument IDs from parent Application node
        let! argIds = pGetApplicationArgs
        do! ensure (argIds.Length >= 2) $"pComparisonOp: Expected 2 args, got {argIds.Length}"

        // PULL model: Recall operand SSAs and types, automatically loading from TMemRef
        let! (lhsLoadOps, lhsSSA, lhsType) = pRecallArgWithLoad argIds.[0]
        let! (rhsLoadOps, rhsSSA, rhsType) = pRecallArgWithLoad argIds.[1]

        // Extract SSA pool from coeffects (5 SSAs for Operators intrinsics)
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pComparisonOp: Expected 1 SSA, got {ssas.Length}"

        // Codata-dependent elision: TargetPlatform determines which Elements to invoke
        let! targetPlatform = getTargetPlatform

        let! state = getUserState
        match targetPlatform with
        | FPGA ->
            // FPGA: combinational comparison — operands must match width
            let lhsBits = match lhsType with TInt (IntWidth b) -> b | _ -> 0
            let rhsBits = match rhsType with TInt (IntWidth b) -> b | _ -> 0
            // Comparison operands must match; result is always i1
            let opBits = max lhsBits rhsBits
            let opTy = if opBits > 0 then TInt (IntWidth opBits) else lhsType

            match opTy with
            | TFloat _ ->
                return! fail (Message $"FPGA does not support float comparison: {predName}")
            | _ ->
                let icmpPred =
                    match predName with
                    | "eq"  -> ICmpPred.Eq
                    | "ne"  -> ICmpPred.Ne
                    | "lt"  -> ICmpPred.Slt
                    | "le"  -> ICmpPred.Sle
                    | "gt"  -> ICmpPred.Sgt
                    | "ge"  -> ICmpPred.Sge
                    | "ult" -> ICmpPred.Ult
                    | "ule" -> ICmpPred.Ule
                    | "ugt" -> ICmpPred.Ugt
                    | "uge" -> ICmpPred.Uge
                    | _ -> failwith $"Unknown FPGA comparison predicate: {predName}"

                // Emit extension ops for narrower operands, then compare
                let needExtLhs = lhsBits > 0 && lhsBits < opBits
                let needExtRhs = rhsBits > 0 && rhsBits < opBits

                let! (extOps, effLhs, effRhs, resultSSA) =
                    match needExtLhs, needExtRhs with
                    | true, true ->
                        parser {
                            let! extL = pExtSI ssas.[0] lhsSSA lhsType opTy
                            let! extR = pExtSI ssas.[1] rhsSSA rhsType opTy
                            return ([extL; extR], ssas.[0], ssas.[1], ssas.[2])
                        }
                    | true, false ->
                        parser {
                            let! extL = pExtSI ssas.[0] lhsSSA lhsType opTy
                            return ([extL], ssas.[0], rhsSSA, ssas.[1])
                        }
                    | false, true ->
                        parser {
                            let! extR = pExtSI ssas.[0] rhsSSA rhsType opTy
                            return ([extR], lhsSSA, ssas.[0], ssas.[1])
                        }
                    | false, false ->
                        parser { return ([], lhsSSA, rhsSSA, ssas.[0]) }

                let! op = pCombICmp resultSSA icmpPred effLhs effRhs opTy
                return (lhsLoadOps @ rhsLoadOps @ extOps @ [op], TRValue { SSA = resultSSA; Type = TInt (IntWidth 1) })

        | _ ->
            // CPU: type-dependent dispatch (int vs float)
            let resultSSA = ssas.[0]
            let! op =
                match lhsType with
                | TFloat _ ->
                    let fcmpPred =
                        match predName with
                        | "eq" -> FCmpPred.OEq
                        | "ne" -> FCmpPred.ONe
                        | "lt" -> FCmpPred.OLt
                        | "le" -> FCmpPred.OLe
                        | "gt" -> FCmpPred.OGt
                        | "ge" -> FCmpPred.OGe
                        | _ -> failwith $"Unknown float comparison predicate: {predName}"
                    pCmpF resultSSA fcmpPred lhsSSA rhsSSA lhsType
                | _ ->
                    let icmpPred =
                        match predName with
                        | "eq"  -> ICmpPred.Eq
                        | "ne"  -> ICmpPred.Ne
                        | "lt"  -> ICmpPred.Slt
                        | "le"  -> ICmpPred.Sle
                        | "gt"  -> ICmpPred.Sgt
                        | "ge"  -> ICmpPred.Sge
                        | "ult" -> ICmpPred.Ult
                        | "ule" -> ICmpPred.Ule
                        | "ugt" -> ICmpPred.Ugt
                        | "uge" -> ICmpPred.Uge
                        | _ -> failwith $"Unknown int comparison predicate: {predName}"
                    pCmpI resultSSA icmpPred lhsSSA rhsSSA lhsType
            return (lhsLoadOps @ rhsLoadOps @ [op], TRValue { SSA = resultSSA; Type = TInt (IntWidth 1) })
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
        let constOp = MLIROp.ArithOp (ArithOp.ConstI (constSSA, 1L, TInt (IntWidth 1)))
        // Emit XOR operation
        let! xorOp = pXorI resultSSA operandSSA constSSA operandType

        return (loadOps @ [constOp; xorOp], TRValue { SSA = resultSSA; Type = operandType })
    }

/// Bitwise complement pattern (~~~, op_LogicalNot) — xori with all-ones
/// Emits: constSSA = constant -1 : operandType; result = xori operand, constSSA
/// SSA extracted from coeffects via nodeId: [0] = constant, [1] = result
let pBitwiseNot (nodeId: NodeId)
                : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! argIds = pGetApplicationArgs
        do! ensure (argIds.Length >= 1) $"pBitwiseNot: Expected 1 arg, got {argIds.Length}"

        let! (loadOps, operandSSA, operandType) = pRecallArgWithLoad argIds.[0]

        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 2) $"pBitwiseNot: Expected 2 SSAs, got {ssas.Length}"
        let constSSA = ssas.[0]
        let resultSSA = ssas.[1]

        // -1 in two's complement is all-ones bits — the bitwise NOT mask
        let constOp = MLIROp.ArithOp (ArithOp.ConstI (constSSA, -1L, operandType))
        let! xorOp = pXorI resultSSA operandSSA constSSA operandType

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
        let dstType = mapNativeTypeWithGraphForArch state.Platform.TargetArch state.Graph state.Current.Type

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
                    pIndexCastS resultSSA srcSSA TIndex dstType
                | _ ->
                    fail (Message $"Unsupported type conversion: {srcType} -> {dstType}")
            return (loadOps @ [convOp], TRValue { SSA = resultSSA; Type = dstType })
    }

// ═══════════════════════════════════════════════════════════
// COMPOSED INTRINSIC PARSERS (per-operation, self-contained)
// ═══════════════════════════════════════════════════════════

/// Parser that succeeds only when the current node's Clef type is unsigned (NTUuint / NTUsize).
/// Fails otherwise, enabling <|> composition with the signed variant.
/// This is the XParsec monadic encoding of DTS signedness: the type IS the selector.
let private pRequireUnsigned : PSGParser<unit> =
    parser {
        let! state = getUserState
        match Types.tryGetNTUKind state.Current.Type with
        | Some (NTUKind.NTUuint _) | Some NTUKind.NTUsize -> return ()
        | _ -> return! fail (Message "Not unsigned type")
    }

/// Unsigned-guarded binary arith: succeeds only for NTUuint, emitting the unsigned op.
/// Composes with pBinaryArithOp (signed) via <|> in pBinaryArithIntrinsic.
let private pIfUnsignedArith (nodeId: NodeId) (unsignedOp: string)
                              : PSGParser<MLIROp list * TransferResult> =
    parser {
        do! pRequireUnsigned
        return! pBinaryArithOp nodeId unsignedOp
    }

/// Unsigned-guarded comparison: succeeds only for NTUuint, emitting the unsigned predicate.
let private pIfUnsignedCmp (nodeId: NodeId) (unsignedPred: string)
                            : PSGParser<MLIROp list * TransferResult> =
    parser {
        do! pRequireUnsigned
        return! pComparisonOp nodeId unsignedPred
    }

/// Binary arithmetic intrinsic — folds over classifyAtomicOp via XParsec catamorphism.
///
/// Signedness-sensitive operations use <|> composition:
///   pIfUnsignedArith (succeeds for NTUuint, fails otherwise)
///   <|> pBinaryArithOp (signed/float — tried on failure of unsigned guard)
///
/// The Clef Dimensional Type System flows through to MLIR operation variants:
///   NTUuint → shrui / divu / remu / ult,ule,ugt,uge
///   NTUint  → shrsi / divi / remi / slt,sle,sgt,sge
///   NTUfloat → shrsi n/a, divf, remf handled inside pBinaryArithOp
let pBinaryArithIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Operators
        do! ensure (argIds.Length >= 2) "Not binary arith (need 2 args)"
        let! node = getCurrentNode
        let category = classifyAtomicOp info
        match category with
        | BinaryArith "shr" ->
            return! (pIfUnsignedArith node.Id "shrui" <|> pBinaryArithOp node.Id "shrsi")
        | BinaryArith "div" ->
            return! (pIfUnsignedArith node.Id "divu" <|> pBinaryArithOp node.Id "div")
        | BinaryArith "rem" ->
            return! (pIfUnsignedArith node.Id "remu" <|> pBinaryArithOp node.Id "rem")
        | BinaryArith op ->
            return! pBinaryArithOp node.Id op
        | Comparison "lt" -> return! (pIfUnsignedCmp node.Id "ult" <|> pComparisonOp node.Id "lt")
        | Comparison "le" -> return! (pIfUnsignedCmp node.Id "ule" <|> pComparisonOp node.Id "le")
        | Comparison "gt" -> return! (pIfUnsignedCmp node.Id "ugt" <|> pComparisonOp node.Id "gt")
        | Comparison "ge" -> return! (pIfUnsignedCmp node.Id "uge" <|> pComparisonOp node.Id "ge")
        | Comparison pred ->
            return! pComparisonOp node.Id pred  // eq/ne are sign-agnostic
        | _ -> return! fail (Message $"Not binary arith: {info.Operation}")
    }

/// Unary arithmetic intrinsic — boolean NOT (xori with 1)
let pUnaryArithIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Operators
        do! ensure (argIds.Length = 1) "Not unary arith (need 1 arg)"
        let! node = getCurrentNode
        let category = classifyAtomicOp info
        match category with
        | UnaryArith "xori"       -> return! pUnaryNot node.Id
        | UnaryArith "complement" -> return! pBitwiseNot node.Id
        | _ -> return! fail (Message $"Not unary arith: {info.Operation}")
    }

/// Type conversion intrinsic — byte(), int(), float(), nativeint()
let pTypeConversionIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Convert
        do! ensure (argIds.Length >= 1) "Convert: Expected 1 arg"
        let! node = getCurrentNode
        return! pTypeConversion node.Id
    }
