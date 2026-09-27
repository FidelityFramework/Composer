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
open Alex.Elements.MemRefElements // pExtractBasePtr (FFI boundary memref→index)
open Core.Types.Dialects         // TargetPlatform
open Alex.CodeGeneration.TypeMapping
open Alex.Patterns.MemoryPatterns // pRecallArgWithLoad (monadic TMemRef auto-load)
module CallableOperands = Alex.Traversal.CallableOperands

/// One admitted physical invocation. The callee is an already witnessed code
/// operand or a settled declaration symbol; this never walks or emits its body.
type CallableInvocation = Direct of string | Indirect of Val

let private pInvoke invocation arguments results = parser {
    match invocation with
    | Direct symbol -> return! pFuncCallResults results symbol arguments
    | Indirect code ->
        do! ensure (code.Type = TFunc(List.map (fun (value: Val) -> value.Type) arguments,
                                     List.map (fun (value: Val) -> value.Type) results))
                "Callable invocation operands disagree with its witnessed physical signature"
        return! pFuncCallIndirectResults results code.SSA arguments
}

/// A function-valued result keeps both operands through the ordinary func
/// operation. Its source-owned carrier was projected at the current occurrence.
let pCallableApplication nodeId invocation arguments shape = parser {
    let code = { SSA = Alex.Traversal.Values.value nodeId 0; Type = CallableOperands.functionType shape }
    let environment = CallableOperands.environmentType shape |> Option.map (fun ty ->
        { SSA = Alex.Traversal.Values.value nodeId 1; Type = ty })
    let! value =
        match CallableOperands.create shape code environment with
        | Result.Ok value -> preturn value
        | Result.Error reason -> fail (Message reason)
    let! operation = pInvoke invocation arguments (CallableOperands.values value)
    return [operation], TRCallable value
}

let pSequenceApplication nodeId invocation arguments shape = parser {
    let code = { SSA = Alex.Traversal.Values.value nodeId 0; Type = Alex.Traversal.SequenceOperands.functionType shape }
    let environment = { SSA = Alex.Traversal.Values.value nodeId 1; Type = Alex.Traversal.SequenceOperands.environmentType shape }
    let! value =
        match Alex.Traversal.SequenceOperands.create shape code environment with
        | Result.Ok value -> preturn value
        | Result.Error reason -> fail (Message reason)
    let! operation = pInvoke invocation arguments (Alex.Traversal.SequenceOperands.values value)
    return [operation], TRSequence value
}

let pLazyApplication nodeId invocation arguments shape = parser {
    let code = { SSA = Alex.Traversal.Values.value nodeId 0; Type = Alex.Traversal.LazyOperands.functionType shape }
    let environment = { SSA = Alex.Traversal.Values.value nodeId 1; Type = Alex.Traversal.LazyOperands.environmentType shape }
    let! value =
        match Alex.Traversal.LazyOperands.create shape code environment with
        | Result.Ok value -> preturn value
        | Result.Error reason -> fail (Message reason)
    let! operation = pInvoke invocation arguments (Alex.Traversal.LazyOperands.values value)
    return [operation], TRLazy value
}

let pIndirectApplication nodeId code arguments resultType = parser {
    let value = { SSA = Alex.Traversal.Values.value nodeId 0; Type = resultType }
    let! operation = pInvoke (Indirect code) arguments [value]
    return [operation], TRValue value
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
/// SSA extracted from coeffects via nodeId: [0] = result. Each argument arrives already at its
/// parameter's width (the witness applied the call's derived meets).
let pDirectCall (nodeId: NodeId) (funcName: string) (args: (SSA * MLIRType) list) (retType: MLIRType)
                (paramNames: string list option)
                : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pDirectCall: Expected 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]
        let castOps : MLIROp list = []
        let finalVals = args |> List.map (fun (ssa, ty) -> { SSA = ssa; Type = ty })

        let! targetPlatform = getTargetPlatform
        match targetPlatform with
        | FPGA ->
            // FPGA: hw.instance (spatial instantiation) instead of func.call (temporal).
            // The port names are the callee's declared parameter names; none are invented here.
            let! names =
                match paramNames with
                | Some names when names.Length = finalVals.Length -> preturn names
                | Some names ->
                    fail (Message $"CCS source checking did not settle one declared port name per argument for hw.instance of '{funcName}' at node {NodeId.value nodeId}: {names.Length} names for {finalVals.Length} arguments")
                | None ->
                    fail (Message $"CCS source checking did not settle the declared port names for hw.instance of '{funcName}' at node {NodeId.value nodeId}")
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

/// Select the FPGA combinational Element for the requested arithmetic operation.
let private pFpgaCombOp (operation: string) (resultSSA: SSA) (lhs: SSA) (rhs: SSA) (opTy: MLIRType) =
    match operation with
    | "add" -> pCombAdd resultSSA lhs rhs opTy
    | "sub" -> pCombSub resultSSA lhs rhs opTy
    | "mul" -> pCombMul resultSSA lhs rhs opTy
    | "div" -> pCombDivS resultSSA lhs rhs opTy
    | "rem" -> pCombMod resultSSA lhs rhs opTy
    | "remu" -> pCombModU resultSSA lhs rhs opTy
    | "andi" -> pCombAnd resultSSA lhs rhs opTy
    | "ori" -> pCombOr resultSSA lhs rhs opTy
    | "xori" -> pCombXor resultSSA lhs rhs opTy
    | "shli" -> pCombShl resultSSA lhs rhs opTy
    | "shrui" -> pCombShrU resultSSA lhs rhs opTy
    | "shrsi" -> pCombShrS resultSSA lhs rhs opTy
    | "divu" -> pCombDivU resultSSA lhs rhs opTy
    | _ -> fail (Message $"Unsupported FPGA arithmetic operation: {operation}")

/// Read one complete source operation; current occurrence and ordered actuals
/// must correspond exactly to the published joint recipe.
let private pNumericOperationAt nodeId : PSGParser<NumericOperationWitness> = parser {
    let! state = getUserState
    let publication = numericProjection state.Graph
    match publication.Operations.TryFind nodeId with
    | None -> return! fail (Message $"Numeric operation {NodeId.value nodeId} has no source-published operation contract.")
    | Some operation ->
        do! ensure (operation.Site = nodeId && state.Current.Id = nodeId)
                "Numeric operation publication names a different occurrence."
        match state.Current.Kind with
        | SemanticKind.Application(callee, actuals) ->
            do! ensure (callee = operation.Callee && actuals = List.map (fun (operand: NumericOperationOperand) -> operand.Actual) operation.Operands)
                    "Numeric operation publication differs from the exact callee or ordered actuals."
            return operation
        | _ -> return! fail (Message "Published numeric operation requires its exact application occurrence.")
}

/// Ordered actual positions reuse a canonical Meet once when the same source
/// value occurs repeatedly. Neither its carrier nor its adaptation is selected here.
let private pNumericOperands (operation: NumericOperationWitness) (operationType: MLIRType) =
    let rec read (witnessed: Map<NodeId, NumericOperationOperand * Val>) (operands: NumericOperationOperand list) = parser {
        match operands with
        | [] -> return [], []
        | operand :: rest ->
            let! operations, value =
                match witnessed.TryFind operand.Actual with
                | Some(previous, value) -> parser {
                    do! ensure (previous = operand) "Repeated numeric actual has different source adaptation premises."
                    return [], value
                  }
                | None -> parser {
                    let! state = getUserState
                    let! loads, ssa, ty = pRecallArgWithLoad operand.Actual
                    let expected =
                        match operand.Carrier with
                        | Some carrier -> scalarCarrierType carrier
                        | None -> valueTypeAt state.Graph operand.Actual
                    do! ensure (ty = expected) "Numeric operand differs from its source-published carrier."
                    let! adaptations, value = pPublishedAdapt operation.Site operand.Actual operand.Adaptation { SSA = ssa; Type = ty }
                    do! ensure (value.Type = operationType) "Numeric operand adaptation does not establish the published operation carrier."
                    return loads @ adaptations, value
                  }
            let! remaining, values = read (witnessed.Add(operand.Actual, (operand, value))) rest
            return operations @ remaining, value :: values
    }
    read Map.empty operation.Operands

/// The exact source result Meet connects an operation carrier to the held
/// result carrier; absence is equality, never a request to choose a cast.
let private pNumericResult (operation: NumericOperationWitness) operations value = parser {
    let! adaptations, result = pPublishedAdapt operation.Site operation.Site operation.ResultAdaptation value
    do! ensure (result.Type = scalarCarrierType operation.Result)
            "Numeric operation result differs from its source-published held carrier."
    return operations @ adaptations, TRValue result
}

let private integerForm kind signed =
    match kind with
    | NumericOperationKind.Add -> Some "add"
    | NumericOperationKind.Subtract -> Some "sub"
    | NumericOperationKind.Multiply -> Some "mul"
    | NumericOperationKind.Divide -> Some(if signed then "div" else "divu")
    | NumericOperationKind.Remainder -> Some(if signed then "rem" else "remu")
    | NumericOperationKind.BitAnd -> Some "andi"
    | NumericOperationKind.BitOr -> Some "ori"
    | NumericOperationKind.BitXor -> Some "xori"
    | NumericOperationKind.ShiftLeft -> Some "shli"
    | NumericOperationKind.ShiftRight -> Some(if signed then "shrsi" else "shrui")
    | _ -> None

let private integerPredicate kind signed =
    match kind with
    | NumericOperationKind.Equal -> Some ICmpPred.Eq
    | NumericOperationKind.NotEqual -> Some ICmpPred.Ne
    | NumericOperationKind.Less -> Some(if signed then ICmpPred.Slt else ICmpPred.Ult)
    | NumericOperationKind.LessOrEqual -> Some(if signed then ICmpPred.Sle else ICmpPred.Ule)
    | NumericOperationKind.Greater -> Some(if signed then ICmpPred.Sgt else ICmpPred.Ugt)
    | NumericOperationKind.GreaterOrEqual -> Some(if signed then ICmpPred.Sge else ICmpPred.Uge)
    | _ -> None

let private realPredicate kind =
    match kind with
    | NumericOperationKind.Equal -> Some FCmpPred.OEq
    | NumericOperationKind.NotEqual -> Some FCmpPred.ONe
    | NumericOperationKind.Less -> Some FCmpPred.OLt
    | NumericOperationKind.LessOrEqual -> Some FCmpPred.OLe
    | NumericOperationKind.Greater -> Some FCmpPred.OGt
    | NumericOperationKind.GreaterOrEqual -> Some FCmpPred.OGe
    | _ -> None

let private pCoreArithOp operation result lhs rhs ty =
    match operation with
    | "add" -> pAddI result lhs rhs ty
    | "sub" -> pSubI result lhs rhs ty
    | "mul" -> pMulI result lhs rhs ty
    | "div" -> pDivSI result lhs rhs ty
    | "divu" -> pDivUI result lhs rhs ty
    | "rem" -> pRemSI result lhs rhs ty
    | "remu" -> pRemUI result lhs rhs ty
    | "andi" -> pAndI result lhs rhs ty
    | "ori" -> pOrI result lhs rhs ty
    | "xori" -> pXorI result lhs rhs ty
    | "shli" -> pShLI result lhs rhs ty
    | "shrui" -> pShRUI result lhs rhs ty
    | "shrsi" -> pShRSI result lhs rhs ty
    | _ -> fail (Message $"No integer Element for published operation {operation}.")

/// Reusable composition of source-resolved scalar operations. Operation width,
/// interpretation, participants and all adaptations come from the publication.
let pNumericOperation nodeId : PSGParser<MLIROp list * TransferResult> = parser {
    let! operation = pNumericOperationAt nodeId
    let! operationType =
        match operation.OperationCarrier |> Option.bind SettledScalar.tryType with
        | Some ty -> preturn ty
        | None -> fail (Message "Numeric operation has no admitted source-published operation carrier.")
    let! operandOps, operands = pNumericOperands operation operationType
    let! result = getNodeSSA nodeId
    let! platform = getTargetPlatform
    let finish operations ty = pNumericResult operation (operandOps @ operations) { SSA = result; Type = ty }
    match operation.Form, operands with
    | NumericOperationForm.Integer signed, [left; right] ->
        match integerPredicate operation.Kind signed, integerForm operation.Kind signed with
        | Some predicate, _ ->
            let! instruction = if platform = FPGA then pCombICmp result predicate left.SSA right.SSA operationType else pCmpI result predicate left.SSA right.SSA operationType
            return! finish [instruction] (TInt(IntWidth 1))
        | None, Some form ->
            let! instruction = if platform = FPGA then pFpgaCombOp form result left.SSA right.SSA operationType else pCoreArithOp form result left.SSA right.SSA operationType
            return! finish [instruction] operationType
        | _ -> return! fail (Message "Published integer operation has no binary Element.")
    | (NumericOperationForm.Boolean | NumericOperationForm.Unit | NumericOperationForm.OpaqueReference), [left; right] ->
        do! ensure (operation.Kind = NumericOperationKind.Equal || operation.Kind = NumericOperationKind.NotEqual)
                "Published identity comparison requires equality or inequality."
        let predicate = if operation.Kind = NumericOperationKind.Equal then ICmpPred.Eq else ICmpPred.Ne
        let! instruction = if platform = FPGA then pCombICmp result predicate left.SSA right.SSA operationType else pCmpI result predicate left.SSA right.SSA operationType
        return! finish [instruction] (TInt(IntWidth 1))
    | NumericOperationForm.Real, [left; right] ->
        do! ensure (platform <> FPGA) "Real numeric operations have no fabric Element."
        match realPredicate operation.Kind with
        | Some predicate ->
            let! instruction = pCmpF result predicate left.SSA right.SSA operationType
            return! finish [instruction] (TInt(IntWidth 1))
        | None ->
            let! instruction =
                match operation.Kind with
                | NumericOperationKind.Add -> pAddF result left.SSA right.SSA operationType
                | NumericOperationKind.Subtract -> pSubF result left.SSA right.SSA operationType
                | NumericOperationKind.Multiply -> pMulF result left.SSA right.SSA operationType
                | NumericOperationKind.Divide -> pDivF result left.SSA right.SSA operationType
                | _ -> fail (Message "Published real operation has no binary Element.")
            return! finish [instruction] operationType
    | _, [operand] when operation.Kind = NumericOperationKind.Identity ->
        return! pNumericResult operation operandOps operand
    | NumericOperationForm.Real, [operand] when operation.Kind = NumericOperationKind.Negate ->
        do! ensure (platform <> FPGA) "Real negation has no fabric Element."
        let! instruction = pNegF result operand.SSA operationType
        return! finish [instruction] operationType
    | NumericOperationForm.Integer _, [operand]
        when operation.Kind = NumericOperationKind.Negate || operation.Kind = NumericOperationKind.Complement ->
        let! names = getNodeSSAs nodeId
        do! ensure (names.Length > 1) "Published unary operation has no canonical constant value name."
        let constant = if operation.Kind = NumericOperationKind.Negate then 0L else -1L
        let! literal = Alex.Elements.MLIRAtomics.pConstI names[1] constant operationType
        let! instruction =
            if operation.Kind = NumericOperationKind.Negate then
                if platform = FPGA then pCombSub result names[1] operand.SSA operationType else pSubI result names[1] operand.SSA operationType
            else
                if platform = FPGA then pCombXor result operand.SSA names[1] operationType else pXorI result operand.SSA names[1] operationType
        return! finish [literal; instruction] operationType
    | NumericOperationForm.Boolean, [operand] when operation.Kind = NumericOperationKind.LogicalNot ->
        let! names = getNodeSSAs nodeId
        do! ensure (names.Length > 1) "Published Boolean operation has no canonical constant value name."
        let! literal = Alex.Elements.MLIRAtomics.pConstI names[1] 1L operationType
        let! instruction = if platform = FPGA then pCombXor result operand.SSA names[1] operationType else pXorI result operand.SSA names[1] operationType
        return! finish [literal; instruction] operationType
    | _ -> return! fail (Message "Published numeric operation form, kind and ordered operands have no corresponding Element.")
}

/// `truncate` (the Math.truncate intrinsic; Dimensional_Range_Design.md §5, a real becomes an
/// integer): arith.fptosi from the operand's float type to the integer type the node carries.
let pTruncate (nodeId: NodeId)
              : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! argIds = pGetApplicationArgs
        do! ensure (argIds.Length >= 1) $"pTruncate: Expected 1 arg, got {argIds.Length}"

        let! (loadOps, srcSSA, srcType) = pRecallArgWithLoad argIds.[0]

        let! ssas = getNodeSSAs nodeId
        do! ensure (ssas.Length >= 1) $"pTruncate: Expected at least 1 SSA, got {ssas.Length}"
        let resultSSA = ssas.[0]

        let! state = getUserState
        let dstType = valueTypeAt state.Graph state.Current.Id

        match srcType, dstType with
        | TFloat _, TInt _ ->
            let! convOp = pFPToSI resultSSA srcSSA srcType dstType
            return (loadOps @ [convOp], TRValue { SSA = resultSSA; Type = dstType })
        | _ ->
            return! fail (Message $"pTruncate: expected a float operand and an integer result, got {srcType} -> {dstType}")
    }

// ═══════════════════════════════════════════════════════════
// TYPE CONVERSION PATTERN (IntrinsicModule.Convert)
// ═══════════════════════════════════════════════════════════

/// A conversion between numeric carriers (`int32 x`, `byte x`, `float x`, `int c`; the
/// width-named conversions are interim boundaries until CS-12 deletes the spellings).
/// Same-family scalar adaptations read the exact source numeric meet; physical
/// widths and operand ranges never select an extension or truncation here.
/// PULL model: extracts argument and result type from XParsec state.
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
        let dstType = valueTypeAt state.Graph state.Current.Id

        if (match srcType, dstType with TInt _, TInt _ | TFloat _, TFloat _ -> true | _ -> false) then
            let! operations, result = pSettledAdaptTo nodeId argIds.[0] dstType { SSA = srcSSA; Type = srcType }
            return loadOps @ operations, TRValue result
        elif srcType = dstType then
            return (loadOps, TRValue { SSA = srcSSA; Type = srcType })
        else
            let! convOp =
                match srcType, dstType with
                | TFloat _, TInt _ ->
                    pFPToSI resultSSA srcSSA srcType dstType
                | TInt _, TFloat _ ->
                    pSIToFP resultSSA srcSSA srcType dstType
                | TIndex, TInt _ ->
                    pIndexCastS resultSSA srcSSA TIndex dstType
                | TInt _, TIndex ->
                    pIndexCastS resultSSA srcSSA srcType TIndex
                | TMemRef _, TIndex | TMemRefStatic _, TIndex ->
                    pExtractBasePtr resultSSA srcSSA srcType
                | _ ->
                    fail (Message $"Unsupported type conversion: {srcType} -> {dstType}")
            return (loadOps @ [convOp], TRValue { SSA = resultSSA; Type = dstType })
    }

// ═══════════════════════════════════════════════════════════
// COMPOSED INTRINSIC PARSERS (per-operation, self-contained)
// ═══════════════════════════════════════════════════════════

/// Binary arithmetic and comparison witnessing requires the published binary
/// operation. Source interpretation selects the exact reusable Elements.
let pBinaryArithIntrinsic : PSGParser<MLIROp list * TransferResult> = parser {
    let! (_, actuals) = pIntrinsicApplication IntrinsicModule.Operators
    do! ensure (actuals.Length = 2) "Not a binary Operators occurrence."
    let! node = getCurrentNode
    return! pNumericOperation node.Id
}

/// Unary operations use the same source-resolved operation composition.
let pUnaryArithIntrinsic : PSGParser<MLIROp list * TransferResult> = parser {
    let! (_, actuals) = pIntrinsicApplication IntrinsicModule.Operators
    do! ensure (actuals.Length = 1) "Not a unary Operators occurrence."
    let! node = getCurrentNode
    return! pNumericOperation node.Id
}

/// Evaluation has already witnessed the argument and its effects. Discarding
/// its value produces the ordinary Clef unit value without inspecting storage.
let pIgnoreIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Operators
        do! ensure (info.Operation = "ignore" && argIds.Length = 1) "Not unary Operators.ignore"
        let! node = getCurrentNode
        let! ssas = getNodeSSAs node.Id
        do! ensure (ssas.Length >= 1) "Operators.ignore has no assigned result value"
        let! state = getUserState
        let unitTy = valueTypeAt state.Graph state.Current.Id
        let! unitValue = Alex.Elements.MLIRAtomics.pConstI ssas.[0] 0L unitTy
        return [unitValue], TRValue { SSA = ssas.[0]; Type = unitTy }
    }

/// `truncate` intrinsic (Math.truncate) — the one Math operation Alex witnesses atomically.
let pTruncateIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Math
        do! ensure (info.Operation = "truncate") "Not Math.truncate"
        do! ensure (argIds.Length = 1) "truncate: Expected 1 arg"
        let! node = getCurrentNode
        return! pTruncate node.Id
    }

/// Type conversion intrinsic — byte(), int(), float(), nativeint()
let pTypeConversionIntrinsic : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! (info, argIds) = pIntrinsicApplication IntrinsicModule.Convert
        do! ensure (argIds.Length >= 1) "Convert: Expected 1 arg"
        let! node = getCurrentNode
        return! pTypeConversion node.Id
    }
