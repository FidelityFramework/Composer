/// MLIR Emitters - Combinator-based MLIR emission
///
/// This module provides higher-level PSG parsers that combine pattern
/// recognition with template-based MLIR emission. Each emitter:
/// 1. Matches a PSG pattern using PSGCombinators
/// 2. Extracts required information
/// 3. Applies the appropriate template
/// 4. Updates the zipper with the emission result
///
/// The emitters compose naturally via the combinator operators,
/// enabling complex patterns to be expressed declaratively.
module Alex.XParsec.MLIREmitters

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.Traversal.MLIRZipper
open Alex.XParsec.PSGCombinators
open Alex.Templates.TemplateTypes

module ArithTemplates = Alex.Templates.ArithTemplates
module LLVMTemplates = Alex.Templates.LLVMTemplates
module MemTemplates = Alex.Templates.MemoryTemplates
module CFTemplates = Alex.Templates.ControlFlowTemplates

// ═══════════════════════════════════════════════════════════════════════════
// EMISSION RESULT TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// Result of an emission operation
type EmissionResult =
    | EmittedValue of ssa: string * mlirType: string
    | EmittedVoid
    | EmittedError of message: string

/// Parser that produces an emission result and updates the zipper
type EmitterParser = PSGParser<EmissionResult>

// ═══════════════════════════════════════════════════════════════════════════
// HELPER FUNCTIONS
// ═══════════════════════════════════════════════════════════════════════════

/// Emit an operation using a template, updating the zipper
let emitWithTemplate (template: MLIRTemplate<'P>) (templateParams: 'P) : PSGParser<string * MLIRZipper> =
    fun state ->
        let text = render template templateParams
        let ssaName, zipper' = MLIRZipper.yieldSSA state.Zipper
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Alex.CodeGeneration.MLIRTypes.Pointer zipper'
        Matched (ssaName, zipper''), { state with Zipper = zipper'' }

/// Recall an SSA value for a previously witnessed node
let recallSSA (nodeId: NodeId) : PSGParser<string * string> =
    fun state ->
        match MLIRZipper.recallNodeSSA (string (NodeId.value nodeId)) state.Zipper with
        | Some (ssa, ty) -> Matched (ssa, ty), state
        | None -> NoMatch (sprintf "Node %A SSA not found" nodeId), state

/// Bind an SSA value to a node
let bindSSA (nodeId: NodeId) (ssa: string) (ty: string) : PSGParser<unit> =
    fun state ->
        let zipper' = MLIRZipper.bindNodeSSA (string (NodeId.value nodeId)) ssa ty state.Zipper
        Matched (), { state with Zipper = zipper' }

// ═══════════════════════════════════════════════════════════════════════════
// LITERAL EMITTERS
// ═══════════════════════════════════════════════════════════════════════════

/// Emit an integer constant
let emitIntConstant (value: int64) (bitWidth: string) : PSGParser<string> =
    fun state ->
        let ssaName, zipper' = MLIRZipper.yieldSSA state.Zipper
        let constParams : ConstantParams = { Result = ssaName; Value = string value; Type = bitWidth }
        let text = render ArithTemplates.Quot.Constant.intConst constParams
        let resultType =
            match bitWidth with
            | "i8" -> Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I8
            | "i16" -> Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I16
            | "i32" -> Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I32
            | "i64" -> Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I64
            | _ -> Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I64
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName resultType zipper'
        Matched ssaName, { state with Zipper = zipper'' }

/// Emit a float constant
let emitFloatConstant (value: float) (bitWidth: string) : PSGParser<string> =
    fun state ->
        let ssaName, zipper' = MLIRZipper.yieldSSA state.Zipper
        let constParams : ConstantParams = { Result = ssaName; Value = string value; Type = bitWidth }
        let text = render ArithTemplates.Quot.Constant.floatConst constParams
        let resultType =
            match bitWidth with
            | "f32" -> Alex.CodeGeneration.MLIRTypes.Float Alex.CodeGeneration.MLIRTypes.F32
            | "f64" -> Alex.CodeGeneration.MLIRTypes.Float Alex.CodeGeneration.MLIRTypes.F64
            | _ -> Alex.CodeGeneration.MLIRTypes.Float Alex.CodeGeneration.MLIRTypes.F64
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName resultType zipper'
        Matched ssaName, { state with Zipper = zipper'' }

// ═══════════════════════════════════════════════════════════════════════════
// ARITHMETIC EMITTERS
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a binary arithmetic operation given the operand SSAs and type
let emitBinaryArith (mlirOp: string) (lhsSSA: string) (rhsSSA: string) (tyStr: string) : PSGParser<string> =
    fun state ->
        let ssaName, zipper' = MLIRZipper.yieldSSA state.Zipper
        let text = sprintf "%s = %s %s, %s : %s" ssaName mlirOp lhsSSA rhsSSA tyStr
        let resultType = Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I64 // Default; could be more precise
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName resultType zipper'
        Matched ssaName, { state with Zipper = zipper'' }

/// Emit a comparison operation given operand SSAs
let emitComparison (predicate: string) (lhsSSA: string) (rhsSSA: string) (tyStr: string) : PSGParser<string> =
    fun state ->
        let ssaName, zipper' = MLIRZipper.yieldSSA state.Zipper
        let text = sprintf "%s = arith.cmpi %s, %s, %s : %s" ssaName predicate lhsSSA rhsSSA tyStr
        let resultType = Alex.CodeGeneration.MLIRTypes.Integer Alex.CodeGeneration.MLIRTypes.I1
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName resultType zipper'
        Matched ssaName, { state with Zipper = zipper'' }

// ═══════════════════════════════════════════════════════════════════════════
// MEMORY EMITTERS
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a load operation
let emitLoad (ptrSSA: string) (resultTypeStr: string) : PSGParser<string> =
    fun state ->
        let ssaName, zipper' = MLIRZipper.yieldSSA state.Zipper
        let loadParams : LoadParams = { Result = ssaName; Pointer = ptrSSA; Type = resultTypeStr }
        let text = render LLVMTemplates.Quot.Memory.load loadParams
        let resultType = Alex.CodeGeneration.MLIRTypes.Pointer // Simplified
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName resultType zipper'
        Matched ssaName, { state with Zipper = zipper'' }

/// Emit a store operation
let emitStore (valueSSA: string) (valueType: string) (ptrSSA: string) : PSGParser<unit> =
    fun state ->
        let storeParams : StoreParams = { Value = valueSSA; Pointer = ptrSSA; Type = valueType }
        let text = render LLVMTemplates.Quot.Memory.store storeParams
        let zipper' = MLIRZipper.witnessVoidOp text state.Zipper
        Matched (), { state with Zipper = zipper' }

/// Emit an alloca operation
let emitAlloca (countSSA: string) (elementType: string) : PSGParser<string> =
    fun state ->
        let ssaName, zipper' = MLIRZipper.yieldSSA state.Zipper
        let allocaParams : AllocaParams = { Result = ssaName; Count = countSSA; ElementType = elementType }
        let text = render LLVMTemplates.Quot.Memory.alloca allocaParams
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Alex.CodeGeneration.MLIRTypes.Pointer zipper'
        Matched ssaName, { state with Zipper = zipper'' }

/// Emit an addressof operation for a global
let emitAddressOf (globalName: string) : PSGParser<string> =
    fun state ->
        let ssaName, zipper' = MLIRZipper.yieldSSA state.Zipper
        let text = render LLVMTemplates.Quot.Memory.addressof {| Result = ssaName; GlobalName = globalName |}
        let zipper'' = MLIRZipper.witnessOpWithResult text ssaName Alex.CodeGeneration.MLIRTypes.Pointer zipper'
        Matched ssaName, { state with Zipper = zipper'' }

// ═══════════════════════════════════════════════════════════════════════════
// COMPOSITE PATTERNS
// ═══════════════════════════════════════════════════════════════════════════

/// Pattern: Binary arithmetic application
/// Matches: Application where func is a binary arithmetic intrinsic
let pBinaryArithApp : PSGParser<IntrinsicInfo * NodeId * NodeId> =
    pApplication >>= fun (funcId, argIds) ->
        match argIds with
        | [lhsId; rhsId] ->
            onChild funcId pIntrinsic >>= fun info ->
                match classifyIntrinsic info with
                | BinaryArith _ -> preturn (info, lhsId, rhsId)
                | _ -> pfail "Not a binary arithmetic intrinsic"
        | _ -> pfail "Binary op requires exactly 2 arguments"

/// Pattern: Comparison application
/// Matches: Application where func is a comparison intrinsic
let pComparisonApp : PSGParser<IntrinsicInfo * NodeId * NodeId> =
    pApplication >>= fun (funcId, argIds) ->
        match argIds with
        | [lhsId; rhsId] ->
            onChild funcId pIntrinsic >>= fun info ->
                match classifyIntrinsic info with
                | Comparison _ -> preturn (info, lhsId, rhsId)
                | _ -> pfail "Not a comparison intrinsic"
        | _ -> pfail "Comparison op requires exactly 2 arguments"

/// Pattern: Console operation application
let pConsoleApp : PSGParser<IntrinsicInfo * NodeId list> =
    pApplication >>= fun (funcId, argIds) ->
        onChild funcId pIntrinsic >>= fun info ->
            match classifyIntrinsic info with
            | ConsoleOp _ -> preturn (info, argIds)
            | _ -> pfail "Not a console intrinsic"

/// Pattern: Platform (Sys) operation application
let pPlatformApp : PSGParser<IntrinsicInfo * NodeId list> =
    pApplication >>= fun (funcId, argIds) ->
        onChild funcId pIntrinsic >>= fun info ->
            match classifyIntrinsic info with
            | PlatformOp _ -> preturn (info, argIds)
            | _ -> pfail "Not a platform intrinsic"

// ═══════════════════════════════════════════════════════════════════════════
// FULL EMITTER COMBINATORS
// ═══════════════════════════════════════════════════════════════════════════

/// Emit a binary arithmetic operation by matching the pattern and generating MLIR
let emitBinaryArithOp : EmitterParser =
    psg {
        let! (info, lhsId, rhsId) = pBinaryArithApp
        let! (lhsSSA, lhsTy) = recallSSA lhsId
        let! (rhsSSA, rhsTy) = recallSSA rhsId
        let mlirOp =
            match classifyIntrinsic info with
            | BinaryArith op -> "arith." + op
            | _ -> "arith.addi" // Fallback
        let! resultSSA = emitBinaryArith mlirOp lhsSSA rhsSSA lhsTy
        return EmittedValue (resultSSA, lhsTy)
    }

/// Emit a comparison operation
let emitComparisonOp : EmitterParser =
    psg {
        let! (info, lhsId, rhsId) = pComparisonApp
        let! (lhsSSA, lhsTy) = recallSSA lhsId
        let! (rhsSSA, _) = recallSSA rhsId
        let predicate =
            match classifyIntrinsic info with
            | Comparison pred -> pred
            | _ -> "eq" // Fallback
        let! resultSSA = emitComparison predicate lhsSSA rhsSSA lhsTy
        return EmittedValue (resultSSA, "i1")
    }

/// Try to emit using one of the available emitters
let tryEmit : EmitterParser =
    emitBinaryArithOp <|> emitComparisonOp <|> (pfail "No matching emitter")
