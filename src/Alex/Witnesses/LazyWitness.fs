/// Lazy Witness - Witness Lazy<'T> values to MLIR
///
/// PRD-14: Deferred computation with memoization
///
/// ARCHITECTURAL PRINCIPLES (Four Pillars):
/// 1. Coeffects: SSA assignment is pre-computed, lookup via context
/// 2. Active Patterns: Match on semantic meaning (LazyExpr, LazyForce)
/// 3. Zipper: Navigate and accumulate structured ops
/// 4. Templates: Return structured MLIROp types, no sprintf
///
/// LAZY STRUCT LAYOUT:
/// !lazy_T = !llvm.struct<(i1, T, !closure_type)>
///   - Field 0: Computed flag (i1)
///   - Field 1: Cached value (T)
///   - Field 2: Thunk closure {code_ptr, env_ptr}
///
/// OPERATIONS:
/// - Lazy.create: Build lazy struct with thunk, flag=false
/// - Lazy.force: Check flag, compute if needed, cache and return
/// - Lazy.isValueCreated: Return the computed flag
module Alex.Witnesses.LazyWitness

open FSharp.Native.Compiler.Checking.Native.SemanticGraph
open Alex.Dialects.Core.Types
open Alex.Dialects.SCF.Templates
open Alex.Traversal.PSGZipper
open Alex.CodeGeneration.TypeMapping

// ═══════════════════════════════════════════════════════════════════════════
// LAZY STRUCT TYPE
// ═══════════════════════════════════════════════════════════════════════════

/// Build the MLIR type for Lazy<T>
/// Layout: { computed: i1, value: T, thunk: {ptr, ptr} }
let lazyStructType (elementType: MLIRType) : MLIRType =
    let closureType = TStruct [TPtr; TPtr]  // {code_ptr, env_ptr}
    TStruct [TInt I1; elementType; closureType]

/// Standard closure type for thunks: {code_ptr, env_ptr}
let closureType = TStruct [TPtr; TPtr]

// ═══════════════════════════════════════════════════════════════════════════
// LAZY.CREATE - Build deferred computation
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Lazy.create: (unit -> 'T) -> Lazy<'T>
/// Takes a thunk closure, builds a lazy struct with computed=false
///
/// Input: thunkVal - the closure representing the deferred computation
/// Output: Lazy struct with computed=false, thunk stored
let witnessLazyCreate
    (appNodeId: NodeId)
    (z: PSGZipper)
    (thunkVal: Val)
    (elementType: MLIRType)
    : (MLIROp list * TransferResult) =

    let lazyType = lazyStructType elementType
    let ssas = requireNodeSSAs appNodeId z
    let resultSSA = requireNodeSSA appNodeId z

    // Pre-assigned SSAs for intermediate values
    let falseSSA = ssas.[0]
    let undefSSA = ssas.[1]
    let withFlagSSA = ssas.[2]
    let withThunkSSA = ssas.[3]

    let ops = [
        // Create false constant for computed flag
        MLIROp.ArithOp (ArithOp.ConstI (falseSSA, 0L, MLIRTypes.i1))

        // Create undef lazy struct
        MLIROp.LLVMOp (LLVMOp.Undef (undefSSA, lazyType))

        // Insert computed=false at index 0
        MLIROp.LLVMOp (LLVMOp.InsertValue (withFlagSSA, undefSSA, falseSSA, [0], lazyType))

        // Insert thunk closure at index 2 (skip value at index 1)
        MLIROp.LLVMOp (LLVMOp.InsertValue (resultSSA, withFlagSSA, thunkVal.SSA, [2], lazyType))
    ]

    (ops, TRValue { SSA = resultSSA; Type = lazyType })

// ═══════════════════════════════════════════════════════════════════════════
// LAZY.FORCE - Evaluate and cache
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Lazy.force: Lazy<'T> -> 'T
/// Checks if computed, returns cached or computes and caches
///
/// MLIR Structure:
/// 1. Extract computed flag
/// 2. Branch based on flag
/// 3. Cached path: extract and return value
/// 4. Compute path: extract thunk, call it, cache result
/// 5. Phi to merge paths
///
/// NOTE: For simplicity, we implement inline force without mutating the lazy struct.
/// A full implementation would use alloca + store for caching.
/// This version evaluates EVERY time (no memoization) - suitable for pure thunks.
/// TODO: Implement proper memoization with alloca for mutable lazy struct.
let witnessLazyForce
    (appNodeId: NodeId)
    (z: PSGZipper)
    (lazyVal: Val)
    (elementType: MLIRType)
    : (MLIROp list * TransferResult) =

    let lazyType = lazyStructType elementType
    let ssas = requireNodeSSAs appNodeId z
    let resultSSA = requireNodeSSA appNodeId z

    // For now, implement a simple "always compute" version
    // This extracts the thunk and calls it every time
    // True memoization requires ptr-based lazy structs with mutation

    // Pre-assigned SSAs
    let computedSSA = ssas.[0]
    let thunkSSA = ssas.[1]
    let codePtrSSA = ssas.[2]
    let envPtrSSA = ssas.[3]

    let ops = [
        // Extract computed flag (unused in this simple impl, but shown for structure)
        MLIROp.LLVMOp (LLVMOp.ExtractValue (computedSSA, lazyVal.SSA, [0], lazyType))

        // Extract thunk closure from lazy struct
        MLIROp.LLVMOp (LLVMOp.ExtractValue (thunkSSA, lazyVal.SSA, [2], lazyType))

        // Extract code_ptr and env_ptr from thunk closure
        MLIROp.LLVMOp (LLVMOp.ExtractValue (codePtrSSA, thunkSSA, [0], closureType))
        MLIROp.LLVMOp (LLVMOp.ExtractValue (envPtrSSA, thunkSSA, [1], closureType))

        // Call thunk: code_ptr(env_ptr) -> 'T
        // The thunk function signature is: (ptr) -> T (env is first arg)
        MLIROp.LLVMOp (LLVMOp.IndirectCall (Some resultSSA, codePtrSSA, [{ SSA = envPtrSSA; Type = TPtr }], elementType))
    ]

    (ops, TRValue { SSA = resultSSA; Type = elementType })

// ═══════════════════════════════════════════════════════════════════════════
// LAZY.ISVALUECREATED - Check if computed
// ═══════════════════════════════════════════════════════════════════════════

/// Witness Lazy.isValueCreated: Lazy<'T> -> bool
/// Returns the computed flag from the lazy struct
let witnessLazyIsValueCreated
    (appNodeId: NodeId)
    (z: PSGZipper)
    (lazyVal: Val)
    (elementType: MLIRType)
    : (MLIROp list * TransferResult) =

    let lazyType = lazyStructType elementType
    let resultSSA = requireNodeSSA appNodeId z

    let ops = [
        // Extract computed flag from lazy struct
        MLIROp.LLVMOp (LLVMOp.ExtractValue (resultSSA, lazyVal.SSA, [0], lazyType))
    ]

    (ops, TRValue { SSA = resultSSA; Type = MLIRTypes.i1 })

// ═══════════════════════════════════════════════════════════════════════════
// LAZY INTRINSIC DISPATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Dispatch Lazy intrinsic operations
/// Called from Application/Witness.fs when encountering IntrinsicModule.Lazy
let witnessLazyOp
    (appNodeId: NodeId)
    (z: PSGZipper)
    (opName: string)
    (args: Val list)
    (resultType: MLIRType)
    : (MLIROp list * TransferResult) option =

    match opName, args with
    | "create", [thunkVal] ->
        // Determine element type from result (Lazy<T> -> T)
        let elementType =
            match resultType with
            | TStruct [_; elemTy; _] -> elemTy  // Extract T from {i1, T, closure}
            | _ -> MLIRTypes.i64  // Fallback
        Some (witnessLazyCreate appNodeId z thunkVal elementType)

    | "force", [lazyVal] ->
        // resultType is the element type T
        Some (witnessLazyForce appNodeId z lazyVal resultType)

    | "isValueCreated", [lazyVal] ->
        // Extract element type from lazy struct
        let elementType =
            match lazyVal.Type with
            | TStruct [_; elemTy; _] -> elemTy
            | _ -> MLIRTypes.i64
        Some (witnessLazyIsValueCreated appNodeId z lazyVal elementType)

    | _ -> None
