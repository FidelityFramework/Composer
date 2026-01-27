/// SyscallOps - Witness platform/syscall operations to MLIR via bindings
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// This is a THIN ADAPTER layer. Platform-specific logic lives in Bindings/*.
/// This module simply:
/// 1. Builds PlatformPrimitive from witness context
/// 2. Calls PlatformDispatch.dispatch (which returns structured MLIROp)
/// 3. Converts BindingResult to (MLIROp list * TransferResult)
///
/// NO sprintf. NO platform-specific logic. Just dispatch coordination.
module Alex.Witnesses.SyscallOps

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Bindings.BindingTypes
open Alex.Bindings.PlatformTypes
open Alex.CodeGeneration.TypeMapping
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes

// SSA lookup alias from BindingTypes
module SSALookup = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════════════════════
// RESULT CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert BindingResult to witness return format
/// Returns (bodyOps, topLevelOps, result)
let private bindingResultToTransfer (result: BindingResult) : (MLIROp list * MLIROp list * TransferResult) option =
    match result with
    | BoundOps (ops, topLevelOps, Some resultVal) ->
        Some (ops, topLevelOps, TRValue resultVal)
    | BoundOps (ops, topLevelOps, None) ->
        Some (ops, topLevelOps, TRVoid)
    | NotSupported _ ->
        None

/// Convert BindingResult with error propagation
/// Returns (bodyOps, topLevelOps, result)
let private bindingResultToTransferWithError (result: BindingResult) : MLIROp list * MLIROp list * TransferResult =
    match result with
    | BoundOps (ops, topLevelOps, Some resultVal) ->
        ops, topLevelOps, TRValue resultVal
    | BoundOps (ops, topLevelOps, None) ->
        ops, topLevelOps, TRVoid
    | NotSupported reason ->
        [], [], TRError reason

// ═══════════════════════════════════════════════════════════════════════════
// PLATFORM BINDING DISPATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a platform binding operation
/// Entry point for SemanticKind.PlatformBinding nodes
/// Uses pre-assigned SSAs from SSAAssignment coeffects
let witnessPlatformBinding
    (appNodeId: NodeId)
    (ssa: SSALookup.SSAAssignment)
    (entryPoint: string)
    (args: Val list)
    (returnType: MLIRType)
    : (MLIROp list * MLIROp list * TransferResult) option =

    let prim: PlatformPrimitive = {
        EntryPoint = entryPoint
        Library = "platform"
        CallingConvention = "ccc"
        Args = args
        ReturnType = returnType
        BindingStrategy = Static
    }

    let result = PlatformDispatch.dispatch appNodeId ssa prim
    bindingResultToTransfer result

/// Witness a platform binding, returning error on failure
/// Uses pre-assigned SSAs from SSAAssignment coeffects
let witnessPlatformBindingRequired
    (appNodeId: NodeId)
    (ssa: SSALookup.SSAAssignment)
    (entryPoint: string)
    (args: Val list)
    (returnType: MLIRType)
    : MLIROp list * MLIROp list * TransferResult =

    let prim: PlatformPrimitive = {
        EntryPoint = entryPoint
        Library = "platform"
        CallingConvention = "ccc"
        Args = args
        ReturnType = returnType
        BindingStrategy = Static
    }

    let result = PlatformDispatch.dispatch appNodeId ssa prim
    bindingResultToTransferWithError result

// ═══════════════════════════════════════════════════════════════════════════
// SYS.* INTRINSIC DISPATCH
// ═══════════════════════════════════════════════════════════════════════════

/// Witness a Sys.* intrinsic operation
/// Dispatches to platform-specific bindings in SyscallBindings.fs
/// This is a THIN ADAPTER - syscall logic lives in Bindings/SyscallBindings.fs
let witnessSysOp
    (appNodeId: NodeId)
    (ssa: SSALookup.SSAAssignment)
    (opName: string)
    (args: Val list)
    (returnType: MLIRType)
    : (MLIROp list * MLIROp list * TransferResult) option =

    // Build entry point as "Sys.{opName}" for dispatch lookup
    let entryPoint = $"Sys.{opName}"

    let prim: PlatformPrimitive = {
        EntryPoint = entryPoint
        Library = "platform"
        CallingConvention = "ccc"
        Args = args
        ReturnType = returnType
        BindingStrategy = Static
    }

    let result = PlatformDispatch.dispatch appNodeId ssa prim
    bindingResultToTransfer result

// NOTE: witnessConsoleOp removed - Console is NOT an intrinsic
// It's Layer 3 user code in Fidelity.Platform that uses Sys.* intrinsics.
// See fsnative-spec/spec/platform-bindings.md

// ═══════════════════════════════════════════════════════════════════════════
// TYPE MAPPING (uses TypeMapping.mapNativeType)
// ═══════════════════════════════════════════════════════════════════════════

/// Witness with NativeType conversion
/// Uses TypeMapping.mapNativeType for authoritative type mapping
/// Uses pre-assigned SSAs from SSAAssignment coeffects
let witnessPlatformBindingNative
    (appNodeId: NodeId)
    (ssa: SSALookup.SSAAssignment)
    (arch: Architecture)
    (entryPoint: string)
    (args: Val list)
    (returnType: NativeType)
    : (MLIROp list * MLIROp list * TransferResult) option =

    let mlirReturnType = mapNativeTypeForArch arch returnType
    witnessPlatformBinding appNodeId ssa entryPoint args mlirReturnType

