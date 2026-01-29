/// ApplicationWitness - Witness function application nodes
///
/// Application nodes emit MLIR function calls (direct or indirect).
/// Post-order traversal ensures function and arguments are already witnessed.
///
/// NANOPASS: This witness handles ONLY Application nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.ApplicationWitness

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.ElisionPatterns
open Alex.Patterns.MemoryPatterns
open Alex.Patterns.PlatformPatterns
open Alex.CodeGeneration.TypeMapping

module SSAAssign = PSGElaboration.SSAAssignment

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Helper: Navigate to actual function node (unwrap TypeAnnotation if present)
let private resolveFunctionNode funcId graph =
    match SemanticGraph.tryGetNode funcId graph with
    | Some funcNode ->
        match funcNode.Kind with
        | SemanticKind.TypeAnnotation (innerFuncId, _) ->
            // Unwrap TypeAnnotation to get actual function
            SemanticGraph.tryGetNode innerFuncId graph
        | _ -> Some funcNode
    | None -> None

/// Witness application nodes - emits function calls
let private witnessApplication (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pApplication ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
    | Some ((funcId, argIds), _) ->
        // Resolve actual function node (unwrap TypeAnnotation if present)
        match resolveFunctionNode funcId ctx.Graph with
        | Some funcNode when funcNode.Kind.ToString().StartsWith("Intrinsic") ->
            // Intrinsic application - dispatch based on category
            match funcNode.Kind with
            | SemanticKind.Intrinsic info ->
                // Recall argument SSAs
                let argsResult =
                    argIds
                    |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)

                let allWitnessed = argsResult |> List.forall Option.isSome
                if not allWitnessed then
                    WitnessOutput.error $"Application: Intrinsic {info.FullName} arguments not yet witnessed"
                else
                    let argSSAs = argsResult |> List.choose id |> List.map fst

                    // Get result SSA
                    match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
                    | None -> WitnessOutput.error $"Application: No SSA for intrinsic {info.FullName}"
                    | Some resultSSA ->
                        // Dispatch based on intrinsic module and operation
                        match info.Module, info.Operation, argSSAs with
                        | IntrinsicModule.NativePtr, "stackalloc", [countSSA] ->
                            match tryMatch (pNativePtrStackAlloc resultSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "NativePtr.stackalloc pattern failed"

                        | IntrinsicModule.NativePtr, "write", [ptrSSA; valueSSA] ->
                            match tryMatch (pNativePtrWrite valueSSA ptrSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "NativePtr.write pattern failed"

                        | IntrinsicModule.NativePtr, "read", [ptrSSA] ->
                            match tryMatch (pNativePtrRead resultSSA ptrSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "NativePtr.read pattern failed"

                        | IntrinsicModule.Sys, "write", [fdSSA; bufferSSA; countSSA] ->
                            match tryMatch (pSysWrite resultSSA fdSSA bufferSSA countSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "Sys.write pattern failed"

                        | IntrinsicModule.Sys, "read", [fdSSA; bufferSSA; countSSA] ->
                            match tryMatch (pSysRead resultSSA fdSSA bufferSSA countSSA) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error "Sys.read pattern failed"

                        // Binary arithmetic intrinsics (+, -, *, /, %)
                        | IntrinsicModule.Operators, _, [lhsSSA; rhsSSA] ->
                            let arch = ctx.Coeffects.Platform.TargetArch
                            match tryMatch (pBuildBinaryArith resultSSA lhsSSA rhsSSA arch) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                            | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                            | None -> WitnessOutput.error $"Binary arithmetic pattern failed for {info.Operation}"

                        | _ -> WitnessOutput.error $"Intrinsic not yet implemented: {info.FullName} with {argSSAs.Length} args"

            | _ -> WitnessOutput.error "Expected Intrinsic SemanticKind"

        | Some funcNode when funcNode.Kind.ToString().StartsWith("VarRef") ->
            // Extract function name from VarRef node
            let funcName =
                match funcNode.Kind.ToString().Split([|'('; ','|]) with
                | parts when parts.Length > 1 ->
                    let name = parts.[1].Trim().Trim([|' '; '"'|])
                    sprintf "@%s" name
                | _ -> "@unknown_func"

            // Recall argument SSAs
            let argsResult =
                argIds
                |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)

            // Ensure all arguments were witnessed
            let allWitnessed = argsResult |> List.forall Option.isSome
            if not allWitnessed then
                WitnessOutput.error "Application: Some arguments not yet witnessed"
            else
                let argSSAs = argsResult |> List.choose id |> List.map fst

                // Get result SSA and return type
                match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
                | None -> WitnessOutput.error "Application: No SSA assigned to result"
                | Some resultSSA ->
                    let arch = ctx.Coeffects.Platform.TargetArch
                    let retType = mapNativeTypeForArch arch node.Type

                    // Emit direct function call by name
                    match tryMatch (pDirectCall resultSSA funcName argSSAs retType) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                    | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                    | None -> WitnessOutput.error "Direct function call pattern emission failed"

        | _ ->
            // Function is an SSA value (indirect call)
            match MLIRAccumulator.recallNode funcId ctx.Accumulator with
            | None -> WitnessOutput.error "Application: Function not yet witnessed"
            | Some (funcSSA, funcTy) ->
                // Recall argument SSAs
                let argsResult =
                    argIds
                    |> List.map (fun argId -> MLIRAccumulator.recallNode argId ctx.Accumulator)

                // Ensure all arguments were witnessed
                let allWitnessed = argsResult |> List.forall Option.isSome
                if not allWitnessed then
                    WitnessOutput.error "Application: Some arguments not yet witnessed"
                else
                    let argSSAs = argsResult |> List.choose id |> List.map fst

                    // Get result SSA and return type
                    match SSAAssign.lookupSSA node.Id ctx.Coeffects.SSA with
                    | None -> WitnessOutput.error "Application: No SSA assigned to result"
                    | Some resultSSA ->
                        let arch = ctx.Coeffects.Platform.TargetArch
                        let retType = mapNativeTypeForArch arch node.Type

                        // Emit indirect call
                        match tryMatch (pApplicationCall resultSSA funcSSA argSSAs retType) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error "Application pattern emission failed"
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// Application nanopass - witnesses function applications
let nanopass : Nanopass = {
    Name = "Application"
    Phase = ContentPhase
    Witness = witnessApplication
}
