/// VarRefWitness - Witness variable reference nodes
///
/// Variable references don't emit MLIR - they forward the binding's SSA.
/// The binding SSA is looked up from the accumulator (bindings witnessed first in post-order).
///
/// FUNCTION REFERENCES: VarRef nodes pointing to function bindings (Lambda nodes) build
/// closure pairs via pNamedFunctionAsClosure, enabling named functions as first-class values.
/// For direct calls, ApplicationWitness navigates to VarRef for name resolution independently.
///
/// NANOPASS: This witness handles ONLY VarRef nodes.
/// All other nodes return WitnessOutput.skip for other nanopasses to handle.
module Alex.Witnesses.VarRefWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.MemRefPatterns  // pLoadMutableVariable
open Alex.Patterns.ClosurePatterns  // pNamedFunctionAsClosure
open Alex.Dialects.Core.Types  // TMemRef
open XParsec
open XParsec.Parsers
open XParsec.Combinators

// ═══════════════════════════════════════════════════════════
// CATEGORY-SELECTIVE WITNESS (Private)
// ═══════════════════════════════════════════════════════════

/// Witness variable reference nodes - forwards binding's SSA
let private witnessVarRef (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    match tryMatch pVarRef ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((name, bindingIdOpt), _) ->
        match bindingIdOpt with
        | Some bindingId ->
            // Check if the binding references a function (Lambda node)
            match SemanticGraph.tryGetNode bindingId ctx.Graph with
            | Some bindingNode ->
                // Check binding type
                match bindingNode.Kind with
                | SemanticKind.PatternBinding _ ->
                    // Check accumulator first — match arm Var bindings are bound to
                    // the scrutinee SSA by MatchWitness, not pre-assigned in coeffects.
                    match MLIRAccumulator.recallNode bindingId ctx.Accumulator with
                    | Some (ssa, ty) ->
                        { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = ssa; Type = ty } }
                    | None ->
                        // Function parameter binding — SSA is in coeffects
                        // Uses platform-aware mapping + per-node width narrowing from coeffects
                        let patternBindingPattern =
                            parser {
                                let! ssa = getNodeSSA bindingId
                                let! state = getUserState
                                let platform = state.Coeffects.TargetPlatform
                                let arch = state.Coeffects.Platform.TargetArch
                                let rawTy = Alex.CodeGeneration.TypeMapping.mapNativeTypeForTarget platform arch state.Graph bindingNode.Type
                                let ty = Alex.XParsec.PSGCombinators.narrowType state.Coeffects bindingId rawTy
                                return ([], TRValue { SSA = ssa; Type = ty })
                            }

                        match tryMatch patternBindingPattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
                        | None -> WitnessOutput.error $"VarRef '{name}': PatternBinding has no SSA in coeffects"

                | SemanticKind.Binding (_, isMut, _, _) ->
                    // Check if binding's child is a Lambda (function binding)
                    let isFunctionBinding =
                        bindingNode.Children
                        |> List.tryHead
                        |> Option.bind (fun childId -> SemanticGraph.tryGetNode childId ctx.Graph)
                        |> Option.map (fun childNode -> childNode.Kind.ToString().StartsWith("Lambda"))
                        |> Option.defaultValue false

                    if isFunctionBinding then
                        // Function reference - check if the binding holds a closure value
                        match MLIRAccumulator.recallNode bindingId ctx.Accumulator with
                        | Some (closureSSA, closureTy) ->
                            // Closure-captured function value — return the closure pair
                            { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = closureSSA; Type = closureTy } }
                        | None ->
                            // Value position vs call position is a COEFFECT (pre-computed).
                            // "The zipper witnesses, it does not decide." — SpeakEZ: Learning to Walk
                            if not (Set.contains node.Id ctx.Coeffects.ValuePosition.FunctionVarRefsInValuePosition) then
                                // Call position — ApplicationWitness handles direct call
                                { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                            else
                            // Value position — build closure pair with thunk wrapper.
                            // The thunk accepts env (ignored) + params, forwarding to the original.
                            // This bridges the calling convention gap: closure calls prepend env,
                            // but named functions don't expect it.
                            let namedFuncClosurePattern =
                                parser {
                                    let! state = getUserState
                                    let! ssas = getNodeSSAs node.Id
                                    // Resolve the QUALIFIED function name: Binding name + ModuleDef parent
                                    // Same logic as LambdaWitness and ApplicationWitness direct call path
                                    let funcName =
                                        match bindingNode.Kind with
                                        | SemanticKind.Binding (bindName, _, _, _) ->
                                            match bindingNode.Parent with
                                            | Some parentId ->
                                                match SemanticGraph.tryGetNode parentId state.Graph with
                                                | Some parentNode ->
                                                    match parentNode.Kind with
                                                    | SemanticKind.ModuleDef (moduleName, _) ->
                                                        sprintf "%s.%s" moduleName bindName
                                                    | _ -> bindName
                                                | None -> bindName
                                            | None -> bindName
                                        | _ -> name
                                    do! ensure (funcName <> "") $"VarRef '{name}': Could not resolve function name"
                                    // Extract param/return types from the binding node's TFun type
                                    let platform = state.Coeffects.TargetPlatform
                                    let arch = state.Coeffects.Platform.TargetArch
                                    let rec extractParamTypes ty acc =
                                        match ty with
                                        | Clef.Compiler.NativeTypedTree.NativeTypes.NativeType.TFun (domain, range) ->
                                            let mlirTy = Alex.CodeGeneration.TypeMapping.mapNativeTypeForTarget platform arch state.Graph domain
                                            extractParamTypes range (mlirTy :: acc)
                                        | _ -> (List.rev acc, Alex.CodeGeneration.TypeMapping.mapNativeTypeForTarget platform arch state.Graph ty)
                                    let (paramMLIRTypes, returnMLIRType) = extractParamTypes bindingNode.Type []
                                    let! (inlineOps, topLevelOps, pairSSA, pairTy) =
                                        pNamedFunctionAsClosure funcName paramMLIRTypes returnMLIRType ssas
                                    return (inlineOps, topLevelOps, TRValue { SSA = pairSSA; Type = pairTy })
                                }
                            match tryMatch namedFuncClosurePattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                            | Some ((inlineOps, topLevelOps, result), _) ->
                                { InlineOps = inlineOps; TopLevelOps = topLevelOps; Result = result }
                            | None ->
                                WitnessOutput.error $"VarRef '{name}': Failed to build closure pair for named function"
                    elif Set.contains bindingId ctx.Coeffects.CurryFlattening.PartialAppBindings then
                        // Partial application binding - no value SSA available
                        // ApplicationWitness handles saturated calls through the coeffect
                        { InlineOps = []; TopLevelOps = []; Result = TRVoid }
                    else
                        // Value binding - post-order: binding already witnessed, recall its SSA
                        match MLIRAccumulator.recallNode bindingId ctx.Accumulator with
                        | Some (ssa, ty) ->
                            // Auto-load ONLY if the Binding is mutable (isMut from PSG).
                            // Mutable bindings hold memref<1xT> cells that need memref.load.
                            // Immutable bindings (including NativePtr.stackalloc results) forward as-is.
                            if isMut then
                                // Mutable cell — extract element type for auto-load
                                let elemTypeOpt =
                                    match ty with
                                    | TMemRef elemType -> Some elemType
                                    | TMemRefStatic (_, elemType) -> Some elemType
                                    | _ -> None
                                match elemTypeOpt with
                                | Some elemType ->
                                    let (NodeId nodeIdInt) = node.Id
                                    match tryMatchWithDiagnostics (pLoadMutableVariable nodeIdInt ssa elemType)
                                                  ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
                                    | Result.Ok ((ops, result), _) ->
                                        { InlineOps = ops; TopLevelOps = []; Result = result }
                                    | Result.Error diagnostic ->
                                        WitnessOutput.error $"VarRef '{name}': {diagnostic}"
                                | None ->
                                    WitnessOutput.error $"VarRef '{name}': Mutable cell has unexpected type {ty}"
                            else
                                // Immutable value (including buffers): forward directly
                                { InlineOps = []; TopLevelOps = []; Result = TRValue { SSA = ssa; Type = ty } }
                        | None ->
                            WitnessOutput.error $"VarRef '{name}': Binding not yet witnessed"

                | _ ->
                    WitnessOutput.error $"VarRef '{name}': Unexpected binding kind {bindingNode.Kind}"
            | None ->
                WitnessOutput.error $"VarRef '{name}': Binding node not found"
        | None ->
            WitnessOutput.error $"VarRef '{name}': No binding ID (unresolved reference)"
    | None ->
        WitnessOutput.skip

// ═══════════════════════════════════════════════════════════
// NANOPASS REGISTRATION (Public)
// ═══════════════════════════════════════════════════════════

/// VarRef nanopass - witnesses variable references
let nanopass : Nanopass = {
    Name = "VarRef"
    Witness = witnessVarRef
}
