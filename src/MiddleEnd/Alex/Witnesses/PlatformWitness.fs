/// PlatformWitness - Witness platform operations to MLIR via XParsec
///
/// Composes per-operation parsers with <|> — no dispatch hub.
/// Intrinsic parsers self-check via pIntrinsicApplication + ensure.
/// ExternCall parser guards on coeffect presence (Platform.Bindings).
///
/// NANOPASS: Handles Sys.write, Sys.read intrinsic applications
/// and [<FidelityExtern>] resolved calls (ExternCall in coeffects).
///
/// STATIC vs DYNAMIC (Mar 2026):
///   Static externs (library = "c") use pExternCallResolved (func.call with extern decl).
///   Dynamic externs (library != "c") use pDynamicExternCallResolved (dlopen/dlsym/call_indirect).
///   Dynamic externs require TopLevelOps for string constant globals (library path, symbol name).
module Alex.Witnesses.PlatformWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.PlatformPatterns
open XParsec.Combinators  // <|>

let private foreignResult (ctx: WitnessContext) (node: SemanticNode) ops topLevel result =
    match result with
    | TRValue value ->
        let meets, ssa, ty = adaptOperand ctx.Coeffects ctx.Graph node.Id node.Id value.SSA value.Type
        { InlineOps = ops @ meets; TopLevelOps = topLevel; Result = TRValue { SSA = ssa; Type = ty } }
    | _ -> { InlineOps = ops; TopLevelOps = topLevel; Result = result }

let private witnessPlatform (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // Try dynamic extern first — returns pending globals that need TopLevelOps emission.
    // Dynamic externs (library != "c") use dlopen/dlsym/call_indirect.
    match tryMatch pDynamicExternCallResolved ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
    | Some ((inlineOps, pendingGlobals, result), _) ->
        // Emit GlobalString for each pending global (deduplicated via accumulator).
        // These are the extern boundary's strings -- the dlopen path and dlsym
        // symbol -- synthesized here from the binding with no literal node in the
        // graph, so no obligation cites them and they carry no anchors. That is a
        // recorded gap at the FFI fence (Witness_Boundary_Audit 4g; C-01 6.7), not
        // something to fill in below the graph.
        let topLevelOps =
            pendingGlobals
            |> List.choose (fun (name, content, storageLen) ->
                MLIRAccumulator.tryEmitGlobal name content storageLen [] ctx.Accumulator)
        foreignResult ctx node inlineOps topLevelOps result
    | None ->
        match Map.tryFind node.Id ctx.Coeffects.Platform.Bindings.Bindings with
        | Some { Resolved = ResolvedBinding.ExternCall (library, _) } when isLinkedExtern ctx.Coeffects.Platform library ->
            match tryMatchWithDiagnostics pExternCallResolved ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Ok ((ops, result), _) -> foreignResult ctx node ops [] result
            | Result.Error message -> WitnessOutput.error message
        | _ ->
        // Fall back to static patterns (syscalls + static extern calls)
        let combined = pSysWriteIntrinsic <|> pSysReadIntrinsic <|> pSysReadlineIntrinsic <|> pExternCallResolved
        match tryMatch combined ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Some ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
        | None -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "Platform"; Witness = witnessPlatform }
