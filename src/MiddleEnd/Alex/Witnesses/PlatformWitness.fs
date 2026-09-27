/// PlatformWitness - Witness platform operations to MLIR via XParsec
///
/// The settled platform binding (Platform.Bindings) selects the owning pattern;
/// a Sys intrinsic application selects its per-operation parser. Each owning
/// pattern either witnesses the call or reports the premise it lacks.
///
/// NANOPASS: Handles Sys.write, Sys.read, Sys.readline intrinsic applications
/// and [<FidelityExtern>] resolved calls (ExternCall in coeffects).
///
/// STATIC vs DYNAMIC (Mar 2026):
///   Static externs (linked library) use pExternCallResolved (func.call with extern decl).
///   Dynamic externs (any other library) use pDynamicExternCallResolved (dlopen/dlsym/call_indirect).
///   Dynamic externs require TopLevelOps for string constant globals (library path, symbol name).
module Alex.Witnesses.PlatformWitness

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Alex.Traversal.TransferTypes
open Alex.Traversal.NanopassArchitecture
open Alex.XParsec.PSGCombinators
open Alex.Patterns.PlatformPatterns

let private foreignResult (ctx: WitnessContext) (node: SemanticNode) ops topLevel result =
    match result with
    | TRValue value ->
        let meets, ssa, ty = adaptOperand ctx.Coeffects ctx.Graph node.Id node.Id value.SSA value.Type
        { InlineOps = ops @ meets; TopLevelOps = topLevel; Result = TRValue { SSA = ssa; Type = ty } }
    | _ -> { InlineOps = ops; TopLevelOps = topLevel; Result = result }

let private witnessPlatform (ctx: WitnessContext) (node: SemanticNode) : WitnessOutput =
    // The settled platform binding selects the owning foreign-call pattern.
    // A linked library is a direct symbol call; any other extern library is a
    // dynamic lookup. Each pattern either witnesses the call or reports the
    // premise it lacks; neither refusal is discarded as a skip.
    match Map.tryFind node.Id ctx.Coeffects.Platform.Bindings.Bindings with
    | Some { Resolved = ResolvedBinding.ExternCall (library, _) } when isLinkedExtern ctx.Coeffects.Platform library ->
        match tryMatchWithDiagnostics pExternCallResolved ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Ok ((ops, result), _) -> foreignResult ctx node ops [] result
        | Result.Error message -> WitnessOutput.error message
    | Some { Resolved = ResolvedBinding.ExternCall (library, symbol) } ->
        // Dynamic externs return pending globals that need TopLevelOps emission.
        match tryMatchWithDiagnostics pDynamicExternCallResolved ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
        | Ok ((inlineOps, pendingGlobals, result), _) ->
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
        | Result.Error message ->
            WitnessOutput.errorDiag (
                Diagnostic.error (Some node.Id) (Some "Platform") (Some "dynamic extern")
                    $"CCS source checking did not settle a native function-address ABI for dynamic extern '{library}'::'{symbol}' at node {NodeId.value node.Id}: {message}")
    | _ ->
    // A Sys intrinsic application belongs to this witness. Its own pattern either
    // witnesses the settled call or reports the premise it lacks; the reason is
    // never discarded as a skip.
    let sysOperation =
        match node.Kind with
        | SemanticKind.Application (callee, _) ->
            match ctx.Graph.Nodes.TryFind callee with
            | Some { Kind = SemanticKind.Intrinsic info } when info.Module = IntrinsicModule.Sys -> Some info.Operation
            | _ -> None
        | _ -> None
    match sysOperation with
    | Some operation ->
        let pattern =
            match operation with
            | "write" -> Some pSysWriteIntrinsic
            | "read" -> Some pSysReadIntrinsic
            | "readline" -> Some pSysReadlineIntrinsic
            | _ -> None
        match pattern with
        | Some pattern ->
            match tryMatchWithDiagnostics pattern ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
            | Ok ((ops, result), _) -> { InlineOps = ops; TopLevelOps = []; Result = result }
            | Result.Error message -> WitnessOutput.error $"Sys.{operation}: {message}"
        | None -> WitnessOutput.error $"Sys.{operation} has no platform witness pattern"
    | None -> WitnessOutput.skip

let nanopass : Nanopass = { Name = "Platform"; Witness = witnessPlatform }
