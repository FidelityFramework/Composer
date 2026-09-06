/// MLIRNanopass - Post-witness declaration collection
///
/// The one MLIR-to-MLIR step the middle end performs: collect the external
/// functions the witnessed ops call and emit their declarations. Nothing
/// semantic happens here. Everything that decides program structure —
/// suspension, closures, net structure, layout, obligations — is settled in
/// the PSG at saturation and witnessed as standard dialects; there is no
/// middle-end lowering to any continuation or net dialect, and no dialect
/// above the witness boundary for one to target (docs/Thin_Middle_End_Design.md).
///
/// PRINCIPLES:
/// 1. Platform-agnostic - no hardcoded backend assumptions
/// 2. Structural - a fold over the op stream, like a PSG nanopass
/// 3. SSA isolation - fresh SSA generation, if any, contained to this layer
module Alex.Pipeline.MLIRNanopass

open Alex.Dialects.Core.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Traversal.TransferTypes

// ═══════════════════════════════════════════════════════════════════════════
// DECLARATION VALIDATION + RELOCATION PASS
// ═══════════════════════════════════════════════════════════════════════════

/// Declaration Validation + Relocation Pass
///
/// Validates that every function call has a matching definition or declaration,
/// then relocates all FuncDecl ops to module top level (deduplicated).
///
/// Patterns that call external functions (write, read, memcpy) emit their own
/// FuncDecl alongside the FuncCall. This pass validates completeness and
/// reorganizes the declarations.
///
/// Hard error on any call to a function with no definition or declaration.
let declarationCollectionPass (operations: MLIROp list) : MLIROp list =

    /// Collect all function/module DEFINITIONS in the module
    let definedFunctions =
        let rec collectDefs (op: MLIROp) =
            match op with
            | MLIROp.FuncOp (FuncOp.FuncDef (name, _, _, _, _)) -> [name]
            | MLIROp.HWOp (HWOp.HWModule (name, _, _, _)) -> [name]
            | _ -> []
        operations |> List.collect collectDefs |> Set.ofList

    /// Collect all FuncDecl names (emitted by patterns for external calls)
    let rec collectDecls (op: MLIROp) : (string * MLIROp) list =
        match op with
        | MLIROp.FuncOp (FuncOp.FuncDecl (name, _, _, _, _)) as decl -> [(name, decl)]
        | MLIROp.FuncOp (FuncOp.FuncDef (_, _, _, body, _)) -> body |> List.collect collectDecls
        | MLIROp.HWOp (HWOp.HWModule (_, _, _, body)) -> body |> List.collect collectDecls
        | MLIROp.SCFOp (SCFOp.If (_, thenOps, elseOps, _)) ->
            let t = thenOps |> List.collect collectDecls
            let e = elseOps |> Option.map (List.collect collectDecls) |> Option.defaultValue []
            t @ e
        | MLIROp.SCFOp (SCFOp.While (condOps, bodyOps)) ->
            (condOps |> List.collect collectDecls) @ (bodyOps |> List.collect collectDecls)
        | MLIROp.SCFOp (SCFOp.For (_, _, _, bodyOps)) -> bodyOps |> List.collect collectDecls
        | MLIROp.Block (_, blockOps) -> blockOps |> List.collect collectDecls
        | MLIROp.Region ops -> ops |> List.collect collectDecls
        | _ -> []

    let allDecls = operations |> List.collect collectDecls
    let declaredNames = allDecls |> List.map fst |> Set.ofList

    /// Combined: all names that have a definition OR declaration
    let knownFunctions = Set.union definedFunctions declaredNames

    /// Recursively collect all function calls
    let rec collectCalls (op: MLIROp) : string list =
        match op with
        | MLIROp.FuncOp (FuncOp.FuncCall (_, name, _, _)) -> [name]
        | MLIROp.FuncOp (FuncOp.FuncDef (_, _, _, body, _)) -> body |> List.collect collectCalls
        | MLIROp.HWOp (HWOp.HWModule (_, _, _, body)) -> body |> List.collect collectCalls
        | MLIROp.SCFOp (SCFOp.If (_, thenOps, elseOps, _)) ->
            let t = thenOps |> List.collect collectCalls
            let e = elseOps |> Option.map (List.collect collectCalls) |> Option.defaultValue []
            t @ e
        | MLIROp.SCFOp (SCFOp.While (condOps, bodyOps)) ->
            (condOps |> List.collect collectCalls) @ (bodyOps |> List.collect collectCalls)
        | MLIROp.SCFOp (SCFOp.For (_, _, _, bodyOps)) -> bodyOps |> List.collect collectCalls
        | MLIROp.Block (_, blockOps) -> blockOps |> List.collect collectCalls
        | MLIROp.Region ops -> ops |> List.collect collectCalls
        | _ -> []

    let calledNames = operations |> List.collect collectCalls |> Set.ofList

    /// Hard error: any call to a function with no definition or declaration
    let undefinedCalls = Set.difference calledNames knownFunctions
    if not (Set.isEmpty undefinedCalls) then
        let names = undefinedCalls |> String.concat ", "
        failwithf "[Alex] ERROR: Calls to undefined functions: %s. All called functions must have a definition or an explicit declaration." names

    /// Deduplicate declarations (patterns may emit the same decl multiple times)
    let uniqueDecls =
        allDecls
        |> List.distinctBy fst
        |> List.map snd

    /// Remove all FuncDecl ops from their original locations (they'll be at module top)
    let rec stripDecls (op: MLIROp) : MLIROp option =
        match op with
        | MLIROp.FuncOp (FuncOp.FuncDecl _) -> None
        | MLIROp.FuncOp (FuncOp.FuncDef (name, args, retTy, body, vis)) ->
            Some (MLIROp.FuncOp (FuncOp.FuncDef (name, args, retTy, body |> List.choose stripDecls, vis)))
        | MLIROp.HWOp (HWOp.HWModule (name, ins, outs, body)) ->
            Some (MLIROp.HWOp (HWOp.HWModule (name, ins, outs, body |> List.choose stripDecls)))
        | MLIROp.SCFOp (SCFOp.If (cond, thenOps, elseOps, result)) ->
            Some (MLIROp.SCFOp (SCFOp.If (cond, thenOps |> List.choose stripDecls, elseOps |> Option.map (List.choose stripDecls), result)))
        | MLIROp.SCFOp (SCFOp.While (condOps, bodyOps)) ->
            Some (MLIROp.SCFOp (SCFOp.While (condOps |> List.choose stripDecls, bodyOps |> List.choose stripDecls)))
        | MLIROp.SCFOp (SCFOp.For (lb, ub, step, bodyOps)) ->
            Some (MLIROp.SCFOp (SCFOp.For (lb, ub, step, bodyOps |> List.choose stripDecls)))
        | MLIROp.Block (label, blockOps) ->
            Some (MLIROp.Block (label, blockOps |> List.choose stripDecls))
        | MLIROp.Region ops ->
            Some (MLIROp.Region (ops |> List.choose stripDecls))
        | _ -> Some op

    let strippedOps = operations |> List.choose stripDecls

    /// Relocated declarations at top, then all other ops
    uniqueDecls @ strippedOps

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS ORCHESTRATION (Apply All Passes)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply MLIR post-witnessing passes
///
/// CURRENT PASS:
/// - Declaration Collection (emit FuncDecl for external functions)
///
/// This is the only MLIR→MLIR transformation in the middle end, and it stays the
/// only one. Suspension (async/seq/receive), closures, and interaction-net
/// structure are settled in the PSG by Baker recipes at saturation and witnessed
/// as standard dialects (func, memref, arith, scf, index). There is no DCont or
/// Inet lowering here and no dialect above the witness boundary for one to target.
/// See docs/Thin_Middle_End_Design.md and docs/Delimited_Continuations_Architecture.md.
let applyPasses (operations: MLIROp list) (platform: PlatformReads) (intermediatesDir: string option) : MLIROp list =
    // Declaration Collection Pass
    //
    // ARCHITECTURAL RATIONALE:
    // Function declarations are MLIR-level structure, not PSG-level semantics.
    // During witnessing, ApplicationWitness emits FuncCall operations with actual types.
    // This pass scans ALL calls, collects unique function signatures, and emits FuncDecl operations.
    //
    // BENEFITS:
    // 1. Deterministic - same calls always produce same declarations (no "first witness wins")
    // 2. Separates concerns - witnessing (codata) vs declaration emission (structural)
    // 3. Codata principle - witnesses return calls, post-pass handles declarations
    // 4. Signature unification - can analyze ALL calls before deciding signature
    let afterDecls = declarationCollectionPass operations

    // Serialize intermediate (if -k flag enabled)
    match intermediatesDir with
    | Some dir ->
        let mlirText = Alex.Dialects.Core.Serialize.moduleToString platform.TargetArch.Pointer "main" afterDecls
        let filePath = System.IO.Path.Combine(dir, "08_after_declaration_collection.mlir")
        System.IO.File.WriteAllText(filePath, mlirText)
        if Clef.Compiler.NativeTypedTree.Infrastructure.PhaseConfig.isVerbose() then
            printfn "[Alex] Wrote nanopass intermediate: 08_after_declaration_collection.mlir"
    | None -> ()

    // Future passes will be composed here:
    // let afterDCont = dcontLoweringPass afterDecls
    // let afterInet = inetLoweringPass afterDCont
    // let afterBackend = backendTargetingPass afterInet

    afterDecls
