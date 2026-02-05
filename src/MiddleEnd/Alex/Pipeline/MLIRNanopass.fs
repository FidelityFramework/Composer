/// MLIRNanopass - Structural MLIR-to-MLIR transformations
///
/// This module establishes the foundation for dual witness infrastructure:
/// - Current: PSG → MLIR (via witnesses)
/// - Future: MLIR → MLIR (via nanopasses) → DCont/Inet dialects
///
/// ARCHITECTURAL PRINCIPLES:
/// 1. Platform-agnostic - no hardcoded backend assumptions
/// 2. Structural transformations - like PSG nanopasses
/// 3. Composable - each pass handles one concern
/// 4. SSA isolation - fresh SSA generation contained to this layer
///
/// LONG-TERM VISION:
/// This is the FIRST component of dual witness infrastructure. Future passes will include:
/// - DCont lowering (sequential/effectful patterns → stack-based async)
/// - Inet lowering (parallel/pure patterns → graph reduction)
/// - Backend targeting (portable dialects → LLVM/SPIR-V/WebAssembly/custom)
/// - Hybrid optimization (mix DCont/Inet based on purity analysis)
module Alex.Pipeline.MLIRNanopass

open Alex.Dialects.Core.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open PSGElaboration.PlatformConfig

// ═══════════════════════════════════════════════════════════════════════════
// DECLARATION COLLECTION PASS (Emit FuncDecl from FuncCall analysis)
// ═══════════════════════════════════════════════════════════════════════════

/// Declaration Collection Pass
///
/// Scans all FuncCall operations and emits unified FuncDecl operations.
/// This eliminates the "first witness wins" coordination during witnessing.
///
/// ARCHITECTURAL RATIONALE:
/// - Function declarations are MLIR-level structure (not PSG semantics)
/// - Witnesses emit calls with actual types (no coordination needed)
/// - This pass analyzes ALL calls before emitting declarations (deterministic)
/// - Signature unification happens here (one place, not scattered across witnesses)
///
/// ALGORITHM:
/// 1. Recursively collect all FuncCall operations
/// 2. Group calls by function name
/// 3. Unify signatures (currently: first signature wins, future: type coercion)
/// 4. Emit FuncDecl operations
/// 5. Remove any existing FuncDecl operations from witnesses (duplicates)
let declarationCollectionPass (operations: MLIROp list) : MLIROp list =

    /// Collect all function DEFINITIONS in the module
    /// These are functions implemented in this module - they need NO declarations
    let definedFunctions =
        operations
        |> List.choose (function
            | MLIROp.FuncOp (FuncOp.FuncDef (name, _, _, _, _)) -> Some name
            | _ -> None)
        |> Set.ofList

    /// Recursively collect all function calls from operations
    let rec collectCalls (op: MLIROp) : (string * MLIRType list * MLIRType) list =
        match op with
        | MLIROp.FuncOp (FuncOp.FuncCall (_, name, args, retTy)) ->
            // Collect this call's signature
            [(name, args |> List.map (fun v -> v.Type), retTy)]

        // Recurse into nested operations
        | MLIROp.FuncOp (FuncOp.FuncDef (_, _, _, body, _)) ->
            body |> List.collect collectCalls

        | MLIROp.SCFOp (SCFOp.If (_, thenOps, elseOps)) ->
            let thenCalls = thenOps |> List.collect collectCalls
            let elseCalls = elseOps |> Option.map (List.collect collectCalls) |> Option.defaultValue []
            thenCalls @ elseCalls

        | MLIROp.SCFOp (SCFOp.While (condOps, bodyOps)) ->
            let condCalls = condOps |> List.collect collectCalls
            let bodyCalls = bodyOps |> List.collect collectCalls
            condCalls @ bodyCalls

        | MLIROp.SCFOp (SCFOp.For (_, _, _, bodyOps)) ->
            bodyOps |> List.collect collectCalls

        | MLIROp.Block (_, blockOps) ->
            blockOps |> List.collect collectCalls

        | MLIROp.Region ops ->
            ops |> List.collect collectCalls

        | _ -> []

    /// Collect all function calls from all operations
    let allCalls = operations |> List.collect collectCalls

    /// Generate declarations ONLY for EXTERNAL functions (not defined in this module)
    /// Internal functions already have FuncDef - declarations would cause redefinition errors
    let declarations =
        allCalls
        |> List.groupBy (fun (name, _, _) -> name)
        |> List.filter (fun (name, _) -> not (Set.contains name definedFunctions))  // ONLY external functions
        |> List.map (fun (name, signatures) ->
            // Pick first signature (deterministic due to post-order)
            let (_, paramTypes, retType) = signatures.Head
            MLIROp.FuncOp (FuncOp.FuncDecl (name, paramTypes, retType, FuncVisibility.Private)))
    
    /// Remove any FuncDecl operations emitted by witnesses (now duplicates)
    let withoutWitnessDecls =
        operations
        |> List.filter (function
            | MLIROp.FuncOp (FuncOp.FuncDecl _) -> false  // Remove witness-emitted declarations
            | _ -> true)

    // Emit only EXTERNAL function declarations (functions called but not defined in this module)
    // Internal functions (with FuncDef) need NO declarations - they're already defined
    if List.isEmpty declarations then
        withoutWitnessDecls
    else
        printfn "[Alex] Declaration collection: Emitted %d external function declarations" (List.length declarations)
        declarations @ withoutWitnessDecls

// ═══════════════════════════════════════════════════════════════════════════
// NANOPASS ORCHESTRATION (Apply All Passes)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply MLIR post-witnessing passes
///
/// CURRENT PASS:
/// - Declaration Collection (emit FuncDecl for external functions)
///
/// FUTURE PASSES (aligned with DCont/Inet Duality vision):
/// - DCont Lowering (async {} → dcont dialect, stack-based continuations)
/// - Inet Lowering (query {} → inet dialect, parallel graph reduction)
/// - Hybrid Optimization (mix DCont/Inet based on purity analysis)
/// - Backend Targeting (portable dialects → LLVM/SPIR-V/WebAssembly/custom)
///
/// ARCHITECTURAL NOTE:
/// This is the integration point for eventual TableGen-based transformations.
/// Future vision: Generate TableGen in MiddleEnd, use it to transform dialects.
let applyPasses (operations: MLIROp list) (platform: PlatformResolutionResult) (intermediatesDir: string option) : MLIROp list =
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
        let mlirText = Alex.Dialects.Core.Serialize.moduleToString "main" afterDecls
        let filePath = System.IO.Path.Combine(dir, "08_after_declaration_collection.mlir")
        System.IO.File.WriteAllText(filePath, mlirText)
        printfn "[Alex] Wrote nanopass intermediate: 08_after_declaration_collection.mlir"
    | None -> ()

    // Future passes will be composed here:
    // let afterDCont = dcontLoweringPass afterDecls
    // let afterInet = inetLoweringPass afterDCont
    // let afterBackend = backendTargetingPass afterInet

    afterDecls
