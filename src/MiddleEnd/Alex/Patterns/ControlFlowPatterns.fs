/// ControlFlowPatterns - Structured control flow constructions
///
/// PUBLIC: Witnesses use these to emit control flow operations (If, While, For, Switch).
/// All control flow constructions compose SCFElements and CFElements.
module Alex.Patterns.ControlFlowPatterns

open XParsec
open XParsec.Parsers
open XParsec.Combinators
open Alex.XParsec.PSGCombinators
open Alex.Dialects.Core.Types
open Alex.Traversal.TransferTypes
open Alex.Elements.SCFElements  // pSCFIf, pSCFWhile, pSCFFor
open Alex.Elements.CFElements   // pSwitch
open Alex.Elements.CombElements // pCombMux (FPGA combinational mux)
open Alex.CodeGeneration.TypeMapping
open Clef.Compiler.NativeTypedTree.NativeTypes  // NodeId
open Core.Types.Dialects                        // TargetPlatform (codata-dependent elision)

// ═══════════════════════════════════════════════════════════
// XPARSEC HELPERS
// ═══════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════
// SEQ FOREACH LOOP
// ═══════════════════════════════════════════════════════════

/// ForEach loop over seq (MoveNext-based iteration)
/// MoveNext should: extract code_ptr[2], alloca seq, store seq, call code_ptr(seq_ptr) -> i1
let pBuildForEachLoop (collectionSSA: SSA) (bodyOps: MLIROp list)
                      (arch: Architecture)
                      : PSGParser<MLIROp list * TransferResult> =
    parser {
        // ForEach is a while loop structure:
        // 1. Extract code_ptr from seq struct at [2]
        // 2. Alloca space for seq, store seq to get pointer
        // 3. Call code_ptr(seq_ptr) -> i1 (returns true if has next)
        // 4. If true, extract current element, execute body, loop
        // 5. If false, exit loop

        // Implementation: Use SCF.While with:
        //   - Condition region: call MoveNext, extract result
        //   - Body region: extract current, execute bodyOps, yield
        return! fail (Message "ForEach MoveNext implementation gap - needs: extract code_ptr[2], alloca/store seq, indirect call")
    }

// ═══════════════════════════════════════════════════════════
// STRUCTURED CONTROL FLOW (SCF)
// ═══════════════════════════════════════════════════════════

/// If/then/else via SCF.If (void — no result value)
let pBuildIfThenElse (cond: SSA) (thenOps: MLIROp list) (elseOps: MLIROp list option) : PSGParser<MLIROp list> =
    parser {
        let! ifOp = pSCFIf cond thenOps elseOps None
        return [ifOp]
    }

/// Expression-valued if/then/else via SCF.If — yields a result from branches
let pBuildIfThenElseWithResult (cond: SSA) (thenOps: MLIROp list) (elseOps: MLIROp list option)
                               (resultSSA: SSA) (resultType: MLIRType)
                               : PSGParser<MLIROp list> =
    parser {
        let! ifOp = pSCFIf cond thenOps elseOps (Some (resultSSA, resultType))
        return [ifOp]
    }

/// FPGA combinational mux: if/then/else elides to comb.mux
/// PULL model: receives node IDs, recalls branch result SSAs from accumulator
let pBuildCombMux (cond: SSA) (thenResultNodeId: NodeId) (elseResultNodeId: NodeId)
                  (resultSSA: SSA) (resultType: MLIRType)
                  : PSGParser<MLIROp list> =
    parser {
        let! (thenSSA, _) = pRecallNode thenResultNodeId
        let! (elseSSA, _) = pRecallNode elseResultNodeId
        let! muxOp = pCombMux resultSSA cond thenSSA elseSSA resultType
        return [muxOp]
    }

/// Unified conditional elision — the TargetPlatform coeffect determines the MLIR residual.
/// CPU: scf.if with nested regions containing branch ops + scf.yield terminators
/// FPGA: branch ops flattened inline + comb.mux selecting between result SSAs
///
/// The witness ALWAYS scope-isolates branches (collecting ops). This pattern decides
/// what to do with those collected ops based on the observed coeffect:
///   - CPU: wrap in scf.if regions (nested structure)
///   - FPGA: return ops inline (flat), append comb.mux
///
/// `result`: Some (resultSSA, resultType) for expression-valued, None for void
let pBuildConditional (condSSA: SSA)
                      (thenOps: MLIROp list) (elseOps: MLIROp list option)
                      (thenValueNodeId: NodeId) (elseValueNodeIdOpt: NodeId option)
                      (result: (SSA * MLIRType) option)
                      : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! targetPlatform = getTargetPlatform
        match targetPlatform, result with
        // ─── FPGA expression-valued: inline ops + comb.mux ───
        | FPGA, Some (resultSSA, resultType) ->
            match elseValueNodeIdOpt with
            | Some elseValueNodeId ->
                let! (thenSSA, _) = pRecallNode thenValueNodeId
                let! (elseSSA, _) = pRecallNode elseValueNodeId
                let! muxOp = pCombMux resultSSA condSSA thenSSA elseSSA resultType
                let allOps = thenOps @ (elseOps |> Option.defaultValue []) @ [muxOp]
                return (allOps, TRValue { SSA = resultSSA; Type = resultType })
            | None ->
                return! fail (Message "FPGA comb.mux requires both branches")

        // ─── FPGA void: not supported (hardware has no side effects without state) ───
        | FPGA, None ->
            return! fail (Message "FPGA: void conditional requires state (seq.compreg)")

        // ─── CPU expression-valued: scf.if with yield terminators ───
        | _, Some (resultSSA, resultType) ->
            let! (thenSSA, _) = pRecallNode thenValueNodeId
            let thenYield = MLIROp.SCFOp (SCFOp.Yield [(thenSSA, resultType)])
            let thenOpsWithYield = thenOps @ [thenYield]
            match elseValueNodeIdOpt with
            | Some elseValueNodeId ->
                let! (elseSSA, _) = pRecallNode elseValueNodeId
                let elseYield = MLIROp.SCFOp (SCFOp.Yield [(elseSSA, resultType)])
                let elseOpsWithYield = elseOps |> Option.map (fun ops -> ops @ [elseYield])
                let! ifOp = pSCFIf condSSA thenOpsWithYield elseOpsWithYield (Some (resultSSA, resultType))
                return ([ifOp], TRValue { SSA = resultSSA; Type = resultType })
            | None ->
                return! fail (Message "Expression-valued if requires else branch")

        // ─── CPU void: scf.if with empty yield terminators ───
        | _, None ->
            let yieldOp = MLIROp.SCFOp (SCFOp.Yield [])
            let thenOpsWithYield = thenOps @ [yieldOp]
            let elseOpsWithYield = elseOps |> Option.map (fun ops -> ops @ [yieldOp])
            let! ifOp = pSCFIf condSSA thenOpsWithYield elseOpsWithYield None
            return ([ifOp], TRVoid)
    }

/// While loop via SCF.While
let pBuildWhileLoop (condOps: MLIROp list) (bodyOps: MLIROp list) : PSGParser<MLIROp list> =
    parser {
        let! whileOp = pSCFWhile condOps bodyOps
        return [whileOp]
    }

/// For loop via SCF.For
let pBuildForLoop (lower: SSA) (upper: SSA) (step: SSA) (bodyOps: MLIROp list) : PSGParser<MLIROp list> =
    parser {
        let! forOp = pSCFFor lower upper step bodyOps
        return [forOp]
    }

// ═══════════════════════════════════════════════════════════
// CONTROL FLOW (CF)
// ═══════════════════════════════════════════════════════════

/// Switch statement via CF.Switch
let pSwitch (flag: SSA) (flagTy: MLIRType) (defaultOps: MLIROp list)
            (cases: (int64 * MLIROp list) list) : PSGParser<MLIROp list> =
    parser {
        // Convert case ops to block refs (would need actual block construction)
        // For now, placeholder structure
        let defaultBlock = BlockRef "default"
        let caseBlocks = cases |> List.map (fun (value, _) ->
            (value, BlockRef $"case_{value}", []))

        let! switchOp = Alex.Elements.CFElements.pSwitch flag flagTy defaultBlock [] caseBlocks
        return [switchOp]
    }
