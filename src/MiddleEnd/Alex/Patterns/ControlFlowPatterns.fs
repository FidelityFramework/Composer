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

// ═══════════════════════════════════════════════════════════
// MATCH ELIMINATION (catamorphism elision)
// ═══════════════════════════════════════════════════════════

/// Extract the tag index from a CaseArm pattern.
/// Union patterns carry tagIndex directly; others default to arm position.
let private getArmTagIndex (armIndex: int) (pattern: Clef.Compiler.PSGSaturation.SemanticGraph.Types.Pattern) : int =
    match pattern with
    | Clef.Compiler.PSGSaturation.SemanticGraph.Types.Pattern.Union (_, tagIndex, _, _) -> tagIndex
    | _ -> armIndex  // Const/Wildcard/Var patterns use positional index

/// Get the DU union type from a CaseArm pattern (for tag extraction).
let private getScrutineeUnionType (arms: Clef.Compiler.PSGSaturation.SemanticGraph.Types.CaseArm list) : NativeType option =
    arms |> List.tryPick (fun arm ->
        match arm.Pattern with
        | Clef.Compiler.PSGSaturation.SemanticGraph.Types.Pattern.Union (_, _, _, unionType) -> Some unionType
        | _ -> None)

/// Build match elimination — tag extract + nested scf.if chain (CPU)
/// or all arms inline + comb.mux chain (FPGA, future).
///
/// Tag extraction and comparisons are emitted HERE at elision time, not in Baker.
/// Baker preserved the structural fold; this pattern decides how to realize it.
///
/// Parameters:
///   scrutineeSSA - SSA of the matched value
///   scrutineeType - MLIR type of the matched value
///   scrutineeNodeId - PSG node ID of the scrutinee (for DUGetTag SSAs)
///   arms - list of (armOps, armBodyValueNodeId, pattern) per arm
///   result - Some (resultSSA, resultType) if expression-valued, None if void
///   nodeId - the CaseElimination node's ID (for SSA allocation)
let pBuildMatchElimination
    (scrutineeSSA: SSA) (scrutineeType: MLIRType) (scrutineeNodeId: NodeId)
    (arms: (MLIROp list * NodeId * Clef.Compiler.PSGSaturation.SemanticGraph.Types.CaseArm) list)
    (result: (SSA * MLIRType) option)
    (nodeId: NodeId)
    : PSGParser<MLIROp list * TransferResult> =
    parser {
        let! targetPlatform = getTargetPlatform

        match targetPlatform with
        | FPGA ->
            return! fail (Message "FPGA match elimination requires comb.mux chain (future)")

        | _ ->
            let numArms = List.length arms

            // Detect record match (all arms are Record/Wildcard patterns — no DU tag)
            let isRecordMatch =
                arms |> List.forall (fun (_, _, arm) ->
                    match arm.Pattern with
                    | Clef.Compiler.PSGSaturation.SemanticGraph.Types.Pattern.Record _ -> true
                    | Clef.Compiler.PSGSaturation.SemanticGraph.Types.Pattern.Wildcard -> true
                    | _ -> false)

            if isRecordMatch then
                // ── Record match path: no DU tag extraction ──
                // Selection is by guard evaluation (or passthrough for single arm)

                // Recall all arm value SSAs upfront
                let! armValueSSAs =
                    match result with
                    | Some _ ->
                        let rec recallAll idx acc =
                            if idx >= numArms then preturn (List.rev acc)
                            else
                                let (_, armValueNodeId, _) = arms.[idx]
                                parser {
                                    let! (armSSA, _) = pRecallNode armValueNodeId
                                    return! recallAll (idx + 1) (armSSA :: acc)
                                }
                        recallAll 0 []
                    | None -> preturn []

                if numArms = 1 then
                    // Single arm: passthrough — just emit arm ops and use arm value directly
                    let (armOps, _, _) = arms.[0]
                    match result with
                    | Some (_, resultType) ->
                        let armSSA = armValueSSAs.[0]
                        return (armOps, TRValue { SSA = armSSA; Type = resultType })
                    | None ->
                        return (armOps, TRVoid)
                else
                    // Multi-arm record match with guards: nested scf.if chain by guard evaluation
                    // Guards were already walked by MatchWitness — recall their SSAs from accumulator
                    let! graph = getGraph

                    // Pre-recall guard SSAs for non-default arms
                    let! guardSSAs =
                        let rec recallGuards idx acc =
                            if idx >= numArms - 1 then preturn (List.rev acc)
                            else
                                let (_, _, arm) = arms.[idx]
                                match arm.Guard with
                                | Some guardId ->
                                    parser {
                                        let guardValueNodeId = findLastValueNode guardId graph
                                        let! (guardSSA, _) = pRecallNode guardValueNodeId
                                        return! recallGuards (idx + 1) (guardSSA :: acc)
                                    }
                                | None ->
                                    // No guard on non-last arm — shouldn't happen but handle gracefully
                                    recallGuards (idx + 1) (V -1 :: acc)
                        recallGuards 0 []

                    // Build nested scf.if from inside-out using recursive builder
                    // Last arm is the exhaustive default (else body)
                    let (lastArmOps, _, _) = arms.[numArms - 1]
                    let lastArmElseOps =
                        match result with
                        | Some (_, resultType) ->
                            let lastSSA = armValueSSAs.[numArms - 1]
                            lastArmOps @ [MLIROp.SCFOp (SCFOp.Yield [(lastSSA, resultType)])]
                        | None ->
                            lastArmOps @ [MLIROp.SCFOp (SCFOp.Yield [])]

                    // Recursive builder: returns ops for else region content (WITH trailing yield)
                    let rec buildElseContent armIndex =
                        if armIndex >= numArms - 1 then
                            lastArmElseOps  // Already has yield
                        else
                            let (armOps, _, _) = arms.[armIndex]
                            let condSSA = guardSSAs.[armIndex]
                            let thenOps =
                                match result with
                                | Some (_, resultType) ->
                                    armOps @ [MLIROp.SCFOp (SCFOp.Yield [(armValueSSAs.[armIndex], resultType)])]
                                | None ->
                                    armOps @ [MLIROp.SCFOp (SCFOp.Yield [])]
                            let innerElse = buildElseContent (armIndex + 1)
                            let ifOp =
                                match result with
                                | Some (resultSSA, resultType) ->
                                    MLIROp.SCFOp (SCFOp.If (condSSA, thenOps, Some innerElse, Some (resultSSA, resultType)))
                                | None ->
                                    MLIROp.SCFOp (SCFOp.If (condSSA, thenOps, Some innerElse, None))
                            // Append yield to propagate inner scf.if result in the else region
                            match result with
                            | Some (resultSSA, resultType) ->
                                [ifOp; MLIROp.SCFOp (SCFOp.Yield [(resultSSA, resultType)])]
                            | None ->
                                [ifOp; MLIROp.SCFOp (SCFOp.Yield [])]

                    // Build outermost if (no trailing yield — this is top-level)
                    let (firstArmOps, _, _) = arms.[0]
                    let firstCondSSA = guardSSAs.[0]
                    let firstThenOps =
                        match result with
                        | Some (_, resultType) ->
                            firstArmOps @ [MLIROp.SCFOp (SCFOp.Yield [(armValueSSAs.[0], resultType)])]
                        | None ->
                            firstArmOps @ [MLIROp.SCFOp (SCFOp.Yield [])]
                    let elseContent = buildElseContent 1
                    let outerIfOp =
                        match result with
                        | Some (resultSSA, resultType) ->
                            MLIROp.SCFOp (SCFOp.If (firstCondSSA, firstThenOps, Some elseContent, Some (resultSSA, resultType)))
                        | None ->
                            MLIROp.SCFOp (SCFOp.If (firstCondSSA, firstThenOps, Some elseContent, None))

                    match result with
                    | Some (resultSSA, resultType) ->
                        return ([outerIfOp], TRValue { SSA = resultSSA; Type = resultType })
                    | None ->
                        return ([outerIfOp], TRVoid)
            else
                // ── DU match path: DUGetTag + nested scf.if chain ──

                // Step 1: Extract tag from scrutinee
                let! allSSAs = getNodeSSAs nodeId
                let tagTy = TInt I8

                // Index 0 is reserved for the result SSA — tag extraction starts at index 1
                let tagExtractOps, tagSSA, tagExtractEnd =
                    match scrutineeType with
                    | TIndex ->
                        let indexZeroSSA = allSSAs.[1]
                        let tagSSA = allSSAs.[2]
                        let memrefI8Ty = TMemRef (TInt I8)
                        let indexZeroOp = MLIROp.ArithOp (ArithOp.ConstI (indexZeroSSA, 0L, TIndex))
                        let loadOp = MLIROp.MemRefOp (MemRefOp.Load (tagSSA, scrutineeSSA, [indexZeroSSA], tagTy, memrefI8Ty))
                        [indexZeroOp; loadOp], tagSSA, 3
                    | _ ->
                        let castSSA = allSSAs.[1]
                        let zeroSSA = allSSAs.[2]
                        let tagSSA = allSSAs.[3]
                        let memrefI8Ty = TMemRef (TInt I8)
                        let castOp = MLIROp.MemRefOp (MemRefOp.ReinterpretCast (castSSA, scrutineeSSA, 0, scrutineeType, memrefI8Ty))
                        let zeroOp = MLIROp.ArithOp (ArithOp.ConstI (zeroSSA, 0L, TIndex))
                        let loadOp = MLIROp.MemRefOp (MemRefOp.Load (tagSSA, castSSA, [zeroSSA], tagTy, memrefI8Ty))
                        [castOp; zeroOp; loadOp], tagSSA, 4

                // Step 2: Recall all arm value SSAs upfront
                let! armValueSSAs =
                    match result with
                    | Some _ ->
                        let rec recallAll idx acc =
                            if idx >= numArms then preturn (List.rev acc)
                            else
                                let (_, armValueNodeId, _) = arms.[idx]
                                parser {
                                    let! (armSSA, _) = pRecallNode armValueNodeId
                                    return! recallAll (idx + 1) (armSSA :: acc)
                                }
                        recallAll 0 []
                    | None -> preturn []

                // Step 3: Build nested scf.if chain from inside-out
                let mutable ssaOffset = tagExtractEnd
                let (lastArmOps, _, _) = arms.[numArms - 1]

                let lastArmElseOps =
                    match result with
                    | Some (_, resultType) ->
                        let lastSSA = armValueSSAs.[numArms - 1]
                        lastArmOps @ [MLIROp.SCFOp (SCFOp.Yield [(lastSSA, resultType)])]
                    | None ->
                        lastArmOps @ [MLIROp.SCFOp (SCFOp.Yield [])]

                let outerOps =
                    List.foldBack (fun i currentElseOps ->
                        let (armOps, _, arm) = arms.[i]
                        let tagIndex = getArmTagIndex i arm.Pattern

                        let tagLitSSA = allSSAs.[ssaOffset]
                        let cmpSSA = allSSAs.[ssaOffset + 1]
                        ssaOffset <- ssaOffset + 2

                        let tagLitOp = MLIROp.ArithOp (ArithOp.ConstI (tagLitSSA, int64 tagIndex, tagTy))
                        let cmpOp = MLIROp.ArithOp (ArithOp.CmpI (cmpSSA, ICmpPred.Eq, tagSSA, tagLitSSA, tagTy))

                        let thenOps =
                            match result with
                            | Some (_, resultType) ->
                                let armSSA = armValueSSAs.[i]
                                armOps @ [MLIROp.SCFOp (SCFOp.Yield [(armSSA, resultType)])]
                            | None ->
                                armOps @ [MLIROp.SCFOp (SCFOp.Yield [])]

                        match result with
                        | Some (resultSSA, resultType) ->
                            let ifOp = MLIROp.SCFOp (SCFOp.If (cmpSSA, thenOps, Some currentElseOps, Some (resultSSA, resultType)))
                            [tagLitOp; cmpOp; ifOp]
                        | None ->
                            let ifOp = MLIROp.SCFOp (SCFOp.If (cmpSSA, thenOps, Some currentElseOps, None))
                            [tagLitOp; cmpOp; ifOp]
                    ) [0 .. numArms - 2] lastArmElseOps

                let allOps = tagExtractOps @ outerOps

                match result with
                | Some (resultSSA, resultType) ->
                    return (allOps, TRValue { SSA = resultSSA; Type = resultType })
                | None ->
                    return (allOps, TRVoid)
    }
