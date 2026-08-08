/// SMTTransfer: Parallel transfer from the ProofObligations coeffect to `smt` dialect IR
///
/// Architecturally parallel to MLIRTransfer and XDCTransfer. All observe coeffects:
///   - MLIRTransfer: coeffects → MLIR ops (via witness traversal)
///   - XDCTransfer:  coeffects → XDC text (pure function, no traversal needed)
///   - SMTTransfer:  coeffects → `smt` dialect verification module (pure function)
///
/// The coeffect IS the pre-computed data. This transfer is transcription:
/// every fact in an ObligationBody was computed at analysis time; this module
/// only renders it as IR. Obligation identity (the anchor name) is preserved
/// verbatim via smt.declare_fun name prefixes, which mlir-translate
/// --export-smtlib carries through to the emitted SMT-LIB text.
///
/// Output: a standalone verification module (one smt.solver scope per
/// obligation, refutation style: `unsat` on every check = all hold).
module Alex.Traversal.SMTTransfer

open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open PSGElaboration.ProofObligations

/// Transcribe one obligation into an isolated smt.solver scope.
/// SSA numbering is fresh per scope (solver regions are isolated).
let private scope (ob: Obligation) : MLIROp list =
    let mutable n = -1
    let v () = n <- n + 1; V n
    let smt op = MLIROp.SMTOp op

    // Anchor discipline: %ob names the obligation; assert (ob = definition)
    // and (not ob). The declare_fun name is the identity that travels.
    let anchor (defSSA: SSA) (body: MLIROp list) : MLIROp list =
        let ob' = v ()
        let bind = v ()
        let neg = v ()
        body
        @ [ smt (SMTDeclareFun (ob', ob.Id, SMTBool))
            smt (SMTEq (bind, ob', defSSA, SMTBool))
            smt (SMTAssert bind)
            smt (SMTNot (neg, ob'))
            smt (SMTAssert neg) ]

    let ops =
        match ob.Body with
        | StorageReservation (len, storage) ->
            let s = v ()
            let l = v ()
            let one = v ()
            let sum = v ()
            let def = v ()
            anchor def
                [ smt (SMTIntConstant (s, int64 storage))
                  smt (SMTIntConstant (l, int64 len))
                  smt (SMTIntConstant (one, 1L))
                  smt (SMTIntAdd (sum, l, one))
                  smt (SMTEq (def, s, sum, SMTInt)) ]
        | ViewContainment (view, len, storage) ->
            let vw = v ()
            let l = v ()
            let s = v ()
            let eq = v ()
            let lt = v ()
            let def = v ()
            anchor def
                [ smt (SMTIntConstant (vw, int64 view))
                  smt (SMTIntConstant (l, int64 len))
                  smt (SMTIntConstant (s, int64 storage))
                  smt (SMTEq (eq, vw, l, SMTInt))
                  smt (SMTIntCmp (lt, SmtLt, vw, s))
                  smt (SMTAnd (def, [eq; lt])) ]
        | NulSentinel lastByte ->
            let b = v ()
            let z = v ()
            let def = v ()
            anchor def
                [ smt (SMTBVConstant (b, int64 lastByte, 8))
                  smt (SMTBVConstant (z, 0L, 8))
                  smt (SMTEq (def, b, z, SMTBV 8)) ]
        | ConsecutiveLayout (storages, span) ->
            let sizes = List.toArray storages
            let n' = sizes.Length
            let bases = [ for i in 0 .. n' - 1 -> v (), sprintf "b%d" i ]
            let baseSSA i = fst bases[i]
            let declOps =
                [ for (ssa, name) in bases -> smt (SMTDeclareFun (ssa, name, SMTInt)) ]
            // b0 >= 0
            let zero = v ()
            let nn = v ()
            let nonNeg =
                [ smt (SMTIntConstant (zero, 0L))
                  smt (SMTIntCmp (nn, SmtGe, baseSSA 0, zero))
                  smt (SMTAssert nn) ]
            // adjacency premises: b[i] = b[i-1] + size[i-1]
            let adjacency =
                [ for i in 1 .. n' - 1 do
                    let c = v ()
                    let a = v ()
                    let eq = v ()
                    yield smt (SMTIntConstant (c, int64 sizes[i-1]))
                    yield smt (SMTIntAdd (a, baseSSA (i-1), c))
                    yield smt (SMTEq (eq, baseSSA i, a, SMTInt))
                    yield smt (SMTAssert eq) ]
            // pairwise disjointness: end_i <= b_j OR end_j <= b_i
            let mutable disjSSAs = []
            let disjoint =
                [ for i in 0 .. n' - 1 do
                    for j in i + 1 .. n' - 1 do
                        let si = v ()
                        let ei = v ()
                        let li = v ()
                        let sj = v ()
                        let ej = v ()
                        let lj = v ()
                        let d = v ()
                        disjSSAs <- disjSSAs @ [d]
                        yield smt (SMTIntConstant (si, int64 sizes[i]))
                        yield smt (SMTIntAdd (ei, baseSSA i, si))
                        yield smt (SMTIntCmp (li, SmtLe, ei, baseSSA j))
                        yield smt (SMTIntConstant (sj, int64 sizes[j]))
                        yield smt (SMTIntAdd (ej, baseSSA j, sj))
                        yield smt (SMTIntCmp (lj, SmtLe, ej, baseSSA i))
                        yield smt (SMTOr (d, [li; lj])) ]
            // span: b[n-1] + size[n-1] = b[0] + span
            let lastSz = v ()
            let hi = v ()
            let spanC = v ()
            let lo = v ()
            let spanEq = v ()
            let def = v ()
            // smt.and is variadic with a two-operand minimum: with exactly one
            // disjointness pair (two strings), use that pair's SSA directly
            let allDisj, allDisjOps =
                match disjSSAs with
                | [only] -> only, []
                | _ -> let a = v () in a, [ smt (SMTAnd (a, disjSSAs)) ]
            let spanOps =
                [ smt (SMTIntConstant (lastSz, int64 sizes[n' - 1]))
                  smt (SMTIntAdd (hi, baseSSA (n' - 1), lastSz))
                  smt (SMTIntConstant (spanC, int64 span))
                  smt (SMTIntAdd (lo, baseSSA 0, spanC))
                  smt (SMTEq (spanEq, hi, lo, SMTInt)) ]
                @ allDisjOps
                @ [ smt (SMTAnd (def, [allDisj; spanEq])) ]
            anchor def (declOps @ nonNeg @ adjacency @ disjoint @ spanOps)

    [ MLIROp.RawMLIR (sprintf "// %s: %s" ob.Id ob.Statement)
      MLIROp.RawMLIR (sprintf "// origin: %s" ob.Source)
      smt (SMTSolver (smt (SMTSetLogic ob.Logic) :: ops @ [ smt SMTCheck ])) ]

/// Generate the verification module from the ProofObligations coeffect.
/// Pure function: ObligationSet → MLIR text.
let transfer (obs: ObligationSet) : string =
    obs.Obligations
    |> List.collect scope
    |> moduleToString "obligations"
