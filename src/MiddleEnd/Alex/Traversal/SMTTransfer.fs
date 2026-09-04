/// SMTTransfer: the build-time dispatch -- the graph's obligations as `smt` dialect IR
///
/// Architecturally parallel to XDCTransfer. Both are transfers: pure functions
/// from settled data to text, with no traversal and no decision.
///   - XDCTransfer: the pin-mapping coeffect -> XDC constraints
///   - SMTTransfer: the graph's obligation nodes -> an `smt` verification module
///
/// Obligations are graph citizens: minted into the PSG by the Baker obligation
/// recipes at saturation (CCS Pass 5), each a node in V with a hyperedge in F
/// whose source set is the structure it constrains (C-01 14.5;
/// Obligation_Residency 3). This module reads those records from the graph and
/// transcribes them. Every fact in an ObligationBody was fixed at saturation;
/// nothing here computes.
///
/// One birth, two dispatches. The anchor name is the identity that travels:
/// CCS renders the same records to SMT-LIB at design time (06b); this module
/// renders them to `smt` dialect at build time (09), and mlir-translate
/// --export-smtlib carries the `smt.declare_fun` names through verbatim, so the
/// two dispatches pair one for one (HelloProof proof-trace 03).
///
/// Output: a standalone verification module beside the program (one
/// smt.solver scope per obligation, refutation style: `unsat` on every check
/// means every obligation holds).
module Alex.Traversal.SMTTransfer

open Alex.Dialects.Core.Types
open Alex.Dialects.Core.Serialize
open Clef.Compiler.PSGSaturation.SemanticGraph.Types

/// Transcribe one obligation into an isolated smt.solver scope.
/// SSA numbering is fresh per scope (solver regions are isolated).
let private scope (ob: ObligationInfo) : MLIROp list =
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
        | ObligationBody.StorageReservation (len, storage) ->
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
        | ObligationBody.ViewContainment (view, len, storage) ->
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
        | ObligationBody.NulSentinel lastByte ->
            let b = v ()
            let z = v ()
            let def = v ()
            anchor def
                [ smt (SMTBVConstant (b, int64 lastByte, 8))
                  smt (SMTBVConstant (z, 0L, 8))
                  smt (SMTEq (def, b, z, SMTBV 8)) ]
        | ObligationBody.ConsecutiveLayout (storages, span, capacity) ->
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
            // where a declared space bounds the layout: span <= capacity
            let fitsSSAs, fitsOps =
                match capacity with
                | Some cap ->
                    let c = v ()
                    let f = v ()
                    [f], [ smt (SMTIntConstant (c, cap)); smt (SMTIntCmp (f, SmtLe, spanC, c)) ]
                | None -> [], []
            let spanOps =
                [ smt (SMTIntConstant (lastSz, int64 sizes[n' - 1]))
                  smt (SMTIntAdd (hi, baseSSA (n' - 1), lastSz))
                  smt (SMTIntConstant (spanC, int64 span))
                  smt (SMTIntAdd (lo, baseSSA 0, spanC))
                  smt (SMTEq (spanEq, hi, lo, SMTInt)) ]
                @ allDisjOps
                @ fitsOps
                @ [ smt (SMTAnd (def, [allDisj; spanEq] @ fitsSSAs)) ]
            anchor def (declOps @ nonNeg @ adjacency @ disjoint @ spanOps)
        | ObligationBody.ConcatCopyBound (leftLen, rightLen) ->
            let ll = v ()
            let lr = v ()
            let al = v ()
            let zero = v ()
            let geL = v ()
            let geR = v ()
            let decls =
                [ smt (SMTDeclareFun (ll, "len_l", SMTInt))
                  smt (SMTDeclareFun (lr, "len_r", SMTInt))
                  smt (SMTDeclareFun (al, "alloc", SMTInt))
                  smt (SMTIntConstant (zero, 0L))
                  smt (SMTIntCmp (geL, SmtGe, ll, zero))
                  smt (SMTAssert geL)
                  smt (SMTIntCmp (geR, SmtGe, lr, zero))
                  smt (SMTAssert geR) ]
            // pin an operand to its concrete byte length where the graph has it
            let pin (ssa: SSA) (len: int option) : MLIROp list =
                match len with
                | Some n ->
                    let c = v ()
                    let eq = v ()
                    [ smt (SMTIntConstant (c, int64 n))
                      smt (SMTEq (eq, ssa, c, SMTInt))
                      smt (SMTAssert eq) ]
                | None -> []
            // emission-contract premise: alloc = len_l + len_r
            let sum = v ()
            let aeq = v ()
            let premise =
                [ smt (SMTIntAdd (sum, ll, lr))
                  smt (SMTEq (aeq, al, sum, SMTInt))
                  smt (SMTAssert aeq) ]
            // conclusion: both copy windows lie within the allocation
            let c1 = v ()
            let c2 = v ()
            let def = v ()
            let conclusion =
                [ smt (SMTIntCmp (c1, SmtLe, ll, al))
                  smt (SMTIntCmp (c2, SmtLe, sum, al))
                  smt (SMTAnd (def, [c1; c2])) ]
            anchor def (decls @ pin ll leftLen @ pin lr rightLen @ premise @ conclusion)

        | ObligationBody.CapacityPositive cap ->
            let c = v ()
            let zero = v ()
            let def = v ()
            anchor def
                [ smt (SMTIntConstant (c, cap))
                  smt (SMTIntConstant (zero, 0L))
                  smt (SMTIntCmp (def, SmtGt, c, zero)) ]
        | ObligationBody.CapacityFits (cap, spaceCap) ->
            let c = v ()
            let sc = v ()
            let def = v ()
            anchor def
                [ smt (SMTIntConstant (c, cap))
                  smt (SMTIntConstant (sc, spaceCap))
                  smt (SMTIntCmp (def, SmtLe, c, sc)) ]
        | ObligationBody.InputBufferBound (count, allocation) ->
            let c = v ()
            let a = v ()
            let def = v ()
            anchor def
                [ smt (SMTIntConstant (c, count))
                  smt (SMTIntConstant (a, allocation))
                  smt (SMTIntCmp (def, SmtLe, c, a)) ]
        | ObligationBody.InputCopyBound (cap, bound) ->
            // for all r, 1 <= r <= cap: r - 1 <= bound
            let r = v ()
            let one = v ()
            let capC = v ()
            let geOne = v ()
            let leCap = v ()
            let rm1 = v ()
            let b = v ()
            let def = v ()
            anchor def
                [ smt (SMTDeclareFun (r, "r", SMTInt))
                  smt (SMTIntConstant (one, 1L))
                  smt (SMTIntCmp (geOne, SmtGe, r, one))
                  smt (SMTAssert geOne)
                  smt (SMTIntConstant (capC, cap))
                  smt (SMTIntCmp (leCap, SmtLe, r, capC))
                  smt (SMTAssert leCap)
                  smt (SMTIntSub (rm1, r, one))
                  smt (SMTIntConstant (b, bound))
                  smt (SMTIntCmp (def, SmtLe, rm1, b)) ]

    [ MLIROp.RawMLIR (sprintf "// %s: %s" ob.Id ob.Statement)
      MLIROp.RawMLIR (sprintf "// origin: %s" ob.Source)
      smt (SMTSolver (smt (SMTSetLogic ob.Logic) :: ops @ [ smt SMTCheck ])) ]

/// The verification module from the graph's obligations.
/// Pure function: ObligationInfo list -> MLIR text. The list is
/// ObligationDischarge.ofGraph, the same list the design-time dispatch rendered.
let transfer (obs: ObligationInfo list) : string =
    obs
    |> List.collect scope
    |> moduleToString "obligations"
