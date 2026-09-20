# The Thin Middle End

> **Doctrine: MLIR is kept as thin as possible in the middle end.**
> Judgments, discharge, and form selection live above the witness boundary, over the saturated graph.
> Below it, MLIR transliterates per target. No semantic dialect crosses the line.
> Companions: [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md) and [C-01 PRD](./PRDs/C-01-Closures.md) Section 14.

## 1. The Doctrine and the Information Argument

The doctrine is stated once, as doctrine: make MLIR as thin as possible in the middle end. Lifting abstraction up into the PSG preserves information the language already has. Pushing lowering down into the backend legs leaves MLIR doing transliteration, which is the work it does well. Between the two is the witness boundary of Section 2, and nothing semantic crosses it as a dialect.

The argument is about information. Selecting a representation form, for a closure, a continuation frame, or a union payload, requires liveness, escape class, extent, and the target profile. All four exist in the saturated graph, computed by CCS and Baker before witnessing. An op carries its types and operands. The liveness that selected the form, the escape class that placed it, and the extent that bounded it stay behind in the graph. A semantic dialect in the middle end therefore forces every downstream pass to re-derive upstream knowledge from the op stream. That re-derivation is the reconstruction Appel identified when he showed that SSA is functional programming: analyses recovering, at cost and approximately, structure the front end held exactly. This pipeline exists to keep the exact structure and skip the recovery.

Above the boundary, each obligation discharges over the graph's own literals (C-01 Section 14.4 for closures, [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md) Section 6 for suspension), form selection reads full information, and witnessing emits the settled result. Below it, each backend leg transliterates the witnessed form into its own idiom. The middle end never holds semantics in ops, because the graph already holds the semantics.

Stochastic programming hosted inside a general-purpose scripting surface has shown the reconstruction cost at language scale. Eric Lippert's account of the Facebook stochastic-programming work, in a January 2026 Hanselminutes interview, describes language tools that represented stochastic quantities with ordinary program arithmetic and updated beliefs against billions of observations. The account closes with the entire division laid off, and the tooling, with no standing apart from its host, was retired with it. The position we take from the episode: a stochastic structure hosted on a scripting surface rests on the halo of the C and C++ beneath it, defensive engineering its author never sees. Geometric algebra is reachable by lowering from proper elaboration in a top-down language structure. From an imperative substrate built upward, it is out of reach. This pipeline lowers from the elaborated side: the probabilistic fragment sits at its tier, and the Bayesian machinery of the admitted "Adaptive Domain Models" paper, closed-form posterior updates, is grounded in graph structure with provable integrity.

## 2. The Witness Boundary as the Formal Line

The five-dialect list below is the baseline, not a ceiling on Alex's expressive
capacity. [M-01](PRDs/M-01-DialectAdmission.md) applies the existing admission
register discipline to complete Elements/Patterns/Witnesses and the information
they carry to each backend. The standard's
[admission requirements](../../clef-lang-spec/spec/backend-lowering-architecture.md#211-operation-and-pathway-admission)
govern each extension. Candidate vocabulary alone does not enable operations.
All semantic decisions remain Baker-owned; the receiving contract covers both
the expression and the correlated graph facts/proof identities it needs.
Alex is target-aware: it selects an admitted Pattern/Witness using the settled
platform/backend facts. One profile may require `scf`, while another warrants
`cf`. Numeric selection and arithmetic construction likewise govern the admitted
`arith`/`math` forms. Portable vocabulary does not require identical IR for every
target. M-01 records the receiving, information-preservation and testing duties.

One normative half of the boundary is in the spec: `clef-lang-spec/spec/backend-lowering-architecture.md` requires that the middle end emit only portable dialects and treats an `llvm.*` op in the middle end as a category error, because a target commitment is lossy and forecloses every other leg.

This document states the second half, the hard-stop rule in full: **no llvm dialect, and no semantic dialect, in the MLIR witnessed out of the PSG.** The standard dialects, `func`, `memref`, `arith`, `scf`, `index`, are the baseline vocabulary of the witnessed region, extended under M-01's operation/profile admission contract. The first ban reserves target-specific encoding for the backend realization stage; it does not prohibit target-aware witnessing. The second keeps the middle end semantics-free in a precise sense: the semantics ride in the graph, the judgments discharge over its literals, and what reaches MLIR is the settled decomposition. A dialect that encoded closure-ness or continuation-ness into the witnessed region would carry semantics past the point of their discharge.

## 3. The Evidence

The doctrine has a record. Each time a semantic dialect has been proposed or built for this pipeline, the standard-dialect decomposition has been found sufficient, and the record now has four entries. The cast/plugin entries below are historical episodes: the current standard prohibits unrealized casts above the boundary and requires no closure cast-resolution plugin. They do not authorize reintroducing those mechanisms.

**The closure dialect dissolved.** C-01 Section 14.3 demonstrates, primitive by primitive, that byte frames, static views, function values, and `func.call_indirect` express every interior closure form. MLIR's own documentation carries the decisive precondition: `memref.view` requires a 1-D i8 source with identity layout and a byte-shift operand. That precondition is exactly the saturated environment's shape, byte extent literal in the type, offsets literal as `arith.constant` shifts. The standard dialect already contained the elaborated form.

**The DCont dialect dissolved.** [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md) retires the op rendering of shift and reset: the published sketch is ill-formed as MLIR, the delimiter is graph structure a builder extent defines, and the witnessed form is a discriminant, a byte frame, and `scf.index_switch`. CMU's removal of its own WAMI dialect on 2026-02-27, unused, in favor of a Coro dialect, is empirical confirmation from the strongest external attempt.

**The anonymous cast resolved into a named pair.** The interim encoding carried function-address and environment conversions as `builtin.unrealized_conversion_cast`, an anonymous op with its meaning in a comment. That episode resolved into the named materialize and scatter pair governed by a round-trip law: scatter after materialize is the identity on the value it carried. The law is checkable. The anonymous cast was not. The episode is the doctrine in miniature: the fix was a name and a law at the boundary, and at no point a dialect above it.

**The historical tail plugins.** `flat-closure-lowering` and `reconcile-ffi-externs` (the mlir-plugins repository) were described as interim cast/fence reconciliation at the tail of the LLVM leg. Their counters served as a mechanism-side audit scaffold. The current closure contract in the standard supersedes the deferred-cast design; per-obligation correspondence remains the preservation requirement. This history establishes neither current plugin use nor permission to restore cast-based closure witnessing.

## 4. Where a Dialect Is Justified

Below the boundary, a dialect is the right tool exactly when it expresses a target's native structure:

- **A runtime that hosts continuations.** WebAssembly stack switching, when its proposal matures, receives a true continuation expression atomized from the PSG into low-level MLIR ([Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md) Section 8.5).
- **Hardware description.** The CIRCT dialects on the FPGA leg.
- **NPU dataflow.** The MLIR-AIE dialects on the NPU leg.

In each case the dialect is transliteration vocabulary for what the target natively is. In this pipeline a dialect never carries language semantics the graph already holds. The test is directional: a dialect qualifies by expressing the target upward, never by expressing Clef downward.

## 5. Consequences

**The count is zero.** The number of semantic dialects in the witnessed region is zero, and it stays zero. This is a checkable property of the pipeline: the witness layer emits from a fixed vocabulary, and additions to that vocabulary are additions to this document first.

**Every dialect question routes through the recipe-first test.** A proposal to build a dialect answers one question before any other: can fan-out and fold-in, plus the standard primitives, express it? In every case to date the answer has been yes: closures, lazy, seq, unions, and now suspension. The burden of proof rests on the proposed dialect.

**The differentiation claim.** We know of no other commercial pipeline that composes the three properties this boundary enforces: the abstraction tower held above the IR, decidable discharge over the graph's own literals, and witnessed decomposition into standard dialects. Individually each exists somewhere in the literature. Composed, they are the pipeline's identity, and the thin middle end is the discipline that keeps them composed.

## 6. Cross-References

- [Closure_Nanopass_Architecture.md](./Closure_Nanopass_Architecture.md) Section 4: the canonical finiteness lemma, and the quantifier-free discharge the graph-side judgments rest on.
- [C-01 PRD](./PRDs/C-01-Closures.md) Section 14: the form family, the standard-dialect correspondence table, and the per-obligation correspondence that supersedes the interim plugins.
- [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md): the suspension instance of the doctrine, and the second dissolved dialect.
- `clef-lang-spec/spec/backend-lowering-architecture.md`: the portable-dialect requirement, operation admission and target realization; its current closure contract retires deferred casts.
- `clef-lang-spec/spec/boundary-constraints-status.md`: the boundary-constraint sequencing under which the interim plugins operate.
- `mlir-plugins/ROADMAP.md`: the tail plugins' own statement of interim scope.
