# Negative and Fractional Types Architecture

> **Status: DRAFT. Design in full. Nothing in this document is implemented.**
> Negative and fractional types as saturated dual-pair structure, witnessed in standard dialects.
> This is the companion treatment that `clef-lang-spec/spec/terms-and-definitions.md` names for its
> proposed negative and fractional type entries. It is not normative, and no normative text depends
> on it. Its horizon is the second, per the [PRD index](./PRDs/README.md).

## 1. Standing and Scope

The spec carries **negative type** and **fractional type** as proposed, non-normative entries whose capsule text ends: "the discipline is the subject of a companion treatment, not a normative chapter of this specification." This document is that companion treatment on the Composer side. It is a draft for review, and it imposes no requirement.

Three sources govern it. The pre-print *Negative and Fractional Types in the Fidelity Framework* (`arxiv-papers/negative-fractional-types-in-fidelity.md`, working draft, June 2026) is the source of record for the surface discipline and for the lowering posture Section 6 reconciles. The site design page (`clef-lang-site/hugo/content/docs/design/types/negative-fractional-types.md`) is the orientation register of the same material. The vocabulary descends from James and Sabry, *The Two Dualities of Computation: Negative and Fractional Types* (2012), with Chen and Sabry's compact closed interpretation (2021) as the categorical development the pre-print builds on.

Two register rules hold throughout. First, everything here is design: the pre-print proposes, the spec entries are capsules, and Composer today contains no parser form, no recipe, and no witness for any construct below. Where a sentence reads as settled, the settled thing is a design decision, never an implementation. Second, the pre-print is quoted exactly where its wording is load-bearing, and Section 6 flags the sentences that resist this document's reading in place of smoothing them.

The graduation path follows the repository's pattern: review of this treatment against the C-01 Section 14 form family and the thin-middle-end doctrine, a findings document from the first exercise (Section 8), then a spec chapter when the discipline earns normative standing. Until then the spec's capsule entries are the only text with standing.

## 2. The Published Surface

The forms below are the pre-print's Sections 6 and 7, quoted exactly, with capsule prose between them. The source titles its Section 6 "Notional Syntax in Clef", and notional is the weight this document preserves.

**Constructors.** The additive dual would be written `Neg<'T>` or, infix, `-'T`. The multiplicative dual would be written `Recip<'T>` or `1/'T`.

```fsharp
// Negative types inherit the dimension of their positive counterpart
// dimension N, reverse direction
type ReverseForce = Neg<float<N>>
// dimension A, reverse direction
type ReverseCurrent = -float<A>

// Fractional types invert the dimension through Kennedy's algebra
// dimension N^-1, constraint
type Compliance = Recip<float<N>>
// dimension ohm^-1, constraint
type Conductance = 1/float<ohm>
```

**Dimension behavior.** The pre-print fixes both transformations in two sentences: "A negative force value carries the same dimension as a positive force value because the reversal is in the direction of evaluation, not in the dimensional algebra. A reciprocal force value carries the inverse dimension because the multiplicative inverse extends the abelian group structure that Kennedy's units of measure already provide." The inference extension is Kennedy's group algebra "lifted from integer to rational exponents", with unification at η and ε sites reducing to algebraic identity checking.

**Primitives.** The four η and ε operations, with their published types:

```fsharp
val eta_plus : unit -> ('T + Neg<'T>)
val epsilon_plus : ('T + Neg<'T>) -> unit
val eta_times : unit -> ('T * Recip<'T>)
val epsilon_times : ('T * Recip<'T>) -> unit
```

"These operations would not be function calls in the conventional sense." They are structural transitions Baker would recognize during elaboration and settle on the PSG as codata.

**Term forms.** `negate(e)`, `unwrap(e)`, `recip(e)`, `eta_times<T>()` (instantiated in the source's Section 7 as `eta_times<Evidence>()`), and `epsilon_times(e1, e2)`.

**Directional judgments.** Inference would extend the existing HM unification with a direction annotation on each judgment:

```
Γ ⊢▸ e : 'T          // forward judgment: e produces a 'T
Γ ⊢◂ e : 'T          // backward judgment: e demands a 'T
```

The published rules for the additive side:

```
       Γ ⊢◂ e : 'T
   ──────────────────────
   Γ ⊢▸ negate(e) : Neg<'T>

   Γ ⊢▸ e : Neg<'T>
   ──────────────────
   Γ ⊢◂ unwrap(e) : 'T

   Γ ⊢▸ e1 : 'T    Γ ⊢◂ e2 : 'T
   ─────────────────────────────
   Γ ⊢▸ epsilon_plus(e1, e2) : unit
```

And for the multiplicative side:

```
   ─────────────────────────────────────────
   Γ ⊢▸ eta_times() : ('T * Recip<'T>)

   Γ ⊢▸ e1 : 'T    Γ ⊢▸ e2 : Recip<'T>
   ─────────────────────────────────────
   Γ ⊢▸ epsilon_times(e1, e2) : unit
```

**One worked example, compressed from the source's Section 7.** A dosage lookup whose conditioning obligation is a fractional value, and the completion that discharges it:

```fsharp
let dosageLookup_bayesian
    (patient : PatientId)
    (prior : Distribution<Dosage>)
    : (Dosage * Recip<Evidence>) =
    let (proposed, demand_evidence) = eta_times<Evidence>()
    let dosage = sampleFromPosterior prior proposed
    (dosage, demand_evidence)

let dosageLookup_complete
    (patient : PatientId)
    (prior : Distribution<Dosage>)
    (evidence : Evidence)
    : Dosage =
    let (dosage, demand) = dosageLookup_bayesian patient prior
    let () = epsilon_times(evidence, demand)
    dosage
```

`Recip<Evidence>` records the unsatisfied conditioning obligation, and the `epsilon_times` at the application site would settle it by unifying the supplied evidence with the demand.

## 3. The Dual-Pair Recipe in Baker

Baker would carry the duality discipline as a recipe, the mechanism C-01 Section 14 establishes for closures and [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md) Section 3 extends to suspension. The recipe is design. Its two halves follow the standing shape.

**Fan-out: η mints the pair.** Fan-out elaborates an η site into two cells, the positive value and its dual, and one audit-log hyperedge connecting them, with a direction annotation on each judgment edge as the introduction rules assign it. The pre-print places this structure directly: "an audit-log structure pairing forward computational events with their negative-typed adjoints would live in the PSG as hyperedges connecting positive and negative cells, with Baker's elaboration carrying the type-level pairing as PSG codata". The hyperedge is the pairing. No operation carries it, so nothing about it survives as op vocabulary, the same relocation the suspension recipe records for the delimiter.

**Fold-in: ε is a cancellation event.** An ε site is a joint constraint on the pairing hyperedge. At saturation, when both member cells are literal, the constraint discharges: the types unify, the dimensional exponents cancel, and the direction annotations at the site are the ones the elimination rule assigns. A discharged pair is closed, and its positive member proceeds as an ordinary value. Realization below the witness boundary is Section 6's concern.

**The boundary saturation check.** The observability constraint is the program-scope form of the same check, and the pre-print states it in full: "A complete program in extended Clef would typecheck by producing only forward judgments at the top level. The presence of unresolved backward judgments or unsatisfied fractional constraints at program scope would produce a design-time error. The discipline matches the observability constraint from the James and Sabry paper (James and Sabry 2012b), where complete programs must have positive non-fractional types at their boundaries." In recipe terms, saturation completes only when every pairing hyperedge is closed or diagnosed: an open backward judgment or an open fractional demand at program scope is a design-time error, per the pre-print, and Section 5 assumes this check has run.

**Obligations, quantifier-free.** The discharge regime is C-01 Section 14.4's: at saturation, over PSG literals, before witnessing.

| VC | Obligation | Fragment | Discharge |
|---|---|---|---|
| VC-PAIR | every η pair carries exactly one ε site on its pairing hyperedge | None; graph | Graph check on the saturated PSG |
| VC-DIR | each pairing edge carries the direction annotation its rule assigns, preserved by every pass | None; per edge | Per-edge check, per pass: the duality dimension as an edge check |
| VC-CANCEL | the pair's exponent vectors are equal at an additive ε site, and sum to zero at a multiplicative one | QF_LRA; QF_LIA in the integer case | Linear arithmetic over exponent literals |
| VC-OBS | at program scope, no unresolved backward judgment and no open fractional demand | None; graph | Boundary check at saturation |

The fourth dimension is an edge check because the pre-print already lists it as one: "Each lowering pass verifies its local edges along all relevant dimensions (compilation, joint constraint, verification strength, execution direction), and the compositionality of the cell complex propagates the guarantee through longer chains." VC-DIR is that sentence in table form.

The dimension algebra routes where the pre-print routes it. For the rational exponents fractional types introduce, "the obligations would be discharged in QF_LRA, quantifier-free linear real arithmetic over the rationals", and the fragment stays linear because "dimensional cancellation reduces to additive constraints on the exponents (the values multiply, the exponents add)". VC-CANCEL inherits that placement, and where the exponents are integers the existing QF_LIA machinery serves.

## 4. The Substrate Mapping

This section states a mapping as design and holds it as hypothesis. The pre-print fixes the categorical invariant and leaves the operational realization to the lowering. The mapping below names the standing machinery this pipeline would realize each dual on, and claims as much as the readings support.

**Negatives ride the suspension machinery.** For the negative side the pre-print records that "the literature converges on control-flow reversal as the operational reading". The pipeline's control-flow reversal machinery is the suspension recipe: continuation frames, which are the C-01 Section 14.1 environment in its second instance; delimiter subgraphs, whose boundary is structure a builder extent defines; and VC-ONE, the resume-exactly-once obligation on each suspended frame. The sources state the same linearity commitment twice. The pre-print, on why ε is never the eraser: "We would not emit ε as an eraser: the eraser realizes weakening, and silent discard would violate the linearity commitment the negative-type discipline rests on." The suspension architecture, on frames: "each suspended frame is resumed exactly once", with multi-shot admitted only as a declared copy carrying its own linear obligation. A `Neg<'T>` under this mapping is a suspended adjoint: a frame whose resumption runs the reverse computation, delimited by the subgraph its pairing hyperedge spans.

Theoretical support comes from the coexponential strategy note (`arxiv-papers/research/coexponential-inference/strategy-net-substrate-cgra-qkb.md`, an internal note whose own opening holds that "None of this is a result"). The note's actor-as-coexponential-server position is carried by two sentences. On the server: "Its server is a greatest fixed point, `¡A = νH_A`, codata, where standard interaction nets handle finite inductive reduction." On where the serving order lives: "the quotient lives in the session type the PSG carries into the hypergraph at Tier 1, the realized order is supplied by the actor mailbox at runtime". A backward obligation served by a suspended context has the same shape as a server obligation realized by mailbox delivery, and the mailbox is one of the four resumption-edge members the suspension recipe already abstracts. The support is theoretical and candidate-grade, and it is quoted here as that and nothing more.

**Fractionals ride the demand machinery.** A `Recip<'T>` is an open demand: the site page's design description is "the demand for a `'T`, carried as a value the substrate must settle". The demand-driven half of the representation family is the standing machinery with that shape: the lazy value's memoization slots, write-once at force; the seq form, where "every resumption source is the caller's pull"; and incremental cutoff by environment closedness (R-04 to R-06), the third member the load chain of Section 8 names. The pre-print's Bayesian section allows settlement past design time: "The discharge happens through the SMT dialect at design time if the evidence is statically known, or propagates to runtime if the evidence depends on dynamic input." Under this mapping the propagated case is carried by a slot in the family, with write-once-at-force as the settlement event and the pull edge as the demand's path.

**The hypothesis boundary.** The readings support the alignments: linearity with linearity on the negative side, demand with pull on the fractional side, and one environment object under both, since the frame and the slot are both instances of the C-01 Section 14.1 environment. The readings support nothing past alignment. The pre-print's initial deployment names Inet cuts and solver discharge, and names neither frames nor slots. The identification of the backward channel with a suspended frame, and of the open demand with a demand slot, is this document's design, and it holds as hypothesis until the Section 8 exercise runs.

## 5. The Witnessed Form

By the time the zipper reaches a dual pair, the Section 3 obligations have discharged, VC-PAIR through VC-OBS. Alex witnesses settled structure, in the posture the spec's closure chapter fixes: "The witness reads the layout; it does not compute it."

What crosses the boundary is standard dialects only, under the hard-stop rule of [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md): "no llvm dialect, and no semantic dialect, in the MLIR witnessed out of the PSG." Applied to the duality discipline:

- **Forward values cross as ordinary values.** The positive member of a closed pair is an ordinary typed value at witness time, and its MLIR is whatever the C-01 Section 14.3 correspondence table gives its type.
- **Backward obligations discharge at saturation.** What a discharged adjoint leaves in the artifact is ordinary structure: the adjoint's environment is a frame in the form family, witnessed as a byte frame, static views, and a function value, per the same table. No dual constructor, no direction annotation, no new op.
- **Nothing dual-shaped crosses.** A `Neg` or `Recip` still open at the witness is the Section 3 boundary check's design-time error surfacing, and a conforming build never reaches the witness carrying one. The enforcement parallel is the other half of the boundary: the spec treats an `llvm.*` op in the middle end as a category error, and dual-shaped structure in the witnessed region is the same class of error under this design.

The per-pass execution-direction check is edge verification, and it composes with the two-sided posture of C-01 Section 14.5 as follows. The PSG side is the discharge of record for every duality obligation, VC-DIR included, checked per edge, per pass, above the boundary. The MLIR side re-checks what the artifact carries, which for this discipline is the frame-layout residue of the adjoint (extent, offsets, sizes) and none of the direction structure, because direction annotations stay behind in the graph. The re-check covers the standard-form residue exactly, and the direction dimension is verified in the graph, where its edges reside.

### 5.1 The Load-Bearing Case

A dual pair is load-bearing when its ε executes at runtime because its operands are not both compile-time literal. The Section 3 obligations hold over structure, types, and exponents, literal at saturation whether or not the member values are, so a load-bearing pair discharges VC-PAIR through VC-OBS exactly as a static one does. Load-bearing never means the ε might not happen: VC-PAIR fixes one ε site per pairing edge, saturation completes only when every pair is closed or diagnosed, and the ε is therefore settled structure whose operand values arrive late. Existence, linearity, and direction are static. What remains for runtime is value transport, and the discharged proof resides where [Obligation_Residency_Design.md](./Obligation_Residency_Design.md) directs, in the graph.

**A load-bearing `Neg<'T>` is a one-shot suspension frame.** The frame's code is the reverse segment's subgraph, delimited by the boundary its pairing hyperedge spans. Its capture list is enumerated at saturation, the live-across read applied to the reverse segment, and fold-in places the frame by its escape class, per [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md) Section 4. The witnessed form is the pair convention of C-01 Section 14.3, a function value and a byte frame, and consumption at the ε site is one `func.call_indirect`. Resume-exactly-once is VC-ONE, the suspension recipe's own obligation, so the linearity commitment Section 4 quotes is already an obligation of the recipe. The capture set is the minimal reverse-needed set, and a `Neg` whose frame amounts to a tape is a design-time error at saturation: a tape's extent grows with the forward run, and VC-EXT discharges ground arithmetic over a literal extent.

**A load-bearing `Recip<'T>` splits on where its supply resides.** When supply is statically ordered against demand inside program scope, the pair resolves to plain SSA dataflow and materializes nothing, the terminal posture of C-01 Section 14.2. A supply that arrives at runtime (an observation, a message, a reply) witnesses as a single-assignment cell of literal extent with a state flag: the memoization slot class of `closure-representation.md` Section 7 under its write-once discipline, with the writer external to the consumer, per the dependency-reference row of the same table. The waiting consumer is a suspension frame whose resumption edge is the store. That compound, one cell and one resumed frame, is the reply shape of the Section 4 coexponential reading, held at the same candidate grade.

**No new carrier.** A load-bearing dual selects among the forms this section already witnesses. The negative rides the hot machinery, the frame and its call. The fractional rides the cold, the cell and its store. The audit of C-01 Section 14.5 extends with the same counters: one materialization to one consumption per `Neg` site, and per runtime `Recip` one cell, one store, at most one resume.

**The fence rule holds unchanged.** Section 7 rejects a load-bearing dual at the `ffi.` fence on the same argument as any other, because the far side holds no judgment forms whether or not the ε would have executed. If a lawful crossing ever exists, it is a BAREWire session whose protocol carries the discharge, in the structural-admissibility posture of the descriptor-shared form of C-01 Section 14.2. That session is future work, stated here as direction and not as design.

## 6. The Seam Reconciliation

The seam: the pre-print's Section 8 sketches an initial deployment through two dialects, Inet cuts for the negative side and the SMT dialect for `epsilon_times`, while the thin-middle-end doctrine holds the witnessed region to standard dialects and counts its semantic dialects at zero: "The number of semantic dialects in the witnessed region is zero, and it stays zero." Read flatly, the two texts are in conflict. Read against the pre-print's own design-time sentences, they compose. This section argues the compatible reading, then flags, verbatim, the sentences that resist it.

**The compatible reading.** Discharge is design-time over the graph, in the pre-print's words. Three sentences carry it:

1. "At each ε morphism site, Baker would record an SMT assertion that unifies the supplied value with the demand the corresponding η introduced. The solver discharges the unification at design time, with the result either entering the verified program (satisfiable) or producing a design-time error at the source location (unsatisfiable core)."
2. "Baker would resolve the structural relationships at the design-time CCS layer, with the middle end eliding the saturated PSG through the dialects the framework targets."
3. On the upstream smt dialect's missing Real sort: "Until it lands, real-sorted discharge would run in the compiler service against the PSG, while integer and bit-vector obligations take both paths."

Sentence 3 is the placement this document adopts as the general rule, and its "both paths" is the two-sided check of C-01 Section 14.5 in the pre-print's own words: the PSG side is the discharge of record, and the smt-dialect side is the re-check encoding, where "The smt module is a verification artifact beside the program, not part of it." QF_LRA therefore sits graph-side, and the SMT dialect appears only as the artifact-side re-check per C-01 Section 14.5. No semantic carrier enters the witnessed region on this path.

For the negative side, the Inet cut is a below-boundary realization. The pre-print's Section 13 states the cut's standing exactly: "the net cut serving as the operational carrier of the type-level pairing and not as the categorical counit itself". A carrier of a settled pairing is target realization, and target realization is where the doctrine licenses a dialect: "The test is directional: a dialect qualifies by expressing the target upward, never by expressing Clef downward." The DCont and Inet classifier of `clef-lang-spec/spec/native-type-mappings.md` is the routing mechanism, and the irregular lane's cut is reached only on a leg whose target natively holds net reduction. The pre-print leaves that choice open by its own lowering posture: "Our nanopass architecture would admit either reading as a target for the saturated PSG, with the choice driven by what the deployment target affords and what the application requires." On the CPU path, ε completes at saturation, the witnessed form is the standard-dialect residue of Section 5, and nothing net-shaped is witnessed.

**Sentences that resist, flagged verbatim.** Three sentences state a dialect-resident deployment the doctrine excludes, and they are flagged here in place of a smoothing paraphrase.

1. Pre-print Section 6: "the initial deployment developed in Section 8 routes `epsilon_times` closures through the SMT dialect for constraint discharge". As written, the dialect is the discharge path. Under the doctrine, the discharge path is graph-side dispatch, with the dialect as re-check.
2. Pre-print Section 8.3: "Constraints would accumulate through subsequent operations as standard SMT assertions, with the SMT dialect maintaining the constraint graph alongside the operation graph the rest of the pipeline operates over." A dialect maintaining a constraint graph alongside the operation graph is a semantic carrier inside the witnessed region. The hard-stop rule excludes it.
3. Pre-print Section 8.2: "The negative type discipline would lower from the saturated PSG into a rewrite-rule layer at the Composer compiler's formal middle end." The phrase "at the Composer compiler's formal middle end" places the rewrite-rule layer above the boundary. The compatible reading needs it below, on the net-target leg.

**Verdict.** The reconciliation holds on the design-time-discharge and elision sentences and fails on the three flagged ones. The disagreement is narrow. It concerns where the deployment's machinery resides. Both texts state the invariant identically: a type-level pairing carried as codata, preserved by every pass. Review owes a settlement, and the candidates are two: the pre-print's initial-deployment sketch revises toward the two-sided posture, or the doctrine records an exception it currently rules out. This document proceeds on the doctrine.

## 7. The Fence Rule

The rule, stated as design: no `Neg` and no `Recip` in the type of any value crossing the `ffi.` fence, in either direction, rejected at compile time.

The argument is the observability constraint applied at the boundary contract. A `Neg<'T>` crossing the fence exports its backward judgment to the far side, and the far side holds no judgment forms: the C ABI party of the C-01 Section 6.7 contract is bytes and calling convention, so the crossing would hand an obligation to a party that cannot discharge it. A `Recip<'T>` crossing exports an open demand whose unification site the far side cannot supply. Either crossing leaves a pairing hyperedge that no saturation can close, which is the condition the pre-print's program-scope check names as a design-time error. We apply the same check per crossing: the code on the far side of an extern declaration is outside the judgment system, so each crossing is a program boundary for the value that crosses, and each crossing value must be positive and non-fractional.

The tie to the boundary-contract tiers is by cross-reference: the fence rule is a precondition on the tier classification of C-01 Section 6.7 (Tiers A, B, C), in the same posture as Tier C, where "the boundary type demands the property instead of assuming it". Tier classification proceeds only over crossing types the fence rule admits.

## 8. Sequencing and Dependencies

**The load-bearing chain.** The PRD index places NFT at the second horizon and names its dependencies: C-01, C-02, C-05, R-04 to R-06. The index's own sentence carries the exact bound: "the flat-closure finiteness lemma (C-01), the lazy slot class (C-05), and incremental cutoff by environment closedness (R-04 to R-06) are the members beneath it, and its guarantees hold exactly as far as those three hold." The pre-print rests its decidability claim on the first member in as many words, "with the verification remaining decidable because the dependencies are structurally explicit in the flat closure representation".

**Placement.** Nothing in this document schedules work. The C-01 recipe migration (C-01 Section 14.7) and the suspension recipe's CPU form ([Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md) Section 9, item 1) are themselves pending, and the dual-pair recipe presupposes both. Second horizon means after them.

**First exercise, sketched.** One reversible sample from the pre-print's Section 7, traced through the dual-pair recipe on the CPU path.

```fsharp
let dosageLookup_reversible
    (patient : PatientId)
    : (Dosage * Neg<PatientId>) =
    let (weight, weight_rev) = getWeight_reversible patient
    let (condition, condition_rev) = getCondition_reversible patient
    let dosage = computeDosage weight condition
    let patient_rev = negate(reconstitute(weight_rev, condition_rev))
    (dosage, patient_rev)
```

- **Fan-out.** The `negate` site mints the pair: a positive cell for the forward chain ending in `dosage`, a negative cell for `patient_rev`, one audit-log hyperedge connecting them, with the direction annotations the negate rule assigns. The adjoint's dependencies (`weight_rev`, `condition_rev`) are captures, so the adjoint carries an environment per C-01 Section 14.1.
- **Fold-in.** The ε site is the audit replay's annihilation site, where the pre-print has the adjoint "annihilated against the forward result" to reproduce the original input. Fold-in selects the adjoint's form from the Section 14.2 family and settles its frame literals.
- **Obligations instantiated.** VC-PAIR: the η at the `negate` site pairs with the replay-site ε, one for one. VC-DIR: the pairing edge carries the annotations the `negate` and `epsilon_plus` rules assign, one side forward and one side backward at the close, preserved by each pass. VC-CANCEL: `PatientId` against `Neg<PatientId>`, dimensionless, so cancellation degenerates to type identity, the pre-print's "algebraic identity checking". VC-OBS: the replay site closes the pair inside program scope. Beneath these, the standing closure conditions on the adjoint's environment: VC-EXT and VC-DIS on the frame layout, and VC-ONE if the adjoint suspends.
- **Witness, CPU path.** The realization is the state-machine form of [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md) Section 8.1. The adjoint witnesses as a function value and byte frame under the C-01 Section 14.3 pair convention. Direction annotations stay in the graph, and nothing net-shaped is witnessed, per Section 6.

The exercise's deliverable is a findings document, on the repository's pattern.

## 9. Open Questions

- **The grade axis.** The pre-print's Section 13 records a negative-grade direction and holds it to four named open items, and its Section 5 composition of reversion with compact closure stands as published. How the duality discipline composes with the grade axis past what the pre-print states is deferred to the author. This document takes none of it up.
- **Multi-shot duals.** The suspension discipline admits multi-shot only as a declared copy: "Multi-shot is never the silent default. It is a declared copy with its own linear obligation." Whether a declared copy of a suspended adjoint mints a second pair with its own VC-PAIR, or is inadmissible because a duplicated obligation is the duplication the linearity commitment excludes, is undecided here. The Section 8 exercise does not reach it, since its sample is one-shot.
- **The quantum sections.** The pre-print's Sections 11 and 12 (multiplicative unitarity, adiabatic schedules) sit past this document's horizon. Their obligations include real-sorted quadratic forms and QF_BV stabilizer membership, theories this treatment does not carry. Further horizon, no treatment here.

## References

- `arxiv-papers/negative-fractional-types-in-fidelity.md`. The pre-print, and the source of record for Sections 2, 3, and 6.
- `clef-lang-site/hugo/content/docs/design/types/negative-fractional-types.md`. The site design page.
- James, R. P., Sabry, A. *The Two Dualities of Computation: Negative and Fractional Types* (Indiana University, 2012). The lineage of the vocabulary.
- Chen, C.-H., Sabry, A. *A computational interpretation of compact closed categories* (POPL 2021). The categorical development of the same lineage.
- [C-01 PRD](./PRDs/C-01-Closures.md) Sections 6.7 and 14. The boundary contract, the form family, the discharge regime, and the two-sided check.
- [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md). The suspension recipe, its verification conditions, and the CPU realization.
- [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md). The witness boundary and the hard-stop rule.
- [Obligation_Residency_Design.md](./Obligation_Residency_Design.md). The residency of discharged obligations, per Section 5.1.
- `clef-lang-spec/spec/closure-representation.md` Sections 7 and 11. The slot-class schema and proof extraction at closure sites.
- `clef-lang-spec/spec/terms-and-definitions.md`. The proposed-entry capsules and the naming of the companion treatment.
- `arxiv-papers/research/coexponential-inference/strategy-net-substrate-cgra-qkb.md`. Internal strategy note, quoted in Section 4 at candidate grade.
