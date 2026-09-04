# Obligation Residency

> **Direction of record: proof obligations become graph citizens.**
> Baker elaborates each obligation beside the structure it constrains, saturation pairs the design-time
> and build-time twins, and the proof structure resides where the rest of the semantics reside, in the graph.
> Not normative. Companions: [Single_Flattening_Design.md](./Single_Flattening_Design.md), [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md), and [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md).

## 1. Standing and Scope

This document is a direction of record, fourth in the set beside [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md), [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md), and [Single_Flattening_Design.md](./Single_Flattening_Design.md). It is not normative: no spec text depends on it, and it imposes no requirement.

It captures a law found under pressure and directs the residency the law requires. The pressure was the HelloProof exercise (the ship-of-theseus repository), which ran the same obligations through both sides of the flattening seam and recorded, in its own ledgers, where the lowering path still holds semantic facts. The direction: proof obligations become graph citizens, elaborated and saturated via Baker in the PSG, so the proof structure matches the rest of the architecture, where closures, suspension, and dual pairs already reside as saturated structure.

Status is stated plainly at each point. The evidence of Section 2 is implemented and on the record in HelloProof's artifacts. The residency of Section 3 is design, resting on the standing plan of C-01 Section 14.5. The consequences of Section 5 are directions taken here, and the open items of Section 6 are open.

## 2. The Law: The Graph Is the Complete Specification of Observable Behavior

The law, stated first: the PSG is the sole specification of the program's observable behavior, and the flattening is refinement — representation is the backend's to choose, observable behavior is not, buffer bounds and output content included. Soundness (every asserted fact holds in the artifact) and completeness (every observable fact of the artifact is asserted) are checked separately, and both are required. Alex witnesses and elides what the graph has settled ([Single_Flattening_Design.md](./Single_Flattening_Design.md) Section 4), and a lowering path that decides an observable fact has taken a decision the graph never held.

The partition of obligations follows from the law, and it is a partition by subject: observable behavior against encoding.

**Observable subjects.** Every obligation over observable behavior is graph-born, discharged at design time, and twinned at build time. A twin-less observable obligation at build time is a detected leak: the artifact carries an observable fact whose semantic source sits below the graph.

**Encoding subjects.** Only observation-invisible encoding facts (rodata tiling, padding, section extents, address arithmetic within an encoding) may track structure, and only because nothing observable rests on them.

**The evidence.** HelloProof's ledgers state the partition in numbers. Thirteen design-time obligations carry source provenance, with `storage_hello` citing `HelloWorld.clef:9:25`, the partial-application site. The design-time ledger's description already names the residency this document directs: "Proof obligations coeffect: born in the PSG, design-time form". Twelve build-time twins exist by design, in the harness's words: "12 ids overlap by design: same obligations, two provenances". Five build-time obligations over observable subjects carry no design-time twin today: `storage_newline`, `view_newline`, `sentinel_newline` (the newline trio), `read_bound`, and `read_copy_bound`. Under the law these five are detected leaks. The semantic facts behind them live today in the Console lowering path: the newline that lowering appends to `writeln` output, and the 1024-byte `readln` buffer with its r-1 trimmed copy. They belong in platform-library source, where the PSG sees them and the recipes mint the obligations at design time. The sixth unpaired build-time id, `rodata_map`, is encoding-class and lawful. The seam is visible even in the layout pair: the design-time `layout_user_strings` states 29 bytes over the four user strings, the build-time `rodata_map` states 31 bytes over five emitted globals, and the two-byte difference is the leaked newline global.

## 3. Obligations as Graph Citizens

**Elaboration.** Each recipe emits its obligation nodes alongside the structure it fans out. An obligation node carries its inventory: subject class (observable or encoding), provenance, logic fragment, discharge target. The mechanism is the standing plan of C-01 Section 14.5: "each obligation generates its own PSG node with dependency edges to the structures it constrains, and dispatch runs over those nodes". Today the ProofObligations nanopass observes the PSG and records each obligation with its source position (HelloProof, stage one), and the ledger's word for that record is coeffect, metadata about the graph. We move the record across the distinction of [Coeffect_Analysis_Architecture.md](./Coeffect_Analysis_Architecture.md), from analysis to enrichment: the recipe that fans out a string global fans out its storage, view, and sentinel obligations in the same firing.

**Saturation.** Twin pairing is a pairing hyperedge, and it fires when both provenances are present. An unpaired observable obligation at the boundary of saturation is a design-time error, the same defect shape as an η site without its ε in [Negative_Fractional_Types_Architecture.md](./Negative_Fractional_Types_Architecture.md) Section 3, where "saturation completes only when every pairing hyperedge is closed or diagnosed". The same machinery serves both. VC-PAIR checks η against ε, obligation pairing checks design-time provenance against build-time provenance, and each is a graph check on the saturated PSG.

**The ledger.** The ledger stops being a harness convention and becomes graph structure. The harness keeps two jobs, link-time layout extraction (the ELF memory map, the concrete addresses) and the artifact side of the two-sided re-check of C-01 Section 14.5, where the PSG side is the discharge of record and the witnessed artifact is re-checkable without trusting the emitter.

## 4. The Witness as Correctness Guarantee

The Huet zipper's traversal is itself a form of guarantee. The zipper is navigation with total reconstruction: at every focus the context holds the remainder, and reconstruction returns the full structure by construction. Applied at the witness, the traversal elides settled structure into standard MLIR primitives, once, and the spec fixes the bound on what elision may drop: "the zipper elides only what the graph has already saturated" (`clef-lang-spec/spec/closure-representation.md` Section 11). The trusted base beneath the traversal is the coeffect and codata carriage of [Single_Flattening_Design.md](./Single_Flattening_Design.md) Section 4, elided into the MLIR with everything else.

The carriage's own description in the sources is an integration of three named concepts: coeffects, codata, and delimited continuations. "The integration of these concepts, using coeffects to identify codata patterns and compiling them via delimited continuations, is the synthesis Composer is built on" (`clef-lang-site`, Coeffects and Codata in Composer). The working phrase "three-tier codata/coeffect library" is the author's characterization of that three-member integration. No source names a tier structure over it, and this document introduces none.

The witness-pattern reading has two grounds in the site documents. The CDL correspondence page states the confirming role directly: "The staged-discharge architecture is the witnessing mechanism that confirms the adjoint pair is well-defined at every edge of the compilation poset." The mode-shifts page, itself a design proposal, states the local form: "each lowering pass verifies its local edges along all relevant dimensions, and the compositionality of the cell complex propagates the guarantee through longer chains." Both sentences describe edge-local confirmation whose composition carries the invariant, and they state that much and no more. The further reading, traversal as guarantee with obligation-paired elision, is this document's direction.

## 5. Consequences

**No judgment call in the seam.** The twin ledger plus the observability test decides residency mechanically. Residency follows from subject alone: observable subjects are graph-born and twinned, encoding subjects stay artifact-side, and a mismatch identifies its site, per the audit consequence of C-01 Section 14.5.

**The re-check ordering rule.** The artifact-side re-check runs immediately post-flattening, before any transform. The re-check reads its literals off the witnessed artifact, and a transform that rewrites the artifact moves the literals out from under the formulas. Check first, then transform.

**The integrity signature.** The signature for backend assignment is the op taxonomy plus the twin-pairing state, per obligation. The witness layer emits from a fixed vocabulary ([Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md) Section 5), so the taxonomy is enumerable, and the pairing state is the correspondence in which "obligations and assertions match one for one" (C-01 Section 14.5).

**The first exercise.** HelloProof's migration is the worked example and the first exercise: the thirteen design-time obligations become recipe-emitted nodes, the twelve twins become pairing hyperedges, and we retire the five leaks by moving the Console facts into platform-library source. The exercise's deliverable is a findings document, on the repository's pattern.

## 6. Open Questions

- **The mode-shift interaction.** The mode-shifts proposal has each shift "carrying with it the proof obligation that the Tier 2 structure at the node admits the Tier 3 refinement claimed". Which obligations lift across a mode shift, and how twin identity survives the shift, is undecided here. The proposal is itself design, and this document takes up none of it past the question.
- **The platform-library residency work.** The work moves the Console facts of Section 2 (the newline global, the `readln` buffer extent, the trimmed copy) into platform-library source the PSG elaborates. It is unscheduled, and the five leaks stay on the record until it lands.

## 7. Cross-References

- [Single_Flattening_Design.md](./Single_Flattening_Design.md): the single flattening, the settled posture the law binds, and the coeffect and codata carriage.
- [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md): the witness boundary, the hard-stop rule, and the fixed witness vocabulary.
- [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md): the suspension recipe and its verification conditions, one source of the obligation nodes Section 3 directs.
- [Negative_Fractional_Types_Architecture.md](./Negative_Fractional_Types_Architecture.md) Section 3: the pairing hyperedge, the boundary saturation check, and the shared defect shape.
- [Coeffect_Analysis_Architecture.md](./Coeffect_Analysis_Architecture.md): the enrichment and coeffect-analysis distinction the promotion in Section 3 crosses.
- [C-01 PRD](./PRDs/C-01-Closures.md) Section 14: the recipe mechanism, the discharge regime (14.4), and the two-sided check (14.5).
- `clef-lang-spec/spec/closure-representation.md` Section 11: proof extraction at closure sites, and the elision bound quoted in Section 4.
- `clef-lang-site`: `mode-shifts.md` and `categorical-deep-learning-adjoint-correspondence.md` for the witness-pattern sentences of Section 4, and `coeffects-and-codata.md` for the three-member integration.
- HelloProof (the ship-of-theseus repository): the ledgers, the harness, and the evidence of Section 2.
