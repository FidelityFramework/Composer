# The Single Flattening

> **Direction of record: the pipeline flattens once.**
> Alex elides the saturated Program Hypergraph into standard MLIR primitives in a single pass,
> and the witnessed primitives descend into the backend targets of the `Fidelity.Platform` description.
> Not normative. Companions: [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md) and [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md).

## 1. Standing and Scope

This document is a direction of record, stated as close to a decision as the author has taken it. Formalization in the literature is deferred, and Section 5 carries the deferral as an open item. It is not normative: no spec text depends on it, and it imposes no requirement. Where the source discussions mark a step as intuition, the text below holds it as intuition.

The PHG paper is expected to undergo significant revision once this residency is established: net reduction resident in the graph, one flattening beneath it. Section 5 states how the paper's Inet lane reads after that revision. Until then the paper stands as published.

## 2. The Furtive Design and Its Lesson

The earlier working view, the furtive design of the source discussions, treated MLIR as established art to be used directly: the typed tree and the AST from F# would be written to whatever high-level MLIR dialect supported the structure, and MLIR transforms would carry the program through the rest of the pipeline.

What the design taught is recorded with why it was left. Carrying significant structure into MLIR implied a sprawl the project always meant to resist. At the time that was intuition. The reconstruction remit analysis later named the reason ([Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md) Section 1): dialects in that culture are reconstruction vessels for structure-poor front ends, and Clef arrives structure-rich. A dialect built to recover structure has no work to do on input that never lost it.

The furtive designs are left, and their lessons and formalism are kept: brought up into the principled hypergraph structure, where the information is preserved in a way that is meaningful in the compiler pathway.

## 3. The Gravity, Twice Observed

Semantic structure in this pipeline settles up into the graph and leaves standard-dialect residue at the witness. The source discussions call that pull the gravity. It has been observed twice.

The first observation is on the record in [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md). Delimited continuations lifted into the PSG as saturated structure, and the op rendering retired. The delimiter is the boundary a builder extent defines, and the witnessed form is a discriminant, a byte frame, and `scf.index_switch`.

The second observation is interaction nets: agents, ports and active pairs give
a concrete model of local graph rewriting. Baker's fan-out/fold-in offers a place
to express admitted rules while preserving their source and joint constraints.
The correspondence must identify the actual rule system and its premises.
Readiness—having the required participants elaborated—does not establish
confluence, termination or equivalence for arbitrary PSG transformations.
Each owning analysis supplies its settlement argument; overlapping rewrites need
admitted conflict handling. The
[rewrite and tape criteria](Nanopass_Incremental_Contract_Direction.md#24-rewrite-independence-and-the-intermediate-tape)
retain participants, premises, replacements and invalidations through intermediate
artifacts. A static annihilation reduces active computation while preserving the
evidence required by remaining consumers and inspection.

The case split mirrors the suspension architecture exactly.

**Static reduction extent.** The net normalizes at saturation and vanishes into the graph, the analog of the vacant and unmaterialized forms in the closure family. Nothing net-shaped is witnessed.

**Dynamic irregular reduction.** The witnessed residue is data plus a driver. The rule set compiles to a finite function table, and the cell population is a typed region whose cell layout is literal in BAREWire terms. The driver is per-target transliteration: a worklist loop on the CPU leg, the redex-bootstrapping kernel loop on the GPU leg.

In either case confluence remains a tier obligation, a property of the logic, never a runtime negotiation.

Below the witness boundary the Inet dialect survives as transliteration vocabulary for targets that natively host nets, the CGRA line. That is the seat WASM stack switching holds for suspension ([Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md) Section 8.5), and each dialect sits in it by the doctrine's directional test, expressing the target upward ([Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md) Section 4). The two dialects bookend the targets, one line lower than the earlier sketches placed them, and the graph above them holds the semantics of both constructs.

## 4. The Finding: One Flattening

Alex, the Library of Alexandria, is a single flattening pass. The Huet zipper, with coeffect and codata carriage, elides the saturated hypergraph into standard MLIR primitives directly, once, and those primitives descend into the backend targets of the `Fidelity.Platform` description.

The consequence is a redistribution of weight. Alex is more constrained than earlier envisioned: it witnesses and elides what the graph has settled, and it performs no semantic transformation. Baker gains significant structure. The recipes carry the semantic inventory: closures, suspension, dual pairs, and nets. The semantic weight of the pipeline resides in elaboration and saturation.

Numeric construction makes this boundary concrete. Dimensions, justified ranges,
demand/effects, storage identity and permitted partial/merge arithmetic meet on
the same source participants. Baker can resolve eligibility and rewrite conflicts
there, retaining the intermediate tape, before Alex witnesses their admitted
form. The [PHG coloring research note](../../arxiv-papers/research/PHG/tractable-conflict-coloring.md)
gives a conditional clique-tree theorem for one tractable conflict class. This
supports keeping joint semantic decisions together before flattening; it does
not assert that all PHG regions belong to that class or that MLIR cannot encode
the same information. Such an encoding would require its own explicit transport
and authority contract at an additional semantic boundary.

The zipper witnesses what is already settled. Semantic decomposition happens
in the graph before that boundary; selected representations, participant
correspondence and proof premises still needed by later consumers travel with
the witnessed form or its checked evidence. Flattening does not authorize
discarding those obligations. This preserves the coherence of the joint judgment
without asking Alex to reconstruct it from emitted operations.

## 5. Consequences

**Design-time services read the graph.** Design-time integrity and developer information live in the graph, before flattening, where the design-time services read, and [Obligation_Residency_Design.md](./Obligation_Residency_Design.md) directs proof obligations into the same residency.

**The count is one.** Flattening happens one time only, on the way to hardware description.

**The zero holds.** The witnessed region stays in standard dialects, and the semantic-dialect count of [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md) Section 5 holds at zero.

**The Inet lane after revision.** After the revision named in Section 1, the PHG paper's Inet lane reads as a coeffect and codata characterization, drawn under analysis over structured source.

Open items:

- **The dynamic-reduction driver.** The per-target designs behind the two shapes of Section 3, the CPU worklist loop and the GPU kernel loop.
- **The below-boundary net vocabulary.** The Inet dialect's form for CGRA-class targets.
- **Formalization in the literature.** Deferred per Section 1. The PHG paper's revision comes first.

## 6. Cross-References

- [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md): the doctrine, the witness boundary and its hard-stop rule, the reconstruction argument, and the zero semantic-dialect count.
- [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md): the first observation of the gravity, the suspension recipe, and the below-boundary seat stack switching holds (Section 8.5).
- [Negative_Fractional_Types_Architecture.md](./Negative_Fractional_Types_Architecture.md) Section 6: the seam reconciliation whose compatible reading this document extends to reduction at large, resting on the pre-print's sentence for the cut's standing: "the net cut serving as the operational carrier of the type-level pairing and not as the categorical counit itself".
- [Closure_Nanopass_Architecture.md](./Closure_Nanopass_Architecture.md) Section 4: the finiteness lemma, which the literal cell layouts of Section 3 rest on.
- [PSG_Nanopass_Architecture.md](./PSG_Nanopass_Architecture.md) addendum: the PSG-to-PHG evolution, hyperedge promotion, and the firing condition Section 3 reads as readiness.
