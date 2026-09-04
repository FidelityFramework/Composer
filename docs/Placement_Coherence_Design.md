# Placement and Coherence

> **Direction of record: placement and coherence traffic are another sheaf over the shared base.**
> Escape classes, co-location hyperedges, alignment obligations, and profile admissibility, each an existing instrument, settle placement at fold-in and leave standard-dialect residue at the witness.
> Not normative. Companions: [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md), [Obligation_Residency_Design.md](./Obligation_Residency_Design.md), and [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md).

## 1. Standing and Scope

This document is a direction of record, fifth in the set beside [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md), [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md), [Single_Flattening_Design.md](./Single_Flattening_Design.md), and [Obligation_Residency_Design.md](./Obligation_Residency_Design.md). It is not normative: no spec text depends on it, and it imposes no requirement.

Its subject is placement (which core holds a value, which line holds a field, which tile holds a frame) and the coherence traffic that follows from those choices. Placement is orthogonal in the sense the corpus already carries for its sibling disciplines: another compatible sheaf over the base poset of compilation stages, in the pattern the site's compilation-sheaf and braid pages establish, connected to the semantic sheaves through the base and through nothing else. Every instrument below is borrowed from a document that specifies it for other work. This document connects them and introduces no machinery of its own: no new pass, no new dialect, no new obligation class. That count is the finding.

## 2. The Instruments

**Escape analysis, read a second time.** The DTS and DMM escape classification is the first instrument. `StackScoped` states that no reference outlives the lexical scope, so no second core ever addresses the value, and the value's coherence traffic is zero. One coeffect carries two consequences: the classification that proves no-tape in the forward-gradient case proves core-privacy here. The standing art records the actor half: per-actor arenas remove false sharing between actors by construction. The escape boundary is the coherence boundary, actor placement maps to cache domains, and each crossing is a BAREWire message of literal extent, sized at design time.

**Co-location as one constraint species.** The PHG paper carries the second instrument in its hyperedge annotation λ_f: a co-location category for spatial targets, all nodes of a source set mapped to one tile, column, or memory region, a joint constraint with no pairwise decomposition, and the paper's own paradigm of jointness is "all four operations share one BRAM block". Cache-line co-location on the CPU leg, the structure-of-arrays planes that coalesced access requires on the GPU leg, and tile residency on the NPU leg are the same species. Fold-in selects the target-indexed realization from the profile, the read the suspension forms already make.

**False sharing as alignment obligations.** At saturation every field offset is a literal, and the line size is a literal of the target profile, so the constraint that two independently written fields share no line is an offset congruence modulo the line size: quantifier-free, emitted at fold-in, discharged in the regime of C-01 Section 14.4. The instrument spans the two ends the standing art names, arena isolation removing the cross-actor case by construction and alignment attributes covering the fields that remain.

**Admissibility against the profile's memory classes.** [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md) Section 8.3 states the shape: admissibility is one comparison, an extent against a memory class, decided at saturation from the target profile. The comparison generalizes to inequalities over the profile's memory classes, L1 and L2 budgets beside tile memory, with the shape unchanged.

## 3. The Witnessed Expression

Placement leaves standard-dialect residue and nothing else, under the hard-stop rule of [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md): alignment attributes and padded literal extents on `memref.alloca` sites, `memref.subview` strides encoding the selected bundle layout, and `scf.for` nesting order carrying the selected traversal. No placement vocabulary crosses the boundary. Vectorization and coalescing are below-boundary work on the target legs, where the doctrine already licenses target-native mechanism.

## 4. Obligation Standing

Under the partition of [Obligation_Residency_Design.md](./Obligation_Residency_Design.md) Section 2, placement facts are realization-class: encoding subjects, lawfully twin-less, permitted to track structure because nothing observable rests on them. A placement fact concerns cycles and never output content, so the law of zero observable degrees of freedom holds with placement fully free. One exception is recorded. In the constant-time cryptographic reach, timing is semantic, so a placement fact there is an observable subject, discharged at the far tier: Tier 4, the relational tier, pRHL judgments.

## 5. The Demonstration Finding

One case demonstrates the reach. The forward-gradient page already states a design-time diagnostic for a K-tangent training step: K independent quires at 100·K bytes on the x86_64 stack, with every tangent's lifetime bounded by its layer's scope. Set against cache geometry, the same figures yield a ceiling. Summing the layer's K tangent buffers and its K quires against the L1 budget of the target profile is one comparison, the admissibility shape of Section 2 and the same check that gates a frame against tile memory. K is a hyperparameter the literature tunes empirically. Under this direction the compiler would derive a per-layer bound on K, decidable and target-indexed, at design time, and would print it in the diagnostic the page already states.

## 6. Cross-References

- [Thin_Middle_End_Design.md](./Thin_Middle_End_Design.md): the witness boundary, the hard-stop rule, and the below-boundary licensing behind Section 3.
- [Obligation_Residency_Design.md](./Obligation_Residency_Design.md): the observability partition and the law behind Section 4.
- [PSG_Nanopass_Architecture.md](./PSG_Nanopass_Architecture.md) addendum, with `arxiv-papers/program-hypergraph-paper.md`: the hyperedge annotation λ_f and the co-location category behind Section 2.
- [Delimited_Continuations_Architecture.md](./Delimited_Continuations_Architecture.md) Section 8: the target realizations and the one-comparison admissibility Section 2 generalizes.
- [Negative_Fractional_Types_Architecture.md](./Negative_Fractional_Types_Architecture.md) Section 5.1: the load-bearing forms this discipline places.
- `arxiv-papers/dts-dmm-paper.md` Section 3.2: the escape classification of Section 2.
- `clef-lang-site`: `docs/internals/hardware/cache-aware-compilation-cpu.md` and `cache-aware-compilation-gpu.md`, the standing art this document sits between; `docs/design/categorical-foundations/forward-gradients-exact-accumulation.md`, the figures of Section 5; `docs/design/categorical-foundations/the-compilation-sheaf.md` and `braid-as-a-fourth-sheaf.md`, the sibling-sheaf pattern of Section 1.
