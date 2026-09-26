# Numeric selection and arithmetic construction: PRD assessment

September 26, 2026. Requested and accepted during F/C completion. The owner
accepted the two feature areas and their expanded validation remit. Registered
contracts are [F-11](F-11-NumericSelection.md) and
[C-08](C-08-ArithmeticConstruction.md), with concrete cases in the
[numeric validation inventory](Numeric_Validation_Cases.md) and visible rows in
the [master index](README.md). This assessment retains the rationale and shared
ownership. Sample numbers are a separate namespace. Registration advances no
implementation or completion claim.

## Accepted decision and ownership

The two accepted owners have lettered acceptance groups inside each:

| Registered PRD | Responsibility | Reason for a separate owner |
|---|---|---|
| **F-11 — Numeric Selection and Numeric Obligations** | Justified ranges; integer width and real representation selection; operation capacity, definedness, scale, rounding/error obligations, boundaries and diagnostics | These apply to ordinary arithmetic independently of collections or concurrency. F-06 parsing and F-07 bitwise operations consume this contract. |
| **C-08 — Arithmetic Construction and Reduction** | Semantic construction of computation regions; residual/accumulator state; permitted term formation, partitions, merges and finalization | This has its own observable result contract and preservation argument. Correct collection iteration and sequence transport do not establish its arithmetic laws. |

Keep [M-01](M-01-DialectAdmission.md#5-numeric-selection-parallelism-and-design-time-projection)
as the owner of admitted Alex vocabulary and downstream correspondence. Actual
scheduling, publication and progress retain their A/T/R and platform owners.
C-08 establishes numerical permission consumed by those realizations, with
their separate execution obligations still required.

More letters beneath F-06/F-07 or C-07 alone would leave the general selector and
construction contract without an appropriate owner. IEEE, fixed-point and posit
do not each need a public feature number: they are families under the same NTU
contract. Two owners with four groups each give a concrete initial inventory.
Existing owners below retain their obligations alongside these registered areas.

## Governing requirements

The normative [Numeric Selection](../../../clef-lang-spec/spec/numeric-selection.md)
chapter governs this assessment, together with
[Width Inference](../../../clef-lang-spec/spec/width-inference.md),
[Rounding](../../../clef-lang-spec/spec/rounding.md),
[Units of Measure](../../../clef-lang-spec/spec/units-of-measure.md) and
[Conformance](../../../clef-lang-spec/spec/conformance.md#6-the-preservation-obligation-through-lowering).
[Pondering Fearless Parallelism](../../../clef-lang-site/hugo/content/blog/pondering-fearless-parallelism.md)
and its [construction companion](../../../clef-lang-site/hugo/content/docs/internals/numerics/arithmetic-construction-and-placement.md)
provide motivation and discriminating examples. The spec carries the requirements.

Width is one part of this contract. Selection filters by capability/policy and
coverage before minimizing the specified worst-case representation error. That
score does not establish computation error. Construction separately establishes
capacity, error, exactness and reproducibility at the scope claimed. Permission
to change a reduction tree differs from permission to schedule its independent
operations concurrently.

Ordinary compilation generates applicable obligations. Inference retains pending
constraints during design; commitment requires settlement or a located diagnostic.
Quantified dimensions are valid schemes, not missing facts. Source kind, physical
dimension, representation scale and range retain distinct correspondence. The
specified unobservable bare `float` exception permits IEEE f64 only under §6's
capability conditions; it does not establish a dimensioned range, coverage of a
known range, or arithmetic safety. There is no additional default unit, integer
width or invented numerical goal.

## Accepted lettered acceptance groups

Letters identify groups within the registered PRDs. Concrete case identifiers,
oracles and requirement mappings are declared in the validation inventory.

| Proposed group | Spec clauses | Required acceptance |
|---|---|---|
| **F-11(a): range and dimensional inference** | Numeric Selection §§3–4, 6, 9.1; Width Inference §§2–3, 6 | Integer and outward-rounded real enclosures; generic dimensional products/quotients and laws; same-context intersection versus alternative-path join; mutable/demand-dependent premise invalidation; terminating widening; progressive narrowing and unresolved cases with source provenance. |
| **F-11(b): representation and boundaries** | Numeric Selection §§2, 5–8, 10.1, 11 | Coverage before selection; ULP-floored score across zero/subnormals and actual posit taper regions; fixed-point scales; deterministic ties; capability/emulation policy; required boundaries and directional exact/lossy transfers; bare/dimensioned seam and suboptimal-boundary explanation. |
| **F-11(c): automatic operation obligations** | Numeric Selection §§10.5.1–10.5.2; Width Inference §§7–8; Rounding §6.1 | Intermediate integer capacity, divisor/shift preconditions, deliberate modular/clamped arithmetic; exact/lossy scale changes; rounding/error propagation with a stated reference, domain and metric; established/refuted/unresolved results independent of build mode. |
| **F-11(d): preservation and tooling** | Numeric Selection §§9–11, 13; Conformance §6 | Codata survives specialization, layout and admitted lowering; M-01 artifact checks preserve actual precision, modes, transfers and arithmetic flags; premise edits retract dependent findings; full/pruned views retain evidence correspondence. |
| **C-08(a): construction admission** | Numeric Selection §§10.2–10.3.1 | Distinguish sequential rounded folds, fixed trees, compensated/error-bounded, reproducible and exact constructions; residual primitives retain operation/environment premises; rounded/exact products have distinct contracts; demand/effects and special-value observations survive elaboration. |
| **C-08(b): accumulator and merge laws** | Numeric Selection §§10.2.1, 10.3.2, 10.5 | Integer/fixed-point, IEEE superaccumulator and declared posit/quire families; exact initialization once, term multiplicity, ingest, permitted merge and finalization; capacity of every admitted partial/merge; product/accumulator dimensions; deterministic result observables at the claimed scope. |
| **C-08(c): decomposition and realization** | Numeric Selection §§10.3–10.4; applicable platform/memory contracts | Eligible partitions/orders/merge trees and preservation of exact partials across boundaries; actual operation capabilities and storage/transfer premises; cost comparison of eligible realizations with the selected representation and arithmetic meaning preserved; separate ownership, publication and progress checks. |
| **C-08(d): functional and parallel acceptance** | Numeric Selection §§10.3–10.5, 11; existing F/C contracts | Inferred measured folds/dot products through closures, records, collections and sequences; vary admitted partitions, completion order and placement; preserve a fixed tree when required; artifact checks and independent numerical oracles, including rejecting cases. ThreeBody supplies later composition pressure. |

Each inventory must enumerate operations, formats, primitive realizations and
contracts. One representative example cannot close a group containing other
required families. A sound refusal tests diagnostics; it does not replace positive
implementation coverage. Finite generated tests complement the preservation
argument. This assessment is not an implementation census or a completion estimate.

## Existing F/C owners that gain concrete cases

| Existing owner | Numeric acceptance responsibility |
|---|---|
| [F-04](F-04-CurryingLambdas.md), [C-01](C-01-Closures.md), [C-02](C-02-HigherOrderFunctions.md) | Inferred dimensions, ranges and premises survive aliases, partial/returned callables, actual environments and demand boundaries. |
| [F-05](F-05-DiscriminatedUnions.md), [F-08](F-08-OptionType.md), [F-09](F-09-ResultType.md), [F-10](F-10-RecordTypes.md) | Selected representations and dimensional relations survive payloads, branches, projections and generic instantiation. |
| [F-06](F-06-InteractiveParsing.md) | Parsing establishes its admitted value/error contract; destination capacity alone supplies no input bound. |
| [F-07](F-07-BitwiseOperators.md) | Shift definedness, range-derived signedness and intended bitwise/modular arithmetic; no width-named source types. |
| [F-02](F-02-ArenaAllocation.md), [C-04](C-04-CoreCollections.md) | Layout/extent, capacity, residence and collector state use selected representation and actual storage authority. |
| [C-03](C-03-Recursion.md) | Recurrences, recursive accumulators and tail calls preserve numeric/dimensional relationships and intermediate bounds. |
| [C-04](C-04-CoreCollections.md), [C-07](C-07-SeqOperations.md) | Fold/reduce ordering, initialization, term multiplicity and finalization; decomposition requires arithmetic permission. |
| [C-05](C-05-Lazy.md), [C-06](C-06-SimpleSeq.md), [C-07](C-07-SeqOperations.md) | Deferred reads use premises valid at demand; cached results retain established dimensions/representation; suspension/current storage and enumeration preserve identities. |

## First discriminating oracles

1. **Capacity:** two `[0,255]` inputs require nine unsigned bits for an exact sum.
   A later modulo cannot justify overflowing an earlier exact intermediate without
   a proof for the transformed expression. Include signed remainder, invalid
   shifts and zero-divisor controls.
2. **Selection:** compare offered IEEE, fixed-point and posit candidates over
   bounded, zero-crossing and extreme ranges. Independently check coverage and
   rounding/error scores; change a capability or boundary and observe dependent
   selection/diagnostics. Exhaustive small formats and exact rational/integer
   reference arithmetic provide independent host-side .NET oracles.
3. **Scale:** `(3/16)*(5/16)` is `15/256`; nearest rescaling to four fractional
   bits gives `16/256`. Preserve the `1/256` error separately from capacity and
   dimension; include a divisibility-proved exact rescale.
4. **Grouping:** the blog's binary64 `(2^54 + -2^54) + 1` and
   `2^54 + (-2^54 + 1)` retain their different specified rounded results. An
   authorized exact construction gives its specified final result under every
   admitted partition/tree. Include a fitting final total with an out-of-capacity
   partial, nonzero initialization, duplicate terms, special values and rounded
   partial-transfer rejection.
5. **Inference and effects:** use measured reductions with unannotated helpers
   and generic records/closures. Change a bound, dimensional substitution, mutable
   dependency or target mode and observe precise retraction. Unused effectful
   operands remain deferred; optimization cannot invent demand.

Independent check jobs fan out across fixtures/targets against one immutable
compiler/dependency snapshot using the .NET process runner. This tests compiler
cases concurrently; native tests separately vary the compiled program's legal
execution/decomposition choices. Use bounded focused checks during development;
close each accepted group with its complete inventory and affected F/C cohort on
recorded inputs. Compare full/pruned serialization for equal semantic outcomes
and traceable evidence, allowing their declared view difference.

## Rewrite primitives, coloring and the retained tape

The immediate requirement is a sound Baker rewrite contract compatible with
interaction-net and incremental work. It selects no net evaluator or new pass
API. The shared [nanopass direction](../Nanopass_Incremental_Contract_Direction.md#24-rewrite-independence-and-the-intermediate-tape)
owns these primitive criteria:

- Rule matches identify exact ordered hyperedge participants, occurrence/scope,
  dimensions, demand/effects, storage identity, arithmetic contract and premises.
  Sharing and annihilation preserve required observations and consumers.
- Fan-out proposals describe reads, writes/rewiring, invalidated facts and boundary
  interfaces against their snapshot. Coloring organizes compatible proposals
  after conflicts have been established. Disjoint IDs do not exclude a shared
  premise, alias, absence observation or publication conflict.
- A deterministic greedy coloring of a finite materialized conflict graph is a
  tractable initial scheduling mechanism. Optimal color count is unnecessary.
  Each batch also satisfies its joint resource/hyperedge constraints; a pairwise
  projection cannot discard a condition involving three or more participants.
  Dependency discovery, numeric proofs and optimization have their own costs.
- Fold-in validates applicability, reconciles overlaps, preserves correspondence,
  retracts invalidated evidence and resumes owning saturation. Readiness and
  coloring do not establish confluence, termination or arithmetic equivalence.
- The **tape** links source snapshot/rule identity to matched premises, proposed
  delta, applied replacement/retirement and resulting obligations. Deferred,
  rejected, superseded and cancelled proposals differ from applied rewrites.
  Physical retention and executable reachability are separate.
- Intermediate manifests preserve the pass/revision chain and inspectable
  correspondence. Pruned views retain necessary evidence or resolvable references
  to the retained trace; they cannot silently destroy a proof's source. Full views
  expose retained history, including retired work. This is compiler evidence,
  not a mandated runtime log in the native program.

Acceptance exercises overlapping/independent proposals, alternative support,
shared captures, cancellation after fan-out, stale results, annihilation with
surviving consumers, and full/pruned trace traversal. Compare firing orders against
the specified observation/evidence relation, allowing consistent fresh-ID renaming.

Baker retains semantic nanopass ownership. Alex's Huet zipper and
Elements/Patterns/Witnesses realize admitted settled structure and correspondence.
Backend circuit construction, register allocation, physical scheduling and storage
placement retain their own constraints. Reusing a coloring algorithm does not
identify these different problems.

The [PHG research note](../../../arxiv-papers/research/PHG/tractable-conflict-coloring.md)
states and proves a conditional polynomial optimal-coloring result for supplied
clique-tree certificates. It includes the semantic correspondence and retained
tape obligations required to apply that classical graph result to real recipes.
This is research support for the primitive design, not another acceptance gate
or a claim about unrestricted hypergraph coloring.

## Delivery order

Finish active generic/dimensional inference, callable/demand and storage
corrections under existing PRDs. Establish the numeric operation/format inventory
and independent oracles alongside that work. Deliver range and selection with
source-linked obligations, then scalar constructions and complete merge contracts.
Admit parallel/target realizations against established arithmetic and execution
premises, preserving the tape throughout. This assessment advances no source,
native-validation or completion claim.
