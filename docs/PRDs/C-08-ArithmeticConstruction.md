# C-08: Arithmetic Construction and Reduction

**Status: In-Progress. Accepted September 26, 2026.**

## 1. Result and governing contract

Composer realizes numeric computation regions with the arithmetic, intermediate
state, decomposition and finalization required by their source contracts. It can
exploit admitted parallel structure while preserving the requested result and
every prerequisite on memory, publication and progress.

[Numeric Selection §§10–11](../../../clef-lang-spec/spec/numeric-selection.md#10-the-preservation-chain-and-arithmetic-construction)
governs this PRD. [F-11](F-11-NumericSelection.md) establishes the shared numeric
facts and operation obligations. [C-04](C-04-CoreCollections.md) and
[C-07](C-07-SeqOperations.md) preserve their collection/sequence operation contracts;
their reductions consume this construction discipline. [M-01](M-01-DialectAdmission.md)
owns physical vocabulary and target admission. A/T/R and platform contracts retain
their actual execution and resource responsibilities.

[Pondering Fearless Parallelism](../../../clef-lang-site/hugo/content/blog/pondering-fearless-parallelism.md)
provides examples and motivation. The source/specification contract determines
whether a rounded fold, fixed tree, error-bounded, reproducible or exact result is
required. Improved accuracy alone does not authorize changing that contract.

## 2. Construction and semantic ownership

A construction identifies its actual input/output representations, admitted
operands, term formation, state denotation, primitive operations, rounding points,
initialization, permitted partitions/orders/merges, finalization, result
observables and proof premises. Its resource and execution requirements accompany
the arithmetic evidence. Selection records which requirements are established
and what would invalidate them.

Baker uses reusable ingredients and recipes to elaborate the construction as PSG
structure. An ingredient exposes its operation and premises; a recipe establishes
them at the actual participants. Recognition creates a candidate. Owning analyses
settle its obligations before commitment. Fan-out/fold-in preserves the joint
relations and [rewrite tape](../Nanopass_Incremental_Contract_Direction.md#24-rewrite-independence-and-the-intermediate-tape).
Source demand, effects, sharing and occurrence-specific context remain intact.

Alex observes settled structure through the Huet zipper, Elements, Patterns and
Witnesses. It preserves actual operand/state/output correspondence and emits the
admitted portable forms. Backend realizations establish their own primitive modes,
circuit operations, scheduling, storage and transport requirements. FPGA/Colibri
integrity supplies acceptance discipline without moving circuit algorithms into
the middle end.

## 3. Acceptance groups

### C-08(a): construction admission and scalar ingredients

- Preserve specified sequential rounded evaluation and fixed trees, including
  source effects, repeated/shared demand and exceptional observations.
- Admit residual primitives such as TwoSum and product residuals only with the
  required precision, rounding, underflow and FMA premises. Preserve their actual
  operation graph against invalid reassociation/contraction.
- Specify compensated, binned/reproducible, superaccumulator and quire constructions
  by their complete contracts. A compensation component alone establishes no
  exactness, arbitrary-merge or order-independence claim.
- Distinguish sums of rounded products from exact products of represented inputs.
  Neither construction recovers input information lost before term formation.
- Account for signed zero, nonfinite values, NaR, exceptional conditions and
  observable status. Exclusions need established input/intermediate contracts.
- Keep recognition, applicable premises, successful discharge and commitment
  distinguishable, with precise diagnostic origins.

### C-08(b): accumulator denotation, capacity and merge

For an exact construction with denotation D and admitted term t, establish:

```text
D(initialize(v)) = v
D(accumulate(a, t)) = D(a) + t
D(merge(a, b)) = D(a) + D(b)
finalize(a) = the contract's prescribed result from D(a)
```

- Include the required nonzero initial value exactly once. Preserve terms and
  multiplicities across empty, singleton, partitioned, nested and repeated use.
- Prove every term, local partial state and merge intermediate fits for every
  permitted decomposition. A small final result after cancellation is insufficient.
- Preserve physical dimensions and fixed-point scale through product formation,
  ingestion, merge and finalization. Different redundant state encodings may
  denote the same exact value; result-bit claims need deterministic finalization
  and all required special-value observations.
  Dimensional normalization does not insert a numeric scale conversion; an actual
  rescaling retains its source/boundary contract and F-11 fidelity obligations.
- Cover exact integer/fixed-point state, IEEE superaccumulators and the declared
  posit/quire families. Standard posit quire width is 16n; the specified bounded
  format uses 800 bits for n>12. Validate actual format field layout and finite
  capacity; width alone supplies neither unlimited accumulation nor primitive
  operation semantics.
- Preserve exact partial information across boundaries. Rounding a partial to an
  output scalar before merging is eligible only when that conversion is proved
  exact on every admitted partial and preserves the contract's observables.

### C-08(c): admissible decomposition, joint constraints and realization

- Establish which operations can overlap without changing the required tree and
  which partitions/orders/merge trees the numerical contract permits.
- Retain exact work/contribution identities, initialization and multiplicities
  through fan-out/fold-in. A retry/duplicate result cannot become another term;
  a late result cannot attach to a new computation instance.
- For compiler rewrite grouping, derive sound read/write/rewiring conflicts,
  including aliases, proof premises, absent lookups and shared publication state.
  Use tractable valid coloring without requiring a minimum color count. A
  [checked clique-tree certificate](../../../arxiv-papers/research/PHG/tractable-conflict-coloring.md)
  can support a restricted optimality claim for its explicit conflict graph.
- Validate complete joint resource conditions when admitting a batch. Pairwise
  compatibility does not establish multi-party capacity. Graph coloring does
  not establish confluence, rewrite termination or arithmetic equivalence.
- Require concrete platform operation facts: precision, rounding, subnormal and
  exception behavior, contraction/reassociation permissions, capacities, lane/carry
  semantics, storage/alignment and transfer/publication requirements.
  Writes or calls that change a dynamic rounding/subnormal mode invalidate the
  dependent operation evidence; an unchanged output-format name does not preserve it.
- Compare complete eligible realizations, including private state, merges,
  movement and coordination. Preserve F-11's selected representation and required
  arithmetic. Distinguish estimates, measurements and justified bounds.
- Keep ownership, publication, lifetime and progress as separate obligations.
  Shared cache-line separation requires actual aligned bases and padded extents.
  Safe numeric merge does not establish race freedom or completion.

### C-08(d): functional, parallel and artifact acceptance

- Compile inferred measured reductions through generic helpers, closures,
  records/tuples, collection operations and deferred sequences. Preserve the
  actual environment, memoized state and demand-valid numeric premises.
- Vary worker count, chunk boundaries, legal tree shape and arrival order under
  each construction's contract. Fixed-tree cases preserve index-defined grouping;
  exact cases preserve their denotation and deterministic specified finalization.
- Validate actual native/target arithmetic modes, state layout, capacities,
  transfers and operation correspondence after lowering. Source-only proof
  success does not certify changed artifact instructions or modes.
- Exercise rejection/retraction for insufficient partial capacity, lossy exact-state
  transport, changed rounding/subnormal/FMA facts, invalid decomposition,
  stale rules/participants and unsupported target realization.
- Inspect full and pruned intermediate tapes, including annihilation and retired
  participants. Maintain the same semantic outcome and resolvable evidence;
  historical records do not resurrect execution or authorize current facts.
- Compare edited-region settlement with a fresh run, including changed terms,
  dimensions, capacities, merge rules and target facts. Revalidate crossing joint
  constraints before reusing a witnessed segment. Superseded workers cannot
  publish terms, proofs, MLIR or executable code into the accepted generation.
  The [segmented publication contract](../Nanopass_Incremental_Contract_Direction.md#25-edit-transactions-proof-reuse-and-segmented-publication)
  governs AOT and REPL realization of these regions.

## 4. Required construction matrix

| Family | Required positive contract | Discriminating control |
|---|---|---|
| Sequential rounded fold | Exact prescribed grouping and rounding points | Cancellation example changes if regrouped; retain original result |
| Fixed tree | Index-defined tree, independent branch scheduling | Vary worker completion without varying arithmetic structure |
| Compensated/error-bounded | Complete initialization/update/finalization error argument | Unsupported underflow/mode or arbitrary merge cannot inherit the bound |
| Reproducible binned | Declared decomposition scope and result observations | Reproducibility alone establishes no exactness or ideal-model accuracy |
| Integer/fixed-point exact reduction | Exact terms, every partial/merge fits, scale retained | Fitting final result with an overflowing partial; lossy rescale |
| IEEE exact accumulation | Exact represented terms and specified final rounding | Exact products differ from rounded-product terms; capacity exhaustion |
| Standard posit quire | Actual selected format, 16n state and adequate intermediates | Rounded posit addition and rounded partial transport lack the exact contract |
| Bounded posit quire | Declared format, 800-bit state for admitted n>12 and adequate intermediates | Same width does not imply the same instruction/format/resource capability |

Scalar/native realizations establish initial positive gates. Target-specific
CPU/SIMD, GPU, FPGA or other realizations require their own complete admitted
profiles; a test on one does not close another's gate. The family inventory cannot
be reduced to a single convenient algorithm or a permanently rejecting path.

## 5. Validation and completion

The [numeric validation inventory](Numeric_Validation_Cases.md) records concrete
reference quantities, admitted domains and exact/error/trace oracles. .NET hosts
independent rational/integer and finite-format reference models; Clef source
retains its native numeric universe and ordinary compilation checks.

ThreeBody supplies later integrated pressure with separate numerical quality,
reproducibility and execution-cost observations. It does not replace small
adversarial cases. A conserved scalar or attractive trajectory alone is not an
independent accuracy oracle; reference precision and timestep need convergence
evidence over the reported interval.

Close C-08 only after all four groups and required construction families pass
source, settlement, witness, artifact and execution gates at their declared
scope, with refusing/mutation controls. Run the complete inventory and affected
F/C cohort against one immutable input snapshot. The parallel .NET compiler-test
harness and parallel execution inside a compiled program have separate oracles.
