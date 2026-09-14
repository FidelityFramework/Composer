# Numeric selection and precision across strata

**Design review: September 2026 — implementation guide**

JavaScript targeting must preserve the numerical meaning established for a computation as it moves through graph elaboration, witnessed operations, target realization and communication. A change in carrier, arithmetic construction or placement can affect different properties. The compiler must explain those effects at design time and carry their justification through the affected transitions.

This guide brings [Pondering Fearless Parallelism](../../../clef-lang-site/hugo/content/blog/pondering-fearless-parallelism.md) and [Carrying Proofs into JavaScript](../../../clef-lang-site/hugo/content/blog/carrying-proofs-into-javascript.md) into Composer's implementation path. [Numeric Selection](../../../clef-lang-spec/spec/numeric-selection.md), [Width Inference](../../../clef-lang-spec/spec/width-inference.md) and the [backend architecture](../../../clef-lang-spec/spec/backend-lowering-architecture.md) govern it. The detailed companions are [Arithmetic Construction and Placement](../../../clef-lang-site/hugo/content/docs/internals/numerics/arithmetic-construction-and-placement.md) and [Proof Preservation Across Actors and Workflows](../../../clef-lang-site/hugo/content/docs/design/javascript-targeting/proof-preservation-across-actors-and-workflows.md).

The obligations below are implementation requirements. Existing integer analysis and codec checks are foundations; they do not establish a completed real-error analysis, construction selector or artifact-bound numerical proof for Composer's proposed JSIR pathway.

## 1. Keep the numerical facts distinct

| Fact | What it establishes |
|---|---|
| Kind and dimension | What quantity is computed and which dimensional operations are meaningful. Meters alone does not establish a magnitude range. |
| Range and relations | A justified enclosure of admitted values, with value identities, scale, path conditions and dependencies. |
| Representation | The available values, encoding, precision profile and exceptional-value behavior of a selected format. |
| Arithmetic construction | Term formation, intermediate state, rounding points, permitted decomposition and finalization. |
| Error contract | Reference quantity, metric, admitted domain and assumptions for an exactness or error claim. |
| Footprint and allocation | Storage extent, placement, ownership and lifetime required by the chosen realization. |

Preserve these relationships in PSG coeffects/codata while subsequent stages need them. Dimensions do not disappear at the AST boundary or become merely host type names. Selection informs footprint, and escape/lifetime evidence informs allocation; neither layout nor a destination's capacity supplies missing evidence about the source value.

An error report must distinguish input uncertainty or quantization, term-formation error, accumulation error, transfer error and numerical-method error. Their propagation requires a justified composition rule. The representation selector's worst-case error score is not an error bound for the complete calculation. A repeatable result is not necessarily accurate.

## 2. Select representation, construction and placement separately

Representation selection first uses justified range evidence and the target's offered formats, filtered by coverage and capability policy. Among eligible candidates, the specified objective minimizes worst-case representation error across the range. Cost, latency and area do not enter that score. A boundary declaration fixes a candidate but still requires coverage and transfer fidelity; a covering, admissible but suboptimal declaration is an informational finding, not permission to replace it.

The selector must handle ranges crossing zero using the specification's mixed absolute/relative treatment. Do not use an undefined pure-relative score at zero, hard-code illustrative posit taper figures, or claim tapered precision improves toward zero. The exact per-family ULP-floor convention remains an open specification item; this guide does not choose it.

Construction selection then establishes which algorithms preserve the operation's contract. A sequential fold with specified rounded additions cannot become an exact reduction simply because the latter is more accurate. Recognition proposes a candidate; it does not authorize reassociation, fusion, removal of rounding or different term formation.

Placement compares complete eligible realizations: their arithmetic, dependency graph, state, transfers and coordination. More accumulator components may remove contention or introduce additional traffic. Cost findings must distinguish estimates, measurements and justified bounds. An idle accelerator or cheaper message format cannot authorize weaker arithmetic.

For JavaScript, [Width Inference §8](../../../clef-lang-spec/spec/width-inference.md#8-target-lowering) specifies binary64 real realization and a documented wide-integer realization beyond the Number envelope. BigInt or a supported multiword construction is an implementation choice to document and validate. Do not infer posit, quire, FMA, rounding-control or accelerator access from the host's processor. Additional constructions need their actual primitive operations, host capabilities and resource contracts established.

## 3. Carry the argument through the compiler

The logical preservation chain is:

```text
Source meaning and justified input facts
    -> range evidence and representation selection
    -> permitted construction, decomposition and placement
    -> settled PSG with numerical obligations and their evidence
    -> Alex's portable witnessed operations + retained codata
    -> JSIR realization and emitted JavaScript
    -> boundary encoding, peer computation or durable reconstruction
```

This is a dependency order. It does not force every fact to settle in a single eager pass: elaboration retains unresolved constraints until the necessary source and platform context is available.

| Transition | Required preservation and ownership |
|---|---|
| Source and foreign inputs into PSG | CCS retains dimensions, range provenance and operation meaning. Generated boundary checks establish only their successful-path predicates. A foreign declaration alone does not prove the input values. |
| Elaboration into saturation | Establish coverage, selected representation and construction premises against the applicable platform facts. Real interval analysis needs sound outward enclosure and terminating procedures; integer transfer functions are not a real-arithmetic analysis. |
| Baker decomposition and fold-in | Fan-out composes recipes from Ingredients; generic fold-in incorporates them. Preserve term identities, operand evaluation, rounding points, capture relationships and the justification for each permitted merge. |
| PSG into portable MLIR | Alex observes settled structure through Library of Alexandria patterns and elements. The Huet-style zipper observes the joint constraints; witnessing does not infer missing ranges or choose a numerical policy. |
| Portable operations into JSIR and source | Realize the selected operations and modes, reading useful codata. Preserve or re-establish properties affected by coercion, precision, ordering or storage changes. Emission does not rerun the representation selector. |
| JavaScript into a foreign or BAREWire boundary | Check directional representation fidelity, encoding, extent and the required failure behavior. Retain the relation to the sending computation and receiving use. |

Alex's vocabulary remains `func`, `scf`, `arith`, `memref` and `index`. JSHIR/JSIR stays below that witness boundary. Numerical constructions must elaborate into supported portable structure; a missing realization is a support requirement, not a reason to introduce a semantic dialect or hide a numerical implementation in emission.

The selected representation is carried, not guessed again from an MLIR width or JavaScript expression. A lowering that can disturb an established property needs preservation evidence or a re-check at that edge. Neither an optimization flag nor a successfully generated solver query supplies that evidence.

## 4. Concrete JavaScript construction obligations

### Integers carried by Number

Integer addition, subtraction and multiplication can be exact when input admission preserves the intended integers and every exact intermediate has magnitude at most `2^53`. Division additionally needs definedness and divisibility where an integer result is required. Bitwise coercions cannot silently replace these operations with narrower arithmetic.

For at most `N` integer terms of magnitude at most `B`, the sufficient bound `sum(abs(x_i)) <= N * B <= 2^53` covers every subtotal in a tree of disjoint partitions. Under the specified zero policy, this supports exact merging independently of the permitted tree. A bound on the final total after cancellation would not establish the same property.

The endpoint `2^53` is representable, but a larger source integer may round to it on entry. Checking only the converted Number's range cannot prove that the original input was preserved. This argument does not redefine the host safe-integer predicate. Diagnostics must identify whether the missing premise concerns original input admission, an intermediate operation or the output boundary.

### Fixed-point scale changes

For `x = X * 2^(-f)` and `y = Y * 2^(-g)`, the exact product is `(X * Y) * 2^(-(f+g))`. Carrier capacity and scale fidelity are separate obligations. Reducing the fractional-bit count by `d` is exact when `2^d` divides the product carrier and all required operations are exact.

Otherwise, nearest rounding to spacing `Delta` has local error at most `Delta / 2`, under the specified tie rule and without clipping. The implementation must justify its emitted rounding sequence and propagate that error through later operations. A convenient host rounding function is not evidence that its tie rule matches.

For example, carriers `3` and `5` at scale `2^-4` multiply to `15/256`. Returning to scale `2^-4` gives `16/256` under nearest rounding: error `1/256`, with no overflow. The diagnostic belongs at the rescaling operation and should explain its downstream contribution to the required error bound.

### Rounded trees and residual arithmetic

With binary64 round-to-nearest/ties-to-even, the represented inputs `2^54`, `-2^54` and `1` produce `1` or `0` under different groupings. A fixed input-indexed tree preserves its specified grouping as completion order changes; changing worker subtotals can still change that tree. Race freedom or a serialized mailbox does not establish arithmetic equivalence.

A rounded tree can have a justified error bound without being exact. The proof companion gives the bound `abs(computed - exact) <= gamma_d * sum(abs(x_i))`, where `gamma_d = d*u/(1-d*u)`, maximum leaf depth is `d`, `u = 2^-53`, and `d*u < 1`. It requires the stated relative-error model at every addition, excluding overflow and underflow cases that invalidate that model. It is an absolute bound relative to the represented inputs; cancellation can still make relative error against the exact sum large.

TwoSum instead retains a rounded high component and its exact residual. Its real-value equality `value(high) + value(low) = value(a) + value(b)` requires finite operands, round-to-nearest/ties-to-even, gradual underflow and no intermediate overflow. Preserve the operation sequence. Immediately adding the two components into one Number can discard the residual again. Two components do not constitute an unlimited exact accumulator or establish associative merging of arbitrary partial states.

### Exact accumulation and finalization

An exact construction needs an interpretation `D` of its accumulator and justified initialization, ingestion, merge and finalization:

```text
D(empty) = 0
D(ingest(q, t)) = D(q) + value(t)
D(merge(q1, q2)) = D(q1) + D(q2)
finish(q) = the prescribed rounding of D(q)
```

These laws apply to admitted states, with capacity established for every permitted local partial and merge intermediate. A nonzero source initial value contributes once to the entire reduction. Equivalent accumulator denotations need not have identical internal bytes; bit-identical output additionally needs deterministic finalization and the same required result observables.

The JavaScript proof companion describes a candidate exact binary64-term construction: decode each finite represented input as `k * 2^-1074`, sum integer coefficients exactly, then round the exact state once. BigInt or multiword storage still requires operation, allocation and execution-budget evidence. Converting the coefficient to Number before applying the scale is not a valid general finalizer: premature rounding or overflow can destroy the result.

Exact sums of rounded products and exact sums of exact products are different contracts. Input quantization and numerical-method error remain in either case. NaNs, infinities, signed zero and any observable exceptional behavior require explicit policies; the finite-value argument does not cover them automatically.

## 5. Preserve precision across transport and suspension

A native endpoint, JavaScript endpoint and BAREWire field may use different representations. Record transfer fidelity directionally: exact transfer requires both coverage and exact representability for every admitted source value. A wider destination does not restore accuracy previously lost by the source, and exact transfer in one direction does not establish an exact round trip.

Intended loss belongs in explicit arithmetic with an analyzed range and an admitted error contract. Do not silently insert a lossy conversion, wrap, clamp or runtime representation switch to satisfy a boundary. A permitted lossy transfer must have its error accounted for at the receiving computation.

An exact global reduction must carry exact partial information until its permitted finalization. It may transmit an accumulator state, a smaller encoding proved exact for all admitted partials, or a final result when the sender completes the whole reduction scope. Sending a rounded subtotal to save bytes changes the construction unless that rounding is proved exact. Perfect byte delivery cannot repair this loss.

BAREWire's structural contract relates dimensions, scale, field meaning and encoding at the endpoints. Useful type and proof information remains in the PSG and lowering context until its preservation work is complete. The final payload needs no self-describing compiler type, dimension, schema or proof tags. Presence flags, lengths and required job/partition identifiers remain ordinary contract data. Encoding agreement, decoding checks and numerical fidelity are separate obligations.

At an `await`, an immutable scalar capture retains its value and established range. A range inferred from mutable contents needs evidence that those contents remain valid, a retained immutable observation, or permitted revalidation before use. Mere shape validation of a returning value does not establish its input snapshot or contribution identity.

For a zero-initialized exact all-results join, the proof companion uses `A subset-of P` and `D(q) = sum(v_p for p in A)`: accepted partitions belong to the expected set, and the accumulator denotes exactly their contributions. Completion requires `A = P`. Associativity and commutativity permit regrouping and reordering, not duplicate insertion. Retry, conflict and stale-result policies preserve the numerical theorem's domain.

Recovery must retain accepted identities and numeric state consistently, or reconstruct them under an established arithmetic contract. Recomputing from saved inputs with a different rounded tree is not automatically equivalent. Cloudflare supplies the declared execution and storage facilities; application proofs depend on their contracts and do not establish host implementation correctness or eventual completion.

## 6. Diagnostics at the point of commitment

[Deferred inference](../../../clef-lang-site/hugo/content/blog/deferred-inference.md) permits consistent partial programs while constraints accumulate. Applicable numeric obligations are generated during ordinary compilation, independently of optimization level or debug assertions. Developers supply missing intent or justified domain facts; they should not need per-operation wrappers to activate the analysis.

Atelier/LSP presents CCS findings with source range, related nodes, reachability, target context and evidence provenance. It does not run a separate numeric selector. A useful readout explains the quantity and range, selected representation, construction and rounding points, permitted decompositions, boundary effects and unresolved premises.

| Finding | What the diagnostic must distinguish |
|---|---|
| Pending range or platform facts | A choice still open during elaboration; locate its origin and dependent use. |
| Known coverage failure | No offered admissible representation covers the established range, or the declared boundary does not cover it. No fallback is permitted. |
| Unresolved proof | Missing input bound, conservative enclosure, unsupported theory or resource limit. These are not successful proofs or demonstrated failing executions. |
| Inconsistent evidence | Conflicting applicable premises; neither source gets authority to override the other. Unreachability requires its own justification. |
| Precision or exactness failure | Coverage may hold while a rounding, rescaling, term-formation or transfer obligation fails. Identify the affected error contract. |
| Missing merge permission | A proposed partition or tree changes specified rounding, or lacks capacity/law evidence. |
| Unsupported realization | A required primitive, arithmetic mode, storage or host capability is unavailable. Do not weaken the operation. |
| Admissible suboptimal boundary | Honor the declaration and report the open selector's alternative, using the specification's informational status. |

For dimensioned reals, unresolved required range evidence becomes a located error at representation commitment. The bare-float `f64` exception is a representation policy when offered and permitted; it proves no range, precision or merge obligation. If an unresolved bare value becomes dimensioned, locate the obligation at that seam and identify the upstream bare source.

These are illustrative diagnostic contents, not new syntax, diagnostic codes or completed tooling:

```text
Pending: force reduction / JavaScript target
  Required: exact accumulation of the admitted represented terms
  Established: each integer term has magnitude at most B
  Missing: a term-count bound or invariant covering every permitted subtotal
  Consequence: Number exact-merge eligibility is not established
  Related source: partition generation and the input-bound declaration
  Required before: committing this construction
```

```text
Cannot preserve exact partial state at this result-message boundary
  Sender: exact accumulator awaiting further merges
  Proposed payload: one rounded binary64 subtotal
  Missing: exact representability of every admitted partial in that payload
  Required: an exact partial encoding, or finalization of the whole scope
  A lossy result requires an explicit different numerical contract
```

Successful diagnostics should be equally concrete: identify the proven bound, rounding rule and permitted partition scope, rather than label a whole program “safe” or “accurate.” Error bounds must name their reference and units or normalization. Suggested rescaling needs a checked transformed range; a unit-name change alone does not prove coverage.

A runtime check can establish a premise only where the source or boundary contract permits it and defines failure. It cannot silently substitute for a required static guarantee. Establish input fidelity before relying on a check after conversion. Preserve fact validity through mutation, foreign calls and suspension.

## 7. Implementation progression and acceptance

The work proceeds along the existing compiler boundaries:

1. **Contracts and evidence carriage.** Retain numeric kinds, dimensions, value identities, bounds, arithmetic requirements and target facts, distinguishing established, refuted and pending obligations. Specify the operation families and host realization being admitted.
2. **Analysis and design-time explanations.** Generate capacity, scale, error and decomposition obligations from supported graphs. Instantiate accepted construction laws against those facts; expose unresolved requirements and invalidate evidence when its dependencies change.
3. **Baker elaboration and Alex witnessing.** Elaborate an eligible construction through the established fan-out/fold-in structure. Check that portable operations preserve terms, captures, rounding and merge relationships before crossing the witness boundary.
4. **JavaScript realization.** Establish correspondence for the emitted primitive operations, wide state, rounding and boundary encoding. Bind preservation evidence and host premises to the selected toolchain and artifact.
5. **Communication and recovery.** Extend the same numeric argument through partial-state transfer, suspension and consistent acceptance. Validate the selected Cloudflare realization using its existing host facilities.

This progression introduces no separate numerical middle end. The exact construction registry, evidence encoding, general error-composition algorithms, some platform fact contracts and ULP-floor details remain open in the governing specification. Resolve those items before claiming the corresponding implementation; do not fill them with undocumented defaults.

Extend [the folder's acceptance sequence](07_dependency_identity_and_validation.md) with these focused cases:

| Case | Required observation |
|---|---|
| Fitting final result with an oversized intermediate | The final bound cannot justify that intermediate realization. |
| Input near the Number exact-integer limit | Detect lost input identity before a post-conversion range check can conceal it. |
| Exact and inexact fixed-point rescaling | Distinguish divisibility from rounding; verify tie, negative-value and propagated-error behavior. |
| Binary64 cancellation under two trees | Preserve a specified tree; do not claim arbitrary regrouping from race freedom. |
| TwoSum with cancellation and near underflow/overflow | Establish its admitted domain and preserve the residual operations through actual emitted code. |
| Exact reduction across permitted partitions | Preserve terms, initial-value multiplicity, every partial's capacity and deterministic finalization. |
| Cross-target partial transfer | Reject unproved rounded subtotals; verify exact encodings, scale, byte order and directional fidelity. |
| Mutation during suspension and restart around acceptance | Invalidate stale premises and preserve the relation between accepted identities and accumulated state. |
| Duplicate, conflicting or stale replies | Preserve the declared contribution set and conflict policy independently of merge associativity. |
| Build-mode and failed-analysis cases | Keep obligations active; timeout or unsupported theory remains unresolved. |

Inspect emitted operations and exercise mutations that alter a rounding point, coerce an integer, lose a residual or round a transmitted partial. Tests must demonstrate that the relevant checks detect those changes. Mathematical construction arguments, solver results, artifact preservation and host execution tests supply different evidence; retain their scope separately.
