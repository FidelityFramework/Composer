# Numeric validation cases

This inventory registers the case identifiers used to close
[F-11: Numeric Selection and Numeric Obligations](F-11-NumericSelection.md) and
[C-08: Arithmetic Construction and Reduction](C-08-ArithmeticConstruction.md).
`F11a01` means a case in F-11(a); `C08b01` means a case in C-08(b). Every row is a
required executable contract. An existing supporting test covers only the
judgment stated here; it does not close a larger representation, construction,
target or incremental gate.

The governing requirements are [Numeric Selection §13](../../../clef-lang-spec/spec/numeric-selection.md#13-normative-requirements),
its cited arithmetic and boundary clauses, [Units of Measure](../../../clef-lang-spec/spec/units-of-measure.md),
[Width Inference](../../../clef-lang-spec/spec/width-inference.md) and the
[incremental contract](../../../clef-lang-spec/spec/incremental-computation.md).
The expressions and equations below specify fixtures and oracles. They do not
introduce a library API, annotation syntax or backend primitive.

## 1. Evidence recorded for every case

Each execution record contains:

- Case identifier and the exact positive, refusal or mutation variant; source,
  dependency, compiler and platform-content identities, including dirty contents.
- Actual native kind and normalized dimension; quantified substitutions or pending
  constraints; value identities, range, scale, representation and applicable context.
- Required obligations, premises, rule versions and outcomes. Established,
  refuted, inconsistent and unresolved results remain distinct. A timeout or proof
  work limit is unresolved; it is not an execution counterexample.
- Diagnostics with source span and relevant participants, including the target or
  boundary when it supplies a premise. A refusal must have the intended cause.
- Full/pruned PSG correspondence, the actual portable operation and target
  realization where claimed, plus native output, bits, trace or error relation.
- For an edit, the changed region, dependency closure, invalidated/reused results
  and interfaces, generation identifiers, and an independent fresh-build result.

Run .NET independent integer/rational and finite-format reference calculations
separately from compiler calls. An oracle passing its own tests supplies no
compiler evidence. Expected inference answers must be compared without unifying
them into the compiler's inference cells. Decimal display and tolerance alone
are insufficient for a bit-exact result; rounded and error-bounded cases name
their metric and reference quantity.

Fixture targets use explicit admitted platform/operation declarations. No case
may obtain a width from the host's `int`, a range from a declared capacity, or a
representation from the spelling of a source kind. Platform profiles and
construction contracts must supply their actual modes and capabilities before
the corresponding positive run can count.

## 2. Existing executable foundations and their limits

The following are concrete starting points, not an assertion that this complete
inventory has passed. Re-run them in the final immutable cohort.

| Evidence | Existing owner and established scope |
|---|---|
| E1 | [DimensionalInferenceCases](../../../clef/tests/Clef.Compiler.Service.Tests/DimensionalInferenceCases.fs): 23 cases passed against the September 26 source-v13 DLL, then in the 1,499-test source checkpoint committed as `d5ae0d9`. Covers inferred compositions, coupled exponents, weak mutable constraints, tuple/record projection, known formal context, alias/partial/inline use and record ambiguity. This is source inference evidence, not numeric selection or incremental execution. |
| E2 | [DimensionalCases](../../../clef/tests/Clef.Compiler.Service.Tests/DimensionalCases.fs), registered through [DimensionalTests](../../../clef/tests/Clef.Compiler.Service.Tests/DimensionalTests.fs): dimensional algebra, nominal identity, generic measure kinds, conversions and exact wide literals. The shared-label fixture now selects explicit nominal owners, retains inferred measures and requires the original ambiguous literal to fail. |
| E3 | [LoopRangeCases / FiniteRecurrenceCases](../../../clef/tests/Clef.Compiler.Service.Tests/LoopRangeCases.fs): finite additive, multiplicative and ordered coupled recurrences, final exhaustion stores, dimensional/provenance mutations and proof-work exhaustion. The [SMT transfer regression](../../tests/SMTTransferRegression.fsx) independently checks finite-recurrence certificate dispatch through both solver routes; its 31 finite-recurrence cases passed in the recorded 116-case solver cohort. Neither result proves a continuous real transfer law. |
| E4 | [Original 15_SimpleSeq](../../samples/console/FidelityHelloWorld/15_SimpleSeq): the unchanged expected output passed with the explicit eager Fibonacci snapshot and source formatting correction recorded in the [waypoints](../Language_Coverage_Waypoints.md). It supplies actual sequence/recurrence execution evidence; it is not a posit, quire or error-analysis case. |
| E5 | [LazyRangeCases](../../../clef/tests/Clef.Compiler.Service.Tests/LazyRangeCases.fs), [CallEffectRangeCases](../../../clef/tests/Clef.Compiler.Service.Tests/CallEffectRangeCases.fs), [EagerWitnessTests](../../tests/Alex.Tests/EagerWitnessTests.fs) and [SequenceTransportTests](../../tests/Alex.Tests/SequenceTransportTests.fs) exercise actual demand, captured identities and occurrence/scoped transport. Use their relevant exact cases as prerequisites; they do not establish arbitrary numeric memoization or reduction laws. |
| E6 | [SpecializationTraceCases](../../../clef/tests/Clef.Compiler.Service.Tests/SpecializationTraceCases.fs) and the [nanopass contract](../Nanopass_Incremental_Contract_Direction.md) exercise/define retained specialization evidence. The source-v13 tape-focused cohort passed; retained history and full/pruned correspondence alone are not incremental reuse. |
| E7 | The [.NET regression runner](../../tests/regression/Runner.fsx) and [ParallelRunnerTests](../../tests/regression/ParallelRunnerTests.fsx) provide bounded independent compiler jobs and isolated artifacts. They test the host harness; compiled-program parallel decomposition needs its own oracle. |
| E8 | [ConstructionOracles.fsx](../../tests/numerics/ConstructionOracles.fsx): 40 exact rational/reference-rounding checks for the named F11c02, C08a01–a03 and C08b01–b02/b05 fixtures. This .NET script validates fixture reference quantities and discriminating controls; it invokes no compiler and certifies no compiler gate. Its [domain](../../tests/numerics/README.md) excludes nonfinite results and signed-zero/status observations. |

Record the precise test method and run manifest used when attaching one of these
foundations to a case below. A later source change retracts an older run as proof
of the changed compiler, even if the test name is unchanged.

## 3. F-11 cases

### F-11(a): inference, ranges and evidence

| ID and case | Required fixture, oracle and discriminating controls | Existing support / completion evidence |
|---|---|---|
| **F11a01 — Principal measured composition** | Infer unannotated inverse/square composition at independent `m` and `s` uses; expect inverse-square dimensions. Solve the two exponent equations from product and quotient jointly, including an integral solution and a individually plausible but jointly nonintegral refusal. Compare native kinds and normalized dimensions without adding expected-answer constraints. Validate each actual substitution in public arguments, result, body and retained captures. | E1/E2 source support. Add native/source-artifact correspondence for the selected representations; scheme-equivalence cases must compare admitted instances, not printed variable names. |
| **F11a02 — Quantified, weak and pending constraints** | Distinguish a generalized measure from an unresolved shared mutable measure and an unresolved range. Add context in separate source revisions to settle a weak dimension; contradictory writes must refuse. Preserve constraints from a shared cell while allowing independent factory instances. An unresolved member relation must remain attached to its scheme/use or produce an explicit pending/refusal result. `let project x = x.Value` must not become an unconstrained `forall a b. a -> b` that admits `project 1`. | E1 covers measured/weak-state distinctions. The member example is a concrete source-v13 admission defect and is a required owning inference regression. Source revisions in E1 are fresh checks, not incremental evidence. |
| **F11a03 — Contextual aggregates and scope** | Infer measured tuple, nested/inline tuple and nominal record projections. A known actual formal must disambiguate a record while sharing the same fresh measure instance with its result; vary direct, alias, partial and inline paths. Include qualified/expected owner, hidden labels, uniquely complete literals, identical complete owners, wrong dimensions and wrong qualified owners. Nested aggregate contexts retain their own structural type relations. | E1/E2 cover the named direct/tuple cases. Require the broader aggregate/call graph through actual source and native consumers; nominal ambiguity is not repaired by choosing declaration order. |
| **F11a04 — Same dimension, distinct magnitudes and scales** | Use two quantities with the same physical dimension and justified but widely separated magnitude intervals. Selection must consume each interval. Separately encode one physical quantity under two explicitly related unit/representation scales and verify the exact value conversion and transformed interval. A generic measure exponent, a numeric multiplier and a fixed-point radix scale are different facts. Mutate the scale relation without changing the dimension and retract dependent bounds/choices. | Required source-analysis and native fixtures with independent rational value/scale oracle. No unit-name heuristic or implicit rescaling is an admissible implementation. |
| **F11a05 — Sound integer and real transfer domains** | Check all multiplication sign combinations, division with a denominator bounded away from zero and a zero-crossing refusal, and admitted `sqrt/log/exp/sin` domain laws. Integer and real domains have distinct transfer rules. For real enclosures, verify containment using exact or outward-rounded reference evaluation, including endpoints, interior extrema, cancellation and values beyond host-word precision. | E2/E3 support integer/dimensional portions. Continuous real/law cases require compiler-level enclosures and premise invalidation, not a standalone calculator. |
| **F11a06 — Conjunction, alternatives and mutable provenance** | Intersect sound enclosures for the same value/context; join branch alternatives. Cross-instance or incompatible-path facts must not intersect. Distinguish inconsistent reachable evidence from a proved unreachable branch. Change a captured value, relevant write/call, law premise or negative lookup and retract exactly affected conclusions. Deferred reads use demand-time premises; memoized results keep their actual computation identity. | E3/E5 provide restricted premise/range controls. Complete real and imported-law support remains part of this case. |
| **F11a07 — Recurrence and bounded proof work** | Check zero/one/many iterations, descending strides, coupled Fibonacci with an actual eager snapshot and multiplicative powers, including final stores after the last yield. Add measured state and dimensionless factors; incompatible measures refuse even if numeric inequalities hold. Cross the analysis-work budget deterministically: leave the obligation pending and block required commitment without selecting a register fallback. | E3/E4 positive and adversarial foundations. Add real recurrences, widening thresholds and continuous-error obligations at their owning domains. |

### F-11(b): selection and boundaries

| ID and case | Required fixture, oracle and discriminating controls |
|---|---|
| **F11b01 — Coverage before optimization** | Supply explicit offered IEEE, posit and fixed-point candidates. Enumerate a finite-format reference domain where feasible and independently establish continuous interval coverage. A candidate with an attractive bounded error score but insufficient range must be removed before scoring. Empty coverage gives CCS8012 with range/offers/boundary; absent platform or unresolved range remains a different obligation. Include exact boundary endpoints and one value beyond each. |
| **F11b02 — Zero crossing and actual format behavior** | Evaluate singleton zero, intervals touching/crossing zero, normal/subnormal transitions, near-unity and regime extremes. Use the specified ULP floor and actual format rounding, posit taper and fixed-point spacing. Check positive/negative asymmetry and ties; no undefined argmin or hard-coded posit endpoint approximation. |
| **F11b03 — Accuracy objective and deterministic choice** | On the same eligible candidate set, vary recorded latency/area/cost without changing the accuracy winner. Vary policy filters separately and recompute eligibility. Permute candidate declaration order; equal-score selection must follow the admitted deterministic tie rule. Distinct same-dimension ranges from F11a04 can select distinct representations. |
| **F11b04 — Required representation and exact direction** | Constrain a boundary to one actual representation. A covered but suboptimal boundary must compile with the specified information diagnostic; a non-covering boundary fails CCS8012. Coverage alone must not admit an inexact transfer claiming exactness. Reverse source/destination direction and test representability again. Include ABI, MMIO and wire instances with their real owning declarations. |
| **F11b05 — Native, emulated and unavailable** | Run native-only, allow-emulated and allow-emulated-warn declarations over the same value. Refuse unavailable choices and absent required exact-accumulation operations. Changing capability/mode facts invalidates the choice. Emulation eligibility does not substitute for per-operation precision, resource or transfer evidence. |
| **F11b06 — Bare exception and dimensioned commitment** | An unobservable bare real uses offered/permitted IEEE f64 only under the specified exception; otherwise report capability failure. A known out-of-range bare value still fails coverage. A dimensioned unobservable real stays pending during elaboration and fails at required commitment unless later actual evidence settles it. No default host float or width is admissible. |
| **F11b07 — Bare-to-dimensioned seam** | Flow an unbounded bare value into a measured result through a helper/alias. The obligation names the dimensioning origin and upstream bare source. Add a justified bound in a later source context to admit it, then remove/change that premise to retract settlement. An unrelated bounded input must not discharge it. |
| **F11b08 — Multi-hop transfer fidelity** | Compile a source representation through an ABI/wire/temporary representation into the destination. Verify each exact transfer or each stated lossy bound and their composition. Mutate only the middle representation, rounding mode or scale. Endpoints having the same type must not hide a lossy middle step. |

### F-11(c): operation obligations

| ID and case | Required fixture, oracle and discriminating controls |
|---|---|
| **F11c01 — Integer intermediates and definedness** | Exercise arbitrary justified integer widths, signed/unsigned boundaries, every relevant intermediate, division by zero, shift validity and signed remainder. A final fitting result cannot cover an overflowing intermediate. Compare exact arithmetic with intentional source modulo/clamp separately; inspect actual extensions and operations. E2/E3 supply limited wide-integer and recurrence support. |
| **F11c02 — Fixed-point scale and information loss** | Establish exact `3/8 + 1/16 = 7/16` and `(3/8)*(5/16) = 15/128` under actual scale/carrier contracts. Rescale `15/128` to sixteenths with the declared rounding/error, and use midpoint cases to distinguish tie/directed modes. Refuse an exactness claim across lossy rescale, inadequate intermediate capacity or unproved divisibility. Keep clipping and quantization distinct. |
| **F11c03 — Error reference and dependency** | State ideal-model, represented-input and specified-rounded references separately. Exercise cancellation, correlated operands, intermediate precision and rounding points. Bound representation, arithmetic and method errors under their own premises; changed input uncertainty or target modes must invalidate the relevant guarantee without asserting a different reference quantity. |
| **F11c04 — Automatic checks and authorized guards** | Compile the same ordinary expressions under every supported build mode, with no enabling wrapper. Assert the same required obligations. Refuted and pending cases must not become successful due to optimization level. A runtime guard can establish only its specified successful-path facts; test failure diagnostics/termination and ensure no silently inserted guard replaces a mandatory static proof. |

### F-11(d): correspondence and incremental execution

| ID and case | Required fixture, oracle and discriminating controls |
|---|---|
| **F11d01 — Representation through actual artifacts** | Match settled kind/dimension/range/representation identities to portable operands, target operations and final result. Mutate width, extension, rounding mode, fast-math flag or rescale after a positive run and require failed correspondence. A matching byte count does not establish semantic equivalence. |
| **F11d02 — Diagnostic ownership and current generation** | Distinguish coverage, capability, inconsistent premises, pending proof and accuracy information at exact source/boundary spans. Repair then rebreak a case while an older proof is outstanding. Publish only diagnostics authorized by the latest generation; an older success/error must not overwrite it. |
| **F11d03 — Full/pruned evidence and deterministic fan-out** | From one immutable input, run full and pruned artifacts and bounded compiler fan-out at several worker counts. Compare semantics, selected representations, obligation outcomes and resolvable tape correspondence, allowing irrelevant allocation/order IDs to differ only under a justified mapping. Retired participants remain evidence, not executable roots. E6/E7 are prerequisites, not proof of numeric grouping or incremental reuse. |
| **F11d04 — Edit closure and fresh equivalence** | Change a local range, measure, capture, write, imported law, absent lookup, target offer or proof rule. Compute the actual change/dependency closure, retract its dependent results, and compare with a fresh compilation of the new source. Add/remove a record construction or array store that changes a shared field/element range or layout consumed in another region; preserve the full joint support including membership/absence. Include live→dead→live, removed/added branches and independent-scope controls. Equal answers alone do not demonstrate reuse; record which owning analyses actually reused valid evidence. |
| **F11d05 — Repartitioned regions and object reuse** | PSG hyperedges and dependency evidence determine the affected semantic region and Alex's actual re-witnessing scope. Compile those settled regions through witnessed segments, object files and LLD. A local edit preserving a validated interface may reuse unaffected objects; prove reuse from actual object/interface identities. Change crossing relations to split, merge and replace regions; allow rebuilding their affected artifact area and require obsolete definitions/initializers to leave the new manifest. A changed numeric representation, ABI, capture/environment layout, demanded effect or publication contract invalidates all affected dependents. Compare symbols, relocations, logical storage contents and behavior with a fresh build, allowing valid relocation/address differences. A common symbol name, flat-module partition or ELF shape is insufficient. |
| **F11d06 — ORC generation and stale materialization** | On the actual native interactive path, pause an old region's materialization, change a numeric carrier/interface or captured premise, and then complete the old job. Refuse stale publication/invocation, preserve the new generation's diagnostics and execute only validated current artifacts. Repeat with a lawful unchanged interface to demonstrate admitted reuse. Retain already-admitted older frames/callbacks until their lifetime ends; incompatible state requires admitted migration or restart/refusal before replacement. Exercise bounded debounce under edit storms, explicit run/flush, failed publication and retirement after the last live reference. Test through the real ORC/resource-owner boundary; a host dictionary simulation alone does not count. |

## 4. C-08 cases

### C-08(a): construction identity and primitives

| ID and case | Required fixture, oracle and discriminating controls |
|---|---|
| **C08a01 — Prescribed rounded order** | Under declared binary64 round-to-nearest-even, distinguish the sequential rounded sum of `[2^54; 1; -2^54]` from an alternative permitted/forbidden grouping. The first yields zero; a grouping that cancels the large terms first yields one. Preserve a specified sequential order or fixed tree, including effects, signed zero and exceptional observations. A more accurate answer is not automatic equivalence. |
| **C08a02 — Exact term formation** | Use represented binary64 operands `a=1+2^-27`, `b=1-2^-27`, plus term `-1`. The exact product sum is `-2^-54`; summing rounded products can give zero. Select the actual source contract and verify term formation, rounding/FMA sites and result. Neither path recovers lost source-input precision. |
| **C08a03 — Residual and compensated constructions** | For admitted TwoSum, verify both returned components against exact represented-input addition (for example `2^53` and `1`), then mutate rounding/subnormal/overflow premises. Validate the complete compensated initialization/update/finalization error bound, not only the residual ingredient. Arbitrary merge or reordering must refuse unless separately established. |
| **C08a04 — Special values and reference scope** | Vary signed zero, admitted nonfinite IEEE inputs, NaR and observable status. Each exclusion requires actual input/intermediate proof. Distinguish error bound, fixed-decomposition reproducibility, all-permitted-decomposition reproducibility and correctly rounded exactness. Identical results on finite examples do not collapse these contracts. |

### C-08(b): state, capacity and exact merge

| ID and case | Required fixture, oracle and discriminating controls |
|---|---|
| **C08b01 — State denotation and initial value** | For each admitted exact construction, check initialize, accumulate, merge and finalize against an independent exact denotation. Use nonzero initial value, empty and singleton inputs and multiple partitions; the initial value contributes once globally. Internal redundant encodings need not have equal bytes if their denotation and prescribed finalization agree. |
| **C08b02 — Every partial and merge fits** | Use cancellation where the final sum fits but an allowed positive prefix or merge exceeds capacity. Vary partitions/order within the advertised contract. Refuse that decomposition or establish sufficient actual state capacity; no final-range-only proof. Include scalar exact integer and fixed-point cases, arbitrary justified bit widths and resource-budget exhaustion. |
| **C08b03 — IEEE exact and reproducible families** | Exercise an IEEE superaccumulator exact contract and a declared reproducible binned contract separately. Compare exact represented terms and specified final rounding; retain bins' promised decomposition scope. Test subnormal endpoints, long repeated sums, cancellation and overflow controls. Reproducibility alone is not an ideal-model accuracy proof. |
| **C08b04 — Standard and bounded posit quires** | For actual declared formats, establish exact product representation and every partial/merge bound. Cover full-gamut `16n` quire state at representative widths, and the bounded format's 800-bit state for admitted `n>12`. Verify format identity/layout, not width alone. Repeated positive products with no bounded-count/invariant proof must not inherit unlimited capacity. Capability refusal and inadequate capacity are distinct. |
| **C08b05 — Exact-state boundary transport** | Move local partial states through the actual admitted transport. An exact partial `2^54+1` rounded to binary64 before merging `-2^54` loses the exact result one; refuse that exact-construction transport unless exact representability is proved for every allowed partial. Change state layout, scale, endianness/alignment or transfer contract and retract correspondence. |

### C-08(c): grouping, target realization and joint admission

| ID and case | Required fixture, oracle and discriminating controls |
|---|---|
| **C08c01 — Complete batches and conflict evidence** | Use overlapping actual read/write/rewiring/proof footprints, including aliases and absent lookups. Check proper grouping; if claiming optimality, validate the explicit clique-tree/ordering premises. Three four-unit allocations into ten units demonstrate why pairwise compatibility does not establish joint capacity. The [PHG .NET checker](../../../arxiv-papers/research/PHG/check-conflict-coloring.fsx) is a mathematical oracle, not a compiler-batch gate. |
| **C08c02 — Work identity and publication** | Vary scheduling, retries, duplicate completion and late completion after a changed computation. Each logical term contributes exactly its source multiplicity. Preserve actual shared memo/cell identities, do not re-form captures on each read, and reject stale publication. Arithmetic correctness must not stand in for race freedom, lifetime or liveness. |
| **C08c03 — Real operation capabilities** | For CPU/SIMD and each admitted GPU/FPGA/other realization, bind precision, rounding, subnormal/FMA modes, state capacities, lane/carry semantics and memory/publication facts. Mutate one required capability to force a located refusal. Compare complete eligible realizations' cost without changing F-11's selected representation or weakening arithmetic. A profile-level family name is insufficient. |
| **C08c04 — Storage and progress premises** | Prove actual alignment/padded extents for separated worker state, distinct ownership/publication and covering lifetimes. Exercise failed allocation, unavailable operation and interrupted worker contracts according to the selected profile. No guessed cache-line size, arbitrary storage alias, or mathematically valid merge can establish completion. |

### C-08(d): composed execution and preservation

| ID and case | Required fixture, oracle and discriminating controls |
|---|---|
| **C08d01 — Functional measured reductions** | Run generic helpers, closures, records/tuples and list/array/sequence reductions with inferred dimensions. Use same-dimension inputs at distinct magnitude/scale regimes. A repeated shared deferred operand may compute once but contribute its specified multiplicity; an unused reduction must not demand its operands. Include independently formed instances and captured mutable views. E1/E4/E5 supply narrower inference/sequence prerequisites only. |
| **C08d02 — Decomposition metamorphisms** | Vary worker count, chunk boundaries, arrival order and every advertised legal merge shape. Preserve index-defined arithmetic for fixed trees; preserve exact denotation and deterministic prescribed finalization for exact contracts. Test empty partitions and nonzero initialization. Independently compare trace, result bits/error relation and all special observations. |
| **C08d03 — Artifact and target modes** | Verify the complete admitted operation graph, actual state fields/extents and lowering modes against source proofs. Mutate FMA contraction, reassociation, scalar-rounding sites, carry width or transfer layout and require the corresponding claim to retract. A source proof cannot certify changed machine operations. |
| **C08d04 — Construction edits through interactive execution** | Apply F11d04–F11d06 to a reduction: change term formation, legal tree, accumulator capacity, capture shape or final rounding during work/materialization. Recompute only justified regions, invalidate dependent interfaces/objects, reject old contributions and compare current execution with a fresh build. A valid old numeric answer cannot authorize an old computation instance. |
| **C08d05 — Integrated numerical quality** | Use ThreeBody or another admitted integrated application after the small cases. Record numerical quality, reproducibility and execution cost separately; establish reference precision/timestep convergence over the reported interval. Conserved scalar values, visually plausible trajectories or faster execution alone are not accuracy or arithmetic-preservation oracles. |

## 5. Complete mapping of Numeric Selection §13

| Requirement | Required cases |
|---|---|
| 1 — Range-driven representation | F11a01, F11a04, F11b01, F11b03 |
| 2 — Coverage filter and hard failure | F11b01, F11b04, F11b06 |
| 3 — Zero-crossing soundness | F11b02 |
| 4 — Accuracy-only objective | F11b03, F11b05, C08c03 |
| 5 — Evidence composition | F11a02, F11a05, F11a06, F11a07, F11d04 |
| 6 — Dimensioned unobservable commitment | F11a02, F11b06, F11d02 |
| 7 — Bare exception and propagation | F11b06, F11a05 |
| 8 — Dimensioning seam | F11b07, F11d02 |
| 9 — Required boundaries | F11b04, F11b08, C08b05 |
| 10 — PSG carriage and preservation | F11d01, F11d03, C08d03 |
| 11 — Capability gating | F11b05, C08b04, C08c03 |
| 12 — Quire adequacy | C08a02, C08b01, C08b02, C08b04, C08b05 |
| 13 — Actual arithmetic semantics | C08a01, C08a02, C08a04, C08d01, C08d03 |
| 14 — Decomposition guarantees | C08a03, C08b01, C08b02, C08d02 |
| 15 — Target realization premises | F11d01, C08c03, C08c04, C08d03 |
| 16 — Separate numeric guarantees | F11c01, F11c02, F11c03, C08a03, C08a04, C08b03 |
| 17 — Automatic obligation checking | F11c04, F11d02; every positive/refusal row runs through ordinary compilation |

All three real representation families and all eight C-08 construction families
have positive and discriminating rows. Every offered/admitted target combination
needs its own recorded evidence; a permanent refusal is not completion of a
required positive family. Integer recurrence or LLVM binary64 success does not
close posit, fixed-point, quire, GPU or FPGA requirements.

## 6. Closing a cohort

First run owning source/analysis regressions, then actual settlement/witness and
backend checks, then native and interactive cases with their input identities
frozen. Fan out independent host jobs with the .NET runner; report failures and
timeouts without changing expected outputs. Check both full and pruned evidence.
Finally run the affected F/C corpus and the complete registered inventory on one
recorded compiler/dependency snapshot.

For incremental gates, repeated fresh checks of edited source are the reference,
not the incremental execution being claimed. Capture actual reuse and invalidation,
segment interfaces, object identities, ORC generation/materialization and diagnostic
publication. Fresh/incremental equivalence and premise-preserving reuse are both
required. Ordinary compiler-test parallelism, mathematical oracle success and
isolated component tests cannot substitute for those execution paths.
