# F-11: Numeric Selection and Numeric Obligations

**Status: In-Progress. Accepted September 26, 2026.**

## 1. Result and governing contract

Ordinary numeric expressions receive justified native representations and the
capacity, definedness, scale and error checks their meaning requires. Source
authors use Clef's `int` and `float` kinds with inferred physical dimensions.
Platform and boundary declarations determine the offered/required encodings.
The compiler preserves the evidence linking those choices to the source.

[Numeric Selection](../../../clef-lang-spec/spec/numeric-selection.md),
[Width Inference](../../../clef-lang-spec/spec/width-inference.md),
[Rounding](../../../clef-lang-spec/spec/rounding.md),
[Units of Measure](../../../clef-lang-spec/spec/units-of-measure.md),
[NTU architecture](../../../clef-lang-spec/spec/ntu-dimensional-architecture.md)
and [Conformance](../../../clef-lang-spec/spec/conformance.md) govern acceptance.
This PRD owns the shared numeric contract consumed by existing F/C features.
[C-08](C-08-ArithmeticConstruction.md) owns whole arithmetic constructions and
their decomposition; [M-01](M-01-DialectAdmission.md) owns admitted witness forms
and downstream correspondence. Neither consumer reimplements selection.

## 2. Architectural ownership

CCS/Baker owns source kind/dimension inference, range provenance, obligation
generation, representation settlement and semantic arithmetic decomposition.
Platform-independent facts can narrow during elaboration. Saturation consumes
the selected declarations when required. A quantified dimensional parameter
remains a valid scheme; an unresolved range remains an obligation.

Separate abstract domains supply integer, real, scale/congruence and error
reasoning. They share graph traversal and provenance rather than reuse integer
transfer functions for continuous intervals. Every imported law retains its
actual type/dimension substitution, argument identities and premises. Proof
resource exhaustion yields an unresolved result, not an assumed bound.

Settled representation and evidence are PSG codata. Alex consumes them through
the existing Huet Element/Pattern/Witness path and its representation resolver.
Witnesses do not reconstruct range proofs or choose a convenient scalar carrier.
Backends preserve operation semantics, representation correspondence and actual
target modes through their transformations. Circuit algorithms remain backend work.

## 3. Acceptance groups

### F-11(a): justified ranges and dimensional inference

- Infer integer and real dimensional relationships through unannotated helpers,
  generic arguments/results, aliases, tuples, records, closures and recurrence.
  Retain scale separately from physical dimension and representation width.
- Preserve principal measure schemes and coupled exponent relations. Instantiate
  arguments, results, captures and body consistently at each use. Weak mutable
  variables and residual member constraints retain their actual dependencies;
  an unresolved constraint cannot be exported as an unconstrained quantifier.
  Use established formal context for aggregate inference across direct, aliased,
  partial and inline calls; retain precise ambiguity when no context selects an owner.
- Propagate exact integer intervals and outward-rounded real enclosures. Include
  multiplication sign cases, reciprocal/division across zero, and admitted
  `sqrt`, `log`, `exp` and `sin` domains and transfer laws.
- Intersect applicable same-value/same-context enclosures; join alternatives.
  Distinguish an inconsistent reachable premise set from established unreachability
  and from a conservative over-approximation awaiting refinement.
- Carry immutable identities and invalidate storage-dependent premises at relevant
  writes or calls. Deferred reads use premises valid at demand. Memoized results
  retain the facts established for their actual computation.
- Use terminating widening with the specified domain thresholds and documented
  proof budgets. Validate bounds in arbitrary-precision/exact or outward-rounded
  analysis arithmetic, including large source literals and intermediate bounds.
- Check domain-law quotations and imported evidence at the actual use, including
  missing, changed, dimensionally incompatible and contradictory premises.

### F-11(b): representation selection and boundary fidelity

- Filter the actual offered set by capability/policy and range coverage before
  applying the accuracy-only objective. Empty coverage, absent platform facts and
  an unresolved range remain distinct outcomes.
- Implement the specified ULP-floored metric, actual IEEE subnormal behavior,
  declared posit taper and fixed-point scale. Check zero-crossing, singleton,
  near-unity, extreme and asymmetric ranges. Resolve equal scores deterministically
  without introducing a hidden performance term.
- Honor valid singleton boundary representations; independently establish coverage
  and directional exact representability or the required lossy-transfer bound.
  A covering suboptimal declaration receives the specified information diagnostic.
- Apply the bare-float f64 exception only under Numeric Selection §6. Preserve
  known-range coverage and operation obligations. Attach an unresolved bare-to-
  dimensioned seam to its origin and upstream source.
- Exercise native-only, allow-emulated and allow-emulated-warn policy, unavailable
  candidates and changed declarations. Performance ranks eligible realizations
  only after the selected representation and arithmetic meaning are fixed.

### F-11(c): automatic arithmetic obligations

- Establish integer intermediate capacity and operation preconditions, including
  nonzero divisors, valid shifts and signed remainder. Deliberate modulo/clamp
  retains its source meaning. A fitting final value does not justify overflowing
  an earlier exact intermediate.
- Establish fixed-point carrier capacity, dimension, scale alignment, exact
  products/division, divisibility and rescaling fidelity. Record every admitted
  rounding contribution and distinguish clipping from quantization.
- Bound rounded arithmetic under a stated reference quantity, metric, input
  domain and arithmetic environment. Preserve dependencies, cancellation,
  intermediate precision and rounding points. Input/model uncertainty,
  representation error, arithmetic error and method error remain distinguishable.
- Generate applicable obligations during ordinary compilation in every build mode.
  No opt-in wrapper enables numeric checking. Source/boundary contracts supply
  intended loss and goals; analysis supplies no invented physical bounds or goals.
- Keep established, refuted, inconsistent and unresolved outcomes distinct.
  Required unresolved facts produce located commitment diagnostics. Runtime
  guards establish successful-path facts only under a contract permitting them
  and defining their failure behavior.

### F-11(d): preservation, diagnostics and incremental evidence

- Preserve exact source/occurrence, scheme, dimension, range, representation,
  boundary and rule identities through recipes, fan-out/fold-in and layout.
- Retain the [rewrite tape](../Nanopass_Incremental_Contract_Direction.md#24-rewrite-independence-and-the-intermediate-tape)
  through full/pruned artifacts. Historical participants remain inspectable without
  becoming executable roots or current proof premises.
- Validate actual emitted widths, extensions, rounding modes, fast-math permissions,
  overflow assertions, rescaling and transfers at every admitted lowering edge.
  Artifact mutation controls invalidate the affected claim.
- Project pending and failed obligations through the shared compiler diagnostics,
  with source spans, related participants and target provenance. Distinguish
  accuracy information from hard coverage/capability failure.
- After edits to ranges, dimensions, mutable effects, operation modes, declarations
  or proof rules, retract dependent results. Keep independent support valid only
  when its premises still hold. Reject stale asynchronous proof publication.
- Check selective re-evaluation against fresh settlement. Representation, layout,
  capture and premise changes propagate through the relevant segment interfaces;
  unchanged independent regions retain their validated evidence and artifacts.
  Follow the [segmented publication contract](../Nanopass_Incremental_Contract_Direction.md#25-edit-transactions-proof-reuse-and-segmented-publication)
  for object reuse, diagnostics and REPL generations.

## 4. Required family and interaction inventory

| Axis | Required distinctions |
|---|---|
| Kind/dimension | Bare and measured int/float; inferred compound units; quantified generic schemes; incompatible dimensions; pending variables; scale-equivalent physical quantities |
| Integer | Negative/nonnegative/mixed range; exact non-native widths; CPU covering widths and FPGA exact widths; boundary minima/maxima; modulo/clamp; intermediate overflow and definedness |
| IEEE | Offered binary32/binary64; normal/subnormal/zero transitions; signed zero; nonfinite and exceptional behavior under admitted contracts |
| Posit | Declared standard/bounded format identity, width, regime/exponent parameters and rounding; actual taper and extremes; NaR contract; native/emulated capability |
| Fixed point | Signed/unsigned carrier where offered, binary scale, exact/inexact alignment/rescaling, multiplication/division and nearest/directed rounding |
| Source composition | Direct/aliased/partial/returned call; branch/recursion; measured aggregate; lazy/sequence demand; mutable captured storage and shared memo identity |
| Boundary | Required representation, ABI/MMIO/wire/entry declaration; exact/lossy direction; multi-hop error; changing source or destination contract |
| Evidence | Established/refuted/pending/inconsistent; alternatives versus conjunction; invalidated support; resource exhaustion; artifact and serialized correspondence |

Select explicit cases for each distinction and representative cross-axis
interactions with a written coverage rationale. A passing bare scalar example
does not cover inferred dimensional composition or every offered format.

## 5. Executable validation and completion

The [numeric validation inventory](Numeric_Validation_Cases.md) supplies concrete
oracles and maps every normative requirement. Tests must call the owning compiler
analysis or compile the source through the actual witness/backend for the gate
claimed. Independent .NET exact arithmetic and finite-format models are oracle
tools; their own success cannot count as a compiler pass.

Each case records source, target/declarations, expected type/dimension, range or
error relation, obligation identity/status, responsible diagnostic and span,
artifact relation, runtime oracle and invalidation controls as applicable.
Refusal cases must fail for the specified reason, not a parser or unrelated error.
Positive cases must execute the advertised admitted behavior.

Use focused owning tests and affected F/C regressions during development; run the
complete F-11 inventory and its dependent cohort on one recorded immutable
compiler/dependency snapshot before closing this PRD. Fan out independent jobs
through the .NET process harness with separate streams, bounded timeouts and
full/pruned artifact comparisons. Report each failure and timeout. Close F-11
only when all four groups and every enumerated required family are satisfied.
