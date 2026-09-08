# Numeric Selection: Composer Implementation Guide

> **Audience:** Composer compiler implementers.
> **Scope:** Building compile-time selection of the numeric *representation* of real-valued quantities (posit / IEEE-754 / fixed-point) as the real-valued sibling of integer width inference.
> **Single source of truth.** The **normative** account of numeric selection — the objective, the side-conditions, range evidence and boundary constraints, the capability gate, the default/unobservable contract, the representation scope split, the quire pass, the preservation chain, and the citation posture — lives in the ratified spec chapter [`numeric-selection.md`](../../clef-lang-spec/spec/numeric-selection.md). **This guide carries implementation-HOW only** (where the code goes, what the passes compute, what data structures carry the results, milestone order). It **links** every normative claim into the spec and does **not** restate it. If a WHAT question arises, the spec answers it; do not duplicate spec text here, or the two drift.

> **What to build toward.** The user-facing payoff is **design-time relative-accuracy preservation** (§10): showing, per target, how much accuracy each candidate representation preserves across a value's justified range, before anything runs. Surface the supported cases early and make unresolved range obligations visible alongside them.

---

## 0. Orientation

Numeric selection extends the range-analysis and PSG-coeffect machinery used for integer width inference. Range propagation runs during elaboration; saturation settles the range and representation against the section's platform declaration before the witness boundary. Later lowering consumes that decision. The real interval domain needs its own transfer functions, sound enclosure method, and terminating widening (§3.2).

This guide is an implementation design, updated in September 2026, rather than a completion ledger. Earlier drafts used `IntervalAnalysis`, `IntWidth 0`, `narrowType`, and fixed `NTUfloat` cases as their starting points. Those names are historical navigation aids, not authority to retain backend selection or width-bearing source types. [NTU Types](../../clef-lang-spec/spec/ntu-types.md) governs the phase boundary: kind and dimension define numeric identity; range and representation travel beside it.

*(The normative model this pipeline implements — objective, range evidence, defaults, quire, preservation chain — is [`numeric-selection.md`](../../clef-lang-spec/spec/numeric-selection.md). This section is the Composer-internal framing of how it maps onto the existing width-inference machinery.)*

---

## 1. The selection resolver — where the objective is computed

**Normative objective:** [`numeric-selection.md` §2](../../clef-lang-spec/spec/numeric-selection.md#2-the-selection-objective) (the argmin, the `R_cov` coverage constraint, the zero-crossing ULP floor). Do not restate the objective or its soundness side-conditions here — implement them.

The resolver is the single choke point the spec's objective is realized at. It is the real-number analogue of `narrowType`:

```fsharp
/// Returns Ok r* | Error <coverage-empty | near-zero-degenerate>
/// Implements numeric-selection.md §2 (objective + both side-conditions).
/// Called after required range/platform facts resolve; pending facts never mean unavailable.
let selectRepresentation
        (target: Target)
        (range: RealInterval)
        (policy: EmulationPolicy)        // §4 capability gate
        : Result<Representation, SelectionError> =
    let candidates =
        R target
        |> List.filter (capabilityAvailable target)                  // R = capability ≠ unavailable
        |> List.filter (fun r -> dynRangeCovers (dynrange r) range)   // R_cov (spec §2.1)
        |> applyEmulationPolicy target policy                         // R_eff (§4)
    match candidates with
    | [] -> Error CoverageEmpty                                       // R_cov/R_eff empty → hard error
    | cs ->
        let score r = worstCaseError r range                         // ULP-floored; finite (spec §2.2)
        cs |> List.minBy score |> Ok
```

**Implementation note on `worstCaseError`.** Do **not** hard-code posit32 figures (`2⁻²⁷` near 1.0, `2⁻⁸` at extremes) — those are continuous-taper anchors, not a two-point model. Compute the taper at the *actual* range endpoints and regime boundaries `[a,b]` crosses. IEEE's `≈2⁻ᵖ` is uniform except near zero where the subnormal/`emin` floor governs (the ULP floor of spec §2.2).

---

<a id="2-tier-plumbing--how-the-range-input-reaches-the-resolver"></a>

## 2. Range-evidence plumbing — how facts reach the resolver

**Normative evidence and boundary contract:** [`numeric-selection.md` §3](../../clef-lang-spec/spec/numeric-selection.md#3-range-evidence-and-boundary-constraints). Carry each fact's justification and applicability on the PSG. Its origin does not give it priority over another fact or determine its verification tier.

Typical inputs and their implementation obligations include:

| Input | Carriage | Admission or use-site obligation |
|---|---|---|
| Dataflow analysis and guards | Sound enclosure or relation, derivation, value identities, branch polarity, and storage dependencies | Establish validity in the current context; invalidate storage-dependent facts after relevant writes |
| Checked domain law | Law identity, dimensional substitution, instantiated premises, accepted justification, and resulting enclosure | Check the law's premises and supported bound evaluation at the use site |
| Input contract or supplied hypothesis | The asserted property, its subject and scope, and whether it is assumed or established by a guard, decoder, or other check | Retain the assumption or discharge its validation obligation; a declaration alone is not a proof about arbitrary input |
| Declared boundary representation | Site, format, capacity, dimensional correspondence, and transfer semantics | Establish that the value fits the declared capacity and satisfies transfer fidelity; capacity alone supplies no bound on an outgoing value |

These are common sources of evidence and constraints, not an exhaustive classification or an authority hierarchy. Keep observed profiling bounds separate from universal range evidence: a sample range is not a sound enclosure of every execution. Admission of profiling evidence still needs its specified trust and validation model.

The range-carriage pass should:

1. Associate each enclosure with the same value identity, dimension, validity context, and premises used to justify it. Preserve relations as well as interval consequences; bounds on different values or mutually exclusive paths cannot simply be intersected.
2. Admit a law's consequence only when its justification applies and its required premises hold. Missing premises remain pending, retaining their provenance for later context or a located diagnostic.
3. Refine jointly with compatible justified enclosures for that value in that context. Their intersection is a sound enclosure. A broad dataflow enclosure extending outside a narrower checked-law enclosure is not evidence of a reachable counterexample; the broad enclosure may include unreachable values.
4. Keep inconsistent premises, unresolved consistency, and established unreachability distinct. An empty intersection requires examining the derivations and context. Conflicting declarations or unjustified assumptions cannot discharge coverage vacuously. An accepted proof of path unreachability instead produces a reachability fact; it is not a selected representation for a reachable value.
5. Pass the justified range and retained evidence to selection. At a declared boundary, restrict selection to its declared candidate and establish coverage separately. Containment of a sound enclosure in the capacity is sufficient; a supported guard or proof may establish coverage when a coarse enclosure does not. If coverage is still unresolved at commitment, compilation fails there. A proven violation and an inability to establish coverage need distinct diagnostic explanations.

**Boundary selection.** A platform description, wire schema, or binding descriptor fixes the representation at a site (spec §5). It does not override an inferred range. A covering-but-suboptimal declaration receives the CCS8014 informational finding only after the required coverage and transfer-fidelity obligations hold. No source seal or width-named numeric type is introduced. Justified bounds must include any required uncertainty allowance; a tolerance cannot waive a known inconsistency. The pass names and sketches in this guide are illustrative implementation planning, not existing public APIs.

**Automatic range-analysis scope.** Closed-form arithmetic can yield bounds when its operands and domain conditions are established. Division by a quantity with no known nonzero lower bound (e.g. `r²`) may remain unbounded. A checked law, guard, or justified input premise can supply the missing fact; merely importing a library cannot establish a fact about every caller. Keep an unresolved obligation pending until the commitment rule in §6 applies.

---

## 3. Riding (and not riding) the width-inference infrastructure

### 3.1 What is genuinely free — replicate these patterns verbatim

The integer twin gives six reusable patterns:

1. **Coeffect settled before witnessing.** Produce `RepresentationSelection` on the PSG at saturation, alongside the accepted range, declaration provenance, and obligations. Transfer structures carry the result rather than decide it. *(A field is cheap; its producer and proof obligations are separate work.)*
2. **Explicit pending state.** Keep unresolved real ranges and platform facts available during elaboration. A sentinel is an internal representation of pending work, never permission to pick a default in type lowering.
3. **Single resolver** — call `selectRepresentation` where saturation has the required range and platform facts; the backend reads the selected representation.
4. **Hard error on unobservability at representation commitment** — retain pending facts during elaboration; apply the dimensioned/bare distinction in §6 when a concrete representation is required.
5. **Check-time diagnostics** — emit codes inside `checkProgram` exactly as `CCS0100`/`FPGA0001` do.
6. **Platform capability facts** — `PlatformContext.RuntimeModel` is the seat for "does this target have b-posit HW / Xposit / quire support?" (§4).

### 3.2 What is NOT free — the real interval domain is a new abstract interpreter [research-grade]

The integer domain and its bit-counting operations are a useful architectural precedent. The real domain requires different transfer functions and infinity/rounding behavior; copying an older `{ Min: int64; Max: int64 }` representation does not provide them. The new domain requires, at minimum:

- **Outward-rounded FP interval arithmetic** — endpoints round outward to remain a sound superset.
- **Sign-crossing reciprocal/division** — `1/[lo,hi]` with `0 ∈ [lo,hi]` can require unbounded pieces and a domain obligation (as when dataflow has no nonzero bound for `r²` in a denominator).
- **Transcendentals** — `sqrt`, `log`, `exp`, `sin`.
- **Terminating widening over a continuous lattice** — the int64 monotone-widening fixpoint does **not** transfer; you need explicit widening operators with thresholds.

```fsharp
/// NEW abstract domain — sibling of IntervalAnalysis, NOT a port of it.
type RealInterval = {
    Lo: Bound        // outward bound, including unbounded endpoints
    Hi: Bound
    Dim: Dimension   // must agree with the measured type on the PSG
}

module RealIntervalDomain =
    let add  : RealInterval -> RealInterval -> RealInterval        // outward-rounded
    let mul  : RealInterval -> RealInterval -> RealInterval
    let recip: RealInterval -> RealInterval list                   // sign-crossing → 0..2 pieces
    let sqrt : RealInterval -> RealInterval
    let widen: RealInterval -> RealInterval -> RealInterval        // thresholded; must terminate
    // transfer over PSG nodes mirrors IntervalAnalysis.analyze's graph walk
```

**Plan accordingly:** this domain is the dominant cost of the whole project and the single biggest research risk. Do not estimate it as "one more codata field."

### 3.3 The three readings of one PSG traversal

Width inference is the *spatial* reading (bits/value); depth/budget inference (`DepthAnalysis.fs`, `foldWithLambdaPreBind`) is the *temporal* reading; numeric selection is the *third* reading (consume the dimensional range, select a representation). "Same traversal" means the same graph walk and carriage — **not** the same transfer functions (§3.2).

Range propagation begins in elaboration and closes at saturation with the section's platform facts. Numeric selection remains visible as PSG metadata and design-time diagnostics; it is not reconstructed from final MLIR types. Clock-depth analysis and numeric accuracy have different transfer functions and obligations even where they share graph traversal.

---

## 4. The performance/capability gate — as code

**Normative rule:** [`numeric-selection.md` §7](../../clef-lang-spec/spec/numeric-selection.md#7-performance-as-a-capability-gate) (performance is a candidate-set filter, never a term in the accuracy objective; a missing capability is a witnessed failure, never a silent swap). Implement it as a three-valued filter:

```fsharp
type Capability = Native | Emulated | Unavailable
type EmulationPolicy = NativeOnly | AllowEmulated | AllowEmulatedWarn   // default AllowEmulated

let applyEmulationPolicy target policy (cands: Representation list) =
    match policy with
    | NativeOnly        -> cands |> List.filter (fun r -> capability target r = Native)
    | AllowEmulated     -> cands
    | AllowEmulatedWarn -> cands   // keep all; the perf diagnostic fires at diagnostic time if r* is emulated
```

Layering (so reviewers don't mistake it for the paper's equation): `R(target) = { r : capability ≠ unavailable }` is the flat `R`; `R_cov` and the emulation policy are refinements layered on top. `allow-emulated-warn` influences *which diagnostic fires*, never *which representation is chosen* — the purity invariant is scoped to the choice.

The hardware evidence and format references live in [`numeric-selection.md` §7](../../clef-lang-spec/spec/numeric-selection.md#7-performance-as-a-capability-gate). Capability descriptions must identify the actual implementation; published decoder results do not establish a complete application's throughput.

---

## 5. The Fidelity.Physics integration surface

**`Fidelity.Physics` remains a planned library.** The current design in [`numeric-selection.md` §4](../../clef-lang-spec/spec/numeric-selection.md#4-the-fidelityphysics-mechanism-design-sketch) carries a typed range-law quotation, its dimensional parameters, its premises, and the provenance of its justification. The precise admission/registration API still needs definition. Implement against that contract rather than reviving the earlier choice between an erased quotation and a value registry.

A native Clef quotation retains the law's measured inputs and result. Where a hosted F# encoding needs an unmeasured quotation plus companion metadata, elaboration must reconstruct and validate that dimensional correspondence before using the law. Missing dimensions are diagnostics. Host encoding restrictions cannot redefine the native type identity.

The bound evaluator admits a terminating expression language with supported arithmetic and defined domain conditions. It computes a sound enclosure; a regime classifier may then classify that enclosure. The spec has settled this direction. Per-transcendental segment splitting and enclosure tightness remain open details. No arbitrary real formula is silently dispatched to QF_LIA because its eventual representation choice is finite.

The implementation sequence is:

1. Locate the registered law and retain its declaration provenance.
2. Check its dimensions and instantiate its input premises at the use site.
3. Evaluate the admitted law over justified bounds, with outward enclosure.
4. Retain the enclosure and justification; perform any regime classification afterward.
5. Combine the justified enclosure with other applicable range evidence (§2), then perform representation selection. An unjustified premise stays pending until required, then receives a diagnostic if unresolved.

The gravitation example obtains a bounded force only when justified mass bounds and a distance bound `r ≥ r_min > 0` are available. A registered library law can connect those bounds to the result; its premises still have to hold for this call. The illustrative source below does not supply that premise merely by opening the library:

```fsharp
open Fidelity.Physics.OrbitalMechanics
let gravForce (m1: float<kg>) (m2: float<kg>) (r: float<m>) : float<N> =
    GravConst * m1 * m2 / (r * r)   // dim N inferred; range requires input bounds including r >= r_min > 0
```

**ML routing.** A planned `Fidelity.ML` sibling supplies a justified enclosure and an asymmetric-bias recommendation. The general selector evaluates the spec's worst-case objective over that enclosure ([`numeric-selection.md` §4](../../clef-lang-spec/spec/numeric-selection.md#4-the-fidelityphysics-mechanism-design-sketch) and Requirement 4). A recommended configuration must pass coverage and capability filtering and need not win the objective. Observed concentration or a distribution's mode cannot justify excluding possible outliers from a hard coverage obligation. A distribution-weighted objective remains optional ML-library future work with its own probability model and error contract.

---

## 6. The default / unobservable case — as code

**Normative contract:** [`numeric-selection.md` §6](../../clef-lang-spec/spec/numeric-selection.md#6-the-default-and-unobservable-case). During elaboration, missing facts remain pending. At representation commitment, a dimensioned real whose range is still unobservable is diagnosed. The bare-float `f64` path is an explicit exception requiring an offered and permitted capability; it is not proof that an unbounded mathematical range fits `f64`.

```fsharp
// Called only at commitment, after available context has been applied.
let resolveUnobservable target policy (node: Node) : Result<Representation, SelectionError> =
    match node.Dimension with
    | Dimensioned _ -> Error (UnboundedDimensionedRange node)
    | Bare when offeredAndPermitted target policy (IEEE F64) -> Ok (IEEE F64)
    | Bare -> Error (MissingBareFloatCapability node)
```

### 6.1 The bare/dimensioned seam — handle it explicitly

Normative seam contract: [`numeric-selection.md` §6.1](../../clef-lang-spec/spec/numeric-selection.md#61-the-baredimensioned-seam). Bare floats participate in the same range propagation. When commitment still requires an unavailable bound, the diagnostic points to the dimensioning boundary and names the upstream bare source:

```
error: y : float<newtons> requires a bounded range; its range derives from
        x (bare float, unbounded at <site>). Annotate x's range, import a
        domain library, or establish a valid bound through the input contract.
```

The source type remains `float` with its dimension; `f32` and `f64` name representations in the platform/lowering vocabulary.

---

## 7. Concrete vs parameterized posits — as a lowering-codomain split

**Normative scope split:** [`numeric-selection.md` §8](../../clef-lang-spec/spec/numeric-selection.md#8-concrete-vs-parameterized-representations). Implementation placement:

1. **Surface:** `float<dim>`, with range and representation as coeffects. A boundary declaration fixes the site's representation; it adds no posit type or seal syntax.
2. **Lowering codomain on fixed-ISA (CPU/SIMD/RISC-V):** concrete `Posit8/16/32/64` representations among the platform's candidates, alongside IEEE/fixed. Such names designate internal carriers or descriptors, never source numeric types. Quire pairing follows the complete declared format, including standard-versus-bounded family and configuration; operand width alone is insufficient.
3. **Parameterized `posit<n,es,rs,bias>`:** FPGA/reconfigurable-only synthesis search — **future work** (CIRCT posit pipeline parameterization not built). The `(rs,es)` grid is enumerable (≤25 points); bias/asymmetry are bounded-but-continuous, explored heuristically.

---

## 8. The quire recognition / sizing / lowering nanopass

**Normative quire semantics:** [`numeric-selection.md` §10.2](../../clef-lang-spec/spec/numeric-selection.md#102-the-quire-pass), particularly the product and partial-sum obligations in §10.2.1. Recognition, layout selection, and adequacy checks occur before target lowering. The accepted facts and their justifications reside on the PSG; witnesses consume them.

```fsharp
let (|QuireMAC|_|) (node: Node) =                  // RECOGNITION — fma/fold/reduce-of-products over a selected posit
    matchFusedProductAccumulation node

// Read the selected format's declared quire layout; operand width alone is insufficient.
// Standard 2022 full-gamut posit: 16n bits (posit32 -> 512 bits).
// Cited b-posit design: 800 bits for n > 12 (b-posit32 -> 25 x 32-bit words).
let quireLayout (format: Representation) = format.DeclaredQuireLayout

type QuireCoeffect = {                             // conceptual internal record
    Layout     : QuireLayout                       // format, field layout, finite range
    Allocation : ByteCount * EscapeClass           // 100 B for b-posit, before target padding
    Lifetime   : Scope
    Capability : ExactAccumulation                 // can this target accumulate exactly?
    Dimension  : Dimension                         // fma of newtons×meters accumulates as joules; dim verified at output
    Adequacy   : AcceptedObligationRefs             // product exactness + every partial sum fits
    Rounding   : DeclaredRoundingSite
}
// LOWERING deferred to target-binding (the fork).
```

The format fixes allocation width; it does not grant unlimited accumulation. Establish that each represented product is exact in the quire layout and every reachable partial sum stays in its finite range. Operand bounds plus a term-count bound can establish this; an accepted invariant can establish it without a fixed count. `k × bits-per-product ≤ Q` is not the addition capacity law, and checking each product alone is insufficient.

Use QF_BV for a faithful finite bit-level encoding and QF_LIA where a bounds obligation has actually reduced to linear integer constraints. Record the encoding and premises with each obligation. A finite representation catalogue does not make arbitrary products or real-valued laws linear. Missing evidence can remain pending during elaboration but must be diagnosed before exact-accumulation commitment. Inserting intermediate rounding would change the operation's contract.

**Per-target capability (implementation table):**

| Target | Quire support | Resolution |
|---|---|---|
| CPU/SIMD | Software or declared instruction support | Allocate the declared layout by escape/lifetime analysis; measure implementation cost |
| FPGA | Synthesized accumulator matching the format | Derive latency and throughput from the actual pipeline and synthesis result |
| RISC-V with a posit extension | Extension-specific quire | Require agreement with the extension's format and instruction semantics |
| Target without admitted exact accumulation | Unavailable | **Capability failure** |

The quire may remain beside the FPGA arithmetic and return a rounded result, or its full state may cross a boundary for further accumulation. Those are different interface contracts: record the rounding site, and specify limb order and format identity whenever the accumulator itself crosses. A quire preserves the represented sum of products; surrounding division, integration, and boundary rounding retain their own error obligations.

---

## 9. Preservation chain — pass ordering

**Normative chain:** [`numeric-selection.md` §10.1](../../clef-lang-spec/spec/numeric-selection.md#101-preservation-chain). Numeric selection is the second arrow; the quire pass realizes the third:

```
Dimension --(range)--> Representation --(width)--> Footprint --(escape)--> Allocation
   DTS      justified range   selection coeffect     declared layout   escape analysis
             and premises    (selectRepresentation)
```

Implementation invariants (all normatively grounded in spec §10.1):
- **Settle before witnessing.** Elaboration can carry pending facts; saturation closes the required constraints against the section's declarations. Later passes consume the recorded choice instead of choosing a new width or representation.
- **Carriage and obligation residency.** Keep representation, range, dimension, grade, lifetime, and the obligations that relate them on the PSG. A transformation must preserve an applicable justification or discharge the affected obligation under its supported theory. The external ledger checks this implementation while the graph mechanism matures; it is not the canonical proof store. No blanket QF_BV re-check covers every property.
- **Transfer fidelity is directional.** Follow [`numeric-selection.md` §10.1](../../clef-lang-spec/spec/numeric-selection.md#101-preservation-chain): range coverage is necessary but does not establish exact representability. Carry any error enclosure and declared rounding behavior. A change in representation must not silently erase the source dimension or introduce undeclared loss.

---

## 10. Design-time surfacing (Lattice / language server)

Numeric selection emits **check-time diagnostics** in the same shape as `CCS0100`/`FPGA0001` (`Severity`, `Range`, `RelatedNodes`, `Reachability`), from a pass inside `checkProgram`, surfaced as squiggles/hovers. "Design time" = continuous Lattice elaboration; representation-adequacy is a warm-rotation elaboration certificate.

Readouts should display computed bounds and errors, not fixed example scores presented as analysis. An illustrative layout:

```
force: float<newtons>
  Range: <sound enclosure>, from <library law and accepted premises>
  Platform: <declared candidate formats and capabilities>
  Selected representation: <format and configuration>
  Worst-case error over this range: <computed bound>
  Accumulation: <quire layout, accepted adequacy, rounding site>
```
```
Error CCS8012: coverage by the representation declared by <boundary> could
  not be established for <value>. Establish a tighter valid bound, change the
  declaration, or express intended loss in arithmetic.
```

**New diagnostic codes** (numeric-selection family, siblings of `CCS0100`/`FPGA0001`):
- `coverage-empty` — `R_cov = ∅`
- `near-zero-degeneracy` — range straddles 0 with no representation resolving it under the ULP floor
- `bare-source-unbounded-at-seam` — §6.1
- `range-evidence-inconsistent` — incompatible premises or declarations, with their derivations and scope (§2)
- `boundary-coverage-unresolved` — available evidence does not establish coverage at commitment; distinguish this from a proven reachable violation (§2)
- `suboptimal-boundary` — informational CCS8014 for a covering but accuracy-suboptimal declaration
- `quire-capability-failure` — §8
- `quire-adequacy-unresolved` — product or partial-sum evidence missing at commitment (§8)

These descriptive names are implementation planning labels; use the normative diagnostic IDs where specified. The b-posit paper cited by the spec supplies hardware evidence for its evaluated designs. It does not supply this application's precision or latency measurements.

---

## 11. Implementation map — files, costs, and where each piece sits

| Component | Location | Cost |
|---|---|---|
| **Representation coeffect** with range and declaration provenance | PSG elaboration/saturation produces it; `Coeffects.fs` / transfer structures carry it into witnessing | Field plus producer and preservation checks |
| **Pending state + resolver** `selectRepresentation` | Saturation, before the witness boundary; `TypeMapping.fs` reads the result | Moderate |
| **New abstract domain** real/dimensional interval (outward FP rounding, sign-crossing division, transcendentals, terminating widening) | new module sibling of `IntervalAnalysis.fs` (which stays int64-only) | **Research-grade — dominant cost** |
| **Diagnostics** numeric-selection family | check-time service, retaining source spans and pending/commitment distinction | Moderate |
| **Quire recognition, declared layout, product/partial-sum adequacy** | Before target fork, with obligations resident on the graph (§8) | Encoding and proof work in addition to recognition |
| **Capability facts** b-posit / extension / quire layout and semantics | Section's platform description, resolved into `PlatformContext` | Declaration and validation |
| **`Fidelity.Physics`** | Typed quotation admission, registration, premise checking and bound evaluation (§5) | Remaining API and bound-method details need specification |

**Source navigation from the original implementation study:** `src/MiddleEnd/PSGElaboration/IntervalAnalysis.fs`, `src/MiddleEnd/Alex/XParsec/PSGCombinators.fs`, `src/MiddleEnd/Alex/CodeGeneration/TypeMapping.fs`, and `src/MiddleEnd/PSGElaboration/Coeffects.fs` in Composer; `PSGSaturation/SemanticGraph/DepthAnalysis.fs`, `NativeTypedTree/NativeTypes.fs`, and `NativeTypedTree/NativeService.fs` under Clef's compiler sources. Revalidate the current paths and phase ownership when implementing; historical backend hooks do not override the spec's saturation boundary.

*(Normative dependencies this implementation carries: [`numeric-selection.md`](../../clef-lang-spec/spec/numeric-selection.md) is the authority; it cross-references `width-inference.md`, `native-type-universe.md` §2.4, `units-of-measure.md`, `ntu-dimensional-architecture.md`, and `incremental-computation.md`. Do not restate their content here — link it.)*

---

## 12. Phasing / milestones — what to build first

Order chosen so each milestone is independently testable and the highest-risk research item is de-risked early.

**Milestone 0 — Carriage.** Add explicit pending/selected representation states and declaration provenance on the PSG, with transfer structures preserving them. Populate capability facts from the platform description. *Exit:* metadata survives the relevant passes without inventing dimensional or representation facts; existing supported behavior remains validated.

**Milestone 1 — Boundary selection and the objective.** Implement `selectRepresentation` (§1) over the concrete formats the target declares, and coverage checks at a declared boundary. Report hard failures when required coverage cannot be established and informational suboptimal-boundary findings after the boundary obligations pass. *Exit:* a measured real with a justified range is checked against a binding or wire declaration and produces a Lattice readout. There is no dependency on new seal syntax.

**Milestone 2 — The real interval domain.** Build `RealIntervalDomain` (§3.2); propagate dataflow-bounded ranges with their derivations; implement the bare/dimensioned split and seam diagnostic (§6). *Exit:* a closed-form measured computation selects automatically; a denominator without a nonzero bound remains pending for later context and is diagnosed if still unresolved at commitment.

**Milestone 3 — The quire nanopass.** Recognize eligible fused-product reductions, obtain the selected format's quire layout, establish product and partial-sum adequacy, and retain lifetime, dimension, rounding, and capability facts (§8). *Exit:* an admitted reduction lowers to matching arithmetic on an available target; insufficient capacity or an unavailable capability is diagnosed. Bit-level reference vectors validate the implementation separately from the obligation encoding.

**Milestone 4 — Checked domain laws (`Fidelity.Physics`).** Settle the remaining quotation-admission and bound-evaluation details (§5), then implement registration, dimensional/premise checking, enclosure evaluation, and evidence-consistency diagnostics. *Exit:* an applicable library law and checked premises supply the justified range for `gravForce` without erasing its measured type; compatible enclosures refine jointly, while missing premises remain pending.

**Future work (do not schedule into the above):** parameterized `posit<n,es,rs,bias>` FPGA synthesis search (§7); the ML distribution-weighted objective (scoped to `Fidelity.ML` only, never the general selector); profiling-evidence provenance/trust model.

**Cross-cutting reminder:** results and obligations reside on the PSG, settled before witnessing and consumed by target lowering (§9). Missing facts are never fabricated to pass an implementation gate.

---

## 13. ThreeBody: precision, useful horizon, and reversal without a tape

**ThreeBody's central experiment is numerical:** show how posit/quire arithmetic can preserve a chaotic gravitational trajectory longer, delay its useful Lyapunov break point, and recompute a more faithful return path after momentum reversal without a stored trajectory tape. The project is currently design-only. Compiler, transport, and proof evidence support the credibility of that experiment; a demonstration of compilation failures is not its purpose.

Here, a longer useful horizon means a later crossing of a stated numerical-error tolerance. It does not mean changing the physical system's Lyapunov exponent. Reduced arithmetic error can delay that crossing, while integration truncation, input error, and later rounding still contribute. Measure the gain for the specified initial conditions, force law, timestep, and comparison method.

Use natural units (`G = 1`) and record the corresponding dimensional scaling and analyzed ranges. Normalization can place relevant values near a posit's high-precision region around magnitude one; it does not itself prove coverage or select the winner. The intended bounded format is b-posit32 with `rS = 6`, `eS = 5`, and an 800-bit quire. Standard full-gamut posit32 (`eS = 2`, 512-bit quire) is a distinct comparison configuration. Each boundary declares the complete format and rounding behavior.

### 13.1 Recompute the return path

Start with a fixed-step, time-reversible mathematical integrator; KDK leapfrog/Verlet is the proposed baseline. For momentum reversal `S(q,p) = (q,-p)`, a reversible map satisfies `Φ_h⁻¹ = S ∘ Φ_h ∘ S`. Advance for `N` steps, flip momentum, and apply the same forward map for another `N` steps. Apply the final momentum flip when comparing the full returned state with the initial state. This constructs the return from the current state and equations, without reading a forward-history buffer. Diagnostic snapshots may be retained for measurement, but never drive the reverse computation.

Finite-precision execution need not satisfy that inverse identity exactly. The return residual is the measurement ThreeBody should make visible. A quire removes intermediate rounding inside an admitted accumulation, subject to §8; other operations and the final rounding remain. A step may contain several such sites. Tape-free reversal is therefore part of the core experiment, independent of a future negative-type facility.

For the initial numerical comparison, keep timestep and force ordering fixed. Later encounter routing, adaptive steps, or different arithmetic on the return path must be included in the reversibility argument for the composite map. A symmetric integrator alone does not establish that a state-dependent scheduler is reversible. See [Hairer and Söderlind, Explicit, Time Reversible, Adaptive Step Size Control](https://www.unige.ch/~hairer/preprints/revstep.pdf).

Negative/fractional types remain proposed and non-normative in [the spec](../../clef-lang-spec/spec/terms-and-definitions.md). A later type discipline may express and check the intended forward/reverse composition, with the required laws stated separately. A momentum sign flip, inverse state evolution, numerical-method adjoint, and differentiation pullback are distinct operations; naming a reverse channel does not establish their equivalence or certify floating-point inversion.

### 13.2 Make the numerical benefit visible

The main display compares trajectory fidelity and reversal residual over physical time, with a stated norm and tolerance marking the useful horizon. Conserved-quantity drift (energy, angular momentum, and linear momentum) and an independently convergence-checked high-precision reference support that display. A close return alone does not establish an accurate forward trajectory.

Compare matched equations, initial conditions, timestep, force ordering, and computational scope:

- **IEEE FP64** supplies the familiar baseline.
- **FP64 with specified compensated summation** tests how much of the difference is due to accumulation. Kahan summation is not an exact quire.
- **Posit without quire** isolates the representation's contribution.
- **The same posit configuration with quire** tests the intended precision and horizon gain. Record formats at force, integration, and return boundaries, rather than changing only an unreported subset of the computation.
- **An explicitly invertible discrete integrator**, if included, provides a bit-exact reversal control. Selecting a fixed-point datatype alone does not make an update invertible; [JANUS](https://arxiv.org/abs/1704.07715) is a relevant construction.

Let the measured curves establish the size and conditions of the advantage. The intended posit/quire result remains the headline; the controls explain which parts of the arithmetic earn it.

### 13.3 Supporting evidence across substrates

BAREWire is the glue layer for memory layout, IPC, and network contracts. Conclave is the platform for intelligent distributed systems on Cloudflare. ThreeBody's CPU/eBPF/FPGA path exercises the same contract continuity that BAREWire supplies across the broader Fidelity framework.

Keep the numerical contract and its obligations attached to the PSG before target-specific lowering. Layout/bounds checks, payload-preservation obligations, kernel admission, and FPGA arithmetic validation establish different parts of the execution. The external ledger checks agreement while the proof-carrying graph matures. These details can support a technical deep dive without displacing the visible precision, horizon, and live-reversal experiment. The [eBPF integration design](ebpf-targeting/05_threebody_integration.md) describes that supporting path.
