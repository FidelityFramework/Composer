# Numeric Selection: Composer Implementation Guide

> **Audience:** Composer compiler implementers.
> **Scope:** Building compile-time selection of the numeric *representation* of real-valued quantities (posit / IEEE-754 / fixed-point) as the real-valued sibling of integer width inference.
> **Single source of truth.** The **normative** account of numeric selection — the objective, the side-conditions, the tiered authority model, the capability gate, the default/unobservable contract, the representation scope split, the quire pass, the preservation chain, and the citation posture — lives in the ratified spec chapter [`numeric-selection.md`](../../clef-lang-spec/spec/numeric-selection.md). **This guide carries implementation-HOW only** (where the code goes, what the passes compute, what data structures carry the results, milestone order). It **links** every normative claim into the spec and does **not** restate it. If a WHAT question arises, the spec answers it; do not duplicate spec text here, or the two drift.

> **What to build toward.** The highest-leverage user-facing payoff is **design-time relative-accuracy preservation** (§10): showing, per target, how much accuracy each candidate representation preserves across a value's actual range, before anything runs. Prioritize that surfacing early, even atop a coarse objective, because that is where "this compiler is materially different" shows.

---

## 0. Orientation

Composer already ships a complete *integer* representation-selection pipeline: `IntervalAnalysis.fs` (interval domain) → an abstract `IntWidth 0` sentinel at type-lowering → a single `narrowType` resolver → a hard error (`FPGA0001`) when a node's range is unobservable → check-time diagnostics → platform capability facts. **Numeric selection is the missing real-number twin of that pipeline.** Reals currently bypass it: `TypeMapping.fs` hard-binds `NTUfloat (Fixed 32/64) → TFloat F32/F64` before any analysis can run. The work is to slot a parallel real-number flow into the *same frame*. Reusable: the PSG traversal skeleton, the coeffect-carriage discipline, the sentinel/resolver pattern, the diagnostic plumbing. Not reusable: the *transfer functions* — the real interval domain is a new, research-grade abstract interpreter, not a port of the int64 one (§3.2). Plan the project around that asymmetry.

*(The normative model this pipeline implements — objective, tiers, defaults, quire, preservation chain — is [`numeric-selection.md`](../../clef-lang-spec/spec/numeric-selection.md). This section is the Composer-internal framing of how it maps onto the existing width-inference machinery.)*

---

## 1. The selection resolver — where the objective is computed

**Normative objective:** [`numeric-selection.md` §2](../../clef-lang-spec/spec/numeric-selection.md#2-the-selection-objective) (the argmin, the `R_cov` coverage constraint, the zero-crossing ULP floor). Do not restate the objective or its soundness side-conditions here — implement them.

The resolver is the single choke point the spec's objective is realized at. It is the real-number analogue of `narrowType`:

```fsharp
/// Returns Ok r* | Error <coverage-empty | near-zero-degenerate>
/// Implements numeric-selection.md §2 (objective + both side-conditions).
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

## 2. Tier plumbing — how the range input reaches the resolver

**Normative tier model + precedence-override composition:** [`numeric-selection.md` §3](../../clef-lang-spec/spec/numeric-selection.md#3-the-tiered-authority-model). The three tiers are provenances of the one resolver's `[a,b]` input, not three algorithms; composition is precedence-override, not intersection. Do not restate the rules — implement the carriage.

```fsharp
type TierClaim =
    | Tier1 of RealInterval          // dataflow, if analysis terminated with a bound
    | Tier2 of RealInterval          // Fidelity.Physics library
    | Tier3 of Representation         // sealed; implies its dynrange as a range claim

/// Highest present tier binds; lower tiers become consistency obligations (diagnostics),
/// never inputs to the bound range. Total and decidable — no empty-intersection state.
/// Implements the composition rule of numeric-selection.md §3.4.
let composeTiers (claims: TierClaim list) : RealInterval * Disagreement list =
    let binding = claims |> List.sortByDescending tierRank |> List.head |> claimRange
    let obligations =
        claims |> List.choose (fun c ->
            let lower = claimRange c
            if not (contains binding (inflate lower tolerance) lower)
            then Some (TierDisagreement (c, binding))   // diagnostic; does NOT change binding
            else None)
    binding, obligations
```

**Tier 3 is sealing (selection-in-reverse):** the developer fixes a representation; the resolver runs the coverage check on a singleton `R` (spec §5, Sealing and Reverse Selection). A covering-but-suboptimal seal still compiles; Lattice witnesses the suboptimality (`suboptimal-seal` diagnostic).

**Honest Tier-1 scope (implementation reality).** Tier 1 succeeds automatically only when the range is bounded by dataflow alone: closed-form, division-free or division-with-known-nonzero-lower-bound, profiled, or annotated. Real physics with division by a quantity whose lower bound is not in dataflow (e.g. `r²` in a denominator) falls through to Tier 2/3 or the §6 error. Do not over-promise the garden path.

---

## 3. Riding (and not riding) the width-inference infrastructure

### 3.1 What is genuinely free — replicate these patterns verbatim

The integer twin gives six reusable patterns:

1. **Coeffect computed pre-emission.** `IntervalAnalysis.analyze` runs once-per-graph as `TransferCoeffects.WidthInference`, computed in `MLIRGeneration.fs`. Add a **sibling field** `RepresentationSelection` on `TransferCoeffects`, computed beside `widthInference`. *(Cheap.)*
2. **Abstract sentinel at type-lowering.** Integers lower to `TInt (IntWidth 0)`, resolved by `narrowType`. Add an abstract `Real`/`FloatWidth 0` sentinel and the `selectRepresentation` resolver. *(Moderate.)*
3. **Single resolver choke point** — resolve at one `selectRepresentation` site, analogous to `narrowType`.
4. **Hard error on unobservability** — `FPGA0001` fires on unbounded ranges; inherit the contract, split on dimensionedness (§6).
5. **Check-time diagnostics** — emit codes inside `checkProgram` exactly as `CCS0100`/`FPGA0001` do.
6. **Platform capability facts** — `PlatformContext.RuntimeModel` is the seat for "does this target have b-posit HW / Xposit / quire support?" (§4).

### 3.2 What is NOT free — the real interval domain is a new abstract interpreter [research-grade]

`IntervalAnalysis.fs` is `{ Min: int64; Max: int64 }` with `modInterval`, `unsignedBitsFor`, and two's-complement bit-counting — **integer-only** (verified: lines 27–31 are the int64 `ValueInterval`). You reuse its *traversal skeleton and carriage discipline* and **nothing of its transfer functions.** The new domain requires, at minimum:

- **Outward-rounded FP interval arithmetic** — endpoints round outward to remain a sound superset.
- **Sign-crossing reciprocal/division** — `1/[lo,hi]` with `0 ∈ [lo,hi]` splits into two unbounded pieces (the `r²`-in-denominator case that makes Tier 1 fall through).
- **Transcendentals** — `sqrt`, `log`, `exp`, `sin`.
- **Terminating widening over a continuous lattice** — the int64 monotone-widening fixpoint does **not** transfer; you need explicit widening operators with thresholds.

```fsharp
/// NEW abstract domain — sibling of IntervalAnalysis, NOT a port of it.
type RealInterval = {
    Lo: float        // outward-rounded toward -inf
    Hi: float        // outward-rounded toward +inf
    Dim: Dimension   // Kennedy unit; carried alongside, not as a type index
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

> **Terminology trap.** "Saturation" is overloaded. `04_saturation_recipes.json` is Baker elaboration, **not** the interval pass. Numeric selection rides `IntervalAnalysis.fs`-style analysis and surfaces only baked into final MLIR types. There is no separate "computation budget" pass — the budget is the clock-period budget spent by depth analysis; numeric selection does not consume it.

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

*(b-posit hardware framing and its qualitative-only citation posture are normative in [`numeric-selection.md` §7](../../clef-lang-spec/spec/numeric-selection.md#7-performance-as-a-capability-gate). Do not reproduce the parity figures here.)*

---

## 5. The Fidelity.Physics integration surface [design not yet ratified — implementation obstacle]

**`Fidelity.Physics` is planned, not built.** Its normative surface is the design sketch in [`numeric-selection.md` §4](../../clef-lang-spec/spec/numeric-selection.md#4-the-fidelityphysics-mechanism-design-sketch); do not write integration code until the binding contract is ratified. This section records the *implementation obstacle* the contract must resolve — which is Composer-HOW, not spec-WHAT.

**The type-checking obstacle.** `Expr<DomainRange<measure>>` **does not type-check**: F# `[<Measure>]` types are a separate kind, not first-class type arguments, and `FSharp.Quotations` reflects over value-level `Expr<'T>`, not measure-level indices. Two coherent designs:

- **Design A — quotation-symbolic (recommended).** The range law is `Expr<float -> ... -> float>` over *unmeasured* floats, its dimension carried by a separate companion attribute (not a measure type-argument). PSG elaboration **symbolically interval-evaluates the quotation AST** to produce `[a,b]`; regime classification then runs over the resulting `[a,b]` *value*, so active patterns are value classifiers, not AST matchers. Matches the Farscape `Expr<PeripheralDescriptor>` precedent.
- **Design B — value-level registry.** The library registers ordinary runtime `DomainRange` values keyed by dimension via an attribute; no quotations. Simpler, but loses compile-time symbolic derivation of derived ranges.

**Decidability precondition (mandatory for Design A).** Symbolic interval evaluation of an arbitrary quotation does not terminate — the dependent-type capability DTS deliberately excludes. The quotation language **must be restricted to a total, terminating sub-language** (closed-form arithmetic + the fixed transcendental set; no recursion, no unbounded fixpoint), or the "decidable-by-construction" claim is unearned. The exact sub-language is OPEN and is a spec deliverable before code.

```fsharp
// Design A surface, five-horsemen attributions preserved:
[<Measure>] type N = kg m / s^2                              // 1. Units of Measure (Kennedy) — keys the range

[<DomainRange("OrbitalMechanics")>]                          // 4. registration glue
let forceLaw : Expr<float -> float -> float -> float> =      // 2. Quotations (Syme) — range LAW over unmeasured floats
    <@ fun gravConst m1Times2 rSquared -> gravConst * m1Times2 / rSquared @>

let (|NearUnityTaper|WideDynamic|MLActivation|)             // 3. Active patterns (Syme) — classify the EVALUATED [a,b]
        (range: RealInterval) = ...

let tier2Claim (psgNode: Node) : RealInterval option =
    lookupDomainRangeAttr psgNode.Dimension
    |> Option.map (fun law ->
        symInterval law (gatherArgIntervals psgNode)          // symbolic interval-eval (terminating sub-language)
        |> tagDimension psgNode.Dimension)
```

**Honest headline.** The `gravForce` example is a **Tier-2 success, not a garden-path success** — Tier 1 cannot lower-bound `r` in the `r²` denominator:

```fsharp
open Fidelity.Physics.OrbitalMechanics
let gravForce (m1: float<kg>) (m2: float<kg>) (r: float<m>) : float<N> =
    GravConst * m1 * m2 / (r * r)   // dim N inferred free; RANGE supplied by the Tier-2 library
```

**ML routing.** A `Fidelity.ML` sibling supplies the range and asymmetric-bias recommendation. Its objective stays on the spec's symmetric argmin fed a **pre-skewed range** (single-objective, per [`numeric-selection.md` §4](../../clef-lang-spec/spec/numeric-selection.md#4-the-fidelityphysics-mechanism-design-sketch) and Requirement 4); a distribution-weighted objective is optional ML-library-only future work, never the general selector.

---

## 6. The default / unobservable case — as code

**Normative contract:** [`numeric-selection.md` §6](../../clef-lang-spec/spec/numeric-selection.md#6-the-default-and-unobservable-case) (report-an-error, no silent default; split on dimensionedness; bare-float→IEEE `f64` is the argmin outcome for an unknown range, not a bypass). Implement:

```fsharp
let resolveUnobservable (node: Node) : Result<Representation, SelectionError> =
    match node.Dimension with
    | Dimensioned _ -> Error (UnboundedDimensionedRange node)   // hard error — like FPGA0001
    | Bare          -> Ok (IEEE F64)                            // the argmin's output for an unknown range
```

### 6.1 The bare/dimensioned seam — handle it explicitly

Normative seam contract: [`numeric-selection.md` §6.1](../../clef-lang-spec/spec/numeric-selection.md#61-the-baredimensioned-seam). Implementation reality: bare floats **do** carry interval analysis (the same real domain of §3.2; "bare works like today" means only that a bare `float` with an unobservable range incurs no error and lowers to f64 — it does not exempt bare floats from range propagation). The error fires **at the dimensioning boundary** with a diagnostic that names the upstream bare source:

```
error: y : float<newtons> requires a bounded range; its range derives from
        x (bare float, unbounded at <site>). Annotate x's range, import a
        domain library, or seal y's representation.
```

*(The `native-type-universe.md §2.4` amendment this seam once forced has been applied: `float`/`float32` are now the bare IEEE lowering targets, representation selected for dimensioned/ranged reals. No further spec change is owed here.)*

---

## 7. Concrete vs parameterized posits — as a lowering-codomain split

**Normative scope split:** [`numeric-selection.md` §8](../../clef-lang-spec/spec/numeric-selection.md#8-concrete-vs-parameterized-representations). Implementation placement:

1. **Surface (Tiers 1–2):** `float<dim>` or a ranged real. Concrete `posit<n,es>` is the Level-3 escape hatch only (seal syntax OPEN).
2. **Lowering codomain on fixed-ISA (CPU/SIMD/RISC-V):** four concrete `Posit8/16/32/64` — direct struct layout, clean SRTP dispatch, clean quire pairing (`Quire32.fma` takes `Posit32` by construction). Selection chooses among the four concretes plus IEEE/fixed. **Concrete types are the codomain of selection, not the surface syntax.**
3. **Parameterized `posit<n,es,rs,bias>`:** FPGA/reconfigurable-only synthesis search — **future work** (CIRCT posit pipeline parameterization not built). The `(rs,es)` grid is enumerable (≤25 points); bias/asymmetry are bounded-but-continuous, explored heuristically.

---

## 8. The quire recognition / sizing / lowering nanopass

**Normative quire semantics** (the `n²/2` adequacy invariant, exactness-not-near-zero-precision, per-target capability): [`numeric-selection.md` §10.2](../../clef-lang-spec/spec/numeric-selection.md#102-the-quire-pass). A Composer nanopass, **downstream of selection** and **before the target-lowering fork**, emitting annotations, not instructions:

```fsharp
let (|QuireMAC|_|) (node: Node) =                  // RECOGNITION — fma/fold/reduce-of-products over a selected posit
    matchFusedProductAccumulation node

let quireWidthBits (n: int) = n * n / 2            // SIZING — n²/2 bits (posit32 → 512 = one cache line)

type QuireCoeffect = {                             // COEFFECT EMISSION — four coeffects on one PSG node
    Allocation : ByteCount * EscapeClass           // 64 B, stack/arena by escape class (reuse escape analysis)
    Lifetime   : Scope
    Capability : ExactAccumulation                 // can this target accumulate exactly?
    Dimension  : Dimension                         // fma of newtons×meters accumulates as joules; dim verified at output
}
// LOWERING deferred to target-binding (the fork).
```

**The sizing invariant is NOT a `k`-cap (implementation trap).** The Posit Standard sizes the quire at `n²/2` precisely so that any practical number of fused products accumulates without overflow — `k` is **not** bounded by the accumulator. **Do not implement a `k·bits-per-product ≤ n²/2` check** (that re-imposes the bound the quire eliminates). The correct QF_BV obligation: for selected width `n`, allocate exactly `n²/2` bits and verify the *per-product intermediate* (full-width product + carry/guard bits) fits the quire field layout — **independent of `k`.** (Normative statement: [`numeric-selection.md` §10.2](../../clef-lang-spec/spec/numeric-selection.md#102-the-quire-pass).)

**Per-target capability (implementation table):**

| Target | Quire support | Resolution |
|---|---|---|
| x86_64 | Software emulation (64 B on stack) | stack; ~50 cycles/FMA |
| Xilinx FPGA | 512-bit fabric pipeline | fabric; 1 cycle/FMA |
| RISC-V + Xposit | Hardware quire instruction | arch. register; 1 cycle/FMA |
| Neuromorphic (Loihi 2) | Not available | **Capability failure** |

---

## 9. Preservation chain — pass ordering

**Normative chain:** [`numeric-selection.md` §10.1](../../clef-lang-spec/spec/numeric-selection.md#101-preservation-chain). Numeric selection is the second arrow; the quire pass realizes the third:

```
Dimension --(range)--> Representation --(width)--> Footprint --(escape)--> Allocation
   DTS        Tier 1/2/3     selection coeffect       n²/2 quire pass    escape analysis
            (composeTiers)   (selectRepresentation)
```

Implementation invariants (all normatively grounded in spec §10.1):
- **Every arrow is a coeffect on the PSG, deferred to target-binding** — later stages have strictly more information. Never resolve a representation eagerly during a pass that only needs to transport it.
- **Carriage.** The representation choice is codata on the PSG beside the dimension, grade, and escape class; a pass that does not touch representation leaves the annotation as it found it. The Composer fixed-point combinator transports the bundle through nanopass lowering; certified proof-transformer passes preserve it by construction; uncertified passes get a per-edge QF_BV Z3 re-check. Codata is navigated, not recomputed (Huet-zipper passive traversal).
- **Transfer fidelity is a separate annotation** — cast fidelity and cross-target transfer fidelity are normatively specified in [`numeric-selection.md` §10.1](../../clef-lang-spec/spec/numeric-selection.md#101-preservation-chain). Implement the transfer-edge annotation (directional, compile-time-derived, 1.0 iff target range covers source range); do not restate the rule here.

---

## 10. Design-time surfacing (Lattice / language server)

Numeric selection emits **check-time diagnostics** in the same shape as `CCS0100`/`FPGA0001` (`Severity`, `Range`, `RelatedNodes`, `Reachability`), from a pass inside `checkProgram`, surfaced as squiggles/hovers. "Design time" = continuous Lattice elaboration; representation-adequacy is a warm-rotation elaboration certificate.

Canonical readouts (figures are illustrative continuous-taper anchors, not a fixed two-point model):

```
force: float<newtons>
  Dimensional range: [1e-11, 1e30] (from gravitational constant + stellar masses; Tier 2)
  ├── x86_64:  float64         (worst-case rel error: 1.11e-16, uniform)      [WideDynamic → IEEE]
  ├── xilinx:  posit<32, es=2> (~2.3e-8 at range extremes, ~1.5e-9 near 1.0)  [NearUnityTaper]
  └── Note: posit gives ~10x better precision in [0.01,100] where 94% of forces reside
```
```
Warning: posit<32,es=2> dynamic range [1e-36,1e36] does NOT cover full dimensional
  range [1e-11,1e72] of astronomicalDistance<meters>          (R_cov filter)
  Consider: float64 (covers full range) or scaling to AU (fits posit range)
```

**New diagnostic codes** (numeric-selection family, siblings of `CCS0100`/`FPGA0001`):
- `coverage-empty` — `R_cov = ∅`
- `near-zero-degeneracy` — range straddles 0 with no representation resolving it under the ULP floor
- `bare-source-unbounded-at-seam` — §6.1
- `tier-disagreement` — lower-tier claim not contained in the bound range (§2)
- `suboptimal-seal` — covering but accuracy-suboptimal Tier-3 seal
- `quire-capability-failure` — §8

> **Citation posture.** The b-posit parity figures and the 5-bit-floor/4-bit-cliff result are qualitative-only and their source identifier is unverified; the normative chapter handles the disclaimer ([`numeric-selection.md` §7](../../clef-lang-spec/spec/numeric-selection.md#7-performance-as-a-capability-gate) and References). Do not reproduce the figures as fact in Composer surfacing or docs.

---

## 11. Implementation map — files, costs, and where each piece sits

| Component | Location | Cost |
|---|---|---|
| **New coeffect** `RepresentationSelection` field on `TransferCoeffects` | `Coeffects.fs` / `TransferTypes.fs`; computed in `MLIRGeneration.fs` beside `widthInference` | Cheap (field + producer) |
| **New sentinel + resolver** abstract `Real`/`FloatWidth 0`; `selectRepresentation` choke point | `TypeMapping.fs` (replace fixed `NTUfloat (Fixed n) → TFloat Fn` ~lines 99–101, 171–173); resolver beside `narrowType` in `PSGCombinators.fs` (~106–157) | Moderate |
| **New abstract domain** real/dimensional interval (outward FP rounding, sign-crossing division, transcendentals, terminating widening) | new module sibling of `IntervalAnalysis.fs` (which stays int64-only) | **Research-grade — dominant cost** |
| **New diagnostics** numeric-selection family | check-time pass merged in `NativeService.checkProgram` (~line 535) | Moderate |
| **New nanopass** quire recognize/size(`n²/2`)/per-product-fit/coeffect-emit | Composer, downstream of selection, before target fork (§8 — no `k`-cap) | Moderate |
| **Capability facts** three-valued b-posit / Xposit / quire-width predicates | extend `PlatformContext` in `NativeTypes.fs` (~357–433, beside `RuntimeModel`/`NsPerWeightUnit`) | Cheap |
| **`Fidelity.Physics`** | ratify Design A vs B + the terminating sub-language *before any code*; do **not** ship `Expr<DomainRange<measure>>` (does not type-check) | Spec-blocked (§5) |

**Verified key source files (absolute):**
`/home/hhh/repos/Composer/src/MiddleEnd/PSGElaboration/IntervalAnalysis.fs` (int64-only `ValueInterval`, lines 27–31 — confirms §3.2) · `/home/hhh/repos/Composer/src/MiddleEnd/Alex/XParsec/PSGCombinators.fs` (`narrowType`/`FPGA0001`) · `/home/hhh/repos/Composer/src/MiddleEnd/Alex/CodeGeneration/TypeMapping.fs` (`IntWidth 0` sentinel; fixed float dispatch ~99–101, 171–173) · `/home/hhh/repos/Composer/src/MiddleEnd/PSGElaboration/Coeffects.fs` · `/home/hhh/repos/clef/src/Compiler/PSGSaturation/SemanticGraph/DepthAnalysis.fs` · `/home/hhh/repos/clef/src/Compiler/NativeTypedTree/NativeTypes.fs` (~357–433) · `/home/hhh/repos/clef/src/Compiler/NativeTypedTree/NativeService.fs` (~535).

*(Normative dependencies this implementation carries: [`numeric-selection.md`](../../clef-lang-spec/spec/numeric-selection.md) is the authority; it cross-references `width-inference.md`, `native-type-universe.md` §2.4, `units-of-measure.md`, `ntu-dimensional-architecture.md`, and `incremental-computation.md`. Do not restate their content here — link it.)*

---

## 12. Phasing / milestones — what to build first

Order chosen so each milestone is independently testable and the highest-risk research item is de-risked early.

**Milestone 0 — Scaffolding (cheap, no behavior change).** Add `RepresentationSelection` to `TransferCoeffects` (trivially populated); add the `Real`/`FloatWidth 0` sentinel behind a flag (current `→ TFloat F32/F64` stays the default resolver output until selection lands); add three-valued capability predicates to `PlatformContext`. *Exit:* full build green, no observable change; the frame exists.

**Milestone 1 — Tier 3 + the objective, on fixed-ISA (smallest end-to-end slice).** Implement `selectRepresentation` (§1) with `R_cov` + the ULP-floored metric over the four concrete `Posit8/16/32/64` + IEEE + fixed; implement Tier-3 sealing (coverage check on a singleton `R`); emit `coverage-empty`, `near-zero-degeneracy`, `suboptimal-seal`. **Blocked on:** seal *syntax* (OPEN — coordinate with the spec). *Exit:* a sealed `float<dim> as posit32` selects/verifies and produces a Lattice readout — proves objective + resolver + diagnostics end-to-end without the hard interval domain.

**Milestone 2 — The real interval domain (research-grade core).** Build `RealIntervalDomain` (§3.2); wire into the PSG traversal skeleton; enable Tier 1 for dataflow-bounded ranges; implement the bare/dimensioned split (§6) and the seam diagnostic (§6.1). *Exit:* a closed-form `float<dim>` computation selects automatically; `r²`-in-denominator correctly falls through with a clear error.

**Milestone 3 — The quire nanopass.** Recognition active pattern, `n²/2` sizing, four-coeffect bundle, per-product-fit QF_BV obligation (§8 — **no `k`-cap**), per-target capability lowering at the fork, `quire-capability-failure`. *Exit:* a `fold`-of-products over `Posit32` lowers to a quire MAC on x86/FPGA/Xposit and fails loudly on Loihi 2.

**Milestone 4 — Tier 2 (`Fidelity.Physics`).** *Spec-blocked:* ratify Design A vs B and the terminating range-expression sub-language **first** (§5). Then implement registration glue, the symbolic interval-evaluator over the restricted sub-language, regime active patterns, and `tier-disagreement` diagnostics. *Exit:* `open Fidelity.Physics.OrbitalMechanics` makes `gravForce` select via Tier 2.

**Future work (do not schedule into the above):** parameterized `posit<n,es,rs,bias>` FPGA synthesis search (§7); the ML distribution-weighted objective (scoped to `Fidelity.ML` only, never the general selector); profiling-evidence provenance/trust model.

**Cross-cutting reminder:** results are coeffects on the PSG, deferred to target-binding, navigated not recomputed (§9). Never resolve a representation eagerly during a pass that only needs to transport it.

---

## 13. The ThreeBody demonstrator (intended accuracy harness)

The **ThreeBody** project (`/home/hhh/repos/ThreeBody`) is meant to be to numeric selection what HelloArty was to width inference: the end-to-end demonstrator of b-posit + quire vs. IEEE-754 accuracy. It is **not yet that** — currently documentation-only (empty `src/.gitkeep`, no `.clef`, no pipeline run), and its docs predate this design and carry drift the spec actively polices. Treat it as a *design proposal to rebuild*, with these guardrails:

- **Cite it (once built and corrected) only for what is true:** quire exactness against catastrophic cancellation (spec §10.2), witnessed by conserved-quantity drift — total energy, angular momentum, and especially **linear momentum** (exactly zero from rest → the tightest pure-arithmetic-drift sensor; the current design omits it). It is a **poor** example for "tapered precision near unity" and not a coverage example until normalization is shown.
- **Normalize to natural units (G = 1)** and state the resulting ranges. Unnormalized SI puts `G ≈ 6.7e-11` into the `[1e-11, 1e30]` wide-dynamic band that routes to **IEEE, not posit** — so without normalization the demo argues against itself.
- **Swap only the number type** across the entire force + integration path; do not confound representation with which subset got exact treatment.
- **Use a symplectic, time-reversible integrator** (leapfrog/Verlet) and an independent high-precision reference trajectory; a non-symplectic method's own secular drift would dwarf the arithmetic signal.
- **Scrub the audit-flagged drift before reuse:** reconcile the b-posit config to a **fixed 800-bit quire** (25×32-bit vector, independent of precision) with **`eS = 5`**; delete the uncited "39% faster decode" claim (cite instead Jonnalagadda et al.'s measured 79% power / 71% area / 60% latency decoder wins, arXiv:2603.01615); remove all "tapered precision near zero/the golden zone" framing.

### 13.1 Reversibility: live re-computation, not tape replay; certification, not identity

If ThreeBody is built as a **negative-types** showcase (typed reversibility) alongside the posit/quire accuracy story, hold these distinctions precisely. **Note:** negative/fractional types are a *proposed, non-normative* extension (see `terms-and-definitions.md`); this section is a demonstrator design sketch, not an implementation commitment.

- **The reversal is live re-computation, never a replayed tape.** At turnaround you negate the momenta and re-run the *same forward operator* (`Φ⁻¹ = S∘Φ∘S`); the backward trajectory is recomputed on the fly, no forward-state history stored. Express it native-Clef: a lazy/thunk-driven reverse pass, region-allocated, with no materialized history buffer. This is the opposite of reverse-mode AD, which tapes a cotangent trace — do not equate them.
- **The negative type *certifies*, it does not *constitute*, the reversal.** The integrator's reversibility is a time-symmetry of the map; the negative type would certify the inverse is structurally complete. Present `S`-conjugacy as "reconstructed from the pairing," never "running the integrator backward."
- **Do not type the velocity flip as `Neg<Velocity>`.** That is value-level additive inverse (`−v`), not a backward-flowing dual. The negative type would ride the step's reverse (adjoint) channel: `SymplecticStep = { forward : State -> State ; adjoint : Neg<State> -> Neg<State> }`, with the N-step reverse as the type-level composition of adjoints.
- **Both IEEE and posit type-check the contract.** A reversibility type is representation-agnostic; what differs is *numerical* reversibility (the measured residual after forward-then-back). The honest claim is **horizon extension, not a typed-contract violation**: b-posit+quire holds the reversal far longer than FP64; neither makes it exact.
- **The quire shrinks the residual; it does not zero it.** Every step still rounds once, and chaos amplifies any residual. "Tracks / returns near" is honest; "bit-reversible" is not — that needs fixed-point/integer invertible steps.

### 13.2 The demo as a curve, with the right panes

Show the **reversal-residual-vs-N curve**, not a single tuned freeze-frame:

- **IEEE FP64** — large, fast-growing residual.
- **b-posit32 + quire** — small, slowly-growing (the headline).
- **fixed-point** — the bit-exact limiting case (the spec offers fixed-point as a selectable representation).
- **FP64 + Kahan summation** — the load-bearing control: the quire is exact compensated summation in the limit, so a skeptic will say Kahan closes the gap. Run this pane and set the thesis strength from the *measured* result. The posit edge that survives Kahan is cancellation in the single subtraction `qᵢ−qⱼ` and in `1/r²`, and posit32 winning there is a genuine taper+accumulation result.

Overlay a design-time **"reversal contract: typechecked ✓"** badge on *both* panes (the structural contract holds for IEEE and posit alike), above the runtime residual curve where only b-posit+quire tracks — separating *structural* reversibility (type, both) from *numerical* reversibility (arithmetic, posit).
