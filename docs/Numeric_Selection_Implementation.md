# Numeric Selection: Composer Implementation Guide

> **Audience:** Composer compiler implementers.
> **Scope:** Building compile-time selection of the numeric *representation* of real-valued quantities (posit / IEEE-754 / fixed-point) as the real-valued sibling of integer width inference.
> **What this is not:** a normative specification. The normative artifact is the future `numeric-selection.md` spec chapter. This guide tells you *where the code goes*, *what the passes compute*, *what data structures carry the results*, and *what to build first*. It marks each item with the thesis's epistemic grade — **[CANON]** (forced by a paper/spec), **[DERIVED]** (forced by canon + the integer twin already in tree), **[PROPOSED]** (a design call not yet ratified) — because some of these will move during spec ratification and you should know which scaffolding is load-bearing versus provisional.

> **The capability to build toward.** The highest-leverage user-facing payoff is **design-time relative-accuracy preservation**: showing, per target, *how much accuracy each candidate representation preserves across the value's actual range*, before anything runs (§10). Tapered (posit) vs. uniform (IEEE) is made visible and quantitative at authoring time — the concrete "this compiler is materially different" demonstration. Prioritize the surfacing (§10) early, even atop a coarse objective, because that is where the leverage shows.
>
> **Governing principle.** Design-time range selection *provisions the runtime envelope*: a representation is selected whose range and precision profile cover the value's range with margin (§1.1), so runtime/training behavior never out-runs it. There is **no runtime re-selection** — preservation is a design-time guarantee, and an out-of-envelope value surfaces as a *loud* condition (coverage-empty, capability-failure), never a silent collapse. Explicit developer seals are sovereign (§2).

---

## 0. The one-paragraph orientation

Composer already ships a complete *integer* representation-selection pipeline: `IntervalAnalysis.fs` (interval domain) → an abstract `IntWidth 0` sentinel at type-lowering → a single `narrowType` resolver → a hard error (`FPGA0001`) when a node's range is unobservable → check-time diagnostics → platform capability facts. **Numeric selection is the missing real-number twin of that pipeline.** Reals currently bypass it entirely: `TypeMapping.fs` hard-binds `NTUfloat (Fixed 32/64) → TFloat F32/F64` before any analysis can run. The work is to slot a parallel real-number flow into the *same frame*. The good news: the PSG traversal skeleton, the coeffect-carriage discipline, the sentinel/resolver pattern, and the diagnostic plumbing are all reusable. The hard news: the *transfer functions* are not — the real interval domain is a new, research-grade abstract interpreter, not a port of the int64 one. Plan the project around that asymmetry.

---

## 1. The selection objective as an algorithm

The core decision procedure is one function. Everything else exists to feed it a range and a candidate set.

**[CANON, dts-dmm §2.6]** the objective over a flat candidate set is:

```
r* = argmin (over r in R(target))  max (over x in [a,b])  |x − round_r(x)| / |x|
```

Properties to preserve in the implementation:
- **Accuracy-only.** No cost or latency term in the score. Performance is handled by *filtering the candidate set* (§4), never by perturbing the objective.
- **Deterministic, compile-time computable** once `[a,b]` is fixed.
- **Per-target.** `R` and therefore `r*` vary by target; the range `[a,b]` does not.

Two side-conditions **[PROPOSED — close genuine soundness gaps, treat as normative in code]** must be wired *into* the objective, not bolted on as diagnostics:

**1.1 Coverage feasibility (`R_cov`).** A non-covering representation saturates to a *finite* relative error (`→ 1.0`), so the raw argmin can pathologically pick it. Filter first:

```fsharp
let R_cov (target: Target) (range: RealInterval) : Representation list =
    R target |> List.filter (fun r -> dynRangeCovers (dynrange r) range)
// If R_cov = [], that is a HARD ERROR (coverage-empty diagnostic), never a near-1.0 silent pick.
```

**1.2 Zero-crossing degeneracy (ULP floor).** `|x − round_r(x)|/|x|` diverges as `x → 0`, and signed dimensioned ranges routinely straddle zero (canon's own `[-80,+40] mV`). Use a mixed absolute/relative metric with a per-representation ULP floor:

```fsharp
// ulpMin r = magnitude of r's smallest positive normal (IEEE) /
//            smallest representable magnitude in the regime covering the range (posit/fixed)
let errAt (r: Representation) (x: float) : float =
    abs (x - roundIn r x) / max (abs x) (ulpMin r)
```

Putting it together — this is the function the resolver (§3) calls:

```fsharp
/// Returns Ok r* | Error <coverage-empty | near-zero-degenerate>
let selectRepresentation
        (target: Target)
        (range: RealInterval)
        (policy: EmulationPolicy)        // §4
        : Result<Representation, SelectionError> =
    let candidates =
        R target
        |> List.filter (capabilityAvailable target)        // CANON: R = capability ≠ unavailable
        |> List.filter (fun r -> dynRangeCovers (dynrange r) range)   // §1.1 R_cov
        |> applyEmulationPolicy target policy              // §4 R_eff
    match candidates with
    | [] -> Error CoverageEmpty                            // R_cov / R_eff empty → hard error
    | cs ->
        // worst-case over [a,b]; sample regime boundaries + endpoints, ULP-floored
        let score r = worstCaseError r range               // uses errAt; finite by ULP floor
        cs |> List.minBy score |> Ok
```

**Implementation note on `worstCaseError`.** Do **not** hard-code posit32 figures (`2⁻²⁷` near 1.0, `2⁻⁸` at extremes). Those are *continuous-taper anchors*, not a two-point model. Compute the taper at the *actual* range endpoints and regime boundaries that `[a,b]` crosses. IEEE's `≈2⁻ᵖ` is uniform except near zero where the subnormal/`emin` floor governs (this is exactly what the ULP floor captures).

---

## 2. The tiered authority model and its PSG mapping

The three tiers are **not three selection algorithms** — they are three *provenances and bindingness levels* of the one function's `[a,b]` input. **[CANON, dts-dmm §2.2]** all provenances feed the same selector and produce identical lowering.

| Tier | Range source | Inference Level | Developer writes | Build status |
|---|---|---|---|---|
| **1 — intrinsic** | interval analysis + profiling | Level 1 (inferred) | nothing | needs new real interval domain (§3) |
| **2 — library** | `Fidelity.Physics` constants/laws | Level 2 (bounded) | `open Fidelity.Physics.X` | needs unbuilt library + contract (§5) |
| **3 — direct** | explicit seal at the site | Level 3 (explicit) | seal form (syntax OPEN) | buildable today *except* seal syntax |

**Honest Tier-1 scope.** Tier 1 succeeds *automatically* only when the range is bounded by dataflow alone: closed-form, division-free or division-with-known-nonzero-lower-bound, profiled, or annotated. Real physics with division by a quantity whose lower bound is not in dataflow (e.g. `r²` in a denominator) **falls through to Tier 2/3 or to the §6 error.** Do not over-promise the garden path.

**Tier 3 is sealing** **[CANON, dts-dmm §2.6 bidirectional]:** the developer fixes a representation; the compiler runs selection *in reverse* — verify the sealed representation's dynrange covers the range (the §1.1 coverage check on a singleton `R`). A covering-but-suboptimal seal still compiles; Lattice witnesses the suboptimality.

### 2.1 Tier composition — precedence-override, NOT intersection [PROPOSED]

Each present tier yields a **range claim**, not a constraint to intersect. Implement a single, total, precedence-ordered refinement:

```fsharp
type TierClaim =
    | Tier1 of RealInterval          // dataflow, if analysis terminated with a bound
    | Tier2 of RealInterval          // library
    | Tier3 of Representation         // sealed; implies its dynrange as a range claim

/// Returns the binding range + any disagreement obligations to report.
let composeTiers (claims: TierClaim list) : RealInterval * Disagreement list =
    let binding =
        // highest present tier wins; it OVERRIDES, it does not merge
        claims |> List.sortByDescending tierRank |> List.head |> claimRange
    let obligations =
        claims
        |> List.choose (fun c ->
            let lower = claimRange c
            // lower-tier observation must sit inside the authoritative range (± tol, OPEN)
            if not (contains binding (inflate lower tolerance) lower)
            then Some (TierDisagreement (c, binding))   // diagnostic; does NOT change binding
            else None)
    binding, obligations
```

Rules to keep straight:
- The binding range is the claim of the **highest present tier** (`Tier3 > Tier2 > Tier1`). Higher tiers override.
- Lower-tier claims become **consistency obligations**, not inputs to the selected range. If observed dataflow `R₁` is not contained in the higher tier's declared range, emit a **tier-disagreement diagnostic** (the real-valued analogue of integer overflow). The diagnostic does **not** change the binding range.
- There is **no empty-intersection state** — composition is total and decidable.

---

## 3. Riding (and not riding) the width-inference infrastructure

### 3.1 What is genuinely free — replicate these patterns verbatim

**[DERIVED]** the integer twin gives you six reusable patterns:

1. **Coeffect computed pre-emission.** `IntervalAnalysis.analyze` runs once-per-graph; the result is carried as `TransferCoeffects.WidthInference`, computed in `MLIRGeneration.fs`. Add a **sibling field** `RepresentationSelection` on `TransferCoeffects`, computed beside `widthInference`. *(Cheap — new field + new producer.)*
2. **Abstract sentinel at type-lowering.** Integers lower to `TInt (IntWidth 0)`, resolved by `narrowType`. Add an abstract `Real` / `FloatWidth 0` sentinel and a `selectRepresentation` resolver. *(Moderate — new sentinel + new resolver.)*
3. **Single resolver choke point.** Resolve at one `selectRepresentation` site, analogous to `narrowType`.
4. **Hard error on unobservability.** `FPGA0001` fires on unbounded ranges; inherit the contract, split on dimensionedness (§6).
5. **Check-time diagnostics.** Emit codes inside `checkProgram` exactly as `CCS0100`/`FPGA0001` do.
6. **Platform capability facts.** `PlatformContext.RuntimeModel` is the seat for "does this target have b-posit HW / Xposit / quire support?" (§4).

### 3.2 What is NOT free — the real interval domain is a new abstract interpreter [PROPOSED, research-grade]

`IntervalAnalysis.fs` is `{ Min: int64; Max: int64 }` with `modInterval`, `unsignedBitsFor`, and two's-complement bit-counting — **integer-only, confirmed in-tree** (verified: lines 27–31 are the int64 `ValueInterval`). You reuse its *traversal skeleton and carriage discipline* and **nothing of its transfer functions.** The new domain requires, at minimum:

- **Outward-rounded FP interval arithmetic** — endpoints must round outward to remain a sound superset.
- **Sign-crossing reciprocal/division** — `1/[lo,hi]` with `0 ∈ [lo,hi]` splits into two unbounded pieces. This is exactly the `r²`-in-denominator case that makes Tier 1 fall through.
- **Transcendentals** — `sqrt`, `log`, `exp`, `sin`. The physics cases need them.
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

**[DERIVED]** width inference is the *spatial* reading (bits/value); depth/budget inference (`DepthAnalysis.fs`, `foldWithLambdaPreBind`) is the *temporal* reading; numeric selection is the *third* reading (consume the dimensional range, select a representation). "Same traversal" means the same graph walk and carriage — **not** the same transfer functions (§3.2).

> **Terminology trap.** "Saturation" is overloaded. `04_saturation_recipes.json` is Baker elaboration, **not** the interval pass. Numeric selection rides `IntervalAnalysis.fs`-style analysis; it surfaces only baked into final MLIR types. There is no separate "computation budget" pass — the budget is the clock-period budget spent by depth analysis; numeric selection does not consume it.

---

## 4. The performance/capability gate (not a cost term) [PROPOSED]

Keep the objective pure-accuracy. Make performance a **three-valued capability filter** on the candidate set: `native | emulated | unavailable`.

```fsharp
type Capability = Native | Emulated | Unavailable
type EmulationPolicy = NativeOnly | AllowEmulated | AllowEmulatedWarn   // default AllowEmulated

let applyEmulationPolicy target policy (cands: Representation list) =
    match policy with
    | NativeOnly      -> cands |> List.filter (fun r -> capability target r = Native)
    | AllowEmulated   -> cands
    | AllowEmulatedWarn ->
        // keep all; record that a perf diagnostic should fire IF r* turns out emulated
        cands   // the warn is emitted at diagnostic time, not selection time
```

The layering against canon, made explicit so reviewers don't mistake it for the paper's equation:
- `R(target)` = `{ r : capability ≠ unavailable }` — **this is canon's flat `R`.**
- `R_cov` (§1.1) and the emulation policy (§4) are **proposed refinements layered on top.**

**Why not a blended `λ·cost` objective** **[DERIVED]:** (1) worst-case relative error is closed-form and µarch-independent; cost is not, and folding it in smuggles delay-model guessing the HelloArty epistemics forbid. (2) "No janky fallback" — a blended objective silently degrades; the capability gate produces a **binary, witnessed** outcome. An emulated accuracy-optimal choice yields a *diagnostic, not a silent swap*. This generalizes **[CANON, dts-dmm §3.5]:** a target lacking the quire triggers a capability-coeffect *failure*, never a lossy fallback.

**b-posit note** **[CANON, adaptive-domain-models §3]:** b-posits constrain `rs ≤ 6` (five MUX-selectable possibilities) with a non-significand field (`1 + rs + es` bits) identical across 16/32/64-bit operands — one decoder, three precisions. **Once the float/posit hardware gap closes, more targets report `capability = Native` and the gate becomes a no-op on those targets.** The design does not load-bear on the exact parity figures (see citation quarantine, §10).

> **`allow-emulated-warn` caveat:** emulated-ness influences *which diagnostic fires*, never *which representation is chosen*. The purity invariant is scoped to the **choice**, not to what gets reported about it.

---

## 5. The Fidelity.Physics integration surface [PROPOSED SKETCH — not yet specifiable]

**[CANON, dts-dmm §2.2/§2.6]** `Fidelity.Physics` is **planned, not built.** Do not write integration code until the binding contract is ratified. This section defines the *surface* the selector consumes and the obstacle the contract must resolve.

**The type-checking obstacle.** `Expr<DomainRange<measure>>` **does not type-check**: F# `[<Measure>]` types are a separate kind, not first-class type arguments, and `FSharp.Quotations` reflects over value-level `Expr<'T>`, not measure-level indices. You must pick one of two coherent designs:

- **Design A — quotation-symbolic (recommended).** The range law is `Expr<float -> ... -> float>` over *unmeasured* floats, with its dimension carried by a **separate companion attribute** (not a measure type-argument). PSG elaboration **symbolically interval-evaluates the quotation AST** to produce `[a,b]`. Regime classification then runs over the resulting `[a,b]` *value* (post-evaluation), so active patterns are *value classifiers*, not AST matchers. Matches the documented Farscape `Expr<PeripheralDescriptor>` precedent.
- **Design B — value-level registry.** Library registers ordinary runtime `DomainRange` values keyed by dimension via an attribute; no quotations. Simpler, but loses compile-time symbolic derivation of derived ranges.

**Decidability precondition (mandatory for Design A).** Symbolic interval evaluation of an arbitrary quotation is **not terminating** — that is the dependent-type capability DTS deliberately excludes. The quotation language **must be restricted to a total, terminating sub-language** (closed-form arithmetic + the fixed transcendental set; no recursion, no unbounded fixpoint). Without this restriction the "decidable-by-construction" claim is unearned. The exact sub-language is OPEN.

The surface the selector consumes (Design A), with five-horsemen attributions preserved:

```fsharp
// 1. Units of Measure (Kennedy) — the dimension that KEYS the range (registered-under, not parameterized-by)
[<Measure>] type N = kg m / s^2

// 2. Quotations (Syme) — a range-LAW expression over UNMEASURED floats, à la Farscape's Expr<PeripheralDescriptor>
//    Symbolically interval-evaluated within the terminating sub-language to a dimensioned [a,b].
[<DomainRange("OrbitalMechanics")>]                       // 4. registration glue
let forceLaw : Expr<float -> float -> float -> float> =
    <@ fun gravConst m1Times2 rSquared -> gravConst * m1Times2 / rSquared @>

// 3. Active patterns (Syme) — classify the EVALUATED [a,b] VALUE into a regime (post-symbolic-eval)
let (|NearUnityTaper|WideDynamic|MLActivation|) (range: RealInterval) = ...
```

How elaboration consumes it:

```fsharp
let tier2Claim (psgNode: Node) : RealInterval option =
    lookupDomainRangeAttr psgNode.Dimension          // find registered law by dimension key
    |> Option.map (fun law ->
        symInterval law (gatherArgIntervals psgNode)  // symbolic interval-eval (terminating sub-language)
        |> tagDimension psgNode.Dimension)            // re-attach the dimension to the bare [a,b]
```

**Honest headline.** The famous `gravForce` example is a **Tier-2 success, not a garden-path success** — Tier 1 cannot lower-bound `r` in the `r²` denominator:

```fsharp
open Fidelity.Physics.OrbitalMechanics
let gravForce (m1: float<kg>) (m2: float<kg>) (r: float<m>) : float<N> =
    GravConst * m1 * m2 / (r * r)   // dim N inferred free (HM+ℤ); RANGE supplied by Tier 2 library
```

**ML routing** **[CANON, adaptive-domain-models §6.9]:** a `Fidelity.ML` sibling ships the asymmetric b-posit (asymmetric es/rs, shifted bias, narrow `[1e-14,1e1]`, 5-bit floor / 4-bit cliff — *citation-flagged*). Note its objective is **distribution-weighted, not §1's symmetric worst-case argmin** — a distinct optimization, scoped strictly to the ML library, never the general selector (OPEN, §11.6 of the thesis).

---

## 6. The default / unobservable case [PROPOSED — split on dimensionedness]

Inherit width inference's "report-an-error, no silent default," split on whether the value carries a dimension:

```fsharp
let resolveUnobservable (node: Node) : Result<Representation, SelectionError> =
    match node.Dimension with
    | Dimensioned _ -> Error (UnboundedDimensionedRange node)   // hard error — like FPGA0001
    | Bare          -> Ok (IEEE F64)                            // NOT a forbidden silent default
```

- **Dimensioned + unobservable → ERROR.** The dimension is a *promise that a domain range exists*; silently picking IEEE discards the dimensional information the developer paid for. Ask for an annotation or a `Fidelity.Physics` import.
- **Bare `float` + unobservable → IEEE `f64`, no error.** This is **the argmin's correct output for an unknown range**, not a bypass: IEEE's uniform `2⁻ᵖ` error is precisely "no commitment about where values cluster" (max-entropy / no-bet). **[CANON, dts-dmm App.C]** the x86 binding names f64 default *and then runs selection and reports "Selection: float64" as the argmin outcome* — understand f64 as a selected outcome for wide/unobservable ranges, not a fallback.

### 6.1 The bare/dimensioned seam — handle it explicitly

Ranges are composition-dependent. `let x : float = bareInput in let y : float<newtons> = x * oneNewton` — `y`'s range derives from `x`'s, which was never bounded. The resolution:

- **Bare floats DO carry interval analysis** (it is the same real domain of §3.2; there is no second analysis to switch off). "Bare works like today" means **only**: *a bare `float` with an unobservable range incurs no error and lowers to f64.* It does **not** exempt bare floats from range propagation.
- **The dimensioning site is where the range obligation crystallizes.** If an inflowing bare value's range is unobservable, the error fires **at the dimensioning boundary** (`x * oneNewton`) with a diagnostic that **names the upstream bare source**:

```
error: y : float<newtons> requires a bounded range; its range derives from
        x (bare float, unbounded at <site>). Annotate x's range, import a
        domain library, or seal y's representation.
```

This points at the real cause, not the wrong node.

**Spec amendment this forces.** `native-type-universe.md` §2.4 (lines ~163–196, currently IEEE-only) must be revised: `float`/`float32` remain the **bare (no-locality-claim) types defaulting to IEEE**, but f64 is a *lowering target selected for wide/unobservable ranges*, not the universal declared representation. The `float = f64` / `float32 = f32` NTU mappings (lines ~1280–1281) remain correct **as lowering targets.** Cross-reference `width-inference.md` §4.

---

## 7. Concrete vs parameterized posits — three-layer scope split [PROPOSED]

Keep concrete and parameterized posits on opposite sides of the CPU/FPGA fork so they never collide:

1. **Surface (Tiers 1–2):** `float<dim>` or a ranged real. No posit width on the garden path. Concrete `posit<n,es>` is the Level-3 escape hatch only (seal syntax OPEN).
2. **Lowering codomain on fixed-ISA (CPU/SIMD/RISC-V):** four concrete `Posit8/16/32/64` — direct struct layout, clean SRTP dispatch, clean quire pairing (`Quire32.fma` takes `Posit32` by construction). Selection chooses *among the four concretes* plus IEEE/fixed. **Concrete types are the codomain of selection, not the surface syntax.**
3. **Parameterized `posit<n,es,rs,bias>`:** FPGA/reconfigurable-only synthesis search. **[CANON future work, dts-dmm §6.5.]** The `(rs,es)` grid is enumerable (≤25 points: `rs ∈ [2,6] × es ∈ [1,5]`); **bias and asymmetry are bounded-but-continuous, explored heuristically, NOT enumerated.** This is type-directed hardware synthesis, and it is **future work** (CIRCT posit pipeline parameterization not built).

**ML routing:** Tier-2 library supplies `[1e-14,1e1]` + asymmetric-bias recommendation → parameterized b-posit on FPGA, nearest concrete `Posit8` on fixed-ISA. The 5-bit floor / 4-bit cliff surfaces as a Lattice coverage warning.

---

## 8. The quire recognition / sizing / lowering pass [PROPOSED placement]

A Composer nanopass, **downstream of selection** (depends on the selected posit width) and **before the target-lowering fork.** It emits *annotations, not instructions* — same coeffect discipline as selection.

**Pipeline stages:**

```fsharp
// 1. RECOGNITION — active pattern over PSG nodes
let (|QuireMAC|_|) (node: Node) =
    // fma / fold / reduce-of-products over a SELECTED posit type
    // e.g. Array.fold (fun acc (a,b) -> acc + a*b) zero  →  fused quire MAC
    matchFusedProductAccumulation node

// 2. SIZING — CANON, dts-dmm §3.5 / Posit Standard 2022: quire width = n²/2 bits
let quireWidthBits (n: int) = n * n / 2        // posit32 → 512 bits = one cache line

// 3. COEFFECT EMISSION — four coeffects on one PSG node [CANON §3.5]
type QuireCoeffect = {
    Allocation : ByteCount * EscapeClass        // 64 B, stack/arena by escape class
    Lifetime   : Scope                          // loop scope unless escaping (reuse escape analysis)
    Capability : ExactAccumulation              // can this target accumulate exactly?
    Dimension  : Dimension                      // fma of newtons×meters accumulates as joules;
}                                               // single rounding at Quire.toPosit; dim verified at output

// 4. LOWERING deferred to target-binding (the fork)
```

### 8.1 The sizing invariant is NOT a `k`-cap — get this right

The Posit Standard sizes the quire at `n²/2` *precisely so that any practical number of fused products accumulates without overflow* — `k` (accumulation length) is **not** bounded by the accumulator. **Do not implement a `k·bits-per-product ≤ n²/2` check** (that re-imposes the bound the quire eliminates; it is arithmetically backwards). The correct obligation:

> **Quire-adequacy:** for selected posit width `n`, allocate exactly `n²/2` bits and verify (QF_BV obligation) that the *per-product intermediate* (full-width product + carry/guard bits) fits the quire field layout — **independent of `k`.**

### 8.2 Per-target capability table [CANON, dts-dmm Table tab:quire-targets, verbatim]

| Target | Quire support | Resolution |
|---|---|---|
| x86_64 | Software emulation (64 B on stack) | stack; ~50 cycles/FMA |
| Xilinx FPGA | 512-bit fabric pipeline | fabric; 1 cycle/FMA |
| RISC-V + Xposit | Hardware quire instruction | arch. register; 1 cycle/FMA |
| Neuromorphic (Loihi 2) | Not available | **Capability failure** |

### 8.3 Why the quire matters — exactness, not near-zero precision

**[CANON, decidable-by-construction §3.2]** IEEE rounding turns Clifford/Cayley *structural zeros* into numerically non-zero phantom grades; the quire's **exact accumulation** keeps structural zeros structurally zero through training. **Do not claim "posit tapered precision near zero"** — that is arithmetically false (posits taper toward **1.0**; near 0 is a regime extreme where they *lose* precision). The quire's value is *exactness of accumulation*, full stop.

---

## 9. The preservation chain — pass ordering

**[CANON, decidable-by-construction §4.1]** the chain — each pass consumes the previous pass's output:

```
Dimension --(range)--> Representation --(width)--> Footprint --(escape)--> Allocation
   DTS        Tier 1/2/3     selection coeffect       n²/2 quire pass    escape analysis
            (composeTiers)   (selectRepresentation)
```

Implementation invariants:
- **Every arrow is a coeffect on the PSG, deferred to target-binding** **[CANON, deferred-optimization, dts-dmm §3.4]** — later stages have strictly more information. Numeric selection is the **second arrow**; the quire pass realizes the third.
- **Carriage** **[CANON, fixed-point-scaffolding §4]:** the representation choice is **codata on the PSG beside the dimension, grade, and escape class.** "A lowering pass that does not touch representation leaves the annotation as it found it." The Composer fixed-point combinator transports the bundle through nanopass lowering; certified proof-transformer passes preserve it by construction (Ohori); uncertified passes get a per-edge QF_BV Z3 re-check. Codata is **navigated, not recomputed** (Huet-zipper passive traversal).
- **Transfer fidelity is a SEPARATE annotation** (dts-dmm §4.5) from representation specs. Label `posit32→f64` as **"cast fidelity 1.0 (one-way widen)"** — posit32's ≤28 significand bits near 1.0 fit f64's 52 — and state that **round-trips `f64→posit32→f64` are < 1.0.** Do not let the 1.0 label imply a posit32 value is f64-accurate.

---

## 10. Design-time surfacing (Lattice / language server) [DERIVED]

Numeric selection emits **check-time diagnostics** in the same shape as `CCS0100`/`FPGA0001` (`Severity`, `Range`, `RelatedNodes`, `Reachability`), from a pass inside `checkProgram`, surfaced as squiggles/hovers. "Design time" = continuous Lattice elaboration; representation-adequacy is a warm-rotation elaboration certificate.

Canonical readouts (figures are CANON-illustrative continuous-taper anchors, not a fixed two-point model):

```
force: float<newtons>
  Dimensional range: [1e-11, 1e30] (from gravitational constant + stellar masses; Tier 2)
  ├── x86_64:  float64         (worst-case rel error: 1.11e-16, uniform)      [WideDynamic → IEEE]
  ├── xilinx:  posit<32, es=2> (~2.3e-8 at range extremes, ~1.5e-9 near 1.0)  [NearUnityTaper]
  └── Note: posit gives ~10x better precision in [0.01,100] where 94% of forces reside
```
```
Warning: posit<32,es=2> dynamic range [1e-36,1e36] does NOT cover full dimensional
  range [1e-11,1e72] of astronomicalDistance<meters>          (R_cov filter, §1.1)
  Consider: float64 (covers full range) or scaling to AU (fits posit range)
```

**New diagnostic codes** (numeric-selection family, siblings of `CCS0100`/`FPGA0001`):
- `coverage-empty` — `R_cov = ∅`
- `near-zero-degeneracy` — range straddles 0 with no representation resolving it under the ULP floor
- `bare-source-unbounded-at-seam` — §6.1
- `tier-disagreement` — `R_lower ⊄ R_binding` (§2.1)
- `suboptimal-seal` — covering but accuracy-suboptimal Tier-3 seal
- `quire-capability-failure` — §8.2

### 10. Citation quarantine — DO NOT reproduce these figures as fact

The b-posit parity numbers (**79% less power / 71% smaller area / 60% reduced latency**, `adaptive-domain-models.tex:177`) and the 5-bit-floor/4-bit-cliff NUS result trace to `jonnalagadda2025` / `jonnalagadda2025bposit`, cited **inconsistently** (a conference paper in one source, an arXiv preprint in the other) against the **future-dated, unverifiable** `arXiv:2603.01615` (a March-2026 ID, past the knowledge cutoff). **The spec chapter must NOT reproduce 79/71/60 as citable fact.** State the *qualitative* claim ("b-posits aim to close the float/posit hardware-efficiency gap; if achieved, native-posit targets grow") and footnote the citation as unverified/inconsistent. The design is robust by construction: if parity proves weaker, more targets report `capability = Emulated`, the gate filters them under `native-only`, and the accuracy-vs-range conclusions are unchanged — only the *size of the native candidate set* moves.

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
| **`Fidelity.Physics`** | specify Design A vs B + terminating sub-language *before any code*; do **not** ship `Expr<DomainRange<measure>>` (does not type-check) | Spec-blocked |

**Verified key source files (absolute):**
`/home/hhh/repos/Composer/src/MiddleEnd/PSGElaboration/IntervalAnalysis.fs` (int64-only `ValueInterval`, confirmed lines 27–31 — confirms §3.2) · `/home/hhh/repos/Composer/src/MiddleEnd/Alex/XParsec/PSGCombinators.fs` (`narrowType`/`FPGA0001`) · `/home/hhh/repos/Composer/src/MiddleEnd/Alex/CodeGeneration/TypeMapping.fs` (`IntWidth 0` sentinel; fixed float dispatch ~99–101, 171–173) · `/home/hhh/repos/Composer/src/MiddleEnd/PSGElaboration/Coeffects.fs` · `/home/hhh/repos/clef/src/Compiler/PSGSaturation/SemanticGraph/DepthAnalysis.fs` · `/home/hhh/repos/clef/src/Compiler/NativeTypedTree/NativeTypes.fs` (~357–433) · `/home/hhh/repos/clef/src/Compiler/NativeTypedTree/NativeService.fs` (~535).

**Spec deliverables:** new chapter `numeric-selection.md` mirroring `width-inference.md`, carrying the corrected objective (`R_cov` + ULP floor), the precedence-override tier composition, and the citation quarantine; amend `native-type-universe.md` §2.4; cross-reference `incremental-computation.md` §12 (Level ladder), `units-of-measure.md`, `ntu-dimensional-architecture.md`.

---

## 12. Phasing / milestones — what to build first

Order chosen so each milestone is independently testable and the highest-risk research item is de-risked early rather than blocking everything.

**Milestone 0 — Scaffolding (cheap, no behavior change).**
- Add the `RepresentationSelection` field to `TransferCoeffects` (computed but trivially populated).
- Add the `Real`/`FloatWidth 0` sentinel to `TypeMapping.fs` behind a flag; current behavior (`→ TFloat F32/F64`) becomes the default resolver output until selection lands.
- Add three-valued capability predicates to `PlatformContext`.
- *Exit criterion:* full build green, no observable change; the frame exists.

**Milestone 1 — Tier 3 + the objective, on fixed-ISA (smallest end-to-end slice).**
- Implement `selectRepresentation` (§1) with `R_cov` + the ULP-floored metric, over the four concrete `Posit8/16/32/64` + IEEE + fixed.
- Implement Tier-3 *sealing* (selection-in-reverse / coverage check on a singleton `R`).
- Emit `coverage-empty`, `near-zero-degeneracy`, `suboptimal-seal` diagnostics.
- **Blocked on:** seal *syntax* (OPEN — the critical-path gap; Tier 3 is otherwise the one tier buildable today). Coordinate with the spec on the seal form before/in parallel.
- *Exit criterion:* a sealed `float<dim> as posit32` selects/verifies and produces a Lattice readout. This proves the objective + resolver + diagnostics end-to-end without the hard interval domain.

**Milestone 2 — The real interval domain (the research-grade core).**
- Build `RealIntervalDomain` (§3.2): outward-rounded FP arithmetic, sign-crossing reciprocal/division, the transcendental set, terminating widening.
- Wire it into the PSG traversal skeleton (reuse the `IntervalAnalysis.analyze` walk shape).
- Enable Tier 1 for dataflow-bounded ranges (closed-form, division-with-known-bound, profiled, annotated).
- Implement the bare/dimensioned split (§6) and the seam diagnostic (§6.1).
- *Exit criterion:* a closed-form `float<dim>` computation selects automatically; `r²`-in-denominator correctly *falls through* with a clear error.

**Milestone 3 — The quire nanopass.**
- Recognition active pattern, `n²/2` sizing, four-coeffect bundle, per-product-fit QF_BV obligation (§8 — **no `k`-cap**), per-target capability lowering at the fork, `quire-capability-failure`.
- *Exit criterion:* a `fold`-of-products over `Posit32` lowers to a quire MAC on x86/FPGA/Xposit and *fails loudly* on Loihi 2.

**Milestone 4 — Tier 2 (`Fidelity.Physics`).**
- *Spec-blocked:* ratify Design A vs B and the terminating range-expression sub-language **first** (§5). Then implement registration glue, the symbolic interval-evaluator over the restricted sub-language, regime active patterns, and `tier-disagreement` diagnostics (§2.1).
- *Exit criterion:* `open Fidelity.Physics.OrbitalMechanics` makes the `gravForce` example select via Tier 2.

**Future work (do not schedule into the above):**
- Parameterized `posit<n,es,rs,bias>` FPGA synthesis search (§7) — `(rs,es)` grid enumerable, bias/asymmetry heuristic; CIRCT pipeline not built.
- The ML asymmetric/distribution-weighted objective (§5) — distinct optimization, scoped to `Fidelity.ML` only, never the general selector.
- Profiling-evidence provenance/trust model (recording/versioning across builds).

**Cross-cutting reminder for every milestone:** results are **coeffects on the PSG, deferred to target-binding**, navigated not recomputed (§9). Never resolve a representation eagerly during a pass that only needs to *transport* it.

---

## 13. The ThreeBody demonstrator (intended accuracy harness)

The **ThreeBody** project (`/home/hhh/repos/ThreeBody`) is meant to be to numeric selection what HelloArty was to width inference: the end-to-end demonstrator of posit + quire vs. IEEE-754 accuracy. It is **not yet that** — it is currently documentation-only (empty `src/.gitkeep`, no `.clef`, no pipeline run), and its docs predate this design and carry drift the spec actively polices. Treat it as a *design proposal to rebuild*, with these guardrails:

- **Cite it (once built and corrected) only for what is true:** **quire exactness against catastrophic cancellation** (§8.3), witnessed by conserved-quantity drift — total energy, angular momentum, and especially **linear momentum** (exactly zero from rest → the tightest pure-arithmetic-drift sensor; the current design omits it). It is a **poor** example for "tapered precision near unity" and not a coverage example until normalization is shown.
- **Normalize to natural units (G = 1)** and state the resulting ranges. Unnormalized SI puts `G ≈ 6.7e-11` into the `[1e-11, 1e30]` wide-dynamic band that §1/§6 route to **IEEE, not posit** — so without normalization the demo argues *against* itself.
- **Swap only the number type** across the entire force + integration path; do not confound representation with which subset got exact treatment.
- **Use a symplectic, time-reversible integrator** (leapfrog/Verlet) and an **independent high-precision reference** trajectory for absolute divergence; a non-symplectic method's own secular drift would dwarf the arithmetic signal.
- **Scrub the audit-flagged drift before reuse:** 800-bit quire → **512-bit** (`n²/2` for posit32); delete the uncited **"39% faster decode"** claim; reconcile the b-posit config to **`es = 2`** (not `es = 5`); and remove all "tapered precision near zero/the golden zone" framing (the same falsehood also sits in `adaptive-domain-models.tex:470` and should be corrected upstream).

When rebuilt on these terms, ThreeBody becomes a *stronger* worked example than the abstract gravitation snippet of §5/§10, because it adds the conserved-quantity error metrics that snippet lacks.

### 13.1 Reversibility: live re-computation, not tape replay; certification, not identity

If ThreeBody is built as a **negative-types** showcase (typed reversibility) alongside the posit/quire accuracy story, hold these distinctions precisely — they are the difference between a sound demo and an equivocation:

- **The reversal is live re-computation, never a replayed tape.** At the turnaround you negate the momenta and re-run the *same forward operator* (`Φ⁻¹ = S∘Φ∘S`). The backward trajectory is *recomputed on the fly*; no forward-state history is stored or read back. Express it the native-Clef way: a **lazy/thunk-driven** reverse pass, region-allocated, with **no materialized history buffer** ("without interference of computational memory"). This is the opposite of reverse-mode AD, which *does* tape a cotangent trace — do not equate them.
- **The negative type *certifies*, it does not *constitute*, the reversal.** The integrator's reversibility is a time-symmetry of the map; the negative type's role is to *certify the inverse is structurally complete* (the §7 completeness obligation, which leapfrog discharges trivially — its reversal information is just the sign convention, with no stored trace). Present `S`-conjugacy as "reconstructed from the pairing," never as "running the integrator backward."
- **Do not type the velocity flip as `Neg<Velocity>`.** That is value-level additive inverse (`−v`), not a backward-flowing dual; mis-typing it bakes the category error into the code. The negative type rides the **step's reverse (adjoint) channel**: `SymplecticStep = { forward : State -> State ; adjoint : Neg<State> -> Neg<State> }`, with the N-step reverse as the type-level composition of adjoints. Use the adjoint (self-inverse) reading, not the backtracking reading.
- **Both IEEE and posit type-check the contract.** A reversibility type is representation-agnostic. What differs is *numerical* reversibility — the measured residual after forward-then-back. So the honest claim is **horizon extension, not a typed-contract violation**: posit+quire holds the reversal far longer than FP64; neither makes it exact.
- **The quire shrinks the residual; it does not zero it.** Every step still rounds once, and chaos amplifies any residual. "Tracks / returns near" is honest; "bit-reversible" is not — that needs fixed-point/integer invertible steps.

### 13.2 The demo as a curve, with the right panes

Show the **reversal-residual-vs-N curve**, not a single tuned freeze-frame:

- **IEEE FP64** — large, fast-growing residual.
- **posit32 + quire** — small, slowly-growing (the headline).
- **fixed-point** — the bit-exact limiting case (the spec already offers fixed-point as a selectable representation).
- **FP64 + Kahan summation** — the load-bearing control: the quire is exact compensated summation in the limit, so a skeptic will say Kahan closes the gap. Run this pane and set the thesis strength from the *measured* result. The posit edge that survives Kahan is cancellation in the single subtraction `qᵢ−qⱼ` and in `1/r²` (which Kahan does not touch but posit's near-unity tapered precision plus quire-exact dot products do), and posit32 winning here is a genuine taper+accumulation result because it is punching above FP64's bit class.

Overlay a design-time **"reversal contract: typechecked ✓"** badge on *both* panes (the structural contract holds for IEEE and posit alike), above the runtime residual curve where only posit+quire tracks. That visual cleanly separates *structural* reversibility (type, both) from *numerical* reversibility (arithmetic, posit) — and is the honest version of the LED go/no-go idea.