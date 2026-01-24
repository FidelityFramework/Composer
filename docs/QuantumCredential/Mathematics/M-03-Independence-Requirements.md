# M-03: Independence Requirements

## Overview

The bias reduction formula (ε → 8ε⁴) from M-02 rests on a critical assumption: **statistical independence** between channels. This document examines what independence means, why it matters, and how violations manifest.

Understanding independence is essential for hardware/software co-design because it defines the *contract* between mathematics and physics. The software can rely on the mathematics only if the hardware delivers independence.

---

## For Software Engineers: What Is Independence?

In software, we often conflate "separate" with "independent." Two threads running in parallel are separate, but they may contend for shared resources in correlated ways. Two network connections are separate, but they may traverse shared infrastructure.

Statistical independence is a precise mathematical property:

> **Two random variables A and B are independent if and only if:**
> **P(A=a and B=b) = P(A=a) × P(B=b) for all values a, b**

In other words, knowing the value of A tells you nothing about the value of B.

### Non-Independence Examples

1. **Shared clock**: Two PRNGs seeded from the same system clock produce correlated outputs
2. **Shared power supply**: Voltage ripple affects both channels similarly
3. **Electromagnetic coupling**: One circuit's switching noise affects its neighbor
4. **Thermal coupling**: Components sharing a heat sink respond to temperature together

Physical independence requires deliberate design; it does not happen automatically just because components are "separate."

---

## Why Independence Matters: The Math

### The Best Case: Perfect Independence

With independent channels, each bit flip in one channel is completely uncorrelated with bit flips in others. The XOR operation combines genuinely different random events, and bias reduction follows M-02's analysis:

```
Output bias = 8ε⁴
```

### The Worst Case: Perfect Correlation

If two channels produce identical outputs (perfect positive correlation):

```
P(A ⊕ A = 1) = 0
```

The XOR of identical values is always zero. We've gone from random to completely deterministic.

### The Moderate Case: Partial Correlation

Real systems exhibit partial correlation. Let ρ represent correlation coefficient (-1 to +1, where 0 means independent).

With correlation ρ between channels:

```
Effective bias reduction ≈ 8ε⁴ × (1 - ρ)⁴  [simplified model]
```

Even modest correlation degrades the benefit:
- ρ = 0.1 (10% correlation): ~66% of theoretical benefit
- ρ = 0.3 (30% correlation): ~24% of theoretical benefit
- ρ = 0.5 (50% correlation): ~6% of theoretical benefit

Correlation compounds through the XOR tree, eroding guarantees at each level.

---

## Sources of Correlation

### Shared Physical Phenomena

| Source | Mechanism | Mitigation |
|--------|-----------|------------|
| **Power supply** | Ripple/noise affects all circuits | Per-channel filtering, local regulation |
| **Ground plane** | Current return paths create coupling | Star grounding, ground planes |
| **Temperature** | Thermal drift affects all diodes | Thermal isolation, compensation |
| **Clock** | Sampling at same instant | Independent timing or true simultaneity |

### Electromagnetic Coupling

| Source | Mechanism | Mitigation |
|--------|-----------|------------|
| **Crosstalk** | Capacitive/inductive coupling between traces | Physical separation, shielding |
| **Radiated noise** | Switching circuits emit RF | Filtering, layout separation |
| **External interference** | WiFi, cellular, etc. | Shielding, differential signaling |

### Quantum Correlations

Interestingly, quantum mechanics does allow for correlations that exceed classical bounds (Bell inequality violations). However:

1. Avalanche diodes produce *classical* quantum noise; the quantum events are uncorrelated
2. Correlations would require deliberate entanglement, which the circuit doesn't create
3. Classical isolation is sufficient for our purposes

---

## Measuring Independence

### Cross-Correlation Function

For two bit sequences A and B, the cross-correlation at lag k:

```
R_AB(k) = E[A(n) × B(n+k)]
```

For independent sequences, R_AB(k) = 0 for all k ≠ 0.

### Mutual Information

Information-theoretic measure of dependence:

```
I(A;B) = H(A) + H(B) - H(A,B)
```

For independent sequences, I(A;B) = 0.

### Practical Test

The simplest test: XOR two channels and measure the bias. If the channels are independent with bias ε each, the XOR should have bias 2ε². If measured bias exceeds this, correlation exists.

```fsharp
// Pseudocode for independence check
let measureCorrelation (ch1: byte[]) (ch2: byte[]) =
    let xorResult = Array.map2 (^^^) ch1 ch2
    let expectedBias = 2.0 * epsilon1 * epsilon1
    let actualBias = measureBias xorResult
    if actualBias > 1.5 * expectedBias then
        Warning "Possible channel correlation detected"
```

---

## Hardware Design for Independence

Document H-03 details the electrical implementation, but the principles are:

### Separate Everything That Can Be Separated

1. **Individual diodes**: Each channel has its own avalanche diode, not a shared noise source
2. **Individual bias networks**: Separate resistors and current paths
3. **Individual decoupling**: Per-channel capacitors close to the source
4. **Individual ground returns**: Star grounding to avoid shared current paths

### Isolate What Can't Be Separated

1. **Power supply filtering**: Each channel filters the shared supply
2. **Physical separation**: Distance reduces capacitive/inductive coupling
3. **Shielding**: Ground planes between sensitive nodes

### Verify Independence

The design should include test points for measuring each channel independently, enabling correlation analysis during development and production testing.

---

## The Defense-in-Depth Perspective

Independence matters for security, not just statistics. Consider an attacker who can influence one channel (through EM injection, power manipulation, etc.):

| Channels | Effect of one-channel attack |
|----------|------------------------------|
| Single channel | Complete compromise |
| Two independent channels | Other channel masks attack |
| Four independent channels | Three channels mask attack |

With four independent channels, an attacker must simultaneously compromise all four to affect the output. The independence requirement creates *architectural* security, not just *algorithmic* security.

---

## Independence vs. Simultaneity

A subtle point: we want **statistical** independence, not necessarily **temporal** independence.

- **Statistical independence**: Channel A's value tells you nothing about Channel B's value
- **Temporal simultaneity**: Channels are sampled at the same moment

These are different properties:

- Channels can be statistically independent while sampled sequentially
- Channels can be sampled simultaneously while being statistically correlated

For QuantumCredential:
- We require **statistical independence** (the math requires it)
- We prefer **temporal simultaneity** (reduces time-varying external influences)
- But simultaneity without independence provides no benefit

---

## Testing Regime

### Development Testing

During hardware development, extensively test for:
1. Per-channel bias (should be within spec)
2. Cross-correlation at lag 0 (should be near zero)
3. Cross-correlation at various lags (should remain near zero)
4. Response to external EM (all channels should respond similarly or not at all)

### Production Testing

Each unit should undergo:
1. Basic bias check per channel
2. XOR bias check (should match 8ε⁴ prediction)
3. Power supply variation test (bias shouldn't correlate with supply)

### Runtime Monitoring

The demo application (D-03) can continuously monitor:
1. Per-channel epsilon values
2. XOR-predicted vs. actual bias
3. Anomaly detection for sudden correlation

---

## Summary

Independence is the **mathematical contract** between hardware and software:

| Responsibility | Component |
|---------------|-----------|
| **Provide independence** | Hardware design (H-03) |
| **Assume independence** | Mathematical analysis (M-01, M-02) |
| **Verify independence** | Testing and monitoring (E-02) |
| **Benefit from independence** | Four-channel XOR tree |

Without independence, the 8ε⁴ formula does not hold, and the security guarantees collapse. This is why hardware/software co-design matters: the mathematics only works if the physics is designed to support it.

---

## Related Documents

- **M-01-XOR-Entropy-Theorem**: The theorem that requires independence
- **M-02-Bias-Reduction-Analysis**: The formula that assumes independence
- **H-03-Channel-Independence**: Hardware implementation of independence
- **E-02-Epsilon-Evaluation**: Measuring per-channel bias (enables correlation detection)
- **V-03-EM-Attack-Resistance**: Why independence provides attack resistance
