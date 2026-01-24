# M-02: Bias Reduction Analysis

## Overview

Real entropy sources are never perfect. An avalanche diode might produce 52% ones and 48% zeros instead of the ideal 50/50 split. This document derives the mathematical relationship between input bias and output bias when combining sources via XOR, culminating in the key result:

> **Four-channel XOR tree: ε → 8ε⁴**

A 5% per-channel bias becomes 0.005% output bias: a 1000× improvement through mathematics alone.

---

## For Software Engineers: Thinking About Bias

Software developers rarely think about bit-level bias. When you call `Random.Next()`, you assume (correctly, for most purposes) that the bits are balanced. But this assumption rests on layers of algorithmic processing that "whiten" the output.

Hardware entropy sources don't have this luxury. The raw output of an avalanche diode reflects the physics of the device:
- Manufacturing variations
- Temperature effects
- Power supply ripple
- Component aging

These physical factors create *bias*, a systematic deviation from perfect randomness. Understanding bias, and how to reduce it mathematically, is essential for hardware/software co-design.

---

## Defining Bias

Let ε (epsilon) represent deviation from perfect balance:

- A source with bias ε produces:
  - **Ones** with probability 0.5 + ε
  - **Zeros** with probability 0.5 - ε

For example:
- ε = 0 means perfect balance (50% ones, 50% zeros)
- ε = 0.05 means 5% bias (55% ones, 45% zeros)
- ε = 0.10 means 10% bias (60% ones, 40% zeros)

The goal is to minimize ε in the output.

---

## Two-Channel XOR: The Basic Case

When two independent sources A and B, each with bias ε, are XOR'd:

### Derivation

```
P(A ⊕ B = 1) = P(A=1)×P(B=0) + P(A=0)×P(B=1)
             = (0.5 + ε)(0.5 - ε) + (0.5 - ε)(0.5 + ε)
             = 2 × (0.25 - ε²)
             = 0.5 - 2ε²
```

### Output Bias

The output has probability 0.5 - 2ε² of being 1, so:

```
Output bias = |0.5 - (0.5 - 2ε²)| = 2ε²
```

### Example

Starting with ε = 0.05 (5% bias):
```
Output bias = 2 × (0.05)² = 2 × 0.0025 = 0.005 = 0.5%
```

Two channels reduce 5% bias to 0.5%, a 10× improvement.

---

## Four-Channel XOR Tree: The Full Analysis

The tree structure combines channels in two levels:

```
Level 1: CH0 ⊕ CH1 → intermediate₁ (bias = 2ε²)
         CH2 ⊕ CH3 → intermediate₂ (bias = 2ε²)

Level 2: intermediate₁ ⊕ intermediate₂ → output
```

### Level 2 Derivation

The two intermediates each have bias 2ε². Applying the two-channel formula again:

```
Output bias = 2 × (2ε²)²
            = 2 × 4ε⁴
            = 8ε⁴
```

### Example

Starting with ε = 0.05 (5% bias per channel):
```
Output bias = 8 × (0.05)⁴
            = 8 × 0.00000625
            = 0.00005
            = 0.005%
```

Four channels reduce 5% bias to 0.005%, a **1000× improvement**.

---

## The Exponential Advantage

The key insight is that ε is raised to the fourth power. This creates enormous sensitivity to per-channel bias:

| Per-channel bias (ε) | Output bias (8ε⁴) | Improvement factor |
|---------------------|-------------------|-------------------|
| 10% | 0.08% | 125× |
| 5% | 0.005% | 1,000× |
| 2% | 0.00013% | 15,625× |
| 1% | 0.000008% | 125,000× |

### Implication for Hardware Design

Small improvements in per-channel bias yield enormous improvements in output quality:
- Reducing ε from 5% to 2% is a 2.5× improvement in input
- But the output improves by 15,625 / 1,000 = 15.6×

This is why hardware tuning matters (see H-04). Each percentage point of per-channel improvement translates to orders of magnitude in output quality.

---

## Why Not More Channels?

If four channels give ε⁴, wouldn't eight channels give even better results?

### Diminishing Returns

The 8ε⁴ formula already produces extremely low bias for reasonable ε values. With ε = 2%:
- Four channels: 0.00013%
- Eight channels: ~0.00000002%

The four-channel output is already below detection thresholds and far exceeds cryptographic requirements.

### Practical Constraints

Each additional channel requires:
- Independent avalanche circuit (cost, board space)
- Independent ADC channel (hardware complexity)
- Verification of independence (see M-03)

Four channels hit the sweet spot: dramatic bias reduction without excessive hardware complexity.

---

## The Independence Assumption

**Critical**: All derivations assume statistical independence between channels.

If channels are correlated (producing similar values at similar times), the bias reduction is compromised. Two perfectly correlated channels produce:

```
P(A ⊕ A = 1) = 0  (always zero!)
```

This is why M-03 (Independence Requirements) and H-03 (Channel Independence) exist: to ensure the mathematical preconditions are met in hardware.

---

## Visualization

```
Per-channel         Two-channel         Four-channel
    ε                  2ε²                  8ε⁴

  5.00% ─────────────> 0.50% ─────────────> 0.005%
    │                    │                    │
    │    10× reduction   │   100× reduction   │
    └────────────────────┴────────────────────┘
              Total: 1000× reduction
```

---

## Connection to Epsilon Evaluation

Document E-02 (Epsilon Evaluation) describes how to *measure* per-channel bias in real-time. This measurement:

1. **Validates hardware tuning** - confirms each channel meets target ε
2. **Enables demonstration** - shows bias reduction visually
3. **Detects degradation** - identifies channel failures

The mathematics here tells us *what to measure*; E-02 tells us *how to measure it*.

---

## Beyond Bias: Higher-Order Statistics

Bias (first-order statistics) is the primary concern, but real entropy sources can exhibit:

- **Serial correlation**: Bit n predicts bit n+1
- **Runs bias**: Too many or too few consecutive identical bits
- **Periodicity**: Patterns that repeat

The XOR combination helps with these issues too:
- Serial correlation in one channel is masked by independence of others
- Runs are broken up by uncorrelated bit flips from other channels

For formal validation, see V-01 (NIST SP 800-22) which tests for these higher-order effects.

---

## Summary

The four-channel XOR tree provides **mathematical amplification** of entropy quality:

| Input | Output | Formula |
|-------|--------|---------|
| ε per channel | 8ε⁴ combined | Proven, not empirical |
| 5% bias | 0.005% bias | 1000× improvement |
| Hardware imperfection | Cryptographic quality | Mathematics provides guarantee |

This is the foundation of QuantumCredential's approach: use mathematics to transform imperfect physical sources into provably high-quality entropy.

---

## Related Documents

- **M-01-XOR-Entropy-Theorem**: Why XOR works for entropy combination
- **M-03-Independence-Requirements**: The independence assumption examined
- **H-04-Per-Channel-Tuning**: Hardware techniques to minimize ε
- **E-02-Epsilon-Evaluation**: Real-time bias measurement
