# E-02: Epsilon Evaluation

## Overview

This document describes the **core capability** of the QuantumCredential demo: real-time measurement and display of per-channel bias (epsilon) before XOR combination. This measurement:

1. **Proves cryptographic validity** - demonstrates that bias reduction works as predicted
2. **Validates hardware design** - confirms each channel meets specifications
3. **Enables visual demonstration** - shows the mathematics in action
4. **Detects degradation** - identifies failing channels before they compromise output

Epsilon evaluation is where the abstraction ceiling meets the abstraction floor: the point where software engineers can observe what normally remains hidden below their APIs.

---

## For Software Engineers: What You've Never Seen

When you call `RandomNumberGenerator.GetBytes()`, what happens?

1. Some entropy source produces values
2. Some mixing algorithm processes them
3. Some buffer fills with bytes
4. You use the bytes

At no point do you see:
- The raw entropy quality
- The per-source bias
- The mixing effectiveness
- Whether the source is degrading

This is the abstraction ceiling: the highest point you can observe. Everything below is trust.

QuantumCredential's epsilon evaluation **lowers the floor**. You see:
- Raw ADC values from each channel
- Computed bias per channel
- Predicted vs. actual combined bias
- Real-time trends and anomalies

This is not debugging information; it is proof of cryptographic quality.

---

## Epsilon: Definition and Measurement

### Mathematical Definition

Bias ε (epsilon) represents deviation from perfect balance:

```
P(bit = 1) = 0.5 + ε
P(bit = 0) = 0.5 - ε
```

For ideal randomness, ε = 0. For real sources, ε is small but non-zero.

### Measurement Approach

Given N sample bits, count the ones:

```
ε_measured = (count_of_ones / N) - 0.5
```

For example:
- 1000 bits with 530 ones: ε = 530/1000 - 0.5 = 0.03 (3% bias toward ones)
- 1000 bits with 485 ones: ε = 485/1000 - 0.5 = -0.015 (1.5% bias toward zeros)

### Statistical Confidence

The measured ε has uncertainty due to finite sample size:

```
Standard error = sqrt(0.25 / N)
```

| Sample size | Standard error | 95% confidence interval |
|-------------|----------------|-------------------------|
| 100 bits | 0.05 | ±10% |
| 1,000 bits | 0.016 | ±3.2% |
| 10,000 bits | 0.005 | ±1% |
| 100,000 bits | 0.0016 | ±0.32% |

For meaningful epsilon measurement, we need at least 10,000 bits (~1,250 bytes) per channel.

---

## Multi-Resolution Evaluation

The demo displays epsilon at multiple time scales:

### Instantaneous (100ms window)
- ~400 samples per channel at 4kHz
- ~3,200 bits (8 bits × 400)
- High variance, shows rapid fluctuations
- Useful for detecting transient interference

### Short-term (1 second window)
- ~4,000 samples per channel
- ~32,000 bits
- Standard error ~0.3%
- Primary display value

### Long-term (10 second window)
- ~40,000 samples per channel
- ~320,000 bits
- Standard error ~0.09%
- Confirms stable channel behavior

### Session cumulative
- All samples since start
- Lowest variance
- Demonstrates long-term stability

---

## Per-Channel Display

Each of the four channels displays:

```
┌──────────────────────────────────────────┐
│ Channel 0                                │
│ ────────────────────────────────────────│
│ Current ε:    +0.023 (2.3% bias)        │
│ 1-sec avg:    +0.019                     │
│ 10-sec avg:   +0.021                     │
│ Session avg:  +0.020                     │
│                                          │
│ [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓░░░░] 52.0% ones      │
│                                          │
│ Status: ✓ Within spec (< 5%)            │
└──────────────────────────────────────────┘
```

Visual elements:
- **Bar graph**: Shows proportion of ones (50% = center)
- **Trend indicator**: Arrow showing if bias is increasing/decreasing
- **Status**: Green check if within spec, yellow warning if marginal, red if failing

---

## Combined Output Display

Below the four channel displays, show the combined output:

```
┌──────────────────────────────────────────┐
│ Combined Output (4-Channel XOR)          │
│ ────────────────────────────────────────│
│                                          │
│ Predicted ε:  0.00011 (from 8ε⁴)        │
│ Measured ε:   0.00008                    │
│ Match: ✓ Within statistical bounds       │
│                                          │
│ [▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓] 50.01% ones     │
│                                          │
│ Bias reduction: 182× (vs best channel)  │
│                                          │
└──────────────────────────────────────────┘
```

The key demonstration:
- Show individual channel epsilons (typically 1-5%)
- Show predicted combined epsilon (8ε⁴ calculation)
- Show actual measured combined epsilon
- Confirm they match within statistical bounds

This **proves** the mathematics works with real hardware.

---

## Implementation Architecture

### F# Data Flow

```fsharp
/// Per-channel statistics accumulator
type ChannelStats = {
    Channel: int
    WindowSize: int
    BitCounts: RingBuffer<int>  // Ones count per sample window
    TotalOnes: int64
    TotalBits: int64
}

/// Compute epsilon from channel stats
let computeEpsilon (stats: ChannelStats) : float =
    let proportion = float stats.TotalOnes / float stats.TotalBits
    proportion - 0.5

/// Predict combined epsilon from channel epsilons
let predictCombinedEpsilon (epsilons: float array) : float =
    // For 4-channel tree: 8 * ε⁴
    // Using geometric mean of individual epsilons
    let avgEpsilon = epsilons |> Array.averageBy abs
    8.0 * pown avgEpsilon 4

/// Sample one byte from each channel, update stats, compute output
let sampleAndEvaluate (adc: ADC) (stats: ChannelStats array) =
    // Read all four channels
    let samples = [| 0..3 |] |> Array.map adc.readChannel

    // Update per-channel statistics
    for i in 0..3 do
        let byte = byte (samples.[i] &&& 0xFFus)
        let ones = popcount byte  // Count ones in byte
        stats.[i] <- updateStats stats.[i] ones

    // Compute XOR combination
    let combined =
        samples
        |> Array.map (fun s -> byte (s &&& 0xFFus))
        |> Array.reduce (^^^)

    // Return combined byte and updated stats
    combined, stats
```

### MLIR Representation

The epsilon computation compiles to efficient native code:

```mlir
// Popcount for ones counting
%ones = math.ctpop %byte : i8

// Accumulate in 64-bit counter (no overflow for hours of operation)
%total_ones_new = arith.addi %total_ones, %ones_ext : i64
%total_bits_new = arith.addi %total_bits, %c8 : i64

// Epsilon = (ones/bits) - 0.5
%proportion = arith.divf %ones_f, %bits_f : f64
%epsilon = arith.subf %proportion, %half : f64
```

---

## Anomaly Detection

### Channel Failure Detection

If a channel's epsilon suddenly changes, it may indicate:
- Component failure (diode degradation)
- Connection problem (intermittent contact)
- External interference (EM attack)

Detection algorithm:
```fsharp
let detectAnomaly (current: float) (baseline: float) (threshold: float) =
    let deviation = abs (current - baseline)
    if deviation > threshold then
        Anomaly (channel, deviation)
    else
        Normal
```

### Correlation Detection

If channels become correlated, the combined epsilon will exceed predictions:

```fsharp
let detectCorrelation (predicted: float) (actual: float) =
    // Allow 3x statistical variance
    let tolerance = 3.0 * statisticalError
    if actual > predicted + tolerance then
        Warning "Possible channel correlation"
    else
        Ok
```

### Attack Detection

Sustained bias in one direction across all channels suggests external influence:

```fsharp
let detectAttack (epsilons: float array) =
    let allPositive = epsilons |> Array.forall (fun e -> e > 0.01)
    let allNegative = epsilons |> Array.forall (fun e -> e < -0.01)
    if allPositive || allNegative then
        Alert "Coherent bias detected - possible EM attack"
    else
        Ok
```

---

## Display Update Strategy

### Refresh Rates

| Element | Refresh rate | Rationale |
|---------|--------------|-----------|
| Current ε | 10 Hz | Shows live behavior |
| 1-sec average | 1 Hz | Stable reading |
| 10-sec average | 0.1 Hz | Trend confirmation |
| Session average | 0.1 Hz | Cumulative stability |
| Status indicators | 1 Hz | Avoid flicker |

### Visual Smoothing

Apply exponential moving average for display to reduce visual noise:

```fsharp
let smoothedEpsilon = 0.7 * previousDisplayed + 0.3 * currentMeasured
```

This smooths the display without hiding real anomalies.

---

## Calibration Mode

Before normal operation, a calibration phase:

1. **Sample each channel independently** (10 seconds each)
2. **Compute baseline epsilon per channel**
3. **Verify independence** (XOR pairs and check correlation)
4. **Compute expected combined epsilon**
5. **Verify combined output matches prediction**
6. **Display calibration results**

If calibration fails, indicate which channel or relationship is problematic.

---

## The Demonstration Value

For software engineers watching the demo:

1. **See what's normally hidden**: Per-channel bias is visible, not abstracted away
2. **Watch mathematics work**: Predicted vs. actual combined epsilon
3. **Understand hardware imperfection**: Channels aren't perfect (2-5% bias)
4. **See imperfection overcome**: Combined output is essentially perfect (<0.01% bias)
5. **Appreciate co-design**: Hardware provides independence, software proves it works

This is the "abstraction floor": the lowest level where you can verify system behavior before trusting the output.

---

## Connection to Other Documents

- **M-02-Bias-Reduction-Analysis**: The mathematics being demonstrated
- **H-04-Per-Channel-Tuning**: Hardware optimization to minimize per-channel epsilon
- **D-03-Entropy-Visualization**: UI design for epsilon display
- **V-02-TRNG-Certification**: Using epsilon data for validation

---

## Summary

Epsilon evaluation transforms QuantumCredential from "an entropy generator" to "a provably valid entropy generator." By measuring and displaying:

| What | Why |
|------|-----|
| Per-channel epsilon | Validates hardware design |
| Predicted combined epsilon | Shows mathematical prediction |
| Actual combined epsilon | Proves prediction is correct |
| Deviation from prediction | Detects correlation or attack |

This is not merely monitoring; it is **proof of cryptographic validity** in real-time.
