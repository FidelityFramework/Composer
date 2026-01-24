# V-03: Electromagnetic Attack Resistance

## Overview

This document describes how the four-channel XOR architecture provides defense against electromagnetic (EM) attacks that could bias entropy output. Single-channel designs are vulnerable to coherent interference; the four-channel design requires simultaneous compromise of all independent channels.

**Key Principle**: The mathematical requirement for independence (M-03) creates architectural security against active EM attacks.

---

## The Single-Channel Vulnerability

### Avalanche Generator Attack Vector

Single-channel avalanche generators have a fundamental weakness: the sensitive diode output operates at microampere levels. External electromagnetic fields can superimpose currents on this signal.

Attack mechanism:
1. Attacker positions an EM source near the device
2. Stray currents induced in the diode output node
3. Repetitive or fixed waveform superimposed on noise signal
4. 1-bit discriminator becomes biased
5. Output correlates with external EM field
6. Generator enters "stuck" state with predictable output

### Modular Noise Generator Attack Vector

Modular noise generators (alternative TRNG design) face a similar threat through a different mechanism:

1. Noise amplifier indiscriminately amplifies all sources
2. EM interference becomes part of amplified signal
3. Attacker times EM pulses to sampling period
4. Comparator voltage reference shifts under EM influence
5. Output bits correlate with external field timing

Both attack modes can force a single-channel generator into predictable output patterns.

---

## Four-Channel Defense Architecture

### Mathematical Foundation

The four-channel XOR architecture defeats both attack modes through redundancy:

```
Channel 0 (compromised?) ──┬── XOR ──┬── XOR ──→ Output
Channel 1 (compromised?) ──┘         │
Channel 2 (independent)   ──┬── XOR ──┘
Channel 3 (independent)   ──┘
```

For the output to be predictable, an attacker must:
1. Identify all four channel locations
2. Generate coherent EM fields at all four
3. Maintain phase coherence across all channels
4. Sustain the attack without detection

### Independence as Defense

The channels operate from separate components:
- Separate avalanche diodes
- Separate bias networks
- Separate decoupling capacitors
- Physical separation on PCB

Influencing one channel provides no leverage over others. The XOR combination means that even complete compromise of three channels leaves the fourth channel's randomness propagating to output.

---

## Attack Scenarios and Defenses

### Scenario 1: Local EM Source

**Attack**: Small EM emitter placed near device

**Effect on single-channel**: High probability of bias

**Effect on four-channel**:
- EM field affects all channels similarly? Unlikely due to different orientations and positions
- Even if all four biased identically, XOR of identical biases still reduces output bias significantly
- Physical separation means field strength varies across channels

**Defense rating**: Strong

### Scenario 2: Swept Frequency Attack

**Attack**: Attacker sweeps frequencies to find resonance

**Effect on single-channel**: If resonance found, high bias possible

**Effect on four-channel**:
- Different trace lengths create different resonances
- No single frequency affects all channels equally
- Broadband attack reduces power at any one frequency

**Defense rating**: Very strong

### Scenario 3: Precisely Targeted Attack

**Attack**: Four separate EM sources, one per channel, precisely positioned

**Effect on single-channel**: N/A (single source sufficient)

**Effect on four-channel**:
- Requires physical access to position four sources
- Requires knowledge of board layout
- Requires phase coherence between sources
- Practically infeasible without physical device access

**Defense rating**: Extremely strong (requires physical access)

---

## Detection Mechanisms

### Per-Channel Bias Monitoring

Epsilon evaluation (E-02) provides attack detection:

```fsharp
/// Detect coherent bias across channels
let detectCoherentAttack (epsilons: float[]) =
    let allPositive = epsilons |> Array.forall (fun e -> e > 0.02)
    let allNegative = epsilons |> Array.forall (fun e -> e < -0.02)

    if allPositive || allNegative then
        // All channels biased in same direction
        // Highly unlikely without external influence
        Alert "Coherent bias detected: possible EM attack"
    else
        Ok
```

### Sudden Change Detection

```fsharp
/// Detect sudden epsilon changes
let detectSuddenChange (current: float[]) (baseline: float[]) =
    let deviations =
        Array.map2 (fun c b -> abs (c - b)) current baseline

    let maxDeviation = Array.max deviations
    let avgDeviation = Array.average deviations

    if maxDeviation > 0.1 then
        Alert (sprintf "Single channel deviation: %.2f" maxDeviation)
    elif avgDeviation > 0.05 then
        Alert (sprintf "Multiple channel deviation: avg %.2f" avgDeviation)
    else
        Ok
```

### Correlation Detection

```fsharp
/// Detect induced correlation between channels
let detectCorrelation (ch1: byte[]) (ch2: byte[]) =
    let correlation = computeCorrelation ch1 ch2

    if abs correlation > 0.1 then
        Alert "Unexpected channel correlation"
    else
        Ok
```

---

## Physical Mitigations

While the four-channel architecture provides mathematical protection, physical mitigations add defense in depth:

### Shielding

| Layer | Implementation | Purpose |
|-------|---------------|---------|
| Board level | Ground plane under sensitive traces | Block PCB coupling |
| Component level | Metal can over diodes | Direct EM blocking |
| Enclosure level | Conductive enclosure | External field attenuation |

### Filtering

| Filter Type | Location | Purpose |
|-------------|----------|---------|
| Low-pass RC | ADC input | Remove RF pickup |
| Decoupling | Power supply | Block conducted interference |
| Ferrite beads | Signal traces | Suppress high-frequency noise |

### Layout

| Technique | Implementation | Purpose |
|-----------|---------------|---------|
| Star ground | Single ground point | Prevent ground loops |
| Separation | >10mm between channels | Reduce mutual coupling |
| Orientation | Channels at 90° | Different field sensitivity |

---

## Comparison: Single vs Four Channel

| Attack Vector | Single Channel | Four Channel |
|---------------|---------------|--------------|
| Local EM source | Vulnerable | Resistant |
| Swept frequency | Vulnerable | Highly resistant |
| Precision targeting | Vulnerable | Requires physical access |
| Detection capability | Limited | Comprehensive via ε monitoring |
| Graceful degradation | None | Continues with 3 channels |

---

## Relationship to Independence

The EM attack resistance emerges from the same property that enables bias reduction: channel independence.

M-03 (Independence Requirements) specifies:
- Separate diodes
- Isolated bias networks
- Independent power filtering

These requirements, driven by mathematical necessity for bias reduction, simultaneously create physical separation that defeats EM attacks.

This is hardware/software co-design in action: a mathematical requirement (independence for XOR) creates a security property (EM attack resistance) as a byproduct.

---

## Connection to Other Documents

- **M-03-Independence-Requirements**: The independence that enables EM resistance
- **H-03-Channel-Independence**: Hardware implementation of isolation
- **E-02-Epsilon-Evaluation**: Detection mechanism for attacks
- **V-04-Fault-Tolerance**: Behavior when channels fail (attack or otherwise)

---

## Summary

The four-channel XOR architecture provides defense against electromagnetic attacks through:

| Property | Mechanism | Result |
|----------|-----------|--------|
| Physical separation | Independent components | No single field affects all |
| Mathematical redundancy | XOR combination | One random channel masks biased ones |
| Continuous monitoring | Epsilon evaluation | Attack detection capability |
| Graceful degradation | Multi-channel design | Continued operation under partial compromise |

Single-channel entropy generators are fundamentally vulnerable to EM attack. The four-channel architecture, designed for mathematical bias reduction, provides attack resistance as an inherent property of its architecture.
