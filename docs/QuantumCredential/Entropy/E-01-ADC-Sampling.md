# E-01: ADC Sampling

## Overview

This document describes how analog avalanche noise is converted to digital samples via the MCP3004 ADC on the YoshiPi platform. The ADC is the bridge between quantum physics and digital computation; its characteristics directly affect entropy quality.

**Key Principle**: The ADC does not create entropy; it captures entropy that exists in the analog signal. ADC configuration determines how faithfully that capture occurs.

---

## Hardware Configuration

### MCP3004 Specifications

| Parameter | Value | Significance |
|-----------|-------|--------------|
| Resolution | 10 bits | 1024 discrete levels |
| Channels | 4 | Matches four-channel architecture |
| Interface | SPI | Standard Linux IIO support |
| Sample rate | Up to 200 ksps | Limited by SPI clock |
| Input range | 0 to Vref | Typically 3.3V |
| INL | ±1 LSB | Linearity affects bias |

### YoshiPi Integration

The YoshiPi carrier board connects the MCP3004 to the Raspberry Pi Zero 2 W:

```
RPi GPIO → SPI0 → MCP3004 → Avalanche Circuits
   |                           |
   +-- CLK (GPIO 11)           +-- CH0: Diode 1
   +-- MOSI (GPIO 10)          +-- CH1: Diode 2
   +-- MISO (GPIO 9)           +-- CH2: Diode 3
   +-- CS0 (GPIO 8)            +-- CH3: Diode 4
```

---

## Linux IIO Subsystem

### Why IIO

The Industrial I/O (IIO) subsystem provides:
- Standardized ADC access across hardware
- Buffered and triggered sampling
- Sysfs interface for configuration
- Kernel-space efficiency

For Phase 1 (Linux-hosted development), IIO provides the abstraction layer. Phase 2 (bare-metal STM32) will use direct register access via Platform.Bindings.

### IIO Device Structure

```
/sys/bus/iio/devices/iio:device0/
├── in_voltage0_raw      # Channel 0 raw value
├── in_voltage1_raw      # Channel 1 raw value
├── in_voltage2_raw      # Channel 2 raw value
├── in_voltage3_raw      # Channel 3 raw value
├── in_voltage_scale     # Conversion factor to millivolts
├── sampling_frequency   # Current sample rate
└── buffer/
    ├── enable           # Buffer enable
    ├── length           # Buffer size
    └── watermark        # Trigger threshold
```

### Single-Shot Sampling

For development and demonstration:

```fsharp
/// Read a single sample from specified channel
let readChannel (ch: int) : uint16 =
    let path = sprintf "/sys/bus/iio/devices/iio:device0/in_voltage%d_raw" ch
    let content = File.ReadAllText path
    uint16 (Int32.Parse content.Trim())
```

This approach:
- Works immediately without kernel module development
- Incurs syscall overhead per sample
- Sufficient for demonstration (hundreds of samples/second)

### Buffered Sampling

For production throughput:

```fsharp
/// Configure buffered sampling for all channels
let configureBuffer (samplesPerChannel: int) =
    // Enable all channels in scan
    File.WriteAllText "/sys/.../scan_elements/in_voltage0_en" "1"
    File.WriteAllText "/sys/.../scan_elements/in_voltage1_en" "1"
    File.WriteAllText "/sys/.../scan_elements/in_voltage2_en" "1"
    File.WriteAllText "/sys/.../scan_elements/in_voltage3_en" "1"

    // Set buffer length
    let totalSamples = samplesPerChannel * 4
    File.WriteAllText "/sys/.../buffer/length" (string totalSamples)

    // Enable buffer
    File.WriteAllText "/sys/.../buffer/enable" "1"

/// Read buffer contents
let readBuffer () : uint16[] =
    use fs = new FileStream("/dev/iio:device0", FileMode.Open)
    // Read interleaved channel data
    // Format: CH0, CH1, CH2, CH3, CH0, CH1, ...
```

Buffered sampling:
- Amortizes syscall overhead
- Enables DMA transfer (hardware dependent)
- Provides consistent inter-sample timing

---

## Sample Timing Considerations

### Independence Through Timing

The XOR combination requires statistically independent samples. Timing affects independence:

**Too fast**: If the avalanche process has not fully randomized between samples, successive samples may correlate.

**Too slow**: Wastes throughput without improving quality.

The avalanche time constant is typically microseconds. At 10 kHz sampling (100 μs period), each sample captures an independent avalanche event.

### Synchronization Options

**Sequential sampling**: Read CH0, then CH1, then CH2, then CH3.
- Simple implementation
- ~40 μs between first and last channel
- Adequate for independence

**Simultaneous sampling**: Hardware-triggered sample-and-hold on all channels.
- Requires S&H circuitry
- Captures same time instant
- Reduces time-varying external influence

For the MCP3004, sampling is inherently sequential (SAR architecture). The inter-channel delay is negligible compared to avalanche time constants.

---

## Value Extraction

### Raw ADC to Entropy Byte

The 10-bit ADC produces values 0-1023. Entropy extraction uses the least significant bits:

```fsharp
/// Extract 8 entropy bits from 10-bit sample
let extractByte (sample: uint16) : byte =
    byte (sample &&& 0xFFus)
```

Why the lower 8 bits:
- MSBs correlate with DC bias voltage
- LSBs contain higher-frequency noise
- LSBs have maximum entropy per bit

Alternative: XOR upper and lower bits:

```fsharp
/// Alternative: XOR upper 2 bits into lower 8
let extractByteXor (sample: uint16) : byte =
    let lower = byte (sample &&& 0xFFus)
    let upper = byte ((sample >>> 8) &&& 0x03us)
    lower ^^^ (upper <<< 6)
```

This slightly increases entropy extraction but complicates analysis. The simple LSB extraction is preferred for the demo.

---

## Platform.Bindings Abstraction

### Interface Definition

```fsharp
module Platform.Bindings.ADC =
    /// Read raw value from ADC channel (0-3)
    val readChannel : int -> uint16

    /// Read all four channels simultaneously (where supported)
    val readAllChannels : unit -> uint16 * uint16 * uint16 * uint16

    /// Configure sample rate (Hz)
    val setSampleRate : int -> unit

    /// Get current sample rate
    val getSampleRate : unit -> int
```

### Platform Implementations

**Linux (YoshiPi)**:
```fsharp
let readChannel ch =
    let path = sprintf "/sys/bus/iio/devices/iio:device0/in_voltage%d_raw" ch
    uint16 (Int32.Parse (File.ReadAllText path).Trim())
```

**STM32L5 (Bare Metal)**:
```fsharp
let readChannel ch =
    // Configure ADC channel
    ADC1.SQR1 <- (uint32 ch) <<< 6
    // Start conversion
    ADC1.CR <- ADC1.CR ||| ADC_CR_ADSTART
    // Wait for completion
    while (ADC1.ISR &&& ADC_ISR_EOC) = 0u do ()
    // Read result
    uint16 ADC1.DR
```

The same F# code compiles to either implementation based on target platform.

---

## Error Handling

### ADC Failures

| Failure Mode | Detection | Response |
|--------------|-----------|----------|
| Stuck value | Same reading repeated | Flag channel as failed |
| Rail voltage | Always 0 or 1023 | Flag channel as failed |
| Erratic readings | Variance exceeds threshold | Check connections |
| Missing device | IIO device not found | Fall back to simulation |

### Graceful Degradation

With four channels, single-channel failures are tolerable:

```fsharp
let readWithFallback () =
    let samples = [| 0..3 |] |> Array.map (fun ch ->
        try
            Some (readChannel ch)
        with
        | :? IOException -> None
    )

    let validSamples = samples |> Array.choose id

    match validSamples.Length with
    | 4 -> // Full quality
        xorAll validSamples
    | 3 -> // Degraded but acceptable
        logWarning "One channel failed"
        xorAll validSamples
    | n when n < 3 ->
        failwith "Insufficient entropy sources"
```

---

## Performance Baseline

### Single-Shot Performance (Linux)

| Operation | Time | Notes |
|-----------|------|-------|
| Open sysfs file | ~50 μs | Kernel overhead |
| Read value | ~20 μs | File read |
| Parse integer | ~5 μs | String to int |
| Total per channel | ~75 μs | |
| Four channels | ~300 μs | Sequential |
| Samples per second | ~3,300 | Single-shot ceiling |

### Buffered Performance (Linux)

| Operation | Time | Notes |
|-----------|------|-------|
| Fill 256-sample buffer | ~25 ms | DMA transfer |
| Read buffer | ~2 ms | Bulk read |
| Process samples | ~1 ms | XOR computation |
| Effective rate | ~9,000 samples/sec | Per-channel |

### Native Performance Target

With Firefly native compilation and direct SPI:

| Operation | Time | Notes |
|-----------|------|-------|
| SPI transaction | ~10 μs | Direct GPIO |
| Four channels | ~40 μs | Pipelined |
| XOR and store | ~1 μs | In-register |
| Target rate | ~24,000 samples/sec | Per-channel |

---

## Connection to Other Documents

- **H-02-Avalanche-Circuit**: Signal source for ADC input
- **E-02-Epsilon-Evaluation**: Uses ADC values for bias calculation
- **PH1-02-Linux-Hardware-Bindings**: Full Linux binding implementation
- **PH2-03-HAL-Bindings**: STM32 direct ADC access

---

## Summary

The ADC captures analog quantum noise and converts it to digital samples. Key design decisions:

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Resolution | 10-bit (use 8 LSBs) | LSBs have maximum entropy |
| Interface | SPI via IIO | Standard Linux support |
| Sampling | Sequential | Adequate independence |
| Abstraction | Platform.Bindings | Same code, multiple targets |

The ADC is the physical boundary between quantum randomness and digital processing. Its faithful capture of avalanche noise is the foundation for all downstream entropy guarantees.
