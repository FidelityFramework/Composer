# PH2-00: Embedded Platform Strategy

## Overview

Phase 2 targets bare-metal embedded platforms for production QuantumCredential devices. The primary target is the **Renesas RA6M5** evaluation kit, with STM32L5 as a secondary option.

## Narrative Guardrails

The documentation should keep three stories visible at the same time:

- YoshiPi proves that Composer can ship a normal, garden-variety Linux app with a screen and ADC-backed hardware I/O.
- RA6M5 proves that Composer can also drive a production embedded target with a stronger root of trust.
- The headline architecture is a **reactive unikernel** built from Clef-native concurrency lowered to a single core; an RTOS (FreeRTOS or Zephyr) stays an *optional bridge*, not the architecture, so the project does not drift into looking like a specialty DSL demo only.

That balance matters because the platform story is part of the market story.

---

## Platform Priority

| Priority | Platform | Status | Rationale |
|----------|----------|--------|-----------|
| **Primary** | Renesas RA6M5 | Active development | Complete feature set, TrustZone, Secure Crypto Engine |
| Secondary | STM32L5 | On hold | Good platform, but RA6M5 better suited for security applications |

---

## Why Renesas RA6M5

The [EK-RA6M5 evaluation kit](https://www.renesas.com/en/products/microcontrollers-microprocessors/ra-cortex-m-mcus/ek-ra6m5-evaluation-kit-ra6m5-mcu-group) provides everything needed for QuantumCredential in a single board:

### Security Features

| Feature | Benefit |
|---------|---------|
| **Hardware Unique Key (HUK)** | Factory-programmed 256-bit key unique to each MCU; enables device-bound credentials |
| **Unique ID** | Immutable 16-byte device identifier; cryptographically verifiable identity |
| **Arm TrustZone** | Hardware isolation between secure and non-secure code |
| **Secure Crypto Engine (SCE9)** | Hardware acceleration; HUK accessible only via private bus |
| **Tamper detection** | Physical security monitoring |
| **Power analysis resistance** | Side-channel attack mitigation |

### The Sovereignty Advantage

The **HUK fundamentally changes the trust model**. Traditional credential systems require trust in external authorities: certificate authorities, identity providers, key escrow services. The RA6M5's factory-provisioned HUK enables a different architecture:

| Traditional Model | HUK-Enabled Model |
|-------------------|-------------------|
| Identity issued by authority | Identity anchored in hardware |
| Keys stored in software/HSM | Keys bound to specific silicon |
| Cloneable with sufficient access | Physically unclonable |
| Trust delegated upward | Trust rooted in device |

With the HUK, the device itself becomes the anchor point for cryptographic identity. Credentials generated and wrapped on a specific RA6M5 are intrinsically bound to that physical device. No external service provisioned this identity; no external service can revoke or clone it. The owner of the device holds the only instance of that cryptographic identity in existence.

This is why the RA6M5 is elevated as the preferred platform: the STM32L5 can implement the same algorithms, but it lacks the factory-provisioned hardware identity that enables this sovereignty model. The HUK is not merely a security feature; it is the foundation for device-anchored self-sovereign credentials.

For a quantum-resistant credential device, these hardware security features are essential; they cannot be replicated in software alone.

### ADC Capabilities

| Specification | Value | Significance |
|---------------|-------|--------------|
| ADC count | 2x 12-bit | Dual independent converters |
| Sample rate | 5 Msps (interleaved) | High-speed entropy sampling |
| Resolution | 12-bit | 4096 levels vs MCP3004's 1024 |
| Channels | Multiple per ADC | Four-channel avalanche support |

The 12-bit resolution and 5 Msps rate exceed the YoshiPi's MCP3004 by significant margins.

### Processing Power

| Specification | Value |
|---------------|-------|
| Core | Arm Cortex-M33 |
| Frequency | 200 MHz |
| Flash | 2 MB |
| SRAM | 512 KB |

The Cortex-M33 includes the DSP extension and single-cycle multiplier, useful for cryptographic operations and entropy conditioning.

### Connectivity

| Interface | Capability |
|-----------|------------|
| USB | Full Speed + High Speed |
| Ethernet | MAC with DMA |
| CAN FD | Automotive-grade |
| Serial | Multiple UART/SPI/I2C |

USB High Speed enables fast credential transfer without the USB Full Speed bottleneck.

---

## Development Environment

### The Clef-Native Direction

The RA6M5 track follows the same **direct register emission** model already proven by the STM32L5 standing-art samples (`samples/embedded/stm32l5-blinky/STM32L5.fs`, `samples/embedded/stm32l5-uart/STM32L5.UART.fs`): hand-written Clef defining register bases/offsets (RCC/GPIO MODER/BSRR/ODR) and using `Ptr.read<uint32>` / `Ptr.write` (Alloy.Memory) with type-safe enums. There is **no binding to a vendor HAL** and **no linking against vendor driver objects** — Clef lowers to the memory-mapped register layer (CMSIS) directly.

The layered architecture, top to bottom:

```
QuantumCredential application (Clef, reactive)
        │
Vendor port library (Clef, thin)   Fidelity.Platform.MCU.Renesas.RA6M5  /  ...ST.STM32L5
   (clock/PLL, pin mux, peripheral config sequences, vector wiring)
        │  uses
Unified CMSIS HAL (Clef-native)    CMSIS-Core: NVIC / SysTick / SCB  +  register-poke primitives
   (shared across all ARM Cortex-M; per-device register defs beneath, derivable from each vendor SVD)
        │  lowers via
Quotation-based memory → direct register emission
   (Ptr.read / Ptr.write ; <@ { Region = Peripheral; Access = WriteOnly; Volatile } @>)
        │
Memory-mapped registers → single-core silicon
```

The "Unified CMSIS HAL" is a **Clef-native register layer**, not a binding to ARM's CMSIS C headers: CMSIS-Core (NVIC/SysTick/SCB) is genuinely common across all Cortex-M, while the per-device register addresses/bitfields are the reusable per-device substrate (derivable from each vendor's SVD). This matures the MCU story across the board — every Cortex-M target shares the same register layer and reactive runtime, not just the RA6M5 credential device.

### Renesas FSP as a Ported Reference (not a binding target)

Renesas FSP is an **imperative driver framework layered over CMSIS**. It is mined as a *roadmap* for higher-level concerns — never bound to, never linked against. We **port** FSP's domain knowledge into clean Clef-native code that lowers to CMSIS:

- **Keep** — per-peripheral module decomposition, the catalogue of config knobs, lifecycle stages, datasheet-aligned vocabulary, and the `Fidelity.Platform.MCU.Renesas.RA…` namespace taxonomy (hand-authored, not generated from FSP).
- **Reimplement over the Clef CMSIS HAL** — clock/PLL/CGC bring-up, pin mux (PFS as Clef data), per-peripheral register config/enable sequences, and NVIC/vector setup dispatched into Clef concurrency.
- **Discard** — the `*_api_t` vtable / `_instance_t` / `_ctrl_t` / `_cfg_t` quartet, configurator codegen, RTOS glue, global ctrl handles, error-code + out-param style, manual init ordering, and callback function pointers.

The principle is **keep the map, redraw the roads.** This is the **PORT** mode of Transcribe (agent-driven Clef-native reimplementation), distinct from the **BIND** mode used for large, well-factored C libraries (e.g. a USB device stack, or LVGL on display-equipped targets), ingested via the Farscape membrane → Clef binding + Alex lowering witness pairs. (The RA6M5 is a headless USB key with status LEDs; a phone provides its display surface over USB — see [D-02-Mobile-Companion](../Demo/D-02-Mobile-Companion.md).)

### Reactive Runtime (replaces RTOS + callbacks)

Concurrency is native to Clef and lowered to a single-core unikernel:

- **Peripheral IRQ → `Observable`** (opaque push event source) instead of an ISR callback.
- **Controller/credential state + derived outputs → `Incremental`** (demand-driven, cutoff-bounded, self-adjusting).
- The model **normatively mandates Observable→Incremental fusion**: the PSG observer edge becomes the incremental staleness edge — no callback indirection, no heap bridge, no materialized subscription.
- **`Async` → LLVM coroutine intrinsics** (compile-time state machines, *no runtime library*) — freestanding/unikernel-perfect.
- **Subscriptions are region-scoped** (deterministic teardown with the arena; no GC/`IDisposable`).
- The **unikernel scheduler *is* the Incremental stabilization loop.** Single-core is the ideal substrate for first standing this up.

> **Forcing-function reality.** This reactive/concurrency model (Observable/Incremental/Actor Olivier-Prospero/Async/Signals/coeffects/BAREWire) is richly *specified* but largely *unimplemented* as CCS intrinsics today (NTUKind has only NTUlazy/NTUseq; IntrinsicModule registers Arena only; Alex has no actor/async/incremental witnesses). This MCU work is therefore the **forcing function** that first stands these intrinsics up — and single-core is the right place to do it.

### RTOS / Zephyr as an Optional Bridge

The RA6M5 has [first-class Zephyr RTOS support](https://docs.zephyrproject.org/latest/boards/renesas/ek_ra6m5/doc/index.html), and FreeRTOS remains available as a fallback bootstrap. These stay **optional bridges**, not the architecture: the headline path is the reactive unikernel. The current minimal C startup is salvageable as-is, but its vector table referencing FreeRTOS port symbols (`vPortSVCHandler` etc.) is a *separable* FreeRTOS coupling to be removed.

### Composer Integration Path

```
Clef Source + CCS
       ↓
   Composer/Alex   (Zipper + XParsec + Platform.Bindings)
       ↓
   MLIR (ARM Cortex-M33 target)
       ↓
   LLVM
       ↓
   ELF Binary (Clef-native register emission + minimal C startup; no vendor driver linkage)
```

The Platform.Bindings pattern applies identically across targets (registered by `(OS, Arch, EntryPoint)`); only the per-device register definitions and config sequences differ.

---

## Phase 2 Document Organization

```
Phase2-Embedded/
├── PH2-00-Embedded-Strategy.md      ← This document
├── RA6M5/
│   ├── README.md                    ← RA6M5 track index
│   ├── PH2-01-Strategy-Overview.md  ← Overall embedded plan
│   ├── PH2-02-Hardware-Platform.md  ← Board and entropy hardware
│   ├── PH2-03-Binding-Surface.md    ← Clef-native CMSIS register surface
│   └── PH2-04-Bootstrap-Options.md  ← Reactive unikernel vs RTOS bridge
└── STM32L5/                         ← Secondary target (on hold)
    ├── PH2-01-Strategy-Overview.md
    ├── PH2-02-Hardware-Platforms.md
    ├── PH2-03-Farscape-Assessment.md
    └── PH2-04-UI-Options.md
```

> The RA6M5 track now follows the **Clef-native CMSIS-HAL + reactive direction** described above: direct register emission (mirroring the STM32L5 standing art), FSP *ported* (not bound), concurrency expressed natively in Clef and lowered to a single-core reactive unikernel. The RA6M5 documents are being updated in place to preserve their lineage of thinking — the hardware/security content (Cortex-M33, HUK, TrustZone, SCE9, ADC) is architecture-neutral and carries forward unchanged.

---

## Migration from YoshiPi

The Phase 1 YoshiPi implementation provides the foundation:

| Component | YoshiPi (Phase 1) | RA6M5 (Phase 2) |
|-----------|-------------------|-----------------|
| Entropy source | 4-channel avalanche | Same circuit, different ADC |
| ADC interface | MCP3004 via SPI/IIO | RA6M5 internal 12-bit ADC |
| Epsilon evaluation | Software (Clef) | Same algorithm |
| XOR combination | Software (Clef) | Same algorithm |
| Crypto operations | Software (ML-KEM/DSA) | Hardware accelerated (SCE9) |
| Credential storage | File system | Secure Flash (TrustZone) |
| Device binding | None (portable) | **HUK-wrapped (clone-proof)** |
| Device identity | MAC address | **Factory Unique ID** |

The Clef application code remains largely unchanged; Platform.Bindings implementations differ.

---

## STM32L5 Status

The STM32L5 documentation in `STM32L5/` remains useful as an earlier bootstrap path, but it is no longer the primary planning target:

- **Historical reference**: The platform analysis and Farscape assessment capture the earlier STM32L5 path
- **Lower priority**: RA6M5's factory-provisioned HUK enables the sovereignty model that STM32L5 cannot match
- **Different use case**: STM32L5 may suit applications where device-bound identity is not required
- **Future option**: May revisit for cost-optimized variants or different product tiers

---

## Next Steps

1. **RA6M5/PH2-02-Hardware-Platform.md**: Document the board and entropy circuit
2. **RA6M5/PH2-03-Binding-Surface.md**: Define the first Clef-native CMSIS register surface (extend the STM32L5 standing-art pattern to RA6M5; factor out unified CMSIS-Core)
3. **RA6M5/PH2-04-Bootstrap-Options.md**: Frame the reactive unikernel as the headline path, with RTOS (FreeRTOS/Zephyr) as the optional bridge
4. Validate avalanche circuit compatibility with RA6M5 ADC sampling

### Open Questions (carried, not resolved here)

These remain open for the RA6M5 track and its sibling design docs:

1. **ISR / vector boundary** — hardware vectors carry no `void*` userdata, so the LVGL-style closure-via-userdata bridge does not transfer; ISR → `Observable` likely needs a captureless top-level handler bound to the vector slot, with peripheral state via the register HAL. Currently unspecified.
2. **Secure world (SCE9 / HUK / TrustZone-M)** — HUK is reachable only via the SCE9 private bus, so pure register-poking may be insufficient; likely needs a thin secure-world call surface (central to the device's sovereignty premise).
3. **Bind/port seam** — the exact line between the unified CMSIS HAL and the per-vendor port library.
4. **Farscape's revised Renesas role** — ingesting the Renesas SVD/CMSIS-Device for register address/bitfield constants (not FSP headers), vs hand-authored constants.
5. **Unikernel scheduler** — cooperative event loop (Incremental stabilization) vs preemption; cooperative is the likely fit for a single core.
6. **Transcribe PORT-as-mode** — confirming PORT lands as a mode of Transcribe alongside BIND.

---

## References

- [Renesas RA6M5 Product Page](https://www.renesas.com/en/products/ra6m5)
- [EK-RA6M5 Evaluation Kit](https://www.renesas.com/en/products/microcontrollers-microprocessors/ra-cortex-m-mcus/ek-ra6m5-evaluation-kit-ra6m5-mcu-group)
- [RA6M5 Security Manual](https://www.renesas.com/en/document/apn/ra6m5-mcu-group-security-manual) - HUK, SCE9, key wrapping details
- [Secure Key Management Tool](https://www.renesas.com/en/document/mat/security-key-management-tool-users-manual)
- [Zephyr RTOS RA6M5 Support](https://docs.zephyrproject.org/latest/boards/renesas/ek_ra6m5/doc/index.html)
- [RA6M5 Datasheet](https://www.renesas.com/en/document/fly/renesas-ra6m5-group)
