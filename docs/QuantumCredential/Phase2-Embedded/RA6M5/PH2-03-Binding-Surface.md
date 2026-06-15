# PH2-03: RA6M5 Binding Surface

## Purpose

This document defines the first hardware-facing surface for the EK-RA6M5 embedded track.

The RA6M5 hardware surface is a **Clef-native register surface**: per-peripheral Clef modules that define register bases, offsets, and bitfields and reach the silicon through `Ptr.read`/`Ptr.write` over the unified CMSIS register layer. This is **not** a binding to a vendor driver framework. Renesas FSP is treated as a *roadmap* — a domain map of peripheral decomposition, config knobs, and silicon bring-up sequences that we **port** into clean Clef, not link against. The canonical direction is **direct register emission**, not linking vendor driver objects (see `Architecture_Canonical.md`, `Quotation_Based_Memory_Architecture.md`).

This document owns the detailed layered architecture and the standing-art register-module pattern for the RA6M5 track. The keep/reimplement/discard port taxonomy and Farscape's SVD role live in [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md).

---

## The Layered Architecture

Hardware access is a stack of thin Clef layers, each lowering to the one beneath it, ending at direct register emission:

```
QuantumCredential application (Clef, reactive)
        │
Vendor port library (Clef)        Fidelity.Platform.MCU.Renesas.RA6M5
   (silicon truths only: clock/PLL/CGC, pin mux, peripheral config sequences, vector wiring)
        │  uses
Unified CMSIS HAL (Clef-native)   CMSIS-Core: NVIC / SysTick / SCB  +  register-poke primitives
   (shared across all ARM Cortex-M; per-device register definitions beneath, derivable from each vendor's SVD)
        │  lowers via
Quotation-based memory → direct register emission
   (Ptr.read / Ptr.write ; <@ { Region = Peripheral; Access = WriteOnly; Volatile } @>)
        │
Memory-mapped registers → single-core silicon
```

Two points the surface depends on:

- **The "Unified CMSIS HAL" is a Clef-native register layer, not a binding to ARM's CMSIS C headers.** CMSIS-*Core* (NVIC/SysTick/SCB) is genuinely common across all Cortex-M and is written once in Clef. The CMSIS-*Device* register addresses and bitfields differ per chip and are the reusable per-device substrate beneath it (derivable from each vendor's SVD).
- **The vendor port library is thin.** Once concurrency is native (see the reactive runtime, below) and the register layer is shared, the per-vendor delta is just the *silicon truths*: the clock/PLL/CGC tree, pin mux (PFS as Clef data), per-peripheral config/enable sequences, and interrupt-vector wiring. The exact line between the unified CMSIS HAL and the per-vendor port library is an open question (see Open Questions).

---

## Standing Art: The ST Register-Module Pattern

The RA6M5 register surface reuses the ST samples verbatim in *form*; only the per-device addresses, bitfields, and bring-up sequences are new. The ST samples are **already** the pattern — they are *not* bindings to ST's HAL:

- `samples/embedded/stm32l5-blinky/STM32L5.fs` — hand-written Clef defining `RCC_BASE`/`GPIOx_BASE` and register offsets (`MODER`/`OTYPER`/`BSRR`/`ODR`/`IDR`), implementing `enableClock`/`configureMode`/`setPin`/`readPin` via `Ptr.read<uint32>` / `Ptr.write` (`Alloy.Memory`) with type-safe enums.
- `samples/embedded/stm32l5-uart/STM32L5.UART.fs` — confirms this is an active, reusable line: register-offset and bitfield modules, a `UartConfig` record, a `UartPort` handle value, and plain Clef `init`/operations over `Ptr.read`/`Ptr.write`.

The shape every RA6M5 peripheral module follows:

```fsharp
module RA6M5.GPIO            // one module per peripheral

open Alloy.Memory

module Addresses =          // base addresses (from the SVD / datasheet)
    let PORT0_BASE = 0x40040000u
    // ...

module Registers =          // register offsets from the peripheral base
    let PDR  = 0x00u        // port direction
    let PODR = 0x02u        // port output data
    let PIDR = 0x04u        // port input data
    // ...

type PinDirection =         // type-safe enums, not raw magic numbers
    | Input  = 0u
    | Output = 1u

let inline configureDirection (port: int) (pin: int) (dir: PinDirection) =
    let addr = nativeint (Addresses.portBase port + Registers.PDR)
    let current = Ptr.read<uint32> addr
    Ptr.write addr ((current &&& ~~~(1u <<< pin)) ||| ((uint32 dir) <<< pin))
```

This is the entire mechanism: type-safe enums for field values, register addresses computed from base + offset, and `Ptr.read`/`Ptr.write` as the only egress to silicon. There is no vtable, no `_ctrl_t`/`_cfg_t` quartet, no global mutable handles, no configurator codegen. Peripheral instances are Clef values (records like `UartPort`), and operations return `Result` or plain values rather than error-code-plus-out-param.

---

## What the Surface Must Expose

The first surface set covers the workload's needs, each as a Clef register module ported from FSP's domain map over the unified CMSIS HAL:

- board and clock initialization (CGC: clock source select, PLL configuration, peripheral clock enables)
- GPIO/IOPort configuration and control (direction, output data, input read, pin mux via PFS)
- ADC setup, channel selection, and scan triggering
- timer and delay primitives (GPT / SysTick from CMSIS-Core)
- USB and serial I/O (SCI-UART)
- flash and secure storage operations
- TrustZone and secure-world handoff points (see Open Question: Secure World)

Each of these maps to a peripheral module in the `Fidelity.Platform.MCU.Renesas.RA…` taxonomy. The module *decomposition* and *vocabulary* are kept from FSP (the "what"); the register pokes are reimplemented in Clef (the "how").

---

## Sample-Pipeline Surface

The quad-sample analog front end (PH2-02) needs special attention; it feeds the credential workload directly. The ADC register module must make the following operations explicit and deterministic:

- select the active mux state
- trigger an ADC conversion
- read the sampled value
- repeat across both ADC channels in a known sequence
- surface timing and calibration data back to Clef

That is the minimal surface needed to make the quad-sample front end intelligible to the compiler and to the reactive runtime. Because acquisition order and channel-switch latency matter (PH2-02), the ADC module exposes scan sequencing as explicit Clef operations rather than hiding it behind a driver state machine. Conversion-complete events are surfaced to the application as an `Observable` (see below), not as an FSP callback.

---

## Module Shape and Naming

The Clef side stays close to the naming and dependency model used elsewhere in Composer:

- `Fidelity.Platform.MCU.Renesas.RA6M5.*` for the device port library (the namespace taxonomy is hand-authored, not Farscape-from-FSP)
- one module per peripheral, register-layer modules holding addresses/offsets/bitfields
- explicit, type-safe enums for register field values; record types for configuration and for peripheral handles
- no hidden dependence on a large runtime just to reach a peripheral — the only egress is `Ptr.read`/`Ptr.write`

See [PH2-08-Naming-Conventions-and-Pilot-Map.md](./PH2-08-Naming-Conventions-and-Pilot-Map.md) for the naming conventions and [PH2-07-Module-Binding-Map.md](./PH2-07-Module-Binding-Map.md) for the per-peripheral module map.

---

## Surface Focus

Keep the surface focused on explicit hardware access, with a clean separation between hardware and scheduling:

- ADC sequencing
- mux control
- security partition boundaries
- the distinction between hardware access (this layer) and scheduler policy (the reactive runtime / unikernel)

Concurrency is **not** part of the register surface. Peripheral IRQs are lifted to `Observable` sources and credential/controller state to `Incremental` graphs; the unikernel scheduler is the Incremental stabilization loop. If FreeRTOS or Zephyr is present, it serves as an optional bootstrap bridge beside these register modules, **not** as the architecture (see [PH2-04-Bootstrap-Options.md](./PH2-04-Bootstrap-Options.md) and PH2-00's narrative guardrail).

---

## Reactive Boundary (ISR → Observable)

The register surface is the steady-state read/write layer; events cross into the application as `Observable` sources. The framework normatively mandates Observable→Incremental fusion (the PSG observer edge becomes the incremental staleness edge — no callback indirection, no heap bridge, no materialized subscription), which is the formal reason FSP-style callbacks disappear. The register modules expose the *enable/clear/status* registers an ISR needs; the lifting of a vector to an `Observable` is a runtime concern, not a register-surface concern.

One boundary detail is unresolved and carried as an Open Question: hardware vectors carry **no** `void*` userdata slot, so the closure-via-userdata bridge used for C library callbacks does not transfer to ISRs.

---

## Salvageable As-Is

The following are CMSIS/bare-metal-neutral and kept unchanged:

- minimal C startup (`Reset_Handler`, vector table, `.data`/`.bss` init)
- the linker memory map (FLASH 0x0 2048K, RAM 0x20000000 512K)
- toolchain flags (`thumbv8m.main-none-eabi`, `-mcpu=cortex-m33`, `-mfpu=fpv5-sp-d16`, `-ffreestanding -nostdlib`)

The current vector table references FreeRTOS port symbols (`vPortSVCHandler`, etc.) — a *separable* FreeRTOS coupling, not a structural dependency (see PH2-04).

---

## Taxonomy Boundary

The peripheral tree follows a thin-branch / leaf split:

- family-wide Renesas RA contracts live in the branch layer (shared register definitions, common config patterns)
- EK-RA6M5 pin maps and quirks live in the leaf layer
- sample-pipeline helpers live above both as workload adapters

The branch/leaf split sits *above* the unified CMSIS HAL; the CMSIS-Core layer (NVIC/SysTick/SCB) is shared across all Cortex-M targets and sits beneath the Renesas branch. See [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) for the concrete work breakdown and the keep/reimplement/discard port taxonomy, and [PH2-06-Leaf-Package-Descriptor.md](./PH2-06-Leaf-Package-Descriptor.md) for the leaf register/pin descriptor schema.

---

## Open Questions

These are carried forward, not resolved here:

1. **Secure world (SCE9 / HUK / TrustZone-M).** The HUK is reachable only via the SCE9 private bus, so pure register-poking may be insufficient for the secure-crypto and key-wrapping operations. The secure surface likely needs a **thin secure-world call surface** — the one place the Clef-native register layer may not cover directly. This is central to the device's sovereignty premise (PH2-00), so the seam between register-level access and secure-world calls must be designed explicitly. Whether that surface is a small set of secure-monitor entry points, a `[<FidelityExtern>]`-style boundary, or a dedicated TrustZone-M call gate is unspecified.

2. **ISR / interrupt-vector boundary.** Hardware vectors carry no `void*` userdata slot, so the closure-via-userdata bridge does not transfer. The likely shape is a captureless top-level Clef handler bound directly to the vector-table slot, reaching peripheral state through the register HAL (not a captured env), and lifting events to an `Observable`. Currently unspecified.

3. **Bind/port seam.** The exact line between the unified CMSIS HAL (generic peripheral ops?) and the per-vendor port library (clock/pinmux/power only?) — this determines how thin the vendor library is.

4. **Farscape SVD ingestion vs hand-authored constants.** Whether Farscape ingests the Renesas SVD/CMSIS-Device to emit register address/bitfield constants, or those constants are hand-authored. (See PH2-05.)

---

## References

- Standing art: `samples/embedded/stm32l5-blinky/STM32L5.fs`, `samples/embedded/stm32l5-uart/STM32L5.UART.fs`.
- `Architecture_Canonical.md`, `Quotation_Based_Memory_Architecture.md` — quotation-based memory + direct register emission, `Platform.Bindings` by `(OS, Arch, EntryPoint)`.
- [PH2-00-Embedded-Strategy.md](../PH2-00-Embedded-Strategy.md) — embedded strategy and narrative guardrail (reactive unikernel; RTOS as optional bridge).
- [PH2-02-Hardware-Platform.md](./PH2-02-Hardware-Platform.md) — board, security features, ADC/sample front end.
- [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) — port taxonomy and Farscape's SVD role.
- [PH2-07-Module-Binding-Map.md](./PH2-07-Module-Binding-Map.md), [PH2-08-Naming-Conventions-and-Pilot-Map.md](./PH2-08-Naming-Conventions-and-Pilot-Map.md) — module map and naming.
