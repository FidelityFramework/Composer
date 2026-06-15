# PH2-06: RA6M5 Leaf Package Descriptor

## Purpose

This document describes the concrete shape of the EK-RA6M5 leaf package as a board **register/pin descriptor** — the hand-authored data that pins the board's identity, clock topology, pin map, and endpoint wiring to the Clef-native register modules that drive the silicon.

The goal is to make the board leaf predictable and explicit. The descriptor is **hand-authored** (or derived from the Renesas SVD / CMSIS-Device for register addresses and bitfields), not generated from a vendor SDK checkout. It is the standing-art register-module pattern (`samples/embedded/stm32l5-blinky/STM32L5.fs`) carried up to a board-level descriptor: the same kind of base addresses, register offsets, and type-safe enums, organized so a board is described once and consumed by the per-peripheral Clef modules.

> See [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) for the keep/reimplement/discard port taxonomy, and the parent retooling direction in [../PH2-00-Embedded-Strategy.md](../PH2-00-Embedded-Strategy.md). Farscape's revised RA6M5 role is to ingest the Renesas **SVD/CMSIS-Device** for register address/bitfield constants — never FSP headers.

## Descriptor Role

The leaf package is the point where the layered architecture becomes concrete for one board.

At this level, the package should answer:

- what board this is
- what family it belongs to
- what physical endpoints it exposes
- what the workload is allowed to assume about those endpoints
- what remains intentionally unbound until a higher layer uses it

The descriptor is data, not behavior. The clock/PLL bring-up, pin mux sequences, and per-peripheral register pokes live in the Clef-native register modules (the vendor port library over the unified CMSIS HAL); the leaf descriptor records the board-specific *constants* and *wiring* those modules consume.

## Predicted Package Shape

The package should look like a board-specific leaf under the Renesas RA branch.

### Namespace Shape

The most likely namespace should remain board-specific, along the lines of:

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5`

Inside that package, the descriptor should live in a small `Platform` module that exposes:

- a board descriptor
- planned endpoint families
- a small set of helper values for board identity and package metadata

This mirrors the standing art: a single Clef module (e.g. the `STM32L5` module's `Addresses` and per-peripheral submodules) owns the board's base addresses, offsets, and enums, and the rest of the code reads from it.

### Package Identity

The package identity should be explicit and stable:

- **Vendor**: Renesas
- **Family**: RA6M5
- **Device**: EK-RA6M5
- **Substrate**: MCU
- **Package**: Board

That identity is what prevents the leaf from turning into a generic RA6M5 package that forgets it is tied to the evaluation kit.

## Descriptor Fields

The current scaffold in `Fidelity.Platform` already hints at the shape we should preserve.

The descriptor should eventually carry:

- `Id`
- `DisplayName`
- `Substrate`
- `Vendor`
- `Family`
- `Device`
- `Package`
- `SpeedGrade`
- `Clocks`
- `Resets`
- `Groups`
- `Uarts`
- `DedicatedPins`
- `Notes`

## What Each Field Means

### `Id`

The stable machine-readable identifier for the leaf package.

This should remain immutable once published.

### `DisplayName`

The human-readable board name, such as `Renesas EK-RA6M5`.

### `Substrate`

The substrate category, which for this board is `MCU`.

### `Vendor`

The silicon vendor, which is `Renesas`.

### `Family`

The MCU family, which is `RA6M5`.

### `Device`

The board or evaluation kit identity, which is `EK-RA6M5`.

### `Package`

The leaf classification, which should stay `Board`.

### `SpeedGrade`

This can stay minimal at first, but it exists to preserve the contract for boards that need speed-bin metadata.

### `Clocks`

The clock topology for the leaf.

For EK-RA6M5, this should eventually describe the board-level clock sources and the way the workload enters its runtime clock domain. The values here feed the Clef-native clock/PLL/CGC bring-up module (the RA equivalent of the ST `RCC` clock-enable path), not a vendor configurator.

### `Resets`

Reset lines and reset-related board controls.

This includes the board-level reset path and MCU-internal reset semantics.

### `Groups`

Logical endpoint groups.

For the RA6M5 leaf, likely groups include:

- GPIO
- ADC
- timers
- serial I/O
- possibly USB and secure IO groupings

### `Uarts`

Serial endpoints exposed by the board.

These should be tied to board wiring and MCU capability.

### `DedicatedPins`

The leaf-specific pins that do not belong to an endpoint group.

This is where board LEDs, mux control lines, and sample-front-end control pins likely live.

### `Notes`

A place for caveats, citations, and provenance policy.

This should be used to record:

- source provenance (SVD/CMSIS-Device version, datasheet/board-manual citations)
- pin-map assumptions
- known board quirks
- any temporary omissions while the descriptor is still partial

## Predictive Contents

The EK-RA6M5 descriptor is likely to grow into three visible sections.

### 1. Board Identity

This is the stable header: vendor, family, device, package, and display name.

### 2. Endpoint Families

This is the leaf's outward-facing surface:

- GPIO
- ADC
- UART
- I2C
- SPI
- timers
- interrupts
- USB

Each family resolves to the per-peripheral Clef register module that owns its base address, register offsets, and type-safe enums — the same decomposition the FSP roadmap suggests (one module per peripheral), reimplemented over the unified CMSIS HAL rather than bound to vendor drivers.

### 3. Board-Specific Wiring

This is the part that differentiates the evaluation kit from other RA6M5 boards.

It should include:

- status LEDs
- user buttons
- ADC channel assignments
- mux select lines for the sample front end
- serial port routing
- any board control signals that matter to the workload

## How the Descriptor Is Produced

The leaf package is authored as a **board identity plus endpoint/register map**, not derived from a vendor SDK dump.

The register address and bitfield *constants* are the reusable, machine-derivable substrate: Farscape's revised RA6M5 role is to ingest the Renesas **SVD/CMSIS-Device** and emit those constants (or they are hand-transcribed from the datasheet). Everything above that — the board-specific decomposition, pin map, and wiring — is hand-authored Clef, exactly as the ST samples are.

The leaf should include:

- the board descriptor
- a board-specific package name
- the concrete endpoint families exposed by the board, each backed by a Clef register module
- the pin and channel references that connect board docs to register modules
- notes or manifests that preserve the provenance of the register/pin constants

## Leaf Responsibilities

The leaf carries the board-specific identity, wiring, and metadata:

- the EK-RA6M5 board descriptor
- the concrete pin and channel map
- board-specific clocks and resets
- board-scoped register/pin metadata and provenance (SVD/datasheet citations)

## Output Shape

A good leaf descriptor should make it easy for later port work to produce these sibling layers:

- a branch package for RA family register/peripheral contracts
- a board leaf for EK-RA6M5
- a workload adapter for sample acquisition and credential bootstrapping

The leaf is the place where all of those concerns become concrete and board-addressable.

## Taxonomy Check

Use one placement rule whenever a symbol is added:

- board knowledge belongs in the leaf
- family-wide behavior belongs in the branch
- workload helpers belong in the adapter layer

## References

- Standing art (the pattern this mirrors): `samples/embedded/stm32l5-blinky/STM32L5.fs`, `samples/embedded/stm32l5-uart/STM32L5.UART.fs` — hand-written Clef register modules (base addresses, offsets, type-safe enums, `Ptr.read`/`Ptr.write`).
- Corrected direction: [../PH2-00-Embedded-Strategy.md](../PH2-00-Embedded-Strategy.md) — Clef-native CMSIS HAL + port taxonomy; FSP as roadmap, not binding target.
- Taxonomy and Farscape's SVD-ingestion role: [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md).
- Module decomposition this descriptor feeds: [PH2-07-Module-Binding-Map.md](./PH2-07-Module-Binding-Map.md).
- `Architecture_Canonical.md`, `Quotation_Based_Memory_Architecture.md` — quotation-based memory + direct register emission.
