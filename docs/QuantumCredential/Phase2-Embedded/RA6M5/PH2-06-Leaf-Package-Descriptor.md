# PH2-06: RA6M5 Leaf Package Descriptor

## Purpose

This document describes the concrete shape of the EK-RA6M5 leaf package as a binding artifact.

The goal is to make the board leaf predictable enough that Farscape can generate it from authoritative Renesas source material.

## Descriptor Role

The leaf package is the point where the binding taxonomy becomes concrete.

At this level, the package should answer:

- what board this is
- what family it belongs to
- what physical endpoints it exposes
- what the workload is allowed to assume about those endpoints
- what remains intentionally unbound until a higher layer uses it

## Predicted Package Shape

The package should look like a board-specific leaf under the Renesas RA branch.

### Namespace Shape

The most likely namespace should remain board-specific, along the lines of:

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5`

Inside that package, the descriptor should live in a small `Platform` module that exposes:

- a board descriptor
- planned endpoint families
- a small set of helper values for board identity and package metadata

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

For EK-RA6M5, this should eventually describe the board-level clock sources and the way the workload enters its runtime clock domain.

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

A place for caveats, citations, and binding policy.

This should be used to record:

- source pack provenance
- pin-map assumptions
- known board quirks
- any temporary omissions while the descriptor is still partial

## Predictive Contents

The EK-RA6M5 descriptor is likely to grow into three visible sections.

### 1. Board Identity

This is the stable header: vendor, family, device, package, and display name.

### 2. Endpoint Families

This is the leaf’s outward-facing surface:

- GPIO
- ADC
- UART
- I2C
- SPI
- timers
- interrupts
- USB

### 3. Board-Specific Wiring

This is the part that differentiates the evaluation kit from other RA6M5 boards.

It should include:

- status LEDs
- user buttons
- ADC channel assignments
- mux select lines for the sample front end
- serial port routing
- any board control signals that matter to the workload

## What Farscape Should Emit

Farscape should generate the leaf package as a **board identity plus endpoint map**, not as a giant dump of everything in the vendor SDK.

The generated leaf should include:

- the board descriptor
- a board-specific package name
- the concrete endpoint families exposed by the board
- the pin and channel references that connect board docs to bindings
- notes or manifests that preserve the provenance of the source material

## Leaf Responsibilities

The leaf carries the board-specific identity, wiring, and metadata:

- the EK-RA6M5 board descriptor
- the concrete pin and channel map
- board-specific clocks and resets
- board-scoped FSP-facing metadata

## Output Shape

A good leaf package should make it easy for later Farscape work to produce these sibling layers:

- a branch package for RA family contracts
- a board leaf for EK-RA6M5
- a workload adapter for sample acquisition and credential bootstrapping

The leaf is the place where all of those concerns become concrete and board-addressable.

## Taxonomy Check

Use one placement rule whenever a symbol is added:

- board knowledge belongs in the leaf
- family-wide behavior belongs in the branch
- workload helpers belong in the adapter layer
