# PH2-05: RA6M5 Taxonomy and Farscape Plan

## Purpose

This document makes the thin-branch / leaf taxonomy explicit for the RA6M5 embedded track and maps the Farscape work needed to build the binding stack.

The key idea is that the binding tree should fan out in layers:

1. shared platform contracts
2. family-wide Renesas/RA binding branches
3. board-specific EK-RA6M5 leaf bindings
4. workload-specific adapters, such as sample acquisition

## Taxonomy

### 1. Shared Contracts

This is the substrate-neutral layer.

It defines things like:

- platform descriptors
- endpoint families
- device capability metadata
- board-independent binding contracts

This layer should stay free of pin maps, board quirks, and vendor-specific register details.

### 2. Family Branch

This is the Renesas RA family layer.

It should contain the generic driver contracts that apply across RA boards:

- IOPORT-style pin and port access
- ADC driver contracts
- timers and delays
- clocks and power initialization
- interrupt and event primitives

This layer captures the Renesas driver model, while the leaf maps the physical pins on EK-RA6M5.

### 3. Board Leaf

This is the EK-RA6M5 package itself.

It should bind:

- the actual board pin map
- the board LEDs, buttons, and headers
- ADC channel routing for the sample front end
- mux select wiring
- board-level clock and reset quirks
- any evaluation-kit-specific IO conventions

The board leaf is where the generic Renesas model becomes a concrete device.

### 4. Workload Adapter

This is the part that turns board capabilities into QuantumCredential behavior.

Examples:

- sample acquisition sequences
- calibration routines
- secure storage access helpers
- workload-facing GPIO or status indicators

This layer stays thin and complements the board leaf.

## Farscape Work Breakdown

Farscape needs to produce or support each layer separately.

### A. Parse FSP and board headers

Farscape must parse the Renesas headers that define:

- IOPORT APIs
- BSP startup APIs
- ADC and timer APIs
- board-configuration headers
- any generated configuration macros

The parser has to preserve:

- functions
- enums
- structs
- constants and macros
- pointer-typed device handles

### B. Normalize the driver vocabulary

The generated output should separate:

- family-wide contracts
- board-specific symbols
- workload helpers

That keeps the binding surface from collapsing into one flat namespace.

### C. Generate the family branch

For RA, Farscape should emit a branch package that includes the generic Renesas API surface.

This is where `IOPORT`, `ADC`, `Timer`, and similar family-level pieces live.

### D. Generate the board leaf

For EK-RA6M5, Farscape should emit the concrete board package.

This package should attach the generic APIs to:

- actual pin numbers
- actual ADC channels
- actual mux select lines
- actual board endpoints

### E. Surface the sample pipeline cleanly

The quad-sample front end should be represented as a workload-oriented helper rather than as a generic GPIO abstraction.

The board leaf should expose the raw control points, and the workload adapter should combine them into:

- mux selection
- ADC scan sequence
- value normalization
- timing/capture metadata

### F. Keep the output auditable

The generated bindings should remain easy to inspect.

That means:

- no hidden board inference
- no guessing pin maps
- no collapsing board identity into the family branch
- no compatibility aliases that obscure the leaf package

## Proposed Shape

A reasonable package shape is:

- `Fidelity.Platform.Contracts`
- `Fidelity.Platform.MCU.Renesas.RA6M5`
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5`
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.SamplePipeline`

The exact namespace names can change, and the structural split stays the same.

For canonical naming and Pilot source-to-namespace mapping, see [PH2-08-Naming-Conventions-and-Pilot-Map.md](./PH2-08-Naming-Conventions-and-Pilot-Map.md).

## Taxonomy Rule

If a binding needs board-specific pin knowledge, it belongs in the leaf.

If a binding can be shared across RA boards, it belongs in the branch.

If a binding only exists to make the workload easier to express, it belongs in the adapter layer.

That rule is the main defense against taxonomy drift.
