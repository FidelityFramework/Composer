# PH2-07: RA6M5 Module Binding Map

## Purpose

This document maps the predicted RA6M5 library shape as precisely as possible, using the thin-branch / leaf taxonomy as the organizing rule.

The binding surface should be readable as a stack:

1. shared platform contracts
2. Renesas/RA branch modules
3. EK-RA6M5 leaf modules
4. workload adapters for sample processing and bootstrapping

## Module Map

### 1. Shared Contract Layer

These modules stay substrate-neutral and provide the shared descriptor vocabulary.

- `Fidelity.Platform.Contracts`
  - `PlatformDescriptor`
  - `EndpointGroup`
  - `PinEndpoint`
  - `ClockEndpoint`
  - `ResetEndpoint`
  - `UartEndpoint`
  - shared descriptor utilities

This layer is the type vocabulary that every leaf package relies on.

### 2. Renesas / RA Branch Layer

These modules are family-wide and shareable across multiple RA boards.

- `Fidelity.Platform.MCU.Renesas.RA.*`
  - generic Renesas MCU contracts
  - board-agnostic endpoint metadata
  - RA family capability declarations

- `Fidelity.Platform.MCU.Renesas.RA.IOPort`
  - pin and port contracts
  - generic pin direction and value operations
  - port-level reads and writes
  - configuration of multiple pins as a group

- `Fidelity.Platform.MCU.Renesas.RA.ADC`
  - ADC driver contracts
  - channel selection
  - scan triggering
  - conversion result retrieval

- `Fidelity.Platform.MCU.Renesas.RA.Timer`
  - timer initialization
  - periodic scheduling primitives
  - delay and capture support

- `Fidelity.Platform.MCU.Renesas.RA.Clock`
  - board-independent clock and power bring-up contracts

- `Fidelity.Platform.MCU.Renesas.RA.Interrupts`
  - interrupt registration and event primitives

These modules should describe the Renesas way of talking to the hardware without binding to a specific board.

### 3. EK-RA6M5 Leaf Layer

These modules are board-specific and own the concrete wiring.

- `Fidelity.Platform.MCU.Renesas.RA6M5`
  - RA6M5 family-specific package anchor
  - namespace that groups the EK-RA6M5 leaf
  - evaluation-kit metadata entry point

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5`
  - actual EK-RA6M5 board identity
  - board descriptor payload
  - board pin map
  - board endpoint grouping
  - evaluation-kit quirks and notes

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Board`
  - LED pins
  - buttons
  - UART wiring
  - board-control GPIO
  - other board-specific endpoints

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.IOPort`
  - concrete mapping from Renesas IOPORT contracts to EK-RA6M5 pins
  - board-specific pin names and numbers
  - port and pin aliases only where they are backed by the board docs

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.ADC`
  - concrete ADC channel mapping
  - ADC group selection
  - mux-dependent sampling configuration

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Clock`
  - board-specific clock source wiring
  - clock tree assumptions that are unique to the evaluation kit

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Reset`
  - reset wiring and reset-related board conventions

This is the layer where the board becomes concrete.

### 4. Workload Adapter Layer

These modules model application behavior on top of the generated bindings.

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.SamplePipeline`
  - mux select sequencing
  - two-ADC sampling choreography
  - timing capture
  - per-line normalization hooks

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Credential`
  - secure storage access helpers
  - bootstrapping of the credential workload
  - device-identity helpers that sit above the raw leaf

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Bootstrap`
  - FreeRTOS coordination helpers if needed
  - startup glue that is more about workload orchestration than hardware semantics

These modules consume the board leaf.

## Binding Assignment

### Parse Once, Emit Many

Farscape should parse the Renesas source material once, then emit into the correct layer:

- shared contract metadata into `Contracts`
- reusable API contracts into the RA branch
- board wiring into the EK-RA6M5 leaf
- workload-specific helpers into adapter modules

### FSP/IOPORT Mapping

The FSP IOPORT model gives the branch layer its shape:

- `open`
- `close`
- `pinsCfg`
- `pinCfg`
- `pinEventInputRead`
- `pinEventOutputWrite`
- `pinRead`
- `pinWrite`
- `portDirectionSet`
- `portEventInputRead`
- `portEventOutputWrite`
- `portRead`
- `portWrite`

Those names belong in the branch contract. The leaf package should instantiate them for the EK-RA6M5 pin map.

### Board Map Mapping

The board leaf should bind:

- actual LED pins to board-facing output helpers
- actual buttons to input helpers
- actual ADC channels to sampling helpers
- actual mux controls to sample helpers
- actual UART routes to serial helpers

If a symbol has a pin number, a board header reference, or a mux line associated with it, place it in the leaf.

## Farscape Emission Plan

Farscape should produce the layers in this order:

1. contracts and shared descriptor helpers
2. Renesas RA branch APIs
3. EK-RA6M5 board leaf descriptor and pin map
4. workload adapters for sample processing and bootstrapping

This order matters because each step consumes the layer beneath it.

## Precision Rule

Whenever Farscape is about to emit a symbol, ask:

- does this symbol depend on the board, or only on the family?
- does this symbol belong to hardware semantics, or workload semantics?
- does this symbol need to know a pin number, mux line, or channel number?

If the answer is board-specific, emit to the leaf.

If the answer is reusable across RA boards, emit to the branch.

If the answer is about the credential workload, emit to an adapter.

For the naming conventions that keep these layers stable, see [PH2-08-Naming-Conventions-and-Pilot-Map.md](./PH2-08-Naming-Conventions-and-Pilot-Map.md).
