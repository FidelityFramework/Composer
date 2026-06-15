# PH2-08: RA6M5 Naming Conventions and Pilot Map

## Purpose

This document defines the naming convention for the RA6M5 embedded tree and shows how the Pilot recipe should map the RA6M5 hardware surface into that tree.

The names in this document are stable regardless of *how* a module is authored. The tree is populated by Clef-native register modules — hand-authored against the device datasheet and SVD, in the same spirit as the ST standing art (`samples/embedded/stm32l5-blinky/STM32L5.fs`, `samples/embedded/stm32l5-uart/STM32L5.UART.fs`). FSP is mined as a *domain map* (which peripherals, which config knobs, which lifecycle stages), not bound as a driver dependency — see [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) for the keep / reimplement / discard split.

The goal is to make the taxonomy machine-readable and stable enough that the module tree reads consistently no matter who authors a given branch or leaf.

## Canonical Naming Rule

Use names that describe the layer’s responsibility, not the source file or vendor framework that happened to define it.

- shared contracts describe the substrate-neutral platform vocabulary
- branch modules describe reusable Renesas / RA behavior
- leaf modules describe the EK-RA6M5 board and its physical wiring
- adapter modules describe workload behavior layered on top of the leaf

That rule keeps the tree thin and prevents namespace drift.

## Canonical Tree Shape

The preferred shape is:

- `Fidelity.Platform.Contracts`
- `Fidelity.Platform.MCU.Renesas.RA`
- `Fidelity.Platform.MCU.Renesas.RA6M5`
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5`
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Board`
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.IOPort`
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.ADC`
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.SamplePipeline`
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Bootstrap`

The `RA` branch is family-wide. `RA6M5` is the device-family leaf boundary. `EK_RA6M5` is the board identity.

Beneath the named branch and leaf modules sits the Clef-native register substrate: the per-device register bases, offsets, and bitfields (`IOPORT`, `ADC`, `CGC`, `GPT`, `SCI`, …), expressed exactly as the ST samples express `RCC_BASE`/`GPIOx_BASE` + `MODER`/`BSRR`/`ODR` offsets and accessed via `Ptr.read<uint32>` / `Ptr.write` (Alloy.Memory) with type-safe enums. These register definitions are derivable from the Renesas SVD / CMSIS-Device; the branch and leaf modules name and compose them.

## Board Name Conventions

The board has two naming forms:

- `EK-RA6M5` in human-facing text and display names
- `EK_RA6M5` in namespace segments, module names, and file stems

This split is intentional.

Hyphens read better in prose, but underscores are safer and more consistent in identifiers.

## Layer Naming Roles

### Shared Contracts

Use `Fidelity.Platform.Contracts` for substrate-neutral types and descriptor vocabulary.

This layer should keep names generic:

- `PlatformDescriptor`
- `EndpointGroup`
- `PinEndpoint`
- `ClockEndpoint`
- `ResetEndpoint`

### Renesas / RA Branch

Use `Fidelity.Platform.MCU.Renesas.RA.*` for family-level behavior that is reusable across RA boards.

Good branch names describe capability:

- `IOPort`
- `ADC`
- `Clock`
- `Timer`
- `Interrupts`

These names stay board-neutral.

### EK-RA6M5 Leaf

Use `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.*` for board-specific wiring and identity.

Good leaf names describe the concrete board concern:

- `Platform` for the board descriptor entry point
- `Board` for LEDs, buttons, headers, and other board endpoints
- `IOPort` for the concrete pin-map binding
- `ADC` for the board’s ADC channel wiring
- `SamplePipeline` for the quad-sample front end choreography
- `Bootstrap` for startup orchestration that is still board-aware

The leaf is where physical pin knowledge belongs.

### Workload Adapters

Use adapter names for behavior that belongs above the board but below the full product workload.

Examples:

- `SamplePipeline`
- `Credential`
- `Bootstrap`

These names should describe intent, not wiring.

## Pilot Mapping Rule

Pilot should be treated as the role-to-namespace map.

It should answer three questions explicitly:

1. which capabilities define the public Renesas surface
2. which symbols belong in the RA branch
3. which symbols are board-specific and therefore belong in the EK-RA6M5 leaf

That means the Pilot recipe should be written around semantic roles, not around raw filenames or vendor entry-point prefixes.

## Porting the FSP Domain Map

FSP's value here is its *decomposition* of the RA hardware surface — one coherent module per peripheral, with a datasheet-aligned vocabulary and an explicit catalogue of the config knobs that matter. That decomposition is what we keep. The FSP code itself (the `*_api_t` / `_instance_t` / `_ctrl_t` / `_cfg_t` quartet, configurator codegen, RTOS glue, callback dispatch) is discarded. See [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) for the full keep / reimplement / discard split.

Mapped onto this naming tree:

- the IOPORT peripheral's domain shape (pin/port direction, read, write, group configuration) belongs in the RA branch as `Fidelity.Platform.MCU.Renesas.RA.IOPort`
- the ADC peripheral's domain shape (channel selection, scan triggering, conversion result retrieval) belongs in the RA branch as `Fidelity.Platform.MCU.Renesas.RA.ADC`
- board-specific identity and configuration belong in `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Board` and related leaf modules
- the concrete board pin map belongs in `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.IOPort`
- workload-oriented use of those pins belongs in the adapter layer (`SamplePipeline`, etc.)

The public naming keeps both source filenames and vendor entry-point prefixes out of the taxonomy. We do not carry `R_IOPORT_*` / `R_ADC_*` / `R_BSP_*` symbol names into the tree; those are imperative FSP entry points, not part of the Clef-native surface. The branch modules name capabilities; the register definitions beneath them are derived from the SVD / CMSIS-Device.

### Recommended Module Roles

- the IOPORT register defs (port direction, output data, input data, drive control) and the branch contract that composes them map to `Fidelity.Platform.MCU.Renesas.RA.IOPort`
- the ADC register defs (control, channel select, conversion result) and the branch contract that composes them map to `Fidelity.Platform.MCU.Renesas.RA.ADC`
- board identity, board-control endpoints, and pin metadata map to `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Board` and related leaf modules
- the concrete board pin table maps to `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.IOPort`
- sample pipeline helpers map to `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.SamplePipeline`

If a symbol needs a board pin number, mux line, or ADC channel number, it is a leaf symbol.

## Pilot Namespace Selection

A Pilot project should select namespaces by responsibility:

- branch namespaces for reusable Renesas register modules and contracts
- leaf namespaces for EK-RA6M5 board wiring
- adapter namespaces for workload helpers

The namespace name should describe the semantic boundary, not a source header path or vendor prefix.

That is the main reason Pilot is useful here: it lets us encode the tree explicitly instead of hoping a parser infers it.

## Preferred Patterns

Use these patterns:

- module names that describe responsibility
- namespace segments that reflect the layer boundary
- separate names for `RA` and `EK_RA6M5`
- board-specific modules for board-specific wiring
- workload modules that sit above the leaf

## Predictive Shape

If the naming convention holds, the module tree should read like this:

- `Contracts` for shared vocabulary
- `RA` for reusable Renesas behavior
- `RA6M5` for device-family organization
- `EK_RA6M5` for the board leaf
- `Board`, `IOPort`, `ADC`, and `SamplePipeline` for concrete board-facing surfaces
- `Bootstrap` and `Credential` for higher-level workload adapters

That gives us a thin branch, a clear leaf, and a stable place to grow the Clef-native register modules — using the FSP decomposition as the map and the SVD / datasheet as the source of truth.

For the staged TOML recipe that implements this naming map, see [PH2-09-Pilot-TOML-Blueprint.md](./PH2-09-Pilot-TOML-Blueprint.md).
