# PH2-08: RA6M5 Naming Conventions and Pilot Map

## Purpose

This document defines the naming convention for the RA6M5 embedded tree and shows how Farscape Pilot should map Renesas FSP / IOPORT source material into that tree.

The goal is to make the taxonomy machine-readable and stable enough that the binding map can be generated cleanly.

## Canonical Naming Rule

Use names that describe the layer’s responsibility, not the source file that happened to define it.

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

Pilot should be treated as the source-to-namespace map.

It should answer three questions explicitly:

1. which headers define the public Renesas surface
2. which symbols belong in the RA branch
3. which symbols are board-specific and therefore belong in the EK-RA6M5 leaf

That means the Pilot recipe should be written around semantic roles, not around raw filenames.

## FSP / IOPORT Source Mapping

The Renesas FSP IOPORT surface should be mapped into the branch and leaf layers in a way that preserves the Renesas vocabulary while still making the tree readable.

At a high level:

- FSP IOPORT API contracts belong in the RA branch
- board pin assignments belong in the EK-RA6M5 leaf
- workload-oriented use of those pins belongs in the adapter layer

The public naming keeps source filenames out of the taxonomy.

### Recommended Source Roles

- `r_ioport.h` and associated IOPORT declarations map to `Fidelity.Platform.MCU.Renesas.RA.IOPort`
- `r_adc.h` and associated ADC declarations map to `Fidelity.Platform.MCU.Renesas.RA.ADC`
- board configuration headers map to `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Board` and related leaf modules
- generated board pin tables map to `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.IOPort`
- sample pipeline helpers map to `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.SamplePipeline`

If a symbol needs a board pin number, mux line, or ADC channel number, it is a leaf symbol.

## Pilot Namespace Selection

A Pilot project should select namespaces by responsibility:

- branch namespaces for reusable Renesas contracts
- leaf namespaces for EK-RA6M5 board wiring
- adapter namespaces for workload helpers

The namespace name should describe the semantic boundary, not the source header path.

That is the main reason Pilot is useful here: it lets us encode the tree explicitly instead of hoping the parser infers it.

## Preferred Patterns

Use these patterns:

- module names that describe responsibility
- namespace segments that reflect the layer boundary
- separate names for `RA` and `EK_RA6M5`
- board-specific modules for board-specific wiring
- workload modules that sit above the leaf

## Predictive Shape

If the naming convention holds, the generated tree should read like this:

- `Contracts` for shared vocabulary
- `RA` for reusable Renesas behavior
- `RA6M5` for device-family organization
- `EK_RA6M5` for the board leaf
- `Board`, `IOPort`, `ADC`, and `SamplePipeline` for concrete board-facing surfaces
- `Bootstrap` and `Credential` for higher-level workload adapters

That gives us a thin branch, a clear leaf, and a stable place for Pilot to grow bindings from the FSP source material.

For the staged TOML recipe that implements this naming map, see [PH2-09-Pilot-TOML-Blueprint.md](./PH2-09-Pilot-TOML-Blueprint.md).
