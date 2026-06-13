# RA6M5 Embedded Track

This directory is the active embedded planning path for QuantumCredential on the Renesas EK-RA6M5 evaluation kit.

It replaces the old STM32L5-centered assumption with a board-specific plan built around:

- the RA6M5 hardware root of trust
- the quad-sample analog front end
- a binding surface centered on Renesas FSP and board support code
- an optional FreeRTOS bootstrap layer, if it helps early bring-up

The broader product story still matters:

- YoshiPi remains the proof that Composer can handle a normal Linux-shaped workload with a screen and ADC-driven I/O.
- RA6M5 is the proof that Composer can also do secure native embedded work on a production-oriented MCU.
- FreeRTOS is a bridge, not the destination, so the embedded track stays aligned with a direct-hardware end state.

## Documents

| Document | Purpose |
|----------|---------|
| [PH2-01-Strategy-Overview.md](./PH2-01-Strategy-Overview.md) | Overall embedded plan and sequencing |
| [PH2-02-Hardware-Platform.md](./PH2-02-Hardware-Platform.md) | Board capabilities and sample front end hardware |
| [PH2-03-Binding-Surface.md](./PH2-03-Binding-Surface.md) | What the generated bindings must expose |
| [PH2-04-Bootstrap-Options.md](./PH2-04-Bootstrap-Options.md) | FreeRTOS vs direct hardware bring-up |
| [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) | Thin-branch / leaf taxonomy and binding work breakdown |
| [PH2-06-Leaf-Package-Descriptor.md](./PH2-06-Leaf-Package-Descriptor.md) | Predictive shape of the EK-RA6M5 leaf package |
| [PH2-07-Module-Binding-Map.md](./PH2-07-Module-Binding-Map.md) | Module-by-module FSP, branch, leaf, and adapter map |
| [PH2-08-Naming-Conventions-and-Pilot-Map.md](./PH2-08-Naming-Conventions-and-Pilot-Map.md) | Canonical names and Pilot source-to-namespace mapping |
| [PH2-09-Pilot-TOML-Blueprint.md](./PH2-09-Pilot-TOML-Blueprint.md) | Draft Pilot project file for the EK-RA6M5 binding pass |

## Working Assumptions

- Farscape can generate usable C and C++ bindings for the hardware-facing libraries we need.
- The first production-oriented surface should be Renesas FSP and adjacent board support code.
- STM32L5 remains a useful historical reference, but it is no longer the primary embedded path.
