# PH2-09: EK-RA6M5 Pilot TOML Blueprint

## Purpose

This document drafts the Pilot recipe that should drive the first-pass Renesas FSP binding generation for the EK-RA6M5 track.

The recipe is intentionally staged:

- branch namespaces capture reusable Renesas / RA contracts
- the board namespace captures EK-RA6M5-specific support code and pin metadata
- workload adapters such as `SamplePipeline` and `Bootstrap` remain downstream modules built on top of the generated leaf

That keeps Pilot focused on the source-driven part of the tree.

## Draft Recipe

```toml
[library]
name = "renesas_fsp"
headers = [
  "/path/to/fsp/ra/fsp/inc/api/r_ioport_api.h",
  "/path/to/fsp/ra/fsp/inc/api/r_adc_api.h",
  "/path/to/fsp/ra/fsp/inc/api/bsp_api.h",
  "/path/to/fsp/ra/boards/ek-ra6m5/board/board.h"
]
include_paths = [
  "/path/to/fsp/ra/fsp/inc/api",
  "/path/to/fsp/ra/fsp/inc",
  "/path/to/fsp/ra/boards/ek-ra6m5/board",
  "/path/to/fsp/ra/boards/ek-ra6m5"
]
transitive_headers = [
  "bsp_cfg.h",
  "bsp_pin_cfg.h"
]

[output]
mode = "fidelity"
directory = "../Bindings"

[options]
opaque_handles = true
flags_enums = true

[[namespace]]
name = "Fidelity.Platform.MCU.Renesas.RA.IOPort"
description = "Reusable Renesas IOPORT contracts"
library = "renesas_fsp"
prefixes = ["R_IOPORT_"]

[[namespace]]
name = "Fidelity.Platform.MCU.Renesas.RA.ADC"
description = "Reusable Renesas ADC contracts"
library = "renesas_fsp"
prefixes = ["R_ADC_"]

[[namespace]]
name = "Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Board"
description = "EK-RA6M5 board support code and pin metadata"
library = "renesas_fsp"
prefixes = ["R_BSP_", "BSP_"]
functions = ["R_BSP_WarmStart"]
```

The paths above are placeholders for the local FSP checkout root and should be replaced with the real board package paths before the recipe is used.

## Why This Shape

### `IOPort`

The `R_IOPORT_` family is the cleanest direct fit for the reusable Renesas branch.

This is where the family-wide pin and port contracts belong, because they are reusable across RA boards and do not require EK-RA6M5-specific pin knowledge.

### `ADC`

The `R_ADC_` family belongs in the same branch layer for the same reason.

The board may choose concrete ADC channels and routing, but the driver contract remains family-wide.

### `Board`

The board namespace is the first leaf boundary.

It should carry:

- board support hooks
- generated board metadata
- pin configuration constants
- the naming surface that later leaf modules can consume

This is the right place for the board identity to become concrete.

## What This Draft Does Not Yet Emit

The draft keeps the first Pilot pass focused on the FSP-backed binding surface.

In particular:

- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.IOPort` is expected to be derived from board pin metadata rather than emitted as a separate raw FSP namespace
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.SamplePipeline` is a workload adapter built on top of the leaf
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Bootstrap` is orchestration glue, not a direct FSP binding target

That separation keeps the binding tree honest.

## Notes For Refinement

This draft should be refined after a real scan of the EK-RA6M5 FSP checkout.

At that point we should confirm:

- whether the board package exposes additional `R_BSP_` entry points worth naming explicitly
- whether any extra public headers need to move into `transitive_headers`
- whether additional branch namespaces such as `Clock`, `Timer`, or `Interrupts` should be added from the same source pack

The important part is that Pilot remains aligned with the thin-branch / leaf taxonomy.
