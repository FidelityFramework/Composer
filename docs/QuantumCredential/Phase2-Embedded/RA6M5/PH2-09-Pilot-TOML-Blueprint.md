# PH2-09: EK-RA6M5 Pilot TOML Blueprint

## Purpose

This document drafts the Pilot recipe that drives the first-pass register-constant ingestion for the EK-RA6M5 track, and shows where the hand-authored Clef-native register modules attach to that output.

The earlier draft of this document assumed Pilot would parse Renesas FSP + BSP headers and emit Clef wrappers over FSP entry points (`R_IOPORT_*`, `R_ADC_*`, `R_BSP_*`). That premise is retired. **We do not bind to FSP.** FSP is an imperative driver framework layered over CMSIS; it is mined as a *roadmap* for domain knowledge (per-peripheral decomposition, config knobs, lifecycle stages, datasheet vocabulary) and then **ported** into clean Clef-native code that lowers to the memory-mapped register layer — exactly as the standing-art ST samples already do. See [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) and [PH2-07-Module-Binding-Map.md](./PH2-07-Module-Binding-Map.md) for the keep / reimplement / discard taxonomy that the recipe below implements.

What Pilot ingests changes accordingly: the source of truth is the **Renesas SVD / CMSIS-Device description** (register addresses, bitfields, peripheral instances) — *not* FSP headers. The output is **register-address/bitfield constants** that the hand-authored Clef register modules consume; the driver behavior itself (clock/PLL bring-up, pin mux, peripheral config/enable sequences, NVIC/vector wiring) is written by hand over the unified CMSIS HAL, never generated from a vendor driver.

The recipe is intentionally staged:

- branch namespaces capture reusable Renesas / RA register surfaces (the constants shared across RA boards)
- the board namespace captures EK-RA6M5-specific pin metadata and board wiring
- workload adapters such as `SamplePipeline` and `Bootstrap` remain downstream modules built on top of the leaf

That keeps Pilot focused on the source-driven part of the tree — the *constants* derivable from the SVD — while the *behavior* lives in hand-written Clef.

## Draft Recipe

The recipe ingests the device's SVD (or an equivalent CMSIS-Device register description) and partitions the emitted register/bitfield constants into the branch, leaf, and board namespaces. Peripheral *behavior* is not emitted here — only the address/bitfield substrate the Clef modules read and write.

```toml
[register-source]
name = "renesas_ra6m5"
# SVD / CMSIS-Device description for the device on the EK-RA6M5.
# Farscape's revised Renesas role is to ingest this for register
# address/bitfield constants, NOT to parse FSP driver headers.
svd = "/path/to/svd/R7FA6M5BH.svd"
# Optional: pin-mux (PFS) tables captured as Clef data rather than
# inferred from FSP board config. May be hand-authored alongside the SVD.
pin_tables = "/path/to/ek-ra6m5/pin_tables.toml"

[output]
mode = "fidelity"
# Emitted artifacts are Clef-native register-constant modules
# (peripheral base addresses, register offsets, bitfield enums),
# consumed by the hand-written register/driver modules.
directory = "../Registers"
kind = "register-constants"   # not driver wrappers

[options]
# Emit bitfields as type-safe Clef enums (mirrors the ST pattern's
# PinMode / OutputType / PullMode enums).
bitfield_enums = true
# Group constants by peripheral instance from the SVD.
group_by_peripheral = true

[[namespace]]
name = "Fidelity.Platform.MCU.Renesas.RA.IOPort"
description = "Reusable RA IOPORT/PORT register constants (base, offsets, PFS bitfields)"
source = "renesas_ra6m5"
# SVD peripheral groups that contribute the family-wide constants.
peripherals = ["PORT0", "PORT1", "PORT2", "PFS"]

[[namespace]]
name = "Fidelity.Platform.MCU.Renesas.RA.ADC"
description = "Reusable RA ADC register constants (base, offsets, control/scan bitfields)"
source = "renesas_ra6m5"
peripherals = ["ADC0", "ADC1"]

[[namespace]]
name = "Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Board"
description = "EK-RA6M5 board pin metadata and board wiring constants"
source = "renesas_ra6m5"
# Board-specific constants come from the pin tables, not from generic
# SVD peripheral instances.
pin_tables = true
```

The paths above are placeholders for the local SVD checkout and the hand-authored pin tables; replace them with the real board package paths before the recipe is used.

## Why This Shape

### `IOPort`

The IOPORT/PORT register surface is the cleanest direct fit for the reusable Renesas branch.

The PORT base addresses, register offsets, and PFS (pin function select) bitfields are reusable across RA boards and do not require EK-RA6M5-specific pin knowledge, so the *constants* belong in the branch. The Clef IOPort module that pokes these registers — the analogue of the ST sample's `configureMode` / `setPin` / `enableClock` over `Ptr.read` / `Ptr.write` — is hand-written and reads these constants. The FSP IOPORT vocabulary (`pinCfg`, `portRead`, `portWrite`, `pinWrite`, …) is preserved as the *shape* of the Clef API surface (per PH2-07), re-expressed in Clef idioms, not as wrappers over `R_IOPORT_*`.

### `ADC`

The ADC register surface belongs in the same branch layer for the same reason.

The board may choose concrete ADC channels and routing, but the register constants (base, scan-control, conversion-result offsets, bitfields) are family-wide. The driver behavior — scan-group setup, trigger, result retrieval — is reimplemented in Clef over those constants, taking FSP's scan/transfer abstractions as the domain map.

### `Board`

The board namespace is the first leaf boundary.

It should carry:

- board pin metadata (the EK-RA6M5 pin map, derived from the pin tables)
- board-specific wiring constants (LEDs, buttons, mux control lines, sample-front-end pins)
- the naming surface that later leaf modules can consume

This is the right place for the board identity to become concrete. The constants here are *board* facts (which physical pin is which LED), distinct from the *family* register facts emitted into the branch.

## What This Draft Does Not Emit

The draft keeps the first Pilot pass focused on the **register-constant substrate** — the part that is genuinely source-driven from the SVD. Everything that is *behavior* is hand-authored Clef, not Pilot output.

In particular:

- **No FSP driver wrappers.** There is no `R_IOPORT_*` / `R_ADC_*` / `R_BSP_*` wrapper surface, no `*_api_t` / `_instance_t` / `_ctrl_t` / `_cfg_t` quartet, no configurator codegen, and no linkage against FSP driver objects. Those are in the DISCARD bucket of the port taxonomy (PH2-05).
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.IOPort` (the concrete pin-map binding) is derived from the board pin tables and the branch constants, not emitted as a separate raw namespace.
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.SamplePipeline` is a workload adapter built on top of the leaf — hand-written Clef choreography, not generated.
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.Bootstrap` is orchestration glue (clock/PLL bring-up sequence, vector wiring), reimplemented in Clef over the unified CMSIS HAL — not a binding target.

That separation keeps the tree honest: Pilot owns the *constants*; the hand-written register modules own the *operations*.

## Relationship to the Standing Art

The output of this recipe plus the hand-written register modules together mirror the ST standing art (`samples/embedded/stm32l5-blinky/STM32L5.fs`, `samples/embedded/stm32l5-uart/STM32L5.UART.fs`):

- the **constants** (`RCC_BASE`, `GPIOx_BASE`, `MODER_OFFSET`, `BSRR_OFFSET`, the `PinMode` / `OutputType` / `PullMode` enums) correspond to what Pilot emits from the SVD;
- the **operations** (`enableClock`, `configureMode`, `setPin`, `clearPin` via `Ptr.read<uint32>` / `Ptr.write`, `Alloy.Memory`) correspond to the hand-written Clef register/driver modules.

The ST samples define both by hand. For RA6M5 we factor the *constant* half out into Pilot (because the SVD makes it mechanical) and write the *operation* half by hand (because that is where FSP's domain knowledge is ported, not bound). This is the canonical direct-register-emission direction from `Architecture_Canonical.md` and `Quotation_Based_Memory_Architecture.md`, reused verbatim; only the per-device addresses/bitfields/sequences are new.

## Notes For Refinement

This draft should be refined after a real scan of the EK-RA6M5 SVD / CMSIS-Device pack and the hand-authored pin tables.

At that point we should confirm:

- whether Farscape's SVD ingestion produces the register/bitfield constants cleanly, or whether some constants are better hand-authored (an open question carried in [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md))
- whether the pin tables are best expressed as the separate Clef data shown above or folded into the board leaf module directly
- whether additional branch namespaces such as `Clock`, `Timer`, or `Interrupts` should be added from the same SVD (their register constants are family-wide; their bring-up sequences are hand-written, dispatched into Clef concurrency rather than FSP callbacks)
- how the ISR / vector-table boundary is wired (hardware vectors carry no `void*` userdata, so the constants here pair with a captureless top-level Clef handler reaching peripheral state via the register HAL — see [PH2-04-Bootstrap-Options.md](./PH2-04-Bootstrap-Options.md); open and unspecified)

The important part is that Pilot remains aligned with the thin-branch / leaf taxonomy and stays scoped to the *constant* substrate: it encodes the tree explicitly and feeds the hand-written Clef register layer, instead of generating wrappers over a vendor driver.

For the canonical naming and source-to-namespace mapping, see [PH2-08-Naming-Conventions-and-Pilot-Map.md](./PH2-08-Naming-Conventions-and-Pilot-Map.md).
