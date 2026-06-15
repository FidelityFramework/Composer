# PH2-07: RA6M5 Module Binding Map

## Purpose

This document maps the predicted RA6M5 library shape as precisely as possible, using the thin-branch / leaf taxonomy as the organizing rule.

The module surface should be readable as a stack:

1. shared platform contracts
2. unified CMSIS HAL (Clef-native register layer)
3. Renesas/RA branch modules
4. EK-RA6M5 leaf modules
5. workload adapters for sample processing and bootstrapping

> **Derivation note.** These modules are **hand-authored Clef** that lower to memory-mapped registers, mirroring the standing art in `samples/embedded/stm32l5-blinky/STM32L5.fs` and `samples/embedded/stm32l5-uart/STM32L5.UART.fs`. They are **not** Farscape-generated wrappers over FSP entry points, and the ELF does **not** link against FSP drivers. FSP is mined as a *domain roadmap* (the **keep / reimplement / discard** taxonomy in [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md)) — its per-peripheral decomposition and datasheet-aligned vocabulary survive; its imperative `*_api_t`/`_ctrl_t`/`_cfg_t` scaffolding does not. "Keep the map, redraw the roads."

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

### 2. Unified CMSIS HAL Layer

This layer is the Clef-native register substrate shared across all ARM Cortex-M targets — it is **not** a binding to ARM's CMSIS C headers. CMSIS-*Core* (NVIC / SysTick / SCB) is genuinely common across Cortex-M; the CMSIS-*Device* register addresses and bitfields differ per chip and live in the per-device modules beneath (derivable from each vendor's SVD).

- `Fidelity.Platform.MCU.ARM.CortexM.*`
  - CMSIS-Core register-poke primitives over `Ptr.read<uint32>` / `Ptr.write` (`Alloy.Memory`)
  - NVIC interrupt enable/disable/priority
  - SysTick configuration
  - SCB (system control block) access
  - quotation-based memory constraints for peripheral regions (`<@ { Region = Peripheral; Access = WriteOnly; Volatile } @>`)

The Renesas RA branch *uses* this layer; the RA branch never re-implements core Cortex-M primitives.

### 3. Renesas / RA Branch Layer

These modules are family-wide and shareable across multiple RA boards.

- `Fidelity.Platform.MCU.Renesas.RA.*`
  - generic Renesas MCU contracts
  - board-agnostic endpoint metadata
  - RA family capability declarations

- `Fidelity.Platform.MCU.Renesas.RA.IOPort`
  - pin and port register operations
  - generic pin direction and value operations
  - port-level reads and writes
  - configuration of multiple pins as a group

- `Fidelity.Platform.MCU.Renesas.RA.ADC`
  - ADC register operations
  - channel selection
  - scan triggering
  - conversion result retrieval

- `Fidelity.Platform.MCU.Renesas.RA.Timer`
  - timer initialization
  - periodic scheduling primitives
  - delay and capture support

- `Fidelity.Platform.MCU.Renesas.RA.Clock`
  - board-independent clock and power bring-up (CGC: clock generation, PLL tree)

- `Fidelity.Platform.MCU.Renesas.RA.Interrupts`
  - NVIC/vector wiring dispatched into Clef concurrency
  - peripheral-event primitives surfaced as `Observable` sources

These modules describe the Renesas way of talking to the hardware, expressed as direct register access over the unified CMSIS HAL, without binding to a specific board. Each module's *shape* (which knobs exist, lifecycle stages, vocabulary) is ported from FSP's per-peripheral decomposition; each module's *implementation* is `Ptr.read`/`Ptr.write` against RA register definitions, not calls into `R_IOPORT_*` / `R_ADC_*` / `R_BSP_*`.

### 4. EK-RA6M5 Leaf Layer

These modules are board-specific and own the concrete wiring.

- `Fidelity.Platform.MCU.Renesas.RA6M5`
  - RA6M5 family-specific package anchor
  - per-device register address/bitfield definitions (the CMSIS-Device substrate, derivable from the Renesas SVD)
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
  - concrete mapping from Renesas IOPORT operations to EK-RA6M5 pins
  - board-specific pin names and numbers
  - pin-mux (PFS) tables expressed as Clef data
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

### 5. Workload Adapter Layer

These modules model application behavior on top of the register layer.

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
  - startup glue that is more about workload orchestration than hardware semantics
  - reactive-unikernel entry wiring (peripheral IRQ → `Observable`, state → `Incremental`; the stabilization loop is the scheduler)
  - RTOS coordination only if present as an *optional* bridge, never as the architecture

These modules consume the board leaf.

> **Reactive concurrency, not callbacks.** The adapter layer expresses concurrency natively in Clef — peripheral interrupts surface as `Observable` push sources and controller/credential state as demand-driven `Incremental` graphs, fused so the observer edge becomes the staleness edge (no callback indirection, no heap bridge). This model is currently **specified but largely unimplemented** in CCS, so the RA6M5 work is the forcing function that first stands these intrinsics up on a single core. See [PH2-04-Bootstrap-Options.md](./PH2-04-Bootstrap-Options.md) for the reactive-unikernel-vs-RTOS axis.

## Module Assignment

### Author Once, Reuse Many

Each peripheral is hand-authored once at the correct layer, then specialized downward:

- core Cortex-M primitives in the unified CMSIS HAL
- shared contract metadata in `Contracts`
- reusable register operations in the RA branch
- per-device register definitions and board wiring in the EK-RA6M5 leaf
- workload-specific helpers in adapter modules

### IOPORT Module Shape (ported from FSP)

FSP's IOPORT decomposition gives the **branch module its shape** — the operations worth modeling. These are ported as Clef register operations, not bound as `R_IOPORT_*` wrappers:

- `pinsCfg` / `pinCfg` — configure pin direction, drive, pull, pin-mux (PFS)
- `pinRead` / `pinWrite` — single-pin read/write via the port data registers
- `portDirectionSet` — port-direction register write
- `portRead` / `portWrite` — port-level read/write
- `pinEventInputRead` / `pinEventOutputWrite` — event-link reads/writes where the device supports them

FSP's lifecycle pair `open` / `close` (and its `_ctrl_t`/`_cfg_t` handles) is **discarded** — there is no driver object to open. Configuration is a register-write sequence; "open" collapses into `pinsCfg` plus clock enable, and there is no resource to "close". The *names that describe the domain* belong in the branch module; the leaf package instantiates them for the EK-RA6M5 pin map.

### Board Map Assignment

The board leaf should wire:

- actual LED pins to board-facing output helpers
- actual buttons to input helpers
- actual ADC channels to sampling helpers
- actual mux controls to sample helpers
- actual UART routes to serial helpers

If a symbol has a pin number, a board header reference, or a mux line associated with it, place it in the leaf.

## Authoring Plan

The layers are authored in dependency order, because each step consumes the layer beneath it:

1. unified CMSIS HAL (CMSIS-Core primitives over `Ptr.read`/`Ptr.write`)
2. contracts and shared descriptor helpers
3. Renesas RA branch register operations
4. EK-RA6M5 per-device register definitions, board leaf descriptor, and pin map
5. workload adapters for sample processing and bootstrapping

Farscape's role here is narrowed to **ingesting the Renesas SVD / CMSIS-Device** to emit register address and bitfield constants (the reusable per-device substrate in layer 4) — not parsing FSP headers. Whether those constants are SVD-ingested or hand-authored is an open question carried in [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md); the *operations* on top of them are hand-authored Clef regardless.

## Precision Rule

Whenever a symbol is about to be authored, ask:

- does this symbol depend on the board, or only on the family?
- is it a core Cortex-M primitive (NVIC/SysTick/SCB), or RA-specific?
- does this symbol belong to hardware semantics, or workload semantics?
- does this symbol need to know a pin number, mux line, or channel number?

If it is a core Cortex-M primitive, author it in the unified CMSIS HAL.

If the answer is board-specific, author it in the leaf.

If the answer is reusable across RA boards, author it in the branch.

If the answer is about the credential workload, author it in an adapter.

## Open Questions

These seams are carried explicitly rather than resolved here:

- **Bind/port seam.** The exact line between the unified CMSIS HAL (generic peripheral ops) and the per-vendor RA branch (clock/pin-mux/power silicon truths) determines how thin the branch stays. See [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md).
- **ISR / vector boundary.** Hardware vectors carry no `void*` userdata, so the closure-via-userdata bridge used for C library callbacks does not transfer. The IOPort/Interrupts `Observable` source likely needs a captureless top-level handler bound to the vector slot, reaching peripheral state through the register HAL — currently unspecified.
- **Secure-world surface.** HUK / SCE9 / TrustZone-M access (central to the credential device's sovereignty premise) is reachable only via the SCE9 private bus and may need a thin secure-world call surface that pure register-poking cannot cover. This affects the `Credential` adapter and is tracked in [PH2-02-Hardware-Platform.md](./PH2-02-Hardware-Platform.md).
- **SVD ingestion vs hand-authored constants.** Whether the per-device register definitions in the leaf are Farscape-SVD-ingested or hand-authored.

For the naming conventions that keep these layers stable, see [PH2-08-Naming-Conventions-and-Pilot-Map.md](./PH2-08-Naming-Conventions-and-Pilot-Map.md). For the keep/reimplement/discard port taxonomy and Farscape's revised SVD role, see [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md).
