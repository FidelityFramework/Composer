# PH2-05: RA6M5 Port Taxonomy and Interop Plan

## Purpose

This document is the durable playbook for how the RA6M5 embedded track turns Renesas
domain knowledge into Clef. It owns two methodology pieces that apply to every
Cortex-M target, not just the credential device:

1. The **keep / reimplement / discard** taxonomy for *porting* FSP — "keep the map,
   redraw the roads."
2. The **bind-vs-port** criterion, expressed as two modes of **Transcribe**, including
   the Farscape boundary as it is actually implemented and Farscape's revised Renesas
   role.

It supersedes the earlier premise that Farscape would parse Renesas FSP/BSP headers
and emit Clef *bindings* over `R_IOPORT_*` / `R_ADC_*` / `R_BSP_*` that the ELF would
link against FSP drivers. **FSP is not a binding target.** It is an imperative driver
framework layered over CMSIS, and we treat it as a *reference map* whose domain
knowledge is ported into clean, Clef-native code that lowers directly to the
memory-mapped register layer — exactly as the standing ST work already does.

For the full corrected direction (the Clef-native CMSIS HAL, the reactive runtime, and
the layered architecture this taxonomy sits inside), see
[PH2-00-Embedded-Strategy.md](../PH2-00-Embedded-Strategy.md). For the board register
surface this taxonomy produces, see
[PH2-03-Binding-Surface.md](./PH2-03-Binding-Surface.md). The standing art it mirrors
is `samples/embedded/stm32l5-blinky/STM32L5.fs` and
`samples/embedded/stm32l5-uart/STM32L5.UART.fs` — hand-written Clef defining register
bases/offsets and using `Ptr.read<uint32>` / `Ptr.write` (Alloy.Memory) with type-safe
enums.

## The Layered Taxonomy

The package tree still fans out in layers; the structure is unchanged. What changed is
the *content* of each layer: every layer is now hand-authored Clef over the unified
CMSIS register HAL, not a Farscape-generated wrapper over FSP entry points.

1. shared platform contracts
2. family-wide Renesas/RA branch
3. board-specific EK-RA6M5 leaf
4. workload-specific adapters, such as sample acquisition

### 1. Shared Contracts

This is the substrate-neutral layer.

It defines things like:

- platform descriptors
- endpoint families
- device capability metadata
- board-independent register/peripheral contracts

This layer should stay free of pin maps, board quirks, and vendor-specific register
details.

### 2. Family Branch

This is the Renesas RA family layer.

It should contain the generic *peripheral* contracts that apply across RA boards,
re-expressed as Clef-native register operations rather than FSP API contracts:

- IOPORT-style pin and port access
- ADC peripheral contracts
- timers and delays
- clocks and power (CGC/PLL) initialization
- interrupt and event primitives

This layer captures the Renesas *silicon model* — the register blocks, bitfields, and
config sequences that are common across the family — while the leaf maps the physical
pins on EK-RA6M5. It is the natural home for the **REIMPLEMENT** bucket below.

### 3. Board Leaf

This is the EK-RA6M5 package itself.

It should define:

- the actual board pin map (as Clef data tables, ported from PFS settings)
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

This layer stays thin and complements the board leaf. It is where the reactive runtime
shows up first: IRQ-driven peripherals surface as `Observable` sources, and
credential/sample state derives via `Incremental` (see
[PH2-00-Embedded-Strategy.md](../PH2-00-Embedded-Strategy.md)).

## FSP as Roadmap: Keep / Reimplement / Discard

FSP is reference material — mined as a *domain map*, then replaced. It is never linked.
Three buckets define the PORT mode (§ Bind vs Port):

| Bucket | What | Examples |
|--------|------|----------|
| **KEEP** (the *what* + decomposition) | Per-peripheral module boundaries; the catalogue of config knobs that matter; lifecycle stages; datasheet-aligned vocabulary; *selected* API-surface shapes where they reflect the domain | One module per peripheral (IOPort, ADC, GPT, SCI-UART, …); scan-group / transfer abstractions; the `Fidelity.Platform.MCU.Renesas.RA…` namespace taxonomy (hand-authored, not Farscape-from-FSP) |
| **REIMPLEMENT** (the silicon truths, over the Clef CMSIS HAL) | Clock/PLL/CGC bring-up; pin mux (PFS → Clef data tables); per-peripheral register config/enable sequences; NVIC/vector setup dispatched into Clef concurrency | The register-poking FSP performs, re-expressed as `Ptr.read` / `Ptr.write` |
| **DISCARD** (imperative mechanisms) | The `*_api_t` vtable / `_instance_t` / `_ctrl_t` / `_cfg_t` quartet; configurator codegen; RTOS glue; global mutable ctrl handles; error-code+out-param style; manual init ordering; **callback function pointers** | Replaced by Clef-native concurrency + the reactive runtime |

Guiding principle for the KEEP bucket: **keep the map, redraw the roads** — preserve
FSP's decomposition and vocabulary, re-express every *operation* in Clef idioms
(`Result` returns, peripheral instances as Clef values, channels/events instead of
callbacks). The taxonomy in
[PH2-08-Naming-Conventions-and-Pilot-Map.md](./PH2-08-Naming-Conventions-and-Pilot-Map.md)
is the KEEP bucket made concrete: a hand-authored namespace tree that survives the
retooling because it describes the *domain*, not FSP's source files.

## Bind vs Port — Two Modes of Transcribe

C/C++ interop splits cleanly by the *nature* of the source. This is recorded as **two
modes of Transcribe** (Atelier `docs/10_transcribe.md`, currently a placeholder scoped
only to binding-ingestion — to be extended to cover PORT):

| Source | Nature | Mode | Mechanism |
|--------|--------|------|-----------|
| **Renesas FSP** | imperative vendor glue; the value is the domain map, not the code | **PORT** | Agent-driven Clef-native reimplementation guided by the keep/reimplement/discard taxonomy above. C/C++ becomes *reference*, not a dependency. |
| **USB device stack** / **LVGL** (display-equipped targets) | large, mature, well-factored C library | **BIND** | Farscape membrane → matched (Clef binding + Alex lowering witness) pairs. C remains behind the boundary. |

The criterion is the *shape and value* of the source. FSP's value is its domain
decomposition and datasheet-aligned vocabulary; its code is imperative scaffolding we
do not want as a dependency, so we PORT. LVGL is a large, well-factored body of C whose
value *is* the code; reimplementing it would be wasteful and lossy, so we BIND and keep
it behind the membrane. LVGL is the canonical BIND exemplar for *display-equipped* MCU
targets (e.g. the YoshiPi/Keystation demo). The RA6M5 credential device itself is
**headless** — status LEDs on-board, with a phone serving as its display surface over USB
(see [D-02-Mobile-Companion](../../Demo/D-02-Mobile-Companion.md)) — so its own BIND
candidates are libraries like a USB device stack, not a GUI. The criterion is
device-independent; a clean BIND story is shared MCU infrastructure either way.

### The Farscape boundary, as implemented (post-dates the spec)

Grounded in the current `Farscape/src`, not the older spec docs:

- **`FnPtr` is gone from generated code.** Function-pointer parameters are marshaled by
  the membrane via one of three mechanisms — **dlsym symbol resolution**,
  **listener-struct builders**, or a **GCHandle / `Marshal.GetFunctionPointerForDelegate`
  trampoline** — all landing at `nativeint` on the ABI.
- The membrane classifies the C idiom as **`CallbackParam` + `UserDataParam`**, and the
  boundary status is **marked into the compute graph** via `[<FidelityExtern>]` PSG
  metadata, resolved at lowering as an **`ExternCall`**.
- **Maturing direction:** the GCHandle/Marshal trampoline is the legacy .NET path; it is
  superseded by **flat-closure marshaling** — the closure's `code_ptr` becomes the C
  callback, the captured environment rides the detected `void*` userdata slot, and
  lifetime is arena/region-scoped (no GCHandle pinning). LVGL's `(cb, user_data)` idiom
  maps directly onto the existing `CallbackParam` / `UserDataParam` detection.

### Farscape's revised Renesas role

Farscape does **not** parse FSP headers for the RA6M5 track. Its Renesas role is reduced
to one job that produces a genuinely reusable substrate: **ingest the Renesas SVD /
CMSIS-Device description** to emit register address and bitfield constants. Those
constants are the per-device layer beneath the unified CMSIS register HAL — the same
data the ST samples currently hand-author. The PORT work (clock/PLL, pin mux, config
sequences, vector wiring) is then authored *over* those constants in Clef.

## Port Work Breakdown

The work is no longer "Farscape parses FSP and emits a binding stack." It is an
agent-driven PORT, layer by layer, with Farscape contributing only the register-constant
substrate. Each layer is produced separately.

### A. Acquire the register substrate (Farscape SVD ingestion)

Ingest the Renesas SVD / CMSIS-Device for the RA6M5 to produce register address and
bitfield constants:

- peripheral base addresses and register offsets
- bitfield positions and masks
- enumerated field values where the SVD provides them
- reset/default values

These constants are board-neutral per-device data. They feed the family branch and the
unified CMSIS register HAL; they are **not** wrappers over FSP entry points. (Whether
this is fully Farscape-driven SVD ingestion or partly hand-authored is an open question —
see below.)

### B. Mine the FSP domain map (reference only)

Read the relevant FSP/BSP material as a *map*, not a build input:

- which peripherals decompose into which modules
- the catalogue of config knobs that actually matter per peripheral
- the lifecycle stages (configure → enable → operate → teardown)
- the datasheet-aligned vocabulary worth preserving

Nothing here is parsed into generated code. This is the **KEEP** bucket sourcing step.

### C. Reimplement the family branch in Clef

Author the RA branch as Clef-native register operations over the §A constants:

- IOPORT pin/port access
- ADC configuration, channel selection, scan triggering
- timer/delay primitives
- clock/PLL/CGC bring-up
- interrupt/event primitives

This is `Ptr.read` / `Ptr.write` over typed register addresses with type-safe enums —
the ST standing-art pattern extended to RA. This is the **REIMPLEMENT** bucket.

### D. Author the board leaf

Bind the generic branch operations to the concrete EK-RA6M5 board:

- actual pin numbers (ported from PFS settings into Clef data tables)
- actual ADC channels
- actual mux select lines
- actual board endpoints

The board leaf consumes the branch; it does not re-derive register layout.

### E. Surface the sample pipeline cleanly

The quad-sample front end is a workload-oriented helper, not a generic GPIO
abstraction. The board leaf exposes the raw control points; the workload adapter
combines them into:

- mux selection
- ADC scan sequence
- value normalization
- timing/capture metadata

In the reactive runtime, the ADC completion IRQ surfaces as an `Observable` source and
the normalized sample value derives via `Incremental` — no callback registration, no
FSP ISR shim.

### F. Keep the output auditable

The ported register modules should remain easy to inspect. That means:

- no hidden board inference
- no guessing pin maps
- no collapsing board identity into the family branch
- no compatibility aliases that obscure the leaf package
- register accesses that read like the datasheet (named bases, named offsets, typed
  enums), so an engineer with the reference manual can verify them by eye

## Proposed Shape

A reasonable package shape is:

- `Fidelity.Platform.Contracts`
- `Fidelity.Platform.MCU.Renesas.RA6M5`
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5`
- `Fidelity.Platform.MCU.Renesas.RA6M5.EK_RA6M5.SamplePipeline`

The exact namespace names can change; the structural split stays the same.

For canonical naming and Pilot source-to-namespace mapping, see
[PH2-08-Naming-Conventions-and-Pilot-Map.md](./PH2-08-Naming-Conventions-and-Pilot-Map.md).

## Taxonomy Rule

If a binding needs board-specific pin knowledge, it belongs in the leaf.

If a binding can be shared across RA boards, it belongs in the branch.

If a binding only exists to make the workload easier to express, it belongs in the
adapter layer.

That rule is the main defense against taxonomy drift, and it is mode-independent: it
holds whether a layer is PORTed (FSP domain → Clef) or BOUND (LVGL → membrane).

## Open Questions

These are carried deliberately and resolved as the retooling lands; do not assume them
settled.

1. **Bind/port seam.** The exact line between the unified CMSIS register HAL (generic
   peripheral ops?) and the per-vendor RA branch (clock/pinmux/power only?). This
   determines how thin the branch is and how much is genuinely shared across all
   Cortex-M.
2. **Farscape SVD vs hand-authored constants.** Whether the §A register substrate is
   fully Farscape-driven SVD/CMSIS-Device ingestion, or partly hand-authored as the ST
   samples are today. The reusability case favors SVD ingestion, but the volume and
   verification cost are not yet measured.
3. **Transcribe PORT-as-mode.** Confirm that PORT lands as a first-class *mode* of
   Transcribe alongside BIND, sharing Farscape's analysis and the design-time decision
   UX — rather than living as an ad-hoc agent workflow outside the Transcribe surface.

## References

- [PH2-00-Embedded-Strategy.md](../PH2-00-Embedded-Strategy.md) — the corrected
  Clef-native CMSIS HAL + reactive-unikernel direction this taxonomy sits inside.
- [PH2-03-Binding-Surface.md](./PH2-03-Binding-Surface.md) — the Clef-native register
  surface this taxonomy produces.
- [PH2-08-Naming-Conventions-and-Pilot-Map.md](./PH2-08-Naming-Conventions-and-Pilot-Map.md) —
  the hand-authored namespace taxonomy (the KEEP bucket made concrete).
- Standing art: `samples/embedded/stm32l5-blinky/STM32L5.fs`,
  `samples/embedded/stm32l5-uart/STM32L5.UART.fs`.
- `Architecture_Canonical.md`, `Quotation_Based_Memory_Architecture.md` —
  quotation-based memory + direct register emission, `Platform.Bindings` by
  `(OS, Arch, EntryPoint)`.
- Atelier `docs/10_transcribe.md` — Transcribe (bind mode today; port mode to be added).
