# RA6M5 Embedded Track

This directory is the active embedded planning path for QuantumCredential on the Renesas EK-RA6M5 evaluation kit.

It replaces the old STM32L5-centered assumption with a board-specific plan built around:

- the RA6M5 hardware root of trust
- the quad-sample analog front end
- a **Clef-native CMSIS register surface** — hand-written register definitions and config sequences that lower to direct register emission, mirroring the STM32L5 standing art (see below)
- a **reactive unikernel** that owns the single core, with any RTOS available only as an optional bring-up bridge

## Direction (corrected)

The earlier framing of this track assumed Composer would **bind to Renesas FSP** — Farscape parsing FSP/BSP headers and emitting wrappers over `R_IOPORT_*` / `R_ADC_*` / `R_BSP_*`, with the ELF linked against FSP driver objects. That premise is retired.

**FSP is not a binding target.** It is an imperative driver framework layered over CMSIS — valuable as a *roadmap* for higher-level concerns (peripheral decomposition, the catalogue of config knobs, lifecycle vocabulary), but not as a dependency. The corrected direction:

- **Port, do not bind.** FSP's *domain knowledge* is reimplemented as clean, Clef-native code that lowers to **CMSIS** — i.e. to the memory-mapped register layer — exactly as the existing **STM32L5 work already does**.
- **Concurrency is expressed natively in Clef** (Observable / Incremental / Actor) and lowered to a **single-core unikernel**. An RTOS becomes an optional bridge, not the architecture.
- The reusable infrastructure this stands up — a Clef-native CMSIS HAL, the reactive runtime, the bind-vs-port interop split — benefits **every** Cortex-M target, not just the RA6M5 credential device.

### Layered architecture

```
QuantumCredential application (Clef, reactive)
        │
Vendor port library (Clef, thin)   Fidelity.Platform.MCU.Renesas.RA6M5
   (clock/PLL, pin mux, peripheral config sequences, vector wiring — the silicon truths)
        │  uses
Unified CMSIS HAL (Clef-native)    CMSIS-Core NVIC / SysTick / SCB + register-poke primitives
   (shared across all Cortex-M; per-device register defs beneath, derivable from each vendor SVD)
        │  lowers via
Quotation-based memory → direct register emission
        │
Memory-mapped registers → single-core silicon
```

The "Unified CMSIS HAL" is a **Clef-native register layer**, not a binding to ARM's CMSIS C headers. CMSIS-*Core* (NVIC/SysTick/SCB) is genuinely common across all Cortex-M; the CMSIS-*Device* register addresses and bitfields differ per chip and form the reusable per-device substrate beneath the HAL, derivable from each vendor's SVD.

The broader product story still matters:

- YoshiPi remains the proof that Composer can handle a normal Linux-shaped workload with a screen and ADC-driven I/O.
- RA6M5 is the proof that Composer can also do secure native embedded work on a production-oriented MCU.
- Any RTOS is a bridge, not the destination, so the embedded track stays aligned with a direct-hardware, reactive-unikernel end state.

## Documents

| Document | Purpose |
|----------|---------|
| [PH2-01-Strategy-Overview.md](./PH2-01-Strategy-Overview.md) | Overall embedded plan and sequencing |
| [PH2-02-Hardware-Platform.md](./PH2-02-Hardware-Platform.md) | Board capabilities and sample front end hardware |
| [PH2-03-Binding-Surface.md](./PH2-03-Binding-Surface.md) | The Clef-native CMSIS register surface the port must expose |
| [PH2-04-Bootstrap-Options.md](./PH2-04-Bootstrap-Options.md) | Reactive unikernel vs optional RTOS bring-up bridge |
| [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) | FSP-as-roadmap port taxonomy (keep / reimplement / discard) and Farscape's SVD-ingestion role |
| [PH2-06-Leaf-Package-Descriptor.md](./PH2-06-Leaf-Package-Descriptor.md) | Predictive shape of the EK-RA6M5 board register / pin descriptor |
| [PH2-07-Module-Binding-Map.md](./PH2-07-Module-Binding-Map.md) | Module-by-module map: peripheral, namespace, register modules, and adapters |
| [PH2-08-Naming-Conventions-and-Pilot-Map.md](./PH2-08-Naming-Conventions-and-Pilot-Map.md) | Canonical names and Pilot source-to-namespace mapping |
| [PH2-09-Pilot-TOML-Blueprint.md](./PH2-09-Pilot-TOML-Blueprint.md) | Draft Pilot project file for the EK-RA6M5 port pass |

## Bind vs Port

Interop splits by the *nature* of the source:

- **FSP → PORT.** Imperative vendor glue whose value is the domain map, not the code. Reimplemented as Clef-native modules guided by the keep / reimplement / discard taxonomy (PH2-05). C/C++ becomes *reference*, not a dependency.
- **Large C libraries → BIND.** Mature, well-factored C (e.g. a USB device stack; or LVGL on display-equipped targets) bound via the Farscape membrane (matched Clef binding + Alex lowering witness pairs); C remains behind the boundary. The RA6M5 itself is headless — status LEDs, with a phone as its display surface over USB (see [D-02-Mobile-Companion](../../Demo/D-02-Mobile-Companion.md)) — so its own bind surface is the host/USB link, not a GUI.

Farscape's revised Renesas role is to ingest the Renesas **SVD / CMSIS-Device** for register address and bitfield constants — **not** FSP headers.

## Working Assumptions

- The hardware-facing surface is **hand-authored Clef** register modules + config sequences that lower to direct register emission — the STM32L5 register-layer pattern, extended to the RA6M5.
- FSP is mined as a *roadmap* for peripheral decomposition, config knobs, and lifecycle vocabulary; its imperative scaffolding (vtable quartet, configurator codegen, RTOS glue, callbacks) is discarded.
- The minimal C startup, linker memory map, and toolchain flags are CMSIS/bare-metal-neutral and carry over as-is; the current FreeRTOS vector-table coupling is separable.
- STM32L5 remains the standing art this track mirrors, not a historical curiosity — `samples/embedded/stm32l5-blinky/STM32L5.fs` and `samples/embedded/stm32l5-uart/STM32L5.UART.fs` are the live reference pattern.

## Open Questions

These are carried unresolved into the detailed per-document work:

1. **ISR / interrupt-vector boundary.** Hardware vectors carry no `void*` userdata slot, so the LVGL-style closure-via-userdata bridge does not transfer. ISR → `Observable` likely needs a captureless top-level handler bound directly to the vector slot, with peripheral state reached via the register HAL. Currently unspecified.
2. **Secure world (SCE9 / HUK / TrustZone-M).** HUK is reachable only via the SCE9 private bus, so pure register-poking may be insufficient; this likely needs a thin secure-world call surface — central to the device's sovereignty premise.
3. **Bind/port seam.** The exact line between the unified CMSIS HAL and the per-vendor port library.
4. **Farscape SVD ingestion vs hand-authored constants.** Whether the register constants come from Farscape SVD ingestion or are hand-authored.
5. **Unikernel scheduler.** Cooperative (Incremental stabilization loop) vs preemptive — cooperative is the likely fit for a single core.
6. **Transcribe PORT-as-mode.** Confirming PORT lands as a mode of Transcribe alongside BIND.

## References

- Standing art: `samples/embedded/stm32l5-blinky/STM32L5.fs`, `samples/embedded/stm32l5-uart/STM32L5.UART.fs`.
- `Architecture_Canonical.md`, `Quotation_Based_Memory_Architecture.md` — quotation-based memory + direct register emission, with `Platform.Bindings` registered by `(OS, Arch, EntryPoint)`.
- `PH2-00-Embedded-Strategy.md` — the embedded strategy this track instantiates.
