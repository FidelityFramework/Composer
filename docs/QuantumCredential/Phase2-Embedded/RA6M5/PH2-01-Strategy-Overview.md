# PH2-01: RA6M5 Strategy Overview

## Purpose

This document defines the active embedded strategy for QuantumCredential on the Renesas EK-RA6M5. The active question is how to shape the RA6M5 workload pipeline on a **Clef-native** hardware-access stack — not how to bind to a vendor driver framework.

This is a correction of an earlier premise. The RA6M5 track was first sketched around binding to Renesas FSP — Farscape parsing FSP/BSP headers and the ELF linking against FSP driver objects. That premise is retired. FSP is an imperative driver framework layered over CMSIS; it is valuable as a *roadmap* for the higher-level concerns (what to configure, in what order, with which knobs), but it is **ported, not bound**. The corrected direction lowers to CMSIS — the memory-mapped register layer — exactly as the existing STM32L5 work already does.

## Why RA6M5

The RA6M5 is the preferred production target because it gives the project a stronger root-of-trust story than the earlier STM32L5 demo board.

- Hardware unique key support anchors device identity in silicon.
- TrustZone separates secure and non-secure workloads.
- Secure crypto hardware reduces the amount of sensitive software we need to carry.
- The board has enough headroom to support sample acquisition, crypto, and workload orchestration together.

## Architecture Posture

The RA6M5 path is a board-specific embedded platform built on Clef-native hardware access. The layering is:

1. **QuantumCredential application** — Clef, reactive. Peripheral events and credential state expressed as Observable/Incremental, not callbacks.
2. **Vendor port library** — Clef, thin. The RA6M5's *silicon truths* ported from FSP's domain knowledge: clock/PLL bring-up, pin mux, per-peripheral config/enable sequences, and interrupt-vector wiring. Lives under `Fidelity.Platform.MCU.Renesas.RA6M5`.
3. **Unified CMSIS HAL** — Clef-native register layer. CMSIS-Core (NVIC/SysTick/SCB) plus register-poke primitives shared across all Cortex-M, with per-device register definitions beneath (derivable from each vendor's SVD). This is a Clef register layer, **not** a binding to ARM's CMSIS C headers.
4. **Quotation-based memory / direct register emission** — `Ptr.read`/`Ptr.write` over quoted memory constraints (`<@ { Region = Peripheral; Access = WriteOnly; Volatile } @>`), with `Platform.Bindings` registered by `(OS, Arch, EntryPoint)`.
5. **Memory-mapped registers → single-core silicon.**

Composer lowers the workload to native code at every layer. The compilation path produces an ELF for the Cortex-M33 target with a minimal C startup and freestanding toolchain — no vendor driver objects in the link.

FreeRTOS (or Zephyr) stays an **optional bridge** for early bring-up only, never the headline architecture. The headline path is a reactive unikernel that owns the single core (see [PH2-04-Bootstrap-Options.md](./PH2-04-Bootstrap-Options.md)).

## Standing Art This Mirrors

The RA6M5 stack reuses an established pattern; it does not invent a parallel mechanism. The STM32L5 samples are hand-written Clef that define register bases/offsets and poke them directly — they are *not* bindings to ST's HAL:

- `samples/embedded/stm32l5-blinky/STM32L5.fs` — hand-written Clef defining `RCC_BASE`/`GPIOx_BASE` and register offsets (`MODER`/`OTYPER`/`BSRR`/`ODR`), implementing `enableClock`/`configureMode`/`setPin` via `Ptr.read<uint32>` / `Ptr.write` (Alloy.Memory) with type-safe enums.
- `samples/embedded/stm32l5-uart/STM32L5.UART.fs` — confirms this is an active, reusable line.

`Architecture_Canonical.md` and `Quotation_Based_Memory_Architecture.md` frame hardware access as quoted memory constraints + direct register emission. The RA6M5 port reuses this verbatim; only the per-device addresses, bitfields, and config sequences are new.

## FSP as Roadmap, Not Target

FSP is mined as a domain map, then replaced — "keep the map, redraw the roads":

- **Keep** — per-peripheral module decomposition, the catalogue of config knobs that matter, lifecycle stages, datasheet-aligned vocabulary, and the hand-authored `Fidelity.Platform.MCU.Renesas.RA…` namespace taxonomy.
- **Reimplement** over the Clef CMSIS HAL — clock/PLL/CGC bring-up, pin mux (PFS as Clef data), per-peripheral register config/enable sequences, and NVIC/vector setup dispatched into Clef concurrency.
- **Discard** — the `*_api_t` vtable / `_instance_t` / `_ctrl_t` / `_cfg_t` quartet, configurator codegen, RTOS glue, global mutable ctrl handles, the error-code+out-param style, manual init ordering, and callback function pointers.

The concrete keep/reimplement/discard breakdown and the agent-driven port workflow are in [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md). The Clef-native register surface this targets is in [PH2-03-Binding-Surface.md](./PH2-03-Binding-Surface.md). Farscape's role on this device is narrowed to ingesting the Renesas **SVD / CMSIS-Device** for register address and bitfield constants — not FSP headers.

A separate interop story applies where the source is a large, well-factored C library rather than vendor glue: such libraries (e.g. a USB device stack, or LVGL on display-equipped targets) are **bound** via the Farscape membrane → matched Clef binding + Alex lowering witness pairs, not ported. The RA6M5 itself is headless — status LEDs plus a phone acting as its display surface over USB (see [D-02-Mobile-Companion](../../Demo/D-02-Mobile-Companion.md)) — so its own bind surface is the host/USB link, not a GUI. Bind-vs-port is the device's two-sided interop posture.

## Reactive Runtime

Concurrency is expressed natively in Clef and lowered to a single-core unikernel:

- Peripheral IRQ → `Observable` (opaque push source) instead of a vendor ISR callback.
- Controller/credential state + derived outputs → `Incremental` (demand-driven, cutoff-bounded).
- Observable→Incremental fusion is normatively mandated: the observer edge becomes the incremental staleness edge — no callback indirection, no heap bridge, no materialized subscription.
- `Async` → LLVM coroutine intrinsics (compile-time state machines, no runtime library) — freestanding-perfect.
- Subscriptions are region-scoped (deterministic teardown with the arena; no GC/`IDisposable`).
- The unikernel scheduler **is** the Incremental stabilization loop.

This reactive model is what makes the RTOS optional: there is no scheduler to import once stabilization is the scheduler.

## This Matures MCU Infrastructure Across the Board

The RA6M5 work is not a one-off port. The Clef-native CMSIS HAL, the bind-vs-port interop split, and the reactive runtime are **shared MCU infrastructure** that benefits every Cortex-M target. The RA6M5 credential device is the first place they all land together.

It is also the **forcing function** for the concurrency model. The reactive/concurrency intrinsics (`Observable`, `Incremental`, `Actor`/Olivier-Prospero, `Async`-as-coroutines, Signals, coeffects, BAREWire) are richly specified but largely unimplemented in CCS today: `NTUKind` registers only `NTUlazy`/`NTUseq`, `IntrinsicModule` registers `Arena` only, and Alex has no actor/async/incremental witnesses. This MCU work is what first stands these intrinsics up — and single-core is the ideal substrate, because coroutines need no runtime and the Incremental loop is the scheduler.

## Planning Goal

The goal is to get real workloads onto the EK-RA6M5 with as little translation loss as possible.

That means:

- preserve the sample-pipeline semantics in software
- expose the device security features as first-class APIs
- keep the register layer explicit and auditable (Clef-native, no hidden vendor runtime)
- keep the design aligned with a clean hardware abstraction

## Immediate Questions

- Which portions of the workload need direct register access on day one?
- Which parts are shared CMSIS-Core / family-branch infrastructure versus board-leaf or workload-specific code?
- What is the exact seam between the unified CMSIS HAL and the per-vendor port library — how thin can the vendor layer be?

These are sequencing questions, not architecture questions. The architecture is the Clef-native register stack above. The remaining documents in this directory work through the answers; [PH2-03-Binding-Surface.md](./PH2-03-Binding-Surface.md) and [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) are the next steps.

## Open Questions

These are carried forward, not resolved here:

- **ISR / vector boundary.** Hardware vectors carry no `void*` userdata, so the LVGL-style closure-via-userdata bridge does not transfer to ISR → `Observable`. This likely needs a captureless top-level handler bound to the vector slot, with peripheral state reached via the register HAL. Currently unspecified.
- **Secure world (SCE9 / HUK / TrustZone-M).** The HUK is reachable only via the SCE9 private bus, so pure register-poking may be insufficient; this likely needs a thin secure-world call surface — central to the device's sovereignty premise.
- **Unikernel scheduler.** Cooperative (Incremental stabilization) versus preemptive; cooperative is the likely fit for a single core.
