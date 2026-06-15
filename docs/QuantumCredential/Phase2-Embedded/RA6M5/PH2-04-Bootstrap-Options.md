# PH2-04: RA6M5 Bootstrap Options

## Two Valid Paths

There are two reasonable ways to start bringing workloads onto the EK-RA6M5. The axis has not changed — **direct bring-up vs. a borrowed RTOS** — but the *direct* pole is no longer "Renesas FSP + generated bindings." It is a **Clef-native register layer driving a reactive unikernel**: peripheral access via direct register emission (the standing-art ST pattern), and concurrency expressed natively in Clef rather than handed to an RTOS scheduler.

This is the headline path. FreeRTOS / Zephyr remain available as an optional bootstrap bridge, but they are no longer one of two co-equal architectural destinations — they are a ramp.

### Option A: Direct Bring-Up (Clef-Native Register Layer + Reactive Unikernel)

This is the architectural destination.

Talk to the board through the **Clef-native CMSIS register layer** — hand-written register bases, offsets, and bitfields, accessed via `Ptr.read` / `Ptr.write` under quotation-based memory constraints — exactly as the STM32L5 standing art already does (`samples/embedded/stm32l5-blinky/STM32L5.fs`, `samples/embedded/stm32l5-uart/STM32L5.UART.fs`). The RA6M5 silicon truths (clock/PLL bring-up, pin mux, peripheral config sequences, vector wiring) are *ported* from FSP's domain knowledge, not linked against FSP drivers. See [PH2-03-Binding-Surface.md](./PH2-03-Binding-Surface.md) for the register surface and [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) for the keep/reimplement/discard port taxonomy.

Concurrency is **native to Clef** and lowers to a single-core unikernel — the scheduler is the Incremental stabilization loop (see [The Reactive Runtime](#the-reactive-runtime), below), not a borrowed RTOS kernel.

Benefits:

- fewer moving parts; no RTOS kernel or driver framework in the critical path
- clearer security story — every register touch is explicit and auditable, with no opaque vendor driver between the workload and the silicon
- less translation between the workload and the board
- this *is* the unikernel end state, not a step toward it

Costs:

- more explicit work up front (clock/PLL, pin mux, and vector wiring are authored, not configured)
- the reactive runtime intrinsics it depends on are largely unimplemented today (see [Forcing-Function Reality](#forcing-function-reality)) — this track is what stands them up

### Option B: RTOS Bootstrap (FreeRTOS / Zephyr)

This is the pragmatic early-bring-up bridge.

Use FreeRTOS (or the RA6M5's first-class Zephyr support) to get tasking, coordination, and peripheral timing under control before the reactive runtime is fully stood up. The RA6M5 has mature support for both, which makes this a low-friction way to reach a testable workload quickly.

Benefits:

- quicker first workload partitioning
- easier coordination of multiple concurrent activities while the native concurrency intrinsics are still maturing
- familiar embedded development model

Costs:

- more runtime surface area
- more abstraction between the workload and the hardware
- it is a bridge, not the destination — work done against the RTOS scheduler must be unwound to reach the reactive unikernel
- the current C startup's vector table references FreeRTOS port symbols (`vPortSVCHandler`, etc.) — a **separable** coupling (see [Minimal C Startup](#minimal-c-startup-salvageable-as-is))

## The Reactive Runtime

This document owns the runtime detail for the direct path. Concurrency on the RA6M5 is expressed in Clef and lowered to a single-core unikernel — there is no RTOS in the critical path and no callback indirection.

### IRQ → Observable → fused → Incremental → Effects

The end-to-end model:

- **Peripheral IRQ → `Observable`.** Each interrupt source is modeled as an opaque, push-style event source — not an ISR callback registered with a driver framework.
- **Controller / credential state + derived outputs → `Incremental`.** Device state and everything computed from it live in a demand-driven, cutoff-bounded self-adjusting graph. Outputs are recomputed only when an input that actually feeds them goes stale.
- **Observable → Incremental fusion (normative).** The compiler rewrites the PSG observer edge into the Incremental node's staleness edge. The consequence is concrete: **no callback indirection, no heap bridge, no materialized subscription object.** The "register a callback, stash a handle, dispatch later" machinery that FSP and an RTOS would carry simply does not exist — the event source is wired directly into the dependency graph at compile time.
- **Effects** (register writes, peripheral re-arming, credential emission) are the leaves of the Incremental graph — they fire as a consequence of stabilization, reaching the hardware through the Clef-native register layer.

### Async as coroutines

`Async` lowers to **LLVM coroutine intrinsics** — compile-time state machines with no runtime library. This is freestanding-perfect: there is no async runtime to link, no thread pool, no allocator dependency. Long-running sequences (e.g. a multi-stage secure handshake or a paced ADC scan campaign) become explicit state machines the compiler emits inline.

### Subscriptions are region-scoped

Subscriptions and any backing state are **region/arena-scoped**. Teardown is deterministic and tied to the arena lifetime — there is no GC, no `IDisposable`, no finalizer. When a region closes, every subscription rooted in it is gone, by construction.

### The scheduler is the stabilization loop

The unikernel scheduler **is** the Incremental stabilization loop. There is no separate task scheduler: the device wakes on an interrupt (an `Observable` push), the staleness propagates through the Incremental graph, effects at the leaves fire, and the system returns to quiescence. On a single core this is the natural fit — no SMP, no preemptive context switching to reason about, and the "scheduler" is just the demand-driven recomputation the Incremental model already specifies.

## Minimal C Startup (Salvageable As-Is)

The bare-metal bring-up scaffolding is CMSIS/architecture-neutral and survives the retooling unchanged:

- **Startup:** `Reset_Handler`, the vector table, and `.data` / `.bss` initialization.
- **Linker memory map:** FLASH `0x0` 2048K, RAM `0x20000000` 512K.
- **Toolchain flags:** `thumbv8m.main-none-eabi`, `-mcpu=cortex-m33`, `-mfpu=fpv5-sp-d16`, `-ffreestanding -nostdlib`.

The one caveat: the **current vector table references FreeRTOS port symbols** (`vPortSVCHandler`, `xPortPendSVHandler`, `xPortSysTickHandler`). That is a *separable* FreeRTOS coupling, not a dependency of the startup scaffolding itself. On the direct path those slots are rebound to Clef handlers (see Open Question 1); on the RTOS bootstrap path they stay as-is.

## Recommendation

Treat an RTOS as a bootstrap helper, not as the architectural destination.

If the Clef-native register layer and the reactive runtime are far enough along to support the sample front end and the device security functions, prefer the direct path — it *is* the unikernel end state, so work invested there is not thrown away.

If the team needs a temporary scheduler and a simpler integration ramp while the reactive intrinsics are still being stood up, use FreeRTOS (or Zephyr) narrowly and plan the exit early.

## Decision Rule

Choose an RTOS bootstrap only if it reduces time to a real, testable workload on the board *and* the reactive runtime is not yet ready to carry that workload.

Choose direct bring-up if the register layer and reactive runtime are ready enough — the extra RTOS layer would only add surface area to unwind on the way to the unikernel we already know we want.

## Forcing-Function Reality

Be honest about maturity. The reactive/concurrency model this document leans on (`Observable`, `Incremental`, Observable→Incremental fusion, `Async`-as-coroutines, region-scoped subscriptions) is **richly specified but largely unimplemented in CCS today**: `NTUKind` registers only `NTUlazy` / `NTUseq`; `IntrinsicModule` registers `Arena` only; Alex has no actor/async/incremental witnesses.

This RA6M5 track is the **forcing function** that first stands these intrinsics up — and single-core is the ideal place to do it: coroutines need no runtime, and the Incremental loop is the scheduler. Until they land, Option B is the honest bridge; the direct path matures alongside the intrinsics. See [PH2-00-Embedded-Strategy.md](../PH2-00-Embedded-Strategy.md) and the framework reactive/concurrency specs.

## Open Questions

These are carried deliberately and not resolved here.

1. **ISR / interrupt-vector boundary.** Hardware vectors carry **no `void*` userdata slot**, so the closure-via-userdata bridge used for C library callbacks (the LVGL `(cb, user_data)` idiom) does **not** transfer to the IRQ → `Observable` edge. The likely shape is a **captureless top-level Clef handler bound directly to the vector-table slot** (a bare `code_ptr`, no captured environment), with peripheral state reached through the register HAL rather than a captured env. The exact mechanism that turns a vector slot into an `Observable` push source is currently unspecified.

2. **Unikernel scheduler — cooperative vs. preemptive.** The reactive model points at a **cooperative** event loop (Incremental stabilization driven by interrupt wakeups), which is the natural single-core fit and avoids preemptive context-switch reasoning. Whether any preemption is ever warranted (e.g. a hard-real-time deadline that cannot wait for stabilization to quiesce) is left open. Cooperative is the working assumption.

## References

- Standing art: `samples/embedded/stm32l5-blinky/STM32L5.fs`, `samples/embedded/stm32l5-uart/STM32L5.UART.fs` — hand-written Clef register layer (`Ptr.read` / `Ptr.write`, type-safe enums).
- [PH2-00-Embedded-Strategy.md](../PH2-00-Embedded-Strategy.md) — platform strategy; FreeRTOS/Zephyr as optional bridge.
- [PH2-03-Binding-Surface.md](./PH2-03-Binding-Surface.md) — the Clef-native CMSIS register surface.
- [PH2-05-Taxonomy-and-Farscape-Plan.md](./PH2-05-Taxonomy-and-Farscape-Plan.md) — keep/reimplement/discard port taxonomy.
- `Architecture_Canonical.md`, `Quotation_Based_Memory_Architecture.md` — quotation-based memory + direct register emission, `Platform.Bindings` by `(OS, Arch, EntryPoint)`.
- Framework reactive/concurrency specs — `Observable` / `Incremental` / Signals, `Async`-as-coroutines, region/arena lifetime, BAREWire.
