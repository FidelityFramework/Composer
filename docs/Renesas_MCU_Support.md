# Renesas RA6M5 Bare-Metal Backend Support

This document describes the Composer backend path for the Renesas EK-RA6M5 development board. The goal is a host-native, command-line workflow that compiles Clef workloads to firmware without relying on vendor IDEs or generated project files.

The current direction is to treat EK-RA6M5 as the primary embedded target because its hardware root of trust gives us a stronger production story than the older STM32L5 demo path. The hardware-access strategy is the same one the ST samples already use: a hand-written, Clef-native register layer that lowers directly to memory-mapped registers. Renesas FSP is *reference material* for the RA family's silicon truths — it is **not** a binding target and is **not** linked into the image.

## Overview

The supported flow is:

1. Lower Clef source to LLVM IR.
2. Compile and link with host-installed `clang` and `ld.lld`.
3. Provide a small C startup layer for reset and vector-table setup.
4. Drive the hardware through a **Clef-native register layer** (the unified CMSIS HAL plus per-device register definitions), mirroring the STM32L5 samples — no vendor driver objects in the link.
5. Run application logic on the **reactive Clef unikernel** (Observable / Incremental); an RTOS is an *optional* bootstrap bridge, not the architecture.
6. Flash the final image with a lightweight programming tool.

This keeps the backend aligned with Composer's preference for reproducible, scriptable toolchains, and with the canonical direction of *direct register emission* rather than linking against a vendor HAL.

## Target

The EK-RA6M5 uses an Arm Cortex-M33 core with Armv8-M Mainline support. For bare-metal builds, the backend targets:

- `thumbv8m.main-none-eabi`
- `-mcpu=cortex-m33`
- `-mfpu=fpv5-sp-d16`
- `-mfloat-abi=hard`
- `-ffreestanding`
- `-nostdlib`

## Hardware Direction

The EK-RA6M5 is better aligned with the production workload plan than the STM32L5 demo board:

- It provides a stronger root-of-trust story through the RA family security features (HUK, SCE9, TrustZone-M).
- It gives us a path toward device-bound workloads rather than just a temporary demo target.

The hardware-access layer is **Clef-native, not a vendor binding**. It is the same pattern already proven in the ST samples:

- `samples/embedded/stm32l5-blinky/STM32L5.fs` — hand-written Clef defining `RCC_BASE` / `GPIOx_BASE` and register offsets (`MODER` / `OTYPER` / `BSRR` / `ODR`), implementing `enableClock` / `configureMode` / `setPin` via `Ptr.read<uint32>` / `Ptr.write` (`Alloy.Memory`) with type-safe enums.
- `samples/embedded/stm32l5-uart/STM32L5.UART.fs` — the same line extended to a second peripheral, confirming this is a reusable pattern rather than a one-off.

The RA6M5 port reuses this verbatim; only the per-device addresses, bitfields, and configuration sequences are new. `Architecture_Canonical.md` (via `Quotation_Based_Memory_Architecture.md`) frames hardware access as quoted memory constraints plus direct register emission — `<@ { Region = Peripheral; Access = WriteOnly; Volatile } @>` — with `Platform.Bindings` registered by `(OS, Arch, EntryPoint)`. That is the canonical direction: direct register emission, not linking vendor driver objects.

### Layered hardware stack

```
QuantumCredential application (Clef, reactive)
        │
Vendor port library (Clef)        Fidelity.Platform.MCU.Renesas.RA6M5
   (silicon truths: clock/PLL/CGC, pin mux, peripheral config sequences, vector wiring)
        │  uses
Unified CMSIS HAL (Clef-native)   CMSIS-Core: NVIC / SysTick / SCB  +  register-poke primitives
   (shared across all ARM Cortex-M; per-device register definitions beneath, derivable from each vendor SVD)
        │  lowers via
Quotation-based memory → direct register emission
   (Ptr.read / Ptr.write ; quoted Region/Access/Volatile constraints)
        │
Memory-mapped registers → single-core silicon
```

The **unified CMSIS HAL is a Clef-native register layer, not a binding to ARM's CMSIS C headers.** CMSIS-*Core* (NVIC / SysTick / SCB) is genuinely common across all Cortex-M; the CMSIS-*Device* register addresses and bitfields differ per chip and are the reusable per-device substrate, derivable from each vendor's SVD. The per-vendor *port library* stays thin: once concurrency is native and the register layer is shared, the per-vendor delta is just the silicon truths (clock/PLL tree, pin mux, peripheral config sequences, interrupt vector wiring).

### FSP as roadmap, not dependency

FSP is an imperative driver framework layered over CMSIS. We mine it as a *domain map* and then replace it; we do not bind to it or link it. The playbook splits cleanly into three buckets:

| Bucket | What | Examples |
|--------|------|----------|
| **KEEP** (the *what* + decomposition) | Per-peripheral module boundaries; the catalogue of config knobs that matter; lifecycle stages; datasheet-aligned vocabulary | One module per peripheral (IOPort, ADC, GPT, SCI-UART, …); scan-group / transfer abstractions; the `Fidelity.Platform.MCU.Renesas.RA…` namespace taxonomy (hand-authored, not Farscape-from-FSP) |
| **REIMPLEMENT** (silicon truths, over the Clef CMSIS HAL) | Clock/PLL/CGC bring-up; pin mux (PFS → Clef data tables); per-peripheral register config/enable sequences; NVIC/vector setup dispatched into Clef concurrency | The register-poking FSP performs, re-expressed as `Ptr.read` / `Ptr.write` |
| **DISCARD** (imperative mechanisms) | The `*_api_t` vtable / `_instance_t` / `_ctrl_t` / `_cfg_t` quartet; configurator codegen; RTOS glue; global mutable ctrl handles; error-code+out-param style; manual init ordering; callback function pointers | Replaced by Clef-native concurrency and the reactive runtime |

The guiding principle is **keep the map, redraw the roads**: preserve FSP's decomposition and vocabulary, re-express every *operation* in Clef idioms. Farscape's revised Renesas role is to ingest the Renesas **SVD / CMSIS-Device** for register address and bitfield constants — *not* FSP headers. (Library bindings that *are* worth taking as-is, such as LVGL and the touch controller for the display demo, go through Farscape's BIND mode; that is a separate track from the FSP **PORT** described here.)

## Memory Layout

The linker script defines the device memory map and the standard bare-metal sections. This is CMSIS/bare-metal-neutral and carries over unchanged.

```ld
/* Memory definitions for Renesas R7FA6M5BH (EK-RA6M5) */
MEMORY
{
    FLASH (rx)  : ORIGIN = 0x00000000, LENGTH = 2048K
    RAM   (rwx) : ORIGIN = 0x20000000, LENGTH = 512K
}

__stack_size = 0x2000;
_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
    .text :
    {
        KEEP(*(.isr_vector))
        *(.text*)
        *(.rodata*)
        _etext = .;
    } > FLASH

    _sidata = LOADADDR(.data);

    .data : AT(_etext)
    {
        _sdata = .;
        *(.data*)
        _edata = .;
    } > RAM

    .bss :
    {
        _sbss = .;
        *(.bss*)
        *(COMMON)
        _ebss = .;
    } > RAM
}
```

## Startup Code

The reset handler copies initialized data into RAM, clears `.bss`, and then enters `main()`. This minimal C startup is bare-metal-neutral and is kept as-is — the only change from earlier drafts is that the vector table no longer references RTOS port symbols.

The system-exception slots (SVCall, PendSV, SysTick) are wired to plain Clef-native handlers rather than to a FreeRTOS port. SysTick drives the unikernel's stabilization tick; SVCall and PendSV are reserved for the scheduler if it ever needs them, and default to a safe no-op until then.

```c
#include <stdint.h>

extern uint32_t _estack;
extern uint32_t _sdata, _edata, _sbss, _ebss, _sidata;

extern int main(void);

/* Clef-native system-exception handlers (no RTOS port symbols).
 * SysTick advances the unikernel stabilization tick; SVCall/PendSV
 * are reserved for the cooperative scheduler and default to no-ops. */
extern void SysTick_Handler(void);
extern void SVC_Handler(void);
extern void PendSV_Handler(void);

void Reset_Handler(void) {
    uint32_t *src = &_sidata;
    uint32_t *dst = &_sdata;

    while (dst < &_edata) {
        *dst++ = *src++;
    }

    dst = &_sbss;
    while (dst < &_ebss) {
        *dst++ = 0;
    }

    main();

    while (1) {
    }
}

__attribute__((section(".isr_vector")))
void (*const vector_table[])(void) = {
    (void (*)(void))(&_estack),
    Reset_Handler,
    0, 0, 0, 0, 0, 0, 0, 0,
    SVC_Handler,
    0, 0,
    PendSV_Handler,
    SysTick_Handler,
};
```

Peripheral interrupt vectors beyond the system block are wired to top-level, captureless Clef handlers (see *Reactive Runtime*, below). Hardware vectors carry no `void*` userdata slot, so a peripheral ISR cannot use the closure-via-userdata bridge that library callbacks rely on; the handler instead reaches peripheral state through the register HAL. The exact ISR → `Observable` boundary is an open design question (see below).

## Reactive Runtime (replaces RTOS + callbacks)

Concurrency is native to Clef and lowered to a single-core unikernel that owns the core. The model:

- **Peripheral IRQ → `Observable`** (an opaque, push event source) instead of a vendor ISR callback.
- **Controller / credential state and derived outputs → `Incremental`** (demand-driven, cutoff-bounded, self-adjusting graph).
- **Observable → Incremental fusion** is normatively specified: the PSG observer edge is rewritten into the incremental node's staleness edge — no callback indirection, no heap bridge, no materialized subscription.
- **`Async` → LLVM coroutine intrinsics** (compile-time state machines, no runtime library) — freestanding/unikernel-perfect.
- **Subscriptions are region-scoped** — deterministic teardown with the arena; no GC, `IDisposable`, or finalizers.
- The **unikernel scheduler is the Incremental stabilization loop**, driven by SysTick.

This reactive/concurrency model is richly specified but largely unimplemented as CCS intrinsics today (`NTUKind` registers only `NTUlazy` / `NTUseq`; `IntrinsicModule` registers `Arena` only; Alex has no actor/async/incremental witnesses). The MCU work is the **forcing function** that first stands these up, and single-core is the ideal substrate — coroutines need no runtime and the Incremental loop *is* the scheduler.

### RTOS as an optional bridge

An RTOS such as FreeRTOS or Zephyr remains available as a *bootstrap bridge* if it materially reduces initial integration risk — task scheduling during bring-up, interrupt-friendly coordination, a simpler first workload partitioning. It is not part of the headline architecture: the long-term path is the reactive unikernel plus the Clef-native register layer. If an RTOS is used during bring-up, its system-exception handlers (the `vPort*` / `xPort*` symbols) are linked at that time and supplied to the vector table by the bridge layer rather than baked into the in-tree startup.

## Open Design Questions

These are carried explicitly and resolved as the retooling lands; do not assume answers here.

1. **ISR / interrupt-vector boundary.** Hardware vectors carry no `void*` userdata slot, so the closure-via-userdata bridge used for library callbacks does not transfer to peripheral ISRs. The ISR → `Observable` source likely needs a captureless top-level handler bound directly to the vector-table slot, reaching peripheral state via the register HAL rather than a captured environment. Currently unspecified.
2. **Secure world (SCE9 / HUK / TrustZone-M).** The HUK is reachable only via the SCE9 private bus, so pure register-poking may be insufficient. This likely needs a thin secure-world call surface — the one place the Clef-native register layer may not cover directly, and central to the device's sovereignty premise.
3. **Bind/port seam.** The exact line between the unified CMSIS HAL (generic peripheral ops?) and the per-vendor port library (clock / pin mux / power only?) determines how thin the vendor library is.
4. **Farscape SVD ingestion.** Whether Farscape emits the register address/bitfield constants from the Renesas SVD / CMSIS-Device, or whether those constants are hand-authored.
5. **Unikernel scheduler.** Cooperative event loop (Incremental stabilization) vs any preemption. Cooperative is the likely fit for a single-core device.

## Programming Options

The generated firmware can be flashed with either OpenOCD or Renesas programming tools.

### OpenOCD

```bash
openocd -f interface/cmsis-dap.cfg \
        -f target/renesas_ra6m5.cfg \
        -c "program build/firmware.hex verify reset exit"
```

### `rfp-cli`

```bash
rfp-cli -device ra -port /dev/ttyUSB0 -file build/firmware.hex
```

## Command Driver

The backend driver wraps compilation, linking, and flashing in one script. The link set is the Clef application plus the minimal C startup and the Clef-native platform support — no vendor driver objects.

```bash
#!/usr/bin/env bash
set -e

TARGET="thumbv8m.main-none-eabi"
CPU="cortex-m33"
CFLAGS="-target $TARGET -mcpu=$CPU -ffreestanding -nostdlib -mthumb -O2 -Wall"
INCLUDES="-I./platform"

BUILD_DIR="./build"
OUTPUT_HEX="$BUILD_DIR/firmware.hex"
OUTPUT_ELF="$BUILD_DIR/firmware.elf"

compile_artifacts() {
    echo "==> Lowering Clef source through Composer"
    mkdir -p "$BUILD_DIR"

    # Application logic plus the Clef-native register layer (RA6M5.fs, the
    # unified CMSIS HAL) are all compiled by Composer; no FSP drivers are linked.
    /path/to/Composer compile ../src/MCU/Program.fidproj -output-kind llvm-ir -o "$BUILD_DIR/clef_output.ll"
    clang $CFLAGS -c "$BUILD_DIR/clef_output.ll" -o "$BUILD_DIR/application.o"

    echo "==> Compiling startup and system support"
    clang $CFLAGS $INCLUDES -c platform/startup_ra6m5.c -o "$BUILD_DIR/startup.o"
    clang $CFLAGS $INCLUDES -c platform/system_ra6m5.c -o "$BUILD_DIR/system.o"
}

link_artifacts() {
    echo "==> Linking firmware"

    ld.lld -T platform/ek_ra6m5.ld \
        "$BUILD_DIR/startup.o" \
        "$BUILD_DIR/system.o" \
        "$BUILD_DIR/application.o" \
        -o "$OUTPUT_ELF"

    llvm-objcopy -O ihex "$OUTPUT_ELF" "$OUTPUT_HEX"
    echo "==> Wrote $OUTPUT_HEX"
}

flash_hardware() {
    echo "==> Flashing board"

    openocd -f interface/cmsis-dap.cfg \
        -f target/renesas_ra6m5.cfg \
        -c "program $OUTPUT_HEX verify reset exit"
}

case "$1" in
    --compile)
        compile_artifacts
        ;;
    --link)
        link_artifacts
        ;;
    --flash)
        flash_hardware
        ;;
    --all)
        compile_artifacts
        link_artifacts
        flash_hardware
        ;;
    *)
        echo "Composer Target Option Required: --compile | --link | --flash | --all"
        exit 1
        ;;
esac
```

If an RTOS bridge is used during bring-up (see *RTOS as an optional bridge*), its sources and include paths are added to `INCLUDES` and the link set at that time; the steady-state build above does not include them.

## Notes

- The startup code is intentionally small so it can stay in-tree and be audited easily, and it carries no RTOS coupling — the system-exception slots point at Clef-native handlers.
- Hardware access is a hand-written Clef-native register layer mirroring `samples/embedded/stm32l5-blinky/STM32L5.fs`; FSP is ported as reference, never linked.
- The workflow assumes host-native LLVM tooling is available.
- The flashing step can be adjusted depending on whether the board is accessed through a debug probe or Renesas programming mode.

## References

- Standing art: `samples/embedded/stm32l5-blinky/STM32L5.fs`, `samples/embedded/stm32l5-uart/STM32L5.UART.fs`, and the shared startup under `samples/embedded/common/`.
- `Architecture_Canonical.md`, `Quotation_Based_Memory_Architecture.md` — quotation-based memory and `Platform.Bindings`.
- QuantumCredential embedded track: `PH2-00-Embedded-Strategy.md`, and the RA6M5 series (`PH2-01` … `PH2-09`) for hardware platform, the Clef-native register surface, bootstrap options, the port taxonomy, and the pilot blueprint.
