# Renesas RA6M5 Bare-Metal Backend Support

This document describes the Composer backend path for the Renesas EK-RA6M5 development board. The goal is a host-native, command-line workflow that compiles Clef workloads to firmware without relying on vendor IDEs or generated project files.

The current direction is to treat EK-RA6M5 as the primary embedded target because its hardware root of trust gives us a stronger production story than the older STM32L5 demo path. Farscape is now capable of producing usable C and C++ bindings, so the main question is not whether a binding layer exists, but which surface we should target first.

## Overview

The supported flow is:

1. Lower Clef source to LLVM IR.
2. Compile and link with host-installed `clang` and `ld.lld`.
3. Provide a small C startup layer for reset and vector-table setup.
4. Generate bindings and descriptors for the chosen hardware surface, starting with Renesas FSP.
5. Optionally integrate FreeRTOS as a bootstrap helper if it shortens bring-up.
6. Flash the final image with a lightweight programming tool.

This keeps the backend aligned with Composer's preference for reproducible, scriptable toolchains.

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

- It provides a stronger root-of-trust story through the RA family security features.
- It gives us a path toward device-bound workloads rather than just a temporary demo target.
- It fits a binding-first workflow because Farscape can now generate usable bindings from C and C++ headers.

That means the first binding layer should likely target Renesas FSP headers and the board-specific support code around them, not a generic HAL abstraction copied from the STM32L5 story.

## Memory Layout

The linker script defines the device memory map and the standard bare-metal sections.

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

The reset handler copies initialized data into RAM, clears `.bss`, and then enters `main()`.

```c
#include <stdint.h>

extern uint32_t _estack;
extern uint32_t _sdata, _edata, _sbss, _ebss, _sidata;

extern int main(void);
extern void xPortSysTickHandler(void);
extern void vPortSVCHandler(void);
extern void xPortPendSVHandler(void);

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
    vPortSVCHandler,
    0, 0,
    xPortPendSVHandler,
    xPortSysTickHandler,
};
```

## FreeRTOS Integration

FreeRTOS is best treated here as a bootstrap helper, not as the final architecture.

If we need it to get early workloads running quickly, it can provide:

- task scheduling during bring-up
- interrupt-friendly coordination primitives
- a simpler path for first workload partitioning

That said, the longer-term direction remains a Clef unikernel and direct board-oriented bindings, so FreeRTOS should be used only when it materially reduces initial integration risk.

```text
third_party/FreeRTOS/
├── include/
└── portable/
    └── GCC/
        └── ARM_CM33_NTZ/
            └── non_secure/
                ├── port.c
                └── portasm.c
```

Even though the portable layer lives under a `GCC/` directory, Clang can assemble the standard GNU syntax used by the port files.

A typical application entry point looks like this:

```c
#include "FreeRTOS.h"
#include "task.h"

void vComposerGeneratedTask(void *pvParameters) {
    (void)pvParameters;

    while (1) {
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

int main(void) {
    system_clock_init();

    xTaskCreate(vComposerGeneratedTask, "ClefWorkload", 512, NULL, 1, NULL);
    vTaskStartScheduler();

    while (1) {
    }
}
```

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

The backend driver wraps compilation, linking, and flashing in one script.

```bash
#!/usr/bin/env bash
set -e

TARGET="thumbv8m.main-none-eabi"
CPU="cortex-m33"
CFLAGS="-target $TARGET -mcpu=$CPU -ffreestanding -nostdlib -mthumb -O2 -Wall"
INCLUDES="-I./platform -I./third_party/FreeRTOS/include -I./third_party/FreeRTOS/portable/GCC/ARM_CM33_NTZ/non_secure"

BUILD_DIR="./build"
OUTPUT_HEX="$BUILD_DIR/firmware.hex"
OUTPUT_ELF="$BUILD_DIR/firmware.elf"

compile_artifacts() {
    echo "==> Lowering Clef source through Composer"
    mkdir -p "$BUILD_DIR"

    /path/to/Composer compile ../src/MCU/Program.fidproj -output-kind llvm-ir -o "$BUILD_DIR/clef_output.ll"
    clang $CFLAGS -c "$BUILD_DIR/clef_output.ll" -o "$BUILD_DIR/application.o"

    echo "==> Compiling startup and system support"
    clang $CFLAGS $INCLUDES -c platform/startup_ra6m5.c -o "$BUILD_DIR/startup.o"
    clang $CFLAGS $INCLUDES -c platform/system_ra6m5.c -o "$BUILD_DIR/system.o"

    echo "==> Compiling FreeRTOS sources"
    clang $CFLAGS $INCLUDES -c third_party/FreeRTOS/tasks.c -o "$BUILD_DIR/rtos_tasks.o"
    clang $CFLAGS $INCLUDES -c third_party/FreeRTOS/queue.c -o "$BUILD_DIR/rtos_queue.o"
    clang $CFLAGS $INCLUDES -c third_party/FreeRTOS/portable/GCC/ARM_CM33_NTZ/non_secure/port.c -o "$BUILD_DIR/rtos_port.o"
    clang $CFLAGS $INCLUDES -c third_party/FreeRTOS/portable/GCC/ARM_CM33_NTZ/non_secure/portasm.c -o "$BUILD_DIR/rtos_portasm.o"
}

link_artifacts() {
    echo "==> Linking firmware"

    ld.lld -T platform/ek_ra6m5.ld \
        "$BUILD_DIR/startup.o" \
        "$BUILD_DIR/system.o" \
        "$BUILD_DIR/application.o" \
        "$BUILD_DIR/rtos_tasks.o" \
        "$BUILD_DIR/rtos_queue.o" \
        "$BUILD_DIR/rtos_port.o" \
        "$BUILD_DIR/rtos_portasm.o" \
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

## Notes

- The startup code is intentionally small so it can stay in-tree and be audited easily.
- The workflow assumes host-native LLVM tooling is available.
- The flashing step can be adjusted depending on whether the board is accessed through a debug probe or Renesas programming mode.
