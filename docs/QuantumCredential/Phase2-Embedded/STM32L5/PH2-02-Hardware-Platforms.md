# Hardware platforms

This page records the STM32L552 device reference and the selected
KeyStation host, Libre Computer AML-S905X-CC-V2. KeyStation's native graphics
port is independent of the choice of credential processor and root-of-trust
integration.

## Execution selections

| Target | Execution plan | Hardware ownership |
| --- | --- | --- |
| STM32L552 reference | Freestanding Cortex-M33 port requirements | MCU startup, Ariel realization and peripheral drivers |
| KeyStation native | Clef unikernel on S905X after an accepted firmware handoff | Native Meson display, USB host/touch and Mali rendering |
| KeyStation hosted | Linux AArch64 application | Linux device drivers and userspace graphics/input interfaces |

Native KeyStation is the primary port objective. Linux provides a separate
hosted route and a reference for device behavior. Both require an implemented
platform environment and an accepted application image.

## Target 1: STM32L552 device reference

### Board and silicon

The board referenced by the sample manifests is **NUCLEO-L552ZE-Q**. ST identifies
it as a Nucleo-144 board carrying an STM32L552ZE MCU. The part has an Arm
Cortex-M33 core, a maximum CPU frequency of 110 MHz, 512 KB of flash and 256 KB
of SRAM. The selected operating frequency depends on the clock configuration.
See the [board page](https://www.st.com/en/evaluation-tools/nucleo-l552ze-q.html)
and [STM32L552ZE specification](https://www.st.com/en/microcontrollers-microprocessors/stm32l552ze).

TrustZone support permits configured security attribution of memory and
peripherals. A linker section named secure does not perform that configuration.
The [STM32L552 datasheet](https://www.st.com/resource/en/datasheet/stm32l552ze.pdf)
describes the reset state and attribution controls. Application memory ownership
and interrupt routing must agree with the selected security configuration.

Cryptographic accelerators must be selected by exact part. ST assigns the
additional AES, PKA and OTFDEC engines to the **STM32L562** branch in its
[L5 family comparison](https://www.st.com/en/microcontrollers-microprocessors/stm32l5x2.html).
The NUCLEO-L552ZE-Q therefore cannot inherit those capabilities from a generic
STM32L5 family description. A credential implementation needs its own supported
cryptographic realization and acceptance evidence.

### Native runtime and Ariel

A native image combines the application, selected runtime services and device
drivers. It can include scheduling, interrupts and memory protection.
[Ariel](../../../../../Fidelity.Platform/Environments/Linux/x86_64/Ariel/README.md)
is Fidelity's scheduling layer. Its current documented realization supplies
bounded synchronous regions through persistent Linux x86_64 pthread carriers.
The [native evidence](../../../../../Fidelity.Platform/tests/Ariel/native/STATUS.md)
records that implementation's lifetime and retirement behavior.

An MCU realization must supply the selected scheduling policy, timer and
interrupt integration for its environment. The application can initially use
one execution owner while independent services retain their own demand. Choosing
a unikernel does not remove scheduling or require stack-only allocation.

Cortex-M startup establishes the initial stack and reset entry, initializes
runtime data and then enters application code. The image also needs a complete
interrupt table, fault handling and the selected clock/security setup. Those
operations are executable runtime and driver work. CCS provides compilation
services, rather than an on-device standard library or a substitute for startup.

### Native MMIO and driver ownership

The Clef-native path implements device operations against the selected
[MMIO contracts](../../../../../Fidelity.Platform/docs/MMIO_CONTRACTS.md).
Silicon declarations own register layout and transaction requirements. Board
wiring identifies the connected devices and pins. The application selects its
mappings, grants and resource budgets. Composer lowers the admitted operations.

A USB device implementation needs clock and pin setup, controller initialization,
endpoint storage and interrupt handling. Enumeration, descriptors and HID report
processing require protocol code. Empty `init` or `sendReport` bodies do not
supply those operations, and a `Platform.Bindings` namespace does not cause Alex
to synthesize a peripheral driver.

[Farscape](../../../../../Farscape/README.md) remains an option for binding an
existing C implementation. That route retains the implementation and its ABI
as dependencies. A native port owns those operations directly in Clef. Neither
route follows automatically from choosing an LLVM target triple.

### Current source and acceptance

The local [Blinky manifest](../../../../samples/embedded/stm32l5-blinky/Blinky.fidproj)
and [UART manifest](../../../../samples/embedded/stm32l5-uart/UARTEcho.fidproj)
identify the NUCLEO board experiment. Their shared
[startup source](../../../../samples/embedded/common/startup/CortexM33.fs)
contains placeholder runtime work and an incomplete peripheral interrupt table.
These sources are unaccepted scaffolds. They do not establish a running
credential device or the device-access contract described above.

The [EK-RA6M5 HelloBlinky profile](../../../../../Fidelity.Platform/Profiles/EK_RA6M5_HelloBlinky/)
is the current concrete MCU composition reference. A supported STM32L552 port
would need its own silicon/product declarations and selected profile, followed
by reset/startup, GPIO/timer and USB acceptance on the physical board. Record
its memory allocation policy, interrupt ownership and clock configuration with
the image. Security and credential claims require additional evidence for the
actual selected implementation.

## Target 2: KeyStation on AML-S905X-CC-V2

The [Fidelity.Platform board entry](../../../../../Fidelity.Platform/Hardware/Products/LibreComputer/AML_S905X_CC_V2/README.md)
is the source of hardware identity: Amlogic S905X, Cortex-A53 CPU cores, Mali-450
GPU and DDR4 on the V2 board. The standard retail memory configuration is 2 GB.
The entry pins the Linux device-tree identity and V2 U-Boot configuration.

### Native execution

The [port plan](../../../../../Fidelity.Platform/docs/SWEET_POTATO_UI_PORT.md)
starts with a documented firmware handoff, serial output, interrupts and timer
operation. CPU rendering establishes the reference output. Native Meson HDMI
and USB touch follow, then restricted Mali acceleration. Each stage has its own
acceptance evidence and resource ownership requirements.

The UI uses cold functional components and reactive areas as described in
[Fidelity.UI](../../../../../Fidelity.UI/docs/09_sweet_potato_keystation.md).
LVGL contributes component and rendering lessons. A Clef-native engine needs its
own layout, text, damage and input implementations. GPU job completion and
scanout release determine when buffers can be reused.

### Hosted execution

A Linux AArch64 environment would supply the application ABI and userspace
bindings. Mesa Lima and Meson DRM/KMS provide the reference graphics stack.
The application can use direct display or a Wayland host, with input delivered
through the selected Linux input interfaces. Farscape bindings expose those
interfaces. They do not replace the corresponding libraries or kernel drivers.

A native implementation instead supplies the device services itself. USB host
control, enumeration and HID parsing are distinct responsibilities. An external
touchscreen does not intrinsically require Linux.

### Display selection

The installation uses the Waveshare 7.9inch HDMI LCD, SKU 17916 (ASIN B087CNJYB4),
in a horizontal 19-inch rack panel. The
[panel entry](../../../../../Fidelity.Platform/Hardware/Products/Waveshare/7_9inch_HDMI_LCD/README.md)
records its identifiers and physical matrix. Panel timing, rotation, USB reports and touch coordinates
must be qualified together. The board's HDMI output and the proposed panel's
USB input are separate interfaces.

### Compiler and driver integration

Hardware facts belong to Fidelity.Platform. The selected execution environment
supplies its ABI, bindings and runtime requirements. Composer must consume that
selection and emit the corresponding artifact. Changing an LLVM target triple
alone does not establish boot, memory access or device support.

Farscape can bind a C implementation through its declared interface. A full
Clef port implements the required device operations in Clef. Neither path should
introduce symbol-name dispatch that treats an empty function body as an MMIO
implementation.

### Implementation sequence

Use the [platform acceptance ladder](../../../../../Fidelity.Platform/docs/SWEET_POTATO_UI_PORT.md#acceptance-ladder)
for boot, CPU rendering, HDMI/touch, reactive areas and Mali. Keep the firmware,
image and device evidence with the board source pack. Add a runnable Composer
sample only when its actual S905X startup and peripheral dependencies exist.

Credential transport, hardware key management and storage encryption have
separate acceptance criteria. The graphics experiment can use synthetic service
state while those integrations are developed.

## References

- [KeyStation UI realization](PH2-04-UI-Options.md)
- [Sweet Potato product and source inventory](../../../../../Fidelity.Platform/Hardware/Products/LibreComputer/AML_S905X_CC_V2/README.md)
- [Platform composition](../../../../../Fidelity.Platform/docs/PLATFORM_COMPOSITION.md)
- [Composer architecture](../../../Architecture_Canonical.md)
