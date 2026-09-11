# Composer-owned Cortex-M build and deployment

`composer compile project.fidproj --deploy -k` performs the complete checked
source-to-board path. The first supported image profile is the EK-RA6M5's secure
Cortex-M33, Thumb, soft-float, 32-bit reset image. Additional devices need their
own reviewed platform/probe contracts; this is not generic support for all MCUs.

## Ownership

The normal frontend and Alex pipeline produce MLIR. `PlatformPipeline` selects
the MCU backend for MCU projects. The orchestrator passes a resolved, typed
`EmbeddedTarget`; the backend owns lowering, image construction and deployment.

`Target.fs` projects `PlatformDescription`, `CortexMImageDescriptor` and the named
vector `StructDescriptor` from CCS's checked graph. BAREWire checks the physical
memory-space projection and AAPCS vector storage in the Composer process. There
is no F# Interactive evaluation of platform source. CCS remains responsible for
the full source declaration diagnostics.

`Layout.fs` emits the linker script and assembler constants. `Image.fs` checks
the entry ABI, runs LLVM optimization/object generation, assembles the explicit
project startup input with ARM GNU `as`, links with LLD, and verifies the resulting
ELF/binary/vector table. Tool processes receive .NET `ArgumentList` arguments;
there is no application shell hook or script interpreter. No C source is compiled.

`Probe.fs` loads the installed SEGGER SDK with .NET native interop. The vendor
supplies SWD and flash algorithms. Composer checks the physical part, original
recovery hashes, image hashes and exact flash readback, preserves option bytes,
and resets only after successful verification. Firmware auto-update and GUI
dialogs are disabled. Inspect/watch are non-programming operations.

The `.S` startup file is a declared native input, not an external build driver.
It owns reset/interrupt semantics and the checked five-word empty-string-array
entry adapter. It remains reviewable application code. `provided_libraries`
accounts for that input's native binding identity; it does not supply host
libraries or bypass the linker's unresolved-symbol rejection.

## Project configuration

```toml
[compilation]
target = "mcu"

[build]
output = "HelloBlinky.elf"
output_kind = "embedded"

[embedded]
startup = "boot/startup.S"
entry_abi = "clef-empty-string-array32"
provided_libraries = ["helloblinky_boot"]
recovery = "recovery/original"

[embedded.vector_handlers]
1 = "Reset_Handler"
15 = "SysTick_Handler"
16 = "Color_Handler"
17 = "Pace_Handler"

[embedded.watch]
blinky_irq_counts = 3
blinky_exit_code = 1
blinky_fault = 6
```

Hardware origins/capacities, vector shape/alignment, stack reservation, reset
symbol, silicon identity and preserved-option extent belong to Fidelity.Platform's
BAREWire declarations. They are not duplicated as project integers.

`embedded.tool_directory` and `embedded.probe_library` optionally name installed
tools, resolved relative to the project. Environment alternatives are
`COMPOSER_ARM_GNU_BIN` and `COMPOSER_JLINK_LIBRARY`. ARM tools otherwise resolve
from PATH or the local Renesas installation; SEGGER otherwise resolves from the
local e² studio installation. LLVM/MLIR/LLD must be installed. These external
vendor/toolchain dependencies are not bundled into the application repository.

## Commands and artifacts

```sh
composer compile HelloBlinky.fidproj -k
composer device HelloBlinky.fidproj --action capture
composer compile HelloBlinky.fidproj --deploy -k
composer device HelloBlinky.fidproj --action watch --seconds 5
```

Capture the original board before its first deployment, selecting a fresh
`embedded.recovery` directory. Capture refuses to replace different earlier bytes.
Other device actions are `inspect` (default), `reset`, and `restore`. Restore
programs only the original code flash, never the option snapshot. It remains
unexercised on the physical acceptance board.

Build-only operation does not access the board. `--deploy` always builds afresh;
it rejects intermediate-only flags. There is no standalone stale-image flash
command. A failed image build invalidates its evidence before linking. Watch
checks board/image agreement before reading declared symbols and confines reads
to SRAM.

The ELF, binary, optimized IR, objects, symbols, disassembly, map, target report,
boot layout and `<name>.build-evidence.json` are retained beside the output.
Deployment writes `<name>.deployment.json` with the binary hash, readback result,
option-preservation result and post-reset snapshot. A build result alone is not
evidence of functional board behavior.

## Validation

```sh
dotnet run --project tests/Mmio/Mmio.Tests.fsproj
dotnet run --project tests/DeviceAccess/DeviceAccess.Tests.fsproj
dotnet run --project tests/MCU/MCU.Tests.fsproj -- /path/to/HelloBlinky.fidproj
dotnet run --project tests/NativeCallbacks/NativeCallbacks.Tests.fsproj
```

These are compiled F# test projects. Build Composer and HelloBlinky first. The
eight MMIO cases check rejection diagnostics and preservation of exact-width
volatile operations through LLVM optimization and ARM lowering. No host process
executes MMIO addresses. The seven hosted callback cases compile and execute
ordinary native programs to check callback and capture regressions.

The fourteen MCU checks use a real freshly built image, reject malformed ELF/vectors, exercise
the linker's data/stack collision assertion, reject a surviving allocator import,
and check that a failed build cannot retain deployment evidence. They do not open
the probe. Physical acceptance used Composer's `--deploy` and `device --action
watch`; the binary matches the earlier accepted 10% PWM image byte for byte.

Composer's `tests/IOMap/IOMap.Tests.fsproj` reads CCS-checked `.clef` declarations
and checks package pins and complete connectivity against the pinned vendor netlist. See the
EK-RA6M5 platform's `docs/IO_MAP.md` for its command and source requirements.

HelloBlinky selects a `DeviceAccessPlan` from Fidelity.Platform.Contracts. CCS
checks each used register's region, mapping, width, permissions, write range and
additional typed predicates. Composer lowers the settled codata. With `-k`,
`targets/intermediates/device-access.json` records those checks and their external
premises. A raw MMIO constructor is rejected when a plan is selected. The
device-access suite also exercises a synthetic 64-bit guest with deliberately
narrow grants; it does not boot a VM or execute MMIO on the host.

The SDK's running-memory reads and post-reset snapshot do not establish calibrated
timing, physical attack resistance, credential security or peripheral driver
coverage beyond the exercised application.
