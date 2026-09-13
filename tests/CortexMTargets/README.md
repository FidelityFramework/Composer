# Cortex-M target contracts

Run `dotnet run --project tests/CortexMTargets/CortexMTargets.Tests.fsproj` from
the Composer checkout after building Composer. LLVM and ARM GNU binutils are
required; the existing Renesas toolchain fallback is supported.

Synthetic CCS-checked platform fixtures build native Cortex-M33 soft-float and
STM32H747 Cortex-M7 FPv5-D16 hard-float images. The M7 fixture emits hardware
double-precision arithmetic and checks target identity, flash/DTCM/AXI SRAM placement,
reserved ARMv7-M vectors, ELF ABI, rejection of unsupported probe operations,
and removal of stale evidence. Negative declarations exercise the real resolver.
The M33 flash-at-zero rule remains required. No probe is opened, and the fixture
images are never executed; they are build contracts, not board firmware.

The AXI SRAM fixture reserves a 346112-byte packed RGB565 logo in BSS alongside
the 8 KiB stack. Checks cover the generated `RAM_ORIGIN` assembly constant,
the selected physical address/name/capacity, and linker rejection of a BSS
reservation that overlaps the stack. AXI SRAM selection establishes placement;
its clock, initialization, cache and peripheral access remain startup obligations.

The evidence directory printed by the runner retains declarations, startup,
compiler logs, LLVM IR, ELF reports and generated layouts.

## Validation checkpoint: 2026-09-13

`dotnet run --project tests/CortexMTargets/CortexMTargets.Tests.fsproj` passed
all 26 checks. The maintained regression runner also passed native compilation
and execution of `01_HelloWorldDirect`:

```sh
# From tests/regression:
dotnet fsi Runner.fsx -- --sample 01_HelloWorldDirect
```

The native binary was separately checked for exit status 0, stdout
`Hello, World!` followed by a newline, and empty stderr. Logs and that execution
record are retained locally under `targets/review-2026-09-13/` in this test
directory, which is ignored by Git. Generated images are not committed.

Restore emitted the existing NU1902 warning for Fidelity.Data's
`Microsoft.Build.Tasks.Git` 8.0.0 dependency and a missing SourceLink metadata
warning. They did not fail the build or tests. This checkpoint validates the
bounded image contracts and selected hosted regression; it does not claim a
full regression-suite pass or execution of the synthetic MCU fixtures.
