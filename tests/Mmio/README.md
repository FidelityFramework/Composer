# Exact-width MMIO

Run `dotnet run --project tests/Mmio/Mmio.Tests.fsproj` after building Composer and its CCS dependency.
The tests compile against the actual EK-RA6M5 platform; they do not execute the
MMIO program on the host. Temporary artifacts are retained at the printed path.

`Mmio.reg8`, `reg16` and `reg32` construct distinct opaque handles from static,
non-null, aligned addresses within the declared pointer space. `read8/16/32`
and `write8/16/32` require the corresponding handle. Reads establish unsigned
hardware ranges. Writes require their full source range to fit; there is no
implicit truncation to make a bad write legal. Explicit masks can express a
bit-field value. Dynamic address construction is intentionally outside this slice.

The negative cases check width mismatch, misalignment, null, out-of-space
addresses, negative/oversized writes and dynamic construction. The positive case
checks all six accessors after LLVM O2 and ARM lowering, including discarded reads
and repeated polling. Width and volatility do not prove a register's legal bit
transitions, permissions, unlock sequence, or general memory synchronization.

The MCU backend applies `nounwind` to function definitions; hardware faults are
handled by the owned vector path, and language exception unwinding is not supplied.
The board application additionally checks the entry ABI and final ELF and retains
physical execution evidence in `MCU/Renesas/EK-RA6M5/HelloBlinky/docs/ACCEPTANCE.md`.
