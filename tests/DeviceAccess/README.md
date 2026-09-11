# Device access checks

From the Composer checkout:

```sh
dotnet run --project tests/DeviceAccess/DeviceAccess.Tests.fsproj
```

The F# runner creates Clef source fixtures and checks them through CCS. The
positive MCU and guest cases also pass through Composer and LLVM optimization;
MMIO is never executed on the host. Optional arguments select case names.

The cases cover region identity and extent, BAREWire alignment/granularity,
transaction width, register and grant permissions, raw-address bypass, plan
ambiguity, mapping establishment/lifetime, byte order and ordering, source
predicate states and type errors, mutation/provenance, unsigned write range,
deferred unused declarations and 32/64-bit address boundaries. The high guest
case reaches the last mapped page of the 64-bit address space while retaining
32-bit register transactions. The guest profile is synthetic and does not boot.

Positive cases compare the emitted access ledger with CCS codata. The MCU case
also checks that a rejected recheck removes an earlier ledger, retaining the
accepted report separately for inspection. Evidence and generated fixtures are
written under a uniquely named temporary directory printed by the runner.

The MCU case also rejects conflicting CLI triples, a conflicting workload
backend and a selected environment with an empty triple before lowering.
Those failures must remove stale evidence instead of falling back to the host.

The complementary `tests/Mmio`, `tests/MCU` and `tests/IOMap` runners check legacy
raw MMIO/volatile ARM lowering, the actual MCU image, and the complete pinned
board netlist respectively. The I/O test projects CCS-checked `.clef` data;
it does not compile production Fidelity.Platform sources as F#.
