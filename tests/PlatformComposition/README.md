# Platform composition checks

Run from Composer:

```text
dotnet run --project tests/PlatformComposition/PlatformComposition.Tests.fsproj
```

The F# runner writes temporary Clef projects and checks them through CCS's
`FidprojLoader`, `SourceResolver` and `ProjectChecker`. Production Clef sources
are parsed and checked by CCS, never compiled as F# test source.

The checks cover dependency diamonds, normalized shared source paths, actual
dependency cycles, link metadata, both orderings of full BAREWire and its
metadata dependency, and explicit `[platform] description` selection across
sibling packages, using both namespace/nested-module and file-level module
exports. They verify that quoted aliases retain the original
description and memory-space node identities, unrelated catalogue records do
not become selected platforms, malformed or ambiguous exports diagnose, and
selected core architecture/OS/runtime claims agree with the manifest. Legacy
directory selection remains covered. The selected manifest's source dependency
closure is tracked separately from the entire workload graph: application and
unrelated dependency exports, aliases and injected core references cannot satisfy
that platform's declaration. `bare` and `freestanding` are compatible
runtime names for this consistency check.

Pin attributes remain application declarations; the pin/clock/device inventory
and C ABI facts that satisfy them come from the selected platform's closure.
Legacy structural pin and ABI discovery is covered separately. Triple checks
cover the current x86_64 and Cortex-M33 architecture spellings and their
Linux/none OS components, including `x86_64-unknown-none` and
`thumbv8m.main-none-eabi`. They are not a general LLVM triple alias validator.

A hardware `Design` regression verifies that `Clock` remains elaboration
metadata, `Step` remains reachable logic, and no native string pool is allocated
for clock endpoint strings. CPU runtime strings still require declared `rodata`.

An explicit root must be a fully qualified immutable module-level binding of
`PlatformDescription` or `PlatformDescriptor`, directly declared, quoted or
aliased. Calls, mutable roots and arbitrary record composition are not evaluated
by the platform reader. The selector belongs in the **selected platform's**
manifest; the workload selects that manifest through its `platform` dependency.

Fixtures remain in the reported temporary directory for inspection. The runner
does not invoke a vendor toolchain, flash hardware or prove boot/runtime behavior.
