# Parallel zipper implementation claim: historical correction

Original record: January 27, 2026. Status corrected September 25, 2026:
**the claimed implementation is not present in the inspected current source**.

The earlier document called the architecture implemented and ready to integrate.
It named `FunctionDiscovery.fs`, `DependencyBatching.fs` and
`ParallelCompilation.fs`. Those modules are absent from the current Composer
`src` tree. No current integration or performance evidence is supplied by that
record. Its ready-to-run instructions, fabricated current signatures and expected
speedup tables are withdrawn; the original claims remain available in version
history for archaeology.

Use the [Alex architecture](Alex_Architecture_Overview.md) for current structure
and the [interactive compiler workbench](Interactive_Compiler_Workbench.md) for
host and scheduling work. This document does not authorize implementing the old
worker-context design.

## Source-backed boundary

| Current source | What it establishes | What it does not establish |
|---|---|---|
| [PSGZipper.fs](../src/MiddleEnd/Alex/Traversal/PSGZipper.fs) | Positional graph navigation. | An independent semantic state or parallel scheduling contract. |
| [TransferTypes.fs](../src/MiddleEnd/Alex/Traversal/TransferTypes.fs) | Witness context and mutable physical emission bookkeeping. | The former record-shaped accumulator or arbitrary worker-state union. |
| [NanopassArchitecture.fs](../src/MiddleEnd/Alex/Traversal/NanopassArchitecture.fs) | Current traversal and witness orchestration. | The three proposed discovery/batching/parallel modules. |
| [MLIRTransfer.fs](../src/MiddleEnd/Alex/Traversal/MLIRTransfer.fs) | Current transfer entry points. | A verified parallel integration or reentrant compiler process. |

The ctx pull discipline, codata, coeffects and Huet zipper remain the witness
path. Baker retains semantic construction and saturation. Emission bookkeeping
must not become a second source of evaluation order, lifetimes, layouts or proof
premises. Missing graph evidence remains an admission failure.

## Why the old worker sketch is insufficient

The sketch created a fresh accumulator per function and merged operations,
visited nodes and bindings. That does not establish isolation of global state,
occurrence-specific observations, lazy witnesses, target mapping, initialization
or physical emission scope. Separate accumulators alone do not make the compiler
reentrant. Nor does an SSA dependency list establish every semantic dependency
needed to partition compilation.

The merge sketch appended operation lists and overwrote map entries. List append
is order-sensitive, and map collisions need an admitted policy. Associativity
alone supplies neither commutativity nor preservation of source effects and proof
correspondence. A Baker enrichment pattern cannot be imported as permission for
Alex to construct missing semantics.

## Evidence required before reviving the idea

A future proposal must name a bounded workload and authoritative graph partition,
retain identity/provenance across boundaries, isolate compiler state, establish a
deterministic merge, and compare against fresh ordinary compilation. Validation
must include effects, scopes, globals, required negative admissions and proof
correspondence, not only equal output text. Performance is measured after those
checks; no speedup is promised.

The workbench may keep a compiler warm and schedule solver workers without
parallelizing Alex. Its initial policy remains serialization or worker isolation.
The [research record](Parallel_Zipper_Research_Summary.md) and
[design record](Parallel_Zipper_Design_Synthesis.md) retain the original questions
and their corrected limits.
