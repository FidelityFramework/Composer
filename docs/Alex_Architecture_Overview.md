# Alex Architecture Overview

This overview states the witness boundary and records its source audit on
2026-09-26. The [pipeline overview](Architecture_Canonical.md),
[Baker contract](../../clef/docs/fidelity/Baker_Saturation_Architecture.md),
[Thin Middle End](Thin_Middle_End_Design.md), and
[M-01](PRDs/M-01-DialectAdmission.md) distinguish the governing architecture from
implemented coverage. M-01's general admission and evidence transport remain
planned; this overview does not close that work.

<a id="the-three-layer-architecture-elements-patterns-witnesses"></a>

## Responsibility

Alex receives the Baker-settled graph and witnesses its admitted computation as
MLIR. Semantic construction, evaluation relationships, captures, layout and proof
premises belong in their owning CCS/Baker stages. Alex must retain the graph's
identities and read the facts needed by the selected target; it cannot reconstruct
a missing source algorithm or manufacture evidence to make emission succeed.

The September 26 audit found violations of that rule: callable and storage
validators reran source settlement during witnessing, and numeric/type helpers
selected widths and reconstructed types. Those operations belong in CCS/Baker.
The corrective work publishes source-validated, eager per-occurrence projections
and changes the corresponding consumers to passive reads. A cached lookup alone
does not enforce the entire boundary: Alex's dependency and input types must also
exclude source analysis entry points, semantic lazy computations, hyperedges and
unresolved inference cells. The source-to-witness handoff is accepted only when
those capability restrictions and the owning source/native regressions hold.

The [closure settlement contract](Closure_Settlement_Contract.md) applies this
boundary to callable values: Baker retains exact capture, environment, call and
destination participants; Alex pulls the settled carrier at the actual occurrence
and keeps its function and environment operands distinct. Environment identity
belongs to the runtime formation, even when two values share implementation code.

The receiving vocabulary has three layers:

| Layer | Responsibility |
|---|---|
| `Elements/` | Atomic MLIR operations and physical operand/type handling through XParsec state. |
| `Patterns/` | Compose Elements into the admitted physical form, pulling its settled facts from the context. |
| `Witnesses/` | Observe the node at its actual graph position and invoke the appropriate Pattern; return operations, a value/void result, a diagnostic, or skip. |

These are Alex's emission layers, not Baker Ingredients/Recipes. There is no
correctness-bearing line-count limit on a Pattern or Witness. Elements use
`module internal`, which restricts assembly visibility. Witnesses compile in the
same Composer assembly, so this **does not** prevent them importing Elements.
Pattern composition is an architectural responsibility, supported by review and
component tests, not an enforced folder-level type firewall.

## Context pull and Huet navigation

[`TransferTypes.fs`](../src/MiddleEnd/Alex/Traversal/TransferTypes.fs) keeps three
different concerns explicit:

- `PSGZipper` contains focus, path and graph. It is navigational, with no mutable
  fields, SSA counter or semantic accumulator.
- `TransferCoeffects` carries platform reads and target selection. Program facts
  are read from graph nodes, layouts, ranges and `Graph.Codata`; they are not the
  old fourteen-field Composer analysis bundle.
- `MLIRAccumulator`, scope references and visited sets coordinate emitted
  operations and operand recall. These are mutable implementation state outside
  the zipper. Their presence does not authorize semantic analysis in Alex.

`WitnessContext` carries these inputs to a witness. XParsec's
[`PSGCombinators.fs`](../src/MiddleEnd/Alex/XParsec/PSGCombinators.fs) threads the
corresponding parser state; the input is graph structure rather than source
characters. Pulling a fact through `ctx` must preserve the origin and position
on which that fact depends. Two occurrences of one shared graph node can have
different enclosing scopes; re-rooting by node ID is not interchangeable with
Huet navigation from the actual occurrence.

SSA names are derived in
[`Values.fs`](../src/MiddleEnd/Alex/Traversal/Values.fs): `V(node, k)` names a
node's emission family and `Arg i` a block argument, with explicit role families
and structural aliases. There is no CCS SSA-preassignment pass or emission
counter. Deriving a name alone does not establish that an operand has been
witnessed; patterns also use the existing accumulator's operand/type recall.

<a id="the-key-distinction-shared-vocabulary-vs-execution-coupling"></a>

## Current traversal and selection

```text
MLIRGeneration.generateWithLinkedLibraries
  -> MLIRTransfer.transfer
  -> target-selected WitnessRegistry
  -> NanopassArchitecture.executeNanopasses
       -> graph declaration roots and classified definitions
       -> post-order Huet traversal / scope-owned traversal
       -> combined witness tries registered witnesses at each node
       -> scope and operation accumulation
       -> reachable-node coverage validation
  -> declaration collection and bounded correspondence checks
  -> serialization
```

The code uses one combined witness over a shared traversal. A witness returning
`TRSkip` lets the next registered witness observe the node; the first non-skip
result is retained. If none handles it, compilation receives a diagnostic.
Reachable nodes missed by traversal produce separate coverage diagnostics.
Scope-owning witnesses use the common traversal callback for their bodies.

This is **not** one parallel traversal per witness. Old IcedTasks examples,
`EnableParallel` pseudocode and claims that sharing Patterns proves parallel
safety do not describe the implementation. The registry coordinates existing
witnesses; it is not permission to add a second emitter or source-name dispatcher.
Share physical vocabulary through Patterns rather than calling another witness
to supply missing source semantics.

[`WitnessRegistry.createRegistry`](../src/MiddleEnd/Alex/Traversal/WitnessRegistry.fs)
now returns an immutable target-selected registry and scope callback owned by
one transfer. Separate transfers cannot overwrite its witness selection.
The type-mapping migration also removes process-global target selection and
diagnostic collection from
[`TypeMapping.fs`](../src/MiddleEnd/Alex/CodeGeneration/TypeMapping.fs).
Shared editor/agent clients do not establish concurrent compilation safety;
all mutable source, witness and realization state must have a compilation or
worker owner. The
[workbench](Interactive_Compiler_Workbench.md#integrity-contract) must serialize
compiler work or isolate workers until a different policy is established.

Full and incremental compilation share the
[parallel execution and measurement contract](Nanopass_Incremental_Contract_Direction.md#26-parallel-full-builds-and-comparative-timing).
The [.NET timing session](../src/Core/Timing.fs) retains independent overlapping
spans per compilation and reports measured wall time separately from their sum.
Registry isolation and instrumentation are individual concurrency prerequisites;
the compiler's segment execution and reconciliation gates establish the complete
parallel path.

## Thin emission and target selection

Alex is target-aware. It observes the selected platform and settled graph facts
to choose an admitted Pattern/Witness form. M-01's admission key is expression
family × platform/backend profile × witness form. Target selection is not a
license to infer missing layout, ownership or source evaluation semantics.

"Flat" or "thin" emission means semantic decomposition is settled above the
witness boundary. It does not prohibit nested regions, results, block arguments
or structured `scf` operations. The standard baseline includes `func`, `memref`,
`arith`, `scf` and `index`; further forms need the operation/profile contract in
M-01. Target-specific realization belongs to the selected backend leg.

The [FPGA workstream](fpga-targeting/README.md) is a guide to preservation and
artifact integrity across these boundaries. Its circuit transformations,
scheduling, handshake/buffer insertion, Colibri component selection/composition,
HDL emission and technology mapping belong to the FPGA backend. They do not
expand Alex's responsibility beyond positional observation and admitted
Element/Pattern/Witness composition. Target-aware selection of admitted portable
physical forms expresses settled facts; target-specific vocabulary and encoding
belong below the declared backend boundary.

Likewise, [incremental nanopass restructuring](Nanopass_Incremental_Contract_Direction.md#8-baker-settlement-and-extensible-alex-witnessing)
preserves Baker's elaboration, dependency propagation and saturation ownership.
Alex's zipper retains the actual occurrence and scope through witnessing;
dependency invalidation, semantic fixed points and design-time scheduling stay
with their owning compiler infrastructure. Reusing a navigation position does
not establish that its graph facts or proofs are still current.

No post-witness MLIR semantic transform is authorized. The former
`MLIRNanopass.fs` path must be removed rather than retained as a place to repair
declarations, closures or continuations. External declarations and signatures
must already have source-owned authority; Alex composes their physical form
through the existing Huet Element/Pattern/Witness path. All MLIR plugin
dependencies must be removed immediately, without waiting for a consumer
migration. Target lowering stays in the selected backend. Current serializer
and target-path reconciliation work remains recorded in M-01; the rule itself
does not establish successful compiler or native gates.

## Existing integrity evidence and its limits

[`SeqWitness.fs`](../src/MiddleEnd/Alex/Witnesses/SeqWitness.fs) rejects unresolved
source suspension forms when Baker has not settled the needed frame, segments
and resumption. It reads exact frame/slot identities and current-read admission
facts rather than rebuilding them from a body shape. Missing semantic premises
must continue to fail in their owning stage.

[`StaticStorageValidation.fs`](../src/MiddleEnd/Alex/Traversal/StaticStorageValidation.fs)
checks bounded correspondence between settled BAREWire storage and emitted
operations before serialization. Coverage diagnostics prevent silent unhandled
reachable nodes. Neither mechanism proves every lowering preserves all source
properties; M-01's general correlated fact/proof transport remains planned.

The [Alex component tests](../tests/Alex.Tests/README.md) exercise public Patterns
and Witnesses, graph/codata/edge preservation, positional navigation, diagnostic
behavior, real MLIR verification and standard lowering. Fixtures supply already
settled facts. They do not establish source admission, proof discharge, capture
lifetime or native behavior. The current array-index Pattern, for example, does
not itself reject absent range evidence; upstream and full-pipeline gates own
that requirement.

The existing [zipper tests](../tests/Alex.Tests/ZipperTests.fs) check shared-node
occurrences with different enclosing scopes and the path loss caused by
re-rooting. Those navigation laws do not establish that every scope-owning
witness preserves the correct occurrence. C-series acceptance also exercises
lambda, match and control-region traversal with scoped operand/block-argument
recall. Deliberate graph-root or cross-reference entry and accidental loss of
structural ancestry must remain distinguishable.

No architecture CI workflow or parallel-equivalence test from the earlier
version of this overview is present as described. Those sketches were proposals,
not installed enforcement. Current evidence belongs in owning tests and
[Language Coverage Waypoints](Language_Coverage_Waypoints.md).

## Source map

| Current file | Role |
|---|---|
| [`MiddleEnd/MLIRGeneration.fs`](../src/MiddleEnd/MLIRGeneration.fs) | Graph/platform ingress, witnessing, correspondence checks and serialization. |
| [`Traversal/PSGZipper.fs`](../src/MiddleEnd/Alex/Traversal/PSGZipper.fs) | Positional Huet navigation. |
| [`Traversal/NanopassArchitecture.fs`](../src/MiddleEnd/Alex/Traversal/NanopassArchitecture.fs) | Combined witness, scope traversal and coverage checks. |
| [`Traversal/WitnessRegistry.fs`](../src/MiddleEnd/Alex/Traversal/WitnessRegistry.fs) | Target-selected witness registration. |
| [`CodeGeneration/TypeMapping.fs`](../src/MiddleEnd/Alex/CodeGeneration/TypeMapping.fs) | Physical type mapping from graph and platform facts. |
| [`Dialects/Core/Types.fs`](../src/MiddleEnd/Alex/Dialects/Core/Types.fs) and [`Serialize.fs`](../src/MiddleEnd/Alex/Dialects/Core/Serialize.fs) | Current operation vocabulary and text serialization. |

The [LLVM backend](LLVM_Backend.md) owns `mlir-opt`, `mlir-translate`, target
bitcode preparation with `opt`, and direct `ld.lld` linking. It does not invoke
Clang or a separate `llc`. Native ORC execution is planned work, not an existing
alternative hidden inside Alex. Deployment modes and runtime inputs are defined
in [`Core/Types/Dialects.fs`](../src/Core/Types/Dialects.fs) and the backend contract.
