# Composer Pipeline Overview

This is the current implementation overview, reconciled with source on
2026-09-25. The [Clef specification](../../clef-lang-spec/README.md) governs
language semantics. [CCS Architecture](CCS_Architecture.md) and the
[Baker contract](../../clef/docs/fidelity/Baker_Saturation_Architecture.md) describe
compiler-owned construction; [Alex](Alex_Architecture_Overview.md) describes the
witness boundary. [Language Coverage Waypoints](Language_Coverage_Waypoints.md)
and the [PRD index](PRDs/README.md) record tested scopes and unfinished work.

## Current service boundary

CCS parses and checks Clef source, including project libraries and selected
platform declarations, constructs the typed PSG, and runs Baker elaboration and
saturation. The graph retains the executable structure, local coeffects, joint
relationships, source provenance and graph-resident obligations used downstream.
Composer orchestrates admission, Alex witnessing and the selected backend.

FCS is not a second semantic stage between CCS and Composer. The bootstrap
implementation is F#, but F# compiler/runtime behavior does not define Clef
semantics. Source libraries and platform declarations remain project inputs;
compiler intrinsic ownership does not make those inputs unnecessary.

[Lattice Integration](Lattice_Integration.md) coordinates the existing editor
services. The planned [interactive workbench](Interactive_Compiler_Workbench.md)
shares their authority and evaluates SageFS as a resident bootstrap host. Native
execution through ORC is planned; an FSI host must not substitute F# or C behavior
for a Clef source case.

## The current pipeline

```text
Clef source + library/platform declarations
  -> CCS ProjectChecker / NativeService
       -> parsing, native checking and PSG construction
       -> intrinsic elaboration and Baker recipe fan-out/fold-in
       -> owning saturation, placement, obligation and admission passes
  -> Composer source-diagnostic and selected-target gates
  -> Alex: graph + platform reads + positional Huet context
       -> Elements / Patterns / Witnesses
       -> admitted physical operations, preserving required graph correspondence
       -> declaration collection and bounded correspondence validation
  -> serialized MLIR + current backend context
  -> selected backend realization and artifact checks
```

The [source admission gate](../tests/SourceAdmission/README.md) requires owning
CCS errors to stop before witnessed MLIR or a native artifact. A graph that
exists, or one for which saturation has become quiescent, is not automatically
admitted for every operation, target or use. Receiving a checked graph likewise
does not certify that every later transformation preserves its properties.

The current backend handoff is MLIR text plus configuration. General correlated
graph-fact/proof transport and typed consumer gates are explicit **Planned**
[M-01](PRDs/M-01-DialectAdmission.md) work. The diagram describes the governing
boundary without claiming that work is complete.

## Ownership by layer

| Layer | Owns | Must not substitute |
|---|---|---|
| CCS checking and construction | Native type/dimension semantics, source admission, symbol relationships and typed graph construction. | An FCS typed-tree overlay or CLR behavior for Clef semantics. |
| Baker and owning CCS nanopasses | Semantic elaboration, evaluation order, capture/residence/layout relationships, range and proof premises under the selected platform. | A backend repair for missing source or graph semantics. |
| Alex | Context-pulled observation of the settled expression and platform facts; admitted Elements/Patterns/Witnesses; physical operation and result composition. | New semantic decomposition or inferred proof premises. |
| Backend | Realize the admitted expression for the named target and preserve/recheck required properties through rewrites and linking. | A successful verifier or link as proof of all source guarantees. |
| Host and clients | Lifecycle, versioned source submission, scheduling, inspection and evidence presentation. | A second checker, graph mutator or emitter. |

Representation decisions can depend on declared platform information before
Alex runs. The earlier rule that CCS/nanopasses cannot know targets was too broad.
Alex is also target-aware: the governing selection key is expression family ×
platform/backend profile × witness form. Profile facts and proof identities
remain dependencies, not incidental architecture-name guesses.

## Baker settles; Alex witnesses

Baker Ingredients/Recipes construct executable graph structure and the exact
participants that justify it. Fan-out and fold-in preserve source identity,
ordered operands, multiplicity and joint relations as required by each operation.
They do not hand Alex an F# evaluation algorithm to reinterpret.

Alex receives `WitnessContext`, containing the graph, `TransferCoeffects`, the
Huet zipper and emission coordination state. Its zipper contains only focus,
path and graph. Program facts are read from nodes, layouts, ranges and codata;
`TransferCoeffects` currently carries platform reads and target selection.
The mutable operation/operand accumulator, scopes and visited sets are separate
from the zipper and do not authorize a mutable semantic reconstruction layer.

[`Values.fs`](../src/MiddleEnd/Alex/Traversal/Values.fs) derives SSA names from
node/role ordinals and block argument positions, following structural aliases.
No CCS pass preassigns Alex SSA numbers. Operand recall remains a separate
physical emission concern: having a name does not prove an operand was emitted.

The current
[`NanopassArchitecture.fs`](../src/MiddleEnd/Alex/Traversal/NanopassArchitecture.fs)
runs a combined witness over a shared post-order traversal, with scope-owning
witnesses invoking the common traversal for their bodies. The registry tries
category-selective witnesses and reports an unhandled-node failure; coverage
validation checks missed reachable nodes. It does not run independent parallel
witness traversals. Sharing Patterns alone establishes no concurrency guarantee.

Witnesses compose public Patterns, which compose atomic Elements. `module
internal` is assembly visibility, not a firewall between folders in the same
Composer assembly. Review and the [Alex component tests](../tests/Alex.Tests/README.md)
support the architectural boundary; no fixed Witness/Pattern line count proves it.

## Thin MLIR and target realization

[Thin Middle End](Thin_Middle_End_Design.md) keeps semantic decisions in the
saturated graph. "Flat" emission does not prohibit structured operations,
nested regions, results or block arguments. The standard `func`/`memref`/`arith`/
`scf`/`index` vocabulary is a baseline, not a claim that every operation in those
dialects is implemented or every target uses identical forms. Further forms
need the M-01 operation/profile admission and preservation contract.

The current middle-end post-witness pass collects and validates function
declarations. Static-storage correspondence is checked before serialization.
Source-level closures, sequences and continuation semantics are not deferred
to a second semantic MLIR pipeline. Existing target/serializer exceptions and
incomplete information transport remain reconciliation work in M-01; they do
not amend the doctrine or establish full support.

For the implemented ELF path, [LLVM/LLD](LLVM_Backend.md) performs:

```text
MLIR -> mlir-opt -> mlir-translate -> LLVM IR
     -> opt (target bitcode) -> ld.lld (LLVM code generation and ELF linking)
```

No Clang or separate `llc` is invoked by that path. Console deployment uses
explicit native startup/runtime inputs, including libc on the supported Linux
profile; freestanding and embedded modes have their own entry/runtime contract.
A binary without the .NET runtime is not necessarily a binary without native
runtime dependencies. Other target pathways retain their own admission and
artifact/execution gates.

## Current source map

| Source | Responsibility |
|---|---|
| [`clef/Project/ProjectChecker.fs`](../../clef/src/Compiler/Project/ProjectChecker.fs) | Project inputs, dependency/platform selection and checking, including volatile source overrides. |
| [`clef/NativeTypedTree/NativeService.fs`](../../clef/src/Compiler/NativeTypedTree/NativeService.fs) | Native checking orchestration and graph pipeline. |
| [`clef/NativeTypedTree/NativeTypes.fs`](../../clef/src/Compiler/NativeTypedTree/NativeTypes.fs) | Native type algebra. |
| [`clef/PSGSaturation/SemanticGraph`](../../clef/src/Compiler/PSGSaturation/SemanticGraph) | Graph structure, codata, incidence and owning graph analyses. |
| [`clef/Baker`](../../clef/src/Compiler/Baker) and [`clef/Nanopass`](../../clef/src/Compiler/Nanopass) | Ingredients, recipes, elaboration/fold-in and owning saturation/admission passes. |
| [`FrontEnd/ProjectLoader.fs`](../src/FrontEnd/ProjectLoader.fs) | Composer front-end entry; calls CCS `ProjectChecker`. |
| [`Core/CompilationOrchestrator.fs`](../src/Core/CompilationOrchestrator.fs) | Diagnostics/target gates and front/middle/backend orchestration. |
| [`MiddleEnd/MLIRGeneration.fs`](../src/MiddleEnd/MLIRGeneration.fs) | Alex ingress, operation collection, correspondence checks and serialization. |
| [`MiddleEnd/Alex`](../src/MiddleEnd/Alex) | Traversal, context, Elements/Patterns/Witnesses and dialect serialization. |
| [`Core/Types/Pipeline.fs`](../src/Core/Types/Pipeline.fs) | Current backend interface and configuration handoff. |
| [`BackEnd/LLVM`](../src/BackEnd/LLVM) | LLVM realization and direct LLD linking. |

The former `Core/FCS`, `Core/PSG/Nanopass`, `Alex/Bindings`, `PSGXParsec` and
`MLIRBuilder` paths in earlier overviews are not current implementation entry
points. Composer no longer runs the old application-flattening/pipe-reduction
sequence after an FCS overlay; inspect the owning CCS/Baker path instead.

## Development and validation

Trace a failure from source admission through graph construction/saturation,
Alex observation and target realization. Fix the first owning stage where the
contract breaks. Do not repair missing recipe semantics by name matching in a
Witness or by a runtime surrogate. Native types, capture modes and selected
representation facts must survive boundaries; raw pointers or CLR delegates
must not replace the admitted Clef contracts.

The [source gate](../tests/SourceAdmission/README.md),
[Alex component suite](../tests/Alex.Tests/README.md),
[regression runner](../tests/regression/README.md), and relevant proof/native
harnesses establish different boundaries. An MLIR verifier checks the emitted
IR, not Baker's source semantics or lifetime premises. A component fixture is
not source-to-native coverage. Record exact revisions and evidence in the
coverage waypoints; do not infer current results from historical sample counts.

## Historical and related material

Alloy is retained as historical context for the earlier native-library design;
current intrinsic/library ownership is recorded in CCS, not inferred from that
archive. Earlier FCS typed-tree-overlay and parallel-witness diagrams are design
history and do not override this pipeline or the current specification.

- [Lattice Integration](Lattice_Integration.md): editor transport and shared authority.
- [Interactive Compiler Workbench](Interactive_Compiler_Workbench.md): planned warm host and native REPL bridge.
- [Proof Composition](Proof_Composition_Architecture.md): shared dispatch, composition and evidence policy.
- [WebView Desktop Architecture](WebView_Desktop_Architecture.md): frontend/native application integration.
- [FPGA Targeting](fpga-targeting/README.md): circuit and artifact verification roadmap.
- [Witness Boundary Audit](Witness_Boundary_Audit.md): dated findings, not a current inventory.
