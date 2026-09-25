# Owned Dynamatic fork and Clef entry contract

Status: **Planned**. Planning record: **2026-09-25**. No fork, compiler pathway,
new component realization or proof-preservation gate is implemented by this
document. The current CIRCT/SystemVerilog HelloArty pathway remains the baseline.
Start with the [workstream overview](README.md) and [roadmap](07_roadmap.md).

## Decision and ownership

Develop an owned Dynamatic fork for the selected dynamically scheduled FPGA
pathway. Clef enters through Baker-settled computation and target evidence;
the fork does not require an application to pass through C or Clang first.
Dynamatic's C frontend and its associated source assumptions are separable
from its Handshake transformations, hardware construction and RTL library.
The implementation still uses C++/LLVM/MLIR and their build dependencies.
Removing the C input requirement does not imply eliminating those languages.

Colibri-derived circuit realizations are central backend material: buffering,
storage, arbitration, stream adaptation and admitted protocol controllers.
Their Clef ownership and semantic contracts are specified in
[Colibri circuit basis](02_colibri_circuit_basis.md). Existing Dynamatic RTL
components remain useful reference implementations and comparison oracles.

The selected pathway is:

```text
Clef source and selected platform/profile
  -> CCS/Baker: settled PSG, realization requirements and resident obligations
  -> passive Alex: admitted Handshake expression + correlated graph evidence
  -> owned fork: permitted optimization, scheduling and buffer realization
  -> selected Colibri-derived / admitted hardware components
  -> closed hw/comb/seq and external-entity interface
  -> VHDL-2008 + artifact correspondence
```

This extends [M-01](../PRDs/M-01-DialectAdmission.md), particularly M-01.a/b/f.
It creates no second source-semantics IR and no independent proof database.
MLIR is the target expression; its evidence refers to the authoritative graph.
The fixed-clock `hw/comb/seq` pathway remains separately admissible. A Mealy
machine with cycle-visible behavior must not silently acquire variable latency
because a Handshake backend is available.

## Bounded input and version contract

The first fork milestone must publish an operation/profile admission table,
not promise acceptance of arbitrary Clef or every Handshake operation.
Each row identifies the source case, Baker owner, exact operation/type/attribute
forms, memory/effect prerequisites, permitted transformations and an oracle.
Use the shared M-01 register rather than adding a central witness dispatcher.

The incoming package must retain:

- The selected graph revision, platform/profile declarations and source origins.
- Value ranges, held widths, signed interpretation, layouts and conversion sites.
- Explicit data/control dependencies, memory identities and effect ordering.
- Clock/reset, interface, latency and throughput requirements where applicable.
- Resident obligation identities, participants, assumptions and evidence status.
- The admitted transformation policy and its selected component contracts.

Alex reads these facts through typed Elements, composed Patterns and passive
Witnesses. Missing facts return a prerequisite diagnostic to the owning CCS/Baker
stage. Recognizing a library symbol such as `Colibri.fifo` or `Spi.transfer` in
code generation is not the mechanism for choosing a hardware implementation.

Pin the fork commit, upstream Dynamatic commit, LLVM/MLIR/CIRCT revisions,
generated dialect definitions, component schemas and tools together. Dynamatic's
Handshake vocabulary and upstream CIRCT Handshake are not assumed identical.
Any adapter must have a versioned, tested operation contract. Record required
build options and optional tools; interface drift fails the admission gate.

## Upstream maintenance boundary

Dynamatic already moved away from a direct CIRCT dependency in its
[January 2024 change](https://github.com/EPFL-LAP/dynamatic/commit/00e6c068ea9960a36e86ff2404b4b0c16a1c6766).
The recorded reasons include its divergent Handshake dialect, limited use of
upstream passes and duplicated LLVM/MLIR build constraints. That supports owning
a focused dependency boundary; it does not establish that VHDL-emitter AST churn
was the cause. The inspected tree carries selected CIRCT-derived infrastructure.

The fork should retain the dialect/type/verifier machinery its admitted passes
need and expose a narrow, versioned structural interface to the VHDL writer.
Keep upstream compatibility edits behind that interface; do not spread access
to unstable internals throughout serialization and proof extraction. LLVM/MLIR
updates remain deliberate migrations with operation, emitted-artifact and proof
replay gates. A fork reduces unwanted coupling, but still owns the imported
code's maintenance and upstream fixes.

## Existing upstream assets and their actual scope

The research inspection identified upstream main at
`2fac2911faf35b84156e08782a9fcfe8fc6174d8`. This is a source-inspection reference,
not a selected or tested fork baseline. Bootstrap must choose and validate its
own coherent revision set.

| Upstream asset | Reuse in the owned fork | Boundary to retain |
|---|---|---|
| Handshake transformations and hardware construction | Selected scheduling, buffering and graph-to-component machinery | Every enabled pass needs its Clef semantic premises and correspondence account. |
| `export-rtl` VHDL writer and RTL library selection | Existing module/instance, parameter and component-emission machinery | This is not evidence of a complete arbitrary `hw/comb/seq` body-to-VHDL emitter. |
| ElasticMiter | Contextual token-sequence comparison and rewrite testbench construction | Current experiments use Handshake models and SMV tools; a Yosys/Rocq certificate route is additional work. |
| Formal property annotation/database | Named circuit references, obligation records and checker-result plumbing | Existing Handshake properties do not establish Clef arithmetic or source-to-circuit preservation. |
| RTL component models and tests | Behavioral references and adversarial protocol stimuli | Simulation or an upstream proof badge is not acceptance of every parameterization. |

Primary references: [Dynamatic source](https://github.com/EPFL-LAP/dynamatic),
[RTL exporter](https://github.com/EPFL-LAP/dynamatic/blob/main/tools/export-rtl/export-rtl.cpp),
[ElasticMiter](https://github.com/EPFL-LAP/dynamatic/blob/main/experimental/tools/elastic-miter/README.md),
[formal properties](https://github.com/EPFL-LAP/dynamatic/blob/main/docs/DeveloperGuide/DynamaticFeaturesAndOptimizations/FormalProperties.md).
Archive the inspected files with the chosen baseline; moving `main` links are
discovery references, not reproducibility records.

The formal-property path currently emits checker predicates through the SMV
writer. It does not already place Clef proofs into VHDL or produce Rocq terms.
Graphiti contributes design lessons about checked rewrites and explicit semantic
relations; its Lean implementation is not a dependency of this workstream.

## Hardware component replacement contract

Select implementations by established operation/resource identity and capability,
not by a textual module-name resemblance. A Colibri-derived buffer can replace a
Dynamatic buffer only when the relevant behavior agrees:

| Concern | Required matching fact |
|---|---|
| Data transfer | Width, packing, accepted-transfer event, ordering and multiplicity |
| Backpressure | Valid/data stability, ready behavior and combinational dependencies |
| State | Capacity, occupancy, initialization, reset priority and initial tokens |
| Time | Fixed latency or permitted latency variation; registered versus bypass paths |
| Memory | Port count, byte enables, collision/read-during-write semantics and ordering |
| Domains | Clock identity, CDC protocol and reset-domain assumptions |

Buffer replacement changes scheduling and timing when it adds/removes storage or
combinational paths. Admission must establish that change before realization.
Colibri's RAM coding patterns are candidates for preserving block-RAM inference;
the selected GHDL/Yosys mapping must demonstrate the intended cells and behavior.
CDC components are admitted at declared crossings, never substituted for an
ordinary same-clock token edge without a new contract.

## Proof and progress obligations

Handshake correctness normally relates ordered sequences of accepted tokens.
It permits stalls and different cycle counts only when the source contract does.
Payload range predicates apply to meaningful tokens (`valid`, or accepted
`valid && ready`, as declared), not unconstrained data wires during invalid cycles.
A fixed-cycle source requirement needs an explicit latency/timing refinement.

Track reset and reachable-state assumptions, input/output token counts, memory
effects, buffer capacity, deadlock, fairness and environmental progress separately.
An equality of token values does not establish eventual completion or a deadline.
ElasticMiter's contextual constraints are useful models for this separation;
its published data abstraction does not replace full-width arithmetic checking.
See the [ElasticMiter paper](https://doi.org/10.1145/3676641.3715993).

Each rewrite must retain its source and obligation correspondence, including
splits, fusion, eliminated nodes and newly introduced storage. An immutable
projection may transport these graph facts to a tool, but is not an alternate
semantic authority. Invalidated assumptions make downstream evidence stale.
[VHDL lowering](03_widths_and_vhdl_lowering.md),
[Rocq checking](04_vhdl_to_rocq.md) and
[artifact verification](05_artifact_verification.md) define the later boundaries.

## Dependencies, provenance and acceptance

Inventory licenses per imported file, tool and generated artifact. Inspected
Dynamatic implementation files use Apache-2.0 with LLVM exception; Colibri RTL
uses CERN-OHL-W-2.0. A Clef port must account for derivative-source obligations.
Do not label the combined pipeline universally permissively licensed.
The build/runtime dependency inventory must explicitly handle optimization
solvers and optional verification tools; an unavailable dependency is not an
implicit permission to weaken the selected contract.

Concrete replacement and audit items already appear in the inspected tree:

- [Synthesis](https://github.com/EPFL-LAP/dynamatic/blob/main/tools/dynamatic/scripts/synthesize.sh)
  invokes Vivado with an out-of-context Kintex-7 part (`xc7k160tfbg484-2`).
  It is not the proposed open Arty implementation flow.
- [Simulation](https://github.com/EPFL-LAP/dynamatic/blob/main/tools/dynamatic/scripts/simulate.sh)
  compiles its reference driver with Clang. A Clef-fed acceptance harness must
  supply its own input/result oracle without requiring a C application.
- [Operator configurations](https://github.com/EPFL-LAP/dynamatic/blob/main/data/components.json)
  include FloPoCo and Vivado floating-point implementations. Bundled
  [integer divide/remainder cores](https://github.com/EPFL-LAP/dynamatic/blob/main/data/vhdl/support/vitis_hls_cores.vhd)
  carry a Xilinx HLS copyright header. Inventory implementation provenance and
  terms per selected operation; replace or exclude dependencies incompatible
  with the open profile before admitting that operation.
- The [build](https://github.com/EPFL-LAP/dynamatic/blob/main/CMakeLists.txt)
  includes a CBC option alongside Gurobi integration. Qualify the open solver
  and the specific enabled optimization passes instead of assuming every pass
  works with either solver.

NuSMV and nuXmv are distinct: the latter's published license restricts use to
academic/noncommercial purposes. The desired open checker pathway must not
silently acquire that dependency. Retargeting reusable formal construction to
the selected tools is planned engineering, not existing compatibility.
See [nuXmv licensing](https://nuxmv.fbk.eu/license.html).

Acceptance proceeds through a small Clef kernel with bounded state and effects,
then only the next demanded expression/profile. Retain the existing HelloArty
SV output as a comparison oracle while adding VHDL. Positive cases must include
stalls, reset, width boundaries and resource realization; negative cases must
reject unsupported operations, missing proofs, stale facts, incompatible resets,
invalid component substitution and forbidden cycle changes. Record simulation,
formal replay, mapped resources and board execution as separate results.
Use [platform/shared edges](06_platform_and_shared_edges.md) for pad and MCU
contracts and the [roadmap](07_roadmap.md) for ordered implementation gates.
