# FPGA verification roadmap and acceptance gates

**Status: all implementation milestones Planned, 2026-09-25.**

This is the execution map for the [FPGA workstream](README.md). The documents
establish direction; they do not establish a working Dynamatic integration,
VHDL backend, Rocq adapter or verified bitstream. Existing HelloArty and HelloProof
evidence remains scoped to the artifacts and claims already recorded.

## Milestone map

The FPGA identifiers below are workstream milestones, not replacements for
language PRDs. Each closes only its named scope. A milestone becomes Complete
when its positive, rejection and evidence gates pass on recorded revisions.

| Milestone | Deliverable | Dependencies | Status |
|---|---|---|---|
| FPGA-01 | Admitted circuit subset, source/evidence handoff and baseline inventory | Applicable M-01.a/b contracts and existing width/proof ownership | Planned |
| FPGA-02 | Owned Dynamatic fork with Clef-derived input and preserved graph contracts | FPGA-01 | Planned |
| FPGA-03 | Canonical Colibri selection and direct VHDL-2008 output | FPGA-01; FPGA-02 for the elastic path | Planned |
| FPGA-04 | Independent combinational VHDL-to-Rocq adapter and checked instance/model relation | FPGA-01 schema; FPGA-03 artifacts for acceptance | Planned |
| FPGA-05 | Sequential component contracts, composition and source-to-circuit proofs | FPGA-03 and FPGA-04 | Planned |
| FPGA-06 | Open synthesis and implementation flow with mapped-artifact verification | FPGA-05 and exact-device primitive models | Planned |
| FPGA-07 | Bitstream-derived circuit correspondence and board acceptance | FPGA-06 and supported configuration decoding | Planned |

Fork isolation, adapter foundations and build-flow exploration can proceed in
parallel once FPGA-01 has fixed their shared boundary. Acceptance follows the
dependencies above. This does not make all language C/A/T/R work a prerequisite
for a bounded hardware oracle or change those families' existing priorities.

The [functional continuation case study](08_functional_continuation_case_study.md)
provides a planned application-level oracle across these milestones: bounded
parallel frame producers, retained captures, suspension, matching joins and
backpressure. It makes the benefit of C-series functional composition observable,
while its additional continuation/parallelism and FPGA contracts remain explicit
prerequisites. It supplements HelloArty's existing fixed-clock oracle; neither
case substitutes for the other's timing or semantic acceptance.

## FPGA-01 — Freeze the first contract, not the whole architecture

**Scope:** HelloArty's single-clock state machine and one small elastic dataflow
oracle with non-machine-word integer payloads. Record a closed operation set,
arithmetic interpretation, initial/reset state, valid-data conditions and required
observations. Reconcile the handoff with M-01; do not add an independent semantic
IR or an ad hoc name-based component dispatcher.

Deliverables:

- Pinned source and tool inventory, build requirements and per-artifact licenses.
- Baseline artifacts and evidence from HelloArty and HelloProof, with exact claim
  scope, inferred widths and clock facts captured separately.
- The admitted profile, component record requirements, proof dependencies and
  a versioned source/graph/circuit correspondence format.
- Fixtures for exact-width arithmetic, reset, a conditional state update and a
  valid/ready channel. Shared diagnostics retain source identities.

**Exit:** required facts are present in the settled graph and survive the declared
Alex/backend handoff. Missing width, ambiguous clock, unsupported operation and
unresolved required proof premise fail with located diagnostics. Existing native
and FPGA evidence is neither overwritten nor promoted by the new fixtures.

## FPGA-02 — Establish the owned Dynamatic boundary

Follow [01](01_dynamatic_fork.md). Pin the fork and its MLIR/LLVM dependencies;
separate the application frontend from reusable dataflow transformations and
component realization. Replace frontend-specific memory/dependence assumptions
with explicit CCS evidence before admitting memory operations.

**Positive gate:** a Clef-derived bounded control/dataflow workload reaches the
selected Handshake subset without invoking C source ingestion, retaining exact
payload widths and stable obligation/operation correspondence. A buffered or
resource-shared variant preserves the declared accepted-transaction behavior.

**Rejection gate:** absent dependence information, unsupported memory shapes,
unjustified index widths and transformation-induced protocol changes fail.

**Exit:** an independently buildable fork slice, tested input contract and
transformation evidence. The compiler may still use C++ and LLVM infrastructure;
this milestone removes C as the required application language. It does not
require adopting Graphiti or Lean.

## FPGA-03 — Make Colibri realization executable

Follow [02](02_colibri_circuit_basis.md) and
[03](03_widths_and_vhdl_lowering.md). Admit the original Colibri `stream_buffer`
at one explicit configuration, along with the closed arithmetic/register basis
needed by the first workload. Both the direct synchronous path and the fork's
elastic path use this implementation policy.

**Positive gate:** strict-subset VHDL-2008 analysis/elaboration and behavioral
checks pass; observed ports, generics, initialization/reset and inferred widths
match the declared contract. Emit the dependency closure and provenance manifest.
Retain the current SystemVerilog oracle during migration.

**Rejection gate:** accidental sign extension, truncation, changed handshake
capacity, unsupported `seq` behavior, unmodeled external entities or a hidden
SV-only pass fail admission. A width or mode change invalidates stale evidence.

**Exit:** executable VHDL generation with bounded behavioral evidence. No claim
of Rocq-checked source equivalence is made until the following milestones pass.

## FPGA-04 — Build our VHDL-to-Rocq implementation

Follow [04](04_vhdl_to_rocq.md). Implement the selected structural combinational
subset independently, using the published VHDL2Rocq method as research input.
State the parser, normalization, model and checking trust boundaries. The
proprietary translator is not downloaded, required or substituted into the build.

**Positive gate:** inspect actual emitted VHDL, produce Rocq definitions and
obligations, and check the chosen width/range/operation correspondence. Include
one component theorem with symbolic width parameters and an exact instantiated
artifact check. Complete the model's representation/precondition proofs rather
than assuming that unbounded integers equal bit-vector operations.

**Rejection gate:** altered operator/constant/connection, mismatched parameter,
multiple drivers, combinational cycles, unsupported signal values or syntax,
and malformed model/certificate input cannot produce accepted evidence.

**Exit:** generated `.v` files compile through the pinned shared Rocq service;
proof dependencies satisfy the selected profile. Record what remains trusted
in extraction. Merely producing theorem statements is not this gate.

## FPGA-05 — Transfer sequential and composition proofs

Extend the model to initial states, clocked transitions and observations.
Reconcile synchronous cycle semantics with the fork's elastic transaction
semantics. Prove source/component correspondence using state relations; do not
infer it from matching signal names or from a finite simulation trace.

**Positive gates:**

- HelloArty: interpreted counter/state representation, initialization and
  next-state behavior preserve the selected source range invariants.
- Colibri buffer: accepted payload order, loss/duplication freedom, stall
  behavior, reset and capacity hold under explicit interface assumptions.
- Composition: connected components discharge their requirements under a sound
  composition rule, retain clock/reset ownership and satisfy the declared
  end-to-end observation relation. Cyclic premise dependencies need a global
  invariant or an appropriate compositional proof.
- If a Clef component port is introduced, its reference correspondence is checked
  before the implementation replaces the original VHDL.

**Rejection gates:** corrupt a next-state equation, reset, ready path or FIFO
capacity; each must invalidate the relevant theorem. A proof relying on missing
fairness cannot establish eventual progress. A bounded check cannot stand in for
induction. Certificate holes, unsupported rules and admitted axioms cannot stand
in for the required derivation.

**Exit:** `CircuitMap.v`/correctness evidence derived from actual artifacts and
checked against source/component obligations. Report safety, equivalence and
progress as separate claims with their actual premises.

## FPGA-06 — Verify the mapped circuit in the open flow

Pin a tested GHDL/Yosys and Artix-7 synthesis/P&R/bitstream-generation tool closure,
including the exact device database, cell models, constraints and options.
`synth_xilinx`, a selected open Artix-7 P&R flow and Project X-Ray-related tooling
are candidates; their presence is not acceptance of the complete Arty A7-100T
flow. Record artifact hashes and tool manifests without promising bit-for-bit
reproducibility until it is measured.

**Positive gate:** prove or check an admitted correspondence from the reference
circuit to the mapped netlist, using explicit LUT/register/memory/pad semantics.
Extract actual widths, memory resources, connectivity and initialization from
the produced artifact. Preserve intentional BRAM mappings and protocol behavior.
Obtain implementation timing/resource reports against the actual platform clock.

**Rejection gate:** unmapped cells, unsupported primitives, altered LUT functions,
missing clocks/constraints, changed reset/initialization or memory collision
behavior invalidate the relevant claims. Stale source-netlist correspondence is
not repaired by matching hashes alone.

**Exit:** evidence identifies precisely the mapped or implemented artifact covered.
P&R reports, formal circuit equivalence and physical timing premises remain
distinct. Bitstream correspondence belongs to FPGA-07.

## FPGA-07 — Inspect the final configuration

Follow [05](05_artifact_verification.md). Qualify the Project X-Ray/fasm2bels-style
extraction path for the exact Arty part and primitive subset. Existing tests on
other Artix-7 parts are useful references, not acceptance of XC7A100T-CSG324.

**Positive gate:** decode the produced bitstream, reconstruct the relevant
logical/physical circuit, and check its correspondence to the verified mapped
design. Bind the Rocq evidence to that artifact and record decoder/database
assumptions. Program the board with the selected open loader and retain observed
hardware behavior and implementation reports as separate acceptance evidence.

**Rejection gate:** mutations that change admitted primitive behavior,
connectivity or initialization fail the corresponding circuit check. Unsupported
features, wrong device databases and omitted cells/routes produce failure or an
explicitly incomplete extraction. Record format/integrity rejection separately:
a changed unused or redundant configuration bit may preserve circuit semantics,
but still invalidates evidence bound to the earlier artifact bytes.

**Exit:** a bounded artifact-derived proof chain for the named design, FPGA part,
component set and assumptions. This establishes no universal physical-silicon
theorem or support for every Colibri component.

## Extensions and shared edges

After the initial gates, expand synchronous FIFOs, arbitration, memory interfaces,
packet/coding kernels and peripherals by the same admission rules. Introduce CDC
only with its digital model and physical premises. Portable Clef SPI/I2C contracts
can be developed alongside their MCU realizations under
[06](06_platform_and_shared_edges.md); they do not require running every bus edge
through dynamically scheduled computation.

GPU/NPU/CGRA work shares evidence identity, proof admission, diagnostics and artifact
extraction discipline. Each still supplies target execution semantics and its own
acceptance oracle. The FPGA adapter does not establish their coverage indirectly.

## Restart and evidence record

Start with this milestone map, the [series baseline](README.md), current
[coverage waypoint](../Language_Coverage_Waypoints.md) and
[M-01](../PRDs/M-01-DialectAdmission.md). Record each accepted slice with:
repository/tool revisions; selected profile and parameters; claim and premises;
input/output artifact identities; positive and rejection results; checker evidence;
remaining trusted boundaries; and the next unimplemented contract.

Keep proof/model provenance through upgrades and invalidate dependencies that
changed. Update the master roadmap and waypoint only for the scope actually
established. This documentation checkpoint ran link/anchor and whitespace checks;
it did not run compiler, solver, Rocq, synthesis or hardware acceptance tests.
