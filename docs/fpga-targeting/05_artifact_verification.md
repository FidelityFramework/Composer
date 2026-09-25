# Verification of emitted FPGA artifacts

**Date:** 2026-09-25. **Implementation status:** Planned.

This document extends the artifact-side pattern of HelloProof to the
[FPGA targeting plan](README.md). It specifies the checks and retained evidence
needed after [VHDL lowering](03_widths_and_vhdl_lowering.md) and the
[Rocq adapter](04_vhdl_to_rocq.md). No FPGA certificate chain or hardware result
is established by this document.

## 1. The HelloProof analogy and its limit

HelloProof's generated [MemoryMap.v](../../../ship-of-theseus/HelloProof/targets/rocq/MemoryMap.v)
contains addresses, extents and bytes extracted from its emitted ELF using
`readelf`/`objdump`, then proves selected storage, sentinel and layout properties.
Its [trusted-base record](../../../ship-of-theseus/HelloProof/proof-trace/06-the-trusted-base.md)
identifies the extraction as trusted code. This is per-artifact validation of
named obligations, not a compiler-correctness theorem or a proof of all machine
execution behavior.

The demo's Alethe outputs and its Rocq proof are distinct evidence paths. A
certificate being written does not show that an independent checker replayed
it; Rocq acceptance of memory-map lemmas does not establish that replay either.
The FPGA plan retains this distinction and strengthens the admission gates.
The local records are existing assets, not tests rerun for this planning change.

For FPGA output, recover gates, registers, memories and connectivity from the
actual artifact, then prove the selected circuit claims against that model.
Producing `Design.vhd` and a model from the same graph does not validate the
serialization. The FPGA analogue of reading the ELF is reading the emitted
VHDL, netlist or final configuration independently of the emitter.

## 2. Obligation identity and the first boundary

Follow [obligation residency](../Obligation_Residency_Design.md): observable
behavior originates in Baker's graph, with correlated artifact-side obligations.
Only observation-invisible encoding facts originate solely on the artifact side.
Changing output latency, buffer capacity or reset behavior is observable when
the contract exposes it; placement bookkeeping is not automatically invisible
if timing or an externally visible resource constraint depends on it.

Retain graph identities, source spans, selected platform facts and proof
dependencies through [M-01's backend handoff](../PRDs/M-01-DialectAdmission.md).
An identifier appearing in two files is insufficient: check subject, parameters,
formula interpretation and participant correspondence. A newly discovered
observable artifact fact without a graph twin diagnoses missing upstream work.

Retain the original emitted VHDL and dependency closure immutably. If the first
artifact re-check consumes an elaborated or flattened form, check those
transformations or name them as trusted extraction steps. Capture that form
before further optimization and identify the boundary actually checked.
Every later transformed artifact needs a new correspondence or a checked
preservation edge; the initial result must not silently follow an altered file.

## 3. Claims at distinct artifact levels

| Level | Independently recovered facts | Required claim |
|---|---|---|
| Emitted and elaborated VHDL | Actual files, resolved entities, generic instances, register/memory updates and ports | The admitted VHDL semantics refines the settled graph contract |
| Technology-mapped circuit | Actual mapped cells, parameters, connections, clocks and initialization | Primitive-model circuit preserves the admitted RTL behavior |
| Placed/routed configuration or bitstream | Decoded configured resources, routing, LUT contents and storage settings | Decoded implementation corresponds to the accepted mapped circuit under the exact device model |
| Configured board observation | Loaded artifact identity, selected board and recorded measurements | The stated deployment/measurement check passed under its test conditions |

Record each level separately. A passed earlier level cannot be displayed as a
passed later one. Elaboration also has an exact boundary: if synthesis performs
optimization while producing a model, its semantic transformation must be
accounted for rather than called parsing.

At the mapping boundary, admit primitive semantics for every used LUT, register,
carry element, RAM, DSP, clock and I/O cell. A missing model cannot become a
black box whose arbitrary behavior makes the desired equivalence vacuous.
Memory collision, startup and reset semantics are part of this inventory.

Placement/routing legality and static timing have separate evidence. Logical
equivalence does not establish setup/hold margins, electrical constraints or
metastability resolution. Retain clock-domain and environmental assumptions
through [platform realization](06_platform_and_shared_edges.md).

## 4. Proposed proof bundle

The following names are a proposed artifact layout, not current compiler output:

| Artifact | Contents and origin |
|---|---|
| `DesignContract.v` | Receiving-system interpretation of settled graph obligations and admitted component laws |
| `CircuitMap.v` | Data/model recovered from the particular VHDL or later circuit artifact, identifying its level |
| `Correctness.v` | Checked theorem instances relating that model to the contract, with explicit assumptions |
| `manifest.json` | Artifact hashes, tool/profile identities, dependency graph, claim status and coverage |
| `certificates/` | Exact queries and emitted certificates, with replay result and checker identity per obligation |
| `reports/` | Mapping, timing, routing, extraction and deployment records, each tied to its input artifact |

The `.v` files here are Rocq source, distinct from Verilog files that use the same
extension. Bundle manifests identify languages and roles explicitly.

Hash the actual artifact bytes, dependency files, component revisions, parameter
values, platform constraints, extraction output, queries and proof inputs. Include
compiler/pass sequence, adapter, solver, checker, Rocq/library versions and the
exact FPGA part/package/speed grade. State canonicalization rules if semantic
normal forms are also hashed; retain the original bytes.

Hashes bind identities and detect changes; they do not prove correspondence.
The checker must consume the named artifacts and demonstrate the claimed
relation. A cache entry is reusable only while every transitive premise and
semantic adapter dependency remains valid.

## 5. Certificate and theorem admission

Use Composer's [managed proof service](../Proof_Composition_Architecture.md#one-managed-proof-service)
and existing library-admission design. Register circuit semantics and transport
laws there; do not create an FPGA-specific competing proof service. Applications
receive automatic source-linked results through the same CCS projection.

For each SMT leaf, record its actual theory fragment and encoding: arithmetic,
bit vectors, finite arrays/memories, or an explicitly supported combination.
Determine which solver proof rules the receiving checker implements. Alethe is
a proof format, not universal replay coverage for every bit-vector or array
encoding produced by a solver.

An accepted result must reconstruct a Rocq proof or pass an admitted checker
whose soundness connection and implementation trust are recorded. Reject missing
premises, altered queries, unsupported rules and unchecked `hole` steps. Do not
insert `Admitted`, a solver-success axiom, or an uninterpreted equivalence
assumption to finish `Correctness.v`. Audit transitive theorem assumptions
against the selected proof profile's permitted foundations.

Solver verdict, certificate generation, external replay and Rocq kernel checking
remain distinct statuses. A timeout or unsupported replay leaves the associated
claim unresolved. Existing solver-dependent checks can be retained as such;
they cannot be relabeled as independently checked derivations.

## 6. Sequential and elastic correctness

Combinational equivalence compares outputs for all admitted inputs. Stateful
claims require initial-state correspondence and transition preservation, or an
admitted simulation/refinement relation. An invariant for unbounded execution
needs an inductive or otherwise complete proof; bounded model checking alone
establishes only its stated finite horizon.

For a fixed-latency pipeline, a proved latency alignment may justify comparison
after a known number of cycles. The [Dynamatic fork](01_dynamatic_fork.md) also
needs elastic behavior: observe accepted/produced tokens and preserve their
identity, order and multiplicity under allowed backpressure. Equality after an
arbitrary fixed delay is not the general contract.

Trace safety and liveness are separate claims. No loss, duplication or invalid
output does not imply eventual completion. Eventual progress requires the
specific fairness, ready/valid peer, clock, reset and resource assumptions used
by the proof. Expose those premises and reject unsupported progress claims.

Reuse [Colibri contracts](02_colibri_circuit_basis.md) through checked component
state/trace relations. Composition also checks connected assumptions against
their providers; individually verified blocks can still be wired incorrectly.
Cyclic assumption dependencies must close through a global invariant or a sound
compositional rule, never by circularly assuming the desired guarantees.

## 7. Bitstream reconstruction opportunity and limits

Project X-Ray documents the device-specific configuration format and database.
Configuration packets and part identity belong to reconstruction alongside frame
contents; a bitstream header is not a semantic proof. See its
[bitstream description](https://f4pga.readthedocs.io/projects/prjxray/en/latest/architecture/bitstream_format.html).

[fasm2bels](https://github.com/chipsalliance/f4pga-xc-fasm2bels/blob/master/fasm2bels/fasm2bels.py)
provides a concrete research starting point: its bitstream input path runs
`bitread` and bit-to-FASM decoding, then reconstructs cells/connectivity; it can
also emit FPGA Interchange representations. Its
[equivalence tests](https://github.com/chipsalliance/f4pga-xc-fasm2bels/blob/master/tests/test_equivalence.py)
exercise `add32` and `alu` on `xc7a35tcpg236-1` with Yosys equivalence commands.
That is not acceptance of the Arty A7-100T configuration used by HelloArty.

The [upstream README](https://github.com/chipsalliance/f4pga-xc-fasm2bels#bels--sites-supported)
declares resource restrictions, including BRAM without FIFO support and limited
I/O standards. Its original workflow targets Vivado import. The planned open
checker must establish its own supported reconstruction path and primitive
models; tool availability does not establish full-device coverage.

Pin the exact part database and decoder revisions. Inventory every used
configuration feature and resource, including defaults and configuration modes;
unknown or unmodeled features block the corresponding bitstream claim. Database
accuracy, decoder behavior, primitive semantics and their checking connection
remain named dependencies until independently justified.

Decode the final bitstream selected for loading, not merely the producer's FASM
intermediate. Matching reconstructed logic still leaves physical timing, pad
behavior, silicon operation and loader/device integrity under explicit models
and assumptions. Board observation adds evidence at its own scope.

## 8. Acceptance gates

First close one emitted-VHDL case with independently recovered state and exact
instance parameters. Then close one mapped design using only admitted primitives.
Bitstream work starts with a separately named part/resource subset; it is not a
prerequisite for honestly reporting the earlier completed levels.

Required negative gates change a port connection, carry bit, reset/init value,
memory address/collision mode, FIFO latency, token handshake or LUT truth table.
Also reject stale hashes, an altered proof query, a certificate hole, a disallowed
axiom, missing primitive model, wrong part database, unsupported configuration
feature and a liveness claim lacking its required assumption.

At each level, apply mutations that violate the selected contract while leaving
the graph-side proof unchanged: the relevant check must fail or become unresolved.
Byte changes always invalidate a byte-bound result, but a semantically equivalent
artifact may earn fresh evidence. Verify that
invalidation reaches both command-line output and the editor's source-linked
claim. Preserve bounded-versus-unbounded proof status and all uncovered features.

The [roadmap](07_roadmap.md) records closure per artifact level, supported profile
and exact oracle. Hardware-unavailable checks remain pending. This documentation
change runs no synthesis, equivalence, Rocq, certificate-replay or board gate.
