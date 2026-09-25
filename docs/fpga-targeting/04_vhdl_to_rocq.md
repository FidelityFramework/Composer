# VHDL to Rocq adapter

**Date:** 2026-09-25. **Implementation status:** Planned.

This document defines Composer's own open adapter from the admitted VHDL
artifact to a Rocq circuit model. It belongs to the
[FPGA targeting plan](README.md), uses the
[Colibri circuit basis](02_colibri_circuit_basis.md), and consumes the artifact
contract from [widths and VHDL lowering](03_widths_and_vhdl_lowering.md).
It adds no implemented translator, admitted theorem, or passing circuit gate.

## 1. Ownership and purpose

Baker settles observable semantics and their obligations in the PSG. Alex
observes that settled structure; the backend expresses the selected circuit.
The adapter independently reads the resulting VHDL and its resolved dependency
closure. It does not choose new widths, scheduling, reset behavior, or protocols.
Those boundaries follow [M-01](../PRDs/M-01-DialectAdmission.md) and
[obligation residency](../Obligation_Residency_Design.md).

The required result is a checked correspondence between the graph's contract
and the circuit described by the actual emitted files. Emitting VHDL and Rocq
from the same internal graph supplies two outputs, but does not establish that
the VHDL implements the model. The artifact reader and its semantic connection
are explicit checking dependencies.

Colibri remains the canonical circuit implementation basis. Its native Clef
port, reference VHDL, contracts, and admitted proof instances belong to one
versioned component family. The adapter is verification infrastructure for that
basis, not a competing circuit library or an alternative source type system.

## 2. VHDL2Rocq precedent and availability

Sankur, Boyer and Faissole's VMCAI 2026 paper describes **VHDL2Rocq**, an internal
Python 3 tool using pyGHDL and Z3. Section 6 states: “We cannot publish our tool
at this point since it is proprietary due to company policy.” Composer cannot
adopt that implementation as an open dependency. See the
[author manuscript, §§2–7](https://hal.science/hal-05369274/file/main.pdf) and
[publication record](https://link.springer.com/chapter/10.1007/978-3-032-15700-3_14).

Its supported designs are combinational, without `process` statements; integer
generics remain symbolic. Supported statements include concurrent assignments,
component instances, and `if`/`for generate`. Binary `std_logic`, ranged integers
and arrays are modeled. SMT checks cover symbolic dimensions, accesses,
assignment coverage and dependencies. Integer assignment ranges generate Rocq
obligations for users to prove; intermediate overflow uses interval analysis.
Translation produces functional Rocq definitions and an IEEE-operation library.

Section 7 handles a fixed pipeline separately through unrolling and ABC
equivalence checking. This is not general sequential VHDL support. The paper
does not establish a mechanically verified Python implementation or Rocq replay
of its Z3/ABC results. Its reusable contribution here is the methodology;
Composer's sequential extension and certificate admission remain new work.

## 3. Admission profile

Begin with the forms the selected backend actually emits and the selected
Colibri kernels require. Publish the exact syntax, typing and semantics profile
with each adapter revision. “VHDL-2008” identifies the serialization standard;
it is not a claim to model every construct accepted by a VHDL compiler.

| Proposed admitted form | Required interpretation |
|---|---|
| Entities, one resolved architecture, explicit instances | Exact generic values or symbolic parameter constraints, port directions, dependency hashes |
| Binary scalar/vector logic, signed and unsigned arithmetic | Bit order, widths, extension, truncation, comparison and shift rules |
| Ranged integers and bounded arrays | Declared range, index direction, conversion and intermediate bounds |
| Concurrent assignments and static generates | Complete drivers, dependency order, no unintended combinational cycle |
| Selected clocked processes/register components | Old-state sampling, simultaneous updates, clock edge, enable and reset priority |
| Selected memories and FIFO kernels | Depth, initialization, read latency, write enables and collision semantics |

The sequential rows extend beyond the paper. They require their own operational
semantics and admission proofs. Start with one clock domain and one explicitly
declared reset discipline. General event-driven processes, arbitrary waits,
delays, shared variables and unrestricted multi-clock designs stay outside the
initial profile. Missing coverage produces an unresolved prerequisite.

Do not silently collapse the nine `std_logic` values to Boolean values. A binary
profile must establish defined inputs and drivers and account for initialization.
Unresolved `U`, `X`, `Z`, weak values or don't-care behavior rejects that profile.
Open-drain and bidirectional pads require a separately admitted electrical/pin
interface contract, including sampled input and output-enable meaning; they
cannot be erased as ordinary wires. See [platform edges](06_platform_and_shared_edges.md).

Uninitialized storage may be modeled nondeterministically only where that
semantics is explicitly admitted and the claimed properties quantify over every
allowed initial value. Inventing zero initialization is never a repair.

## 4. Circuit model schema

Use a small versioned semantic vocabulary shared by the importer, component
contracts and artifact checkers. The following is a schema, not implemented API:

| Field | Meaning |
|---|---|
| `Parameters` / `WellFormed` | Generic domains, dimensions and construction prerequisites |
| `State` | Registers, finite memories, controller state and retained protocol data |
| `Inputs` / `Environment` | Sampled values, admitted clock events and environmental constraints |
| `Init` | Predicate over all permitted initial states and reset conditions |
| `Step` | Transition relation; a total function where the profile is deterministic |
| `Observe` | Pins, transferred tokens, accepted requests/responses and specified errors |
| `Assumptions` | Clock/reset, peer protocol, memory and device premises |
| `Origin` | Component, artifact and source/PSG identities with revisions |

Evaluate combinational logic from the current state and inputs, then apply the
specified state update simultaneously. VHDL signal assignment and immediate
variable assignment must not acquire the same meaning accidentally. Separate
the sampling event from physical propagation delay; timing admission is another
claim, with its own constraints and evidence.

For elastic circuits, observations include transfers when `valid` and `ready`
hold at the admitted sampling event. Preserve token identity, ordering and
multiplicity, including cancellation/error behavior where specified. A proof of
equal arithmetic results alone does not establish the protocol refinement.

## 5. Parametric proofs and concrete instances

Retain a theorem over every parameter valuation satisfying `WellFormed` when a
component family supports such a proof. Discharge width compatibility, array
bounds, driver completeness and dependency constraints under those hypotheses.
Retain counterexample parameter valuations in diagnostics when a check fails.

Every emitted instance still needs validation: selected generic values, resolved
architecture, exact port association, casts, library binding and reset/clock
wiring must instantiate that theorem. A generic theorem for depth greater than
one does not cover a depth-one specialization without its own admitted rule.
Instantiation identifies the actual artifact, not only the generator template.

Range lemmas must cover intermediate operations and conversions as well as final
values. A representable final output does not justify an overflowing intermediate.
Use the existing range/proof machinery and its accepted evidence; the adapter
does not invent a second numeric-selection policy.

## 6. Colibri correspondence and proof reuse

For an isomorphic Clef port, establish an explicit relation `R` between reference
VHDL state and ported state. Check initialization, preservation of `R` by each
corresponding step, and equality of observations. A claimed state isomorphism
also needs its inverse laws on the admitted states. Names and matching diagrams
do not establish these properties.

Keep a cycle-preserving kernel port as the initial correspondence target. A later
retiming, buffering or elastic scheduling change needs a separately admitted
trace refinement, including any allowed stuttering. Do not describe a changed
controller as the same kernel merely because its final data values match.

Import each selected Colibri contract with its assumptions, parameter coverage,
reset semantics, upstream revision and proof method. Translate supported PSL
properties into the receiving model and check that translation. Rerunning an
upstream formal harness is useful evidence; its success is not automatically a
Rocq theorem. Transport a proved property only through an admitted state/trace
correspondence and discharged premises. Simulation-only components retain that
status until their required proof gate is implemented.

## 7. Adapter trust and implementation

The first prototype may use a pinned GHDL/pyGHDL parser or elaborator, provided
its role and transformations are recorded. [GHDL's synthesis interface](https://ghdl.github.io/ghdl/using/Synthesis.html)
is an available extraction route, not a correctness theorem for this adapter.
Parsing original VHDL, importing an elaborated design and importing a synthesized
netlist are different boundaries; identify which claim each check establishes.

A restricted, checked importer or a certificate-producing import with a small
sound checker is the intended route to reducing trust. Until that connection is
established, label the parser, elaborator, normalizer and model translation as
trusted dependencies. Differential simulation is a development check, not a
replacement for their semantic correspondence.

Choose the implementation through Composer's existing compiler integration,
Clef-native direction and type-preserving boundaries. The paper's Python choice
does not prescribe a permanent integration language. Reuse current orchestration
and typed contracts; avoid shell-driven parallel proof infrastructure.

Use the [managed proof service and library admission](../Proof_Composition_Architecture.md#one-managed-proof-service).
Pin Rocq, semantic libraries, importer and checker revisions; register circuit
laws in the same dependency graph as other obligations. The editor consumes the
shared CCS projection. No additional application-facing service or Lean
dependency is introduced. Timeouts leave claims unresolved.

## 8. Acceptance and rejection gates

The first accepted profile must demonstrate a parameterized combinational
component and a sequential Colibri-derived kernel. Produce models from emitted
files, check their declared correspondences, and retain all assumptions.

Required negative cases include changed signedness, truncated carry, reversed
array direction, invalid generic boundary, missing/multiple driver, combinational
cycle, wrong reset priority, invented initialization, changed FIFO latency,
unresolved pad values, stale component hashes and unsupported syntax.

Changing emitted VHDL must invalidate its artifact-bound evidence even when the
PSG and its design proof are unchanged. A mutation that violates the selected
contract must fail rechecking; a changed but equivalent artifact may earn new
evidence. Apply the same gate to a ported kernel's reference correspondence.
A rejected form must not be
silently omitted, replaced by an assumption, or emitted as an uninterpreted stub.

The deliverable records exact admitted forms and theorem assumptions, source and
artifact provenance, positive/negative results, and remaining trusted adapters.
Continue through [artifact verification](05_artifact_verification.md); passing
this adapter gate alone establishes neither mapped-circuit nor bitstream claims.
This planning change has not run those acceptance gates.
