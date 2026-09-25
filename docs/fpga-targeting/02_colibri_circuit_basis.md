# Colibri as the canonical circuit implementation library

**Status: Planned implementation, 2026-09-25.**

This document owns component selection and reuse in the
[FPGA workstream](README.md). The objective is to concentrate circuit realization
on CERN's maintained implementations and verification assets. Colibri is not an
optional peripheral appendix: every admitted FPGA realization follows the same
component policy, including generated logic and the direct HelloArty path.

## A single controlled realization boundary

The [Dynamatic fork](01_dynamatic_fork.md) constructs and transforms an admitted
dataflow circuit. Its backend selects components from a versioned implementation
set centered on Colibri. Each selected operation has a behavioral contract,
parameter constraints, an implementation identity and a preservation obligation.
The [VHDL emitter](03_widths_and_vhdl_lowering.md) serializes that selection.

Colibri does not currently supply every arithmetic operation, elastic control
element, memory controller or FPGA primitive required by Dynamatic. The initial
basis therefore explicitly enumerates Colibri components, the minimum structural
`hw`/`comb`/`seq` operations, and any necessary admitted extensions. An extension
must use the same contract/evidence route; a generator name is not an exemption.
There is no fallback that silently selects unreviewed IP or emits arbitrary HDL.

Retain original Colibri VHDL as the first implementation and regression reference.
Extract only the dependency closure needed for each admitted component. Propose
upstream improvements where useful. A later Clef port remains traceable to that
reference and is admitted by correspondence, rather than by resemblance of code.
This avoids prematurely maintaining a second complete circuit library.

## What is worth reusing

| Family | Colibri assets | Backend use and contract to retain |
|---|---|---|
| Elastic flow | `stream_buffer`, skid and pipeline configurations | Timing-path separation; capacity, fall-through, latency and ready/valid behavior |
| Storage | Synchronous/asynchronous FIFOs, packet FIFOs, RAM/ROM, dual-port and asymmetric memories | Storage realization; occupancy, data order, collision behavior, initialization and inference style |
| Clock boundaries | Reset/pulse/handshake synchronizers and asynchronous FIFOs | Declared crossings with protocol and physical assumptions |
| Data distribution | Round-robin arbiter, interleaver/deinterleaver, broadcaster | Contention, packet boundaries, ordering and progress premises |
| Packet processing | Header/trailer operations, packet join/delay, stream/RAM adapters | Exact accepted payload and metadata transformations |
| Coding and integrity | CRC, scramblers, PRBS/BERT, RLE, 8b/10b | Recognized stream operations with explicit arithmetic, polynomial and framing parameters |
| External interfaces | SPI, I2C, UART, JTAG and selected protocol blocks | Timed controllers behind portable transaction contracts and platform pads |

See the upstream inventories for [common](https://gitlab.com/colibri-cern/colibri/-/blob/master/src/common/readme.md),
[memory](https://gitlab.com/colibri-cern/colibri/-/blob/master/src/memory/readme.md),
[packet](https://gitlab.com/colibri-cern/colibri/-/blob/master/src/packet/readme.md),
[communications](https://gitlab.com/colibri-cern/colibri/-/blob/master/src/comms/readme.md)
and [coding](https://gitlab.com/colibri-cern/colibri/-/blob/master/src/endec/readme.md).
The research revision is pinned in the [series index](README.md#research-and-revision-record).
This inventory does not imply a general numerical/DSP library or interchangeability
with all Dynamatic operation implementations.

## Component admission record

This is required information, not an implemented new descriptor type or an
instruction to create a second semantic registry. Its carrier must be reconciled
with [M-01](../PRDs/M-01-DialectAdmission.md) and established platform identities.

| Required information | Purpose |
|---|---|
| Source/component identity, revision, source closure and license | Reproduce exactly the implementation being claimed |
| Admitted operation and semantic model | Define what a compiler operation means at this boundary |
| Generics and legality conditions | Tie inferred widths, depths, modes and clock parameters to this instance |
| Ports, representation and observations | Identify payload bits, control events, clock/reset and valid data |
| Initial-state and transition relation | Specify reset, enables, simultaneous updates and allowed internal state |
| Capacity, latency and handshake behavior | Preserve flow control, data loss/duplication freedom and permitted timing changes |
| Environmental assumptions | Distinguish input stability, clock/reset discipline and fairness from proved conclusions |
| Properties, proof kind and checked configurations | Separate simulation, bounded checking, induction and parametric theorems |
| Source/graph/circuit correspondence | Connect operation participants to actual component instances and implementation signals |
| Synthesis and target restrictions | Retain memory inference patterns, pad realization and relevant primitive models |

An FPGA payload width is settled upstream. A component requiring a larger physical
port gets an explicit, checked encoding adapter; that does not change the source
value range. A narrower port requires a representability proof. No implicit
truncation, signedness change or default machine-word width is admitted.

## Transfer of proofs and properties

For a literal port, define a relation between reference and replacement states,
show that initial/reset states are related, and show that each admitted transition
preserves the relation and the specified observations. A true state isomorphism
can make this a direct transfer. Retiming, buffering and resource sharing normally
need a more general refinement or accepted-transaction trace relation.

The transferred claim retains all its premises. For elastic channels, observations
usually occur on `valid && ready`; meaningless payload while invalid need not
satisfy the same range invariant. Stability during a stall, packet boundaries,
ordering and capacity are separate obligations. Eventual delivery additionally
depends on the stated progress assumptions. A safe buffer does not automatically
make a cyclic network live.

Composition must discharge the assumptions of each instance from its neighbors
and platform contract. Cyclic dependencies require a global invariant or a sound
compositional rule; components cannot establish each other's premises circularly.
Proving each component separately does not alone prove their wiring, reset
sequencing, clock crossings or global deadlock freedom.
The [Rocq adapter](04_vhdl_to_rocq.md) and
[artifact verifier](05_artifact_verification.md) own those correspondences.

## Verification assets are scoped

Colibri provides VHDL, self-checking simulation, PSL properties and SBY tasks.
Coverage is component-specific. For example, the current memory inventory marks
`fifo` and `ring_buffer` as having formal verification, while asynchronous FIFOs
and dual-port RAM do not. The [FIFO properties](https://gitlab.com/colibri-cern/colibri/-/blob/master/fv/memory/fifo.psl)
check reset, occupancy and pointer behavior at full/empty boundaries. They are
not a complete theorem that every accepted payload emerges exactly once and in
order. Preserve these assets and add the contract needed by the compiler.

The [common verification tasks](https://gitlab.com/colibri-cern/colibri/-/blob/master/fv/common/run.sby)
exercise selected configurations. A successful concrete run is not a theorem
over all widths/depths. Admission either instantiates a checked parametric theorem
or checks the exact selected configuration with supported proof evidence.
SBY success is also distinct from a Rocq-checkable certificate. Never relabel
simulation coverage or a solver verdict as a kernel-checked theorem.

## Preserve physical implementation value

Colibri includes different RAM coding styles to obtain the intended primitive
inference. A behavioral port can preserve values while losing BRAM inference,
latency or area properties. Validate logical behavior and actual mapped resources
separately. Resource reports establish measured configurations, not universal
cost theorems.

CDC and pad components additionally depend on physical premises. Formal digital
models must state clock relationships, synchronization assumptions and reset
discipline. They do not establish analog metastability bounds or board timing by
construction. Generated constraints and post-route timing evidence must correspond
to the same selected platform and artifact.

## First admitted components and rejection gates

Begin with a single-clock `stream_buffer` configuration at a non-machine-word
payload width, plus the arithmetic/register subset needed by HelloArty. Establish
reset, valid-data range, exact accepted payload sequence, stall behavior, latency
and capacity. Introduce synchronous FIFO and arbitration only after their stronger
contracts are expressed. CDC and bidirectional peripheral pads are later gates.

Reject or invalidate evidence when any of these changes without an accepted
preservation step:

- Payload width, bit order, signedness or arithmetic interpretation.
- Component revision, generic values, reset polarity, clock or initial state.
- FIFO depth, fall-through mode, register placement or response buffering.
- A proof assumption, relevant tool option or required primitive model.
- A formerly supported operation becoming an unmodeled external instance.

The application author selects behavior and platform requirements. Component
selection, proof instantiation and failure diagnostics are compiler/framework
responsibilities. Ordinary Clef applications do not acquire VHDL glue or manual
proof annotations to participate in this profile.
