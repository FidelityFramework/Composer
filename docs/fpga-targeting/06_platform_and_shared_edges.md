# Platform contracts and shared compiler boundaries

**Status: Planned — 2026-09-25.** This chapter defines the shared contracts for
the FPGA workstream. It adds no executable backend or peripheral support and
does not reorder the C/A/T/R language and runtime acceptance gates. The latest
[coverage waypoint](../Language_Coverage_Waypoints.md) remains the implementation
record; [the roadmap](07_roadmap.md) owns this workstream's staged acceptance.

## 1. One admission contract across the workstream

[M-01](../PRDs/M-01-DialectAdmission.md) admits an **expression family × selected
platform/backend profile × witness form**. A Dynamatic-derived transformation,
VHDL construct or imported peripheral needs that same record: demand, governing
semantics, Baker owner, graph participants, required evidence, typed Alex form,
backend carrier, rewrite correspondence and acceptance scope.

Baker settles computation, numeric construction, state and protocol relationships.
Alex receives and faithfully expresses those facts through Elements, Patterns
and passive Witnesses. Missing semantics return to their graph or declaration
owner; a target emitter must not infer a protocol from a function name. Target
awareness remains necessary: a spatial circuit and an MCU peripheral transaction
can realize the same admitted contract through different operation sequences.

The demanded part of M-01.b must carry immutable graph facts and proof identities
alongside executable IR. Today's [backend interface](../../src/Core/Types/Pipeline.fs)
does not provide a general obligation-correspondence input. A VHDL printer alone
cannot close that gap. [The fork boundary](01_dynamatic_fork.md) and
[width/lowering contract](03_widths_and_vhdl_lowering.md) must name their readers
and preservation rules for every required fact.

## 2. Preserve inference and graph-owned evidence

[Numeric selection](../../../clef-lang-spec/spec/numeric-selection.md) and
[width inference](../../../clef-lang-spec/spec/width-inference.md) remain the
source contracts. Retain CCS interval facts, selected representations and
intermediate arithmetic requirements through hardware construction. A narrowed
wire is justified by its reachable-value and representation evidence, including
signedness and conversion behavior; an integer width supplied by another tool
does not supersede that evidence.

Keep value range, physical register capacity, pointer representation, memory
transaction width and transmitted word width distinct. HelloESP already sends
an eight-bit SPI payload through a 32-bit MMIO transaction. The same distinction
applies to a narrow FPGA datapath connected to a wider bus or storage interface.
Depth estimates, resource estimates and proved bounds retain different labels.

[Obligation residency](../Obligation_Residency_Design.md) supplies the ownership
direction: obligations about observable behavior originate in the graph and
remain correlated with their artifact checks. Encoding facts can be extracted
at the stage that creates them. Preserve source participants and stable
identities across splitting, fusion, buffering and elimination; record what a
transformation proves about the resulting circuit, rather than retaining an
attribute whose referent disappeared. Historical counts in that design document
describe its original HelloProof exercise, not current coverage totals.

For a circuit invariant, retain the source-to-circuit representation relation,
initial-state establishment and preservation by the admitted transition relation.
State encoding, reset behavior, pipeline latency and transaction acceptance are
part of that relation. [VHDL-to-Rocq](04_vhdl_to_rocq.md) owns the formal model;
[artifact verification](05_artifact_verification.md) owns extraction and checking.

## 3. Platform declarations retain their source identity

Use the implemented [Fidelity.Platform taxonomy](../../../Fidelity.Platform/PLATFORM_STRUCTURE.md)
and [compiler integration contract](../../../Fidelity.Platform/docs/CANONICAL_PLATFORM_SPEC.md).
Their ownership split is already useful for this work:

| Owner | Facts and behavior owned here |
| --- | --- |
| `Hardware/Silicon` | Part/package, available controllers, register requirements, clock capabilities and fabric resources |
| `Hardware/Products` | Installed components, connectors, board routes, oscillator and revision applicability |
| `Protocols` | Reusable transaction meanings, sequencing and communication layouts |
| `Environments` and realization bindings | ABI/runtime services and mechanisms implementing admitted operations |
| `Profiles` and workload | Explicit selection, budgets, controller ownership, rates, mappings and grants |

A selected `[platform] description` names one immutable declaration within its
package's transitive source closure. Keep original declaration identities when
reusing memory spaces, mappings, grants, clocks and pins; equal field values do
not substitute for the same resource. Unselected inventories confer no access.
General hardware instances and multi-target composition remain outside current
[single-target composition](../../../Fidelity.Platform/docs/PLATFORM_COMPOSITION.md).

Contracts and BAREWire are currently distinct, partly overlapping schemas. The
Arty Contracts pin map remains operative even when the selected profile supplies
a BAREWire description. The smaller BAREWire surface does not replace it.
[PlatformBindings.pins](../../../clef/src/Compiler/PSGSaturation/SemanticGraph/PlatformBindings.fs)
joins pin declarations to `[<Pin>]`/`[<Pins>]` field attributes as `Codata.Pins`;
[XDCTransfer](../../src/MiddleEnd/Alex/Traversal/XDCTransfer.fs) consumes that result.
New target constraints must preserve the same identity and selected-source rules.

## 4. Arty is the baseline, with explicit physical boundaries

[HelloArty](../../../HelloArty/README.md) supplies the existing clocked Mealy-machine
oracle. Its [product bindings](../../../Fidelity.Platform/Hardware/Products/Digilent/ArtyA7_100T/ArtyA7_100T.Bindings.clef)
describe the 100 MHz oscillator on E3, LEDs, switches, buttons, USB-UART and
ChipKit digital GPIO. Its [Prelude](../../../Fidelity.Platform/Hardware/Products/Digilent/ArtyA7_100T/ArtyA7_100T.Prelude.clef)
exposes the state/step/clock design shape and pin metadata. Existing bring-up
evidence does not establish the proposed open synthesis and routing flow.

The [project](../../../HelloArty/src/FPGA/HelloArty.fidproj) sets `clock_mhz = 25`.
[FidprojLoader](../../../clef/src/Compiler/Project/FidprojLoader.fs) describes that
setting as an override for depth analysis; [DepthAnalysis](../../../clef/src/Compiler/PSGSaturation/SemanticGraph/DepthAnalysis.fs)
is a structural heuristic. The physical clock declaration remains 100 MHz.
A 25 MHz analysis setting neither inserts a divider/PLL nor proves a derived
clock exists. Reconcile the selected clock, circuit clock generation and timing
constraints before accepting a new timing claim; retain post-route evidence.

ChipKit `InOut` metadata supplies package-pin and electrical facts, not a complete
bidirectional pad implementation. Current SPI/I2C controller contracts, selected
pin routes and pad semantics are still missing for Arty. Pmod coverage is also
outside the current [product binding scope](../../../Fidelity.Platform/Hardware/Products/Digilent/ArtyA7_100T/README.md).
Do not select SPI/I2C pins solely from familiar Arduino names.

An I2C realization must distinguish sampled level, drive-low intent and output
enable, then establish the selected pad's open-drain behavior and pull-up
assumptions. Clock stretching, arbitration and recovery require the capabilities
actually admitted. Clock-domain crossing has its own reset, metastability and
timing assumptions; FIFO presence alone does not discharge them.

## 5. Shared Clef SPI/I2C contracts, distinct realizations

[Colibri](https://gitlab.com/colibri-cern/colibri/) is a source for reusable circuit
implementations and design experience. It is not an existing universal Clef
protocol library. Porting selected behavior into Clef requires an explicit
semantic comparison and conformance evidence for that selection, as described
in [the circuit basis](02_colibri_circuit_basis.md).

The shared layer should describe transaction requests, results, ordering,
completion, bounded storage and errors. SPI needs admitted mode, bit order,
word size, chip-select extent and rate constraints. I2C needs address convention,
start/repeated-start/stop sequencing, acknowledgement and admitted error/recovery
behavior. Device-specific register addressing and byte order remain explicit.

| Realization | Mechanism and separate acceptance |
| --- | --- |
| FPGA timed kernel | A Clef state machine realizes serial edges, sampling, counters and pads under the selected clock/reset contract; synthesis and circuit correspondence are checked. |
| MCU hardware controller | A Clef driver programs the selected controller through declared MMIO, waits or handles completion, and satisfies the shared transaction contract under that device's semantics. |
| Optional software-driven pins | A separately selected implementation requires its own timing and interference evidence; MCU support does not imply this implementation. |

The reusable transaction layer does not require every MCU to execute the FPGA's
serial-edge machine in software. Conversely, a controller register map does
not supply a proof that its driver implements the shared contract. Selected
bindings own controller capabilities, pin multiplexing, clock setup and legal
accesses; workload/profile selection owns sharing and resource allocations.

There is concrete native standing art: [HelloESP's SPI driver](../../../MCU/Espressif/CCC2026Badge/HelloESP/src/Spi.clef)
uses ESP32-S3 SPI2, mode 0, MSB-first, write-only transfers, one byte through W0,
at a configured 8 MHz. [Display](../../../MCU/Espressif/CCC2026Badge/HelloESP/src/Display.clef)
consumes that byte operation for ST7735S commands and data;
[Access](../../../MCU/Espressif/CCC2026Badge/HelloESP/src/Access.clef) supplies explicit
mappings, grants and `Mmio.bind32`. Its [hardware record](../../../MCU/Espressif/CCC2026Badge/HelloESP/README.md)
establishes the narrow badge workload. Polling is presently unbounded; receive,
general modes and a reusable transaction/error API are further work.

The [STM32H7 roadmap](../../../Fidelity.Platform/docs/STM32H7_DRIVER_ROADMAP.md)
likewise calls native I2C4 control unimplemented. Its one-owner, bounded-transfer,
7-bit-address and named-error requirements inform the shared contract without
claiming an existing generalized I2C implementation.

## 6. BAREWire and one source-linked proof service

[BAREWire](../../../BAREWire/README.md) supplies shared layout, representation,
buffer and transfer contracts. Those can describe a controller request or a
hardware/software handoff without making SPI or I2C carry a BAREWire envelope.
The device's own on-wire command framing remains authoritative. Proof identities
and dimensions remain compiler evidence; they need not become payload tags.

Follow [Proof Composition](../Proof_Composition_Architecture.md): one managed,
pinned Rocq service alongside supported solver dispatch, with registered rules
and explicit semantic adapters. FPGA checking extends that service; it does not
install a second application-facing proof system. Evidence is a derivation,
never an axiom created from a solver success flag or an emitted certificate.
Kernel/library versions, encodings, queries and remaining device assumptions
are part of each result's dependencies.

[Lattice integration](../Lattice_Integration.md) consumes the same CCS projection
as command-line checking. Show source participants, selected platform, inferred
width justification, law, checked revision and pending premises. Invalidate
dependent evidence when a clock, pin route, arithmetic construction, controller
contract or source snapshot changes. Distinguish inferred facts, solver verdicts,
certificate emission and checked derivations; retained evidence opens without
rebuilding. Missing coverage remains visible at its commitment boundary.

## 7. Reuse the method across targets; stage implementation narrowly

The common asset is the obligation graph, admitted proof service and artifact
adapter contract. CPU/MCU code needs instruction, MMIO and memory-model semantics;
GPU code needs launch/address-space/synchronization semantics; NPU and CGRA work
needs tile, route, DMA/FIFO and scheduling semantics. Their artifacts may be
ELF, device images or configurations. Neither a shared container format nor a
passing layout proof establishes complete execution correctness on every target.

Start with the HelloArty circuit baseline, a bounded C-free DHLS oracle and a
stream/buffer example under separately stated cycle or transaction observations.
Do not equate dynamic token scheduling with a cycle-exact board machine.
General peripheral contracts, board routes and CDC follow their own admission
gates. This provides shared compiler infrastructure as demanded, while leaving
the wider language/runtime priorities and unsupported target claims unchanged.
