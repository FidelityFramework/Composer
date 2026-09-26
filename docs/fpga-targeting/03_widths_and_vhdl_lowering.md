# Width preservation and closed VHDL-2008 lowering

Status: **Planned**. Planning record: **2026-09-25**. This design adds no emitter,
admitted operation or proof-preserving backend implementation. Preserve the
existing CIRCT/SystemVerilog pathway and HelloArty evidence while developing
the [owned Dynamatic fork](01_dynamatic_fork.md).

## Existing seam and governing contract

[TypeMapping](../../src/MiddleEnd/Alex/CodeGeneration/TypeMapping.fs) reads
`RangeAnalysis.heldWidth` and settled aggregate layouts from CCS. The hardware
[witness](../../src/MiddleEnd/Alex/Witnesses/HardwareModuleWitness.fs) and
[patterns](../../src/MiddleEnd/Alex/Patterns/HardwareModulePatterns.fs) produce
CIRCT structure. This is an existing implementation seam to reconcile with the
[portable middle-end boundary](../../../clef-lang-spec/spec/backend-lowering-architecture.md#2-portable-middle-end-target-committing-backend),
which places target-specific CIRCT operations in FPGA backend realization. New
target work must not extend that seam as an architectural exception. Current
[lowering](../../src/BackEnd/CIRCT/Lowering.fs) maps
residual arithmetic to `comb`, canonicalizes/CSEs, then lowers through `sv` and
calls `export-verilog`. None of that establishes a VHDL preservation result.

[M-01](../PRDs/M-01-DialectAdmission.md) supplies the admission and information
transport contract. Its M-01.b work must extend the current
[backend handoff](../../src/Core/Types/Pipeline.fs), which principally accepts
MLIR text and target configuration, with the required immutable graph/evidence
projection. The [witness boundary](../Witness_Boundary_Audit.md) remains binding:
proof obligations reside in the semantic graph; serialization does not invent
them or move semantic authority into a parallel circuit-proof IR.

## Width and arithmetic invariants

CCS/Baker settles value ranges, held widths, operation widths, dimensions,
representation choices and conversion obligations. Alex faithfully observes
those decisions. The VHDL emitter preserves them; it does not re-run range
inference, choose a convenient CPU word or infer signedness from source names.

- A signless MLIR `iN` is an N-bit carrier. Signed/unsigned comparisons,
  extensions, division and shifts are determined by their admitted operations.
- Fixed-width `comb.add`/`comb.mul` results are bit-vector results. Prove that
  chosen widths preserve the source arithmetic, or retain the source's explicit
  modular operation. Never justify accidental truncation as an optimization.
- An intermediate may need more bits than either stored operand/result. Consume
  Baker's extension, operation-width and narrowing plan in order.
- The missing-width sentinel is an error, not a zero-width VHDL object or a
  default integer. Aggregates retain settled field order, widths and offsets.
- Mathematical range facts and physical register widths remain distinct;
  truncation, slicing and width-changing interfaces retain their own evidence.

Use `ieee.numeric_std` with explicit casts and size operations for the admitted
two-state datapath. Do not route arbitrary N-bit payloads through VHDL `integer`
or `to_integer`: their range is not a substitute for the circuit's representation.
Name escaping, literal sizing, zero/sign extension and concatenation order need
deterministic, reviewed rules.

## Closed operation/profile matrix

The table is a **planned admission checklist**, not current emitter support.
Each enabled row needs exact pinned dialect forms, a semantic rule, source and
graph prerequisites, VHDL construction, correspondence and discriminating tests.
All operations, types, attributes and nested forms outside enabled rows fail.

| CIRCT family | VHDL realization and admission requirements |
|---|---|
| `hw.module`, `hw.output` | Entity/architecture and port assignments; explicit order, direction, width and clock type handling. |
| `hw.constant` | Exactly sized bit-vector literal, including high-bit and negative two's-complement cases where applicable. |
| `hw.instance`, `hw.module.extern` | Declared entity/library binding, generic substitution and port ABI; no implicit component discovery. |
| `hw` arrays/structs and access operations | Admit separately with fixed shapes and explicit packing/index rules; no blanket aggregate support. |
| `comb.and`, `or`, `xor`, `add`, `mul` | Preserve variadic operands, result width and bit-vector behavior; explicit intermediate sizing. |
| `comb.concat`, `extract`, `replicate` | Preserve operand order, bit numbering, bounds and repetition count. |
| `comb.mux` | Exact condition/result widths and selected two-state/extended-value semantics; distinguish selection from VHDL simulation unknowns. |
| `comb.icmp` | Enumerate predicates; signed/unsigned operands and equality variants use their declared semantics. |
| Division, remainder and shifts | Separate signed/unsigned forms, zero divisor, signed minimum/-1, rounding/remainder convention and oversized shift behavior. |
| `seq.compreg`, `seq.compreg.ce` | Only exact admitted clock/reset/enable/initialization forms; state semantics described below. |
| Memory and analog/pad forms | Realize through a separately admitted memory/pad/component contract; `hw/comb/seq` presence alone does not admit them. |

Primary definitions: [HW](https://circt.llvm.org/docs/Dialects/HW/),
[Comb](https://circt.llvm.org/docs/Dialects/Comb/) and
[Seq](https://circt.llvm.org/docs/Dialects/Seq/).
Pin the actual definitions and transformation implementations with the toolchain;
these moving documentation links do not freeze semantics.

Division, remainder and oversized shifts must not inherit VHDL host/operator
behavior accidentally. A seed emitter may reject these forms until their
operation-specific conditions and tests exist. Where the source excludes a case,
the exclusion is a checked prerequisite or explicit environmental assumption.
Unknown/don't-care behavior must have a declared formal interpretation. A
`twoState` flag and a VHDL `std_logic` declaration are not interchangeable claims.

## State, clocks and reset

The first state profile should name a single rising-edge clock and a restricted
reset/enable form. This restriction is planned scope, not a reinterpretation of
every `seq` operation. Admission must reconcile the pinned operation semantics,
selected implementation and source reset contract before emission.

Record clock identity and polarity; reset kind, polarity and priority; reset
value; enable/hold behavior; initialization and permitted unconstrained state.
For a reset-plus-enable register, confirm whether reset wins when enable is low.
Power-up initialization, synchronous reset and asynchronous reset are different
behaviors. An absent reset/init is not permission to initialize everything to
zero. FPGA configuration initialization requires a supported physical mapping.

The witness may select an already admitted form using settled platform facts;
it may not create reset synchronizers, clock dividers or a new scheduling plan
to repair missing semantics. Multi-clock designs require explicit crossing and
reset-domain contracts. Synchronizer attributes express implementation intent;
they do not constitute a digital proof of analog metastability behavior.

## External circuits, memory and pads

Every external entity needs a versioned binding describing library/entity and
architecture where relevant, generic names/types/values, flattened or structured
port ABI, numeric interpretation, latency, reset and clock roles. Its behavioral
contract, source provenance and selected proof/model evidence travel together.
Unresolved externs cannot pass formal checking as unconstrained black boxes
while being reported as proved implementations.

[Colibri realizations](02_colibri_circuit_basis.md) include resource-sensitive
choices: first-word-fall-through versus registered FIFOs, RAM output registers,
mixed-width ports and read-during-write behavior. Preserve the selected choice
through GHDL/Yosys and inspect the mapped primitive configuration. A valid VHDL
array does not establish block-RAM inference or compatible collision semantics.

Pad realization belongs to the platform boundary: separate input sampling,
output data and output enable before resolving a top-level bidirectional pad.
Open-drain operation means pull low or release, with actual pin readback; it is
not an ordinary Boolean output. Pin identity, electrical standard, pull-up
wiring and external timing requirements come from the selected product/profile.
See [platform/shared edges](06_platform_and_shared_edges.md).

## Pass isolation and serialization

Publish one permitted pass sequence for each admitted pathway. The VHDL route
must end in its supported structural operations and component bindings before
serialization. Do not invoke `lower-seq-to-sv`, `lower-hw-to-sv` or `export-verilog`
as hidden prerequisites. Reject residual `sv` operations, procedural fragments,
unrealized casts and unknown attributes requiring semantic interpretation.

Validate input and post-pass MLIR using the pinned verifiers plus the stricter
operation/profile gate. Neither a verifier success nor canonicalization/CSE
alone proves source preservation. The exporter should construct a typed VHDL
representation with explicit declarations and dependencies, then serialize it;
it must not recover semantics through source symbol matching or raw text repair.
GHDL analysis/elaboration is a further gate, not the whole correctness check.

## Correspondence and evidence transport

Transport graph node/value identities, obligation IDs, source origins, selected
width facts and implementation contracts through the shared M-01 handoff.
VHDL attributes may expose identifiers to tools that preserve them, but are not
the proof payload or the only evidence carrier. Bind generated artifacts to the
authoritative graph projection and the exact tool/profile manifest.

Correspondence is many-to-many: one semantic value can split across bits/cells,
several values can share optimized logic, and a proved constant can disappear.
Track transformations and justification for these outcomes instead of promising
stable signal names or one source node per register. Preserve explicit origins
for added state, buffers, resets, adapters and interface conversions. Rewrites
must not silently discard an obligation when its original operation disappears.

[VHDL-to-Rocq](04_vhdl_to_rocq.md) defines checked circuit interpretation and
certificate replay; [artifact verification](05_artifact_verification.md) extends
the account to synthesized and final artifacts. A checked descriptor or retained
attribute is not, by itself, an equivalence theorem for the generated circuit.

## Acceptance boundary

Retain width-sensitive HelloArty behavior as a baseline. Add small discriminating
cases for a nonnegative counter without an unnecessary sign bit, signed boundary
comparisons, an intermediate wider than its result, extension/truncation, mux
selection, reset/enable priority and initialization. Include widths exceeding
VHDL's implementation integer range and unsupported-operation rejection.

For each newly admitted row, retain source/graph facts, pre/post-pass MLIR,
VHDL, elaboration result, semantic comparison and correspondence evidence.
Mutating a width, signed predicate, reset priority, generic or entity binding
must invalidate the appropriate check. Simulation, induction/certificate replay,
resource mapping and board execution remain separately recorded gates.
Follow the [roadmap](07_roadmap.md); this plan closes none of those gates.
