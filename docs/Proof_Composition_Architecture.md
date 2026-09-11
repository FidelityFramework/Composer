# Proof composition and the Rocq toolchain

**Status:** engineering design and tooling research, 2026-09-11. The existing compiler/editor proof slices are described in [Lattice integration](Lattice_Integration.md). The library integrations and composition gates below are proposed; upstream proofs do not establish their integration into Fidelity.

## Purpose and documentation ownership

Fidelity can reuse substantial mechanized foundations for concurrent resources, message protocols, distributed transformations, probabilistic relations and machine execution. This changes the implementation opportunity: framework authors can build semantic adapters and reusable domain rules on existing proofs instead of developing every underlying logic themselves. The intended result is automatic, source-linked verification across supported substrates, from distributed services to freestanding firmware.

This document owns the Composer integration decisions, capability assessment and engineering gates. The [verification internals](https://clef-lang.com/docs/internals/verification/proof-composition-and-tooling/) explain their relationship to Clef's PSG and mode shifts. [Conformance §6.1](https://clef-lang.com/spec/draft/conformance/#61-verification-evidence-and-composition) states the tool-independent evidence contract. The working *Decidable by Construction* manuscript, §5.3, develops the proof-theoretic scope; its research record is `arxiv-papers/research/DBC/proof-composition-and-tier-four.md`. None of these documents establishes an implementation milestone by describing it.

## Decisions retained and reconciled

| Keep | Extend or replace |
|---|---|
| Clef's ML-family inference, dimensional facts and PSG coeffects | Connect supported resource and protocol judgments to those same program identities; do not introduce a replacement source type system |
| Automatic obligation generation and the existing mode-shift design | Carry checked theorem instances between modes, with explicit semantic mappings and transitive premises |
| cvc5 for the supported solver fragments | Reuse Rocq-founded libraries where arithmetic solving alone cannot justify the judgment |
| Clef Proofs as the application-facing inspection surface | Show composed evidence, its scope, assumptions and freshness through the same source links |
| Quotation-based law authoring for framework/domain authors | Also admit registered Rocq theorems through a checked binding; a handwritten Clef quotation need not wrap every upstream theorem |
| Tier 3 reusable parameterized domain lemmas; Tier 4 relational judgments | Include supported concurrent/distributed domain theorems at Tier 3; replace the categorical claim that Rocq enters the trust base only at Tier 4 |
| Fidelity.Platform declarations and BAREWire layout/representation contracts | Supply the device, memory-order, transport and failure semantics needed by the selected proof model |

Application developers write their application and select meaningful domain requirements. The compiler derives the supported obligations, identifies registered laws, instantiates them and checks their premises. Ordinary use does not require a proof attribute, a theorem choice, a Rocq file or a second editor extension. Theorem development belongs to framework and domain-library authoring, except where a developer introduces a genuinely new domain requirement. An optional analyzer may suggest a changed requirement or a repair; it does not authorize or enable checks that should already be automatic.

That authoring community will initially be small, beginning with the framework's author. Admission should therefore grow incrementally around actual framework uses: one checked law, a supported construction, explicit coverage and regression cases. Upstream reuse reduces the foundational work per addition. No milestone should presume that a large community first supplies a comprehensive theorem catalogue. Later contributors use the same admission and maintenance requirements, and application consumers benefit without taking on that authoring role.

## Four tiers, explicit proof dependencies

| Tier | Role | Typical evidence |
|---|---|---|
| 1 | Admitted structural inference | Dimensional substitutions, structural/resource derivations under their specified rules |
| 2 | Local conditions in supported analysis/solver fragments | Range, representability, layout and ordering conditions, including suitable wait-for rank checks |
| 3 | Parameterized domain and system theorem applications | Resource handoff, protocol invariants, fault-model preservation, restricted probabilistic or termination results with checked premises |
| 4 | Relational reasoning about executions, distributions or realizations | Derivations in the supported compiler-relational (`cRHL`) or probabilistic-relational (`pRHL`) discipline and other explicitly admitted relational rules |

This is an organization of obligations, not a requirement to visit every tier in order. Concurrency is not wholly Tier 3: a local ownership rule may be structural, a queue-capacity condition arithmetic, and a refinement between implementations relational. A distributed transformer may provide a relational foundation for a Tier 3 invariant-preservation theorem. The instance retains that dependency.

A tier number does not identify the trusted computing base. A Tier 3 theorem proved in Rocq still depends on that foundation unless an alternative accepted derivation replaces it. Checking only its arithmetic premises with cvc5 does not remove this dependency. Conversely, an arithmetic result needs no unrelated distributed library merely because its consumer is an actor.

## Evidence is a derivation, never a new axiom

The proof profile declares its permitted foundational axioms and accepted checking mechanisms. Library admission checks the theorem and its transitive assumptions against that profile. Tool-generated evidence is used as a proved lemma; it is never admitted as an axiom because a solver returned `unsat`, a package was installed, or a certificate file exists.

Three records must remain distinguishable:

- **Logical foundation:** the permitted axioms and the checked laws built on them.
- **Execution premises:** the device, host, ingress, failure and fairness assumptions under which the program claim holds. These remain visible hypotheses or justified contracts, not hidden extensions of the logical foundation.
- **Checking dependencies:** the kernel, solver or independent checker, encodings and semantic adapters actually relied upon. A trustworthy proof of the wrong encoding still says nothing about the intended program.

A theorem instance needs its law identity, actual parameters, source/PSG participants, discharged premises, remaining hypotheses, conclusion, meaning in the receiving mode and checked evidence. A lower-tier certificate must either reconstruct a proof acceptable to that receiving discipline or pass a checker with an admitted soundness connection to it. An external checker's success flag cannot simply become a Rocq axiom. Unsupported reconstruction stays unresolved. Existing solver verdicts remain honestly labelled as solver-dependent results; they must not be relabelled as independently checked certificates.

Mode shifts are the existing mechanism for this composition. A proved implication may transport a numerical result into a protocol premise, or a resource theorem into an access permission. An adjunction alone does not prove that a selected Clef encoding has these meanings, nor imply that every round trip is invertible. The semantic interpretation and applicable laws need their own proofs. Linear ownership evidence cannot be duplicated merely because the implementation stores proof references in a graph.

## What the upstream work supplies

These are reuse candidates with different roles, not seven application-facing verification tools.

| Component | Demonstrated upstream contribution | Fidelity integration and overlap |
|---|---|---|
| [Iris](https://iris-project.org/) | A Rocq-verified framework for higher-order concurrent separation logic | Candidate foundation for ownership, invariants and resource composition. Requires an interpretation of the relevant Clef operations, states and observations, plus an adequacy connection to execution |
| [Actris](https://iris-project.org/actris/) | Dependent separation protocols for message-passing reasoning over Iris | Candidate laws for message order and resource handoff. Does not infer Clef protocols or connect them to Prospero automatically |
| [Aneris](https://github.com/logsem/aneris) | Distributed separation logic for partial correctness and refinement | Candidate for network-facing invariants under its stated network semantics. Overlaps Actris at protocol reasoning; neither name alone supplies liveness |
| [Verdi](https://github.com/uwplse/verdi) | Mechanized distributed systems and verified system transformers | Candidate for preservation across declared fault models. Its handler/extraction/shim route is not Fidelity's execution route; using its theorems requires a correspondence, not adoption of its runtime |
| [Clutch](https://clutch-project.org/logics-and-examples.html) | A family of Iris-based probabilistic logics, including relational reasoning by couplings | Candidate foundations for selected Tier 4 laws. Its judgments are not automatically identical to Fidelity's pRHL or cRHL; probabilistic error bounds are not floating-point roundoff bounds |
| [Islaris](https://github.com/rems-project/islaris) | Machine-code reasoning combining Iris with Isla and detailed ISA semantics, including MMIO examples | Evidence that this proof approach reaches device interaction. Existing Armv8-A/RISC-V coverage does not establish Cortex-M33, RA6M5, x86_64 or FPGA coverage |
| [RefinedC](https://plv.mpi-sws.org/refinedc/) | Automated foundational verification with restricted, structured proof automation over Iris | An engineering precedent for predictable automation. Fidelity retains its PSG-derived obligations and application workflow rather than importing RefinedC's annotation model |

Iris supplies logical machinery; Actris and Aneris build more specific program logics. Verdi supplies a different system-model/transformer approach. They need not all appear in one proof. Choose one semantic route for an initial obligation family, and add another only when it provides coverage worth the bridge. Shared Rocq implementation language does not make the models interchangeable.

An imported theorem may bypass reauthoring its proof or quotation. It cannot bypass declaring its parameters, proposition, semantics, permitted assumptions and correspondence to a PSG operation. Resource algebras and ghost state can express logical ownership without allocating corresponding runtime objects. Actual counters, logs, queues and retries still incur storage and execution costs. Iris resource structure also does not identify Clef's fractional values with permissions by itself.

The existing [McErlang comparison](https://clef-lang.com/docs/design/concurrency/the-three-layer-actor-contract/#fit-with-the-existing-architecture) remains useful: model exploration, structural checks and reusable deductive proofs establish different scoped results. A wait-for rank addresses the represented synchronous dependencies; it is not a proof of every form of distributed progress. Library reuse extends the available judgments without discarding the cheaper graph-derived checks or requiring model exploration for every edit.

## First composition example: a distributed numerical join

The strongest initial demonstration connects the numerical guarantees motivating the JavaScript work to actor and protocol guarantees. A supported partition-and-join construction should generate these obligations automatically:

1. **Identify the job and its partition.** Every result refers to the same input snapshot, computation version and partition scheme. Partitions are disjoint and cover the declared input.
2. **Prove each local computation.** For an exact accumulator, prove input encoding, product or term formation, capacity of every reachable partial state, exact merge and the final rounding contract. A compensated or fixed-tree float reduction instead carries its own accuracy/reproducibility theorem; it is not silently promoted to exact accumulation.
3. **Prove resource handoff.** Sender and receiver permissions, arena lifetime, suspension and completion refer to those same buffers. Prospero owns actor/arena orchestration; Ariel supplies eligible execution turns; Olivier actors perform the work.
4. **Prove acceptance and merge.** Accepted results belong to the current job, and a repeated partition result cannot contribute twice. Mathematical identities and set membership must agree with the actual encoded keys and state transitions.
5. **Account for failures.** Durable deduplication state and the corresponding merge update must satisfy the chosen recovery/atomicity contract. A retry-safe operation is not thereby guaranteed eventually to complete. Delivery, fairness and termination require separate premises and results.

The conclusion can then be specific: under the declared ingress, arithmetic, transport and recovery contracts, every completed accepted join equals the stated sequential specification, or satisfies the selected error relation. Completion itself is a further claim.

Verdi's [sequence-number correctness development](https://github.com/uwplse/verdi/blob/master/theories/Systems/SeqNumCorrect.v) is a useful concrete reference: it proves a simulation from duplicate-message executions back to an asynchronous model, supporting invariant transfer. Its [sequence-number implementation](https://github.com/uwplse/verdi/blob/master/theories/Systems/SeqNum.v) uses mathematical natural numbers and retained identifiers. A finite target carrier therefore needs a bound or a proved epoch/reuse discipline, including delayed messages and retained state. Crash recovery and bounded storage must be established for the actual application; the duplicate-suppression theorem alone does not supply them.

This example makes the joint constraint consequential. Independent proofs of a buffer bound, a valid protocol and an accurate sum are insufficient if they concern different buffers, job epochs or terms. The PSG must connect the same participants across modes and lowering stages. The richer obligations should arise from that composition and registered construction semantics, without an analyzer rediscovering a theorem from source text.

## From hosted actors to MMIO and server unikernels

Fidelity.Platform supplies target facts and operation contracts. BAREWire supplies the layout, widths, encodings and transfer relationships to which proofs refer. A device register needs more than an address and a bit width: access permissions, read/write side effects, legal state transitions, ordering and required initialization participate in the contract.

| Substrate | Relevant obligations | Boundary that remains explicit |
|---|---|---|
| Native actors | Ownership, arena lifetime, turn eligibility, wait dependencies, synchronization and supported protocol laws | Scheduler, atomics and memory-model realization; cooperative single-core execution does not eliminate interrupts or DMA |
| JavaScript and hosted workflows | Numeric bounds and merge laws, continuation validity, partition identity, acceptance and durable reconstruction | Host scheduling, isolation, transactions and failure semantics. Cloudflare supplies its facilities; Fidelity does not install a new scheduler there |
| Cortex-M33 / EK-RA6M5 | Vector placement/alignment, initial stack and reset path, memory map, legal access widths, initialization order, interrupt/resource interaction | The exact Arm M-profile and Renesas device semantics, board configuration and generated instruction correspondence; Islaris is a research lead, not an existing proof of HelloBlinky |
| Firecracker guest unikernel | Startup and address-space layout, virtio descriptor/ring ownership, permitted device accesses, publication/completion order and buffer lifetime | The selected transport, guest architecture and VMM implementation contract. [Firecracker's design](https://github.com/firecracker-microvm/firecracker/blob/main/docs/design.md) is an engineering source, not a proof of the VMM |
| FPGA numerical coprocessor | Exact datapath/quire capacity, state transitions, transfer fidelity, ownership across requests and completion | Timing/resource evidence, actual transport and deployed bitstream correspondence; a network hop does not inherit a local-memory protocol automatically |

Ordinary separation-logic heap rules cannot model an MMIO read that clears a status bit, a write-one-to-clear register, or an asynchronously changing device as plain RAM. Likewise, a volatile access is not a complete synchronization or device-ordering proof. These semantics belong in the device operation model and its interpretation. Hardware documentation grounds that model; it is not machine-checked evidence that every physical implementation satisfies it.

Machine-artifact checks are one stage in this architecture. They matter where lowering or deployment can disturb a claim, but do not replace the resource, numerical and distributed proofs that motivated the design.

## One managed proof service

Composer should direct a consolidated OPAM/Rocq toolchain alongside cvc5. Packages are pinned, resolved and built during toolchain preparation or library updates, not during keystrokes. A release declares a tested set of compiler, kernel, libraries and adapter revisions. Compatibility across these research projects must be demonstrated; if separate internal environments are temporarily necessary, evidence crossing between them still requires an accepted checked interface.

Framework authors establish reusable rules in that environment. Application checking uses registered instances and bounded procedures. The service routes work by the obligation's declared logic and admitted rule, not by trying every available prover until one says yes. A possible division is a generated Rocq term checked by the kernel, or a restricted derivation checker whose soundness and implementation trust are explicitly accounted for. Neither route is selected merely by drawing a Tier 4 box.

The editor path needs:

- Stable semantic identities and a dependency DAG for obligations, laws, premises, encodings, target facts and source snapshots.
- Warm workers, cancellation and bounded queues; immediate reuse of still-valid results while affected obligations become pending.
- Separate latency budgets for structural checks, supported solver leaves and larger composed checks. Measure cold and warm p50/p95, invalidation fan-out, memory use and certificate-check time on representative editing workloads before promising a latency bound.
- Bounded, construction-specific instantiation and leaf solving. No unbounded theorem synthesis or interactive tactic development in the application editing loop.
- Identical required checks for editor and command-line builds. An expired budget yields a visible pending/unresolved result; at a required commitment boundary it blocks the unestablished claim, never supplies a success.

The Clef Proofs view reports claim, participants, checked source version, premises, selected law, evidence kind and remaining host assumptions. Opening retained evidence must not rebuild the application. It should distinguish inference results, solver verdicts, emitted certificates and checked derivations. Hiding the display changes no checking obligation.

External process orchestration is consistent with Composer's backend model. It does not require Python or shell scripts as a permanent integration layer, nor a Rocq runtime in the deployed application. Package licenses, notices and redistribution terms should be recorded for the revisions actually selected; invoking a tool out of process is not itself a licensing conclusion.

## Engineering gates and permitted claims

1. **Freeze an initial proof profile.** Pin a compatible Rocq/library set, enumerate permitted axioms and checking dependencies, and build a minimal upstream example. This research has not installed or compatibility-tested the combined set.
2. **Establish the semantic bridge.** Give one Clef construction its operational interpretation and show that its inferred PSG facts instantiate the selected library judgment. Prove the bridge and its adequacy for the observation being claimed.
3. **Complete certificate admission.** Reconstruct/check one supported cvc5 proof family or generate an equivalent receiving-system derivation. Reject mismatched queries, missing premises, altered certificates and disallowed assumptions. Do not pass an unchecked Boolean verdict into a higher theorem as an axiom.
4. **Demonstrate the numerical join.** Cover arithmetic, ownership and accepted-result identity in one composition. Include negative cases: missing/duplicated partition, stale epoch, overflow, wrong buffer identity and interrupted persistence. State exactly which safety, relational and progress claims pass.
5. **Exercise the application workflow.** Ordinary Clef code automatically produces the instance and source-linked evidence. Change a premise and verify invalidation, cancellation and bounded response; run the same obligations without the editor. Use the repository's F# test infrastructure for these compiler/service gates.
6. **Add substrate realizations independently.** Establish hosted workflow premises, native resource transfer, a modeled MMIO sequence, and a Firecracker guest device path against their own contracts. A successful route is evidence for that route, not all hardware.

The capabilities above substantially reduce the amount of foundational logic Fidelity must invent. The remaining work is precise: semantic correspondence, automatic elaboration, proof admission, evidence preservation, measured dispatch and substrate-specific validation. Until a gate is implemented and checked, describe it as a reuse opportunity or design requirement. Once it is checked, report the actual theorem, supported construction and assumptions through the same developer-facing mechanism.
