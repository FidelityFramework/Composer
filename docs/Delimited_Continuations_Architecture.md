# Delimited Continuations Architecture

> **Delimited continuations as saturated graph structure, witnessed in standard dialects.**
> This document extends the closure saturation form family of [C-01 PRD](./PRDs/C-01-Closures.md) Section 14
> to suspension. The [sequence representation contract](../../clef-lang-spec/spec/seq-representation.md)
> specifies the caller-pulled instance; native frame and resumption construction remain pending
> as recorded in the [language coverage waypoints](Language_Coverage_Waypoints.md).

## 1. A Concurrency Language First

Clef is a concurrency language first. Every coordination surface in the language reduces to one object: a delimited continuation, captured at a suspension point and resumed by a delivery.

| Surface | Suspension | Resumption trigger |
|---|---|---|
| `async { }` `let!` | I/O issue | completion delivery |
| Actor `receive` (Olivier) | mailbox wait | message arrival |
| Supervision (Olivier) | failure continuation captured at spawn | child failure |
| Synchronous RPC | reply wait | reply delivery |
| Thread and process handoff (Prospero) | scheduling point | dispatcher resume |

The formalism under this table is settled. Danvy and Filinski supply the control operators, shift and reset: reset delimits a context, and shift captures the continuation up to the nearest enclosing delimiter as a first-class value. Dybvig, Peyton Jones, and Sabry supply the monadic framework that generalizes the operators to multiple prompts and fixes their equational theory. The structure is sound logically and in the formalism. It is also unifying in practice: an `async` `let!`, an actor `receive`, and a `let!` in any computation expression each capture a delimited remainder, and only the resumption trigger differs, so one compilation story serves every row of the table.

The surface those rows share has a lineage of its own: the computation-expression builders descend from Petricek and Syme's *Computation Expression Zoo*, and the `async` row from the F# asynchronous model of Syme, Petricek, and Lomov. Behind the scenes, Petricek, Orchard, and Mycroft supply the coeffect calculus for the frame decisions of Section 4: the escape class read at fold-in is a context requirement in their sense, carried on the graph as the concepts chapter [Coeffects and Codata](https://clef-lang.com/docs/internals/concepts/coeffects-and-codata/) describes.

Fidelity does not abandon delimited continuations. What it abandons is the expression of them as IR operations. We move the construct from op vocabulary to graph structure, the same relocation C-01 Section 14 records for the closure environment.

## 2. The Op Rendering, Two Exhibits

Two renderings of the formalism as IR operations are on the record, one ours and one CMU's.

### 2.1 Exhibit A: The Published Chapter Sketch

The design chapter [Delimited Continuations](https://clef-lang.com/docs/design/concurrency/delimited-continuations/) sketches a `dcont` dialect: `dcont.shift` capturing a continuation as an SSA result, `dcont.resume` delivering a value, `dcont.reset` marking the boundary. The sketch is ill-formed as MLIR on four counts.

1. **Op-result self-reference.** `%k1 = dcont.shift { ... dcont.resume %k1 ... }` uses the op's own result inside its defining region. MLIR scoping admits into a region only block arguments and values that dominate the enclosing op. An op's results are defined after the op and are invisible to its own regions.
2. **SSA values crossing region boundaries.** `%response` is defined inside the first shift's region and consumed in the outer body after it. A region-interior value reaches the enclosing region only through the op's declared results, and the sketch declares none.
3. **Reset in value position.** `dcont.reset %result` appears as the final operation, applied to a computed value, and the delimiter therefore encloses nothing. Both shifts sit outside any reset, the capture is undelimited, and the captured "rest of the computation" extends past the function into its callers. Undelimited capture has no bounded extent, and every guarantee in Sections 3 through 7 assumes bounded extent.
4. **No tag mechanism.** Two nested delimiters cannot be told apart by the shifts they enclose. The multi-prompt structure of Dybvig, Peyton Jones, and Sabry has no representation in the sketch.

Each count is a defect of the rendering, and none is a defect of the formalism. The operators are as sound as they were in 1990. The sketch establishes something narrower: under MLIR's scoping rules, a shift op's own result is out of scope at exactly the point where the formalism requires the captured continuation.

### 2.2 Exhibit B: The WAMI Dialect

The external exhibit is the WAMI dialect (Kang, Desai, Jia, and Lucia, *Compilation to WebAssembly through MLIR without Losing Abstraction*, arXiv:2506.16048; the mlir-wasm-dialect repository). The design is instructive twice over, in what its authors built and in what they omitted.

What they built is a handler calculus: suspend plus resume-with-handler, where the handler block receives the suspended continuation as a block argument, and a mutable storage cell holds continuations between uses. Shift and reset appear nowhere in it. That surface is closer to effect handlers than to the Danvy and Filinski operators named in the paper's framing.

Five omissions are on the record:

- **No verifiers.** The ops file's verifier section is a TODO, leaving the dialect's structural rules unenforced.
- **One hardcoded tag.** A single `"yield"` tag is baked in. Nested delimiters are inexpressible.
- **One function type per module.** A module-level constraint forces every continuation in a module to share one function type.
- **No one-shot enforcement.** A consumed continuation can be resumed again without complaint.
- **One lowering.** The only implemented target is ssawasm for the WebAssembly stack-switching proposal.

Upstream removed the dialect on 2026-02-27 as unused and moved to a Coro dialect, a reversion to C-dominated imperative framing that Fidelity declines to follow. We read the removal as empirical confirmation from the builders themselves: the strongest external attempt at a mid-pipeline continuation dialect ended with its authors setting it aside. Two of its lowering patterns are retained as reference material in Section 8.5.

## 3. The Suspension Recipe in Baker

Baker carries the formalism as a recipe, the mechanism C-01 Section 14 established for closures. Fan-out elaborates a computation-expression region into suspension structure on the PSG. Fold-in (Section 4) selects the frame form at saturation.

Fan-out splits the region into **segments** at its suspension points. Each `let!` marks a cut. The code between two cuts is a segment, and the code from the last cut to the builder's return is the final segment. The delimiter is structure: the boundary of the subgraph the builder's extent defines. No operation carries it, so the ill-formed shapes of Section 2.1 cannot arise: there is no op result for a continuation to self-reference and no region boundary for a live value to cross. Reset is where the region ends, by construction.

Per-segment analysis computes the **live-across set**: the bindings live on the path from each suspension point to the region's end. These become frame slots. The continuation frame is not a new object. It is an instance of the general environment of C-01 Section 14.1, second of the three instances named there, carrying the state-machine slot class whose discipline the spec's `closure-representation.md` Section 7 schema already fixes: captures read-only in MoveNext, internal state read-modify-write between yields, `current` and `state` written at yield. The suspension recipe instantiates that schema. It does not restate it.

Resumption sources are abstracted. The frame's awaited delivery is an edge class with four members today: I/O completion, mailbox delivery, interrupt, DMA completion. The rows of the Section 1 table differ only in that edge. One recipe serves the full table.

## 4. Fold-In: Layout and Placement

Fold-in reads the fan-out structure and settles three things, all literal at saturation.

**State count.** The number of states is the number of suspension points, literally. A region with N cuts folds to a frame whose discriminant ranges over N+2 values: not-started, one per suspension, done.

**Frame layout.** Slot assignment is interference coloring over segment liveness: two live-across values whose lifetimes do not overlap share a slot. This is the existing graph-coloring machinery, applied to segments in place of basic blocks. The result is a byte frame with literal extent and literal offsets, the shape the closure form family already discharges.

**Placement.** Fold-in places the frame by the lifetime lattice of `closure-representation.md` Section 3.3. A continuation that does not escape its delimiter lives on the stack. A continuation that does escape (a mailbox holding suspended receives, a stored future) lives in a region whose lifetime covers it. The escape class is a coeffect read at fold-in, the same read the closure forms make.

## 5. Static Resolution of Nested Delimiters

In the saturated graph, every suspension point carries an edge to its delimiter by construction: fan-out created the suspension inside exactly one builder extent, and the edge records that extent. An inner `async` inside an actor `receive` yields two extents, and each cut belongs to the extent fan-out cut it from. Nesting is settled before any code exists.

The literature settles the same association at run time: the implementation searches the dynamic context at each shift for the matching reset, guided in the multi-prompt calculi by prompt tags. The saturated graph carries the association statically.

Claim of record: in this pipeline, delimiter association is a compile-time property of the graph. The compiled program contains no prompt tag and no dynamic search, on any realization in Section 8.

## 6. Verification Conditions

Every obligation the suspension forms generate is quantifier-free, in the discharge regime of C-01 Section 14.4. For a frame with slots s_0 .. s_(n-1), offsets off_i, sizes size_i, extent E, and suspension count N, all literals at saturation:

| VC | Obligation | Fragment | Discharge |
|---|---|---|---|
| VC-EXT | size_0 + ... + size_(n-1) + pad = E | QF_LIA | Ground arithmetic over literals |
| VC-STATE | every store to the discriminant writes a literal in [-1, N] | QF_LIA | Finite conjunction over literals |
| VC-ACC | for each state k, the slots read by segment k are within live(k) | None; sets | Per-state check against segment liveness, enumerated |
| VC-DOM | the delimiter node dominates every suspension it encloses | None; graph | Dominance check on the saturated graph |
| VC-ONE | each suspended frame is resumed exactly once | None; linear | Linear obligation on the frame value |

Because VC-ONE is stated on the frame value, multi-shot is well-defined where it is declared: a frame copy is a byte copy of E bytes, legitimate because the frame is flat with literal extent, and each copy carries its own VC-ONE. Multi-shot is never the silent default. It is a declared copy with its own linear obligation.

## 7. The Witnessed Form

What crosses the witness boundary is standard dialects only: a discriminant, a byte frame with static `memref.view`s, function values, and `scf.index_switch` over the discriminant. No continuation dialect, no llvm dialect, no new op. The correspondence table of C-01 Section 14.3 covers every constituent. The suspension form adds the discriminant switch and nothing else.

The [seq specification](../../clef-lang-spec/spec/seq-representation.md) fixes the pair `(moveNext, {state, current, captures, internal_state})`: the function value is separate from its environment, with no code pointer stored as an environment field. `state` is the discriminant, `current` is the in-flight value, the captures and internal state are the live-across slots, and MoveNext is resume with a narrowed signature. `seq { }` is the instance in which every resumption source is the caller's pull. The suspension recipe generalizes the resumption edge and keeps that representation. This is the settled design contract, not evidence that native sequence frame construction is complete.

## 8. Target Realizations

Below the witness boundary, each backend leg realizes the same saturated structure in its own shape. The realizations below are design, stated with the target-profile conditions that select them.

### 8.1 CPU: The State-Machine Form

The witnessed form runs as written: `scf.index_switch` dispatches on the discriminant, each case is a segment, and resume is a call that loads the frame, switches, and runs to the next cut. This is the seq CFG generalized, and it is the first realization to build (Section 9).

### 8.2 GPU: The Launch-Graph Form

The launch-graph form is designed for the GPU leg: each inter-suspension segment becomes a kernel, the frame becomes the transfer buffer between kernels in BAREWire layout, and the continuation becomes a dependency edge in the launch graph. The DCont and Inet classifier of `clef-lang-spec/spec/native-type-mappings.md` already routes sequential effects away from parallel targets: effectful continuation structure stays host-side, and the launch graph carries dependency structure only.

### 8.3 NPU: The Dataflow Form

The dataflow form is designed for MLIR-AIE class targets. The frame maps to a tile-local buffer, and suspension aligns with DMA completion: the segment ends where the transfer begins, and the transfer's completion is the resumption edge. Admissibility is one comparison, frame extent against the tile memory class, decided at saturation from the target profile.

### 8.4 MCU: Static Frames, Interrupt Resumption

On a freestanding single-core leg, frames are static (the program-lifetime placement of the lattice), and the resumption source is an interrupt service routine, the discipline `clef-lang-spec/spec/dcont-representation.md` Section 7 states normatively. VC-ONE holds against concurrent delivery only if the resume path is exclusive, so the single-forcer discipline reappears as an interrupt-masking obligation across the resume window.

### 8.5 WASM Stack Switching: A Future Realization

Stack switching is the one target whose native structure is itself a delimited continuation. When the proposal matures, the realization is a true continuation expression, atomized from the PSG into low-level MLIR below the boundary, with no state-machine reification. This is explicitly outside current remit. Two WAMI lowering patterns are recorded here as reference material for that day: the block nest whose branches carry an on-tag successor and a fallback successor, and the recursive cont-type declarations that let a continuation's type mention itself. Both belong below the boundary.

### 8.6 FPGA: Bounded Concurrent Frame Composition

The [Arty continuation case study](fpga-targeting/08_functional_continuation_case_study.md)
is a planned FPGA admission experiment: independent bounded computations suspend,
retain request-specific captures and join under backpressure. Baker owns delimiter,
segment, frame and resumption semantics; Alex retains the portable witnessed form.
The FPGA backend maps the settled structure to admitted circuit state/dataflow and
checks its preservation. Continuation suspension does not itself establish
hardware parallelism or progress; those require the case study's separate gates.
The existing imperative HelloArty remains its own fixed-clock regression oracle.

## 9. Status and Sequencing

The [delimited-continuation specification](../../clef-lang-spec/spec/dcont-representation.md) already adopts this graph-resident architecture and explicitly retires the earlier `cont.*` operation surface above the witness boundary. Its frame, segment and proof contracts are normative; target realizations in Section 8 remain design work.

Current sequence implementation establishes source element constraints, resident generator formals, producer capture timing, delimiter ownership, owner-local delegation iteration and local evaluation contracts. The [language coverage waypoints](Language_Coverage_Waypoints.md) record their separate source, editor and witness gates. Native sequence frame construction and execution are not complete. The obsolete shape coeffect and Alex's guessed frame reconstruction have been retired; Alex refuses unsettled sequence suspension nodes even when ownership, delegation provenance and local evaluation facts are present.

The next upstream steps compose local evaluation contracts to establish value availability, dominance and suspension segments, then per-cut liveness, frame layout/placement, Boolean resumption and their proof obligations. These prerequisites belong in Baker before the native CPU realization can pass its witness and execution gates.

Sequencing:

1. **CPU state-machine form first.** First consumer: monitor-map's IPC and timer paths, whose socket request-response waits and revert countdown are suspensions with completion-delivery resumption.
2. **Target profiles, then NPU and MCU.** The Section 8.3 admissibility comparison and the Section 8.4 masking obligation both read the profile. The profile machinery lands before either leg.
3. **GPU launch-graph form when Three Body requires it.**
4. **WASM stack switching when the proposal and its runtimes mature.** Not scheduled.

Each stage produces a findings document, on the pattern this repository follows.

## References

- Danvy, O., Filinski, A. *Abstracting Control* (LFP 1990). Shift and reset.
- Dybvig, R. K., Peyton Jones, S., Sabry, A. *A Monadic Framework for Delimited Continuations* (JFP 2007). Multiple prompts and the equational theory.
- Syme, D., Petricek, T., Lomov, D. *The F# Asynchronous Programming Model* (PADL 2011). The async surface of the Section 1 table.
- Petricek, T., Syme, D. *The F# Computation Expression Zoo* (PADL 2014). The builder surface the table's rows share.
- Petricek, T., Orchard, D., Mycroft, A. *Coeffects: Unified static analysis of context-dependence* (ICALP 2013). Context requirements as static analysis.
- Petricek, T., Orchard, D., Mycroft, A. *Coeffects: a calculus of context-dependent computation* (ICFP 2014). The calculus behind the escape-class read at fold-in.
- Kang, B., Desai, H., Jia, L., Lucia, B. *WAMI: Compilation to WebAssembly through MLIR without Losing Abstraction* (2025), arXiv:2506.16048. The dialect of Section 2.2.
- Appel, A. W. *SSA is Functional Programming* (SIGPLAN Notices, 1998). The state-machine and CFG equivalence the CPU realization rests on.
- [C-01 PRD](./PRDs/C-01-Closures.md) Section 14. The environment as the general object, the form family, and the discharge regime.
- `clef-lang-spec/spec/closure-representation.md` Section 7. The slot-class schema the continuation frame instantiates.
- [Sequence Representation](../../clef-lang-spec/spec/seq-representation.md). The caller-pulled suspension contract; implementation status is tracked separately in the language coverage waypoints.
