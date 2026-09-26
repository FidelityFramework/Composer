# Functional continuation case study: two-bank animation frames

**Status: Planned case study — 2026-09-25.** This is an acceptance proposal,
not an implemented sample, new source API or claim of FPGA continuation support.
Its transaction, cancellation and profile contracts must be settled by their
owners before implementation. It applies the [C-series acceptance discipline](../PRDs/C-Series-Acceptance.md)
to a useful extension of HelloArty without replacing its existing Mealy baseline.

## Purpose

HelloArty currently spells out four similar brightness calculations in
[Behavior.clef](../../../HelloArty/src/FPGA/Behavior.clef). The proposed engine
computes an animation frame from one captured phase/color/master snapshot using
two independent producers, one for LEDs 0–1 and one for LEDs 2–3. A collector
awaits both results and publishes one complete frame to a bounded sink.

The source should express the shared brightness transform through ordinary
functions, captured offsets and finite collection/sequence operations. Delimited
continuations express waiting for producer delivery and sink capacity. This
tests functional composition, retained state and parallel coordination instead
of asking the application to spell out four producer state machines.

The frame sink is initially a checked simulation interface. Connecting it to PWM
later requires a declared frame-commit boundary that preserves the live display's
timing contract. UART is not an initial dependency: the current application emits
`UartReport = ValueNone`, and the [profile acceptance](../../../Fidelity.Platform/Profiles/ArtyA7_HelloArty/README.md)
explicitly does not establish a working transmitter.

## Bounded transaction contract to settle

- Admit at most **two requests in flight**. Each accepted request carries an
  identity `(epoch, request)` and immutable phase, color and master-enable values.
  Later button/switch changes cannot change an already accepted snapshot.
- Form two distinct producer continuations per request. Each captures its lane
  offsets and request snapshot, computes two brightness values in lane order,
  and delivers one result with its request identity and lane identity.
- The collector's delimiter owns both awaited deliveries. It may retain the
  first result while suspended for the second. It joins only the two different
  lanes belonging to the same `(epoch, request)`; duplicate or stale deliveries
  cannot satisfy the missing participant.
- Publish complete frames in request-acceptance order within an epoch, regardless
  of producer completion order. No partially updated four-LED frame is visible.
- Sink backpressure suspends publication. The pending frame and capture values
  remain stable; a stall does not repeat mapping, delivery or output effects.
  A request retires only after its output is accepted or its cancellation is
  established. Completed arithmetic alone does not release its storage.
- Source suspension/resumption is one-shot per awaited delivery. A readiness
  level held high is not permission to resume the same continuation twice.

**Proposed reset semantics:** reset cancels every accepted but uncommitted request,
clears pending publications, and begins a new epoch; committed frames remain
historical observations. No pre-reset delivery can complete a post-reset join.
This cancellation rule must be adopted explicitly in the transaction/continuation
contract before implementation, including what observers see at reset. Finite
epoch/request representations require proved retirement before identifier reuse;
wraparound cannot make an old delivery current. The backend reset implementation
must clear or invalidate all relevant continuation, queue and output state.
Until these requirements are settled and checked, mid-flight reset is an open
acceptance case, not behavior inferred from an FPGA register's reset pin.

## Functional and trace oracles

Use an independent integer reference for the existing triangle/smoothstep
formula, preserving its division order. With the current floor 0 and ceiling
256, proposed smoothed-brightness vectors include phase 0: `[0; 255; 0; 0]`, and
phase 128: `[128; 126; 0; 0]`. Phase here is the captured next-phase value supplied
to the transform, before PWM and color/master policy. These are reference
expectations, not recorded executable results. Apply the captured color/master
policy at the declared frame boundary.

Run contrasting completion schedules for the same two captured requests: lane 0
first, lane 1 first, simultaneous delivery, and a second request whose arithmetic
finishes before the first. Change the live inputs after acceptance. Stall the
output while both request slots are occupied, then release it. Values and output
order must agree with the reference; producer completion order may differ.

Trace observations identify request acceptance, each lane's delivery, join,
output acceptance, retirement and cancellation. Check the required partial order
and exact multiplicity rather than imposing an arbitrary total order on the two
independent producers. Each committed request has two distinct lane deliveries,
one matching join and one output acceptance. A cancelled request has no later
output acceptance. No third request is accepted while both slots remain live.

Negative controls change a retained capture, substitute a request/lane identity,
duplicate a resume, drop a join participant, reuse storage early or retain an old
epoch through reset. The corresponding graph, artifact or trace claim must fail.
Output equality alone cannot discharge these identity and lifetime obligations.

## Storage, progress and clocks

The logical bound is two snapshots, four producer frames, two collector frames,
four partial-result slots and one pending output slot. Actual physical sharing
requires the settled liveness/layout relation; the compiler derives bytes,
alignment, initialization and peak live capacity from the admitted graph and
target. There is no unbounded producer creation, heap queue or implicit arena.

The existing [Arty description](../../../Fidelity.Platform/Profiles/ArtyA7_HelloArty/Description.clef)
has `ProgramLifetime = None`. Its BRAM/register/flash inventory does not establish
frame storage or immutable-image authority. This experiment needs an explicit
profile contract for its supported storage, initialization, access and residence
before those objects can be placed. Capacity failures remain visible.

Safety must hold while the sink stalls indefinitely. Eventual completion needs
separate premises: admitted producer execution, fair delivery/scheduling where
applicable, an eventually accepting sink and no further reset. A bounded latency
claim additionally needs explicit stall and execution bounds. Do not derive
progress from capacity, local component correctness or a passing finite trace.

Start with one selected clock and explicit reset polarity/priority. Board inputs
require their declared synchronization/sampling contract. The existing 100 MHz
oscillator and 25 MHz structural heuristic are different facts in the profile
record; neither supplies a divider or timing closure for this engine.

## Compiler ownership and acceptance stages

[Baker's suspension recipe](../Delimited_Continuations_Architecture.md) and the
[DCont contract](../../../clef-lang-spec/spec/dcont-representation.md) own cuts,
delimiter association, live-across state, capture identity, one-shot use and
placement obligations. Alex observes the settled graph through ctx/Huet
Elements/Patterns/Witnesses and expresses **portable MLIR**. The selected FPGA
backend performs Handshake/CIRCT conversion, scheduling, buffering, component
selection and circuit/HDL realization under the [FPGA admission contract](README.md).
No lane scheduler, Colibri selection or circuit algorithm is added to Alex.

1. Settle the bounded source protocol and graph obligations; establish an
   independent reference and discriminating source/graph tests.
2. Obtain native CPU value/trace acceptance of the same semantic contract.
   This validates neither FPGA progress nor physical parallelism.
3. Admit the FPGA realization and check emitted artifacts against the graph and
   reference traces under backpressure and the adopted reset contract.
4. Establish actual circuit overlap separately: identify the realized producer
   resources and show both requests/lanes can make the claimed overlapping
   progress. Two source branches or correct sequential output is insufficient.
5. Check mapped storage, timing and implementation correspondence, then record
   board evidence separately under the [artifact-verification plan](05_artifact_verification.md).

## Dependencies and limits

| Workstream | Contribution and boundary |
|---|---|
| C-01 / C-02 | Captured callable identity, independent formations, HOF invocation and retained environments |
| C-04 / C-06 / C-07 | The selected finite transforms, producer/consumer demand and retained values; no claim that all collection/sequence families work |
| A-01 / A-02 / A-03 | Async formation, suspension/delivery and ordered parallel join; historical LLVM-coroutine and CPU-thread sketches do not define FPGA realization |
| A-04–A-06 | Relevant lifetime, bounded storage and escape obligations; OS-backed allocation is not assumed on FPGA |
| T-03 / T-04 / T-05 | A later persistent mailbox/reply version; the first bounded join does not require or establish the actor capstone or OS threading |
| M-01 / FPGA-01–FPGA-07 | Profile admission, backend conversion/component scheduling, preservation and progressively stronger artifact/board evidence |

C-03 and C-05 are dependencies only if the chosen source actually requires their
recursion or memoization surfaces. The proposal does not establish a general FPGA
scheduler, unbounded actors, multi-clock delivery or production display support.
