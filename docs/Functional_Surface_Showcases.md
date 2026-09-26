# Functional surface showcases

**Planning record — 2026-09-25.** These are bounded application slices for
exercising the [C-series contracts](PRDs/C-Series-Acceptance.md) and their
successors. They are not completed samples or newly admitted UI/CE APIs.
Existing application behavior and recorded artifacts remain regression oracles.
This plan changes neither application source nor hardware state.

## What the showcases should demonstrate

The language should let an application express a model, compose transformations,
retain handlers safely and describe a coherent view without spelling out every
intermediate state mutation. Target ownership, lifetime and timing remain
explicit at the effect boundary. Shorter source alone is not the acceptance test:
the composition must preserve values, effects, demand, identity and resources.

For UI, use the [Fidelity.UI architecture](../../Fidelity.UI/docs/00_architecture.md)
and [Elmish/selective-state direction](../../Fidelity.UI/docs/07_elmish_signal_hybrid.md):

```text
typed action/event -> serialized pure domain transition -> model revision
                   -> explicit commands                   -> selected projections
                   -> owning platform executor            -> owned view bindings/areas
```

Domain state has an identified authority. Local drafts, focus and expansion can
belong to a mounted component. Related fields publish coherently; a signal is
not permission to read partially updated domain state across a bridge. Commands
and events retain occurrence semantics. Only explicitly designated latest-value
state may be coalesced.

The visual model should be understandable across realizations: named controls,
grouped settings, visible pending/error state and stable identity for repeated
items. Functions, child collections and modifiers compose that model. Optional
layout CEs describe the same model with the same lifecycle; reactive and async
builders retain their distinct meanings. Native and DOM backends may lay out or
paint differently while preserving the admitted control behavior.

## Four concrete application slices

| Application | First functional slice | Additional gate beyond C features |
|---|---|---|
| HelloWayland | Frame/resize events become a typed frame plan through pure projections and bounded folds; the owner executes resize/draw/present commands | Wayland callback/buffer ownership, Ariel borrow retirement and exact render equivalence |
| WrenHello | Native model transitions publish coherent read-only projections through a calm signal facade; a small control/telemetry panel consumes them | Bridge delivery/versioning, owned subscriptions and Fidelity.UI mount/identity contracts |
| HelloDISCO | Typed preference projection and pure event updates drive a bounded control view and the existing LED/palette executor | MCU storage/timing budgets, display transaction ownership and actual board evidence |
| HelloArty | Two captured frame producers suspend independently and join under backpressure | Continuation/parallelism contracts and separate FPGA circuit-overlap/preservation gates |

### HelloWayland: a pure frame plan with owned effects

Use the current typed CPU path, specifically
[Host.clef](../../HelloWayland/src/Cpu/Host.clef),
[Layout.clef](../../HelloWayland/src/Cpu/Typed/Layout.clef) and
[Anim.clef](../../HelloWayland/src/Cpu/Typed/Anim.clef). Bounded sample projection
and min/max folds can produce a typed bounds/layout result. A synchronous reducer
can then turn frame/resize events into an immutable plan plus explicit commands.
This does not require waiting for a full event-stream or async surface.

Preserve fixed-point integer division order, padding, sample stride and the
agreement of animation bounds with trace/tile bounds. Compare the plan and
rendered output against the retained CPU oracle before changing the authoring
surface further. Functional composition must not introduce an unbounded
intermediate collection or lose a source numeric premise.

[Window.clef](../../HelloWayland/src/Cpu/Window.clef) owns native callback dispatch.
Frame-ready and buffer-release are distinct signals. A pending resize must still
allow the old buffer pair to retire; waiting for resize cannot deadlock that
release. [MappedFill.clef](../../HelloWayland/src/Cpu/Typed/MappedFill.clef) requires
all Ariel carrier accesses to retire before callback return and unmap. A deferred
sequence, lazy computation or event handler cannot retain that mapped borrow.
Shutdown still joins carriers, retires compositor ownership and frees storage
in the established order. A reactive facade must preserve these relationships.

### WrenHello: signals over an explicit message boundary

The current [backend](../../WrenHello/src/Backend/Main.fs) owns counter state;
the [frontend](../../WrenHello/src/Frontend/App.fs) mirrors native events into
Solid signals through the existing ASCII WebKit bridge. Start with that real
transport. BAREWire/WebSocket and Composer's JavaScript frontend are separate
work, not assumed prerequisites for testing the state/projection contract.

Grow a bounded control-and-telemetry panel: a counter/control card, a few telemetry
rows, an expandable bounded history and pending/error feedback. A pure native
update produces a model revision and commands; snapshots/deltas update frontend
projections. The frontend owns transient editing state while the native domain
remains authoritative. The calmer API hides routine subscription plumbing from
components while retaining typed delivery, revision, ownership and failure rules.

Acceptance conserves burst commands, bounds telemetry/history queues, rejects
stale revisions and starts resubscription from a current snapshot. Related fields
must never display mixed revisions. Updating telemetry must preserve unrelated
focus and drafts. Closing a panel retires its subscription; separately owned
collection may continue under its own explicit demand and budget. Late replies
cannot update a replacement component instance.

Keep the actual [embedded-artifact gate](../../WrenHello/tests/native-ui/README.md):
pin compiler, native host and bundled frontend; exercise malformed/oversized
messages, duplicate handlers and close with queued commands. A source-level
frontend test does not establish that the delivered host contains that bundle.

### HelloDISCO: bounded Elmish state on the existing MCU owner

The [interactive application](../../MCU/ST/STM32H747I-DISCO/HelloDISCO/experiments/interactive/README.md)
already separates pure joystick/behavior transitions from one M7 foreground
owner that drives LEDs and palette transactions. Preserve this useful boundary.
The inspected working tree also contains typed `Settings.Snapshot`, validation,
`Behavior.snapshot`/`restore`, and `Option.map`/`filter`/`exists` composition.
Those uncommitted edits are existing work to preserve; this review neither
reimplements them nor claims their native or board acceptance.

Use palette, cadence, LED pattern and pause as a small logical model. Compose
read-only selectors and typed actions into the same control roles used by the
desktop/DOM showcase, under a bounded embedded capability profile. The first
step can remain a pure transition/projection with the current physical controls.
A richer on-screen control view follows only when its native UI operations,
storage and rendering path are admitted. No browser DOM is required on the MCU.

Acceptance preserves real-sample debounce, one action per held contact, missed
sample handling, modulo clock premises and the independence of palette controls
from LED pause. Keep foreground ownership of CLUT transactions, vertical-blank
hide/update/reveal, deadlines and immutable scanout storage. Carry RAM, stack,
queue and retained-view budgets through the actual image and record cold-start
and control observations separately, as in the
[accepted hardware checkpoint](../../MCU/ST/STM32H747I-DISCO/HelloDISCO/evidence/hardware/2026-09-13-interactive-initial-palette/README.md).

The [settings persistence design](../../MCU/ST/STM32H747I-DISCO/HelloDISCO/experiments/persistent-settings/DESIGN.md)
remains a separate storage contract. A typed settings record does not establish
durability. This showcase also does not presume M4 ownership, touch, audio,
cache coherence or a general reactive-area engine from the accepted M7 image.

### HelloArty: functional composition with concurrent circuit realization

The [two-bank continuation case study](fpga-targeting/08_functional_continuation_case_study.md)
specifies two in-flight frame requests, distinct captured producer instances,
matching joins, backpressure and a proposed reset/epoch contract. Its independent
value and trace oracles distinguish source concurrency from actual circuit
overlap. Retain the original fixed-clock Mealy design as a separate regression.

C-01/C-02 provide the retained callable basis; selected C-04/C-06/C-07 operations
provide finite transforms and demand protocols. Suspension, delivery, parallel
join and FPGA realization retain their additional owners. Completing a sequence
pipeline alone cannot establish async coordination or physical parallelism.

## Borrowing from Ripple without changing Clef's execution contract

[Fable.Ripple's builder](../../Fable.Ripple/src/Fable.Ripple/Builder.fs) and
[DOM composition](../../Fable.Ripple/src/Fable.Ripple.Dom/Dom.fs) are useful source
references for ordinary function/list authoring, selective content/property
updates and a separate signal CE. Use these ergonomic ideas alongside the
[Fidelity.UI component contract](../../Fidelity.UI/docs/02_component_model.md).
Ripple's eager DOM creation and scheduler are not Clef's activation policy.

Author one nontrivial panel in both function and CE forms. Both must remain cold
until explicitly activated and establish setup once per activated logical
identity. Effects retire with their owning scope: removing a keyed child retires
its child scope, while a separately owned prepared/service instance may outlive
its presentation. Keyed rows preserve identity through
reordering while same-key replacement updates current payload. Reusing the key
alone does not make an initial captured value current. Construction-time `if`/`for`
and dynamic branches/keyed children need distinct contracts.

Layout CEs, reactive `let!`/`and!`, resource scopes and async waits must not borrow
one another's lifetime or scheduling meaning implicitly. In particular, `and!`
does not prove threads or FPGA parallelism. `Lazy` memoization is not tracked
Incremental invalidation, and `Seq` demand is not asynchronous event delivery.
The function and CE versions must agree on visible values, effects, ownership
and diagnostics, including rejected escapes and late completions.

## Continuation order and evidence

1. Complete the shared callable gates from unchanged 16h and the current C-series
   baseline. Use a captured selector/handler or staged transform from these
   applications as a supplementary case when it exercises the same contract.
2. Take one small synchronous slice in HelloWayland or HelloDISCO: pure model/plan,
   typed projection and existing owned effects. Preserve the real oracle before
   expanding UI machinery. Register the case with its owning language gates.
3. Establish WrenHello's versioned command/projection boundary and one owned
   panel. Add the same function/CE authoring contract to the native UI experiment
   as its admitted primitives become available.
4. After the required Async/Threading contracts are admitted, use R-04's tracked
   demand/invalidation core alongside R-01/R-02 delivery where required, then
   their R-03/R-05/R-06 extensions. A/T contracts own actual suspension,
   scheduling and cross-domain lifetime. UI mount/render behavior
   remains Fidelity.UI work; a C checkpoint does not complete these successors.
5. Admit the Arty continuation case through its CPU semantic and FPGA artifact
   gates. Its backend work may proceed independently once its premises exist.

Each slice records its inferred types, graph relationships, settled obligations,
actual Huet occurrence and portable Element/Pattern/Witness output, target
artifact, observed behavior and storage/lifetime evidence. Design-time edits
exercise [scope-aware continuation](PRDs/C-Series-Acceptance.md#42-scope-aware-design-time-nanopass-continuation)
with fresh-check comparison as selective reuse is introduced. Do not substitute
an attractive UI or equivalent single output for missing delivery evidence.

These cases supplement the enumerated PRD oracles. They neither require every
showcase to finish before a C PRD can close nor allow an attractive subset to
close an unfinished capability. Record revisions and implementation status in
[Language Coverage Waypoints](Language_Coverage_Waypoints.md).
