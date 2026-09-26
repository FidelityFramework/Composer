# Evaluation strategy and dependency contracts

The Clef specification governs these distinctions. Shared closure representation
does not give ordinary functions, cold work, lazy values, sequences, incremental
nodes and observers the same activation or reuse semantics. Baker settles each
protocol and its participants; Alex witnesses that protocol at the actual Huet
occurrence through Elements, Patterns and Witnesses.

## Lazy-by-default demand

Ordinary bindings and arguments denote shared deferred computations. Supplying
an argument or capturing a binding does not force it. First demand evaluates the
required computation; later demands share its result. In particular,
`let ignoreArg x = 0` followed by
`ignoreArg (Console.writeln "ran"; 42)` does not print. The presence of an effect
does not itself establish demand for an otherwise unused argument.

`eager expr` establishes explicit shallow demand at its activated frontier. A
direct eager initializer runs when its binding is reached; a direct eager actual
runs when its application is activated, before the entered body. Explicit eager
actuals retain source order after callee resolution. Their results retain sharing.
Parentheses and type annotations do not hide the marker.

The marker is local. `ignoreArg (work (eager expr))` does not run the undemanded
outer argument merely because it contains an eager subexpression. A deferred
lambda, lazy body, sequence body or unselected branch still bounds activation.
Eager construction demands the outer value, not all nested payloads:
`eager (lazy expr)` forms a lazy value; `eager (Lazy.force value)` demands its
memoized result. Direct eager fields/elements are explicit demands at their
activated construction boundary. This intent survives elaboration as semantic
evidence rather than a target optimization hint.

Explicit sequencing, entry execution, startup actions and effect sinks supply
their specified demand and ordering obligations. Once a sequential expression
is demanded, its required effects cannot disappear because their values are
unused. Closure formation retains deferred binding identities and original
mutable cells; it does not silently force a capture to take a value snapshot.

Strictness analysis can remove unnecessary deferred machinery where it proves
the same demand, sharing, effect order, failure and termination behavior.
Guaranteed eventual demand alone does not authorize moving an effect ahead of
another effect, a mutation, cleanup or a potentially nonterminating operation.
The governing rules are in [expression evaluation](../../clef-lang-spec/spec/expressions.md#evaluation-of-elaborated-forms).

## Independent semantic questions

| Question | Required distinction |
|---|---|
| Formation and activation | Preserve each binding's deferred identity, sharing and lexical capture relationships. Invoking a callback, forcing a thunk, starting cold work, pulling an iterator and installing a subscription are separate demand operations. Formation alone does not demand operand effects. |
| Delivery and demand | An active observable producer invokes its observers. A sequence consumer pulls; a lazy consumer forces; incremental demand selects the graph that must be validated. Push delivery does not establish when subscription or producer activation occurs. |
| Reuse | Lazy memoization retains one successful result for the same instance. Incremental reuse requires validated dependencies and effects. Fresh sequence enumeration creates independent progress. Bare observable delivery has no implicit value cache. |
| Change | An observable emission is a delivery obligation. An incremental invalidation marks potentially stale computation; it is not itself a new output value or permission to recompute an undemanded node. |
| Scheduling and ownership | Synchronous delivery, batching, actor admission, parallel execution and target realization each retain their own contracts. None follows solely from the word push, pull, eager or lazy. |

The governing chapters are [closures](../../clef-lang-spec/spec/closure-representation.md),
[lazy values](../../clef-lang-spec/spec/lazy-representation.md),
[sequences](../../clef-lang-spec/spec/seq-representation.md),
[incremental computation](../../clef-lang-spec/spec/incremental-computation.md),
[observable computation](../../clef-lang-spec/spec/observable-computation.md) and
[reactive signals](../../clef-lang-spec/spec/reactive-signals.md).

## Protocols that shared machinery must preserve

`Lazy<'T>` computes on its first successful force and caches that instance's
result. Aliases share the cache; separate formations do not. Mutable captures
read their original cells during that evaluation. A later cell write does not
invalidate an already computed lazy value. The specification's single-forcer
ownership obligation also applies when force sites cross threads or actors.

`seq<'T>` forms a deferred producer. Each enumeration has fresh progress and
retains the prescribed deferred bindings and shared external cells. Pulls
perform body effects in order until yield or exhaustion; current requires the
successful-pull premise for that iterator. An eager consumer such as `Seq.fold`
drives those pulls when its result or specified effect is demanded. Calling it eager does not make the sequence body
run at construction or turn its callback into a construction-time effect.

`Incremental<'T>` has a cache that can become stale. Demand selects the work;
dependency validation determines whether the cache is reusable. Tracked reads,
including reads through helper calls and captured aggregates, determine active
dependencies. Merely capturing a handle does not establish a read dependency.
Dynamic dependency changes retain their own instance, disposal and lifetime
obligations. Cutoff can suppress an unchanged output's propagation only while
preserving independent invalidations and required effects. It supplies no fixed
work, time or hardware resource bound by itself.

`Observable<'T>` delivers every emission to active observers in subscription
order, synchronously by default. Duplicate suppression and buffering require
their explicit operator contracts. Fusion into an incremental invalidation edge
must preserve delivery and effects; a stale bit alone cannot represent an event
trace whose individual emissions matter. A cold wrapper can defer subscription
without changing the producer-driven delivery contract after activation.

The Signals surface composes these protocols: `Signal.set` invalidates,
tracked `Signal.get` reads establish dependencies, `Memo` supplies incremental
caching, and `Effect` supplies an always-demanded sink. Required effects cannot
be eliminated because their result type is unit. `Batch` controls stabilization;
it does not grant general permission to discard observable emissions. Detachment
and scope exit preserve deterministic logical cleanup independently of physical
storage reclamation.

## C-series and successor acceptance

The [C-series acceptance matrix](PRDs/C-Series-Acceptance.md) applies these
distinctions to every new callable/storage/continuation form. The following
oracles prevent a shared implementation from changing the protocol:

| Composition | Discriminating evidence |
|---|---|
| Unused ordinary argument containing an effect | The argument and effect remain deferred, including through aliases, partial application and higher-order calls. |
| Explicit eager binding or actual | Reaching its binding or activating its actual application boundary demands the marked outer value once, even when the receiving binding/formal is unused; eager actuals preserve specified order. |
| Nested eager marker inside deferred or unselected work | No recursive search for eager work bypasses a deferred argument, function, lazy/sequence body or branch. Demanding an outer constructor does not force ordinary nested payloads. |
| Multiple demands for one ordinary binding or argument | The required computation occurs once and its result is shared; separate dynamic binding instances remain distinct. |
| Demanded sequential effects around a deferred callback or producer | Explicit sequencing preserves required effect order; an unused ordinary operand is not forced solely by source position or effect classification. Callback/body effects occur at their admitted demand. |
| Mutable capture before and after lazy force | A pre-force write is observed; post-force mutation does not replay the thunk. An alias shares the original cache; a separate factory has a separate cache. |
| Repeated or interleaved enumeration | Each iterator has independent progress and current validity; original external cells remain shared; no body work is introduced at formation. |
| Incremental diamond with an unchanged branch and a changed branch | Cutoff on the first branch does not remove the second branch's invalidation. Undemanded stale nodes remain unevaluated. |
| Observable feeding an incremental node | Required event counts, order and effects survive fusion, including duplicate emissions and multiple emissions before demand. |
| Cold observable activation and teardown | No registration or producer connection precedes activation; delivery follows the active subscription contract; cleanup excludes later invocations. |
| Signal/Memo/Effect composition | Reads determine dependencies; independent writes survive joins; batches stabilize consistently; unit-valued effects and cleanup retain their required executions. |

C-01–C-07 must preserve their applicable formation, capture, invocation, lazy
and sequence behavior now. Reactive, cold and scheduling successors add their
own executable gates; a C component pass does not establish those successor
protocols. The [completion ledger](C_F_Completion_Ledger.md) records these existing
PRD requirements and their shared C/F machinery; it does not establish a separate
source of requirements.

## Realization cost and coordination

Semantic eligibility precedes cost selection. A target policy compares admitted
realizations using computation, allocation, retained bytes and lifetimes, memory
traffic, coordination and latency. Avoiding repeated evaluation can retain a
large environment or concentrate writes to cache state. A successfully computed
lazy value may become read-only; incremental validation and dependency updates
have separate continuing costs. Neither fewer evaluations nor more parallel
workers establishes a performance improvement by itself.

Local recomputation is eligible only when proof establishes observational
equivalence, including effects, mutable reads, identity, failures and termination.
It must not duplicate an effectful deferred computation or turn a shared source
instance into observably separate instances. Earlier execution requires the same
proof of demand and ordering; a performance estimate supplies no such authority.

Actor ownership identifies possible readers and writers. Physical cache-line
separation additionally requires target facts, aligned allocation bases and
suitable extents. Queues, publication, reclamation and allocator metadata carry
their own obligations. A working set fitting a cache is a capacity estimate,
not a residency guarantee. See the [coordination case study](../../clef-lang-site/hugo/content/blog/counting-the-cost-of-coordination.md)
and [CPU cache policy](../../clef-lang-site/hugo/content/docs/internals/hardware/cache-aware-compilation-cpu.md).

Baker settles semantic demand, sharing, effects, ownership and their proof
dependencies. Target cost analysis may consume those facts without redefining
them. Alex observes the admitted physical contract through its Huet zipper and
composes Elements/Patterns/Witnesses; it does not infer strictness, schedule
effects or choose semantic equivalence while emitting MLIR. CPU cache policy and
FPGA realization remain target concerns rather than language rules.

## Design-time explanations and diagnostics

The specified source surface gives `lazy expr` and `eager expr` distinct contracts:
lazy exposes an explicit memoized value and later force boundary, while eager
establishes demand at an activated frontier and retains the operand's type.
Ordinary lazy-by-default bindings need no annotation. `Lazy.create` and
`Lazy.force` operate on explicit memoized values; sequencing orders the effects
of a demanded computation. See [lazy values](../../clef-lang-spec/spec/lazy-representation.md),
[expression evaluation](../../clef-lang-spec/spec/expressions.md#evaluation-of-elaborated-forms)
and [lexical rules](../../clef-lang-spec/spec/lexical-analysis.md).

Design-time information should make the settled graph understandable without
requiring developers to annotate every evaluation choice:

| Presentation | Required evidence and purpose |
|---|---|
| On-demand explanation or informational hint | Locate the binding, its demands, sharing scope and retained captures. Explain why it stays deferred or why a thunk can be elided. A proven semantic fact needs no performance claim. |
| Optional performance advisory | State a concrete opportunity and tradeoff, the target assumptions and whether supporting cost evidence is static, estimated or measured. Repeated enumeration can repeat work; caching can instead retain storage or change freshness. Do not label either idiom universally better. |
| Warning | Identify a violated declared performance/resource requirement or an explicitly enabled analysis policy with actionable evidence. An uncertain cost preference is not a default warning about valid lazy code. |
| Error | Identify a language, ownership, lifetime or other mandatory admission obligation that cannot be satisfied. Performance policy cannot downgrade correctness. |

An explanation follows source identity through nanopass elaboration and points
to the joint premises that establish demand and reuse. A code action must state
whether it is proven behavior-preserving. Adding a force, cache or parallel
boundary can change effect timing, freshness, failures or ownership; such changes
are semantic design choices and cannot be offered as automatic equivalent fixes.

Reports retain graph snapshot and source provenance, target/profile identity and
the premises supporting the conclusion. Scoped edits invalidate affected
conclusions, including negative premises and changed alternatives. A previous
profile is historical evidence, not a fresh measurement of the edited program.
Cache-line estimates belong to the target analysis report; Alex consumes settled
contracts rather than generating new source-level advice by inspecting MLIR.

Acceptance requires an explainable unused-effect case, a shared repeated-demand
case, fresh sequence enumeration versus cached reuse, and a retained-environment
tradeoff. Target-dependent suggestions must change or disappear when the target
facts are unavailable, while semantic explanations remain valid. Source edits
that alter demands or captures must retract stale advice. Equivalent code actions
must preserve the same observable traces used by the underlying C-series gates.
An eager result that is later unused is not automatically redundant: timing,
effects or an intentional readiness boundary may be its purpose. A lazy value
that is eventually demanded is likewise not automatically a poor choice. Advice
explains the specific tradeoff and respects both forms as intentional designs.

## Evaluation requirements through the owning stages

The specification's [ordinary call-by-need and sharing rules](../../clef-lang-spec/spec/expressions.md#default-demand-and-sharing)
and [explicit eager frontiers](../../clef-lang-spec/spec/expressions.md#eager-expressions)
apply to the existing PRDs: capture identity in [C-01](PRDs/C-01-Closures.md),
application and partial-application boundaries in [C-02](PRDs/C-02-HigherOrderFunctions.md),
[F-03](PRDs/F-03-PipeOperators.md) and [F-04](PRDs/F-04-CurryingLambdas.md),
recursive demand in [C-03](PRDs/C-03-Recursion.md), collection and payload demand
in [C-04](PRDs/C-04-CoreCollections.md), [F-05](PRDs/F-05-DiscriminatedUnions.md),
[F-08](PRDs/F-08-OptionType.md), [F-09](PRDs/F-09-ResultType.md) and
[F-10](PRDs/F-10-RecordTypes.md), and the explicit lazy and sequence protocols in
[C-05](PRDs/C-05-Lazy.md), [C-06](PRDs/C-06-SimpleSeq.md) and
[C-07](PRDs/C-07-SeqOperations.md). These are shared requirements of those PRDs,
not a separate planning gate. Their implementation must replace blanket eager
assumptions rather than merely add a keyword to an otherwise eager evaluator.
The following contracts define the delivery order through the owning stages:

1. **Source construction.** Parsing and checking retain an explicit eager marker,
   its operand type, source range and activated binding/application/construction
   frontier. Resolve transparent grouping before identifying that frontier.
   Preserve ordinary deferred identities and the boundaries of function-valued
   results. Recognizing a function named `eager` or a source spelling during MLIR
   emission does not implement the syntax contract.
2. **Demand elaboration.** Baker recipes build relations for activation, value
   demand, explicit effect order and shared dynamic binding identity. A source
   node can have several structural uses without denoting repeated evaluation;
   separate loop iterations or activations can instantiate distinct bindings
   from the same static node. Capture and forwarding relations carry identity
   without automatically adding demand. Constructor/tag demand stays distinct
   from field or payload demand.
3. **Saturation and admission.** Owning nanopasses compose these relations to a
   fixed point across known callees, joins, recursion, captures and returns.
   Strictness and single-use conclusions retain all supporting participants and
   exclusions. An unknown callable alternative, changed demand path or changed
   ownership premise retracts dependent conclusions. A recursive dependency
   cannot be discharged by a traversal visited set. Reachability alone does not
   establish execution, sharing, freshness or effect order.
4. **Deferred storage and realization.** Each required shared computation has
   an admitted cache/state identity, typed result, initialization guard and
   covering lifetime. Aliases, captures and forwarded arguments retain it.
   Strictness can eliminate this storage only with preservation evidence.
   Memoization state contains no code pointer as untyped data; callable code and
   the actual environment remain separate operands. Concurrent force and
   recursive force require their owning protocol, not an inferred property of
   the chosen flag or layout.
5. **Passive witnessing and target selection.** Alex observes the settled
   protocol at its Huet occurrence and composes existing or extended physical
   Elements/Patterns/Witnesses. It must not decide which ordinary argument was
   unused, reconstruct a thunk, or walk a body recursively to emit its effects.
   Target lowering selects an admitted realization while preserving the same
   demand and sharing evidence. Hardware-specific coordination costs remain
   attached to that selection.
6. **Native and design-time evidence.** Compile discriminating sources through
   the real pipeline and execute the resulting artifacts. Compare effect traces,
   values, termination cases and storage identity for ordinary versus explicit
   eager demand, repeated aliases, partial application, unused payloads, mutable
   captures, nested deferred boundaries and recursive calls. Then test source
   edits that change each premise against a fresh check and retract stale
   explanations. Component graph shapes alone cannot certify runtime semantics.

Existing ordered-evaluation recipes must be audited at their owning operation:
explicit sequencing and strict primitives keep their required order; ordinary
applications and aggregate formation cannot inherit blanket operand demand.
Likewise, an immutable capture described as a snapshot may contain a computed
value or a deferred binding identity. Materializing its bits must preserve which
one it denotes. Program startup selects declared actions and demanded values;
the mere presence of an effectful initializer is insufficient to activate an
otherwise ordinary unused binding.

Historical native traces that depended on implicit eager argument evaluation
need semantic review. Where the intended example requires eager ordering, make
that intent explicit in source and gate the marker through the compiler. Where
the example is intended to demonstrate ordinary lazy behavior, use the specified
demand trace. Neither changing an expected string alone nor preserving an old
eager trace satisfies these source demand requirements.

## Runtime incrementality and compiler incrementality

Source `Incremental<'T>` specifies application behavior: cached runtime instances,
tracked reads, active dependencies, demand and stabilization. Scope-aware
incremental compilation specifies how source edits invalidate and rebuild
compiler conclusions, proofs and artifacts. They share the need for complete
dependencies and sound reuse, but they are separate contracts.

In particular, the runtime incremental chapter's dependency DAG is not authority
to discard recursive dependencies in compiler analyses. Compiler nanopass
saturation must retain the owning domains' fixed points, negative premises,
alternative support, retractions and snapshot identities. Alex's pull of settled
facts through a Huet zipper is graph observation; it does not choose the runtime
program's push/pull strategy. See the
[incremental compiler direction](Nanopass_Incremental_Contract_Direction.md).
