# Nanopass boundaries, provenance and incremental restructuring

September 19, 2026; reachability and segmented-artifact criteria extended September
26. Design direction for the existing
[completion roadmap](Clef_Language_Completion_Analysis.md), following the owner's
request to defer a formal pass protocol until more of the pipeline and its
incremental restructuring have been exercised. This note records requirements
and acceptance cases. It does not declare selective recompilation, distributed
proof migration or a truth-maintenance engine implemented.

## 1. Existing boundaries to preserve

The [evaluation strategy contract](Evaluation_Strategy_Contract.md) distinguishes
runtime `Incremental<'T>` and observable delivery from compiler reuse after edits.
Alex's ctx pull and Huet observation do not select the source program's runtime
push/pull strategy.

Baker ingredients, patterns and recipes elaborate semantics through nanopass
fan-out and fold-in. Today a [Recipe](../../clef/src/Compiler/Nanopass/Recipe.fs)
names its original node, replacement root, new nodes and elaboration origin.
[Fan-out](../../clef/src/Compiler/Nanopass/FanOut.fs) deliberately allocates nodes
sequentially because the current fresh-ID allocator is global.
[Fold-in](../../clef/src/Compiler/Nanopass/FoldIn.fs) produces a fresh graph,
repoints structural and hyperedge references, and clears analyses that subsequent
passes rebuild. These are concrete starting points, not a persistent incremental
identity or dependency protocol.

The [Huet zipper](../src/MiddleEnd/Alex/Traversal/PSGZipper.fs) contains focus,
path and graph. It supplies structural attention and reconstruction. Dependency
selection, proof support, work scheduling and solver state have their own owners;
they must not become fields of a semantic accumulator hidden in the zipper.
Alex continues to pull settled facts through elements and patterns.

The current editor contract already requires immutable versioned observations,
current-query identity, cancellation and stale-result exclusion.
[Lattice integration](Lattice_Integration.md) explicitly starts with serialized
full-document checks. The current server invalidates its proof-task cache as a
whole. [Proof composition](Proof_Composition_Architecture.md) already calls for
stable semantic identities and dependencies among obligations, laws, premises,
encodings and target facts. Selective reuse extends those requirements.

## 2. Topology defines the recompilation frontier

Before stratifying segmented compilation, establish an overlay describing
candidate semantic regions and their interfaces. Whether that overlay is a new
hypergraph relation, an index over existing relations, or a combination remains
open. Physical dimension algebra does not acquire new meaning from this use of
the word dimension.

A region must account for every relationship crossing its boundary: resolved
bindings and callable identity; captures and storage identity; effects, access
and lifetime; target declarations; and the participants and ordered roles of
joint obligations. An AST subtree, lexical scope, or dominator alone does not
establish this boundary. A spanning hyperedge retains all its participants when
projected into separate checking procedures. Splitting its storage cannot split
away a premise of its judgment.

The exported interface includes the facts downstream work consumes and the
evidence dependencies supporting them. A local rebuild may stop propagating
only when that interface is equivalent under the applicable consumer contract
and its evidence remains valid. Equal numeric output alone is insufficient when
ownership, effects, callable identity or proof support changes. An unchanged
input at a join cannot conceal a changed independent input.

Recursive dependency components need a defined stabilization boundary. Their
interior may require a local fixed point before an interface can be published;
an acyclic partition must not be asserted by discarding cycles. Complete dependency
discovery, a conservative frontier and deterministic settlement precede finer
cuts. Region size and placement can then be tuned against measured recomputation,
proof-dispatch and transfer costs without changing semantic validity.

The existing [closure boundary](Closure_Nanopass_Architecture.md#4-why-flat-the-finiteness-lemma)
contributes enumerated crossing participants, extent and release requirements.
It does not by itself establish every dependency of an analysis. The corrected
[incremental contract](../../clef-lang-spec/spec/incremental-computation.md)
requires actual reads/effects and preserves independent invalidations at a join.
That application-language contract supplies useful acceptance pressure here;
it is not evidence that compiler-region reuse is implemented.

### 2.1 Reachability is a supported conclusion

The current whole-graph check supplies a reference execution of the analyses.
Incremental reuse must establish why a reachability conclusion still holds after
an edit. Record its roots, relevant execution/declaration relationships and
supporting derivations. Adding or removing a reference, startup effect, callback
origin or platform declaration can change this support beyond the edited scope.
Deleting one incoming relationship does not retire a node with another valid
support; a cycle detached from every applicable root cannot support itself.
Dead-to-live transitions require renewed elaboration and admission of the newly
demanded region, including facts that a prior executable did not need.

Fan-out proposals belong to their observed snapshot and dependencies. Fold-in
must validate those premises, reconcile competing/duplicate replacements, retire
unsupported generated structure and invalidate dependent facts before renewed
settlement. Source and generated identity correspondence must survive that
reconciliation. Deterministic allocation in today's complete run does not supply
stable identity across an edit or across independently scheduled regions.

A proposed `full`/`reachable` intermediate-dump option is a serialization policy.
It must leave the compiler graph, reachability judgments and admission unchanged.
A filtered dump declares its selection policy and omitted-node counts and either
retains supporting evidence/participants or explicitly identifies external
references. Full dumps remain available for investigating lost support and
tree-shaking mistakes. Compact diagnostic output cannot serve as the dependency
closure for incremental compilation merely because it contains the live nodes.

### 2.2 Settled regions through segmented MLIR, objects and ELF

The prospective CPU path is settled semantic regions → Alex's portable MLIR
segments → backend object segments → LLD → the final ELF. These are distinct
boundaries: a nanopass focus, a dependency region, an MLIR region/module and an
object file need not have the same partition. This direction does not claim that
the current compiler emits independently reusable object segments.

Before reusing a segment, validate its semantic interface and its realization
dependencies: exported/imported symbol identity and linkage, argument/result and
data representations, capture/environment/storage identity, initialization and
effect ordering, relevant platform facts, and proof/correspondence support.
Changing a callee body may leave a caller's interface valid; changing its range,
layout, capture protocol or a fact used to specialize the caller can invalidate
the caller too. Whole-program optimization, constant propagation, inlining and
link-time transformations add dependencies that the reuse decision must include.
Bind reuse to artifact content identity and the relevant compiler, component,
backend and linker versions/configuration, target features and optimization
options, and resolved runtime/library/startup/linker-script inputs as applicable.
Unchanged PSG facts do not establish that a changed realization input is reusable.

Alex receives the settled demanded region, admitted boundary facts and its actual
occurrence context. A node ID re-rooted without its Huet path is insufficient for
shared structure. Existing Element/Pattern/Witness composition remains the
emission path. CCS/Baker own semantic dependency discovery, retraction, saturation
and readiness; the backend owns realization/link dependency tracking and
invalidation, object partition realization and the link contract.
An absent semantic premise remains an owning-stage failure.

Publish a final artifact from a manifest identifying the exact object/bitcode
and evidence contents, their compatible interfaces/configurations and the accepted
ELF. An older segment can be reused in a new generation only after its
dependencies have been validated against that generation. Withdraw obsolete
objects and symbols when their source support disappears; a successful LLD run
alone cannot detect an accidentally retained initializer, stale constant or
invalidated proof. Publish the accepted manifest and ELF together so cancellation
or a late worker completion cannot expose a mixture of incompatible generations.

### 2.3 Joint constraints as candidate region boundaries

Investigate expressing a region's boundary judgment through joint PSG relations:
the exported/imported facts, their ordered participants and the derivations that
support their compatibility. Existing joint incidence is a starting point for
that experiment, not evidence that it already records the full dependency
footprint of each analysis. A segmentation proposal must explain what supports
every boundary summary and which changes invalidate it.

For example, a closure environment's representation depends jointly on its
capture participants, their types/ranges and storage requirements, and the
declared platform. A changed capture can invalidate that judgment and its
consumers even when an observed numeric output remains equal. Reconsider the
joint judgment and propagate through the affected exported facts; preserve an
independent valid support when another derivation is withdrawn. Cycles still
require an admitted stabilization boundary, including rejection of unsupported
self-justification after a root is removed.

Record semantic participants, roles, observed lookup/absence facts and derivation
support in or against the identified graph snapshot. Keep scheduling state—dirty
versions, queued tasks, cancellation and publication ownership—with the service
that executes those relationships. A change announcement can identify an affected
relation and revision without introducing an event loop into the PSG zipper or
making Alex infer the dependency. Pure structural incidence alone misses negative
observations, alternative selection and realization dependencies.

Exercise this candidate with a changed participant in a multi-input constraint,
an unchanged independent participant, alternative support, a newly present
declaration, and a region boundary crossed by a joint obligation. Check whether
the declared dependencies predict the necessary reconsideration and whether
unchanged valid interfaces stop further propagation. Several partitions may be
correct; recomputation and transfer measurements can choose among them after
semantic and evidence closure are established. No general region schema or new
nanopass protocol is selected by this experiment.

## 3. Direction for nanopass input and output

The eventual contract needs to make the following observable, without fixing
record layouts or scheduling machinery now:

| Boundary | Required account |
|---|---|
| Input | Graph snapshot and semantic region; observed facts and relationships; applicable rules, declarations and target facts; readiness requirements |
| Dependency footprint | Actual observations, including lookup scope, absent declarations, alternative selection and relevant rule availability; a newly added binding or rule can invalidate a prior conclusion |
| Output | Proposed replacements and additions, codata, obligations and diagnostics; source origins; participant roles; support for each derived fact; effects on the exported region interface |
| Fold-in | Validate that the input premises still hold, integrate the result consistently, preserve identity correspondence, and publish affected dependency/frontier information |
| Outcome | Distinguish not applicable, waiting for facts, a rejected construction and a settled result; record what can make waiting work ready |

An output's complete read dependencies govern whether its derivation can be
reused. Conflict explanations may expose a smaller supporting subset; an SMT
unsat core alone is not a complete dependency footprint for the nanopass that
selected and encoded the query.

Within a revision, each analysis retains its own refinement and settlement laws.
Source edits retract premises across revisions; they do not justify treating
every analysis as an arbitrary in-place update. Alternative derivations matter:
removing one justification need not invalidate a fact that still has an admitted
independent justification. Keep the withdrawn derivation available for explanation.

[Radul and Sussman's propagator report](https://groups.csail.mit.edu/mac/users/gjs/propagators/)
provides the relevant distinction: dependency-bearing partial information supports
retraction, while recorded inconsistent premise sets support dependency-directed
search. Incremental invalidation supplies part of that machinery. A full search
or truth-maintenance claim also requires explicit justifications, alternatives
and conflict handling; the nanopass label alone does not supply them.

## 4. Cross-family solver invalidation

Invalidate by dependency, across reasoning families. A changed range premise can
leave an independent dimensional derivation reusable while invalidating a layout
or relational theorem that used that range. Tier labels are not isolation walls.
Both premise withdrawal and replacement must reach the obligation's actual
participants, dependent projections and admitted composed proofs.

A reusable solver result needs the exact claim and assumptions, semantic input
fingerprints, encoding and rule versions, relevant target facts, and the accepted
solver/checker context. Snapshot and request generations govern publication;
they must not be confused with a reusable logical-result key. An unrelated edit
can retain a result only after its dependencies and correspondence have been
validated against the current snapshot.

Retraction marks affected results stale, withdraws their use from dependent
claims and cancels or supersedes pending requests. Late replies cannot settle
current work merely because an obligation name matches. An incremental solver
session must deactivate the withdrawn assumptions through its supported protocol
or rebuild from the current premise set. Solver frames are an implementation
choice, not the compiler's semantic dependency model.

Keep solver outcomes precise. For an encoding of assumptions and a negated claim,
a satisfiable result supplies a counterexample in the encoded model. A
conservative abstraction may admit assignments that are not executable source
states; source-level counterexample presentation needs that correspondence.
This differs from inconsistent premises. Unknown, timeout, cancelled, missing evidence,
solver verdict and independently checked derivation retain distinct meanings.
Premise consistency and the accepted trust context cannot disappear when results
are composed.

## 5. Lattice explains a dependency failure

Extend the compiler-owned proof projection with a trace from the failed claim to
its contributing source and platform declarations. The first presentation is a
source diagnostic plus a focused proof drawer, not a mandatory whole-graph view.
For example, a capacity failure should show:

1. The operation and required capacity judgment, with actual participants.
2. The derived range and the target's declared capacity, including units and
   representation where relevant.
3. The derivations and source sites supplying those facts; distinguish an exact
   conflicting support set from a larger explanatory dependency slice.
4. Dependent obligations awaiting repair and independent evidence still valid
   for the current snapshot.

Navigation follows semantic identities and elaboration origins through synthesized
nodes. An optional local graph view can show the crossing relationships at the
region frontier. Counterexample details, retained query and available certificate
remain inspectable without rerunning compilation. Lattice presents the result;
CCS owns the judgment, dependency closure and freshness decision.

## 6. Suspending or transferring work

For a future Composer actor, describe a suspended semantic work item explicitly:
graph snapshot identity; region/interface identity; focused node and structural
path; rule and nanopass continuation point; required dependency slice; premises,
pending obligations and evidence references; declared target; and request/ownership
generation. The schema and storage representation remain deferred.

Reconstruct the zipper against the identified immutable snapshot. A focus ID alone
does not recover the path through shared structure. Current process-local fresh
IDs are not a distributed identity scheme. A graph slice is sufficient only when
its omitted context is represented by an admitted boundary summary; otherwise
the receiving worker must obtain that context.

The suspended nanopass/proof work is separate from the zipper position. Transfer
an explicit continuation description and replayable solver request with retained
evidence, rather than depending on a live process stack or solver session being
portable. The receiving worker verifies schema, rule, target and dependency
compatibility. Publication returns through the owning snapshot's fold-in/admission
boundary, which excludes stale or duplicate completions. Worker placement does
not change the source program's target contract.

BAREWire is the existing candidate for the typed transport contract. Applying it
here needs an exercised schema and lifecycle; no actor migration or unikernel
deployment is established by this note.

## 7. Acceptance pressure before formalization

Add gates as these capabilities enter the pipeline: changed and unchanged region
interfaces; independent and shared proof supports; a withdrawn premise with a
remaining alternative derivation; a cross-family dependent claim; a newly added
binding/rule; cyclic stabilization; stale solver replies; wrong target/rule/query
identity; missing crossing hyperedge participants; and suspended work resumed
against an incompatible snapshot or path. Repeated scheduling and duplicate
completion must preserve the same admitted graph result.

Reachability and segmented-artifact gates include live→dead→live edits, deletion
of the last supporting root, deletion with an independent support remaining,
detached recursive components, changed capture/storage identity, a changed
exported representation, and changed startup ordering. Exercise undo/redo,
failed edits followed by repair, reordered/duplicate worker completions and
replacement of a segment during an in-flight link. Verify actual symbol sets,
relocations and startup behavior as applicable, including rejection of stale
object/evidence combinations. Cover both unchanged interfaces that permit reuse
and changed interfaces that force dependents to rebuild. Change a toolchain or
link input while keeping PSG facts unchanged and check the resulting invalidation.

Compare incremental results and diagnostics with a fresh check of the same inputs.
Measure affected-region size, unnecessary recomputation, retained evidence,
dispatch count, memory and latency on the FidelityHello variants and larger
application oracles. Equivalent artifacts or specified behavior remain a separate
gate after graph equivalence and proof freshness. These observations will inform
the eventual formal input/output and recompilation-boundary contracts.
Fresh/incremental agreement establishes consistency between the paths. Retain
independent source/specification, negative-admission and artifact/behavior gates
to detect a defect shared by both implementations.

## 8. Baker settlement and extensible Alex witnessing

Baker construction and Alex observation have different responsibilities and
topologies. Reusable navigation laws can serve both, but sharing one concrete
zipper type is not a requirement. Baker's ingredients and recipes construct and
relate facts; its pass machinery owns propagation, readiness and settlement.
Attention to a joint constraint may involve several graph participants beyond
the structural path. Alex's positional zipper serves the settled program
structure and scope required by its compositional pull witnesses. Neither
responsibility requires placing a solver or mutable semantic state in a zipper.

Baker saturation needs an explicit account of when applicable work has settled,
when progress depends on unavailable facts, and when constraints conflict.
Quiescence on a partial graph is not sufficient evidence of witness readiness.
An editor can retain that partial graph and expose its remaining prerequisites;
a required commitment boundary checks the readiness of the demanded region.
Budget exhaustion or cancellation cannot stand in for a fixed point. The laws
for refinement, cycles and termination remain obligations of the affected
analysis families, including dimension, range and placement separately.

A lazy fixed-point binding between witness functions connects compositional
references. It establishes neither a second semantic saturation engine nor
permission to build an imperative or recursive subtree emitter. Alex continues
to consume settled consequences. Preserving source claims through changed target
operations is a separate obligation with its own correspondence and checking.

Before witnessing, an eventual capability analysis should establish that the
selected Elements/Patterns/Witnesses cover the demanded settled constructs for
the selected target, and that their prerequisites are present. MLIR verification
and applicable lowering/correspondence checks remain subsequent gates. The
coverage analysis itself is subject to dependencies on the component catalogue,
target profile and rule versions.

External component assemblies can extend that catalogue under the same contracts
as built-in components. Their future admission record needs component/provider
identity and version, supported construct/target combinations, required settled
facts, consumed and produced evidence, applicable preservation method, and a
versioned documentation reference. This note does not select a loading ABI.
An assembly that needs a new semantic fact must identify the owning Baker
extension and its obligations; it cannot reconstruct that fact in a witness.

Lattice should preserve this distinction in failures:

| Failure | Explanation and attribution |
|---|---|
| A required PSG fact is unresolved or absent | Show the requested fact, its source/elaboration origin, participants, owning analysis and pending or failed premise; a boundary-contract defect may be in the compiler rather than in the source program |
| No applicable witness covers a settled construct | Show the construct and selected target, required capability, considered component identities and their declared applicability; do not report a false type error |
| A selected component fails verification or correspondence | Show its provider/version, element/pattern/witness identity, original PSG site, affected claim and generated operation/artifact location; link the component's matching documentation |

The compiler service supplies those structured causes and snapshot identities.
The editor can present a concise source diagnostic, related source/declaration
locations, and an expandable component trace. A deep MLIR location needs retained
origin correspondence to reach the correct Baker construction; guessing from an
operation name or source line is insufficient. Missing correspondence is itself
a boundary failure. Generated operations with multiple origins retain those
contributors rather than assigning an arbitrary single cause.

The ledger direction in [Obligation Residency](Obligation_Residency_Design.md)
comes from HelloProof's source/artifact reconciliation. A concrete current family
is [static-storage correspondence](../src/MiddleEnd/Alex/Traversal/StaticStorageValidation.fs),
with [artifact mutation tests](../tests/StaticStorageRegression.fsx). Such checks
cover their implemented families; the proposed external-component admission
must supply its own applicable coverage. The ledger is neither a general sandbox
nor a universal semantic proof.
An external or built-in lowering can proceed under an admitted preservation
argument or the applicable rechecking route; redispatch must establish the actual
lowered claim and its correspondence, not merely repeat the source query.
Upstreaming a component changes its maintenance home, not its proof status.
Add rejecting cases for absent coverage, incorrect prerequisites, dropped
participants/anchors, mismatched provider/rule versions and corrupted operations.

## 9. Native Incremental and bootstrap investigations

The owner's subsequent direction makes native `Incremental<'T>` the destination
for this language work and a prospective building block for Composer's own
incremental execution. REPL use, larger builds and CI supply future workloads.
The governing work already exists in [R-04](PRDs/R-04-IncrementalFoundations.md),
[R-05](PRDs/R-05-IncrementalDynamism.md),
[R-06](PRDs/R-06-IncrementalIntegration.md) and the
[incremental specification](../../clef-lang-spec/spec/incremental-computation.md).
That specification already acknowledges adaptive computation and IcedTasks'
cold-first influence. Influence is distinct from adopting a bootstrap package.

The capability gates are:

- Establish the ordinary functional surface and its NTU admission, captured
  behavior, derived/custom cutoff functions, tracked reads/effects, cached state,
  demand and ownership. A CE facade is a later expression of those same operations.
- Exercise native stabilization with static joins, independent invalidations,
  equal-output cutoff, cold construction and multiple instances. Then exercise
  dynamic dependencies and replacement with explicit retirement and bounded
  storage. Baker owns the plan and obligations; Alex witnesses its settled
  portable forms. Fidelity.UI supplies compositional pressure and FidelityHello
  variants supply precise positive and rejecting oracles.
- Before Composer consumes that machinery, supply the topological region
  interfaces, analysis read dependencies, proof supports and revision publication
  contracts above. Application-value equality alone cannot validate a cached
  compiler judgment. Keep a fresh-check reference for REPL edits and build/CI
  cache results. Any host/native boundary needs its own admitted representation
  and lifetime contract.

These are dependencies to establish, not a fixed calendar or a requirement to
complete every actor/CE feature before a useful native incremental slice.

The September 25 [interactive compiler workbench](Interactive_Compiler_Workbench.md)
adds a separate Composer-owned bootstrap investigation: evaluate SageFS as a
persistent host for the real CCS/Baker/Alex pipeline while coverage work proceeds.
Its [WB milestones](Interactive_Compiler_Workbench.md#milestones) distinguish the
host pilot, shared sessions/proof responsiveness, native ORC execution and client
integration. Keeping the compiler resident establishes neither selective
recompilation nor native retained-value semantics. Compiler hot patches require
generation-aware invalidation; fresh persisted-source checks remain the reference.
This work reuses the proof service and the contracts in this document rather than
introducing an alternative dependency engine or placing work/solver state in Alex.

Two requested .NET bootstrap candidates were reviewed on September 19:

| Candidate | Evidence and bounded role | Roadmap disposition |
|---|---|---|
| [Incremental.NET](https://github.com/fsprojects/Incremental.NET/tree/67c7e69cdbcb50ad5cce2b880d970356f131a96e) | The reviewed head is `67c7e69c`, September 6, 2018. Its [README](https://github.com/fsprojects/Incremental.NET/blob/67c7e69cdbcb50ad5cce2b880d970356f131a96e/README.md) leaves dynamic `Bind` and package publication on the roadmap. Its [implementation](https://github.com/fsprojects/Incremental.NET/blob/67c7e69cdbcb50ad5cce2b880d970356f131a96e/Core.fs) supplies fixed map arities and explicit stabilization, without compiler support/provenance or regional retraction contracts. | Retain as an evaluated reference. The observed failures below exclude production adoption for the proposed compiler substrate without substantial repair and new conformance evidence. |
| [IcedTasks](https://github.com/TheAngryByrd/IcedTasks/releases/tag/v0.11.9) | Latest release observed: v0.11.9, September 5, 2025. [Reviewed current source](https://github.com/TheAngryByrd/IcedTasks/tree/c51e9fc63fb0c78887f5c418b0530453a2b2cf80) is newer. Cold/cancellable builders can help host proof-worker composition; they do not supply dependency invalidation or Clef CE semantics. | A bounded optional experiment beneath the existing proof service, after package/toolchain compatibility and lifecycle tests. No package dependency was added. |

The Incremental.NET review loaded its unmodified pinned `Core.fs` into .NET SDK
10.0.401 FSI. Updating only one independent `map2` input raised
`NullReferenceException`; an initial mapped `.Value` of 5 had a
`.VersionedValue.Value` of 0; a throwing computation left the source updated and
its derived value old; and an acyclic 105-map chain hit the iteration guard.
These are observed characterization results, not upstream test-suite results.
The temporary probe, output and source-identity metadata are retained at
`/tmp/incremental-net-review-opsuvf21/`; no package install or project restore was
used. Its `netstandard2.0` source loaded in this host, but the original project
and old test toolchain were not compatibility-tested.

For IcedTasks, a useful experiment would compare a token-aware worker with the
current [ProofDispatch](../src/CCS.Editor/ProofDispatch.fs), preserving the
[server's](../src/Lattice.Server/Server.fs) generation-owned shared query tasks,
individually cancellable waits and final generation/version checks. Test solver
timeout/termination, queue admission, failed siblings, cleanup and late replies.
From the reviewed [task combinators](https://github.com/TheAngryByrd/IcedTasks/blob/c51e9fc63fb0c78887f5c418b0530453a2b2cf80/src/IcedTasks/CancellableTask.fs),
concurrent starts do not establish sibling cancellation and draining on failure.
Cooperative cancellation does not terminate cvc5; existing process ownership must
remain explicit. Keep multi-waiter cached results as started `Task` values;
pooling requires a separately demonstrated use case.

The [released project](https://github.com/TheAngryByrd/IcedTasks/blob/v0.11.9/src/IcedTasks/IcedTasks.fsproj)
targets .NET Standard, net6 and net9. Current upstream has net10 tests, but this
audit did not build the released package with Composer's net10.0/FSharp.Core
10.1.401 combination. Measure allocation and latency against the current host
before selecting it. Neither package changes Baker's semantic authority, the
positional zipper contract, or Alex's pull witnessing architecture.
