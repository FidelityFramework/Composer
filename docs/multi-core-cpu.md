# Multi-Core CPU Through Ariel

Status: native typed pthread carrier, renderer equivalence and live animated
window acceptance gates pass, September 9, 2026.
The compiler integration is merged into the normal Clef repository at `534429798`.
Composer and Lattice.Server build through the standard sibling compiler path;
the temporary compiler override and worktree have been removed. The complete
compiler suite passes 169 tests, and native mapped-carrier and animated-window
acceptance were repeated after reconciliation. See the
[repository and recovery record](Ariel_Integration_Changes.md).
HelloWayland's CPU window uses persistent Ariel carriers. The current scheduling
layer implementation is `Fidelity.Platform/Environments/Linux/x86_64/Ariel/Region.clef`;
its typed renderer gate is `HelloWayland/tests/ariel-typed`. The older raw-pointer
experiment remains archived outside the production source lists.

## Current implementation and boundary gap

- BAREWire `DispatchRegions` validates allocation identity, extents, aliases,
  complete inputs, partition coverage and capture-layout agreement. Its hosted
  .NET/JavaScript checks pass. Spatial validity does not authorize a lifetime.
- CCS retains named `FnPtr` entries, actual foreign parameter/result
  representations and bounded scalar-array reference arguments. Composer lowers
  the boundary through LLVM/LLD. Fresh native probes exercise allocation/release,
  actual pthread callback entry and join, empty-output rejection before the C
  call, and process affinity under inherited and restricted masks.
- Farscape generates opaque handles, synchronization layouts, callback signatures,
  allocator specializations and ownership metadata from the installed headers.
  These declarations are in the dedicated `Fidelity.Pthread.fidproj` package.
- Ordinary Clef closures carry typed arrays with their extents, scalar captures,
  mutable cells, record layouts and captured function values. Annotated anonymous
  functions receive closure pairs; function aliases snapshot values. Native
  callback regressions live in `tests/NativeCallbacks`. `FnPtr` still denotes a
  native entry rather than an implicitly captured closure.
- The bounded lifecycle model explores 1,747 states. Fresh native gates pass for
  startup, partial create rollback, shutdown, 100 repeated dispatch generations
  with one/two/four carriers, callback failure/reuse and participant retirement.
  HelloWayland's real pixel calculation matches serial output under the tested
  strides, bands, angles and carrier counts, with a noncaller pthread witness.
  Detailed results and remaining coverage limits are recorded in
  `Fidelity.Platform/tests/Ariel/native/STATUS.md` and the typed renderer README.

The actual animated CPU window is selected by
[`HelloWayland.fidproj`](../../HelloWayland/HelloWayland.fidproj). Its typed host
retains listener state, joins Ariel rendering before presentation, and retires
Wayland buffers on release. A 30-second native window observation identified all
31 Ariel worker threads separately from driver helper threads and measured active
CPU time on each. Resize to 820 × 960 succeeded; normal close joined the carriers
and exited with status zero. Observed resident memory was approximately 66–69 MiB.
Focused captures confirmed animation. Direct pixel comparisons established that
the static caption and panel are byte-identical between equal-size screenshots
and between native mapped frames. An apparent lettering loss during image
inspection was not present in the actual pixel data.
The separate [`HelloWayland.CpuCarriers.fidproj`](../../HelloWayland/HelloWayland.CpuCarriers.fidproj)
is the successful headless renderer gate: it writes a PPM and exits. That gate
alone does not establish the animated window or general closure reclamation.

The mapped path now uses a nominal `BorrowedView<Schema>` supplied only by a
generated acquisition/callback/release scope. Its native element representation
comes from a BAREWire `ViewLayoutDescriptor`, independently of ordinary integer
array storage. The compiler keeps a bounded descriptor in a stack header, checks
native stride, extent, alignment and address arithmetic, and performs the paired
release after the callback returns. HelloWayland writes directly into GBM storage;
there is no application frame staging array or frame-copy step in this path.
The host requests a preserving native read/write mapping when it updates only
the animated region of an already seeded buffer. This native preservation
requirement is separate from the callback's narrower write-only view capability.
Native gates have verified eight maps against the actual bytes before unmapping,
including row padding, and rejection of an index equal to the view's extent.
A separate partial-write gate verified all 128 visible pixels through eight
remaps after changing only one pixel per frame; preservation of driver-owned row
padding is not assumed. The four-participant carrier gate verified 64 mappings,
exact U32 output, matching worker joins and no heap allocation in either scoped
callback body.

The graph-generated `mapped-element-span` obligation shares the emitter's
`MappedSpans` model. Its QF_LIA theorem proves complete element containment,
alignment and nonwrapping address addition, conditional on the checked extent.
It does not prove GBM's native allocation contract, variable stride multiplication,
worker disjointness or scheduler retirement. Source and transferred solver paths
agree on valid claims and weakened-guard counterexamples. Scoped callback
declarations are trusted library retirement contracts, with caller-use validation
and separate lifecycle evidence; they are not a general verified borrow checker.

The display migration also exercises general compiler boundary machinery:
Farscape's measured record layouts and explicit read-only references, native
aggregate-by-value calls, and nullable pointer output cells. Clef keeps tagged
`Option` values. Native pointer encoding and copy-back occur only in the foreign
adapter; packed payload accesses retain byte alignment. Listener records retain
their native function entries and explicit context lifetime. Source unit results
and native C `void` callback entries use separate compiler-generated signatures.
The focused resvg transform/pixel and callback gates pass. A projected writable
record, multiple pointer output references, or an unsupported aggregate calling
convention requires a further explicit adapter; these cases are rejected rather
than silently changing alias or write-back semantics.

The implementation follows the [foreign boundary](../../clef-lang-spec/spec/ffi-boundary.md):
opaque `CHandle` values, declared scalar representations, compiler-owned bounded
array projection for out parameters, and an explicitly retained typed callback
environment. Restoring `NativePtr` or treating addresses as ordinary numeric
values would contradict that design. The narrow native metadata and dispatch
packages compile existing declarations and shared scalar equations; this does
not establish migration of every BAREWire or display operation.

[Platform predicates](../../clef-lang-spec/spec/platform-predicates.md) describe
declared capabilities. Current-process affinity is a runtime input to carrier
admission, capped by deployment policy. Neither an x86_64 target name nor an
online CPU count establishes the resources available to this process. The
experimental realization records these inputs separately; it is not an already
integrated compiler predicate-selection path.

A concrete system profile should compose its reusable CPU, GPU and NPU
descriptions. For example, a Strix Halo system profile can reference the x86_64
CPU component and describe shared-memory relationships across all three engines.
UMA and coherence facts belong to that whole-system profile, rather than being
duplicated under each engine or assumed for every x86_64 target. BAREWire consumes
these facts when establishing a particular mapping's access and lifetime contract.
UMA alone does not prove that a driver mapping avoids internal staging. This
profile composition is a design direction, not an implemented UMA selector.

The first milestone is synchronous CPU execution of bounded parallel regions for
the logo animation. Actor construction, `MailboxProcessor`, supervision trees
and full scheduler conformance are outside this milestone. The larger Ariel
contract guides the extension points; it is not a prerequisite implementation
checklist for getting this region mechanism running.

## Extract common machinery from the application

HelloArty forced platform facts into declarations that both code generation and
constraint generation could consume. [BAREWire's platform description design](../../BAREWire/docs/11%20Platform%20Description.md)
generalized those facts. HelloWayland now provides the same discipline for
parallel dispatch: complete input, exclusive output slices, captured layout and
retirement before reuse. Their common home is
[BAREWire Dispatch Regions](../../BAREWire/docs/13%20Dispatch%20Regions.md).

The application retains its rendering algorithm. Shared library and compiler
machinery must apply to another bounded map without recognizing `Trace.pixel`,
`Fill.frame`, a shadow-table slot or a Wayland function name.

## Authority and existing foundations

[Ariel Under Prospero](../../clef-lang-site/hugo/content/docs/design/concurrency/ariel-under-prospero.md),
[Surfacing the Scheduler](../../clef-lang-site/hugo/content/blog/surfacing-the-scheduler.md)
and the [scheduler contract](../../clef-lang-spec/spec/scheduler-contract.md)
give Ariel its remit. Olivier owns actor semantics, Prospero owns supervision
policy and actor lifecycles, and Ariel provides dispatch. Its clients are the
supervisor and compiler. The initial CPU substrate adapter may explicitly call
the internal region mechanism, as the GPU adapter explicitly invokes dispatch
today. This does not add a language-level scheduling API. Automatic insertion
by Baker is a subsequent integration step.

| Owner | Responsibility for this feature |
| --- | --- |
| BAREWire | Common layout, bounded-view and dispatch-boundary vocabulary; pure spatial validators and obligation forms |
| Fidelity.Platform / Farscape | Target ABI and synchronization declarations, corrected foreign bindings, native carrier support |
| Clef / CCS / Baker | Carry worker-entry, capture-layout and boundary evidence for the explicit path; later recognize and insert eligible parallel regions |
| Ariel | Assign bounded ranges, publish inputs, join and retire regions, report supported failures and document substrate assumptions |
| Prospero / Olivier | Future integration boundary; no actor or mailbox implementation in this milestone |
| Composer / Alex | Witness the settled graph into ordinary operations and check preservation through lowering |
| Lattice | Display compiler-owned evidence at the region, call and declaration sites |
| HelloWayland | The concrete workload, frame-equivalence cases and performance measurements |

Current code already has closure environments and external-call lowering.
`src/MiddleEnd/Alex/Dialects/Core/Types.fs` represents serial SCF control flow;
there is no current Ariel dispatch recipe. The generated pthread bridge's
unresolved arguments and discarded environment have been repaired, alongside
target-provided storage layouts and direct pthread error codes. The base x86_64
platform source list does not include pthread. The dedicated typed package and
Ariel's carrier package make that dependency explicit.

BAREWire's existing region and descriptor checks establish spatial facts. Its
static-storage integration establishes a concrete pool/emission correspondence.
The new feature must extend that discipline to dynamic backing regions and
runtime synchronization; those guarantees are not already implemented.

## Compiler path and later automation

First validate the explicit internal dispatch call from the CPU adapter using
existing function and external-call machinery. Compiler changes should close
specific worker-entry, captured-layout or boundary-evidence gaps exposed there.
The archived raw-pointer prototype is a lifecycle experiment, not the source API
to promote into the application.
The descriptor and obligations are shared framework machinery applied to this
call, without manually authored per-demo proofs.

For subsequent automatic insertion, Baker identifies a finite map with an independent
calculation per index and exclusive output writes. The loop itself has a write
effect: eligibility requires reasoning about its footprint and aliases, including
called functions and mutation outside the region. Purity of the returned pixel
alone is insufficient.

The graph settles the iteration domain, captured environment, worker entry,
read/write footprints and required dispatch boundary. Layout comes from
BAREWire and the target description. Dynamic size premises become checked inputs
or guards where necessary. Unsupported independence analysis retains serial
execution; required memory obligations remain required in either realization.

Alex witnesses that structure as ordinary functions, calls, loops and memory
operations. This follows the [thin middle end](./Thin_Middle_End_Design.md):
parallel eligibility and outlining are compiler decisions before the witness
boundary. No new MLIR dialect or OpenMP dependency is required by this design.
The [LLVM/LLD backend](./LLVM_Backend.md) still produces the native ELF, with
runtime library inputs explicit. LLD itself does not turn a serial loop into
multi-core execution.

## A bounded first Ariel implementation

Implement one synchronous region at a time over persistent CPU workers. Model
its participant states: ready, executing, completed and released, with start and
failure outcomes. A small lifecycle harness should exercise alternate completion
orders. This tests the region mechanism without constructing the full simulated
actor scheduler. A worker may complete its output before releasing its region
references; retirement waits for both.

The hosted realization can initially use persistent pthread carriers and
mutex/condition-variable synchronization. It must publish state under a specified
lock protocol, recheck wait predicates, distinguish successive regions, handle
partial startup failure, and join carriers at shutdown. Worker count is runtime
resource policy informed by allowed CPU affinity and deployment constraints;
online CPU count alone is not that budget. A single-carrier realization must work.

The calling thread may help compute ranges, then wait until all participants
release the region. A region descriptor and bounded assignment state suffice;
no actor identities, mailbox delivery or actor resumption are needed here.
Define rejection of overlapping region submissions initially so reentrancy
cannot create an implicit queue or deadlock.

Record the scope and assumptions: OS carrier progress, the synchronization
contract, one active region, and supported startup/submission failures. Native
process faults are not made recoverable by adding this worker pool. Output
equivalence across worker orders is the determinism check needed here.

The [scheduler contract](../../clef-lang-spec/spec/scheduler-contract.md) remains
the destination for actor dispatch, control-plane immunity, supervised budgets,
mailbox admission, the event record and full simulated replay. This milestone
does not claim those clauses implemented. Its completion/retirement boundary
provides the later integration point for Prospero's lifetime decisions.

## Implementation sequence and acceptance

1. **Common spatial contract.** Extend the existing BAREWire vocabulary only for
   missing allocation/slice relationships. Exercise overlap, incomplete input,
   representation overflow and environment-layout rejection through the shared
   library gates. Preserve the byte model across its supported hosts.
2. **Region lifecycle checks.** Exercise exact assignment, completion, retirement,
   rejected submissions and partial startup. Include a delayed worker that
   accesses bookkeeping after its last output write; no region storage may be
   reused yet. These are bounded tests of this mechanism.
3. **Hosted platform realization.** Repair generated pthread support upstream,
   validate the worker entry/environment ABI and synchronization storage, and
   run the same lifecycle cases using native carriers. Retain compiler-owned
   entry references; string lookup alone cannot be the reachability authority.
4. **HelloWayland integration.** Call the internal region mechanism from the CPU
   `Fill` adapter. Use compiler-derived layout and boundary evidence; make only
   the compiler fixes this path needs. Keep the common calculation and host loop.
5. **HelloWayland acceptance.** Compile the CPU build through that path.
   Compare serial and multi-core frame bytes for fixed tables, angles, strides
   and sizes. Exercise zero/tail partitions, resize, failed admission and teardown.
   Measure table preparation, rendering, join and presentation separately at
   several worker counts, keeping shadow quality unchanged.

   Worker retirement releases the dispatch's use of storage. Buffer reuse must
   also respect the compositor's release, according to the platform boundary
   contract; a frame callback or worker join alone is not that release.

The first deliverable is the multi-core logo animation using a reusable Ariel
region mechanism with explicit evidence and coverage. Automatic Baker insertion,
actor/mailbox integration, asynchronous UI suspension and GPU scheduler federation
are later steps. No frame-rate guarantee follows from a core count.

The [HelloWayland application note](../../HelloWayland/docs/multi-core-cpu.md)
provides the concrete boundaries and the historical shadow-table regression.
Cross-repository links assume sibling checkouts.
