# Interactive compiler workbench and native REPL bridge

Status: **Planned**, September 25, 2026. Composer owns this workstream. The first
milestone evaluates SageFS as a host for the current .NET bootstrap compiler
while language coverage work continues. No adapter, native JIT session, shared
daemon or latency improvement is established by this document.

The purpose is to shorten the feedback loop for completing Clef semantics in
CCS/Baker and Composer/Alex. Keep the compiler resident, submit actual Clef
examples, inspect construction and witnessing, dispatch the associated proofs,
and eventually invoke native code without restarting the bootstrap host for
every submission. A warm process does not itself make checking incremental.

This is a cross-cutting development workstream alongside C-01 through C-07,
not another source-language feature or a prerequisite to completing all of them.
F-06's interactive console parsing is a separate feature. Full native
self-hosting and notebook support remain broader milestones; a bounded bootstrap
workbench and ORC integration need not wait for them.

The interactive CLI is named **`clefx`**, matching the **`.clefx`** script
extension. This is the selected name, not an implemented command or a commitment
to particular flags. F#'s `.fsx` scripts, `fsi` interactive tool and `.fsi`
signature files are separate concepts; Clef has no separate signature files.

## Authority and scope

The [Clef specification](../../clef-lang-spec/spec/interactive-development.md)
owns interactive language semantics. The current
[Baker contract](../../clef/docs/fidelity/Baker_Saturation_Architecture.md),
[backend lowering contract](../../clef-lang-spec/spec/backend-lowering-architecture.md),
[M-01 admission gates](PRDs/M-01-DialectAdmission.md), and
[incremental provenance requirements](Nanopass_Incremental_Contract_Direction.md)
govern this host just as they govern compilation. The
[PRD index](PRDs/README.md) owns feature status; a successful workbench experiment
does not close a language PRD.

SageFS is the first hosting candidate, not the semantic service. Its current
engine uses FSI/FCS for F# execution. Shayan Habibi's
[Fable.SageFs](https://github.com/shayanhabibi/Fable.SageFs/tree/53ed66275a267d35bee2044a30550bbb5b2ffd6c)
demonstrates hosting a compiler inside that engine; it preserves Fable's FCS
fork alongside the host's compiler. It is an integration, not a replacement
compiler or an independent SageFS implementation. Its published Windows
experiment is precedent, not evidence of compatibility with CCS or this Linux
workstation.

The local original SageFS checkout reviewed for this planning was from February
2026. The September upstream
[session interface](https://github.com/WillEhrendreich/SageFs/blob/1b685a3d4bc440830398e829403b83d8cc020d5e/SageFs.Core/FsiSession.fs)
and MCP behavior are materially newer. WB-01 must pin and record the actual
version it evaluates. Do not infer availability from the old checkout or import
the upstream F# workflow skill as Clef's acceptance policy.

## Ownership

| Owner | Responsibility |
|---|---|
| Composer | Coordinate the roadmap, own the initial host adapter and shared compilation/execution service, integrate ORC, and orchestrate proof workers and acceptance evidence. Start in this solution; no separate integration repository is required for the pilot. |
| Clef / CCS | Own source admission, Baker construction/saturation, graph facts, obligations, diagnostic projections and semantic invalidation. Keep SageFS and FSI dependencies outside the compiler's public semantic contracts. |
| Lattice clients and server | Carry editing requests and present versioned compiler/proof/execution observations through the same service. They do not maintain another checker, proof ledger or emitter. |
| Toolchain build/packaging owners | Supply tested LLVM/MLIR/ORC/JITLink and cvc5 profiles, dependency identities and compatibility checks. Packaging does not own language semantics. |
| Clef specification | Define binding/redefinition, initialization, effects, lifetime and target rules for native interactive execution. A host convenience cannot amend those rules. |

[Lattice Integration](Lattice_Integration.md) owns editor integration. The
[proof-service design](Proof_Composition_Architecture.md) owns dispatch,
composition and evidence policy. This document coordinates their workbench
requirements rather than creating parallel services. The
[bootstrap investigations](Nanopass_Incremental_Contract_Direction.md#9-native-incremental-and-bootstrap-investigations)
remain the home for related .NET scheduling and incremental experiments.

## Existing foundations and missing work

| Foundation inspected | Present boundary | Work still required |
|---|---|---|
| [ProjectChecker](../../clef/src/Compiler/Project/ProjectChecker.fs) | Checks a project with volatile source overrides through CCS and its selected platform/dependencies. | Interactive submission ownership and source mapping; safe reuse must satisfy the incremental contract. |
| [CCS.Editor](../src/CCS.Editor/README.md) | Serialized whole-project checks, immutable display snapshots, compiler identity and stale-read exclusion. | Shared host lifecycle, compiler-generation identity and projections needed for deeper Baker/Alex inspection. |
| [Lattice server](../src/Lattice.Server/README.md) | LSP transport and generation-owned proof scheduling for the current editor service. | Share scheduling and session authority with MCP/REPL clients instead of copying it into an adapter. |
| [Alex](../src/MiddleEnd/MLIRGeneration.fs) | Witnesses the graph and performs bounded correspondence checks before serialization. | M-01 still owns general admission/information transport; a REPL must not claim those gates are complete. |
| [LLVM pipeline](../src/BackEnd/LLVM/Pipeline.fs) | Lowers MLIR and emits a native artifact through the AOT path. | Native JIT integration, invocation boundary, result presentation and session lifetimes. |

The proposed execution path is:

```text
editor / agent / dashboard
  -> shared Clef session (initial host adapter: SageFS/.NET)
  -> CCS and Baker -> versioned PSG and graph-resident obligations
       -> existing proof service -> cvc5 -> revision-bound evidence
       -> Alex ctx pull witnesses -> admitted MLIR -> LLVM/ORC -> native invocation
```

Both consumers observe compiler-owned facts. Any solver result used to settle or
authorize further work must enter through the owning compiler/evidence contract;
it is not an independent mutation of the graph or an instruction to Alex to
invent missing semantics.

LLVM's [ORC](https://llvm.org/docs/ORCv2.html) and
[JITLink](https://llvm.org/docs/JITLink.html) supply compilation, runtime linking
and executable-memory machinery. ORC can consume LLVM IR or relocatable objects;
this does not require loading a finished ELF executable. MLIR's
[ExecutionEngine](https://mlir.llvm.org/doxygen/classmlir_1_1ExecutionEngine.html)
is an integration candidate to assess against the required session lifecycle.
The .NET host invokes native Clef functions; it does not supply their semantics.

## Integrity contract

1. **One semantic pipeline.** A result accepted as Clef behavior must originate
   in Clef source and pass the ordinary CCS/Baker and Composer/Alex gates for its
   target. F# evaluation can inspect and test the compiler implementation.
   Hand-built graph fixtures and isolated recipe/witness calls remain component
   evidence, not source acceptance or proof of native behavior. Missing language
   behavior cannot be supplied by a CLR surrogate or a C helper in the REPL.
2. **Baker settles; Alex witnesses.** Construction, evaluation relationships,
   captures, layout and proof premises belong in Baker's owning stages. Alex
   retains its ctx pull discipline through codata, coeffects and the positional
   Huet zipper. Missing prerequisites stay explicit failures. No session-specific
   emitter, mutable semantic accumulator in the zipper, or late reconstruction
   is admitted. Flat emission retains target-admitted structured operations;
   it does not require replacing every structured form with explicit blocks.
3. **Inspection preserves identity.** Read views must retain source/elaboration
   origin, graph revision, node and occurrence/path identity, selected platform,
   codata and obligation correspondence. Do not publish mutable compiler cells
   or an already-forced lazy value as timeless session data. Returning the same
   output is insufficient when effects, residence or proof premises changed.
4. **Compiler edits create a new generation.** Record CCS and Composer build
   identities and runtime modifications. The current editor identity hashes the
   CCS assembly on disk; that alone cannot identify a hot-patched implementation.
   Compiler edits must invalidate affected graph, proof and JIT products. Use
   conservative full invalidation until dependency-aware reuse is demonstrated.
   Experimental hot patches are labelled and require saved-source rebuild and
   fresh-session replay before acceptance. SageFS patch eligibility is a pilot
   question, not a guarantee for compiler functions.
5. **Shared clients do not imply reentrant compilation.** CCS currently has
   process-global allocation/configuration, and Alex has mutable target selection
   and registration. Serialize compiler work per worker or isolate workers.
   Solvers consume frozen queries outside the compiler critical section. Worktree,
   project and target routing must be explicit; one client's cancellation must
   not cancel another client's identical current proof request.
6. **Report the boundary actually checked.** Distinguish source admission, graph
   invariants, generated obligations, solver verdicts, checked certificates,
   witnessed-artifact correspondence and native oracle results. An emitted SMT
   file is not a discharged obligation. MLIR verification alone does not establish
   preservation, and a successful native return does not prove lifetime safety.
   Required unresolved evidence blocks its commitment boundary under the owning
   contract; it never becomes success to keep the interface responsive.

These are workbench acceptance requirements. Some current protections are bounded
tests and conventions, not unforgeable API boundaries. In particular, assembly
visibility does not enforce an Ingredients-only construction boundary, and
M-01's general evidence transport remains planned. MCP lifecycle guards and a
skill can reinforce the workflow; they cannot establish semantic correctness or
prevent arbitrary source edits through other tools.

## Proof responsiveness

cvc5 dispatch is a design-time cross-application over the PSG's own obligations
and premises. The workbench consumes the existing proof service and must expose
the same required checks without an editor. Opening a graph or retained proof
must not silently rebuild or redispatch it.

The current [scheduler](../src/Lattice.Server/Server.fs) debounces checks by
150 ms, cancels superseded proof generations, shares identical query tasks
within a generation, and permits two concurrent dispatches. Each unique
[dispatch](../src/CCS.Editor/ProofDispatch.fs) starts a cvc5 process with a
two-second query limit and five-second wall limit. The document response waits
for its selected batch, and an edit clears the entire cache. In-progress CCS
checks are not interruptible through the current service; obsolete results are
discarded after computation. These are observed implementation boundaries, not
measured latency claims.

WB-02 applies the proof design's performance requirements in measured steps:

- Publish individual current results as they finish, while graph inspection and
  editing remain available. Show pending, counterexample, unknown, error and
  cancelled/stale states honestly; never leave old green evidence current.
- Bound queues and concurrency, coalesce superseded work, and prioritize current
  relevant requests without dropping required checks. Account for time still
  spent finishing an obsolete compiler check or terminating solver work.
- Evaluate reuse of an exact complete query/context with solver version/options
  and compiler/encoding identity. Reassociate a reused answer with the current
  obligation and premises. An anchor or node ID alone is not a cache key.
  Unknown/time-limited results must not suppress later work with a larger budget.
- Evaluate bounded warm solver workers against the existing process-per-query
  baseline. cvc5 supports [incremental solving](https://cvc5.github.io/docs/latest/options.html)
  and [per-query limits](https://cvc5.github.io/docs/latest/resource-limits.html),
  but assertion isolation, reset, cancellation and the selected proof profile
  need explicit tests. A resident process does not automatically provide useful
  learned-state reuse or incremental compiler checking.
- Introduce dependency-directed compiler/proof reuse only with complete premise
  and invalidation coverage under the existing incremental contract. Keep fresh
  checking as the reference for each supported workload.

Measure cold and warm edit-to-snapshot, edit-to-first-current-verdict and
edit-to-complete-required-evidence p50/p95. Separate compiler time, query encoding,
queue wait, solver startup, solving and publication. Retain query counts, cache
hits, obsolete-work drain time and memory use. No latency bound or speedup is
promised before a representative coverage workload is measured.

## Milestones

WB labels below identify this workstream only; they are not new language PRD
categories. Every milestone is **Planned**.

| Milestone | Deliverable | Entry and completion boundary |
|---|---|---|
| WB-01: SageFS compiler-workbench pilot | A pinned host and small adapter in Composer, running an existing Clef coverage case through real compiler APIs with Baker/Alex inspection. | Can start alongside C-01–C-07. Complete the pilot acceptance below, record Linux compatibility and cold/warm costs, and make an evidence-based hosting decision. No broad REPL or language-completion claim. |
| WB-02: Shared session and responsive proofs | One versioned service shared by Lattice and MCP/workbench consumers, with compiler-generation identity and measured proof scheduling. | Reuse the current editor/proof service. Complete stale-result, cancellation, isolation and exact-query reuse gates before enabling the corresponding optimization. |
| WB-03: Native interactive execution | Connect the ordinary Composer lowering to ORC/JITLink, then establish invocation, initialization, result presentation and retained-value lifetimes for a named supported slice. | Use the proven host boundary and the required WB-02 identity/admission subset. Compare JIT and AOT behavior; reject unsupported redefinition/lifetime cases. No dependency on completing self-hosting or all WB-02 optimizations. |
| WB-04: Clef-facing clients and host transition | Direct Clef submissions, shared editor/agent observation and lifecycle, with a host-independent session contract. | Extend Lattice protocol/client gates; retire bootstrap-specific hosting when the native host can satisfy the same contracts. Notebook support remains separately scoped. |

Native execution does not inherently require a compile-and-launch stage first.
A subprocess oracle can still help validation. Replaying accumulated source in
fresh processes is not retained native state: initialization and effects would
run again. WB-03 must specify the actual interactive behavior through the language
contract before admitting persistent values or replacing definitions.

## Pilot acceptance

Choose one existing computation coverage case and its owning PRD. Sequence
current-read admission provides a useful integrity control: the existing
[CCS premise tests](../../clef/tests/Clef.Compiler.Service.Tests/SequenceCurrentAdmissionCases.fs),
[Alex boundary tests](../tests/Alex.Tests/SequenceBoundaryTests.fs) and
[source-admission harness](../tests/SourceAdmission/README.md) establish different
boundaries and must keep those distinctions. Pair such controls with a real
coverage defect; do not change expected semantics just to make the pilot green.

1. Record worktree/project/target, compiler sources and build identities, runtime
   patch generation, SageFS/SDK, solver and LLVM identities. Establish an ordinary
   command-line baseline using the same source inputs and owning tests.
2. Reproduce the Clef behavior through the adapter. Inspect the first incorrect
   Baker construction or Alex observation, its source origin and related proof
   premises. Fix the owning stage; repeat the public source path. Synthetic
   fixture exploration remains explicitly component-level evidence.
3. Remove or alter a required premise in the owning negative fixture and verify
   that admission/certification is withdrawn for the expected reason. Confirm
   that unelaborated sequence forms remain Alex failures rather than acquiring
   an alternate lowering. Restore and recheck.
4. For WB-01, change source, target and compiler implementation; verify the
   adapter invalidates earlier products, using conservative reset where needed.
   Record which existing host/client interactions were actually exercised.
   WB-02 additionally requires rapid edits during a slow proof, rejection of late
   evidence and simultaneous editor/agent consumers with independent waiter
   cancellation; WB-01 does not require building that shared service first.
5. Persist the fix, rebuild the actual compiler as required, and replay in a
   fresh session. Run the owning graph, witness, solver and native gates with
   ordinary tooling. A hot-patched pass or narrowed/empty test run cannot close
   the language feature. WB-03 additionally compares the admitted JIT/AOT slice.
6. Record the latency decomposition above and the time to reproduce, inspect and
   verify the defect against the ordinary workflow. Keep or reject the SageFS
   hosting choice based on those results and integration cost. If it is unsuitable,
   retain the Clef service contracts without making compiler progress depend on it.

Installing SageFS, connecting tools or adding an agent skill alone completes none
of these gates. The pilot's eventual Clef-specific guidance must preserve normal
build/test acceptance and allow a documented fallback when the host is broken;
it must not force agents to bypass a compiler or proof boundary to stay in FSI.

## Planning evidence and next action

This planning checkpoint inspected Composer `9d5c85266a5392da345d69ee9297a3536282b4cc`
and Clef `e94fa905f7f13c2b8216b355e9d84b17dceef5c8`, including existing uncommitted
Clef work. Those are inspection references, not a tested combined integration.
No SageFS installation, compiler implementation change or performance run is
part of this checkpoint. Validation and subsequent implementation evidence belong
in [Language Coverage Waypoints](Language_Coverage_Waypoints.md).

Resume at WB-01: select the coverage case and establish its ordinary baseline,
pin the evaluated host, then implement the smallest adapter that exercises the
real pipeline. Keep all later milestones planned until their own gates pass.
