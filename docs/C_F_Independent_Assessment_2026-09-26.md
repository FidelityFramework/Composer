# Independent assessment of the September 26 C/F checkpoint

This is an independent assessment, prepared for the owner, of the intermediate
checkpoint recorded in [C_F_Checkpoint_2026-09-26.md](C_F_Checkpoint_2026-09-26.md)
at Clef `fd4ee1b`, Composer `b4f6396`, specification `d3f1d88` and BAREWire
`61b0bf7`. The governing authority is the C/F PRDs and their specification
clauses. The checkpoint states that its acceptance work is open, and this
assessment reads it as a mid-process record. It measures the state at HEAD,
identifies what the tracked list omits, and places each item on one scale:

| Class | Meaning |
|---|---|
| Regression | A behavior that passed in an earlier cohort and fails at HEAD. |
| Divergence | Delivered code that departs from the spec or the owner's architecture rules. |
| Sharpen | A tracked item whose cause, owner or oracle is stated too loosely. |
| Broaden | A tracked item whose remit is narrower than the defect. |
| Latent | Live code that will misbehave when a promised form is exercised. |
| Blind spot | A requirement that no code, tracked item or PRD next-work entry covers. |
| Vestigial | Retired design or dead code still present. |
| Decision | A question the owner must settle before implementation proceeds. |

Method: the full manifest, CCS suite and Alex suite were rerun at HEAD. Six
auditors examined the changes since Clef `14fb7c7` and Composer `f6d391d`, and
an adversarial verifier re-checked each serious finding. Severities below are
the verifier's corrected values. Further sweeps covered code the audited
commits did not change, organized by area in "Untracked items" below. Those
sweeps had no separate verifier, so each serious item drawn from them was re-read
at HEAD before inclusion.

## Measured state at HEAD

| Gate | Result |
|---|---|
| CCS source suite | 1,583 passed, 0 failed, 0 skipped |
| Alex component suite | 211 passed, 91 failed, 302 total |
| Manifest compilation | 27 of 51 entries compile. 23 fail, and 1 times out at 300 s. |
| Manifest execution | 26 of 27 compiled samples match exit status and normalized stdout. 04c mismatches. |

The manifest extraction, with per-sample outcomes and compiler hashes, is
[evidence/2026-09-26-head-manifest-audit.json](evidence/2026-09-26-head-manifest-audit.json).
Against the 21/48 baseline at Composer `6d54764`:

| Movement | Samples |
|---|---|
| Now passing | 06, 13, 15, 16b, 16g, and the new 04a and 04b |
| Regressed from a baseline pass | 12_HigherOrderFunctions, 16a_SequenceOperations |
| Repaired in the `f65907e` cohort, failing again at HEAD | 11a_DirectCaptures |
| Compile timeout at 300 s (baseline: compile errors at 202 s) | 16_SeqOperations |
| Still failing | 04, 08a–e, 09a–c, 11, 11b, 16c, 16e, 17–23 |
| Compiles, runtime demand oracle mismatches (tracked) | 04c_DefaultDemand |

Causes of the regressions, from the HEAD compile logs:

- **04, 11a, 12.** Alex refuses returned closures with "Callable copy does not
  preserve its settled code, environment owner, and source type" on the callable
  result path. The checkpoint's 11a and 12 passes belong to the `f65907e` cohort,
  before the passive-witness rewrite in `b4f6396`.
- **16a.** `CCS8403 Source settlement is required before witnessing: UnsettledField`
  at SequenceOperations.clef:7, from the new publication gate.
- **16c, 16e.** Alex reports unwitnessed callable and sequence operands.

The committed cohort ran two native controls (NominalIdentity and
MixedDimensionSchemes), neither of which exercises closures, Lazy, Seq or
console output. Disposition: add these four regressions to the tracked table
under C-01/C-02 and C-07, and make a full manifest run part of each checkpoint
that changes shared callable, witness or publication code.

## Delivered work that conforms

- Every count quoted in the checkpoint matches its cited log, and the v22
  cohort was built from the committed source. The red Alex suite is disclosed.
- The direct-capture signature correction (`f65907e`) validates capture
  provenance and has red and green regressions.
- Nominal types are keyed by module and name. Specialization derivations are
  recorded as `SchemeSpecialization` hyperedges, and retired generic nodes are
  kept as historical records.
- Unresolved `HasMember` constraints are no longer accepted silently.
- The explicit Lazy memoization protocol is constructed as PSG structure through
  a Baker ingredient.
- Exhausting the loop-recurrence work limit produces CCS8011.
- Alex no longer re-runs CCS source validators, the process-global witness
  registry is removed, and the timing instrumentation is confined to the host.

## Divergences in the delivered work

| Item | Anchor | Disposition |
|---|---|---|
| **Witness authority is a process-global table keyed by graph object identity** (high). `prepare` writes the ordinary, callable and storage projections into `Codata`, and about 25 Alex consumers read a `ConditionalWeakTable` instead. A structurally identical graph copy loses its authority. The same mechanism accounts for about 81 of the 91 Alex failures. A second seal, `OrdinaryDemand.sealEmission`, has no production caller, and a test depends on it. | clef `WitnessEmission.fs:16`, `:33-36`, `:49-52`, `:91-94`; `OrdinaryDemand.fs:284-309`; Composer `TraversalOccurrenceTests.fs:73` | Alex reads the settled `Codata` projection. The planned contracts assembly carries graph content. Retire both tables. |
| **Ordinary demand runs with inverted polarity** (high). The delivered mechanism omits a formal only when it has zero uses in a closed family of direct calls. Every other binding, used actual and capture is evaluated strictly, and compilation reports success with no diagnostic. 04c records each consequence: unused binding, binding before first demand, argument before body, capture at formation, early cell read. | clef `OrdinaryDemand.fs:97-130`, `:173-180`; Composer `ApplicationWitness.fs:76`; spec expressions.md "Default Demand and Sharing", "Evaluating Function Applications" | Record in the tracked row that the delivered omission is an optimization over strict evaluation. Deferral with shared storage is the default the spec requires, and earlier evaluation needs proof. |
| **A proof mismatch becomes an empty reading** (medium). `OrdinaryDemand.read` returns no omissions and no diagnostic when the published edges differ from a fresh analysis. The direction is conservative. | clef `OrdinaryDemand.fs:195-200`; consumers `RangeAnalysis.fs:1118`, `StaticStringLayout.fs:15`, `CallableCarriers.fs:238` | Raise a located diagnostic, as `projectValidated` already does. |
| **Readers recompute settled facts** (medium). `read` reruns `analyze`, `CallableEmission.project` reruns callable settlement, and eight graph-keyed caches were added. | clef `OrdinaryDemand.fs:195-200`, `CallableEmission.fs` | Read the settled hyperedges and `Codata`. |
| **Published argument ordinals ignore omitted and multi-component formals** (medium, tracked as "Physical argument correspondence"). Two Alex fallbacks read those ordinals without a guard. No test places a used formal after an omitted one. | clef `CallableEmission.fs:210-215`; Composer `VarRefWitness.fs:131-137`, `LambdaWitness.fs:270-277` | Publish physical ordinals from the settled convention, and add the omitted-then-used native case. |
| **Tuple parameters are lowered at the syntax level with counter-derived names** (medium). | clef `Bindings.fs:159-191`, `:169` | Lower through a Baker recipe with structural identity. |
| **BAREWire `61b0bf7` adds retired numeric forms inside the Clef source closure** (medium). `int64`, `0L`, `System.Int64.MaxValue` and mutable loops appear, reported only as unreachable information. | BAREWire `61b0bf7` | Sweep under D10 with the rest of BAREWire/src. |

## Tracked items to sharpen or broaden

**Ordinary call-by-need (04c).** Sharpen: besides the polarity above, four
specification decisions are missing. They are mutable-read deferral (only the
04c oracle decides it), the demand contract of assignment and mutable
initializers, operand demand and order for each primitive (including prefix
`(&&)` and `(||)`), and the representation of deferred aggregate components.
No effect or termination classification exists to license earlier evaluation,
and `IntrinsicCategory.Pure` is attached to partial operations such as
`List.head` and `Map.key` (clef `Primitives.fs:123`, `:138`, `:606-651`).
Broaden: eager evaluation is built into structure outside the Option/Result/Seq
recipes named in the ledger:

- `Primitives.evaluateBefore`, with more than 30 call sites, whose docstring
  states the superseded rule.
- the match scrutinee snapshot and `SequenceEvaluationRecipes`.
- inline expansion, which binds every argument "even when unused"
  (`Applications.fs:120-123`, `:181-192`).
- `Curry.DeferredArgNodes`, which re-emits a partial application's supplied
  arguments at every saturated call (`Curry.fs:125-132`,
  `NanopassArchitecture.fs:201-209`).

The corpus self-check idiom binds a predicate over a mutable counter and then
resets the counter. Examples are 16a SequenceOperations.clef:31-48, 16d
SequenceCollect.clef:10-18, 15d SequenceAdditive.clef:23-30 and the
NativeCallbacks controls. Under the specified law each predicate reads the
counter when demanded, so these gates will change result when call-by-need
lands. Migrate the probes to explicit `eager` bindings or immediate assertions
under the documented-oracle rule, together with the eager oracles outside F-08:
IgnoreValues, 08d, 09b, PipeEvaluationCases, CallEffectRangeCases and
LocalUnusedCases.

**Whole Alex boundary.** Sharpen: about 81 of the 91 failures carry the
missing-publication diagnostic. The fixture helper performs the seal itself.
Two negative tests pass for that reason rather than their stated one
(`EagerWitnessTests.fs:279-300`, `ControlFlowOccurrenceTests.fs:59-86`).

**F-05 binding patterns.** Sharpen: the `let true = value` crash has no retained
regression, and the test that exposed it was rewritten to a one-arm match. Keep
the failing form as a red regression.

**Nominal identity.** Broaden: name and rendered-type identity remain in
`MappedBindings`, the `ClosedCallbacks` record count, the `graph.Types` name
index and `formatType`. Hovers render distinct nominal types identically.

**C-04 Map and Set.** Broaden:

- `Set.remove` returns the left subtree when both children are non-empty, so the
  right subtree's elements are lost (`SetRecipes.fs:133-137`).
- AVL insertion never rotates (`Patterns.fs:557`, `:617`).
- Map has no `remove` recipe.
- 35 recursive self-references are built as `VarRef(name, None)`, which Alex
  rejects (`Patterns.fs:74` and following).
- List, Map and Set have no source schemes. `resolveCollectionOp` returns
  `NotAnIntrinsic` (`Intrinsics.fs:919-922`), so registering 13a will fail at name
  resolution. Scheme admission belongs first in the C-04 order.

**C-07 successors.** The dormant `min`, `max`, `minBy` and `maxBy` recipes use
the unresolved recursion idiom. Rebuild them rather than extend them.

**Failure attribution.** Sharpen: 18–23 fail at D10 source admission (CCS8706,
CCS8018, CCS8009) before any C/F stage. The "26 failures directly exercise C/F
contracts" figure is 19. Sample 17 needs its foreign-boundary owner, and sample
06 now passes.

## Untracked items

### Sample and library corpus

- **D10 ruling 5 was executed before its corpus sweep** (blind spot). Clef
  `ca167f9` made width spellings and suffixes errors, while the Dimensional
  Handoff §6 orders the sweep first. Samples 17–23, BAREWire RoundTrip and
  13a_BAREWireCollections stop at source admission. BAREWire/src holds about 420
  width-named sites. Disposition: add the sweep as a tracked row owned by the
  sample corpus and BAREWire, then rerun 17–23, RoundTrip and 13a before
  attributing residual failures to C/F.
- **The drift gate cannot fail on these forms** (vestigial). Its D10 pattern
  omits `byte`, `nativeint`, `uint`, `single`, `double` and literal suffixes.
  SCHEDULED rows never fail. CCS8019 and `interimWarnings` survive from the alias
  period (clef `Types.fs:155-163`, Composer `Output.fs:85-92`).
- **The full-profile Platform leaf compiles dead legacy code into every
  full-profile sample** (vestigial). It includes the MLIR text templates in
  Helpers.clef, `nativeint` WebView conduits, a width-named Types.clef and the
  full BAREWire codec. The sweep estimates about 626 unreachable diagnostics per
  compile and 85 s compiles against 13–25 s for CompilerSurface.
- **Sample 15 changed without the documented-oracle record** (sharpen).
  `6d8a4f0` made `temp` an explicit `eager` binding, reordered output writes and
  removed trailing spaces. The eager edit is consistent with the spec.
  Numeric_Validation_Cases.md:59 points to a waypoint record that does not exist.

### Verification infrastructure

- **Python remains the only driver for several gates** (blind spot). These are
  tests/ForeignReferences, ForeignScalarArrays and MemoryArrays `run.py`,
  `StaticStorageNativeRegression.py` (cited at LLVM_Backend.md:50) and BAREWire
  `tests/dispatch_projection.py`. Compiled `.pyc` files are tracked, and
  RoundTrip.fidproj:6 cites the removed `native_gate.py`. Port each driver to .NET
  with its oracle, and remove the Python files.
- **The runner ignores `compile_timeout`**, and the 30 s default is below
  measured compile times for 14, 14a, 14b and 16d.
- **Exact-stream and pinned-snapshot oracles exist only in private `/tmp`
  drivers**, which the checkpoint cites as reproduction commands. Primary evidence
  is on tmpfs.
- **NativeCallbacks compares output for 2 of 29 cases.** `Process.run`
  concatenates stdout and stderr (`Process.fs:18`), and NominalIdentity's only
  structural check is `arith.cmpf`. Exit-code oracles cannot discriminate demand
  traces.
- **MutualRecursion and RecursionSimple are on disk but unregistered**, although
  C-03 §5 places mutual recursion in its acceptance surface (13_Recursion
  directory; Manifest.toml:501).
- **The MCU and Cortex-M harnesses were edited only to build** after shared
  witness and pipeline changes, and have no recorded run.

### Tooling

- The editor, analyzer and live-LSP gates were last run at v20 and are absent
  from later verification tables.
- Every functional tooling fixture targets `library`, which maps to no platform
  (clef `ProjectChecker.fs:34-36`). No platform-gated settlement or publication
  runs in any tooling gate (`NativeService.fs:1192-1197`, `:1313`).
- The design-time demand explanations required by
  [Evaluation_Strategy_Contract.md](Evaluation_Strategy_Contract.md#design-time-explanations-and-diagnostics)
  exist nowhere, and `Codata.OrdinaryDemand` has no tooling reader.
- The lattice-analyzers Option corpus keeps "already evaluated fallback" eager
  framing.
- CCS8403 publication failures surface in the editor as user source errors.

### Specification, records and order

- **Startup demand** (decision). The spec is self-consistent. Its startup
  chapter keeps observable module initializers eager (program-structure-and-execution.md
  "Program Execution"), and expressions.md assigns startup its own contract.
  Evaluation_Strategy_Contract.md:248-250 says unused ordinary bindings are not
  activated at startup. The 04a oracle and the tooling fixtures follow the spec.
  The owner either keeps the eager carve-out and corrects the contract document,
  or extends call-by-need to module scope and revises the startup chapter and 04a.
- **Spec contradictions** (decision). native-type-universe.md says mutable
  bindings cannot be captured, and closure-representation.md specifies
  by-reference capture. Field lookup, object construction and `while` keep legacy
  strict wording.
- **Companion revisions are unrecorded** (sharpen). Spec `645c15a`, `0e1f866` and
  `d3f1d88` and BAREWire `61b0bf7` are absent from the resume pairing. Composer
  HEAD requires `61b0bf7` (`StorageCommitment.fs:39`, `:294`).
- **Decision records** (sharpen). F-11, C-08 and the numeric inventory are marked
  "Accepted September 26". WB-02 and WB-03 are In-Progress, and spec §5.1 is cited
  as governing the incremental design. The repository holds no decision record
  where these documents cite acceptance. Record the owner's decision in the PRD
  index.
- **C-02 open work** (sharpen). The §4 open-work paragraph was replaced with
  present-tense pair-transport text, which 04, 08a–e, 09a–c, 11, 11b and 12
  contradict at HEAD. Restore the open-work statement.
- **Order** (decision). The resume paragraph and the "acceptance work stays
  visible" table place the Alex boundary and incremental work ahead of item 1
  without changing the numbered order. F-11(d) and C-08(d) completion now
  depends on incremental compilation.

### Incremental and parallel compilation under the PHG criterion

No incremental or parallel implementation exists yet, and the design document
names many whole-program premises. The criterion here is the PHG design: a
spanning concern is a hyperedge whose ordered participants include every
contributor across segments, and its incidence drives both invalidation and the
obligation from which the proof is generated. Against that criterion:

- **Whole-program joins held without participant incidence** (latent). These are
  FieldRanges, ElementRanges, Layouts, Escaping, the StaticStringPool field, the
  `constantsOf` widening set, the writable-space capacity reservation (checked by
  a BAREWire call, with no obligation) and the unused-formal caller census.
  `OrdinaryDemand.fs:182-184` records only the participants present at proof time.
- **Obligation edges lack roles and discharge status** (sharpen). `Constrains`
  edges have ordinal 0 and deduplicated sources. StorageWitness locates the pool
  obligation by body equality.
- **Hyperedges have no incidence index** (sharpen). They are held as a flat list,
  with 162 linear scans across 58 files.
- **Linked symbol names embed process-global node numbers** (latent).
  Examples are `FunctionPointers.fs:22`, `CallableSymbols.fs:12` and
  `TransferTypes.fs:733`, along with monomorphization clone ordinals.
- **RangeAnalysis widening reads a global constant set**, so a partitioned
  fixed point would make widths depend on the worker schedule (latent).
- **Program-wide artifacts have no owning segment** (sharpen). These are the
  static pool, the startup spine, program storage, the monolithic Codata record
  and the source seal.
- **Schedule independence** (blind spot). No normative clause states that results
  are independent of the worker schedule. Equivalence oracles exist only in
  F-11(d) and C-08.
- **Conflict graph, certificate and rewrite tape** (blind spot). None is present
  in Baker. `Recipe` records no snapshot, read footprint, premises or rule version.

The research note below supplies a construction that closes the last gap and
gives segmentation a mechanical boundary.

### Unchanged Alex code

These mechanisms predate the audited commits and appear nowhere in the
checkpoint. Each anchor below was re-read at HEAD.

| Item | Class | Anchor | Disposition |
|---|---|---|---|
| **The post-order traversal is the working strict evaluator.** Every non-omitted child is witnessed before its parent. BindingWitness evaluates `let _ = e` "for side effects". The new EagerWitness forwards an operand that is already evaluated, so `eager` and ordinary demand emit identical code. | `NanopassArchitecture.fs:128-160`; `BindingWitness.fs:42-45`; `EagerPatterns.fs:39-59` | Sharpen the 04c row. Baker publishes the demand sites, shared deferred-initializer storage and proven-strict edges, and the traversal witnesses only where the graph places evaluation. |
| **VarRef "Phase 2" places an unvisited binding's initializer at its first reference.** Branch arms share one visited set, so a later reference in the other arm can recall a value defined in a non-dominating region. | `NanopassArchitecture.fs:162-195`; `ControlFlowWitness.fs:77-79` | Latent. Initializer placement becomes a settled fact, and Phase 2 is removed. |
| **`ignore` assumes its argument was already evaluated.** The spec's `ignore (g (eager work()))` example requires the opposite. | `ApplicationPatterns.fs:830-842` | Broaden the per-primitive demand decision to intrinsics. |
| **The packed closure pair is still constructed**, as a `memref<2xindex>` with `FuncToIndex` and heap allocation for escaping closures. Its invoker `pClosureCall` has no callers. | `LambdaWitness.fs:536`, `:677`; `ApplicationPatterns.fs:147` | Vestigial. It contradicts the C-01 row's claim of replacement. Remove it after confirming the mapped-view consumer's migration. |
| **Unbound type variables become `TIndex`.** The AX1001 errors are collected into a process-global list, then drained and discarded. | `TypeMapping.fs:40-46`, `:248-258`; `TransferTypes.fs:773` | Latent silent failure. Raise AX1001 as a located diagnostic that halts. |
| **Process-global mutable state** remains in `currentTargetPlatform`, `typeMappingErrors` and emission error lists, outside the registry-isolation check. | `TypeMapping.fs:52-63`; `MLIRGeneration.fs:65` | Broaden the worker-owned-state row. |
| **SSA names are drawn from a node family by cursor and by index arithmetic.** `MappedViewPatterns` allocates with a mutable cursor, and `PlatformPatterns` computes names as `22 + 13*i`. The fixed 512-value family with role bands underlies both. | `MappedViewPatterns.fs:27-32`; `PlatformPatterns.fs:1123-1128`; `Values.fs:16-41` | Decision. The cursor and the index arithmetic conflict with the rule against pools, counters and index arithmetic on names. The owner confirms whether the banded family is the accepted structural derivation. |
| **Target witnesses decide semantics.** The MMIO witness chooses width adaptation from operation-name prefixes. The FPGA witnesses choose aggregate layouts and arm widths. The NPU kernel witness infers kernel structure from field names, a float test by string prefix, and operation names by suffix. | `MmioPatterns.fs:20-36`; `TypeMapping.fs:308-310`; `KernelModuleWitness.fs:89-97`, `:222-243` | Blind spot outside the C/F remit, recorded for the Alex boundary work. |
| **Escaping values go to the heap with no source authority or release.** | `MemoryPatterns.fs:291-293`; `ContinuationPatterns.fs:270-271` | Sharpen the F-02 row. |
| **Witness selection is first match by registration order**, with no ambiguity check. | `WitnessRegistry.fs:108-112`, `:140-157` | Latent. Enforce single ownership per kind. |
| **Dead code encodes retired policies**, such as the whole-node-map `runNanopass`, which has no callers. | `NanopassArchitecture.fs:286-315` | Vestigial. Remove it. |

### Design documents, specification and backend configuration

The drift gate, run in its read-only warning mode, reports one failing line
(error-handling.md:315, FS0058) and 4,038 SCHEDULED lines exempt from failure.

- **The retired closure-cast plugin loads whenever its file exists** (latent).
  Composer's LLVM lowering adds `--load-pass-plugin` and the
  `resolve-closure-casts` pass when `flat-closure-lowering.so` is present
  (Lowering.fs:27, :75-88). The file is present on this machine. Native results
  can therefore depend on a retired design and differ between machines. Remove
  the plugin path together with the packed closure pair in Alex.
- **The normative PSG chapter describes retired machinery** (vestigial).
  program-semantic-graph.md describes FCS `FSharpExpr` correlation, an SSA
  assignment pass, yield-state analysis and coeffects "computed when Alex needs"
  them (:22-29, :61-66), which contradicts the settled-publication model.
- **No PSG or PHG vocabulary exists for ordinary demand** (blind spot). The spec
  requires demand and effect relations to survive Baker and Alex
  (expressions.md:2893-2895), and program-hypergraph.md defines no relation kind
  for them, although the code now publishes `OrdinaryDemandProjection`.
- **Missing demand clauses extend further** (broaden). Besides the four noted
  above, the foreign and platform call boundary, tuple construction and inline
  expansion have no demand clause (expressions.md:2877-2878, :3024-3026).
- **The Baker contract document still prescribes eager snapshots** (vestigial).
  Baker_Saturation_Architecture.md:145-147 describes producers that evaluate
  supplied expressions in the forming scope. No Baker, Alex or pipeline document
  describes the source publication seam.
- **The register and drift gate were not updated for this campaign** (blind
  spot). No row retires the numbered gate series or eager ordinary evaluation.
  Several SCHEDULED and ALLOW rows exempt deleted files, such as the
  PSGElaboration directory and SSAAssignment.fs
  (Design_Supersession_Register.md:17-72, drift-gate.sh:42-72).
- **Legacy design documents present Alex-owned strategies as current**
  (vestigial). CCS_Lazy_Seq_Coroutine_Intrinsics.md assigns compilation strategy
  to Alex (:51, :646). In-code comments cite the deleted SSAAssignment pass
  (LambdaWitness.fs:666, PSGCombinators.fs:214).
- **The F#-to-Clef guide does not mention call-by-need or `eager`** (blind spot).
  From_FSharp_to_Clef.md is where an F# developer meets the departure.

### Companion libraries

- **Console output sits in discarded bindings** (latent, high). Every console
  write in Fidelity.Platform is `let _ = Sys.write STDOUT s` followed by `()`
  (Console.clef:28, :34, :39, :45). Under the specified law a discarded ordinary
  binding is not an effect root, so all console output stops executing once
  ordinary deferral lands. BAREWire's Envelope and Btf encoders use the same shape,
  which puts RoundTrip's frame oracle in the same position. The platform library
  must move to statement sequencing or explicit `eager` in the same changeset as
  the demand work.
- **No declaration carries the foreign-boundary demand contract** (blind spot).
  expressions.md:3024-3026 requires a foreign or platform call to declare its
  materialization and effect order. No descriptor, endpoint record, spec chapter
  (ffi-boundary.md, platform-bindings.md), Farscape output or CCS reader carries
  it.
- **CCS removed the `int` of `char` kind function that D10 keeps** (latent).
  Dimensional_Range_Design.md:525 retains it. Platform `b1aaf62` replaced it in
  Parse.clef with a ten-arm match, which is why 06 now passes. The checkpoint
  records neither the library change nor the removed kind function.
- **Sample 17 belongs to the Fidelity.Libc regeneration** (sharpen). CMemory.clef
  is a subset of Farscape-generated bindings. Its owner is the D10 Farscape leg,
  not fixture reconciliation.
- **The drift gate does not cover the libraries** (broaden). drift-gate.sh:18-31
  includes BAREWire/docs only, so BAREWire/src, BAREWire/samples,
  Fidelity.Platform and Farscape are outside it.
- **The platform-description vocabulary is ill-typed Clef in every profile**
  (latent). BAREWire's Description.fs uses `int64` fields, the errors are demoted,
  and CCS reads the values syntactically (Description.fs:27-36).
- **13a_BAREWireCollections tests collection patterns BAREWire's Clef sources do
  not use** (vestigial).

### PRD inventory against code

Coverage runs through four stages: source admission, Baker recipe, Alex witness
and a native case. The prior sections already cover the bare absence of List,
Map and Set schemes, the `Set.remove` loss, AVL rotation, `Map.remove`, the
unresolved recursion idiom and `Pure` on partial operations.

| PRD | Source | Recipe | Witness | Native |
|---|---|---|---|---|
| C-04 List | `::`, `@` and literals only | 15 operations, none reachable | Witnesses match `Intrinsic` nodes with children, while Baker emits `Application(Intrinsic, args)` | None |
| C-04 Map and Set | None | Partial, as above | `isEmpty` rejected, and no node stores height | None |
| C-04 Option | All but `toList` | All admitted, `get` unguarded | Present | 08a–e, failing at HEAD |
| C-04 Array | 12 operations; `map`, `fold`, `sum`, `sumBy` absent | Six admitted names have no recipe or witness | zeroCreate, get, set, sub, length, blit | 20, blocked by D10 |
| C-04 ranges | Counted `for` only; list and array ranges rejected with CCS8401 | None | None | None |
| C-06/C-07 successors | All but `maxBy` | All, with unguarded Current reads | `Seq.empty` absent | None |

Further findings:

- **Partial selectors are source-admitted with unguarded reads** (latent).
  `Seq.head`, `min`, `max` and `minBy` bind the `moveNext` result and never test it
  before reading Current (SeqRecipes.fs:437). `Option.get` reads the payload with
  no guard.
- **Recipes emit intrinsics that one-pass saturation never decomposes** (latent).
  `toList` and `toArray` emit `List.rev` and `List.toArray`, and `List.toArray`
  has no implementation (NativeService.fs:1043-1048).
- **List recipes invent key and state types** (latent). `minBy`, `sumBy` and
  `forall2` default to the element type or `int` (ListRecipes.fs:547-551).
- **Unsupported patterns throw instead of diagnosing** (broaden). The `let true`
  crash is one case of a family. The checker throws `failwith` for unsupported
  let, tuple and parameter patterns (Bindings.fs:62-141), and Baker throws
  `failwithf` in MatchRecipes. Or, As, And, array, type-test and cons patterns
  reach these paths. Refutable or record parameter patterns silently become `_`
  (Bindings.fs:227-238). Literal-identifier patterns become a tag-0 union test
  (Patterns.fs:78-99).
- **F-06 ordered input passes only because the runner paces stdin** (blind spot).
  RunnerCore writes each stdin line with a 50 ms delay (RunnerCore.fsx:143-153),
  which hides the multi-line `readln` defect. The CCS Parse and Format resolvers
  are dead code, and F-06 still describes Parse as a compiler intrinsic.
- **`Bits.*` is registered with width-named types and retired bitcasts, and has
  no witness** (vestigial, F-07). No obligation enforces the valid-shift-count
  precondition (numeric-selection.md:326).
- **C-03 has no tail-call mechanism to prove**, and recursive value bindings are
  accepted without distinction (Bindings.fs:870-907).
- **Operators and intrinsics outside Option, Result and Seq are never reified as
  callback values** (blind spot, C-02) (BakerSaturation.fs:359-386).
- **C-04 §3.2 gives `Map.keys` and `Map.values` list results**, while the spec,
  recipe and samples use `seq` (sharpen).

### Recursion and productivity under call-by-need

From the Fixed-Point Scaffolding and negative/fractional-type cross-check. The
`let rec` contradiction was re-read at HEAD.

- **Re-entrant demand has no native contract** (blind spot, decision). The spec
  does not say what happens when a value is demanded while its own computation
  is in progress. The inherited Recursive Safety Analysis
  (inference-constraint-solving.md:358-427) is F# text for eager evaluation:
  it rewrites to `lazy`, inserts runtime checks and raises an exception. Module
  value cycles are equally unspecified. This blocks C-03 §5 and C-05 §7, and
  neither the checkpoint nor its order schedules it.
- **The spec both removes and specifies `let rec`** (decision).
  namespaces-and-modules.md:16-21 states normatively that Clef has no `let rec`
  and no `and` grouping. The expressions grammar (expressions.md:46, :219), the
  inference chapter, C-03 and sample 13 use `let rec ... and`. The C-03 acceptance
  surface depends on which statement holds.
- **C-03's two stack gates are one dependency** (sharpen). Under call-by-need an
  accumulator loop reaches bounded stack only with a strictness proof, and that
  proof needs the same recurrence-range evidence original13 lacks.
- **Non-strict payloads conflict with the collection layouts** (blind spot). The
  list and union layouts store the tail at construction and assert that no node
  is mutated (VC-RO). Deferred payloads and productive cyclic values have no
  representation or lifetime rule there.

This cross-check also produced notes for the owner's hand review in
`arxiv-papers/research/FPS/`. One covers productivity stated as PSG guardedness,
together with `eager` as a `seq`-like operator. The other covers incremental
re-stabilization as a fourth fixed-point construction, with parallel segment
fold-in as the paper's extension problem. A proposed note for the
negative/fractional-type paper awaits a research folder. It concerns the §7.2
recursive knot and the §6.1 copying boundary under call-by-need sharing.

### Dimensional typing, memory management and discharge

From the DTS/DMM and Decidable by Construction cross-check. The dimensional core
conforms to the papers. `solveDim` (DimensionAlgebra.fs:199-276) is an abelian
unifier with a stated termination argument. Exponent overflow reports CCS8048.
Measure binders stay out of specialization keys, so dimensions never change
emitted code. Ranges, widths and escape never pass through the unifier. The
gaps are in memory management and in discharge:

- **An unobservable integer range is information on cores** (sharpen, decision).
  CCS8011 is an error on fabric and Info on cores (RangeAnalysis.fs:2241), and
  `heldWidthOf` holds such a value at the Register width (:2676-2692). The code
  records this as the single interim of Dimensional_Range_Design §1.3, pending a
  migration inventory. The owner's ruling keeps unobservable ranges as errors
  with no silent default, and no C/F item schedules the promotion. original13
  (C-03) and F-07 widths depend on it.
- **No build discharges obligations** (blind spot). `ObligationDischarge.emit`
  writes the ledger and SMT module only when intermediates are enabled
  (ObligationDischarge.fs:374-383), and the build-time module pairs obligations
  by anchor name. A compile never runs the solver, and pairing by name is the
  identifier match the Decidable by Construction paper excludes as discharge.
- **Arena lifetime is a measure-sorted parameter unified by the group solver**
  (vestigial). memory-regions.md:61 and units-of-measure.md:197 retain the
  retired reading that every axis is an abelian group. Lifetime belongs with the
  propagated lattice axes.
- **Escape analysis conflates cause with lifetime** (blind spot). The ByRef and
  Closure causes are never produced, classes never join, no lifetime obligation
  is emitted, and Alex treats a missing class as stack-scoped
  (Types.fs:1362-1372, Escape.fs:76-100).
- **Platform descriptors disagree on the meaning of MinMagnitude** (latent). On
  ESP32-S3 the float-literal coverage check rejects 0.0 and every negative
  literal (ObligationElaboration.fs:339-345, Literals.fs:50).
- **The spec gives CCS8012 two severities** (decision). error-handling.md:331
  and native-type-universe.md:126 disagree. The papers and code treat it as an
  error.

This group's paper notes are placed for the owner's hand review:

- `research/DTS-DMM/`: capturing an unsettled binding retains shared memo
  storage whose lifetime covers every capturer, and deferred member relations are
  a third condition on generalization;
- `research/DBC/`: early evaluation is a termination and effect obligation
  admitted through a conservative fragment, and Tier 2 separates incomplete
  synthesis from decidable checking with target-relative enclosures;
- `research/grade-axis/06-discharge-rows-and-inference-exchange.md`: the Width
  and escape discharge rows, and fact exchange between lattice axes during
  inference.

## Research note for the owner

`arxiv-papers/research/PHG/structural-hull-certificates.md`, with the checker
`research/PHG/check-hull-certificates.fsx`, is placed for the owner's hand
review. The existing note's theorem is correct, and its checker reproduces the
known counts of chordal graphs. The new note constructs the clique-tree
certificate from the PSG. Each hyperedge's participants are closed to their
minimal subtree in one structural tree: the containment tree for rewrites, the
dominance tree for runtime values. The result is chordal by Gavril's theorem,
the per-node membership sets form the certificate, and a top-down pass colors
optimally without building the conflict graph. Strict SSA register allocation
is the established instance (Hack, Grund and Goos 2006). The price is
conservativeness where hulls meet at nodes no participant occupies. A hyperedge
spans segments exactly when its hull crosses a segment boundary. The checker
passes every labeled tree on one to five nodes with one to four hyperedges
(6,607,900 families) and 20,000 seeded larger instances.

## Evidence

- HEAD manifest: [evidence/2026-09-26-head-manifest-audit.json](evidence/2026-09-26-head-manifest-audit.json),
  extracted from runner results SHA-256 recorded in that file.
- Full runner artifacts, CCS and Alex logs, and the structured audit reports are
  under `/tmp/claude-1000/-home-hhh-repos-clef/402241eb-a4c4-43b3-9a24-eeb5e3905ddb/scratchpad/audit/`.
  That location is volatile, and the extraction above is the durable record.
