# Independent assessment: PSG-only separation at the September 27 notch

This assessment is for the owner and the implementer continuing the
rearchitecture. It audits the rewritten
[C_F_Checkpoint_2026-09-26.md](C_F_Checkpoint_2026-09-26.md) and the code at
**clef `b77f888`, Composer `b655e38`, Fidelity.Platform `d7cc3b7`**. It replaces
every earlier version of this document. Those versions described earlier
revisions and remain in history at Composer `3a5b7af` and before.

The question under audit is whether transforms are fully removed from Alex
Elements, Patterns and Witnesses, and whether Baker Ingredients and Recipes now
fully elaborate and saturate the graph and hold every programming decision in
the hypergraph.

## 1. The rule

The owner's architecture, stated without exception:

1. **Only the PSG, or elements of the PSG, is transferred to Alex.**
2. **No MiddleEnd code opens a CCS namespace or calls a CCS function.**

Anything Alex obtains beyond the PSG in order to lower has exactly one of two
causes, and each cause has one disposition:

| Cause | Disposition |
|---|---|
| The PSG is not fully elaborated and saturated | Baker establishes the fact as graph structure and publishes it. Alex then reads it. |
| Alex duplicates work Baker already does | The Alex code is vestigial and is deleted. |

No third disposition exists. A CCS call that "only reads" still breaks the
rule, because it delivers a fact by executing CCS code inside the middle end
rather than by transferring the PSG.

The outward transfer from Alex is serialization and flattening. The witnessed
operations are flattened into a module and spelled as MLIR text. Both are
necessary, and both are tightly scoped. Serialization spells each operation
that was constructed from PSG facts, one to one. It does not select an
operation, rewrite one, name a value outside the derivation or decide a size,
width or layout. Flattening assembles the witnessed operations at their
settled scopes. For incremental compilation, serialization and flattening will
be segmented so that neither remains one monolithic step. The segment
boundaries are source-authorized PSG regions. Neither the serializer nor the
flattener chooses them.

The design of record states the same contract:

- clef-lang-spec
  [backend-lowering-architecture.md:57](../../clef-lang-spec/spec/backend-lowering-architecture.md):
  "Witnessing SHALL remain a passive observation through the Huet zipper;
  missing semantic prerequisites SHALL be reported, not reconstructed in Alex."
- clef-lang-spec
  [program-semantic-graph.md:572](../../clef-lang-spec/spec/program-semantic-graph.md):
  "CCS/Baker elaboration and saturation → settled publication → passive Alex
  witnessing → Composer backend".
- clef-lang-site
  [learning-to-walk.md:122](../../clef-lang-site/hugo/content/docs/internals/pipeline/learning-to-walk.md):
  the passive zipper, in which every ordering, strategy and dependency decision
  was made during PSG construction.
- clef-lang-site
  [baker-saturation-engine.md:47](../../clef-lang-site/hugo/content/docs/internals/pipeline/baker-saturation-engine.md):
  Alex "cannot repair missing source semantics from a physical slot or a
  familiar operation name".
- [Composer AGENTS.md](../AGENTS.md) lines 8-11 and
  [clef AGENTS.md](../../clef/AGENTS.md) lines 8-9.

The checkpoint's own statement at lines 142-143 is weaker than the rule. It
treats a CCS call from Alex as a violation only "if that function performs
selection or semantic repair". Under the rule, every call is a violation.

## 2. Verdict

**No.** Transforms are not fully removed from Alex, and Baker does not yet hold
every decision. The notch removed large decision paths. The shared width
selector, the string algorithms, the array loops and the hardware and kernel
declaration analysis are gone from Alex, and that work is real. What remains is
systemic.

| Measure | Result |
|---|---:|
| MiddleEnd source files | 88 |
| Files that reference CCS | 79 |
| Files that open a CCS namespace | 75 |
| Distinct CCS namespaces opened | 10 |
| Files with at least one confirmed decision, transform or CCS call | 66 |
| Confirmed MiddleEnd findings | 248 |
| Of which the PSG is incomplete | 146 |
| Of which Alex duplicates Baker | 77 |
| Of which a target form belongs to the backend | 11 |
| Of which the code is dead | 13 |
| Confirmed clef findings: publication-time analysis and claim gaps | 30 |
| Confirmed Composer Core and BackEnd findings | 14 |
| Checkpoint removal-table rows: hold, partial, fail | 6, 8, 2 |

The checkpoint discloses some residue: BorrowedView (lines 207 and 493) and the
FPGA function, record and conditional Patterns (line 500). It does not disclose
the CCS dependence at all, and it claims two rows that the code contradicts
(section 4).

## 3. The failure mode

The rule is not enforced by the build, so every rewrite can re-grow the
dependence it was meant to remove.

**Composer references the whole compiler.** `src/Composer.fsproj:229` is a
`ProjectReference` to `Clef.Compiler.Service`. Every CCS analysis, resolution
and checker module is therefore visible to every Witness, Pattern and Element.

**The PSG types live in CCS namespaces beside checker functions.** Alex cannot
name a `SemanticNode` without opening a namespace that also exports analysis.

| CCS namespace opened in MiddleEnd | Open lines | What else the namespace exports |
|---|---:|---|
| `PSGSaturation.SemanticGraph.Types` | 69 | Projection types, beside helper modules |
| `NativeTypedTree.NativeTypes` | 58 | `PlatformContext.tryWidth` and other platform functions |
| `PSGSaturation.SemanticGraph.Core` | 17 | `SemanticGraph.tryGetNode` (32 calls), graph edits, `invalidateWitness` |
| `PSGSaturation.SemanticGraph.WitnessEmission` | 4 | `tryRead` and seven domain readers (66 qualified calls) |
| `NativeTypedTree.UnionFind` | 3 | `applySubst`, the checker's type substitution |
| `SemanticGraph.PlatformResolution` | 1 | Declaration readers |
| `SemanticGraph.MappedBindings` | 1 | Whole-graph mapping discovery with a graph-identity cache |
| `SemanticGraph.ExplicitDemand` | 1 | Eager-marker resolution |
| `SemanticGraph.BorrowedViews` | 1 | Schema discovery with a graph-identity cache |
| `NativeTypedTree.DimensionAlgebra` | 1 | The dimensional solver |

**The projection is delivered by calling CCS code, not by transferring data.**
The 66 `WitnessEmission.try*` calls each run `tryRead`
(clef `WitnessEmission.fs:93-106`), which checks provenance and materialization
and then returns the projection held in `graph.Codata`. The data is PSG
content. The delivery is a CCS call at every site, repeated per node.

**Facts outside the PSG are handed to Alex beside it.** `MLIRGeneration.fs`
receives a `PlatformContext` with the graph and builds `TransferCoeffects` from
it (`:33-35`, `:61-64`). Every Pattern can branch on the target enum and on
widths read through `PlatformContext.tryWidth`, none of which arrives as PSG.

**The required correction was dropped from the record.** The checkpoint at
Composer `3a5b7af` listed "isolate Alex behind an immutable contracts assembly
with no source-analysis dependency" as required work. The rewritten checkpoint
omits it.

The structural requirement follows directly from the rule:

1. The PSG is carried in a contract that contains only immutable PSG types and
   published projections. CCS produces it. It exposes no analysis, resolution,
   substitution or solver.
2. The MiddleEnd compiles against that contract alone and has no reference to
   `Clef.Compiler.Service`.
3. The contract is admitted once at transfer entry, and the published
   projection travels with the occurrence. No per-site call reaches back into
   CCS.

With that boundary in place, every finding in section 5 that calls or opens CCS
becomes a compile error. Each error then resolves in one of the two ways in
section 1. The build enforces the rule, and review no longer carries it alone.

## 4. The checkpoint's removal table, row by row

The table is at checkpoint lines 123-140.

| Row | Claim | Verdict | Evidence |
|---|---|---|---|
| 125 | Shared type and width selection moved to Baker | **Partial** | Baker materializes carriers as rows, but it evaluates the selection by calling the PSGSaturation readers `RangeAnalysis.heldWidth` and `selectedRepresentation` (clef `NumericCarrierRecipes.fs:59-61`, `:89-99`). The selection rule is not a Baker ingredient. Alex still reads Register and Pointer widths through `PlatformContext.tryWidth` (`MLIRGeneration.fs:33-35`) and offers a platform-word width independent of any carrier (`PSGCombinators.fs:67`, `:100-107`, `TransferTypes.fs:35-50`). |
| 126 | No local MemoryWitness width correction | Holds | clef `MemoryAccessRecipes.fs:187`, `:238-256` take the shared carrier. `MemoryWitness.fs:14-24` only dispatches. |
| 127 | Numeric operation signedness, carrier and adaptation in Baker | **Partial** | Rows exist only for `Operators` intrinsics (clef `NumericOperations.fs:7-28`). `Convert.toFloat` and real operations are refused (`NumericOperationRecipes.fs:128-129`). `integerSlot` (`NumericOperations.fs:29-46`) is a second selection rule. Alex chooses conversion signedness (`ApplicationPatterns.fs:392-412`, `:345-366`) and float comparison ordering (`:246-254`). |
| 128 | Memory field-name and type interpretation removed | **Partial** | Holds for the migrated memory families. Records still resolve fields by name in Alex (`RecordPatterns.fs:32-35`, `:104-112`, `RecordWitness.fs:85-120`). Collections still decide layout and case numbering (`CollectionPatterns.fs:27-305`). |
| 129 | No address by copying a scalar | **Partial** | clef `MemoryAccessRecipes.fs:52-74`, `:92-96`, `:497-499` hold. `ApplicationPatterns.fs:398-411` still chooses `fptosi`/`sitofp` signedness and index casts, and extracts a base pointer for a conversion. |
| 130 | Array loops and copies constructed by Baker | Holds | clef `ArrayConstructionRecipes.fs:94-227`, `ArrayMemoryRecipes.fs:161-356`. |
| 131 | String equality constructed by Baker | Holds | clef `StringComparisonRecipes.fs:108-192`. |
| 132 | String search, concatenation and indexing algorithms deleted | Holds | `Patterns/StringPatterns.fs` is deleted. `StringIntrinsicWitness.fs:16-21` refuses what is not published. The capabilities remain missing, as the checkpoint states. |
| 133 | fromBytes and toBytes identity shortcuts removed | **Partial** | Baker validates the Meet at `valueSite` but publishes the raw operand (clef `MemoryAccessRecipes.fs:238-262` against `:415`, `:446-448`). Alex passes the raw operand to `pPublishedAdapt` (`MemoryPatterns.fs:492-495`, `:542-545`). `lastValueNode` then rediscovers the site (`PSGCombinators.fs:140-158`). The checkpoint's toBytes failure (lines 431-440) is this defect. |
| 134 | One passive dispatch selector for scalar, lazy and sequence dispatch | Holds | clef `NumericCarrierRecipes.fs:157-195` publishes `NumericIndexTransport`. `ControlFlowPatterns.fs:57` is the shared selector. The Boolean conditionals are a separate finding (section 5.3). |
| 135 | Structural, annotation and environment shape discovery removed | **Fails** | The authority is computed at publication, not written by Baker. clef `CallableEmission.fs` derives value shapes through `CallableCarriers.valueShape` over `applySubst node.Type` (`CallableCarriers.fs:127-133`), symbols from ModuleDef parents and metadata (`:13-59`), and transports by walking graph structure (`:103-139`). Alex still discovers shape (`StructuralWitness.fs:99-131`, `TypeAnnotationWitness.fs:53-55`). |
| 136 | VarRef classification removed | **Fails** | The classification moved into the publication reader. clef `CallableEmission.fs:217-239` tests for Lambda children, `Codata.Closures` membership and captures, and `:282-305` walks VarRef and Binding chains. Alex still selects the read path by structure (`VarRefWitness.fs:39-213`). The checkpoint's own lines 144-145 state that moving a transform into a publication reader does not make it passive. |
| 137 | Raw constant match normalized by Baker | Holds | clef `MatchRecipes.fs:118-152`, `:187-196`. |
| 138 | Hardware declaration analysis moved to Baker | **Partial** | clef `HardwareModuleRecipes.fs:13-247` settles the Mealy contract. It is published as one opaque record on one edge with ordinal 0 (`SpatialValues.fs:15-17`), not as roles. The output component is computed (`:178`) but not published, so the backend re-derives it (`HardwareRealization.fs:90-92`) and chooses reset semantics (`:40-49`). Unpinned outputs are dropped without a diagnostic (`HardwareModuleRecipes.fs:172-178`). |
| 139 | Kernel opcode discovery moved to Baker | **Partial** | clef `KernelModuleRecipes.fs:82-138` builds the steps with a mutable `ResizeArray` and publishes one opaque record on one edge with ordinal 0. `Nanopass/KernelDeclarations.fs:8-13` swallows the ingress diagnostic. |
| 140 | Platform signature discovery and marshalling removed | **Partial** | The import and call boundary holds (clef `BoundaryRecipes.fs:103-229`, `:338-499`). Platform core and target facts are still re-read from declarations in Composer (`CompilationOrchestrator.fs:106`, `:260-279`, `BackEnd/MCU/Target.fs:53-119`). |

## 5. What Alex still does

Every anchor below is confirmed by an independent refuter or was verified by
the auditor. Appendix A lists all 248 MiddleEnd findings by file.

### 5.1 CCS calls and checker reads from the MiddleEnd

These eighteen findings, shown in seventeen rows, break the rule by execution,
not only by dependency.

| Anchor | CCS function or state | What Alex obtains | Disposition |
|---|---|---|---|
| `Witnesses/BorrowedViewWitness.fs:9-11` | `BorrowedViews.operation` | Operation kind, view operand and schema layout, found by whole-graph descriptor scan with a graph-identity cache (clef `BorrowedViews.fs:63-64`) | PSG incomplete |
| `Patterns/BorrowedViewPatterns.fs:15-82` | BorrowedViews layout | Element width, range, alignment and access permission | PSG incomplete |
| `Patterns/BorrowedViewPatterns.fs:25-28` | `MappedSpans.forLayout` | Pointer bits, maximum extent, load/store alignment | PSG incomplete |
| `Witnesses/MappedViewWitness.fs:4`, `:12` | `MappedBindings.tryFindCall` | Mapping discovery by `readDescriptors` over the whole graph, with a graph-identity cache (clef `MappedBindings.fs:55`, `:163-176`) | PSG incomplete |
| `Witnesses/MmioWitness.fs:7-9` | `Mmio.operation` | Operation classification through `Predicates.valueOf` | Duplicates Baker |
| `Patterns/MmioPatterns.fs:9-12`, `:35-37`, `:46-51` | `applySubst` (`:35`), with an unused `PlatformResolution` open (`:12`) | Address and transaction width | Duplicates Baker |
| `Patterns/MmioPatterns.fs:10`, `:12`, `:35-37` | `applySubst` | Width agreement with the register handle | Duplicates Baker |
| `Patterns/EagerPatterns.fs:19`, `:27-30` | `ExplicitDemand.operand` | The operand of an explicit demand | Duplicates Baker |
| `Patterns/EagerPatterns.fs:10`, `:45-46`, `:52` | `applySubst` | Callable-forward versus scalar and unit paths | Duplicates Baker |
| `Patterns/EnvironmentPatterns.fs:41-44` | Source type inspection | Callable versus data treatment of an environment source | Duplicates Baker |
| `Patterns/ContinuationPatterns.fs:153-172` | Source type inspection | Descriptor-only versus ordinary slot write path | Duplicates Baker |
| `MLIRGeneration.fs:33-35` and uses | `PlatformContext.tryWidth` | Register and Pointer widths for every layout and the serialized module | Duplicates Baker |
| `MLIRGeneration.fs:30-35`, `:50`, `:61-63` | Platform context beside the graph | Width context handed to every Pattern | PSG incomplete |
| `MLIRGeneration.fs:57-59`, `:99-105` | `ObligationDischarge.ofGraph` | The obligation inventory for the SMT artifact | PSG incomplete |
| `MLIRGeneration.fs:97`, `:104`, `:119` | `PhaseConfig.isVerbose` | Diagnostic gating, no semantic effect | Duplicates Baker |
| `Traversal/MLIRTransfer.fs:58` | Width read through `PlatformContext` | Pointer-integer width for MMIO address spelling | Duplicates Baker |
| `XParsec/PSGCombinators.fs:22` | `open ... UnionFind` | Nothing today, but exposes checker cells to every Pattern | Delete (dead) |

Two of these reach CCS caches keyed by graph object identity
(clef `BorrowedViews.fs:63`, `MappedBindings.fs:163`). Earlier assessments
asked for that mechanism to be retired.

### 5.2 Platform and target facts outside the PSG

- `MLIRGeneration.fs:162-172` re-decides the CCS8203 refusal per target and
  exempts FPGA, which duplicates clef `PlatformDeclaration.fs:164-192`.
- `MLIRGeneration.fs:127-130` chooses the activation kind from the target enum.
- `Traversal/WitnessRegistry.fs:71-79`, `:99-125`, `:165-177` decides which
  language constructs are admitted per substrate. That is a source admission
  fact.
- `Traversal/XDCTransfer.fs:10-82` and `MLIRGeneration.fs:108-124` produce the
  FPGA constraint artifact in the middle end. The clock period rounding at
  `XDCTransfer.fs:39-41` is a decision.
- Eleven findings are target forms in the MiddleEnd that belong to the backend. Among them:
  - `ApplicationPatterns.fs:118-138` and `:147-163` choose a spatial instance
    or a temporal call, and the dialect of each numeric operation.
  - `ControlFlowPatterns.fs:121-205` chooses a mux or structured regions.
  - `ClosurePatterns.fs:34-86` chooses `hw.module` or `func.func`.
  - `RecordPatterns.fs:56-84` and `DUPatterns.fs:71-133` choose the fabric
    aggregate form.
  - `Dialects/Core/Serialize.fs:518-527` spells an LLVM volatile device access.

### 5.3 Representation and layout decisions

Ninety-one findings are decisions. The recurring ones:

- **Union tags and case numbering.** Tag width, offset and case numbers are
  chosen in `ControlFlowPatterns.fs:368-391`, `MemoryPatterns.fs:49-71`,
  `CollectionPatterns.fs:27-305`, `DUWitness.fs:73-74`,
  `MapWitness.fs:41-125`, `SetWitness.fs:41-110`, `TypeMapping.fs:39` and
  `Serialize.fs:69-75`.
- **Scalar carriers and sizes.** Unit and Char are i32 (`TypeMapping.fs:62`).
  Bool, Char and Unit widths are set in `Dialects/Core/Types.fs:56-67`. Byte
  sizes and pointer size are computed in `Types.fs:97-133`. Access alignment is
  hard-coded in `MLIRAtomics.fs:44-93`.
- **Residence and allocation kind.** `MemoryPatterns.fs:211-232` places
  escaping aggregates on the heap. `MemRefPatterns.fs:34-57` places mutable
  cells. `ContinuationPatterns.fs:272-276` places escaping frames.
  `MutableCallablePatterns.fs:63-92` lays out the callable cell.
- **Signedness of conversions and index transport.**
  `ApplicationPatterns.fs:345-412`, `MemoryPatterns.fs:354-358`, `:448-451`, and
  the BorrowedView guards all choose it.
- **Boolean conditionals over lazy and sequence values.**
  `LazyPatterns.fs:189-196` and `SequencePatterns.fs:117-125` choose an index
  switch and a fixed unsigned cast of the condition.

### 5.4 Calling conventions and callable identity

Twenty-three findings post-process a projection into a further fact. Most of
them assemble the calling convention:

- `Traversal/CallableOperands.fs:29-156` expands code and environment
  components. At `:119-129` it gives unit formals zero components, which
  contradicts Baker's `Arguments`.
- `Witnesses/LambdaWitness.fs:68-91` and `:239-371` decide the function
  signature, return convention, visibility, unit returns and native callback
  adapters.
- `Witnesses/ApplicationWitness.fs:99-164` decides the call form.
- `Traversal/LazyOperands.fs:36-41` and `Traversal/SequenceOperands.fs:24-26`
  hold second derivations of their conventions.

These follow from rows 135 and 136. The convention is computed in clef
`CallableEmission.fs:240-281` at publication, and Alex completes it.

### 5.5 Structure discovery

Thirty-nine findings navigate beyond the current occurrence to find a fact
that should be published:

- `PSGCombinators.fs:922-935` finds the value node of a block, branch, arm,
  initializer or body by last-value discovery.
- `PSGCombinators.fs:140-158` rediscovers which Meet applies.
- `PSGCombinators.fs:328-348` classifies intrinsic applications.
- `VarRefWitness.fs:39-213` selects the read path by structure.
- `MutableAssignmentWitness.fs:73-96` infers the storage class of a target from
  the MLIR type of an earlier emission.
- `CoverageValidation.fs:19-30` decides which occurrences must be witnessed.
- `NanopassArchitecture.fs:356-414` decides traversal roots and order, and
  therefore which occurrence first witnesses a shared node.
- `NanopassArchitecture.fs:137-138` and `LambdaWitness.fs:51-62`, `:211-216`
  decide whether a shared occurrence is re-witnessed per function.

### 5.6 Constructions, fallbacks and silent failures

Twenty constructions build algorithms in Alex:

- `ControlFlowPatterns.fs:243-478` builds union case elimination.
- `MutableCallablePatterns.fs:151-177` builds a dispatch with a default arm.
- `RecordPatterns.fs:154-232` builds a copy through a runtime call.
- `BorrowedViewPatterns.fs:24-104` builds guard algorithms.
- `ContinuationPatterns.fs:208-215` and `:287-298` build enumerator
  acquisition and discriminant initialization.
- `Traversal/SMTTransfer.fs:34-804` builds the clause content of every proof
  obligation at build time. Baker publishes only the obligation, so Alex
  constructs its logic.

Seventeen fallbacks supply a value when a fact is absent. These take priority,
because a swallowed failure hides every defect behind it:

| Anchor | What is supplied when the fact is absent |
|---|---|
| `XParsec/PSGCombinators.fs:150-167`, `:850-858` | No Meet is read as no adaptation |
| `Traversal/CallableOperands.fs:64-80` | An absent demand row is read as nothing omitted |
| `Patterns/SequencePatterns.fs:19-22` | A missing publication is classified as not a sequence |
| `Patterns/ClosurePatterns.fs:92-99` | A unit function returns a fabricated zero |
| `Patterns/LiteralPatterns.fs:66-76` | A unit value 0 is materialized where no operation produced one |
| `Traversal/WitnessRegistry.fs:113-114` | Ownership is chosen by first match and emitted type |
| `Witnesses/IntrinsicWitness.fs:21-26` | An intrinsic occurrence is treated as carrying no value |
| clef `Nanopass/KernelDeclarations.fs:8-13` | An ingress diagnostic is swallowed |
| clef `Baker/Recipes/HardwareModuleRecipes.fs:172-178` | Unpinned outputs are dropped |

### 5.7 Name minting

Twenty-three findings construct names outside a derivation:

- `Traversal/Values.fs:16-43` defines banded ordinal families that can
  collide. A meet ordinal of 100 or more reaches the 1100 and 1200 bands.
- Program storage and declaration symbols are spelled in Alex:
  `MemoryPatterns.fs:159`, `TransferTypes.fs:720-724`, `VarRefWitness.fs:163`,
  `BindingWitness.fs:111`, `MutableAssignmentWitness.fs:61`,
  `CallablePatterns.fs:127`, `EnvironmentPatterns.fs:61-63` and
  `CodeGeneration/CallableSymbols.fs`.
- Symbol identity is a source fact. Value names are a derivation of the
  settled graph, not a Pattern's choice.

### 5.8 Serialization beyond spelling

`Dialects/Core/Serialize.fs` does more than spell:

| Anchor | What the serializer does |
|---|---|
| `Serialize.fs:69-75` | Chooses the fabric union tag width |
| `Serialize.fs:271-277` | Renames SSA values in `hw.module` bodies by text replacement of `%argN` with port names |
| `Serialize.fs:483-503` | Rewrites a constructed `ReinterpretCast` into `arith.constant` plus `memref.view`, and mints `_off` names |
| `Serialize.fs:518-527` | Spells an LLVM volatile access (target form) and chooses its alignment |
| `Serialize.fs:576-591` | Decides string literal storage bytes and terminator |
| `Serialize.fs:600-610` | Decides the initial image of writable program storage |
| `Serialize.fs:674-677` | Spells the `scf.for` induction variable with the step's name, because `SCFOp.For` carries no induction value |

`Dialects/Core/Types.fs:56-133` and `:476-494` decide carrier widths, byte
sizes, pointer size and the parameter order of the intrinsic write
declaration. Once those facts are published and the operations carry them,
the serializer returns to its scope.

### 5.9 Duplicated validation and admission bypass

Three findings re-prove a Baker result in Alex:

- `EnvironmentPatterns.fs:21-24` re-proves the environment layout.
- `LazyPatterns.fs:85-88` re-encodes the formation rule.
- `MmioPatterns.fs:27` re-checks admissible widths.

Nine findings read settled `Codata` fields directly where the read feeds a
decision. A direct `Codata` read is a PSG read, so the refuters rejected twenty
further claims of this kind. Section 3's single admission at entry makes the
distinction moot.

## 6. Baker ownership in clef

The owner's question has a second half: whether Baker Ingredients and Recipes
hold every decision. A PSGSaturation reader is CCS, but it is not Baker. When a
reader derives a fact at publication, the PSG is not saturated at handoff, and
the decision has no hyperedge, no participants and no retraction record.
Seventeen confirmed findings are of this kind:

| clef anchor | What is computed at publication instead of written by a recipe |
|---|---|
| `PSGSaturation/SemanticGraph/CallableEmission.fs:13-59`, `:171` | Callable symbols and declaration records |
| `CallableEmission.fs:73-98`, `:140-170` | Carriers, joins, flows, call targets, parameters and dimensional instances |
| `CallableEmission.fs:183-199`, `:240-281` | The calling convention and retention residence |
| `CallableEmission.fs:217-239`, `:282-312` | Definition-only classification, SSA alias identity, unit and closed-data occurrences |
| `CallableIngress.fs:38-511` | Closed-input admission and retained continuation identity |
| `OrdinaryDemand.fs:34-308` | Omitted and eager actuals and the deferred-only set, which also governs coverage |
| `StorageWitness.fs:90-173` | Lazy occurrences, definition-only thunks, the startup plan and the literal-pool anchor |
| `Meets.fs:37-461` | Extension or truncation kind and width for every value-to-slot Meet outside numeric operations |
| `StringByteStorage.fs:8-31` | Byte element slot and range through alias traversal |
| `MemoryPublication.fs:26`, `:48-49` | Which copy constructions are executable |
| `SpatialPublication.fs:10-95` | Proof-body and adaptation correctness, re-solved during publication |
| `WitnessEmission.fs:10-31` | Runs all of the above inside `prepare` |

Three Baker structures are present but are not hypergraph structure:

- `Baker/Ingredients/SpatialValues.fs:15-17` stores a whole hardware or kernel
  plan as one opaque record on one edge with ordinal 0 and flattened sources.
  Participants, roles and order are not visible to invalidation.
- `Baker/Recipes/KernelModuleRecipes.fs:82-138` accumulates steps in a mutable
  collection.
- `Baker/Recipes/NumericCarrierRecipes.fs:89-99` delegates the selection rule
  to a reader rather than holding it as an ingredient.

## 7. Composer Core and BackEnd

The rule in section 1 governs the MiddleEnd. The backend may realize published
facts, but it may not recover missing source semantics. Fourteen findings fall
here:

- `Core/CompilationOrchestrator.fs:106` and `:259-279` call
  `PlatformResolution.resolve` and `runtimeModel` to obtain the triple, CPU,
  OS, pointer bits and runtime model. These are source platform facts to be
  published.
- `BackEnd/MCU/Target.fs:13-128` selects image descriptors, spaces and vector
  layout from declarations.
- `Core/WitnessArtifacts.fs:78-79` and `:115` call `ObligationDischarge`.
  `Core/WitnessArtifacts.fs:141-147` calls `ProgramInitialization.read`.
- `BackEnd/CIRCT/HardwareRealization.fs:40-49` chooses reset semantics.
  `:90-92` re-derives the Step output component (section 4, row 138).
- AIE and CIRCT name and construct inside their own realization. That is
  backend-owned and consistent with the rule.

## 8. Required corrections, in order

1. **Stop the silent failures first.** These are the nine entries in the table
   in section 5.6. Each becomes a located diagnostic at its owner before other
   work proceeds.
2. **Establish the PSG contract.** The MiddleEnd compiles against a contract of
   immutable PSG types and published projections, with no reference to
   `Clef.Compiler.Service`. Admission happens once at transfer entry. The
   `PlatformContext` side channel in `TransferCoeffects` is removed. Every
   remaining CCS call then fails to compile.
3. **Move publication-time analysis into Baker.** The derivations in section 6
   become recipes that write typed rows with roles, ordinals, participants and
   rewrite provenance. Publication then validates and projects those rows. This
   resolves rows 135 and 136, and it is the prerequisite for most of section
   5.4.
4. **Publish the 146 missing facts by theme:**
   - representation and layout: tags, case numbers, carriers, sizes, alignment
     and residence
   - calling conventions
   - value occurrences: the value node, and the Meet keyed on the actual
     operand
   - symbols
   - obligation clause content
   - target admission
5. **Delete what Alex duplicates (77) and what is dead (13).**
6. **Move the 11 target forms to the backend.**
7. **Restrict serialization to spelling and flattening to assembly at settled
   scopes.** Segment both along source-authorized PSG regions when incremental
   compilation arrives.
8. **Restore the contracts-assembly item to the checkpoint and correct its
   removal table** to the verdicts in section 4.

## 9. What this notch teaches

1. **A boundary that the build does not enforce is not a boundary.** The notch
   removed substantial decision paths. Seventy-nine of eighty-eight MiddleEnd
   files still depend on CCS, because nothing prevents it.
2. **Relocation is not ownership.** A decision moved from Alex into a
   PSGSaturation reader satisfies "not in Alex" and fails "established in the
   hypergraph". The checkpoint's own lines 144-145 say this, and rows 135 and
   136 contradict them.
3. **Passivity is judged by what crosses the boundary.** It is not judged by
   what a function is called or whether it appears to only read. The PSG
   crosses, and nothing else does.
4. **Absence is a refusal, never an identity.** "No Meet, so no adaptation"
   and "no demand row, so nothing omitted" convert a missing fact into a
   plausible answer.
5. **Serialization is spelling.** A serializer that selects, rewrites or names
   is a second emitter.

## 10. Method, evidence and limits

- **Scope.** This is a read-only audit of the named revisions. No build, test,
  sample or gate was run, and the checkpoint's test counts were not re-run.
- **Mechanical inventory.** A grep of every `open Clef.`, module alias and
  qualified `Clef.Compiler.` reference in `Composer/src/MiddleEnd`, `Core` and
  `BackEnd`.
- **Readers.** Thirteen independent readers covered all 88 MiddleEnd files,
  each file read completely. Five further readers audited the Baker side of
  each removal-table row, the publication seam and Composer Core and BackEnd.
- **Refuters.** Every reported violation went to a separate refuter instructed
  to refute it and to correct anchors, class and disposition. Of 354 claims,
  316 were confirmed and 38 refuted. Refuted claims are excluded. Among them
  are the short-circuit claim (CCS rewrites `&&` and `||` at clef
  `NativeService.fs:1480-1492`) and most direct-`Codata` claims.
- **Merging and additions.** Findings reported by more than one unit were
  merged by file and first line, leaving 288. The auditor verified four more
  by reading the code: three serializer items and the `PhaseConfig` call.
  Appendix A holds 292 rows.
- **Appendix text.** The "What it decides" column is the classifier's
  description. Anchors, class and disposition carry the refuter's corrections.
- **Out of scope.** Tests, the specification and BAREWire were not audited
  against the rule.
- **Volatile evidence.** The run transcript is under
  `/home/hhh/.claude/projects/-home-hhh-repos-clef/7ffcb8b0-af49-4aed-9734-e779f1615b5d/subagents/workflows/wf_64138467-ce2/`.
  This document is the durable record.

## Appendix A. Confirmed findings by file

Classes: CCS call, Checker state, Structure discovery, Decision, Construction,
Fallback, Name minting, Target form, Duplicated validation, Projection
post-processing, Publication-time analysis, Claim partial, Claim fails,
Admission bypass, Dead vestige, Seam. Dispositions: PSG incomplete (Baker must
establish and publish the fact), Duplicates Baker (delete the Alex code),
Backend-owned (move to the backend), Delete (dead).

### A.1 Composer MiddleEnd

| File | Lines | Class | Disposition | What it decides |
|---|---|---|---|---|
| Alex/CodeGeneration/CallableSymbols.fs | 5-6, 8-12, 16-26 | Name minting | PSG incomplete | The spelling and uniqueness scheme of emitted callable symbols. |
| Alex/CodeGeneration/TypeMapping.fs | 39 | Decision | PSG incomplete | The bit width of the fabric DU tag discriminant. |
| Alex/CodeGeneration/TypeMapping.fs | 62 | Decision | PSG incomplete | The Unit carrier width (i32). The Char width (i32) duplicates the Baker constant. |
| Alex/CodeGeneration/TypeMapping.fs | 63-70 | Dead vestige | Delete (dead) | Integer width as IntWidth. This only repeats scalarCarrierType. |
| Alex/CodeGeneration/TypeMapping.fs | 73-76 | Projection post-processing | PSG incomplete | The storage representation of aggregates (a byte view rather than a typed record). |
| Alex/Dialects/Core/Serialize.fs | 69-75 | Decision | PSG incomplete | Bit width of a union tag discriminant on fabric (TTag is only produced for fabric representations). |
| Alex/Dialects/Core/Serialize.fs | 271-277 | Construction | Duplicates Baker | Renames SSA values in a serialized hw.module body by text replacement of %argN with port names. Port identity must arrive as published names, not a post-emission text rewrite (auditor-verified). |
| Alex/Dialects/Core/Serialize.fs | 483-503 | Decision | PSG incomplete | Rewrites a constructed ReinterpretCast into arith.constant plus memref.view when element types differ or an i8 view has a nonzero offset, and mints a new name by appending _off. The serializer selects and rewrites the operation (auditor-verified). |
| Alex/Dialects/Core/Serialize.fs | 518-527 | Construction | Backend-owned | Target form of a volatile device access (LLVM pointer materialization and volatile load/store) and the integer carrier used for the address. |
| Alex/Dialects/Core/Serialize.fs | 576-591 | Construction | Delete (dead) | Storage bytes and terminator layout of an immutable string literal. |
| Alex/Dialects/Core/Serialize.fs | 600-610 | Decision | PSG incomplete | Initial image / storage class (.bss-style uninitialized vs zero-initialized) of program-lifetime writable storage, resting on an unpublished before-read premise. |
| Alex/Dialects/Core/Serialize.fs | 674-677 | Name minting | PSG incomplete | Spells the scf.for induction variable with the step value name, because SCFOp.For carries no induction-variable value. The loop index has no derived name (auditor-verified). |
| Alex/Dialects/Core/Types.fs | 22, 112-133 | Decision | Duplicates Baker | Byte size of scalars, memref descriptors and index values used for allocation and layout. |
| Alex/Dialects/Core/Types.fs | 56-67 | Decision | Duplicates Baker | MLIR carrier width for Bool, Char and Unit slots. |
| Alex/Dialects/Core/Types.fs | 97-102, 113-116 | Decision | Duplicates Baker | Pointer size in bytes. |
| Alex/Dialects/Core/Types.fs | 476-478 | Decision | Duplicates Baker | Physical width of a Boolean foreign-boundary parameter/result. |
| Alex/Dialects/Core/Types.fs | 486-494 | Decision | PSG incomplete | Calling-convention parameter order and buffer carrier type of the intrinsic write declaration. |
| Alex/Elements/FuncElements.fs | 63-81 | Construction | Duplicates Baker | Nothing succeeds. The API keeps Pattern-side construction of runtime helper declarations and calls (memcpy) in place as dead, failing paths. |
| Alex/Elements/MLIRAtomics.fs | 19-25 | Decision | Duplicates Baker | Field layout and access descriptor: field index as element index, with memref type TMemRefStatic(1, fieldTy). |
| Alex/Elements/MLIRAtomics.fs | 44-93 | Decision | Duplicates Baker | The access alignment of record and DU field loads and stores. |
| Alex/Elements/MLIRAtomics.fs | 98-103 | Decision | Drops a published fact | Storage and residence for a StackScoped aggregate, and its alignment (none). |
| Alex/Elements/MemRefElements.fs | 6-9 | Decision | Duplicates Baker | The source descriptor type and the loaded element type of memref.load. |
| Alex/Elements/MemRefElements.fs | 145-151 | Projection post-processing | Delete (dead) | The result memref type of a subview. |
| Alex/Patterns/ApplicationPatterns.fs | 118-138 | Decision | Backend-owned | The target realization of a direct call (spatial instance versus temporal call). |
| Alex/Patterns/ApplicationPatterns.fs | 131-133 | Name minting | PSG incomplete | The instance symbol name and the output port name and arity. |
| Alex/Patterns/ApplicationPatterns.fs | 147-163, 256-271, 283, 289-338 | Decision | Backend-owned | The target dialect or Element family for each published numeric operation, and the FPGA admission of real operations. |
| Alex/Patterns/ApplicationPatterns.fs | 246-254, 301-321 | Decision | PSG incomplete | Ordered versus unordered float comparison semantics, and float arithmetic Element selection. |
| Alex/Patterns/ApplicationPatterns.fs | 345-366 | Decision | PSG incomplete | The signedness of real-to-integer truncation, with no published definedness or range premise. |
| Alex/Patterns/ApplicationPatterns.fs | 392-412 | Decision | PSG incomplete | The conversion operation, its signedness (always signed fptosi/sitofp/index_cast), and FFI memref-to-index marshalling for Convert intrinsics. |
| Alex/Patterns/BorrowedViewPatterns.fs | 15, 24, 26, 31, 43, 63, 72, 82 | CCS call | PSG incomplete | Element width, schema range, alignment and access permission of the view |
| Alex/Patterns/BorrowedViewPatterns.fs | 24-28, 46-52, 61-62 | Construction | PSG incomplete | Guard algorithm and requirement realization for view access |
| Alex/Patterns/BorrowedViewPatterns.fs | 25-28 | CCS call | PSG incomplete | Pointer bits, maximum representable extent and aligned load/store alignment |
| Alex/Patterns/BorrowedViewPatterns.fs | 38, 41, 46-60 | Decision | PSG incomplete | Signedness of index/extent transport |
| Alex/Patterns/BorrowedViewPatterns.fs | 43-44 | Decision | Duplicates Baker | Whether the operation is permitted |
| Alex/Patterns/BorrowedViewPatterns.fs | 68-104 | Construction | PSG incomplete | Check width, signedness, range-guard algorithm and narrowing adaptation |
| Alex/Patterns/BorrowedViewPatterns.fs | 106-108 | Decision | Duplicates Baker | Unit carrier |
| Alex/Patterns/CallablePatterns.fs | 31-48 | Structure discovery | PSG incomplete | Whether an annotation occurrence is an applied intrinsic's callee position (witnessed as no value) or a first-class operand. |
| Alex/Patterns/CallablePatterns.fs | 72, 91-94, 147-151 | Projection post-processing | PSG incomplete | The physical function type, environment operand type and copy admissibility of a callable occurrence. |
| Alex/Patterns/CallablePatterns.fs | 127 | Name minting | PSG incomplete | The module-scope storage symbol of a program-lifetime callable environment. |
| Alex/Patterns/CallablePatterns.fs | 135, 156-159 | Name minting | PSG incomplete | The emitted declaration/ABI symbol spelling of callable code. |
| Alex/Patterns/ClosurePatterns.fs | 34-48, 67-86 | Decision | Backend-owned | Target form of a function definition (hw.module vs func.func) and whether a unit function has an output. |
| Alex/Patterns/ClosurePatterns.fs | 43, 78-81 | Name minting | PSG incomplete | Declared output port identity of an FPGA module. |
| Alex/Patterns/ClosurePatterns.fs | 92-99 | Fallback | PSG incomplete | The runtime value returned by a unit-typed function (zero at retTy). |
| Alex/Patterns/CollectionPatterns.fs | 27-89 | Decision | Duplicates Baker | Tag width, case numbering, the tag-compare algorithm and the payload field placement for Option. |
| Alex/Patterns/CollectionPatterns.fs | 29-35 | Decision | PSG incomplete | The Option case numbering used in construction. |
| Alex/Patterns/CollectionPatterns.fs | 95-151 | Decision | PSG incomplete | List layout (tag width, case numbers, field positions) and the isEmpty algorithm. |
| Alex/Patterns/CollectionPatterns.fs | 157-305 | Decision | PSG incomplete | The physical layout and case numbering of the Map and Set tree representation. |
| Alex/Patterns/CollectionPatterns.fs | 311-319 | Decision | Duplicates Baker | The Result case numbering. |
| Alex/Patterns/ContinuationPatterns.fs | 26-32, 416-421 | Decision | Duplicates Baker | The scalar carrier width of continuation slots and of the sequence current value. |
| Alex/Patterns/ContinuationPatterns.fs | 45-61 | Decision | Duplicates Baker | The physical descriptor carrier (static frame-sized byte memref or the occurrence representation) that a ValueView continuation slot holds. |
| Alex/Patterns/ContinuationPatterns.fs | 139-172 | Projection post-processing | PSG incomplete | Whether an environment or sequence slot write may store a descriptor-only value and which SSA supplies it. |
| Alex/Patterns/ContinuationPatterns.fs | 153-172 | Checker state | Duplicates Baker | Whether a slot write takes the sequence environment/descriptor-only path or the ordinary recall path. |
| Alex/Patterns/ContinuationPatterns.fs | 186-193, 329-337 | Decision | PSG incomplete | A representation adaptation (static-extent to dynamic descriptor) at slot writes and capture initialization. |
| Alex/Patterns/ContinuationPatterns.fs | 208-215, 368-382 | Construction | Duplicates Baker | The enumerator acquisition algorithm (fresh frame, per-slot capture marshalling) and the MoveNext calling form. |
| Alex/Patterns/ContinuationPatterns.fs | 218-224, 253-262, 462-465 | Structure discovery | Duplicates Baker | Whether the settled frame and region placement is admissible, which re-proves a Baker layout. |
| Alex/Patterns/ContinuationPatterns.fs | 237-244 | Structure discovery | Duplicates Baker | That an unwitnessed operand is a function formal, and what carrier it has. |
| Alex/Patterns/ContinuationPatterns.fs | 272-276 | Decision | Delete (dead) | The storage class and allocation kind of a continuation frame, including heap allocation for escaping frames. |
| Alex/Patterns/ContinuationPatterns.fs | 287-298, 360, 372, 486-497 | Construction | PSG incomplete | The initial and fresh-entry discriminant value of every sequence frame, and the store that establishes it. |
| Alex/Patterns/ContinuationPatterns.fs | 325-329 | Fallback | PSG incomplete | Which of two candidate carriers counts as the actual initializer representation. |
| Alex/Patterns/ControlFlowPatterns.fs | 121-131, 143-205, 256-270 | Decision | Backend-owned | The target realization of conditionals and matches (mux versus structured regions), and FPGA void-conditional admission. |
| Alex/Patterns/ControlFlowPatterns.fs | 143-205 | Decision | Backend-owned | target form |
| Alex/Patterns/ControlFlowPatterns.fs | 150-172 | Decision | PSG incomplete | Evaluation strategy of if/then/else on fabric (speculative both-arm evaluation). |
| Alex/Patterns/ControlFlowPatterns.fs | 243-247, 256-478 | Construction | PSG incomplete | How a union CaseElimination is realized: the tag read, the per-arm tag comparisons, their order, and the nested control-flow or mux structure. |
| Alex/Patterns/ControlFlowPatterns.fs | 246-247, 298-344 | Construction | PSG incomplete | The construction of the match decision, the default-arm fallback and SSA name positions. |
| Alex/Patterns/ControlFlowPatterns.fs | 298-306, 314-315, 334-336, 368, 375-389, 418-423, 442-444 | Name minting | PSG incomplete | Which SSA name carries each tag literal, comparison, cast, zero index and intermediate if/mux result. |
| Alex/Patterns/ControlFlowPatterns.fs | 365-391 | Decision | PSG incomplete | tag width, tag offset, extraction form |
| Alex/Patterns/ControlFlowPatterns.fs | 368-391 | Decision | PSG incomplete | The union tag's width (i8), its byte offset (0), its storage view, and how it is extracted. |
| Alex/Patterns/DUPatterns.fs | 53-58, 139 | Fallback | Duplicates Baker | Whether and how the payload/read value is extended or truncated. |
| Alex/Patterns/DUPatterns.fs | 62,98,119 | Decision | PSG incomplete | DU tag width and DU storage allocation kind/residence. |
| Alex/Patterns/DUPatterns.fs | 71-99, 106-120, 128-133 | Target form | Backend-owned | Target form of DU construction, tag extraction and payload extraction (including zero-initialized aggregate for payload-less struct cases). |
| Alex/Patterns/DUPatterns.fs | 73-95, 108-116, 130-133 | Decision | Duplicates Baker | Union value representation on fabric. |
| Alex/Patterns/EagerPatterns.fs | 10, 45-46, 52 | Checker state | Duplicates Baker | The evaluation path (callable forward versus scalar) for an explicit demand. |
| Alex/Patterns/EagerPatterns.fs | 19, 27-30 | CCS call | Duplicates Baker | Which source node is the operand of the current explicit demand. |
| Alex/Patterns/EagerPatterns.fs | 28-38 | Structure discovery | PSG incomplete | Whether the current explicit demand has a unique admitted expression frontier. |
| Alex/Patterns/EagerPatterns.fs | 52 | Checker state | Duplicates Baker | The unit-result path for an explicit demand. |
| Alex/Patterns/EnvironmentPatterns.fs | 21-24 | Duplicated validation | Duplicates Baker | Whether the environment layout is proof-backed (re-verification of Baker's proof). |
| Alex/Patterns/EnvironmentPatterns.fs | 32, 54, 73 | Admission bypass | PSG incomplete | No independent decision. Spells Baker's residence and destination facts. |
| Alex/Patterns/EnvironmentPatterns.fs | 41-44 | Checker state | Duplicates Baker | Whether the environment source is treated as an unwitnessed callable or a data value. |
| Alex/Patterns/EnvironmentPatterns.fs | 61-63 | Name minting | PSG incomplete | Module-scope storage symbol for a program-lifetime environment. |
| Alex/Patterns/LazyPatterns.fs | 85-88 | Duplicated validation | Duplicates Baker | Which lazy slots are initialized at formation. |
| Alex/Patterns/LazyPatterns.fs | 189-196 | Decision | PSG incomplete | The control form (index switch) and the selector cast signedness for lazy-valued if/then/else. |
| Alex/Patterns/LiteralPatterns.fs | 66-76 | Fallback | PSG incomplete | Materializes a unit value (0) where the witnessed operation produced none. |
| Alex/Patterns/LiteralPatterns.fs | 83-95 | Construction | Delete (dead) | string encoding, byte length, symbol naming and C-style sentinel storage |
| Alex/Patterns/LiteralPatterns.fs | 85-95 | Name minting | Delete (dead) | String literal global symbol. |
| Alex/Patterns/LiteralPatterns.fs | 93-95 | Decision | Duplicates Baker | String encoding extent. |
| Alex/Patterns/LiteralPatterns.fs | 102-104, 110 | Decision | Duplicates Baker | Representation of a string literal value. |
| Alex/Patterns/MemRefPatterns.fs | 34-57 | Decision | PSG incomplete | Residence and allocation kind of mutable cells |
| Alex/Patterns/MemRefPatterns.fs | 118-119 | Decision | Duplicates Baker | Physical storage shape of module-level value slots |
| Alex/Patterns/MemoryPatterns.fs | 49-71, 247-250 | Decision | PSG incomplete | DU tag width and pointer-vs-inline union access form |
| Alex/Patterns/MemoryPatterns.fs | 52, 248 | Decision | PSG incomplete | Width and offset of the union tag |
| Alex/Patterns/MemoryPatterns.fs | 54-62 | Decision | PSG incomplete | Union layout (indirect versus inline) and how the tag is read |
| Alex/Patterns/MemoryPatterns.fs | 117-131, 317-331 | Structure discovery | PSG incomplete | Arena extent, stack residence and carrier (which disagrees with the published nint representation) |
| Alex/Patterns/MemoryPatterns.fs | 152-193, 222-228 | Decision | Duplicates Baker | Program-storage residence and storage shape of a static aggregate |
| Alex/Patterns/MemoryPatterns.fs | 159, 225, 515, 533 | Name minting | PSG incomplete | Target symbol identity of program storage |
| Alex/Patterns/MemoryPatterns.fs | 163-176 | Projection post-processing | PSG incomplete | Physical storage type of a writable program entry |
| Alex/Patterns/MemoryPatterns.fs | 211-232 | Decision | PSG incomplete | allocation kind/residence (stack, static, heap) for constructed values |
| Alex/Patterns/MemoryPatterns.fs | 219-231 | Decision | PSG incomplete | Heap residence and allocation kind for escaping aggregates (DU, record, environment, continuation and lazy callers via EnvironmentPatterns, ContinuationPatterns, LazyPatterns and NanopassArchitecture) |
| Alex/Patterns/MemoryPatterns.fs | 242-266 | Name minting | PSG incomplete | SSA names and placement of each payload field |
| Alex/Patterns/MemoryPatterns.fs | 285-298 | Construction | Delete (dead) | Bulk copy algorithm and calling convention |
| Alex/Patterns/MemoryPatterns.fs | 286-298 | Construction | Delete (dead) | bulk copy algorithm (runtime memcpy) |
| Alex/Patterns/MemoryPatterns.fs | 351-358 | Decision | PSG incomplete | index-cast signedness for BorrowedView get/set |
| Alex/Patterns/MemoryPatterns.fs | 354-358 | Decision | PSG incomplete | Signedness of index transport |
| Alex/Patterns/MemoryPatterns.fs | 448-451 | Decision | PSG incomplete | signedness and narrowing of array extent to its result carrier |
| Alex/Patterns/MmioPatterns.fs | 9-12, 35-37, 46-51 | CCS call | Duplicates Baker | Address and transaction width of the access |
| Alex/Patterns/MmioPatterns.fs | 10, 12, 35-37 | Checker state | Duplicates Baker | Whether the access width matches the register handle |
| Alex/Patterns/MmioPatterns.fs | 20-21, 34-37 | Structure discovery | Duplicates Baker | Operation kind (register handle, load or store) |
| Alex/Patterns/MmioPatterns.fs | 27 | Duplicated validation | Duplicates Baker | Admissible transaction widths for the target |
| Alex/Patterns/MmioPatterns.fs | 46-52 | Decision | PSG incomplete | Adaptation kind and signedness of the stored value |
| Alex/Patterns/MmioPatterns.fs | 53-55 | Decision | Duplicates Baker | Unit carrier |
| Alex/Patterns/MutableCallablePatterns.fs | 43-48 | Projection post-processing | PSG incomplete | The physical function and environment carrier of each write value and read occurrence. |
| Alex/Patterns/MutableCallablePatterns.fs | 63-65, 74, 88-92, 138 | Decision | PSG incomplete | Discriminator width/type (index), environment cell layout and allocation kind/residence of the mutable callable cell. |
| Alex/Patterns/MutableCallablePatterns.fs | 151-177 | Construction | PSG incomplete | The dispatch algorithm and default arm for reading a mutable callable. |
| Alex/Patterns/MutableCallablePatterns.fs | 161 | Name minting | PSG incomplete | Declaration symbol spelling for dispatch alternatives. |
| Alex/Patterns/PlatformPatterns.fs | 55 | Projection post-processing | PSG incomplete | Which intrinsic write declarations are emitted at this scope, and their order. |
| Alex/Patterns/RecordPatterns.fs | 33-35, 105-112, 216-222, 254-256, 293-295 | Structure discovery | PSG incomplete | Which field (index, type and therefore offset) an occurrence addresses |
| Alex/Patterns/RecordPatterns.fs | 56-84, 152-165, 290-300 | Decision | Backend-owned | Target form of record construction, update and access |
| Alex/Patterns/RecordPatterns.fs | 101-103, 212-214 | Name minting | PSG incomplete | SSA naming of per-field view temporaries |
| Alex/Patterns/RecordPatterns.fs | 104, 215 | Fallback | Duplicates Baker | Field-value adaptation, and whether any adaptation happens at all |
| Alex/Patterns/RecordPatterns.fs | 154-232 | Construction | PSG incomplete | Copy algorithm, runtime-library call, calling convention and pointer-cast signedness |
| Alex/Patterns/RecordPatterns.fs | 244-271 | Fallback | PSG incomplete | The store's carrier. A value/field width mismatch is taken as equality |
| Alex/Patterns/SequencePatterns.fs | 19-22 | Fallback | Duplicates Baker | Sequence and non-sequence classification of an occurrence when the publication is missing. |
| Alex/Patterns/SequencePatterns.fs | 117-125 | Decision | PSG incomplete | The control form and selector cast signedness for sequence-valued if/then/else. |
| Alex/Traversal/CallableOperands.fs | 29-61, 83-108, 111-133, 143-156 | Projection post-processing | PSG incomplete | The physical calling convention (TFunc argument/result list, code+environment expansion of callable/lazy/sequence components, the common convention of a join or flow) of every callable occurrence and direct call. |
| Alex/Traversal/CallableOperands.fs | 64-80, 147-155 | Fallback | Duplicates Baker | Which formals are omitted (absent demand row defaults to 'none omitted') and whether the carrier matches its declaration. |
| Alex/Traversal/CallableOperands.fs | 92-105 | Projection post-processing | Duplicates Baker | Whether the callable's first formal is its environment and what its physical type is. |
| Alex/Traversal/CallableOperands.fs | 96-104, 218 | Admission bypass | PSG incomplete | No decision itself. Feeds the environment and transport re-validation above. |
| Alex/Traversal/CallableOperands.fs | 119-129 | Decision | PSG incomplete | That unit-typed formals and results have zero physical components, which contradicts Baker's Arguments convention where every non-omitted Data formal occupies one component. |
| Alex/Traversal/CallableOperands.fs | 208-240 | Projection post-processing | Duplicates Baker | Whether one callable occurrence may carry another occurrence's code/environment operands. |
| Alex/Traversal/CoverageValidation.fs | 19-30, 80-83 | Structure discovery | PSG incomplete | Which PSG occurrences must be witnessed, which is the executable coverage obligation. |
| Alex/Traversal/LazyOperands.fs | 36-41 | Projection post-processing | PSG incomplete | The lazy thunk calling convention and environment representation (a second derivation beside the thunk's own callable carrier used by CallableOperands.thunkDeclaration). |
| Alex/Traversal/MLIRTransfer.fs | 58 | CCS call | Duplicates Baker | The pointer-integer width used when spelling MMIO address conversion. |
| Alex/Traversal/NanopassArchitecture.fs | 137-138 | Decision | PSG incomplete | Evaluation strategy and placement of shared DAG occurrences: whether a shared value is recomputed inside each function body or reused, and whether the target platform changes that. |
| Alex/Traversal/NanopassArchitecture.fs | 160-162 | Projection post-processing | PSG incomplete | Whether structural children of an occurrence are traversed. |
| Alex/Traversal/NanopassArchitecture.fs | 164-172 | Projection post-processing | Duplicates Baker | Which call actuals are not evaluated at this call site (deferred evaluation) and which child positions they occupy. |
| Alex/Traversal/NanopassArchitecture.fs | 314-328 | Structure discovery | Delete (dead) | Nothing semantic. Diagnostic text only. It still re-walks the alias chain Baker already resolves. |
| Alex/Traversal/NanopassArchitecture.fs | 356-368, 371-374, 402-414 | Decision | PSG incomplete | Which nodes are traversal roots and their visit order. Because globalVisited dedupes, the order also decides which structural occurrence first witnesses a shared node, and so the scope it lands in. |
| Alex/Traversal/SMTTransfer.fs | 28-29, 395-431, 432-475 | Construction | PSG incomplete | The per-axis dimensional equations that make up an application-dimension or dimensional-relation claim. |
| Alex/Traversal/SMTTransfer.fs | 34-804 | Construction | PSG incomplete | The logical content of every proof obligation at build time: its clause set, premise and conclusion structure, and quantified variables. |
| Alex/Traversal/SMTTransfer.fs | 35-36 | Name minting | PSG incomplete | SSA identity of every value in the SMT verification module. |
| Alex/Traversal/SMTTransfer.fs | 52-76, 391-394, 799-803 | Decision | PSG incomplete | The solver logic and the arithmetic encoding of real-valued obligations, which contradicts the published ObligationInfo.Logic. |
| Alex/Traversal/SMTTransfer.fs | 510-574 | Construction | PSG incomplete | The exact conjuncts of the static string pool layout claim at build time. |
| Alex/Traversal/SMTTransfer.fs | 710-751 | Construction | PSG incomplete | The premise that the concat allocation size equals the operand lengths. |
| Alex/Traversal/SequenceOperands.fs | 24-26 | Decision | PSG incomplete | The sequence invocation convention: environment byte-buffer representation and a one-bit has-next result. |
| Alex/Traversal/SequenceOperands.fs | 55-65 | Projection post-processing | PSG incomplete | Whether one sequence occurrence may carry another's code and environment operands. |
| Alex/Traversal/StaticStorageValidation.fs | 5-6, 28, 47, 57-74 | Name minting | Duplicates Baker | Which emitted SSA plays which role in a literal view, by index arithmetic on names. A second private copy of the literal emission convention. |
| Alex/Traversal/StaticStorageValidation.fs | 80,105 | Projection post-processing | PSG incomplete | The physical storage type of each writable program allocation, using target Architecture outside the PSG. |
| Alex/Traversal/TransferTypes.fs | 35-41 | Decision | Duplicates Baker | The integer width of platform-word values (lengths, record/memory word fields) from the target Register dimension instead of the occurrence's published representation; LinkedLibraries is an unread second channel for link requirements. |
| Alex/Traversal/TransferTypes.fs | 364-391 | Fallback | PSG incomplete | Which of two conflicting representations (Element-chosen static residence extent/byte buffer vs published logical representation) an SSA value has. The Element's allocation shape wins without diagnostic. |
| Alex/Traversal/TransferTypes.fs | 418-424 | Decision | PSG incomplete | Storage layout and discriminator width (platform index, not a range-selected width over the finite alternative count) of a mutable callable cell. |
| Alex/Traversal/TransferTypes.fs | 720-724 | Name minting | PSG incomplete | The program-storage symbol identity of a module-level value slot. |
| Alex/Traversal/Values.fs | 16-43 | Name minting | PSG incomplete | How many values a node may emit, which names they get, and which one counts as the node's result. The bands can collide: meetValue i>=100 hits 1100/1200, prologueValue k>=1000 hits hardware 3000, continuation lane>=16 hits the next slot, and solverValue collides with NodeId 0's family. NodeId 0 is used in clef (Boundaries.fs:167, PlatformResolution.fs:766). SMTTransfer.fs:36 increments a counter n into solverValue. `undefined` (line 43) has no callers. |
| Alex/Traversal/WitnessRegistry.fs | 71-79, 99-125, 165-177 | Decision | PSG incomplete | Language-construct admission per substrate (mutable assignment, memory, strings, platform calls, MMIO, collections, lazy/seq) and the target witness set, without reading graph.Platform. |
| Alex/Traversal/WitnessRegistry.fs | 78-79, 99-125, 165-177 | Decision | PSG incomplete | Which language constructs are admitted on each target. |
| Alex/Traversal/WitnessRegistry.fs | 113-114 | Fallback | PSG incomplete | Which lowering path owns an occurrence, chosen by first match and by an emitted MLIR type instead of the published operation. |
| Alex/Traversal/XDCTransfer.fs | 10-82 | Target form | Backend-owned | That the target-specific constraint artifact (XDC) is produced by the portable middle end. |
| Alex/Traversal/XDCTransfer.fs | 19-26 | Fallback | Duplicates Baker | Whether the published pin directions are settled. |
| Alex/Traversal/XDCTransfer.fs | 39-41 | Projection post-processing | PSG incomplete | The clock period constraint handed to synthesis, including how it is rounded. |
| Alex/Witnesses/ApplicationWitness.fs | 99-103 | Decision | Duplicates Baker | Result width of a direct call. |
| Alex/Witnesses/ApplicationWitness.fs | 107-110 | Decision | PSG incomplete | Result carrier of an indirect call. |
| Alex/Witnesses/ApplicationWitness.fs | 123-133 | Projection post-processing | PSG incomplete | Physical parameter component expansion (count and naming) of the call. |
| Alex/Witnesses/ApplicationWitness.fs | 151-156 | Projection post-processing | Duplicates Baker | Saturated-call target and actuals. Partial applications emit nothing. |
| Alex/Witnesses/ApplicationWitness.fs | 157-164 | Decision | Duplicates Baker | Direct versus indirect calling form. |
| Alex/Witnesses/ArithIntrinsicWitness.fs | 19-39 | Structure discovery | Duplicates Baker | Numeric operation kind (add/sub/cmp/unary) and which parser owns the site. |
| Alex/Witnesses/ArithIntrinsicWitness.fs | 377-414 | Decision | PSG incomplete | That a conversion/truncation occurs and which conversion parser composes it (source/target carriers and sign then resolved in the pattern). |
| Alex/Witnesses/BindingWitness.fs | 43-46 | Decision | PSG incomplete | Discard semantics (no SSA binding). |
| Alex/Witnesses/BindingWitness.fs | 71-76 | Fallback | Duplicates Baker | Whether a function binding is definition-only or a value. |
| Alex/Witnesses/BindingWitness.fs | 78, 96 | Admission bypass | PSG incomplete | Partial-application bindings emit nothing. |
| Alex/Witnesses/BindingWitness.fs | 103-105 | Structure discovery | PSG incomplete | The initializer value occurrence of a program-lifetime slot. |
| Alex/Witnesses/BindingWitness.fs | 111 | Name minting | PSG incomplete | Declaration symbol of the slot being initialized. |
| Alex/Witnesses/BindingWitness.fs | 151-158 | Fallback | Delete (dead) | Whether an unwitnessed binding value is an error or a definition-only entry binding. |
| Alex/Witnesses/BorrowedViewWitness.fs | 9-11 | CCS call | PSG incomplete | Whether the node is a BorrowedView operation, and implicitly its op, view operand and schema layout (element bits, range, alignment, access). |
| Alex/Witnesses/ControlFlowWitness.fs | 163, 178-179 | Structure discovery | PSG incomplete | Condition value occurrence and branch yield occurrences of a conditional. |
| Alex/Witnesses/DUWitness.fs | 73-74 | Decision | PSG incomplete | Union tag encoding (tag value per case). |
| Alex/Witnesses/DUWitness.fs | 82 | Structure discovery | Duplicates Baker | Which meet (width adaptation) applies to the DU payload. |
| Alex/Witnesses/EnvironmentWitness.fs | 27-29, 78, 109, 113, 117-120, 124-125 | Admission bypass | PSG incomplete | Nothing beyond validation. Environment layout and callable-environment relation are taken from unpublished codata. |
| Alex/Witnesses/EnvironmentWitness.fs | 77-98 | Projection post-processing | PSG incomplete | Which generator implementation symbol forms the code half of the sequence value read from an environment slot. |
| Alex/Witnesses/FunctionPointerWitness.fs | 21-22, 34, 45-50 | Admission bypass | PSG incomplete | Whether the node is a native callback address or invocation, plus the callback symbol and lambda. |
| Alex/Witnesses/FunctionPointerWitness.fs | 45 | Structure discovery | Duplicates Baker | Argument width adaptation at the native callback boundary. |
| Alex/Witnesses/IntrinsicWitness.fs | 21-26 | Fallback | PSG incomplete | That an intrinsic occurrence carries no runtime value. |
| Alex/Witnesses/LambdaWitness.fs | 51-62, 211-216 | Structure discovery | PSG incomplete | Function operand/operation scope membership: which shared occurrences are re-witnessed per function versus reused from an outer scope. |
| Alex/Witnesses/LambdaWitness.fs | 68-91 | Projection post-processing | PSG incomplete | The function's physical parameter signature (number, order and MLIR type of every component per formal). |
| Alex/Witnesses/LambdaWitness.fs | 112, 332 | Decision | PSG incomplete | Declaration ABI visibility (calling convention/linkage settlement). |
| Alex/Witnesses/LambdaWitness.fs | 152-155 | Name minting | Duplicates Baker | Which Arg n each formal component occupies. |
| Alex/Witnesses/LambdaWitness.fs | 212-214 | Decision | PSG incomplete | Target-specific evaluation strategy / value residence across function boundaries. |
| Alex/Witnesses/LambdaWitness.fs | 237 | Structure discovery | PSG incomplete | The function's result occurrence (which value is returned). |
| Alex/Witnesses/LambdaWitness.fs | 239-271, 283-293, 333-341 | Decision | Duplicates Baker | Return convention and result component types of the function. |
| Alex/Witnesses/LambdaWitness.fs | 283-297 | Decision | Duplicates Baker | Return width/representation of the function (and assumes every return meet is integer, ignoring ExtendFloat/TruncateFloat meets). |
| Alex/Witnesses/LambdaWitness.fs | 306-307, 343-344 | Decision | PSG incomplete | Calling convention for unit-returning functions (return a zero value versus void). |
| Alex/Witnesses/LambdaWitness.fs | 327-331 | Name minting | PSG incomplete | Hardware port names for multi-component formals. |
| Alex/Witnesses/LambdaWitness.fs | 351-371 | Construction | PSG incomplete | Native callback adapter construction and its ABI (void return, private linkage). |
| Alex/Witnesses/LazyWitness.fs | 31-37 | Projection post-processing | PSG incomplete | Traversal coverage: that the Computed/Cached field declarations are definition-only and need no executable witness. |
| Alex/Witnesses/LiteralWitness.fs | 31-72 | Admission bypass | PSG incomplete | (no selection. Reads settled pool offsets, lengths and anchors) |
| Alex/Witnesses/LiteralWitness.fs | 47-72 | Structure discovery | PSG incomplete | Which obligation anchors the static byte pool global carries. |
| Alex/Witnesses/MapWitness.fs | 41-125 | Decision | PSG incomplete | Physical Map node layout (case tag, field order, height field omission) and operand arity. |
| Alex/Witnesses/MappedViewWitness.fs | 4, 12 | CCS call | PSG incomplete | Whether an application is a mapped-binding call and which mapping (acquire/release/layout) it belongs to, which selects the MappedView pattern path. |
| Alex/Witnesses/MatchWitness.fs | 51-68 | Structure discovery | Duplicates Baker | Whether a terminal refutable arm sits under its requirement frontier |
| Alex/Witnesses/MatchWitness.fs | 58-66 | Structure discovery | Duplicates Baker | Whether a single refutable arm is admitted (frontier placement re-validation). |
| Alex/Witnesses/MatchWitness.fs | 104 | Structure discovery | Duplicates Baker | Which occurrence supplies each arm's result value for the match join. |
| Alex/Witnesses/MemoryIntrinsicWitness.fs | 18-27 | Structure discovery | Duplicates Baker | Ownership of Array intrinsic sites (which witness emits the published memory operation). |
| Alex/Witnesses/MemoryIntrinsicWitness.fs | 28-33 | Structure discovery | PSG incomplete | Allocation kind and construction path for arena creation and arena allocation. |
| Alex/Witnesses/MmioWitness.fs | 7-9 | CCS call | Duplicates Baker | Whether the node is an MMIO read/write/register operation and which op it is. |
| Alex/Witnesses/MutableAssignmentWitness.fs | 32-55, 73-77 | Structure discovery | PSG incomplete | Whether the assignment stores into a program-lifetime module slot (memref.global) or into another cell kind. |
| Alex/Witnesses/MutableAssignmentWitness.fs | 39-48 | Projection post-processing | PSG incomplete | Whether this Set is a mutable-callable write and which cell owns it. |
| Alex/Witnesses/MutableAssignmentWitness.fs | 60, 83 | Structure discovery | Duplicates Baker | The width adaptation (extend/truncate meet) applied to the assigned value. |
| Alex/Witnesses/MutableAssignmentWitness.fs | 61 | Name minting | PSG incomplete | The declaration symbol of a program-lifetime storage slot. |
| Alex/Witnesses/MutableAssignmentWitness.fs | 73-80, 91-96 | Structure discovery | PSG incomplete | Storage class of the assignment target (local mutable cell vs error) and the stored element type, both inferred from the MLIR type of an earlier emission. |
| Alex/Witnesses/OptionWitness.fs | 38 | Structure discovery | Duplicates Baker | The payload width adaptation into the option's payload slot. |
| Alex/Witnesses/RecordWitness.fs | 7-8, 85-120 | Decision | PSG incomplete | Access form/representation of a field read or write (struct field vs memory/string/closure/DU path) and field selection by name. |
| Alex/Witnesses/RecordWitness.fs | 70-71 | Decision | Duplicates Baker | Correspondence of tuple element position to physical struct field. |
| Alex/Witnesses/SeqWitness.fs | 36-41, 67, 70, 84, 89-90, 136, 140, 144-156 | Admission bypass | PSG incomplete | Nothing beyond the decisions reported separately. The frame layout, initializers, generator and callable carrier are taken from unpublished codata. |
| Alex/Witnesses/SeqWitness.fs | 66-82 | Projection post-processing | PSG incomplete | Which generator implementation symbol forms the code half of the (code, environment) sequence value at a descriptor-only frame read. |
| Alex/Witnesses/SetWitness.fs | 41-110 | Decision | PSG incomplete | Physical Set node layout (case tag, field order, height field omission) and operand arity. |
| Alex/Witnesses/StringIntrinsicWitness.fs | 13-16 | Structure discovery | Duplicates Baker | Ownership of String intrinsic sites. |
| Alex/Witnesses/StructuralWitness.fs | 99-131 | Structure discovery | Duplicates Baker | Evaluation strategy for tuple projection (elide the tuple and forward the element vs extract a field) and index validity. |
| Alex/Witnesses/StructuralWitness.fs | 103, 121-127 | Decision | PSG incomplete | Correspondence of tuple index to physical struct field (layout/field selection). |
| Alex/Witnesses/StructuralWitness.fs | 140-167 | Structure discovery | Duplicates Baker | Which meet (width adaptation) applies to the forwarded tuple element. |
| Alex/Witnesses/TypeAnnotationWitness.fs | 53-55 | Structure discovery | PSG incomplete | Whether a callable annotation is an applied-intrinsic callee (no value) or a forwarded callable operand |
| Alex/Witnesses/VarRefWitness.fs | 39-53, 93, 103, 156 | Structure discovery | Duplicates Baker | Whether this reference is a callee position (emit nothing) or an assignment destination (no load). |
| Alex/Witnesses/VarRefWitness.fs | 46-53, 61-64, 68, 73-75, 79, 92-97, 109, 116-119, 145, 155, 159, 182 | Structure discovery | PSG incomplete | Read kind of this occurrence: memref cell load versus value forward. |
| Alex/Witnesses/VarRefWitness.fs | 88-91,112-115 | Fallback | Duplicates Baker | Callable value materialization form at this reference. |
| Alex/Witnesses/VarRefWitness.fs | 101, 171 | Admission bypass | PSG incomplete | Partial-application binding references emit nothing. |
| Alex/Witnesses/VarRefWitness.fs | 119-143 | Fallback | PSG incomplete | Which value a pattern-bound name denotes. |
| Alex/Witnesses/VarRefWitness.fs | 163 | Name minting | PSG incomplete | Declaration symbol of a program-lifetime value slot. |
| Alex/Witnesses/VarRefWitness.fs | 182-204 | Decision | PSG incomplete | Mutable-cell load element carrier |
| Alex/XParsec/PSGCombinators.fs | 22 | CCS call | Delete (dead) | Nothing today. It is a vestigial open that gives downstream Patterns access to checker cells. |
| Alex/XParsec/PSGCombinators.fs | 67, 100-107, 181-182, 228, 757, 772, 821 | Decision | Duplicates Baker | The platform word width, offered as a width source independent of any published carrier |
| Alex/XParsec/PSGCombinators.fs | 96-107, 184-191 | Decision | Delete (dead) | width of main return / nativeint |
| Alex/XParsec/PSGCombinators.fs | 109-112 | Admission bypass | PSG incomplete | feeds signedness decisions |
| Alex/XParsec/PSGCombinators.fs | 138-167, 169-174 | Structure discovery | PSG incomplete | which node's adaptation applies. Absence becomes equality |
| Alex/XParsec/PSGCombinators.fs | 140-158 | Structure discovery | PSG incomplete | Which settled meet (extension or truncation) applies to an operand, by rediscovering the key Baker used |
| Alex/XParsec/PSGCombinators.fs | 150-167, 850-858 | Fallback | PSG incomplete | That no extension or truncation is needed when no meet is present |
| Alex/XParsec/PSGCombinators.fs | 184-191 | Decision | Delete (dead) | Entry-point return ABI type and the nativeint width |
| Alex/XParsec/PSGCombinators.fs | 231-234 | Decision | Backend-owned | The target dialect form chosen inside portable Patterns (codata-dependent elision) |
| Alex/XParsec/PSGCombinators.fs | 328-348 | Structure discovery | PSG incomplete | Whether an application is an intrinsic call, and which IntrinsicInfo (module/operation) governs its lowering |
| Alex/XParsec/PSGCombinators.fs | 423-436 | Structure discovery | Duplicates Baker | The emitted symbol name of a lambda |
| Alex/XParsec/PSGCombinators.fs | 623-681 | Structure discovery | Duplicates Baker | Operation kind, comparison predicate family, int-versus-float form and shift signedness of every Operators intrinsic |
| Alex/XParsec/PSGCombinators.fs | 755-760 | Name minting | PSG incomplete | none directly (reads a derived fact) |
| Alex/XParsec/PSGCombinators.fs | 921-935 | Structure discovery | PSG incomplete | the value node of a binding initializer, condition, branch, arm and lambda body |
| Alex/XParsec/PSGCombinators.fs | 922-935 | Structure discovery | PSG incomplete | Which node supplies the value of a block, a branch, a match arm, a binding initializer or a lambda body |
| MLIRGeneration.fs | 19, 33-35, 50, 62, 88, 90, 137, 143-173 | CCS call | Duplicates Baker | The Register and Pointer widths Alex uses for every layout, boundary and serialized module (moduleToString arch.Pointer). |
| MLIRGeneration.fs | 30-35, 50, 61-63, 158-173 | CCS call | PSG incomplete | The pointer/index and register widths used by Alex patterns and by serialization. |
| MLIRGeneration.fs | 33-35, 50, 61-64 | Decision | PSG incomplete | The target form (FPGA, NPU or CPU) and width context available to every Pattern. |
| MLIRGeneration.fs | 57-59, 99-105 | CCS call | PSG incomplete | Which proof obligations exist and are carried into the proof envelope and the SMT artifact. |
| MLIRGeneration.fs | 84-90 | Target form | Backend-owned | The target container form and target constraint-file generation. |
| MLIRGeneration.fs | 97, 104, 119 | CCS call | Duplicates Baker | Calls the CCS infrastructure function PhaseConfig.isVerbose to gate diagnostic printing. A MiddleEnd dependency on CCS with no semantic effect (auditor-verified). |
| MLIRGeneration.fs | 108-124 | Target form | Duplicates Baker | Physical pin constraints and clock period for the FPGA artifact. |
| MLIRGeneration.fs | 127-130 | Decision | Duplicates Baker | Activation kind: TargetModuleActivation for FPGA and NPU, CheckedProgramStartup for everything else. |
| MLIRGeneration.fs | 162-172 | Decision | Duplicates Baker | Whether a target leg needs Register and Pointer widths: FPGA does not, all other legs do. |

### A.2 Composer Core and BackEnd

| File | Lines | Class | Disposition | What it decides |
|---|---|---|---|---|
| Composer/BackEnd/AIE/KernelRealization.fs | 61, 63, 71, 86, 93, 118-119, 127, 159-191 | Name minting | Backend-owned | SSA identities in the AIE text. |
| Composer/BackEnd/AIE/KernelRealization.fs | 139-142, 156 | Target form | Backend-owned | Which declared kernel targets and iteration counts are realizable. |
| Composer/BackEnd/AIE/KernelRealization.fs | 153-191 | Construction | Backend-owned | Element-wise application and transport direction, as target realization of the published partition. Role recovery by position. |
| Composer/BackEnd/CIRCT/HardwareRealization.fs | 16-20 | Name minting | Backend-owned | SSA identities of registers, reset logic, struct packing and extracts. |
| Composer/BackEnd/CIRCT/HardwareRealization.fs | 31-58 | Target form | Backend-owned | The target form of the reset. |
| Composer/BackEnd/CIRCT/HardwareRealization.fs | 40-49 | Target form | PSG incomplete | Reset semantics: a one-cycle POR pulse and the cycle-0 register state, chosen in the backend. |
| Composer/BackEnd/CIRCT/HardwareRealization.fs | 90-92 | Projection post-processing | PSG incomplete | Step result decomposition: next-state versus output component. |
| Composer/BackEnd/CIRCT/HardwareRealization.fs | 121-128 | Seam | Backend-owned | Target form of the Step implementation (hw.module with named ports), decided in common Alex and assumed by the backend. |
| Composer/BackEnd/MCU/Target.fs | 13,42-128 | CCS call | PSG incomplete | The image descriptor, flash/RAM spaces, vector layout and stack reservation. |
| Composer/Composer.fsproj | 229 | Seam | PSG incomplete | Whether Alex can reach analysis, resolution and checker state (it currently can). |
| Composer/Core/CompilationOrchestrator.fs | 89-117 | CCS call | PSG incomplete | The target triple, CPU model, OS and pointer bits given to the backend, and target admission. |
| Composer/Core/CompilationOrchestrator.fs | 259-261, 279 | Fallback | PSG incomplete | The runtime model given to the backend (libc, freestanding, bare, rocm or xdna). |
| Composer/Core/WitnessArtifacts.fs | 78-79, 115 | CCS call | Duplicates Baker | The proof-envelope validity, and whether a backend may run. |
| Composer/Core/WitnessArtifacts.fs | 141-147 | CCS call | Duplicates Baker | The expected startup definition and symbol for catalog validation. |

### A.3 clef publication and Baker

| File | Lines | Class | Disposition | What it decides |
|---|---|---|---|---|
| clef/Baker/Ingredients/SpatialValues.fs | 15-17 | Claim partial | PSG incomplete | The hardware Mealy contract. |
| clef/Baker/Recipes/BoundaryRecipes.fs | 106,260-279 | Claim partial | PSG incomplete | Platform declaration interpretation. |
| clef/Baker/Recipes/HardwareModuleRecipes.fs | 13-247 | Claim partial | PSG incomplete | Mealy contract: state/input/output representations, reset values, register capacity, clock/reset/pin identities, port paths, metadata-only census. The decision is made in Baker, not at publication. Gaps: (a) Step output-component presence is computed (99-105) but not published; (b) the internal POR contract is only asserted active-high (143), and its realization is not published; (c) unpinned output fields are silently dropped (172-176); (d) endpoint shapes are matched by short display name via applySubst (59-67), which KernelDeclarations.fs:10-11 itself rejects; (e) signedness of an undeclared register is chosen from the range sign (218-221), outside NumericSettlement; (f) the Step body is still lowered by common Alex FPGA Patterns (see remaining-3 findings). |
| clef/Baker/Recipes/HardwareModuleRecipes.fs | 172-178 | Fallback | PSG incomplete | Which Step output components are observable: unpinned outputs are dropped. |
| clef/Baker/Recipes/KernelModuleRecipes.fs | 11-203 | Claim partial | PSG incomplete | The kernel scalar construction order, alias adaptations, ingress transport carriers, partition tiles, and output coverage. The decision is made in Baker (the SpatialSettlement and KernelDeclarations nanopasses), not at publication. It is not represented as nodes or edges. Per-element traversal, FIFO and DMA exist only as backend construction. |
| clef/Baker/Recipes/MemoryAccessRecipes.fs | 238-262 | Claim partial | PSG incomplete | which occurrence is the numeric operand of a memory store/initializer adaptation |
| clef/Baker/Recipes/MemoryAccessRecipes.fs | 398-411 | Claim partial | PSG incomplete | address of a buffer (base only, offset dropped) and index-cast signedness in a conversion |
| clef/Baker/Recipes/NumericCarrierRecipes.fs | 59-61, 80-99, 115-131, 145-146, 196-211 | Claim partial | PSG incomplete | scalar and composite width and representation for every value occurrence |
| clef/Baker/Recipes/NumericOperationRecipes.fs | 23-46, 48-141 | Claim partial | PSG incomplete | operation carrier, signedness, adaptations |
| clef/Nanopass/KernelDeclarations.fs | 8-13 | Fallback | PSG incomplete | Whether an ingress failure is reported: its source-contract diagnostic is swallowed. |
| clef/PSGSaturation/SemanticGraph/CallableEmission.fs | 13-59, 85, 103-139, 171-179 | Claim fails | PSG incomplete | Whether an occurrence is callable, lazy, sequence or data. The emission symbol identity. Which source occurrence a wrapper transports |
| clef/PSGSaturation/SemanticGraph/CallableEmission.fs | 73-83 | Publication-time analysis | PSG incomplete | Callable carriers, joins, flows and mutable callable storage |
| clef/PSGSaturation/SemanticGraph/CallableEmission.fs | 85-170 | Publication-time analysis | PSG incomplete | Call targets and parameters, dimensional instances, value-transport paths and value shapes. |
| clef/PSGSaturation/SemanticGraph/CallableEmission.fs | 92-98, 140-170 | Publication-time analysis | PSG incomplete | Which implementation, parameters, arguments and dimensional instance each call site uses |
| clef/PSGSaturation/SemanticGraph/CallableEmission.fs | 183-190, 240-281 | Publication-time analysis | PSG incomplete | Calling convention (physical argument components) and callable retention residence |
| clef/PSGSaturation/SemanticGraph/CallableEmission.fs | 191-199, 282-312 | Publication-time analysis | PSG incomplete | Alias endpoints for SSA naming, which nodes are unit-typed, which occurrences are closed data |
| clef/PSGSaturation/SemanticGraph/CallableEmission.fs | 217-239 | Claim fails | PSG incomplete | Whether a VarRef names a definition-only function, a function binding, or code that needs an environment |
| clef/PSGSaturation/SemanticGraph/CallableEmission.fs | 222-234, 287-305 | Publication-time analysis | PSG incomplete | Whether a binding or lambda is definition-only, and the SSA alias identity of each occurrence. |
| clef/PSGSaturation/SemanticGraph/CallableEmission.fs | 240-280 | Publication-time analysis | PSG incomplete | The calling convention: how many physical arguments each formal gets, and their positions. |
| clef/PSGSaturation/SemanticGraph/CallableEmission.fs | 306-312 | Publication-time analysis | PSG incomplete | Which occurrences carry no value (unit) and which are closed data, which then drives Alex's value/no-value path. |
| clef/PSGSaturation/SemanticGraph/CallableIngress.fs | 38-511 | Publication-time analysis | PSG incomplete | Whether a callable occurrence and a formal's inputs are closed and admitted. Retained continuation source identity |
| clef/PSGSaturation/SemanticGraph/Meets.fs | 37-461 | Publication-time analysis | Duplicates Baker | extension/truncation kind and target width for every value-to-slot meet outside numeric operations |
| clef/PSGSaturation/SemanticGraph/MemoryPublication.fs | 26 | Publication-time analysis | PSG incomplete | which array copy constructions are executable, and so required to have receipts |
| clef/PSGSaturation/SemanticGraph/OrdinaryDemand.fs | 34-187, 202-223, 253-308 | Publication-time analysis | PSG incomplete | Which formals and actuals are omitted, which actuals are eager, and which nodes are deferred-only. The deferred-only set excludes nodes from coverage and from the numeric/memory required sets. |
| clef/PSGSaturation/SemanticGraph/SpatialPublication.fs | 10-30, 38-44, 68-95 | Publication-time analysis | PSG incomplete | Proof-body and adaptation correctness, re-solved at publication (a partial second solver used as validation). |
| clef/PSGSaturation/SemanticGraph/SpatialPublication.fs | 92 | Publication-time analysis | Duplicates Baker | Kernel ingress validity (Compute use census, transport-record correspondence), re-derived by graph scan during publication. |
| clef/PSGSaturation/SemanticGraph/StorageWitness.fs | 90-123, 137-150, 165-173, 201-224 | Publication-time analysis | PSG incomplete | Lazy value occurrences, definition-only thunks, the startup plan, program instances and the literal-pool proof anchor. |
| clef/PSGSaturation/SemanticGraph/StringByteStorage.fs | 8-31 | Publication-time analysis | PSG incomplete | byte element slot and 0..255 range for an array occurrence reached through aliases |
| clef/PSGSaturation/SemanticGraph/WitnessEmission.fs | 10-31, 45-64, 108-120 | Publication-time analysis | PSG incomplete | The contents of the Ordinary, Callable and Storage domains that Alex reads. |
| clef/PSGSaturation/SemanticGraph/WitnessEmission.fs | 66-103 | Seam | PSG incomplete | Whether a graph is admitted to witnessing, and so whether every downstream WitnessEmission.tryX read is current. |
