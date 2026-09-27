# Clef / Composer architecture checkpoint

Updated 2026-09-27 for the owner and the implementer continuing this
rearchitecture. This replaces the accumulated checkpoint history with the
architecture, implementation, evidence and remaining work of the current tree.

**This is a recorded implementation notch. The shared representation-selection
path and substantial source-semantic construction have been removed from Alex.
Baker establishes the decisions in the PSG; Alex witnesses those resolved facts.
F01 has completed the three native proof stages and executed successfully.
Complete compiler acceptance and sample parity remain unfinished.**

The final bounded checks are **Array source 11/12, linked Composer controls
48/54, spatial controls 18/18, and the previously completed SMT transfer
controls 163/163**. The six Composer failures and one source failure remain
visible below. This is a documented implementation notch, not a green
acceptance checkpoint.

## The north star: Baker constructs; Alex witnesses; the backend realizes

The architectural correction is the central accomplishment of this work. The
early form of Alex combined witnessing with decisions and transformations that
belong to the source compiler. Removing that vestigial responsibility is
substantive rearchitecture: the implementation now has explicit, inspectable
source contracts where witness-time selection or repair previously occurred.

CCS/Baker owns source semantics through elaboration and saturation: numeric
selection, operation construction, declaration and ABI settlement, demand,
storage residence, callable correspondence and proof premises. The PSG retains
the actual source occurrences, scope, joint constraints, hyperedges, proof
citizens and intermediate rewrite records that establish those decisions.
Publication makes the settled facts available; it must not become a second
semantic solver.

Alex is a passive witness of those facts through the Huet zipper:

- **Elements** provide typed atomic operations and physical spelling for settled
  facts. They do not select a source width, storage policy or semantic meaning.
- **Patterns** compose Elements using the published operands, adaptations,
  relationships and guards. A Pattern can serve multiple Witnesses. Shared
  composition is the intended reuse mechanism.
- **Witnesses** observe the current occurrence and select its applicable
  published composition. They do not reconstruct an algorithm, inspect mutable
  checker cells, infer a type, rediscover declarations or repair absent facts.

The backend owns target-specific realization. A selected platform can inform
portable witnessing; that does not authorize LLVM, CIRCT or AIE target
commitments in the common middle end. Target realization preserves the source
contract and its proof correspondence.

A missing fact is information about the owning source contract. A green build
does not establish acceptance, and a failed gate never authorizes a fallback.
Clef's native dimensional type universe and lazy-default semantics remain
authoritative. .NET is the compiler/tooling host, not the language model.

The three governing documentation sources agree:

- Composer [C/F ownership](PRDs/C-Series-Acceptance.md),
  [M-01 admission](PRDs/M-01-DialectAdmission.md) and
  [core collections](PRDs/C-04-CoreCollections.md).
- clef-lang-spec [backend boundary](../../clef-lang-spec/spec/backend-lowering-architecture.md),
  [numeric selection](../../clef-lang-spec/spec/numeric-selection.md),
  [native type universe](../../clef-lang-spec/spec/native-type-universe.md) and
  [native mappings](../../clef-lang-spec/spec/native-type-mappings.md).
- clef-lang-site [Baker Saturation Engine](../../clef-lang-site/hugo/content/docs/internals/pipeline/baker-saturation-engine.md),
  [Nanopass Navigation](../../clef-lang-site/hugo/content/docs/internals/concepts/nanopass-navigation.md),
  [Learning to Walk](../../clef-lang-site/hugo/content/docs/internals/pipeline/learning-to-walk.md)
  and the [Weaving the Braid blog](../../clef-lang-site/hugo/content/blog/weaving-the-braid.md).

These are design oracles, not retrospective explanations of whatever the
current code happens to do. Baker Saturation Engine places executable
algorithms and their joint relations in source recipes and preserves identity
through fold-in. Nanopass Navigation establishes the sequence from settled PSG
publication through passive Elements/Patterns/Witnesses to the target backend.
Learning to Walk requires ordering, captures, lifetime and SSA decisions before
zipper traversal. Weaving the Braid requires every joint premise to refer to
the same instantiated facts and distinguishes an obligation from its discharge
and from preservation of that evidence downstream.

Those principles determine both the implementation and the tests: publication
must retract when the actual source premise changes; witnessing must preserve
that same identity; each proof stage must have its own checked evidence.
Read those contracts before targeted implementation work. The old code is not
architectural authority.

## Repository revisions and owner boundaries

| Repository | Base revision before this notch |
|---|---|
| clef | `38d184a7b85c501b4d7e1bd1b7481ea3292c3b48` |
| Composer | `3a5b7af4f80a614a564bdb47d6865520593e0e4a` |
| Fidelity.Platform | `e69948daec67ccc3bcabf3fa9c6616e9b54c875d` |

The corresponding notch revisions are:

| Repository | Notch revision |
|---|---|
| clef | `b77f8885d62db447f3473f92338f112fe7aa5cce` |
| Fidelity.Platform | `d7cc3b77f20f80720cea6e79c2bed1e6ee756be1` |
| Composer | The commit containing this rewritten checkpoint and the witness/backend implementation |

Work was performed in the existing trees. The commits include the required new
source and test files, not only previously tracked modifications.
Fidelity.Platform changes are in `Contracts/PlatformContracts.clef` and
`Environments/Linux/x86_64/Console.clef`. The boundary/backend ownership
correction in the base revisions is the foundation for this source and witness
rewrite.

No Python, side worktrees, deleted implementation recovery or custom MLIR
plugins were used for this slice. No evidence directory was created. The owner
explicitly authorized committing and pushing these changes with the checkpoint
on 2026-09-27. Production scope is frozen at this notch. The bounded final
verification is complete; remaining failures are retained for the next
source-contract corrections. No background work remains running.

## Decisions and transforms removed from Alex

The following inventory names the architectural change, its source authority,
and the passive composition that replaces the vestige. It includes the boundary
ownership foundation already in the base revisions and this notch's
numeric, memory, string, callable and spatial reimplementation.

| Removed decision or transform | Authority established in the PSG | Alex's remaining role |
|---|---|---|
| Shared type/width selection from ranges and native type cells | Baker Numeric settlement publishes immutable scalar and composite representations with exact source premises | TypeMapping spells the held representation; PSGCombinators checks correspondence |
| The proposed local MemoryWitness width correction | One source-settled result carrier, shared with other occurrences and operations | MemoryWitness reads the published operation; no independent width path |
| Numeric operation signedness, intermediate carrier and adaptation choices | NumericOperationRecipes retains ordered operands, operation/result carriers, selected declaration, Meets and proof obligations | Shared numeric composition follows the exact operation and adaptations |
| Memory field-name/type interpretation and collection-specific width guessing | Memory settlement retains element slot, actual descriptor, access guard, residence and extent | Shared memory Pattern composes typed accesses |
| Taking an address by copying a scalar into new storage | Source actual-place identity: mutable cell, array element, record field or existing reference | Witness the real descriptor base, offset, stride, index and element-byte extent |
| Duplicate Array intrinsic loops and SubViewCopy construction | Baker creates allocation, initialization and copy source graphs, with exact construction/copy receipts | Dynamic allocation and access are small published memory operations |
| String equality/substring helper paths, including the old memcmp route | Source equality uses descriptor lengths and guarded readonly byte traversal; a proved empty operand has its own source length construction | Witness the ordinary source operations |
| String search loop, concatenation allocation/copy and byte-based character indexing | These require proper source constructions; Unicode codepoint semantics remain required | The old algorithms are deleted; missing positive capabilities remain explicit |
| fromBytes/toBytes identity shortcuts based on equal carriers | Source snapshot/copy lineage, byte/text evidence, exact source/result carriers and independent storage | StringIntrinsicWitness uses the shared memory projection and admitted view transport |
| Scalar/lazy/sequence dispatch choosing index-cast signedness from ranges | Numeric index transport retains sign, selected Pointer dimension and an actual range-coverage proof | All three use one passive dispatch-selector Pattern |
| Structural/annotation/environment shape discovery | Published value shape, callable symbol and actual child correspondence | Forward the published value or symbol; no recursive last-value or Lambda discovery |
| VarRef classification by looking for child Lambdas/captures | Published DefinitionOnlyBindings/DefinitionOnlyLambdas and complete alias/formal identity paths | Read the current occurrence's declared role |
| Raw constant-match reconstruction in the middle end | Baker normalizes it into typed comparisons, ordinary control flow and requirements | Existing control-flow composition witnesses the source construction |
| Hardware declaration traversal, reset/pin/clock interpretation and module construction in its Pattern | Hardware source recipe publishes the Mealy contract and source proof citizens | Thin Hardware Witness, shared spatial Pattern and Element; CIRCT backend realization |
| Kernel opcode discovery and target text construction | Source publication retains the full ordered scalar computation, declared ingress and partition | Thin Kernel Witness and the same spatial composition; AIE backend realization |
| Platform signature discovery, marshalling construction, system-call selection and inline declaration construction | Source boundary facts retain declaration identity/scope, ordered actual/formal correspondence and ABI adaptations | Published import/call Patterns; declarations appear directly at their settled scope |

This is the architecture to preserve. Calling a CCS function from Alex would
still violate it if that function performs selection or semantic repair.
Likewise, renaming a transform or moving it into a publication reader would
not make it passive.

The reusable lower tiers are part of this correction. Memory and String
witnesses share published memory composition; Hardware and Kernel share spatial
composition; scalar, lazy and sequence dispatch share the published index
selector. The individual Witness does not need its own variant of a decision
already established by Baker. Further rewrites should simplify the Witness
around these contracts and add shared composition only where the resolved
facts really are common.

### Implementation map for the next implementer

These are entry points into the retained architecture, not alternative places
to solve the same semantics.

| Contract | Source construction and settlement | Passive publication/composition |
|---|---|---|
| Value representations and operations | clef `Baker/Ingredients/ValueRepresentations.fs`, `NumericValues.fs`, `NumericOperations.fs`; `Baker/Recipes/NumericCarrierRecipes.fs`, `NumericOperationRecipes.fs`; `Nanopass/NumericSettlement.fs` | clef `PSGSaturation/SemanticGraph/NumericPublication.fs`; Composer `Alex/CodeGeneration/TypeMapping.fs`, `Alex/XParsec/PSGCombinators.fs` |
| Actual storage, extents and access | clef `Baker/Ingredients/MemoryValues.fs`, `ArrayShapes.fs`; `Baker/Recipes/MemoryExtentRecipes.fs`, `MemoryAccessRecipes.fs`; `Nanopass/MemoryAccessElaboration.fs`, `MemorySettlement.fs` | clef `PSGSaturation/SemanticGraph/MemoryPublication.fs`; Composer `Alex/Patterns/MemoryPatterns.fs`, `Alex/Elements/MemRefElements.fs`, `Alex/Witnesses/MemoryWitness.fs` |
| Array algorithms and retained copy authority | clef `Baker/Recipes/ArrayConstructionRecipes.fs`, `ArrayMemoryRecipes.fs`; `Nanopass/ArrayConstruction.fs` | Same Memory publication and composition; `Alex/Dialects/Core/Types.fs` and `Serialize.fs` carry the typed dynamic stack operation |
| String semantics and byte/text lineage | clef `Baker/Ingredients/StringBytes.fs`; `Baker/Recipes/StringComparisonRecipes.fs`, `StringViewRecipes.fs`, `StringBorrowRecipes.fs`; existing callable reification and Array construction | clef Memory/Boundary publication; Composer `Alex/Witnesses/StringIntrinsicWitness.fs` and shared Memory composition; old `Alex/Patterns/StringPatterns.fs` deleted |
| Closed callable inputs and dispatch transport | clef `PSGSaturation/SemanticGraph/CallableIngress.fs`, `CallableEmission.fs`, `Curry.fs`; source callable recipes and NumericIndexTransport | Composer `Alex/Patterns/ControlFlowPatterns.fs` shared by ControlFlow, Lazy and Sequence; exact adaptation checks in PSGCombinators |
| Hardware and kernel declarations | clef `Baker/Ingredients/SpatialValues.fs`; `Baker/Recipes/HardwareModuleRecipes.fs`, `KernelDeclarations.fs`, `KernelModuleRecipes.fs`; `Nanopass/SpatialSettlement.fs` | clef `PSGSaturation/SemanticGraph/SpatialPublication.fs`; Composer `Alex/Patterns/SpatialPatterns.fs`, `Alex/Elements/SpatialElements.fs`, thin Hardware/Kernel Witnesses |

All clef paths in this table are relative to `src/Compiler`; Composer `Alex/`
paths are relative to `src/MiddleEnd`. The source recipes construct executable
meaning and its retained relationships. Settlement publishes a coherent domain.
Publication checks that the retained facts still belong to the current graph.
The final column witnesses that authority. None is permission to create a
parallel range solver or alias analyzer in a downstream reader.

### Shared representation authority

The removed shared selector affected Application, Lambda, FunctionPointer,
Binding, VarRef, MutableAssignment, Match, ControlFlow, Record, DU, Option and
HardwareModule in addition to Memory. Fixing one memory operation's width
would have left the architectural defect in all of these paths. The correction
therefore lives at their shared source/publication/composition boundary.

Source entry points include `NumericCarrierRecipes.fs`,
`NumericOperationRecipes.fs`, `NumericSettlement.fs` and
`NumericPublication.fs`. Alex's `CodeGeneration/TypeMapping.fs` and
`XParsec/PSGCombinators.fs` read and check immutable facts. They no longer
invoke range selection, follow mutable native type cells or complete
placeholder representations.

The source premises include exact integer/real literal bits, record field
names/types/order, constructor placement, native type identities, ranges and
actual-place identity. Changing one retracts the affected publication. Width,
signedness, dimension and layout are not interchangeable pieces of metadata.

### Memory and actual places

`MemoryWitness.fs` now dispatches through
`MemoryPatterns.pPublishedMemoryOperation`. The source contract identifies
the actual operand and its complete access/storage authority. A descriptor's
nonzero offset and stride survive address witnessing; a convenient scalar
copy cannot stand in for the original location.

For the migrated memory families, readonly global inventory, mutable local
residence, source guards and exact element-byte extents are held facts. The
Pattern checks that recalled values agree with them. It does not choose a
replacement storage class when a contract is missing. BorrowedView still has
separate access/guard/width vestiges and is explicitly queued below.

### Array construction belongs to Baker

`ArrayConstructionRecipes.fs` creates explicit count/offset requirements,
fresh storage, default initialization and ordinary guarded loops.
`ArrayShapes.fs` preserves actual descriptor extents from literal cardinality
or the original allocation Count, including their complete dependencies.
Ranges enclose that count; they never replace the runtime descriptor extent.

`ArrayMemoryRecipes.fs` validates the retained construction and publishes
allocation/copy receipts. Allocation has layout, capacity, index and address
coverage proof citizens. Copies retain actual reads, stores, advance/loop
structure, Requirements, participants and their intermediate access rewrites.
The per-read/per-store numeric adaptations remain the existing source authority.

This distinction matters for byte conversions: a byte source and a native
integer destination can have the same logical element meaning and different
physical storage widths. The source read/store Meets govern that conversion.
Requiring equal physical widths would be an incorrect extra restriction;
letting Alex invent a cast would be a second architectural error.

Overlapping `Array.blit` completes a fresh source snapshot before writeback.
Empty copies retain their zero-count authority without phantom read/write
loops. Unforced unused construction must remain deferred and must not demand
writable storage. The executable/deferred inventory is shared with the existing
source demand owner.

The positive unused-let control currently fails: an unforced
`let unused: int array = Array.zeroCreate 3` followed by `0` is still treated
as requiring writable stack storage. This is an open source demand defect,
not an accepted exception to lazy-default semantics. The empty eager
zeroCreate/sub cases and the retained guard/copy retractions pass.

Alex's dynamic allocation Pattern recalls the held Count, uses the published
index conversion, element and alignment, and composes the declared stack
allocation. Initialization and copying remain ordinary source operations.
Escaping/actor allocation and unsupported native default values are capability
gaps, not excuses for heap or CLR/null defaults.

### Strings and independent snapshots

The entire old `Patterns/StringPatterns.fs` was deleted. That removed both
dead alternative implementations and active middle-end algorithms. Search,
concatenation and character access have not been falsely counted as complete:
their proper source constructions are still required. Character indexing must
respect Unicode codepoints, not reinterpret a UTF-8 byte index as a character.

Source equality uses exact descriptor lengths and readonly byte traversal.
A proved empty operand selects the source length-only construction. One
validation failure exposed a literal occurrence shared across disjoint
branches; the source recipe now creates distinct occurrences, preserving
valid SSA scope rather than repairing the resulting operations afterward.

Ordinary, qualified, local-alias, curried and first-class length forms are
required to use source extent authority. First-class extent reification uses
the existing source closure recipe and field operation. The final registry
check found a remaining local-alias carrier mismatch, recorded below; the
other tested callable forms passed.

`StringViewRecipes.fs` requires the admitted snapshot allocation/copy and its
complete byte/text evidence. Local comparison retains that lineage, including
constructed local immutable strings. Static origins are not fabricated for
them. The stricter system-write static-storage contract remains separate.

`String.toBytes` reuses source `Array.sub` rather than acquiring another copy
algorithm. Tests mutate an original byte array after fromBytes and compare
the resulting text's contents. Tests also create two toBytes results, write
512 into one and require the other to remain 65 at that position. These are
independence and native-element-domain oracles, not merely length checks.
The final fromBytes content/mutation control passes. The final toBytes control
currently stops at exact numeric-Meet correspondence before execution. Its
strong content oracle remains in place; independent toBytes execution is not
claimed from source construction alone.

### Callable, demand and dispatch correspondence

Readonly aliases and formal references retain their complete source identity
paths. Mutable/program-slot reads retain occurrence identity across writes.
Curried calls preserve saved/current argument correspondence; already-complete
factory calls retain their distinct source boundary.

Array extents and static string origins now share
`CallableIngress.closedFormalInputs`. ParameterInputs is an observed census,
not proof that every possible caller has been accounted for. The existing
closed implementation proof and exact formal/actual positions must agree.
The retained FormalInputs correspondence includes the formal, every actual
and call, and all participants of the unchanged closure proof.

The focused tests caught omission of the formal identity in that new retained
correspondence. That was corrected at the source helper, not waived in its
users.

Scalar, lazy and sequence dispatch share a NumericIndexTransport. It contains
an actual source coverage obligation for the selected index domain, not only
a width. A 33-bit value is refused for a 32-bit index. The three Patterns use
the same passive selector composition.

The retained index contract names the dispatch site and operand, scalar carrier,
selected Pointer declaration and bits, signedness, capacity, obligation and
complete participants. Its proof is an `IntegerRepresentationCoverage` citizen
in the PSG. Publication checks the actual formula, selected dimension,
source occurrence and rewrite correspondence. The common Pattern
`pPublishedDispatchSelector` follows that contract. Editing sign or width in
matching rows, changing the operand, removing the proof, or altering the source
dispatch cannot preserve admission merely because the resulting MLIR would
still parse.

### Spatial declarations and backend commitments

Hardware and Kernel witnesses now share `SpatialPatterns.fs` and
`SpatialElements.fs`. Baker publishes exact declaration scope and identity,
state/reset/clock/pins, computation and partition. Hardware reset/register
ranges and kernel partitions have proof citizens; partition formulas agree
between source discharge and MLIR SMT.

`BackEnd/CIRCT/HardwareRealization.fs` realizes the published Mealy contract,
including explicit reset behavior and exact Step signature. The old
HardwareModule Pattern's declaration-analysis/module-construction code is gone.

`BackEnd/AIE/KernelRealization.fs` preserves the full ordered scalar computation,
not the first recognizable arithmetic operation. Declared input and result
transports remain distinct. External kernel formals use their explicitly
declared ingress domain; result capacity never supplies a fabricated result
range. Ingress paths, uses, selected declaration files and field ranges are
retained for retraction. The npu2 target now selects aie2p in its backend.

## Proof discharge and native artifact publication

The required order is:

1. Discharge source obligations in the PSG before Alex.
2. Discharge the corresponding MLIR SMT obligations before target realization.
3. Discharge Rocq against the generated artifact .v before publishing the
   executable.

The LLVM backend links to a fresh provisional file, checks its actual ELF and
fresh .v/.vo, and publishes atomically only after success. Failure preserves
the previous executable. Source and MLIR receipts retain executable identity,
hash/version/arguments and ordered source/anchor/solver outcomes, not hashes
alone. Artifact validation checks the current source/catalog/operation/receipt
identities.

Exact source pool symbols remain local via LLVM internal linkage and llvm.used.
ELF checks cover symbol identity, bytes, sentinels, alignment and readonly
placement. A typed memory census refuses unsupported residual operations.

F01 completed all three stages and executed with stdout `Hello, World!\n`,
empty stderr and exit 0. The recorded executable is
`/tmp/composer-artifact-fresh-5af8155dfd6c43b2b422e68102140023/hello`.
Its proof directory is
`hello.proof-e0bb34d83db74eee9cff62778b623a8e` beside it.

Its full 8192-byte .rodata is accounted for as a 4096-byte trusted foreign
prefix and the exact 4096-byte source pool. Twelve source-linked Rocq claims
produce 25 closed theorems; 25 numeric/dimensional anchors remain solver-only.
This does not prove machine-instruction or syscall equivalence. ELF extraction,
LLVM/LLD and runtime remain explicit trusted components.

Arrays, actual-address/writable storage, lazy/sequence frames and spatial
artifacts still need their artifact contracts. The snapshot tests use stock
MLIR execution as a witness-behavior oracle; that does not establish native
artifact/Rocq admission. HelloProof/ship-of-theseus is the reference for the
three-stage design; its retired Alex implementation is not authority.

## Verification at the notch

These are focused scopes, not complete suite or manifest acceptance.
Temporary logs are in /tmp; source test names and asserted contracts are the
reproducible evidence. No new evidence directory is required.

| Scope | Result | Temporary log |
|---|---|---|
| Source controls before Array integration | 101/101 passed | `/tmp/clef-string-callable-focused-9.log` |
| First integrated source Array/index/ingress run | 138/139 passed; formal-participant omission corrected | `/tmp/clef-array-source-focused-3.log` |
| First final source run after demand-inventory reuse | 124/142 passed; 18 failures exposed stale/throwing demand reads, owning sequencing correction made | `/tmp/clef-array-source-final-1.log` |
| Focused source rerun after demand lifecycle correction | 70/74 passed; two empty cases subsequently corrected, guard oracle corrected, unused-let defect remains | `/tmp/clef-array-source-final-2.log` |
| Final Array source controls on the frozen production DLL | 11/12 passed; unused-let demand failure retained | `/tmp/clef-array-source-final-4.log` |
| Composer ArrayAllocation/StringView/index integration | Production build passed, zero errors | `/tmp/composer-array-snapshot-build-final-2.log` |
| Linked snapshot/string/artifact/MLIR component controls | 48/54 passed; exact six failures below | `/tmp/composer-array-snapshot-focused-final-2.log` |
| Native artifact controls within that 54-case run | 17/17 passed, including fresh Hello World stdout/stderr/exit | Same log |
| Snapshot and StringView controls within that run | 8/9 passed; toBytes exact-Meet correspondence fails | Same log |
| String comparison and callable extent controls within that run | 19/20 passed; local-alias carrier mismatch | Same log |
| MLIR component controls within that run | 4/8 passed; both dynamic allocations and both opaque copies pass; four older array-read fixtures fail preparation | Same log |
| Final kernel/hardware controls | 18/18 passed: Kernel 12, Hardware 6 | `/tmp/composer-spatial-final-1.log` |
| Prior actual kernel source/registry/stock MLIR/native AIE verifier controls | 10/10 passed | `/tmp/composer-kernel-ingress-focused-2.log` |
| Prior hardware source/registry/CIRCT and retraction controls | 6/6 passed | `/tmp/composer-kernel-ingress-hardware-1.log` |
| Prior string comparison/extent and native artifact controls | 20/20 and 17/17 passed | `/tmp/composer-kernel-ingress-focused-1.log` |
| Actual source/MLIR SMT transfer, including false claims and spatial partitions | 163/163 passed | `/tmp/composer-smt-transfer-spatial.log` |
| Actual memory registry and nonzero descriptor offset/stride address controls | 5/5 and 1/1 passed | `/tmp/composer-memory-registry-verification.log`, `/tmp/composer-memory-address-component-2.log` |

The demand-lifecycle failure was useful: re-reading settled demand after each
Array rewrite asks a stale domain to justify the changed graph. The correction
takes the complete eligible candidate census before the batch rewrite, then
refreshes the owning source phases. Passive publication reads a validated
demand Result and refuses damaged authority; it must not throw, repair it or
invent a new runtime classification.

The guard-retraction oracle now checks the actual earlier owning refusal:
`PSG settlement (OrdinaryDemand)` reports that the retained unused-formal/actual
rows no longer match the complete use proof, with an occurrence and nonempty
participants. It still requires refusal of the corrupted predicate. The other
five construction/receipt/proof/access retractions retain their existing
specific refusal checks. No damaged graph was admitted to make a test pass.

### Exact failures to resume from

| Failing positive control | Observed failure | Required architectural correction |
|---|---|---|
| `ArrayConstructionCases`: unforced unused construction | `CCS8403`: dynamic array requires exactly one declared writable stack space, despite unused lazy binding | Repair the ordinary-demand/executable inventory at its source owner; do not add storage authority to this test or eagerly execute an unused binding |
| `StringBoundaryTests`: independent toBytes snapshots | `Published adaptation disagrees with its exact source numeric meet`, twice, before the runner | Reconcile the retained source copy read/store adaptations with the exact Meet used by shared composition; do not equate logical element identity with equal physical width |
| `StringComparisonWitnessTests`: callable extent `alias` | `AX4001` at node 106: witnessed `TInt (IntWidth 64)` versus published `TInt (IntWidth 8)` | Trace the alias/extent occurrence and held result correspondence; Alex must not select a replacement width |
| `MlirComponentTests`: four signed/unsigned array-read cases at Pointer 32/64 | Source preparation lacks complete callable carriers, a finite required range, MemorySettlement and SpatialSettlement | Restore valid source-owned fixture preparation and retain the signed/unsigned stock-lowering assertions; do not seed replacement publication facts in Alex |

The last row is four failures, so the table accounts for all six Composer
failures. These failures have not been excluded, weakened or reclassified as
passing. The new dynamic-allocation controls preserve the Count, element and
alignment through stock lowering at Pointer 32/i8/alignment 1 and Pointer
64/i32/alignment 4. They do not establish the older array-read cases.

A bounded read-only diagnosis narrowed the two registry failures further
(`/tmp/composer-snapshot-notch-diagnosis.log`):

- For toBytes, store 325 retains `Operand=314`, `From=8`, `To=32`,
  `ExtendUnsigned`. The Pattern instead passes value 230, whose source form is
  `Sequential [Require 313; read 314]`. Store 347 has the same disagreement
  between retained operand 336 and the enclosing sequence 277.
  `MemoryAccessRecipes.valueSite`/adaptation establishes the continuation as
  the numeric operand; `MemoryPatterns` passes the surrounding access frontier
  into `pPublishedAdapt`. The source memory fact must preserve both the actual
  value occurrence and exact numeric operand/result transport, and the Pattern
  must use those identities. Adding recursive alias/last-value discovery to
  the Pattern would restore the removed defect.
- For aliased length, node 106 is `Sequential [102;105]` and publishes i8;
  terminal call 105 publishes and witnesses i64. The generated length field
  operation 139 also publishes i64. This is a source composition/result
  correspondence gap. Settle the equality or adaptation relation at the
  sequence frontier and retain it. Choosing a width in StructuralWitness would
  hide the disagreement and violate the architecture.

These diagnoses made no production edits. They identify concrete source
correspondence work for the next slice and explain why the passive checks are
correct to stop.

The final 142-case source scope was not rerun after the last two empty-case
corrections. The final 11/12 Array result is a subset, not an inferred 141/142
result. Production builds passed; the initial final Composer test attempt also
encountered a test-only F# offside syntax error before execution. That syntax
was corrected without changing the negative oracle, and the recorded 54-case
run executed afterward.

### Reproduce the final bounded checks

Build the current source/compiler projects in dependency order before using
`--no-build`. The recorded final Array command rebuilt only the test project
against the frozen source DLL; the Composer and spatial commands used the
corresponding coherent compiled assemblies. In the repository named on each
line, these are the exact test scopes:

```sh
# clef
dotnet test tests/Clef.Compiler.Service.Tests/Clef.Compiler.Service.Tests.fsproj --no-restore -p:BuildProjectReferences=false --filter 'FullyQualifiedName~ArrayConstructionCases'

# Composer
dotnet test tests/Alex.Tests/Alex.Tests.fsproj --no-build --no-restore --filter 'FullyQualifiedName~StringBoundaryTests|FullyQualifiedName~StringComparisonWitnessTests|FullyQualifiedName~ArtifactProofTests|FullyQualifiedName~MlirComponentTests'
dotnet test tests/Alex.Tests/Alex.Tests.fsproj --no-build --no-restore --filter 'FullyQualifiedName~KernelModuleTests|FullyQualifiedName~HardwareModuleTests'
```

These commands document the scoped oracles. They are not an instruction to run
the manifest or full gate chain before correcting the named source contracts.

The owner's post-cleanup baseline was 0/51 samples compiling, 300/1605 CCS
failures and 51/307 Alex failures. Those are historical observations. No current
full manifest or complete suite run establishes restored parity. Eighteen
older lazy/sequence witness fixtures need current source declaration,
representation and residence prerequisites; they have not been waived.

## Remaining boundaries and next work

The accomplished cleanup should be preserved and extended. It is not a claim
that every Alex path is already passive or every language capability recovered.

1. Correct the exact Array/snapshot/alias failures recorded at this notch at their
   owning source contract. Preserve content-sensitive mutation/independence,
   empty/deferred behavior, complete ingress and guard/proof retraction oracles.
2. Rewrite BorrowedView using the already settled header/element/access facts.
   Its Pattern still constructs access guards, check widths, narrowing and a
   fixed unit carrier. Its mapped-span theorem assumes checked native
   extent/index; it supplies no executable acquisition/guard/callback/release
   correspondence. Establish the source scope/requirement contract and reuse
   it before claiming positive get/set capability. Seeded headers cannot prove
   lifetime. This work is queued, not implemented at this notch.
3. Reconcile remaining FPGA function, record and conditional Patterns with
   the portable/backend boundary. The new spatial declaration witnesses are
   passive; those common Patterns still contain target-specific forms.
   Preserve actual hardware/kernel positive controls through this correction.
4. Complete source constructions for missing String operations and positive
   lazy/sequence fixtures. A refusal alone does not implement a capability.
5. Extend artifact contracts and restore sample parity under all three required
   discharge stages. Do not substitute stock-MLIR execution for native artifact
   acceptance, or run a full gate chain merely because the compiler builds.

HelloNappy/device execution is not restored. Actual StrixHalo still has
Core=None; HelloNappy's host buffers and descriptorless XRT boundary need proper
declarations and encoding. A Peano probe for full int32-by-int32 multiplication
with int64 result produced an elf32-aie/aie2p relocatable object with undefined
__muldi3. The current pathway lacks executable core linking/runtime composition.
Do not change widths or restrict the source math to hide that requirement.
No xclbin, device, FPGA bitstream or sample restoration is claimed from verifier
success.

The independent assessment remains separate and unchanged. No owner decision
is pending on the ownership direction. Continue the reimplementation by relying
on Baker's established PSG decisions, publishing missing correspondence and
simplifying the shared Elements/Patterns/Witnesses around that authority.
