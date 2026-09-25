# Costanich front-end investigation

September 19, 2026. Read-only investigation of the adjacent F# fork for the
owner's question about declaration order and earlier construction of Clef's
semantic graph. No fork code was ported, no dependency was added, and no compiler
was built or executed for this investigation.

The useful contribution is a concrete declaration/reference discovery stage
between parsing and checking, with dependency ordering and explicit recursive
file components. This is relevant to Clef's front-end restructuring. The reviewed
implementation does not establish a compilation speedup, general freedom from
type-declaration order, or a complete incremental semantic dependency contract.

## 1. Source identity and evidence boundary

The local repository is `/home/hhh/repos/costanich-fsharp`, with origin
`https://github.com/bryancostanich/fsharp.git`. Its clean working tree is on
`main`, commit `a0cce495d32e0f5f52c71fbda7313fa515c86c68`, dated March 27, 2026.
That checkout does **not** contain the feature under review.

The feature is present in the locally available remote-tracking branch
`origin/fix_dogmatic_file_order_nonsense`, at
[`8bcc791eaa89050d7af4226a912dcc085d641a32`](https://github.com/bryancostanich/fsharp/commit/8bcc791eaa89050d7af4226a912dcc085d641a32),
authored April 27, 2026, 10:40:17 -07:00: “Unify file-order-auto walker with
FileContentMapping.” The feature's first commit is `699d8d063`; its parent,
`a5df95f9fdc611690273c3495b613ba302bd4108`, is the comparison base. That bounded
diff changes 55 files, with 4,342 additions and 16 deletions; comparing against
the older checkout's main branch also includes unrelated upstream changes.

These identities were verified with local `git rev-parse`, `git show`,
`git diff` and `git status`. No fetch or branch checkout was performed. Pinned
links below identify the inspected Git objects, not a claim about the current
public PR or remote branch. Branch files were extracted for line-numbered reading
under `/tmp/costanich-frontend-review-hqi03a1z/`.

Clef's comparison checkpoint is clean commit
`259d4786c037704c5b9622a7efd1bc2a75e1b0aa`. Composer is being developed
concurrently; its existing
[nanopass/incremental direction](Nanopass_Incremental_Contract_Direction.md)
governs the interpretation here.

## 2. What the fork actually changes

The option defaults off. The build driver calls `applyAutoFileOrder` after
parsing and initial environment creation, before `CheckClosedInputSet`. It skips
this path when compiling FSharp.Core. The parser itself is unchanged by the
feature. [Driver, lines 142–180](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Driver/fsc.fs#L142).

The active build path has four steps:

1. Collect declarations and reference paths from every parsed file. Declaration
   summaries include names, arities, accessibility, kinds, members, opens and
   nested modules. Collection uses `Array.Parallel.mapi`; subsequent stub
   insertion folds through the type environment.
2. Build a declaration export map, separate AutoOpen alias map and dependency
   sets. Preserve signature/implementation pairing, redirect signature references
   to the implementation for ordering, and compute strongly connected components
   with iterative Tarjan traversal.
3. Emit dependency-first single files. Eligible multi-file components become a
   synthetic recursive namespace; groups containing signature files or namespaces
   that would need wrapping fall back to original order within the group.
4. Mark the reordered last implementation as `IsLastCompiland` and pass the result
   to the existing checker.

The complete implementations are
[SymbolCollection.fs](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Checking/SymbolCollection.fs)
and
[CycleGroupProcessing.fs](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Checking/CycleGroupProcessing.fs).
The active ordering entry is `computeCompilationUnits` at line 1557. The older
`computeDependencyOrder` at line 1465, with its opens-only retry after a cycle,
is not the entry used by either `applyAutoFileOrder` or
`computeReorderedFileNames`.

The enter phase creates F# `Entity` shells with rigid parameters and no completed
representation. At this tip its active `buildTopLevel` path includes public
types directly under namespaces; it skips top-level named modules and their
nested contents. Earlier module-stub helper functions remain in the file, but
that does not mean they are used by `buildTopLevel`.
[Stub construction, lines 574–681](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Checking/SymbolCollection.fs#L574).

The discovery graph is a file-scheduling approximation from syntax. Its name
matching incorporates prefixes and declaration kinds; it is not a graph of
settled type judgments. Bare expression identifiers are deliberately omitted
except at function-application heads. Opens and captured reference paths are
flattened into each file's summary; local-name suppression uses file-level
declarations, not a full lexical binding analysis. Consequently, completeness
for aliases, passed function values and nested shadowing requires separate
evidence before this technique could define Clef's checking frontier.
[Reference collection, lines 683–797](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Checking/SymbolCollection.fs#L683),
[resolution, lines 1163–1270](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Checking/SymbolCollection.fs#L1163),
[expression walker, lines 459–487](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Driver/GraphChecking/FileContentMapping.fs#L459).

## 3. File order, type order and removal of `and` are distinct claims

| Claim | Evidence at the inspected tip | Limit |
|---|---|---|
| A consumer file can be listed before its provider | Acyclic dependency sorting; component test lists A before B and expects compilation success | Inherits the reference-discovery limitations above |
| Mutually recursive types can be split across files | Tree/Forest component test; eligible file SCCs become a recursive namespace | Build-only synthesis; signature and namespace guards retain original ordering |
| Arbitrary type declarations inside one file can be reordered | No declaration-level sorter was found; a single-file SCC passes through unchanged | This broader claim is not established |
| `and` is unnecessary everywhere with the flag | A warning is emitted on existing `and` tails; tests check warning/silence | No component test replaces a same-file `and` group with separate mutually recursive declarations |
| All source permutations preserve semantics | Some deliberately misordered fixtures and a self-host shuffle script exist | Not a demonstrated universal property, particularly for initialization effects |

The
[component tests, lines 22–73](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/tests/FSharp.Compiler.ComponentTests/TypeChecks/FileOrderAuto/FileOrderAutoTests.fs#L22)
exercise file ordering, cross-file type recursion and signature pairing. The
[`and` tests, lines 213–241](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/tests/FSharp.Compiler.ComponentTests/TypeChecks/FileOrderAuto/FileOrderAutoTests.fs#L213)
retain the original `and` syntax. In this upstream prior art, the checker changes add FS3887 warnings;
they do not collect separate same-file declarations into a recursive group.
[CheckDeclarations.fs, lines 5203 and 5295 onward](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Checking/CheckDeclarations.fs#L5203).

For example, replacing `and Forest = Tree list` with `type Forest = Tree list`
in the single named-module warning fixture leaves a forward `Forest` reference.
Static tracing shows that this file receives neither named-module stubs nor
multi-file cycle synthesis. The reviewed changes therefore do not provide the
mechanism advertised by that broad reading of “no and.” This is an implementation
inference, not a reproduced compiler failure: the fork was not executed here.

The migration guide's instruction to replace `and` chains with separate
declarations is broader than this evidence. Its design companion also says
within-file recursive semantics are unchanged. Record the discrepancy; do not
use the warning itself as evidence that the replacement compiles.
[Migration, lines 72–87](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/docs/file-order-auto-migration.md#L72),
[design, lines 298–308](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/docs/file-order-auto-design.md#L298).

## 4. Build, editor and incremental behavior

The legacy FCS `IncrementalBuilder` hook eagerly pre-parses disk files when the
builder is created, computes a filename order, then hands that order to normal
lazy processing. It does not hand those parsed trees into a new parse cache.
An exception during this prepass silently restores the original source list.
There is no new incremental dependency-graph invalidation mechanism in this
hook. [IncrementalBuild.fs, lines 1594–1619](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Service/IncrementalBuild.fs#L1594).

There are two additional differences from the build path:

- FCS neither pre-populates `TcEnv` nor synthesizes cycle groups. It keeps original
  order inside each component, while ordering the components themselves.
- At this tip, build discovery consumes `FullPathIdentifier`, but FCS's
  `computeReorderedFileNames` still consumes `PrefixedIdentifier` and ignores the
  new full-path entry. Thus discovery itself differs, even for acyclic projects.
  [FCS collector, lines 405–482](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Checking/CycleGroupProcessing.fs#L405).

Eight added incremental component tests cover a misordered diamond, dependent
changes, signature shielding, file addition/removal, one added edge and project
reordering. The edge-addition test changes Third to depend on Second, where
Second already precedes Third. It does not establish that an unsaved edit which
requires reversing the cached order recomputes that order.
[Incremental tests, lines 18–123](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/tests/FSharp.Compiler.ComponentTests/FSharpChecker/FileOrderAutoIncremental.fs#L18).

The separate TransparentCompiler path still calls the inherited
`DependencyResolution.mkGraph` on snapshot source order, with no call to either
auto-order entry. Its inherited resolver restricts dependencies to earlier
indices. The synthetic-project harness selects that compiler through its
experimental/environment setting, so a future reproduction must record which
FCS engine actually ran; the new tests' presence alone does not establish both
paths. [TransparentCompiler.fs, lines 1280–1379](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Service/TransparentCompiler.fs#L1280),
[inherited resolver, lines 203–227](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Driver/GraphChecking/DependencyResolution.fs#L203),
[harness selection](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/tests/FSharp.Test.Utilities/CompilerAssert.fs#L342).

## 5. Validation and timing claims

The fork's release notes report 15,404 upstream tests passing. Its OSS record
reports seven clean projects; Suave retains the same 30 baseline errors, and
five other projects retain baseline toolchain/environment failures. These are
the author's recorded results, not tests rerun during this investigation.
[Release notes, lines 24–36 and 81–101](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/docs/file-order-auto-release-notes.md#L24),
[OSS results, lines 3–25](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/tests/file-order-auto-test/oss-sweep/RESULTS.md#L3).

The OSS reproduction instructions build projects with the flag; they do not
shuffle their source lists. The separate self-host script shuffles only
single-line `<Compile Include=.../>` entries with seed 42; multiline entries
stay in place. The script's existence is not a retained successful-run log.
Neither source substantiates a completed randomized-permutation campaign across
all listed OSS projects. [OSS reproduction, lines 27–57](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/tests/file-order-auto-test/oss-sweep/RESULTS.md#L27),
[self-host script, lines 26–86](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/tests/file-order-auto-test/self-host-test.sh#L26).

The end-to-end script explicitly passes `OtherFlags=--file-order-auto+` because
the installed SDK's build task does not yet translate the new MSBuild property.
It therefore exercises compiler invocation and output, not standalone sufficiency
of the new property on a stock SDK.
[End-to-end script, lines 56–82](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/tests/file-order-auto-test/end-to-end/run.sh#L56).

Performance characterization is explicitly listed as **not measured** in the
release notes. The new build prepass parallelizes collection; it introduces no
new parallel typechecker. At the reviewed tip the build collection walks the
shared FileContentMapping once for opens and again for full references. FCS adds
an eager parse before its ordinary processing. None of these observations alone
determines total latency or memory.
[Performance caveat, lines 126–135](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/docs/file-order-auto-release-notes.md#L126),
[collection, lines 758–797](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Checking/SymbolCollection.fs#L758).

The inherited `GraphChecking/Docs.md` describes a **different** feature:
parallel checking within F#'s existing permitted file-order dependencies, using
per-file state deltas. Its .NET 7 benchmarks must not be attributed to Costanich's
auto-order change or to Clef's native checker.
[Inherited graph-checking design, lines 62–170 and 292–332](https://github.com/bryancostanich/fsharp/blob/8bcc791eaa89050d7af4226a912dcc085d641a32/src/Compiler/Driver/GraphChecking/Docs.md#L62).

## 6. Relation to the current Clef front end and Baker

Clef has inherited GraphChecking helper files in its project, but the native
path does not invoke them. Its project explicitly removes `CompilerConfig`,
`ParseAndCheckInputs`, the old `fsc` driver and service orchestration. Searches of
`src/Compiler` find none of `SymbolCollection`, `CycleGroupProcessing`,
`FullPathIdentifier`, `fileOrderAuto` or `applyAutoFileOrder`.
[Clef project, lines 358–400](../../clef/src/Compiler/Clef.Compiler.Service.fsproj).
The Costanich implementation has not been inherited into this active path.

The present path is concrete:

- [SourceResolver](../../clef/src/Compiler/Project/SourceResolver.fs), lines 51–55,
  122–129 and 147–184, follows transitive `.fidproj` dependencies and preserves
  each project's declared source order.
- [ProjectChecker](../../clef/src/Compiler/Project/ProjectChecker.fs), lines
  129–225 and 258–303, reads or overlays sources, parses them with a sequential
  `List.map`, builds platform context and invokes the native multi-file checker.
- [NativeService](../../clef/src/Compiler/NativeTypedTree/NativeService.fs), lines
  163–257, uses the inherited lexer/parser to produce syntax. Lines 2540–2693
  thread the native `TypeEnv` through declarations, modules and files, construct
  PSG nodes and solve constraints. Recursive binding groups and union/record
  type groups already pre-register their own identities at lines 1953–2016 and
  2077–2149; that is local group handling, not whole-project discovery.
- [NameResolution](../../clef/src/Compiler/NativeTypedTree/NameResolution.fs),
  lines 195–297, implements lexical paths, explicit opens, aliases, shadowing and
  canonical exports. The native checker rejects AutoOpen at NativeService lines
  1910–1915. Costanich's AutoOpen recovery policy is therefore not a Clef language
  requirement.
- NativeService lines 950–1090 resolve node types, form the structural graph,
  expand/prune, run intrinsic and Baker fan-out/fold-in, elaborate obligations,
  read platform declarations and run range analysis. Declaration scheduling is
  upstream of these semantic responsibilities.

There is also no working Clef signature-checking contract to inherit from the
fork's `.fsi` pairing: the multi-file API currently skips signature inputs at
lines 2674–2676, while the single-input API rejects them at lines 2709–2721.
This is a present discrepancy to keep visible, not evidence of admitted
signature-based parallel boundaries.

The reusable idea is **separating declaration identity discovery from body
checking**, and making recursive dependency components explicit. A Clef version
must use native type/measure identities, native lexical lookup and source
provenance. Body inference, dimensional and other admission judgments, captured
behavior, effects and initialization order remain owned by the existing native
semantic machinery. F# `TcEnv`/`Entity` shells and synthetic namespace rewrites
are not a portable implementation of those contracts.

The existing [Baker architecture](../../clef/docs/fidelity/Baker_Saturation_Architecture.md)
separates ingredients, patterns, recipes and fold-in. The newer
[incremental direction, sections 1–4 and 8](Nanopass_Incremental_Contract_Direction.md)
requires complete crossing relationships, actual observations, missing-name
dependencies, support for derived facts, revision freshness and explicit
settlement. Those requirements distinguish an initial scheduling graph from the
PSG hypergraph and its reusable judgments. A file SCC is useful discovery data;
it is not a proof-region boundary or an analysis fixed point by itself.

Before attributing a timing benefit, a bounded experiment would need separate
measurements of parsing, discovery, native resolution/constraint solving,
structural PSG construction and Baker settlement, plus retained memory and
recomputation under edits. Current shared constraint stores, the module-level
`solvedConstraintCount` and reset-based node allocation are further reasons that
changing the file loop to parallel execution is not the feature reviewed here.
[NativeService, lines 363–378 and 2655–2693](../../clef/src/Compiler/NativeTypedTree/NativeService.fs),
[current fan-out boundary](Nanopass_Incremental_Contract_Direction.md#1-existing-boundaries-to-preserve).

The relevant acceptance pressure is: permutations of acyclic source files;
separate same-file type declarations versus true recursive groups; recursive
cross-file bodies and types; aliases and function values under nested shadowing;
unsaved edits that reverse a dependency; unchanged versus changed exported NTU
schemes; initialization effects; and identical build/editor diagnostics and
source identities. Compare each with a fresh native check and the applicable
FidelityHello behavior oracle. This records the investigation boundary rather
than choosing an implementation or importing a foreign type system.

## 7. Read inventory and remaining uncertainty

Fully read from the pinned feature branch:

- `docs/file-order-auto-design.md` (308 lines), migration (155), release notes
  (173), and `tests/file-order-auto-test/oss-sweep/RESULTS.md` (133).
- `Checking/SymbolCollection.fs` (1,703) and `.fsi` (88);
  `Checking/CycleGroupProcessing.fs` (482) and `.fsi` (57).
- `TypeChecks/FileOrderAuto/FileOrderAutoTests.fs` (241),
  `FSharpChecker/FileOrderAutoIncremental.fs` (123), both FCS smoke `Program.fs`
  files (88 and 208), self-host script (118), and end-to-end script (102).
- Inherited `Driver/GraphChecking/Docs.md` (332), to distinguish its mechanism and
  old performance evidence from this feature.

Fully read locally for the comparison: Clef `ProjectChecker.fs` (363),
`SourceResolver.fs` (203), `NameResolution.fs` (335),
`Baker_Saturation_Architecture.md` (777); Composer
`Nanopass_Incremental_Contract_Direction.md` (338 at reading). The fork checkout's
main README was also read; it is the inherited upstream overview.

Targeted, not whole-file reads: all feature diff hunks in the existing checker,
driver, config/options, FileContentMapping, GraphChecking types/resolution,
IncrementalBuild, project wiring and MSBuild files; TransparentCompiler dependency
construction and test-harness engine selection; Clef NativeService parser,
constraint/generalization and recursive registration sections plus the
construction/saturation and project-check entry paths; Clef project inclusions.
The remainder of the upstream checker and parser was not reread in full. Linked
`conductor/tracks/...` documents mentioned by the migration guide are absent from
this branch's Git tree. External post/PR identity and any evidence newer than
the locally available feature tip are outside this local audit.

No fresh runtime, upstream suite, randomized OSS, compatibility or performance
result is claimed. The note distinguishes inspected implementation, author-recorded
results and static inferences; the known build/editor divergence and the broad
same-file/no-`and` claim remain unresolved by executed evidence here.
