# CCS editor read service

This .NET-hosted service uses the same `ClefCompilerServiceProject` selection as Composer. The normal build uses the sibling `clef/src/Compiler/Clef.Compiler.Service.fsproj`. The compiler integration is merged into Clef's `main`, and this workstation uses that normal checkout without a local override.

For an intentional alternative compiler checkout, pass an MSBuild property or create an ignored `Directory.Build.local.props` at the Composer root:

```xml
<Project>
  <PropertyGroup>
    <ClefCompilerServiceProject>/absolute/checkout/src/Compiler/Clef.Compiler.Service.fsproj</ClefCompilerServiceProject>
  </PropertyGroup>
</Project>
```

`EditorSession(projectPath)` checks `.fidproj` inputs through CCS's `checkProjectWithVolatile`. CCS owns source ordering, project dependencies, platform facts, inference, diagnostics and obligation construction. The service serializes checks across its sessions because the current compiler has process-global state. It freezes display values before releasing that lock; it does not expose mutable compiler type cells or lazily computed graphs.

`CheckAsync(Map<absolutePath, unsavedText>)` reserves a revision immediately and clears `Current` while checking. A superseded check returns `None`. `Invalidate()` also clears current reads. `TryHover(revision, filePath, line, character)` accepts zero-based lines and UTF-16 columns and rejects stale revisions. Lookup uses compiler source intervals and resolved reference identities, including shadowed bindings. Missing positions return `None`; compiler type formatting supplies the displayed dimensions.

Snapshots retain the exact source strings, the compiler assembly hash, original and effective diagnostic severities, and structured compiler ranges. Current parser failures remain separate file-associated message lists because CCS does not expose structured parser ranges here. Clients must show that limitation without inventing positions by parsing message text.

`InputFiles` lists those source paths and the exact root, platform and transitive dependency manifests selected through `FidprojLoader` metadata. Consumers must watch this set beyond their workspace folder and replace it after every check. Known missing referenced paths remain listed on failure so creating or repairing an input can trigger a check. Version-only dependencies currently skipped by CCS do not acquire invented local paths.

Obligations retain their compiler statement, logic, origin, references, graph-source premises and exact compiler-generated SMT-LIB query. `ProofDispatch.checkAsync` runs the selected cvc5 executable against one query and returns a separate result keyed by its query hash. It distinguishes proved, counterexample, unknown and error; caller cancellation remains cancellation. The current `Solver` field identifies the configured executable. A successful query establishes this source obligation under its encoded premises; it does not establish preservation through native lowering. Consumers must associate results with the snapshot revision and discard results after invalidation.

Run the focused executable from Composer with .NET 10 and cvc5 on `PATH`:

```sh
dotnet run --project tests/CCS.Editor.Tests/CCS.Editor.Tests.fsproj
```

The tests check ordered two-file inference, shadowed references, unsaved dimensional errors and import removal/restoration, retained snapshots, UTF-16/CRLF positions, invalidation, parser failures, generated obligations, and real cvc5 verdict/cancellation boundaries. They do not build Composer's native backend.

`--loop-obligations` checks the finite additive recurrence projection: both
obligation kinds retain navigation to the initial cells, loop and exact stores.
An unsaved bound change refreshes their queries; replacing the additive update
retracts those obligations, and repair restores them without changing an earlier
snapshot. This checks revision behavior of the current whole-project service;
it does not claim dependency-directed incremental recomputation.

`--program-lifetime` checks selected-platform storage designations and startup
graph projections, including declaration locations, unsaved repairs, ordered
initializer identities and pending native prerequisites. `EditorSnapshot`
publishes these immutable PSG observations; the Lattice server exposes them via
the versioned, read-only `clef/programInitialization` query. Storage intent and
proved writable authority are separate fields. Shared fixtures live in
[`tests/Fixtures/ProgramLifetime`](../../tests/Fixtures/ProgramLifetime/README.md).

To check the GUI sample and dispatch its source obligations:

```sh
dotnet run --project tests/CCS.Editor.Tests/CCS.Editor.Tests.fsproj -- \
  --sample samples/lattice/HelloDimensionsProof/HelloDimensionsProof.fidproj
```

The sample requires the sibling platform and source dependencies declared by its project. The [Lattice integration plan](../../docs/Lattice_Integration.md) describes the surrounding server and editor work.
