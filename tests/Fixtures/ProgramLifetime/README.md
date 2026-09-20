# Selected program-lifetime declaration projection

This shared fixture exercises the real `.fidproj` platform selection and source
dependency path through CCS.Editor and Lattice. Like the compiler's
`ProgramLifetimeCases`, it uses a fabric selection to isolate declaration
checking from CPU representation offerings; it makes no native image claim.

Immutable and mutable roles name `constant-vault` and `state-vault`, then both
are renamed through unsaved edits to `Names.clef`. Definition queries retain the
exact source declarations. `Cases.json` supplies compiler-confirmed CCS8206 and
CCS8207 messages, with markers around the expected exclusive source spans.
Each error is followed by an unsaved repair, retaining the original disk files.
The expected message substitutes `{platform}` with the selected fixture
directory's platform ID, retaining the complete public project diagnostic.

`Startup.fidproj` is a separate source-only check. It exposes the compiler's
ordered startup plan, exact initializer IDs and source spans, and storage intent
without native space authority. An unsaved edit makes an initializer callable
opaque; the graph's pending dependency fact remains queryable without turning
the native settlement requirement into a source error. Repair restores the plan.
The editor snapshot and `clef/programInitialization` LSP query project these PSG
relations; neither client reconstructs initialization order.

After the corresponding compiler/editor/server outputs are built, the three
public projection gates are:

```sh
# Composer
dotnet run --project tests/CCS.Editor.Tests --no-build -- --program-lifetime
# lattice-analyzers
dotnet run --project tests/Lattice.CCS.ProgramLifetime -p:BuildProjectReferences=false
# lattice-vscode/client, Node.js 22+
npm run test:platform
```

The analyzer companion and LSP gate retain evidence under `/tmp`. All three
consume these same source files and diagnostic expectations.
