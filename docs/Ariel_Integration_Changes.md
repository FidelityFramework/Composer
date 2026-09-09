# Ariel integration: reconciled repository record

Reconciliation completed on September 9, 2026. The compiler used for the native
Ariel and HelloWayland work is integrated into the normal Clef checkout.

- Clef `main` and `fidelity` both point to merge commit `534429798`, whose parents
  are the previous `main` (`ad17be2d8`) and `fidelity` (`d86a3e3a0`). Main's PHG,
  predicate propagation and backward range refinement are retained.
- Composer and CCS.Editor use the normal sibling project,
  `../clef/src/Compiler/Clef.Compiler.Service.fsproj`. The temporary ignored
  `Directory.Build.local.props` override was removed. A restore-aware build
  confirmed the canonical path in the actual dependency records.
- `/home/hhh/repos/clef.worktrees/clef-scope-integration` was removed after source
  comparison and native acceptance. It is no longer registered as a worktree.
  The other Clef checkout is clean and aligned with `main`.
- Fidelity.Data's hosted library and test projects now use `.fs` filenames.
  All 39 renamed source bodies are byte-identical to their predecessors.

## Repository commits

| Repository | Integration commit | One-line commit message |
| --- | --- | --- |
| clef | `534429798` | Merge scoped native mapping and dimensional compiler integration into main |
| Composer | `0663126` | Lower typed foreign boundaries and scoped mapped storage through LLVM and LLD |
| BAREWire | `d895cfd` | Describe typed foreign ownership, callbacks and borrowed mappings with focused metadata projects |
| Farscape | `d29ba7c` | Generate typed pthread and display bindings with ABI projections and scoped GBM mappings |
| Fidelity.Platform | `9366692` | Add persistent Ariel CPU carriers and generated Linux display bindings |
| HelloWayland | `aeb64fe` | Render the CPU Wayland window through Ariel carriers and scoped BAREWire mappings |
| Fidelity.Data | `213918a` | Restore .fs source names for hosted Fidelity.Data builds and tests |

The final documentation commits use these messages:

- Composer: `Reconcile compiler checkout and record native Ariel acceptance`
- HelloWayland: `Record Ariel window acceptance from the reconciled compiler`

## Validation through normal repositories

- The complete Clef compiler regression project passes **169 tests**.
- Fidelity.Data passes **374 tests**, with one existing ignored test.
- Composer and Lattice.Server build successfully, using the normal sibling
  compiler and dependency repositories. No compiler DLL was copied from the
  temporary checkout.
- The fresh native mapped-carrier gate passes **64 frames with four carriers**,
  exact U32 pixel readback, matching worker creation/join and no heap allocations
  in either scoped callback body.
- The complete HelloWayland CPU application compiles from the reconciled tree.
  A 20-second GUI observation identifies all **31 Ariel worker threads** and
  confirms CPU work on each. Animation changes 19,495 orange glyph pixels;
  static caption and panel pixels remain byte-identical. Resize to 820 × 960
  succeeds and normal close joins the carriers with exit status zero.

Reproduction entry points, run from each named repository:

```sh
# clef
dotnet test tests/Clef.Compiler.Service.Tests/Clef.Compiler.Service.Tests.fsproj
# Composer
dotnet build src/Composer.fsproj
dotnet build src/Lattice.Server/Lattice.Server.fsproj
# HelloWayland
python3 tests/ariel-window/run_mapped_carriers.py
../Composer/src/bin/Debug/net10.0/Composer compile HelloWayland.fidproj -k --no-color
./targets/CPU-HelloWayland
```

The recorded native evidence is under
`/tmp/hello-wayland-mapped-carriers-r7m2u58p` and
`/tmp/hello-wayland-window-main-reconciled`. See
[the implementation and proof scope](multi-core-cpu.md) and
[the application acceptance record](../../HelloWayland/docs/multi-core-cpu.md).

## Recovery history

Before removal, both the original index tree and complete nonignored working
state were retained in Clef's local Git recovery history:

- Ref: `refs/archive/ariel-scope-integration-20260909`
- Working snapshot: `b6893e6f484b9eb89d67d862f469b98e7d25dd83`
- Index, metadata and the superseded validated patch handoff:
  `.git/reconciliation-backups/ariel-scope-20260909` under the normal Clef repo.

The archived working source matches reconciled `main`; the sole additional
tracked file on `main` is the completed reconciliation handoff document.
The thousands of inherited staged deletions already matched `main`'s cleanup.
Ignored files were build outputs and regenerated parser files, with no manual
assets or active configuration omitted. The archive is recovery history, not
another checkout or a build dependency. The obsolete patch handoff was removed
from Composer's pending files after integration.
