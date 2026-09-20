# Public source admission gate

This standalone harness invokes the existing Composer executable against fresh
CompilerSurface projects. It links `tests/Infrastructure/Process.fs` for direct
arguments, concurrent output drainage, timeouts and process-tree termination;
it has no compiler project reference and does not rebuild Composer.

From Composer, after the coordinated compiler build:

```sh
dotnet run --project tests/SourceAdmission/SourceAdmission.Tests.fsproj -- src/bin/Debug/net10.0/Composer
```

Exact case names may follow the compiler path. For the initial false-acceptance
regression alone:

```sh
dotnet run --project tests/SourceAdmission/SourceAdmission.Tests.fsproj -- src/bin/Debug/net10.0/Composer function-return
```

Seven negative cases cover a plain function used with CE return, let! or do!, a
lexically shadowed `seq`, return/yield outside an owning computation, and an
ordinary `use` binding without an admitted resource lifecycle (`CCS8401`). Three
sequence typing cases reject incompatible yielded dimensions (`CCS8040`), a
scalar `yield!` operand, and a yield!-only result contradicting its annotation
(`CCS8003`). Two producer cases reject a `Seq.map` callback whose input dimension
differs from its sequence, and `Seq.append` inputs with different element
dimensions (`CCS8040`). Their selectors are `sequence-map-dimensions` and
`sequence-append-dimensions`. Each negative requires exit 1, its exact set of effective diagnostics
with the expected codes and messages at the expected project-relative
filenames and start lines, the matching error-count source-gate summary,
and absence of a native executable or witnessed MLIR. An unrelated nonzero exit
does not pass. The CLI currently prints only the start line; complete start/end
spans remain the responsibility of CCS and editor projection tests.
Type-mismatch messages also embed the source path and start column; the harness
substitutes the generated absolute source filename into that exact expectation.

Three C-06 boundary selectors cover the public source-to-native path:

- `sequence-current-is-internal` requires CCS8009 for the unavailable source
  name `SeqEnumerator.current`. This does not test the internal current-read
  certificate; CCS graph tests own that prerequisite.
- `sequence-unknown-input-region` requires CCS8403 when an iterator's sequence
  parameter has no established backing region.
- `sequence-factory-local-cell` requires the explicit CCS8403 factory-local
  capture lifetime failure and both dependent residence findings. Moving the
  result frame cannot establish lifetime for the returning activation's cell.

The first two require exactly one error each; the factory case requires exactly
three. Graph-residence diagnostics include generated node IDs. Explicit
`{node:name}` placeholders match only decimal IDs; the complete surrounding
reason, code, source line and total remain exact. Evidence retains both the
expected templates and every actual diagnostic, so extra errors cannot pass.

The coordinated 2026-09-20 full run passed 12/16 cases on CCS `5dad941b…40f44`;
four sequence type failures received extra target-settlement diagnostics. After
the compiler retained the source rejection and skipped target frame synthesis,
all four passed their unchanged single-error expectations on CCS
`08d547524f7c76f61bb82e4a67e2f04ffe64b6ca37da7637ba2c4b2c07384482`.
This records twelve earlier passes plus four corrected passes, not a full rerun
on the final hash. Evidence is retained at
`/tmp/composer-source-admission-000edec5a3b74f4286abaac3d3dba11c/evidence.json`
and `/tmp/composer-source-admission-15540117b5aa4ec4928ea13d70ce4f38/evidence.json`.

The positive control combines ordinary function calls, `Result.iter` and a
counted loop. It must compile, pass stock `mlir-opt --verify-each`, exit zero and
produce exact output. Sequence type and graph checks live in CCS and editor
tests; sequence execution has a separate coordinated native oracle. This gate's
negative results do not establish native frame or resumption behavior.

The [C-06 checkpoints](../../docs/Language_Coverage_Waypoints.md) record ownership
in Baker's graph, retire the old shape coeffect, and elaborate admitted delegation
into owner-local iteration. The delegated yield retains both origin and delimiter
relations. This harness checks the specified upstream source and settlement
refusals; it does not substitute an Alex error for an expected CCS diagnostic.
The graph prerequisites and valid native executions have separate gates.

Each run retains source, projects, logs and compiler hashes in its printed
temporary directory. `evidence.json` records every selected result, including
failures; verifier/native exit -1 means that stage was not run. Cases execute
sequentially, and an unknown selector fails before compilation.
