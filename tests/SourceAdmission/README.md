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
`sequence-append-dimensions`. Each negative requires exit 1, exactly one effective diagnostic
with its expected code and exact message at the expected project-relative
filename and start line, the single-error source-gate summary,
and absence of a native executable or witnessed MLIR. An unrelated nonzero exit
does not pass. The CLI currently prints only the start line; complete start/end
spans remain the responsibility of CCS and editor projection tests.
Type-mismatch messages also embed the source path and start column; the harness
substitutes the generated absolute source filename into that exact expectation.

The positive control combines ordinary function calls, `Result.iter` and a
counted loop. It must compile, pass stock `mlir-opt --verify-each`, exit zero and
produce exact output. Existing intrinsic `seq` remains a separate source-only
admission control; successful sequence type and graph checks live in CCS and
editor tests. This gate does not claim that native sequence frames work.

Each run retains source, projects, logs and compiler hashes in its printed
temporary directory. `evidence.json` records every selected result, including
failures; verifier/native exit -1 means that stage was not run. Cases execute
sequentially, and an unknown selector fails before compilation.
