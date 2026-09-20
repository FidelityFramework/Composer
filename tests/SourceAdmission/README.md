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
ordinary `use` binding without an admitted resource lifecycle. Each
requires exit 1, exactly one effective `CCS8401` diagnostic with its exact message
at the expected project-relative filename and start line, the single-error source-gate summary,
and absence of a native executable or witnessed MLIR. An unrelated nonzero exit
does not pass. The CLI currently prints only the start line; complete start/end
spans remain the responsibility of CCS and editor projection tests.

The positive control combines ordinary function calls, `Result.iter` and a
counted loop. It must compile, pass stock `mlir-opt --verify-each`, exit zero and
produce exact output. Existing intrinsic `seq` remains a separate source-only
admission control; this gate does not claim that native sequence frames work.

Each run retains source, projects, logs and compiler hashes in its printed
temporary directory. `evidence.json` records every selected result, including
failures; verifier/native exit -1 means that stage was not run. Cases execute
sequentially, and an unknown selector fails before compilation.
