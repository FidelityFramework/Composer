# Regression check scope

Choose checks from the semantics and compiler contracts changed in the current
step. Run the focused checks before advancing that step; expand when the impact
or results warrant it. A full repository or sample run is a delivery gate, not
the default response to every edit. This policy selects evidence efficiently;
it does not reduce the [C-series acceptance contract](PRDs/C-Series-Acceptance.md)
or the corresponding F-series obligations.

## Checkpoint cadence

A complete C-xx gate warrants a completion checkpoint. A coherent change spanning
several feature gates can warrant an intermediate checkpoint once its affected
checks pass. Record the actual source, component, artifact and execution evidence
with the coordinated repository revisions; an intermediate checkpoint does not
close a whole feature gate.

Keep a coupled migration in an isolated worktree when completing its contract
would exceed the verified checkpoint. Preserve its code and failing evidence,
then continue the migration against the new baseline. Do not invent a missing
origin, relax a lifetime proof, weaken an oracle or introduce an alternate emitter
to make the checkpoint green. An authorized push contains the reproducible,
verified scope; the worktree retains the ongoing integration.

## Checks for an implementation step

Start with a discriminating case for the defect or new behavior. State what it
must establish, which stage owns that fact, and which neighboring behavior could
change. Use existing cases where they cover the contract; add a regression when
the changed behavior lacks a meaningful oracle.

| Changed contract | Focused evidence |
|---|---|
| Source admission, typing, application or Baker construction | Source-driven CCS tests for the changed form; inspect the actual settled graph, ordered occurrences, captures, generated formals and relevant recipes. Include the corresponding rejection case when admission changes. |
| Ranges, dimensions, storage, lifetime or proof premises | Owning CCS checks and affected proof tests, including invalidated or contradicted premises. A projected obligation or emitted SMT module does not establish discharge. |
| Editor-visible graph facts, diagnostics or incremental invalidation | The relevant CCS.Editor mode and, where affected, Lattice integration. Check edits, current revisions, source locations and retained snapshots; a fresh batch result alone does not establish invalidation. |
| Alex observation, operand recall or physical composition | Focused Alex component tests at actual Huet occurrences, including affected scopes and supplied evidence. Verify/lower the resulting MLIR with the real tools. Component fixtures establish only the supplied settled forms. |
| Executable semantics or backend preservation | Compile the relevant source cases and execute the fresh artifacts. Assert values, effects, evaluation order, call counts, storage behavior or failure behavior as the contract requires. Inspect the generated artifact when representation or operation preservation is part of the claim. |
| Runner selection, concurrency, isolation, timeout or result reporting | Focused .NET harness tests plus a small real compiler/native cohort exercising the changed runner path. Harness success alone is not language acceptance. |

These rows are selected by impact, not a checklist to run in full after every
edit. A Baker change with native consequences needs both its owning source/graph
test and a source-to-native check. An editor-only correction need not rerun every
native sample. A documentation-only change records link/content validation and
does not claim new execution evidence.

After a focused failure, fix the owning stage and rerun the affected selection.
When a shared compiler type or contract changes, rebuild its consumers with
project references enabled before acceptance. A stale F# union consumer can
misread a new case tag even when both assemblies load successfully. A
`--no-build` test run is evidence only for the coordinated assemblies it loads.
Expand to neighboring controls when a shared ingredient, recipe, graph relation,
capture/storage representation, witness or backend transformation changes.
Choose controls that distinguish the risk: direct and indirect calls, captured
and capture-free values, repeated enumeration, empty and nonempty input, nested
scopes, or the relevant full platform versus CompilerSurface startup path.
Preserve affected F behavior during C work and affected C behavior during F work.
An unexpected control failure expands the investigation; it is not removed from
the reported selection.

## Practical focused commands

Run from the Composer root and coordinate builds that share Composer or CCS
outputs. These are examples for sequence application and callable-occurrence
work, not a mandatory bundle for every change:

```bash
# Source admission and settled graph checks for the selected family.
dotnet test ../clef/tests/Clef.Compiler.Service.Tests/Clef.Compiler.Service.Tests.fsproj \
  --filter 'FullyQualifiedName~SequenceApplicationCases'

# Actual-occurrence operand recall and verified physical forms.
dotnet test tests/Alex.Tests/Alex.Tests.fsproj \
  --filter 'FullyQualifiedName~LambdaOccurrenceTests'

# The editor projection/invalidation checks for this source family.
dotnet run --project tests/CCS.Editor.Tests/CCS.Editor.Tests.fsproj \
  -- --sequence-applications

# Fresh source-to-native behavior with a bounded independent control.
dotnet fsi tests/regression/Runner.fsx -- \
  --sample 16h_SequenceApplications --sample 16a_SequenceOperations \
  --jobs 2 --results /tmp/composer-focused-checks
```

Confirm that test filters discover the intended tests and sample filters select
the intended manifest entries. The sample runner uses substring matching and
rejects unmatched filters; test-framework filters also need a nonzero discovered
count. A green empty selection provides no evidence.

For changed proof transport, run the applicable solver/integration cohort, for
example `dotnet run --project ../lattice-analyzers/tests/Lattice.CCS.Integration/Lattice.CCS.Integration.fsproj`.
The editor also exposes `--direct-captures`, `--closure-environments`,
`--loop-obligations`, `--call-effects`, `--program-lifetime` and
`--string-encoding`; choose the changed contract. Its default invocation runs
the projection checks together with source-input, visibility and editing checks. The
[Alex README](../tests/Alex.Tests/README.md) defines component evidence limits;
the [runner README](../tests/regression/README.md) documents harness checks and
native case selection.

## When to run the broader gates

Run the full relevant regression suites at a major integration checkpoint and
before claiming a whole C-xx PRD complete. For the Linux compiler pathway this
includes the unfiltered CCS and Alex suites and the full sample manifest, plus
the editor, proof and specialized native cohorts required by the delivered
contracts. Extend to the owning target gates when making claims for that target.
Run a broad gate earlier if a cross-cutting change cannot be bounded to known
consumers, focused results contradict the expected dependency boundary, or
neighboring controls expose effects beyond the selected family.

```bash
dotnet test ../clef/tests/Clef.Compiler.Service.Tests/Clef.Compiler.Service.Tests.fsproj
dotnet test tests/Alex.Tests/Alex.Tests.fsproj
dotnet fsi tests/regression/Runner.fsx -- --jobs 4 --results /tmp/composer-delivery-checks
```

Those commands are the common broad baseline, not a replacement for the PRD's
remaining editor/proof/native/target gates. A full manifest run reports its
skipped entries as well as failures; it cannot close promised behavior omitted
or skipped by that manifest. A full C-xx closure reconciles every promised
operation and composition against actual evidence and preserves the relevant
cross-family controls. Passing selected samples cannot stand in for that review.

## Concurrency, evidence and honest outcomes

Use the runner's `--jobs N` for independent compiler processes and then native
processes, bounded by available CPU and memory. This does not authorize concurrent
compiler requests within one process. Coordinate shared builds; the runner's
cooperative build/snapshot lock does not cover unrelated manual builds. Samples
that write shared runtime resources need serialization or their own isolation.
When resource contention is suspected, a serial rerun can distinguish it from
the semantic failure; retain both results and state any timeout changes.

Keep the printed run directory: selection, `run.json`, compiler snapshot hashes,
ordered `results.json`, per-job streams/status, binaries and retained `-k`
artifacts identify what was checked. For tests outside this runner, record the
command, selected cases/counts, source/build revisions, platform selection and
relevant logs/artifacts. Record the scope and result at completed
[coverage checkpoints](Language_Coverage_Waypoints.md), including known failures
and checks not run. Old results remain dated evidence, not a fresh run.

Never silently weaken an oracle to obtain green results: removing a failing
case, adding a skip, changing expected behavior, relaxing proof requirements,
substituting a lighter platform or accepting a nonzero native exit requires an
explicit rationale grounded in the owning contract and a recorded scope change.
Correct rejection of valid conforming source remains an implementation gap. A
targeted pass advances only its stated step; an unresolved required gate keeps
the PRD open.
