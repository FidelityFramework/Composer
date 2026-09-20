# Native sequence acceptance

This standalone runner compiles the permanent
[15a_SequenceSemantics](../../samples/console/FidelityHelloWorld/15a_SequenceSemantics/README.md)
oracle by default using the existing Composer binary. The optional
`--sample 15b_SequenceElements` selects the companion scalar element oracle. It has no compiler project reference.
It copies the source and project into a fresh temporary directory and uses the
shared process runner for concurrent stdout/stderr drainage, timeouts and process
tree termination.

After a coordinated compiler build:

```sh
dotnet run --project tests/NativeSequences/NativeSequences.Tests.fsproj -- src/bin/Debug/net10.0/Composer
dotnet run --project tests/NativeSequences/NativeSequences.Tests.fsproj -- src/bin/Debug/net10.0/Composer --sample 15b_SequenceElements
```

Acceptance requires successful compilation, retained MLIR accepted by stock
`mlir-opt --verify-each`, native exit zero and exact `ExpectedOutput.txt` text
(only CRLF is normalized). The source checks the actual consumed values before
printing pass lines. A compile/verification failure, nonzero native exit,
timeout or output mismatch fails the runner.

Each run retains source, project, expected output, logs, MLIR and compiler hashes
with `evidence.json`. A stage exit of -1 means it did not finish or was not run.
The semantic groups exercise literals, captures, conditional and loop suspension,
empty inputs, factories, delegation and independent enumerations. It does not substitute for the original
formatter-dependent `15_SimpleSeq`, or for upstream frame/lifetime proof gates.

The [15b_SequenceElements](../../samples/console/FidelityHelloWorld/15b_SequenceElements/README.md)
selection checks boolean order, two observable unit yields, real fractions,
measured integers and inverse measured reals. Unknown sample names fail before
compilation. Fractional real values do not claim fractional measure exponents.

`--sample 15c_SequenceTemplateBorrows` checks surrounding-scope captured
templates, repeated deferred delegation, shared mutable source cells and nested
capture levels. The covering activation must outlive every template use;
escaping or unknown inputs require additional evidence.

`--sample 15d_SequenceAdditive` checks finite additive recurrences whose guard,
induction step and actual stores supply joint range and proof evidence. It covers
triangular, negative and mixed state, zero trips, and both directions with
non-unit steps. It leaves the original Fibonacci and powers-of-two gates visible.

`--sample 16a_SequenceOperations` selects the C-07 composition/consumer oracle:
append operand and empty-input effects, ordered `iter`/`fold`, independent fold
state types, short-circuit `exists`/`forall`, and `take` demand boundaries.
Its source and expected output remain separate from the C-06 oracles.

`--sample 16b_SequenceCallbacks` checks materialized callback environments across
suspension: mapper aliases, mutable capture identity, repeated formation and
filter/map/take demand. Neither C-07 fixture's presence implies a passing gate;
the runner retains the actual compilation, verification and execution results.

`--sample 16c_SequenceComposition` checks the original even-square sum 220 and
deep composition sum 400, both map/filter orders, empty/singleton/rejected inputs,
multiple captures and per-stage counts. It checks that a downstream limit prevents
the next upstream post-yield effect, including when the formed pipeline is
enumerated again. `collect` and escaping callback environments remain separate
acceptance areas.

`--sample 16_SeqOperations` selects the original source coverage, now packaged as
`.clef` with the full default profile. Its factory functions, module-level initialization,
while recurrences and expected values are unchanged and remain a separate,
unpassed gate. Its exact output retains the spaces emitted by the original print
loops; see the [sample README](../../samples/console/FidelityHelloWorld/16_SeqOperations/README.md).

`--sample 16d_SequenceCollect` checks returned child capture initializers, shared
mutable effects, variable-length and empty inner sequences, repeated enumeration
and a limit reached inside a child. The retained evidence records whether the
compiler satisfies both residence and exact native demand behavior.

`--sample 16e_SequenceSearch` selects optional first-value/chooser search, empty
effects, eager operand factories, distinct input/result dimensions and nested
Option selection. Registration does not claim a passing native result.

`--sample 16f_SequenceOptions` checks scalar-payload Option transport: retained
values after later pulls and exhaustion, repeated construction at one loop site,
independent enumerators, re-enumeration and delegation. Its manual consumers
observe integer, boolean and inverse measured real payloads without introducing
aggregate callback captures. The five source groups use failure codes 191–195;
registration alone does not establish native acceptance.

`--sample 16g_SequenceStartup` selects two ordered source files with multiple
namespace/module declarations, an effectful unreferenced initializer and a
module sequence. It checks once-only startup, deferred pulls, repeated direct
and named-function enumeration, and caller effects after all initializers.
The runner copies both sources; the full default platform profile supplies
the declared program-lifetime storage authority. The source remains a strict
oracle while startup implementation and native validation are pending.
