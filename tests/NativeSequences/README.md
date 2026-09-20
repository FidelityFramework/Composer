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
