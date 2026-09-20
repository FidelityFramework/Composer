# Original sequence operations coverage

This is the original C-07 source fixture, repackaged as Clef source with the
full default platform profile (including Format) platform dependency. Its executable expressions, ordering,
printed checks and expected numeric values are unchanged. Comments now refer to
the shared sequence and closure contracts instead of prescribing inline code
pointers, entire-environment copies or operation-specific field layouts.

The fixture retains coverage beyond the focused 16a–16c oracles:

- Module-level formed sequences and eager consumers.
- The parameterized `range`/`naturals` factories and their original mutable
  `while` recurrence.
- Functions receiving sequence values and returning transformed sequences,
  including callbacks capturing those functions' parameters.
- `collect` callbacks returning variable-length child sequences and capturing
  caller values.
- Formatter output and the original manual-consumer comparisons, empty inputs,
  capture combinations and deep pipelines.

These paths have not passed the coordinated native gate. Packaging does not
establish their origin, callback, frame-range or lifetime premises, and the
passing focused fixtures do not replace this coverage. No factory, recurrence
or top-level expression has been rewritten into an easier accepted form.

```sh
dotnet run --project tests/NativeSequences/NativeSequences.Tests.fsproj -- src/bin/Debug/net10.0/Composer --sample 16_SeqOperations
```

The harness requires fresh compilation, stock MLIR verification, native exit
zero and exact `ExpectedOutput.txt` (only CRLF is normalized). The expected file
retains the final space printed by each original multi-element print loop.
The regression manifest denotes those spaces with `\u0020`; its parsed text is
identical. Expected values remain those in the original verification summary,
including even-square sum 220 and deep-pipeline sum 400. This migration itself
has not run the native fixture.
