# Optional sequence search

This C-07 oracle checks `Seq.tryHead` and `Seq.tryPick` through consumed values
and observable demand. Fixed output appears only after all five groups pass;
failures return distinct codes 181–185.

Coverage includes a first value without a second pull, an effectful empty body,
one chooser invocation per current until the first `Some`, complete exhaustion
when no value matches, eager chooser/input factories in source order, measured
integer input with an independent inverse measured real result, and preservation
of `Some None` as a selected nested Option value. Per-element flags detect
duplicate callback invocations without imposing a numeric counter bound.

Native validation is pending the coordinated compiler build and runner gate.
The fixture uses the CompilerSurface platform dependency and supplements
`16_SeqOperations`; it does not establish the nonempty contracts of `head`,
`min` or `max`.

```sh
dotnet run --project tests/NativeSequences/NativeSequences.Tests.fsproj -- src/bin/Debug/net10.0/Composer --sample 16e_SequenceSearch
```
