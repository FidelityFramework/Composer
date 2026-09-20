# Option sequence elements

This oracle checks that scalar-payload Option elements remain values across
subsequent pulls, exhaustion, repeated construction at one loop site, independent
enumerators, re-enumeration and delegation. Integer, boolean and inverse measured
real payloads are observed through ordinary `for` consumption. Stored Option
values belong to the consuming activation; this fixture introduces no aggregate
callback captures.

The five groups return failure codes 191–195. Fixed output is emitted only after
every value and ordering check succeeds. Native validation is pending the
coordinated compiler build and retained runner evidence. This supplements
`16e_SequenceSearch`, whose source remains unchanged.

```sh
dotnet run --project tests/NativeSequences/NativeSequences.Tests.fsproj -- src/bin/Debug/net10.0/Composer --sample 16f_SequenceOptions
```
