# Sequence operation semantics

This C-07 oracle checks append, ordered eager consumers, short-circuit predicates
and demand-limited production through Clef source and the complete native pipeline.
Its fixed output is printed only after the consumed values and effects agree.

Coverage includes eager operand formation in source order, effectful empty
prefixes/suffixes, repeated enumeration, mutable callback captures, independent
fold state/payload types, sum/product, inverse measured real state over measured
integer elements, and the distinct stopping polarities of `exists` and `forall`.
`take` checks positive, nonpositive, shorter-input and repeated-use boundaries.
Both callback counts and upstream effects are observed, including the absence
of a pull after a deciding predicate or exhausted demand.

Run from Composer after a coordinated compiler build:

```sh
dotnet run --project tests/NativeSequences/NativeSequences.Tests.fsproj -- src/bin/Debug/net10.0/Composer --sample 16a_SequenceOperations
```

The runner requires fresh compilation, stock MLIR verification, native exit zero
and exact `ExpectedOutput.txt`. Its evidence records the compiler artifacts.
This oracle supplements `16_SeqOperations`; callback-producing transformations
and the broader C-07 exit gates remain tracked in the PRD and language waypoints.
