# Sequence semantics

This C-06 oracle consumes the generated state machines and checks their actual
values before printing any pass lines. Exit code 1 reports a semantic failure.
It covers literal yields, repeated and nested independent enumeration, delayed
pre/post-yield effects, shared mutable captures, conditional loop suspension,
empty effects, counted yields, caller-owned factories, repeated delegation,
retained child captures, factories inside generators, and chained empty inputs.
The decimal trace cells in deferred children have explicit finite bounds.

Types and range information belong to the compiler graph. The native frames
contain unboxed state, values and retained storage descriptors; the descriptors
carry addresses, offsets, extents and strides, with no runtime type objects.
Baker establishes layout and residence before Alex observes those relationships.
The native result complements the graph and proof tests; execution alone does
not establish every frame layout, lifetime or resumption obligation.

The project uses CompilerSurface and fixed Console output. Its companion runner
is [NativeSequences](../../../../tests/NativeSequences/README.md), which requires
fresh compilation, stock MLIR verification, native exit zero and exact output
from `ExpectedOutput.txt`. Current validation is recorded in the
[language coverage waypoints](../../../../docs/Language_Coverage_Waypoints.md).

The original `15_SimpleSeq` remains a separate acceptance oracle. The current
full-profile run rejects its accumulating generators because their frame values
lack settled observable ranges (including triangular, Fibonacci and power
recurrences). Its formatter/profile dependencies remain separate from this
CompilerSurface oracle. A pass here must not be reported as a pass of the
original sample. See `/tmp/composer-c06-final-regression.log` for that run.
