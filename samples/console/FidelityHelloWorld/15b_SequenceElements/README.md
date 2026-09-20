# Sequence element values

This companion to `15a_SequenceSemantics` checks native consumption of `seq<bool>`,
`seq<unit>`, `seq<float>`, `seq<int<distance>>`, and `seq<float<1/duration>>`.
Two unit yields remain observable through their pull count and ordered effects;
unit does not become an empty sequence. Real checks use exactly representable
binary fractions, including negative values. Inverse measures have integral
exponents; fractional numeric values do not imply fractional measure admission.

The five groups validate actual values and exit 71–75 on failure before printing
fixed Console lines. CompilerSurface avoids the unrelated `Format.int` migration.
Native acceptance requires compilation, stock MLIR verification, exit zero and
exact `ExpectedOutput.txt` through `tests/NativeSequences`. That gate passed for
all five groups with CCS SHA-256
`5DAD941BBEB27341242D826D1D4D0CC935B039A7CF60FA31EA3A81494D240F44`.
The companion waypoint records subsequent final regression evidence.
This oracle adds scalar coverage; it does not establish
aggregate payload support or replace graph layout and lifetime obligations.
