# Sequence composition and demand

This C-07 oracle preserves the original sample's even-square sum **220** and
five-stage filter/map/filter/take/fold sum **400**. It also checks uncaptured maps,
map/filter ordering, first-three demand, multiple immutable callback captures,
effectful empty input, singleton mapping and all-rejected input.

Every stage records its actual invocations. The deep pipeline stops after source
value 50: 50 source pulls, 50 first predicates, ten maps, ten second predicates
and five folder calls. Only 49 source post-yield effects have run. Enumerating
the same formed pipeline again restarts progress while retaining the original
shared counter cells. Bounded counter arithmetic keeps the observation code's
integer requirements explicit; it does not weaken the expected counts.

Run the native sequence harness with `--sample 16c_SequenceComposition`.
Acceptance requires fresh compilation, stock MLIR verification, native exit zero
and exact `ExpectedOutput.txt`. No completed result is implied by registration.
This fixture does not cover `collect`, opaque callback pairs or escaping callback
environments; those have separate representation and residence requirements.
