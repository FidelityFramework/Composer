# Finite additive sequence state

This native oracle consumes additive source recurrences whose finite bounds
follow from the actual monotone guard, induction update and
accumulator stores. It checks triangular values `1,3,6,10,15,21`, repeated
enumeration, negative and mixed deltas, zero iterations, and ascending and
descending strides of two. No accumulator inside these generators is clamped
or reduced modulo a bound. The observer's count and decimal trace use explicit
moduli so instrumentation has its own independent finite range.

The runner requires successful compilation, stock `mlir-opt --verify-each`,
native exit zero, and exact `ExpectedOutput.txt`. It records the compiler hashes
and retained artifacts. Registration alone is not a passing result.

The original formatter-dependent `15_SimpleSeq` remains separate and unchanged.
Coupled Fibonacci state and multiplicative powers of two require their own
recurrence evidence; this additive oracle does not close those gates.
