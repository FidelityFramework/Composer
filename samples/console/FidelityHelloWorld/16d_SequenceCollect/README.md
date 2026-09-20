# Sequence collect oracle

C-07 acceptance fixture for callback-produced child sequences. It checks exact
outer/inner order, repeated enumeration, a captured multiplier, variable inner
lengths, effectful empty children and downstream demand stopping inside a child.

Expected values are independent of the compiler implementation. Exit codes
91–95 identify the failing semantic group. Admission, stock MLIR verification
and native execution are required; fixture presence alone is not a passing gate.
