# Callback environments across suspension

This C-07 source oracle exercises mapper aliases, immutable and mutable captures,
repeat enumeration, filter/map/take demand and repeated callback formation in a
loop. It checks actual upstream callback counts and values before printing its
fixed output. A shared mutable capture must retain its original storage; callbacks
formed by separate loop iterations must carry the values of their own formations.

Run with the native sequence harness and `--sample 16b_SequenceCallbacks`.
Fresh compilation, stock MLIR verification, native exit zero and exact output are
required. The language waypoint records implementation and test status; this
fixture does not substitute for unknown-callee or escaping-environment admission.
