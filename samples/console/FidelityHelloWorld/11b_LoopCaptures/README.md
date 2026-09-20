# Loop captures

Status: registered; native compilation, zero exit and exact output pass.

This variant belongs beside `11_Closures` and `11a_DirectCaptures`. It checks the
immutable per-iteration source-binding contract adopted in specification
`25e2954`: saved callbacks retain distinct values from ascending, descending and
integer range loops. Nested loops with the same identifier retain distinct
bindings. Local direct functions capture the current iteration, while a separately
mutable captured cell remains shared after the loop.

The source uses ordinary local function slots and Platform's CompilerSurface.
It does not prescribe closure storage or establish lifetime proof discharge.
The companion `tests/NativeCallbacks/LoopCaptures.clef` has six focused groups;
this sample combines the same expectations into four groups (exit codes 1–4).

Successful execution must exit zero and print exactly:

```text
=== Loop captures Test ===
Counted iteration snapshots: pass
Range iteration snapshots: pass
Nested binding identities: pass
Direct calls and shared captures: pass
```

The companion native fixture also passes stock MLIR verification. Compiler,
native and exact-output gate evidence is recorded in the
[waypoint ledger](../../../../docs/Language_Coverage_Waypoints.md).
