# Result elimination

Status: registered; native compilation, zero exit and exact output pass.

This variant exercises `Result.defaultValue`, `Result.defaultWith` and
`Result.iter` under specification `82cd330`'s
[native Result contracts](../../../../../clef-lang-spec/spec/error-handling.md#native-result-operations).
Default values are eager; recovery receives the actual Error payload only on
Error; actions consume the Ok payload only on Ok. Source and pipe order, stored
fallback/callback snapshots, shared captures, independent measured cases, unit
results and failure propagation remain observable.

The project uses Platform's CompilerSurface and fixed Console text. Five groups
have distinct failure codes 1–5. The companion
`tests/NativeCallbacks/ResultElimination.clef` covers 15 groups, including explicit
generics, callable/record/unit Error payloads, preserved callable success payloads
and extra application after the two-operand default operation boundary. Neither
oracle prescribes an aggregate layout or proves closure lifetime obligations.

Successful execution must exit zero and print exactly:

```text
=== Result elimination Test ===
Success values and error recovery: pass
Factory and pipe order: pass
Stored defaults and shared captures: pass
Independent measured cases: pass
Unit actions and failure propagation: pass
```

The companion native fixture also passes stock MLIR verification. Compiler and
exact-output gates are recorded in the
[waypoint ledger](../../../../docs/Language_Coverage_Waypoints.md).
