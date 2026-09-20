# Result callbacks

Status: registered in the regression manifest; compilation, native zero exit and
exact six-line output pass at the Result callback waypoint.

This variant exercises `Result.map`, `Result.mapError` and `Result.bind` under the
[Result operation contracts](../../../../../clef-lang-spec/spec/error-handling.md).
Callbacks run only for their selected case. Supplied operands retain source and
pipe evaluation order; partials snapshot callback values while captured mutable
cells remain shared. Success and error payloads retain independent types and
dimensions, including inverse measures.

An untouched payload is preserved. The enclosing typed case may be reconstructed
when the resulting type or layout changes; the oracle assumes no container
identity, byte size or storage class. Pattern matching observes each result.
The project uses Platform's CompilerSurface and fixed Console text.

Successful execution must exit zero and print exactly:

```text
=== Result callbacks Test ===
Selected cases and preserved payloads: pass
Factory and pipe order: pass
Stored callbacks and shared captures: pass
Independent measured cases: pass
Explicit failure propagation: pass
```

The companion `tests/NativeCallbacks/ResultCallbacks.clef` has 12 groups (failure
codes 191–202), including all three bare aliases and partials, explicit generic
argument order, record/function/unit payloads, and success/failure pipelines.
Both fixtures pass native execution; the callback runner additionally verifies
the retained MLIR module with stock `mlir-opt --verify-each`.
