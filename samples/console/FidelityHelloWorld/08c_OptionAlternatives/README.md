# Optional alternatives

This FidelityHello variant exercises `Option.orElse` and `Option.orElseWith`:
eager optional operands, deferred thunk invocation, pipe order, stored snapshots,
shared mutable captures, nested optional results, record/function payloads and
measured values with inverse dimensions. Capturing function values use ordinary
anonymous functions. The project depends on Platform's CompilerSurface.

```sh
dotnet fsi tests/regression/Runner.fsx -- --sample 08c_OptionAlternatives
```

Run from Composer. The regression manifest checks exit status and all seven output
lines. The companion `tests/NativeCallbacks/OptionAlternatives.clef` covers the
larger branch/effect matrix. The [waypoint record](../../../../docs/Language_Coverage_Waypoints.md)
pins the compiler revision, native results and companion tooling checks.
