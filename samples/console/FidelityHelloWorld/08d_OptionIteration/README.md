# Optional actions

This FidelityHello variant exercises `Option.iter`: `Some` invokes its action once,
`None` skips it, and forming an action retains source evaluation order. It also
checks stored action identity, shared mutable captures, measured payloads with
inverse dimensions, and unit results that are bound, consumed or discarded.
The project depends on Platform's CompilerSurface.

```sh
dotnet fsi tests/regression/Runner.fsx -- --sample 08d_OptionIteration
```

Run from Composer; this variant is registered in the regression manifest.
The six fixed output lines accompany a zero exit status. The companion
`tests/NativeCallbacks/OptionIteration.clef` covers additional branch, generic
record and function-payload cases. Native verification is pending the compiler
implementation; adding this oracle does not assert that verification has passed.
