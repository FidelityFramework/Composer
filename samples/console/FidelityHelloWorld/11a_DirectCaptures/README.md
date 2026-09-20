# Direct named captures

This FidelityHello variant exercises immutable direct capture elaboration through
Baker and Alex: repeated calls, recursive forwarding, returned anonymous closures,
bounded array descriptors and inverse dimensions. The regression manifest checks
its native exit status and exact output.

```sh
dotnet fsi tests/regression/Runner.fsx -- --sample 11a_DirectCaptures
```

Run from Composer. The companion `tests/NativeCallbacks/DirectCaptures.clef` adds
shadowing, records, captured function values, nested named declarations and ordered
argument effects, then requires stock MLIR verification as well as native execution.
The older `11_Closures` remains a separate oracle with its full Platform dependency.
