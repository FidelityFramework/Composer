# Native callback and capture gates

Run `python3 tests/NativeCallbacks/run.py src/bin/Debug/net10.0/Composer` from
the Composer checkout. Each case compiles a fresh LLVM/LLD executable and runs
it with a timeout. The gates cover named entries, bounded array descriptors,
mutable cells, records, captured function values, alias snapshots and function
fields followed by later allocations. `IgnoreValues` checks that discarding
ordinary and optional opaque-handle values preserves evaluation effects and
produces a usable Clef unit value.

`ListenerEntry` consumes the same `CallbackDescriptor` vocabulary emitted by
Farscape for native listener fields. It checks an ordinary Clef unit call, a
separate C `void` entry, field aliases and another address of the same handler,
the full unsigned 32-bit argument range, and negative signed 32-bit results.
The source retains its logical unit result; the native entry thunk discards it.
Declared scalar representations also govern indirect-call arguments and results.

The focused listener gate passed as a fresh native executable on 2026-09-09
using Composer Debug build 28. Retained local evidence is
`/tmp/clef-listener-entry-tvbgj__e`: `build.log`, `run.log`, the executable and
`targets/intermediates/10_output.mlir`. These temporary artifacts may expire.

`IgnoreValues` passed on Debug build 32 as a fresh native executable in
`/tmp/clef-ignore-values-n3yh6t9n`. Its effect counts and three unit consumers
passed; the retained MLIR contains both effect-producing calls.

Listener declaration provenance currently follows record fields, ordinary
aliases and repeated `FnPtr.ofFunction` addresses of the declared entry. An
ABI-specialized pointer transported through an unrelated, unannotated `FnPtr`
parameter does not yet carry that declaration. This gate does not establish
that higher-order transport. Native entries require named module functions
without captures; foreign context remains an explicit opaque handle. Ordinary
Clef closures retain their code/environment pairs and bounded capture views.
