# Typed foreign reference gate

Run `python3 tests/ForeignReferences/run.py src/bin/Debug/net10.0/Composer` from
Composer. The gate compiles fresh native executables through LLVM/LLD.

`PointerCells` uses the real `posix_memalign`/`free` boundary. It checks initial
`None`, an unsuccessful call that preserves the output, successful `Some` copy-back,
and an immutable option alias that survives replacement of the array cell.
The native executable runs with both ordinary and perturbed allocator contents.
Retained LLVM must use byte alignment for packed option payloads. A separate
empty output array must take the extent rejection branch before the native call.

These checks passed with Composer Debug build 32 on 2026-09-09. They exercise
source options and call-scoped native pointer cells; they do not establish general
option reclamation or arbitrary foreign record mutation. Rich record projection
currently requires a read-only reference or supported by-value contract; a
writable record needs identical native storage or an explicit copy-back adapter.
Multiple pointer output references are rejected until their aliasing is preserved.
