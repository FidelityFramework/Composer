# Direct LLVM/LLD backend

Composer's ELF backend passes its witnessed MLIR through LLVM lowering, prepares target bitcode, and invokes LLD directly:

```text
MLIR -> mlir-opt -> mlir-translate -> LLVM IR
     -> opt -mtriple=<target> -passes=no-op-module -> LLVM bitcode
     -> ld.lld --lto-O0 --lto-CGO0 -> ELF
```

LLVM's TargetMachine supplies a missing data layout from the selected target during bitcode preparation. An explicit conflicting IR target is rejected. The preparation pass verifies and serializes IR without an optimization pipeline. Native Linux builds retain host CPU selection through LLD; cross builds use the selected target's default CPU. LLD then invokes LLVM code generation internally and performs symbol resolution, relocations and section/segment layout. Composer does not invoke a C compiler or a separate `llc` in this backend. Use matching LLVM `opt` and `ld.lld` versions; `composer doctor` checks their availability.

`-k` retains the `.bc` handoff beside `08_output.ll`. Arguments are passed as individual process arguments, including paths containing spaces. Missing tools, undefined symbols and missing entry points fail the build; there is no alternative compiler-driver fallback.

## Runtime and layout inputs

The console deployment retains its existing hosted runtime behavior. Native Linux discovers installed `crt1.o`, `crti.o`, `crtn.o`, libc and the runtime loader in target library directories. These are already-built runtime inputs; their use does not require a C compiler. Freestanding and embedded modes supply their own `_start` and add no implicit libc or startup objects. Shared-library mode uses `--shared` and the libraries requested by binding resolution.

The CLI exposes direct link inputs:

| Option | Meaning |
| --- | --- |
| `--sysroot PATH` | Target runtime root; default library paths are rooted here. |
| `--link-library-path PATH` | Additional target library directory; repeatable. |
| `--link-start-file PATH` | Object preceding the program bitcode; repeatable, in order. Providing these replaces automatic startup discovery. |
| `--link-end-file PATH` | Object following the libraries; repeatable, in order. |
| `--dynamic-linker PATH` | Loader path recorded in the ELF, expressed as a path on the target, without a sysroot prefix. |
| `--linker-script PATH` | LLD script defining section placement, regions and assertions. |

Cross-target console builds require a sysroot or explicit startup inputs and never implicitly search host runtime directories. Runtime discovery is a Linux convenience profile. The current direct backend emits ELF; PE/COFF, Mach-O and Wasm need their own LLD link profiles and are diagnosed explicitly. An embedded board's startup/vector objects and linker script remain required target inputs; a successful ELF link alone does not establish board bootability. These CLI link inputs are not yet automatically projected from Fidelity.Platform declarations or persisted as a new `.fidproj` schema.

## Proof preservation and regression gates

BAREWire's pool offsets remain compiler-owned facts. LLD controls where that pool is placed in the final image. The existing emission correspondence gate and source/native obligations remain in force. Supplying a linker script does not itself prove that every platform constraint is enforced; the artifact must satisfy the corresponding checks.

`dotnet fsi tests/LLDBackendRegression.fsx` tests hosted execution, shared-library calls, libc-free freestanding execution, an ARM cross-link with scripted placement, paths containing spaces, and failure cases. Executable traps reject any attempt to invoke Clang, GCC or `llc` during the tests.

After compiling HelloDimensionsProof with `-k`, `python3 tests/StaticStorageNativeRegression.py <sample>/targets` checks the exact pool bytes and alignment in the ELF, read-only `PT_LOAD` permissions, both proof dispatches, and program output. This gate passes with the direct LLD backend.
