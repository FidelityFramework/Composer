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

## Required diagnostics

Alex retains required checks as portable `cf.assert` operations with their specified diagnostics. The backend receives both the typed operations and their portable serialization. Before LLVM lowering, [RequirementRealization](../src/BackEnd/LLVM/RequirementRealization.fs) realizes each typed assertion for the selected process runtime and writes a separate `*.runtime.mlir` input. The portable artifact remains intact. This target pass does not traverse the source graph or recover operations by parsing MLIR text.

The admitted process profile requires checked `os = linux`, `runtime_model = libc`, the 64-bit Pointer dimension, console or library deployment, and an AMD64 Linux GNU/musl target ABI. These platform selections come from the [Linux profile](../../Fidelity.Platform/Profiles/Linux_x86_64_Default/Fidelity.Platform.fidproj). The [Linux environment](../../Fidelity.Platform/Environments/Linux/x86_64/Environment.clef) declares write syscall 1, exit-group syscall 231, negative errno returns, and process teardown; [Console](../../Fidelity.Platform/Environments/Linux/x86_64/Console.clef) declares stderr descriptor 2. The backend profile supplies the matching AMD64 syscall register convention. A target without an admitted diagnostic and termination capability is rejected when an assertion requires one; a target triple alone cannot grant that capability.

The success edge performs no diagnostic IO. The failure edge writes the exact UTF-8 message plus a newline to stderr using an explicit byte length, retaining embedded NUL bytes. It continues after short writes, retries interrupted writes, and terminates the process with exit status 1 through `exit_group`. An unrecoverable diagnostic IO error still terminates; it cannot turn a failed requirement into successful execution. Diagnostic storage is immutable, generated symbols avoid existing typed and opaque symbol spellings, and the syscall operations retain side effects and memory clobbers through lowering. No libc buffering, C compiler driver, optional assertion flag, or release-build removal participates in this path.

[LLVMRequirementTests](../tests/Alex.Tests/LLVMRequirementTests.fs) verifies capability retraction, original condition/order preservation, symbol collisions, stock MLIR verification and lowering, and actual native execution. The native cases require silent success and exact failure bytes, including quotes, newlines, Unicode, embedded NUL, and a diagnostic larger than a pipe buffer.

These runtime diagnostic globals and helper functions add image storage beyond Baker's `StaticStringPool`. The pool's correspondence and capacity evidence covers exactly its source string members; it does not cover the whole `.rodata` section or the generated helper code. The selected [Linux compatibility budgets](../../Fidelity.Platform/Profiles/Linux_x86_64_Default/CompatibilityBudgets.clef) declare finite `.rodata` and `.text` capacities, with [no general exemption for a hosted process](../../Fidelity.Platform/Profiles/Linux_x86_64_Default/README.md). Whole-image acceptance must account for generated diagnostics, helper code, alignment, and linker contributions against the selected resource declarations and validate the resulting artifact. The large-diagnostic regression establishes diagnostic transport and termination behavior; it is not acceptance against that profile's section budgets.

## Proof preservation and regression gates

BAREWire's pool offsets remain compiler-owned facts. LLD controls where that pool is placed in the final image. The existing emission correspondence gate and source/native obligations remain in force. Supplying a linker script does not itself prove that every platform constraint is enforced; the artifact must satisfy the corresponding checks.

`dotnet fsi tests/LLDBackendRegression.fsx` tests hosted execution, shared-library calls, libc-free freestanding execution, an ARM cross-link with scripted placement, paths containing spaces, and failure cases. Executable traps reject any attempt to invoke Clang, GCC or `llc` during the tests.

After compiling HelloDimensionsProof with `-k`, `python3 tests/StaticStorageNativeRegression.py <sample>/targets` checks the exact pool bytes and alignment in the ELF, read-only `PT_LOAD` permissions, both proof dispatches, and program output. This gate passes with the direct LLD backend.
