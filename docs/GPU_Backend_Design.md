# GPU Backend Design

> **Status**: Implemented (August 2026) — `src/BackEnd/GPU/{Lowering,Pipeline}.fs`
> **Substrate**: AMD GPU via ROCDL (validated on gfx1151, Radeon 8060S / Strix Halo)
> **Replaces**: the `| GPU -> Error "GPU backend not yet implemented."` stub

## What the leg does

Takes the portable MLIR that Alex emits, assembles a `gpu.module` around the
kernel's call-graph closure, and lowers it through ROCDL to an AMD code
object:

```
Clef  ──CCS/PSG/nanopasses──▶  Alex  ──▶  portable MLIR
                                              │  (func, scf, arith, memref, index)
                                              ▼
                                       BackEnd/GPU/Lowering.fs
                                              │
        reachability ─▶ wrap ─▶ mlir-opt --convert-gpu-to-rocdl
                                       ─▶ mlir-opt --gpu-module-to-binary
                                              ▼
                                        <output>.hsaco
```

Five steps, one external tool:

1. **reachability** — device call-graph closure from the kernel entry
2. **wrap** — closure plus a generated `gpu.func` inside a chip-targeted `gpu.module`
3. **mlir-opt** — `--convert-scf-to-cf --convert-gpu-to-rocdl`
4. **mlir-opt** — `--gpu-module-to-binary`
5. **extract** — lift the object out of the `#gpu.object<…>` attribute into a `.hsaco`

It is materially simpler than the AIE leg: one tool instead of seven, no
`bootgen`/`xclbinutil` packaging, no per-core ELF patching, and the emitted
text is checkable by the same binary that consumes it.

## Where it sits, and why the middle end defers

The MiddleEnd commits to nothing. Alex emits portable dialects for a GPU
target exactly as it does for every other target, and the commitment to
AMDGPU happens in the leg. This is not merely layering hygiene — it is the
same discipline the language applies to evaluation.

Clef's deferred computation is **call-by-need**, and the reason that matters
here is not deferral but *sharing*: a thunk is a settled value whose identity
exists before its content does. Because the computation has not run, **its
placement is still open**. Something eagerly evaluated has already run
somewhere, and all that remains is to copy the result.

A middle end that committed to a target would be the eager case. Alex holds
the program as a settled, unforced structure in portable dialects, and the
leg is where forcing happens — where the placement decision is finally
discharged. Committing to a target discards information, so the commitment
belongs at the last possible moment, in exactly one place per substrate.

This is also why witnesses stay platform-agnostic while *patterns* observe
the `TargetPlatform` coeffect: the walk observes rather than forces, which is
the compiler-internal statement of the same rule.

## No Python

`BackEnd/AIE/Lowering.fs` opens with *"Invokes the AIE toolchain natively (no
Python)"* — MLIR-AIE ships `aiecc.py` as its driver and the AIE leg
deliberately bypasses it. The GPU leg holds the same line: F# driving
`mlir-opt` through `System.Diagnostics.Process`, the same `runTool` shape,
the same `Result` railway.

A wrapper script that manufactures MLIR sits *in the lowering path* while
being invisible to every discipline the compiler enforces: it is not a
witness, it observes no coeffects, it produces no elision, and nothing
type-checks it. Once one exists, the cheapest way to add the next target is a
second script. A bring-up harness written during development was deleted
rather than kept as scaffolding, for that reason.

## The kernel contract

The application supplies the compute function; the toolchain supplies the
data movement — the same division of labour as the NPU leg.

```fsharp
/// i = global index, x = input[i], n = element count
let kernel (i: int) (x: int) (n: int) : int
```

The leg generates the `gpu.func` that reads `input[i]`, calls this, and
writes `output[i]`. **Nothing on the Clef side names a buffer**, and that is
precisely why it lowers: a `nativeptr` parameter arrives as a bare `index`
carrying neither extent nor address space, and the ROCDL pipeline cannot
legalise it — measured at 17 unresolved `index → memref` casts on the first
attempt. In the generated wrapper the buffers are genuine `memref` values the
toolchain can place and size. Those 17 casts are the finiteness lemma of
`Closure_Nanopass_Architecture.md` Section 4 read operationally: a bare index
carries no extent for any judgment or any legaliser to close over, which is
why the spec demotes `nativeptr` to internal `TNativePtr` plumbing and
length-carried `memref` takes its place at every surface.

Kernel entry is located by trailing dotted segment (`FIDELITY_GPU_KERNEL`,
default `kernel`); ambiguity and absence are both diagnostics that list the
candidates found. Chip via `FIDELITY_GPU_CHIP`, default `gfx1151`.

### Device tree shaking

The emitted module carries the whole host program — entry point, console I/O,
FFI shims. `reachableFrom` walks `func.call` edges from the kernel entry and
keeps only that closure, so nothing host-side reaches the device object. FFI
declarations have no body and are skipped by construction.

## Validation

A Clef kernel compiled through this leg and executed on the iGPU:

```
Backend:  GPU
  GPU: gfx1151 kernel entry 'kernel' -> TestKernel.hsaco
GPU code object generated: targets/TestKernel.hsaco
```

```
Machine: EM_AMDGPU        Flags: 0x4a, gfx1151
clef_kernel               GLOBAL PROTECTED FUNC     (dispatch entry)
clef_kernel.kd            OBJECT                    (kernel descriptor)
TestKernel.kernel         GLOBAL FUNC               (the Clef function)
clef_kernel.num_vgpr = 5
```

Loaded via `hipModuleLoad`/`hipModuleGetFunction` and launched over 16
elements, the results matched the Clef definition exactly. The same source
compiled to a CPU binary through the LLVM leg produced the matching scalar
answer — one definition, two substrates.

## Wiring

Five sites, no MiddleEnd changes:

| Site | Change |
|---|---|
| `src/BackEnd/GPU/Lowering.fs` | new |
| `src/BackEnd/GPU/Pipeline.fs` | new — the `BackEnd` record value |
| `src/Core/Types/Pipeline.fs` | `BackEndArtifact.GpuCodeObject of path` |
| `src/Core/PlatformPipeline.fs` | `\| GPU -> BackEnd.GPU.Pipeline.backend` |
| `src/Composer.fsproj` | two `<Compile Include>` lines after the AIE pair |
| `src/Core/CompilationOrchestrator.fs` | one artifact reporting arm |

## Known limits and next steps

**The contract is elementwise.** A gather kernel — every thread reading a
shared table, as a rasteriser does — cannot be expressed, because the Clef
side would have to index a buffer and `array` is not denotable in a
signature (`Unknown type constructor: array`). Note the standing art: an
`arrayTyCon` with `NTUKind.NTUarray` and `TypeLayout.FatPointer` already
exists, every `Array.*` intrinsic is typed against it, and array locals
already elide to `memref.alloc`/`load`/`store`. Three gaps stand between that
and gather kernels:

1. `"array"` is absent from `tryResolveBuiltinTypeConstructor`, and
   `resolveSynType` has no `SynType.Array` case
2. `Array.length` is typed but has no elision — the enclosing function emits
   `func.return` with zero operands against a declared result
3. array-typed *parameters* through Baker/PSG are unproven: SSA
   pre-assignment for a fat pointer, and the calling convention

On (3) the recommendation is that `array<'T>` pass as a single `memref`
value rather than `{ptr, len}` scalars, because a memref descriptor already
carries the extent — which is exactly what lets the same parameter cross to
a device kernel unchanged. That would make the array the substrate-crossing
construct and leave the thunk host-side, where its memo cell belongs: many
device threads forcing one thunk is a race, and there is no host runtime on
the device to run the forcing code.

**Other targets.** `memref` is portable, so array elision needs no
target-specific witness. The one nuance is CIRCT: `hw` has no `memref`, so
the FPGA leg must commit arrays to BRAM or registers. That is leg work, not
witness work.

**Not yet plumbed.** The chip is env-driven because `fidproj` has no
per-target section; `[platform].device` is parsed into `PlatformSection` but
dropped by clef's `ProjectChecker`, so no chip string reaches a backend
today. Threading it through `setupContext` is the principled fix.
