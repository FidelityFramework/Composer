# Composer AIE Backend Design -- HelloNappy

## Scope

Minimum viable backend for compiling a Clef kernel to an XDNA2 xclbin via the
MLIR-AIE toolchain. The goal is end-to-end: Clef source in, xclbin + insts.bin
out, NPU runs the kernel, host collects the result.

Constraints for this iteration:
- No async, no referential transparency analysis, no auto-parallelism
- Tile count is statically declared by the programmer (4 tiles for hello world)
- Single operation: element-wise multiply on int32 buffers (64 samples, 16 per tile)

## Design Principles

### Pure Functions as the Primary Abstraction

The kernel source follows the same principle as HelloArty's FPGA pattern: the
programmer writes a *pure function*, and the compiler generates the spatial
coordination infrastructure. In HelloArty, `Design<'State>.Step` is a pure
function from state to state; the witness generates `hw.module`, `seq.compreg`,
clock domains, and reset logic. The programmer never writes `always_ff`.

For NPU, the `Compute` field of an `ElementKernel<'T>` is a pure binary
function from `'T -> 'T -> 'T`. The witness generates `aie.device`, `aie.tile`,
`aie.core`, `aie.objectfifo`, acquire/release synchronization, and the DMA
runtime sequence. The programmer never writes acquire/release or manages FIFOs.

The lower strata (MLIR-AIE with its imperative coordination protocol) is
generated infrastructure, not authored surface syntax.

### Relationship to Python IRON

The IRON framework establishes four concepts for kernel authoring:

| IRON Concept          | Clef Equivalent                                    |
|-----------------------|----------------------------------------------------|
| Core body function    | `Compute`: a pure function `'T -> 'T -> 'T`        |
| ObjectFifo            | Generated from `Shape.Grain` (one set per tile)     |
| Worker                | Generated from `Compute` (wrapped in `aie.core`)    |
| Runtime / sequence    | Generated from `Shape.Elements` and tile count       |
| TensorAccessPattern   | Generated from `Elements / Grain` partitioning       |

The Clef programmer does not interact with any of the right column's generated
artifacts. They write the left column's pure function and a shape descriptor.

### ML-Family Idioms

The kernel source uses ML-family conventions:
- The compute function is curried and composable
- The kernel descriptor is a record value (a binding, not a procedure)
- Type inference flows from the `Compute` field's signature to the element type
- No mutable state in the kernel source; all mutation is in the lowered MLIR

### Dimensional Types (Future)

The `Shape` record currently uses bare integers. When Clef acquires dimensional
types or units of measure, these become typed quantities:

```fsharp
// Future: compile-time checked divisibility
Shape = { Elements = 64<samples>; Grain = 16<samples> }
```

The type system can then enforce `Elements mod Grain = 0` at compile time.
The current design uses a record structure that accommodates this evolution
without breaking changes.

## Kernel Source Pattern

### ElementKernel

```fsharp
module HelloNappy.Kernel

/// The computation: a pure binary function.
/// This is the entire algorithmic content of the kernel.
/// In the ML sense, this is map2 over paired buffers.
let multiply (a: int32) (b: int32) : int32 = a * b

/// Shape descriptor: how many elements total, how many per tile.
/// The compiler derives tile count as Elements / Grain.
type Shape = {
    Elements: int       // total buffer length
    Grain: int          // elements per compute tile
}

/// ElementKernel<'T>: apply a binary function element-wise
/// over two input buffers of 'T, producing an output buffer of 'T.
///
/// Analogous to Design<'State> for FPGA:
///   Design<'S>         = { InitialState: 'S; Step: 'S -> 'S }
///   ElementKernel<'T>  = { Compute: 'T -> 'T -> 'T; Shape: Shape }
///
/// The Compute field is a pure function. The witness lowers it
/// to arith/memref/scf ops inside each aie.core, surrounded by
/// the object FIFO acquire/release protocol.
type ElementKernel<'T> = {
    Compute: 'T -> 'T -> 'T
    Shape: Shape
}

[<KernelModule>]
let emul : ElementKernel<int32> = {
    Compute = multiply
    Shape = { Elements = 64; Grain = 16 }
}
```

The witness extracts from the `ElementKernel<int32>` record:
- `Shape.Elements` = 64, `Shape.Grain` = 16 => 4 tiles
- `Compute` = `multiply` => lowered to `arith.muli %a, %b : i32`
- `'T` = `int32` => `memref<16xi32>` object FIFO element type

### Future Kernel Types

The `ElementKernel` pattern generalizes. Each kernel type implies a different
tiling and data-movement strategy:

```fsharp
// Unary map: buffer -> buffer
type MapKernel<'A, 'B> = {
    Compute: 'A -> 'B
    Shape: Shape
}

// Reduction: buffer -> scalar
type ReductionKernel<'T, 'Acc> = {
    Init: 'Acc
    Accumulate: 'Acc -> 'T -> 'Acc
    Shape: Shape
}

// Stencil: windowed access pattern
type StencilKernel<'T, 'N> = {
    Compute: Buffer<'T, 'N> -> 'T    // 'N is window size
    Shape: Shape
    Halo: int                        // overlap between tiles
}
```

Each would have its own witness strategy for generating the appropriate
MLIR-AIE patterns (different object FIFO depths, accumulator chains for
reduction, halo exchange for stencils). For hello world, only
`ElementKernel` is implemented.

## Platform Attributes and Target Routing

### `[<KernelModule>]` as a Backend-Agnostic Attribute

The `[<KernelModule>]` attribute marks a binding as the root of a compute kernel
dispatched to an accelerator. It is intentionally backend-agnostic: the same
attribute serves both NPU and (future) GPU targets. The backend selection is
determined by the fidproj's `target=` field, not by the attribute itself.

```
Attribute         DeclRoot          Backend routing (via fidproj target)
─────────────────────────────────────────────────────────────────────────
[<EntryPoint>]    EntryPoint        target=cpu  → LLVM backend → ELF
[<HardwareModule>] HardwareModule   target=fpga → CIRCT backend → Verilog
[<KernelModule>]  KernelModule      target=npu  → AIE backend → xclbin
[<KernelModule>]  KernelModule      target=gpu  → GPU backend (future)
```

The WitnessRegistry gates which witness runs per platform:
- `target=npu` registers `KernelModuleWitness` (emits MLIR-AIE)
- `target=gpu` would register `GPUKernelWitness` (emits GPU dialect)

Both witnesses match `DeclRoot.KernelModule`. The source stays portable; the
compilation project carries the target configuration.

This mirrors how `[<HardwareModule>]` works today: the attribute is structural
("this is a hardware design root"), and `HardwareModuleWitness` only activates
when `targetPlatform = FPGA`.

### Host + Kernel Split

A heterogeneous application has two compilation units:

| Unit       | fidproj target | Attribute        | Produces           |
|------------|---------------|------------------|--------------------|
| Host       | cpu           | [<EntryPoint>]   | Native ELF binary  |
| Kernel     | npu           | [<KernelModule>] | xclbin + insts.bin |

The host links XRT and loads the kernel artifact at runtime. The kernel fidproj
produces a standalone xclbin; it does not link against any host libraries.

## Architecture

### Compilation Pipeline

```
Clef kernel source (.clef)
    |
    v
[FrontEnd] CCS: .clef -> PSG
    |
    v
[MiddleEnd] Alex: PSG -> MLIR-AIE dialect
    |  KernelModuleWitness extracts the compute body
    |  and generates aie.device/tile/core/objectfifo/runtime_sequence
    |
    v
MLIR-AIE text (aie.device, aie.core with arith+memref+scf body)
    |
    v
[BackEnd] AIE: aiecc.py drives the rest
    |  aie-opt passes (objectfifo transform, lock/BD assignment, buffer addressing)
    |  Peano (clang++ --target=aie2p): core MLIR -> ELF per tile
    |  bootgen: CDO -> PDI
    |  xclbinutil: package xclbin
    |
    v
emul.xclbin + insts.bin
```

### Comparison with Other Backends

| Aspect           | CPU (LLVM)              | FPGA (CIRCT)            | NPU (AIE)                  |
|------------------|-------------------------|-------------------------|----------------------------|
| DeclRoot         | EntryPoint              | HardwareModule          | KernelModule               |
| Attribute        | [<EntryPoint>]          | [<HardwareModule>]      | [<KernelModule>]           |
| Witness          | (none special)          | HardwareModuleWitness   | KernelModuleWitness        |
| Source pattern   | main : int              | Design<'S>              | ElementKernel<'T>          |
| Primary field    | (function body)         | Step: 'S -> 'S          | Compute: 'T -> 'T -> 'T   |
| Metadata fields  | (none)                  | InitialState: 'S        | Shape: Shape               |
| Alex emits       | func/memref/arith/scf   | hw/comb/seq             | aie.*/arith/memref/scf     |
| Backend tool     | mlir-opt + llc + clang  | circt-opt               | aiecc.py (aie-opt + Peano) |
| Output artifact  | NativeBinary (ELF)      | Verilog (.sv)           | Xclbin (.xclbin + .bin)    |

### MLIR-AIE Output (Generated by KernelModuleWitness)

The witness generates the full MLIR-AIE module. The `Compute` function body
is lowered to vanilla arith/memref/scf ops inside each `aie.core` body.
The spatial infrastructure (tiles, object FIFOs, DMA) is generated from the
`Shape` descriptor.

For HelloNappy with 4 tiles, 16 elements per tile, `multiply`:

```mlir
module {
  aie.device(npu2) {
    // Shim tiles (one per column used)
    %shim_0 = aie.tile(0, 0)
    %shim_1 = aie.tile(1, 0)
    %shim_2 = aie.tile(2, 0)
    %shim_3 = aie.tile(3, 0)

    // Compute tiles (row 2, one per column)
    %tile_0 = aie.tile(0, 2)
    %tile_1 = aie.tile(1, 2)
    %tile_2 = aie.tile(2, 2)
    %tile_3 = aie.tile(3, 2)

    // Object FIFOs: per-tile input/output channels
    // Depth 2 for double-buffering (compute overlaps with DMA)
    aie.objectfifo @in1_0(%shim_0, {%tile_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2_0(%shim_0, {%tile_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_0(%tile_0, {%shim_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.objectfifo @in1_1(%shim_1, {%tile_1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2_1(%shim_1, {%tile_1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_1(%tile_1, {%shim_1}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.objectfifo @in1_2(%shim_2, {%tile_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2_2(%shim_2, {%tile_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_2(%tile_2, {%shim_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    aie.objectfifo @in1_3(%shim_3, {%tile_3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @in2_3(%shim_3, {%tile_3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @out_3(%tile_3, {%shim_3}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // Compute core (tile 0) -- replicated per tile with different FIFO names
    %core_0 = aie.core(%tile_0) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c_inf = arith.constant 9223372036854775807 : index

      scf.for %iter = %c0 to %c_inf step %c1 {
        %sub_a = aie.objectfifo.acquire @in1_0(Consume, 1)
            : !aie.objectfifosubview<memref<16xi32>>
        %buf_a = aie.objectfifo.subview.access %sub_a[0]
            : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

        %sub_b = aie.objectfifo.acquire @in2_0(Consume, 1)
            : !aie.objectfifosubview<memref<16xi32>>
        %buf_b = aie.objectfifo.subview.access %sub_b[0]
            : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

        %sub_c = aie.objectfifo.acquire @out_0(Produce, 1)
            : !aie.objectfifosubview<memref<16xi32>>
        %buf_c = aie.objectfifo.subview.access %sub_c[0]
            : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

        // ---- Lowered from: Compute = multiply = fun a b -> a * b ----
        scf.for %i = %c0 to %c16 step %c1 {
          %a = memref.load %buf_a[%i] : memref<16xi32>
          %b = memref.load %buf_b[%i] : memref<16xi32>
          %r = arith.muli %a, %b : i32
          memref.store %r, %buf_c[%i] : memref<16xi32>
        }

        aie.objectfifo.release @in1_0(Consume, 1)
        aie.objectfifo.release @in2_0(Consume, 1)
        aie.objectfifo.release @out_0(Produce, 1)
      }
      aie.end
    }

    // Cores 1-3: identical structure, different FIFO names
    // (generated by the witness in a loop over tile index)

    %core_1 = aie.core(%tile_1) {
      // ... same body with @in1_1, @in2_1, @out_1
      aie.end
    }
    %core_2 = aie.core(%tile_2) {
      // ... same body with @in1_2, @in2_2, @out_2
      aie.end
    }
    %core_3 = aie.core(%tile_3) {
      // ... same body with @in1_3, @in2_3, @out_3
      aie.end
    }

    // Runtime sequence: host-side DMA configuration
    // Partitions the 64-element host buffers into 4x16 chunks,
    // one per tile, using strided DMA descriptors.
    aie.runtime_sequence(
        %A: memref<64xi32>,
        %B: memref<64xi32>,
        %C: memref<64xi32>) {

      // DMA tasks for tile 0: elements [0..15]
      %t0_a = aiex.dma_configure_task_for @in1_0 {
        aie.dma_bd(%A : memref<64xi32>, 0, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0_a)

      %t0_b = aiex.dma_configure_task_for @in2_0 {
        aie.dma_bd(%B : memref<64xi32>, 0, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0_b)

      // DMA tasks for tile 1: elements [16..31]  (offset = 16)
      %t1_a = aiex.dma_configure_task_for @in1_1 {
        aie.dma_bd(%A : memref<64xi32>, 16, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1_a)

      %t1_b = aiex.dma_configure_task_for @in2_1 {
        aie.dma_bd(%B : memref<64xi32>, 16, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1_b)

      // DMA tasks for tile 2: elements [32..47]  (offset = 32)
      %t2_a = aiex.dma_configure_task_for @in1_2 {
        aie.dma_bd(%A : memref<64xi32>, 32, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2_a)

      %t2_b = aiex.dma_configure_task_for @in2_2 {
        aie.dma_bd(%B : memref<64xi32>, 32, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2_b)

      // DMA tasks for tile 3: elements [48..63]  (offset = 48)
      %t3_a = aiex.dma_configure_task_for @in1_3 {
        aie.dma_bd(%A : memref<64xi32>, 48, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3_a)

      %t3_b = aiex.dma_configure_task_for @in2_3 {
        aie.dma_bd(%B : memref<64xi32>, 48, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3_b)

      // Output DMA: collect results from all 4 tiles
      %t0_c = aiex.dma_configure_task_for @out_0 {
        aie.dma_bd(%C : memref<64xi32>, 0, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0_c)

      %t1_c = aiex.dma_configure_task_for @out_1 {
        aie.dma_bd(%C : memref<64xi32>, 16, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1_c)

      %t2_c = aiex.dma_configure_task_for @out_2 {
        aie.dma_bd(%C : memref<64xi32>, 32, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t2_c)

      %t3_c = aiex.dma_configure_task_for @out_3 {
        aie.dma_bd(%C : memref<64xi32>, 48, 16,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 16, stride = 1>])
          {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t3_c)

      // Wait for all output transfers, then free input tasks
      aiex.dma_await_task(%t0_c)
      aiex.dma_await_task(%t1_c)
      aiex.dma_await_task(%t2_c)
      aiex.dma_await_task(%t3_c)
      aiex.dma_free_task(%t0_a)
      aiex.dma_free_task(%t0_b)
      aiex.dma_free_task(%t1_a)
      aiex.dma_free_task(%t1_b)
      aiex.dma_free_task(%t2_a)
      aiex.dma_free_task(%t2_b)
      aiex.dma_free_task(%t3_a)
      aiex.dma_free_task(%t3_b)
    }
  }
}
```

## Compiler Changes

### 1. DeclRoot extension

**File:** `clef/src/Compiler/PSGSaturation/SemanticGraph/Types.fs`

```fsharp
type DeclRoot =
    | EntryPoint
    | HardwareModule
    | KernelModule      // NPU kernel
```

### 2. Attribute detection

**File:** `clef/src/Compiler/NativeTypedTree/Expressions/Types.fs`

Add `hasKernelModuleAttribute` following the same pattern as
`hasHardwareModuleAttribute`. Detects `[<KernelModule>]` on bindings.

### 3. KernelModuleWitness (new)

**File:** `Composer/src/MiddleEnd/Alex/Witnesses/KernelModuleWitness.fs`

Registered in WitnessRegistry when `targetPlatform = NPU`, inserted before
BindingWitness (same pattern as HardwareModuleWitness for FPGA).

Matches `SemanticKind.Binding(_, _, _, Some DeclRoot.KernelModule)`.

Extracts from the `ElementKernel<'T>` record:
- `Shape.Elements` (literal int) and `Shape.Grain` (literal int)
- Derives tile count as `Elements / Grain`
- `Compute` (VarRef to pure function; Lambda body lowered to arith ops)
- `'T` (element type from record type parameter)

Generates complete MLIR-AIE module text:
- `aie.device(npu2)` wrapper
- Per-tile: shim tile, compute tile, 3 object FIFOs, `aie.core` with lowered
  compute body
- `aie.runtime_sequence` with per-tile DMA descriptors (strided, offset by
  `tileIndex * Grain`)

The compute body lowering walks the `Compute` function's expression tree
and emits the corresponding arith/memref/scf ops inside the `aie.core` region.
For `multiply = fun a b -> a * b` this produces `arith.muli`. For more complex
expressions, the full expression tree is lowered (additions, shifts, comparisons,
nested arithmetic).

### 4. BackEndArtifact extension

**File:** `Composer/src/Core/Types/Pipeline.fs`

```fsharp
type BackEndArtifact =
    | NativeBinary of path: string
    | Verilog of path: string
    | Xclbin of xclbinPath: string * instsPath: string
    | IntermediateOnly of format: string
```

### 5. AIE Backend (new)

**Files:**
- `Composer/src/BackEnd/AIE/Pipeline.fs`
- `Composer/src/BackEnd/AIE/Lowering.fs`

Pipeline:
1. Write MLIR-AIE text to file
2. Invoke aiecc.py with appropriate flags
3. Return Xclbin artifact

Lowering:
- Resolves tool paths from `AIE_TOOLCHAIN` env var or `~/aie-toolchain`
- Sets `PEANO_INSTALL_DIR` for Peano cross-compilation
- Runs: `aiecc.py --aie-generate-xclbin --aie-generate-npu-insts
  --no-compile-host --no-xchesscc --no-xbridge
  --xclbin-name=<output>.xclbin --npu-insts-name=<output>.bin <input>.mlir`

### 6. PlatformPipeline update

**File:** `Composer/src/Core/PlatformPipeline.fs`

```fsharp
| NPU -> BackEnd.AIE.Pipeline.backend
```

### 7. Orchestrator update

**File:** `Composer/src/Core/CompilationOrchestrator.fs`

Add `Xclbin` case to artifact match:

```fsharp
| Xclbin (xclbinPath, instsPath) ->
    printfn "Xclbin generated: %s" xclbinPath
    printfn "NPU instructions: %s" instsPath
    Ok ()
```

## Host-Side Changes (HelloNappy)

The host program (Program.clef) needs to be updated for the MLIR_AIE kernel
dispatch interface. The xclbin metadata shows the kernel interface:

```xml
<kernel name="MLIR_AIE" type="dpu" dpu_kernel_id="0x901">
  <arg name="opcode"  offset="0x00" type="uint64_t"/>
  <arg name="instr"   offset="0x08" type="char *"/>
  <arg name="ninstr"  offset="0x10" type="uint32_t"/>
  <arg name="bo0"     offset="0x14" type="void*"/>
  <arg name="bo1"     offset="0x1c" type="void*"/>
  <arg name="bo2"     offset="0x24" type="void*"/>
</kernel>
```

Host dispatch sequence:
1. Load xclbin (`xrtDeviceLoadXclbinFile`)
2. Load `insts.bin` from disk into a buffer
3. Allocate instruction BO, map it, copy instruction data
4. Allocate data BOs for A, B, C
5. Open kernel `"MLIR_AIE"` (not `"emul"`)
6. Set args: opcode=0, instr BO, ninstr, bo_A, bo_B, bo_C
7. Sync input BOs to device
8. Run and wait
9. Sync output BO from device
10. Read results

## Project Structure

Two fidproj files:
- `HelloNappy/HelloNappy.fidproj` -- host program (target=cpu), links xrt_coreutil
- `HelloNappy/kernel/HelloNappyKernel.fidproj` -- kernel (target=npu)

Build order: kernel first (produces xclbin + insts.bin), then host.

## Toolchain Dependencies

- `aie-toolchain` venv at `/home/hhh/aie-toolchain/`
  - Peano (llvm-aie 19.0.0): `~/aie-toolchain/lib/python3.12/site-packages/llvm-aie/`
  - mlir-aie (0.0.1.2026020504): aie-opt, aie-translate, aiecc.py, bootgen
  - xclbinutil: system-installed (`/usr/bin/xclbinutil`)
- Environment variables (resolved by Lowering.fs):
  - `AIE_TOOLCHAIN` -- path to aie-toolchain venv (default: `~/aie-toolchain`)
  - `PEANO_INSTALL_DIR` -- path to llvm-aie (derived from pip show)
  - `MLIR_AIE_INSTALL_DIR` -- path to mlir_aie package (derived from pip show)
