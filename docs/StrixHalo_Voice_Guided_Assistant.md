# Strix Halo Voice-Guided Assistant

## Status: Active Development (Updated 2026-03-10)

This document describes a local MoE inference system with voice I/O, compiled entirely through the Fidelity/Composer stack. No llama.cpp, no vLLM, no Python runtime. The long-term vision is a complete inference pipeline in Clef source compiled to native code via MLIR.

The design principle is complete hardware utilization. Every processor on the die does meaningful inference work: the CPU runs a ternary routing model that makes hardware-aware dispatch decisions in microseconds, the NPU runs quantized experts at 50 TOPS, and the GPU runs dense language models at full FP16 precision. The ternary router is not a decision tree; it is a small, fast model whose job is to make the agentic flow responsive to what each processor does best.

### Current Reality vs. Vision

This document now distinguishes between **what exists** and **what is planned**. The 2026 deliverable is a pragmatic audio agent using existing inference runtimes (ONNX Runtime) on proven hardware, not the full native-compiled MoE pipeline. The native pipeline remains the north star but requires ~25 more Composer PRDs to reach.

---

## Hardware Target

AMD Strix Halo APU (ASUS ROG Flow Z13 GZ302EA) with unified memory:

| Processor | Role | Capability | Driver Status |
|-----------|------|------------|---------------|
| **Zen 5 CPU** | MoE routing, TTS synthesis, orchestration | AVX-512, 32 threads @ 5.2 GHz | **Working** — fully operational |
| **XDNA 2 NPU** | ASR inference, ternary expert inference | 50 TOPS INT8, AIE2P, 48 tiles | **Working** — `amdxdna-dkms` 7.0, `/dev/accel/accel0` |
| **RDNA 3.5 GPU** | Dense LLM inference, attention-heavy computation | gfx1151, Radeon 8060S, FP16/FP32 | **Working** — ROCm 7.2, HIP operational |
| **Unified LPDDR5X** | Shared memory for all processors | 64 GB, zero-copy between all three | **Available** — DMA-BUF interop possible |

All three compute substrates are proven operational on this machine as of 2026-03-10.

### Driver Stack Details

| Component | Package | Version | Notes |
|-----------|---------|---------|-------|
| NPU kernel driver | `amdxdna-dkms` | 7.0 (drm-misc-fixes-02-26-27) | DKMS from superm1 (AMD), replaces broken in-kernel 6.19 driver |
| NPU firmware | `linux-firmware` | 20260221-1 | `amdnpu/17f0_11/npu.sbin` (rev 11 for Strix Halo) |
| NPU userspace | `xrt` + `xrt-plugin-amdxdna` | 2.21.75 | XRT runtime; `xrt-smi` has mmap API skew with DKMS driver (non-blocking) |
| GPU kernel driver | `amdgpu` (in-kernel) | 6.19.6-arch1-1 | Mainline, stable |
| GPU userspace | ROCm 7.2 | 7.2.0 | hipBLAS, rocBLAS, rocFFT, RCCL installed |
| MLIR-AIE reference | `~/repos/mlir-aie/` | — | Xilinx MLIR-AIE dialect for NPU kernel compilation |

All models reside in unified memory. No copies between processors. BAREWire zero-copy semantics map directly to the hardware topology.

---

## Architecture

### Conversational Pipeline

The pipeline is sequential by nature of conversation, but every stage can float across substrates based on load, latency constraints, and what the driver stack supports. The diagram below shows the logical flow; substrate assignment is a scheduling decision, not a structural one.

```
Microphone
    │
    ▼
┌──────────────────────────────────────────────────────┐
│  Audio Capture (PipeWire)                            │
└──────────┬───────────────────────────────────────────┘
           │ PCM frames (unified memory)
           ▼
┌──────────────────────────────────────────────────────┐
│  Mel Spectrogram                                     │  FFT + mel filterbank
│  preferred: CPU │ possible: NPU tiles                │
└──────────┬───────────────────────────────────────────┘
           │ mel features (unified memory)
           ▼
┌──────────────────────────────────────────────────────┐
│  ASR (Whisper)                                       │  INT8 quantized
│  preferred: NPU │ possible: GPU, CPU                 │
└──────────┬───────────────────────────────────────────┘
           │ text tokens (unified memory)
           ▼
┌──────────────────────────────────────────────────────┐
│  BitNet Categorical Router                           │  1.58-bit, L2-resident
│  always: CPU (AVX-512)                               │  dispatches to expert + substrate
└──────────┬───────────────────────────────────────────┘
           │ routing decision: expert ID + target substrate
           ├───────────────────────┐
           ▼                       ▼
┌────────────────────┐  ┌────────────────────────┐
│  Ternary Expert    │  │  Dense LLM             │
│  preferred: NPU    │  │  preferred: GPU (FP16) │
│  possible: CPU     │  │  possible: CPU         │
└────────┬───────────┘  └──────────┬─────────────┘
         │                         │
         └──────────┬──────────────┘
                    │ response tokens (streaming, unified memory)
                    ▼
┌──────────────────────────────────────────────────────┐
│  TTS + Vocoder (VITS/Piper + HiFi-GAN)              │  mel gen + waveform synthesis
│  preferred: CPU │ possible: NPU tiles (when idle)    │
└──────────┬───────────────────────────────────────────┘
           │ PCM audio
           ▼
┌──────────────────────────────────────────────────────┐
│  Audio Output (PipeWire)                             │
└──────────────────────────────────────────────────────┘
           │
           ▼
        Speaker
```

Every stage except the BitNet router has substrate flexibility. Unified memory means there is no copy penalty for moving work between processors — the scheduling decision is purely about latency, throughput, and what's currently occupied.

### Substrate Roles

The three substrates are not rigidly assigned; they have affinities based on what each does well. The system adapts based on what's available and what the query demands.

**CPU (Zen 5, AVX-512, 32 threads)** — The router's permanent home and the default substrate for stages that aren't latency-critical or don't benefit from accelerator throughput. The BitNet categorical router is always CPU because it must be L2-cache-resident for microsecond dispatch decisions. Its job extends beyond MoE expert selection: it provides categorical reach into dense models on the GPU, deciding whether a query needs a ternary expert (fast, power-efficient, NPU) or a dense model (higher quality, attention-heavy, GPU). Mel spectrogram extraction and TTS synthesis default to CPU but can migrate when accelerators are idle.

**NPU (XDNA 2, 32 compute tiles, 58 TOPS)** — The power-efficient workhorse for quantized inference. Whisper ASR is its primary resident workload. Ternary experts dispatch here when the router selects them. The tile grid (8 columns × 6 rows) supports **concurrent spatial partitioning** — verified via DRM UAPI on 2026-03-10. Each `CREATE_HWCTX` claims a column partition; multiple contexts coexist on disjoint partitions with driver-managed migration and preemption. A 10 TOPS STT context + a 10 TOPS TTS context leaves ~38 TOPS for additional workloads. The supervisor queries live capacity (`QUERY_RESOURCE_INFO` → `npu_tops_curr`) before hydrating new contexts, and REALTIME-priority contexts (Whisper) can preempt NORMAL-priority experts. For lightweight queries, the NPU alone can handle the entire ASR → expert → TTS path while the GPU stays available for dense work. See `Farscape/docs/roadmap/02_farscape-phase4-npu-xrt-binding.md` Section 11 for the full spatial scheduling architecture.

**GPU (RDNA 3.5, gfx1151, FP16/FP32)** — The throughput substrate for attention-heavy computation. Dense LLMs with full-precision attention and large KV caches run here. This is not a fallback from NPU; dense models are first-class workloads that the router dispatches to the GPU because that's where they run best. The GPU also serves as overflow capacity for any stage when the NPU is saturated.

### Scheduling Dynamics

The substrate preferences above represent steady-state operation. The interesting cases are the transitions:

- **During active expert inference**: If both NPU and GPU are occupied with expert/LLM work, TTS runs on CPU (overlap — the user hears audio while inference continues). This is the common case during long responses.
- **Short response completed**: NPU and GPU finish quickly. TTS migrates to idle NPU tiles for lower-latency synthesis. CPU is freed for the next mel extraction or router evaluation.
- **Multi-expert activation**: Router selects both a ternary expert (NPU) and a dense LLM (GPU) for different aspects of the response. Both run concurrently; results merge in the token stream. This is the hardware utilization sweet spot.
- **Cold start / first query**: All models load from mmap'd weight files on NVMe. Always-resident models (Whisper, router, TTS) are warm after first query. Expert/LLM loading uses `madvise` prefetch on routing decisions.

The pipeline adapts rather than prescribes. A rigid "stage X always runs on substrate Y" model leaves capacity on the table and creates bottlenecks when assumptions break down.

### Model Inventory

| Model | Task | Parameters | Size | Preferred | Can Also Run On | Residency |
|-------|------|-----------|------|-----------|-----------------|-----------|
| Whisper small | ASR | 244M | ~244 MB (INT8) | NPU | GPU, CPU | Always resident |
| BitNet router | MoE dispatch + categorical reach | 10-50M | ~5-25 MB (1.58-bit) | CPU (L2) | — | Always resident |
| Language expert | General text | 100-500M | ~50-250 MB (1.58-bit) | NPU | CPU | Hot-loadable |
| Code expert | Programming tasks | 100-500M | ~50-250 MB (1.58-bit) | NPU | CPU | Hot-loadable |
| Reasoning expert | Logic, planning | 200-800M | ~100-400 MB (1.58-bit) | NPU | CPU | Hot-loadable |
| General LLM (dense) | Open-ended generation | 1-3B | ~2-6 GB (INT8/FP16) | GPU | CPU (slow) | Hot-loadable |
| Long-context LLM (dense) | Multi-turn, summarization | 1-7B | ~2-14 GB (INT8/FP16) | GPU | CPU (slow) | Hot-loadable |
| VITS/Piper | TTS (mel gen) | 25-60M | ~25-60 MB | CPU | NPU (idle tiles) | Always resident |
| HiFi-GAN | Vocoder | 14M | ~14 MB | CPU | NPU (idle tiles) | Always resident |

Total always-resident: under 400 MB. With 64 GB unified memory (128 GB in higher configs), multiple experts and dense LLMs can be loaded simultaneously. Every model lives in the same physical memory regardless of which processor runs it — the scheduling decision has zero memory-copy cost.

---

## Expert Hot-Loading

### Memory-Mapped Weight Files

Expert weights are stored as memory-mapped files on NVMe. The OS virtual memory subsystem handles paging:

- **Resident expert**: weights are in physical memory. Inference starts immediately.
- **Non-resident expert**: weights are memory-mapped but paged out. First access triggers page-in from NVMe.
- **Prefetch on routing**: when the router selects an expert, issue `madvise(MADV_WILLNEED)` on that expert's weight file. NVMe starts reading while the current expert finishes.

### Page-In Latency

| Model Size | NVMe Read (14 GB/s PCIe 5.0) | Acceptable? |
|------------|-------------------------------|-------------|
| 50 MB (ternary expert) | ~4 ms | Yes |
| 250 MB (ternary expert) | ~18 ms | Yes |
| 500 MB (ternary expert) | ~36 ms | Marginal |
| 2 GB (dense LLM, GPU) | ~143 ms | Noticeable; use prefetch or keep resident |
| 7 GB (dense LLM, GPU) | ~500 ms | Keep resident; first-load only |

For ternary experts under 250 MB, page-in latency is imperceptible. The router's prediction accuracy determines the effective cache hit rate. If the router consistently selects the same 2-3 experts for a conversation, those experts stay resident after first use.

### Eviction Policy

Simple LRU by expert. When memory pressure requires eviction, drop the least-recently-used expert's pages. The OS handles this natively for memory-mapped files; the application just tracks access order for prefetch decisions.

---

## Actor Topology

Each pipeline stage is a `MailboxProcessor` actor. Actors are substrate-agnostic — they define *what* computation happens, not *where*. Substrate affinity is a runtime scheduling property carried as a coefficient, not baked into the actor's identity. Messages are BAREWire-serialized references to unified memory buffers (zero-copy regardless of which substrate produced or consumes them).

```fsharp
// Pipeline actors — substrate affinity is a scheduling hint, not a type constraint
let audioCapture = AudioCapture.spawn deviceId
let melExtractor = MelSpectrogram.spawn { sampleRate = 16000; fftSize = 400; hopLength = 160 }
let asrActor     = Inference.spawn { model = whisperModel; affinity = Prefer NPU }
let routerActor  = Inference.spawn { model = routerModel;  affinity = Require CPU }  // L2-resident
let expertPool   = ExpertPool.spawn expertConfigs    // manages hot-loading + substrate dispatch
let ttsActor     = Inference.spawn { model = ttsModel;     affinity = Prefer CPU }
let vocoderActor = Inference.spawn { model = vocoderModel;  affinity = Prefer CPU }
let audioOutput  = AudioOutput.spawn deviceId

// Wiring — types enforce data flow correctness; substrate is orthogonal
audioCapture  |> pipeTo melExtractor     // PCMFrame → MelFeatures
melExtractor  |> pipeTo asrActor         // MelFeatures → TokenSequence
asrActor      |> pipeTo routerActor      // TokenSequence → RoutingDecision
routerActor   |> pipeTo expertPool       // RoutingDecision → TokenStream (streaming)
expertPool    |> pipeTo ttsActor         // TokenStream → MelFrames (streaming)
ttsActor      |> pipeTo vocoderActor     // MelFrames → PCMFrame
vocoderActor  |> pipeTo audioOutput      // PCMFrame → speaker
```

The `Inference.spawn` actor accepts any model — the substrate it runs on is determined by the affinity hint and current system load. `Require` pins a stage to a substrate (the router must stay L2-resident on CPU). `Prefer` expresses an affinity that the scheduler can override when capacity shifts. The `ExpertPool` actor manages the additional complexity of hot-loading and substrate dispatch across the expert/LLM boundary.

### Type Safety Across the Pipeline

Each actor boundary has a distinct message type. The compiler rejects wiring errors. Substrate scheduling is invisible at this level — the type system enforces data flow, not hardware topology.

```fsharp
type PCMFrame       = { samples: float32 array; sampleRate: int }
type MelFeatures    = { frames: float32 array; numMels: int; numFrames: int }
type TokenSequence  = { tokens: int array; confidence: float32 }
type RoutingDecision = {
    expertId: ExpertId
    substrate: SubstratePreference   // Prefer NPU | Prefer GPU | Require CPU
    query: TokenSequence
    priority: Priority
}
type TokenStream    = IAsyncEnumerable<int>            // streaming, token-by-token
type MelFrames      = IAsyncEnumerable<float32 array>  // streaming, frame-by-frame
```

Attempting to pipe `MelFeatures` to `expertPool` is a compile-time error. The types document and enforce the data flow. Substrate placement is a scheduling concern resolved at runtime by the supervision tree, not a structural concern resolved at compile time.

### Streaming Response

The expert generates response tokens incrementally. Each token flows to TTS immediately:

```fsharp
// Expert inference streams tokens as they are generated
let! responseStream = expertPool.InferStreaming(routingDecision)

// TTS processes tokens as they arrive (not waiting for full response)
responseStream
|> AsyncSeq.bufferByCount 4          // accumulate a few tokens for natural prosody
|> AsyncSeq.map ttsActor.Synthesize  // mel generation per phrase
|> AsyncSeq.iter vocoderActor.Vocalize  // waveform per phrase → speaker
```

The user hears the beginning of the response while the expert is still generating the end. The pipeline latency from first generated token to first audible output is: TTS inference time (~20-40ms for a short phrase on CPU) plus vocoder time (~5-10ms). Under 50ms from token to sound.

---

## What This Replaces

The architecture evolves in stages. The 2026 pragmatic path already replaces the glue and orchestration layers; the 2027+ native path replaces the inference runtimes themselves.

| Conventional Stack | 2026 (Pragmatic) | 2027+ (Native) |
|-------------------|------------------|-----------------|
| whisper.cpp (C++) | ONNX Runtime on NPU via Clef bindings | Whisper compiled from Clef via MLIR-AIE |
| llama.cpp / vLLM (C++/Python) | ONNX Runtime on NPU/GPU via Clef bindings | Ternary + dense inference compiled from Clef via MLIR |
| Python glue scripts | Clef function pipeline | Clef actor pipeline, compile-time wired |
| GGML/GGUF format | ONNX format (standard) | BAREWire-serialized weight tensors (FWGT) |
| Multiple processes, IPC | Single process, Clef orchestration | Single process, actors on unified memory |
| nvidia-smi / htop monitoring | Standard profiling | Per-actor latency signals, Fidelity.UI dashboard |
| Hard-coded substrate assignment | Substrate affinity hints | Adaptive scheduling via supervision tree |

### What the Clef Inference Engine Must Implement (Native Path)

The native inference engine is the long-term engineering deliverable. It replaces the C++ inference runtimes (GGML, vLLM) with Clef compiled through MLIR:

| Component | Complexity | Notes |
|-----------|-----------|-------|
| Tokenizer (BPE) | Low | String processing; well-understood algorithm |
| Ternary layer forward pass | Low | Add-subtract on packed integers; ~30 lines Clef per layer type |
| Attention mechanism | **High** | KV cache management, multi-head attention, softmax |
| Layer normalization | Medium | Element-wise; needs vectorization for throughput |
| Activation functions (GELU, SiLU) | Low | Single-pass element-wise |
| Weight loading (memory-mapped) | Medium | mmap integration via platform binding |
| NPU dispatch | Medium | AMD XDNA driver API; platform binding |
| GPU dispatch (dense LLMs) | **High** | ROCm/HIP integration for RDNA 3.5 |
| KV cache management | **High** | Per-expert cache, context-length-dependent sizing |

Attention and KV cache management are the hardest problems. For ternary experts, attention heads may use reduced precision (INT8 activations), which simplifies the computation but still requires correct implementation. The KV cache for a 4K context window with 32 heads and 128-dim keys is ~32 MB per expert.

---

## Distillation Pipeline

The user distills their own models using a conventional training stack (PyTorch), then exports weights for the Clef inference engine.

### Training (External, PyTorch)

1. **Teacher model**: Large pretrained model (e.g., Llama 70B on rented GPU)
2. **Ternary experts**: Distill task-specific experts with ternary quantization-aware training (NPU targets)
3. **Dense models**: Fine-tune or quantize (INT8/FP16) general-purpose LLMs for GPU inference
4. **Router distillation**: Train a small router on the expert selection and hardware dispatch task
5. **Export**: Save weights in a flat binary format (BAREWire-compatible); quantization field identifies the target processor

### Weight Format

```
[header: 64 bytes]
  magic: u32          ("FWGT")
  version: u16
  quantization: u8    (0=FP32, 1=FP16, 2=INT8, 3=Ternary)
  num_layers: u16
  hidden_dim: u16
  num_heads: u16
  vocab_size: u32

[layer_offsets: num_layers * 8 bytes]
  offset: u64         (byte offset from file start)

[weight_data: contiguous]
  layer 0 weights...
  layer 1 weights...
  ...
```

Memory-mapping this file gives direct pointer access to each layer's weights. No deserialization. The mmap region is the weight tensor. BAREWire's zero-copy principle extends to model loading: the bits on disk are the bits in memory are the bits the NPU reads.

---

## Gap Analysis (2026-03-10)

### Compiler (Composer) Readiness

Composer has completed its **Foundation phase** (11 of 39 PRDs). Samples 01-09 compile and execute; the FPGA backend is proven through HelloArty synthesis and flash deployment. Closures (C-01) are in progress.

| Category | PRDs | Status | NPU Audio Relevance |
|----------|------|--------|---------------------|
| **Foundation (F-01 to F-10)** | 11 | Complete | Baseline compilation working |
| **Closures (C-01)** | 1 | In Progress | Required for callback-style APIs and actor behaviors |
| **Computation (C-02 to C-07)** | 6 | Planned | Collections, sequences needed for streaming pipelines |
| **Async (A-01 to A-06)** | 6 | Planned | Actor message loops, zero-copy scoped regions |
| **Threading (T-01 to T-05)** | 5 | Planned | MailboxProcessor actors (the capstone) |
| **IO (I-01, I-02)** | 2 | Planned | Audio device I/O |
| **Desktop/Reactive/Embedded** | 8 | Planned/Future | Not on critical path |
| **NPU Backend** | 0 | Not yet PRD'd | `TargetPlatform.NPU` is a stub returning `Error` |
| **GPU Compute Backend** | 0 | Not yet PRD'd | Same — stub only |

**Critical-path dependency chain for native actor pipeline:**
C-01 → C-02/C-03 → C-05 → C-06/C-07 → A-01 → A-02 → T-01 → T-02 → T-03 → T-04 → T-05 (MailboxProcessor)

This chain represents ~25 PRDs. The native-compiled MoE actor pipeline is a 2027 deliverable.

### Fidelity.Platform Binding Coverage

| Substrate | Quality | Files | What Exists | What's Missing |
|-----------|---------|-------|-------------|----------------|
| **CPU/Linux/x86_64** | Comprehensive | 89 `.clef` | libc, pthread, DRM, GBM, Wayland, WebSocket, WebView | Nothing critical |
| **GPU/AMD/RDNA3.5** | Substantial | ~13K lines | HIP runtime API (device, memory, stream, event, module) | hipBLAS, hipFFT, MIOpen, rocBLAS; no kernel compilation from Clef |
| **NPU/AMD/XDNA2** | Scaffolding | 2 files | Empty `PlatformDescriptor`, fidproj manifest | Everything: XRT bindings, tile topology, buffer mgmt, DMA |
| **FPGA/Xilinx** | Functional | — | Pin bindings + XDC constraints, validated | Working |

### Farscape (Binding Generator) Status

Farscape is validated for C library binding generation (libc, ~80+ functions). The three-layer architecture (Declarations → Types → Api) and `.pilot.toml` project system are proven patterns.

| Farscape Phase | Target | Status |
|----------------|--------|--------|
| Phase 0-3 | libc, Wayland, core Linux | Working |
| **Phase 4** | XRT/XDNA NPU bindings | **Designed** (`docs/roadmap/02_farscape-phase4-npu-xrt-binding.md`), not built |
| (Unplanned) | PipeWire audio bindings | Not started |
| (Unplanned) | ONNX Runtime C API bindings | Not started |

### Audio Infrastructure

**Nothing exists.** No PipeWire/ALSA bindings, no ONNX Runtime bindings, no DSP/FFT primitives, no audio capture/playback code anywhere in the Fidelity ecosystem. `Fidelity.Signal` is reactive signals (SolidJS-style), not audio signals.

---

## Remediation Plan

The gap analysis reveals two timelines: a **pragmatic 2026 path** using existing inference runtimes, and the **native 2027+ path** using Composer-compiled inference kernels. Both paths share the same hardware validation and audio I/O foundation.

### Phase 0: Hardware Validation (Now — 2 weeks)

All three substrates have working drivers. Remaining validation:

- [ ] Fix XRT/DKMS mmap API skew (rebuild XRT against DKMS driver, or use raw accel ioctls)
- [ ] Run MLIR-AIE IRON passthrough example on NPU (`~/repos/mlir-aie/` has examples)
- [ ] Run HIP compute test on gfx1151 (hipMemcpy + kernel launch, not just rocminfo)
- [ ] Verify DMA-BUF interop path between GPU and NPU buffer objects

### Phase 1: Audio I/O Foundation (Parallel with Composer PRDs — 4 weeks)

This is Farscape work, independent of Composer PRD progress:

- [ ] Create `pipewire.pilot.toml` targeting `/usr/include/pipewire-0.3/pipewire/stream.h`
- [ ] Generate PipeWire Clef bindings via Farscape (capture + playback)
- [ ] Create `onnxruntime.pilot.toml` targeting ONNX Runtime C API headers
- [ ] Generate ONNX Runtime Clef bindings via Farscape
- [ ] Write standalone audio loopback test: mic → buffer → speaker (proves PipeWire bindings)
- [ ] Write standalone ONNX Runtime test: load model → run inference → verify output

PipeWire is the correct audio backend for Omarchy/Arch (native, low latency, handles both capture and playback). ONNX Runtime is the pragmatic inference runtime — it has an XDNA execution provider for AMD NPUs.

### Phase 2: NPU Inference Bindings (After Phase 0 — 4 weeks)

Execute Farscape Phase 4 as designed:

- [ ] Generate XRT C shim bindings (`xrt_device.h`, `xrt_bo.h`, `xrt_kernel.h`, `xrt_run.h`)
- [ ] Build "hello NPU" test: load xclbin → create buffer objects → run → read results
- [ ] Test DMA-BUF zero-copy buffer sharing between HIP and XRT (`xrt_bo_import`)
- [ ] Verify ONNX Runtime XDNA execution provider works through generated bindings
- [ ] Profile NPU inference latency for Whisper-small (target: <100ms for 5s audio)

### Phase 3: Pragmatic Audio Agent (2026 Deliverable — 8 weeks)

A working audio agent using **ONNX Runtime on NPU** for inference, **PipeWire for audio I/O**, orchestrated by Clef host code. This does not require the full actor pipeline.

- [ ] Whisper-small ASR running on NPU via ONNX Runtime XDNA EP
- [ ] Audio capture → mel spectrogram → ASR pipeline (single-threaded, function-call chaining)
- [ ] Response generation via single expert model (NPU or GPU, depending on model size)
- [ ] TTS synthesis via Piper/VITS on CPU
- [ ] End-to-end: speak → transcribe → infer → speak response
- [ ] Latency profiling: target <2s end-to-end for simple queries

**Architecture**: Simple function pipeline, not actors. Each stage calls the next. This works after C-01 (closures) lands in Composer. No MailboxProcessor needed.

```clef
// Phase 3 architecture: function pipeline, not actors
let processQuery (audio: PcmBuffer) : PcmBuffer =
    audio
    |> MelSpectrogram.extract
    |> Whisper.transcribe npuSession      // ONNX Runtime on NPU
    |> Expert.generate expertSession      // ONNX Runtime on NPU or GPU
    |> Piper.synthesize ttsModel          // CPU
```

### Phase 4: Native Pipeline (2027+ — Requires Composer PRDs)

Graduate from ONNX Runtime to Composer-compiled inference kernels:

- [ ] Composer NPU backend PRD (MLIR → MLIR-AIE → xclbin)
- [ ] Composer GPU compute backend PRD (MLIR → HIP kernels)
- [ ] MailboxProcessor actors (T-05) for pipeline stages
- [ ] Native ternary forward pass compiled to NPU tiles
- [ ] Native attention mechanism compiled to GPU
- [ ] MoE routing with hardware-aware expert dispatch
- [ ] Expert hot-loading with mmap and prefetch
- [ ] Full actor pipeline replacing function-call chain

---

## Incremental Milestones (Native Pipeline — Long-Term)

The milestones below describe the **native-compiled** path. Phase 3 above delivers a working audio agent before these milestones are reached.

### Milestone 1: CPU-Only Pipeline

All inference on CPU via AVX-512. No NPU, no GPU. Proves the actor pipeline, weight loading, and inference engine work end-to-end.

- [ ] BPE tokenizer in Clef
- [ ] Ternary weight loader (memory-mapped)
- [ ] Ternary forward pass (add-subtract, AVX-512)
- [ ] Attention mechanism (scalar Clef, then vectorized)
- [ ] Whisper inference on CPU (slow but correct)
- [ ] TTS inference on CPU (VITS/Piper)
- [ ] Actor pipeline wiring
- [ ] End-to-end: speak → transcribe → route → infer → speak response

### Milestone 2: NPU Acceleration

Move ASR and ternary expert inference to XDNA 2 NPU.

- [ ] AMD XDNA driver platform binding
- [ ] NPU model loading and context management
- [ ] ASR on NPU (Whisper)
- [ ] Ternary expert inference on NPU
- [ ] Latency profiling and optimization

### Milestone 3: GPU LLM Inference

Dense language models on RDNA 3.5. These are not fallbacks; they are the primary path for open-ended generation, multi-turn conversation, and tasks where attention precision dominates quality.

- [ ] ROCm/HIP platform binding for RDNA 3.5
- [ ] Dense forward pass on GPU (FP16 matmul)
- [ ] GPU attention with KV cache
- [ ] Router dispatch logic: hardware-aware expert selection (NPU for ternary, GPU for dense)
- [ ] Concurrent NPU + GPU inference for multi-expert activations

### Milestone 4: Hot-Loading and Adaptive Routing

Expert pool management and learned routing.

- [ ] Memory-mapped expert pool with LRU eviction
- [ ] Prefetch on routing decision (`madvise`)
- [ ] Usage-based expert residency (keep frequently-used experts in memory)
- [ ] Fidelity.UI dashboard showing per-actor latency, expert residency, processor utilization

---

## Honest Constraints

### Resolved (as of 2026-03-10)

**NPU driver stack: operational.** The in-kernel 6.19 `amdxdna` driver had an SMU power-on bug (cmd 4 returned 0xff). This was resolved by installing `amdxdna-dkms` 7.0 from Mario Limonciello's (AMD) DKMS package. The NPU probes cleanly, `/dev/accel/accel0` exists, and the driver is bound. XRT userspace has a minor mmap API skew with the DKMS driver — functional but needs alignment.

**GPU compute stack: operational.** ROCm 7.2 with full HIP stack (hipBLAS, rocBLAS, rocFFT, RCCL) installed. `gfx1151` detected and functional via `rocminfo`.

### Active Risks

**NPU multi-context scheduling is unvalidated.** The XDNA 2 has 48 AIE tiles that theoretically support concurrent model contexts on separate tile columns. Whether the current driver stack (amdxdna 7.0 + XRT 2.21) actually supports this is unknown. Phase 0 must validate: can two xclbin overlays run on disjoint tile columns simultaneously? If not, the adaptive TTS placement and concurrent ASR+expert paths are sequential, not parallel.

**ONNX Runtime XDNA execution provider maturity.** ONNX Runtime has an AMD XDNA EP, but its model coverage, operator support, and performance characteristics on AIE2P specifically are not well-documented. Phase 2 must validate: does Whisper-small run correctly and at acceptable latency through the XDNA EP?

**Composer is ~28% through its PRD roadmap.** The native-compiled inference pipeline (Clef → MLIR → NPU/GPU kernels) requires closures, async, threading, actors, plus entirely new NPU and GPU compute backends. This is a 2027 deliverable. The 2026 path uses ONNX Runtime as the inference engine, with Clef providing the host orchestration.

**Model quality depends on distillation.** Ternary quantization works well for encoder models and feed-forward-dominant architectures. The BitNet router on CPU provides categorical reach into dense GPU models for tasks where attention precision dominates quality. This is hardware-aware scheduling, not a quality fallback.

**Attention is the engineering bottleneck (for native path).** The 2026 pragmatic path sidesteps this entirely by using ONNX Runtime for attention-heavy inference. The native Clef implementation (2027+) will need correct KV cache management before it can replace the runtime.

**The training pipeline remains Python.** Distillation requires PyTorch and GPU access. The Clef stack handles inference, not training. The boundary between training and inference is the weight file format.

---

## Cross-References

### Composer Docs
- [Architecture_Canonical.md](./Architecture_Canonical.md): Two-layer model, platform bindings
- [Platform_Binding_Model.md](./Platform_Binding_Model.md): How NPU and GPU bindings would be structured

### Fidelity Ecosystem
- `Fidelity.Platform/GPU/AMD/RDNA3_5/StrixHalo_iGPU/`: HIP/ROCm bindings (~13K lines, Farscape-generated)
- `Fidelity.Platform/NPU/AMD/XDNA2/StrixHalo_NPU/`: NPU platform descriptor (scaffolding only)
- `Farscape/docs/roadmap/02_farscape-phase4-npu-xrt-binding.md`: **DRM UAPI + XRT binding, spatial partitioning vision** (Section 11: Supervised Spatial Scheduling)
- `Farscape/docs/roadmap/05_farscape-phase4c-pipewire-audio.md`: PipeWire audio capture/playback binding
- `Farscape/docs/roadmap/06_farscape-phase4d-onnxruntime.md`: ONNX Runtime C API binding (CPU/GPU/NPU as equal-opportunity targets)
- `Farscape/docs/roadmap/99_composer-transcribe-vision.md`: Long-term Farscape → Transcribe/Transpose evolution

### Driver Stack
- `amdxdna-dkms` 7.0: [superm1/amdxdna-dkms](https://github.com/superm1/amdxdna-dkms) (Mario Limonciello, AMD)
- MLIR-AIE reference: `~/repos/mlir-aie/` ([Xilinx/mlir-aie](https://github.com/Xilinx/mlir-aie))
- XRT runtime: [amd/XRT](https://github.com/Xilinx/XRT) (xrt-smi, xclbinutil, libxrt_core)

### SpeakEZ Blog
- [A Unified Vision for Ternary Models](/blog/a-unified-vision-for-ternary-models/): Ternary quantization, MoE routing, Strix Halo deployment model
- [Bringing Posit Arithmetic to Clef](/blog/bringing-posit-arithmetic-to-fsharp/): Posit accumulation for precision-critical paths
- [Fidelity.Rx / Signal-Actor Isomorphism](/blog/fidelityrx-native-reactivity-in-fidelity/): Actor pipeline model

### External
- [AMD XDNA Architecture](https://www.amd.com/en/technologies/xdna): NPU specifications
- [Whisper](https://github.com/openai/whisper): ASR model family
- [Piper TTS](https://github.com/rhasspy/piper): Fast, local neural TTS
- [BitNet](https://arxiv.org/abs/2310.11453): Ternary quantization-aware training
- [ONNX Runtime](https://github.com/microsoft/onnxruntime): Inference runtime with XDNA execution provider
