/// PlatformPipeline - Resolves target platform to backend
///
/// This is the ONE place where TargetPlatform maps to a BackEnd value.
/// It runs at pipeline assembly time, not during compilation.
/// The orchestrator never sees this match — it receives the assembled BackEnd.
module Core.PlatformPipeline

open Core.Types.Dialects
open Core.Types.Pipeline

/// Resolve a target platform to its backend.
/// Configuration, not dispatch — called once at pipeline assembly time.
let resolveBackEnd (targetPlatform: TargetPlatform) : BackEnd =
    match targetPlatform with
    | FPGA -> BackEnd.CIRCT.Pipeline.backend
    | MCU -> BackEnd.MCU.Pipeline.backend
    | CPU | TargetPlatform.Library -> BackEnd.LLVM.Pipeline.backend
    | GPU -> BackEnd.GPU.Pipeline.backend
    | NPU -> BackEnd.AIE.Pipeline.backend
