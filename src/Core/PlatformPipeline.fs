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
    | CPU | MCU -> BackEnd.LLVM.Pipeline.backend
    | GPU -> { Name = "GPU"; Compile = fun _ _ -> Error "GPU backend not yet implemented." }
    | NPU -> { Name = "NPU"; Compile = fun _ _ -> Error "NPU backend not yet implemented." }
