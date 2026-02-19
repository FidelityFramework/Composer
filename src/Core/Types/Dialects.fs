module Core.Types.Dialects

type MLIRDialect =
    | Standard | LLVM | Func | Arith | SCF
    | MemRef | Index | Affine | Builtin

/// Target platform — determines which compilation pipeline (backend) to use.
/// This is the orchestrator's vocabulary. Console vs freestanding, hosted vs
/// unikernel — those are deployment modes internal to each backend.
type TargetPlatform =
    | CPU           // General-purpose processor → LLVM backend
    | FPGA          // Field-programmable gate array → CIRCT backend
    | GPU           // Graphics/compute processor → future
    | MCU           // Microcontroller → LLVM backend (different config)
    | NPU           // Neural processing unit → future

/// Deployment mode — backend-internal concern from fidproj configuration.
/// Determines linker flags, runtime dependencies, entry point handling.
/// Does NOT affect pipeline selection.
type DeploymentMode =
    | Console       // Linked with libc, can use stdio
    | Freestanding  // No libc, syscalls only
    | Embedded      // No OS, bare metal
    | Library       // Shared library (.so/.dll)
