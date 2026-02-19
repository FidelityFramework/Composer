/// Pipeline Types - Backend abstraction for multi-target compilation
///
/// A backend is a function value: MLIR text + context → artifact.
/// Each target provides its own backend. The orchestrator composes
/// and runs the pipeline without knowing which backend it is.
module Core.Types.Pipeline

/// Result of a backend compilation pass
type BackEndArtifact =
    | NativeBinary of path: string
    | Verilog of path: string
    | IntermediateOnly of format: string

/// Context passed to a backend for compilation.
/// Contains backend-internal configuration — the orchestrator assembles
/// this but doesn't interpret it.
type BackEndContext = {
    OutputPath: string
    IntermediatesDir: string option
    /// CLI target override (e.g., --target x86_64-pc-windows-gnu for cross-compilation).
    /// Backend-specific: LLVM uses it, CIRCT ignores it.
    TargetTripleOverride: string option
    DeploymentMode: Dialects.DeploymentMode
    /// Stop after intermediate generation (e.g., --emit-llvm for LLVM, Verilog-only for CIRCT)
    EmitIntermediateOnly: bool
}

/// A backend is a function value that compiles MLIR text to a target artifact.
/// Each target (LLVM, CIRCT, ...) provides its own BackEnd value.
/// No dispatch in the orchestrator — the pipeline is assembled once at startup.
type BackEnd = {
    /// Human-readable name for logging
    Name: string
    /// Compile MLIR text to target artifact
    Compile: string -> BackEndContext -> Result<BackEndArtifact, string>
}
