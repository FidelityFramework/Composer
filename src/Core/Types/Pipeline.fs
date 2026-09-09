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
    | Xclbin of xclbinPath: string * instsPath: string
    | GpuCodeObject of path: string
    | IntermediateOnly of format: string

/// Resolved declarations, not executable build hooks. MCU hardware facts come
/// from Fidelity.Platform's BAREWire records in the checked semantic graph.
type EmbeddedTarget = {
    PlatformId: string
    Image: BAREWire.Hardware.CortexMImageDescriptor
    Vectors: BAREWire.Hardware.StructDescriptor
    Flash: BAREWire.Platform.MemorySpace
    Ram: BAREWire.Platform.MemorySpace
    StartupSource: string
    ProvidedLibraries: Set<string>
    VectorHandlers: Map<int, string>
    RecoveryDirectory: string
    ToolDirectory: string option
    ProbeLibrary: string option
    WatchSymbols: Map<string, int>
}

/// Explicit ELF link inputs. Paths name target files; cross links never search host libraries.
type NativeLinkOptions = {
    Sysroot: string option
    LibraryPaths: string list
    StartFiles: string list
    EndFiles: string list
    DynamicLinker: string option
    LinkerScript: string option
} with
    static member Empty =
        { Sysroot = None; LibraryPaths = []; StartFiles = []; EndFiles = []
          DynamicLinker = None; LinkerScript = None }

/// Context passed to a backend for compilation.
/// Contains backend-internal configuration — the orchestrator assembles
/// this but doesn't interpret it.
type BackEndContext = {
    OutputPath: string
    IntermediatesDir: string option
    /// CLI target override (e.g., --target x86_64-pc-windows-gnu for cross-compilation).
    /// Backend-specific: LLVM uses it, CIRCT ignores it.
    TargetTripleOverride: string option
    TargetPointerBits: int option
    TargetCpu: string option
    DeploymentMode: Dialects.DeploymentMode
    /// Stop after intermediate generation (e.g., --emit-llvm for LLVM, Verilog-only for CIRCT)
    EmitIntermediateOnly: bool
    /// External library dependencies accumulated during binding resolution.
    /// Used to generate data-driven linker flags (e.g., {"c"; "wayland-client"} → -lc -lwayland-client)
    ExternLibraries: Set<string>
    NativeLink: NativeLinkOptions
    EmbeddedTarget: EmbeddedTarget option
    Deploy: bool
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
