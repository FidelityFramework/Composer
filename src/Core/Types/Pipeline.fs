/// Pipeline Types - Backend abstraction for multi-target compilation
///
/// A backend is a function value: typed MLIR and portable text + context → artifact.
/// Each target provides its own backend. The orchestrator composes
/// and runs the pipeline without knowing which backend it is.
module Core.Types.Pipeline

/// The exact witnessed module and its portable serialization travel together.
/// Target realization consumes operations; it never reconstructs source semantics
/// or parses diagnostic text to recover lost operations.
type BackEndInput = {
    Operations: Alex.Dialects.Core.Types.MLIROp list
    PointerBits: Result<int, string>
    ModuleName: string option
    Text: string
    /// Exact source writable objects and the witnessed symbol that realizes
    /// each one. A backend still owes placement/capacity correspondence.
    WritableStorage: (string * Clef.Compiler.PSGSaturation.SemanticGraph.Types.ProgramStorageEntry) list
}

/// Result of a backend compilation pass
type BackEndArtifact =
    | NativeBinary of path: string
    | Verilog of path: string
    | Xclbin of xclbinPath: string * instsPath: string
    | GpuCodeObject of path: string
    | IntermediateOnly of format: string

/// The two accepted address-table Cortex-M image contracts. These are explicit
/// backend capabilities, not a claim that every Cortex-M part shares one map.
[<RequireQualifiedAccess>]
type CortexMProfile =
    | CortexM33SoftFloat
    | Stm32H747HardFloat

/// Resolved declarations, not executable build hooks. MCU hardware facts come
/// from Fidelity.Platform's BAREWire records in the checked semantic graph.
type EmbeddedTarget = {
    PlatformId: string
    Profile: CortexMProfile
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

/// Resolved declarations for the ROM-loaded Xtensa image path. A sibling of
/// EmbeddedTarget, not a widening of it: this target has no flash at address
/// zero to execute from, three SRAM banks of which one is dual-mapped, and a
/// vector block that is code rather than an address table.
type XtensaTarget = {
    PlatformId: string
    Image: BAREWire.Hardware.XtensaImageDescriptor
    Vectors: BAREWire.Hardware.StructDescriptor
    /// Instruction-bus-only bank.
    Sram0: BAREWire.Platform.MemorySpace
    /// Dual-mapped bank, instruction-bus view; owns the bank's capacity.
    Sram1: BAREWire.Platform.MemorySpace
    /// The dual-mapped bank's base on the data bus.
    Sram1DataBase: int64
    /// End of the data window: the ROM's and the cache's memory begins here.
    DataLimit: int64
    /// Data-bus-only bank.
    Sram2: BAREWire.Platform.MemorySpace
    /// The download target, addressed by offset rather than mapped.
    FlashStore: BAREWire.Platform.MemorySpace
    StartupSource: string
    ProvidedLibraries: Set<string>
    /// Named vector entry -> handler symbol. Names, not slot indices: the
    /// Xtensa block has ten named entries at fixed offsets, and "slot 15"
    /// means nothing here.
    VectorEntries: Map<string, string>
    RecoveryDirectory: string
    /// Directory holding the LLVM tools for this target, when not on PATH.
    ToolDirectory: string option
    /// -mcpu, when the toolchain has a model for this part.
    Cpu: string option
    /// -mattr features, used when no CPU model exists. Upstream LLVM has 26 of
    /// the esp32s3 bundle's 29 features but no esp32s3 CPU.
    Features: string list
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
    /// Explicit checked platform selections, distinct from the target triple.
    PlatformOS: string option
    RuntimeModel: Clef.Compiler.NativeTypedTree.NativeTypes.RuntimeModel option
    DeploymentMode: Dialects.DeploymentMode
    /// Stop after intermediate generation (e.g., --emit-llvm for LLVM, Verilog-only for CIRCT)
    EmitIntermediateOnly: bool
    /// External library dependencies accumulated during binding resolution.
    /// Used to generate data-driven linker flags (e.g., {"c"; "wayland-client"} → -lc -lwayland-client)
    ExternLibraries: Set<string>
    NativeLink: NativeLinkOptions
    EmbeddedTarget: EmbeddedTarget option
    /// Set instead of EmbeddedTarget when the selected platform is Xtensa.
    /// Exactly one of the two is populated; the MCU backend dispatches on it.
    XtensaTarget: XtensaTarget option
    Deploy: bool
}

/// A backend is a function value that compiles a witnessed MLIR module to a target artifact.
/// Each target (LLVM, CIRCT, ...) provides its own BackEnd value.
/// No dispatch in the orchestrator — the pipeline is assembled once at startup.
type BackEnd = {
    /// Human-readable name for logging
    Name: string
    /// Compile typed MLIR plus its portable serialization to a target artifact
    Compile: BackEndInput -> BackEndContext -> Result<BackEndArtifact, string>
}
