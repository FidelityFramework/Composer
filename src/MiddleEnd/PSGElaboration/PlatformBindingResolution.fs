/// Platform Binding Resolution Nanopass
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// This nanopass resolves ALL platform decisions BEFORE witnessing.
/// Witnesses just lookup pre-resolved bindings - they don't decide.
///
/// PSG should be complete before witnessing. Platform decisions
/// (freestanding vs console, syscall vs libc) are resolved here.
///
/// NOTE: Entry point elaboration (_start wrapper for freestanding mode)
/// is NOT handled here. That's a PSG-level concern handled by
/// IntrinsicElaboration.fs in CCS. See Phase 7 of XParsec-centric
/// Baker remediation plan.
module PSGElaboration.PlatformBindingResolution

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open PSGElaboration.PlatformConfig
open Alex.Dialects.Core.Types

// ═══════════════════════════════════════════════════════════════════════════
// INTRINSIC RESOLUTION
// ═══════════════════════════════════════════════════════════════════════════

/// Check if intrinsic is a platform intrinsic that needs resolution
let private isPlatformIntrinsic (info: IntrinsicInfo) : bool =
    match info.Module with
    | IntrinsicModule.Sys -> true
    | IntrinsicModule.DateTime ->
        // DateTime.now and utcNow need platform resolution (syscall)
        match info.Operation with
        | "now" | "utcNow" -> true
        | _ -> false  // Component extraction is pure arithmetic
    // NOTE: Console is NOT an intrinsic - see fsnative-spec/spec/platform-bindings.md
    | _ -> false

/// Resolve a single intrinsic to a concrete binding based on runtime mode
let private resolveIntrinsic
    (mode: RuntimeMode)
    (os: OSFamily)
    (info: IntrinsicInfo)
    : ResolvedBinding option =

    match mode with
    | Freestanding ->
        // Direct syscall - look up syscall number for OS
        match info.Module, info.Operation with
        | IntrinsicModule.Sys, "write" ->
            let syscallNum = SyscallNumbers.getWriteSyscall os
            Some (Syscall (syscallNum, "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"))
        | IntrinsicModule.Sys, "read" ->
            let syscallNum = SyscallNumbers.getReadSyscall os
            Some (Syscall (syscallNum, "={rax},{rax},{rdi},{rsi},{rdx},~{rcx},~{r11},~{memory}"))
        | IntrinsicModule.Sys, "exit" ->
            let syscallNum = SyscallNumbers.getExitSyscall os
            Some (Syscall (syscallNum, "={rax},{rax},{rdi},~{rcx},~{r11},~{memory}"))
        // NOTE: Sys.stackArgc and Sys.stackArgv are witnessed directly by Alex
        // based on target architecture. No pre-resolution needed here.
        // NOTE: Console.* is NOT an intrinsic - it's Layer 3 user code in Fidelity.Platform
        // that uses Sys.* intrinsics. See fsnative-spec/spec/platform-bindings.md
        | _ -> None  // Not a platform-resolvable intrinsic

    | Console ->
        // libc call - use standard function names
        match info.Module, info.Operation with
        | IntrinsicModule.Sys, "write" -> Some (LibcCall "write")
        | IntrinsicModule.Sys, "read" -> Some (LibcCall "read")
        | IntrinsicModule.Sys, "exit" -> Some (LibcCall "exit")
        // NOTE: Console.* is NOT an intrinsic - it's Layer 3 user code in Fidelity.Platform
        // that uses Sys.* intrinsics. See fsnative-spec/spec/platform-bindings.md
        | _ -> None  // Not a platform-resolvable intrinsic

// ═══════════════════════════════════════════════════════════════════════════
// FIDELITYEXTERN RESOLUTION
// ═══════════════════════════════════════════════════════════════════════════

/// Try to resolve an Application's target function to a [<FidelityExtern>] binding.
/// Follows the reference chain: funcNodeId → VarRef → definition Binding → metadata.
/// Returns (library, symbol) if the Binding carries FidelityExtern metadata.
let private tryResolveExternCall (graph: SemanticGraph) (funcNodeId: NodeId) : (string * string) option =
    let rec followToBinding (nodeId: NodeId) : SemanticNode option =
        match Map.tryFind nodeId graph.Nodes with
        | Some node ->
            match node.Kind with
            | SemanticKind.TypeAnnotation (innerFuncId, _) ->
                followToBinding innerFuncId
            | SemanticKind.VarRef (_, Some definitionId) ->
                Map.tryFind definitionId graph.Nodes
            | SemanticKind.Binding _ -> Some node
            | _ -> None
        | None -> None

    match followToBinding funcNodeId with
    | Some bindingNode ->
        match Map.tryFind "FidelityExtern.Library" bindingNode.Metadata,
              Map.tryFind "FidelityExtern.Symbol" bindingNode.Metadata with
        | Some (MetadataValue.String library), Some (MetadataValue.String symbol) ->
            Some (library, symbol)
        | _ -> None
    | None -> None

// ═══════════════════════════════════════════════════════════════════════════
// MAIN ANALYSIS ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════

/// Analyze the semantic graph and resolve all platform bindings
/// This is the main nanopass entry point
let analyze
    (graph: SemanticGraph)
    (mode: RuntimeMode)
    (os: OSFamily)
    (arch: Architecture)
    : PlatformResolutionResult =

    // Resolve platform word type for this architecture
    // This is the authoritative source for what PlatformWord means on this target
    let wordType = platformWordType arch

    // Walk graph, find Application nodes that call platform intrinsics.
    // Bindings are keyed by the Application node ID (the call site),
    // since witnesses process Application nodes during traversal.
    let resolveFunc (funcNodeId: NodeId) : IntrinsicInfo option =
        match Map.tryFind funcNodeId graph.Nodes with
        | Some funcNode ->
            match funcNode.Kind with
            | SemanticKind.TypeAnnotation (innerFuncId, _) ->
                match Map.tryFind innerFuncId graph.Nodes with
                | Some { Kind = SemanticKind.Intrinsic info } when isPlatformIntrinsic info -> Some info
                | _ -> None
            | SemanticKind.Intrinsic info when isPlatformIntrinsic info -> Some info
            | _ -> None
        | None -> None

    let bindings =
        graph.Nodes
        |> Map.toSeq
        |> Seq.choose (fun (nodeId, node) ->
            let (NodeId nodeIdInt) = nodeId
            match node.Kind with
            | SemanticKind.Application (funcNodeId, _) ->
                // First: try intrinsic resolution (Sys.write, Sys.read, etc.)
                match resolveFunc funcNodeId with
                | Some intrinsicInfo ->
                    match resolveIntrinsic mode os intrinsicInfo with
                    | Some resolved ->
                        let entryPoint = sprintf "%s.%s" (string intrinsicInfo.Module) intrinsicInfo.Operation
                        Some (nodeIdInt, {
                            NodeId = nodeIdInt
                            EntryPoint = entryPoint
                            Mode = mode
                            Resolved = resolved
                        })
                    | None -> None
                | None ->
                    // Second: try [<FidelityExtern>] resolution
                    match tryResolveExternCall graph funcNodeId with
                    | Some (library, symbol) ->
                        Some (nodeIdInt, {
                            NodeId = nodeIdInt
                            EntryPoint = symbol
                            Mode = mode
                            Resolved = ExternCall (library, symbol)
                        })
                    | None -> None
            | _ -> None)
        |> Map.ofSeq

    // Note: _start wrapper generation moved to CCS IntrinsicElaboration.fs
    // Entry point elaboration is a PSG-level concern, not code generation
    let needsStart = (mode = Freestanding)

    // Accumulate external library dependencies from resolved bindings
    let externLibs =
        bindings
        |> Map.toSeq
        |> Seq.choose (fun (_, binding) ->
            match binding.Resolved with
            | LibcCall _ -> Some "c"
            | ExternCall (library, _) -> Some library
            | Syscall _ | InlineAsm _ -> None)
        |> Set.ofSeq

    {
        RuntimeMode = mode
        TargetOS = os
        TargetArch = arch
        PlatformWordType = wordType
        Bindings = bindings
        ExternLibraries = externLibs
        NeedsStartWrapper = needsStart
    }

// ═══════════════════════════════════════════════════════════════════════════
// LOOKUP HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Lookup a resolved binding by node ID
let lookupBinding (nodeId: int) (result: PlatformResolutionResult) : BindingResolution option =
    Map.tryFind nodeId result.Bindings

/// Check if a node has a platform binding resolution
let hasBinding (nodeId: int) (result: PlatformResolutionResult) : bool =
    Map.containsKey nodeId result.Bindings

/// Get all resolved bindings
let allBindings (result: PlatformResolutionResult) : BindingResolution list =
    result.Bindings |> Map.toList |> List.map snd
