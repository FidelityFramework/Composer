/// BindingTypes - Platform binding types for witness-based MLIR generation
///
/// ARCHITECTURAL PRINCIPLE (January 2026):
/// Bindings RETURN structured MLIROp lists - they do NOT emit.
/// The fold accumulates what bindings return via withOps.
///
/// CANONICAL ARCHITECTURE (January 2026):
/// Bindings receive SSAAssignment (pre-computed, immutable) for SSA lookups.
/// They do NOT receive PSGZipper - zipper is for navigation, not coeffects.
module Alex.Bindings.BindingTypes

open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.Bindings.PlatformTypes

// Alias for SSA lookup coefficient
module SSALookup = PSGElaboration.SSAAssignment

// ===================================================================
// Binding Strategy
// ===================================================================

type BindingStrategy =
    | Static
    | Dynamic

// ===================================================================
// Binding Result - What bindings RETURN
// ===================================================================

/// Result of a binding: operations to emit and optional result value
/// topLevelOps are for extern declarations that need to be emitted at module level
type BindingResult =
    | BoundOps of ops: MLIROp list * topLevelOps: MLIROp list * result: Val option
    | NotSupported of reason: string

// ===================================================================
// Platform Primitive Types
// ===================================================================

/// Platform primitive call (uses typed Val instead of string tuples)
type PlatformPrimitive = {
    EntryPoint: string
    Library: string
    CallingConvention: string
    Args: Val list
    ReturnType: MLIRType
    BindingStrategy: BindingStrategy
}

/// External function declaration
type ExternalDeclaration = {
    Name: string
    Signature: string
    Library: string option
}

// ===================================================================
// SSA Lookup Helpers - Used by bindings
// ===================================================================

/// Look up pre-assigned SSA for a node
let lookupSSA (nodeId: NodeId) (ssa: SSALookup.SSAAssignment) : SSA option =
    SSALookup.lookupSSA nodeId ssa

/// Require SSA for a node (fail if not found)
let requireSSA (nodeId: NodeId) (ssa: SSALookup.SSAAssignment) : SSA =
    match lookupSSA nodeId ssa with
    | Some s -> s
    | None -> failwithf "No SSA assignment for node %A" nodeId

/// Look up all SSAs for a node (for multi-SSA operations)
let lookupSSAs (nodeId: NodeId) (ssa: SSALookup.SSAAssignment) : SSA list option =
    SSALookup.lookupSSAs nodeId ssa

/// Require all SSAs for a node
let requireSSAs (nodeId: NodeId) (ssa: SSALookup.SSAAssignment) : SSA list =
    match lookupSSAs nodeId ssa with
    | Some ssas -> ssas
    | None -> failwithf "No SSA allocation for node %A" nodeId

// ===================================================================
// Binding Signature - RETURNS ops, does not emit
// ===================================================================

/// A binding takes nodeId (for pre-assigned SSAs), SSAAssignment (coeffects), and primitive, RETURNS ops
/// SSAs are pre-allocated during SSAAssignment pass (coeffects pattern)
type Binding = NodeId -> SSALookup.SSAAssignment -> PlatformPrimitive -> BindingResult

// ===================================================================
// Platform Dispatch Registry
// ===================================================================

module PlatformDispatch =
    let mutable private bindings: Map<(OSFamily * Architecture * string), Binding> = Map.empty
    let mutable private currentPlatform: TargetPlatform option = None

    let register (os: OSFamily) (arch: Architecture) (entryPoint: string) (binding: Binding) =
        let key = (os, arch, entryPoint)
        bindings <- Map.add key binding bindings

    let registerForOS (os: OSFamily) (entryPoint: string) (binding: Binding) =
        register os X86_64 entryPoint binding
        register os ARM64 entryPoint binding

    let setTargetPlatform (platform: TargetPlatform) =
        currentPlatform <- Some platform

    let getTargetPlatform () =
        currentPlatform |> Option.defaultValue (TargetPlatform.detectHost())

    /// Dispatch: Returns BindingResult (ops + result or NotSupported)
    /// nodeId is used to get pre-assigned SSAs from SSAAssignment coeffects
    let dispatch (nodeId: NodeId) (ssa: SSALookup.SSAAssignment) (prim: PlatformPrimitive) : BindingResult =
        let platform = getTargetPlatform()
        let key = (platform.OS, platform.Arch, prim.EntryPoint)
        match Map.tryFind key bindings with
        | Some binding -> binding nodeId ssa prim
        | None ->
            let fallbackKey = (platform.OS, X86_64, prim.EntryPoint)
            match Map.tryFind fallbackKey bindings with
            | Some binding -> binding nodeId ssa prim
            | None -> NotSupported $"No binding for {prim.EntryPoint} on {platform.OS}/{platform.Arch}"

    let hasBinding (entryPoint: string) : bool =
        let platform = getTargetPlatform()
        let key = (platform.OS, platform.Arch, entryPoint)
        Map.containsKey key bindings ||
        Map.containsKey (platform.OS, X86_64, entryPoint) bindings

    let clear () =
        bindings <- Map.empty
        currentPlatform <- None

    let getRegisteredEntryPoints () : string list =
        let platform = getTargetPlatform()
        bindings
        |> Map.toList
        |> List.filter (fun ((os, arch, _), _) -> os = platform.OS && (arch = platform.Arch || arch = X86_64))
        |> List.map (fun ((_, _, ep), _) -> ep)
        |> List.distinct
