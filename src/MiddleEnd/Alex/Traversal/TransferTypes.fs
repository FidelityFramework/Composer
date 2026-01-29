/// Transfer Types - Core types for MLIR Transfer
///
/// CANONICAL ARCHITECTURE (January 2026):
/// This file defines the types that witnesses receive. It compiles BEFORE
/// witnesses so they can elegantly take `ctx: WitnessContext` rather than
/// explicit parameter threading.
///
/// The Three Concerns:
/// - PSGZipper: Pure navigation (Focus, Path, Graph) - defined in PSGZipper.fs
/// - TransferCoeffects: Pre-computed, immutable coeffects
/// - MLIRAccumulator: Mutable fold state
///
/// See: mlir_transfer_canonical_architecture memory
module Alex.Traversal.TransferTypes

open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Types
open FSharp.Native.Compiler.PSGSaturation.SemanticGraph.Core
open FSharp.Native.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open PSGElaboration.PlatformConfig
open Alex.CodeGeneration.TypeMapping
open Alex.Traversal.PSGZipper

// ═══════════════════════════════════════════════════════════════════════════
// MODULE ALIASES (for type definitions)
// ═══════════════════════════════════════════════════════════════════════════

module MutAnalysis = PSGElaboration.MutabilityAnalysis
module SSAAssign = PSGElaboration.SSAAssignment
module StringCollect = PSGElaboration.StringCollection
module PatternAnalysis = PSGElaboration.PatternBindingAnalysis
module YieldStateIndices = PSGElaboration.YieldStateIndices

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFER COEFFECTS (Pre-computed, Immutable)
// ═══════════════════════════════════════════════════════════════════════════

/// Pre-computed coeffects - computed ONCE before traversal, NEVER modified
type TransferCoeffects = {
    SSA: SSAAssign.SSAAssignment
    Platform: PlatformResolutionResult
    Mutability: MutAnalysis.MutabilityAnalysisResult
    PatternBindings: PatternAnalysis.PatternBindingAnalysisResult
    Strings: StringCollect.StringTable
    YieldStates: YieldStateIndices.YieldStateCoeffect
    EntryPointLambdaIds: Set<int>
}

// ═══════════════════════════════════════════════════════════════════════════
// EXECUTION TRACE (For Debugging Pattern Failures)
// ═══════════════════════════════════════════════════════════════════════════

/// Execution trace entry - records each step in pattern execution
type ExecutionTrace = {
    /// Hierarchy depth: 0=Witness, 1=Pattern, 2=Element
    Depth: int
    
    /// Component name: "LiteralWitness", "pBuildStringLiteral", "pAddressOf"
    ComponentName: string
    
    /// PSG NodeId (if witness-level, otherwise None)
    NodeId: NodeId option
    
    /// Serialized parameters for inspection
    Parameters: string
    
    /// Sequential execution order
    Timestamp: int
}

module ExecutionTrace =
    /// Format trace entry for display
    let format (trace: ExecutionTrace) : string =
        let indent = String.replicate trace.Depth "  "
        let nodeInfo = match trace.NodeId with Some nid -> sprintf "[Node %d] " (NodeId.value nid) | None -> ""
        sprintf "%s%s%s(%s)" indent nodeInfo trace.ComponentName trace.Parameters

/// Trace collector - mutable accumulator for execution traces
type TraceCollector = ResizeArray<ExecutionTrace>

module TraceCollector =
    let create () : TraceCollector = ResizeArray<ExecutionTrace>()
    
    let add (depth: int) (componentName: string) (nodeId: NodeId option) (parameters: string) (collector: TraceCollector) =
        collector.Add({
            Depth = depth
            ComponentName = componentName
            NodeId = nodeId
            Parameters = parameters
            Timestamp = collector.Count
        })
    
    let toList (collector: TraceCollector) : ExecutionTrace list =
        collector |> Seq.toList

// ═══════════════════════════════════════════════════════════════════════════
// STRUCTURED DIAGNOSTICS
// ═══════════════════════════════════════════════════════════════════════════

/// Diagnostic severity levels
type DiagnosticSeverity =
    | Error
    | Warning
    | Info

/// Structured diagnostic capturing WHERE and WHAT went wrong
type Diagnostic = {
    /// Severity level
    Severity: DiagnosticSeverity

    /// NodeId where error occurred (if known)
    NodeId: NodeId option

    /// Source component (e.g., "Literal", "Arithmetic", "ControlFlow")
    Source: string option

    /// Phase/operation that failed (e.g., "pBuildStringLiteral", "SSA lookup")
    Phase: string option

    /// Human-readable message
    Message: string

    /// Optional: Expected vs Actual for validation errors
    Details: (string * string) option
}

module Diagnostic =
    /// Create an error diagnostic with full context
    let error nodeId source phase message =
        { Severity = Error
          NodeId = nodeId
          Source = source
          Phase = phase
          Message = message
          Details = None }

    /// Create an error diagnostic with just a message
    let errorSimple message =
        error None None None message

    /// Create an error diagnostic with expected/actual details
    let errorWithDetails nodeId source phase message expected actual =
        { Severity = Error
          NodeId = nodeId
          Source = source
          Phase = phase
          Message = message
          Details = Some (expected, actual) }

    /// Format diagnostic to human-readable string
    let format (diag: Diagnostic) : string =
        let parts = [
            // Severity
            match diag.Severity with
            | Error -> Some "[ERROR]"
            | Warning -> Some "[WARNING]"
            | Info -> Some "[INFO]"

            // NodeId
            match diag.NodeId with
            | Some nid -> Some (sprintf "Node %d" (NodeId.value nid))
            | None -> None

            // Source
            match diag.Source with
            | Some src -> Some (sprintf "(%s)" src)
            | None -> None

            // Phase
            match diag.Phase with
            | Some phase -> Some (sprintf "in %s" phase)
            | None -> None

            // Message
            Some diag.Message

            // Details
            match diag.Details with
            | Some (expected, actual) ->
                Some (sprintf "Expected: %s, Actual: %s" expected actual)
            | None -> None
        ]
        parts
        |> List.choose id
        |> String.concat " "

// ═══════════════════════════════════════════════════════════════════════════
// MLIR ACCUMULATOR (Mutable Fold State)
// ═══════════════════════════════════════════════════════════════════════════

/// Scope for nested regions
type AccumulatorScope = {
    VarAssoc: Map<string, SSA * MLIRType>
    NodeAssoc: Map<NodeId, SSA * MLIRType>
    CapturedVars: Set<string>
    CapturedMuts: Set<string>
}

/// Mutable accumulator - the `acc` in the fold
type MLIRAccumulator = {
    mutable TopLevelOps: MLIROp list
    mutable Errors: Diagnostic list
    mutable Visited: Set<NodeId>
    mutable ScopeStack: AccumulatorScope list
    mutable CurrentScope: AccumulatorScope
}

module MLIRAccumulator =
    let empty () : MLIRAccumulator =
        let globalScope = {
            VarAssoc = Map.empty
            NodeAssoc = Map.empty
            CapturedVars = Set.empty
            CapturedMuts = Set.empty
        }
        {
            TopLevelOps = []
            Errors = []
            Visited = Set.empty
            ScopeStack = []
            CurrentScope = globalScope
        }

    let addTopLevelOp (op: MLIROp) (acc: MLIRAccumulator) =
        acc.TopLevelOps <- op :: acc.TopLevelOps

    let addTopLevelOps (ops: MLIROp list) (acc: MLIRAccumulator) =
        acc.TopLevelOps <- List.rev ops @ acc.TopLevelOps

    let addError (err: Diagnostic) (acc: MLIRAccumulator) =
        acc.Errors <- err :: acc.Errors

    let markVisited (nodeId: NodeId) (acc: MLIRAccumulator) =
        acc.Visited <- Set.add nodeId acc.Visited

    let isVisited (nodeId: NodeId) (acc: MLIRAccumulator) =
        Set.contains nodeId acc.Visited

    let bindVar (name: string) (ssa: SSA) (ty: MLIRType) (acc: MLIRAccumulator) =
        acc.CurrentScope <- { acc.CurrentScope with VarAssoc = Map.add name (ssa, ty) acc.CurrentScope.VarAssoc }

    let recallVar (name: string) (acc: MLIRAccumulator) =
        Map.tryFind name acc.CurrentScope.VarAssoc

    let bindNode (nodeId: NodeId) (ssa: SSA) (ty: MLIRType) (acc: MLIRAccumulator) =
        acc.CurrentScope <- { acc.CurrentScope with NodeAssoc = Map.add nodeId (ssa, ty) acc.CurrentScope.NodeAssoc }

    let recallNode (nodeId: NodeId) (acc: MLIRAccumulator) =
        Map.tryFind nodeId acc.CurrentScope.NodeAssoc

    let isCapturedVariable (name: string) (acc: MLIRAccumulator) =
        Set.contains name acc.CurrentScope.CapturedVars

    let isCapturedMutable (name: string) (acc: MLIRAccumulator) =
        Set.contains name acc.CurrentScope.CapturedMuts

    let pushScope (scope: AccumulatorScope) (acc: MLIRAccumulator) =
        acc.ScopeStack <- acc.CurrentScope :: acc.ScopeStack
        acc.CurrentScope <- scope

    let popScope (acc: MLIRAccumulator) =
        match acc.ScopeStack with
        | prev :: rest ->
            acc.CurrentScope <- prev
            acc.ScopeStack <- rest
        | [] -> ()

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFER RESULT (Result of witnessing a node)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of witnessing a PSG node
type TransferResult =
    | TRValue of Val                    // Produces a value (SSA + type)
    | TRVoid                             // Produces no value (effect only)
    | TRError of Diagnostic              // Error with structured context

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS OUTPUT (What witnesses return)
// ═══════════════════════════════════════════════════════════════════════════

/// Codata returned by witnesses
type WitnessOutput = {
    InlineOps: MLIROp list
    TopLevelOps: MLIROp list
    Result: TransferResult
}

module WitnessOutput =
    let empty = { InlineOps = []; TopLevelOps = []; Result = TRVoid }
    let inline' ops result = { InlineOps = ops; TopLevelOps = []; Result = result }
    let value v = { InlineOps = []; TopLevelOps = []; Result = TRValue v }

    /// Create error output with simple message
    let error msg = { InlineOps = []; TopLevelOps = []; Result = TRError (Diagnostic.errorSimple msg) }

    /// Create error output with full diagnostic context
    let errorDiag diag = { InlineOps = []; TopLevelOps = []; Result = TRError diag }

    /// Skip this node (not handled by this nanopass)
    let skip = empty

    let withTopLevel topOps (output: WitnessOutput) : WitnessOutput = 
        { output with TopLevelOps = topOps @ output.TopLevelOps }
    let combine (a: WitnessOutput) (b: WitnessOutput) =
        { InlineOps = a.InlineOps @ b.InlineOps
          TopLevelOps = a.TopLevelOps @ b.TopLevelOps
          Result = b.Result }
    let combineAll outputs = outputs |> List.fold combine empty

// ═══════════════════════════════════════════════════════════════════════════
// WITNESS CONTEXT (What witnesses receive)
// ═══════════════════════════════════════════════════════════════════════════

/// Context passed to witnesses - the elegant single parameter
type WitnessContext = {
    Coeffects: TransferCoeffects
    Accumulator: MLIRAccumulator
    Graph: SemanticGraph
    Zipper: PSGZipper              // Navigation state (created ONCE by fold)
}

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT ACCESSORS (Convenience functions)
// ═══════════════════════════════════════════════════════════════════════════

/// Get single pre-assigned SSA for a node
let requireSSA (nodeId: NodeId) (ctx: WitnessContext) : SSA =
    match SSAAssign.lookupSSA nodeId ctx.Coeffects.SSA with
    | Some ssa -> ssa
    | None -> failwithf "No SSA for node %A" nodeId

/// Get all pre-assigned SSAs for a node
let requireSSAs (nodeId: NodeId) (ctx: WitnessContext) : SSA list =
    match SSAAssign.lookupSSAs nodeId ctx.Coeffects.SSA with
    | Some ssas -> ssas
    | None -> failwithf "No SSAs for node %A" nodeId

/// Get target architecture from coeffects
let targetArch (ctx: WitnessContext) : Architecture =
    ctx.Coeffects.Platform.TargetArch

/// CANONICAL TYPE MAPPING - the ONLY way to map NativeType to MLIRType
/// Uses graph-aware mapping that correctly handles records by looking up
/// field types from TypeDef nodes. ALL type mapping should go through this.
let mapType (ty: NativeType) (ctx: WitnessContext) : MLIRType =
    mapNativeTypeWithGraphForArch ctx.Coeffects.Platform.TargetArch ctx.Graph ty

/// Get platform-aware word width for string length, array length, etc.
let wordWidth (ctx: WitnessContext) : IntWidth =
    platformWordWidth ctx.Coeffects.Platform.TargetArch
