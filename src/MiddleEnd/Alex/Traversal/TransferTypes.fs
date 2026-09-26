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

open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.NativeTypedTree.NativeTypes
open Alex.Dialects.Core.Types
open Alex.CodeGeneration.TypeMapping
open Alex.Traversal.PSGZipper
open Alex.Traversal.ScopeContext

// ═══════════════════════════════════════════════════════════════════════════
// MODULE ALIASES (for type definitions)
// ═══════════════════════════════════════════════════════════════════════════


// ═══════════════════════════════════════════════════════════════════════════
// TRANSFER COEFFECTS (Pre-computed, Immutable)
// ═══════════════════════════════════════════════════════════════════════════

/// The platform as emission reads it: the instruction set and the declared Register and Pointer
/// widths (from the CCS context), and the call-site resolutions the graph carries (Codata.Bindings).
type PlatformReads = {
    LinkedLibraries: Set<string>
    TargetArch: Architecture
    Bindings: PlatformBindings
}
with
    /// The platform word type: the declared Register width as an MLIR integer.
    member this.PlatformWordType : MLIRType = TInt (declaredWordWidth this.TargetArch)

/// What the traversal carries beside the graph. Every fact about the program is read from the
/// graph (its nodes, layouts, ranges and Codata); these are the target and the platform reads.
type TransferCoeffects = {
    Platform: PlatformReads
    /// Target platform — determines which MLIR dialects Patterns emit
    /// CPU → func/arith/scf, FPGA → hw/comb/seq (codata-dependent elision)
    TargetPlatform: Core.Types.Dialects.TargetPlatform
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

/// Formal Alex error codes — AX prefix, 4-digit numeric
/// Ranges:
///   AX1xxx — Type resolution errors
///   AX2xxx — Witness/traversal errors
///   AX3xxx — SSA/accumulator errors
///   AX4xxx — Pattern match errors
///   AX5xxx — Scope/control flow errors
type AlexErrorCode =
    // AX1xxx: Type resolution
    | AX1001  // Unbound type variable
    | AX1002  // Type mapping failure

    // AX2xxx: Witness/traversal
    | AX2001  // Arguments not yet witnessed (accumulator recall failure)
    | AX2002  // Function node not resolved
    | AX2003  // Direct call pattern failure
    | AX2004  // Indirect call pattern failure

    // AX3xxx: SSA/accumulator
    | AX3001  // SSA count mismatch
    | AX3002  // SSA not found in coeffects

    // AX4xxx: Pattern match
    | AX4001  // Pattern emission failure

    // AX5xxx: Scope/control flow
    | AX5001  // Scope isolation failure

module AlexErrorCode =
    let format (code: AlexErrorCode) : string =
        match code with
        | AX1001 -> "AX1001" | AX1002 -> "AX1002"
        | AX2001 -> "AX2001" | AX2002 -> "AX2002" | AX2003 -> "AX2003" | AX2004 -> "AX2004"
        | AX3001 -> "AX3001" | AX3002 -> "AX3002"
        | AX4001 -> "AX4001"
        | AX5001 -> "AX5001"

/// Structured diagnostic capturing WHERE and WHAT went wrong
type Diagnostic = {
    /// Severity level
    Severity: DiagnosticSeverity

    /// Formal error code
    Code: AlexErrorCode option

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
          Code = None
          NodeId = nodeId
          Source = source
          Phase = phase
          Message = message
          Details = None }

    /// Create an error diagnostic with formal error code and full context
    let coded code nodeId source phase message =
        { Severity = Error
          Code = Some code
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
          Code = None
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

            // Error code
            match diag.Code with
            | Some code -> Some (AlexErrorCode.format code)
            | None -> None

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

/// A callable's code and actual environment are distinct typed SSA operands.
/// Construction is internal to the graph-reading callable projection; there is
/// no scalar/packed fallback and code-only values carry no environment operand.
type CallableBoundary =
    | Exact of CallableCarrier
    | Joined of CallableJoin
    | Flow of CallableFlow
    member this.Occurrence = match this with Exact value -> value.Occurrence | Joined value -> value.Occurrence | Flow value -> value.Occurrence
    member this.SourceType = match this with Exact value -> value.SourceType | Joined value -> value.SourceType | Flow value -> value.SourceType

type CallableOperand = internal {
    Carrier: CallableBoundary
    Code: Val
    Environment: Val option
}

/// Only data descriptors reside in a mutable callable cell. The source
/// contract supplies the finite dispatch which reads its function value.
type CallableCellOperand = internal {
    Contract: MutableCallableStorage
    Discriminator: Val
    Environment: Val option
}

/// A sequence keeps the independently typed pull function and its actual
/// environment. The source family fixes layout; it never selects an instance.
type SequenceOperand = internal {
    Flow: SequenceFlow
    Family: SequenceFamily
    Code: Val
    Environment: Val
}

/// An explicit lazy occurrence retains its separately typed thunk and actual
/// instance environment. Layout identity alone never identifies that instance.
type LazyOperand = internal {
    Occurrence: NodeId
    Layout: LazyLayout
    Code: Val
    Environment: Val
}

/// A successfully witnessed void result at one graph occurrence and operation
/// scope. This is emission completion, not source demand or global visitation.
type VoidCompletion = internal {
    Graph: SemanticGraph
    Path: (NodeId * NodeId list * NodeId list) list
    Scope: ScopeContext ref
}

/// One operation scope's complete operand reading. All three maps must move
/// together when a shared graph body is witnessed at a different occurrence.
type OperandSnapshot = {
    Scalars: Map<NodeId, SSA * MLIRType>
    Callables: Map<NodeId, CallableOperand>
    CallableCells: Map<NodeId, CallableCellOperand>
    Sequences: Map<NodeId, SequenceOperand>
    Lazies: Map<NodeId, LazyOperand>
    Voids: Map<NodeId, VoidCompletion>
    Types: Map<SSA, MLIRType>
}

/// Flat accumulator - all operations in single stream with scope markers
/// SSA bindings are global (shared across all witnesses and scopes)
/// NOTE: Visited set is NOT in accumulator - each nanopass gets its own visited set
/// CRITICAL: This is a CLASS (reference type) not a record, so mutations propagate correctly
[<AllowNullLiteral>]
type MLIRAccumulator() =
    member val AllOps: MLIROp list = [] with get, set                      // Flat operation stream with markers
    member val Errors: Diagnostic list = [] with get, set
    member val NodeAssoc: Map<NodeId, SSA * MLIRType> = Map.empty with get, set  // Global SSA bindings (PSG nodes)
    member val CallableAssoc: Map<NodeId, CallableOperand> = Map.empty with get, set
    member val CallableCellAssoc: Map<NodeId, CallableCellOperand> = Map.empty with get, set
    member val SequenceAssoc: Map<NodeId, SequenceOperand> = Map.empty with get, set
    member val LazyAssoc: Map<NodeId, LazyOperand> = Map.empty with get, set
    member val VoidAssoc: Map<NodeId, VoidCompletion> = Map.empty with get, set
    member val SSATypes: Map<SSA, MLIRType> = Map.empty with get, set            // SSA → type reverse index (for monadic type derivation in Elements)

    // Witnessing Coordination State (Dependent Transparency)
    member val EmittedGlobals: Set<string> = Set.empty with get, set              // Track emitted global strings (by symbol name)
    member val EmittedStaticGlobals: Map<string, MLIRType * ProgramStorageEntry option> = Map.empty with get, set
    member val PendingStaticGlobals: MLIROp list = [] with get, set                // memref.global decls emitted by a parser, awaiting drain to TopLevelOps by the witness (module-scope placement)
    // NOTE: Function declarations now handled by MLIR Declaration Collection Pass (no coordination needed)

    // Deferred InlineOps: Partial app arguments whose InlineOps are suppressed at their
    // original scope and re-emitted at the saturated call site (MLIR region isolation)
    member val DeferredInlineOps: System.Collections.Generic.Dictionary<int, MLIROp list> = System.Collections.Generic.Dictionary<int, MLIROp list>() with get
    // Emission history is not operand state: restoring a value scope must not
    // hide an operation withheld anywhere while a subtree was witnessed.
    member val DeferredEmissionStamp: obj = obj() with get, set

module MLIRAccumulator =
    let empty () : MLIRAccumulator =
        MLIRAccumulator()

    /// Add a single operation to the flat stream
    let addOp (op: MLIROp) (acc: MLIRAccumulator) =
        acc.AllOps <- op :: acc.AllOps

    /// Add multiple operations to the flat stream
    let addOps (ops: MLIROp list) (acc: MLIRAccumulator) =
        acc.AllOps <- List.rev ops @ acc.AllOps

    /// Add an error diagnostic
    let addError (err: Diagnostic) (acc: MLIRAccumulator) =
        acc.Errors <- err :: acc.Errors

    /// Bind a PSG node to its SSA value (global binding)
    /// Also populates SSATypes reverse index for monadic type derivation in Elements
    let bindNode (nodeId: NodeId) (ssa: SSA) (ty: MLIRType) (acc: MLIRAccumulator) =
        acc.VoidAssoc <- acc.VoidAssoc.Remove nodeId
        acc.CallableAssoc <- acc.CallableAssoc.Remove nodeId
        acc.CallableCellAssoc <- acc.CallableCellAssoc.Remove nodeId
        acc.SequenceAssoc <- acc.SequenceAssoc.Remove nodeId
        acc.LazyAssoc <- acc.LazyAssoc.Remove nodeId
        acc.NodeAssoc <- Map.add nodeId (ssa, ty) acc.NodeAssoc
        // Preserve physical SSA type if already registered by an Element (pAlloca, pAlloc, etc.)
        // Elements register physical types (TMemRefStatic from alloca); bindNode carries semantic types
        // (TMemRef for mutable cells). pLoad/pLoadFrom derive memrefType from SSATypes.
        match Map.tryFind ssa acc.SSATypes with
        | Some existingTy when existingTy <> ty ->
            // Type collision detected — same SSA registered with different type.
            // Benign case: TMemRefStatic(n, elem) vs TMemRef(elem) — physical vs semantic type.
            // Elements (pAlloca) register physical TMemRefStatic; bindNode carries semantic TMemRef.
            // Physical type is correct for pLoadFrom — preserve it silently.
            // Real collision: fundamentally different types — indicates SSATypes scope leak.
            let isBenignMemRefRefinement =
                match existingTy, ty with
                | TMemRefStatic (_, elemA), TMemRef elemB when elemA = elemB -> true
                | TMemRef elemA, TMemRefStatic (_, elemB) when elemA = elemB -> true
                // Physical TMemRefStatic from alloca → logical TStruct from witness (record types)
                // Prefer TStruct: it carries field names needed by RecordWitness FieldGet
                | TMemRefStatic _, TStruct _ ->
                    acc.SSATypes <- Map.add ssa ty acc.SSATypes
                    true
                | _ -> false
            if not isBenignMemRefRefinement then
                let ssaStr = Alex.Dialects.Core.Serialize.ssaToString ssa
                let diag = Diagnostic.errorWithDetails (Some nodeId) (Some "SSATypes") (Some "bindNode")
                            (sprintf "SSA type collision: %s already registered as %A, new type %A (keeping existing)" ssaStr existingTy ty)
                            (sprintf "%A" existingTy) (sprintf "%A" ty)
                acc.Errors <- diag :: acc.Errors
        | Some _ -> () // Same type — no conflict
        | None ->
            acc.SSATypes <- Map.add ssa ty acc.SSATypes

    /// Recall the SSA binding for a PSG node (global lookup)
    let recallNode (nodeId: NodeId) (acc: MLIRAccumulator) =
        Map.tryFind nodeId acc.NodeAssoc

    /// Bind a graph-projected callable atomically. A mismatched physical SSA
    /// must not leave a half-pair or silently replace its registered type.
    let bindCallable (nodeId: NodeId) (value: CallableOperand) (acc: MLIRAccumulator) =
        let operands = value.Code :: Option.toList value.Environment
        if value.Carrier.Occurrence <> nodeId then
            Result.Error "Callable operand belongs to a different source occurrence."
        elif operands |> List.exists (fun value ->
            acc.SSATypes.TryFind value.SSA |> Option.exists (fun ty -> ty <> value.Type)) then
            Result.Error "Callable operand conflicts with an already witnessed SSA type."
        else
            acc.NodeAssoc <- acc.NodeAssoc.Remove nodeId
            acc.CallableCellAssoc <- acc.CallableCellAssoc.Remove nodeId
            acc.SequenceAssoc <- acc.SequenceAssoc.Remove nodeId
            acc.LazyAssoc <- acc.LazyAssoc.Remove nodeId
            acc.CallableAssoc <- acc.CallableAssoc.Add(nodeId, value)
            acc.VoidAssoc <- acc.VoidAssoc.Remove nodeId
            for value in operands do acc.SSATypes <- acc.SSATypes.Add(value.SSA, value.Type)
            Result.Ok ()

    let recallCallable nodeId (acc: MLIRAccumulator) = acc.CallableAssoc.TryFind nodeId

    let bindCallableCell nodeId (value: CallableCellOperand) (acc: MLIRAccumulator) =
        let operands = value.Discriminator :: Option.toList value.Environment
        let expectedEnvironment = value.Contract.EnvironmentBytes |> Option.map (fun bytes ->
            TMemRefStatic(1, TMemRefStatic(bytes, TInt(IntWidth 8))))
        if value.Contract.Binding <> nodeId || value.Discriminator.Type <> TMemRefStatic(1, TIndex) ||
           Option.map (fun (environment: Val) -> environment.Type) value.Environment <> expectedEnvironment then
            Result.Error "Mutable callable cell does not match its settled source protocol."
        elif operands |> List.exists (fun value ->
            acc.SSATypes.TryFind value.SSA |> Option.exists (fun ty -> ty <> value.Type)) then
            Result.Error "Mutable callable cell conflicts with an already witnessed SSA type."
        else
            acc.NodeAssoc <- acc.NodeAssoc.Remove nodeId
            acc.CallableAssoc <- acc.CallableAssoc.Remove nodeId
            acc.SequenceAssoc <- acc.SequenceAssoc.Remove nodeId
            acc.LazyAssoc <- acc.LazyAssoc.Remove nodeId
            acc.CallableCellAssoc <- acc.CallableCellAssoc.Add(nodeId, value)
            acc.VoidAssoc <- acc.VoidAssoc.Remove nodeId
            for value in operands do acc.SSATypes <- acc.SSATypes.Add(value.SSA, value.Type)
            Result.Ok ()

    let recallCallableCell nodeId (acc: MLIRAccumulator) = acc.CallableCellAssoc.TryFind nodeId

    let bindSequence nodeId (value: SequenceOperand) (acc: MLIRAccumulator) =
        let operands = [value.Code; value.Environment]
        if value.Flow.Occurrence <> nodeId then
            Result.Error "Sequence operand belongs to a different source occurrence."
        elif operands |> List.exists (fun operand ->
            acc.SSATypes.TryFind operand.SSA |> Option.exists (fun ty -> ty <> operand.Type)) then
            Result.Error "Sequence operand conflicts with an already witnessed SSA type."
        else
            acc.NodeAssoc <- acc.NodeAssoc.Remove nodeId
            acc.CallableAssoc <- acc.CallableAssoc.Remove nodeId
            acc.CallableCellAssoc <- acc.CallableCellAssoc.Remove nodeId
            acc.LazyAssoc <- acc.LazyAssoc.Remove nodeId
            acc.SequenceAssoc <- acc.SequenceAssoc.Add(nodeId, value)
            acc.VoidAssoc <- acc.VoidAssoc.Remove nodeId
            for operand in operands do acc.SSATypes <- acc.SSATypes.Add(operand.SSA, operand.Type)
            Result.Ok ()

    let recallSequence nodeId (acc: MLIRAccumulator) = acc.SequenceAssoc.TryFind nodeId

    let bindLazy nodeId (value: LazyOperand) (acc: MLIRAccumulator) =
        let operands = [value.Code; value.Environment]
        if value.Occurrence <> nodeId then
            Result.Error "Lazy operand belongs to a different source occurrence."
        elif operands |> List.exists (fun operand ->
            acc.SSATypes.TryFind operand.SSA |> Option.exists (fun ty -> ty <> operand.Type)) then
            Result.Error "Lazy operand conflicts with an already witnessed SSA type."
        else
            acc.NodeAssoc <- acc.NodeAssoc.Remove nodeId
            acc.CallableAssoc <- acc.CallableAssoc.Remove nodeId
            acc.CallableCellAssoc <- acc.CallableCellAssoc.Remove nodeId
            acc.SequenceAssoc <- acc.SequenceAssoc.Remove nodeId
            acc.LazyAssoc <- acc.LazyAssoc.Add(nodeId, value)
            acc.VoidAssoc <- acc.VoidAssoc.Remove nodeId
            for operand in operands do acc.SSATypes <- acc.SSATypes.Add(operand.SSA, operand.Type)
            Result.Ok ()

    let recallLazy nodeId (acc: MLIRAccumulator) = acc.LazyAssoc.TryFind nodeId

    let private occurrencePath (position: PSGZipper) =
        position.Path |> List.map (fun step -> step.Parent.Id, step.LeftSiblings, step.RightSiblings)

    let forgetVoid nodeId (acc: MLIRAccumulator) = acc.VoidAssoc <- acc.VoidAssoc.Remove nodeId

    let completeVoid (position: PSGZipper) scope (acc: MLIRAccumulator) =
        let nodeId = position.Focus.Id
        // Transparent witnesses may bind a value directly and return TRVoid to
        // indicate that they emitted no operations. Preserve that result.
        if not (acc.NodeAssoc.ContainsKey nodeId || acc.CallableAssoc.ContainsKey nodeId ||
                acc.CallableCellAssoc.ContainsKey nodeId || acc.SequenceAssoc.ContainsKey nodeId ||
                acc.LazyAssoc.ContainsKey nodeId) then
            acc.VoidAssoc <- acc.VoidAssoc.Add(nodeId, { Graph = position.Graph; Path = occurrencePath position; Scope = scope })

    let completedVoid (position: PSGZipper) scope (acc: MLIRAccumulator) =
        acc.VoidAssoc.TryFind position.Focus.Id |> Option.exists (fun completion ->
            obj.ReferenceEquals(completion.Graph, position.Graph) &&
            obj.ReferenceEquals(completion.Scope, scope) &&
            completion.Path = occurrencePath position)

    let snapshotOperands (acc: MLIRAccumulator) =
        { Scalars = acc.NodeAssoc; Callables = acc.CallableAssoc; CallableCells = acc.CallableCellAssoc
          Sequences = acc.SequenceAssoc; Lazies = acc.LazyAssoc; Voids = acc.VoidAssoc; Types = acc.SSATypes }

    let restoreOperands (snapshot: OperandSnapshot) (acc: MLIRAccumulator) =
        acc.NodeAssoc <- snapshot.Scalars
        acc.CallableAssoc <- snapshot.Callables
        acc.CallableCellAssoc <- snapshot.CallableCells
        acc.SequenceAssoc <- snapshot.Sequences
        acc.LazyAssoc <- snapshot.Lazies
        acc.VoidAssoc <- snapshot.Voids
        acc.SSATypes <- snapshot.Types

    /// Recall the type of an SSA value (reverse index lookup)
    /// Used by Elements (e.g. pLoad) to derive memref types monadically from the accumulator
    let recallSSAType (ssa: SSA) (acc: MLIRAccumulator) =
        Map.tryFind ssa acc.SSATypes

    /// Register an SSA value's type directly (for intermediate SSAs created by Elements)
    /// Called by Elements like pAlloca/pAlloc that create new SSAs not bound to PSG nodes
    let registerSSAType (ssa: SSA) (ty: MLIRType) (acc: MLIRAccumulator) =
        acc.SSATypes <- Map.add ssa ty acc.SSATypes

    // ═══════════════════════════════════════════════════════════
    // WITNESSING COORDINATION (Dependent Transparency Support)
    // ═══════════════════════════════════════════════════════════

    /// Try to emit a global string (returns Some op if not already emitted, None if duplicate)
    /// This implements dependent transparency coordination: witnesses check before emitting module-level declarations
    let tryEmitGlobal (name: string) (content: string) (byteLength: int) (obligations: string list) (acc: MLIRAccumulator) : MLIROp option =
        if Set.contains name acc.EmittedGlobals then
            None  // Already emitted by another witness
        else
            acc.EmittedGlobals <- Set.add name acc.EmittedGlobals
            Some (MLIROp.GlobalString (name, content, byteLength, obligations))


    /// Register a module-level memref.global static-storage decl for a program-lifetime value,
    /// deduplicated by symbol name. A memref.global is only valid at module scope, but this is
    /// called from the PSGParser layer (which has no module-scope handle), so the decl is queued
    /// in PendingStaticGlobals on the shared accumulator; the owning witness drains it to its
    /// WitnessOutput.TopLevelOps (which the nanopass driver places at module root). The caller
    /// emits the matching memref.get_global inline. Idempotent per symbol name.
    let tryEmitGlobalMemref (name: string) (declType: MLIRType) (authority: ProgramStorageEntry option) (acc: MLIRAccumulator) : unit =
        match acc.EmittedStaticGlobals.TryFind name with
        | Some(oldType, oldAuthority) when oldType = declType && oldAuthority = authority -> ()
        | Some _ -> failwithf "Writable global '%s' has conflicting type or source storage identity" name
        | None ->
            acc.EmittedStaticGlobals <- acc.EmittedStaticGlobals.Add(name, (declType, authority))
            acc.PendingStaticGlobals <- MLIROp.GlobalMemref (name, declType, authority) :: acc.PendingStaticGlobals

    /// Drain any pending memref.global decls queued during parser emission, clearing the queue.
    /// The witness routes the returned ops into WitnessOutput.TopLevelOps for module-scope placement.
    let drainPendingStaticGlobals (acc: MLIRAccumulator) : MLIROp list =
        let pending = acc.PendingStaticGlobals
        acc.PendingStaticGlobals <- []
        pending

    /// Store deferred InlineOps for a node (suppressed at original scope, re-emitted at saturated call site)
    let deferInlineOps (nodeId: NodeId) (ops: MLIROp list) (acc: MLIRAccumulator) =
        let key = NodeId.value nodeId
        acc.DeferredInlineOps.[key] <- ops
        if not ops.IsEmpty then acc.DeferredEmissionStamp <- obj()

    /// Retrieve deferred InlineOps for a node (returns empty list if none)
    let getDeferredInlineOps (nodeId: NodeId) (acc: MLIRAccumulator) : MLIROp list =
        let key = NodeId.value nodeId
        match acc.DeferredInlineOps.TryGetValue(key) with
        | true, ops -> ops
        | false, _ -> []

    /// NOTE: Function declaration coordination removed - now handled by MLIR Declaration Collection Pass
    /// This eliminates "first witness wins" race condition and separates concerns:
    /// - Witnesses emit FuncCall operations (codata)
    /// - Declaration Collection Pass analyzes calls and emits FuncDecl (structural MLIR transformation)

    /// NOTE: Scope markers removed - single-phase execution with nested accumulators
    /// Scope-owning witnesses (Lambda, ControlFlow) create nested accumulators for body operations.
    /// Operations naturally nest; bindings remain global for cross-scope lookups.

    /// Backward compatibility aliases
    let addTopLevelOp = addOp
    let addTopLevelOps = addOps

    /// Property accessor for compatibility
    let topLevelOps (acc: MLIRAccumulator) = acc.AllOps

    /// Recursively count all operations (including nested in FuncDef, SCFOp, etc.)
    let rec countOperations (ops: MLIROp list) : int =
        ops |> List.sumBy (fun op ->
            match op with
            | MLIROp.FuncOp (FuncOp.FuncDef (_, _, _, body, _)) ->
                1 + countOperations body
            | MLIROp.SCFOp (SCFOp.If (_, thenOps, elseOps, _)) ->
                let elseCount = match elseOps with Some ops -> countOperations ops | None -> 0
                1 + countOperations thenOps + elseCount
            | MLIROp.SCFOp (SCFOp.While (condOps, bodyOps)) ->
                1 + countOperations condOps + countOperations bodyOps
            | MLIROp.SCFOp (SCFOp.For (_, _, _, bodyOps)) ->
                1 + countOperations bodyOps
            | MLIROp.SCFOp (SCFOp.IndexSwitch (_, cases, fallback, _)) ->
                1 + countOperations ((cases |> List.collect snd) @ fallback)
            | MLIROp.Block (_, blockOps) ->
                1 + countOperations blockOps
            | MLIROp.Region ops ->
                1 + countOperations ops
            | _ -> 1)

    /// Get total operation count from accumulator (including all nested operations)
    let totalOperations (acc: MLIRAccumulator) : int =
        countOperations acc.AllOps

// ═══════════════════════════════════════════════════════════════════════════
// TRANSFER RESULT (Result of witnessing a node)
// ═══════════════════════════════════════════════════════════════════════════

/// Result of witnessing a PSG node
type TransferResult =
    | TRValue of Val                    // Produces a value (SSA + type)
    | TRCallable of CallableOperand    // Code and actual environment remain separate operands
    | TRCallableCell of CallableCellOperand // Shared data storage, never a packed function value
    | TRSequence of SequenceOperand    // Pull function and actual continuation environment
    | TRLazy of LazyOperand            // Thunk function and actual memoization environment
    | TRVoid                             // Produces no value (effect only)
    | TRError of Diagnostic              // Error with structured context
    | TRSkip                             // Node not handled (try next witness)

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

    /// Create error output with formal error code and full context
    let errorCoded code nodeId source phase msg =
        { InlineOps = []; TopLevelOps = []; Result = TRError (Diagnostic.coded code nodeId source phase msg) }

    /// Create error output with full diagnostic context
    let errorDiag diag = { InlineOps = []; TopLevelOps = []; Result = TRError diag }

    /// Skip this node (not handled by this nanopass)
    let skip = { InlineOps = []; TopLevelOps = []; Result = TRSkip }

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
    Accumulator: MLIRAccumulator     // Shared for SSA bindings (global)
    RootAccumulator: MLIRAccumulator // Root/module-level accumulator (constant across all scopes)
    ScopeContext: ref<ScopeContext>  // Current scope for operation accumulation (mutable for traversal)
    RootScopeContext: ref<ScopeContext>  // Root module-level scope (constant, for TopLevelOps like GlobalString, nested FuncDef)
    Graph: SemanticGraph
    Zipper: PSGZipper                // Navigation state (created ONCE by fold)
    GlobalVisited: ref<Set<NodeId>>  // Global visited set (shared across all nanopasses and function bodies)
    TraversalVisited: ref<Set<NodeId>>  // Traversal visited set: global on CPU, per-function on FPGA
}

// ═══════════════════════════════════════════════════════════════════════════
// MODULE-LEVEL VALUE SLOTS
// ═══════════════════════════════════════════════════════════════════════════

/// A module-level `let` value (EmissionStrategy.MainPrologue, not a function) lives in a
/// program-lifetime slot: a one-element memref.global. Its initializer runs in the entry
/// point's prologue and every reference, in any function, reloads from the slot, so no SSA
/// value ever crosses a function boundary. Function values (Lambda children) are emitted as
/// functions and are not slots.
module ModuleValues =
    /// One shared observation determines both slot value naming and access.
    let isSlotBinding = Alex.Traversal.Values.isModuleValueSlot

    /// The memref.global symbol for a slot: readable name plus the binding's node id for uniqueness.
    let globalName (bindingName: string) (bindingId: NodeId) : string =
        let sanitized =
            bindingName
            |> String.map (fun c -> if System.Char.IsLetterOrDigit c || c = '_' then c else '_')
        sprintf "__clef_module_value_%s_%d" sanitized (NodeId.value bindingId)

// ═══════════════════════════════════════════════════════════════════════════
// COEFFECT ACCESSORS (Convenience functions)
// ═══════════════════════════════════════════════════════════════════════════

/// The result value a node names (Alex.Traversal.Values)
let requireSSA (nodeId: NodeId) (ctx: WitnessContext) : SSA =
    Alex.Traversal.Values.resultOf ctx.Coeffects.TargetPlatform ctx.Graph nodeId

/// The values a node names (Alex.Traversal.Values)
let requireSSAs (nodeId: NodeId) (ctx: WitnessContext) : SSA list =
    Alex.Traversal.Values.valuesOf ctx.Coeffects.TargetPlatform ctx.Graph nodeId

/// The escape kind of an allocating site (Codata.Escapes); stack-scoped where the graph records none.
let escapeOf (graph: SemanticGraph) (nodeId: NodeId) : EscapeKind =
    graph.Codata.Value.Escapes |> Map.tryFind nodeId |> Option.defaultValue EscapeKind.StackScoped

/// The meet the graph derived for a consumer's operand, with the value emission names for it;
/// None where the widths agree. A read of a slot names the consumer as its own operand.
let meetFor (graph: SemanticGraph) (consumer: NodeId) (operand: NodeId) : (Meet * SSA) option =
    graph.Codata.Value.Meets
    |> Map.tryFind consumer
    |> Option.bind (fun meets ->
        meets |> List.tryFindIndex (fun m -> m.Operand = operand)
        |> Option.map (fun i -> meets.[i], Alex.Traversal.Values.meetValue consumer i))

/// Get target architecture from coeffects
let targetArch (ctx: WitnessContext) : Architecture =
    ctx.Coeffects.Platform.TargetArch

/// Witness-layer type mapping — delegates to mapNativeTypeForTarget.
/// Extracts platform, architecture, and graph from WitnessContext.
/// Drains any AX1001 diagnostics collected during mapping into the accumulator.
let mapType (ty: NativeType) (ctx: WitnessContext) : MLIRType =
    let result = mapNativeTypeForTarget ctx.Coeffects.TargetPlatform ctx.Coeffects.Platform.TargetArch ctx.Graph ty
    // Drain type mapping diagnostics (AX1001: unbound TVar).
    // These are non-fatal — CCS may leave type variables unresolved for unused bindings
    // (e.g. `| Error err -> ...` where err is never referenced). Alex continues with TIndex.
    // CCS diagnostics surface these at the appropriate level; Alex doesn't re-report.
    drainTypeMappingErrors () |> ignore
    result

/// A value's specialized physical carrier is keyed by its exact graph use.
/// Source type alone cannot name a continuation frame or a settled byte buffer.
let mapTypeAt (nodeId: NodeId) (ty: NativeType) (ctx: WitnessContext) : MLIRType =
    let codata = ctx.Graph.Codata.Value
    match codata.LazyOrigins.TryFind nodeId, codata.EnvironmentOrigins.TryFind nodeId, codata.SequenceOrigins.TryFind nodeId with
    | Some _, _, _ when (match ty with NativeType.TLazy _ -> true | _ -> false) ->
        failwithf "Lazy value %d requires its separately witnessed thunk and environment operands" (NodeId.value nodeId)
    | Some owner, _, _ ->
        match codata.LazyLayouts.TryFind owner with
        | Some layout when layout.Bytes > 0 && layout.Alignment > 0 -> TMemRefStatic(layout.Bytes, TInt(IntWidth 8))
        | _ -> failwithf "Lazy environment %d has no settled layout %d" (NodeId.value nodeId) (NodeId.value owner)
    | None, Some owner, _ ->
        match codata.EnvironmentLayouts |> Map.tryFind owner with
        | Some layout when layout.Bytes >= 0 && layout.Alignment > 0 -> TMemRefStatic(layout.Bytes, TInt(IntWidth 8))
        | _ -> failwithf "Environment value %d has no settled layout %d" (NodeId.value nodeId) (NodeId.value owner)
    | None, None, Some owner ->
        match ctx.Graph.Codata.Value.ContinuationFrames |> Map.tryFind owner with
        | Some frame when frame.Bytes > 0 -> TMemRefStatic(frame.Bytes, TInt(IntWidth 8))
        | _ -> failwithf "Sequence value %d has no settled frame for origin %d" (NodeId.value nodeId) (NodeId.value owner)
    | None, None, None ->
        match tryArrayElementTypeAt ctx.Graph nodeId with
        | Some element -> TMemRef element
        | None -> mapType ty ctx

/// Get platform-aware word width for string length, array length, etc.
let wordWidth (ctx: WitnessContext) : IntWidth =
    declaredWordWidth ctx.Coeffects.Platform.TargetArch
