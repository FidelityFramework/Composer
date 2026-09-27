/// Snapshot-scoped correspondence. Semantic scope comes from the checked PSG;
/// a backend partition is never authority to choose a source invalidation region.
module Core.Types.WitnessArtifacts

open System
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Alex.Dialects.Core.Types

/// The current full-check/full-witness pipeline supplies the checked graph.
/// witnessRun distinguishes emission bookkeeping, not an accepted source
/// revision or temporal freshness. Smaller scopes require a CCS/Baker contract;
/// a publication coordinator must separately check the accepted source revision.
[<NoEquality; NoComparison>]
type SemanticScope =
    | WholeCheckedGraph of witnessRun: Guid * graph: SemanticGraph

/// Actual external-tool evidence for one stage of this compile invocation.
/// These records describe solver-dependent checks, not Rocq certificates.
type ProofTool = {
    Executable: string
    Sha256: string
    Version: string
    Arguments: string list
}

type ProofOutcome = {
    Anchor: string
    Source: string
    Verdict: string
}

type ProofStageEvidence = {
    Invocation: Guid
    Stage: string
    InputSha256: string
    QuerySha256: string
    Tools: ProofTool list
    Outcomes: ProofOutcome list
    StandardOutput: string
    StandardError: string
}

/// Created only by the current-invocation source solver dispatch. Keeping the
/// actual graph object prevents a receipt authorizing a different PSG snapshot.
type SourceProofReceipt internal
    (graph: SemanticGraph, obligations: ObligationInfo list, query: string, evidence: ProofStageEvidence) =
    member _.Graph = graph
    member _.Obligations = obligations
    member _.Query = query
    member _.Evidence = evidence

/// Created only after the witnessed MLIR has been exported and checked by cvc5.
type MlirProofReceipt internal
    (scope: SemanticScope, source: SourceProofReceipt, operations: MLIROp list,
     text: string, query: string, evidence: ProofStageEvidence) =
    member _.Scope = scope
    member _.Source = source
    member _.Operations = operations
    member _.Text = text
    member _.Query = query
    member _.Evidence = evidence

/// The complete source obligation inventory and its passive typed transcription.
/// No target realization may change or omit a required obligation.
[<NoEquality; NoComparison>]
type ProofEnvelope = {
    Scope: SemanticScope
    Source: SourceProofReceipt option
    Obligations: ObligationInfo list
    Operations: MLIROp list
    Text: string
    Mlir: MlirProofReceipt option
}

[<NoEquality; NoComparison>]
type Occurrence = {
    Scope: SemanticScope
    Focus: SemanticNode
    /// The actual traversal root captured with the path. Keeping it separate
    /// makes a truncated/re-rooted breadcrumb list observably different.
    Anchor: SemanticNode
    /// Actual Huet breadcrumbs, nearest parent first. Node.Parent is not used.
    Path: (SemanticNode * NodeId list * NodeId list) list
}

[<NoEquality; NoComparison>]
type EmittedDefinition = {
    Operation: MLIROp
    Occurrence: Occurrence
}

/// Exact typed declaration, including ABI attributes, not a signature inferred
/// from a call or recovered from serialized text.
type FunctionImport = {
    Symbol: string
    Arguments: MLIRType list
    Results: MLIRType list
    Visibility: FuncVisibility
    Byval: ByvalParam list
    /// Retained verbatim from Baker; physical signless types cannot encode this ABI.
    Boundary: BoundaryImport option
    IntrinsicWrite: IntrinsicWriteImport option
}

type ActivationOwnership =
    | CheckedProgramStartup
    | TargetModuleActivation

[<NoEquality; NoComparison>]
type Unit = {
    Id: string
    Scope: SemanticScope
    Definitions: EmittedDefinition list
    Imports: FunctionImport list
    SpatialModules: SpatialModuleWitness list
    WritableStorage: (string * ProgramStorageEntry) list
    Startup: (NodeId * string) option
    /// Existing target-owned opaque modules have source/content correspondence,
    /// but their internals do not acquire a typed symbol inventory by being here.
    HasOpaqueTargetContent: bool
    ContentHash: string
}

[<NoEquality; NoComparison>]
type Catalog = {
    Scope: SemanticScope
    Activation: ActivationOwnership
    /// Initially exactly one unit: no cache, partial-check or reuse promise.
    Units: Unit list
    /// Isolated physical correspondence fixtures may omit this. Production
    /// requires it whenever the current graph carries proof obligations.
    Proof: ProofEnvelope option
}
