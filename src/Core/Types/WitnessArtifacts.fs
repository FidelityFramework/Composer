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
}
