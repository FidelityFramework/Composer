namespace Clef.Compiler.Editor

/// Zero-based lines and UTF-16 columns, with an exclusive end. No dummy ranges.
type SourceSpan = {
    FilePath: string
    StartLine: int
    StartCharacter: int
    EndLine: int
    EndCharacter: int
}

type SourceFile = { FilePath: string; Content: string }

type DiagnosticView = {
    Severity: string
    EffectiveSeverity: string
    Code: string
    Message: string
    Range: SourceSpan option
    RelatedNodeIds: int list
    Reachability: string
    IsUnnecessary: bool
}

/// The current CCS parser returns messages, not structured diagnostic ranges.
/// Preserve them separately; consumers must not manufacture source locations.
type ParseFailure = { FilePath: string; Messages: string list }

type NodeView = {
    NodeId: int
    Name: string option
    Kind: string
    Type: string
    Range: SourceSpan option
    IsReachable: bool
    ValueRange: string option
}

type HoverView = {
    NodeId: int
    Name: string option
    Kind: string
    Type: string
    Range: SourceSpan
    Definition: SourceSpan option
    IsReachable: bool
    ValueRange: string option
    ObligationIds: string list
}

/// A compiler-authored obligation and its exact design-time query. This record
/// contains no solver verdict; presence of an obligation is not a proof result.
type ObligationView = {
    Id: string
    NodeId: int
    Kind: string
    Logic: string
    Statement: string
    Source: string
    Refs: string list
    Range: SourceSpan option
    Premises: NodeView list
    SmtLib: string
    QueryHash: string
}

/// Only immutable display values leave a check. No compiler union-find cells,
/// mutable arrays, lazy graph computations, or compiler object references escape.
/// NodeId is meaningful only together with this snapshot's revision.
type ProgramInitializerView = {
    Ordinal: int
    Module: NodeView
    Binding: NodeView
    Value: NodeView
    RequiresProgramStorage: bool
    HasProgramAuthority: bool
}

type ProgramInitializationView = {
    Entry: NodeView
    SourceEntry: NodeView
    SpineNodeId: int
    EntryCallNodeId: int
    Initializers: ProgramInitializerView list
}

type ProgramInitializationPendingView = {
    Site: NodeView
    Sources: NodeView list
    Reason: string
}

type EditorSnapshot = {
    Revision: int64
    ProjectPath: string
    CompilerIdentity: string
    Sources: SourceFile list
    /// Exact local source and manifest paths selected by CCS project metadata.
    /// Includes known missing inputs on a failed check so repair can be watched.
    InputFiles: string list
    Diagnostics: DiagnosticView list
    ParseFailures: ParseFailure list
    Obligations: ObligationView list
    ProgramInitialization: ProgramInitializationView option
    ProgramInitializationPending: ProgramInitializationPendingView list
    Failure: string option
}
