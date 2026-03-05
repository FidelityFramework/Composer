/// CCS Integration Layer
/// Thin interface between Composer and F# Native Compiler Services.
///
/// CCS provides:
/// - Native type checking with types attached during construction
/// - SRTP resolution during type checking (not post-hoc)
/// - Hard-pruned SemanticGraph (only reachable nodes)
/// - No BCL types, no IL imports, no obj
/// - Baker enrichment (module classification metadata)
///
/// Composer receives SemanticGraph and applies:
/// - Lowering nanopasses (FlattenApplications, LowerStrings, etc.)
/// - Alex emission (Zipper + XParsec + Bindings → MLIR)
module Core.CCS.Integration

// Re-export CCS types for use throughout Composer
open Clef.Compiler.NativeTypedTree.NativeTypes
open Clef.Compiler.PSGSaturation.SemanticGraph.Types
open Clef.Compiler.PSGSaturation.SemanticGraph.Core
open Clef.Compiler.PSGSaturation.SemanticGraph.Traversal
open Clef.Compiler.PSGSaturation.SemanticGraph.Diagnostics
open Clef.Compiler.NativeService
// Baker - Post-construction semantic enrichment (peer to NativeTypedTree)
// ModuleClassifications now computed lazily on SemanticGraph itself
module BakerPipeline = Clef.Compiler.Baker.Pipeline

/// Type aliases for cleaner code
type CCSNode = SemanticNode
type CCSGraph = SemanticGraph
type CCSType = NativeType
type CCSKind = SemanticKind
type CCSNodeId = NodeId
type CCSLiteralValue = NativeLiteral

/// Check result from CCS
type CCSCheckResult = CheckResult

/// Module classification (computed lazily on SemanticGraph)
type CCSModuleClassification = ModuleClassification

/// Extract int from NodeId
let nodeIdToInt (NodeId id) = id

/// Get the ID of a node
let nodeId (node: CCSNode) : CCSNodeId = node.Id

/// Get the type of a node
let nodeType (node: CCSNode) : CCSType = node.Type

/// Get the kind of a node
let nodeKind (node: CCSNode) : CCSKind = node.Kind

/// Get the SRTP resolution if present
let nodeSRTP (node: CCSNode) : WitnessResolution option = node.SRTPResolution

/// Get children of a node
let nodeChildren (node: CCSNode) : NodeId list = node.Children

/// Get a node by ID
let getNode (id: NodeId) (graph: CCSGraph) : CCSNode option =
    SemanticGraph.tryGetNode id graph

/// Get a node by ID (throws if not found)
let getNodeExn (id: NodeId) (graph: CCSGraph) : CCSNode =
    SemanticGraph.getNode id graph

/// Fold over graph in post-order (children before parents - required for SSA emission)
let foldPostOrderGraph (folder: 'State -> CCSNode -> 'State) (state: 'State) (graph: CCSGraph) : 'State =
    foldPostOrder folder state graph

/// Fold over graph in pre-order
let foldPreOrderGraph (folder: 'State -> CCSNode -> 'State) (state: 'State) (graph: CCSGraph) : 'State =
    foldPreOrder folder state graph

/// Map over all nodes in the graph
let mapNodes (f: CCSNode -> CCSNode) (graph: CCSGraph) : CCSGraph =
    map f graph

/// Filter nodes in the graph
let filterNodes (predicate: CCSNode -> bool) (graph: CCSGraph) : CCSGraph =
    filter predicate graph

/// Get all binding nodes
let bindings (graph: CCSGraph) : CCSNode list =
    SemanticGraph.bindings graph

/// Check if result has errors
let hasErrors (result: CCSCheckResult) : bool =
    CheckResult.hasErrors result

/// Get error diagnostics
let errors (result: CCSCheckResult) : Diagnostic list =
    CheckResult.errors result

/// Get warning diagnostics
let warnings (result: CCSCheckResult) : Diagnostic list =
    CheckResult.warnings result

/// Get platform context from check result (for Alex code generation)
let platformContext (result: CCSCheckResult) : PlatformContext option =
    result.PlatformContext

/// Format a type for display
let formatTypeStr (ty: CCSType) : string =
    formatType ty

/// Check if a node is a platform binding
let isPlatformBinding (node: CCSNode) : bool =
    match node.Kind with
    | SemanticKind.PlatformBinding _ -> true
    | _ -> false

/// Get platform binding name if present
let platformBindingName (node: CCSNode) : string option =
    match node.Kind with
    | SemanticKind.PlatformBinding name -> Some name
    | _ -> None

/// Check if a node is a function application
let isApplication (node: CCSNode) : bool =
    match node.Kind with
    | SemanticKind.Application _ -> true
    | _ -> false

/// Check if a node is a binding (let/do)
let isBinding (node: CCSNode) : bool =
    match node.Kind with
    | SemanticKind.Binding _ -> true
    | _ -> false

/// Get binding info (name, isMutable, isRecursive) if present
let bindingInfo (node: CCSNode) : (string * bool * bool) option =
    match node.Kind with
    | SemanticKind.Binding (name, isMutable, isRec, _declRoot) -> Some (name, isMutable, isRec)
    | _ -> None

/// Check if a node is a literal
let isLiteral (node: CCSNode) : bool =
    match node.Kind with
    | SemanticKind.Literal _ -> true
    | _ -> false

/// Get literal value if present
let literalValue (node: CCSNode) : NativeLiteral option =
    match node.Kind with
    | SemanticKind.Literal v -> Some v
    | _ -> None

/// Check if a node is a lambda
let isLambda (node: CCSNode) : bool =
    match node.Kind with
    | SemanticKind.Lambda _ -> true
    | _ -> false

/// Check if a node is a variable reference
let isVarRef (node: CCSNode) : bool =
    match node.Kind with
    | SemanticKind.VarRef _ -> true
    | _ -> false

/// Get variable reference name if present
let varRefName (node: CCSNode) : string option =
    match node.Kind with
    | SemanticKind.VarRef (name, _) -> Some name
    | _ -> None

/// Get variable reference definition node ID if present
let varRefDefinition (node: CCSNode) : CCSNodeId option =
    match node.Kind with
    | SemanticKind.VarRef (_, defId) -> defId
    | _ -> None

/// Check if type is a function type
let isFunType (ty: CCSType) : bool =
    isFunctionType ty

/// Check if type is a type variable
let isTypeVariable (ty: CCSType) : bool =
    isTypeVar ty

/// Check if a node is an interpolated string
let isInterpolatedString (node: CCSNode) : bool =
    match node.Kind with
    | SemanticKind.InterpolatedString _ -> true
    | _ -> false

/// Get interpolated string parts if present
let interpolatedStringParts (node: CCSNode) : InterpolatedPart list option =
    match node.Kind with
    | SemanticKind.InterpolatedString parts -> Some parts
    | _ -> None

/// Re-export InterpolatedPart type for use in Composer
type CCSInterpolatedPart = InterpolatedPart

// ═══════════════════════════════════════════════════════════════════════════
// Parsing API - Exposed for ProjectLoader
// ═══════════════════════════════════════════════════════════════════════════

/// Parse options for source files
type CCSParseOptions = ParseOptions

/// Parse result
type CCSParseResult = ParseResult

/// Combined parse and check result
type CCSParseAndCheckResult = ParseAndCheckResult

/// Default parse options (from NativeService module)
let defaultCCSParseOptions : CCSParseOptions = Clef.Compiler.NativeService.defaultParseOptions

/// Parse a source string with default options
let parseSource (source: string) (fileName: string) : CCSParseResult =
    parseStringWithDefaults source fileName

/// Parse a source string with custom options
let parseSourceWithOptions (source: string) (fileName: string) (options: CCSParseOptions) : CCSParseResult =
    parseString source fileName options

/// Parse and type-check a source string in one step
let parseAndCheckSource (source: string) (fileName: string) : CCSParseAndCheckResult =
    parseAndCheck source fileName

/// Check multiple parsed inputs together with shared type environment
/// This is the correct way to compile multi-file projects - type abbreviations,
/// bindings, etc. from earlier files are visible when checking later files.
let checkMultipleInputs (inputs: Clef.Compiler.Syntax.ParsedInput list) : CCSCheckResult =
    checkParsedInputs inputs

/// Check if parse result succeeded
let parseSucceeded (result: CCSParseResult) : bool =
    match result with
    | ParseSuccess _ -> true
    | ParseError _ -> false

/// Get parsed input from successful parse
let parsedInput (result: CCSParseResult) : Clef.Compiler.Syntax.ParsedInput option =
    match result with
    | ParseSuccess input -> Some input
    | ParseError _ -> None

/// Get parse errors
let parseErrors (result: CCSParseResult) : string list =
    match result with
    | ParseSuccess _ -> []
    | ParseError errs -> errs

// ═══════════════════════════════════════════════════════════════════════════
// Module Classification - Coeffect Observation (computed lazily on PSG)
// ═══════════════════════════════════════════════════════════════════════════

/// Emit Baker intermediates to a directory (for -k flag debug output).
let emitBakerIntermediates (graph: CCSGraph) (outputDir: string) (baseName: string) : unit =
    BakerPipeline.emitIntermediates graph outputDir baseName

/// Get module classification for a specific module (observes lazy field).
let getModuleClassification (moduleId: CCSNodeId) (graph: CCSGraph) : CCSModuleClassification option =
    Map.tryFind moduleId graph.ModuleClassifications.Value

/// Get all module classifications (observes lazy field).
let allModuleClassifications (graph: CCSGraph) : Map<CCSNodeId, CCSModuleClassification> =
    graph.ModuleClassifications.Value

let moduleInitBindings (classification: CCSModuleClassification) : CCSNodeId list =
    classification.ModuleInit

let moduleDefinitions (classification: CCSModuleClassification) : CCSNodeId list =
    classification.Definitions

let moduleDeclarationRoot (classification: CCSModuleClassification) : (CCSNodeId * DeclRoot) option =
    classification.DeclarationRoot
