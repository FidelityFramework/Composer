# PSG Enrichment Architecture

> **Enrichment** is the parent concept encompassing **Elaboration** (PLT term) and **Saturation** (Fidelity term) - both describing the process of synthesizing PSG structure that wasn't written by the user.

## Overview

The Fidelity compiler transforms Clef source code to native binaries through a multi-phase pipeline. A critical aspect of this pipeline is **enrichment** - the process by which the Program Semantic Graph (PSG) gains structure beyond what the user explicitly wrote.

This document establishes the canonical terminology and architecture for enrichment in the Fidelity framework.

## Terminology: PLT vs Fidelity

### Elaboration (Programming Language Theory)

In PLT, **elaboration** refers to the process of "fleshing out" a program with implicit structure:
- Type annotations made explicit
- Implicit arguments inserted
- Syntactic sugar desugared
- Pattern matching expanded to decision trees

The term emphasizes making the implicit explicit - taking a surface-level representation and producing a fully-explicit internal representation.

### Saturation (Fidelity Framework)

**Saturation** is the Fidelity-specific term for filling the PSG "to the brim" before lowering:
- Every node has complete type information
- Every intrinsic operation has its implementation structure
- Every higher-order function has been decomposed to primitives
- Every closure has explicit capture representation

The metaphor is of a sponge absorbing water until it can hold no more - the PSG absorbs semantic information until it is fully saturated and ready for lowering to MLIR.

### Enrichment (The Parent Concept)

**Enrichment** encompasses both elaboration and saturation. When we say a node is "enriched," we mean:
1. It may have been elaborated (in the PLT sense) - e.g., pattern matching expanded
2. It may have been saturated (in the Fidelity sense) - e.g., intrinsic implementation synthesized

Both result in PSG structure that wasn't in the original source but is necessary for compilation.

## Two Kinds of Enrichment

The Fidelity compiler performs two distinct kinds of enrichment, tracked via metadata:

### 1. Intrinsic Elaboration (`Elaboration.Kind = "Intrinsic"`)

When the compiler encounters an intrinsic operation (defined in CCS), it may need to synthesize PSG structure to implement that operation's semantics.

**Examples:**
- `Console.write "Hello"` - The user wrote a simple call, but the intrinsic implementation requires:
  - String length extraction
  - Buffer pointer calculation
  - Syscall invocation with fd, buffer, length
  - Return value handling

- `a + b` for generic numeric types - SRTP resolution may require:
  - Type-specific operation dispatch
  - Overflow checking infrastructure (if enabled)

**Key characteristic:** The user wrote something high-level; the compiler synthesizes the low-level implementation structure.

### 2. Baker Saturation (`Elaboration.Kind = "Baker"`)

Baker is responsible for **decomposing language features to primitives**. This includes:
- HOF decomposition (List.map, Seq.fold)
- Seq expression state machines
- Lazy thunk structures
- Async workflows

**Examples:**
- `List.map f xs` decomposes to:
  - Empty check (`List.isEmpty`)
  - Recursive traversal (`List.head`, `List.tail`)
  - Result construction (`List.cons`)
  - Recursive call structure

- `seq { for x in xs do yield f x }` decomposes to:
  - State machine structure
  - State variable nodes
  - Yield point tracking
  - MoveNext implementation

- `lazy expr` decomposes to:
  - A separate thunk value and typed environment with computed state, cache and captures
  - Source-owned guarded computation, result storage, completion publication and hot read

**Key characteristic:** The user wrote a high-level expression; Baker synthesizes the implementation structure.

## Metadata Architecture

Enriched nodes carry metadata that enables "pierce the veil" debugging:

```fsharp
module ElaborationMetadata =
    /// What kind of enrichment: "Intrinsic" or "Baker"
    [<Literal>]
    let Kind = "Elaboration.Kind"

    /// What construct triggered enrichment (e.g., "List.map", "Console.write")
    [<Literal>]
    let For = "Elaboration.For"

    /// Links related nodes from the same enrichment expansion
    [<Literal>]
    let Id = "Elaboration.Id"
```

### Source-Based vs Enriched Nodes

| Node Type | Metadata | Meaning |
|-----------|----------|---------|
| Source-based | None | User wrote this directly |
| Intrinsic-elaborated | Kind="Intrinsic" | Synthesized to implement intrinsic semantics |
| Baker-saturated | Kind="Baker" | Synthesized for HOF decomposition |

### The Expansion ID Pattern

When enrichment creates multiple related nodes, they share an `Elaboration.Id`:

```
Node 42: Kind="Baker", For="List.map", Id=7
Node 43: Kind="Baker", For="List.map", Id=7   ← Same expansion
Node 44: Kind="Baker", For="List.map", Id=7   ← Same expansion
Node 45: (no metadata)                         ← Source-based
Node 46: Kind="Intrinsic", For="syscall", Id=8 ← Different expansion
```

This enables tooling to:
- Group related enriched nodes
- Show expansion boundaries in debugging views
- Reconstruct the "user wrote this / compiler synthesized this" boundary

## Pipeline Placement

| Boundary | Owning work |
|---|---|
| CCS source construction | Native types, declaration identities and source constraints |
| CCS/Baker elaboration and saturation | Intrinsic and operation recipes, demand/control, captures, numeric selection, layout, residence, declaration/ABI settlement and proof premises |
| Source publication | Immutable settled PSG codata and structure, with ordered joint premises and the intermediate rewrite record |
| Alex | Passive Huet Element/Pattern/Witness composition into admitted portable MLIR |
| Selected backend | Target-specific realization and artifact preservation checks |

### Source-owned analysis and readiness

CCS/Baker analyses and graph construction follow their semantic dependencies.
Recipes can introduce participants that require renewed analysis; changed
premises invalidate dependent results before publication. Each domain owns its
refinement and convergence rules. Missing prerequisites, contradictions and
settled facts remain distinct; merely exhausting work establishes no readiness.

Alex reads the immutable settled publication. It neither computes coeffects nor
queries hyperedges to reconstruct semantic facts. Physical SSA identities derive
from settled graph roles and block arguments without a semantic preassignment
pass. Target-specific lowering belongs to the backend. Custom MLIR plugins and
middle-end semantic transformations are retired.

## Implementation: Marking API

The unified marking API in `SemanticGraph.Elaboration`:

```fsharp
/// Mark a node as Baker-saturated
let markBaker (forConstruct: string) (id: int) (node: SemanticNode) : SemanticNode

/// Mark a node as Intrinsic-elaborated
let markIntrinsic (forConstruct: string) (id: int) (node: SemanticNode) : SemanticNode

/// Generate a fresh expansion ID for grouping related nodes
let freshId () : int
```

### Usage in Baker

```fsharp
let ctx = mkContext range elemType platform "List.map" inspiringNode
let expandedNode = mkExpandedNode ctx (SemanticKind.Application (...)) resultType
// expandedNode automatically has Kind="Baker", For="List.map", Id=ctx.ExpansionId
```

### Usage in Intrinsic Elaboration

```fsharp
let expansionId = Elaboration.freshId()
let synthesizedNode = baseNode |> markIntrinsic "Console.write" expansionId
```

## Phase Output Serialization

The CCS phase emitter includes enrichment metadata in JSON output:

```json
{
  "id": 42,
  "kind": "Application",
  "type": "unit",
  "elaborationKind": "Baker",
  "elaborationFor": "List.map",
  "elaborationId": 7
}
```

This enables external tooling (IDEs, debuggers, visualization) to understand enrichment boundaries.

## Design Principles

### 1. Transparency Over Magic

Every synthesized node is marked. There is no hidden structure. Developers can always see what the compiler added.

### 2. Separation of Concerns

- **Intrinsic elaboration** handles operation semantics
- **Baker saturation** handles algorithmic structure
- **CCS/Baker analyses** settle semantic and proof facts over the graph and publish immutable codata
- **Alex** passively composes the admitted physical form from that publication
- **Backend realization** preserves the source contract through target-specific transformations

### 3. Composability

Baker-saturated structure composes with intrinsic elaboration:
- `List.map Console.write xs` has Baker structure (map decomposition) containing Intrinsic nodes (Console.write implementation)

### 4. Debuggability

The expansion ID links enable:
- "Collapse" views that hide enriched structure
- "Expand" views that show full detail
- Step-through debugging that skips synthesized code

## Related Documents

- [Coeffect_Analysis_Architecture.md](./Coeffect_Analysis_Architecture.md) - Source-owned analysis and publication
- [PSG_Nanopass_Architecture.md](./PSG_Nanopass_Architecture.md) - PSG construction pipeline
- [CCS_Architecture.md](./CCS_Architecture.md) - Clef Compiler Services
- [Architecture_Canonical.md](./Architecture_Canonical.md) - Overall system architecture

## Serena Memories

- `elaboration_infrastructure` - Condensed reference for enrichment metadata
- `baker_component` - Baker saturation component details
- `psg_elaboration_architecture` - PSG elaboration nanopass details
