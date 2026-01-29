# PSG Scope Boundary Representation: Comprehensive Exploration

**Date:** January 29, 2026  
**Purpose:** Understand how PSG represents scope boundaries and whether existing infrastructure supports scope-aware witness implementation  
**Status:** READ-ONLY EXPLORATION COMPLETE

---

## EXECUTIVE SUMMARY

**THE GOOD NEWS:** The PSG ALREADY has the information we need. Scope boundaries are NOT implicit - they are EXPLICIT through the Children list in SemanticNode.

**KEY INSIGHT:** Lambda, IfThenElse, While, and other scope-containing nodes have their child nodes (the nodes that form their scope bodies) explicitly listed in `SemanticNode.Children`. This is sufficient for a pass to understand scope ownership WITHOUT physical barriers.

**THE ARCHITECTURAL PATTERN:**
1. Scopes are defined by **ParentOf edges** (stored in SemanticNode.Children)
2. Witnesses can check scope ownership by examining `node.Parent` and `node.Children`
3. The Zipper provides bidirectional navigation to understand scope boundaries
4. Sub-graph witnessing (in ControlFlowWitness) demonstrates the correct pattern

---

## PART 1: How PSG REPRESENTS Scope Boundaries

### 1.1: SemanticNode Structure

Each node in the PSG carries:

```fsharp
type SemanticNode = {
    Id: NodeId
    Kind: SemanticKind              // What semantic construct this is
    Children: NodeId list           // Child nodes it "owns"
    Parent: NodeId option           // Optional parent node
    IsReachable: bool               // Reachability mark
    EmissionStrategy: EmissionStrategy  // How to emit this node
    // ... other fields
}
```

**KEY POINT:** `Children` is the **explicit scope boundary marker**. Every edge in the Children list is a **parent → child (scope ownership) relationship**.

### 1.2: Lambda Nodes Structure

```fsharp
| Lambda of 
    parameters: (string * NativeType * NodeId) list  // Parameter name, type, definition NodeId
    * body: NodeId                                     // Body expression (THE SCOPE)
    * captures: CaptureInfo list                       // Captured variables
    * enclosingFunction: string option                 // For debugging
    * context: LambdaContext                           // Regular | Lazy | Seq
```

**In the PSG Builder:**

```fsharp
| SemanticKind.Lambda (params', body, _, _, _) ->
    let paramIds = params' |> List.map (fun (_, _, nodeId) -> nodeId)
    paramIds @ [body]  // Children = parameter defs + body
```

**Scope Ownership:** All nodes reachable from `body` are "owned" by this Lambda.

### 1.3: IfThenElse Nodes Structure

```fsharp
| IfThenElse of 
    guard: NodeId              // Condition to test
    * thenBranch: NodeId       // Then block (SCOPE)
    * elseBranch: NodeId option // Else block (SCOPE) - optional
```

**In the PSG Builder:**

```fsharp
| SemanticKind.IfThenElse (guard, thenB, elseB) ->
    guard :: thenB :: (Option.toList elseB)  // Children = guard + branches
```

**Scope Ownership:** 
- `guard` nodes are evaluated outside the scope
- `thenBranch` nodes form their own scope
- `elseBranch` nodes (if present) form another scope

### 1.4: WhileLoop Nodes Structure

```fsharp
| WhileLoop of 
    guard: NodeId  // Loop condition (SCOPE)
    * body: NodeId  // Loop body (SCOPE)
```

**In the PSG Builder:**

```fsharp
| SemanticKind.WhileLoop (guard, body) -> [guard; body]  // Children = both
```

**Scope Ownership:**
- `guard` is re-evaluated each iteration (scope boundary repeats)
- `body` nodes form the loop body scope

### 1.5: Other Scope-Containing Nodes

| Node Kind | Scope Children | Notes |
|-----------|---|---|
| **ForLoop** | `[start; finish; body]` | Range and body form scopes |
| **ForEach** | `[collection; body]` | Collection and body are scopes |
| **Match** | `scrutinee :: cases` | Each case body is a scope |
| **TryWith** | `[body; handler]` | Both body and handler are scopes |
| **TryFinally** | `[body; cleanup]` | Both body and cleanup are scopes |
| **Sequential** | `[nodes...]` | All sequential statements are child scopes |

---

## PART 2: Are Scope Boundaries Already Representable?

### 2.1: The Children List IS the Scope Boundary

**ANSWER: YES.**

The `Children: NodeId list` field in SemanticNode is NOT a generic "all children" list - it's specifically the nodes that belong to the scope owned by this node.

Evidence from PSG Builder:

```fsharp
let private extractImpliedChildren (kind: SemanticKind) : NodeId list =
    // Extracts only nodes semantically "owned" by this node
    match kind with
    | SemanticKind.Lambda (params', body, _, _, _) ->
        let paramIds = params' |> List.map (fun (_, _, nodeId) -> nodeId)
        paramIds @ [body]  // ← Only the nodes "inside" this scope
```

**This is NOT a general graph traversal list - it's a scope ownership list.**

### 2.2: The Parent Field Enables Scope Discovery

Conversely, if a witness needs to know "am I inside a scope boundary?", it can check:

```fsharp
let isInsideScope (nodeId: NodeId) (parentId: NodeId) (graph: SemanticGraph) : bool =
    match SemanticGraph.tryGetNode parentId graph with
    | None -> false
    | Some parentNode ->
        List.contains nodeId parentNode.Children  // Is this node owned by parent?
```

**Or equivalently:**

```fsharp
let scopeOwner (nodeId: NodeId) (graph: SemanticGraph) : SemanticNode option =
    match SemanticGraph.tryGetNode nodeId graph with
    | None -> None
    | Some node ->
        Option.bind (fun pid -> SemanticGraph.tryGetNode pid graph) node.Parent
```

---

## PART 3: How Witnesses Handle Scope Boundaries

### 3.1: The Sub-Graph Combinator Pattern (ControlFlowWitness)

From `NanopassArchitecture.fs` (lines 85-114):

```fsharp
/// Witness a sub-graph rooted at a given node
/// Used by control flow witnesses to materialize branch sub-graphs
let witnessSubgraph
    rootNodeId
    (ctx: WitnessContext)
    (witness: WitnessContext -> SemanticNode -> WitnessOutput)
    : MLIROp list =

    // Create ISOLATED accumulator for this sub-graph
    let subAccumulator = MLIRAccumulator.empty()

    match SemanticGraph.tryGetNode rootNodeId ctx.Graph with
    | None -> []
    | Some rootNode ->
        match PSGZipper.focusOn rootNodeId ctx.Zipper with
        | None -> []
        | Some focusedZipper ->
            // Create sub-context with isolated accumulator
            let subCtx = {
                Graph = ctx.Graph
                Coeffects = ctx.Coeffects
                Accumulator = subAccumulator  // ← ISOLATED SCOPE
                Zipper = focusedZipper
            }

            // Traverse sub-graph starting from root
            visitAllNodes witness subCtx rootNode subAccumulator

            // Return collected operations
            List.rev subAccumulator.TopLevelOps
```

**How This Enforces Scope Boundaries:**

1. **Isolated Accumulator:** Each scope gets its own `MLIRAccumulator`, preventing cross-scope binding contamination
2. **Zipper Focus:** The zipper is focused on the scope root, establishing scope position
3. **Visitor Traversal:** `visitAllNodes` only traverses nodes reachable from the scope root
4. **Return as Block:** The result is a `MLIROp list` representing the scope as a cohesive unit

**This is how ControlFlowWitness handles scopes:**

```fsharp
// From ControlFlowWitness.fs (lines 47-57)
match tryMatch pIfThenElse ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
| Some ((condId, thenId, elseIdOpt), _) ->
    match MLIRAccumulator.recallNode condId ctx.Accumulator with
    | None -> WitnessOutput.error "IfThenElse: Condition not yet witnessed"
    | Some (condSSA, _) ->
        let thenOps = witnessSubgraph thenId ctx subGraphCombinator
        let elseOps = elseIdOpt |> Option.map (fun elseId -> witnessSubgraph elseId ctx subGraphCombinator)

        match tryMatch (pBuildIfThenElse condSSA thenOps elseOps) ctx.Graph node ctx.Zipper ctx.Coeffects.Platform with
        | Some (ops, _) -> { InlineOps = ops; TopLevelOps = []; Result = TRVoid }
        | None -> WitnessOutput.error "IfThenElse pattern emission failed"
```

**Pattern:**
1. Pattern match extracts the child scope node IDs
2. Call `witnessSubgraph` on each child scope root
3. Each scope is witnessed through the nanopass chain independently
4. Results are collected as operation blocks
5. Parent witness composes the scope blocks into MLIR control flow ops

---

## PART 4: Scope Ownership Questions Answered

### Q1: Do Lambda nodes have children? What are they?

**YES.** Lambda nodes have:
- **Parameter definition nodes:** The NodeIds from the parameter tuples
- **Body node:** The expression forming the lambda body

These are explicitly listed in `node.Children` via the Builder's `extractImpliedChildren`.

**Example:**
```
Lambda(
  parameters = [("x", IntTy, paramNodeId_x)],
  body = bodyNodeId
)

→ Children = [paramNodeId_x, bodyNodeId]
```

### Q2: How are IfThenElse, While nodes structured?

**They are scope containers:**

```
IfThenElse(guard, thenBranch, elseBranch)
  Children = [guard, thenBranch, elseBranch]

WhileLoop(guard, body)
  Children = [guard, body]
```

Each child represents a logically distinct **scope** that can be witnessed independently.

### Q3: Is there an existing concept of "scope ownership"?

**YES, implicitly through Children:**

The PSG Builder explicitly constructs `Children` lists to represent "nodes owned by this scope". This is NOT a generic graph edges list - it's a **scope ownership declaration**.

Evidence:
- Parameter definitions are included (part of Lambda scope)
- Body nodes are included (they ARE the scope)
- Guard nodes are included (must be evaluated in scope)
- Siblings are NOT included
- Cousin nodes are NOT included

**The scope ownership concept exists - it's encoded in the ParentOf relationship (Children list).**

### Q4: Can witnesses know their position in scope hierarchy?

**YES, via the Zipper and Parent field:**

```fsharp
let getScopeContext (node: SemanticNode) (graph: SemanticGraph) : (SemanticNode option * SemanticKind) =
    match node.Parent with
    | None -> (None, SemanticKind.Error "No parent")
    | Some parentId ->
        match SemanticGraph.tryGetNode parentId graph with
        | None -> (None, SemanticKind.Error "Parent not found")
        | Some parentNode ->
            // Now witness knows:
            // - parentNode.Kind tells what kind of scope it's in
            // - parentNode.Children tells all siblings in scope
            // - Parent owns this node
            (Some parentNode, parentNode.Kind)
```

The Zipper provides this navigation automatically.

---

## PART 5: What Witnesses Actually Need

### 5.1: No Physical Barriers Needed

**Finding:** Witnesses do NOT need physical Type Barriers or ModuleInternal restrictions to enforce scope boundaries.

**Why:** The PSG structure itself defines scope boundaries through the Children/Parent relationship.

### 5.2: What DOES Need to be Understood

Witnesses need to understand:

1. **Scope Ownership via Children:** A node's Children are its scope members
2. **Scope Traversal via Zipper:** Navigation through scope hierarchy
3. **Isolated Accumulation:** Each scope boundary creates an isolated accumulator
4. **Sub-Graph Witnessing:** Scopes are witnessed independently, then composed

### 5.3: Enforcement via Architecture

The architecture already enforces scope boundaries through:

1. **Category-Selective Witnesses:** Each witness only handles specific node kinds
2. **Sub-Graph Combinator Pattern:** Scope bodies are witnessed through a combinator
3. **Isolated Accumulators:** No cross-scope binding contamination
4. **Zipper Navigation:** Proper position awareness

---

## PART 6: Pattern Examples from Existing Code

### 6.1: ControlFlowWitness Sub-Graph Pattern

**File:** `/src/MiddleEnd/Alex/Witnesses/ControlFlowWitness.fs`

Pattern:
1. Match on IfThenElse/WhileLoop to extract scope node IDs
2. Create sub-graph combinator from nanopass list
3. Call `witnessSubgraph` on each child scope
4. Compose results using patterns

This ALREADY demonstrates proper scope handling.

### 6.2: LazyWitness Single-Node Pattern

**File:** `/src/MiddleEnd/Alex/Witnesses/LazyWitness.fs`

Pattern:
1. Match LazyExpr to extract body node
2. Recall body from accumulator (must be witnessed first)
3. Pattern match to compose

This pattern works because:
- LazyExpr's children include the body node
- That body is witnessed as a complete unit
- Results are recalled and composed

---

## PART 7: Answer to the Original Question

**"Does the PSG ALREADY have the information we need to know 'this subtree is owned by a scope boundary'?"**

### YES. DEFINITIVELY.

The PSG has:

1. **Explicit Children Lists:** Each node lists its owned children
2. **Parent Pointers:** Bidirectional navigation
3. **SemanticKind Classification:** Witness can pattern-match on scope kind
4. **Zipper Navigation:** Positional awareness through scope hierarchy
5. **Sub-Graph Witnessing Pattern:** Infrastructure already exists

### NO additional representation is needed.

The architecture does NOT require:
- Physical type barriers
- Module internal declarations at the witness level
- New metadata fields
- Alternative node structures

The information is ALREADY sufficient.

---

## PART 8: Architectural Implications

### 8.1: What This Means for Witness Refactoring

When implementing witnesses for scope-containing nodes (Lambda, ControlFlow, etc.):

1. **Extract child scope node IDs** via pattern matching on SemanticKind
2. **Use `witnessSubgraph`** to materialize each scope independently
3. **Compose via patterns** that accept operation lists as parameters
4. **Accumulate result** in parent witness's inline operations

This is the pattern already demonstrated in ControlFlowWitness.

### 8.2: What This Means for Future Features

Passes that need to enforce scope boundaries (e.g., capture analysis, lifetime verification):

1. Can trust the Children/Parent relationship as the scope boundary
2. Can traverse Children to find all scope members
3. Can use Parent to navigate to enclosing scopes
4. Don't need to ask the witness layer for scope verification - it's in the PSG

### 8.3: Why This Works

The PSG is built by the Nanopass framework with scope awareness. Each nanopass pass that constructs semantic nodes explicitly populates Children with the nodes "owned" by that scope. By the time Alex witnesses the PSG, scope boundaries are already correctly defined.

---

## REFERENCE: Key Code Locations

| Component | File | Key Function | Lines |
|-----------|------|--------------|-------|
| SemanticNode Type | `/fsnative/src/Compiler/PSGSaturation/SemanticGraph/Types.fs` | `type SemanticNode` | 315-328 |
| Children Extraction | `/fsnative/src/Compiler/PSGSaturation/SemanticGraph/Builder.fs` | `extractImpliedChildren` | Various |
| Sub-Graph Witnessing | `/src/MiddleEnd/Alex/Traversal/NanopassArchitecture.fs` | `witnessSubgraph` | 85-114 |
| Zipper Navigation | `/src/MiddleEnd/Alex/Traversal/PSGZipper.fs` | `PSGZipper` type | 54-63 |
| Control Flow Pattern | `/src/MiddleEnd/Alex/Witnesses/ControlFlowWitness.fs` | `witnessControlFlowWith` | 44-79 |
| Lazy Pattern | `/src/MiddleEnd/Alex/Witnesses/LazyWitness.fs` | `witnessLazy` | 24-71 |

---

## CONCLUSION

**The PSG already has everything needed to represent and understand scope boundaries.**

The architecture does not need additional type-level barriers or module-internal restrictions. Scope ownership is explicitly encoded in the Children/Parent relationship of each SemanticNode. Witnesses can safely rely on this structure, which is guaranteed by the PSG's construction process.

The existing sub-graph witnessing pattern (used in ControlFlowWitness) demonstrates the correct way to handle scopes: witness each scope independently with an isolated accumulator, then compose the results in the parent witness.

No architectural changes are required. Witnesses simply need to follow the pattern already demonstrated.
