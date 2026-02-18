# Elaboration Fold Architecture: Intrinsics & Saturation

> **January 2026**: Defines the four-pass pipeline for both **Intrinsic Elaboration** (language-level semantics) and **Baker Saturation** (HOF decomposition). Same infrastructure, two phases.

## The Five Artifacts

```
1. PSG₀  - Original PSG from type checking
2. Intrinsic Recipes  - New node structures for intrinsic elaboration (isolated)
3. PSG₁  - PSG₀ with intrinsic structures folded in
4. Saturation Recipes - New node structures for Baker saturation (isolated)
5. PSG₂  - PSG₁ with saturation structures folded in → delivered to Alex
```

## The Four Discrete Passes

```
PSG₀ (artifact 1: fncs_phase_N_original.json)
  │
  ├─[Pass 1: Intrinsic Fan-Out]─────────────────────────────┐
  │   • Parallel via MailboxProcessor                       │
  │   • Identifies nodes needing intrinsic elaboration      │
  │   • Creates Recipe for each                             │
  │                                                         ▼
  │                              Intrinsic Recipes (artifact 2: fncs_intrinsic_recipes.json)
  │                                                         │
  ├─[Pass 2: Intrinsic Fold-In]─────────────────────────────┤
  │   • Single traversal                                    │
  │   • Builds fresh PSG₁ from (PSG₀ + Recipes)             │
  │   • Original elaborated nodes excluded                  │
  │                                                         ▼
PSG₁ (artifact 3: fncs_phase_N_intrinsic.json)
  │
  ├─[Pass 3: Saturation Fan-Out]────────────────────────────┐
  │   • Parallel via MailboxProcessor                       │
  │   • Identifies Match, HOF patterns needing saturation   │
  │   • Creates Recipe for each                             │
  │                                                         ▼
  │                              Saturation Recipes (artifact 4: fncs_saturation_recipes.json)
  │                                                         │
  ├─[Pass 4: Saturation Fold-In]────────────────────────────┤
      • Single traversal                                    │
      • Builds fresh PSG₂ from (PSG₁ + Recipes)             │
      • Match → IfThenElse, HOFs → primitives               │
                                                            ▼
PSG₂ (artifact 5: fncs_phase_N_saturated.json) → Alex
```

## Core Principle: Single Live PSG

- Only ONE PSG version in memory at a time
- After fold-in, old PSG is discarded (GC'd)
- Intermediates emitted to disk for inspection (`-k` flag)
- No accumulation of `graph'`, `graph''`, `graph'''`

## Why Recipes Are Separate Artifacts

The Recipe artifacts (2 and 4) enable:

1. **Debugging** - See exact structures BEFORE fold-in
2. **Validation** - Verify recipes independently
3. **Tooling** - IDE can show "Match elaborates to IfThenElse chain"
4. **Diffing** - Compare recipe outputs across compiler versions

## Recipe Structure

```fsharp
type Recipe = {
    OriginalNodeId: NodeId        // Node being replaced
    NewNodes: SemanticNode list   // Replacement structure
    ReplacementRootId: NodeId     // What parents should reference
    ElaborationKind: string       // "Intrinsic" | "Baker"
    ElaborationSource: string     // "Console.writeln" | "Match" | "List.map"
}

type RecipeSet = {
    Kind: string                          // "Intrinsic" | "Saturation"
    Recipes: Map<NodeId, Recipe>          // All recipes
    ReplacementMap: Map<NodeId, NodeId>   // originalId → replacementRootId
}
```

## Fan-Out Pass (Parallel Recipe Creation)

```fsharp
let fanOut kind shouldElaborate createRecipe graph =
    graph.Nodes
    |> findReachable
    |> filter shouldElaborate
    |> Async.Parallel (map createRecipe)  // Embarrassingly parallel
    |> RecipeSet.fromList kind
```

- Each recipe created independently (referential transparency)
- No coordination between workers
- Result: RecipeSet artifact

## Fold-In Pass (Single Traversal)

```fsharp
let foldIn recipeSet graph =
    // Build fresh graph:
    // - Nodes with recipes → include recipe's NewNodes, exclude original
    // - Nodes without recipes → include with updated child references
    // - Update all NodeId references via ReplacementMap
```

- Single traversal builds complete new PSG
- Original elaborated nodes simply not included
- No orphans, no patching, no mutation

## Implementation Location

```
clef/src/Compiler/Nanopass/
├── Recipe.fs        - Recipe, RecipeSet types
├── FanOut.fs        - Parallel recipe creation
├── FoldIn.fs        - Generic fold-in
└── Serialization.fs - JSON emission for intermediates
```

## Integration with Pipeline

In `NativeService.fs`:

```fsharp
// Phase 1-4: Type checking, reachability (existing)
let psg0 = ...

// Pass 1: Intrinsic Fan-Out
let intrinsicRecipes = IntrinsicElaboration.fanOut psg0
emitRecipeSet "fncs_intrinsic_recipes.json" intrinsicRecipes

// Pass 2: Intrinsic Fold-In
let psg1 = FoldIn.foldIn intrinsicRecipes psg0
emitPhase "fncs_phase_N_intrinsic.json" psg1

// Pass 3: Saturation Fan-Out
let saturationRecipes = BakerSaturation.fanOut psg1
emitRecipeSet "fncs_saturation_recipes.json" saturationRecipes

// Pass 4: Saturation Fold-In
let psg2 = FoldIn.foldIn saturationRecipes psg1
emitPhase "fncs_phase_N_saturated.json" psg2

// Deliver to Composer/Alex
{ Graph = psg2; ... }
```

## See Also

- `PSG_Nanopass_Architecture.md` - Overall nanopass principles
- `Baker_Architecture.md` - Baker saturation details
- Serena memory: `psg_elaboration_fold_architecture` - Quick reference
