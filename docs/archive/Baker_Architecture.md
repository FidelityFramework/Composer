# Baker: The Chef of the Semantic Graph

> **Status**: Current Architecture (January 2026)
> **Role**: Semantic Saturation & HOF Decomposition
> **Position**: Post-Reachability (Nanopass)

## Overview

**Baker** is the semantic saturation engine of the Firefly compiler. Its name comes from the metaphor of "baking in" meaning: taking the raw structural skeleton of the Program Semantic Graph (PSG) and enriching it with the full semantic implementation of F# language features.

While **Alex** (the backend) witnesses the graph to generate MLIR, Baker prepares that graph by decomposing high-level abstractions into low-level primitives that Alex can understand.

## The Philosophy: "Only Pay For What You Use"

In standard .NET implementations, using a feature like `List.map` often drags in the entire `FSharp.Core.dll` runtime dependencies—the "kitchen sink" approach. Firefly takes a radically different path.

Baker operates **post-reachability**. It only "bakes" the code that is actually used by your application. If your program uses `List.map` but never `List.sort`, the sorting logic is never even saturated into the graph, let alone compiled. This "book-ending" of reachability analysis and Baker's saturation results in concise, lean native binaries that execute close to the metal.

## Architecture: The Nanopass Pipeline

Gone are the rigid numbered phases and complex "two-tree zippers" of early experiments. The current pipeline embraces a fluid **nanopass infrastructure** designed for parallelism and safety.

The core architectural pattern is **Fan-Out / Fold-In**:

1.  **Fan-Out (Discovery)**: A zipper traverses the graph to find "decomposable" nodes (like HOFs or pattern matching). This happens in parallel.
2.  **Saturation**: Baker "Recipes" expand these nodes into sub-graphs of primitives.
3.  **Fold-In**: The new sub-graphs are merged back into the main PSG.

This separation ensures that graph mutations are safe and predictable.

## The Kitchen Metaphor: Recipes & Ingredients

Baker's internal structure is organized around a strict culinary metaphor that enforces architectural layering:

### 1. Ingredients (The Primitives)
Ingredients are the atomic building blocks—the "raw food" of the compiler. These are wrappers around low-level PSG operations (like `cons`, `head`, `tail`, `ifThenElse`).
*   **Rule**: Only Ingredients are allowed to create raw PSG nodes.

### 2. Recipes (The Combinations)
Recipes are the instructions for combining Ingredients to create a finished dish (a language feature).
*   **Rule**: Recipes *never* touch raw nodes. They only compose Ingredients.

For example, a `List.map` isn't a black box in the runtime; it's a **Recipe** composed of `foldRight`, `cons`, and `app`.

## The Secret Sauce: XParsec & Saturation Combinators

How do we write these Recipes without creating a mess of imperative code? We borrowed a principle from our parser: **Parser Combinators**.

Baker uses **XParsec** not to parse text, but to parse *and generate* semantic logic. We define a `SaturationParser` monad that threads the graph state automatically.

```fsharp
// Actual code from ListRecipes.fs
let listMapRecipe mapper list elemTy =
    foldRight
        (emptyList elemTy)                 // Base case: []
        (fun head recurse ->
            saturation {                   // The Saturation Monad
                let! mapped = app1 mapper head
                return! cons mapped recurse
            })
        list
```

This declarative style allows us to express complex logic (like state machines for `Seq` or recursion for `List`) in just a few lines of readable, type-safe code.

## From Functional Abstraction to Direct Computation

Baker bridges the gap between high-level functional programming and low-level machine code.

1.  **FCS**: Gives us the Typed Tree.
2.  **PSG Builder**: Converts it to a semantic graph.
3.  **Reachability**: Prunes the graph to only what matters.
4.  **Baker**: "Saturates" abstractions. A `List.map` becomes a specific loop structure; a `Seq` expression becomes a state machine.
5.  **Alex**: Lowers this fully explicit graph to MLIR.

By "baking in" this information, we allow the backend to perform powerful type, memory, and computational pattern inference, turning functional abstractions into direct, efficient computation.
