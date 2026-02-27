# Graph Pattern Matching via State-Threaded Combinators

> Documenting the novel computational model in Composer's Alex layer,
> assessing a derivative-based formalization, and motivating it through
> the future Transcribe feature area.

---

## 1. What Composer Does Today

### The Empty String Trick

Alex uses XParsec — a generic parser combinator library — for PSG pattern matching.
The input is an empty string:

```fsharp
let reader = Reader.ofString "" state  // No characters to consume
parser reader
```

No character is ever read. `Reader.Peek()` always returns `ValueNone`. The entire
parsing machinery — bind, choice, backtracking — operates through the **user state
channel**, not the input channel:

```fsharp
type PSGParserState = {
    Graph: SemanticGraph          // the structure being navigated
    Zipper: PSGZipper             // the cursor (replaces Reader.index)
    Current: SemanticNode         // the "current token" (replaces Reader.Peek())
    Coeffects: TransferCoeffects  // pre-computed observations
    Accumulator: MLIRAccumulator  // collected MLIR results
}
```

The state IS the input. Navigation IS consumption. XParsec's backtracking — which
normally restores `(index, state)` atomically on `<|>` failure — restores
`(0, previous_graph_position)`. The index restoration is a no-op. The state
restoration is the real backtracking.

### Three Layers, One Monad

The Element/Pattern/Witness architecture composes through XParsec's `parser { }` CE:

- **Elements** (`module internal`): Atomic MLIR operations. Produce `MLIROp` values.
- **Patterns**: Compose Elements via monadic sequencing. Navigate the graph, read
  coeffects, pull child results from the accumulator. ~50 lines each.
- **Witnesses**: Thin observers that compose Patterns with `<|>` (choice). ~20 lines
  each. The `<|>` chain IS the dispatch — no central hub.

```fsharp
// Witness: compose patterns with choice
let combined =
    pBinaryArithIntrinsic <|> pUnaryArithIntrinsic <|> pTypeConversionIntrinsic
match tryMatch combined ctx.Graph node ctx.Zipper ctx.Coeffects ctx.Accumulator with
| Some ((ops, result), _) -> { InlineOps = ops; Result = result }
| None -> WitnessOutput.skip
```

Each pattern self-checks (tests node kind, guards on type, reads coeffects) and
fails via `return! fail (Message ...)` if preconditions aren't met. XParsec's `<|>`
catches the failure, restores state, tries the next pattern. The type system is the
selector:

```fsharp
let pRequireUnsigned : PSGParser<unit> =
    parser {
        let! state = getUserState
        match Types.tryGetNTUKind state.Current.Type with
        | Some (NTUKind.NTUuint _) | Some NTUKind.NTUsize -> return ()
        | _ -> return! fail (Message "Not unsigned type")
    }

// Signedness dispatch via backtracking: type IS the selector
return! (pIfUnsignedArith node.Id "shrui" <|> pBinaryArithOp node.Id "shrsi")
```

### Codata-Dependent Elision

Coeffects are pre-computed by CCS before Alex runs. Patterns **observe** them
monadically — they never compute them:

```fsharp
let! targetPlatform = getTargetPlatform   // reads coeffect
match targetPlatform with
| FPGA -> // emit comb.* (combinational logic)
| _    -> // emit arith.* (temporal operations)
```

Same Pattern, different Elements, selected by observation. This is the codata
principle: the photograph is taken before the walk; the walk merely looks at it.

### Navigation and Restoration

The `onChild` combinator navigates to a child, runs a sub-parser, and restores
focus — all within a single parser computation:

```fsharp
let onChild (childId: NodeId) (p: PSGParser<'T>) : PSGParser<'T> =
    parser {
        let! state = getUserState
        let savedCurrent = state.Current
        do! setCurrentNode childNode
        let! result = p
        do! setCurrentNode savedCurrent   // restore, not backtrack
        return result
    }
```

This is deliberate save-and-restore during **successful** matching. Text parsers
never do this — you don't "read ahead, extract something, then un-read." Tree
automata descend and return, but they don't carry monadic accumulation state through
the round trip.

---

## 2. Where This Sits Relative to Known Models

### Three Computational Models Compared

| Property | Sequence Parser | Tree Automaton | PSG Pattern Matching |
|----------|----------------|----------------|---------------------|
| Input structure | Linear | Tree | DAG / general graph |
| Consumption | Advance index | Descend subtree | Navigate + observe (non-destructive) |
| State | Position | Position + stack | Position + coeffects + accumulator + zipper |
| Backtracking | Restore index | Restore index + stack | Restore full state snapshot |
| Result | Parsed value | Matched subtree | MLIR operations (accumulated) |
| Navigation | Forward only | Up/down | Arbitrary (up, down, left, right, jump) |

The PSG model is neither sequence recognition nor tree recognition. It is **graph
observation with monadic state accumulation**: the parser navigates a graph,
observes pre-computed properties at each node, and accumulates output operations.
Pattern matching and code generation are a single fused pass — the MLIR is the
residual of observation (elision, not emission).

### What is Genuinely Novel

1. **Parser combinator backtracking as graph pattern dispatch.** The `<|>` chain in
   witnesses uses parser alternation for dispatch. The NTU type system selects which
   pattern succeeds, and backtracking is the dispatch mechanism. No pattern matching
   literature we are aware of uses parser combinator backtracking this way.

2. **State-as-input fusion.** The empty string trick exploits XParsec's generic type
   parameters to repurpose a text parser as a graph traversal engine. It works
   because XParsec's `Position<'State>` bundles index with state atomically — state
   restoration on backtracking is correct by construction.

3. **Monadic accumulation during pattern matching.** Text parsers produce parse
   trees. PSG patterns produce MLIR operations. The accumulator threads through the
   parser monad, fusing recognition and code generation into a single pass.

### What is Well-Applied Existing Art

4. **Zipper-based traversal.** Huet zippers are well-known. Using one for compiler
   IR traversal exists in Stratego/XT and Kiama. The contribution here is combining
   it with parser combinators rather than rewrite rules.

5. **Coeffect-driven dispatch.** Reading pre-computed metadata to select code paths
   is standard in staged compilation. The contribution is threading it through parser
   combinator state rather than as function parameters.

6. **Soft-delete reachability.** Phase 3 of the PSG pipeline (CCS) computes which
   nodes are reachable from declaration roots and marks unreachable nodes with
   `IsReachable = false`. This is a nanopass, not a gap — it runs before Alex
   exists. Structure preservation (soft-delete rather than hard prune) is a
   deliberate architectural decision: the daemon/LSP needs the full graph for
   incremental recomputation when code changes, and the typed tree zipper (Phase 4)
   needs full structure for navigation and FSharpExpr correlation. Alex simply skips
   unreachable nodes during its walk. This is established, correct, and load-bearing.

---

## 3. The Derivative Analogy

### Where It Holds

XParsec's choice operator on PSG patterns is operationally analogous to computing
Brzozowski derivatives:

```
At node N with state S, observe node kind K:
  D_K(p1 | p2) = D_K(p1) | D_K(p2)
```

- If `p1` accepts observation `K` → result is `p1`'s continuation
- If `p1` rejects → restore state, try `p2` with same observation
- If both reject → derivative is empty (fail)

The backtracking-with-state-restoration IS the derivative computation, executed
eagerly rather than computed as a symbolic transformation.

### Where It Breaks Down

Brzozowski derivatives are **purely structural** — they transform the pattern without
running it. This gives you memoization (same pattern + same observation = same
derivative), static analysis (inspect the derivative without executing), and boolean
closure (intersection and complement distribute over derivatives trivially).

PSG pattern matching is **effectful**. Patterns read coeffects, update accumulators,
navigate the zipper. The "derivative" at each node is a function of the entire state,
not just the observation. You cannot memoize `pBinaryArithOp` because its result
depends on `state.Coeffects.Platform` and it produces `state.Accumulator` updates.

### What Boolean Operators Would Add

Current XParsec provides union (`<|>`). It does not provide intersection or
complement. To express "match A AND NOT B," you must write manual guards:

```fsharp
// Current: imperative guard
let pNonIntrinsicApp = parser {
    let! (funcId, argIds) = pApplication
    let! state = getUserState
    if isIntrinsic funcId state then
        return! fail (Message "Is intrinsic")
    else
        return (funcId, argIds)
}
```

With boolean operators:

```fsharp
// Hypothetical: declarative intersection + complement
let pNonIntrinsicApp = pApplication <&> (~~pIntrinsicApplication)
```

The derivative formulation makes this trivial:
`D_K(P1 & ~P2) = D_K(P1) & ~D_K(P2)`.

This matters most for the Transcribe feature area (Section 5), where pattern-matching
foreign ASTs involves complex structural constraints that intersection and complement
express naturally.

---

## 4. Formalization: Within the Fused Model, Not Against It

### The Recognition/Emission Split is the Wrong Path

An earlier draft of this analysis proposed separating patterns into a pure
"recognition" layer (algebraic, derivative-compatible) and an effectful "emission"
layer (monadic), connected by bind (`>>=`). This is formally clean and
architecturally destructive.

Every `>>=` boundary between recognition and emission is a seam. Seams accumulate:

1. Helper functions to bridge the two phases
2. Intermediate types to carry context across the bind
3. Registries to coordinate which emission pairs with which recognition
4. Parameter passing to thread data that the fused model carries in state

This is the exact pattern that produced PSGEmitter and PSGScribe — central dispatch
hubs that were removed **twice** because they emerged from coordination pressure at
architectural seams. The helper functions cleaned out during the XParsec remediation
(5,773 → 2,225 witness lines) were the same phenomenon.

The fused model — one `parser { }` block per Pattern, recognition and emission in a
single monadic flow — works because it has **no internal seams**. There is no place
for a registry to form. There is no boundary where parameter pressure builds. The
architecture is structurally resistant to corruption, not merely disciplinarily.

A codata architecture resists decomposition. You observe it whole.

### Boolean Operators as Combinators Within the Fused Model

The capabilities that derivatives provide — intersection and complement — do not
require splitting the model. They can be expressed as combinators within it:

```fsharp
/// Complement: succeed if inner pattern FAILS on the current node
let pNot (inner: PSGParser<'T>) : PSGParser<unit> =
    parser {
        let! state = getUserState
        match tryMatch inner state.Graph state.Current state.Zipper
                       state.Coeffects state.Accumulator with
        | Some _ -> return! fail (Message "Complement: inner matched")
        | None -> return ()
    }

/// Intersection: succeed only if BOTH patterns match the current node
/// Runs p1, restores state, runs p2. Returns p2's result.
let pBoth (p1: PSGParser<'A>) (p2: PSGParser<'B>) : PSGParser<'B> =
    parser {
        let! state = getUserState
        match tryMatch p1 state.Graph state.Current state.Zipper
                       state.Coeffects state.Accumulator with
        | Some _ ->
            // p1 matched. Now try p2 from the same position.
            return! p2
        | None -> return! fail (Message "Intersection: first pattern failed")
    }
```

Usage in a Pattern — still one `parser { }` block, no split:

```fsharp
let pForLoopToMap : TranscribeParser<ClefExpr> =
    parser {
        let! (init, cond, step, body) = pForLoop
        do! pBoth (onChild cond pArrayIterationTest)
                  (pNot (onChild body pContainsBreakOrContinue))
        let! bodyExpr = onChild body pTransposeExpression
        return ClefExpr.ArrayMap(arrayTarget, bodyExpr)
    }
```

This is `<&>` and `~~` without the split. The `parser { }` block stays whole.
Recognition and emission remain fused. The combinators are just things you `let!`
or `do!` inside the same monadic flow.

### What This Does NOT Give You

Derivative-based frameworks provide two things that fused-model combinators do not:

1. **Static pattern analysis.** With an algebraic pattern AST, you can inspect
   combinator chains for coverage and overlap without running them. Fused-model
   combinators are opaque functions — you can execute them but not inspect their
   structure. This means questions like "do these patterns cover all SemanticKind
   cases?" remain runtime-discoverable only.

2. **Recognition memoization.** If recognition is pure and separated, you can
   cache `D_K(patterns)` per node kind. Fused-model combinators are effectful
   and cannot be memoized this way.

These are real losses. They are also theoretical. In practice: pattern coverage is
validated by the regression runner (10 samples, all passing). Memoization matters
for very large PSGs with repeated node kinds — a performance concern that has not
manifested.

### The Derivative Algebra as a Reference Model

The derivative construction remains valuable as a **formal reference** even if it
is never implemented as a split:

```
D_K(P) = { p ∈ P | p accepts observation K at this position }
```

A pattern algebra with Observe, Sequence, Choice, Intersect, Complement, and
Navigate describes what the fused-model combinators **do** — it just describes
them algebraically rather than implementing them as a separate layer.

This means:
- The algebra can serve as a specification for fused-model combinator behavior
- Proofs about pattern properties (completeness, non-overlap) can be stated in
  the algebraic framework even if the implementation remains fused
- If XParsec's author builds derivative-based internals upstream, the fused model
  benefits without Composer having to restructure

The formal model is the map. The fused implementation is the territory. The map
does not need to become the territory to be useful.

### What It Costs to Wait

If boolean combinators (`pNot`, `pBoth`) are deferred until Transcribe demands
them, compound constraints are expressed as manual guards — the current model:

```fsharp
// Current: manual guard (verbose, correct, structurally protected)
let! hasBreak = onChild body pContainsBreakOrContinue
if hasBreak then return! fail (Message "Contains break")
```

This is more verbose than `do! pNot (onChild body pContainsBreak)`. It is also
impossible to misuse, impossible for an agent to "improve" with a coordination
layer, and structurally identical to every other Pattern in the system.

The cost of waiting is ergonomic, not architectural. The patterns still compose.
The four pillars remain intact. The 34 PRDs proceed without risk.

---

## 5. The Transcribe Feature Area

Transcribe is a future Composer feature for porting programs from other languages
into Clef source.

### Three Vectors

**Vector 1: Native Inbound (Farscape lineage)**

```
C/C++/Rust/Go source → Clef source → Composer → MLIR → LLVM → native binary
```

Farscape already generates declaration-level bindings for C via clang JSON AST +
XParsec post-processing. Transcribe extends this to complete source transposition.

**Vector 2: Edge Inbound (CloudEdge lineage)**

```
TypeScript source → Clef source → Composer → Fable → JavaScript → Cloudflare Workers
```

Glutinum generates F# bindings from `@cloudflare/workers-types`. Hawaii generates
management API clients from OpenAPI specs. Transcribe extends this to porting
TypeScript application code. Xantham (under development) provides a more structured
TypeScript parsing path that would serve as the frontend adapter for this vector.

**Vector 3: Notebook Inbound (Jupyter/GPU lineage)**

```
Python notebooks → Clef notebooks → Composer → { CPU via ORC JIT, GPU via ROCm, NPU via AIE }
```

Python ML notebooks are the lingua franca of the field. Transposing them to Clef
notebooks that dispatch to heterogeneous compute on Strix Halo (unified LPDDR5X,
zero-copy between CPU/GPU/NPU) is compelling — especially because Clef's DTS
(Dimensional Type System) would enforce tensor dimension constraints that Python
leaves implicit.

### Language Progression

| Language | Vector | Existing Art | Key Translation Challenge |
|----------|--------|-------------|--------------------------|
| C | Native | Farscape (declarations) | `malloc/free` → arena scoping |
| C++ | Native | Farscape (partial) | Templates → SRTP, RAII → arena lifetime |
| TypeScript | Edge | Glutinum + Hawaii (SDKs), Xantham (future) | `any`/union → DU, `Promise` → async CE |
| Rust | Native | None yet | Ownership → DMM coeffects (near 1:1) |
| Python | Notebook | None yet | Dynamic typing → inferred NTU types |
| Go | Native | None yet | Goroutines → Prospero actors |

### Pipeline Architecture

```
Foreign Source
    │
    ├── Foreign Frontend Adapter (per language)
    │   C/C++: clang JSON AST
    │   TypeScript: Xantham AST
    │   Python: cpython AST module
    │   Rust: syn / rustc
    │   Go: go/parser + go/types
    │
    ├── Normalize → TranscribeGraph (shared representation)
    │
    ├── Semantic Transposer (shared)
    │   Idiom recognition, memory model translation, type mapping
    │
    └── Emit → Clef Source (.clef files)
```

### Why Transcribe Has Different Coupling Than the Main Pipeline

This is the architectural crux. The main pipeline (PSG → MLIR) and Transcribe
(Foreign AST → Clef AST) have fundamentally different coupling between recognition
and output:

**Main pipeline: tight coupling.** "This is an Application node with unsigned
operands" immediately determines "emit `arith.divui`." The observation and the
output are in the same domain — compiler IR in, compiler IR out. One PSG node kind
maps to a narrow set of MLIR operations. Recognition and emission are naturally
fused because the mapping is direct.

**Transcribe: loose coupling.** "This is a C `for` loop iterating over an array
with no `break`" could produce `Array.map`, `Array.iter`, a recursive function, or
a mutable `while` — depending on the loop body, the surrounding context, what Clef
idioms are preferred, and what the developer's intent appears to be. The foreign
pattern and the Clef output live in **different semantic domains**. One foreign
construct maps to many possible Clef constructs. Recognition and emission are
naturally separable because the mapping is indirect.

This asymmetry means:

| Property | Main Pipeline (Alex) | Transcribe |
|----------|---------------------|------------|
| Source domain | PSG (Clef compiler IR) | Foreign AST (C, TS, Python, etc.) |
| Target domain | MLIR (same ecosystem) | Clef AST (different ecosystem) |
| Recognition → output | 1:1 or 1:few | 1:many |
| Coupling | Tight (fuse naturally) | Loose (separate naturally) |
| Coeffects needed during recognition? | Yes (platform, SSA, widths) | No (foreign AST is read-only) |
| Accumulator mutations during recognition? | Yes (MLIR ops) | No (just identifying patterns) |
| Split would be forced? | Yes — introduces seam | No — follows natural boundary |

The last three rows are decisive. In the main pipeline, recognition reads coeffects
and mutates the accumulator — it's effectful, which is why splitting it from emission
is artificial. In Transcribe, recognition over a foreign AST is genuinely pure: you
are reading a foreign graph to identify idioms, not threading compiler state. The
foreign AST is immutable input with no coeffects attached. Recognition can be
algebraic because the foreign domain doesn't carry the state that makes the main
pipeline's patterns effectful.

### The Two-Model Architecture

The resolution: **fused model for the main pipeline, split model for Transcribe.**

```
Main Pipeline (fused):                    Transcribe (split):

  PSG Node                                  Foreign AST Node
     │                                           │
     ▼                                           ▼
  parser { }                               Recognition (algebraic)
     observe coeffects                        pure pattern matching
     navigate zipper                          boolean operators
     emit MLIR                                derivative-compatible
     accumulate results                        │
     ─── single block ───                      │  on match:
                                               ▼
                                            Emission (monadic)
                                              parser { }
                                              produce Clef AST
                                              accumulate fragments
                                              ─── fused block ───
```

The split in Transcribe is at a **different level** than what Section 4 rejected.
Section 4 rejected splitting recognition from emission **within a single pattern
over a single graph**. Transcribe's split is between **two different graphs in two
different domains**: foreign AST recognition (pure, algebraic) and Clef AST emission
(effectful, monadic). These are genuinely separable concerns, not an artificial
factoring of a naturally fused operation.

The emission layer in Transcribe is itself fused — a `parser { }` block that
produces Clef AST using the same accumulator model as Alex. The split is above it:
which emission block runs is determined by which recognition pattern matched.

### What the Split Model Enables for Transcribe

**Boolean operators on idiom recognition.** Since recognition over foreign ASTs is
pure, intersection and complement work correctly:

```fsharp
// Algebraic recognition patterns (pure, no state effects)
type IdiomPattern =
    | ForLoopOverArray                     // C: for(i=0;i<n;i++) a[i]
    | ForLoopOverArrayNoBreak              // ForLoopOverArray AND NOT ContainsBreak
    | AssignmentToFieldWithCast            // TS: (obj as T).field = expr
    | NumpyBroadcast                       // Python: arr + scalar
    | ...

// Derivative-compatible: D_K(P1 & ~P2) = D_K(P1) & ~D_K(P2)
let forLoopToMap =
    ForLoopOverArray <&> (~~ContainsBreak) <&> (~~ContainsGoto)
```

**Static pattern coverage analysis.** Since idiom patterns are algebraic data, you
can analyze whether a Transcribe adapter covers all foreign construct kinds. "Does
the C adapter have patterns for every C statement kind?" is answerable by inspecting
the pattern algebra, not by running the test suite.

**Idiom pattern reuse across languages.** "For-loop-as-map" is the same idiom in C,
Go, and Python. The algebraic recognition layer can share patterns across adapters
where foreign constructs have equivalent structure.

**Academic rigor.** The split model for Transcribe is a formally clean instance of
derivative-based graph pattern matching. Recognition patterns form an algebra with
union, intersection, complement, and navigation. Derivatives compute which patterns
survive after observing a node kind. This is publishable, rigorous, and directly
useful for the transliteration capability.

### The Showcase Relationship

The fused model in the main pipeline **demonstrates** that the architecture works
under tight coupling and effectful constraints. It is the proven foundation: 34 PRDs,
self-hosting, daemon mode, all executing through fused `parser { }` blocks.

The split model in Transcribe **formalizes** the derivative algebra as a showcase
that the fused model supports. The emission side of Transcribe uses the same
`parser { }` CE, the same accumulator threading, the same state model. The
recognition side adds the formal layer that the main pipeline doesn't need but
Transcribe benefits from.

The two models are not in tension. They are the same architecture applied at
different coupling strengths:

- Tight coupling (same domain) → fuse recognition and emission → main pipeline
- Loose coupling (cross-domain) → separate recognition from emission → Transcribe

### Compound Constraints in Foreign AST Pattern Matching

Foreign AST pattern matching involves compound constraints that arise naturally
because foreign languages have constructs Clef expresses differently:

**Intersection**: "Match a function call that is BOTH a known library function AND
has exactly two arguments of numeric type."

**Complement**: "Match any expression EXCEPT a literal or a simple variable
reference." Used for identifying expressions that need temporary bindings during
transposition.

**Both together**: "Match a loop that iterates over an array AND does NOT contain
break/continue/goto." Identifies loops that can be transposed to `Array.map` or
`Array.iter`. This is the idiom recognition problem — the core of Transcribe's
intelligence.

In the main pipeline, these are expressible as manual guards within fused `parser
{ }` blocks. In Transcribe, they are first-class algebraic operators on the
recognition layer — because that layer is pure and supports them natively.

### The RE# Connection

RE# (ReSharp) demonstrates that Brzozowski derivatives provide boolean operators
(intersection, complement) naturally for string-level regex. The same algebraic
framework extends to Transcribe's idiom recognition patterns. RE# also serves as
a potential component:

- **Lexer for partial-code scenarios**: When the foreign compiler isn't available
  (incomplete snippets, notebook magic commands, mixed-language cells), an RE#-style
  derivative-based lexer provides O(n) tokenization feeding XParsec structural
  parsing
- **Python regex transposition**: Python ML code uses regex heavily (tokenizers,
  data preprocessing). A derivative-based Clef regex engine maps Python regex
  patterns to Clef equivalents with better performance guarantees (O(n) vs.
  backtracking)
- **DTS-annotated captures**: A regex engine returning dimensionally-typed captures
  (`"3.14 m/s"` → `float<m/s>`) would be valuable for scientific data processing
  in Clef notebooks

### Sequencing

Transcribe depends on primitives that do not yet exist. The 34 remaining PRDs
establish the foundation. The natural ordering:

1. **34 PRDs** — complete the compiler's core feature set (fused model at full form)
2. **Self-hosting** (Clef and Composer fully supported) — the bootstrap milestone
3. **Daemon mode** (Stage 3) — LSP, incremental recomputation over the full PSG
4. **Jupyter kernel** (Stage 4) — interactive execution, ORC JIT, notebook protocol
5. **Transcribe** — split model introduced here, after the fused model is proven

Within Transcribe:
- C first (Farscape foundation), then C++ (same adapter)
- TypeScript next (Glutinum/Xantham foundation, CloudEdge integration)
- Python timed to Jupyter kernel availability (Stage 4)
- Rust and Go as the ecosystem matures

The fused model completes its full form through the 34 PRDs. The split model for
Transcribe arrives after, building on a proven foundation. The derivative algebra
formalizes what the fused model does intuitively — and Transcribe is the first
context where that formalization earns its keep as implementation rather than
specification.

If XParsec gains derivative-based internals upstream (its author is investigating
RE#), Transcribe's recognition layer composes with it directly. The fused model in
the main pipeline benefits from any performance improvements in XParsec's `<|>`
without adopting the split.

---

## 6. Summary

Composer's PSG pattern matching is a novel computational model: parser combinator
backtracking repurposed for graph observation with monadic state accumulation. It
sits between sequence parsers and tree automata, optimized for the specific demands
of compiler IR traversal where recognition and code generation are fused.

The fused model — one `parser { }` block per Pattern, no internal seams — is not
an accident of implementation. It is the result of removing factorings that failed
(PSGEmitter, PSGScribe, hidden helpers). The architecture is structurally resistant
to corruption because there is no boundary where coordination infrastructure can
form. A codata architecture resists decomposition; you observe it whole. The main
compiler pipeline — through all 34 PRDs, self-hosting, and daemon mode — stays
fused.

The Transcribe feature area introduces a split model **only for cross-domain
transposition** (Foreign AST → Clef AST), where the coupling between recognition
and emission is naturally loose. Recognition over foreign ASTs is genuinely pure —
the foreign graph is read-only, carries no coeffects, and requires no accumulator
mutations. This makes the split natural rather than forced, and enables the
derivative algebra with boolean operators (intersection, complement) and static
pattern analysis that the main pipeline neither needs nor benefits from.

The two models are the same architecture applied at different coupling strengths:

- **Tight coupling** (same domain: PSG → MLIR) → fuse → main pipeline
- **Loose coupling** (cross-domain: Foreign AST → Clef AST) → split → Transcribe

The fused model is the proven core. The split model for Transcribe is the formal
showcase — demonstrating with academic rigor that derivative-based graph pattern
matching works, built on a foundation that was already proven in its fused form.
Transcribe's emission layer is itself fused `parser { }` blocks, using the same
accumulator model as Alex. The split exists above it, at the domain boundary.

This document records the analysis for future reference. The derivative algebra is
a formalization target realized through Transcribe. The fused model is the
implementation target for the main compiler pipeline, now and permanently.
