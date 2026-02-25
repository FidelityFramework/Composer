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

## 4. A Possible Formalization

To get a true derivative-based framework, separate two concerns that PSG patterns
currently fuse:

### Recognition (Algebraic, Derivative-Compatible)

Given a node kind `K` and a pattern set `P`, which patterns survive?

```
D_K(P) = { p ∈ P | p accepts observation K at this position }
```

Define a pattern algebra:

```fsharp
type GraphPattern<'Obs, 'R> =
    | Observe of ('Obs -> bool) * 'R       // leaf: accept/reject on observation
    | Sequence of GraphPattern * GraphPattern
    | Choice of GraphPattern * GraphPattern       // union  (existing <|>)
    | Intersect of GraphPattern * GraphPattern    // AND    (new)
    | Complement of GraphPattern                  // NOT    (new)
    | Navigate of Direction * GraphPattern        // move, then match
    | Recursive of (unit -> GraphPattern)         // fixed point for cycles
```

Derivative operation:

```fsharp
let rec derivative (obs: 'Obs) = function
    | Observe (pred, r) -> if pred obs then Epsilon r else Empty
    | Choice (p1, p2) -> Choice (derivative obs p1, derivative obs p2)
    | Intersect (p1, p2) -> Intersect (derivative obs p1, derivative obs p2)
    | Complement p -> Complement (derivative obs p)
    | Sequence (p1, p2) ->
        if nullable p1
        then Choice (Sequence (derivative obs p1, p2), derivative obs p2)
        else Sequence (derivative obs p1, p2)
    | Navigate (dir, inner) -> Navigate (dir, derivative obs inner)
    | Recursive f -> derivative obs (f ())
```

Intersection and complement fall out for free — the same algebraic closure that
RE#'s string-level derivatives provide.

### Emission (Effectful, Monadic)

The effectful part — reading coeffects, accumulating MLIR, updating state — stays
monadic. It executes only after recognition succeeds:

```
Pattern = Recognition × Emission

Recognition: algebraic, derivative-compatible, statically analyzable
Emission: effectful, state-dependent, runs on successful recognition
```

Connected by bind:

```fsharp
let pBinaryArithOp nodeId operation =
    recognizeBinaryArith >>= fun _ -> emitBinaryArith nodeId operation
```

The recognition layer gets derivatives, intersection, complement.
The emission layer stays as `parser { }` CE blocks.
The `>>=` at the boundary connects them.

### What This Enables

- **Intersection**: "match A AND B at the same node" — useful for type-guarded
  patterns where multiple properties must hold simultaneously
- **Complement**: "match anything EXCEPT A" — useful for catch-all patterns that
  exclude known cases
- **Pattern set analysis**: With algebraic representations, analyze combinator chains
  for coverage (do patterns cover all `SemanticKind` cases in a witness?) and overlap
  (do two patterns both match the same node kind + type combination?). This is
  analysis of the **pattern algebra itself**, not of PSG node reachability — which is
  a completely different concern handled upstream by CCS Phase 3
- **Recognition memoization**: If recognition is pure (separated from emission),
  cache `D_K(patterns)` per node kind `K` to accelerate witness dispatch

### What It Costs

- **Separating recognition from emission breaks the current elegance.** A pattern is
  currently a single `parser { }` block that recognizes and emits in one flow. The
  monadic style makes this natural. Forcing a separation adds conceptual overhead.
- **Derivative compaction for graphs.** The Might et al. paper (ICFP 2011) needed
  careful fixed-point handling for recursive CFG productions. Graph patterns with
  cycles would need analogous treatment.
- **The current model works.** 6,250 lines across layers, 10 passing regression
  samples, FPGA and CPU targets generating correct code. The engineering case for
  refactoring the recognition model is weak unless a concrete feature requires
  intersection/complement.

---

## 5. The Transcribe Feature Area

Transcribe is a future Composer feature for porting programs from other languages
into Clef source. It is the concrete motivating case for whether the formalization
in Section 4 is worth pursuing.

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
    │   XParsec graph patterns over TranscribeGraph
    │   Idiom recognition, memory model translation, type mapping
    │   Same architecture as Alex's PSG→MLIR walk, different source and target
    │
    └── Emit → Clef Source (.clef files)
```

The Semantic Transposer is architecturally identical to Alex: zipper navigation,
XParsec pattern matching, monadic state accumulation. The difference is that Alex
consumes a PSG and produces MLIR, while Transcribe consumes a TranscribeGraph and
produces Clef AST.

### Why Transcribe Needs Boolean Pattern Operators

Foreign AST pattern matching requires constraints that current XParsec cannot express
declaratively:

**Intersection**: "Match a function call that is BOTH a known library function AND
has exactly two arguments of numeric type." In current XParsec, this requires nested
`if` guards. With intersection: `pKnownLibCall <&> pBinaryNumericArgs`.

**Complement**: "Match any expression EXCEPT a literal or a simple variable
reference." Used for identifying expressions that need temporary bindings during
transposition. Currently requires enumerating all positive cases. With complement:
`pAnyExpr <&> (~~(pLiteral <|> pSimpleVarRef))`.

**Intersection + Complement together**: "Match a loop that iterates over an array
AND does NOT contain break/continue/goto." Identifies loops that can be transposed to
`Array.map` or `Array.iter`. This is the idiom recognition problem — the core of
Transcribe's intelligence.

These compound constraints arise naturally when pattern-matching foreign ASTs because
foreign languages have constructs that Clef expresses differently. The transposer
must recognize specific combinations of foreign constructs and map them to idiomatic
Clef. Boolean operators on patterns make these recognizers compositional rather than
monolithic.

### The RE# Connection

RE# (ReSharp) demonstrates that Brzozowski derivatives provide boolean operators
(intersection, complement) naturally for string-level regex. The same algebraic
framework extends to the graph pattern algebra described in Section 4. RE# also
serves as a potential component:

- **Lexer for partial-code scenarios**: When the foreign compiler isn't available
  (incomplete snippets, notebook magic commands, mixed-language cells), an RE#-style
  derivative-based lexer provides O(n) tokenization feeding XParsec structural
  parsing
- **Python regex transposition**: Python ML code uses regex heavily (tokenizers,
  data preprocessing). A derivative-based Clef regex engine maps Python regex patterns
  to Clef equivalents with better performance guarantees (O(n) vs. backtracking)
- **DTS-annotated captures**: A regex engine returning dimensionally-typed captures
  (`"3.14 m/s"` → `float<m/s>`) would be valuable for scientific data processing in
  Clef notebooks

### Sequencing

Transcribe depends on primitives that do not yet exist. The natural ordering:

1. **Self-hosting** (Clef and Composer fully supported) — the bootstrap milestone
2. **Daemon mode** (Stage 3) — LSP, incremental recomputation over the full PSG
3. **Jupyter kernel** (Stage 4) — interactive execution, ORC JIT, notebook protocol
4. **Transcribe** — consumes the above as infrastructure

Within Transcribe:
- C first (Farscape foundation), then C++ (same adapter)
- TypeScript next (Glutinum/Xantham foundation, CloudEdge integration)
- Python timed to Jupyter kernel availability (Stage 4)
- Rust and Go as the ecosystem matures

The formalization work (Section 4) can proceed independently as research. If XParsec
gains derivative-based internals (its author is investigating RE#), the recognition
layer comes as a library upgrade. If not, the recognition/emission separation can be
built as a Composer-local abstraction over XParsec's existing `<|>`.

---

## 6. Summary

Composer's PSG pattern matching is a novel computational model: parser combinator
backtracking repurposed for graph observation with monadic state accumulation. It
sits between sequence parsers and tree automata, optimized for the specific demands
of compiler IR traversal where recognition and code generation must be fused.

The derivative analogy is real at the `<|>` level but breaks at the effectful
emission level. A full formalization would separate recognition (algebraic,
derivative-compatible) from emission (effectful, monadic), connected by bind. This
would unlock boolean operators (intersection, complement) and pattern set analysis
that the current model cannot express.

The Transcribe feature area — porting C/C++, TypeScript, Python, Rust, and Go
programs into Clef — is the concrete motivating case for this formalization. Foreign
AST pattern matching requires compound constraints (match A AND B, match anything
EXCEPT C) that boolean operators express naturally. Transcribe follows after
self-hosting, daemon mode, and Jupyter kernel infrastructure are in place.

This document records the analysis for future reference. It is a formalization
target, not a near-term engineering task.
