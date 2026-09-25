# Composer Compiler - Claude Context

## Language Coverage Checkpoints

When continuing language/compiler work, read the latest entries in
`docs/Language_Coverage_Waypoints.md` and their referenced contracts alongside
the roadmap. Keep that record current at each completed checkpoint: concrete
scope, validation results, companion repository revisions, and the next unsettled
contract. Distinguish pending implementation from tested behavior so later
sessions can resume from the recorded state.

The Clef specification governs language semantics. Current implementation
summaries are [Architecture_Canonical.md](docs/Architecture_Canonical.md),
[Alex_Architecture_Overview.md](docs/Alex_Architecture_Overview.md), and the
[Baker contract](../clef/docs/fidelity/Baker_Saturation_Architecture.md). Historical
FCS overlay, parallel-witness and SSA-preassignment sketches are not instructions
for current work. M-01 admission/evidence transport remains planned.

## Alex Architecture

The Element/Pattern/Witness model uses XParsec for graph observation and physical
operation composition:

```text
Elements/    (module internal)  -> Atomic MLIR operations
Patterns/    (public)           -> Compose Elements for an admitted physical form
Witnesses/   (public)           -> Observe through ctx and invoke Patterns
```

`module internal` restricts assembly visibility, not access by sibling folders.
Witnesses are in the same assembly and can import Elements; layering is maintained
through API use, review and tests. There is no correctness-bearing line-count
limit for Patterns or Witnesses.

### Golden Rules

1. Baker settles; Alex witnesses. Preserve ctx pull through graph/codata,
   coeffects and the actual Huet position. Do not reconstruct source semantics,
   captures, layout or proof premises during emission.
2. Missing semantic prerequisites remain explicit failures in the owning stage.
   Fix CCS/Baker when settlement is wrong; extend Elements/Patterns/Witnesses when
   the admitted settled form lacks physical expression. Do not create an alternate
   emitter to bypass either boundary.
3. **NEVER create git commits** — that is the user's responsibility.

## Decision-Making Discipline

Apply the architectural constraints before presenting alternatives:

1. Use Clef-native contracts, including typed `FnPtr<'F>` and declared native
   dimensions/representations, rather than importing CLR delegates, default-value
   semantics or untyped pointer substitutes.
2. Preserve type, source/graph identity and evidence at boundaries. Target widths
   and layouts come from the declared platform and owning compiler contracts;
   do not insert a convenient CPU width as a language rule.
3. Compose current Ingredients/Recipes and Elements/Patterns/Witnesses where their
   contracts match. A new semantic protocol needs its owning graph construction,
   not a copied special case below the witness boundary.
4. Distinguish the current type algebra and tested coverage from planned features.
   Do not claim an old sketch or a successful component fixture establishes native
   source-language behavior.

The compiler bootstrap is implemented in F#. That implementation language is not
Clef's execution model. In the planned
[workbench](docs/Interactive_Compiler_Workbench.md), FSI may inspect and exercise
compiler functions; accepted Clef behavior still traverses the real compiler and
native gates. A hot patch requires saved-source rebuild and fresh-session replay
before acceptance. Ordinary build/test fallback remains available.

## Architectural Principles

### Fix the owning stage

Trace the defect through the actual pipeline:

```text
Clef source + project/platform inputs
  -> CCS checking and typed PSG construction
  -> Baker elaboration/fold-in and owning saturation/admission passes
  -> Alex ctx pull / Huet traversal -> admitted MLIR
  -> selected backend realization -> native artifact and execution
```

There is no separate FCS stage or FSharpExpr typed-tree overlay in this pipeline.
Before a fix, establish where the first incorrect fact or missing relationship
appears. Do not patch a later symptom by matching source/library spellings in
code generation or inventing new proof premises.

### Layer Separation

| Layer | Does | Does NOT |
|---|---|---|
| CCS checking | Parse/check Clef, resolve project inputs, construct native types and typed graph relationships. | Delegate Clef semantics to FCS or the CLR. |
| Baker / owning CCS nanopasses | Elaborate and saturate computation, evaluation/capture/layout relationships and obligations under declared platform facts. | Defer semantic repair to Alex or LLVM. |
| Alex | Observe settled graph and target facts at the Huet focus; compose the admitted physical form. | Infer a missing source algorithm or obligation premise. |
| Backend | Realize the selected target and preserve/recheck affected properties. | Treat a successful MLIR verifier/link as proof of all source properties. |
| Host / editor | Schedule versioned requests and present compiler/evidence results. | Maintain a second semantic checker or emitter. |

### Traversal, values and state

The Huet zipper contains focus, path and graph, with no mutable fields or counters.
`TransferCoeffects` currently carries platform reads and target selection;
program facts are read from nodes, layouts, ranges and codata. The emission
accumulator, scopes and visited sets are separate mutable bookkeeping, not
permission to construct semantics in Alex.

`Values.fs` derives SSA names from graph node/role ordinals and block arguments.
CCS does not preassign Alex SSA numbers. Operand/type recall remains a distinct
emission concern. The current traversal combines registered witnesses at each
node in one post-order walk, with scope-owned callbacks. Do not assume parallel
witness traversals or reentrant compilation: target selection and registration
still include process-global mutable state.

Flat/thin MLIR retains admitted structured operations, regions and block
arguments. [M-01](docs/PRDs/M-01-DialectAdmission.md) governs expression family ×
platform/backend profile × witness form; it is not a blanket five-dialect support
claim. The current post-witness middle-end pass collects declarations. General
fact/proof transport and reconciliation of existing target paths remain planned.

## Negative Examples

- Source-name special cases in codegen instead of settled node kinds, intrinsic
  markers and binding facts.
- Semantic decomposition in Alex when the missing behavior belongs in Baker.
- Re-rooting a shared graph node and assuming its occurrence-dependent Huet scope
  survived.
- Calling a mutable operation accumulator a semantic analysis or putting it into
  the zipper. Physical result recall does not supply missing source facts.
- Reintroducing an FCS typed-tree overlay, Composer application-flattening pass,
  SSA-preassignment pass or second emitter from historical documentation.
- Deleting graph structure/provenance without the owning reachability/fold-in
  contract. Current soft-delete traversal reads `IsReachable`; it does not invent
  reachability while emitting.
- Replacing Clef behavior with F# evaluation or C helpers to make a REPL example
  pass. Legitimate host/foreign boundaries retain their own explicit contracts.

## Current Source and Documentation

| Resource | Purpose |
|---|---|
| [Clef specification](../clef-lang-spec/README.md) | Language and representation authority. |
| [Baker architecture](../clef/docs/fidelity/Baker_Saturation_Architecture.md) | Ingredients/Recipes, graph construction and saturation boundary. |
| [CCS architecture](docs/CCS_Architecture.md) | Compiler service and graph facts. |
| [Alex overview](docs/Alex_Architecture_Overview.md) | Context, traversal, physical expression and current evidence limits. |
| [Thin Middle End](docs/Thin_Middle_End_Design.md) | Semantic/witness/backend boundary. |
| [M-01](docs/PRDs/M-01-DialectAdmission.md) | Planned admission and information-transport work. |
| [LLVM backend](docs/LLVM_Backend.md) | Direct LLVM/LLD ELF path and runtime inputs. |
| [Workbench](docs/Interactive_Compiler_Workbench.md) | Planned resident host and native REPL bridge. |

| Current file | Purpose |
|---|---|
| `src/Composer.fsproj` | Main compiler project and compile order. |
| `src/Core/CompilationOrchestrator.fs` | Source/target gates and pipeline orchestration. |
| `src/FrontEnd/ProjectLoader.fs` | Calls CCS ProjectChecker. |
| `../clef/src/Compiler/Project/ProjectChecker.fs` | Project/dependency/platform checking and volatile inputs. |
| `../clef/src/Compiler/NativeTypedTree/NativeService.fs` | Native checking and graph pipeline. |
| `../clef/src/Compiler/Baker/` and `../clef/src/Compiler/Nanopass/` | Construction, recipes and saturation/admission passes. |
| `src/MiddleEnd/MLIRGeneration.fs` | Alex ingress and serialization. |
| `src/MiddleEnd/Alex/Traversal/` | Huet navigation, witness context, derived values, traversal and coverage checks. |
| `src/MiddleEnd/Alex/Patterns/` and `Witnesses/` | Physical forms and passive graph observation. |
| `src/BackEnd/LLVM/` | LLVM realization, target bitcode and direct LLD linking. |

F# upstream sources and specifications can explain bootstrap/parser history;
they do not override Clef's own contracts. Alloy is historical context. Use
available code-navigation tools and text search against current paths.

## Build & Test

Coordinate shared compiler builds before running tests that rebuild Composer or
its CCS dependency. From the Composer root:

```bash
dotnet build src/Composer.fsproj
src/bin/Debug/net10.0/Composer compile samples/console/FidelityHelloWorld/01_HelloWorldDirect/HelloWorld.fidproj -k
samples/console/FidelityHelloWorld/01_HelloWorldDirect/targets/helloworld
```

The [regression runner](tests/regression/README.md) is the main sample gate:

```bash
cd tests/regression
dotnet fsi Runner.fsx
dotnet fsi Runner.fsx -- --verbose
dotnet fsi Runner.fsx -- --sample 05_AddNumbers
```

It builds the compiler and runs samples sequentially; `--parallel` is unsupported.
Compiler changes require the owning graph, source-admission, Alex, proof and
native gates as applicable. The regression runner must pass and binaries must
execute correctly; a narrowed or empty selection cannot establish full coverage.
Document-only work records document validation rather than inventing runtime
results. The [Alex suite](tests/Alex.Tests/README.md) explicitly distinguishes
component fixtures from source-language and native acceptance.

### Intermediate Artifacts

With `-k`, inspect the sample's `targets/intermediates/` in pipeline order:

| Artifact | Stage |
|---|---|
| `01_psg0.json` | Initial typed PSG/reachability view. |
| `02_intrinsic_recipes.json`, `03_psg1.json` | Intrinsic recipes and fold-in. |
| `04_saturation_recipes.json`, `05_psg2.json` | Baker recipes and final graph view. |
| `07_output.mlir` | MLIR retained by the orchestrator; transfer also writes its diagnostic intermediate at this path. |
| `08_after_declaration_collection.mlir` | Middle-end declaration collection. |
| `09_obligations.mlir` | Optional SMT obligation module; emission is not discharge. |
| `10_output.mlir` | Final middle-end serialization. |
| `08_output.ll` and associated bitcode | LLVM backend handoff. |

Additional owning passes retain their own diagnostics/evidence. The old
`06_coeffects.json` name remains reserved in PhaseConfig; do not assume a current
separate Composer coeffect-analysis pass or an emitted file from that reservation.

## Project Configuration

Use an existing project/profile appropriate to the target. A CPU project uses
`target = "cpu"`; its platform dependency is part of checking, not an optional
host inference. For example, the
[direct HelloWorld project](samples/console/FidelityHelloWorld/01_HelloWorldDirect/HelloWorld.fidproj)
selects Fidelity.Platform explicitly. `output_kind` controls deployment/runtime
inputs and does not define Clef semantics.
