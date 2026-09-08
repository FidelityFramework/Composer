# CCS: Clef Compiler Service

CCS parses and checks Clef source, constructs the Program Semantic Graph (PSG), and owns the semantic
facts consumed by Composer and the local Lattice server. This page describes that boundary and the
current implementation. Language requirements come from the [Clef specification](https://github.com/FidelityFramework/clef-lang-spec),
particularly [units of measure](https://clef-lang.com/spec/draft/units-of-measure/),
[NTU types](https://clef-lang.com/spec/draft/ntu-types/), and
[preservation through lowering](https://clef-lang.com/spec/draft/conformance/#6-the-preservation-obligation-through-lowering).

See [Architecture_Canonical.md](Architecture_Canonical.md) for Composer's pipeline and
[Lattice_Integration.md](Lattice_Integration.md) for the coordinated editor implementation and acceptance gates.

## Current implementation boundary

Composer's [project reference](../src/Composer.fsproj#L186) selects the peer
`clef/src/Compiler/Clef.Compiler.Service.fsproj`. The source descriptions below refer to that checkout's
`main` implementation. The dimensional work on `fidelity` has separate commits and unfinished worktree
changes. The local editor demo uses Composer's existing main reference; it does not reconcile those two
lines of work. A reviewed, compatible compiler revision remains necessary before claiming that alignment.

CCS and Composer currently run on .NET. The compiler's implementation language and host libraries do not
define the semantics or storage layout of a Clef program. The service constructs native types and graph
nodes directly; Lattice consumes those results without checking the document with a separate FCS instance.

| Entry point | Current responsibility |
|---|---|
| [FidprojLoader](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/Project/FidprojLoader.fs) | Read project sources, dependencies, platform metadata, and output context |
| [SourceResolver](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/Project/SourceResolver.fs) | Resolve sources in dependency order |
| [ProjectChecker](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/Project/ProjectChecker.fs) | Check ordered sources together; accept unsaved source overrides; return graph, diagnostics, source texts, and parse errors |
| [NativeService](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/NativeTypedTree/NativeService.fs) | Parse and type-check, construct the graph, and run semantic enrichment and saturation |
| [Composer integration](../src/FrontEnd/CCS/Integration.fs) | Expose CCS results to Composer's lowering and emission code |

`checkProjectWithVolatile` already accepts a map of absolute paths to unsaved text in both compiler lines.
It is a project-checking function, not a persistent workspace session. Project loading and platform inputs
remain part of the check; the editor must not maintain a competing interpretation of them.

## Dimensional types, ranges, and representations

### Dimensional identity

Dimensions use the free abelian-group algebra of declared base measures and measure variables. Addition
requires equal dimensions; multiplication adds their exponents; division subtracts them. The current
[DimensionAlgebra](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/NativeTypedTree/DimensionAlgebra.fs)
normalizes these expressions and solves measure equations using integer exponents and substitutions.
The type checker combines this measure algebra with ordinary type constraints and polymorphism.

The implemented numeric form is `TNum(carrier, dimension)` in
[NativeTypes](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/NativeTypedTree/NativeTypes.fs#L1445).
A dimension participates in source type identity; it is not merely a hover label. Measure-sorted arguments
on other type constructors use `TMeasure`, with parameter kinds retained through instantiation.

The supported measure algebra provides a decidable constraint problem. That fact alone supplies neither
a language-server latency bound nor a completeness claim for every program query. Parsing, project
resolution, range analysis, pending constraints, and solver work have separate costs and failure states.
An editor response-time claim needs measurements against the implemented session and workloads.

### Range and representation coeffects

A value's numeric kind and dimension are distinct from its possible values and the representation selected
for those values. The intended order is:

```text
source kind and dimension + justified range + declared target capabilities
    -> representation and layout decisions in CCS
    -> recorded graph facts consumed by target lowering
```

[Width Inference](https://clef-lang.com/spec/draft/width-inference/) and
[Numeric Selection](https://clef-lang.com/spec/draft/numeric-selection/) govern these decisions. A boundary
can fix the permitted representation; CCS must still establish coverage and the applicable transfer-fidelity
obligations. A dimension alone selects neither an integer width nor a posit configuration. A platform
capacity also does not prove that an arbitrary value fits it.

The current main implementation has an integer
[RangeAnalysis](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/PSGSaturation/SemanticGraph/RangeAnalysis.fs)
pass. It writes node `ValueRange`, aggregate `FieldRanges`, `ElementRanges`, and escaping-value facts.
`PlatformDeclaration.fill` obtains declared platform facts before that pass; `Placement.settle` follows it
and supplies aggregate layouts. Required unresolved integer ranges and declaration failures produce compiler
diagnostics. The real interval domain, general real-format selection, and quire work have their own
implementation scope in [Numeric_Selection_Implementation.md](Numeric_Selection_Implementation.md).

The internal carrier representation still contains width-bearing `NTUKind` cases and named constructors.
The `TNum` declaration explicitly identifies their width/name treatment as interim. Those internal cases
are not a source-level prescription to write `int32`, or a license for Alex to choose a machine-word default.
The remaining carrier-identity work must be reconciled with the source rules and tested on the aligned
compiler revision.

### Preservation through compilation

Composer consumes the facts CCS establishes. Each lowering must preserve an applicable justification or
re-check the affected obligation, as required by
[Conformance §6](https://clef-lang.com/spec/draft/conformance/#6-the-preservation-obligation-through-lowering).
A property established in the source checker is not, by itself, evidence that a later transformation preserves it.

Dimensions, ranges, representations, and the dependencies of their proofs must remain available until their
consumers and preservation checks are satisfied. They may be carried in compiler metadata or checked
witnesses. Final machine code or wire payloads need not retain runtime type tags. This requirement does not
assert that every current lowering has a complete `clef.dim` attribute path, debug-metadata encoding, or
solver-backed preservation check.

## The Native Type Universe

The NTU is the compiler's vocabulary for native type and representation work. Source identity, target
representation, and host implementation types must remain distinct:

| Concern | Owner and source of truth |
|---|---|
| Numeric kind and dimension | Clef typing rules; CCS type and measure constraints |
| Integer range and chosen width | CCS range analysis against justified inputs and platform declarations |
| Real representation | Numeric-selection requirements and the supported implementation for the target |
| Pointer, region, and access identity | [FFI boundary](https://clef-lang.com/spec/draft/ffi-boundary/) and [memory regions](https://clef-lang.com/spec/draft/memory-regions/) |
| Aggregate and closure placement | CCS layout and lifetime analysis; target lowering realizes the result |
| Host storage used by the compiler | An implementation choice, not the numeric or memory model of Clef source |

For example, a Clef string's native layout follows its
[representation contract](https://clef-lang.com/spec/draft/native-type-mappings/#strings), rather than the layout of
`System.String` used by the hosted compiler. Similarly, pointer width follows the declared target boundary.
Internal raw-pointer plumbing is not a user-facing replacement for region/access-aware pointer contracts.
Concrete formats and fixed-width carriers belong in the representation implementation and boundary
descriptors, not in a table of architecture-dependent source defaults.

## DMM: lifetime, escape, and placement

Memory placement is derived from a value's lifetime requirements and the target's available storage.
The [closure representation](https://clef-lang.com/spec/draft/closure-representation/) and
[memory-region](https://clef-lang.com/spec/draft/memory-regions/) chapters state the capture and placement
requirements. Immutable captures preserve their values and sharing; mutable captures preserve shared storage.
A closure environment must live long enough for every closure that uses it.

Main contains compiler-owned [escape analysis](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/PSGSaturation/SemanticGraph/Escape.fs)
and [placement](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/PSGSaturation/SemanticGraph/Placement.fs).
Its `EscapeKind` distinguishes `StackScoped`, `EscapesViaClosure`, `EscapesViaReturn`, `EscapesViaByRef`,
and `StaticLifetime`. These are analysis results, not a one-to-one table assigning every return or reference
escape to a particular arena. Placement also depends on the available lifetime and storage context.

The implementation records escape results, closure placement, and related information in
`SemanticGraph.Codata`, with aggregate layouts in `Layouts`. Some of these fields are lazy compiler-owned
projections. Their presence establishes the analysis owner. The local `CCS.Editor` service freezes its
supported type, range and obligation read views before a subsequent check can mutate compiler cells;
further layout/lifetime presentations need the same discipline.

This architecture does not make every proposed lifetime obligation complete. The specification still names
remaining lifetime-ordering rules, and implementation coverage must be established by concrete cases.
A source `inline` annotation or suggested restructuring is not a universal proof that an allocation is safe.
Any such transformation must preserve the relevant capture and lifetime constraints.

## PSG facts and their current status

The [graph types](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/PSGSaturation/SemanticGraph/Types.fs)
carry nodes with source ranges, types, parents/children, SRTP resolution, layout hints, metadata, and
reachability. Modules, declaration roots, type indexes, and compiler-owned projections provide further
structure for consumers.

### Reachability

The current node field is `IsReachable: bool`. NativeService selects soft marking or pruning according to
phase configuration. `Live`, `Latent`, and `Fresh` are not the current node-state API, and a per-target
reachability bitvector is not present in this graph type. No boundary-size reactivation complexity is promised.

Diagnostics have a separate `ReachabilityContext` of `Reachable`, `Unreachable`, or `Unknown`.
[Diagnostic.effectiveSeverity](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/PSGSaturation/SemanticGraph/Diagnostics.fs)
provides the compiler's presentation severity. The editor preserves this result rather than deriving its own
reachability policy. Pending inference and parse failure must also remain distinguishable from unreachability.

### Coeffects computed during elaboration and saturation

| Fact or projection | Current main implementation and limit |
|---|---|
| Types, dimensions, and SRTP | Attached during checking; diagnostics expose failures, while a general editor-facing pending-state contract remains to be implemented |
| Capture information and emission strategy | Present on graph structures used by closure and emission processing; validate the relevant source form |
| Integer ranges and declared representations | `RangeAnalysis.run` after platform declaration processing; separate from the planned general real-selection domain |
| Aggregate layouts and closure placement | `Placement.settle`, `Placement.closures`, and `Layouts`; supported forms require their own adequacy tests |
| Escape, curry normalization, meets, bindings, pins, declaration roots | Compiler producers feed `Codata`; several projections are lazy |
| Obligation nodes and hyperedges | `ObligationElaboration` constructs supported obligations and their graph relationships |
| Solver-input artifacts | `ObligationDischarge` projects those nodes to JSON and SMT-LIB when artifact emission is enabled; it returns no solver verdict |
| SSA names | Derived during emission from node identity in [Values.fs](../src/MiddleEnd/Alex/Traversal/Values.fs); not a pre-assigned CCS coeffect |
| Workspace versions, cancellation, live notifications | An editor-session deliverable in [Lattice_Integration.md](Lattice_Integration.md), not fields supplied by the current project-check result |

The producer sequence is visible in
[NativeService](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/NativeTypedTree/NativeService.fs#L990).
Alex observes the selected facts and realizes them in target operations. It owns emission mechanics; it
must not silently reconstruct a missing semantic fact or choose a different representation.

### Proof obligations and verdicts

[ObligationDischarge.ofGraph](https://github.com/FidelityFramework/clef/blob/main/src/Compiler/Nanopass/ObligationDischarge.fs#L125)
returns the graph's obligations; `ledgerJson` and `smtLib` produce their projections. `emit` writes
`06a_obligations.json` and `06b_obligations.smt2` when enabled. It does not invoke cvc5, record a verdict in
`CheckResult`, or provide a live editor notification.

Composer's [SMTTransfer](../src/MiddleEnd/Alex/Traversal/SMTTransfer.fs) participates in the source/lowering
correspondence scaffold. A paired anchor or generated SMT file does not itself prove that the emitted host
operations implement the modeled contract. A reported verdict needs an actual dispatch result tied to the
obligation, its premises, its encoding, and the checked input revision. Live dispatch and editor evidence
status are separate integration work.

## The editor service boundary

CCS supplies the semantics; the .NET-hosted [Lattice server](../src/Lattice.Server/README.md) in Composer's
solution supplies the LSP adapter. [CCS.Editor](../src/CCS.Editor/README.md) wraps project checking with
serialized sessions, immutable read projections, compiler identity and input revisions. The adapter binds
those checks to document versions and watches root, platform and dependency inputs. Superseded checks
finish and are discarded; the current compiler has process-global state and no safe interruption API.
Parser errors remain file-associated strings, shown in output and as a failed semantic/proof request.
Structured parser locations are still needed for reliable squiggles.

The editor projection indexes compiler source intervals for hover and follows resolved reference identities
for definitions, including shadowing and cross-file references. Completion and rename need further
compiler-facing queries; `getBindings` alone does not establish the bindings visible at a source position.

The local VSCode gate exercises a two-file project using unsaved source overrides: dimensional hover,
CCS8040 appearing and clearing, resolved definitions and expandable compiler obligations checked by cvc5.
The shared [integration plan](Lattice_Integration.md#implementation-gates) records the wider gates, including
Neovim semantic checks and compiler reconciliation. Stale checks cannot replace newer responses, and
incomplete source must not silently switch to the ordinary F# checker.

There is no installable Lattice server command or completed per-target representation panel assumed here.
Cache-locality predictions, restructuring suggestions, and cross-target latency displays require their own
compiler facts and validation before an editor can present them as analysis.

## Layer separation

| Layer | Responsibility | Boundary |
|---|---|---|
| CCS | Project context, source typing, dimensions, ranges, saturation, layout/lifetime facts, diagnostics, and supported obligations | Reads target declarations; does not emit target machine code |
| Composer | Compile requests, lowering, artifact generation, and preservation checks; initial host for the Lattice server | Consumes the selected CCS revision and its semantic facts |
| Alex/Zipper | Traverse graph structure and realize settled facts through target bindings | Does not perform independent type inference or silently choose missing widths/layouts |
| Fidelity.Platform | Target capabilities, representations, memory spaces, bindings, and boundary contracts | Declarations constrain compiler reasoning; a capacity declaration alone is not a source-value bound |
| Lattice | Document transport, bounded analysis requests, and presentation of versioned compiler results | CCS remains the authority for semantic facts and their evidence |

The same division applies to analyzer migration: compiler diagnostics own semantic judgments, graph queries
supply established facts, and clients handle presentation. Substantial analysis can be added through concrete,
bounded requests once type resolution and Baker/saturation have carried the required facts. Each addition
needs a stated input snapshot, compiler stage, premises, result contract, and acceptance cases. Optional
requests and display choices leave mandatory compiler checks in force. This stages useful analysis without
requiring a general plugin framework for the first editor integration.

A type-checked declaration or accepted library lemma still needs its use-site premises; admitting arbitrary
predicates is not part of dimensional algebra. Hosting these services on .NET supplies implementation
facilities, not permission to substitute FCS semantics for a Clef document.

## Related documentation

- [Lattice_Integration.md](Lattice_Integration.md): compiler alignment, session design, repository responsibilities, and acceptance gates.
- [Architecture_Canonical.md](Architecture_Canonical.md): Composer pipeline and the Alex traversal boundary.
- [Numeric_Selection_Implementation.md](Numeric_Selection_Implementation.md): range evidence, selection, and quire implementation work.
- [NTU_Architecture.md](NTU_Architecture.md): compiler representation detail; reconcile implementation notes with the normative NTU source rules.
- [PSG_Nanopass_Architecture.md](PSG_Nanopass_Architecture.md): nanopass organization and graph construction.
- [Platform_Binding_Model.md](Platform_Binding_Model.md): platform declaration and binding integration.
- [Lattice consumer contract](https://github.com/FidelityFramework/clef/blob/main/docs/fidelity/phg/Lattice_Consumer_Contract.md): compiler ownership of editor facts, with current implementation status qualified above.
