# Alex component tests

Tests run serially inside this process because source-backed fixtures and node
builders share CCS's process-local identity and variable supplies. Parallelize
independent check jobs in isolated processes, as described in the
[regression policy](../../docs/Regression_Check_Policy.md#concurrency-evidence-and-honest-outcomes).

Run from Composer after coordinating any shared compiler builds:

```sh
dotnet test tests/Alex.Tests/Alex.Tests.fsproj
```

The project references the current Composer project and its selected CCS project.
`mlir-opt` must be on `PATH`; unavailable or failing tooling fails the component
tests. Tooling was inspected with LLVM/MLIR 22.1.8 when this suite was introduced.

These tests cover the existing public observation boundary:

- Huet navigation preserves graph identity, range facts and obligation edges;
  the same shared node has the enclosing scope of its actual position.
- The binding-name parser observes the zipper parent and restores its focus.
- `pArrayGetIntrinsic` reads the index range from the graph, composes the internal
  load element, and preserves the supplied facts and previously recalled values.
- Missing recalled operands and missing memory carrier types produce the existing
  diagnostics. The current index pattern falls back to signed extension when a
  range is absent; this suite does **not** claim it rejects missing range evidence.
  The operand diagnostic names the missing graph node; the memory-type diagnostic
  names the missing SSA carrier. They are checked for these specific reasons.
- Alex's serializer produces complete functions from those pattern results. The
  real MLIR verifier and standard lowering preserve signed/unsigned extension at
  32- and 64-bit index widths.
- Mutable function bindings use a separate cell containing the existing closure
  carrier. Binding, reference and assignment witnesses preserve a loaded immutable
  snapshot across reassignment. Missing initializer/cell values require their
  specific diagnostics. The composed output verifies and lowers with real MLIR at
  32- and 64-bit index widths.
- The unit-result pattern preserves ordered conditional effects and appends the
  canonical i32 zero value. It preserves the body's missing-operand diagnostic
  and rejects an already returned value. Its composed conditional output verifies
  and lowers through standard MLIR without changing graph facts or operand recall.
- The index-switch pattern preserves supplied case labels, arm effects and ordered
  result values, with an explicit default region. It rejects incompatible selector
  and arm types, duplicate labels and already terminated arms. Zero, one and multiple
  results verify and lower through stock MLIR; declarations inside its arms still
  reach the existing module collection pass. These are dialect component checks,
  not continuation segmentation or sequence execution tests.
- Continuation patterns read supplied slot placement, descriptor identity and
  selector ranges. Scalar and captured-cell access verifies and lowers through
  stock MLIR; missing placement and mismatched carriers fail explicitly. Fresh
  enumerators allocate distinct state, copy only capture fields, and require
  explicit residence and resident layout obligations. These component fixtures
  do not establish the upstream liveness, lifetime or proof-discharge premises.
  Caller-owned factory storage is allocated without initialization, then its
  exact descriptor is initialized by the constructor. Empty activation storage
  verifies at extent zero and rejects every attempted slot access.
  Owned child regions use the supplied parent formal and distinct settled offsets
  without allocating child storage; their typed views verify and lower through
  stock MLIR. Inconsistent extents, alignment and containment fail explicitly.

Continuation descriptor fields retain physical address, offset, extent and stride
data. They contain no runtime type object, type tag or boxed scalar payload.
The current rank-one carrier's five-word cost is a target representation choice
settled upstream; it is not a Clef type-size rule. These tests claim no elimination
of statically known descriptor fields. Literal frame offsets and extents come from
the supplied graph, in accord with the closure and continuation representation
contracts; MLIR verification establishes neither target layout admission nor
captured-storage lifetime.

Fixtures supply already settled graph facts and previously witnessed operands.
The current parser API requires `MLIRAccumulator` for operand recall; this suite
uses it as the existing input table. The index pattern leaves that table unchanged;
the mutable binding pattern registers its allocation's physical type through the
existing Element API. Every observed witness preserves the graph, codata and edges.
The harness neither introduces nor reproduces a traversal driver, semantic
accumulator or recursive subtree emitter. Elements remain internal; tests reach
them through public patterns.

These are component contracts, not source-language, array-bounds, proof-discharge,
native-execution or complete dialect-support claims. In particular, the signed
index fixture is verified and lowered, never executed as an array access. CCS
tests establish the source and Baker contracts; FidelityHello and NativeCallbacks
provide the separate source-to-native behavioral oracles. Additional dialects
require their own admitted settled graph forms and associated tests.

The fixture's obligation edge is checked for preservation, not discharge or
admission. The current array pattern does not itself check proof state, and these
tests introduce no such semantic responsibility in Alex. Upstream missing-range
and obligation-rejection coverage must exercise the owning CCS/Baker pass and the
full pipeline; an MLIR verifier accepting this component cannot establish either.
