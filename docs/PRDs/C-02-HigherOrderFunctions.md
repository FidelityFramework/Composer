# C-02: Higher-Order Functions

> **Status:** In-Progress. HOFs, callback recipes and native oracles are implemented;
> general stored/returned callable composition remains open.
> **Sample:** `12_HigherOrderFunctions`, with native callback and C-07 application variants.
> **Dependencies:** [C-01](C-01-Closures.md), source type/application contracts and
> the storage contracts of retained values.
> **Criteria realignment:** 2026-09-25; documentation only, with no new runtime results.

## 1. Executive Summary

C-02 completes functions as arguments, results and stored values, including
composition and partial application. Existing HOF and Option/Result/Seq native
callback paths supply the starting point. Remaining work makes them compose
across callable origins, environment lifetimes and application forms.

The [closure specification](../../../clef-lang-spec/spec/closure-representation.md),
[application checking](../../../clef-lang-spec/spec/inference-application-resolution.md)
and each operation's chapter govern semantics. Use the
[shared C-series criteria](C-Series-Acceptance.md) and
[coverage waypoints](../Language_Coverage_Waypoints.md) for completion and evidence.
The former claim that HOF support is largely free once closures exist is retired:
staging, representation, effects and residence are observable obligations.

## 2. Language Feature Specification

### 2.1 HOF Patterns

| Use | Required behavior |
|---|---|
| Function parameter | Invoke the supplied value with its actual environment and checked argument/result types |
| Function result | Retain implementation/environment with residence covering subsequent uses |
| Composition | Preserve callback order, independent environments and intermediate types/dimensions |
| Conditional/matched selection | Select the whole function value; equal layouts do not imply equal implementations |
| Stored alias | Retain the shared deferred computation; first demand establishes the selected value, and later demands share it. Explicit `eager` establishes a snapshot at its activated frontier |
| Partial application | Retain supplied computations and their identities until demand; direct eager actuals establish ordered shallow demand at the activated application frontier |
| Bare operation value | Admit its full function type and subsequent stages through the ordinary graph contract |

Direct syntax and forward/backward pipes preserve demand and required effect
order. Producer formation retains ordinary supplied computations without forcing
their effects. Iteration and callback invocation occur at their specified
activation boundaries. Aliasing a partial cannot replay a successfully evaluated
shared operand initializer. The [evaluation contract](../Evaluation_Strategy_Contract.md)
governs the distinction between ordinary supply and explicit eager demand.

### 2.2 Type Representation

Source function types remain `TFun` with public generic/NTU signatures. Independent
argument, intermediate, accumulator and result dimensions survive specialization,
capture, storage and elimination. Physical carriers do not replace those types.

A materialized callable is `(fn, env)`, separate middle-end values. Known-callee
elision and direct capture passing require C-01's complete graph premises. A
function type or code identity alone does not establish a storage field,
environment instance or lifetime.

### 2.3 Calling Convention Unification

One truthful application contract connects callee, actual environment, hidden
formals where applicable, explicit arguments and results. Baker constructs it
before Alex. Pair parameters/returns, joins and aggregate/DU elimination agree
with that contract. No interior numeric code-pointer packing, implicit null
environment or cast-resolution convention is introduced.

Foreign `FnPtr<'F>` entries retain the distinct
[C-01 boundary contract](C-01-Closures.md#6-ffi-boundary-marshaling). Ordinary HOF
support does not automatically transport specialized foreign ABI provenance.

## 3. CCS Layer Implementation

CCS owns typing, generic instantiation, source identity and argument admission.
Callable application and operation recipes establish stages at actual declared
boundaries. They preserve original operand identities through Baker fan-out/fold-in;
tracing an alias to a source expression does not authorize forcing it, replaying
it, or moving its effects ahead of its demand frontier.

Known-callee environment admission supports bounded scalar captures and complete
direct uses in sequence producers. Stored/bare Seq operations and broader
returned, opaque, aggregate or nested callable captures retain open contracts.
[Closure values as data](../Closure_As_Data.md) and [C-07](C-07-SeqOperations.md)
identify the boundaries; later waypoints supersede interim native-status prose.

Front-end/Baker work is more than checking an existing `TFun`: callable origins,
stage frontiers, shared demand, environments, types, effects and residence must survive
each transformation.

## 4. Composer/Alex Layer Implementation

The bounded environment path supplies explicit formals and actual operands to
ordinary lambda/application witnesses. Alex pulls at the actual Huet position;
it does not reconstruct partial applications, discover captures or select source
algorithms by library spelling.

General pair support still needs multi-value recall/signatures, call/return
results, branch joins, stored values and target lowering. The single-result model
and legacy packed closure-call path are migration work, not source restrictions
or evidence of canonical general HOF acceptance.

## 5. MLIR Output Specification

Direct calls use graph-established operands. Unknown values use a function-typed
value and actual environment through `func.call_indirect`. Both halves survive
parameters, returns, joins and eliminators whenever elision is unavailable.
Typed views access placed slots. Historical direct `llvm.*` listings and raw-pointer
closure structs are retired implementation sketches.

[M-01](M-01-DialectAdmission.md) governs realization and preserved information.
Require verification, backend lowering and native behavior; verifier success
alone does not establish timing, identity or storage lifetime. The
[FPGA/Colibri](../fpga-targeting/README.md) and
[eBPF](../ebpf-targeting/README.md) plans inform these criteria: transformed circuits
or bytecode must retain the settled callable/effect/storage contracts with their
own artifact checks. An existing host pass does not certify those target paths.

## 6. Validation

### 6.1 Sample Code

Retain original12's `applyTwice`, pair mapping, conditional selection, factories
and both composition orders. Native callback and lettered FidelityHello variants
extend its scope; they do not replace it.

The unchanged [16h oracle](../../samples/console/FidelityHelloWorld/16h_SequenceApplications/README.md)
is the next staging gate: direct/pipe/stored/bare forms, both fold frontiers,
independent dimensions, immutable aliases, formation effects, fresh enumeration
and stopping. Original16 remains the factory/capture composition gate.

### 6.2 Expected Output

Keep original12's exact [manifest output](../../tests/regression/Manifest.toml)
and 16h/original16 expected outputs. New cases check ordered effects and actual
environment identity as well as return values.

| Completion row | Positive cases | Rejection/preservation cases |
|---|---|---|
| Parameter/result transport | Named/literal/captured callbacks, returned functions, repeated invocation | Wrong dimensions or missing environment facts produce the responsible located diagnostic |
| Stages | Direct/both pipe directions, bare aliases, multiple partial frontiers, ordinary and explicit eager actuals | Unused ordinary effects stay deferred; explicit eager order and shared successful results survive staging |
| Selection/storage | Different implementations/equal layouts; same code/distinct environments; deferred and explicit eager reads of mutable selection | First demand establishes the selection unless explicit eager demand occurred earlier; later demands share that value; joins retain both halves |
| Composition | Captured HOFs, nested functions, generic/measure instantiations | Independent parameters, captures and intermediate dimensions remain distinct |
| Deferred operations | Eleven C-07 core operations in admitted forms, including children and stopping | Complete-use/residence failures stay explicit; no post-decision pull/callback |
| Aggregates | Functions through records, tuples, options/results and collections with valid storage | Elimination preserves code identity/lifetime; a descriptor alone cannot admit arbitrary `TFun` storage |
| Tooling/evidence | Public types, capture navigation, precise errors and unsaved repair | Hidden parameters do not replace source signatures; stale/altered participants and evidence are refused |

## 7. Files to Create/Modify

| Owner | Existing implementation to extend |
|---|---|
| Source typing | CCS `NativeTypedTree/Expressions/Applications.fs`, binding/generalization and scheme owners |
| Staging/demand | `Nanopass/CallableApplications.fs`, `Baker/Recipes/ApplicationRecipes.fs`, owning operation recipes |
| Captures/residence | C-01 construction/settlement; C-04/C-06 storage owners |
| Physical expression | Alex `Traversal/Values.fs`, `Dialects/Core/Types.fs`, lambda/application/environment witnesses/patterns |
| Acceptance | NativeCallbacks, original12/16/16h, source/graph negatives, Alex/proof and shared editor/client gates |

These are extension owners, not a second application lowerer or per-operation emitter.

## 8. Implementation Checklist

The [C-06 checkpoint](../Language_Coverage_Waypoints.md#c-06-native-continuation-settlement--2026-09-20)
records original12 in its passing 23/28 compilation cohort. Option/Result and
[C-07](../Language_Coverage_Waypoints.md#c-07-sequence-operations--implementation-waypoint-acceptance-open-2026-09-20)
native fixtures establish their specific callback paths; original16 and 16h remain
unpassed. This criteria update reruns none of those results.

- [ ] Complete staged stored/bare Seq applications without replaying supplied effects.
- [ ] Establish general transport and actual-environment identity through C-01's contract.
- [ ] Admit returned/retained/aggregate storage under owning lifetime/placement contracts.
- [ ] Pass §6 through source, graph, Alex, backend/native and exact negative gates.
- [ ] Pass editor/analyzer/live-LSP projection and repair for newly admitted forms.
- [ ] Record final regression controls, artifacts, companion revisions and profile limits under the shared criteria.

Unchecked rows are completion gates; they do not revoke existing support or imply
function typing and callback execution are absent.
Accurately refusing a promised conforming use does not close its positive gate.
C-02 retains the same delivery obligation as the F-series; dependency order does
not make these missing compositions optional.

## 9. Risk Assessment

The principal risk is losing a premise while composing working mechanisms:
operand order, complete-use classification, signature truth, environment occurrence,
effect invalidation or residence. C-01 pair/direct mutable-cell work and C-04
storage contracts dominate that risk.

Proceed in bounded source-to-artifact increments with fixed observations. A
literal callback, source scheme, component fixture or hover does not close its
retained callable counterpart. Foreign registration, Async/actors and descriptor
transfer retain their applicable gates; their constraints inform current design.

## 10. Related PRDs

- [C-01](C-01-Closures.md): captures, representation, residence and foreign distinctions.
- [C-03](C-03-Recursion.md): recursive values of function type and capture/effect forwarding.
- [C-04](C-04-CoreCollections.md): HOF algorithms and persistent storage.
- [C-05](C-05-Lazy.md), [C-06](C-06-SimpleSeq.md), [C-07](C-07-SeqOperations.md): deferred application, lifetime and demand.
- [M-01](M-01-DialectAdmission.md): admitted forms and preserved information.
