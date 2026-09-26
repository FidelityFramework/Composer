# Async continuation settlement and backend coroutines

Clef async semantics follow the source-owned continuation and evaluation
contracts. CCS/Baker constructs and settles those semantics in the PSG; Alex
passively witnesses the immutable publication. LLVM coroutine operations are a
possible target realization governed by Composer's backend admission contract.

## Source-owned async construction

CCS checks async expressions and their native types. Baker's owning ingredients
and recipes establish:

- Deferred body identity, capture timing and activation boundaries.
- Bind, return, suspension, resumption and completion relationships.
- Live values, definite initialization, frame membership and storage lifetime.
- Callable declarations, physical signatures, cleanup and result publication.
- Complete joint premises, scope and rewrite correspondence.

The absence of suspension points does not authorize execution at formation.
Unresolved facts prevent the commitment that needs them. Alex does not recognize
async builders, insert suspension points, choose allocation residence or
reconstruct a state machine from a source body.

## Passive witnessing and target realization

Alex observes source-settled operations through the Huet zipper and composes
Elements through Patterns and Witnesses. It reads node-local codata, without
querying the joint constraint set or running source analysis.

Composer's backend may realize admitted continuation operations using LLVM
coroutines. The backend owns target intrinsics, attributes, splitting and physical
artifact production. Any introduced frame or layout transformation must preserve
the source-owned storage, lifetime and call contracts and the required
source-to-artifact correspondence.

A coroutine intrinsic does not establish a scheduling, memory-allocation or
runtime-dependency policy. Those obligations must be settled under the declared
target profile and verified for the actual artifact. All custom MLIR plugins and
their compatibility dependencies are retired.

## Acceptance

Source and graph checks establish demand, capture, suspension and lifetime
semantics. Witness checks establish faithful composition of the published facts.
Backend and native checks establish realization, completion, ordering and resource
behavior for the admitted target. A focused passing check is not complete async
or F/C acceptance.

Keep exact failed cases and observations. Repair missing premises at the source
owner rather than adding emitter fallbacks or weakening the oracle.

## Related contracts

- [Evaluation strategy](Evaluation_Strategy_Contract.md)
- [Delimited continuations](Delimited_Continuations_Architecture.md)
- [A-01: Basic async](PRDs/A-01-BasicAsync.md)
- [A-02: Async await](PRDs/A-02-AsyncAwait.md)
- [A-03: Async parallel](PRDs/A-03-AsyncParallel.md)
- [M-01: Dialect admission and backend transport](PRDs/M-01-DialectAdmission.md)
