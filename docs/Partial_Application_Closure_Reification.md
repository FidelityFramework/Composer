# Partial application closure reification

The [closure settlement contract](Closure_Settlement_Contract.md),
[C-01](PRDs/C-01-Closures.md), [C-02](PRDs/C-02-HigherOrderFunctions.md) and
[application semantics](../../clef-lang-spec/spec/expressions.md#evaluating-function-applications)
govern partial application. Baker owns semantic reification. Alex passively
witnesses the settled result.

## Required behavior

A partially applied function retains the actual callable and the shared deferred
identities of its supplied arguments. Direct explicit eager arguments are
demanded at their specified activated application frontier. Later application
retains those results and identities; it cannot replay an initializer.

The callable's declared argument boundary determines partial application. A
function-valued result introduces a later boundary; flattening every arrow in a
source type is not evidence of one native call signature.

Local, returned, stored, aliased, aggregate-held and higher-order partial values
retain the same semantics. Separate runtime formations remain separate instances
even when they share implementation code and layout.

## Source-owned construction

CCS establishes source types, lexical identity and declared application
boundaries. Baker ingredients and recipes construct any required residual
callable, forwarding body, captures and invocation structure in the PSG. Fan-out
and fold-in preserve source correspondence and every joint proof participant.

Owning nanopasses settle complete callable alternatives, supplied/formal
correspondence, effects, demand, physical signatures, typed capture layout and
covering storage lifetime. A retained descriptor does not extend its backing
storage. Unknown alternatives or incomplete use information remain explicit
premises to resolve before the dependent commitment.

Materialized callables retain separate function and environment operands. A
proved direct form may elide an unnecessary carrier while preserving actual
environment identity and the same demand and lifetime obligations.

Alex consumes the published residual function, capture accesses, storage and call
facts at their actual Huet occurrences. Elements/Patterns/Witnesses compose their
physical form. Missing semantic construction is an owning CCS/Baker failure;
there is no thunk-generation alternative in Alex or in a middle-end MLIR pass.

## Acceptance

The C-01/C-02 gates cover every declared partial frontier, returned functions,
stored and forwarded residual values, mutable shared cells, distinct formations,
and independent dimensional arguments/results. Demand traces distinguish unused
ordinary arguments, first and repeated demand, and explicit eager frontiers.

Source edits that change a supplied argument, callable alternative, capture,
consumer or lifetime premise retract dependent facts. Source, graph, witness,
artifact, native and tooling results retain their exact scope and cohort under
the [C-series acceptance contract](PRDs/C-Series-Acceptance.md).
