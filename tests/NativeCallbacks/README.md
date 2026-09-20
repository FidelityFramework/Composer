# Native callback and capture gates

Run `dotnet run --project tests/NativeCallbacks/NativeCallbacks.Tests.fsproj` from
the Composer checkout. Each case compiles a fresh LLVM/LLD executable and runs
it with a timeout. The gates cover named entries, bounded array descriptors,
mutable cells, records, captured function values, alias snapshots and function
fields followed by later allocations. `IgnoreValues` checks that discarding
ordinary and optional opaque-handle values preserves evaluation effects and
produces a usable Clef unit value.

To select cases, pass the compiler executable followed by exact case names:

```sh
dotnet run --project tests/NativeCallbacks/NativeCallbacks.Tests.fsproj -- src/bin/Debug/net10.0/Composer OptionDefaults OptionPartials
```

Unknown names fail before compilation. With no names, the runner selects all
cases. A selected passing subset does not establish the remaining cases. Each
retained MLIR module must also pass `mlir-opt --verify-each`; missing tooling is
a failed gate. The evidence records the executable and compiler assembly hashes.

`DirectCaptures` checks Baker's immutable direct capture form: repeated calls,
recursive forwarding, shadowed bindings, nested declarations, returned anonymous
closures, array descriptors, records, captured function values and inverse
dimensions. Two explicit effectful arguments retain their source order. Separate
local functions deliberately share names, requiring definition identity to survive
target symbol assignment. Exit codes 221–230 identify the ten groups. Mutable
capture frontiers and named functions used as values remain outside this recipe's
admission; this gate does not establish complete closure residence obligations
or retire the interim representation used by returned function values.

`CallEffects` checks range preservation after local, higher-order, transitive and
recursive writes, including negative values, loop iterations and ordered argument
evaluation. Earlier value snapshots survive later writes. Saved Boolean predicates
retain their historical observations and cannot constrain the current value of
mutable storage. Effectful guard operands and predicate calls cannot turn an old
observation into a bound on the new value. Exit codes 1–13 distinguish the groups.

`OptionAlternatives` checks `orElse` and `orElseWith`, which retain the optional
result. Its 25 groups (exit codes 231–255) cover both branches, empty alternatives,
eager operand formation, deferred invocation, both pipe directions, partial
snapshots, shared mutable captures, independently specialized bare aliases,
measured and inverse-dimensional values, nested options, records and function
payloads. Selection preserves callable identity without invoking the payload;
ordered effects distinguish selection from later application.

`OptionDefaults` checks eager fallback evaluation for Some and None, direct and
piped evaluation order, partial-formation snapshots, independent measured
specialization, nested options, record payloads and stored function fields.
Function selection does not invoke the payload; subsequent invocation observes
the selected closure's captures. Extra source arguments apply that selected
function after the operation consumes its fallback and option. Exit codes
181–191 distinguish the eleven groups. The harness checks retained `scf.if` and
`func.call_indirect` operations as well as the fresh executable's exit status.

`OptionDefaultWith` checks eager thunk construction and deferred invocation:
Some skips the thunk, None invokes it once per call, and stored partials retain
the original thunk value while sharing its mutable captures. It covers forward
and backward pipes, independently specialized measured values, nested options,
records, stored function fields, unit-returning effects and measured reals.
Function payloads and staged overapplications check that all supplied operands
evaluate before invocation, with ordered traces across each returned function.
Exit codes 201–218 identify its eighteen groups. The retained MLIR must contain
`scf.if` and `func.call_indirect`; native execution establishes the branch effects
and evaluation order. This gate does not establish complete closure proof
discharge or eliminate the interim closure representation.

`OptionCallbacks` is a language acceptance gate for `Option.map`, `bind`,
`filter`, `exists`, and `forall`. Each operation checks None without invoking
its callback and Some with exactly one callback invocation, recording the count
after each call. It covers true and false predicates, retained filter payloads,
`Some false` from map, None returned by bind, and vacuous truth for `forall None`.
Callbacks capture a record containing a boolean and numeric fields. A generic
mapping helper specializes to different input/output types; measured values
exercise type-changing map/bind and dimensional preservation. Exit codes
101–106 identify map, bind, filter, exists, forall, and generic/dimensional
failures respectively. The harness also requires portable conditional and
indirect-call operations in the retained MLIR.

`OptionEvaluation` separates eager argument evaluation from callback invocation.
For each of the five HOFs, direct calls and backward pipes evaluate an effectful
callback factory before an effectful option-producing expression; forward pipes
evaluate the option expression first. Both expressions run exactly once. None
skips the callback; Some invokes it once after both argument expressions. Ordered
event traces detect omitted, duplicate, or reordered evaluation. Nested
`Option.filter` over `Option.map`, plus chained forward and backward pipes,
check both factories, the input, and conditional callbacks in one expression.
Exit codes 111–116 identify map failures, 117–122 bind, 123–128 filter,
129–134 exists, and 135–140 forall; within each group the order is direct
None/Some, forward None/Some, backward None/Some. Codes 141–146 cover nested
filter/map, forward chains, and backward chains, each None/Some. Native execution
remains a required acceptance gate.

`OptionPartials` checks stored partial applications of all five HOFs, callback
construction once, reuse across Some/None, higher-order transport, immutable
callback snapshots, and sharing of mutable captures. Returned generic partials
preserve distinct and measured payload types. All eight admitted operations
(`map`, `bind`, `filter`, `exists`, `forall`, `isSome`, `isNone`, `get`) are also
used as bare function values. Unannotated polymorphic aliases and explicit type
applications exercise specialization before Baker. Exit codes 151–167 identify
the individual acceptance groups in source order.

`OptionFunctionPayloads` checks functions inside options: map produces and
consumes captured functions, bind returns Some/None functions, filter retains
the original callable, and exists/forall invoke predicates over callable
payloads. Captured values survive callback returns; shared mutable captures
remain shared. None skips both the predicate and payload invocation. Direct and
explicitly typed `Option.get` calls can immediately invoke their function payload,
including a payload that returns another closure; ordered effects check eager
argument evaluation. Exit codes 171–177 identify the seven groups. `Option.get`
is tested on Some values, consistent with its specified unchecked extraction contract.

`GenericRecords` exercises distinct concrete layouts of the same generic record,
including numeric and record payloads, nested options, copy updates, and returned
unit closures that retain each concrete record. It checks phantom parameters,
field order differing from declaration parameter order, repeated parameters,
and measured fields. Numeric values exceed an eight-bit range so premature
narrowing is observable. Exit codes 81–89 identify these cases in source order.

`ListenerEntry` consumes the same `CallbackDescriptor` vocabulary emitted by
Farscape for native listener fields. It checks an ordinary Clef unit call, a
separate C `void` entry, field aliases and another address of the same handler,
the full unsigned 32-bit argument range, and negative signed 32-bit results.
The source retains its logical unit result; the native entry thunk discards it.
Declared scalar representations also govern indirect-call arguments and results.

The focused listener gate passed as a fresh native executable on 2026-09-09
using Composer Debug build 28. Retained local evidence is
`/tmp/clef-listener-entry-tvbgj__e`: `build.log`, `run.log`, the executable and
`targets/intermediates/10_output.mlir`. These temporary artifacts may expire.

`IgnoreValues` passed on Debug build 32 as a fresh native executable in
`/tmp/clef-ignore-values-n3yh6t9n`. Its effect counts and three unit consumers
passed; the retained MLIR contains both effect-producing calls.

Listener declaration provenance currently follows record fields, ordinary
aliases and repeated `FnPtr.ofFunction` addresses of the declared entry. An
ABI-specialized pointer transported through an unrelated, unannotated `FnPtr`
parameter does not yet carry that declaration. This gate does not establish
that higher-order transport. Native entries require named module functions
without captures; foreign context remains an explicit opaque handle. Ordinary
Clef closures retain their code/environment pairs and bounded capture views.
