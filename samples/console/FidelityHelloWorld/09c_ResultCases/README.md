# Result case predicates

Status: native compilation, exact-output and zero-exit gates pass.

This variant exercises `Result.isOk` and `Result.isError` under specification
`5f49a02`'s [native Result contracts](../../../../../clef-lang-spec/spec/error-handling.md#native-result-operations).
Each predicate evaluates its Result input once and observes its case tag. Payload
construction remains eager; callable payloads remain uninvoked. Every Result
value fixes both payload types, including independent dimensions and unit.

The CompilerSurface project prints fixed Console text. Five groups have failure
codes 1–5. The companion `tests/NativeCallbacks/ResultCases.clef` adds record
payloads, both directions for each predicate, and lexical `Result` shadowing
across eight groups (141–148). Its runner requires `arith.cmpi`,
`func.call_indirect`, stock MLIR verification and native exit zero. The oracles
establish observable behavior; they do not prescribe layout or discharge
closure lifetime obligations.

Successful execution must exit zero and print exactly:

```text
=== Result cases Test ===
Case tags and unit payloads: pass
Factory and pipe order: pass
Stored predicates and measured payloads: pass
Callable payloads remain uninvoked: pass
Short-circuit composition: pass
```
