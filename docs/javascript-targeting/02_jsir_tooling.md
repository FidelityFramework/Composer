# JSIR Tooling: What Composer Actually Invokes

**SpeakEZ Technologies | Fidelity Framework**
**April 2026**

This document fixes a set of misconceptions that accumulated in early drafts of Composer's backend design. The misconceptions traced to reading Google's JSIR RFC before the repository was cloned and examined. The corrected picture below is what Composer's backend actually targets.

## The Single Binary

JSIR ships one executable: `jsir_gen`. It lives at `maldoca/js/ir/jsir_gen.cc` in the upstream repository. There is no separate `jsir-lift` tool. Early Composer drafts referenced `jsir-lift` as though it were distinct; it isn't. Everything JSIR does — parsing source, generating MLIR, lowering MLIR back to source — flows through `jsir_gen` with different pass arguments.

```
jsir_gen --input=<file> --passes=<pass-list> --output=<file>
```

## The Four Named Conversions

`jsir_gen`'s `--passes` argument accepts a comma-separated list of conversion names. Four named passes span the representations JSIR deals with:

| Pass | Input | Output | Purpose |
|:-----|:------|:-------|:--------|
| `source2ast` | JavaScript source | Babel AST JSON | Uses Babel to parse JavaScript |
| `ast2hir` | Babel AST JSON | JSHIR (region-based MLIR) | Lifts Babel AST into the high-level MLIR dialect with regions |
| `hir2ast` | JSHIR | Babel AST JSON | Lowers JSHIR back into Babel AST |
| `ast2source` | Babel AST JSON | JavaScript source | Uses Babel's printer to produce JavaScript |

The representations are:

1. **JavaScript source** — `.js` text.
2. **Babel AST** — JSON representation of a Babel AST (Google explicitly uses Babel, not ESTree; the C++ code uses `BabelAstString`, `BabelParseRequest`, and lives under `maldoca/js/babel/`).
3. **JSHIR** — the high-level region-based JSIR dialect. Control flow uses MLIR regions. Ops like `jshir.if_statement`, `jshir.while_statement`, `jshir.switch_statement`.
4. **JSIR** — a separate lower-level dialect that JSIR can lower JSHIR into. Most ops live in `jsir.*`; region-based structural ops live in `jshir.*`. Composer primarily emits and consumes the JSHIR level for backend purposes because it's the representation that round-trips cleanly to source.

There is no third "low-level" dialect beyond these. The JSIR repository has exactly two MLIR dialects registered: `Jsir_Dialect` (namespace `jsir`) and `Jshir_Dialect` (namespace `jshir`).

## Forward vs. Reverse Pipelines

**Forward pipeline (source to MLIR):**

```
jsir_gen --passes=source2ast,ast2hir --input=file.js --output=file.mlir
```

This is what Google uses internally for analysis: decompiling Hermes bytecode, running dataflow analysis for deobfuscation, classifying malicious JavaScript. Composer does not need the forward pipeline for production compilation; it uses it only during development to validate that hand-emitted JSIR ops round-trip correctly.

**Reverse pipeline (MLIR to source):**

```
jsir_gen --passes=hir2ast,ast2source --input=module.mlir --output=module.js
```

This is what Composer invokes in its backend. Alex's witnesses emit JSHIR ops into an MLIR module. Composer writes that module to a temporary `.mlir` file, runs `jsir_gen` against it with the `hir2ast,ast2source` pass list, and receives JavaScript source.

## Babel, Not ESTree

An early draft of `JSIR_Backend_Design.md` described JSIR as using "ESTree" as the AST substrate. This was wrong. Inspection of the upstream repository confirms:

- `maldoca/js/babel/babel.h` — the Babel integration layer
- `maldoca/js/babel/babel.pb.h` — protobuf definitions for Babel AST
- `maldoca/js/ast/ast_util.h` — comment: "Parses the source using Babel and returns a string representing the AST"
- `BabelAstString`, `BabelParseRequest`, `BabelGenerateOptions` — the types the C++ code uses throughout
- `third_party/babel_standalone/` — the vendored Babel runtime

The distinction matters because Babel and ESTree differ in places that affect emission:

| Construct | ESTree | Babel |
|:----------|:-------|:------|
| Class methods | `MethodDefinition` | `ClassMethod` (and `ClassPrivateMethod`, `ClassProperty`) |
| Object methods | `Property` with `method: true` | `ObjectMethod` as a distinct node kind |
| Numeric literal | `Literal` | `NumericLiteral` |
| String literal | `Literal` | `StringLiteral` |
| Boolean literal | `Literal` | `BooleanLiteral` |
| Null literal | `Literal` | `NullLiteral` |

JSIR's op set mirrors Babel. Ops like `jsir.class_method`, `jsir.object_method`, `jsir.numeric_literal`, `jsir.string_literal`, `jsir.boolean_literal`, `jsir.null_literal` exist; `MethodDefinition`-style ops do not. When writing witnesses, the reference is Babel's node kinds, not ESTree's.

## What's in the Dialect

The JSIR dialect contains roughly 85 operations across `jsir.*` and `jshir.*`. A representative partial list:

**Literals and identifiers:**
- `jsir.numeric_literal`, `jsir.string_literal`, `jsir.boolean_literal`, `jsir.null_literal`, `jsir.big_int_literal`, `jsir.reg_exp_literal`
- `jsir.identifier` (r-value), `jsir.identifier_ref` (l-value), `jsir.private_name`

**Expressions:**
- `jsir.binary_expression`, `jsir.unary_expression`, `jsir.update_expression`, `jsir.assignment_expression`
- `jsir.call_expression`, `jsir.optional_call_expression`, `jsir.new_expression`
- `jsir.member_expression`, `jsir.optional_member_expression`, `jsir.member_expression_ref`
- `jsir.array_expression`, `jsir.object_expression`, `jsir.sequence_expression`
- `jsir.template_literal`, `jsir.tagged_template_expression`
- `jsir.arrow_function_expression`, `jsir.function_expression`, `jsir.class_expression`
- `jsir.this_expression`, `jsir.super`, `jsir.import`, `jsir.yield_expression`, `jsir.await_expression`
- `jsir.spread_element`, `jsir.meta_property`, `jsir.parenthesized_expression`

**Statements:**
- `jsir.expression_statement`, `jsir.empty_statement`, `jsir.debugger_statement`, `jsir.throw_statement`, `jsir.return_statement`
- `jsir.variable_declaration`, `jsir.variable_declarator`
- `jsir.function_declaration`, `jsir.class_declaration`, `jsir.class_body`, `jsir.class_method`, `jsir.class_private_method`, `jsir.class_property`, `jsir.class_private_property`
- `jsir.import_declaration`, `jsir.export_named_declaration`, `jsir.export_all_declaration`, `jsir.export_default_declaration`

**Control flow (JSHIR):**
- `jshir.block_statement`, `jshir.with_statement`, `jshir.labeled_statement`
- `jshir.if_statement`, `jshir.switch_statement`, `jshir.switch_case`
- `jshir.while_statement`, `jshir.do_while_statement`, `jshir.for_statement`, `jshir.for_in_statement`, `jshir.for_of_statement`
- `jshir.try_statement`, `jshir.catch_clause`
- `jshir.break_statement`, `jshir.continue_statement`
- `jshir.logical_expression`, `jshir.conditional_expression`

**Object and pattern refs (for destructuring and l-value contexts):**
- `jsir.object_property`, `jsir.object_property_ref`, `jsir.object_method`
- `jsir.object_pattern_ref`, `jsir.array_pattern_ref`, `jsir.assignment_pattern_ref`, `jsir.rest_element_ref`

**Region terminators:**
- `jsir.expr_region_end`, `jsir.exprs_region_end`

This set covers essentially all Babel-AST-representable JavaScript. It does not cover JSX: JSIR has no `jsx_element`, `jsx_attribute`, or similar operations. JSX is a source-level surface that every JSX-based framework lowers before the JavaScript reaches JSIR.

## What Composer Invokes

Composer's backend, at build time:

1. Alex emits JSHIR ops via its JSIR witness set into an in-memory MLIR module.
2. The module is written to `build/<project>/<unit>.mlir`.
3. Composer shells out to `jsir_gen --passes=hir2ast,ast2source --input=<unit>.mlir --output=<unit>.js`.
4. The resulting `.js` file is the compilation artifact, wrapped in a `JavaScriptModule` back-end artifact record.

No other JSIR tooling is invoked. JSIR's analysis passes (constant propagation, dataflow analyses, deobfuscation transforms) are not part of Composer's backend. They exist in the JSIR repository for Google's internal use cases and could be invoked for JavaScript analysis tasks if Composer ever needed them (e.g., Transcribe lifting foreign JavaScript into Clef — Horizon 3), but they have no role in the emission path.

## Build Integration

JSIR builds with Bazel against `@llvm-project`. Composer's build currently uses `bazel` as well for the native MLIR pipeline, so integration is straightforward: the JSIR dialect's TableGen definitions (`maldoca/js/ir/jsir_dialect.td`, `jsir_ops.td`, `jsir_ops.generated.td`, `jshir_ops.*`, `jsir_attrs.td`, `jsir_types.td`, `interfaces.td`) compile against Composer's MLIR version, and the `jsir_gen` binary is produced by the same build system.

The dialect will be upstreamed into MLIR core once Google's RFC (published April 6, 2026) completes review. Until then, Composer depends on JSIR as an external dialect. Post-upstream, JSIR becomes an MLIR core component like EmitC and the dependency shortens.

## Cross-References

- [JSIR_Backend_Design.md](../JSIR_Backend_Design.md) — the backend specification that invokes this tooling
- [01_two_models.md](./01_two_models.md) — when Composer uses this pipeline vs. when Fable is in charge
- [Fidelity.CloudEdge/docs/10_jsir_strategic_assessment.md](../../../Fidelity.CloudEdge/docs/10_jsir_strategic_assessment.md) — the ecosystem-level framing and RFC context
- Upstream: https://github.com/google/jsir
- RFC: https://discourse.llvm.org/t/rfc-jsir-a-high-level-ir-for-javascript/90456
