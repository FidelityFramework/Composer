# Compiler Surface Gaps, September 2026

This note records the gaps in the Clef-to-native surface that the BAREWire rebuild hit
against the HelloProof snapshot (`v0.0.2+9a1a60b1`), what each one turned out to be at the
source, what was changed, and which regression sample guards it. The rule ids are those of
BAREWire's `docs/12 Intersection Subset.md`; each fixed gap lets its rule retire and its
`SUBSET(rule)` sites migrate back to the preferred spelling.

Every repro below is a reachable program compiled with `Composer compile <fidproj>` and run;
"snapshot" is the pinned HelloProof binary, "rebuilt" is `src/bin/Debug/net10.0/Composer`
built from this tree together with the sibling `clef` tree. The acceptance programs are
BAREWire's `samples/RoundTrip/RoundTrip.fidproj` (every shared tier, its stdout diffed against
`samples/RoundTrip/expected.txt`, which the .NET build produces) and the Encoding/Framing gate
(`Encoder`, `Decoder`, `Codec`, `Envelope` with a probe `Main.clef`, compared with a .NET run of
the same operations). With the rebuilt compiler both pass: RoundTrip's transcript is identical
to the .NET one, and the Encoding gate agrees with .NET on every line except `strlen`, where
`String.length "héllo"` is 6 natively (UTF-8 bytes, the Clef string model) and 5 on .NET
(UTF-16 units); that is a substrate difference, not a compiler defect, and BAREWire's own
`Text` shim is the place that reconciles it.

Status at the end of the round: every repro of the four library lanes (103 programs: the
Schema lane's `bisect/*`, the Hardware lane's `repro*`, the Platform lane's `p*`/`probe*`, the
shared `repros/*`, and samples 18-23) compiles and prints the expected values with the rebuilt
compiler; on the snapshot 74 of them do not compile and 9 more compile and print wrong numbers
(`Array.length` as 0, a string payload's length as 2^56). HelloProof is unchanged: the same 19
obligation ids, 19 `unsat`, `mlir-opt` clean, and `Hello, Houston!`.

## Two defects that hid every other one

Before the gaps themselves, two changes in the middle end explain why the snapshot's failure
messages pointed everywhere except at the cause.

### Unhandled nodes inside function bodies were skipped silently

`Alex/Traversal/WitnessRegistry.fs`, the Y-combinator that walks function bodies
(`lazyCombinator`), returned `WitnessOutput.skip` when no witness handled a node. The top-level
combiner (`NanopassArchitecture.combineWitnesses`) reports a coverage error in the same
situation; inside a body the node simply produced no value, and the failure surfaced two or
three nodes later as "Binding not yet witnessed" or "arguments not yet witnessed". The body
combinator now reports `No witness handled node N — Kind: ... Type: ...`.

### A variable pattern that matched every type

`Alex/Witnesses/LambdaWitness.fs` decided whether a function returns unit with
`NativeType.TApp ({ NTUKind = Some NTUunit }, [])`. `NTUKind` is `[<RequireQualifiedAccess>]`,
so the bare `NTUunit` is a *variable pattern* and the arm matched every `TApp` with a kind.
A non-unit function whose body produced no value was therefore treated as unit, and
`ClosurePatterns.pFunctionDef` fabricated `arith.constant 0` for its return value: the
program compiled and printed zeros. `Fn.len (bytes: byte array) = Array.length bytes` returned
0 on the snapshot. Both sites now read `NTUKind.NTUunit` (the other witnesses already did).

Together these two turned several genuine gaps below into silent miscompiles rather than
errors. Any sample that previously "passed" by printing a wrong number was passing through
these two holes.

## Fixed

### `generalization`: module-level generic functions were fixed at their first instantiation

Repro: `repros/generalization` (`Gen.firstOf` at `int`, `string` and a record; `applyTwice`
at `int` and `int64`) and the `Some v` with `v: 'a` case (`generic-option`).

Root cause, front end (`clef`):

- `NativeTypedTree/Expressions/Bindings.fs` (`checkBinding`, function branch, the comment
  "Generalization disabled") never generalized a function binding; every use site unified the
  definition's type variables, so the second instantiation reported `expected int, got string`
  at the use and at the definition.
- `NativeTypedTree/UnionFind.fs` `generalizeType` collected union-find *root* records while
  `applySubst` leaves an unbound variable as its *original* record; `TypeParam` has reference
  equality, so `NativeTypes.instantiate`'s substitution table could not find variables that had
  ever been unioned, and they leaked into every instantiation.
- Constraints are solved once at the end (`Expressions/Types.fs` `addConstraint` only
  accumulates), so a scheme cannot be formed at the binding without solving first.

Fix:

- `UnionFind.canonicalizeVars` rewrites unbound variables to their root record;
  `generalizeType` canonicalizes the body before collecting parameters.
- `NativeService.checkModuleDecl` (non-recursive path) solves the constraints accumulated so
  far incrementally (`solveNewConstraints`) after each top-level function binding, then
  generalizes it (`generalizeTopLevelFunction`): non-recursive, non-`inline`, non-extern,
  non-entry-point function bindings whose type still has free variables get a `TForall`
  scheme on the Binding node (`NodeBuilder.SetType`) and in the environment. `Identity`
  already instantiated `TForall` bindings freshly at every use.
- New nanopass `Nanopass/Monomorphization.fs`, run from `NativeService.buildResult` on the
  resolved nodes before reachability and Baker: for every generic binding it groups the use
  sites by recovered type arguments (`matchTypeArgs`), clones the Lambda subtree once per
  group with the arguments substituted into every node type, lambda parameter type and pattern
  type (`name__monoN`), repoints the use sites, replaces the original in its `ModuleDef`, and
  drops the original. Generic functions with no use site are dropped; a use site whose type
  cannot be matched leaves the binding untouched.

Caveat: generalization considers every free variable of a top-level function's type. A
module-level *value* with an unresolved type used by a later function (`let empty = [||]`)
would be over-generalized; such values are rejected by F#'s value restriction anyway.

Sample: `18_Generalization`.

### `option-payload-shared`: every `Some v` pattern in a program shared one payload type

Repro: the Hardware lane's `reproA2`/`reproA3` (`int64 option` and `float option` matched in
one program: `expected 'int64', got 'float'`), and, once the Schema tier's types resolved
(below), every `match o with Some v -> v` over an `int64 option` in a program that also
splices BAREWire: `expected 'Schema.SchemaType', got 'int64'`.

Root cause (`clef` `NativeTypedTree/Expressions/Patterns.fs`, the `SynArgPats.Pats` arm of the
union-case pattern): `extractDomains` unwrapped the constructor's `TForall` and used the
quantified variable itself (`optA`, created once at environment creation) as the payload type.
Every `Some v` pattern therefore bound `v` to the same type variable; the first constraint to
bind it typed all the others, and which one came first depended on constraint order, so the
symptom moved when unrelated code changed. Expression uses were already instantiated freshly
(`Types.instantiateTForall` from `Identity`).

Fix: the pattern instantiates the scheme freshly (`Types.instantiateTForall`) before taking the
constructor's domain types.

Samples: `18_Generalization` (generic option), `22_UnionPayloads` (`int64 option` and
`float option` patterns in one program).

### `literal-values`: module-level values did not lower

Repro: `repros/option-statement-match`, `repros/statement-match-and-record-mutation`
(module-level record and array values in a library module), WrenHello finding 4, and the
`FrameKind.Tell` bindings in `BAREWire.Framing`. On the snapshot even a same-module value
failed: `let k = 42` at module level emitted `%v0 = arith.constant 42` at *module scope* and
`useK` referenced `%v0` across a function boundary.

Root cause (`Composer`):

- `PSGElaboration/SSAAssignment.fs` Pass 1 (`findModuleLevelValueBindings`) assigned SSAs only
  to value bindings of the ModuleDef that contains `main`; values in any other module had no
  SSA ("has no SSA allocated" surfaced from a string literal inside them).
- `Alex/Traversal/NanopassArchitecture.fs` `runAllNanopasses` visited `ModuleInit` bindings at
  the root scope, so their ops were serialized at module level; references reused the SSA name
  from another function.

Fix: a module-level value binding (`EmissionStrategy.MainPrologue`, non-Lambda value) is a
program-lifetime slot, a one-element `memref.global` (`MemRefPatterns.pGlobalSlotInit/Load/
Store`, `TransferTypes.ModuleValues`). The entry point is walked first and its `LambdaWitness`
prologue visits every module's `ModuleInit` bindings inside `main`'s body scope; each
`VarRef` (`VarRefWitness`) and each assignment (`MutableAssignmentWitness`) to such a value
reloads or stores through the global in whatever function it occurs. SSA Pass 1 covers every
module; slot bindings get their own SSAs (no aliasing of the child), a VarRef to a slot gets 3,
a `Set` gets 2. Function-valued bindings (partial applications, eta-expanded lambdas) are not
slots and keep their existing path.

Second round, the initializer with local bindings (`let linux : Desc = let a = ... in { ... }`,
the Platform lane's `probe0`/`probe1`): the front end marks *every* binding outside a function
`MainPrologue` (`Bindings.fs`, `env.EnclosingFunction.IsNone`), so the locals of an
initializer were classified as slots too, and `SSAAssignment.assignFunctionBody` skipped every
`MainPrologue` child ("main handles it"), leaving the initializer's calls and literals without
SSAs. A slot is now exactly a direct member of a `ModuleDef` (its module classification's
`ModuleInit` set; `ModuleValues.isSlotBinding` and `SSAAssignment.isModuleValueSlotBinding`
agree), nested `MainPrologue` bindings are ordinary locals of the initializer, and the slot's
initial value is the initializer's last value node (`findLastValueNode`), not the block node.

Samples: `19_ModuleValues`, `23_RecordSurface` (initializer with locals and calls).

### `array-intrinsics`, `array-length`: `Array.length`, `Array.blit`, `.[i]` indexers

Repro: `Fn.len (bytes: byte array) = let n = Array.length bytes in n + 1` ("Binding 'n': Value
not yet witnessed"); `Array.length` in a returned or compared expression lowered to 0 (through
the unit-pattern hole above), so every `Cursor.fits` was false; `Array.blit` compiled and did
nothing; `data.[i]` read and `data.[0] <- v` write unwitnessed.

Root cause (`Composer`): `Alex/Witnesses/MemoryIntrinsicWitness.fs` composed
`zeroCreate/get/set/sub` only; no witness existed for `IndexGet`/`IndexSet`
(`SSAAssignment.producesValue` also returned false for `IndexSet`, so it had no SSA).

Fix: `MemoryPatterns.pArrayLengthIntrinsic` (`memref.dim` + `index.casts`),
`pArrayBlitIntrinsic` (base pointers, element-size scaled offsets, `memcpy`),
`pIndexGetArray`/`pIndexSetArray` (same elision as `Array.get`/`Array.set`), wired into
`MemoryIntrinsicWitness` and `MemoryWitness`.

Sample: `20_ArraySurface`.

### Array literals (`[| a; b; c |]`)

Repro: `let xs = [| 1; 2; 3 |]` printed `count=1`, `first=3` after the witness was added.

Root cause: two layers. `clef` `Expressions/Collections.fs` `checkArrayOrListComputed`
received the literal as one `Sequential` chain (the parser's `ArrayOrListComputed` form) and
built `ArrayExpr [seq]` with a single element; `Composer` had no witness for `ArrayExpr` at
all (`SSAAssignment` gave it a fixed 20 SSAs).

Fix: the front end flattens the `Sequential` chain into elements and reuses the literal path
(`checkArrayOrList`), leaving comprehension bodies (`for`, `while`, ranges, `yield`) alone;
`MemoryPatterns.pBuildArrayLiteral` allocates like `Array.zeroCreate` and stores each element
(`2 + n` SSAs). Element types of arrays of records use the record's physical storage type (see
"Two record representations").

Sample: `20_ArraySurface`.

### `record-arrays`: a record array reached through a let-bound field

Repro: the Hardware lane's `reproC1`, `reproC2`, `reproC5` (`let bits = f.Bits` where `f` is an
element of a `Fld array`, then `Array.get bits b` and `bf.Position`): `Unknown field name:
Position` from `MemoryPatterns.pStructFieldGet`, the fallback for a field access on a value
that is not a `TStruct`. Passing `f.Bits` or `bf` to a function instead worked, which is the
clue.

Root cause (`clef`): `Expressions/Types.fs` `resolveFieldType` defers `x.Field` to a
`Constraint.HasMember(x, Field, result)` when the type of `x` is still a variable (the
constraint that determines it, here from `Array.get`'s instantiation, has been accumulated but
not solved), and `Unify.solveConstraint` accepts `HasMember` with `Ok ()` without ever binding
`result`. `bits` was typed `array<'?3130>` and `bf` `'?3130` in the final PSG, the middle end
mapped the element as a raw memref, and the field lookup fell through. Passing the value to a
function unified the variable with the parameter type, which is why those variants worked.

Fix (`NativeService.dischargeMemberConstraints`): after the equality constraints are solved
(the incremental solve before generalization and the final solve), every `HasMember` whose
base type now resolves to a record with that field (or to `Length` on a string or array) has
its result unified with the field's type, repeated until no constraint makes progress, with the
environment that holds the record table (the fold's final environment, not the initial one).
A constraint whose base never resolves stays open, as before.

Sample: `23_RecordSurface`.

### `record-inference`: a record literal typed as the last record sharing its labels

Repro: the Hardware lane's `reproB3`: `type N = { Name; Count }` and `type F = { Name; Count;
Extra }`; `let r : N = { Name = n; Count = c }` failed with `expected 'F', got 'N'`.

Root cause (`clef` `Expressions/Types.fs` `resolveRecordTypeFromFields`, the multi-candidate
arm): among the record types that have every label of the expression, "last definition wins"
was applied directly, and `F` has both labels.

Fix: a fresh record expression must set every field of its type, so a candidate with more
fields than the expression names is not admissible; "last definition wins" is applied among
the exact matches. Two records with identical field sets are still disambiguated by
declaration order (the `let` annotation is not consulted, as in F# without it); the qualified
label `{ N.Name = ... }` remains the explicit spelling.

Sample: `23_RecordSurface`.

### `recursive-union`: a union that mentions itself, directly or through an `and` group

Repro: the Schema lane's `bisect/a` (`Node = Leaf of int | Branch of Node`) and `bisect/b`
(`Field = { Name; Type: T } and T = P of string | S of Field array`): `NativeType.TError:
Unknown type: Node` after type checking.

Root cause (`clef` `NativeService.checkModuleDecl`, `SynModuleDecl.Types`): a type group was
folded one definition at a time, and a union's case payloads (and a record's fields) were
resolved before the definition's own constructor was registered, so a self-reference or a
forward reference inside the group resolved to `TError`.

Fix: every union and record member of the group is registered twice. First as a placeholder
(`Opaque` layout, so a reference to it counts as one word in the layout estimates), against
which the members' payload and field types are resolved and the final constructors computed;
then the final constructors are registered before the fold, and the fold reuses them
(`groupTycons`), so every reference to a group member, from inside or outside the group,
carries the same `TypeConRef`. `Unify` compares constructors by name and module, so the
placeholder round is invisible to type checking.

Middle end: with the union's cases now visible in its `TypeDef`, the CPU representation of a
user union is sized from the emitted payload representation (next section), so a union that
contains itself through a payload needs no recursion: a nested union or record payload is a
memref descriptor whose size does not depend on the pointee.

Sample: `22_UnionPayloads`.

### `string-payload`: a string (or record) payload read in another function miscompiled

Repro: the Schema lane's `bisect/c1`, `bisect/c1b`: `let prim (k: string) : T = P k` then
`match t with P k -> String.length k` printed 2^56 or 2^60 for `"u8"`.

Root cause (`Composer` `Alex/CodeGeneration/TypeMapping.fs`): a DU with payloads was mapped
to `memref<Nxi8>` with `N` from the front end's `Inline(size)` layout, which counts a string
or record payload as one pointer (`1 + 8 = 9` bytes). The construction pattern (`pDUCase`)
stores the payload at offset 1 through a `memref<1xmemref<?xi8>>` view, that is the
payload's memref *descriptor* (five words, 40 bytes), so `P k` wrote 40 bytes into a 9-byte
allocation, and the length read back through the same view was whatever the heap held.
A module-level `let u8 = P "u8"` happened to survive (`bisect/c`), a payload built from a
parameter did not.

Fix: `TypeMapping.tryGetUnionCases` reads the union's cases from its `TypeDef`, and both
graph-aware mappers (`mapNativeTypeWithGraphForArch`, `mapNativeTypeForTarget` on CPU/MCU)
size a user union as `1 + max case payload` where a payload slot is the scalar's width for a
scalar and the descriptor size for every memory-backed value (`unionPayloadSlotBytes`,
`unionRepresentation`); a union of nullary cases stays an enumeration tag. `T` is now
`memref<41xi8>` and the length is 2.

Sample: `22_UnionPayloads`.

### `record-union-field`: a record with an enumeration field

Repro: `type Kind = Tell | Ask | Reply; type Frame = { Kind: Kind; ... }` failed with
`pAllocValue: expected TMemRefStatic or TStruct, got TTag 3`.

Root cause (`Composer`): `Alex/CodeGeneration/TypeMapping.fs` mapped a nullary-cases-only DU
to the abstract `TTag` on every target; `TTag` is the FPGA representation, and the CPU
patterns (`pDUCase`, `pAllocValue`, record field views) only know memrefs.

Fix: `TypeMapping.setTargetPlatform` is called by `MLIRGeneration` and the enum arm returns
`TTag` for FPGA and a one-byte memref (`memref<1xi8>`) otherwise, the same shape every other
DU has, so construction, tag extraction and record field storage share one path.

Sample: `21_RecordsAndTags`.

### `three-case-match`: three-or-more-arm matches in value position (nested `scf.if`)

Repro: `match k with | Tell -> 0 | Ask -> 1 | Reply -> 2` failed in mlir-opt
("region branch point has 0 operands, but region successor needs 1 inputs"); the Schema
lane's `bisect/c1`, `bisect/c1a`.

Root cause (`Composer`): `Alex/Patterns/ControlFlowPatterns.fs` `pBuildMatchElimination`, DU
path, emitted `[tagLit; cmp; scf.if]` for each intermediate arm without the `scf.yield` that
terminates the enclosing else region, and reused one result SSA for every nesting level (the
constant-match path had the SSA reuse too). Two-arm matches never nest and were fine.

Fix: both paths allocate a distinct result SSA per level from the node's SSA pool and end
every inner else region with a yield of that level's result; the outermost trailing yield is
stripped.

Samples: `21_RecordsAndTags`, `22_UnionPayloads`.

### `module-union-value`: a module-level union value used from another module

Repro: the Schema lane's `bisect/c3` (`let three : T = Q 3` in a library module, matched in
`main`): "Binding 'three': Value not yet witnessed", "DUConstruct: Node has no SSA".

Root cause and fix: the module-level value slots above (`literal-values`); the value's
initializer is a `DUConstruct` and is witnessed in the entry point's prologue like any other.

Sample: `19_ModuleValues` (record and array values), `22_UnionPayloads` (union values built
by functions).

### `while`: `for i = a to b` loops

Repro: `for i = 0 to n - 1 do acc <- acc + i`: `ControlFlowWitness` reported "ForLoop needs
step constant" and the loop variable's `VarRef` had no binding node.

Root cause: `clef` `Expressions/ControlFlow.fs` `checkFor` added the loop variable to the
environment with `None` as its node, and `Composer`'s `ForLoop` witness was a stub.

Fix (front end): `checkFor` desugars the loop into a counted `while` over a mutable cell
(`let __for_end = <end>; let mutable i = <start>; while i <= __for_end do body; i <- i + 1`,
with `>=`/`- 1` for `downto`), so the loop variable is a real binding and the proven
while/mutable machinery lowers it. `ForLoop` no longer reaches the middle end.

Sample: `20_ArraySurface`.

### `immutable-records`: `r.Field <- v` through a parameter

Repro: `let push (w: Writer) b = Array.set w.Data w.Position b; w.Position <- w.Position + 1`
produced an `Error "Cannot assign to 'w.Position' (not found or not mutable)"` node, which the
silent skip then ignored: the program compiled and `Position` stayed 0.

Root cause: `clef` `Expressions/Bindings.fs` `checkLongIdentSet` looked the whole dotted name
up as a mutable binding; `Composer` had no `FieldSet` witness.

Fix: `checkLongIdentSet` resolves the longest prefix that names a binding and builds
`FieldGet`s for the intermediate fields and a `FieldSet` for the last;
`RecordPatterns.pRecordFieldSet` stores through a typed view at the field's byte offset
(records are memref-backed, so the mutation is visible through every reference). Field
mutability is not checked (the record definition does not carry it).

Sample: `21_RecordsAndTags`.

### Qualified member access `Module.value.Field`

Repro: `Spaces.rodata.Name` — "The value or constructor 'Spaces.rodata.Name' is not defined".

Root cause: `clef` `Expressions/Identity.fs` `resolveBinding` tried only the first identifier
as the base binding. Fix: try the longest proper prefix that names a binding, the rest as a
member path.

Sample: `19_ModuleValues`, `21_RecordsAndTags`.

### `string-equality`, `string-tags`: `=` and `<>` on strings

Repro: `s.Kind = Kind.Arena` failed in mlir-opt: `arith.cmpi 'lhs' must be signless-integer-
like, but got 'memref<?xi8>'`. Every string-alias tag comparison in BAREWire goes through this.

Root cause (`Composer`): `ApplicationPatterns.pComparisonOp` emitted `arith.cmpi` for every
non-float operand type. Fix: `StringPatterns.pStringEquality` compares lengths and then bytes
with `memcmp` (16 SSAs; `SSAAssignment` sizes `op_Equality`/`op_Inequality` on string or array
operands accordingly); the comparison pattern dispatches to it for memref operands.

Sample: `21_RecordsAndTags`.

### `string-constant`: a string constant in a `[<RequireQualifiedAccess>]` module

Repro: the Platform lane's `p4d`/`p4e` (`[<Literal>] let Stack: string = "stack"` and the
plain `let Stack: string = "stack"`, compared with `=` from another function). Root cause and
fix: the literal case is the string-equality witness above; the plain value is a module-level
slot (`literal-values`). Both spellings lower.

Sample: `21_RecordsAndTags` (`Kind.Arena`, `Kind.Stack`).

### `short-circuit`: `||` and `&&` evaluated both operands

Not a rule of `docs/12` yet; found by the RoundTrip gate, which died with SIGFPE in
`Check.run` at `b.Framing <> Framing.Ring || (b.Slot > 0 && ... b.Capacity % int64 b.Slot = 0L)`
for a delimited buffer (`Slot = 0`).

Root cause (`Composer` `Alex/XParsec/PSGCombinators.fs`): `op_BooleanOr` and `op_BooleanAnd`
were classified as binary arithmetic (`arith.ori`, `arith.andi`) over operands that had
already been evaluated in the enclosing block, so a guarded division ran unconditionally.

Fix (`clef` `NativeService.checkExpr`): `a || b` and `a && b` are desugared, before checking,
to the conditionals F# defines them as (`if a then true else b`, `if a then b else false`),
whichever spelling the parser produced for the operator (`SynExpr.Ident` or
`SynExpr.LongIdent` with operator trivia). The existing if-expression lowering (`scf.if`)
evaluates the right operand only when the left does not decide.

Sample: `23_RecordSurface`. Proposed rule text for `docs/12`, now retired on arrival: "the
right operand of `||`/`&&` ran unconditionally on the snapshot (a guarded division faulted)".

### Unsigned widening and shift amounts

Repro: `int (b: byte)` gave `-84` for `0xAC`; `(v: uint32) >>> 8` failed in mlir-opt
(`%v99 expects different type: i32 vs i64`).

Root cause (`Composer`): `ApplicationPatterns.pTypeConversion` always sign-extended;
`pBinaryArithOp` passed the platform-`int` shift amount to an i32 shift.

Fix: widening follows the source's Clef signedness (`extui` for `NTUuint`, `size_t`, `bool`,
`char`); shift amounts are brought to the operand's width.

Samples: `21_RecordsAndTags` (byte widening), the BAREWire encoding gate (shifts).

### `string-to-bytes`

Repro: `Text.toUtf8 s = String.toBytes s` produced no result. Root cause:
`StringIntrinsicWitness` had `fromBytes` but not `toBytes`. Fix: `pStringToBytesIntrinsic`,
the identity on the memref (a string is its UTF-8 bytes).

Sample: `20_ArraySurface` indirectly (byte arrays); the Platform lane's `p15b`, `p4a`.

### A user type shadowed by a spliced abbreviation

Repro: `21_RecordsAndTags` declares `type FrameKind = | Tell | Ask | Reply`; every dependent
of Fidelity.Platform splices BAREWire first, whose `BAREWire.Framing` declares `type FrameKind
= byte`. The sample's own annotation `(k: FrameKind)` resolved to `byte`: `expected 'uint8',
got 'FrameKind'`.

Root cause (`clef` `Expressions/Types.fs` `resolveTypeName`): abbreviations are consulted
before type definitions, and `addTypeDef` left an earlier same-named abbreviation in place.
Fix: registering a type definition retires the same-named abbreviation, so the later
declaration wins in both directions (an abbreviation declared after a type already won).
Record definitions are still keyed by short name (`RecordDefs`), so two records of the same
short name in different namespaces still collide; see Open issues.

Sample: `21_RecordsAndTags`.

### An apostrophe in a function name

Repro: BAREWire's `Lifecycle.process'` (RoundTrip): `mlir-opt` rejected `func.func
@Lifecycle.process'(...)` ("unexpected character"). Fix: `Alex/Dialects/Core/Serialize.fs`
`symbolName` spells `'` as `$` (legal in an MLIR bare id and an ELF symbol, present in no Clef
name) at every symbol site (`func.func`, `func.func private`, `func.call`, `func.constant`).

Sample: `22_UnionPayloads` (`Options.half'`, `Options.scale'`).

### Two record representations

Not a rule in `docs/12`, but the cause of the last failures in the module-value repros:
`TypeMapping.mapNativeTypeForArch` mapped a record to the front end's `Inline(size)` layout
(a string field counted as 8 bytes) while the graph-aware mappers build a `TStruct` whose
string field is a 40-byte memref descriptor; the `array`, `option` and `Result` arms of the
graph-aware mappers delegated their element types to the arch-only mapper, so `Space array`
became `memref<?xmemref<16xi8>>` while a `Space` occupies 48 bytes. `physicalStorageType` is
now the single definition of a record's storage type and the container arms use it.

## Not gaps on this snapshot

The following rules of `docs/12` were re-verified with isolated probes and pass on the
snapshot unchanged; their failures in the encoding gate were cascades of the gaps above.

- `if-argument`: an if-expression as a call argument, as a tuple element, and as the last
  expression after a `while` all lower (the WrenHello 6 finding was an older build).
- `value-match`: a `match` in statement position with unit arms that call functions lowers,
  for a three-case union and for `option`.
- `option-argument`: a `Some`-bound record passed as an argument (the Platform lane's `p13`)
  lowers on the snapshot; the reported failure was a cascade.
- `array-length-in-array-returning-function` (`p9a`, `p9c`, `p10b*`, `p12`): the
  `array-length` fix.

## Deferred

### `arrays` (the List surface)

`List.length`, `List.head`, `List.tail`, `List.isEmpty`, `List.map` are "not defined" for user
source; `samples/console/FidelityHelloWorld/13a_SimpleCollections` fails on the snapshot for
this reason and is not in the regression manifest.

Root cause: `clef` `Expressions/Intrinsics.fs` `resolveCollectionOp` returns `NotAnIntrinsic`
for `List`, `Map`, `Set`, `Option`, `Result` ("resolved through Baker elaboration"), so the
name falls through to binding lookup and fails. Downstream, `Composer`'s `ListWitness`
matches a bare `Intrinsic` node with children (the shape Baker's recipes create), not the
`Application (Intrinsic List.op, args)` shape user source produces, and `ListExpr` has no
witness. A fix needs (1) typed List intrinsics in `resolveCollectionOp` (`TList` types), (2)
either a Baker normalization of user-source applications into the recipe shape or an
`Application` arm in `ListWitness`, (3) a `ListExpr` witness (cons chain), and (4) a
`Composer` cost table entry per operation. BAREWire's tables stay arrays until then.

### Samples 14, 15, 16

`14_Lazy` ("pBuildLazyForce: Expected at least 5 SSAs, got 3"), `15_SimpleSeq` (3 type
errors) and `16_SeqOperations` ("Unhandled intrinsic 'SeqEnumerator.moveNext'") fail
identically on the snapshot and on this build; they are outside the BAREWire surface and were
not investigated.

### Samples 06, 11, 12 and 13

`11_Closures` and `12_HigherOrderFunctions` compile on both builds and print the same wrong
values on both (`Hello, !`, `add10 5: 5`, then a segfault in 12): captured values inside
closures are read back as zero or empty. `13_Recursion` (`Recursion.fidproj`) fails in
`mlir-opt` on both builds ("region entry argument"); its `MutualRecursion.fidproj` compiles and
runs on both. The regression manifest's expected output does not match either build. Not
investigated; the BAREWire tiers use no closures.

`06_AddNumbersInteractive` is unchanged: `Console.readln` returns everything one `read`
delivers, so a pipe that already holds both input lines hands both back on the first call
(`983285.29... + 0`, on the snapshot and the rebuilt compiler alike, with or without the
BAREWire splice); the regression harness writes stdin one line at a time with a pause for
exactly that reason (`Runner.fsx`, `runProcess`) and the sample passes under it. Whether
`readline` should stop at the delimiter and keep the rest for the next call is a platform
question outside this lane.

### Compile time with `-k`

`Composer compile -k` takes about 65 s per sample on both builds (85 MB of intermediates, the
`%A`-formatted PSG dumps of the whole spliced program, BAREWire included) against 6 s without
`-k`; the regression harness passes `-k`, so a full manifest run is now about 25 minutes. The
dump format, not this round's changes, sets that cost.

### Rules not exercised

`nested-pattern`, `two-cases`, `single-payload`: not reproduced in this round; the
documented workarounds stand.

## Open issues

- Record definitions and field labels are keyed by the record's short name (`RecordDefs`,
  `FieldLabels` in `Expressions/Types.fs`), so a user record named like a spliced one (`Frame`,
  `Region`, `View`) replaces the spliced definition for field lookup. Type definitions are
  registered under every module-path suffix, which is what saved `FrameKind`; records need the
  same, keyed by the most qualified name with the short name as an alias.
- `tryResolveRecordFieldType` returns a generic record's field type without substituting the
  record's type arguments (`_typeArgs` is ignored), so field access on `Box<'a>` shares the
  definition's parameter across uses, the same shape as the option-pattern defect above.
- `String.length` counts UTF-8 bytes natively and UTF-16 units on .NET; a transcript that
  prints a non-ASCII string's length differs by substrate. BAREWire's `Text` shim owns that
  reconciliation; the Encoding gate's `strlen` line is the only cross-substrate difference left.
- The CPU union representation stores a record payload as a descriptor (`memref<1xmemref<Nxi8>>`
  view) while a record *field* of record type is stored inline in the enclosing `TStruct`;
  both are consistent with their own readers, but the two conventions should converge before
  unions carry large records.

## HelloProof

`HelloProof/sample/HelloWorld.fidproj` compiles with the rebuilt compiler, its
`06a_obligations.json` lists the same 19 obligation ids as the committed one, `cvc5` reports
19 `unsat` on `06b_obligations.smt2`, `mlir-opt` accepts `09_obligations.mlir`, and the binary
prints `Enter your name: Hello, Houston!` for `echo Houston`. The committed intermediates
`01_psg0/03_psg1/05_psg2/06_coeffects.json` differ from a fresh `-k` run by timestamps,
string-table ordering and SSA numbering (VarRef, Set and ArrayExpr costs; nested-binding SSAs),
so a re-snapshot should re-commit those intermediates together with the compiler. The
`ConcatCopyBound` obligations, the observer over all reachable literals and the FSharp.Core
10.1.400 pin were already in the working tree the snapshot was taken from (HelloProof
`PINS.md`) and are unchanged by this round.

## Regression samples added

| Sample | Guards |
| --- | --- |
| `18_Generalization` | let-polymorphism, monomorphization, generic `option` |
| `19_ModuleValues` | module-level values across modules and functions, mutable module value, qualified member access |
| `20_ArraySurface` | `Array.length`, literals, indexers, `Array.blit`, `for`/`downto`, arrays of records |
| `21_RecordsAndTags` | field assignment through a parameter, enum union field, nested matches, string-tag `=`/`<>`, byte widening, a user type shadowing a spliced abbreviation |
| `22_UnionPayloads` | string and record union payloads, a self-referencing union, an `and` group, option patterns at two payload types, an apostrophe in a function name |
| `23_RecordSurface` | nested record arrays through a let-bound field, record literal inference by exact field set, a module value with local bindings and calls, short-circuit `||`/`&&` |
