# Mutable cells in direct capture passing

**Status: implementation direction, 2026-09-19. Not implemented or a new language surface.**
The current [direct-capture admission](../../clef/src/Compiler/PSGSaturation/SemanticGraph/DirectCaptures.fs#L40)
excludes mutable captures. This note defines the next bounded contract; it does not
claim completion of C-01 or select `TCell` as an approved representation.

## Governing decisions

[Closure representation §8](../../clef-lang-spec/spec/closure-representation.md#L203)
requires leading capture parameters when all uses establish the nonescaping direct
form; binding parentage alone is insufficient. [§2.2 and §3.3](../../clef-lang-spec/spec/closure-representation.md#L43)
require shared mutable storage and a lifetime covering every capturing closure.
[§11](../../clef-lang-spec/spec/closure-representation.md#L279) preserves unresolved
storage premises; a finite frontier does not prove transitive storage lifetime.

[C-01 §14.2–14.4](PRDs/C-01-Closures.md#L813) supplies the unmaterialized form,
portable typed views, application agreement and residence obligations. Its
unmaterialized environment needs no environment allocation; that does not eliminate
the captured cell's own storage obligations. The [closure retooling mechanism](../../clef/docs/fidelity/phg/Closure_Retooling_Plan.md#L149)
owns construction through Baker ingredients, recipes and fan-out/fold-in. Alex
pulls settled facts at Huet positions through witnesses, patterns and Elements.

## Representation choice to settle

`Cell<T>` below means a compiler-internal reference to existing mutable storage.
It is neither a source constructor nor a raw address. Two representations remain
possible:

| Choice | How application types remain truthful | Cost / constraint |
| --- | --- | --- |
| Internal storage type, such as `NativeType.TCell T` | Hidden formals and forwarding operands have `Cell<T>`; the elaborated `TFun` scheme includes those domains. Reads produce `T`; writes require `T`. | Extend substitution, equality, generalization/instantiation, range classification, layout and all type consumers. No source spelling or implicit conversion to/from `T`. |
| Typed graph facts on hidden formals | Keep the source scheme and give the elaborated callable one authoritative typed signature containing distinct value and cell domains. Call facts bind every hidden actual to its cell formal; VC-APP checks this signature. | Every signature consumer must use that contract. Appending cell operands to ordinary `Application` while leaving its scalar-only `TFun` unchanged is invalid. A witness-only annotation is insufficient. |

The second choice can avoid a new `NativeType` case, but cannot avoid an explicit
storage sort in the graph's application contract. Choose one authority before
implementation; do not maintain competing signatures. Neither `TByref` mapped to
`index`, scalar `AddressOf` materialization, nor a foreign BAREWire `BorrowedView`
establishes this lexical-cell contract.

## Semantic nodes and facts

The minimal operations are explicit regardless of the chosen encoding:

| Operation | Result / meaning |
| --- | --- |
| `CellRef(owner)` | `Cell<T>`: forward the existing binding cell or hidden formal; no load, allocation or copy of `T`. |
| `CellRead(cell)` | `T`: read that cell at this program point. |
| `CellWrite(cell, value)` | Unit: evaluate `value : T`, then update the same cell. |

The original mutable binding remains the allocation site. Baker rewrites captured
value reads and assignments into these operations, and supplies `CellRef` at each
direct call. A hidden formal binds the cell descriptor; reading the formal as a
reference must not silently load its element.

Before range analysis, retain a typed origin relation from each hidden formal and
cell reference to its source binding, with call-specific actual/formal edges.
After range and placement, project a `CellView` for each admitted owner/formal:
`{ Origin: NodeId; Element: NativeType; Extent: 1; Slot: SettledSlot }`.
The first slice permits only resolved scalar slots. Element dimensions remain
intact; a cell reference itself is not a numeric value with the element's range.
Residence evidence remains in the owning graph obligations, not a Boolean flag
claiming proof. Missing or conflicting facts prevent commitment upstream.

An origin ID identifies an allocation **site**, not a single global runtime cell.
Different activations of an enclosing function have different storage instances.
Forwarding the actual descriptor preserves that distinction. Recursive calls pass
their incoming cell formal; they do not allocate another cell or look up an outer
function's SSA by origin ID. Mutually recursive call groups need a consistent
capture-parameter contract and a fixed point of their write effects before admission.

## Range, effects and provenance

All initializer and assignment values, including writes through lifted formals,
contribute to the original cell's storage range. Placement selects one element
carrier from that join. A guarded read may have a narrower range without narrowing
the shared cell's carrier. Map alias writes back to their origin and compute
transitive may-write effects over the admitted direct-call graph; a call invalidates
guard refinements for every cell it may write. A missing effect summary is not
evidence of purity. Recursive groups must settle together.

Retain original read/write node identities and source ranges where possible.
Provenance and obligation participants include the source binding, lambda, hidden
formal, forwarding operand, call and relevant writes. Origin links are reference
or provenance edges, not structural children that cause the source allocation to
be emitted again. Fold-in must remap these references and proof participants.

## Current gaps and the first admission boundary

- [Lambda parameter mapping](../src/MiddleEnd/Alex/Witnesses/LambdaWitness.fs#L299)
  currently reads only native type/range. It needs the settled cell formal view;
  [parameter references](../src/MiddleEnd/Alex/Witnesses/VarRefWitness.fs#L41) currently
  forward values without a cell operation distinction.
- [Direct call composition](../src/MiddleEnd/Alex/Patterns/ApplicationPatterns.fs#L48)
  already passes typed operands. [Mutable load/store patterns](../src/MiddleEnd/Alex/Patterns/MemRefPatterns.fs#L58)
  already compose the memory Elements. New cell observation must reuse those
  operations, retaining the exact `memref<1xT>` view, without casts or witness inference.
- [Range analysis](../../clef/src/Compiler/PSGSaturation/SemanticGraph/RangeAnalysis.fs)
  now computes finite may-write summaries for existing calls and invalidates
  affected guard facts in operand evaluation order. Saved Boolean observations
  cannot reinstate stale mutable bounds. Assignment collection still recognizes
  `Set` through `VarRef`; future cell aliases must join through their canonical
  origins and participate in those same effect summaries before admission.

The first slice is direct, fully accounted, nonescaping calls over scalar mutable
cells whose uses remain within the originating storage lifetime. Preserve the
existing source argument evaluation order. A returned closure that recaptures the
cell is outside this initial admission: it requires established residence covering
the closure's full lifetime and a settled bounded view at the recapture site.
Directness of the intermediate helper does not prove either condition.

Completion needs CCS graph/type and negative admission tests; range tests for
writes after guards and through recursive calls; Alex position-to-pattern tests
for the settled view and missing evidence; and source-to-native oracles for shared
updates and independent enclosing activations. MLIR verification/lowering checks
carrier agreement, not residence or source-language conformance.
