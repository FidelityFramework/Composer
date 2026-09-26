# Native Type Universe architecture

Clef's native type universe is governed by the [native type specification](../../clef-lang-spec/spec/native-type-universe.md),
[dimensional architecture](../../clef-lang-spec/spec/ntu-dimensional-architecture.md)
and [numeric selection contract](../../clef-lang-spec/spec/numeric-selection.md).
.NET supplies the compiler host; its type and runtime model does not define Clef.

## Source identity and representation

Source `int` and `float` kinds retain their inferred physical dimensions. Source
kind, physical dimension, scale, justified range and selected representation have
distinct roles. A target's register width does not establish a value's range or
select its representation. Platform and boundary declarations supply capabilities
and constraints that CCS/Baker consumes before representation commitment.

CCS/Baker owns:

- Native kind and dimensional inference, including quantified measure schemes.
- Range, scale, error and operation-definedness reasoning with exact provenance.
- Selection among admitted integer, IEEE, posit and fixed-point representations.
- Storage extent, alignment, capacity, access, lifetime and ABI settlement.
- The obligations and complete joint premises supporting those conclusions.

Required unresolved facts remain explicit and prevent the commitment that needs
them. Neither a convenient machine width nor a matching representation name is
proof of coverage or boundary fidelity. The specification's bounded bare-float
exception supplies no general permission to default unresolved representations.

## Publication and witnessing

Owning nanopasses elaborate and saturate the PSG through Baker ingredients,
recipes, fan-out and fold-in. Local results and joint relations retain their
participants, order, multiplicity, source identity and proof status. Changes to
those premises invalidate their dependent conclusions.

The source-owned publication contains immutable settled types, representations,
layouts, physical signatures and occurrence-specific facts. Alex observes those
facts through the Huet zipper and composes Elements through Patterns and
Witnesses. It does not infer dimensions, select widths, analyze escape, choose a
lifetime, evaluate platform predicates or repair missing settlement.

Physical operand naming and emission correspondence belong to witnessing. They
do not create semantic authority. Target-specific instructions, encodings and
lowering belong to Composer's selected backend, which preserves or re-establishes
the required source-to-artifact correspondence.

```text
Clef source + declared platform/boundary contracts
  -> CCS native type and dimensional checking
  -> Baker elaboration and owning nanopass saturation
       range, numeric selection, storage/layout, ABI and proof settlement
  -> immutable source-owned publication
  -> Alex: passive Huet Element/Pattern/Witness composition
  -> selected backend: target realization and artifact checks
```

## Platform predicates

CCS/Baker establishes compile-time platform predicates from their declared
authority and settles the resulting executable structure. Alex witnesses that
structure; it does not decide which source branch or numerical policy applies.
Changing a declaration retracts the conclusions and artifacts that depend on it.

## Delivery and evidence

[F-11](PRDs/F-11-NumericSelection.md) owns numeric selection and obligations;
[C-08](PRDs/C-08-ArithmeticConstruction.md) owns arithmetic construction;
[M-01](PRDs/M-01-DialectAdmission.md) owns admitted physical forms and downstream
information transport. Their source, proof, artifact and native gates establish
different observations. No representation table alone establishes conformance.

- [CCS architecture](CCS_Architecture.md)
- [Baker saturation](../../clef/docs/fidelity/Baker_Saturation_Architecture.md)
- [Alex architecture](Alex_Architecture_Overview.md)
- [Compiler pipeline](Architecture_Canonical.md)
