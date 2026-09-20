# Native math admission fixture

Status: proposed, unregistered, and not compiled or run. No math witness or
lowering implementation is added by these files.

`Sine.clef` exercises dimensionless `Math.sin` at `0.0`, `-0.0`, `0.25` and
`-0.25`, with finite result intervals and an odd-symmetry residual check. Exit
codes 1–5 identify those checks; success prints exactly:

```text
math.sin finite scalar: pass
```

The zero comparisons check the numerical zero result for both source inputs.
They do not distinguish the IEEE sign bit of the result. A signed-zero bit-level
claim needs a separate target artifact/ABI oracle. The quarter-point bounds are
regression tolerances, not a compiler-derived enclosure or a rounding guarantee.
No non-finite behavior, general real-range analysis, posit/fixed-point selection,
or additional math operation is claimed.

## Existing dependency path

The fixture selects the Linux x86_64 CompilerSurface and explicitly declares
`[link] libraries = ["m"]`. This uses the existing contract, without assuming
that every target supplies libm:

- `clef/src/Compiler/Project/FidprojLoader.fs:151` parses `link.libraries`.
- `SourceResolver.fs:184` combines project and dependency declarations.
- `Composer/src/MiddleEnd/MLIRGeneration.fs:142` carries them into the native
  external-library set; `BackEnd/LLVM/Codegen.fs:117` emits the corresponding
  linker arguments. Automatic console linkage adds `c`, not `m`.

This declaration identifies a link dependency. It does not itself prove a
numeric range, rounding property, or operation capability.

## Governing integration requirements

Read in full for this slice: the relevant numeric implementation guide sections
0–12, [Thin Middle End](../../docs/Thin_Middle_End_Design.md), the language review's
[NTU/dialect sections](../../docs/Clef_Language_Completion_Review_2026-09-19.md),
and the dimensional design's selection, boundary, rounding and witness sections
3–6/8.3. The implementation guide explicitly defers normative decisions to
[numeric-selection.md](../../../clef-lang-spec/spec/numeric-selection.md).

1. **Admit the operation and its actual operands.** Current
   `clef/src/Compiler/NativeTypedTree/Expressions/Intrinsics.fs:632` resolves `Math.sin` as dimensionless
   `float -> float`; the unqualified library-scheme table does not add bare
   `sin`. Preserve that source identity and the actual call/argument/result
   relations. A library symbol string cannot replace those relations.
2. **Settle representation and applicable premises in CCS.** The
   [numeric implementation guide](../../docs/Numeric_Selection_Implementation.md)
   §§2–3/6/9/12 requires declaration provenance, justified range facts, validity
   context, capability and selection before witnessing. The specified bare-float
   path permits offered and policy-allowed IEEE `f64` when range remains
   unobservable; record that basis rather than selecting a default in Alex.
   Any known bounds and transfer requirements still apply. A finite test input
   does not establish the domain contract of every possible call.
3. **Keep distinct proof claims distinct.** Current
   `ObligationRecipes.realLiteral` records an exact rational singleton;
   `realCoverage` relates that literal to the declared representation capacity
   and explicitly provides no rounding guarantee. Neither establishes the image
   or accuracy of `sin`. `Intrinsics.RangeSources` currently leaves transcendental
   images untabled. Full real image evaluation requires the real interval domain
   and its supported enclosure rules; integer `ValueRange` is not that domain.
   Numeric-selection §4 requires terminating image computation, not a sine
   formula mislabeled as QF_LRA/QF_LIA. Attach accepted facts and obligations to
   their actual participants; Alex consumes the settled projection and does not
   discover or discharge them by inspecting graph edges.
4. **Register portable math vocabulary.** Add a `math` admission row/design
   citation/sample under the Thin Middle End vocabulary rule and the closure
   retooling plan's dialect gate. Its historical five-dialect list must be updated
   with the implementation. Introduce typed `MathOp.Sin` and serialization,
   internal `MathElements`, `MathPatterns`, a registered `MathIntrinsicWitness`,
   and project compile ordering. The pattern consumes assigned values, recalled
   operands, settled physical types and required meets; it cannot infer a new
   representation or invent a proof.
5. **Lower only on an admitted target path.** Local upstream sources at
   `circt/llvm/mlir/lib/Conversion/MathToLibm/MathToLibm.cpp` map scalar sine to
   `sinf`/`sin` and introduce `func` calls/declarations. Place that conversion
   before `convert-func-to-llvm`, retaining `math.sin` in witnessed MLIR first.
   `MathToLLVM/MathToLLVM.cpp` alternatively maps it to `LLVM::SinOp`; that route
   does not establish that native codegen has no library requirement. Composer's
   current LLVM pipeline contains neither math conversion. Do not add fast-math
   assumptions or silently promote unsupported representations; the local libm
   pass also contains promotion patterns outside this proposed scalar gate.

The guide's milestone order remains carriage, boundary selection, real domain,
then later quire/domain-law work. This finite operation fixture is evidence for
one realization; it cannot mark those milestones complete. Historical warnings
or range-provenance hierarchies in older dimensional notes do not override the
current guide/spec's joint evidence and hard coverage rules.

## Missing scalar admission contract

The following are existing mechanisms, not a settled scalar real call plan:

- `clef/src/Compiler/NativeTypedTree/NativeTypes.fs:809` carries a declared
  representation's name, capability, family, bits, bounds and boundary behavior.
  `PlatformResolution.DeclaredRepresentation` also retains its declaration node.
  Linux x86_64 declares both native IEEE representations in
  `Fidelity.Platform/Hardware/Silicon/CPU/x86_64/Architecture.clef:100`.
- `ObligationElaboration.fs:311` starts with a real literal's existing
  `NTUfloat(Fixed bits)` and finds the matching offered IEEE declaration. The
  resulting literal coverage obligation is useful evidence; it is not a real
  expression selection pass or a math provider admission.
- `SemanticGraph/Placement.fs:144` constructs `SettledSlot.Real bits` from the
  kind, including posit kinds. This slot retains neither representation family
  nor declaration identity. `Alex/CodeGeneration/TypeMapping.fs:146` maps 32 to
  `f32` and other real slots to `f64`; scalar kind mapping similarly reads fixed
  widths. These existing mappings cannot authorize this new math path.
- `Codata` has no scalar real selection map. `PlatformBindings.resolve` handles
  selected system intrinsics and attributed extern calls, not `Math.sin`.
  Project `link.libraries` establishes linkage, not a typed math provider.

The smallest proposed Baker prerequisite is a call plan for this admitted
operation, produced from the selected platform declaration and the resolved
intrinsic identity. It must retain the call, operand and result identities; the
actual declared representation with its provenance; the applicable selection
basis and capability policy; a typed provider descriptor and its provenance;
and the applicable facts/obligations. Operand production, callable parameter and
result realization must agree with that plan. Existing fixed-width kinds alone
cannot stand in for this evidence.

A selected-platform mapped `Math.sin` descriptor can supply provider authority
if it explicitly relates this operation and its argument/result representations
to the provider ABI. Validate that relation upstream and preserve it in the
plan. A descriptor containing only a library/symbol name does not establish the
representation match or numeric behavior. The declared link requirement must
also reach the native linker. No such math descriptor or plan is added here.

For a dimensionless bare value with an unobservable range, the spec's explicit
offered-and-permitted IEEE `f64` policy can be recorded as the selection basis.
That limited basis cannot replace known range/transfer requirements, justify a
dimensioned fallback, or claim that arbitrary values fit finite bounds. The
provider's supported domain and any required accuracy premises must remain
explicit. This fixture does not supply a general transcendental enclosure or a
rounding proof. Pending or unsupported premises must not become Alex defaults.

## Planned gates

- CCS admission of `Math.sin 0.25`, rejecting a measured argument such as
  `Math.sin 1.0<distance>`, integer/bool arguments, and scalar overapplication;
  preserve the public effective diagnostic code and complete source span.
  These are proposed cases; their results and exact diagnostic spans are untested.
- Missing/unresolved selection or unsupported capability must not produce a
  guessed float representation. A missing recalled operand gets a specific Alex
  component diagnostic. Existing literal evidence remains attached unchanged.
- Inspect retained MLIR for `math.sin` in `sine`, verify it with stock `mlir-opt`,
  lower through the selected standard math conversion, and verify the resulting
  LLVM dialect. Exercise supported physical scalar types in component tests.
- Compile this project, require exit zero and exact output, and retain compiler,
  target, library and tool identities. No such gate has been run yet.

Once the compiler implementation is ready, from Composer:

```sh
src/bin/Debug/net10.0/Composer compile tests/NativeMath/Sine.fidproj -k --no-color
tests/NativeMath/targets/native-sine
```
